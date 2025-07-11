#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
import torch
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.nets.builder import build_net
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.agents.builder import build_agent
from trademaster.optimizers.builder import build_optimizer
from trademaster.losses.builder import build_loss
from trademaster.trainers.builder import build_trainer
from trademaster.utils import plot
from trademaster.utils import set_seed
import ray
from ray.tune.registry import register_env
from trademaster.environments.portfolio_management.environment import (
    PortfolioManagementEnvironment,
)
import shutil

# Set seed for reproducibility
set_seed(2023)

# Set up paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)


def env_creator(env_name):
    if env_name == "portfolio_management":
        env = PortfolioManagementEnvironment
    else:
        raise NotImplementedError
    return env


def create_regime_config(base_config_path, regime_name, output_dir):
    """Create a regime-specific config file by modifying the base MCAD config"""
    print(f"Creating config for {regime_name} regime...")

    # Load base config
    cfg = Config.fromfile(base_config_path)

    # Modify paths for the specific regime
    regime_data_path = f"data/portfolio_management/mcad/{regime_name}"
    cfg.data.data_path = regime_data_path
    cfg.data.train_path = f"{regime_data_path}/train.csv"
    cfg.data.valid_path = f"{regime_data_path}/valid.csv"
    cfg.data.test_path = f"{regime_data_path}/test.csv"
    cfg.data.test_dynamic_path = f"{regime_data_path}/test_with_label.csv"

    # Update tech indicators to match what's available in regime datasets
    # Remove auto-generated features that don't exist in individual regime datasets
    regime_tech_indicators = [
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30",
    ]
    cfg.data.tech_indicator_list = regime_tech_indicators

    # Update work directory to include regime name
    original_work_dir = (
        cfg.work_dir if hasattr(cfg, "work_dir") else cfg.trainer.work_dir
    )
    new_work_dir = f"{original_work_dir}_{regime_name}"
    if hasattr(cfg, "work_dir"):
        cfg.work_dir = new_work_dir
    cfg.trainer.work_dir = new_work_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the regime-specific config
    regime_config_path = os.path.join(
        output_dir, f"portfolio_management_mcad_ppo_ppo_adam_mse_fg_{regime_name}.py"
    )
    cfg.dump(regime_config_path)

    print(f"✅ Created config for {regime_name}: {regime_config_path}")
    return regime_config_path


def run_ppo_training(config_path, regime_name):
    """Run PPO training for a specific MCAD regime"""
    print(f"\n{'='*60}")
    print(f"Starting PPO Training for MCAD {regime_name.upper()} Regime")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    try:
        # Load config
        cfg = Config.fromfile(config_path)
        cfg = replace_cfg_vals(cfg)

        # Build dataset
        print(f"Building dataset for MCAD {regime_name}...")
        dataset = build_dataset(cfg)

        # Initialize Ray and register environment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        work_dir = os.path.join(ROOT, cfg.trainer.work_dir)

        ray.init(ignore_reinit_error=True)
        register_env(
            "portfolio_management",
            lambda config: env_creator("portfolio_management")(config),
        )

        # Create work directory
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        cfg.dump(osp.join(work_dir, osp.basename(config_path)))

        # Build trainer with numpy compatibility fix
        print(f"Building trainer for MCAD {regime_name}...")
        trainer = build_trainer(cfg, default_args=dict(dataset=dataset, device=device))

        # Add the crucial fix for numpy compatibility - this prevents the np.bool error
        # This is the same fix used in PortfolioManagementSARLTrainer
        if hasattr(trainer, "configs"):
            trainer.configs["disable_env_checking"] = True
        else:
            # If configs doesn't exist, create it
            trainer.configs = {"disable_env_checking": True}

        # Run training and validation
        print(f"Starting training for MCAD {regime_name}...")
        trainer.train_and_valid()

        # Run testing
        print(f"Running test for MCAD {regime_name}...")
        try:
            trainer.test()

            # Generate plots
            print(f"Generating plots for MCAD {regime_name}...")
            plot(
                trainer.test_environment.save_asset_memory(),
                alg=f"PPO_MCAD_{regime_name}",
            )
        except Exception as test_error:
            print(
                f"⚠️  Testing failed for MCAD {regime_name}, but training completed successfully: {str(test_error)}"
            )
            # Training was successful even if testing failed

        print(f"✅ MCAD {regime_name.upper()} regime training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error in MCAD {regime_name.upper()} regime training: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Don't shutdown Ray here - let individual processes handle it
        pass


def main():
    """Main function to run PPO training for all three MCAD regimes"""

    # Define the three MCAD regimes
    mcad_regimes = ["trade_war_i", "covid", "trade_war"]

    # Base config file (the original MCAD PPO config)
    base_config_path = os.path.join(
        ROOT,
        "configs/portfolio_management/portfolio_management_mcad_ppo_ppo_adam_mse_fg.py",
    )

    # Directory to store regime-specific configs
    regime_configs_dir = os.path.join(ROOT, "temp_regime_configs")

    print("🚀 Starting PPO Training for All Three MCAD Regimes")
    print("MCAD Regimes to train:")
    for regime in mcad_regimes:
        print(f"  - {regime.upper()}")

    if not os.path.exists(base_config_path):
        print(f"❌ Base config file not found: {base_config_path}")
        return

    results = {}
    regime_config_paths = {}

    try:
        # Step 1: Create regime-specific config files
        print(f"\n{'='*60}")
        print("STEP 1: Creating regime-specific configuration files")
        print(f"{'='*60}")

        for regime in mcad_regimes:
            regime_config_path = create_regime_config(
                base_config_path, regime, regime_configs_dir
            )
            regime_config_paths[regime] = regime_config_path

        # Step 2: Run training for each regime
        print(f"\n{'='*60}")
        print("STEP 2: Running PPO training for each regime")
        print(f"{'='*60}")

        for regime in mcad_regimes:
            config_path = regime_config_paths[regime]
            results[regime] = run_ppo_training(config_path, regime)

        # Print final summary
        print(f"\n{'='*70}")
        print("FINAL TRAINING SUMMARY - MCAD REGIMES")
        print(f"{'='*70}")

        for regime, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"MCAD {regime.upper():<12}: {status}")

        successful_regimes = sum(results.values())
        total_regimes = len(results)

        print(
            f"\nOverall: {successful_regimes}/{total_regimes} MCAD regimes completed successfully"
        )

        if successful_regimes == total_regimes:
            print("🎉 All MCAD regimes trained successfully!")
        else:
            print("⚠️  Some MCAD regimes failed. Check the logs above for details.")

    finally:
        # Clean up temporary config files
        if os.path.exists(regime_configs_dir):
            print(f"\n🧹 Cleaning up temporary config files in {regime_configs_dir}")
            shutil.rmtree(regime_configs_dir)


if __name__ == "__main__":
    main()
