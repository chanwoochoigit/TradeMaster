#!/usr/bin/env python3
"""
Comprehensive Mask SAC MCAD Pipeline
====================================

This script provides a complete end-to-end pipeline for:
1. Setting up MCAD dataset configurations
2. Running Mask SAC training on all 3 regimes (COVID, Trade War, Trade War I)
3. Monitoring training progress
4. Exporting allocation histories to CSV
5. Generating summary reports

Usage:
    python mask_sac_mcad_complete_pipeline.py [--quick-test]

Options:
    --quick-test: Run with reduced episodes for testing (50 instead of 1000)
"""

import os
import sys
import argparse
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.regime_dates import REGIME_DATES, get_regime_dates, get_all_regimes


class MaskSACMCADPipeline:
    """Complete pipeline for Mask SAC MCAD training and evaluation"""

    def __init__(self, quick_test: bool = False):
        self.quick_test = quick_test
        self.regimes = get_all_regimes()
        self.root_dir = Path(__file__).parent
        self.workdir = "workdir"
        self.data_dir = "data/portfolio_management/mcad"

        # Training parameters
        self.num_episodes = 50 if quick_test else 200
        self.max_training_hours = 1 if quick_test else 6

        # Results tracking
        self.results = {"setup": {}, "training": {}, "export": {}, "summary": {}}

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "PROGRESS": "🔄",
        }.get(level, "📝")

        print(f"[{timestamp}] {prefix} {message}")

    def run_command(
        self,
        command: str,
        description: str,
        background: bool = False,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Execute shell command with logging"""
        self.log(f"Running: {description}", "PROGRESS")
        self.log(f"Command: {command}", "INFO")

        try:
            if background:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                return process
            else:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=timeout
                )

                if result.returncode == 0:
                    self.log(f"{description} completed successfully", "SUCCESS")
                else:
                    self.log(
                        f"{description} failed (exit code: {result.returncode})",
                        "ERROR",
                    )
                    if result.stderr:
                        self.log(f"Error output: {result.stderr[-500:]}", "ERROR")

                return result

        except subprocess.TimeoutExpired:
            self.log(f"{description} timed out after {timeout} seconds", "WARNING")
            return None
        except Exception as e:
            self.log(f"{description} failed with exception: {str(e)}", "ERROR")
            return None

    def setup_data_structure(self) -> bool:
        """Prepare MCAD data in the required format"""
        self.log("Setting up MCAD data structure", "PROGRESS")

        for regime in self.regimes:
            regime_path = f"{self.data_dir}/{regime}"

            # Check if data files exist
            required_files = ["train.csv", "valid.csv", "test.csv"]
            missing_files = []

            for file in required_files:
                if not os.path.exists(f"{regime_path}/{file}"):
                    missing_files.append(file)

            if missing_files:
                self.log(f"Missing data files for {regime}: {missing_files}", "ERROR")
                self.results["setup"][regime] = False
                continue

            try:
                # Prepare individual stock files
                self.log(f"Preparing data structure for {regime} regime", "PROGRESS")

                # Read data
                train_df = pd.read_csv(f"{regime_path}/train.csv")
                valid_df = pd.read_csv(f"{regime_path}/valid.csv")
                test_df = pd.read_csv(f"{regime_path}/test.csv")

                # Combine all data
                combined_df = pd.concat(
                    [train_df, valid_df, test_df], ignore_index=True
                )
                stocks = sorted(combined_df["tic"].unique())

                self.log(f"Found {len(stocks)} stocks for {regime}: {stocks}", "INFO")

                # Create stocks.txt
                with open(f"{regime_path}/stocks.txt", "w") as f:
                    for stock in stocks:
                        f.write(f"{stock}\n")

                # Create individual stock CSV files
                for stock in stocks:
                    stock_data = combined_df[combined_df["tic"] == stock].copy()
                    stock_data = stock_data.sort_values("date")

                    # Select feature columns
                    feature_cols = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjcp",
                        "volume",
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

                    # Add temporal features
                    stock_data["weekday"] = pd.to_datetime(
                        stock_data["date"]
                    ).dt.dayofweek
                    stock_data["day"] = pd.to_datetime(stock_data["date"]).dt.day
                    stock_data["month"] = pd.to_datetime(stock_data["date"]).dt.month

                    # Add labels
                    stock_data["ret1"] = stock_data["close"].pct_change().fillna(0)
                    stock_data["mov1"] = (
                        stock_data["close"] > stock_data["close"].shift(1)
                    ).astype(int)

                    # Create final dataframe
                    final_cols = (
                        ["date"]
                        + feature_cols
                        + ["weekday", "day", "month", "ret1", "mov1"]
                    )
                    stock_df = stock_data[final_cols].copy()
                    stock_df = stock_df.set_index("date")
                    stock_df.index.name = "Date"

                    # Save stock file
                    stock_df.to_csv(f"{regime_path}/{stock}.csv")

                # Create aux_stocks directory and file
                aux_dir = f"{regime_path}/aux_stocks_files"
                os.makedirs(aux_dir, exist_ok=True)

                with open(f"{aux_dir}/1_all.txt", "w") as f:
                    for stock in stocks:
                        f.write(f"{stock}\n")

                self.results["setup"][regime] = True
                self.log(f"Data structure setup completed for {regime}", "SUCCESS")

            except Exception as e:
                self.log(f"Failed to setup data for {regime}: {str(e)}", "ERROR")
                self.results["setup"][regime] = False

        success_count = sum(self.results["setup"].values())
        self.log(
            f"Data setup completed: {success_count}/{len(self.regimes)} regimes successful",
            "SUCCESS" if success_count == len(self.regimes) else "WARNING",
        )

        return success_count == len(self.regimes)

    def create_configs(self) -> bool:
        """Create configuration files for all regimes"""
        self.log("Creating configuration files", "PROGRESS")

        # Use shared regime date configurations

        config_template = """# base parameters (do not modify)
root = None
workdir = "workdir"
tag = "mask_sac_mcad_{regime}"
num_stocks = 5  # MCAD has 5 stocks: spy, qqq, dbc, agg, gld
num_envs = 1
num_features = 20  # MCAD features (17 tech indicators + 3 temporals)
temporal_dim = 3 # weekday, day, month
train_start_date = "{train_start_date}"
val_start_date = "{val_start_date}"
test_start_date = "{test_start_date}"
test_end_date = None
if_use_per = False
if_use_rep = True
if_use_beta = True
if_norm = True
if_norm_temporal = False
save_freq = 20
repeat_times = 128
action_wrapper_method = "reweight"
T = 1.0
n_steps_per_episode = 1024

# train parameters (adjust mainly)
num_episodes = {num_episodes}
days = 10
batch_size = 64
buffer_size = 4096
horizon_len = 64
embed_dim = 128
decoder_embed_dim = 128
depth = 1 # 1 transformer
decoder_depth = 1
lr = 5e-5 # act_lr, cri_lr
act_lr = 5e-5
cri_lr = 5e-5
rep_lr = 5e-5
beta_lr = 5e-5
rep_loss_weight = 0.01
beta_loss_weight = 0.01
seed = 10

# size
feature_size = (days, num_features)
patch_size = (days, num_features)

transition = ["state", "action", "mask", "ids_restore", "reward", "done", "next_state"]
transition_shape = dict(
    state=dict(shape=(num_envs, num_stocks, days, num_features), type="float32"),
    action=dict(shape=(num_envs, num_stocks + 1), type="float32"),
    mask=dict(shape=(num_envs, num_stocks), type="int32"),
    ids_restore=dict(shape=(num_envs, num_stocks), type="int64"),
    reward=dict(shape=(num_envs,), type="float32"),
    done=dict(shape=(num_envs,), type="float32"),
    next_state=dict(shape=(num_envs, num_stocks, days, num_features), type="float32"),
)

dataset = dict(
    type="PortfolioManagementDataset",
    root=root,
    data_path="data/portfolio_management/mcad/{regime}",
    stocks_path="data/portfolio_management/mcad/{regime}/stocks.txt",
    aux_stocks_path="data/portfolio_management/mcad/{regime}/aux_stocks_files",
    features_name=[
        "open",
        "high",
        "low",
        "close",
        "adjcp",
        "volume",
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
    ],
    temporals_name=[
        "weekday",
        "day",
        "month",
    ],
    labels_name=[
        "ret1",
        "mov1",
    ],
)

environment = dict(
    type="EnvironmentPV",
    dataset=None,
    mode="train",
    if_norm=if_norm,
    if_norm_temporal=if_norm_temporal,
    scaler=None,
    days=days,
    start_date=None,
    end_date=None,
    initial_amount=1e3,
    transaction_cost_pct=1e-3,
)

rep_net = dict(
    type="MaskTimeState",
    embed_type="TimesEmbed",
    feature_size=feature_size,
    patch_size=patch_size,
    t_patch_size=1,
    num_stocks=num_stocks,
    pred_num_stocks=num_stocks,
    in_chans=1,
    input_dim=num_features,
    temporal_dim=temporal_dim,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=4,
    decoder_embed_dim=decoder_embed_dim,
    decoder_depth=decoder_depth,
    decoder_num_heads=8,
    mlp_ratio=4.0,
    norm_pix_loss=False,
    cls_embed=True,
    sep_pos_embed=True,
    trunc_init=False,
    no_qkv_bias=False,
    mask_ratio_min=0.6,
    mask_ratio_max=0.8,
    mask_ratio_mu=0.7,
    mask_ratio_std=0.1,
)

act_net = dict(
    type="ActorMaskSAC",
    embed_dim=decoder_embed_dim,
    depth=depth,
    cls_embed=True,
)

cri_net = dict(
    type="CriticMaskSAC",
    embed_dim=decoder_embed_dim,
    depth=depth,
    cls_embed=True,
)

criterion = dict(type="MSELoss", reduction="none")
scheduler = dict(
    type="MultiStepLRScheduler",
    multi_steps=[
        120 * n_steps_per_episode,
        200 * n_steps_per_episode,
        280 * n_steps_per_episode,
    ],
    t_initial=num_episodes * n_steps_per_episode,
    decay_t=500 * n_steps_per_episode,
    gamma=0.1,
    t_mul=1.0,
    lr_min=0.0,
    decay_rate=1.0,
    warmup_t=60 * n_steps_per_episode,
    warmup_lr_init=1e-8,
    warmup_prefix=False,
    cycle_limit=0,
    t_in_epochs=False,
    noise_range_t=None,
    noise_pct=0.67,
    noise_std=1.0,
    noise_seed=42,
    initialize=True,
)
optimizer = dict(type="AdamW", params=None, lr=lr)

agent = dict(
    type="AgentMaskSAC",
    act_lr=act_lr,
    cri_lr=cri_lr,
    rep_lr=rep_lr,
    beta_lr=beta_lr,
    rep_net=rep_net,
    act_net=act_net,
    cri_net=cri_net,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    if_use_per=if_use_per,
    if_use_rep=if_use_rep,
    if_use_beta=if_use_beta,
    rep_loss_weight=rep_loss_weight,
    beta_loss_weight=beta_loss_weight,
    num_envs=num_envs,
    transition_shape=transition_shape,
    max_step=1e4,
    gamma=0.99,
    reward_scale=2**0,
    repeat_times=repeat_times,
    batch_size=batch_size,
    clip_grad_norm=3.0,
    soft_update_tau=5e-3,
    state_value_tau=0,
    device=None,
    action_wrapper_method=action_wrapper_method,
    T=T,
)
"""

        try:
            # Create configs directory if it doesn't exist
            config_dir = "configs/earnmore"
            os.makedirs(config_dir, exist_ok=True)

            for regime in self.regimes:
                dates = get_regime_dates(regime)
                config_content = config_template.format(
                    regime=regime,
                    num_episodes=self.num_episodes,
                    train_start_date=dates["train_start"],
                    val_start_date=dates["val_start"],
                    test_start_date=dates["test_start"],
                )

                config_path = f"{config_dir}/mask_sac_mcad_{regime}.py"
                with open(config_path, "w") as f:
                    f.write(config_content)

                self.log(f"Created config file: {config_path}", "SUCCESS")

            return True

        except Exception as e:
            self.log(f"Failed to create config files: {str(e)}", "ERROR")
            return False

    def train_regime(self, regime: str) -> bool:
        """Train a specific regime with real-time log monitoring"""
        config_path = f"configs/earnmore/mask_sac_mcad_{regime}.py"
        tag = f"mask_sac_mcad_{regime}"

        # Check if already trained
        checkpoint_path = f"{self.workdir}/{tag}/best.pth"
        if os.path.exists(checkpoint_path):
            self.log(f"Training already completed for {regime}, skipping...", "SUCCESS")
            return True

        self.log(f"Starting training for {regime} regime", "PROGRESS")

        # Start training
        train_command = f"python tools/earnmore/train.py --config {config_path}"

        try:
            # Start process with real-time output
            process = subprocess.Popen(
                train_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            self.log(
                f"Training process started for {regime} (PID: {process.pid})", "SUCCESS"
            )

            # Monitor training with real-time logs
            max_wait_seconds = self.max_training_hours * 3600
            start_time = time.time()
            last_log_time = start_time
            log_check_interval = 30  # Check for new logs every 30 seconds

            while True:
                elapsed = time.time() - start_time

                # Check if training completed
                if os.path.exists(checkpoint_path):
                    self.log(f"Training completed for {regime}!", "SUCCESS")
                    if process.poll() is None:
                        process.terminate()
                    return True

                # Check if process finished
                if process.poll() is not None:
                    # Process finished, get final output
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        self.log(f"Final output from {regime}:", "INFO")
                        for line in remaining_output.strip().split("\n"):
                            if line.strip():
                                self.log(f"  {line}", "INFO")

                    if process.returncode == 0:
                        self.log(
                            f"Training process finished successfully for {regime}",
                            "SUCCESS",
                        )
                        return os.path.exists(checkpoint_path)
                    else:
                        self.log(
                            f"Training process failed for {regime} (exit code: {process.returncode})",
                            "ERROR",
                        )
                        return False

                # Show real-time output
                current_time = time.time()
                if current_time - last_log_time >= log_check_interval:
                    # Try to read available output without blocking
                    try:
                        # Read available lines
                        output_lines = []
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            output_lines.append(line.strip())
                            if len(output_lines) >= 10:  # Limit to last 10 lines
                                break

                        if output_lines:
                            # Try to estimate completion time based on episode progress
                            time_estimate = ""
                            try:
                                log_file_path = f"{self.workdir}/{tag}/train_infos.txt"
                                if os.path.exists(log_file_path):
                                    with open(log_file_path, "r") as f:
                                        info_lines = f.readlines()
                                    if info_lines:
                                        last_info = json.loads(info_lines[-1].strip())
                                        if "episode" in last_info:
                                            episode_num = (
                                                last_info["episode"][0]
                                                if isinstance(
                                                    last_info["episode"], list
                                                )
                                                else last_info["episode"]
                                            )
                                            if episode_num > 0:
                                                avg_time_per_episode = (
                                                    elapsed / episode_num
                                                )
                                                remaining_episodes = (
                                                    self.num_episodes - episode_num
                                                )
                                                eta_seconds = (
                                                    remaining_episodes
                                                    * avg_time_per_episode
                                                )
                                                time_estimate = f", ETA: {eta_seconds//3600:.0f}h {(eta_seconds%3600)//60:.0f}m"
                            except:
                                pass

                            self.log(
                                f"Training progress for {regime} ({elapsed//60:.0f} min elapsed{time_estimate}):",
                                "PROGRESS",
                            )
                            for line in output_lines[-5:]:  # Show last 5 lines
                                if line.strip():
                                    self.log(f"  {line}", "INFO")
                        else:
                            self.log(
                                f"Training {regime} in progress... ({elapsed//60:.0f} min elapsed)",
                                "PROGRESS",
                            )

                    except:
                        self.log(
                            f"Training {regime} in progress... ({elapsed//60:.0f} min elapsed)",
                            "PROGRESS",
                        )

                    # Check training log file if it exists
                    log_file_path = f"{self.workdir}/{tag}/train_log.txt"
                    if os.path.exists(log_file_path):
                        try:
                            # Show last few lines from training log
                            with open(log_file_path, "r") as f:
                                lines = f.readlines()
                                if lines:
                                    self.log(
                                        f"Recent training metrics for {regime}:", "INFO"
                                    )
                                    for line in lines[-3:]:  # Last 3 lines
                                        if line.strip():
                                            self.log(f"  {line.strip()}", "INFO")
                        except:
                            pass

                    last_log_time = current_time

                # Check timeout
                if elapsed >= max_wait_seconds:
                    self.log(
                        f"Training timeout for {regime} after {self.max_training_hours} hours",
                        "WARNING",
                    )
                    self.log("Showing final training status...", "INFO")

                    # Show final logs before terminating
                    log_file_path = f"{self.workdir}/{tag}/train_log.txt"
                    if os.path.exists(log_file_path):
                        try:
                            with open(log_file_path, "r") as f:
                                lines = f.readlines()
                                if lines:
                                    self.log(
                                        f"Final training status for {regime}:",
                                        "WARNING",
                                    )
                                    for line in lines[-10:]:  # Last 10 lines
                                        if line.strip():
                                            self.log(f"  {line.strip()}", "INFO")
                        except:
                            pass

                    # Show tensorboard files if they exist
                    tb_dir = f"{self.workdir}/{tag}"
                    if os.path.exists(tb_dir):
                        tb_files = [
                            f
                            for f in os.listdir(tb_dir)
                            if f.startswith("events.out.tfevents")
                        ]
                        if tb_files:
                            self.log(f"Tensorboard logs available in: {tb_dir}", "INFO")

                    process.terminate()
                    return False

                time.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            self.log(f"Training interrupted for {regime}", "WARNING")
            if "process" in locals() and process.poll() is None:
                process.terminate()
            return False
        except Exception as e:
            self.log(f"Training failed for {regime} with exception: {str(e)}", "ERROR")
            return False

    def export_allocation_history(self, regime: str) -> bool:
        """Export allocation history for a trained regime"""
        config_path = f"configs/earnmore/mask_sac_mcad_{regime}.py"

        self.log(f"Exporting allocation history for {regime}", "PROGRESS")

        export_command = f"python tools/export_allocations.py --config {config_path}"
        result = self.run_command(
            export_command, f"Exporting allocation history for {regime}"
        )

        if result and result.returncode == 0:
            # Check if CSV file was created
            tag = f"mask_sac_mcad_{regime}"
            csv_path = f"{self.workdir}/{tag}/mask_sac_mcad_{regime}.csv"

            if os.path.exists(csv_path):
                # Get file info
                file_size = os.path.getsize(csv_path)
                df = pd.read_csv(csv_path)

                self.log(f"Allocation history exported: {csv_path}", "SUCCESS")
                self.log(f"  - File size: {file_size/1024:.1f} KB", "INFO")
                self.log(f"  - Records: {len(df)}", "INFO")
                self.log(f"  - Columns: {list(df.columns)}", "INFO")

                return True
            else:
                self.log(f"CSV file not found after export: {csv_path}", "ERROR")
                return False
        else:
            self.log(f"Failed to export allocation history for {regime}", "ERROR")
            return False

    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report"""
        self.log("Generating summary report", "PROGRESS")

        # Collect results
        setup_success = sum(self.results["setup"].values())
        training_success = sum(self.results["training"].values())
        export_success = sum(self.results["export"].values())
        total_regimes = len(self.regimes)

        # Create summary
        summary = {
            "pipeline_run": {
                "timestamp": datetime.now().isoformat(),
                "quick_test_mode": self.quick_test,
                "num_episodes": self.num_episodes,
                "max_training_hours": self.max_training_hours,
            },
            "results": {
                "setup": f"{setup_success}/{total_regimes}",
                "training": f"{training_success}/{total_regimes}",
                "export": f"{export_success}/{total_regimes}",
                "overall": f"{min(setup_success, training_success, export_success)}/{total_regimes}",
            },
            "regimes": {},
        }

        # Add regime-specific details
        for regime in self.regimes:
            summary["regimes"][regime] = {
                "setup": self.results["setup"].get(regime, False),
                "training": self.results["training"].get(regime, False),
                "export": self.results["export"].get(regime, False),
            }

        # Save summary to JSON
        summary_path = f"{self.workdir}/pipeline_summary.json"
        os.makedirs(self.workdir, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        self.log("=" * 60, "INFO")
        self.log("PIPELINE SUMMARY", "SUCCESS")
        self.log("=" * 60, "INFO")

        self.log(f"Setup Success: {setup_success}/{total_regimes}", "INFO")
        self.log(f"Training Success: {training_success}/{total_regimes}", "INFO")
        self.log(f"Export Success: {export_success}/{total_regimes}", "INFO")

        self.log("\nRegime Details:", "INFO")
        for regime in self.regimes:
            setup_status = "✅" if self.results["setup"].get(regime, False) else "❌"
            training_status = (
                "✅" if self.results["training"].get(regime, False) else "❌"
            )
            export_status = "✅" if self.results["export"].get(regime, False) else "❌"

            self.log(
                f"  {regime:<15}: Setup {setup_status} | Training {training_status} | Export {export_status}",
                "INFO",
            )

        # Show output files
        if export_success > 0:
            self.log("\nGenerated Files:", "INFO")
            for regime in self.regimes:
                if self.results["export"].get(regime, False):
                    tag = f"mask_sac_mcad_{regime}"
                    csv_path = f"{self.workdir}/{tag}/mask_sac_mcad_{regime}.csv"
                    if os.path.exists(csv_path):
                        self.log(f"  📊 {regime}: {csv_path}", "INFO")

        self.log(f"\n📄 Summary report saved: {summary_path}", "SUCCESS")

        overall_success = min(setup_success, training_success, export_success)
        if overall_success == total_regimes:
            self.log("🎉 All regimes completed successfully!", "SUCCESS")
        else:
            self.log(f"⚠️ {total_regimes - overall_success} regimes failed", "WARNING")

    def run_pipeline(self) -> bool:
        """Run the complete pipeline"""
        start_time = time.time()

        self.log("🚀 Starting Mask SAC MCAD Complete Pipeline", "SUCCESS")
        self.log(
            f"Mode: {'Quick Test' if self.quick_test else 'Full Training'}", "INFO"
        )
        self.log(f"Regimes: {', '.join(self.regimes)}", "INFO")
        self.log(f"Episodes per regime: {self.num_episodes}", "INFO")

        try:
            # Step 1: Setup data structure
            self.log("\n" + "=" * 60, "INFO")
            self.log("STEP 1: Data Structure Setup", "PROGRESS")
            self.log("=" * 60, "INFO")

            if not self.setup_data_structure():
                self.log("Data setup failed, aborting pipeline", "ERROR")
                return False

            # Step 2: Create configurations
            self.log("\n" + "=" * 60, "INFO")
            self.log("STEP 2: Configuration Creation", "PROGRESS")
            self.log("=" * 60, "INFO")

            if not self.create_configs():
                self.log("Config creation failed, aborting pipeline", "ERROR")
                return False

            # Step 3: Training
            self.log("\n" + "=" * 60, "INFO")
            self.log("STEP 3: Model Training", "PROGRESS")
            self.log("=" * 60, "INFO")

            for regime in self.regimes:
                success = self.train_regime(regime)
                self.results["training"][regime] = success

                if not success:
                    self.log(
                        f"Training failed for {regime}, continuing with next regime",
                        "WARNING",
                    )

            # Step 4: Export allocation histories
            self.log("\n" + "=" * 60, "INFO")
            self.log("STEP 4: Allocation History Export", "PROGRESS")
            self.log("=" * 60, "INFO")

            for regime in self.regimes:
                if self.results["training"].get(regime, False):
                    success = self.export_allocation_history(regime)
                    self.results["export"][regime] = success
                else:
                    self.log(
                        f"Skipping export for {regime} - training not completed",
                        "WARNING",
                    )
                    self.results["export"][regime] = False

            # Step 5: Generate summary
            self.log("\n" + "=" * 60, "INFO")
            self.log("STEP 5: Summary Generation", "PROGRESS")
            self.log("=" * 60, "INFO")

            self.generate_summary_report()

            # Final timing
            elapsed_time = time.time() - start_time
            self.log(
                f"\n⏱️ Total pipeline time: {elapsed_time/3600:.2f} hours", "SUCCESS"
            )

            # Determine overall success
            successful_exports = sum(self.results["export"].values())
            overall_success = successful_exports == len(self.regimes)

            if overall_success:
                self.log(
                    "🎉 Pipeline completed successfully for all regimes!", "SUCCESS"
                )
            else:
                self.log(
                    f"⚠️ Pipeline completed with {successful_exports}/{len(self.regimes)} successful regimes",
                    "WARNING",
                )

            return overall_success

        except KeyboardInterrupt:
            self.log("Pipeline interrupted by user", "WARNING")
            return False
        except Exception as e:
            self.log(f"Pipeline failed with exception: {str(e)}", "ERROR")
            return False

    def check_training_status(self, regime: str = None) -> None:
        """Check current training status and show recent logs"""
        if regime:
            regimes_to_check = [regime]
        else:
            regimes_to_check = self.regimes

        self.log("=" * 60, "INFO")
        self.log("TRAINING STATUS CHECK", "INFO")
        self.log("=" * 60, "INFO")

        for reg in regimes_to_check:
            tag = f"mask_sac_mcad_{reg}"
            workdir_path = f"{self.workdir}/{tag}"

            self.log(f"\n📊 Checking {reg.upper()} regime:", "INFO")

            # Check if training directory exists
            if not os.path.exists(workdir_path):
                self.log(f"  ❌ Training not started (no workdir)", "WARNING")
                continue

            # Check for checkpoint
            checkpoint_path = f"{workdir_path}/best.pth"
            if os.path.exists(checkpoint_path):
                checkpoint_size = os.path.getsize(checkpoint_path)
                self.log(
                    f"  ✅ Training completed (checkpoint: {checkpoint_size/1024/1024:.1f} MB)",
                    "SUCCESS",
                )
            else:
                self.log(f"  🔄 Training in progress (no checkpoint yet)", "PROGRESS")

            # Check training log
            log_file_path = f"{workdir_path}/train_log.txt"
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, "r") as f:
                        lines = f.readlines()

                    if lines:
                        # Get file info
                        log_size = os.path.getsize(log_file_path)
                        self.log(
                            f"  📝 Training log: {len(lines)} lines ({log_size/1024:.1f} KB)",
                            "INFO",
                        )

                        # Show recent lines
                        self.log(f"  📈 Recent training metrics:", "INFO")
                        for line in lines[-5:]:  # Last 5 lines
                            if line.strip():
                                self.log(f"    {line.strip()}", "INFO")
                    else:
                        self.log(f"  📝 Training log exists but empty", "WARNING")

                except Exception as e:
                    self.log(f"  ❌ Error reading training log: {str(e)}", "ERROR")
            else:
                self.log(f"  📝 No training log found", "WARNING")

            # Check training info log
            info_file_path = f"{workdir_path}/train_infos.txt"
            if os.path.exists(info_file_path):
                try:
                    with open(info_file_path, "r") as f:
                        lines = f.readlines()

                    if lines:
                        info_size = os.path.getsize(info_file_path)
                        self.log(
                            f"  📊 Training info: {len(lines)} entries ({info_size/1024:.1f} KB)",
                            "INFO",
                        )

                        # Try to parse the last entry
                        try:
                            last_info = json.loads(lines[-1].strip())
                            if "episode" in last_info:
                                episode_num = (
                                    last_info["episode"][0]
                                    if isinstance(last_info["episode"], list)
                                    else last_info["episode"]
                                )
                                progress_pct = (episode_num / self.num_episodes) * 100
                                self.log(
                                    f"    Episode: {episode_num}/{self.num_episodes} ({progress_pct:.1f}%)",
                                    "INFO",
                                )
                        except:
                            pass

                except Exception as e:
                    self.log(f"  ❌ Error reading training info: {str(e)}", "ERROR")

            # Check tensorboard logs
            tb_files = [
                f
                for f in os.listdir(workdir_path)
                if f.startswith("events.out.tfevents")
            ]
            if tb_files:
                total_tb_size = sum(
                    os.path.getsize(f"{workdir_path}/{f}") for f in tb_files
                )
                self.log(
                    f"  📈 Tensorboard: {len(tb_files)} files ({total_tb_size/1024:.1f} KB)",
                    "INFO",
                )
            else:
                self.log(f"  📈 No tensorboard logs", "WARNING")

            # Check for process
            try:
                result = subprocess.run(
                    f"ps aux | grep 'train.py.*{reg}' | grep -v grep",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    self.log(f"  🔄 Training process active", "SUCCESS")
                else:
                    self.log(f"  ⏸️ No active training process", "WARNING")
            except:
                pass

    def tail_training_logs(self, regime: str, lines: int = 20) -> None:
        """Show recent training logs for a specific regime"""
        tag = f"mask_sac_mcad_{regime}"
        log_file_path = f"{self.workdir}/{tag}/train_log.txt"

        self.log(f"📖 Showing last {lines} lines from {regime} training log:", "INFO")
        self.log("=" * 60, "INFO")

        if not os.path.exists(log_file_path):
            self.log(f"❌ Training log not found: {log_file_path}", "ERROR")
            return

        try:
            with open(log_file_path, "r") as f:
                all_lines = f.readlines()

            if not all_lines:
                self.log("📝 Training log is empty", "WARNING")
                return

            # Show last N lines
            recent_lines = all_lines[-lines:]
            for i, line in enumerate(recent_lines):
                line_num = len(all_lines) - len(recent_lines) + i + 1
                self.log(f"{line_num:4d}: {line.rstrip()}", "INFO")

        except Exception as e:
            self.log(f"❌ Error reading training log: {str(e)}", "ERROR")


def main():
    parser = argparse.ArgumentParser(description="Mask SAC MCAD Complete Pipeline")
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run in quick test mode (50 episodes instead of 1000)",
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = MaskSACMCADPipeline(quick_test=args.quick_test)
    success = pipeline.run_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
