import numpy as np
import pandas as pd
import torch
import sys
import os
import gym
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

# Import from pm modules
from pm.registry import ENVIRONMENT, AGENT, DATASET
from pm.utils import update_data_root, load_checkpoint


def export_allocation_history_old(agent, envs, output_path):
    """
    Legacy interface: Export allocation history by running the agent through the environments

    Args:
        agent: The trained agent
        envs: Vector of environments
        output_path: Path to save the allocation history CSV

    Returns:
        pandas.DataFrame: DataFrame containing allocation history with dates
    """

    allocation_history = []
    date_history = []

    # Reset environments
    observations = envs.reset()

    # Convert to tensor if needed
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(
            observations, dtype=torch.float32, device=agent.device
        )

    done = False
    step = 0

    print("Exporting allocation history...")

    while not done:
        # Get actions from agent
        with torch.no_grad():
            actions = agent.select_action(observations)

        # Take step in environment
        next_observations, rewards, dones, infos = envs.step(actions.cpu().numpy())

        # Store allocation (actions represent portfolio weights)
        allocation_history.append(actions.cpu().numpy().copy())

        # Extract date from environment info if available
        if infos and len(infos) > 0 and "date" in infos[0]:
            date_history.append(infos[0]["date"])
        elif hasattr(envs.envs[0].env, "date"):
            # Get date from environment directly
            date_history.append(envs.envs[0].env.date)
        else:
            # Use step number as fallback
            date_history.append(f"step_{step}")

        # Update observations
        observations = torch.tensor(
            next_observations, dtype=torch.float32, device=agent.device
        )

        # Check if any environment is done
        done = any(dones)
        step += 1

        if step % 100 == 0:
            print(f"Processed {step} steps...")

    # Convert to DataFrame
    if allocation_history:
        # Stack all allocations
        all_allocations = np.vstack(allocation_history)

        # Create DataFrame with dates and allocations
        df_data = {"date": date_history}

        # Add allocation columns
        for i in range(all_allocations.shape[1]):
            df_data[f"allocation_{i}"] = all_allocations[:, i]

        df = pd.DataFrame(df_data)

        print(f"Exported {len(df)} allocation records")
        return df
    else:
        print("No allocation history to export")
        return pd.DataFrame()


def export_allocation_history(
    config_path, model_path, output_path, asset_names, dataset_mode="test"
):
    """
    New interface: Export allocation history from a trained model

    Args:
        config_path: Path to config file
        model_path: Path to trained model (.pth file)
        output_path: Path to save the allocation history CSV
        asset_names: List of asset names for column naming
        dataset_mode: 'train', 'val', or 'test' dataset to run inference on

    Returns:
        pandas.DataFrame: DataFrame containing allocation history with proper column names
    """
    try:
        # Get project root and add to path
        project_root = Path(config_path).parent.parent.parent
        sys.path.insert(0, str(project_root))

        # Load configuration
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", config_path)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)

        # Convert module attributes to config object
        cfg = type("Config", (), {})()
        for attr in dir(cfg_module):
            if not attr.startswith("_"):
                setattr(cfg, attr, getattr(cfg_module, attr))

        # Set root if not already set
        if not hasattr(cfg, "root") or cfg.root is None:
            cfg.root = str(project_root)

        # Update data root
        update_data_root(cfg, root=str(project_root))

        # Ensure dataset root is set correctly
        if hasattr(cfg, "data") and isinstance(cfg.data, dict):
            cfg.data["root"] = str(project_root)
        elif hasattr(cfg, "dataset") and isinstance(cfg.dataset, dict):
            cfg.dataset["root"] = str(project_root)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Building dataset...")
        # Handle both 'data' (standard) and 'dataset' (masked SAC) config structures
        if hasattr(cfg, 'data'):
            dataset = DATASET.build(cfg.data)
        elif hasattr(cfg, 'dataset'):
            dataset = DATASET.build(cfg.dataset)
        else:
            raise AttributeError("Config must have either 'data' or 'dataset' attribute")

        print("Building environments...")
        # Create training environment to get the scaler
        cfg.environment.update(
            dict(
                mode="train",
                if_norm=True,
                dataset=dataset,
                start_date=cfg.train_start_date,
                end_date=cfg.val_start_date,
            )
        )
        scaler_env = ENVIRONMENT.build(cfg.environment)
        scaler = scaler_env.scaler

        # Create inference environment for the specified dataset
        if dataset_mode == "train":
            start_date = cfg.train_start_date
            end_date = cfg.val_start_date
        elif dataset_mode == "val":
            start_date = cfg.val_start_date
            end_date = cfg.test_start_date
        else:  # test
            start_date = cfg.test_start_date
            end_date = cfg.test_end_date

        cfg.environment.update(
            dict(
                mode="val",  # Use val mode for deterministic inference
                if_norm=True,
                dataset=dataset,
                scaler=scaler,
                start_date=start_date,
                end_date=end_date,
            )
        )

        def make_env(env_id, env_params):
            def thunk():
                env = gym.make(env_id, **env_params)
                return env

            return thunk

        inference_environment = ENVIRONMENT.build(cfg.environment)
        inference_envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    "PortfolioManagement-v0",
                    env_params=dict(
                        env=deepcopy(inference_environment),
                        transition_shape=cfg.transition_shape,
                    ),
                )
                for i in range(len(inference_environment.aux_stocks))
            ]
        )

        print("Building agent...")
        cfg.agent.update(dict(device=device))
        agent = AGENT.build(cfg.agent)

        # Set up aux_stocks
        if hasattr(inference_envs.envs[0], "aux_stocks"):
            agent.aux_stocks = inference_envs.envs[0].aux_stocks

        print(f"Loading model from {model_path}...")
        episode = load_checkpoint(agent, model_path)
        print(f"Loaded model from episode {episode}")

        # Set agent to evaluation mode
        if hasattr(agent, "eval"):
            agent.eval()

        # Run inference and get allocation history
        print(f"Running inference on {dataset_mode} dataset...")
        allocation_history = []
        date_history = []

        # Reset environments
        observations = inference_envs.reset()

        # Convert to tensor for agent
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(
                observations, dtype=torch.float32, device=agent.device
            )

        done = False
        step = 0

        while not done:
            # Get actions from agent (portfolio weights)
            with torch.no_grad():
                # For Mask SAC, need to process through representation network
                if hasattr(agent, "rep"):
                    # Handle different observation shapes
                    if len(observations.shape) == 4:  # (e, n, d, f)
                        e, n, d, f = observations.shape
                        state = observations
                    elif len(observations.shape) == 5:  # (b, e, n, d, f)
                        b, e, n, d, f = observations.shape
                        state = observations.view(b * e, n, d, f)
                    else:
                        state = observations

                    rep_state, _, _ = agent.rep.forward_state(state)
                    actions = agent.forward_action(x=rep_state)
                else:
                    actions = agent.select_action(observations)

            # Take step in environment
            next_observations, rewards, dones, infos = inference_envs.step(
                actions.cpu().numpy()
            )

            # Check if done BEFORE storing allocation to avoid duplicates
            done = any(dones)
            
            # Only store allocation if not done
            if not done:
                # Store allocation (first environment only for single-asset-class case)
                allocation_history.append(actions.cpu().numpy()[0].copy())

                # Extract date from environment info
                if infos and len(infos) > 0 and "date" in infos[0]:
                    date_history.append(infos[0]["date"])
                else:
                    date_history.append(f"step_{step}")

            # Update observations
            observations = torch.tensor(
                next_observations, dtype=torch.float32, device=agent.device
            )

            step += 1

            if step % 100 == 0:
                print(f"Processed {step} steps...")

        # Create DataFrame with proper column names
        if allocation_history:
            all_allocations = np.vstack(allocation_history)

            # Create DataFrame
            df_data = {"date": date_history}

            # Add allocation columns with asset names (lowercase)
            # Layout: [cash, asset1, asset2, ..., assetN] but we want [asset1, asset2, ..., assetN, cash]
            for i, asset in enumerate(asset_names):
                if i + 1 < all_allocations.shape[1]:
                    df_data[asset.lower()] = all_allocations[:, i + 1]

            # Add cash as the last column (first column in allocations)
            df_data["cash"] = all_allocations[:, 0]

            df = pd.DataFrame(df_data)

            # Save to file
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if there is one
                os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_path, index=False)

            print(f"✅ Exported {len(df)} allocation records to {output_path}")
            return df
        else:
            print("❌ No allocation history to export")
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None
