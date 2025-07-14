#!/usr/bin/env python3
"""
Export EIIE results from trained TradeMaster models
"""
import sys
import os
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from datetime import datetime

# Add TradeMaster to path
ROOT = str(Path(__file__).resolve().parent)
sys.path.append(ROOT)

from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.datasets.builder import build_dataset
from trademaster.environments.builder import build_environment
from trademaster.agents.builder import build_agent
from trademaster.nets.builder import build_net

def export_eiie_model(config_path, model_path, output_path):
    """Export results from a single EIIE model"""
    print(f"  Loading config: {config_path}")
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    
    print(f"  Building dataset...")
    dataset = build_dataset(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build test environment
    test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task="test"))
    
    action_dim = test_environment.action_dim
    state_dim = test_environment.state_dim
    input_dim = len(test_environment.tech_indicator_list)
    time_steps = test_environment.time_steps

    cfg.act.update(dict(input_dim=input_dim, time_steps=time_steps))
    cfg.cri.update(dict(input_dim=input_dim, action_dim=action_dim, time_steps=time_steps))

    act = build_net(cfg.act)
    cri = build_net(cfg.cri)
    
    # Load trained model
    print(f"  Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'act' in checkpoint:
        act.load_state_dict(checkpoint['act'])
    else:
        act.load_state_dict(checkpoint)
    
    act.to(device)
    act.eval()
    
    # Run inference
    print(f"  Running inference...")
    state = test_environment.reset()
    states = []
    actions = []
    rewards = []
    portfolio_values = []
    
    done = False
    total_steps = 0
    
    with torch.no_grad():
        while not done and total_steps < 10000:  # Safety limit
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from model
            action = act(state_tensor)
            if hasattr(action, 'detach'):
                action = action.detach().cpu().numpy().flatten()
            
            # Step environment
            next_state, reward, done, info = test_environment.step(action)
            
            # Store results
            states.append(state.copy() if hasattr(state, 'copy') else state)
            actions.append(action.copy())
            rewards.append(reward)
            portfolio_values.append(info.get('total_asset', test_environment.portfolio_value))
            
            state = next_state
            total_steps += 1
    
    # Create results DataFrame
    print(f"  Creating results DataFrame with {len(actions)} steps...")
    
    # Asset names (SPY, QQQ, DBC, AGG, GLD)
    asset_names = ['SPY', 'QQQ', 'DBC', 'AGG', 'GLD']
    
    results_data = []
    for i, (action, reward, portfolio_value) in enumerate(zip(actions, rewards, portfolio_values)):
        row_data = {
            'step': i,
            'portfolio_value': portfolio_value,
            'reward': reward
        }
        
        # Add allocation percentages
        if len(action) >= 5:
            for j, asset in enumerate(asset_names):
                row_data[asset.lower()] = action[j] if j < len(action) else 0.0
        
        # Add cash (1 - sum of other allocations)
        total_allocation = sum(row_data.get(asset.lower(), 0) for asset in asset_names)
        row_data['cash'] = max(0, 1.0 - total_allocation)
        
        results_data.append(row_data)
    
    df = pd.DataFrame(results_data)
    
    # Save results
    print(f"  Saving to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df

def main():
    """Export all trained EIIE models"""
    
    models = {
        'covid': {
            'config': 'configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_covid.py',
            'model': 'work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_covid/checkpoints/best.pth'
        },
        'trade_war': {
            'config': 'configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war.py',
            'model': 'work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war/checkpoints/best.pth'
        },
        'trade_war_i': {
            'config': 'configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i.py',
            'model': 'work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i/checkpoints/best.pth'
        }
    }
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"eiie_results_{timestamp}"
    
    print(f"🚀 Exporting EIIE results to: {output_dir}")
    print("=" * 60)
    
    results_summary = {}
    
    for regime, paths in models.items():
        print(f"\n📊 Processing {regime.replace('_', ' ').title()} regime...")
        
        config_path = paths['config']
        model_path = paths['model']
        output_path = f"{output_dir}/{regime}/eiie_test_results.csv"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"  ❌ Model not found: {model_path}")
            results_summary[regime] = "failed - model not found"
            continue
            
        try:
            df = export_eiie_model(config_path, model_path, output_path)
            print(f"  ✅ Success! Exported {len(df)} steps")
            results_summary[regime] = f"success - {len(df)} steps"
            
            # Print summary statistics
            final_portfolio_value = df['portfolio_value'].iloc[-1]
            total_return = (final_portfolio_value / df['portfolio_value'].iloc[0] - 1) * 100
            print(f"     Final portfolio value: ${final_portfolio_value:,.2f}")
            print(f"     Total return: {total_return:.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results_summary[regime] = f"failed - {str(e)}"
    
    # Save summary
    summary_path = f"{output_dir}/export_summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump({
            'timestamp': timestamp,
            'results': results_summary
        }, f, indent=2)
    
    print(f"\n✅ Export completed!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📋 Summary: {summary_path}")
    
    # Print final summary
    print(f"\n📊 Summary:")
    for regime, status in results_summary.items():
        print(f"  {regime}: {status}")

if __name__ == "__main__":
    main() 