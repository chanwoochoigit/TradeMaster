#!/usr/bin/env python3

import os
import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.append('.')
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.datasets.builder import build_dataset
from trademaster.environments.builder import build_environment
from trademaster.agents.builder import build_agent
from trademaster.trainers.builder import build_trainer

def test_eiie_model(config_path, regime_name):
    print(f"\n🚀 Testing EIIE model for {regime_name}")
    print(f"Config: {config_path}")
    
    try:
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Fix the dataset paths
        cfg.data.test_dynamic_path = f'data/portfolio_management/mcad/{regime_name}/test_with_label.csv'
        cfg.data.train_path = f'data/portfolio_management/mcad/{regime_name}/train.csv'
        cfg.data.valid_path = f'data/portfolio_management/mcad/{regime_name}/valid.csv'
        cfg.data.test_path = f'data/portfolio_management/mcad/{regime_name}/test.csv'
        
        # Fix environment type
        cfg.environment.type = 'PortfolioManagementEIIEEnvironment'
        
        cfg = replace_cfg_vals(cfg)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build components
        dataset = build_dataset(cfg)
        test_environment = build_environment(cfg, default_args=dict(dataset=dataset, task='test'))
        
        agent = build_agent(cfg, default_args=dict(
            action_space=test_environment.action_space,
            state_space=test_environment.state_space,
            device=device
        ))
        
        trainer = build_trainer(cfg, default_args=dict(
            dataset=dataset,
            test_environment=test_environment,
            agent=agent,
            device=device
        ))
        
        # Load trained model
        model_path = f'work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_{regime_name}/checkpoints/best.pth'
        trainer.agent.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded from {model_path}")
        
        # Run test
        trainer.test()
        print(f"✅ Test completed for {regime_name}")
        
        # Get allocation results
        allocations = trainer.test_environment.save_asset_memory()
        
        # Save results
        output_path = f'eiie_final_results/eiie_{regime_name}_allocations.csv'
        allocations.to_csv(output_path, index=False)
        
        print(f"✅ Results saved to {output_path}")
        print(f"   Shape: {allocations.shape}")
        print(f"   Columns: {allocations.columns.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {regime_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    models_to_test = [
        ('configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_covid.py', 'covid'),
        ('configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war.py', 'trade_war'),
        ('configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i.py', 'trade_war_i'),
    ]
    
    print("🎯 Testing all trained EIIE models...")
    
    results = {}
    for config_path, regime_name in models_to_test:
        success = test_eiie_model(config_path, regime_name)
        results[regime_name] = success
    
    print("\n📊 Final Results Summary:")
    for regime_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {regime_name}: {status}")
    
    successful_tests = sum(results.values())
    print(f"\n🎉 {successful_tests}/{len(models_to_test)} models tested successfully!")
    
    if successful_tests > 0:
        print("\n📁 Results saved in eiie_final_results/ directory:")
        for file in os.listdir('eiie_final_results'):
            if file.endswith('.csv'):
                print(f"  - {file}")

if __name__ == "__main__":
    main() 