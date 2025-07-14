#!/usr/bin/env python3
"""
Get EIIE allocation results from trained models
Simple approach using the existing training infrastructure
"""
import os
import sys
import pandas as pd
import torch
from pathlib import Path

# Add TradeMaster to path
ROOT = Path(__file__).parent
sys.path.append(str(ROOT))

from tools.portfolio_management.test_eiie import main as test_eiie

def run_eiie_test(config_path):
    """Run EIIE test and capture results"""
    print(f"Running EIIE test for: {config_path}")
    
    # Store original argv
    original_argv = sys.argv.copy()
    
    try:
        # Set up arguments for test_eiie
        sys.argv = ['test_eiie.py', '--config', config_path, '--task_name', 'test']
        
        # Run the test
        test_eiie()
        
    except Exception as e:
        print(f"Error running test: {e}")
        
    finally:
        # Restore original argv
        sys.argv = original_argv

def main():
    """Get EIIE results for all regimes"""
    
    # EIIE model configurations  
    configs = [
        "configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_covid.py",
        "configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war.py", 
        "configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i.py"
    ]
    
    print("=" * 80)
    print("🚀 Getting EIIE Results from Trained Models")
    print("=" * 80)
    
    for config_path in configs:
        if os.path.exists(config_path):
            run_eiie_test(config_path)
        else:
            print(f"❌ Config not found: {config_path}")
    
    print("=" * 80)
    print("✅ EIIE Tests Complete!")
    print("Check work_dir/ for test_result.csv files")
    
    # Find and display results
    import glob
    result_files = glob.glob("work_dir/*/test_result.csv")
    if result_files:
        print(f"\n📊 Found {len(result_files)} result files:")
        for f in result_files:
            print(f"  {f}")
    else:
        print("\n❌ No test result files found")

if __name__ == "__main__":
    main() 