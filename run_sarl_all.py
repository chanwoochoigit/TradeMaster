#!/usr/bin/env python3
"""
Simple script to run SARL training for all three market regimes
"""
import subprocess
import sys
import os

def run_sarl_training():
    """Run SARL training for all three market regimes"""
    
    regimes = ['trade_war_i', 'covid', 'trade_war']
    
    for regime in regimes:
        print(f"\n{'='*50}")
        print(f"🚀 Starting SARL training for {regime}")
        print(f"{'='*50}")
        
        cmd = [
            'python', 'tools/portfolio_management/train_sarl.py',
            '--config', f'configs/portfolio_management/portfolio_management_mcad_sarl_sarl_adam_mse_{regime}.py',
            '--task_name', 'train',
            '--verbose', '1'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ SARL training completed successfully for {regime}")
        except subprocess.CalledProcessError as e:
            print(f"❌ SARL training failed for {regime}: {e}")
            print(f"Command: {' '.join(cmd)}")
            continue
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user for {regime}")
            break
    
    print(f"\n{'='*50}")
    print("🎉 All SARL training completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_sarl_training() 