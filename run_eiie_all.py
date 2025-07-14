#!/usr/bin/env python3
"""
Simple script to run EIIE training for all three market regimes
"""
import subprocess
import sys
import os

def run_eiie_training():
    """Run EIIE training for all three market regimes"""
    
    regimes = ['trade_war_i', 'covid', 'trade_war']
    
    for regime in regimes:
        print(f"\n{'='*50}")
        print(f"🚀 Starting EIIE training for {regime}")
        print(f"{'='*50}")
        
        cmd = [
            'python', 'tools/portfolio_management/train_eiie.py',
            '--config', f'configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_{regime}.py',
            '--task_name', 'train',
            '--verbose', '1'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ EIIE training completed successfully for {regime}")
        except subprocess.CalledProcessError as e:
            print(f"❌ EIIE training failed for {regime}: {e}")
            print(f"Command: {' '.join(cmd)}")
            continue
        except KeyboardInterrupt:
            print(f"\n🛑 Training interrupted by user for {regime}")
            break
    
    print(f"\n{'='*50}")
    print("🎉 All EIIE training completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_eiie_training() 