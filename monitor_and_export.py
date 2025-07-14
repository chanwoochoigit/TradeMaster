#!/usr/bin/env python3
"""
Training Completion Monitor & Auto-Export Script

This script monitors EIIE training processes and automatically exports allocation
histories when models finish training.
"""
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_model_completion(model_name, regime):
    """Check if a model has completed training"""
    workdir = f"work_dir/portfolio_management_mcad_{model_name}_{model_name}_adam_mse_{regime}"
    best_model_path = Path(workdir) / "best.pth"
    return best_model_path.exists()

def check_training_process(model_name, regime):
    """Check if training process is still running"""
    try:
        result = subprocess.run(
            ["ps", "aux"], 
            capture_output=True, 
            text=True
        )
        pattern = f"train_{model_name}.py --config configs/portfolio_management/portfolio_management_mcad_{model_name}_{model_name}_adam_mse_{regime}.py"
        return pattern in result.stdout
    except:
        return False

def export_model_allocations(model_name, regime):
    """Export allocation history for a completed model"""
    print(f"\n🎉 {model_name.upper()} {regime} training completed!")
    print(f"📊 Exporting allocation history...")
    
    try:
        cmd = [
            sys.executable, 
            "tools/export_allocations.py",
            "--model", model_name,
            "--regime", regime
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {model_name.upper()} {regime} allocations exported successfully!")
            print(result.stdout)
        else:
            print(f"❌ Export failed for {model_name} {regime}:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Exception during export: {e}")

def main():
    """Main monitoring loop"""
    print("🔍 Starting Training Completion Monitor")
    print("=" * 60)
    
    # Models and regimes to monitor
    monitor_configs = [
        ("eiie", "covid"),
        ("eiie", "trade_war"), 
        ("eiie", "trade_war_i"),
        # Add other models when they're working
        # ("sarl", "covid"),
        # ("sarl", "trade_war"),
        # ("sarl", "trade_war_i"),
        # ("deeptrader", "covid"),
        # ("deeptrader", "trade_war"),
        # ("deeptrader", "trade_war_i"),
    ]
    
    completed = set()
    check_interval = 60  # Check every minute
    
    print(f"📋 Monitoring {len(monitor_configs)} model-regime combinations:")
    for model, regime in monitor_configs:
        print(f"  • {model.upper()} - {regime}")
    print(f"⏰ Check interval: {check_interval} seconds")
    print("=" * 60)
    
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_count = 0
        
        for model, regime in monitor_configs:
            key = f"{model}_{regime}"
            
            if key in completed:
                continue
                
            # Check if model has completed
            if check_model_completion(model, regime):
                if not key in completed:
                    export_model_allocations(model, regime)
                    completed.add(key)
            else:
                # Check if still training
                if check_training_process(model, regime):
                    active_count += 1
                else:
                    print(f"⚠️  {model.upper()} {regime} process not found (may have crashed)")
        
        # Status update
        remaining = len(monitor_configs) - len(completed)
        print(f"\n📊 [{current_time}] Status: {len(completed)}/{len(monitor_configs)} completed, {active_count} active, {remaining} remaining")
        
        # Exit if all completed
        if len(completed) == len(monitor_configs):
            print("\n🎉 All monitored trainings completed!")
            break
            
        # Wait before next check
        time.sleep(check_interval)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}") 