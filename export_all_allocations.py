#!/usr/bin/env python3
"""
Export all model allocation histories to consistent exports directory structure

This script exports allocation histories for all trained models:
- mask_sac
- sarl  
- eiie
- deeptrader (if available)

Output structure:
    exports/
    ├── covid/
    │   ├── mask_sac.csv
    │   ├── sarl.csv
    │   └── eiie.csv
    ├── trade_war/
    │   ├── mask_sac.csv
    │   ├── sarl.csv
    │   └── eiie.csv
    ├── trade_war_i/
    │   ├── mask_sac.csv
    │   ├── sarl.csv
    │   └── eiie.csv
    └── export_summary.json

Usage:
    python export_all_allocations.py
"""
import subprocess
import sys
import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def export_model_allocations(model_name, regimes=None):
    """Export allocation histories for a specific model"""
    
    if regimes is None:
        regimes = ['covid', 'trade_war', 'trade_war_i']
    
    print(f"\n🤖 Exporting {model_name.upper()} allocation histories...")
    
    results = {}
    
    for regime in regimes:
        print(f"  📊 {regime} regime...")
        
        # Use the export_allocations.py script
        cmd = [
            'python', 'tools/export_allocations.py',
            '--model', model_name,
            '--regime', regime,
            '--dataset', 'test'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check if file was created
            temp_file = f"{model_name}_{regime}_test_allocations.csv"
            if os.path.exists(temp_file):
                # Create target directory structure
                target_dir = f"exports/{regime}"
                os.makedirs(target_dir, exist_ok=True)
                
                # Move file to target location
                target_path = f"{target_dir}/{model_name}.csv"
                shutil.move(temp_file, target_path)
                
                print(f"    ✅ Exported to: {target_path}")
                results[regime] = {
                    "status": "success",
                    "file": target_path,
                    "size_kb": round(os.path.getsize(target_path) / 1024, 1)
                }
            else:
                print(f"    ❌ Export failed - file not created")
                results[regime] = {"status": "failed", "reason": "file_not_created"}
                
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Export failed: {e}")
            results[regime] = {"status": "failed", "reason": str(e)}
        except Exception as e:
            print(f"    ❌ Export failed with exception: {e}")
            results[regime] = {"status": "failed", "reason": str(e)}
    
    return results

def check_model_availability():
    """Check which models have trained checkpoints available"""
    
    models = {
        'mask_sac': ['covid', 'trade_war', 'trade_war_i'],
        'sarl': ['covid', 'trade_war', 'trade_war_i'],
        'eiie': ['covid', 'trade_war', 'trade_war_i'],
        'deeptrader': ['covid', 'trade_war', 'trade_war_i']
    }
    
    available_models = {}
    
    for model, regimes in models.items():
        available_regimes = []
        for regime in regimes:
            if model == 'mask_sac':
                checkpoint_path = f"workdir/mask_sac_mcad_{regime}/best.pth"
            else:
                checkpoint_path = f"work_dir/portfolio_management_mcad_{model}_{model}_adam_mse_{regime}/best.pth"
            
            if os.path.exists(checkpoint_path):
                available_regimes.append(regime)
        
        if available_regimes:
            available_models[model] = available_regimes
    
    return available_models

def export_all_allocations():
    """Export allocation histories for all available models"""
    
    print("🚀 Exporting allocation histories for all models...")
    print("📁 Output directory: exports/")
    print("=" * 60)
    
    # Check model availability
    available_models = check_model_availability()
    
    if not available_models:
        print("❌ No trained models found. Please train models first.")
        return False
    
    print("📋 Available models:")
    for model, regimes in available_models.items():
        print(f"  {model.upper()}: {', '.join(regimes)}")
    
    # Export each model
    all_results = {}
    
    for model, regimes in available_models.items():
        model_results = export_model_allocations(model, regimes)
        all_results[model] = model_results
    
    # Generate summary
    summary = {
        "export_timestamp": datetime.now().isoformat(),
        "models_exported": list(available_models.keys()),
        "results": all_results
    }
    
    # Save summary
    summary_path = "exports/export_summary.json"
    os.makedirs("exports", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Export completed!")
    print(f"📋 Summary saved to: {summary_path}")
    
    # Show file structure
    print("\n📂 File structure:")
    # Group by regime instead of model
    all_regimes = set()
    for regimes in available_models.values():
        all_regimes.update(regimes)
    
    for regime in sorted(all_regimes):
        print(f"  exports/{regime}/")
        for model in available_models.keys():
            if regime in available_models[model] and all_results[model][regime]["status"] == "success":
                print(f"    ├── {model}.csv")
    
    # Show overall results
    total_exports = sum(len(regimes) for regimes in available_models.values())
    successful_exports = sum(
        sum(1 for result in model_results.values() if result["status"] == "success")
        for model_results in all_results.values()
    )
    
    print(f"\n📊 Overall result: {successful_exports}/{total_exports} exports successful")
    
    return successful_exports == total_exports

if __name__ == "__main__":
    success = export_all_allocations()
    sys.exit(0 if success else 1) 