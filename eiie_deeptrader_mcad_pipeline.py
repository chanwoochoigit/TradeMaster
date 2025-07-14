#!/usr/bin/env python3
"""
EIIE & DeepTrader MCAD Complete Pipeline
========================================

This script trains EIIE and DeepTrader models on all MCAD regimes and exports their allocation histories.

Usage:
    python eiie_deeptrader_mcad_pipeline.py [--quick-test] [--model eiie|deeptrader|all]
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json


class EIIEDeepTraderPipeline:
    def __init__(self, quick_test=False, models=None):
        self.quick_test = quick_test
        self.models = models or ["eiie", "deeptrader"]
        self.regimes = ["covid", "trade_war", "trade_war_i"]
        self.root_dir = Path(__file__).parent
        self.num_epochs = 50 if quick_test else 200

    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def run_command(self, command, description):
        """Run shell command with logging"""
        self.log(f"Running: {description}", "PROGRESS")
        self.log(f"Command: {command}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode == 0:
                self.log(f"✅ {description} completed successfully", "SUCCESS")
                return True
            else:
                self.log(f"❌ {description} failed", "ERROR")
                if result.stderr:
                    self.log(f"Error: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log(f"💥 {description} failed with exception: {str(e)}", "ERROR")
            return False

    def train_model(self, model, regime):
        """Train a specific model on a specific regime"""
        config_path = f"configs/portfolio_management/portfolio_management_mcad_{model}_{model}_adam_mse_{regime}.py"
        train_script = f"tools/portfolio_management/train_{model}.py"
        
        self.log(f"Training {model.upper()} on {regime} regime")
        
        # Use the trademaster environment python
        command = f"/opt/anaconda3/envs/trademaster/bin/python {train_script} --config {config_path}"
        
        return self.run_command(command, f"Training {model} on {regime}")

    def check_model_exists(self, model, regime):
        """Check if trained model exists"""
        work_dir = f"work_dir/portfolio_management_mcad_{model}_{model}_adam_mse_{regime}"
        model_path = self.root_dir / work_dir / "best.pth"
        
        exists = model_path.exists()
        if exists:
            self.log(f"✅ Found model: {model_path}")
        else:
            self.log(f"❌ Model not found: {model_path}")
        
        return exists

    def run_pipeline(self):
        """Run the complete training pipeline"""
        start_time = time.time()
        
        self.log("🚀 Starting EIIE & DeepTrader MCAD Pipeline", "SUCCESS")
        self.log(f"Models: {', '.join([m.upper() for m in self.models])}")
        self.log(f"Regimes: {', '.join(self.regimes)}")
        self.log(f"Epochs: {self.num_epochs}")
        
        results = {}
        
        try:
            # Train all model-regime combinations
            for model in self.models:
                results[model] = {}
                
                for regime in self.regimes:
                    self.log(f"\n🎯 Processing {model.upper()} on {regime}")
                    
                    # Check if already trained
                    if self.check_model_exists(model, regime):
                        self.log("Model already exists, skipping training")
                        results[model][regime] = "already_exists"
                        continue
                    
                    # Train the model
                    success = self.train_model(model, regime)
                    results[model][regime] = "success" if success else "failed"
                    
                    if success:
                        self.log(f"✅ {model.upper()} on {regime} completed")
                    else:
                        self.log(f"❌ {model.upper()} on {regime} failed")
            
            # Summary
            end_time = time.time()
            total_time = (end_time - start_time) / 3600  # Convert to hours
            
            self.log(f"\n📊 Pipeline completed in {total_time:.2f} hours")
            
            # Count successes
            total_jobs = len(self.models) * len(self.regimes)
            successful_jobs = sum(1 for model_results in results.values() 
                                for status in model_results.values() 
                                if status in ["success", "already_exists"])
            
            self.log(f"Successful jobs: {successful_jobs}/{total_jobs}")
            
            # Save results
            results_file = f"eiie_deeptrader_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.log(f"Results saved to: {results_file}")
            
            return successful_jobs == total_jobs
            
        except KeyboardInterrupt:
            self.log("🛑 Pipeline interrupted by user", "WARNING")
            return False
        except Exception as e:
            self.log(f"💥 Pipeline failed: {str(e)}", "ERROR")
            return False


def main():
    parser = argparse.ArgumentParser(description="EIIE & DeepTrader MCAD Pipeline")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Quick test mode (50 epochs)")
    parser.add_argument("--model", choices=["eiie", "deeptrader", "all"], 
                       default="all", help="Which model(s) to train")
    
    args = parser.parse_args()
    
    # Determine models to train
    if args.model == "all":
        models = ["eiie", "deeptrader"]
    else:
        models = [args.model]
    
    # Run pipeline
    pipeline = EIIEDeepTraderPipeline(quick_test=args.quick_test, models=models)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 