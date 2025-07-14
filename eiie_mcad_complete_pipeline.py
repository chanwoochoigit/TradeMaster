#!/usr/bin/env python3
"""
Complete EIIE MCAD Pipeline

This script provides a complete end-to-end pipeline for:
1. Running EIIE training on all 3 MCAD regimes (COVID, Trade War, Trade War I)
2. Monitoring training progress
3. Exporting allocation histories to CSV in the same format as masked SAC
4. Generating summary reports

Usage:
    python eiie_mcad_complete_pipeline.py
"""

import os
import sys
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class EIIEMCADPipeline:
    def __init__(self, quick_test: bool = False):
        self.project_root = project_root
        self.quick_test = quick_test
        self.regimes = ["trade_war_i", "covid", "trade_war"]
        self.epochs = 50 if quick_test else 200
        
        # Results tracking
        self.setup_results = {}
        self.training_results = {}
        self.export_results = {}
        self.training_times = {}
        
        # Ensure directories exist
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        for regime in self.regimes:
            os.makedirs(f"exports/{regime}", exist_ok=True)
            
        os.makedirs("work_dir", exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if level == "ERROR":
            icon = "❌"
        elif level == "SUCCESS":
            icon = "✅"
        elif level == "PROGRESS":
            icon = "🔄"
        elif level == "WARNING":
            icon = "⚠️"
        else:
            icon = "ℹ️"
            
        print(f"[{timestamp}] {icon} {message}")
        
    def run_command(self, command: str, description: str) -> Optional[subprocess.CompletedProcess]:
        """Run a command and return the result"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=7200  # 2 hour timeout
            )
            return result
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out: {description}", "ERROR")
            return None
        except Exception as e:
            self.log(f"Command failed: {description} - {str(e)}", "ERROR")
            return None

    def fix_eiie_configs(self) -> bool:
        """Fix EIIE configuration files for MCAD dataset"""
        self.log("Fixing EIIE configurations for MCAD dataset", "PROGRESS")
        
        configs_fixed = 0
        for regime in self.regimes:
            config_path = f"configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_{regime}.py"
            
            if os.path.exists(config_path):
                # Read the existing config
                with open(config_path, 'r') as f:
                    content = f.read()
                
                # Update epochs for quick test
                if self.quick_test and "'epochs': 200" in content:
                    content = content.replace("'epochs': 200", f"'epochs': {self.epochs}")
                    with open(config_path, 'w') as f:
                        f.write(content)
                
                self.log(f"Fixed config for {regime}", "SUCCESS")
                configs_fixed += 1
            else:
                self.log(f"Config not found for {regime}: {config_path}", "WARNING")
                
        return configs_fixed == len(self.regimes)

    def train_eiie_regime(self, regime: str) -> bool:
        """Train EIIE for a specific regime"""
        start_time = time.time()
        self.log(f"Starting EIIE training for {regime} regime", "PROGRESS")
        
        # Check if already trained
        checkpoint_dir = f"work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_{regime}/checkpoints"
        if os.path.exists(os.path.join(checkpoint_dir, "best.pkl")):
            self.log(f"EIIE model already trained for {regime}, skipping training", "SUCCESS")
            self.training_results[regime] = True
            self.training_times[regime] = 0
            return True
        
        # Run EIIE training
        training_command = f"/opt/anaconda3/envs/trademaster/bin/python tools/portfolio_management/train_eiie.py --config configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_{regime}.py --task_name train --verbose 1"
        
        self.log(f"Running: EIIE training for {regime}", "PROGRESS")
        self.log(f"Command: {training_command}", "INFO")
        
        result = self.run_command(training_command, f"EIIE training for {regime}")
        training_time = time.time() - start_time
        self.training_times[regime] = training_time / 60  # Convert to minutes
        
        if result and result.returncode == 0:
            self.log(f"EIIE training succeeded for {regime} after {training_time/60:.1f} minutes", "SUCCESS")
            self.training_results[regime] = True
            return True
        else:
            error_output = result.stderr if result else "Unknown error"
            self.log(f"EIIE training for {regime} failed", "ERROR")
            self.log(f"Error output: {error_output}", "ERROR")
            self.log(f"EIIE training failed for {regime} after {training_time/60:.1f} minutes", "ERROR")
            self.training_results[regime] = False
            return False

    def export_allocation_history(self, regime: str) -> bool:
        """Export allocation history for a trained regime"""
        self.log(f"Exporting allocation history for {regime}", "PROGRESS")

        # Export ALL datasets (train+val+test) directly to the final location
        export_command = f"/opt/anaconda3/envs/trademaster/bin/python tools/export_allocations.py --model eiie --regime {regime} --dataset all --output exports/{regime}/eiie.csv"
        result = self.run_command(
            export_command, f"Exporting allocation history for {regime}"
        )

        if result and result.returncode == 0:
            # Verify CSV file was created and has content
            csv_path = f"exports/{regime}/eiie.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    record_count = len(df)
                    file_size = os.path.getsize(csv_path) / 1024  # KB
                    
                    if record_count > 0:
                        self.log(f"Export successful for {regime}: {record_count} records ({file_size:.1f} KB)", "SUCCESS")
                        self.export_results[regime] = True
                        return True
                    else:
                        self.log(f"Export file is empty for {regime}", "ERROR")
                        self.export_results[regime] = False
                        return False
                except Exception as e:
                    self.log(f"Error reading exported CSV for {regime}: {str(e)}", "ERROR")
                    self.export_results[regime] = False
                    return False
            else:
                self.log(f"Export CSV file not found for {regime}: {csv_path}", "ERROR")
                self.export_results[regime] = False
                return False
        else:
            error_output = result.stderr if result else "Unknown error"
            self.log(f"Export command failed for {regime}: {error_output}", "ERROR")
            self.export_results[regime] = False
            return False

    def generate_summary(self) -> Dict:
        """Generate and save summary report"""
        self.log("Generating summary report", "PROGRESS")
        
        # Count successes
        setup_success = sum(1 for v in self.setup_results.values() if v)
        training_success = sum(1 for v in self.training_results.values() if v)
        export_success = sum(1 for v in self.export_results.values() if v)
        
        # Create summary
        summary = {
            "pipeline_type": "EIIE_MCAD_Complete_Pipeline",
            "timestamp": datetime.now().isoformat(),
            "quick_test": self.quick_test,
            "epochs_per_regime": self.epochs,
            "regimes": self.regimes,
            "results": {
                "setup": {
                    "total": len(self.regimes),
                    "successful": setup_success,
                    "details": self.setup_results
                },
                "training": {
                    "total": len(self.regimes),
                    "successful": training_success,
                    "details": self.training_results,
                    "times_minutes": self.training_times
                },
                "export": {
                    "total": len(self.regimes),
                    "successful": export_success,
                    "details": self.export_results
                }
            }
        }
        
        # Save summary
        summary_path = "work_dir/eiie_pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

    def run_pipeline(self):
        """Run the complete EIIE MCAD pipeline"""
        pipeline_start = time.time()
        
        print("🚀 Starting EIIE MCAD Complete Pipeline")
        print(f"Mode: {'Quick Test' if self.quick_test else 'Full Training'}")
        print(f"Regimes: {', '.join(self.regimes)}")
        print(f"Epochs per regime: {self.epochs}")
        
        # Step 1: Configuration Setup
        self.log("\n" + "="*60, "INFO")
        self.log("STEP 1: Configuration Setup", "PROGRESS")
        self.log("="*60, "INFO")
        
        setup_success = self.fix_eiie_configs()
        for regime in self.regimes:
            self.setup_results[regime] = True  # Config fixes are regime-independent
        
        # Step 2: EIIE Model Training
        self.log("\n" + "="*60, "INFO")
        self.log("STEP 2: EIIE Model Training", "PROGRESS")
        self.log("="*60, "INFO")
        
        for regime in self.regimes:
            success = self.train_eiie_regime(regime)
            if not success:
                self.log(f"Training failed for {regime}, continuing with next regime", "WARNING")
        
        # Step 3: Allocation History Export
        self.log("\n" + "="*60, "INFO")
        self.log("STEP 3: Allocation History Export", "PROGRESS")
        self.log("="*60, "INFO")
        
        for regime in self.regimes:
            if self.training_results.get(regime, False):
                self.export_allocation_history(regime)
            else:
                self.log(f"Skipping export for {regime} - training not completed", "WARNING")
                self.export_results[regime] = False
        
        # Step 4: Summary Generation
        self.log("\n" + "="*60, "INFO")
        self.log("STEP 4: Summary Generation", "PROGRESS")
        self.log("="*60, "INFO")
        
        summary = self.generate_summary()
        
        # Final Report
        pipeline_time = time.time() - pipeline_start
        
        self.log("="*60, "INFO")
        self.log("EIIE PIPELINE SUMMARY", "SUCCESS")
        self.log("="*60, "INFO")
        self.log(f"Setup Success: {summary['results']['setup']['successful']}/{summary['results']['setup']['total']}", "INFO")
        self.log(f"Training Success: {summary['results']['training']['successful']}/{summary['results']['training']['total']}", "INFO")
        self.log(f"Export Success: {summary['results']['export']['successful']}/{summary['results']['export']['total']}", "INFO")
        self.log("", "INFO")
        self.log("Regime Details:", "INFO")
        for regime in self.regimes:
            setup_icon = "✅" if self.setup_results.get(regime, False) else "❌"
            training_icon = "✅" if self.training_results.get(regime, False) else "❌"
            export_icon = "✅" if self.export_results.get(regime, False) else "❌"
            self.log(f"  {regime:<15}: Setup {setup_icon} | Training {training_icon} | Export {export_icon}", "INFO")
        
        self.log("", "SUCCESS")
        self.log(f"📄 Summary report saved: work_dir/eiie_pipeline_summary.json", "SUCCESS")
        
        # Check for failures
        total_failures = (len(self.regimes) * 3) - (
            summary['results']['setup']['successful'] + 
            summary['results']['training']['successful'] + 
            summary['results']['export']['successful']
        )
        
        if total_failures > 0:
            self.log(f"⚠️ {total_failures} operations failed", "WARNING")
        
        self.log("", "SUCCESS")
        self.log(f"⏱️ Total pipeline time: {pipeline_time/3600:.2f} hours", "SUCCESS")
        
        if total_failures > 0:
            self.log("⚠️ Pipeline completed with some failures", "WARNING")
        else:
            self.log("🎉 Pipeline completed successfully!", "SUCCESS")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete EIIE MCAD Pipeline")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run in quick test mode (50 epochs instead of 200)")
    
    args = parser.parse_args()
    
    pipeline = EIIEMCADPipeline(quick_test=args.quick_test)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 