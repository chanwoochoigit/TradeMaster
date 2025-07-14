#!/usr/bin/env python3
"""
Export SARL allocation histories to consistent exports directory structure
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

def export_sarl_allocations():
    """Export SARL allocation histories for all regimes"""
    
    print("🚀 Exporting SARL allocation histories...")
    print("📁 Output directory: exports/{regime}/")
    
    regimes = ['covid', 'trade_war', 'trade_war_i']
    
    for regime in regimes:
        print(f"\n📊 Exporting SARL {regime} regime...")
        
        # Use the export_allocations.py script
        cmd = [
            'python', 'tools/export_allocations.py',
            '--model', 'sarl',
            '--regime', regime,
            '--dataset', 'test'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check if file was created
            temp_file = f"sarl_{regime}_test_allocations.csv"
            if os.path.exists(temp_file):
                # Create target directory structure
                target_dir = f"exports/{regime}"
                os.makedirs(target_dir, exist_ok=True)
                
                # Move file to target location
                target_path = f"{target_dir}/sarl.csv"
                shutil.move(temp_file, target_path)
                
                print(f"✅ SARL {regime} exported to: {target_path}")
            else:
                print(f"❌ SARL {regime} export failed - file not created")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ SARL {regime} export failed: {e}")
            if e.stdout:
                print(f"Output: {e.stdout}")
            if e.stderr:
                print(f"Error: {e.stderr}")
        except Exception as e:
            print(f"❌ SARL {regime} export failed with exception: {e}")
    
    print(f"\n✅ SARL allocation export completed!")
    print(f"📁 Files saved in: exports/{{regime}}/sarl.csv")

if __name__ == "__main__":
    export_sarl_allocations() 