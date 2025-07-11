#!/usr/bin/env python3
"""
Comprehensive pipeline for training Mask SAC on all 3 MCAD regimes and exporting allocation histories
"""
import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(command, description, background=False):
    """Run a shell command with proper logging"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")

    if background:
        # Run in background and return the process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return process
    else:
        # Run and wait for completion
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Show last 500 chars
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr[-500:])
        return result


def check_training_status(workdir, tag):
    """Check if training has completed by looking for best.pth"""
    checkpoint_path = os.path.join(workdir, tag, "best.pth")
    return os.path.exists(checkpoint_path)


def wait_for_training_completion(workdir, tag, max_wait_hours=6):
    """Wait for training to complete, checking every 5 minutes"""
    max_wait_seconds = max_wait_hours * 3600
    check_interval = 300  # 5 minutes

    elapsed = 0
    while elapsed < max_wait_seconds:
        if check_training_status(workdir, tag):
            print(f"✅ Training completed for {tag}")
            return True

        print(
            f"⏳ Waiting for {tag} training to complete... ({elapsed//60} min elapsed)"
        )
        time.sleep(check_interval)
        elapsed += check_interval

    print(f"⚠️ Training for {tag} did not complete within {max_wait_hours} hours")
    return False


def main():
    """Main pipeline function"""

    print("🎯 MASK SAC MCAD Pipeline Starting")
    print(
        "This will train Mask SAC on all 3 MCAD regimes and export allocation histories"
    )

    # Configuration
    regimes = ["covid", "trade_war", "trade_war_i"]
    workdir = "workdir"

    # Training configurations
    configs = {
        "covid": "configs/earnmore/mask_sac_mcad_covid.py",
        "trade_war": "configs/earnmore/mask_sac_mcad_trade_war.py",
        "trade_war_i": "configs/earnmore/mask_sac_mcad_trade_war_i.py",
    }

    training_results = {}

    # Step 1: Train all regimes
    print(f"\n📚 STEP 1: Training Mask SAC on {len(regimes)} MCAD regimes")

    for regime in regimes:
        config_path = configs[regime]
        tag = f"mask_sac_mcad_{regime}"

        # Check if already trained
        if check_training_status(workdir, tag):
            print(f"✅ {regime} already trained, skipping...")
            training_results[regime] = True
            continue

        # Start training
        train_command = f"python tools/earnmore/train.py --config {config_path}"
        print(f"\n🏋️ Starting training for {regime} regime...")

        # Run training in background
        process = run_command(
            train_command, f"Training Mask SAC on {regime} regime", background=True
        )

        # Wait for completion (with timeout)
        success = wait_for_training_completion(workdir, tag, max_wait_hours=3)
        training_results[regime] = success

        if not success:
            print(f"⚠️ Training timeout for {regime}, moving to next regime...")
            continue

    # Step 2: Export allocation histories for successfully trained models
    print(f"\n📊 STEP 2: Exporting allocation histories")

    export_results = {}
    for regime in regimes:
        if not training_results.get(regime, False):
            print(f"⏭️ Skipping {regime} - training not completed")
            export_results[regime] = False
            continue

        config_path = configs[regime]
        export_command = f"python tools/export_allocations.py --config {config_path}"

        result = run_command(
            export_command, f"Exporting allocation history for {regime}"
        )
        export_results[regime] = result.returncode == 0

    # Step 3: Summary
    print(f"\n📋 PIPELINE SUMMARY")
    print("=" * 60)

    print("\n🏋️ Training Results:")
    for regime in regimes:
        status = "✅ SUCCESS" if training_results.get(regime, False) else "❌ FAILED"
        print(f"  {regime:<15}: {status}")

    print("\n📊 Export Results:")
    for regime in regimes:
        status = "✅ SUCCESS" if export_results.get(regime, False) else "❌ FAILED"
        print(f"  {regime:<15}: {status}")

    # Overall success
    successful_regimes = sum(
        1
        for r in regimes
        if training_results.get(r, False) and export_results.get(r, False)
    )
    total_regimes = len(regimes)

    print(
        f"\n🎯 OVERALL RESULT: {successful_regimes}/{total_regimes} regimes completed successfully"
    )

    if successful_regimes == total_regimes:
        print("🎉 All MCAD regimes completed successfully!")

        # Show where allocation files are saved
        print("\n📁 Allocation files saved to:")
        for regime in regimes:
            if export_results.get(regime, False):
                tag = f"mask_sac_mcad_{regime}"
                alloc_file = os.path.join(workdir, tag, f"mask_sac_mcad_{regime}.csv")
                print(f"  {regime}: {alloc_file}")
    else:
        print("⚠️ Some regimes failed. Check the logs above for details.")

    return successful_regimes == total_regimes


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
