#!/usr/bin/env python3
"""
Training Monitor for Mask SAC MCAD Pipeline
==========================================

This script monitors training progress in real-time for all MCAD regimes.
Run this in a separate terminal while training is happening.

Usage:
    python monitor_training.py [--regime REGIME] [--follow] [--interval SECONDS]

Options:
    --regime REGIME: Monitor specific regime (covid, trade_war, trade_war_i)
    --follow: Continuously monitor and update (like tail -f)
    --interval SECONDS: Update interval for --follow mode (default: 30)
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path


class TrainingMonitor:
    """Monitor training progress for MCAD regimes"""

    def __init__(self):
        self.regimes = ["covid", "trade_war", "trade_war_i"]
        self.workdir = "workdir"

    def log(self, message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "PROGRESS": "🔄",
        }.get(level, "📝")

        print(f"[{timestamp}] {prefix} {message}")

    def get_training_status(self, regime: str) -> dict:
        """Get detailed training status for a regime"""
        tag = f"mask_sac_mcad_{regime}"
        workdir_path = f"{self.workdir}/{tag}"

        status = {
            "regime": regime,
            "started": False,
            "completed": False,
            "checkpoint_exists": False,
            "checkpoint_size": 0,
            "log_lines": 0,
            "log_size": 0,
            "last_episode": None,
            "recent_metrics": [],
            "process_active": False,
            "tensorboard_files": 0,
            "errors": [],
        }

        # Check if training started
        if not os.path.exists(workdir_path):
            return status

        status["started"] = True

        # Check checkpoint
        checkpoint_path = f"{workdir_path}/best.pth"
        if os.path.exists(checkpoint_path):
            status["completed"] = True
            status["checkpoint_exists"] = True
            status["checkpoint_size"] = os.path.getsize(checkpoint_path)

        # Check training log
        log_file_path = f"{workdir_path}/train_log.txt"
        if os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r") as f:
                    lines = f.readlines()

                status["log_lines"] = len(lines)
                status["log_size"] = os.path.getsize(log_file_path)

                # Get recent metrics (last 3 lines)
                for line in lines[-3:]:
                    if line.strip():
                        status["recent_metrics"].append(line.strip())

            except Exception as e:
                status["errors"].append(f"Error reading log: {str(e)}")

        # Check training info for episode number
        info_file_path = f"{workdir_path}/train_infos.txt"
        if os.path.exists(info_file_path):
            try:
                with open(info_file_path, "r") as f:
                    lines = f.readlines()

                if lines:
                    # Parse last entry
                    try:
                        last_info = json.loads(lines[-1].strip())
                        if "episode" in last_info:
                            episode_num = (
                                last_info["episode"][0]
                                if isinstance(last_info["episode"], list)
                                else last_info["episode"]
                            )
                            status["last_episode"] = episode_num
                    except:
                        pass

            except Exception as e:
                status["errors"].append(f"Error reading info: {str(e)}")

        # Check for active process
        try:
            import subprocess

            result = subprocess.run(
                f"ps aux | grep 'train.py.*{regime}' | grep -v grep",
                shell=True,
                capture_output=True,
                text=True,
            )
            status["process_active"] = bool(result.stdout.strip())
        except:
            pass

        # Check tensorboard files
        try:
            tb_files = [
                f
                for f in os.listdir(workdir_path)
                if f.startswith("events.out.tfevents")
            ]
            status["tensorboard_files"] = len(tb_files)
        except:
            pass

        return status

    def display_status(self, regimes: list = None):
        """Display current training status"""
        if regimes is None:
            regimes = self.regimes

        print("\n" + "=" * 80)
        print("🔍 MASK SAC MCAD TRAINING MONITOR")
        print("=" * 80)

        for regime in regimes:
            status = self.get_training_status(regime)

            print(f"\n📊 {regime.upper()} REGIME")
            print("-" * 40)

            if not status["started"]:
                print("  ⏸️  Training not started")
                continue

            # Training status
            if status["completed"]:
                print(f"  ✅ Training completed")
                print(f"  💾 Checkpoint: {status['checkpoint_size']/1024/1024:.1f} MB")
            elif status["process_active"]:
                print(f"  🔄 Training in progress")
                if status["last_episode"]:
                    print(f"  📈 Episode: {status['last_episode']}")
            else:
                print(f"  ⏸️  Training stopped (no active process)")

            # Log information
            if status["log_lines"] > 0:
                print(
                    f"  📝 Log: {status['log_lines']} lines ({status['log_size']/1024:.1f} KB)"
                )

                # Show recent metrics
                if status["recent_metrics"]:
                    print(f"  📊 Recent metrics:")
                    for metric in status["recent_metrics"]:
                        print(f"    {metric}")
            else:
                print(f"  📝 No training log")

            # Tensorboard
            if status["tensorboard_files"] > 0:
                print(f"  📈 Tensorboard: {status['tensorboard_files']} files")

            # Errors
            if status["errors"]:
                print(f"  ❌ Errors:")
                for error in status["errors"]:
                    print(f"    {error}")

    def tail_logs(self, regime: str, lines: int = 10):
        """Show recent log lines for a regime"""
        tag = f"mask_sac_mcad_{regime}"
        log_file_path = f"{self.workdir}/{tag}/train_log.txt"

        print(f"\n📖 Last {lines} lines from {regime} training log:")
        print("=" * 60)

        if not os.path.exists(log_file_path):
            print(f"❌ Log file not found: {log_file_path}")
            return

        try:
            with open(log_file_path, "r") as f:
                all_lines = f.readlines()

            if not all_lines:
                print("📝 Log file is empty")
                return

            # Show last N lines
            recent_lines = all_lines[-lines:]
            for i, line in enumerate(recent_lines):
                line_num = len(all_lines) - len(recent_lines) + i + 1
                print(f"{line_num:4d}: {line.rstrip()}")

        except Exception as e:
            print(f"❌ Error reading log: {str(e)}")

    def follow_training(self, regimes: list = None, interval: int = 30):
        """Continuously monitor training progress"""
        if regimes is None:
            regimes = self.regimes

        print(f"🔄 Following training progress (updating every {interval} seconds)")
        print("Press Ctrl+C to stop")

        try:
            while True:
                # Clear screen (works on most terminals)
                os.system("clear" if os.name == "posix" else "cls")

                self.display_status(regimes)

                # Check if all completed
                all_completed = True
                for regime in regimes:
                    status = self.get_training_status(regime)
                    if status["started"] and not status["completed"]:
                        all_completed = False
                        break

                if all_completed:
                    print(f"\n🎉 All training completed!")
                    break

                print(f"\n⏰ Next update in {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Mask SAC MCAD training progress"
    )
    parser.add_argument(
        "--regime",
        choices=["covid", "trade_war", "trade_war_i"],
        help="Monitor specific regime only",
    )
    parser.add_argument(
        "--follow", action="store_true", help="Continuously monitor (like tail -f)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval for --follow mode (seconds)",
    )
    parser.add_argument(
        "--tail", type=int, metavar="N", help="Show last N lines from training log"
    )

    args = parser.parse_args()

    monitor = TrainingMonitor()

    # Determine which regimes to monitor
    regimes = [args.regime] if args.regime else None

    if args.tail:
        if not args.regime:
            print("❌ --tail requires --regime to be specified")
            sys.exit(1)
        monitor.tail_logs(args.regime, args.tail)
    elif args.follow:
        monitor.follow_training(regimes, args.interval)
    else:
        monitor.display_status(regimes)


if __name__ == "__main__":
    main()
