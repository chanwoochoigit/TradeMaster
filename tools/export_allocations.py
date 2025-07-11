#!/usr/bin/env python3
"""
Export allocation history from trained Mask SAC MCAD models to CSV.

Default behavior exports ALL regimes and ALL datasets (train, val, test):
    python tools/export_allocations.py

Single regime examples:
    python tools/export_allocations.py --regime covid --dataset test --output covid_test.csv
    python tools/export_allocations.py --regime trade_war --dataset all  # All datasets for trade_war

Output format: date,spy,qqq,dbc,agg,gld,cash (lowercase, cash at end)
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pm.utils.export import export_allocation_history


def export_all_regimes_all_datasets(output_dir="allocation_exports"):
    """Export allocations for all regimes and all datasets (train, val, test)"""

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Map regime names to config files and model paths
    regime_info = {
        "covid": {
            "config": "configs/earnmore/mask_sac_mcad_covid.py",
            "workdir": "workdir/mask_sac_mcad_covid",
            "name": "COVID",
        },
        "trade_war": {
            "config": "configs/earnmore/mask_sac_mcad_trade_war.py",
            "workdir": "workdir/mask_sac_mcad_trade_war",
            "name": "Trade War",
        },
        "trade_war_i": {
            "config": "configs/earnmore/mask_sac_mcad_trade_war_i.py",
            "workdir": "workdir/mask_sac_mcad_trade_war_i",
            "name": "Trade War I",
        },
    }

    datasets = ["train", "val", "test"]
    asset_names = ["SPY", "QQQ", "DBC", "AGG", "GLD"]

    results = {}

    print(f"\n🚀 Exporting allocations for ALL regimes and ALL datasets")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)

    for regime, info in regime_info.items():
        print(f"\n📊 Processing {info['name']} regime...")

        config_path = project_root / info["config"]
        model_dir = project_root / info["workdir"]
        best_model_path = model_dir / "best.pth"

        # Check if model exists
        if not best_model_path.exists():
            print(f"❌ Model not found: {best_model_path}")
            results[regime] = {"status": "failed", "reason": "model_not_found"}
            continue

        results[regime] = {"status": "success", "files": {}}

        for dataset in datasets:
            print(f"  📈 Exporting {dataset} dataset...")

            output_file = f"{regime}_{dataset}_allocations.csv"
            output_path = os.path.join(output_dir, output_file)

            try:
                df = export_allocation_history(
                    config_path=str(config_path),
                    model_path=str(best_model_path),
                    output_path=output_path,
                    asset_names=asset_names,
                    dataset_mode=dataset,
                )

                if df is not None and not df.empty:
                    print(f"    ✅ {len(df)} records → {output_file}")
                    results[regime]["files"][dataset] = {
                        "file": output_file,
                        "records": len(df),
                        "columns": list(df.columns),
                    }
                else:
                    print(f"    ❌ No data for {dataset}")
                    results[regime]["files"][dataset] = {"error": "no_data"}

            except Exception as e:
                print(f"    ❌ Failed {dataset}: {str(e)}")
                results[regime]["files"][dataset] = {"error": str(e)}

    # Save summary
    summary_file = os.path.join(output_dir, "export_summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"✅ Export completed!")
    print(f"📁 All files saved in: {output_dir}")
    print(f"📋 Summary: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export allocation history from Mask SAC MCAD models"
    )
    parser.add_argument(
        "--regime",
        choices=["covid", "trade_war", "trade_war_i", "all"],
        default="all",
        help="MCAD regime to export allocations for (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (only used for single regime/dataset)",
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Dataset to run inference on (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="allocation_exports",
        help="Output directory for exports (default: allocation_exports)",
    )

    args = parser.parse_args()

    # Default behavior: export all regimes and all datasets
    if args.regime == "all" or args.dataset == "all":
        export_all_regimes_all_datasets(args.output_dir)
        return 0

    # Single regime, single dataset export
    regime_info = {
        "covid": {
            "config": "configs/earnmore/mask_sac_mcad_covid.py",
            "workdir": "workdir/mask_sac_mcad_covid",
            "name": "COVID",
        },
        "trade_war": {
            "config": "configs/earnmore/mask_sac_mcad_trade_war.py",
            "workdir": "workdir/mask_sac_mcad_trade_war",
            "name": "Trade War",
        },
        "trade_war_i": {
            "config": "configs/earnmore/mask_sac_mcad_trade_war_i.py",
            "workdir": "workdir/mask_sac_mcad_trade_war_i",
            "name": "Trade War I",
        },
    }

    if args.regime not in regime_info:
        print(f"❌ Unknown regime: {args.regime}")
        sys.exit(1)

    info = regime_info[args.regime]
    config_path = project_root / info["config"]
    model_dir = project_root / info["workdir"]
    best_model_path = model_dir / "best.pth"

    # Auto-generate output path if not provided
    if not args.output:
        args.output = f"{args.regime}_{args.dataset}_allocations.csv"

    print(f"🚀 Exporting allocations for {info['name']} regime")
    print(f"📁 Config: {config_path}")
    print(f"🤖 Model: {best_model_path}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"📄 Output: {args.output}")

    # Check if model exists
    if not best_model_path.exists():
        print(f"❌ Model not found: {best_model_path}")
        print("Please train the model first.")
        sys.exit(1)

    # Check if config exists
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)

    try:
        # Add asset names for MCAD (will be converted to lowercase in export)
        asset_names = ["SPY", "QQQ", "DBC", "AGG", "GLD"]

        # Export allocation history
        print("🔄 Running inference and exporting allocations...")
        df = export_allocation_history(
            config_path=str(config_path),
            model_path=str(best_model_path),
            output_path=args.output,
            asset_names=asset_names,
            dataset_mode=args.dataset,
        )

        if df is not None and len(df) > 0:
            print(f"✅ Successfully exported {len(df)} allocation records")
            print(f"📊 Columns: {list(df.columns)}")

            # Show summary statistics
            asset_cols = [
                col
                for col in df.columns
                if col.upper() in [a.upper() for a in asset_names] or col == "cash"
            ]
            if asset_cols:
                print(f"\n📈 Average allocations:")
                for asset in asset_cols:
                    if asset in df.columns:
                        avg_alloc = df[asset].mean()
                        print(f"  {asset}: {avg_alloc:.3f}")

            print(f"📁 Saved to: {args.output}")
        else:
            print("❌ No allocation data exported")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
