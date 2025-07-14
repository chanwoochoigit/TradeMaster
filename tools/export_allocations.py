#!/usr/bin/env python3
"""
Export allocation history from trained portfolio management models to CSV.

Supports multiple models: mask_sac, eiie, deeptrader

Default behavior exports ALL models for ALL regimes:
    python tools/export_allocations.py

This creates the following directory structure:
    allocation_exports_YYYYMMDD_HHMMSS/
    ├── covid/
    │   ├── mask_sac.csv
    │   ├── eiie.csv
    │   └── deeptrader.csv
    ├── trade_war/
    │   ├── mask_sac.csv
    │   ├── eiie.csv
    │   └── deeptrader.csv
    ├── trade_war_i/
    │   ├── mask_sac.csv
    │   ├── eiie.csv
    │   └── deeptrader.csv
    └── export_summary.json

Single model examples:
    python tools/export_allocations.py --model mask_sac
    python tools/export_allocations.py --model eiie --regime covid
    python tools/export_allocations.py --model deeptrader --regime trade_war --dataset test

Output format: date,dataset,spy,qqq,dbc,agg,gld,cash (lowercase, cash at end)
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pm.utils.export import export_allocation_history


def get_model_info():
    """Return model configuration mapping"""
    return {
        "mask_sac": {
            "name": "Mask SAC",
            "regimes": {
                "covid": {
                    "config": "configs/earnmore/mask_sac_mcad_covid.py",
                    "workdir": "workdir/mask_sac_mcad_covid",
                },
                "trade_war": {
                    "config": "configs/earnmore/mask_sac_mcad_trade_war.py",
                    "workdir": "workdir/mask_sac_mcad_trade_war",
                },
                "trade_war_i": {
                    "config": "configs/earnmore/mask_sac_mcad_trade_war_i.py",
                    "workdir": "workdir/mask_sac_mcad_trade_war_i",
                },
            }
        },
        "eiie": {
            "name": "EIIE",
            "regimes": {
                "covid": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_covid.py",
                    "workdir": "work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_covid",
                },
                "trade_war": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war.py",
                    "workdir": "work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war",
                },
                "trade_war_i": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i.py",
                    "workdir": "work_dir/portfolio_management_mcad_eiie_eiie_adam_mse_trade_war_i",
                },
            }
        },
        "deeptrader": {
            "name": "DeepTrader",
            "regimes": {
                "covid": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_deeptrader_deeptrader_adam_mse_covid.py",
                    "workdir": "work_dir/portfolio_management_mcad_deeptrader_deeptrader_adam_mse_covid",
                },
                "trade_war": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_deeptrader_deeptrader_adam_mse_trade_war.py",
                    "workdir": "work_dir/portfolio_management_mcad_deeptrader_deeptrader_adam_mse_trade_war",
                },
                "trade_war_i": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_deeptrader_deeptrader_adam_mse_trade_war_i.py",
                    "workdir": "work_dir/portfolio_management_mcad_deeptrader_deeptrader_adam_mse_trade_war_i",
                },
            }
        },
        "sarl": {
            "name": "SARL",
            "regimes": {
                "covid": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_sarl_sarl_adam_mse_covid.py",
                    "workdir": "work_dir/portfolio_management_mcad_sarl_sarl_adam_mse_covid",
                },
                "trade_war": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_sarl_sarl_adam_mse_trade_war.py",
                    "workdir": "work_dir/portfolio_management_mcad_sarl_sarl_adam_mse_trade_war",
                },
                "trade_war_i": {
                    "config": "configs/portfolio_management/portfolio_management_mcad_sarl_sarl_adam_mse_trade_war_i.py",
                    "workdir": "work_dir/portfolio_management_mcad_sarl_sarl_adam_mse_trade_war_i",
                },
            }
        }
    }


def export_all_models_all_regimes(output_dir="exports", models=None, regimes=None):
    """Export allocations for all models and regimes with clean separation"""
    import tempfile

    # Use consistent exports directory structure
    os.makedirs(output_dir, exist_ok=True)

    model_info = get_model_info()
    datasets = ["train", "val", "test"]
    asset_names = ["SPY", "QQQ", "DBC", "AGG", "GLD"]

    # Filter models and regimes if specified
    if models is None:
        models = list(model_info.keys())
    if regimes is None:
        regimes = ["covid", "trade_war", "trade_war_i"]

    results = {}

    print(f"\n🚀 Exporting allocations for models: {', '.join(models)}")
    print(f"📊 Regimes: {', '.join(regimes)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📂 Structure: {output_dir}/{{regime}}/{{model}}.csv")
    print("=" * 80)

    # Process each regime, then each model within that regime
    for regime in regimes:
        regime_name = regime.replace("_", " ").title()
        print(f"\n📊 Processing {regime_name} regime...")
        
        if regime not in results:
            results[regime] = {}
        
        # Create regime directory
        regime_dir = os.path.join(output_dir, regime)
        os.makedirs(regime_dir, exist_ok=True)
        
        # Process each model for this regime
        for model_key in models:
            if model_key not in model_info:
                print(f"  ❌ Unknown model: {model_key}")
                continue
                
            model_config = model_info[model_key]
            model_name = model_config["name"]
            
            print(f"  🤖 {model_name}...")
            
            if regime not in model_config["regimes"]:
                print(f"    ❌ {model_name}: No config for {regime}")
                results[regime][model_key] = {"status": "failed", "reason": "no_config"}
                continue

            regime_config = model_config["regimes"][regime]
            config_path = project_root / regime_config["config"]
            model_workdir = project_root / regime_config["workdir"]
            best_model_path = model_workdir / "best.pth"

            # Check if model exists
            if not best_model_path.exists():
                print(f"    ❌ Model not found: {best_model_path}")
                results[regime][model_key] = {"status": "failed", "reason": "model_not_found"}
                continue

            results[regime][model_key] = {"status": "success", "datasets": {}}
            combined_dfs = []

            # Export each dataset and collect DataFrames
            for dataset in datasets:
                print(f"    📈 Exporting {dataset} dataset...")

                # Use temporary file for individual dataset export
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                    temp_path = tmp_file.name

                try:
                    df = export_allocation_history(
                        config_path=str(config_path),
                        model_path=str(best_model_path),
                        output_path=temp_path,
                        asset_names=asset_names,
                        dataset_mode=dataset,
                    )

                    if df is not None and not df.empty:
                        # Add dataset column to identify train/val/test
                        df['dataset'] = dataset
                        combined_dfs.append(df)
                        
                        print(f"      ✅ {len(df)} records from {dataset}")
                        results[regime][model_key]["datasets"][dataset] = {
                            "records": len(df),
                            "date_range": f"{df['date'].min()} to {df['date'].max()}",
                        }
                    else:
                        print(f"      ❌ No data for {dataset}")
                        results[regime][model_key]["datasets"][dataset] = {"error": "no_data"}

                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

                except Exception as e:
                    print(f"      ❌ Failed {dataset}: {str(e)}")
                    results[regime][model_key]["datasets"][dataset] = {"error": str(e)}
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            # Combine all datasets for this model/regime
            if combined_dfs:
                print(f"    🔗 Combining {len(combined_dfs)} datasets...")
                
                # Concatenate all datasets
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                
                # Sort by date to maintain chronological order
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df = combined_df.sort_values('date')
                combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
                
                # Reorder columns to put dataset right after date
                cols = ['date', 'dataset'] + [col for col in combined_df.columns if col not in ['date', 'dataset']]
                combined_df = combined_df[cols]
                
                # Save with simple model name as filename
                output_file = f"{model_key}.csv"
                output_path = os.path.join(regime_dir, output_file)
                combined_df.to_csv(output_path, index=False)
                
                total_records = len(combined_df)
                print(f"    ✅ {model_name}: {total_records} records → {regime}/{output_file}")
                
                results[regime][model_key]["combined_file"] = {
                    "file": f"{regime}/{output_file}",
                    "total_records": total_records,
                    "columns": list(combined_df.columns),
                    "datasets_included": list(combined_df['dataset'].unique()),
                }
            else:
                print(f"    ❌ {model_name}: No data exported")
                results[regime][model_key]["status"] = "failed"
                results[regime][model_key]["reason"] = "no_datasets_exported"

    # Save summary
    summary_file = os.path.join(output_dir, "export_summary.json")
    summary_data = {
        "export_timestamp": datetime.now().isoformat(),
        "models_exported": models,
        "regimes_exported": regimes,
        "results": results
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("=" * 80)
    print(f"✅ Export completed!")
    print(f"📁 All files saved in: {output_dir}")
    print(f"📋 Summary: {summary_file}")
    
    # Print file structure summary
    print(f"\n📂 File structure:")
    for regime in regimes:
        if regime in results:
            print(f"  {regime}/")
            for model_key in models:
                if model_key in results[regime] and results[regime][model_key].get("status") == "success":
                    print(f"    ├── {model_key}.csv")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export allocation history from portfolio management models"
    )
    parser.add_argument(
        "--model",
        choices=["mask_sac", "eiie", "deeptrader", "sarl", "all"],
        default="all",
        help="Model to export allocations for (default: all)",
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
        help="Output CSV file path (only used for single model/regime/dataset)",
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Dataset to run inference on (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Output directory for exports (default: exports)",
    )

    args = parser.parse_args()

    # Determine models and regimes to export
    models = [args.model] if args.model != "all" else ["mask_sac", "eiie", "deeptrader", "sarl"]
    regimes = [args.regime] if args.regime != "all" else ["covid", "trade_war", "trade_war_i"]

    # Default behavior: export multiple models/regimes with clean separation
    if len(models) > 1 or len(regimes) > 1 or args.dataset == "all":
        export_all_models_all_regimes(args.output_dir, models, regimes)
        return 0

    # Single model, single regime, single dataset export
    model_info = get_model_info()
    model_key = models[0]
    regime = regimes[0]

    if model_key not in model_info:
        print(f"❌ Unknown model: {model_key}")
        sys.exit(1)

    model_config = model_info[model_key]
    if regime not in model_config["regimes"]:
        print(f"❌ No config for {model_config['name']} in {regime} regime")
        sys.exit(1)

    regime_config = model_config["regimes"][regime]
    config_path = project_root / regime_config["config"]
    model_dir = project_root / regime_config["workdir"]
    best_model_path = model_dir / "best.pth"

    # Auto-generate output path if not provided
    if not args.output:
        args.output = f"{model_key}_{regime}_{args.dataset}_allocations.csv"

    print(f"🚀 Exporting allocations for {model_config['name']} - {regime.replace('_', ' ').title()} regime")
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
