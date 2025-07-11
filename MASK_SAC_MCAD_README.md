# Mask SAC MCAD Complete Pipeline

This repository provides a comprehensive end-to-end pipeline for training Mask SAC (Soft Actor-Critic with Masking) on MCAD (Multi-regime Crisis Asset Dynamics) datasets and exporting allocation histories.

## Overview

The pipeline handles 3 different market regimes:
- **COVID** (2020 pandemic period)
- **Trade War** (US-China trade tensions)
- **Trade War I** (Initial trade war phase)

Each regime contains 5 assets: SPY, QQQ, DBC, AGG, GLD

## Features

✅ **Automated Data Setup**: Converts MCAD data to required format  
✅ **Configuration Generation**: Creates training configs for all regimes  
✅ **Training Management**: Monitors training progress with timeouts  
✅ **Allocation Export**: Generates CSV files with portfolio allocations  
✅ **Comprehensive Reporting**: JSON summary and detailed logs  
✅ **Quick Test Mode**: Reduced episodes for development/testing  

## Quick Start

### Full Training (Recommended)
```bash
python mask_sac_mcad_complete_pipeline.py
```

### Quick Test (50 episodes)
```bash
python mask_sac_mcad_complete_pipeline.py --quick-test
```

## What the Pipeline Does

### 1. Data Structure Setup
- Validates MCAD data files exist
- Creates individual stock CSV files
- Generates stocks.txt and aux_stocks configuration
- Adds temporal features (weekday, day, month)

### 2. Configuration Creation  
- Generates training configs for each regime
- Sets up proper data paths and parameters
- Configures Mask SAC agent with transformer architecture

### 3. Model Training
- Trains separate models for each regime
- Monitors training progress with checkpoints
- Handles timeouts and failures gracefully
- Saves best model weights

### 4. Allocation Export
- Loads trained models
- Runs backtesting on historical data
- Exports portfolio allocation history to CSV
- Includes proper asset names and dates

### 5. Summary Generation
- Creates comprehensive JSON report
- Shows success/failure for each regime
- Lists output file locations
- Provides timing information

## Output Files

After successful completion, you'll find:

```
workdir/
├── mask_sac_mcad_covid/
│   ├── best.pth                    # Trained model
│   ├── mask_sac_mcad_covid.csv    # Allocation history
│   └── train_log.txt              # Training logs
├── mask_sac_mcad_trade_war/
│   ├── best.pth
│   ├── mask_sac_mcad_trade_war.csv
│   └── train_log.txt
├── mask_sac_mcad_trade_war_i/
│   ├── best.pth
│   ├── mask_sac_mcad_trade_war_i.csv
│   └── train_log.txt
└── pipeline_summary.json          # Overall results
```

## Allocation CSV Format

Each allocation CSV contains:
- **date**: Trading date (lowercase)
- **spy, qqq, dbc, agg, gld**: Asset allocation percentages (lowercase)
- **cash**: Cash allocation percentage (final column)

Example:
```csv
date,spy,qqq,dbc,agg,gld,cash
2007-09-26,0.25,0.20,0.15,0.15,0.10,0.15
2007-09-27,0.28,0.22,0.13,0.16,0.09,0.12
...
```

**Export Commands**:
```bash
# Export ALL regimes and ALL datasets (default)
python tools/export_allocations.py

# Single regime/dataset
python tools/export_allocations.py --regime covid --dataset test --output covid_test.csv
```

## Training Parameters

### Default (Full Training)
- **Episodes**: 1000 per regime
- **Training Time**: Up to 6 hours per regime
- **Batch Size**: 64
- **Buffer Size**: 4096
- **Horizon Length**: 64

### Quick Test Mode
- **Episodes**: 50 per regime  
- **Training Time**: Up to 1 hour per regime
- **Other parameters**: Same as full training

## Architecture Details

### Mask SAC Agent
- **Representation Network**: MaskTimeState with transformer
- **Actor Network**: ActorMaskSAC
- **Critic Network**: CriticMaskSAC
- **Masking**: Variable masking ratio (0.6-0.8)
- **Optimizer**: AdamW with learning rate scheduling

### Data Features
- **Technical Indicators**: 17 features (OHLCV + normalized versions)
- **Temporal Features**: 3 features (weekday, day, month)
- **Total Features**: 20 per timestep
- **Lookback Window**: 10 days

## Monitoring and Debugging

### Real-time Monitoring
The pipeline provides timestamped logs for:
- Data setup progress
- Configuration creation
- Training status updates
- Export progress
- Error messages with details

### Troubleshooting

**Training Fails to Start**
- Check CUDA/GPU availability
- Verify data files exist in `data/portfolio_management/mcad/`
- Ensure sufficient disk space

**Training Times Out**  
- Reduce episodes with `--quick-test`
- Check GPU memory usage
- Monitor system resources

**Export Fails**
- Verify `best.pth` checkpoint exists
- Check model compatibility
- Review error logs in workdir

## System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible (recommended)
- **Memory**: 8GB+ RAM
- **Storage**: 2GB+ free space
- **Dependencies**: PyTorch, pandas, numpy, mmengine

## Advanced Usage

### Custom Parameters
Edit the pipeline script to modify:
- Number of episodes
- Batch size
- Learning rates
- Network architecture
- Training timeouts

### Individual Regime Training
```bash
python tools/earnmore/train.py --config configs/earnmore/mask_sac_mcad_covid.py
```

### Manual Export
```bash
python tools/export_allocations.py --config configs/earnmore/mask_sac_mcad_covid.py
```

## Results Interpretation

### Training Metrics
- **ARR%**: Annualized Return Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest loss from peak
- **Volatility**: Return standard deviation

### Allocation Analysis
- **Diversification**: Check allocation spread across assets
- **Temporal Patterns**: Analyze regime-specific behaviors
- **Risk Management**: Monitor cash allocation levels
- **Asset Rotation**: Track allocation changes over time

## Support

For issues or questions:
1. Check the pipeline summary JSON for detailed error information
2. Review training logs in the workdir
3. Verify data integrity and system requirements
4. Test with `--quick-test` mode first

---

**Note**: This pipeline is designed for research and educational purposes. The allocation strategies should be thoroughly backtested before any real trading applications. 