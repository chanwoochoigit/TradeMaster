# Mask SAC MCAD Pipeline Setup Guide

This guide provides instructions for setting up and running the complete Mask SAC MCAD (Multi-regime Crisis Asset Discovery) pipeline.

## 🚀 Quick Start

### Option 1: Use Updated Requirements (Recommended)

```bash
# Install all dependencies for the complete TradeMaster project
pip install -r requirements.txt

# OR install only Mask SAC specific dependencies  
pip install -r mask_sac_requirements.txt
```

### Option 2: Manual Installation (if pip install fails)

```bash
# Core dependencies that must be exact versions
pip install gym==0.17.3
pip install mmengine==0.7.2
pip install "einops>=0.6.0"
pip install timm==0.9.2
pip install seaborn
pip install tensorboard
pip install iopath
pip install prettytable
```

## 📋 Requirements Summary

### Critical Version Dependencies
These versions are **required** for compatibility:

- `gym==0.17.3` (NOT 0.21.0 or 0.26.x)
- `mmengine==0.7.2`
- `einops>=0.6.0` (NOT 0.3.0)
- `timm==0.9.2`

### Environment
- Python 3.9+ 
- PyTorch 1.13.1+
- TensorFlow 2.11.0+

## 🏃‍♂️ Running the Pipeline

### Complete Pipeline (All 3 Regimes)

```bash
# Full training (200 episodes per regime, ~6 hours)
python mask_sac_mcad_complete_pipeline.py

# Quick test mode (50 episodes per regime, ~1 hour)  
python mask_sac_mcad_complete_pipeline.py --quick-test
```

### Check Training Status

```bash
# Check all regimes
python mask_sac_mcad_complete_pipeline.py --check-status all

# Check specific regime
python mask_sac_mcad_complete_pipeline.py --check-status covid
```

### Export Allocations Only

```bash
# Export all regimes and datasets
python tools/export_allocations.py

# Export specific regime
python tools/export_allocations.py --regime covid --dataset test
```

## 📊 Pipeline Stages

1. **Data Structure Setup** - Prepares MCAD dataset for 3 regimes:
   - `covid` (2020-2021 COVID-19 pandemic)
   - `trade_war` (2018-2019 US-China trade war)  
   - `trade_war_i` (2016-2017 pre-trade war)

2. **Configuration Creation** - Generates training configs for each regime

3. **Model Training** - Trains Mask SAC agents on portfolio management
   - Uses 5 assets: SPY, QQQ, DBC, AGG, GLD
   - 20 technical indicators + 3 temporal features
   - Transformer-based masked autoencoder architecture

4. **Allocation Export** - Exports trained model allocation histories to CSV

5. **Summary Generation** - Creates comprehensive pipeline report

## 📁 Output Files

### Training Outputs
```
workdir/
├── mask_sac_mcad_covid/
│   ├── best.pth                 # Trained model checkpoint
│   ├── train_log.txt           # Training metrics
│   └── events.out.tfevents.*   # Tensorboard logs
├── mask_sac_mcad_trade_war/
└── mask_sac_mcad_trade_war_i/
```

### Allocation Exports
```
allocation_exports_YYYYMMDD_HHMMSS/
├── covid_train_allocations.csv
├── covid_val_allocations.csv  
├── covid_test_allocations.csv
├── trade_war_train_allocations.csv
├── trade_war_val_allocations.csv
├── trade_war_test_allocations.csv
├── trade_war_i_train_allocations.csv
├── trade_war_i_val_allocations.csv
├── trade_war_i_test_allocations.csv
└── export_summary.json
```

### CSV Format
```csv
date,spy,qqq,dbc,agg,gld,cash
2020-01-02,0.2,0.3,0.1,0.2,0.1,0.1
2020-01-03,0.25,0.25,0.15,0.15,0.1,0.1
...
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'mmengine'**
   ```bash
   pip install mmengine==0.7.2
   ```

2. **ValueError: too many values to unpack (expected 2)**  
   ```bash
   pip install gym==0.17.3  # Downgrade from 0.26.x
   ```

3. **AttributeError: module 'keras.backend' has no attribute 'is_tensor'**
   ```bash
   pip install "einops>=0.6.0"  # Upgrade from 0.3.0
   ```

4. **No module named 'seaborn'**
   ```bash
   pip install seaborn
   ```

### Verify Installation
```bash
python -c "
from mmengine.config import Config
import einops, timm, seaborn, gym
print('✅ All imports successful!')
print(f'gym version: {gym.__version__}')
print(f'einops version: {einops.__version__}')
"
```

## 📈 Performance Notes

- **Training Time**: ~1.5 hours per regime (quick test), ~4-6 hours (full)
- **Memory**: ~4-8GB GPU memory recommended  
- **CPU**: Multi-core recommended for faster training
- **Storage**: ~2GB for models + logs + exports

## 🔧 Configuration

### Training Parameters (Configurable)
```python
num_episodes = 50      # Quick test mode
num_episodes = 200     # Full training mode  
batch_size = 64
lr = 5e-5
embed_dim = 128
```

### Asset Configuration
Currently configured for 5 assets:
- **SPY**: S&P 500 ETF
- **QQQ**: Nasdaq ETF  
- **DBC**: Commodities ETF
- **AGG**: Aggregate Bond ETF
- **GLD**: Gold ETF

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify all dependencies are correct versions
3. Check training logs in `workdir/mask_sac_mcad_*/train_log.txt`
4. Use `--check-status` to monitor training progress

---

*Last updated: December 2024*
*Compatible with TradeMaster v1.0.0* 