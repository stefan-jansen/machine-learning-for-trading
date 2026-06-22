# Firm Characteristics (Chen-Pelger-Zhu 2020)

Anonymized panel of ~1.2M stock-month observations with 46 firm
characteristics and forward returns, spanning 1967-2016. Built from the
replication archive of Chen, Pelger, and Zhu (2020), *Deep Learning in
Asset Pricing*. Used throughout the book for ML-based asset-pricing
examples where a standard, reproducible benchmark matters more than
symbol-level interpretation.

## Dataset

- **Source**: GitHub replication repo
  (https://github.com/jasonzy121/Deep_Learning_Asset_Pricing), which
  itself ships the published dataset via Google Drive
- **Coverage**: 1967-01 → 2016-12, monthly observations, ~1.2M rows
- **Features**: 46 firm characteristics (accounting ratios, price-based
  measures, momentum variants), 178 macro indicators, forward returns
- **Size on disk**: ~1.1 GB raw CSV; converted to ~500 MB parquet
- **Access**: Public, no API key required
- **Canonical schema**: `symbol` (anonymous integer id), `timestamp`
  (monthly Date), 46 characteristic columns, `return`, `split`

## Pre-defined Splits

The dataset ships with deterministic train/valid/test splits aligned to
the original paper:

| Split | Period       | Share |
| ----- | ------------ | ----- |
| train | 1967-1989    | ~70%  |
| valid | 1990-1999    | ~15%  |
| test  | 2000-2016    | ~15%  |

## Download

```bash
# Pull raw CSVs + NPZ arrays (~1.1 GB over the network)
uv run python data/equities/firm_characteristics/download.py

# Verify what's already on disk, do not refetch
uv run python data/equities/firm_characteristics/download.py --check

# Force a re-download even if files exist
uv run python data/equities/firm_characteristics/download.py --force

# Convert the raw CSVs to the canonical firm_characteristics_*.parquet
# files that the loader consumes (run once after a fresh download)
uv run python data/equities/firm_characteristics/download.py --convert
```

Output layout under `$ML4T_DATA_PATH/equities/firm_characteristics/`:

```
firm_characteristics_all.parquet      # full panel
firm_characteristics_train.parquet    # 1967-1989
firm_characteristics_test.parquet     # 2000-2016
dl_asset_pricing/                     # raw CSV + NPZ staging
    RetChar.csv
    Macro.csv
    char/Char_{train,valid,test}.npz
    macro/macro_{train,valid,test}.npz
    RF/RF_{train,valid,test}_normalized_task_1.npz
```

The staging directory is kept so experiments that need the pre-split
NPZ arrays (the original Chen-Pelger-Zhu format) can read them
directly.

## Loading

```python
from data import load_firm_characteristics

df = load_firm_characteristics()                        # full panel
df = load_firm_characteristics(split="train")
df = load_firm_characteristics(split="test", include_macro=True)
```

If the canonical parquets are missing, the loader raises
`DataNotFoundError` pointing at the download command.

## Consumers

- Chapters 10-16 — standard benchmark for linear and nonlinear
  asset-pricing models.
- `case_studies/us_firm_characteristics/` — Chen-Pelger-Zhu replication
  pipeline (CV, GBM, latent factor, deep models).
