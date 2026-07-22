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
- **Coverage**: 1967-01 to 2016-12, monthly observations, ~1.2M rows
- **Features**: 46 firm characteristics (accounting ratios, price-based
  measures, momentum variants), 178 macro indicators, forward returns
- **Size on disk**: ~1.1 GB raw CSV; converted to ~500 MB parquet
- **Access**: Public, no API key required
- **Canonical schema**: `symbol` (anonymous integer id), `timestamp`
  (monthly Date), 46 characteristic columns, `ret`, `split`
- **Identity scope**: `symbol` is persistent within each tensor released by
  the authors. Separate numeric namespaces prevent accidental linking across
  tensors because the archive does not publish a cross-split map.

## Pre-defined Splits

The dataset ships with deterministic train/valid/test splits aligned to
the authors' released tensors:

| Split | Period    |
| ----- | --------- |
| train | 1967-1986 |
| valid | 1987-1991 |
| test  | 1992-2016 |

## Download

This is the largest free dataset in the book: ~1.5 GB pulled from the
authors' Google Drive folder (RetChar.csv alone is ~1.1 GB), followed by
a tensor-to-Parquet conversion. The tensor conversion recovers persistent
anonymous firm identifiers that the flattened CSV omits. How long it takes
depends on your bandwidth and disk speed. Per-file progress is printed as it
runs. A single command downloads and converts; no separate `--convert` pass
is needed.

```bash
# Download the ~1.5 GB folder and convert to parquet in one step
uv run python data/equities/firm_characteristics/download.py

# Verify what's already on disk, do not refetch
uv run python data/equities/firm_characteristics/download.py --check

# Force a re-download even if files exist
uv run python data/equities/firm_characteristics/download.py --force

# Re-run only the tensor-to-Parquet conversion (files already downloaded)
uv run python data/equities/firm_characteristics/download.py --convert
```

It is also fetched automatically as part of `data/download_all.py`. To
skip it there (e.g. on a metered or space-constrained connection), pass
`--skip-firm-characteristics`.

Output layout under `$ML4T_DATA_PATH/equities/firm_characteristics/`:

```
firm_characteristics_all.parquet      # full panel
firm_characteristics_train.parquet    # 1967-1986
firm_characteristics_valid.parquet    # 1987-1991
firm_characteristics_test.parquet     # 1992-2016
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

- Chapters 10-16 - standard benchmark for linear and nonlinear
  asset-pricing models.
- `case_studies/us_firm_characteristics/` - Chen-Pelger-Zhu replication
  pipeline (CV, GBM, latent factor, deep models).
