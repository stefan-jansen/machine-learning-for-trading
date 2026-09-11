# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Firm Characteristics Dataset
#
# Academic dataset of anonymized firm characteristics for ML-based asset pricing.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | GitHub (Chen, Pelger, Zhu 2020) |
# | **Asset Class** | US Equities (anonymized) |
# | **Frequency** | Monthly |
# | **Firms** | Anonymized |
# | **Coverage** | 1967-2016 |
# | **Size** | ~258 MB |
# | **API Key** | None (free) |
# | **Loader** | `load_firm_characteristics()` |
#
# **NOTE**: This is a **static academic dataset**. Firms are anonymized and not updateable.

# %%
"""Firm Characteristics - download, explore, and update workflow."""

import polars as pl

# %% [markdown]
# ## 1. Configuration
#
# This is a **static academic dataset** from Chen, Pelger, and Zhu (2020)
# "Deep Learning in Asset Pricing". No local configuration file.
#
# ### Dataset Characteristics
#
# - **94 firm characteristics**: Accounting ratios, technical indicators, etc.
# - **Anonymized firms**: No stock identifiers to prevent data mining
# - **Pre-split periods**: Train (1967-1989), Test (2000-2016)
# - **Gap period**: 1990-1999 excluded to prevent look-ahead bias

# %%
print("=== Firm Characteristics Configuration ===")
print("Provider: GitHub (Chen, Pelger, Zhu 2020)")
print("Paper: 'Deep Learning in Asset Pricing'")
print("Coverage: 1967-1989 (train), 2000-2016 (test)")
print("Features: 94 firm characteristics")
print("Frequency: Monthly")
print("\nThis is a static academic dataset. Firms are anonymized.")

# %% [markdown]
# ## 2. API Key Setup
#
# **No API key required.** This dataset is freely available on GitHub.

# %%
print("No API key required - data is hosted on GitHub.")
print("Source: https://github.com/jasonzy121/Deep_Learning_Asset_Pricing")

# %% [markdown]
# ## 3. Download Data
#
# `data/equities/firm_characteristics/download.py` is the only path that produces what
# `load_firm_characteristics()` reads. It fetches the archive from the Google Drive folder
# the paper's repository links, then converts the published `char/*.npz` tensors - not
# `RetChar.csv` - into `equities/firm_characteristics/firm_characteristics_{train,valid,test,all}.parquet`.
#
# The tensors are what carry firm identity. Each block has a fixed anonymous firm axis whose
# positions are persistent within the block, so the converter can emit a `symbol` column; the
# CSV drops that axis and can only emit `permno`, which the loader rejects. A split offset keeps
# the three blocks' identifier namespaces disjoint, because the archive publishes no mapping
# between them.
#
# ```bash
# uv run python data/equities/firm_characteristics/download.py           # fetch and convert
# uv run python data/equities/firm_characteristics/download.py --check   # verify what is there
# uv run python data/equities/firm_characteristics/download.py --convert # convert an existing archive
# ```
#
# This card used to carry a second downloader of its own, writing
# `firm_characteristics_{all,train,test}.parquet` into an `academic/` directory from
# `RetChar.csv`. Nothing read that directory and the loader rejects that schema, so the files
# it produced were unreachable whichever way a reader arrived at them.

# %%
from utils import ML4T_DATA_PATH
from utils.paths import display_path

parquet_dir = ML4T_DATA_PATH / "equities" / "firm_characteristics"
present = sorted(path.name for path in parquet_dir.glob("firm_characteristics_*.parquet"))

print("=== Firm Characteristics Download ===")
print("Downloader: data/equities/firm_characteristics/download.py")
print(f"Writes to:  {display_path(parquet_dir)}")
print(f"Present:    {present or 'nothing yet - run the downloader'}")

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loader throughout the book:

# %%
from data import load_firm_characteristics

# Load the full dataset
df = load_firm_characteristics()

print(f"Shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Count features (exclude timestamp, symbol, ret)
feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol", "ret"]]
print(f"Features: {len(feature_cols)}")
print(f"Memory: {df.estimated_size('mb'):.1f} MB")

# %%
# Schema
df.schema

# %%
# Preview
df.head(10)

# %% [markdown]
# ### Feature Overview

# %%
# Feature statistics
feature_cols = [c for c in df.columns if c not in ["timestamp", "symbol", "split", "ret"]]
print(f"Number of features: {len(feature_cols)}")
print("\nFeature names (first 20):")
for col in feature_cols[:20]:
    print(f"  {col}")
if len(feature_cols) > 20:
    print(f"  ... and {len(feature_cols) - 20} more")

# %% [markdown]
# ### Train/Test Split Coverage

# %%
# Yearly coverage
yearly = (
    df.with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.len().alias("n_observations"),
    )
    .sort("year")
)
print("Yearly coverage:")
yearly

# %% [markdown]
# ## 5. Data Profile

# %%
from ml4t.data.storage.data_profile import load_profile

from utils import ML4T_DATA_PATH

profile_path = (
    ML4T_DATA_PATH / "equities" / "firm_characteristics" / "firm_characteristics_all_profile.json"
)
profile = load_profile(profile_path)

if profile is None:
    print(f"No profile at {display_path(profile_path)}")
    print(
        "The downloader above writes it, next to the data, through\n"
        "ml4t.data.storage.data_profile. Re-run it to produce one; there is no separate\n"
        "profile-generating script."
    )
else:
    print("=== Firm Characteristics Profile ===")
    print(f"Written by {profile.source}")
    print(profile.summary())

# %% [markdown]
# ## 6. Loader Options
#
# The loader supports loading the full dataset or pre-defined splits:

# %%
# Load train split (1967-1989)
train = load_firm_characteristics(split="train")
print(f"Train: {train.shape}, {train['timestamp'].min()} to {train['timestamp'].max()}")

# %%
# Load test split (2000-2016)
test = load_firm_characteristics(split="test")
print(f"Test: {test.shape}, {test['timestamp'].min()} to {test['timestamp'].max()}")

# %%
# Load full dataset (default)
full = load_firm_characteristics()
print(f"Full: {full.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### Source
#
# - **Paper**: Chen, Pelger, Zhu (2020) "Deep Learning in Asset Pricing"
# - **GitHub**: https://github.com/jasonzy121/Deep_Learning_Asset_Pricing
# - **Published**: Management Science, 2024
#
# ### Dataset Columns
#
# | Column | Description |
# |--------|-------------|
# | `date` | Month-end date |
# | `permno` | Anonymized firm identifier |
# | `ret` | Monthly stock return |
# | `me` | Market equity |
# | `bm` | Book-to-market ratio |
# | `mom12m` | 12-month momentum |
# | `... (94 total)` | Various firm characteristics |
#
# ### Train/Test Split
#
# | Split | Period | Purpose |
# |-------|--------|---------|
# | Train | 1967-1989 | Model training |
# | Gap | 1990-1999 | Excluded (prevents look-ahead) |
# | Test | 2000-2016 | Out-of-sample evaluation |
#
# ### Data Quality Notes
#
# - **Anonymized firms**: No stock identifiers to prevent data mining
# - **Cross-sectional ranking**: Features are rank-transformed
# - **Missing values**: Some characteristics have gaps
# - **Survivorship**: Includes delisted firms

# %% [markdown]
# ## 8. Updating Data
#
# **This dataset is NOT updateable.**
#
# This is a static academic dataset published with a research paper.
# The data cannot be extended beyond the original publication period (2016).
#
# ### Related Resources
#
# For more recent firm characteristics data, consider:
#
# | Resource | Coverage | Access |
# |----------|----------|--------|
# | WRDS/CRSP | 1926-present | Subscription |
# | Open Source Asset Pricing | Varies | Free |
# | Ken French Library | 1926-present | Free (factors only) |

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Features | 94 firm characteristics |
# | Frequency | Monthly |
# | Coverage | 1967-2016 (with gap 1990-1999) |
# | Provider | GitHub (Chen, Pelger, Zhu 2020) |
# | Loader | `load_firm_characteristics(split=None)` |
#
# **Primary use**: Deep learning asset pricing research (Chapters 10-15).
# **Limitation**: Anonymized firms, static dataset ends 2016.
