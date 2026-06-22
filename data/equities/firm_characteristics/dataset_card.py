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

import json
import os
import sys
import zipfile
from pathlib import Path

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
# The dataset is hosted on Google Drive via the GitHub repository.
# Download can be automatic (with `gdown`) or manual.

# %%
# Google Drive file ID for the data archive
GDRIVE_FILE_ID = "1nYHpJ2lNm-qDX5iq18-HaL1H6z7lPGVi"
GITHUB_REPO = "https://github.com/jasonzy121/Deep_Learning_Asset_Pricing"


def download_firm_characteristics(dry_run: bool = False, force: bool = False, convert: bool = True):
    """Download Chen-Pelger-Zhu firm characteristics dataset.

    Args:
        dry_run: If True, show what would be downloaded without doing it
        force: If True, re-download even if data exists
        convert: If True, convert CSV to parquet after download
    """
    from utils import ML4T_DATA_PATH

    output_dir = ML4T_DATA_PATH / "academic"
    dl_dir = output_dir / "dl_asset_pricing"
    parquet_path = output_dir / "firm_characteristics_all.parquet"

    print("=== Firm Characteristics Download ===")
    print("Dataset: Chen-Pelger-Zhu (2020)")
    print("Coverage: 1967-1989 (train), 2000-2016 (test)")
    print("Features: 94 firm characteristics + returns")
    print("Estimated size: ~1.5 GB (archive), ~258 MB (parquet)")
    print(f"Output: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        print("  - data.zip from Google Drive (~1.5 GB)")
        print(f"  - Extract to: {dl_dir}")
        print(f"  - Convert to parquet: {parquet_path}")
        print("\nManual download:")
        print(f"  1. Go to: {GITHUB_REPO}")
        print("  2. Download data.zip from Google Drive link")
        print(f"  3. Extract to: {dl_dir}")
        return

    # Check existing
    if parquet_path.exists() and not force:
        existing = pl.read_parquet(parquet_path)
        print(f"\nData already exists ({len(existing):,} rows).")
        print("Use force=True to re-download.")
        return

    # Try automatic download with gdown
    try:
        import gdown
    except ImportError:
        print("\nWARNING: gdown not installed for automatic download")
        print("Install with: pip install gdown")
        print("\nManual download instructions:")
        print(f"  1. Go to: {GITHUB_REPO}")
        print("  2. Download data.zip from Google Drive link")
        print(f"  3. Extract to: {dl_dir}")
        print("  4. Run: download_firm_characteristics(convert=True)")
        return

    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "data.zip"

    print("\nDownloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        print("ERROR: Download failed")
        return

    print("\nExtracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dl_dir)

    # Cleanup zip
    zip_path.unlink()

    print("Download complete!")

    # Convert to parquet
    if convert:
        _convert_to_parquet(output_dir)


def _convert_to_parquet(output_dir: Path):
    """Convert RetChar.csv to parquet format with train/test splits."""
    dl_dir = output_dir / "dl_asset_pricing"
    retchar_path = dl_dir / "RetChar.csv"

    if not retchar_path.exists():
        # Try nested path
        nested_path = dl_dir / "data" / "RetChar.csv"
        if nested_path.exists():
            retchar_path = nested_path
        else:
            print(f"ERROR: RetChar.csv not found at {retchar_path}")
            return

    print("\nConverting to parquet format...")

    # Read CSV
    df = pl.read_csv(retchar_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Date is in YYYYMMDD format (integer)
    df = df.with_columns(
        pl.col("Date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("date"),
        pl.col("Date").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("year"),
    )

    # Rename permno column if it exists differently
    if "Permno" in df.columns:
        df = df.rename({"Permno": "permno"})
    if "RET" in df.columns:
        df = df.rename({"RET": "ret"})

    # Create splits based on paper
    train_df = df.filter(pl.col("year") < 1990)
    test_df = df.filter(pl.col("year") >= 2000)

    # Drop helper columns
    train_df = train_df.drop(["Date", "year"])
    test_df = test_df.drop(["Date", "year"])
    all_df = df.drop(["Date", "year"])

    # Save parquet files
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df.write_parquet(output_dir / "firm_characteristics_all.parquet")
    train_df.write_parquet(output_dir / "firm_characteristics_train.parquet")
    test_df.write_parquet(output_dir / "firm_characteristics_test.parquet")

    print("  Created:")
    print(f"    firm_characteristics_all.parquet: {len(all_df):,} rows")
    print(f"    firm_characteristics_train.parquet: {len(train_df):,} rows (1967-1989)")
    print(f"    firm_characteristics_test.parquet: {len(test_df):,} rows (2000-2016)")


# %% [markdown]
# ### Download

# %%
# Uncomment to download
# download_firm_characteristics()

# %% [markdown]
# ### Dry Run (Preview)

# %%
download_firm_characteristics(dry_run=True)

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
from utils import ML4T_DATA_PATH

# Check for existing profile
profile_path = ML4T_DATA_PATH / "academic" / "firm_characteristics_profile.json"

if profile_path.exists():
    profile = json.loads(profile_path.read_text())
    print("=== Firm Characteristics Profile ===")
    print(f"Rows: {profile['rows']:,}")
    print(f"Columns: {profile['columns']}")
    print(f"Memory: {profile['memory_mb']:.1f} MB")
else:
    print(f"Profile not found at {profile_path}")
    print("Generate with: python generate_profiles.py --dataset firm_characteristics")

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
