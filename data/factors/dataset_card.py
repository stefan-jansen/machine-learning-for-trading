# %% [markdown]
# # Academic Factor Data Dataset
#
# Fama-French and AQR factor returns for benchmarking and risk adjustment.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | Ken French Library, AQR |
# | **Asset Class** | Factor Returns |
# | **Frequency** | Monthly (daily available) |
# | **Factors** | FF3, FF5, Momentum, QMJ, BAB |
# | **Coverage** | 1926-present (FF), varies (AQR) |
# | **Size** | ~5 MB |
# | **API Key** | None (free) |
# | **Loader** | `load_ff_factors()`, `load_aqr_factors()` |

# %%
"""Academic Factor Data - download, explore, and update workflow."""

import json
from pathlib import Path

import polars as pl

# %% [markdown]
# ## 1. Configuration
#
# Academic factor data is **provider-defined** (no local config file). Each provider
# maintains their own factor definitions and data format.

# %%
print("=== Academic Factor Configuration ===")
print("\nFama-French (Ken French Library):")
print("  - FF3: Mkt-RF, SMB, HML")
print("  - FF5: FF3 + RMW, CMA")
print("  - Momentum: MOM")
print("  - Coverage: 1926-present")
print("\nAQR Research:")
print("  - QMJ: Quality Minus Junk")
print("  - BAB: Betting Against Beta")
print("  - VME: Value Minus Everything")
print("  - HML Devil: Industry-adjusted value")
print("  - Coverage: varies by factor")

# %% [markdown]
# ## 2. API Key Setup
#
# **No API key required.** Both Ken French Library and AQR provide free public access.

# %%
print("Ken French Library: Free, no API key required")
print("  URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
print("\nAQR Research: Free, no API key required")
print("  URL: https://www.aqr.com/Insights/Datasets")

# %% [markdown]
# ## 3. Download Data
#
# The `ml4t-data` library handles downloading and caching factor data.


# %%
def download_ff_factors(
    datasets: list[str] | None = None, frequency: str = "monthly", dry_run: bool = False
):
    """Download Fama-French factor data.

    Args:
        datasets: Specific datasets to download (default: core factors)
        frequency: "monthly" or "daily"
        dry_run: If True, show what would be downloaded
    """
    from ml4t.data.providers.fama_french import FamaFrenchProvider

    from utils import ML4T_DATA_PATH

    output_dir = ML4T_DATA_PATH / "factors" / "fama-french"

    # Default core datasets
    if datasets is None:
        datasets = ["ff3", "ff5", "mom"]

    print("=== Fama-French Download ===")
    print(f"Datasets: {datasets}")
    print(f"Frequency: {frequency}")
    print(f"Output: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for ds in datasets:
            print(f"  - {ds}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    provider = FamaFrenchProvider(cache_path=output_dir, use_cache=True)

    print(f"\nDownloading {len(datasets)} datasets...")
    for dataset in datasets:
        print(f"  {dataset}...", end=" ", flush=True)
        try:
            df = provider.fetch(dataset, frequency=frequency)
            print(f"OK ({len(df):,} rows)")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n=== Complete ===")
    print(f"Data saved to: {output_dir}")


def download_aqr_factors(datasets: list[str] | None = None, dry_run: bool = False):
    """Download AQR factor data.

    Args:
        datasets: Specific datasets to download (default: core factors)
        dry_run: If True, show what would be downloaded
    """
    from ml4t.data.providers.aqr import AQRProvider

    from utils import ML4T_DATA_PATH

    output_dir = ML4T_DATA_PATH / "factors" / "aqr"

    # Default core datasets
    if datasets is None:
        datasets = ["qmj", "bab"]

    print("=== AQR Download ===")
    print(f"Datasets: {datasets}")
    print(f"Output: {output_dir}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for ds in datasets:
            print(f"  - {ds}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    provider = AQRProvider(cache_path=output_dir)

    print(f"\nDownloading {len(datasets)} datasets...")
    for dataset in datasets:
        print(f"  {dataset}...", end=" ", flush=True)
        try:
            df = provider.fetch(dataset)
            print(f"OK ({len(df):,} rows)")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n=== Complete ===")
    print(f"Data saved to: {output_dir}")


# %% [markdown]
# ### Download Fama-French Factors

# %%
# Uncomment to download
# download_ff_factors()

# %% [markdown]
# ### Download AQR Factors

# %%
# Uncomment to download
# download_aqr_factors()

# %% [markdown]
# ### Dry Run (Preview)

# %%
download_ff_factors(dry_run=True)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loaders throughout the book:

# %%
from data import load_aqr_factors, load_ff_factors

# %% [markdown]
# ### Fama-French Factors

# %%
# Load Fama-French factors
ff = load_ff_factors()

print(f"Shape: {ff.shape}")
print(f"Columns: {ff.columns}")
print(f"Date range: {ff['timestamp'].min()} to {ff['timestamp'].max()}")
print(f"Memory: {ff.estimated_size('mb'):.1f} MB")

# %%
# Preview
ff.tail(10)

# %%
# Factor statistics (annualized)
factor_cols = [c for c in ff.columns if c not in ["timestamp", "date"]]
print("Factor Annualized Statistics (%):")
for col in factor_cols[:6]:
    series = ff[col].drop_nulls()
    mean_annual = series.mean() * 12  # Monthly to annual
    vol_annual = series.std() * (12**0.5)
    sharpe = mean_annual / vol_annual if vol_annual > 0 else 0
    print(f"  {col:8s}: mean={mean_annual:6.2f}, vol={vol_annual:6.2f}, SR={sharpe:.2f}")

# %% [markdown]
# ### AQR Factors

# %%
# Load AQR factors
aqr = load_aqr_factors()

print(f"Shape: {aqr.shape}")
print(f"Columns: {aqr.columns}")
print(f"Date range: {aqr['timestamp'].min()} to {aqr['timestamp'].max()}")

# %%
# Preview
aqr.tail(10)

# %% [markdown]
# ## 5. Data Profile

# %%
from utils import ML4T_DATA_PATH

# Check for profiles
for provider, subdir in [("Fama-French", "fama-french"), ("AQR", "aqr")]:
    profile_path = ML4T_DATA_PATH / "factors" / subdir / "profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        print(f"=== {provider} Profile ===")
        print(f"Files: {len(profile.get('files', []))}")
    else:
        print(f"{provider} profile not found at {profile_path}")

# %% [markdown]
# ## 6. Loader Options
#
# The loaders support filtering by frequency and date range:

# %%
# Daily frequency
ff_daily = load_ff_factors(frequency="daily")
print(f"FF daily: {ff_daily.shape}")

# %%
# Date range
recent_ff = load_ff_factors(start_date="2020-01-01")
print(f"FF 2020+: {recent_ff.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### Fama-French Factors
#
# From Ken French's Data Library:
#
# | Factor | Description |
# |--------|-------------|
# | Mkt-RF | Market excess return |
# | SMB | Small Minus Big (size) |
# | HML | High Minus Low (value) |
# | RMW | Robust Minus Weak (profitability) |
# | CMA | Conservative Minus Aggressive (investment) |
# | Mom | Momentum (12-1 month return) |
#
# [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
#
# ### AQR Factors
#
# Alternative factors from AQR Capital:
#
# | Factor | Description |
# |--------|-------------|
# | QMJ | Quality Minus Junk (profitability, growth, safety) |
# | BAB | Betting Against Beta (low-beta premium) |
# | VME | Value Minus Everything (alternative value) |
# | HML Devil | Value with industry adjustment |
#
# [AQR Datasets](https://www.aqr.com/Insights/Datasets)

# %% [markdown]
# ## 8. Updating Data
#
# To update with the latest data:
#
# ```python
# # Update Fama-French factors
# download_ff_factors()
#
# # Update AQR factors
# download_aqr_factors()
# ```
#
# Factor data is typically updated monthly.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Providers | Ken French, AQR |
# | Frequencies | Monthly, Daily |
# | Coverage | 1926-present (FF), varies (AQR) |
# | API Key | None (free) |
# | Loaders | `load_ff_factors()`, `load_aqr_factors()` |
#
# **Primary use**: Risk attribution, alpha measurement, factor investing research.
