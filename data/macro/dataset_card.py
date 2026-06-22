# %% [markdown]
# # FRED Macro Indicators Dataset
#
# Treasury yields and economic indicators for regime filtering.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | FRED (Federal Reserve) |
# | **Asset Class** | Macro/Economic |
# | **Frequency** | Daily (treasury), Monthly (economic) |
# | **Series** | 17+ indicators |
# | **Coverage** | 2000-2025 |
# | **Size** | ~5 MB |
# | **API Key** | `FRED_API_KEY` (free) |
# | **Loader** | `load_macro()` |

# %%
"""FRED Macro Indicators - download, explore, and update workflow."""

import json
import os
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# %% [markdown]
# ## 1. Configuration
#
# The macro series are defined in `config.yaml`. Primary use: Treasury yields
# for regime filtering (risk-on/risk-off based on yield curve slope).

# %%
# Load and display configuration
config_path = Path("config.yaml")
config = yaml.safe_load(config_path.read_text())
macro_config = config["macro"]

print("=== Macro Configuration ===")
print(f"Provider: {macro_config['provider']}")
print(f"Date range: {macro_config['start']} to {macro_config['end']}")
print("\nSeries groups:")
for group_name, info in macro_config["series"].items():
    if isinstance(info, dict) and "symbols" in info:
        symbols = info["symbols"]
        print(f"  {group_name}: {info.get('description', '')}")
        for s in symbols:
            print(f"    - {s}")

# %% [markdown]
# ## 2. API Key Setup
#
# FRED requires a free API key.
#
# ### Getting a FRED API Key
#
# 1. Go to [FRED API Key Signup](https://fredaccount.stlouisfed.org/login/secure/)
# 2. Create a free account or sign in
# 3. Navigate to **API Keys** and create a new key
# 4. Add to your `.env` file in the repository root:
#
# ```bash
# FRED_API_KEY=your-32-character-api-key
# ```
#
# FRED is free with generous rate limits (120 requests/minute).

# %%
# Verify API key is configured
api_key = os.getenv("FRED_API_KEY")
if api_key:
    print(f"FRED_API_KEY: {api_key[:8]}... (configured)")
else:
    print("WARNING: FRED_API_KEY not set in environment")
    print("Get free key at: https://fredaccount.stlouisfed.org/login/secure/")
    print("Add to .env file: FRED_API_KEY=your-key-here")

# %% [markdown]
# ## 3. Download Data
#
# The download fetches multiple economic series and aligns them to a daily calendar.
# Different series have different native frequencies (daily, weekly, monthly, quarterly).

# %%
# Key macro indicators with native frequency
FRED_SERIES = {
    # Daily series
    "DFF": ("Fed Funds Rate", "daily"),
    "DGS10": ("10-Year Treasury", "daily"),
    "DGS2": ("2-Year Treasury", "daily"),
    "DGS5": ("5-Year Treasury", "daily"),
    "DGS30": ("30-Year Treasury", "daily"),
    "T10Y2Y": ("10Y-2Y Spread", "daily"),
    "VIXCLS": ("VIX Volatility Index", "daily"),
    # Weekly series
    "ICSA": ("Initial Jobless Claims", "weekly"),
    # Monthly series
    "CPIAUCSL": ("CPI All Urban Consumers", "monthly"),
    "UNRATE": ("Unemployment Rate", "monthly"),
    "PAYEMS": ("Non-Farm Payrolls", "monthly"),
    "INDPRO": ("Industrial Production", "monthly"),
    # Quarterly series
    "GDP": ("Gross Domestic Product", "quarterly"),
}


def download_macro_data(
    dry_run: bool = False, force: bool = False, series: list[str] | None = None
):
    """Download macro data from FRED.

    Args:
        dry_run: If True, show what would be downloaded without doing it
        force: If True, re-download even if data exists
        series: Specific series to download (default: all from FRED_SERIES)
    """
    from ml4t.data.providers import FREDProvider

    from utils import ML4T_DATA_PATH

    api_key = os.getenv("FRED_API_KEY")
    if not api_key and not dry_run:
        raise ValueError("FRED_API_KEY not set. See API Key Setup section.")

    # Load config for date range (resolved relative to this script for cwd-independence;
    # __file__ is undefined in papermill/notebook execution, so fall back to cwd).
    try:
        here = Path(__file__).parent
    except NameError:
        here = Path.cwd()
    config = yaml.safe_load((here / "config.yaml").read_text())
    macro_config = config["macro"]

    if series is None:
        series_to_download = FRED_SERIES
    else:
        series_to_download = {s: FRED_SERIES[s] for s in series if s in FRED_SERIES}

    output_dir = ML4T_DATA_PATH / "macro"
    output_path = output_dir / "fred_macro.parquet"

    print("=== Macro Download ===")
    print(f"Series: {len(series_to_download)}")
    print(f"Date range: {macro_config['start']} to {macro_config['end']}")
    print(f"Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for series_id, (name, freq) in series_to_download.items():
            print(f"  {series_id:12s} ({freq:9s}) {name}")
        return

    # Check existing
    if output_path.exists() and not force:
        existing = pl.read_parquet(output_path)
        print(f"\nData already exists ({len(existing):,} rows).")
        print("Use force=True to re-download.")
        return existing

    # Initialize provider
    provider = FREDProvider(api_key=api_key)

    # Download each series
    all_series = []
    print(f"\nDownloading {len(series_to_download)} series...")
    for series_id, (name, frequency) in series_to_download.items():
        print(f"  {series_id}...", end=" ", flush=True)
        try:
            df = provider.fetch_ohlcv(
                series_id,
                start=macro_config["start"],
                end=macro_config["end"],
                frequency=frequency,
            )
            # Rename close to series_id
            series_df = df.select(
                [
                    pl.col("timestamp").cast(pl.Date).alias("date"),
                    pl.col("close").alias(series_id.lower()),
                ]
            )
            all_series.append(series_df)
            print(f"OK ({len(df):,} obs)")
        except Exception as e:
            print(f"ERROR: {e}")

    provider.close()

    if not all_series:
        raise RuntimeError("No series downloaded!")

    # Create daily date range for alignment
    from datetime import datetime

    dates = pl.date_range(
        datetime.strptime(macro_config["start"], "%Y-%m-%d"),
        datetime.strptime(macro_config["end"], "%Y-%m-%d"),
        eager=True,
    )
    result = pl.DataFrame({"date": dates})

    # Join all series and forward-fill
    for series_df in all_series:
        series_col = [c for c in series_df.columns if c != "date"][0]
        result = result.join(series_df, on="date", how="left")
        result = result.with_columns(pl.col(series_col).forward_fill())

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)

    print("\n=== Complete ===")
    print(f"Total rows: {len(result):,}")
    print(f"Columns: {len(result.columns)}")
    print(f"Saved to: {output_path}")

    return result


# %% [markdown]
# ### Download All Series

# %%
# Uncomment to download all macro data
# download_macro_data()

# %% [markdown]
# ### Dry Run (Preview)

# %%
download_macro_data(dry_run=True)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loader throughout the book:

# %%
from data import load_macro

# Load all macro data
df = load_macro()

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Memory: {df.estimated_size('mb'):.1f} MB")

# %%
# Schema
df.schema

# %%
# Preview
df.head(10)

# %% [markdown]
# ### Treasury Yield Statistics

# %%
# Treasury yield summary
yield_cols = [c for c in df.columns if c.startswith("dgs")]
if yield_cols:
    print("Treasury Yield Summary:")
    for col in yield_cols:
        series = df[col].drop_nulls()
        print(
            f"  {col.upper()}: mean={series.mean():.2f}%, min={series.min():.2f}%, max={series.max():.2f}%"
        )

# %% [markdown]
# ### Yield Curve Slope

# %%
# Yield curve slope (10Y - 2Y)
if all(c in df.columns for c in ["dgs10", "dgs2"]):
    df_with_slope = df.with_columns((pl.col("dgs10") - pl.col("dgs2")).alias("yield_curve_slope"))

    slope = df_with_slope["yield_curve_slope"].drop_nulls()
    print("\nYield Curve Slope (10Y - 2Y):")
    print(f"  Mean: {slope.mean():.2f}%")
    print(f"  Current: {slope[-1]:.2f}%")
    print(f"  % Inverted (< 0): {(slope < 0).sum() / len(slope) * 100:.1f}%")

# %% [markdown]
# ## 5. Data Profile

# %%
from utils import ML4T_DATA_PATH

# Check for existing profile
profile_path = ML4T_DATA_PATH / "macro" / "profile.json"

if profile_path.exists():
    profile = json.loads(profile_path.read_text())
    print("=== Macro Profile ===")
    print(f"Rows: {profile['rows']:,}")
    print(f"Columns: {profile['columns']}")
    print(f"Memory: {profile['memory_mb']:.1f} MB")
else:
    print(f"Profile not found at {profile_path}")
    print("Generate with: python generate_profiles.py --dataset macro")

# %% [markdown]
# ## 6. Loader Options
#
# The loader supports filtering by series and date range:

# %%
# Specific series
yields_only = load_macro(series=["DGS2", "DGS10", "DGS30"])
print(f"Treasury yields only: {yields_only.shape}")

# %%
# Date range
recent = load_macro(start_date="2020-01-01")
print(f"2020 onwards: {recent.shape}")

# %%
# Combined filters
filtered = load_macro(
    series=["DGS10", "DGS2", "VIXCLS"], start_date="2020-01-01", end_date="2023-12-31"
)
print(f"Yields + VIX, 2020-2023: {filtered.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### FRED API
# - [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
# - [API Key Request](https://fredaccount.stlouisfed.org/login/secure/)
# - Rate limit: 120 requests/minute (generous)
#
# ### Regime Filtering
#
# The yield curve slope is commonly used for regime detection:
#
# | Slope Range | Regime | Interpretation |
# |-------------|--------|----------------|
# | > 0.5% | Risk-on | Normal economic expansion |
# | 0% to 0.5% | Caution | Late cycle |
# | < 0% | Risk-off | Inverted curve, recession signal |
#
# Chapter 6 strategies use this for conditional signal weighting.
#
# ### Data Quality Notes
# - Treasury yields are daily (excluding weekends/holidays)
# - Economic series are forward-filled to daily alignment
# - VIX is close price (not intraday high)

# %% [markdown]
# ## 8. Updating Data
#
# To update with the latest data:
#
# ```python
# # Update all series
# download_macro_data()
#
# # Force full re-download
# download_macro_data(force=True)
# ```
#
# **Tip**: Update the `end` date in `config.yaml` before re-downloading.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Series | 13+ (treasury yields, economic indicators) |
# | Frequency | Daily (aligned from native frequencies) |
# | Coverage | 2000-2025 |
# | Provider | FRED (free API key) |
# | Config | `config.yaml` |
# | Loader | `load_macro(series, start_date, end_date)` |
#
# **Primary use**: Yield curve slope for regime filtering in strategy signals.
