# %% [markdown]
# # US Equities Dataset
#
# Historical US equity data from NASDAQ Data Link (formerly Quandl Wiki Prices).
#
# | Property | Value |
# |----------|-------|
# | **Provider** | NASDAQ Data Link |
# | **Asset Class** | US Equities |
# | **Frequency** | Daily |
# | **Symbols** | 3,199 |
# | **Coverage** | 1962-2018 |
# | **Size** | ~662 MB |
# | **API Key** | `QUANDL_API_KEY` (free) |
# | **Loader** | `load_us_equities()` |
#
# **NOTE**: This is a **frozen dataset** ending in 2018. It is not updateable.

# %%
"""US Equities - download, explore, and update workflow."""

import json
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# %% [markdown]
# ## 1. Configuration
#
# US Equities is a **frozen dataset** from NASDAQ Data Link (formerly Quandl Wiki Prices).
# No local configuration file - the dataset is downloaded as-is.
#
# ### Dataset Characteristics
#
# - **Survivorship-bias free**: Includes delisted companies
# - **Coverage**: 1962-01-02 to 2018-03-27 (frozen)
# - **Adjusted prices**: Split and dividend adjusted
# - **Data quality**: Some gaps for illiquid stocks

# %%
print("=== US Equities Configuration ===")
print("Provider: NASDAQ Data Link (Wiki Prices)")
print("Coverage: 1962-01-02 to 2018-03-27 (FROZEN)")
print("Symbols: ~3,199 US companies")
print("Frequency: Daily OHLCV")
print("\nThis dataset is not updateable - frozen at March 2018.")

# %% [markdown]
# ## 2. API Key Setup
#
# NASDAQ Data Link requires a free API key.
#
# ### Getting an API Key
#
# 1. Go to [NASDAQ Data Link](https://data.nasdaq.com/sign-up)
# 2. Create a free account
# 3. Navigate to **Account Settings** → **API Key**
# 4. Add to your `.env` file in the repository root:
#
# ```bash
# QUANDL_API_KEY=your-api-key-here
# ```
#
# **Note**: The environment variable is `QUANDL_API_KEY` (legacy name) or
# `NASDAQ_DATA_LINK_API_KEY` (new name). Both are supported.

# %%
# Verify API key is configured
api_key = os.getenv("QUANDL_API_KEY") or os.getenv("NASDAQ_DATA_LINK_API_KEY")
if api_key:
    print(f"API Key: {api_key[:8]}... (configured)")
else:
    print("WARNING: No API key found in environment")
    print("Get free key at: https://data.nasdaq.com/sign-up")
    print("Add to .env file: QUANDL_API_KEY=your-key-here")

# %% [markdown]
# ## 3. Download Data
#
# This is a one-time download. The dataset is frozen and won't be updated.


# %%
def download_us_equities(dry_run: bool = False, force: bool = False):
    """Download US Equities dataset from NASDAQ Data Link.

    Args:
        dry_run: If True, show what would be downloaded without doing it
        force: If True, re-download even if data exists
    """
    from ml4t.data.providers.wiki_prices import WikiPricesProvider

    from utils import ML4T_DATA_PATH

    api_key = os.getenv("QUANDL_API_KEY") or os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key and not dry_run:
        raise ValueError(
            "No API key found. Set QUANDL_API_KEY environment variable.\n"
            "Get free key at: https://data.nasdaq.com/sign-up"
        )

    output_dir = ML4T_DATA_PATH / "equities" / "market" / "us_equities"
    output_path = output_dir / "us_equities.parquet"

    print("=== US Equities Download ===")
    print("Dataset: Quandl WIKI Prices")
    print("Coverage: 1962-01-02 to 2018-03-27 (frozen)")
    print("Symbols: ~3,199 US companies")
    print("Estimated size: ~650 MB")
    print(f"Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        print("  - Full Wiki Prices dataset")
        print("  - One-time download (frozen dataset)")
        return

    # Check existing
    if output_path.exists() and not force:
        import polars as pl

        existing = pl.read_parquet(output_path)
        print(f"\nData already exists ({len(existing):,} rows).")
        print("Use force=True to re-download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading... (this may take several minutes)\n")

    downloaded_path = WikiPricesProvider.download(
        output_path=output_dir,
        api_key=api_key,
    )

    # Rename to canonical name if needed
    if downloaded_path.name != "us_equities.parquet":
        final_path = output_dir / "us_equities.parquet"
        downloaded_path.rename(final_path)
        downloaded_path = final_path

    # Print stats
    provider = WikiPricesProvider(parquet_path=downloaded_path)
    stats = provider.get_dataset_stats()

    print("\n=== Complete ===")
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Symbols: {stats['total_symbols']}")
    print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"File size: {stats['file_size_mb']:.1f} MB")
    print(f"Saved to: {downloaded_path}")


# %% [markdown]
# ### Download (One-Time)

# %%
# Uncomment to download
# download_us_equities()

# %% [markdown]
# ### Dry Run (Preview)

# %%
download_us_equities(dry_run=True)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loader throughout the book:

# %%
from data import load_us_equities

# Load all US equities data
df = load_us_equities()

print(f"Shape: {df.shape}")
print(f"Symbols: {df['symbol'].n_unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Memory: {df.estimated_size('mb'):.1f} MB")

# %%
# Schema
df.schema

# %%
# Preview
df.head(10)

# %% [markdown]
# ### Coverage by Year

# %%
# Active symbols per year
yearly = (
    df.with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.col("symbol").n_unique().alias("n_symbols"),
        pl.len().alias("n_observations"),
    )
    .sort("year")
)
print("Coverage by year (last 20):")
yearly.tail(20)

# %% [markdown]
# ### Top Symbols by Volume

# %%
# Top symbols by average daily volume
top_volume = (
    df.group_by("symbol")
    .agg(
        pl.col("volume").mean().alias("avg_volume"),
        pl.col("timestamp").min().alias("first_date"),
        pl.col("timestamp").max().alias("last_date"),
        pl.len().alias("n_days"),
    )
    .sort("avg_volume", descending=True)
)
top_volume.head(20)

# %% [markdown]
# ## 5. Data Profile

# %%
from utils import ML4T_DATA_PATH

# Check for existing profile
profile_path = ML4T_DATA_PATH / "equities" / "market" / "us_equities" / "us_equities_profile.json"

if profile_path.exists():
    profile = json.loads(profile_path.read_text())
    print("=== US Equities Profile ===")
    rows = profile.get("total_rows", profile.get("rows"))
    cols = profile.get("total_columns", profile.get("columns"))
    if isinstance(cols, list):
        cols = len(cols)
    print(f"Rows: {rows:,}" if rows is not None else "Rows: unknown")
    print(f"Columns: {cols}")
    mem = profile.get("memory_mb")
    if mem is not None:
        print(f"Memory: {mem:.1f} MB")
else:
    print(f"Profile not found at {profile_path}")
    print("Generate with: python generate_profiles.py --dataset us_equities")

# %% [markdown]
# ## 6. Loader Options
#
# The loader supports filtering by symbols and date range:

# %%
# Specific symbols
tech_stocks = load_us_equities(symbols=["AAPL", "MSFT", "GOOGL"])
print(f"Tech stocks: {tech_stocks.shape}")

# %%
# Date range
recent = load_us_equities(start_date="2015-01-01")
print(f"2015 onwards: {recent.shape}")

# %%
# Combined filters
filtered = load_us_equities(
    symbols=["AAPL", "MSFT", "AMZN", "GOOGL", "FB"], start_date="2010-01-01", end_date="2018-12-31"
)
print(f"5 tech stocks, 2010-2018: {filtered.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### NASDAQ Data Link
# - [NASDAQ Data Link](https://data.nasdaq.com/)
# - [Wiki Prices Documentation](https://data.nasdaq.com/databases/WIKIP)
# - [API Documentation](https://docs.data.nasdaq.com/)
#
# ### Dataset Columns
#
# | Column | Description |
# |--------|-------------|
# | `date` | Trading date |
# | `symbol` | Ticker symbol |
# | `open` | Opening price (adjusted) |
# | `high` | High price (adjusted) |
# | `low` | Low price (adjusted) |
# | `close` | Closing price (adjusted) |
# | `volume` | Trading volume |
# | `adj_open` | Split/dividend adjusted open |
# | `adj_high` | Split/dividend adjusted high |
# | `adj_low` | Split/dividend adjusted low |
# | `adj_close` | Split/dividend adjusted close |
# | `adj_volume` | Adjusted volume |
# | `ex_dividend` | Ex-dividend amount |
# | `split_ratio` | Stock split ratio |
#
# ### Data Quality Notes
#
# - **Survivorship-bias free**: Includes delisted companies
# - **Adjusted prices**: Split and dividend adjusted
# - **Coverage gaps**: Some illiquid stocks have missing days
# - **End date**: March 27, 2018 (dataset frozen)

# %% [markdown]
# ## 8. Updating Data
#
# **This dataset is NOT updateable.**
#
# The Wiki Prices dataset was frozen in March 2018 when Quandl discontinued
# free maintenance. The data cannot be extended beyond 2018-03-27.
#
# ### Alternatives for Recent Data
#
# For US equity data after 2018, consider:
#
# | Provider | Coverage | Cost |
# |----------|----------|------|
# | AlgoSeek | 2017-2021 | Licensed |
# | Yahoo Finance | Current | Free |
# | Polygon.io | Current | Paid |
# | Tiingo | Current | Freemium |
#
# See `09_algoseek_licensed.py` for AlgoSeek S&P 500 daily data.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Symbols | 3,199 US companies |
# | Frequency | Daily OHLCV |
# | Coverage | 1962-2018 (frozen) |
# | Provider | NASDAQ Data Link (free API key) |
# | Loader | `load_us_equities(symbols, start_date, end_date)` |
#
# **Primary use**: Historical backtests, ML training on pre-2019 data.
# **Limitation**: Frozen dataset - ends March 2018.
