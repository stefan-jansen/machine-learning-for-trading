# %% [markdown]
# # ETF Universe Dataset
#
# 100 diversified ETFs across 9 categories for momentum and cross-asset strategies.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | Yahoo Finance |
# | **Asset Class** | Multi-asset (Equity, Fixed Income, Commodities, Currency) |
# | **Frequency** | Daily |
# | **Symbols** | 100 ETFs |
# | **Coverage** | 2006-2025 |
# | **Size** | ~16 MB |
# | **API Key** | None (free) |
# | **Loader** | `load_etfs()` |

# %%
"""ETF Universe - download, explore, and update workflow."""

import json
from pathlib import Path

import polars as pl
import yaml

# %% [markdown]
# ## 1. Configuration
#
# The ETF universe is defined in `config.yaml`. This is the **candidate pool** -
# strategy definition (Chapter 6) filters this down based on liquidity, history,
# and correlation clustering.

# %%
# Load and display configuration
config_path = Path("config.yaml")
config = yaml.safe_load(config_path.read_text())
etf_config = config["etfs"]

print("=== ETF Configuration ===")
print(f"Provider: {etf_config['provider']}")
print(f"Date range: {etf_config['start']} to {etf_config['end']}")
print(f"Frequency: {etf_config['frequency']}")
print(f"\nCategories ({len(etf_config['tickers'])}):")
for category, info in etf_config["tickers"].items():
    symbols = info["symbols"]
    print(f"  {category}: {len(symbols)} ETFs")

total_etfs = sum(len(info["symbols"]) for info in etf_config["tickers"].values())
print(f"\nTotal: {total_etfs} ETFs")

# %% [markdown]
# ## 2. API Key Setup
#
# **No API key required.** Yahoo Finance data is free and publicly accessible.
#
# The `ml4t-data` library handles rate limiting automatically to avoid
# being blocked by Yahoo Finance.

# %%
print("Yahoo Finance requires no API key - data is publicly available.")

# %% [markdown]
# ## 3. Download Data
#
# The download uses the `ml4t-data` library which handles:
# - Rate limiting (1 second delay between batches)
# - Retry logic for failed requests
# - Consistent schema output
#
# **Note**: First-time download takes ~2-3 minutes for 100 ETFs.


# %%
def download_etf_data(dry_run: bool = False, force: bool = False, symbols: list[str] | None = None):
    """Download ETF data from Yahoo Finance.

    Args:
        dry_run: If True, show what would be downloaded without doing it
        force: If True, re-download even if data exists
        symbols: Specific symbols to download (default: all from config)
    """
    from ml4t.data.providers import YahooFinanceProvider

    from utils import ML4T_DATA_PATH

    # Load config
    config = yaml.safe_load(config_path.read_text())
    etf_config = config["etfs"]

    # Flatten symbols list
    if symbols is None:
        symbols = []
        for category_info in etf_config["tickers"].values():
            symbols.extend(category_info["symbols"])

    output_dir = ML4T_DATA_PATH / "etfs" / "market"
    output_path = output_dir / "etf_universe.parquet"

    print("=== ETF Download ===")
    print(f"Symbols: {len(symbols)}")
    print(f"Date range: {etf_config['start']} to {etf_config['end']}")
    print(f"Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for i, symbol in enumerate(symbols, 1):
            print(f"  {i:3}. {symbol}")
        return

    # Check existing data
    if output_path.exists() and not force:
        existing = pl.read_parquet(output_path)
        existing_symbols = set(existing["symbol"].unique().to_list())
        missing = [s for s in symbols if s not in existing_symbols]
        if not missing:
            print(f"\nAll {len(symbols)} ETFs already downloaded.")
            print("Use force=True to re-download.")
            return existing
        print(f"Found {len(existing_symbols)} existing, downloading {len(missing)} missing...")
        symbols = missing

    # Initialize provider and download
    provider = YahooFinanceProvider()
    print(f"\nDownloading {len(symbols)} ETFs...")

    etf_data = provider.fetch_batch_ohlcv(
        symbols=symbols,
        start=etf_config["start"],
        end=etf_config["end"],
        frequency="daily",
        chunk_size=50,
        delay_seconds=1.0,
    )

    # Combine with existing data if applicable
    if output_path.exists() and not force:
        existing = pl.read_parquet(output_path)
        etf_data = pl.concat([existing, etf_data])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    etf_data.write_parquet(output_path)

    print("\n=== Complete ===")
    print(f"Total rows: {len(etf_data):,}")
    print(f"Symbols: {etf_data['symbol'].n_unique()}")
    print(f"Date range: {etf_data['timestamp'].min()} to {etf_data['timestamp'].max()}")
    print(f"Saved to: {output_path}")

    return etf_data


# %% [markdown]
# ### Download All ETFs

# %%
# Uncomment to download all ETF data
# download_etf_data()

# %% [markdown]
# ### Dry Run (Preview)
#
# See what would be downloaded without actually downloading:

# %%
download_etf_data(dry_run=True)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loader throughout the book:

# %%
from data import load_etfs

# Load all ETF data
df = load_etfs()

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
# ### Coverage by Symbol

# %%
# Coverage and basic stats by symbol
coverage = (
    df.group_by("symbol")
    .agg(
        pl.col("timestamp").min().alias("first_date"),
        pl.col("timestamp").max().alias("last_date"),
        pl.len().alias("n_bars"),
        pl.col("volume").mean().alias("avg_daily_volume"),
    )
    .sort("avg_daily_volume", descending=True)
)
coverage.head(20)

# %% [markdown]
# ### Category Summary

# %%
# Build category mapping from config
category_map = {}
for category, info in etf_config["tickers"].items():
    for symbol in info["symbols"]:
        category_map[symbol] = category

df_with_cat = df.with_columns(pl.col("symbol").replace(category_map).alias("category"))

category_summary = (
    df_with_cat.group_by("category")
    .agg(
        pl.col("symbol").n_unique().alias("n_symbols"),
        pl.col("timestamp").min().alias("earliest"),
        pl.col("timestamp").max().alias("latest"),
        pl.col("volume").mean().alias("avg_volume"),
    )
    .sort("n_symbols", descending=True)
)
category_summary

# %% [markdown]
# ## 5. Data Profile
#
# Profiles document the dataset structure, statistics, and quality metrics.
# They are stored alongside the data files.

# %%
from utils import ML4T_DATA_PATH

# Check for existing profile
profile_path = ML4T_DATA_PATH / "etfs" / "market" / "profile.json"

if profile_path.exists():
    profile = json.loads(profile_path.read_text())
    print("=== ETF Universe Profile ===")
    print(f"Dataset: {profile['dataset']}")
    print(f"Rows: {profile['rows']:,}")
    print(f"Columns: {profile['columns']}")
    print(f"Memory: {profile['memory_mb']:.1f} MB")
    print("\nSchema:")
    for col, dtype in profile["schema"].items():
        print(f"  {col}: {dtype}")
    print(f"\nDate range: {profile['column_stats']['timestamp']['min']}")
    print(f"         to {profile['column_stats']['timestamp']['max']}")
else:
    print(f"Profile not found at {profile_path}")
    print("Generate with: python generate_profiles.py --dataset etfs")

# %% [markdown]
# ### Generate/Refresh Profile
#
# To regenerate the profile after downloading new data:
#
# ```bash
# python generate_profiles.py --dataset etfs --force
# ```

# %% [markdown]
# ## 6. Loader Options
#
# The loader supports filtering by symbols and date range:

# %%
# Specific symbols
spy_qqq = load_etfs(symbols=["SPY", "QQQ"])
print(f"SPY + QQQ only: {spy_qqq.shape}")

# %%
# Date range
recent = load_etfs(start_date="2024-01-01")
print(f"2024 onwards: {recent.shape}")

# %%
# Combined filters
filtered = load_etfs(
    symbols=["SPY", "QQQ", "IWM", "TLT", "GLD"], start_date="2020-01-01", end_date="2023-12-31"
)
print(f"5 ETFs, 2020-2023: {filtered.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### Yahoo Finance
# - [Yahoo Finance API (unofficial)](https://python-yahoofinance.readthedocs.io/)
# - Rate limits: ~2000 requests/hour (handled by ml4t-data)
#
# ### ETF Categories
#
# | Category | Count | Description |
# |----------|-------|-------------|
# | `us_equity_broad` | 10 | Large, mid, small cap, equal weight |
# | `us_equity_style` | 10 | Value, growth, momentum, dividend |
# | `us_sectors` | 13 | SPDR sector ETFs + real estate |
# | `international_developed` | 18 | EAFE, Europe, Japan, country ETFs |
# | `emerging_markets` | 11 | EM broad + China, Brazil, India, etc. |
# | `fixed_income` | 15 | Treasury, corporate, high yield, TIPS |
# | `commodities` | 9 | Gold, silver, oil, broad commodity |
# | `specialty` | 10 | Biotech, semiconductors, regional banks |
# | `currency` | 4 | USD, EUR, JPY, GBP currency ETFs |
#
# ### Data Quality Notes
# - Volume represents actual ETF trading volume
# - Adjusted close accounts for dividends and splits
# - Some ETFs have shorter history (check `first_date` in coverage)

# %% [markdown]
# ## 8. Updating Data
#
# To update with the latest data, re-run the download:
#
# ```python
# # Update to latest available data
# download_etf_data()
#
# # Force full re-download
# download_etf_data(force=True)
# ```
#
# **Tip**: Update the `end` date in `config.yaml` before re-downloading
# to extend the coverage period.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Symbols | 100 ETFs across 9 categories |
# | Frequency | Daily |
# | Coverage | 2006-2025 |
# | Provider | Yahoo Finance (free) |
# | Config | `config.yaml` |
# | Loader | `load_etfs(symbols, start_date, end_date)` |
# | Profile | `$ML4T_DATA_PATH/etfs/profile.json` |
#
# **Note**: This is the **candidate pool**. Chapter 6 filters to ~80 ETFs
# based on liquidity, history, and correlation clustering.
