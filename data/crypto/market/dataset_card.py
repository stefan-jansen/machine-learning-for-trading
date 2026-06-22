# %% [markdown]
# # Crypto Premium Index Dataset
#
# Perpetual futures OHLCV and premium index data for funding rate arbitrage strategy.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | Binance Public API |
# | **Asset Class** | Cryptocurrency |
# | **Frequency** | 1h (OHLCV), 8h (premium) |
# | **Symbols** | 20 perpetual futures |
# | **Coverage** | 2020-2025 |
# | **Size** | ~70 MB |
# | **API Key** | None (free) |
# | **Loader** | `load_crypto_perps()`, `load_crypto_premium()` |

# %%
"""Crypto Premium Index - download, explore, and update workflow."""

import json
from pathlib import Path

import polars as pl
import yaml

# %% [markdown]
# ## 1. Configuration
#
# The crypto universe is defined in `config.yaml`. Organized by market segment:
# major cryptocurrencies, DeFi tokens, and Layer 1 blockchains.

# %%
# Load and display configuration
config_path = Path("config.yaml")
config = yaml.safe_load(config_path.read_text())
crypto_config = config["crypto"]

print("=== Crypto Configuration ===")
print(f"Provider: {crypto_config['provider']}")
print(f"Market: {crypto_config['market']}")
print(f"Date range: {crypto_config['start']} to {crypto_config['end']}")
print(f"Premium interval: {crypto_config['interval']}")
print("\nCategories:")
for category, info in crypto_config["symbols"].items():
    symbols = info["symbols"]
    print(f"  {category}: {len(symbols)} tokens - {info['description']}")
    print(f"    {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

total_symbols = sum(len(info["symbols"]) for info in crypto_config["symbols"].values())
print(f"\nTotal: {total_symbols} symbols")

# %% [markdown]
# ## 2. API Key Setup
#
# **No API key required.** Binance Public API provides free access to historical data
# through data.binance.vision.
#
# The `ml4t-data` library handles rate limiting automatically.

# %%
print("Binance Public API requires no API key - data is freely available.")
print("Source: data.binance.vision")

# %% [markdown]
# ## 3. Download Data
#
# The download uses the `ml4t-data` library which handles:
# - Rate limiting
# - Data validation
# - Consistent schema output
#
# Two types of data are available:
# - **OHLCV** (hourly): Price and volume for perpetual futures
# - **Premium Index** (8-hourly): Basis between perpetual and spot prices


# %%
def download_crypto_ohlcv(
    dry_run: bool = False, force: bool = False, symbols: list[str] | None = None
):
    """Download crypto perpetual futures OHLCV from Binance.

    Args:
        dry_run: If True, show what would be downloaded without doing it
        force: If True, re-download even if data exists
        symbols: Specific symbols to download (default: all from config)
    """
    from utils import ML4T_DATA_PATH

    # Load config
    config = yaml.safe_load(config_path.read_text())
    crypto_config = config["crypto"]

    # Flatten symbols list
    if symbols is None:
        symbols = []
        for category_info in crypto_config["symbols"].values():
            symbols.extend(category_info["symbols"])

    output_dir = ML4T_DATA_PATH / "crypto" / "market"
    output_path = output_dir / "perps_1h.parquet"

    print("=== Crypto OHLCV Download ===")
    print(f"Symbols: {len(symbols)}")
    print("Frequency: 1h")
    print(f"Date range: {crypto_config['start']} to {crypto_config['end']}")
    print(f"Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for symbol in symbols:
            print(f"  {symbol}")
        return

    # Check existing
    if output_path.exists() and not force:
        existing = pl.read_parquet(output_path)
        print(f"\nData already exists ({len(existing):,} rows).")
        print("Use force=True to re-download.")
        return existing

    # Initialize provider
    from ml4t.data.providers import BinancePublicProvider

    provider = BinancePublicProvider(market="spot")

    # Download each symbol
    all_data = []
    print(f"\nDownloading {len(symbols)} symbols...")
    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            df = provider.fetch_ohlcv(
                symbol=symbol,
                start=crypto_config["start"],
                end=crypto_config["end"],
                frequency="hourly",
            )
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            all_data.append(df)
            print(f"OK ({len(df):,} rows)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_data:
        raise RuntimeError("No data downloaded!")

    # Combine and save
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pl.concat(all_data)
    combined.write_parquet(output_path)

    print("\n=== Complete ===")
    print(f"Total rows: {len(combined):,}")
    print(f"Symbols: {combined['symbol'].n_unique()}")
    print(f"Saved to: {output_path}")

    return combined


def download_crypto_premium(
    dry_run: bool = False, force: bool = False, symbols: list[str] | None = None
):
    """Download crypto premium index from Binance.

    Args:
        dry_run: If True, show what would be downloaded without doing it
        force: If True, re-download even if data exists
        symbols: Specific symbols to download (default: all from config)
    """
    from utils import ML4T_DATA_PATH

    # Load config
    config = yaml.safe_load(config_path.read_text())
    crypto_config = config["crypto"]

    # Flatten symbols list
    if symbols is None:
        symbols = []
        for category_info in crypto_config["symbols"].values():
            symbols.extend(category_info["symbols"])

    output_dir = ML4T_DATA_PATH / "crypto" / "market"
    output_path = output_dir / "premium_index_8h.parquet"

    print("=== Crypto Premium Index Download ===")
    print(f"Symbols: {len(symbols)}")
    print("Frequency: 8h (funding rate interval)")
    print(f"Date range: {crypto_config['start']} to {crypto_config['end']}")
    print(f"Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for symbol in symbols:
            print(f"  {symbol}")
        return

    # Check existing
    if output_path.exists() and not force:
        existing = pl.read_parquet(output_path)
        print(f"\nData already exists ({len(existing):,} rows).")
        print("Use force=True to re-download.")
        return existing

    # Initialize provider (futures market for premium index)
    from ml4t.data.providers import BinancePublicProvider

    provider = BinancePublicProvider(market="futures")

    print(f"\nDownloading premium index for {len(symbols)} symbols...")
    premium_data = provider.fetch_premium_index_multi(
        symbols=symbols,
        start=crypto_config["start"],
        end=crypto_config["end"],
        interval="8h",
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    premium_data.write_parquet(output_path)

    print("\n=== Complete ===")
    print(f"Total rows: {len(premium_data):,}")
    print(f"Symbols: {premium_data['symbol'].n_unique()}")
    print(f"Saved to: {output_path}")

    return premium_data


# %% [markdown]
# ### Download OHLCV Data

# %%
# Uncomment to download OHLCV data
# download_crypto_ohlcv()

# %% [markdown]
# ### Download Premium Index

# %%
# Uncomment to download premium index
# download_crypto_premium()

# %% [markdown]
# ### Dry Run (Preview)

# %%
download_crypto_ohlcv(dry_run=True)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loaders throughout the book:

# %%
from data import load_crypto_perps, load_crypto_premium

# %% [markdown]
# ### Perpetual Futures OHLCV

# %%
# Load hourly OHLCV data
perps = load_crypto_perps()

print(f"Shape: {perps.shape}")
print(f"Symbols: {perps['symbol'].n_unique()}")
print(f"Date range: {perps['timestamp'].min()} to {perps['timestamp'].max()}")
print(f"Memory: {perps.estimated_size('mb'):.1f} MB")

# %%
# Schema
perps.schema

# %%
# Preview
perps.head(10)

# %%
# Volume by symbol (USD notional)
volume_by_symbol = (
    perps.group_by("symbol")
    .agg(
        (pl.col("volume") * pl.col("close")).sum().alias("total_volume_usd"),
        pl.len().alias("n_observations"),
        pl.col("timestamp").min().alias("first_date"),
        pl.col("timestamp").max().alias("last_date"),
    )
    .sort("total_volume_usd", descending=True)
)
volume_by_symbol

# %% [markdown]
# ### Premium Index

# %%
# Load 8-hourly premium index data
premium = load_crypto_premium()

print(f"Shape: {premium.shape}")
print(f"Symbols: {premium['symbol'].n_unique()}")
print(f"Date range: {premium['timestamp'].min()} to {premium['timestamp'].max()}")
print(f"Memory: {premium.estimated_size('mb'):.1f} MB")

# %%
# Preview
premium.head(10)

# %%
# Premium statistics by symbol
# Premium index captures basis between perpetual and spot
premium_stats = (
    premium.group_by("symbol")
    .agg(
        pl.col("premium_index_close").mean().alias("mean_premium"),
        pl.col("premium_index_close").std().alias("std_premium"),
        pl.col("premium_index_close").min().alias("min_premium"),
        pl.col("premium_index_close").max().alias("max_premium"),
    )
    .sort("mean_premium", descending=True)
)
premium_stats

# %% [markdown]
# ## 5. Data Profile
#
# Profiles document the dataset structure, statistics, and quality metrics.

# %%
from utils import ML4T_DATA_PATH

# Check for existing profiles
for dataset, filename in [
    ("OHLCV", "perps_1h_profile.json"),
    ("Premium", "premium_index_8h_profile.json"),
]:
    profile_path = ML4T_DATA_PATH / "crypto" / "market" / filename
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        print(f"=== Crypto {dataset} Profile ===")
        rows = profile.get("total_rows", profile.get("rows"))
        cols = profile.get("total_columns", profile.get("columns"))
        if isinstance(cols, list):
            cols = len(cols)
        print(f"Rows: {rows:,}" if rows is not None else "Rows: unknown")
        print(f"Columns: {cols}")
        mem = profile.get("memory_mb")
        if mem is not None:
            print(f"Memory: {mem:.1f} MB")
        print()
    else:
        print(f"Profile not found: {profile_path}")
        print(f"Generate with: python generate_profiles.py --dataset crypto_{dataset.lower()}\n")

# %% [markdown]
# ## 6. Loader Options
#
# The loaders support filtering by symbols and date range:

# %%
# Specific symbols
btc_eth = load_crypto_perps(symbols=["BTCUSDT", "ETHUSDT"])
print(f"BTC + ETH only: {btc_eth.shape}")

# %%
# Date range
recent = load_crypto_premium(start_date="2024-01-01")
print(f"Premium 2024+: {recent.shape}")

# %%
# Combined filters
filtered = load_crypto_perps(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], start_date="2023-01-01", end_date="2023-12-31"
)
print(f"3 symbols, 2023: {filtered.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### Binance Public API
# - [Binance Public Data](https://data.binance.vision/)
# - [API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
#
# ### Premium Index
#
# The premium index measures the basis between perpetual futures and spot prices:
#
# $$\text{Premium} = \frac{P_{perp} - P_{spot}}{P_{spot}}$$
#
# Key properties:
# - **Positive premium**: Perpetual trades at premium (bullish sentiment)
# - **Negative premium**: Perpetual trades at discount (bearish sentiment)
# - **Funding rate**: Derived from premium, settles every 8 hours
#
# ### Data Quality Notes
# - Volume represents Binance exchange volume only
# - BTC/ETH/major alts: Data from Jan 2020 (6 years history)
# - Newer tokens (APT, SUI, INJ, ARB, OP): Data from listing date (2022-2023)
# - MATICUSDT renamed to POLUSDT in Sept 2024 (data ends there)
# - 8-hour intervals align with funding rate settlement times (00:00, 08:00, 16:00 UTC)

# %% [markdown]
# ## 8. Updating Data
#
# To update with the latest data, re-run the download:
#
# ```python
# # Update OHLCV data
# download_crypto_ohlcv()
#
# # Update premium index
# download_crypto_premium()
#
# # Force full re-download
# download_crypto_ohlcv(force=True)
# download_crypto_premium(force=True)
# ```
#
# **Tip**: Update the `end` date in `config.yaml` before re-downloading.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Symbols | 20 perpetual futures (major, DeFi, L1) |
# | Frequencies | 1h OHLCV, 8h premium index |
# | Coverage | 2020-2025 (6 years for BTC/ETH) |
# | Provider | Binance Public (free) |
# | Config | `config.yaml` |
# | Loaders | `load_crypto_perps()`, `load_crypto_premium()` |
#
# **Use case**: Funding rate arbitrage strategy exploiting premium mean reversion.
