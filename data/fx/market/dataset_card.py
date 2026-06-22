# %% [markdown]
# # FX Pairs Dataset
#
# Foreign exchange OHLCV data for G10 majors and crosses.
#
# | Property | Value |
# |----------|-------|
# | **Provider** | OANDA |
# | **Asset Class** | Currency |
# | **Frequency** | Daily, 4-hourly |
# | **Symbols** | 20 FX pairs |
# | **Coverage** | 2011-2025 |
# | **Size** | ~17 MB |
# | **API Key** | `OANDA_API_KEY` (free) |
# | **Loader** | `load_fx_pairs()` |

# %%
"""FX Pairs - download, explore, and update workflow."""

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
# The FX dataset configuration defines which pairs to download, the date range,
# and available frequencies. All parameters are stored in `config.yaml`.

# %%
# Load and display configuration
config_path = Path("config.yaml")
config = yaml.safe_load(config_path.read_text())

print("=== FX Configuration ===")
print(f"Provider: {config['fx']['provider']}")
print(f"Date range: {config['fx']['start']} to {config['fx']['end']}")
print(f"Frequencies: {config['fx']['frequencies']}")
print("\nPairs by category:")
for category, info in config["fx"]["pairs"].items():
    pairs = info["pairs"]
    print(f"  {category.capitalize()} ({len(pairs)}): {', '.join(pairs)}")

total_pairs = sum(len(info["pairs"]) for info in config["fx"]["pairs"].values())
print(f"\nTotal: {total_pairs} pairs")

# %% [markdown]
# ## 2. API Key Setup
#
# OANDA provides free API access for historical FX data.
#
# ### Getting an OANDA API Key
#
# 1. Create a free practice account at [OANDA](https://www.oanda.com/)
# 2. Navigate to **Manage API Access** in your account settings
# 3. Generate a new API token
# 4. Add to your `.env` file in the repository root:
#
# ```bash
# OANDA_API_KEY=your-api-key-here
# ```
#
# The key format is typically: `xxxxxxxx-yyyyyyyy` (two parts separated by hyphen)

# %%
# Verify API key is configured
api_key = os.getenv("OANDA_API_KEY")
if api_key:
    # Show partial key for verification (first 8 chars)
    print(f"OANDA_API_KEY: {api_key[:8]}... (configured)")
else:
    print("WARNING: OANDA_API_KEY not set in environment")
    print("Add to .env file: OANDA_API_KEY=your-key-here")

# %% [markdown]
# ## 3. Download Data
#
# The download uses the `ml4t-data` library which handles:
# - Rate limiting (OANDA allows 100 requests/second)
# - Data validation
# - Consistent schema output
#
# **Note**: First-time download takes ~30 seconds per frequency (20 pairs each).


# %%
def download_fx_data(frequency: str = "4h", dry_run: bool = False):
    """Download FX data from OANDA.

    Args:
        frequency: "daily" or "4h"
        dry_run: If True, show what would be downloaded without doing it
    """
    from ml4t.data.providers.oanda import OandaProvider

    from utils import ML4T_DATA_PATH

    api_key = os.getenv("OANDA_API_KEY")
    if not api_key:
        raise ValueError("OANDA_API_KEY not set. See API Key Setup section.")

    # Load config
    config = yaml.safe_load(config_path.read_text())
    fx_config = config["fx"]

    # Flatten pairs list
    pairs = []
    for category_info in fx_config["pairs"].values():
        pairs.extend(category_info["pairs"])

    output_dir = ML4T_DATA_PATH / "fx" / "market"
    output_path = output_dir / f"{frequency}.parquet"

    print(f"=== FX Download ({frequency}) ===")
    print(f"Pairs: {len(pairs)}")
    print(f"Date range: {fx_config['start']} to {fx_config['end']}")
    print(f"Output: {output_path}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for pair in pairs:
            print(f"  {pair}")
        return

    # Initialize provider
    provider = OandaProvider(api_key=api_key)

    # Download each pair
    all_data = []
    print(f"\nDownloading {len(pairs)} pairs...")
    for pair in pairs:
        print(f"  {pair}...", end=" ", flush=True)
        try:
            # OANDA uses format: EUR_USD (with underscore)
            oanda_pair = f"{pair[:3]}_{pair[3:]}"
            df = provider.fetch_ohlcv(oanda_pair, fx_config["start"], fx_config["end"], frequency)
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
    print(f"Pairs: {combined['symbol'].n_unique()}")
    print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print(f"Saved to: {output_path}")

    return combined


# %% [markdown]
# ### Download Daily Data

# %%
# Uncomment to download daily data
# download_fx_data(frequency="daily")

# %% [markdown]
# ### Download 4-Hourly Data

# %%
# Uncomment to download 4-hourly data
# download_fx_data(frequency="4h")

# %% [markdown]
# ### Dry Run (Preview)
#
# See what would be downloaded without actually downloading:

# %%
download_fx_data(frequency="4h", dry_run=True)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loader throughout the book:

# %%
from data import load_fx_pairs

# Load 4-hourly data (default)
df = load_fx_pairs()

print(f"Shape: {df.shape}")
print(f"Pairs: {df['symbol'].n_unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Memory: {df.estimated_size('mb'):.1f} MB")

# %%
# Schema
df.schema

# %%
# Preview
df.head(10)

# %%
# Available pairs
print("Available pairs:")
for i, pair in enumerate(sorted(df["symbol"].unique().to_list()), 1):
    print(f"  {i:2}. {pair}")

# %% [markdown]
# ### Coverage by Pair

# %%
# Coverage and basic stats by pair
coverage = (
    df.group_by("symbol")
    .agg(
        pl.col("timestamp").min().alias("first_date"),
        pl.col("timestamp").max().alias("last_date"),
        pl.len().alias("n_bars"),
        pl.col("close").mean().alias("avg_price"),
    )
    .sort("symbol")
)
coverage

# %% [markdown]
# ### Volatility Analysis

# %%
# Annualized volatility by pair
fx_vol = (
    df.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
    .group_by("symbol")
    .agg(
        (pl.col("returns").std() * (252 * 6) ** 0.5).alias("annual_vol"),  # 6 bars/day for 4h
    )
    .sort("annual_vol", descending=True)
)
fx_vol

# %% [markdown]
# ## 5. Data Profile
#
# Profiles document the dataset structure, statistics, and quality metrics.
# They are stored alongside the data files.

# %%
from utils import ML4T_DATA_PATH

# Check for existing profile
profile_path = ML4T_DATA_PATH / "fx" / "market" / "4h_profile.json"

if profile_path.exists():
    profile = json.loads(profile_path.read_text())
    print("=== FX 4h Profile ===")
    rows = profile.get("total_rows", profile.get("rows"))
    cols_field = profile.get("columns")
    n_cols = profile.get(
        "total_columns",
        len(cols_field) if isinstance(cols_field, list) else cols_field,
    )
    print(f"Rows: {rows:,}" if rows is not None else "Rows: unknown")
    print(f"Columns: {n_cols}")
    if isinstance(cols_field, list):
        print("\nSchema:")
        for c in cols_field:
            print(f"  {c['name']}: {c['dtype']}")
        ts = next((c for c in cols_field if c.get("name") == "timestamp"), None)
        if ts:
            print(f"\nDate range: {ts['min']}")
            print(f"         to {ts['max']}")
else:
    print(f"Profile not found at {profile_path}")
    print("Generate with: python generate_profiles.py --dataset fx_pairs_4h")

# %% [markdown]
# ### Generate/Refresh Profile
#
# To regenerate the profile after downloading new data:
#
# ```bash
# python generate_profiles.py --dataset fx_pairs_4h --force
# python generate_profiles.py --dataset fx_pairs_daily --force
# ```

# %% [markdown]
# ## 6. Loader Options
#
# The loader supports filtering by frequency, pairs, and date range:

# %%
# Daily frequency
daily = load_fx_pairs(frequency="daily")
print(f"Daily data: {daily.shape}")

# %%
# Specific pairs
majors = load_fx_pairs(pairs=["EUR_USD", "GBP_USD", "USD_JPY"])
print(f"Majors only: {majors.shape}")

# %%
# Date range
recent = load_fx_pairs(start_date="2024-01-01")
print(f"2024 onwards: {recent.shape}")

# %%
# Combined filters
filtered = load_fx_pairs(
    frequency="daily", pairs=["EUR_USD", "GBP_USD"], start_date="2020-01-01", end_date="2023-12-31"
)
print(f"EUR/GBP daily 2020-2023: {filtered.shape}")

# %% [markdown]
# ## 7. Documentation
#
# ### OANDA API
# - [OANDA REST API Documentation](https://developer.oanda.com/rest-live-v20/introduction/)
# - [Instrument List](https://developer.oanda.com/rest-live-v20/pricing-ep/)
#
# ### FX Market Conventions
# - Pairs are quoted as BASE/QUOTE (e.g., EUR/USD = euros per dollar)
# - Major pairs include USD; crosses exclude USD
# - Standard lot = 100,000 units of base currency
#
# ### Data Quality Notes
# - OANDA provides mid-prices (average of bid/ask)
# - Volume represents OANDA's internal trading volume, not global FX volume
# - Weekend gaps are normal (FX markets close Friday 5pm ET to Sunday 5pm ET)

# %% [markdown]
# ## 8. Updating Data
#
# To update with the latest data, re-run the download:
#
# ```python
# # Update to latest available data
# download_fx_data(frequency="4h")
# download_fx_data(frequency="daily")
# ```
#
# The `ml4t-data` library handles incremental updates automatically when the
# end date in the config extends beyond existing data.
#
# **Tip**: Update the `end` date in `config.yaml` before re-downloading
# to extend the coverage period.

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Pairs | 20 (4 majors, 3 commodity, 13 crosses) |
# | Frequencies | Daily, 4-hourly |
# | Coverage | 2011-2025 |
# | Provider | OANDA (free API key) |
# | Config | `config.yaml` |
# | Loader | `load_fx_pairs(frequency, pairs, start_date, end_date)` |
# | Profile | `$ML4T_DATA_PATH/fx/{frequency}_profile.json` |
