# %% [markdown]
# # Tick Data Dataset
#
# Market-by-order tick data for order book analysis and microstructure research.
#
# | Property | Value |
# |----------|-------|
# | **Providers** | Databento (MBO), NASDAQ FTP (ITCH) |
# | **Asset Class** | US Equities |
# | **Frequency** | Tick (microsecond) |
# | **Coverage** | Point-in-time |
# | **Size** | MBO: ~500 MB, ITCH: ~5 GB/day |
# | **API Key** | MBO: `DATABENTO_API_KEY` (**PAID**), ITCH: None |
# | **Loaders** | `load_mbo_data()`, `load_nasdaq_itch()` |
#
# **NOTE**: MBO data is expensive (~$0.50/symbol/day). ITCH is free but requires parsing.

# %%
"""Tick Data - download, explore, and update workflow."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# %% [markdown]
# ## 1. Configuration
#
# Two tick data sources are available:
#
# | Source | Type | Cost | Parsing |
# |--------|------|------|---------|
# | Databento MBO | Order book events | ~$0.50/symbol/day | Pre-processed |
# | NASDAQ ITCH | Raw exchange feed | Free | Requires parsing |
#
# **MBO (Market-By-Order)**: Clean order book events - add, modify, cancel, trade.
# **ITCH**: Native NASDAQ TotalView-ITCH 5.0 messages - requires binary parsing.

# %%
print("=== Tick Data Configuration ===")
print()
print("Databento MBO:")
print("  Provider: Databento API (XNAS.ITCH dataset)")
print("  Cost: ~$0.45-0.50 per symbol per day")
print("  Default symbols: NVDA, SPY, TSLA")
print("  Format: Pre-processed Parquet")
print()
print("NASDAQ ITCH:")
print("  Provider: NASDAQ FTP (free)")
print("  Cost: Free (requires parsing)")
print("  File size: 4-6 GB per day (compressed)")
print("  Format: Binary (requires parsing)")

# %% [markdown]
# ## 2. API Key Setup
#
# ### Databento MBO (Paid)
#
# Databento requires a paid API key. New accounts receive $125 free credit.
#
# 1. Sign up at [Databento](https://databento.com/signup)
# 2. Navigate to **API Keys** in your dashboard
# 3. Add to your `.env` file:
#
# ```bash
# DATABENTO_API_KEY=db-your-api-key-here
# ```
#
# ### NASDAQ ITCH (Free)
#
# No API key required. Data is freely available from NASDAQ's FTP server.

# %%
# Verify Databento API key
databento_key = os.getenv("DATABENTO_API_KEY")
if databento_key:
    print(f"DATABENTO_API_KEY: {databento_key[:8]}... (configured)")
else:
    print("DATABENTO_API_KEY: Not configured")
    print("  Get key at: https://databento.com/signup ($125 free credit)")

print()
print("NASDAQ ITCH: No API key required (free FTP access)")

# %% [markdown]
# ## 3. Download Data
#
# **WARNING**: MBO data is expensive. Always estimate cost before downloading!

# %%
# Default configuration
DEFAULT_MBO_SYMBOLS = ["NVDA", "SPY", "TSLA"]
DEFAULT_MBO_DAYS = 10


def get_trading_dates(start_date: str, end_date: str) -> list[str]:
    """Generate list of trading dates (weekdays) between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Skip weekends
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def estimate_mbo_cost(
    symbols: list[str] | None = None,
    dates: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Estimate MBO download cost.

    Args:
        symbols: List of symbols (default: NVDA, SPY, TSLA)
        dates: List of dates in YYYYMMDD format
        start_date: Start date (YYYY-MM-DD) if not providing dates list
        end_date: End date (YYYY-MM-DD) if not providing dates list

    Returns:
        Cost estimate dictionary
    """
    if symbols is None:
        symbols = DEFAULT_MBO_SYMBOLS

    if dates is None and start_date and end_date:
        dates = get_trading_dates(start_date, end_date)
    elif dates is None:
        # Default: 10 trading days
        end = datetime(2024, 11, 15)
        start = end - timedelta(days=20)
        dates = get_trading_dates(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        dates = dates[-DEFAULT_MBO_DAYS:]

    cost_per_symbol_day = 0.50
    total_symbol_days = len(symbols) * len(dates)
    estimated_cost = total_symbol_days * cost_per_symbol_day

    print("=== MBO Cost Estimate ===")
    print(f"Symbols: {symbols}")
    print(f"Days: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"Total symbol-days: {total_symbol_days}")
    print(f"Cost per symbol-day: ${cost_per_symbol_day:.2f}")
    print("")
    print(f"ESTIMATED COST: ${estimated_cost:.2f}")

    return {
        "symbols": symbols,
        "num_days": len(dates),
        "total_symbol_days": total_symbol_days,
        "estimated_cost_usd": estimated_cost,
    }


def download_mbo_data(
    symbols: list[str] | None = None,
    dates: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    dry_run: bool = True,  # Default True for safety!
    force: bool = False,
):
    """Download MBO tick data from Databento.

    Args:
        symbols: List of symbols (default: NVDA, SPY, TSLA)
        dates: List of dates in YYYYMMDD format
        start_date: Start date (YYYY-MM-DD) if not providing dates list
        end_date: End date (YYYY-MM-DD) if not providing dates list
        dry_run: If True, show cost estimate without downloading (DEFAULT: True)
        force: If True, re-download even if data exists
    """
    import databento as db

    from utils import ML4T_DATA_PATH

    if symbols is None:
        symbols = DEFAULT_MBO_SYMBOLS

    if dates is None and start_date and end_date:
        dates = get_trading_dates(start_date, end_date)
    elif dates is None:
        end = datetime(2024, 11, 15)
        start = end - timedelta(days=20)
        dates = get_trading_dates(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        dates = dates[-DEFAULT_MBO_DAYS:]

    output_dir = ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "market_by_order"

    print("=== MBO Data Download ===")
    print(f"Symbols: {symbols}")
    print(f"Dates: {len(dates)} days ({dates[0]} to {dates[-1]})")
    print(f"Output: {output_dir}")

    # Always show cost estimate
    estimate = estimate_mbo_cost(symbols, dates)

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for symbol in symbols:
            print(f"  {symbol}: {len(dates)} days")
        print("\nSet dry_run=False to actually download.")
        print(f"WARNING: This will cost ~${estimate['estimated_cost_usd']:.2f}")
        return

    # Verify API key
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set. See API Key Setup section.")

    client = db.Historical(api_key)

    print(f"\nDownloading {len(symbols)} symbols x {len(dates)} days...")
    for symbol in symbols:
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for date in dates:
            output_file = symbol_dir / f"{date}.parquet"

            if output_file.exists() and not force:
                print(f"  Skipping {symbol}/{date} (exists)")
                continue

            print(f"  {symbol}/{date}...", end=" ", flush=True)
            try:
                date_str = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                data = client.timeseries.get_range(
                    dataset="XNAS.ITCH",
                    schema="mbo",
                    symbols=[symbol],
                    start=f"{date_str}T00:00:00",
                    end=f"{date_str}T23:59:59",
                )

                df = data.to_df()
                if len(df) > 0:
                    pl_df = pl.from_pandas(df.reset_index())
                    pl_df.write_parquet(output_file)
                    print(f"OK ({len(df):,} events)")
                else:
                    print("No data")
            except Exception as e:
                print(f"ERROR: {e}")

    print("\n=== Complete ===")
    print(f"Data saved to: {output_dir}")


def download_nasdaq_itch(date: str = "01302020", dry_run: bool = False):
    """Download NASDAQ ITCH sample data.

    Args:
        date: Date in MMDDYYYY format (default: 01302020)
        dry_run: If True, show what would be downloaded
    """
    from ml4t.data.providers.nasdaq_itch import ITCHSampleProvider

    from utils import ML4T_DATA_PATH

    output_dir = ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "nasdaq_itch" / "raw"

    print("=== NASDAQ ITCH Download ===")
    print("Source: NASDAQ TotalView-ITCH 5.0")
    print("URL: https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/")
    print(f"Date: {date}")
    print(f"Output: {output_dir}")
    print()
    print("WARNING: Files are 4-6 GB each!")
    print("   Download may take 30-60 minutes.")
    print("   Requires parsing before use (see Chapter 4 notebooks)")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        print(f"  {date}.NASDAQ_ITCH50.gz (~5 GB)")
        print("\nSet dry_run=False to actually download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    provider = ITCHSampleProvider(download_path=output_dir)

    print(f"\nDownloading {date}.NASDAQ_ITCH50.gz...")
    output_path = provider.download(date_or_filename=date, output_path=output_dir)

    print("\n=== Complete ===")
    print(f"Data saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Parse the binary data using the Rust parser or Python")
    print("  2. See Chapter 4 notebooks for parsing examples")


# %% [markdown]
# ### Estimate MBO Cost (ALWAYS DO THIS FIRST!)

# %%
# Estimate cost for default symbols and dates
estimate_mbo_cost()

# %% [markdown]
# ### Download MBO Data

# %%
# Dry run (default) - shows what would be downloaded
download_mbo_data(dry_run=True)

# %%
# Uncomment to actually download (after reviewing cost!)
# download_mbo_data(dry_run=False)

# %% [markdown]
# ### Download NASDAQ ITCH

# %%
# Dry run
download_nasdaq_itch(dry_run=True)

# %%
# Uncomment to download (4-6 GB, takes 30-60 minutes)
# download_nasdaq_itch(dry_run=False)

# %% [markdown]
# ## 4. Load and Explore
#
# Once downloaded, use the loaders throughout the book:

# %%
from data import load_mbo_data, load_nasdaq_itch

# %% [markdown]
# ### MBO Data (Databento)

# %%
# Load MBO data (if available)
mbo = load_mbo_data(symbols=["NVDA"], start_date="2024-11-04", end_date="2024-11-04")

print(f"Shape: {mbo.shape}")
print(f"Columns: {mbo.columns}")
print(f"Memory: {mbo.estimated_size('mb'):.1f} MB")

# %%
# Preview
mbo.head(10)

# %%
# Event type distribution
if "action" in mbo.columns:
    print("Event types:")
    mbo.group_by("action").len().sort("len", descending=True)

# %% [markdown]
# ### NASDAQ ITCH

# %%
# Load ITCH data (if available and parsed)
itch = load_nasdaq_itch()

print(f"Shape: {itch.shape}")
print(f"Memory: {itch.estimated_size('mb'):.1f} MB")

# %%
# Message type distribution
if "msg_type" in itch.columns:
    print("Message types:")
    itch.group_by("msg_type").len().sort("len", descending=True)

# %% [markdown]
# ## 5. Data Profile

# %%
from utils import ML4T_DATA_PATH

# Check for MBO profile
mbo_profile_path = (
    ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "market_by_order" / "profile.json"
)
if mbo_profile_path.exists():
    profile = json.loads(mbo_profile_path.read_text())
    print("=== MBO Profile ===")
    print(f"  Rows: {profile.get('rows', 'N/A'):,}")
else:
    print("MBO profile not found")

# Check for ITCH profile
itch_profile_path = (
    ML4T_DATA_PATH / "equities" / "market" / "microstructure" / "nasdaq_itch" / "profile.json"
)
if itch_profile_path.exists():
    profile = json.loads(itch_profile_path.read_text())
    print("\n=== ITCH Profile ===")
    print(f"  Rows: {profile.get('rows', 'N/A'):,}")
else:
    print("ITCH profile not found")

# %% [markdown]
# ## 6. Loader Options
#
# Both loaders support filtering by symbols and dates.

# %% [markdown]
# ### MBO Data

# %%
# Load specific symbol and date
nvda = load_mbo_data(symbols=["NVDA"], start_date="2024-11-04", end_date="2024-11-04")
print(f"NVDA Nov 4: {nvda.shape}")

# %%
# Load multiple symbols
multi = load_mbo_data(symbols=["NVDA", "SPY"], start_date="2024-11-04", end_date="2024-11-04")
print(f"NVDA + SPY: {multi.shape}")

# %% [markdown]
# ### NASDAQ ITCH

# %%
# Load all parsed messages
itch_day = load_nasdaq_itch()
print(f"ITCH messages: {itch_day.shape}")

# %%
# Load specific message types
# trades = load_nasdaq_itch(message_types=["P", "Q"])

# %% [markdown]
# ## 7. Documentation
#
# ### Databento MBO
#
# - [Databento Documentation](https://databento.com/docs/)
# - [XNAS.ITCH Dataset](https://databento.com/docs/datasets/xnas-itch)
# - [MBO Schema](https://databento.com/docs/schemas/mbo)
#
# ### MBO Event Types
#
# | Action | Description |
# |--------|-------------|
# | `A` | Add order to book |
# | `M` | Modify existing order |
# | `C` | Cancel order |
# | `T` | Trade execution |
# | `F` | Order fully filled |
#
# ### NASDAQ ITCH
#
# - [NASDAQ TotalView-ITCH](https://www.nasdaq.com/docs/TotalView-ITCH-5-0.pdf)
# - [NASDAQ FTP](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/)
#
# ### ITCH Message Types
#
# | Type | Description |
# |------|-------------|
# | `S` | System event |
# | `R` | Stock directory |
# | `A` | Add order (no MPID) |
# | `F` | Add order (with MPID) |
# | `E` | Order executed |
# | `C` | Order executed with price |
# | `X` | Order cancel |
# | `D` | Order delete |
# | `U` | Order replace |
# | `P` | Trade (non-cross) |
# | `Q` | Cross trade |
#
# ### Data Quality Notes
#
# - **MBO**: Pre-processed, clean timestamps, consistent schema
# - **ITCH**: Raw binary, requires parsing, timestamp normalization needed
# - **MBO cost**: ~$0.45-0.50 per symbol per day
# - **ITCH size**: 4-6 GB per day (compressed)

# %% [markdown]
# ## 8. Updating Data
#
# Tick data is typically downloaded **point-in-time** for specific analysis periods.
#
# ### MBO Updates
#
# ```python
# # Download specific dates (ALWAYS estimate first!)
# estimate_mbo_cost(symbols=["AAPL"], start_date="2024-12-01", end_date="2024-12-05")
# download_mbo_data(symbols=["AAPL"], start_date="2024-12-01", end_date="2024-12-05", dry_run=False)
# ```
#
# ### ITCH Updates
#
# ```python
# # Download specific date
# download_nasdaq_itch(date="01152025", dry_run=False)
# ```
#
# ### Cost Considerations
#
# | Data Type | Cost | Recommendation |
# |-----------|------|----------------|
# | MBO | ~$0.50/symbol/day | Download only needed periods |
# | ITCH | Free | Download as needed (5 GB/day) |
#
# **Best practice**: Use ITCH for broad market analysis, MBO for specific symbols.

# %% [markdown]
# ## Summary
#
# | Source | Cost | Format | Use Case |
# |--------|------|--------|----------|
# | Databento MBO | ~$0.50/symbol/day | Pre-processed Parquet | Specific symbol analysis |
# | NASDAQ ITCH | Free | Raw binary (5 GB/day) | Broad market analysis |
#
# **Loaders**:
# - `load_mbo_data(symbols, start_date, end_date)` - Databento MBO
# - `load_nasdaq_itch(date)` - NASDAQ ITCH
#
# **Primary use**: Order book reconstruction, microstructure research, market making analysis.
# **Critical**: Always run `estimate_mbo_cost()` before downloading MBO data!
