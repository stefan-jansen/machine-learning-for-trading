# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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
# # Data Management: From Download to Production Pipeline
#
# **Docker image**: `ml4t`
#
# **Chapter 2: The Financial Data Universe**
#
# Previous notebooks fetched and validated data. This notebook shows how to
# **manage** it at scale using ml4t-data's production features:
#
# - **DataManager**: Unified entry point for fetching, storing, and updating
# - **Universe**: Predefined symbol lists (S&P 500, NASDAQ 100, etc.)
# - **HiveStorage**: Partitioned Parquet for fast queries and incremental writes
# - **Incremental Updates**: Keep data fresh without re-downloading history
# - **CLI**: Command-line interface for scripted workflows
#
# ## Learning Objectives
#
# By completing this notebook, you will:
# 1. Use `DataManager` as a single entry point for all data operations
# 2. Load predefined universes with the `Universe` class
# 3. Store and query data with Hive-partitioned Parquet
# 4. Perform incremental updates and detect gaps
# 5. Use the `ml4t-data` CLI for scripted workflows
#
# ## Why This Matters
#
# A one-time download is fine for a tutorial. A trading system needs:
# - **Daily updates** that only fetch new data (10x faster than full refresh)
# - **Partitioned storage** that supports fast date-range queries
# - **Gap detection** to ensure completeness before backtesting
# - **Metadata tracking** so you know what you have and when it was updated
#
# > **ml4t-data docs**: See the [Incremental Updates Guide](https://ml4trading.io/docs/data/user-guide/incremental-updates/)
# > and [Storage Guide](https://ml4trading.io/docs/data/user-guide/storage/) for full reference.
#
# **Prerequisites**: ml4t-data installed; live network access for Yahoo Finance.

# %% [markdown]
# ## Setup

# %%
"""Data Management — DataManager, Universe, HiveStorage, and incremental updates."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog

# ml4t-data emits structured debug logs on every fetch/store; route them
# through stdlib logging at WARNING so the notebook output stays focused on
# the demonstration.
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
)

# ml4t-data core imports
from ml4t.data import DataManager
from ml4t.data.storage import HiveStorage
from ml4t.data.storage.backend import StorageConfig
from ml4t.data.universe import Universe

from utils.paths import get_output_dir

# Working directory for this notebook's storage examples. Wipe any artifacts
# from a previous run so the demo is fully reproducible.
STORAGE_DIR = get_output_dir(2, "data_management")
if STORAGE_DIR.exists():
    shutil.rmtree(STORAGE_DIR)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Storage directory: {STORAGE_DIR}")


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ---
#
# ## 1. DataManager: The Unified Entry Point
#
# `DataManager` abstracts away provider selection, storage, and updates
# behind a single interface. Compare:
#
# ```python
# # Without DataManager (manual)
# provider = YahooFinanceProvider()
# df = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31", "daily")
#
# # With DataManager (unified)
# dm = DataManager()
# df = dm.fetch("AAPL", "2024-01-01", "2024-12-31")
# ```
#
# The real power shows with batch operations, storage integration, and updates.

# %% [markdown]
# ### Fetch: Single Symbol

# %%
# DataManager without storage — pure fetch mode
dm = DataManager()

# Fetch a single symbol (defaults to Yahoo Finance for equities)
aapl = dm.fetch("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")

print(f"AAPL: {aapl.shape[0]} rows, {aapl.shape[1]} columns")
print(f"Date range: {aapl['timestamp'].min().date()} to {aapl['timestamp'].max().date()}")
print(f"Columns: {aapl.columns}")
aapl.head(3)

# %% [markdown]
# ### Batch Fetch: Multiple Symbols
#
# `batch_load` fetches multiple symbols in parallel and returns a single
# stacked DataFrame — the standard multi-asset format used throughout the book.

# %%
# Fetch 5 ETFs in parallel
etf_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
etf_data = dm.batch_load(
    symbols=etf_symbols,
    start="2024-01-01",
    end="2024-12-31",
    provider="yahoo",
    max_workers=4,
)

print(f"Combined: {etf_data.shape[0]:,} rows across {etf_data['symbol'].n_unique()} symbols")

etf_data.group_by("symbol").len().sort("symbol")

# %% [markdown]
# ---
#
# ## 2. Universe: Predefined Symbol Lists
#
# Instead of maintaining symbol lists in YAML or hardcoding them, ml4t-data
# ships curated universes that stay current with index rebalances.

# %%
# List available universes
print("Available universes:")
for name in Universe.list_universes():
    symbols = Universe.get(name)
    print(f"  {name}: {len(symbols)} symbols")

# %%
# Access a universe directly
sp500 = Universe.SP500
print(f"\nS&P 500: {len(sp500)} symbols")
print(f"First 10: {sp500[:10]}")
print(f"Last 10:  {sp500[-10:]}")

# %%
# Use with DataManager.batch_load_universe for one-line loading
# (fetches all 503 S&P 500 symbols — use a smaller slice for demo)
sp500_sample = dm.batch_load(
    symbols=sp500[:5],
    start="2024-06-01",
    end="2024-12-31",
    provider="yahoo",
)
print(
    f"S&P 500 sample: {sp500_sample.shape[0]:,} rows, {sp500_sample['symbol'].n_unique()} symbols"
)

# %%
# Custom universes for your strategy
Universe.add_custom("etf_momentum", ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"])
Universe.add_custom("crypto_arb", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])

print("\nCustom universes registered:")
for name in ["etf_momentum", "crypto_arb"]:
    print(f"  {name}: {Universe.get(name)}")

# %% [markdown]
# ---
#
# ## 3. HiveStorage: Partitioned Parquet
#
# For data you'll query repeatedly, Hive-partitioned Parquet is the storage
# layer used throughout ml4t-data. The HiveStorage backend collapses the
# logical key `equities/daily/AAPL` to a filesystem-safe directory name and
# nests Hive-style year/month partitions underneath:
#
# ```
# hive_demo/
# ├── .metadata/
# │   └── equities_daily_AAPL.json
# └── equities_daily_AAPL/
#     ├── year=2024/month=1/data.parquet
#     ├── year=2024/month=2/data.parquet
#     └── ...
# ```
#
# **Benefits over flat files**:
# - **Partition pruning**: Query "last 30 days" reads 1 file, not all of history
# - **Incremental writes**: New data appends without rewriting existing partitions
# - **Metadata tracking**: Know when each symbol was last updated

# %% [markdown]
# ### DataManager with Storage

# %%
# Initialize storage
storage_config = StorageConfig(
    base_path=STORAGE_DIR / "hive_demo",
    compression="zstd",
    partition_granularity="month",
)
storage = HiveStorage(config=storage_config)

# DataManager with storage — enables load/update/metadata operations
dm_stored = DataManager(storage=storage)

# %% [markdown]
# ### Load and Store
#
# `DataManager.load()` fetches from the provider and writes to Hive
# partitions in one call. The storage key encodes the asset class, frequency,
# and symbol.

# %%
symbols = ["AAPL", "MSFT", "GOOGL"]
stored_keys = {}
for symbol in symbols:
    key = dm_stored.load(symbol, "2023-01-01", "2024-12-31", provider="yahoo")
    stored_keys[symbol] = key
    print(f"  Stored {symbol} → key: {key}")

# %% [markdown]
# ### Query Stored Data

# %%
# List what's in storage. `storage.list_keys()` walks the on-disk layout, so it
# reports every symbol regardless of metadata-file contents.
stored_symbols = sorted(storage.list_keys())
print(f"Symbols in storage: {stored_symbols}")

# Read back with a date-range filter — only the matching month=k partitions
# touch disk, so reading 2024 from a 2-year archive halves the I/O.
aapl_2024 = storage.read(
    stored_keys["AAPL"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
).collect()
print(f"\nAAPL 2024 only: {len(aapl_2024)} rows (partition-pruned)")
print(f"Date range: {aapl_2024['timestamp'].min().date()} to {aapl_2024['timestamp'].max().date()}")

# %% [markdown]
# ### Metadata

# %%
# Check metadata for stored symbols
for symbol in symbols:
    meta = dm_stored.get_metadata(symbol)
    if meta:
        print(f"\n{symbol}:")
        for k, v in list(meta.items())[:5]:
            print(f"  {k}: {v}")

# %% [markdown]
# ### Inspect Partition Structure

# %%
# See the actual file layout
hive_root = STORAGE_DIR / "hive_demo"
parquet_files = sorted(hive_root.rglob("*.parquet"))
print(f"Total Parquet files: {len(parquet_files)}")
print("\nExample partition paths (first 8):")
for f in parquet_files[:8]:
    rel = f.relative_to(hive_root)
    size_kb = f.stat().st_size / 1024
    print(f"  {rel}  ({size_kb:.1f} KB)")

# %% [markdown]
# ---
#
# ## 4. Incremental Updates
#
# The key workflow: download history once, then **update daily** with only new data.

# %% [markdown]
# ### Update a Symbol

# %%
# update() checks what's already stored and only fetches new data
for symbol in symbols:
    key = dm_stored.update(symbol, lookback_days=7, provider="yahoo")
    print(f"  Updated {symbol} → {key}")

# Verify data is current
for symbol in symbols:
    meta = dm_stored.get_metadata(symbol)
    if meta and "last_updated" in meta:
        print(f"  {symbol} last updated: {meta['last_updated']}")

# %% [markdown]
# ### Understanding Update Strategies
#
# ml4t-data supports four update strategies:
#
# | Strategy | Behavior | Use Case |
# |----------|----------|----------|
# | `INCREMENTAL` | Only fetch data after last stored timestamp | Daily updates (default) |
# | `APPEND_ONLY` | Never modify existing rows | Audit-safe archives |
# | `FULL_REFRESH` | Replace all data for the symbol | Recovery after corruption |
# | `BACKFILL` | Fill gaps in historical data | Fix missing periods |
#
# The default `INCREMENTAL` strategy is correct for most workflows.
# `DataManager.update()` uses it automatically.

# %% [markdown]
# ### Gap Detection
#
# Before backtesting, verify data completeness. The IncrementalUpdater
# can detect missing trading days.

# %%
from ml4t.data.update_manager import GapDetector

# Pass `exclude_weekends=True` so Saturdays and Sundays don't count as gaps.
# The cached series here is calendar-dense (each non-trading day carries the
# prior close forward), so the detector reports no gaps. For a sparse,
# trading-days-only feed it would instead flag every missing session, including
# holidays — without an exchange calendar it cannot tell a holiday from a true
# gap, so pair it with a calendar-aware check for end-of-day pipelines.
gap_detector = GapDetector(exclude_weekends=True)

for symbol, key in stored_keys.items():
    df = storage.read(key).collect()
    gaps = gap_detector.detect_gaps(df, frequency="daily")
    if gaps:
        print(f"{symbol}: {len(gaps)} gap(s) detected")
        for gap in gaps[:3]:
            print(f"  {gap['start'].date()} to {gap['end'].date()} ({gap['size_days']} days)")
    else:
        print(f"{symbol}: No gaps (complete)")

# %% [markdown]
# ---
#
# ## 5. Command-Line Interface
#
# ml4t-data includes a CLI for scripted workflows and cron jobs.
# Here are the key commands:
#
# ### Fetch Data
# ```bash
# # Single symbol
# ml4t-data fetch AAPL --start 2024-01-01 --end 2024-12-31
#
# # Multiple symbols
# ml4t-data fetch SPY QQQ IWM TLT --provider yahoo --output data/etfs.parquet
# ```
#
# ### Update Stored Data
# ```bash
# # Update a symbol (incremental — only fetches new data)
# ml4t-data update AAPL --storage-path ./data
#
# # Update all stored symbols
# ml4t-data update --all --storage-path ./data
# ```
#
# ### Validate Data Quality
# ```bash
# # Run OHLCV validation on stored data
# ml4t-data validate ./data/etfs.parquet
# ```
#
# ### List Available Data
# ```bash
# # List symbols in storage
# ml4t-data list --storage-path ./data
#
# # List available providers
# ml4t-data info --providers
# ```
#
# ### Automated Daily Updates (Cron)
# ```bash
# # Daily at 6 PM EST (after US market close), Monday-Friday
# 0 18 * * 1-5 cd ~/ml4t && ml4t-data update --all --storage-path ./data >> logs/update.log 2>&1
# ```

# %% [markdown]
# ---
#
# ## 6. Putting It Together: Production Data Pipeline
#
# Here's the complete workflow combining everything above — the pattern
# used by the book's `data/download_all.py` orchestrator.


# %%
def production_pipeline(
    universe_name: str,
    start: str,
    end: str,
    storage_path: Path,
) -> pl.DataFrame:
    """Fetch, store, validate, and assemble a stacked DataFrame for a universe.

    The same pattern drives `data/download_all.py` for every asset class —
    only the universe and provider differ.
    """
    from ml4t.data.validation import OHLCVValidator

    symbols = Universe.get(universe_name)
    print(f"Universe '{universe_name}': {len(symbols)} symbols")

    config = StorageConfig(base_path=storage_path, compression="zstd")
    store = HiveStorage(config=config)
    manager = DataManager(storage=store, enable_validation=True)

    stored = {}
    for symbol in symbols:
        stored[symbol] = manager.load(symbol, start, end, provider="yahoo")
    print(f"Fetched: {len(stored)} symbols")

    validator = OHLCVValidator(max_return_threshold=0.5)
    issues = 0
    for symbol, key in stored.items():
        df = store.read(key).collect()
        result = validator.validate(df)
        if not result.passed:
            issues += result.error_count
            print(f"  {symbol}: {result.error_count} validation issues")
    print(f"Validated: {issues} total issue(s) across {len(stored)} symbols")

    frames = [
        store.read(key).collect().with_columns(pl.lit(symbol).alias("symbol"))
        for symbol, key in stored.items()
    ]
    combined = pl.concat(frames)
    print(f"Result: {combined.shape[0]:,} rows, {combined['symbol'].n_unique()} symbols")
    return combined


# %%
# Run pipeline on a small universe
pipeline_output = production_pipeline(
    universe_name="etf_momentum",
    start="2024-01-01",
    end="2024-12-31",
    storage_path=STORAGE_DIR / "pipeline_demo",
)

pipeline_output.head()

# %% [markdown]
# A single validation issue per symbol on this 2024 ETF panel comes from the
# `OHLCVValidator(max_return_threshold=0.5)` flagging the largest 1-day move
# in each series — a sanity check, not a data error. The validator surfaces
# candidates; downstream code decides whether to drop, winsorize, or pass
# through. Section 2.6 (data quality) covers the trade-offs.

# %% [markdown]
# ---
#
# ## Summary
#
# | Component | Purpose | Key Method |
# |-----------|---------|------------|
# | **DataManager** | Unified entry point | `fetch()`, `batch_load()`, `load()`, `update()` |
# | **Universe** | Predefined symbol lists | `Universe.SP500`, `Universe.get("nasdaq100")` |
# | **HiveStorage** | Partitioned Parquet | `read()`, `write()`, partition pruning |
# | **GapDetector** | Gap detection in time series | `detect_gaps()`, `detect_gaps_in_storage()` |
# | **CLI** | Scripted workflows & cron | `ml4t-data fetch`, `ml4t-data update` |
#
# ### The ml4t-data Workflow
#
# ```
# 1. Initial load:    dm.load("AAPL", "2020-01-01", "2024-12-31")
# 2. Daily update:    dm.update("AAPL", lookback_days=7)
# 3. Gap check:       gap_detector.detect_gaps(df, frequency="daily")
# 4. Batch load:      dm.batch_load_universe("sp500", start, end)
# 5. Automate:        cron + ml4t-data update --all
# ```
#
# ### Key Takeaways
#
# - **One entry point, many providers.** `DataManager.fetch()` hides whether the
#   bytes come from Yahoo, Binance, AlgoSeek, or local Hive parquet; the user
#   code does not change when providers do.
# - **`load()` is cache-first, `fetch()` is provider-first.** Use `load()` for
#   research / backtesting (fast, offline, deterministic); use `fetch()` only
#   when the cache must be refreshed.
# - **Universes are first-class.** `Universe.SP500` and friends keep symbol
#   lists out of notebook code and version-controlled in the library.
# - **Gap detection is a separate concern.** `GapDetector` runs against
#   already-stored data; missing trading days surface as findings, not silent
#   nulls.
# - **The CLI is the production surface.** Cron-driven `ml4t-data update --all`
#   is the same code path the notebook exercises.
#
# ### Further Reading
#
# - **Incremental updates**: `19_incremental_updates` walks the update strategies
#   from this notebook in detail and shows how to schedule them.
# - **Storage formats**: `20_storage_benchmark_file` compares Parquet, CSV, and HDF5;
#   `21_storage_benchmark_database` benchmarks Postgres-backed alternatives.
# - **Data quality**: `13_data_quality_framework` covers validation and anomaly detection.
# - **Provider comparison**: `16_provider_comparison` demonstrates multi-source acquisition.
# - **ml4t-data docs**: [ml4trading.io/docs/data/](https://ml4trading.io/docs/data/)
