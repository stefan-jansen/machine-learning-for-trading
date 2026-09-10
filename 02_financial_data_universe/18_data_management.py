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

import plotly.graph_objects as go
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

from utils.downloading import update_through_last_complete_bar
from utils.paths import REPO_ROOT, get_output_dir
from utils.style import COLORS, show_plotly_with_alt


def _rel(path):
    """Repo-relative display path (keeps absolute machine paths out of outputs)."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


# Working directory for this notebook's storage examples. Wipe any artifacts
# from a previous run so the demo is fully reproducible.
STORAGE_DIR = get_output_dir(2, "data_management")
if STORAGE_DIR.exists():
    shutil.rmtree(STORAGE_DIR)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Storage directory: {_rel(STORAGE_DIR)}")


# %% [markdown]
# ### Declared parameters
#
# The demo universe, its window and the batch worker count are the three things a CI run
# would want to narrow, so they are declared rather than written into the call that uses
# them. `REBASE_LEVEL` is the value each series is rebased to in the figure below, and it
# appears in the axis label as well as the arithmetic.

# %% tags=["parameters"]
ETF_SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
DEMO_START = "2024-01-01"
DEMO_END = "2024-12-31"
DEMO_PROVIDER = "yahoo"
MAX_WORKERS = 4
DEMO_SYMBOL = "AAPL"
PIPELINE_UNIVERSE = "etf_momentum"
REBASE_LEVEL = 100.0

# %% [markdown]
# ---
#
# ## 1. DataManager: The Unified Entry Point
#
# `DataManager` puts provider selection, storage and updates behind one interface. Compare:
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
aapl = dm.fetch(DEMO_SYMBOL, DEMO_START, DEMO_END, provider=DEMO_PROVIDER)

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
etf_symbols = ETF_SYMBOLS
etf_data = dm.batch_load(
    symbols=etf_symbols,
    start=DEMO_START,
    end=DEMO_END,
    provider=DEMO_PROVIDER,
    max_workers=MAX_WORKERS,
)

print(f"Combined: {etf_data.shape[0]:,} rows across {etf_data['symbol'].n_unique()} symbols")

# %% [markdown]
# A row count says the call returned something; it does not say what. Rebasing each series to
# a common level at the first session makes the frame legible, and colouring by asset class
# rather than by ticker makes the point of a multi-asset loader visible: the three equity
# lines travel together and the bond and gold lines do not, which is the dispersion a batch
# spanning three asset classes exists to capture and a single-asset loader cannot show.

# %%
etf_rebased = etf_data.sort("timestamp").with_columns(
    (pl.col("close") / pl.col("close").first().over("symbol") * REBASE_LEVEL).alias("rebased")
)

etf_class = {
    "SPY": "Equities",
    "QQQ": "Equities",
    "IWM": "Equities",
    "TLT": "Long bonds",
    "GLD": "Gold",
}
class_color = {"Equities": COLORS["blue"], "Long bonds": COLORS["copper"], "Gold": COLORS["amber"]}

fig = go.Figure()
seen: set[str] = set()
for sym in etf_symbols:
    s = etf_rebased.filter(pl.col("symbol") == sym)
    cls = etf_class[sym]
    fig.add_trace(
        go.Scatter(
            x=s["timestamp"].to_list(),
            y=s["rebased"].to_list(),
            mode="lines",
            line=dict(color=class_color[cls], width=1.5),
            name=cls,
            legendgroup=cls,
            showlegend=cls not in seen,
            text=sym,
            hovertemplate="%{text}: %{y:.1f}<extra></extra>",
        )
    )
    seen.add(cls)
fig.add_hline(y=REBASE_LEVEL, line=dict(color=COLORS["neutral"], width=1, dash="dot"))
fig.update_layout(
    title="Batch-loaded ETFs, rebased and coloured by asset class",
    xaxis_title="Date",
    yaxis_title=f"Rebased close (first session = {REBASE_LEVEL:.0f})",
    height=420,
    legend_title="Asset class",
)
show_plotly_with_alt(
    fig,
    "Five lines over one year, rebased to a common level at the first session and coloured "
    "in three groups for equities, long bonds and gold, with a dotted reference line at "
    "the rebase level. The three equity lines rise together into the autumn and then "
    "separate, two finishing well above the reference line and one falling back toward "
    "it. The gold line runs highest through the autumn and eases back to join the "
    "equity leaders. The bond line spends almost the whole year below the reference "
    "line and ends furthest below it.",
)

# %% [markdown]
# ---
#
# ## 2. Universe: Predefined Symbol Lists
#
# Instead of maintaining symbol lists in YAML or hardcoding them, ml4t-data
# ships curated universes that stay current with index rebalances.

# %%
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

# %% [markdown]
# A universe feeds straight into `batch_load`. The full list is the whole index, which is
# more fetching than a demonstration needs, so this takes a slice of it.

# %%
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
# layer used throughout ml4t-data. The HiveStorage backend encodes a logical key such as
# `equities/daily/AAPL` as a filesystem-safe directory name and nests Hive-style year and
# month partitions underneath:
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

# Given a storage backend, the same manager gains load, update and metadata operations.
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

# %% [markdown]
# `storage.list_keys()` walks the on-disk layout rather than reading a manifest, so it
# reports what is actually there even if a metadata file is missing or stale.
#
# The read below carries a date range, and only the partitions overlapping that range are
# opened. Reading one year out of a two-year archive touches about half the files.

# %%
stored_symbols = sorted(storage.list_keys())
print(f"Symbols in storage: {stored_symbols}")
aapl_2024 = storage.read(
    stored_keys[DEMO_SYMBOL],
    start_date=datetime.strptime(DEMO_START, "%Y-%m-%d"),
    end_date=datetime.strptime(DEMO_END, "%Y-%m-%d"),
).collect()
print(f"\n{DEMO_SYMBOL} {DEMO_START} to {DEMO_END}: {len(aapl_2024)} rows (partition-pruned)")
print(f"Date range: {aapl_2024['timestamp'].min().date()} to {aapl_2024['timestamp'].max().date()}")

# %% [markdown]
# ### Metadata

# %%
for symbol in symbols:
    meta = dm_stored.get_metadata(symbol)
    if meta:
        print(f"\n{symbol}:")
        for k, v in list(meta.items())[:5]:
            print(f"  {k}: {v}")

# %% [markdown]
# ### Inspect Partition Structure

# %% [markdown]
# The directory names are not addressable from outside: the key is encoded for filesystem
# safety, and each write commits into a new generation directory so a failed write cannot
# leave a half-written partition visible. `partitions()` is therefore how a caller asks what
# the store wrote, rather than listing a path it guessed.

# %%
for symbol in symbols:
    parts = storage.partitions(stored_keys[symbol])
    print(f"{symbol}: {len(parts)} partitions, {sum(p.size_bytes for p in parts) / 1024:.1f} KB")

print("\nAAPL partitions (first 8):")
for part in storage.partitions(stored_keys[DEMO_SYMBOL])[:8]:
    print(f"  {part.label}  ({part.size_bytes / 1024:.1f} KB)")

# %% [markdown]
# The two-year load lands as one Parquet file per calendar month, which is the
# `partition_granularity="month"` setting above. Two things follow from that and the
# chart below shows both. A date-range query reads only the months it overlaps, so the
# partition-pruned read printed earlier touched a fraction of the files. And an
# incremental update writes one new file rather than rewriting anything, because a month
# that has closed never changes.
#
# The near-uniform file sizes are the visible consequence: every month holds about the
# same number of trading days, so no partition is large enough to dominate a read.

# %%
aapl_sizes = pl.DataFrame(
    [
        {"period": part.label, "size_kb": part.size_bytes / 1024}
        for part in storage.partitions(stored_keys[DEMO_SYMBOL])
    ]
)

fig = go.Figure(
    go.Bar(
        x=aapl_sizes["period"].to_list(),
        y=aapl_sizes["size_kb"].to_list(),
        marker_color=COLORS["blue"],
    )
)
fig.update_layout(
    title=f"{DEMO_SYMBOL} Hive partitions, one per month",
    xaxis_title="Partition (year-month)",
    yaxis_title="Partition size (KB)",
    height=420,
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "A bar per calendar month showing the size in kilobytes of that month's Parquet "
    "partition. The bars are of similar height across the whole span, with small "
    "variation and no month standing out.",
)

# %% [markdown]
# ---
#
# ## 4. Incremental Updates
#
# The key workflow: download history once, then **update daily** with only new data.

# %% [markdown]
# ### Update a Symbol
#
# The delta is everything since the last stored bar, up to the newest bar the
# vendor has actually published. That bound is not optional: Yahoo returns the
# current exchange date as a row with accumulating volume and no
# open/high/low/close, and the provider rejects a bar whose prices are null.
# `DataManager.update()` fetches to `datetime.now()`, so it asks for that row on
# every trading day. `update_through_last_complete_bar` asks for the same delta
# and finds the end of the window instead of computing it - it steps back a day
# at a time while the provider refuses the window, because the placeholder row
# usually resolves a few hours after the close and sometimes does not. A refused
# window is logged at error level, so an error line followed by a row count is
# the retreat working rather than a failure.

# %%
# Fetch every bar since the last stored one, up to the last complete session.
for symbol in symbols:
    rows = update_through_last_complete_bar(
        dm_stored, storage, symbol, provider="yahoo", lookback_days=7
    )
    print(f"  Updated {symbol} → {rows} rows")

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
# `exclude_weekends=True` stops Saturdays and Sundays counting as gaps, which is the only
# part of the trading calendar a detector can infer without being handed one.
#
# What it reports therefore depends on the shape of the series it is given, and the two
# shapes give opposite answers. The cached series here is calendar-dense: every non-trading
# day carries the prior close forward, so there is nothing missing to find and the detector
# reports none. A sparse feed carrying only trading days would have the detector flag every
# holiday, because a holiday and an outage are the same absence to anything without an
# exchange calendar. Neither answer is wrong and neither is a coverage check on its own.
#
# Before backtesting, verify data completeness. The IncrementalUpdater
# can detect missing trading days.

# %%
from ml4t.data.update_manager import GapDetector

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
pipeline_output = production_pipeline(
    universe_name=PIPELINE_UNIVERSE,
    start=DEMO_START,
    end=DEMO_END,
    storage_path=STORAGE_DIR / "pipeline_demo",
)

pipeline_output.head()

# %% [markdown]
# The issue count printed above is the extreme-return check firing, not corrupt data. The
# validator is configured with a maximum daily move, and on a one-year ETF panel the largest
# move in each series is the one it flags. That is the check working: it surfaces candidates
# and says nothing about which of them are defects.
#
# Deciding what to do with a candidate is a separate step, and it belongs downstream where
# the context is. `13_data_quality_framework` scores exactly this kind of flag against the
# corporate-action columns of the same file, which is the difference between a candidate
# list and a defect list.

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
