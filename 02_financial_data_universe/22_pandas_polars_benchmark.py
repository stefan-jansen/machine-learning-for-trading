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
# # pandas vs Polars: DataFrame Library Benchmark
#
# **Docker image**: `ml4t`
#
# **Purpose**: Compare pandas and Polars for financial data operations typical in
# ML for trading pipelines. This run measures the pinned environment (pandas
# 2.3.3, Polars 1.41+); the version-detection cell below reports whether the
# pandas-3.0 performance features (Copy-on-Write, PyArrow strings) are active.
#
# **Learning Objectives**:
# - Understand performance characteristics of each library for different operations
# - Know when to use pandas vs Polars based on operation type and data scale
# - Recognize what pandas 3.0's Copy-on-Write and PyArrow strings change, and
#   detect whether the running pandas has them enabled
# - Measure memory efficiency for large financial datasets
#
# **Book Reference**: Chapter 2, Section 2.4 (Storing Data) — engine choice
# trade-offs alongside file and database benchmarks.
#
# **Prerequisites**: Familiarity with pandas/Polars basics; existing storage benchmarks.
#
# ## Key Categories Tested
#
# | Category | Operations | Financial Use Case |
# |----------|-----------|-------------------|
# | A: Rolling | SMA, EMA, rolling std, Sharpe | Time-series features |
# | B: GroupBy | OHLCV resampling, cross-sectional stats | Bar construction |
# | C: Window | Z-scores, percentile ranks, lags | Normalized features |
# | D: Filtering | Multi-condition predicates | Options chain filtering |
# | E: Joins | ASOF (trade-quote), anti-joins | Tick data matching |
# | F: Lazy/Streaming | Parquet scan, predicate pushdown | Large file processing |
# | G: Memory | Peak usage, allocation patterns | Resource constraints |
# | H: Strings | Contains, extract, replace | Ticker manipulation |
#
# ## Quick Start
#
# ```bash
# # Development (S scale)
# BENCHMARK_SCALE=S docker compose run --rm ml4t python 02_financial_data_universe/22_pandas_polars_benchmark.py
#
# # Standard benchmark (L scale)
# BENCHMARK_SCALE=L docker compose run --rm ml4t python 02_financial_data_universe/22_pandas_polars_benchmark.py
#
# # Scale test (XL scale)
# BENCHMARK_SCALE=XL docker compose run --rm ml4t python 02_financial_data_universe/22_pandas_polars_benchmark.py
# ```

# %% [markdown]
# ## Setup and Version Detection

# %%
"""Pandas vs Polars Benchmark — systematic performance comparison across financial data operations."""

import gc
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import psutil
from IPython.display import display
from plotly.subplots import make_subplots

from utils.reproducibility import set_global_seeds
from utils.storage_benchmarks import (
    ACTIVE_SCALE,
    BENCHMARK_DIR,
    N_ROWS_PER_SYMBOL,
    N_SYMBOLS,
    RESULTS_DIR,
    TIMING_RUNS,
    estimate_memory_mb,
    generate_ohlcv_data,
    generate_tick_data,
    get_scale_config,
    time_operation,
)
from utils.style import COLORS

warnings.filterwarnings("ignore")


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ### Version and Feature Detection
#
# pandas 3.0 introduces changes that affect performance, and the cell below
# reports whether the *installed* pandas has them turned on (this pinned run is
# pandas 2.3.3, so they are not):
# - **Copy-on-Write (CoW)**: default in 3.0 — internal views with copy-on-modify semantics
# - **PyArrow-backed strings**: default string dtype in 3.0 — better memory and string operations
# - **New `pd.col()` API**: cleaner column references in assign/groupby

# %%
# Version detection
PANDAS_VERSION = pd.__version__
POLARS_VERSION = pl.__version__

print("=" * 70)
print("DATAFRAME LIBRARY BENCHMARK")
print("=" * 70)
print(f"\npandas version: {PANDAS_VERSION}")
print(f"Polars version: {POLARS_VERSION}")

# pandas 3.0 feature detection
PANDAS_MAJOR = int(PANDAS_VERSION.split(".")[0])
IS_PANDAS_3 = PANDAS_MAJOR >= 3

# Check Copy-on-Write status (enabled by default in pandas 3.0)
try:
    COW_ENABLED = pd.options.mode.copy_on_write
except AttributeError:
    COW_ENABLED = False

# Check for PyArrow string dtype (default in pandas 3.0)
try:
    test_series = pd.Series(["test"])
    PYARROW_STRINGS = "pyarrow" in str(test_series.dtype) or test_series.dtype == "string"
except Exception:
    PYARROW_STRINGS = False

print("\npandas 3.0 features:")
print(f"  Copy-on-Write: {'enabled' if COW_ENABLED else 'disabled'}")
print(f"  PyArrow strings: {'yes' if PYARROW_STRINGS else 'no'}")

# Configure Polars streaming (opt-in to new engine in 1.37+)
POLARS_STREAMING = False
try:
    pl.Config.set_engine_affinity(engine="streaming")
    POLARS_STREAMING = True
    print("\nPolars streaming engine: enabled")
except Exception:
    print("\nPolars streaming engine: not available (requires 1.37+)")

# %% [markdown]
# ## Data Generation
#
# We use the same synthetic OHLCV and tick data generators as the storage benchmarks
# to ensure comparable results across all benchmarks.

# %%
scale_cfg = get_scale_config(ACTIVE_SCALE)
print(f"\nScale: {ACTIVE_SCALE} ({scale_cfg['target_memory']} target)")
print(f"OHLCV: {N_SYMBOLS} symbols × {N_ROWS_PER_SYMBOL:,} rows/symbol")

print("\n=== Generating synthetic data ===\n")

# Generate OHLCV data (Polars native)
ohlcv_pl = generate_ohlcv_data(n_symbols=N_SYMBOLS, n_rows=N_ROWS_PER_SYMBOL)
total_rows = len(ohlcv_pl)
print(f"OHLCV: {total_rows:,} rows ({estimate_memory_mb(ohlcv_pl):.1f} MB)")

# Convert to pandas (triggers CoW in pandas 3.0)
ohlcv_pd = ohlcv_pl.to_pandas()
print(f"pandas memory: {ohlcv_pd.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Generate tick data for join benchmarks
trades_pl, quotes_pl = generate_tick_data(
    n_symbols=min(N_SYMBOLS, 50),  # Limit symbols for tick data
    seed=42,
)
n_trades = len(trades_pl)
n_quotes = len(quotes_pl)
print(f"Trades: {n_trades:,} rows")
print(f"Quotes: {n_quotes:,} rows")

# Convert tick data to pandas
trades_pd = trades_pl.to_pandas()
quotes_pd = quotes_pl.to_pandas()

# Store results
results = []


# %% [markdown]
# ## Helper Functions
#
# Force materialization to ensure fair timing comparisons. Both libraries
# use lazy evaluation in some contexts (Polars explicitly, pandas via CoW).


# %%
def force_eval_pandas(df: pd.DataFrame) -> None:
    """Force pandas DataFrame evaluation by touching all data."""
    # Numeric columns: sum
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        _ = df[numeric_cols].sum().sum()
    # String columns: length (only actual string types)
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols[:2]:  # Limit to avoid slow string ops
        try:
            if df[col].dtype == "object" or "string" in str(df[col].dtype):
                _ = df[col].astype(str).str.len().sum()
        except Exception:
            pass  # Skip if not actually string-like


# %%
def force_eval_polars(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Force Polars DataFrame evaluation."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    # Touch numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Int64)]
    if numeric_cols:
        _ = df.select([pl.col(c).sum() for c in numeric_cols[:5]]).to_dict()
    return df


# %% [markdown]
# ### Benchmark Runner
# Time an operation on both libraries and collect results.


# %%
def benchmark_operation(
    name: str,
    category: str,
    pandas_func,
    polars_func,
    n_runs: int = TIMING_RUNS,
) -> dict:
    """Benchmark an operation on both libraries.

    Returns dict with timing results for both libraries.
    """
    # pandas benchmark
    gc.collect()
    pd_time, pd_result = time_operation(pandas_func, n_runs=n_runs)

    # Polars benchmark
    gc.collect()
    pl_time, pl_result = time_operation(polars_func, n_runs=n_runs)

    # Calculate speedup
    speedup = pd_time / pl_time if pl_time > 0 else float("inf")

    result = {
        "category": category,
        "operation": name,
        "pandas_time": pd_time,
        "polars_time": pl_time,
        "speedup": speedup,
    }

    print(f"  {name}: pandas={pd_time:.4f}s, polars={pl_time:.4f}s, speedup={speedup:.1f}x")

    return result


# %% [markdown]
# ## Category A: Rolling Calculations
#
# Rolling window operations are fundamental to time-series feature engineering.
# We test single-window, multi-horizon, and compound calculations (Sharpe ratio).

# %%
print("\n" + "=" * 70)
print("CATEGORY A: ROLLING CALCULATIONS")
print("=" * 70)

rolling_results = []

# %% [markdown]
# ### A1: Simple Rolling Mean (20-day SMA)
#
# Basic moving average - the foundation of many trading signals.


# %%
def pd_rolling_mean():
    result = ohlcv_pd.groupby("symbol")["close"].rolling(20).mean().reset_index(drop=True)
    _ = result.sum()  # Force evaluation
    return result


# %%
def pl_rolling_mean():
    """Compute 20-day rolling mean per symbol using Polars window expressions."""
    result = ohlcv_pl.with_columns(pl.col("close").rolling_mean(20).over("symbol").alias("sma_20"))
    _ = result.select(pl.col("sma_20").sum()).item()
    return result


r = benchmark_operation("rolling_mean_20", "A_rolling", pd_rolling_mean, pl_rolling_mean)
rolling_results.append(r)

# %% [markdown]
# ### A2: Rolling Standard Deviation (Volatility)
#
# Volatility estimation - critical for risk management and signal normalization.


# %%
def pd_rolling_std():
    result = ohlcv_pd.groupby("symbol")["close"].rolling(20).std().reset_index(drop=True)
    _ = result.sum()
    return result


# %%
def pl_rolling_std():
    """Compute 20-day rolling standard deviation per symbol for volatility estimation."""
    result = ohlcv_pl.with_columns(pl.col("close").rolling_std(20).over("symbol").alias("vol_20"))
    _ = result.select(pl.col("vol_20").sum()).item()
    return result


r = benchmark_operation("rolling_std_20", "A_rolling", pd_rolling_std, pl_rolling_std)
rolling_results.append(r)

# %% [markdown]
# ### A3: Multi-Horizon Rolling (1, 5, 21, 63, 126, 252 days)
#
# Real feature engineering requires multiple lookback windows simultaneously.
# This tests the ability to compute many windows in a single pass.

# %%
HORIZONS = [1, 5, 21, 63, 126, 252]


def pd_multi_horizon():
    result = ohlcv_pd.copy()
    for h in HORIZONS:
        result[f"ret_{h}"] = result.groupby("symbol")["close"].pct_change(h)
    force_eval_pandas(result)
    return result


# %%
def pl_multi_horizon():
    """Compute returns at six horizons in a single with_columns call using Polars expressions."""
    # Polars: all horizons in single with_columns call
    result = ohlcv_pl.with_columns(
        [pl.col("close").pct_change(h).over("symbol").alias(f"ret_{h}") for h in HORIZONS]
    )
    _ = result.select([pl.col(f"ret_{h}").sum() for h in HORIZONS]).to_dict()
    return result


r = benchmark_operation("multi_horizon_returns", "A_rolling", pd_multi_horizon, pl_multi_horizon)
rolling_results.append(r)

# %% [markdown]
# ### A4: Rolling Sharpe Ratio
#
# Compound calculation: rolling mean / rolling std. Tests chained operations.


# %%
def pd_rolling_sharpe():
    result = ohlcv_pd.copy()
    returns = result.groupby("symbol")["close"].pct_change()
    result["sharpe"] = (returns.rolling(63).mean() / returns.rolling(63).std()) * np.sqrt(252)
    force_eval_pandas(result)
    return result


# %%
def pl_rolling_sharpe():
    """Compute 63-day rolling Sharpe ratio via chained Polars window expressions."""
    result = ohlcv_pl.with_columns(
        pl.col("close").pct_change().over("symbol").alias("returns")
    ).with_columns(
        (
            pl.col("returns").rolling_mean(63).over("symbol")
            / pl.col("returns").rolling_std(63).over("symbol")
        )
        .mul(np.sqrt(252))
        .alias("sharpe")
    )
    _ = result.select(pl.col("sharpe").sum()).item()
    return result


r = benchmark_operation("rolling_sharpe_63", "A_rolling", pd_rolling_sharpe, pl_rolling_sharpe)
rolling_results.append(r)

# %% [markdown]
# ### A5: Exponential Moving Average
#
# EMA with span=20 - popular for trend-following signals.


# %%
def pd_ewm():
    result = (
        ohlcv_pd.groupby("symbol")["close"].ewm(span=20, adjust=False).mean().reset_index(drop=True)
    )
    _ = result.sum()
    return result


# %%
def pl_ewm():
    """Compute exponential moving average (span=20) per symbol using Polars ewm_mean."""
    result = ohlcv_pl.with_columns(
        pl.col("close").ewm_mean(span=20, adjust=False).over("symbol").alias("ema_20")
    )
    _ = result.select(pl.col("ema_20").sum()).item()
    return result


r = benchmark_operation("ewm_span_20", "A_rolling", pd_ewm, pl_ewm)
rolling_results.append(r)

results.extend(rolling_results)

# %% [markdown]
# ## Category B: GroupBy Aggregations
#
# GroupBy operations are essential for cross-sectional analysis and resampling.

# %%
print("\n" + "=" * 70)
print("CATEGORY B: GROUPBY AGGREGATIONS")
print("=" * 70)

groupby_results = []

# %% [markdown]
# ### B1: OHLCV Resampling (1-min to daily)
#
# Aggregate minute bars to daily bars - common in bar construction pipelines.


# %%
def pd_resample():
    result = (
        ohlcv_pd.groupby([ohlcv_pd["timestamp"].dt.date, "symbol"])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .reset_index()
    )
    force_eval_pandas(result)
    return result


# %%
def pl_resample():
    """Resample OHLCV to daily bars using Polars group_by with first/last/min/max/sum aggregations."""
    result = ohlcv_pl.group_by([pl.col("timestamp").dt.date().alias("timestamp"), "symbol"]).agg(
        [
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ]
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("ohlcv_resample_daily", "B_groupby", pd_resample, pl_resample)
groupby_results.append(r)

# %% [markdown]
# ### B2: Cross-Sectional Statistics by Date
#
# Compute market-wide statistics for each timestamp.


# %%
def pd_cross_sectional():
    result = ohlcv_pd.groupby("timestamp").agg(
        {
            "close": ["mean", "std", "min", "max"],
            "volume": ["sum", "mean"],
        }
    )
    result.columns = ["_".join(col) for col in result.columns]
    return result.reset_index()


# %%
def pl_cross_sectional():
    """Compute cross-sectional statistics (mean, std, min, max) per timestamp using Polars."""
    result = ohlcv_pl.group_by("timestamp").agg(
        [
            pl.col("close").mean().alias("close_mean"),
            pl.col("close").std().alias("close_std"),
            pl.col("close").min().alias("close_min"),
            pl.col("close").max().alias("close_max"),
            pl.col("volume").sum().alias("volume_sum"),
            pl.col("volume").mean().alias("volume_mean"),
        ]
    )
    force_eval_polars(result)
    return result


r = benchmark_operation(
    "cross_sectional_stats", "B_groupby", pd_cross_sectional, pl_cross_sectional
)
groupby_results.append(r)

# %% [markdown]
# ### B3: Symbol-Level Statistics
#
# Per-symbol summary statistics across all time periods.


# %%
def pd_symbol_stats():
    result = ohlcv_pd.groupby("symbol").agg(
        {
            "close": ["mean", "std", "min", "max", "count"],
            "volume": ["sum", "mean"],
            "high": "max",
            "low": "min",
        }
    )
    result.columns = ["_".join(col) for col in result.columns]
    return result.reset_index()


# %%
def pl_symbol_stats():
    """Compute per-symbol summary statistics (close, volume, high, low) using Polars group_by."""
    result = ohlcv_pl.group_by("symbol").agg(
        [
            pl.col("close").mean().alias("close_mean"),
            pl.col("close").std().alias("close_std"),
            pl.col("close").min().alias("close_min"),
            pl.col("close").max().alias("close_max"),
            pl.col("close").count().alias("close_count"),
            pl.col("volume").sum().alias("volume_sum"),
            pl.col("volume").mean().alias("volume_mean"),
            pl.col("high").max().alias("high_max"),
            pl.col("low").min().alias("low_min"),
        ]
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("symbol_stats", "B_groupby", pd_symbol_stats, pl_symbol_stats)
groupby_results.append(r)

results.extend(groupby_results)

# %% [markdown]
# ## Category C: Window Functions
#
# Window functions compute values relative to other rows in a group.
# These are essential for cross-sectional normalization.

# %%
print("\n" + "=" * 70)
print("CATEGORY C: WINDOW FUNCTIONS")
print("=" * 70)

window_results = []

# %% [markdown]
# ### C1: Cross-Sectional Z-Score
#
# Normalize returns relative to cross-section at each timestamp.


# %%
def pd_zscore():
    result = ohlcv_pd.copy()
    result["returns"] = result.groupby("symbol")["close"].pct_change()
    grouped = result.groupby("timestamp")["returns"]
    result["zscore"] = (result["returns"] - grouped.transform("mean")) / grouped.transform("std")
    force_eval_pandas(result)
    return result


# %%
def pl_zscore():
    """Compute cross-sectional z-score of returns at each timestamp using Polars .over() windows."""
    result = ohlcv_pl.with_columns(
        pl.col("close").pct_change().over("symbol").alias("returns")
    ).with_columns(
        (
            (pl.col("returns") - pl.col("returns").mean().over("timestamp"))
            / pl.col("returns").std().over("timestamp")
        ).alias("zscore")
    )
    _ = result.select(pl.col("zscore").sum()).item()
    return result


r = benchmark_operation("cross_sectional_zscore", "C_window", pd_zscore, pl_zscore)
window_results.append(r)

# %% [markdown]
# ### C2: Percentile Rank
#
# Rank each symbol's return within the cross-section.


# %%
def pd_rank():
    result = ohlcv_pd.copy()
    result["returns"] = result.groupby("symbol")["close"].pct_change()
    result["rank"] = result.groupby("timestamp")["returns"].rank(pct=True)
    force_eval_pandas(result)
    return result


# %%
def pl_rank():
    """Compute percentile rank of returns within each timestamp cross-section using Polars."""
    result = (
        ohlcv_pl.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
        .with_columns(pl.col("returns").rank().over("timestamp").alias("rank_raw"))
        .with_columns(
            (pl.col("rank_raw") / pl.col("rank_raw").max().over("timestamp")).alias("rank_pct")
        )
    )
    _ = result.select(pl.col("rank_pct").sum()).item()
    return result


r = benchmark_operation("percentile_rank", "C_window", pd_rank, pl_rank)
window_results.append(r)

# %% [markdown]
# ### C3: Lagged Values
#
# Create multiple lag columns (1, 5, 21 days) - common for autoregressive features.

# %%
LAGS = [1, 5, 21]


def pd_lags():
    result = ohlcv_pd.copy()
    for lag in LAGS:
        result[f"close_lag_{lag}"] = result.groupby("symbol")["close"].shift(lag)
    force_eval_pandas(result)
    return result


# %%
def pl_lags():
    """Create multiple lag columns (1, 5, 21 days) per symbol using Polars shift with .over()."""
    result = ohlcv_pl.with_columns(
        [pl.col("close").shift(lag).over("symbol").alias(f"close_lag_{lag}") for lag in LAGS]
    )
    _ = result.select([pl.col(f"close_lag_{lag}").sum() for lag in LAGS]).to_dict()
    return result


r = benchmark_operation("lagged_values", "C_window", pd_lags, pl_lags)
window_results.append(r)

results.extend(window_results)

# %% [markdown]
# ## Category D: Filtering
#
# Filter operations select subsets of data based on conditions.
# Complex predicates are common in options chain processing.

# %%
print("\n" + "=" * 70)
print("CATEGORY D: FILTERING")
print("=" * 70)

filter_results = []

# %% [markdown]
# ### D1: Simple Price Filter

# %%
# Precompute price threshold (median)
price_threshold = float(ohlcv_pl.select(pl.col("close").median()).item())


def pd_simple_filter():
    result = ohlcv_pd[ohlcv_pd["close"] > price_threshold]
    force_eval_pandas(result)
    return result


# %%
def pl_simple_filter():
    """Filter rows where close exceeds median price threshold using Polars filter."""
    result = ohlcv_pl.filter(pl.col("close") > price_threshold)
    force_eval_polars(result)
    return result


r = benchmark_operation("simple_filter", "D_filter", pd_simple_filter, pl_simple_filter)
filter_results.append(r)

# %% [markdown]
# ### D2: Multi-Condition Filter
#
# Combine price, volume, and symbol conditions.

# %%
# Get list of symbols for filtering
symbol_list = ohlcv_pl.select("symbol").unique().head(N_SYMBOLS // 2)["symbol"].to_list()
volume_threshold = float(ohlcv_pl.select(pl.col("volume").median()).item())


def pd_multi_filter():
    result = ohlcv_pd[
        (ohlcv_pd["close"] > price_threshold)
        & (ohlcv_pd["volume"] > volume_threshold)
        & (ohlcv_pd["symbol"].isin(symbol_list))
    ]
    force_eval_pandas(result)
    return result


# %%
def pl_multi_filter():
    """Apply compound filter on price, volume, and symbol membership using Polars boolean expressions."""
    result = ohlcv_pl.filter(
        (pl.col("close") > price_threshold)
        & (pl.col("volume") > volume_threshold)
        & (pl.col("symbol").is_in(symbol_list))
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("multi_condition_filter", "D_filter", pd_multi_filter, pl_multi_filter)
filter_results.append(r)

# %% [markdown]
# ### D3: Range Filter (Options-Style)
#
# Simulate filtering an options chain by moneyness and expiry.


# %%
def pd_range_filter():
    # Simulate: price between 95-105% of reference, volume in range
    ref_price = price_threshold
    result = ohlcv_pd[
        (ohlcv_pd["close"] >= ref_price * 0.95)
        & (ohlcv_pd["close"] <= ref_price * 1.05)
        & (ohlcv_pd["volume"] >= volume_threshold * 0.5)
        & (ohlcv_pd["volume"] <= volume_threshold * 2.0)
    ]
    force_eval_pandas(result)
    return result


# %%
def pl_range_filter():
    """Filter by price and volume ranges using Polars is_between for options-style moneyness bands."""
    ref_price = price_threshold
    result = ohlcv_pl.filter(
        pl.col("close").is_between(ref_price * 0.95, ref_price * 1.05)
        & pl.col("volume").is_between(volume_threshold * 0.5, volume_threshold * 2.0)
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("range_filter", "D_filter", pd_range_filter, pl_range_filter)
filter_results.append(r)

results.extend(filter_results)

# %% [markdown]
# ## Category E: Joins
#
# Join operations are critical for tick data processing (trade-quote matching)
# and panel data operations.

# %%
print("\n" + "=" * 70)
print("CATEGORY E: JOINS")
print("=" * 70)

join_results = []

# %% [markdown]
# ### E1: ASOF Join (Trade-Quote Matching)
#
# Match each trade to the most recent quote - fundamental for tick data analysis.

# %%
# Sort data for ASOF join (required)
# pandas merge_asof requires left keys to be sorted by the "on" column
trades_pd_sorted = trades_pd.sort_values("timestamp").reset_index(drop=True)
quotes_pd_sorted = quotes_pd.sort_values("timestamp").reset_index(drop=True)
# Polars requires sort by both by and on columns
trades_pl_sorted = trades_pl.sort(["symbol", "timestamp"])
quotes_pl_sorted = quotes_pl.sort(["symbol", "timestamp"])


def pd_asof_join():
    result = pd.merge_asof(
        trades_pd_sorted,
        quotes_pd_sorted,
        on="timestamp",
        by="symbol",
        direction="backward",
    )
    force_eval_pandas(result)
    return result


# %%
def pl_asof_join():
    """Match trades to most recent quotes via Polars join_asof with backward strategy."""
    result = trades_pl_sorted.join_asof(
        quotes_pl_sorted,
        on="timestamp",
        by="symbol",
        strategy="backward",
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("asof_join", "E_join", pd_asof_join, pl_asof_join)
join_results.append(r)

# %% [markdown]
# ### E2: Anti-Join (Find Unmatched Trades)
#
# Find trades without matching quotes - useful for data quality checks.
# pandas 3.0 introduces `how='left_anti'` in merge.


# %%
def pd_anti_join():
    # pandas 3.0 anti-join (or fallback for older versions)
    if IS_PANDAS_3:
        try:
            result = (
                pd.merge(
                    trades_pd_sorted,
                    quotes_pd_sorted[["timestamp", "symbol"]].drop_duplicates(),
                    on=["timestamp", "symbol"],
                    how="left",
                    indicator=True,
                )
                .query("_merge == 'left_only'")
                .drop("_merge", axis=1)
            )
        except Exception:
            # Fallback
            merged = trades_pd_sorted.merge(
                quotes_pd_sorted[["timestamp", "symbol"]].drop_duplicates(),
                on=["timestamp", "symbol"],
                how="left",
                indicator=True,
            )
            result = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
    else:
        merged = trades_pd_sorted.merge(
            quotes_pd_sorted[["timestamp", "symbol"]].drop_duplicates(),
            on=["timestamp", "symbol"],
            how="left",
            indicator=True,
        )
        result = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)
    force_eval_pandas(result)
    return result


# %%
def pl_anti_join():
    """Find trades without matching quotes using Polars native anti-join."""
    result = trades_pl_sorted.join(
        quotes_pl_sorted.select(["timestamp", "symbol"]).unique(),
        on=["timestamp", "symbol"],
        how="anti",
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("anti_join", "E_join", pd_anti_join, pl_anti_join)
join_results.append(r)

# %% [markdown]
# ### E3: Inner Join
#
# Standard inner join for combining related tables.

# %%
# Create a smaller lookup table for join benchmark
symbols_df_pd = pd.DataFrame(
    {
        "symbol": ohlcv_pd["symbol"].unique(),
        "sector": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], size=N_SYMBOLS),
    }
)
symbols_df_pl = pl.DataFrame(
    {
        "symbol": ohlcv_pl.select("symbol").unique()["symbol"],
        "sector": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], size=N_SYMBOLS),
    }
)


def pd_inner_join():
    result = ohlcv_pd.merge(symbols_df_pd, on="symbol", how="inner")
    force_eval_pandas(result)
    return result


# %%
def pl_inner_join():
    """Inner-join OHLCV with sector lookup table using Polars join on symbol."""
    result = ohlcv_pl.join(symbols_df_pl, on="symbol", how="inner")
    force_eval_polars(result)
    return result


r = benchmark_operation("inner_join", "E_join", pd_inner_join, pl_inner_join)
join_results.append(r)

results.extend(join_results)

# %% [markdown]
# ## Category F: Lazy Evaluation and Streaming
#
# Test lazy evaluation benefits on parquet files. This is where Polars
# typically shows largest advantages through predicate pushdown.

# %%
print("\n" + "=" * 70)
print("CATEGORY F: LAZY/STREAMING")
print("=" * 70)

lazy_results = []

# Save data to parquet for lazy benchmarks
parquet_path = BENCHMARK_DIR / f"ohlcv_{ACTIVE_SCALE.lower()}.parquet"
ohlcv_pl.write_parquet(parquet_path)
print(f"Parquet file: {parquet_path.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ### F1: Lazy Filter (Predicate Pushdown)
#
# Filter during scan - Polars can push predicates into the parquet reader.


# %%
def pd_lazy_filter():
    # pandas: must read all, then filter
    df = pd.read_parquet(parquet_path)
    result = df[df["close"] > price_threshold]
    force_eval_pandas(result)
    return result


# %%
def pl_lazy_filter():
    """Scan parquet with predicate pushdown, filtering during read via Polars lazy API."""
    # Polars: predicate pushdown - filter during read
    result = pl.scan_parquet(parquet_path).filter(pl.col("close") > price_threshold).collect()
    force_eval_polars(result)
    return result


r = benchmark_operation("lazy_filter", "F_lazy", pd_lazy_filter, pl_lazy_filter)
lazy_results.append(r)

# %% [markdown]
# ### F2: Column Projection
#
# Read only needed columns - both libraries optimize this.


# %%
def pd_column_projection():
    df = pd.read_parquet(parquet_path, columns=["timestamp", "symbol", "close", "volume"])
    force_eval_pandas(df)
    return df


# %%
def pl_column_projection():
    """Read only selected columns from parquet using Polars lazy scan with column projection."""
    df = pl.scan_parquet(parquet_path).select(["timestamp", "symbol", "close", "volume"]).collect()
    force_eval_polars(df)
    return df


r = benchmark_operation("column_projection", "F_lazy", pd_column_projection, pl_column_projection)
lazy_results.append(r)

# %% [markdown]
# ### F3: Combined Filter + Aggregation (Query Optimization)
#
# Complex query that benefits from Polars' query optimizer.


# %%
def pd_combined_query():
    df = pd.read_parquet(parquet_path)
    result = (
        df[df["close"] > price_threshold]
        .groupby("symbol")
        .agg({"close": "mean", "volume": "sum"})
        .reset_index()
    )
    force_eval_pandas(result)
    return result


# %%
def pl_combined_query():
    """Filter and aggregate in one lazy query, leveraging Polars query optimizer."""
    result = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("close") > price_threshold)
        .group_by("symbol")
        .agg(
            [
                pl.col("close").mean(),
                pl.col("volume").sum(),
            ]
        )
        .collect()
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("combined_query", "F_lazy", pd_combined_query, pl_combined_query)
lazy_results.append(r)

results.extend(lazy_results)

# %% [markdown]
# ## Category G: Memory Efficiency
#
# Measure memory usage during operations. Polars typically uses less memory
# due to its columnar layout and streaming capabilities.

# %%
print("\n" + "=" * 70)
print("CATEGORY G: MEMORY EFFICIENCY")
print("=" * 70)

memory_results = []

# %% [markdown]
# ### G1: Baseline Memory Usage

# %%
process = psutil.Process()

# pandas memory
gc.collect()
mem_before = process.memory_info().rss / 1e6
_ = ohlcv_pd.copy()  # Copy triggers CoW in pandas 3.0
mem_after = process.memory_info().rss / 1e6
pd_mem = mem_after - mem_before

# Polars memory
gc.collect()
mem_before = process.memory_info().rss / 1e6
_ = ohlcv_pl.clone()
mem_after = process.memory_info().rss / 1e6
pl_mem = mem_after - mem_before

print("Copy/Clone operation memory:")
print(f"  pandas: {pd_mem:.1f} MB")
print(f"  Polars: {pl_mem:.1f} MB")

# Estimated DataFrame memory
pd_estimated = ohlcv_pd.memory_usage(deep=True).sum() / 1e6
pl_estimated = ohlcv_pl.estimated_size("mb")

print("\nDataFrame estimated size:")
print(f"  pandas: {pd_estimated:.1f} MB")
print(f"  Polars: {pl_estimated:.1f} MB")
print(f"  Ratio: {pd_estimated / pl_estimated:.2f}x")

memory_results.append(
    {
        "category": "G_memory",
        "operation": "df_estimated_size",
        "pandas_time": pd_estimated,  # Using time fields for memory (MB)
        "polars_time": pl_estimated,
        "speedup": pd_estimated / pl_estimated if pl_estimated > 0 else 1.0,
    }
)

# %% [markdown]
# ## Category H: String Operations
#
# String operations are often a bottleneck. pandas 3.0's PyArrow strings
# should improve performance here.

# %%
print("\n" + "=" * 70)
print("CATEGORY H: STRING OPERATIONS")
print("=" * 70)

string_results = []

# %% [markdown]
# ### H1: String Contains


# %%
def pd_str_contains():
    result = ohlcv_pd[ohlcv_pd["symbol"].str.contains("SYM_0", regex=False)]
    force_eval_pandas(result)
    return result


# %%
def pl_str_contains():
    """Filter rows by literal substring match on symbol using Polars str.contains."""
    result = ohlcv_pl.filter(pl.col("symbol").str.contains("SYM_0", literal=True))
    force_eval_polars(result)
    return result


r = benchmark_operation("str_contains", "H_string", pd_str_contains, pl_str_contains)
string_results.append(r)

# %% [markdown]
# ### H2: String Replace


# %%
def pd_str_replace():
    result = ohlcv_pd.copy()
    result["symbol_new"] = result["symbol"].str.replace("SYM_", "SYMBOL_", regex=False)
    force_eval_pandas(result)
    return result


# %%
def pl_str_replace():
    """Replace substring in symbol column using Polars str.replace with literal mode."""
    result = ohlcv_pl.with_columns(
        pl.col("symbol").str.replace("SYM_", "SYMBOL_", literal=True).alias("symbol_new")
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("str_replace", "H_string", pd_str_replace, pl_str_replace)
string_results.append(r)

# %% [markdown]
# ### H3: String Extract (Pattern Matching)


# %%
def pd_str_extract():
    result = ohlcv_pd.copy()
    result["symbol_num"] = result["symbol"].str.extract(r"SYM_(\d+)", expand=False)
    force_eval_pandas(result)
    return result


# %%
def pl_str_extract():
    """Extract numeric suffix from symbol via regex capture group using Polars str.extract."""
    result = ohlcv_pl.with_columns(
        pl.col("symbol").str.extract(r"SYM_(\d+)", group_index=1).alias("symbol_num")
    )
    force_eval_polars(result)
    return result


r = benchmark_operation("str_extract", "H_string", pd_str_extract, pl_str_extract)
string_results.append(r)

results.extend(string_results)

# %% [markdown]
# ## Results Summary

# %%
print("\n" + "=" * 70)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 70)

# Convert to DataFrame
results_df = pl.DataFrame(results)

# Add memory results if available
if memory_results:
    memory_df = pl.DataFrame(memory_results)
    results_df = pl.concat([results_df, memory_df])

# Summary by category
print("\n### By Category (Mean Speedup)")
category_summary = (
    results_df.group_by("category")
    .agg(
        [
            pl.col("speedup").mean().alias("mean_speedup"),
            pl.col("speedup").min().alias("min_speedup"),
            pl.col("speedup").max().alias("max_speedup"),
            pl.len().alias("n_ops"),
        ]
    )
    .sort("mean_speedup", descending=True)
)
display(category_summary)

# Overall statistics
print("\n### Overall Statistics")
overall_speedup = results_df.select(pl.col("speedup").mean()).item()
print(f"Mean speedup (Polars vs pandas): {overall_speedup:.1f}x")

operations_faster = results_df.filter(pl.col("speedup") > 1.0).height
operations_slower = results_df.filter(pl.col("speedup") < 1.0).height
print(f"Operations where Polars is faster: {operations_faster}/{len(results_df)}")
print(f"Operations where pandas is faster: {operations_slower}/{len(results_df)}")

# Detailed results
print("Detailed Results (sorted by speedup):")
display(results_df.sort("speedup", descending=True))

# %% [markdown]
# ## Visualization

# %%
# Create visualization
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Speedup by Category",
        "Operation Times (log scale)",
        "Speedup Distribution",
        "pandas vs Polars Times",
    ],
    specs=[
        [{"type": "bar"}, {"type": "bar"}],
        [{"type": "histogram"}, {"type": "scatter"}],
    ],
)

# 1. Speedup by category (bar chart)
cat_data = category_summary.sort("mean_speedup", descending=True)
fig.add_trace(
    go.Bar(
        x=cat_data["category"].to_list(),
        y=cat_data["mean_speedup"].to_list(),
        marker_color=COLORS["blue"],
        text=[f"{s:.1f}x" for s in cat_data["mean_speedup"].to_list()],
        textposition="outside",
    ),
    row=1,
    col=1,
)
fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

# 2. Operation times (grouped bar)
sorted_results = results_df.sort("speedup", descending=True).head(15)
fig.add_trace(
    go.Bar(
        name="pandas",
        x=sorted_results["operation"].to_list(),
        y=sorted_results["pandas_time"].to_list(),
        marker_color=COLORS["amber"],
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Bar(
        name="Polars",
        x=sorted_results["operation"].to_list(),
        y=sorted_results["polars_time"].to_list(),
        marker_color=COLORS["blue"],
    ),
    row=1,
    col=2,
)

# 3. Speedup distribution
fig.add_trace(
    go.Histogram(
        x=results_df["speedup"].to_list(),
        nbinsx=20,
        marker_color=COLORS["blue"],
        opacity=0.7,
    ),
    row=2,
    col=1,
)
fig.add_vline(x=1.0, line_dash="dash", line_color="red", row=2, col=1)

# 4. pandas vs Polars scatter
fig.add_trace(
    go.Scatter(
        x=results_df["pandas_time"].to_list(),
        y=results_df["polars_time"].to_list(),
        mode="markers",
        marker=dict(color=COLORS["blue"], size=10),
        text=results_df["operation"].to_list(),
        hovertemplate="%{text}<br>pandas: %{x:.4f}s<br>Polars: %{y:.4f}s<extra></extra>",
    ),
    row=2,
    col=2,
)
# Add diagonal (equal performance line)
max_time = max(results_df["pandas_time"].max(), results_df["polars_time"].max())
fig.add_trace(
    go.Scatter(
        x=[0, max_time],
        y=[0, max_time],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        showlegend=False,
    ),
    row=2,
    col=2,
)

# Update layout
fig.update_xaxes(title_text="Category", row=1, col=1)
fig.update_yaxes(title_text="Speedup (Polars/pandas)", row=1, col=1)

fig.update_xaxes(title_text="Operation", tickangle=45, row=1, col=2)
fig.update_yaxes(title_text="Time (s)", type="log", row=1, col=2)

fig.update_xaxes(title_text="Speedup", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)

fig.update_xaxes(title_text="pandas time (s)", row=2, col=2)
fig.update_yaxes(title_text="Polars time (s)", row=2, col=2)

fig.update_layout(
    title_text=f"pandas {PANDAS_VERSION} vs Polars {POLARS_VERSION} Benchmark (Scale: {ACTIVE_SCALE})",
    height=800,
    showlegend=True,
    barmode="group",
)

fig.show()

# %% [markdown]
# ## Save Results

# %%
# Save detailed results
csv_path = RESULTS_DIR / f"pandas_polars_{ACTIVE_SCALE.lower()}.csv"
results_df.write_csv(csv_path)
print(f"Results saved to: {csv_path}")

# Save summary
summary_df = pl.DataFrame(
    {
        "metric": [
            "pandas_version",
            "polars_version",
            "scale",
            "total_rows",
            "mean_speedup",
            "operations_tested",
            "polars_faster_count",
            "pandas_faster_count",
            "cow_enabled",
            "pyarrow_strings",
        ],
        "value": [
            PANDAS_VERSION,
            POLARS_VERSION,
            ACTIVE_SCALE,
            str(total_rows),
            f"{overall_speedup:.2f}",
            str(len(results_df)),
            str(operations_faster),
            str(operations_slower),
            str(COW_ENABLED),
            str(PYARROW_STRINGS),
        ],
    }
)
summary_path = RESULTS_DIR / f"pandas_polars_summary_{ACTIVE_SCALE.lower()}.csv"
summary_df.write_csv(summary_path)
print(f"Summary saved to: {summary_path}")

# %% [markdown]
# ## Key Takeaways

# %%
# Surface this run's category ranking dynamically; the category table and overall
# counts were already displayed in the Summary cell above, so this cell only names
# the top-2 / bottom-2 categories so the takeaways stay in sync with the table.
_ranked = category_summary.sort("mean_speedup", descending=True)
_top2 = _ranked.head(2)["category"].to_list()
_bot2 = _ranked.tail(2)["category"].to_list()
print(f"Scale: {ACTIVE_SCALE} ({total_rows:,} rows)")
print(f"Fastest-on-Polars categories at this scale: {', '.join(_top2)}")
print(f"Smallest-gap (or pandas-faster) categories at this scale: {', '.join(_bot2)}")

# %% [markdown]
# ### What the table above shows
#
# Each row is the mean Polars-over-pandas speedup for the operation category at
# the scale chosen for this run (`ACTIVE_SCALE`). The ordering *depends on
# scale* and the takeaways printed above name this run's top-2 / bottom-2
# categories so the prose stays in sync with the actual numbers:
#
# - At **S (10K rows)** fixed Python overhead compresses the gap but does not
#   erase it: Polars still leads every category on average and is faster on the
#   large majority of individual operations. The margins are widest on strings
#   and joins and narrowest on the simpler rolling/window/groupby/filter
#   operations, where a handful of individual ops dip below parity — those are
#   the only places pandas wins. The cell above prints this run's top-2 and
#   bottom-2 categories so the prose tracks the actual numbers.
# - At **L / XL (≥1M rows)** Polars' parallelization widens the gap on string,
#   groupby, join, and lazy operations; the narrow-margin categories at S pull
#   further ahead as parallelization amortizes the Python overhead.
#
# Running this notebook at `BENCHMARK_SCALE=S` and again at a larger scale makes
# the trend visible: mean speedup grows with row count as parallelization
# amortizes Python overhead.
#
# ### Decision framework
#
# | Data size | Choice | Reason |
# |-----------|--------|--------|
# | < 100K rows | Either library | pandas stays competitive; Polars adds learning curve |
# | 100K – 1M rows | Prefer Polars | Larger gap on join / groupby / string operations |
# | > 1M rows | Polars | Parallelization advantage is largest here |
# | Visualization | Convert to pandas | matplotlib / seaborn compatibility |
#
# ### Migration notes
#
# 1. **pandas 2.x → 3.0**: free speedup from Copy-on-Write + PyArrow strings.
# 2. **pandas 3.0 → Polars**: migrate production pipelines processing >100K rows
#    or any pipeline dominated by string / groupby / join operations.
# 3. **New projects**: start with Polars; convert to pandas at the visualization
#    boundary only.
#
# **Book Reference**: Section 2.4 (*Storing Data*) discusses DataFrame engine
# selection alongside on-disk format and database choices.

# %%
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
print(f"pandas {PANDAS_VERSION} vs Polars {POLARS_VERSION}, scale {ACTIVE_SCALE}")
print(f"Results: {csv_path}")
