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

# %%
"""The ml4t Library Ecosystem - unified data, feature, and diagnostic libraries for Chapters 7-12."""

# %% [markdown]
# # The ml4t Library Ecosystem
#
# **Docker image**: `ml4t`
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.1 - Data Preprocessing and Encodings
#
# ## Purpose
#
# This notebook introduces the **ml4t library ecosystem** used throughout
# Chapters 7-12: data loaders (`ml4t-data`), feature computation
# (`ml4t-engineer`), and evaluation tools (`ml4t-diagnostic`).
#
# ## Learning Objectives
#
# 1. Load datasets using the unified `ml4t-data` loaders
# 2. Discover and compute features via the `ml4t-engineer` registry
# 3. Configure features with lists, dicts, or YAML for reproducibility
# 4. Validate library features against manual implementations
# 5. Preview feature evaluation with `ml4t-diagnostic` (full treatment in
#    `05_signal_evaluation`)
# 6. Understand when to use the library vs manual code
#
# ## Data Policy
#
# All examples use **real ETF data** (no synthetic data).
#
# ## Prerequisites
#
# - `01_data_quality_diagnostics` - establishes the ETF dataset shape used here.
# - Polars basics (`with_columns`, `over`, `lazy`).

# %%
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import polars as pl
from IPython.display import display
from ml4t.diagnostic.signal import analyze_signal
from ml4t.engineer import compute_features
from ml4t.engineer.core.registry import get_registry

from data import load_etfs

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
SPY_START_DATE = "2015-01-01"

# %% [markdown]
# ## 1. ml4t-data: Unified Data Loaders
#
# The `data` module provides consistent interfaces for all seven
# datasets introduced in Chapter 2. Each loader returns a Polars DataFrame
# with standardized column names.

# %%
etfs = load_etfs()
print(f"ETF universe: {len(etfs):,} rows, {etfs['symbol'].n_unique()} symbols")
print(f"Columns: {etfs.columns}")
print(f"Date range: {etfs['timestamp'].min()} to {etfs['timestamp'].max()}")

# %%
# For demos: SPY only
spy = etfs.filter(
    (pl.col("symbol") == "SPY")
    & (pl.col("timestamp") >= datetime.fromisoformat(SPY_START_DATE).date())
).sort("timestamp")

print(f"SPY: {len(spy):,} rows from {spy['timestamp'].min()} to {spy['timestamp'].max()}")

# %% [markdown]
# ## 2. ml4t-engineer: Feature Registry
#
# The `ml4t-engineer` library provides 120+ pre-built features with
# consistent naming, validation against reference implementations (TA-Lib
# where applicable), and self-documenting metadata.

# %%
registry = get_registry()
all_features = registry.list_all()

# Features by category
categories = {}
for name in all_features:
    metadata = registry.get(name)
    categories.setdefault(metadata.category, []).append(name)

print(f"Total features available: {len(all_features)}\n")
print("Features by category:")
for cat, feats in sorted(categories.items()):
    examples = ", ".join(feats[:4])
    suffix = ", ..." if len(feats) > 4 else ""
    print(f"  {cat:20s}: {len(feats):3d}  ({examples}{suffix})")

# %% [markdown]
# ### 2.1 Feature Metadata
#
# Each registry entry carries its default parameters, input requirements, and
# a description, plus a closed-form formula where a standard one exists (about
# a quarter of the catalog: momentum and price transforms tend to have one,
# while estimator-based volatility and microstructure features do not). This
# makes features discoverable without reading source code.

# %%
for feature_name in ["rsi", "atr", "garman_klass_volatility"]:
    meta = registry.get(feature_name)
    print(f"\n{'=' * 50}")
    print(f"Feature:     {meta.name}")
    print(f"Category:    {meta.category}")
    print(f"Description: {meta.description}")
    print(f"Formula:     {meta.formula or '(not recorded)'}")
    print(f"Parameters:  {meta.parameters}")
    print(f"Input type:  {meta.input_type}")

# %% [markdown]
# ## 3. Config-Driven Feature Computation
#
# `compute_features()` accepts three input formats, from simplest to most
# reproducible:
#
# 1. **List of names** - default parameters
# 2. **List of dicts** - custom parameters per feature
# 3. **YAML config** - stored configuration for pipelines

# %% [markdown]
# ### 3.1 Simple Feature List

# %%
result = compute_features(spy, ["rsi", "sma", "ema", "atr"])

new_cols = [c for c in result.columns if c not in spy.columns]
print(f"Computed {len(new_cols)} features: {new_cols}")
display(result.select(["timestamp", "close"] + new_cols).tail(5))

# %% [markdown]
# ### 3.2 Parameterized Features
#
# A list of dicts sets explicit parameters per feature. Each feature name
# resolves to one output column per call, so a single call holds one parameter
# set per feature. To compute several horizons of the *same* indicator (a
# common multi-timeframe setup), call `compute_features` once per parameter set
# and suffix the columns before joining.

# %%
# Distinct features, each with an explicit parameter set. Bollinger bands take the
# two deviations separately, following TA-Lib, so the upper and lower band need not
# sit the same distance from the moving average:
parameterized = [
    {"name": "rsi", "params": {"period": 10}},
    {"name": "atr", "params": {"period": 20}},
    {"name": "bollinger_bands", "params": {"period": 20, "nbdevup": 2.0, "nbdevdn": 2.0}},
]

result = compute_features(spy, parameterized)

new_cols = [c for c in result.columns if c not in spy.columns]
print(f"Parameterized features ({len(new_cols)} columns): {', '.join(new_cols)}")

# %%
# Multiple horizons of one indicator: one call per period, suffix, then join.
rsi_multi = spy.select(["timestamp"])
for period in (10, 21, 63):
    horizon = compute_features(spy, [{"name": "rsi", "params": {"period": period}}])
    rsi_multi = rsi_multi.join(
        horizon.select(["timestamp", pl.col("rsi").alias(f"rsi_{period}")]),
        on="timestamp",
    )

rsi_cols = [c for c in rsi_multi.columns if c != "timestamp"]
print(f"Multi-horizon RSI columns: {rsi_cols}")
display(rsi_multi.tail(5))

# %% [markdown]
# ### 3.3 YAML Configuration (Production)
#
# For reproducibility across notebooks and case studies, store feature
# configurations in YAML:
#
# ```yaml
# features:
#   - name: rsi
#     params:
#       period: 14
#
#   - name: macd
#     params:
#       fast: 12
#       slow: 26
#       signal: 9
#
#   - name: atr
#     params:
#       period: 14
# ```
#
# Load with: `compute_features(df, config_path="features.yaml")`

# %% [markdown]
# ## 4. Validation: Library vs Manual
#
# The library uses Wilder's smoothing (matching TA-Lib) while a naive
# implementation might use EWM span. Let's compare to understand the
# difference.


# %%
def manual_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Manual RSI using EWM span (not Wilder's smoothing)."""
    return (
        df.with_columns(pl.col("close").diff().alias("delta"))
        .with_columns(
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"),
        )
        .with_columns(
            pl.col("gain").ewm_mean(span=period, adjust=False).alias("avg_gain"),
            pl.col("loss").ewm_mean(span=period, adjust=False).alias("avg_loss"),
        )
        .with_columns(
            (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("rsi_manual"),
        )
    )


manual_result = manual_rsi(spy.select(["timestamp", "close"]))
library_result = compute_features(spy, [{"name": "rsi", "params": {"period": 14}}])

comparison = manual_result.join(
    library_result.select(["timestamp", "rsi"]),
    on="timestamp",
    how="inner",
).select(["timestamp", "rsi_manual", "rsi"])

print("RSI comparison (manual EWM vs library Wilder's):")
display(comparison.filter(pl.col("rsi").is_not_null()).tail(10))

# Both columns can carry a leading NaN/null from the warm-up window; in Polars
# NaN is distinct from null, so guard against both before averaging.
diff = comparison.filter(
    pl.col("rsi").is_not_null()
    & pl.col("rsi").is_not_nan()
    & pl.col("rsi_manual").is_not_null()
    & pl.col("rsi_manual").is_not_nan()
).with_columns((pl.col("rsi") - pl.col("rsi_manual")).abs().alias("abs_diff"))
print(f"Mean absolute difference: {diff['abs_diff'].mean():.4f}")
print("Differences are due to smoothing method (EWM span vs Wilder's).")

# %% [markdown]
# ### When to use the library vs manual code
#
# | Use ml4t-engineer | Implement manually |
# |---|---|
# | Standard indicators (RSI, MACD, ATR) | Custom alpha factors |
# | Production pipelines | Pedagogical demonstrations |
# | Cross-validation with TA-Lib | Non-standard variations |

# %% [markdown]
# ## 5. ml4t-diagnostic: Feature Evaluation (Preview)
#
# The third library, `ml4t-diagnostic`, closes the loop: once a feature is
# computed, `analyze_signal()` measures whether it predicts forward returns in
# the cross-section, reporting the Information Coefficient (rank correlation of
# factor to forward return), its t-statistic, and quantile spreads. Here we run
# a single call to show the `ml4t-engineer` to `ml4t-diagnostic` handoff on the
# full ETF panel. Notebook `05_signal_evaluation` develops the full workflow
# (ICIR, quantile monotonicity, turnover, half-life).

# %%
# Compute one momentum factor across the whole ETF cross-section, then evaluate.
panel = etfs.sort(["symbol", "timestamp"])
factor = (
    panel.group_by("symbol", maintain_order=True)
    .map_groups(lambda g: compute_features(g, [{"name": "mom", "params": {"period": 21}}]))
    .select(["timestamp", "symbol", pl.col("mom").alias("factor")])
    .drop_nulls()
)
prices = panel.select(["timestamp", "symbol", pl.col("close").alias("price")])

signal = analyze_signal(
    factor,
    prices,
    periods=(1, 5, 21),
    quantiles=5,
    date_col="timestamp",
    asset_col="symbol",
    price_col="price",
    factor_col="factor",
)

print(f"Evaluated on {signal.n_assets} assets over {signal.n_dates:,} dates")
for horizon in ("1D", "5D", "21D"):
    print(
        f"  {horizon:>3s}  IC={signal.ic[horizon]:+.4f}  "
        f"t-stat={signal.ic_t_stat[horizon]:+.2f}  "
        f"quantile spread={signal.spread[horizon]:+.4f}"
    )

# %% [markdown]
# The 21-day momentum factor carries near-zero cross-sectional IC on this ETF
# universe, and the t-statistics do not clear conventional significance. A
# single raw indicator rarely predicts returns on its own; the feature
# engineering and model-based combination methods in Chapters 8 through 12 are
# what turn raw features into usable signals. The point here is the interface:
# `ml4t-engineer` produces the factor and `ml4t-diagnostic` scores it, with no
# glue code in between.

# %% [markdown]
# ## 6. Key Polars Patterns for Feature Engineering
#
# These three patterns appear in 90% of feature engineering code.
# The `09_pandas_polars_benchmark` notebook provides full performance
# comparisons.

# %% [markdown]
# ### 6.1 GroupBy + Rolling via `.over()`
#
# Polars' `.over()` expression is the window function syntax - parallel
# and significantly faster than pandas' `groupby().transform()`.

# %%
multi_symbol = etfs.filter(pl.col("symbol").is_in(["SPY", "QQQ", "IWM"])).sort(
    ["symbol", "timestamp"]
)

features = multi_symbol.with_columns(
    pl.col("close").pct_change().over("symbol").alias("ret_1d"),
    (pl.col("close").pct_change().rolling_std(21).over("symbol") * np.sqrt(252)).alias("vol_21d"),
    pl.col("close").rolling_mean(21).over("symbol").alias("sma_21"),
)

print("Window function features:")
display(features.select(["symbol", "timestamp", "close", "ret_1d", "vol_21d"]).tail(10))

# %% [markdown]
# **Key pattern**: All transformations in ONE `with_columns()` call
# for parallel execution - never chain separate calls.

# %% [markdown]
# ### 6.2 ASOF Joins (Point-in-Time Matching)
#
# ASOF joins match by the closest timestamp. Critical for:
# - Trade-quote matching
# - Fundamental data alignment (announcement date to trading date)
# - Macro data alignment (release date to trading date)

# %%
trades = spy.select(["timestamp", "close"]).rename({"close": "trade_price"}).head(1000)

quotes = (
    spy.select(["timestamp", "close"])
    .rename({"close": "quote_price"})
    .with_columns(pl.col("timestamp").cast(pl.Date))
    .head(1000)
)

matched = trades.sort("timestamp").join_asof(
    quotes.sort("timestamp"),
    on="timestamp",
    strategy="backward",
)

print("ASOF join result (most recent quote for each trade):")
display(matched.head(5))

# %% [markdown]
# **Requirements**: Both DataFrames sorted by join key; use
# `strategy="backward"` for point-in-time safety.

# %% [markdown]
# ### 6.3 Lazy Evaluation (Large File Processing)
#
# For large files, `scan_parquet()` pushes filters to the storage layer.
# Here we demonstrate this using a loader to first get the data, then
# showing the lazy API pattern with `LazyFrame`.

# %%
# Demonstrate lazy API pattern with in-memory data
# (In production, use pl.scan_parquet on the actual file)
spy_lazy = (
    load_etfs(symbols=["SPY"], start_date="2020-01-01")
    .lazy()
    .select(["timestamp", "close", "volume"])
    .with_columns(pl.col("close").pct_change().alias("returns"))
)

result = spy_lazy.collect()
print(f"Lazy query: {len(result):,} rows")
print(f"\nQuery plan:\n{spy_lazy.explain()}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **ml4t-data** provides unified loaders for all seven datasets
# 2. **ml4t-engineer** offers 120+ validated features via a registry API
# 3. **Config-driven** computation (list, dict, YAML) ensures reproducibility
# 4. **ml4t-diagnostic** scores features against forward returns via
#    `analyze_signal`; notebook `05_signal_evaluation` covers the full workflow
# 5. **Library for production**, manual code for teaching and custom factors
# 6. **Polars patterns** (`.over()`, ASOF joins, lazy scans) power the pipelines
#
# **Next**: Chapter 8 notebooks build features manually to explain the
# economics, then use the registry for case study pipelines.
