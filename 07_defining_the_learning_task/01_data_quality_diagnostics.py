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
# # Data Quality Diagnostics
#
# **Docker image**: `ml4t`
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.1 - Data Preprocessing and Encodings
#
# ## Purpose
#
# This notebook provides a **standardized diagnostic survey** across all ML4T datasets.
# No cleaning is performed - just assessment. The output is a "health report" identifying
# which datasets need active preprocessing versus which are already in good shape.
#
# ## Learning Objectives
#
# 1. Build reusable diagnostic functions for financial panel data
# 2. Understand common data quality issues: missingness, duplicates, outliers, gaps
# 3. Identify dataset-specific quirks that affect ML pipeline design
# 4. Create a systematic framework for data validation
#
# ## Book Reference
#
# Section 7.1 emphasizes that **data quality determines model quality**. This notebook
# operationalizes that principle with concrete diagnostics before any feature engineering.
#
# ## Prerequisites
#
# - Familiarity with the seven ML4T datasets and their loaders (`data` module).
# - Polars basics (`pl.DataFrame`, `group_by`, `with_columns`).
# - No prior notebook output is required; this is the entry point of the pipeline.
#
# ## Data Contract
#
# - **Input**: Raw datasets from data loaders (ETFs, US Equities, Crypto, etc.)
# - **Output**: Diagnostic reports and quality assessments (no mutations)

# %%
"""Data Quality Diagnostics - Survey all ML4T datasets."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display

from data import (
    load_cme_futures,
    load_crypto_perps,
    load_crypto_premium,
    load_etfs,
    load_firm_characteristics,
    load_fx_pairs,
    load_us_equities,
)
from utils.style import COLORS  # importing utils.style activates the ml4t Plotly template

warnings.filterwarnings("ignore")
# Suppress chatty INFO logs from ml4t.diagnostic so output cells stay focused on results.
logging.disable(logging.INFO)

# %%
# %% tags=["parameters"]
# Production defaults
ETF_START_DATE = "2015-01-01"
US_EQUITIES_START_DATE = "1970-01-01"
CRYPTO_START_DATE = "2021-01-01T00:00:00+00:00"
CME_FUTURES_START_DATE = "2015-01-01"
FX_START_DATE = "2015-01-01T00:00:00+00:00"
FIRM_CHARACTERISTICS_START_DATE = "1990-01-01"


# %% [markdown]
# ## 1. Diagnostic Function Library
#
# These functions are designed to be **reusable** across datasets and chapters.
# They follow a consistent interface: input DataFrame, return diagnostic results.

# %% [markdown]
# ### 1.1 Index Integrity Check
#
# Validates that the time index has correct dtype, monotonicity, and uniqueness.
# For panel data, also checks that (date, symbol) pairs are unique.


# %%
def check_index_integrity(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    symbol_col: str | None = "symbol",
) -> dict[str, Any]:
    """Check index integrity: dtype, monotonicity, uniqueness.

    Args:
        df: Input DataFrame
        time_col: Name of the time column
        symbol_col: Name of the symbol column (None for single-asset data)

    Returns:
        Dictionary with diagnostic results
    """
    results = {
        "time_col": time_col,
        "symbol_col": symbol_col,
        "n_rows": len(df),
        "checks": {},
    }

    # Check time column dtype
    time_dtype = df[time_col].dtype
    results["time_dtype"] = str(time_dtype)
    results["checks"]["valid_time_dtype"] = time_dtype in [pl.Date, pl.Datetime]

    # Check for null time values
    null_times = df[time_col].is_null().sum()
    results["null_times"] = null_times
    results["checks"]["no_null_times"] = null_times == 0

    if symbol_col and symbol_col in df.columns:
        # Panel data: check uniqueness of (date, symbol)
        n_unique = df.select([time_col, symbol_col]).unique().height
        results["unique_pairs"] = n_unique
        results["checks"]["unique_date_symbol"] = n_unique == len(df)

        # Check monotonicity within each symbol, in the order the rows arrive.
        # Sorting by [symbol, time] before asking whether time is sorted makes the
        # check unfalsifiable - it would return True for any input, including a
        # frame whose rows are in the wrong order. `maintain_order=True` keeps each
        # group's rows in their original sequence so the comparison means something.
        mono_check = df.group_by(symbol_col, maintain_order=True).agg(
            is_mono=(pl.col(time_col) == pl.col(time_col).sort()).all()
        )
        all_mono = mono_check["is_mono"].all()
        results["checks"]["monotonic_per_symbol"] = all_mono

        # Symbol statistics
        n_symbols = df[symbol_col].n_unique()
        results["n_symbols"] = n_symbols
    else:
        # Single asset: check global monotonicity
        is_sorted = (df[time_col] == df[time_col].sort()).all()
        results["checks"]["monotonic"] = is_sorted

        # Check uniqueness
        n_unique = df[time_col].n_unique()
        results["unique_times"] = n_unique
        results["checks"]["unique_times"] = n_unique == len(df)

    # Date range
    results["date_min"] = str(df[time_col].min())
    results["date_max"] = str(df[time_col].max())

    # Overall pass/fail
    results["passed"] = all(results["checks"].values())

    return results


# %% [markdown]
# ### 1.2 Duplicate Detection
#
# Finds exact duplicates (identical rows) and near-duplicates (same index, different values).


# %%
def check_duplicates(
    df: pl.DataFrame,
    key_cols: list[str],
    value_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Detect exact and near-duplicates.

    Args:
        df: Input DataFrame
        key_cols: Columns that define uniqueness (e.g., ['timestamp', 'symbol'])
        value_cols: Columns to check for value differences in near-duplicates

    Returns:
        Dictionary with duplicate counts and examples
    """
    results = {"key_cols": key_cols, "n_rows": len(df)}

    # Exact duplicates (all columns identical)
    n_exact = len(df) - df.unique().height
    results["exact_duplicates"] = n_exact

    # Key duplicates (same key, potentially different values)
    key_counts = df.group_by(key_cols).len()
    dupes = key_counts.filter(pl.col("len") > 1)
    n_key_dupes = dupes["len"].sum() - len(dupes) if len(dupes) > 0 else 0
    results["key_duplicates"] = n_key_dupes

    # If there are key duplicates and value columns specified, check for near-dupes
    if n_key_dupes > 0 and value_cols:
        # Near-duplicates: same key, different values
        dupe_keys = dupes.select(key_cols)
        dupe_rows = df.join(dupe_keys, on=key_cols, how="inner")

        # Group by key and check if values vary
        near_dupes = dupe_rows.group_by(key_cols).agg(
            [pl.col(c).n_unique().alias(f"{c}_nunique") for c in value_cols]
        )
        # Count rows where any value column has more than 1 unique value
        vary_cols = [f"{c}_nunique" for c in value_cols]
        has_variation = near_dupes.select([(pl.col(c) > 1) for c in vary_cols]).select(
            pl.any_horizontal(pl.all())
        )

        n_near = has_variation.sum().item() if len(has_variation) > 0 else 0
        results["near_duplicates"] = n_near

        # Sample of duplicates
        if n_key_dupes > 0 and len(dupe_rows) > 0:
            results["sample_duplicates"] = dupe_rows.head(5).to_dicts()
    else:
        results["near_duplicates"] = 0

    results["passed"] = n_exact == 0 and n_key_dupes == 0

    return results


# %% [markdown]
# ### 1.3 Coverage Report
#
# Analyzes missingness patterns by field, by asset, and by time period.


# %%
def coverage_report(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    symbol_col: str | None = "symbol",
    value_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Generate coverage report: missingness by field, asset, time.

    Args:
        df: Input DataFrame
        time_col: Time column name
        symbol_col: Symbol column name (None for single-asset)
        value_cols: Columns to check for missingness (default: numeric columns)

    Returns:
        Dictionary with coverage statistics
    """
    results = {"n_rows": len(df), "n_cols": len(df.columns)}

    # Default to numeric columns if not specified
    if value_cols is None:
        value_cols = [
            c
            for c in df.columns
            if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            and c not in [time_col, symbol_col]
        ]

    results["checked_cols"] = value_cols

    # Missingness by column
    col_missing = {}
    for col in value_cols:
        n_null = df[col].is_null().sum()
        pct_null = 100.0 * n_null / len(df) if len(df) > 0 else 0
        col_missing[col] = {"n_null": n_null, "pct_null": round(pct_null, 2)}

    results["by_column"] = col_missing

    # Worst columns
    sorted_cols = sorted(col_missing.items(), key=lambda x: x[1]["pct_null"], reverse=True)
    results["worst_columns"] = sorted_cols[:5]

    # Missingness by asset (if panel)
    if symbol_col and symbol_col in df.columns:
        # For each symbol, count rows with any missing value
        symbol_missing = df.group_by(symbol_col).agg(
            [
                pl.len().alias("n_rows"),
                *[pl.col(c).is_null().sum().alias(f"{c}_null") for c in value_cols[:5]],
            ]
        )
        results["n_symbols"] = df[symbol_col].n_unique()

        # Find symbols with most missing data
        null_cols = [f"{c}_null" for c in value_cols[:5]]
        if null_cols:
            symbol_missing = symbol_missing.with_columns(
                total_null=pl.sum_horizontal(null_cols)
            ).sort("total_null", descending=True)
            results["worst_symbols"] = symbol_missing.head(5).to_dicts()

    # Missingness by time period
    time_missing = df.group_by(time_col).agg(
        [pl.col(c).is_null().sum().alias(f"{c}_null") for c in value_cols[:3]]
    )
    if value_cols:
        null_col = f"{value_cols[0]}_null"
        worst_dates = time_missing.sort(null_col, descending=True).head(5)
        results["worst_dates"] = worst_dates.to_dicts()

    # Overall coverage
    total_cells = len(df) * len(value_cols)
    total_missing = sum(v["n_null"] for v in col_missing.values())
    results["overall_coverage_pct"] = (
        round(100 * (1 - total_missing / total_cells), 2) if total_cells > 0 else 100
    )

    return results


# %% [markdown]
# ### 1.4 Coverage Heatmap
#
# Visualizes data availability across time × asset for panel data.


# %%
def coverage_heatmap(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    symbol_col: str = "symbol",
    value_col: str = "close",
    title: str = "Data Coverage Heatmap",
    max_symbols: int = 50,
    order_by: Literal["symbol", "first_observation"] = "first_observation",
) -> tuple[go.Figure, np.ndarray]:
    """Create time × asset coverage heatmap.

    Args:
        df: Input DataFrame
        time_col: Time column name
        symbol_col: Symbol column name
        value_col: Column to check for presence
        title: Plot title
        max_symbols: Maximum symbols to show (for readability)
        order_by: Column order along the x axis. Alphabetical order scatters entry
            dates at random, so a survivorship-free panel reads as a ragged skyline
            with no visible structure. Ordering by first observation is what turns
            the same data into the listing staircase the pattern actually is.

    Returns:
        The figure, and the presence matrix it draws (months x assets), so the
        notebook can quantify the pattern rather than assert it.
    """
    # Aggregate to reduce data size - use monthly periods
    coverage_df = (
        df.with_columns(period=pl.col(time_col).dt.truncate("1mo"))
        .group_by(["period", symbol_col])
        .agg(has_data=(pl.col(value_col).is_not_null().sum() > 0).cast(pl.Int8))
    )

    # Column order
    if order_by == "first_observation":
        symbols = (
            coverage_df.filter(pl.col("has_data") == 1)
            .group_by(symbol_col)
            .agg(first_seen=pl.col("period").min())
            .sort(["first_seen", symbol_col])[symbol_col]
            .to_list()
        )
    else:
        symbols = coverage_df[symbol_col].unique().sort().to_list()

    if len(symbols) > max_symbols:
        # Even sample across the ordering, so the thinned axis keeps its shape
        symbols = symbols[:: len(symbols) // max_symbols][:max_symbols]
        coverage_df = coverage_df.filter(pl.col(symbol_col).is_in(symbols))

    pivot_df = coverage_df.pivot(
        values="has_data",
        index="period",
        on=symbol_col,
    ).sort("period")

    # Reindex onto a continuous month range before anything reads the matrix.
    #
    # The pivot emits one row per month that HAS data, so a month in which the whole
    # panel is missing - a vendor outage, an ingestion gap - is simply absent from the
    # matrix rather than present as an all-zero row. Any count of "months missing for
    # every asset" taken off the un-reindexed matrix is therefore zero by
    # construction, which is the same unfalsifiable-check defect this notebook exists
    # to find, committed by the notebook itself.
    if len(pivot_df):
        full_months = pl.DataFrame(
            {
                "period": pl.datetime_range(
                    pivot_df["period"].min(),
                    pivot_df["period"].max(),
                    interval="1mo",
                    eager=True,
                ).cast(pivot_df["period"].dtype)
            }
        )
        pivot_df = full_months.join(pivot_df, on="period", how="left").fill_null(0).sort("period")

    # Pivot emits columns in order of first appearance, which is not `symbols`.
    # Select explicitly, or the x labels would name different assets than the
    # columns they sit under.
    symbols = [s for s in symbols if s in pivot_df.columns]
    periods = pivot_df["period"].to_list()
    period_labels = [str(p)[:7] for p in periods]  # YYYY-MM format
    matrix = pivot_df.select(symbols).to_numpy()

    # Create heatmap - present bars are filled (blue), missing periods stay white, so the
    # listing/delisting staircase reads as "where do we have data" rather than the reverse.
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=symbols,
            y=period_labels,
            colorscale=[[0, "white"], [1, COLORS["blue"]]],
            showscale=False,
            hovertemplate="Symbol: %{x}<br>Period: %{y}<br>Has data: %{z}<extra></extra>",
        )
    )

    # Cap the pixel height so a multi-decade monthly panel renders as a readable block
    # instead of an unusable vertical strip (588 months × 15px would be > 8000px tall).
    fig.update_layout(
        title=title,
        xaxis_title="Symbol",
        yaxis_title="Period",
        height=min(800, max(400, len(periods) * 15)),
        width=min(900, max(600, len(symbols) * 15)),
    )

    return fig, matrix


# %% [markdown]
# ### 1.5 Distribution Summary
#
# Computes summary statistics and flags extreme values.


# %%
def distribution_summary(
    df: pl.DataFrame,
    fields: list[str],
    extreme_threshold: float = 5.0,
) -> pl.DataFrame:
    """Compute distribution statistics with extreme value flags.

    Args:
        df: Input DataFrame
        fields: Columns to summarize
        extreme_threshold: Z-score threshold for flagging extremes

    Returns:
        DataFrame with summary statistics
    """
    stats_list = []

    for field in fields:
        if field not in df.columns:
            continue

        col = df[field].drop_nulls()
        if len(col) == 0:
            continue

        # Basic stats
        stats = {
            "field": field,
            "count": len(col),
            "null_pct": round(100 * df[field].is_null().sum() / len(df), 2),
            "mean": round(col.mean(), 6),
            "std": round(col.std(), 6),
            "min": round(col.min(), 6),
            "q25": round(col.quantile(0.25), 6),
            "median": round(col.median(), 6),
            "q75": round(col.quantile(0.75), 6),
            "max": round(col.max(), 6),
        }

        # Skewness and kurtosis (using numpy for robustness)
        arr = col.to_numpy()
        if len(arr) > 2 and np.std(arr) > 0:
            stats["skew"] = round(float(np.mean(((arr - np.mean(arr)) / np.std(arr)) ** 3)), 4)
            stats["kurtosis"] = round(
                float(np.mean(((arr - np.mean(arr)) / np.std(arr)) ** 4) - 3), 4
            )
        else:
            stats["skew"] = 0.0
            stats["kurtosis"] = 0.0

        # Extreme value count
        if stats["std"] > 0:
            z_scores = np.abs((arr - np.mean(arr)) / np.std(arr))
            n_extreme = int(np.sum(z_scores > extreme_threshold))
            stats["n_extreme"] = n_extreme
            stats["pct_extreme"] = round(100 * n_extreme / len(arr), 4)
        else:
            stats["n_extreme"] = 0
            stats["pct_extreme"] = 0.0

        stats_list.append(stats)

    return pl.DataFrame(stats_list)


# %% [markdown]
# ### 1.6 Outlier Flags
#
# Detects domain violations and spike anomalies (single-bar reversals).


# %%
def outlier_flags(
    df: pl.DataFrame,
    price_cols: list[str] | None = None,
    volume_col: str | None = "volume",
    return_col: str | None = None,
    max_return_threshold: float = 2.0,
) -> dict[str, Any]:
    """Flag domain violations and spike anomalies.

    Args:
        df: Input DataFrame
        price_cols: Price columns to check for domain violations
        volume_col: Volume column to check for negative values
        return_col: Return column to check for extreme spikes
        max_return_threshold: Maximum plausible return (e.g., 2.0 = 200%)

    Returns:
        Dictionary with outlier counts and examples
    """
    results = {"n_rows": len(df), "flags": {}}

    # Check for negative prices
    if price_cols:
        for col in price_cols:
            if col in df.columns:
                n_negative = (df[col] < 0).sum()
                n_zero = (df[col] == 0).sum()
                results["flags"][f"{col}_negative"] = n_negative
                results["flags"][f"{col}_zero"] = n_zero

    # Check for negative volume
    if volume_col and volume_col in df.columns:
        n_neg_vol = (df[volume_col] < 0).sum()
        results["flags"]["negative_volume"] = n_neg_vol

    # Check for impossible returns
    if return_col and return_col in df.columns:
        impossible = df[return_col].abs() > max_return_threshold
        n_impossible = impossible.sum()
        results["flags"]["impossible_returns"] = n_impossible

        if n_impossible > 0:
            examples = df.filter(impossible).head(5)
            results["impossible_return_examples"] = examples.to_dicts()

    # Spike detection: single-bar reversals
    # A spike is when price moves sharply then reverts (e.g., OHLC where H > 2*C or L < C/2).
    # Resolve the high/low/close names so roll-adjusted panels (adj_*) are covered too;
    # the spike ratio is adjust-invariant (all OHLC in a bar share one adjustment factor).
    def _resolve(*names: str) -> str | None:
        return next((n for n in names if n in df.columns), None)

    hi_col = _resolve("high", "adj_high")
    lo_col = _resolve("low", "adj_low")
    cl_col = _resolve("close", "adj_close")
    if hi_col and lo_col and cl_col:
        # Detect bars where high is >50% above close
        high_spike = df[hi_col] > df[cl_col] * 1.5
        low_spike = df[lo_col] < df[cl_col] * 0.5
        n_spikes = (high_spike | low_spike).sum()
        results["flags"]["price_spikes"] = n_spikes

    # Count total flags
    results["total_flags"] = sum(results["flags"].values())
    results["passed"] = results["total_flags"] == 0

    return results


# %% [markdown]
# ## 2. Dataset Registry
#
# Central registry mapping dataset names to loader functions and metadata.


# %%
def load_dataset_safely(loader_func, *args, **kwargs):
    """Attempt to load a dataset, returning None with message if unavailable."""
    try:
        return loader_func(*args, **kwargs)
    except Exception as e:
        error_type = type(e).__name__
        print(f"  WARNING: Could not load: {error_type}")
        return None


# Define dataset registry
DATASET_REGISTRY = {
    "etfs": {
        "loader": load_etfs,
        "time_col": "timestamp",
        "symbol_col": "symbol",
        "freq": "daily",
        "price_cols": ["open", "high", "low", "close"],
        "volume_col": "volume",
        "description": "100 ETFs from Yahoo Finance",
    },
    "us_equities": {
        "loader": load_us_equities,
        "time_col": "timestamp",
        "symbol_col": "symbol",
        "freq": "daily",
        "price_cols": ["open", "high", "low", "close", "adj_close"],
        "volume_col": "volume",
        "description": "3,199 US equities (1962-2018)",
    },
    "crypto_perps": {
        "loader": lambda: load_crypto_perps(frequency="8h"),
        "time_col": "timestamp",
        "symbol_col": "symbol",
        "freq": "8h",
        "price_cols": ["open", "high", "low", "close"],
        "volume_col": "volume",
        "description": "Crypto perpetuals from Binance",
    },
    "crypto_premium": {
        "loader": lambda: load_crypto_premium(frequency="8h"),
        "time_col": "timestamp",
        "symbol_col": "symbol",
        "freq": "8h",
        "price_cols": [
            "premium_index_open",
            "premium_index_high",
            "premium_index_low",
            "premium_index_close",
        ],
        "volume_col": None,
        "description": "Crypto premium index from Binance",
    },
    "cme_futures": {
        "loader": lambda: load_cme_futures(tenors=[0]),
        "time_col": "session_date",
        "symbol_col": "product",
        "freq": "daily",
        # Diagnostics run on the adjusted series (the canonical downstream series);
        # per-row OHLC share one cum_ratio, so integrity checks are adjust-invariant.
        "price_cols": ["adj_open", "adj_high", "adj_low", "adj_close"],
        "volume_col": "volume",
        "description": "CME futures (front month)",
    },
    "fx_pairs": {
        "loader": lambda: load_fx_pairs(frequency="4h"),
        "time_col": "timestamp",
        "symbol_col": "symbol",
        "freq": "4h",
        "price_cols": ["open", "high", "low", "close"],
        "volume_col": "volume",
        "description": "FX pairs from OANDA",
    },
    "firm_characteristics": {
        "loader": load_firm_characteristics,
        "time_col": "timestamp",
        "symbol_col": "symbol",
        "freq": "monthly",
        "price_cols": [],
        "volume_col": None,
        "description": "Chen-Pelger-Zhu firm characteristics",
    },
}

# %% [markdown]
# ## 3. Per-Dataset Diagnostics
#
# Run all diagnostics for each available dataset. Results are stored for the
# summary dashboard.

# %%
diagnostic_results = {}


# %%
def ensure_symbol_alias(df: pl.DataFrame) -> pl.DataFrame:
    """Expose the canonical asset identifier under the symbol name when needed."""
    if "asset" in df.columns and "symbol" not in df.columns:
        return df.with_columns(pl.col("asset").alias("symbol"))
    return df


# %%
def filter_from_start(df: pl.DataFrame, time_col: str, start_value: str) -> pl.DataFrame:
    """Apply a start-date filter without tripping over timezone/unit differences."""
    start_date = datetime.fromisoformat(start_value).date()
    dtype = df.schema[time_col]
    if dtype == pl.Date:
        return df.filter(pl.col(time_col) >= pl.lit(start_date).cast(pl.Date))
    return df.filter(pl.col(time_col).dt.date() >= pl.lit(start_date).cast(pl.Date))


# %% [markdown]
# ### 3.1 ETF Universe
#
# Yahoo Finance data covering 100 ETFs across 9 asset classes. Key concerns:
# adjustment artifacts from corporate actions and ticker changes.

# %%
etfs = load_dataset_safely(DATASET_REGISTRY["etfs"]["loader"])

if etfs is not None:
    config = DATASET_REGISTRY["etfs"]
    tcol = config["time_col"]
    etfs = ensure_symbol_alias(etfs)
    etfs = filter_from_start(etfs, tcol, ETF_START_DATE)

    print(f"Loaded {len(etfs):,} rows, {etfs['symbol'].n_unique()} symbols")
    print(f"Date range: {etfs[tcol].min()} to {etfs[tcol].max()}")

# %%
if etfs is not None:
    # Index integrity and duplicates
    idx_check = check_index_integrity(etfs, config["time_col"], config["symbol_col"])
    dupe_check = check_duplicates(
        etfs,
        key_cols=[config["time_col"], config["symbol_col"]],
        value_cols=config["price_cols"],
    )
    print(f"Index Integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")
    print(
        f"Duplicates:      {'PASSED' if dupe_check['passed'] else 'FAILED'} "
        f"(exact: {dupe_check['exact_duplicates']}, key: {dupe_check['key_duplicates']})"
    )

# %%
if etfs is not None:
    # Coverage and outlier flags
    cov_report = coverage_report(
        etfs,
        config["time_col"],
        config["symbol_col"],
        config["price_cols"] + [config["volume_col"]],
    )
    outliers = outlier_flags(
        etfs,
        price_cols=config["price_cols"],
        volume_col=config["volume_col"],
    )
    print(f"Coverage: {cov_report['overall_coverage_pct']}%")
    print(f"Outliers: {'PASSED' if outliers['passed'] else 'FLAGGED'}")
    for flag, count in outliers["flags"].items():
        if count > 0:
            print(f"  {flag}: {count}")

# %%
if etfs is not None:
    # Distribution summary
    dist_summary = distribution_summary(etfs, config["price_cols"][:3])
    display(dist_summary)

    diagnostic_results["etfs"] = {
        "index": idx_check,
        "duplicates": dupe_check,
        "coverage": cov_report,
        "outliers": outliers,
    }

# %% [markdown]
# ETFs pass index integrity and duplicate checks. Coverage is near-complete for
# OHLCV fields. Any outlier flags likely reflect Yahoo Finance adjustment
# artifacts (split ratios applied to historical bars) rather than genuine
# price anomalies.

# %% [markdown]
# ### 3.2 US Equities
#
# Longest dataset (56 years) with survivorship-free panel. Key concerns:
# penny stocks, stock splits, and delisting events.

# %%
us_equities = load_dataset_safely(DATASET_REGISTRY["us_equities"]["loader"])

if us_equities is not None:
    config = DATASET_REGISTRY["us_equities"]
    us_equities = ensure_symbol_alias(us_equities)
    us_equities = filter_from_start(us_equities, "timestamp", US_EQUITIES_START_DATE)

    print(f"Loaded {len(us_equities):,} rows, {us_equities['symbol'].n_unique()} symbols")
    print(f"Date range: {us_equities['timestamp'].min()} to {us_equities['timestamp'].max()}")

# %%
if us_equities is not None:
    # Index integrity
    idx_check = check_index_integrity(us_equities, config["time_col"], config["symbol_col"])
    print(f"Index Integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")

    # Duplicates and coverage (same battery as the other panels; this is the
    # largest dataset and the one that most needs the full check).
    dupe_check = check_duplicates(
        us_equities,
        key_cols=[config["time_col"], config["symbol_col"]],
        value_cols=config["price_cols"],
    )
    cov_report = coverage_report(
        us_equities,
        config["time_col"],
        config["symbol_col"],
        config["price_cols"] + [config["volume_col"]],
    )
    print(
        f"Duplicates:      {'PASSED' if dupe_check['passed'] else 'FAILED'} "
        f"(exact: {dupe_check['exact_duplicates']}, key: {dupe_check['key_duplicates']})"
    )
    print(f"Coverage: {cov_report['overall_coverage_pct']}%")

    # Penny stocks and extreme returns
    penny = us_equities.filter(pl.col("close") < 1)
    us_equities_ret = us_equities.sort(["symbol", "timestamp"]).with_columns(
        returns=pl.col("close").pct_change().over("symbol")
    )
    extreme_ret = us_equities_ret.filter(pl.col("returns").abs() > 1.0)
    print(
        f"Penny stocks (close < $1): {len(penny):,} rows ({100 * len(penny) / len(us_equities):.1f}%)"
    )
    print(f"Extreme returns (>100%):   {len(extreme_ret):,} rows")

# %% [markdown]
# ### A one-sided filter, and the column that shows it
#
# It is tempting to read "extreme returns" as "corporate actions" and move on. This
# dataset ships a `split_ratio` column, so we do not have to guess - and checking
# turns the reading around.
#
# A raw return is bounded below by $-1$: a price cannot fall by more than 100%. So a
# symmetric-looking filter on $|r| > 1$ **can only ever fire on the upside**. A
# reverse split, which multiplies the price, clears the threshold easily. A forward
# split, which divides it, produces a return near $-0.5$ and is invisible to the
# filter no matter how large the split.

# %%
if us_equities is not None:
    splits = us_equities_ret.filter(
        (pl.col("split_ratio") != 1) & pl.col("split_ratio").is_not_null()
    )
    forward = splits.filter(pl.col("split_ratio") > 1)
    reverse = splits.filter(pl.col("split_ratio") < 1)

    print(f"Rows flagged by |return| > 1.0:  {len(extreme_ret):,}")
    print(f"  smallest flagged return:      {extreme_ret['returns'].min():+.4f}")
    print(
        f"  that are a split day:         {splits.filter(pl.col('returns').abs() > 1.0).height:,}"
    )
    print(f"\nRows with split_ratio != 1:      {len(splits):,}")
    print(
        f"  forward splits (ratio > 1):   {len(forward):,}, "
        f"median same-day return {forward['returns'].median():+.4f}, "
        f"caught by the filter: {forward.filter(pl.col('returns').abs() > 1.0).height:,}"
    )
    print(
        f"  reverse splits (ratio < 1):   {len(reverse):,}, "
        f"median same-day return {reverse['returns'].median():+.4f}, "
        f"caught by the filter: {reverse.filter(pl.col('returns').abs() > 1.0).height:,}"
    )

# %% [markdown]
# The smallest flagged return is positive, which is the whole story in one number:
# not a single row was flagged for falling. The filter catches most reverse splits
# and essentially none of the forward splits, even though forward splits are the far
# more common event here by roughly eight to one.
#
# So "extreme returns signal corporate actions" is the wrong direction of inference.
# Most flagged rows are *not* splits, and the great majority of splits are *not*
# flagged. The lesson is the general one: **when a dataset records an event
# directly, detect it from that column rather than inferring it from a threshold on
# something else** - and check the bounds of the quantity you are thresholding before
# assuming a two-sided rule is two-sided.

# %%
if us_equities is not None:
    # Outlier flags
    outliers = outlier_flags(
        us_equities,
        price_cols=["open", "high", "low", "close"],
        volume_col="volume",
    )
    print(f"Outliers: {'PASSED' if outliers['passed'] else 'FLAGGED'}")
    for flag, count in outliers["flags"].items():
        if count > 0:
            print(f"  {flag}: {count}")

    diagnostic_results["us_equities"] = {
        "index": idx_check,
        "duplicates": dupe_check,
        "coverage": cov_report,
        "outliers": outliers,
        "penny_stocks": len(penny),
        "extreme_returns": len(extreme_ret),
    }

# %% [markdown]
# US Equities is the dataset that most needs active cleaning. Penny stocks
# ($<1) should be filtered to avoid microstructure noise dominating cross-sectional
# models. Extreme returns (>100% daily) typically reflect stock splits or
# data errors and require investigation. See `02_preprocessing_pipeline`
# for the full cleaning pipeline.

# %% [markdown]
# ### Coverage Heatmap: US Equities
#
# Visualize data availability across time and assets. Shaded cells mark months with
# data; white marks their absence. **Assets are ordered left to right by the month
# they first appear**, which is what makes the pattern legible: in alphabetical order
# the same panel is a ragged skyline with no visible structure, because a ticker's
# spelling says nothing about when it listed.

# %%
if us_equities is not None:
    fig, matrix_for_summary = coverage_heatmap(
        us_equities,
        time_col="timestamp",
        symbol_col="symbol",
        value_col="close",
        title="US equities is survivorship-free: assets list and delist mid-panel",
        max_symbols=50,
    )
    fig.update_layout(
        xaxis_title="Asset (sampled)",
        yaxis_title="Month",
    )
    fig.show()

# %% [markdown]
# Two things to read off it. The **staircase rising to the right** is the listing
# history: each column begins at the month its asset first traded, and because the
# columns are ordered by that month the start dates form a step rather than
# scattering. The **broken top edge** is delisting - a column that stops short of the
# present is an asset that left the panel, and it is still in the data. That is what
# survivorship-free means, and it is the reason a universe must be rebuilt as of each
# decision date rather than taken from today's index membership.
#
# What is *not* here is equally informative, but only one of the two absences is
# something this figure can report. There are **no full-width white rows**: the month
# axis is a continuous range built from the panel's own bounds, so a month in which
# nothing traded would be drawn as a blank row, and none is - no vendor outage or
# exchange closure is written into this panel.
#
# The columns cannot answer the matching question. The symbol axis is assembled from
# the assets that *appear in the data*, so an asset missing throughout would not be a
# white column - it would not be a column at all. "No asset is entirely absent" is
# therefore not a finding this figure can produce, and checking it needs an
# independently defined expected universe to compare against, which this notebook
# does not have. The cell below counts only what is countable here.
#
# Interior gaps, where an asset goes quiet and comes back, do exist but are rare.
# Missingness is overwhelmingly a question of when each asset entered and left, which
# makes the Section 7.1 strategy a universe-construction problem rather than an
# imputation one.

# %%
# Quantify what the heatmap shows, so the reading above is checkable
if us_equities is not None:
    _m = np.nan_to_num(matrix_for_summary.astype(float), nan=0.0)
    _interior = 0
    for _j in range(_m.shape[1]):
        _idx = np.flatnonzero(_m[:, _j] == 1)
        if len(_idx):
            _interior += int((_m[_idx[0] : _idx[-1] + 1, _j] == 0).sum())
    print(f"Sampled panel: {_m.shape[0]} months x {_m.shape[1]} assets")
    # The month axis is a continuous range, so this count can actually fire; the
    # symbol axis is the observed universe, so "an asset absent in every month" is
    # not expressible here and is deliberately not printed as a guaranteed zero.
    print(f"  Months missing for every asset:       {int((_m.sum(axis=1) == 0).sum())}")
    print(
        f"  Assets listing after the first month: {int(sum(1 for j in range(_m.shape[1]) if _m[0, j] == 0))}"
    )
    print(
        f"  Assets ending before the last month:  {int(sum(1 for j in range(_m.shape[1]) if _m[-1, j] == 0))}"
    )
    print(f"  Interior gaps (absent, then back):    {_interior} cells")

# %% [markdown]
# ### 3.3 Crypto Perpetuals
#
# 24/7 trading with 8-hour funding settlement cycles. Bars should
# align exactly to 00:00, 08:00, 16:00 UTC.

# %%
crypto_perps = load_dataset_safely(DATASET_REGISTRY["crypto_perps"]["loader"])

if crypto_perps is not None:
    config = DATASET_REGISTRY["crypto_perps"]
    crypto_perps = ensure_symbol_alias(crypto_perps)
    crypto_perps = filter_from_start(crypto_perps, "timestamp", CRYPTO_START_DATE)

    print(f"Loaded {len(crypto_perps):,} rows, {crypto_perps['symbol'].n_unique()} symbols")

    idx_check = check_index_integrity(crypto_perps, config["time_col"], config["symbol_col"])
    cov_report = coverage_report(
        crypto_perps,
        config["time_col"],
        config["symbol_col"],
        config["price_cols"],
    )
    hours = crypto_perps["timestamp"].dt.hour()
    aligned = hours.is_in([0, 8, 16]).all()

    print(f"Index Integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")
    print(f"Coverage: {cov_report['overall_coverage_pct']}%")
    print(f"8-hour alignment: {'PASSED' if aligned else 'FAILED'}")

    diagnostic_results["crypto_perps"] = {
        "index": idx_check,
        "coverage": cov_report,
        "aligned_8h": aligned,
    }

# %% [markdown]
# ### 3.4 Crypto Premium Index
#
# Premium index for funding arbitrage strategies. Same 8-hour frequency
# as perpetuals; values represent the basis between spot and futures.

# %%
crypto_premium = load_dataset_safely(DATASET_REGISTRY["crypto_premium"]["loader"])

if crypto_premium is not None:
    config = DATASET_REGISTRY["crypto_premium"]
    crypto_premium = ensure_symbol_alias(crypto_premium)
    crypto_premium = filter_from_start(crypto_premium, "timestamp", CRYPTO_START_DATE)

    print(f"Loaded {len(crypto_premium):,} rows, {crypto_premium['symbol'].n_unique()} symbols")

    idx_check = check_index_integrity(crypto_premium, config["time_col"], config["symbol_col"])
    dist = distribution_summary(crypto_premium, config["price_cols"])
    print(f"Index Integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")
    display(dist)

    diagnostic_results["crypto_premium"] = {"index": idx_check}

# %% [markdown]
# Both crypto datasets pass integrity checks. The 24/7 schedule means no
# weekend gaps to worry about, but delisting events mid-panel (coins removed
# from the exchange) can create sudden coverage drops.

# %% [markdown]
# ### 3.5 CME Futures
#
# Session-aligned daily bars with roll continuity across contract months.

# %%
cme_futures = load_dataset_safely(DATASET_REGISTRY["cme_futures"]["loader"])

if cme_futures is not None:
    config = DATASET_REGISTRY["cme_futures"]
    cme_futures = filter_from_start(cme_futures, "session_date", CME_FUTURES_START_DATE)

    products = cme_futures["product"].unique().sort().to_list()
    print(f"Loaded {len(cme_futures):,} rows, {len(products)} products")
    print(f"Products: {', '.join(products)}")

    idx_check = check_index_integrity(cme_futures, config["time_col"], config["symbol_col"])
    cov_by_product = (
        cme_futures.group_by("product")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("adj_close").is_null().sum().alias("n_null_close"),
        )
        .sort("product")
    )
    print(f"Index Integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")
    display(cov_by_product)

    diagnostic_results["cme_futures"] = {"index": idx_check, "n_products": len(products)}

# %% [markdown]
# ### 3.6 FX Pairs
#
# 4-hour bars with expected weekend gaps (market closes Friday 17:00 EST,
# reopens Sunday 17:00 EST).

# %%
fx_pairs = load_dataset_safely(DATASET_REGISTRY["fx_pairs"]["loader"])

if fx_pairs is not None:
    config = DATASET_REGISTRY["fx_pairs"]
    fx_pairs = ensure_symbol_alias(fx_pairs)
    fx_pairs = filter_from_start(fx_pairs, "timestamp", FX_START_DATE)

    print(f"Loaded {len(fx_pairs):,} rows, {fx_pairs['symbol'].n_unique()} pairs")

    idx_check = check_index_integrity(fx_pairs, config["time_col"], config["symbol_col"])
    # Polars dt.weekday() is ISO: Monday=1 ... Saturday=6, Sunday=7. The FX week
    # reopens Sunday 17:00 EST, which lands at 21:00-22:00 UTC across daylight
    # saving, so Sunday bars at or after 21:00 UTC are the legitimate weekly
    # reopen. Saturday bars and Sunday-daytime bars are the true anomalies.
    FX_REOPEN_HOUR_UTC = 21
    fx_pairs_with_weekday = fx_pairs.with_columns(
        weekday=pl.col("timestamp").dt.weekday(),
        hour=pl.col("timestamp").dt.hour(),
    )
    weekend_data = fx_pairs_with_weekday.filter(pl.col("weekday").is_in([6, 7]))
    reopen_bars = weekend_data.filter(
        (pl.col("weekday") == 7) & (pl.col("hour") >= FX_REOPEN_HOUR_UTC)
    )
    anomalous_weekend = weekend_data.filter(
        (pl.col("weekday") == 6) | (pl.col("hour") < FX_REOPEN_HOUR_UTC)
    )
    cov_report = coverage_report(
        fx_pairs,
        config["time_col"],
        config["symbol_col"],
        config["price_cols"],
    )

    print(f"Index Integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")
    print(
        f"Weekend bars: {len(weekend_data)} "
        f"({len(reopen_bars)} Sunday reopen, expected; "
        f"{len(anomalous_weekend)} anomalous)"
    )
    print(f"Coverage: {cov_report['overall_coverage_pct']}%")

    diagnostic_results["fx_pairs"] = {
        "index": idx_check,
        "coverage": cov_report,
        "weekend_data": len(weekend_data),
        "reopen_bars": len(reopen_bars),
        "anomalous_weekend": len(anomalous_weekend),
    }

# %% [markdown]
# FX trades continuously from the Sunday 17:00 EST reopen to the Friday close, so
# Sunday bars at or after 21:00 UTC are expected market data. Saturday bars and
# Sunday-daytime bars fall outside the trading week and are the genuine anomalies
# to filter.

# %% [markdown]
# ### 3.7 Firm Characteristics
#
# Chen-Pelger-Zhu academic monthly panel with 46 pre-computed characteristics
# (plus `ret`, `timestamp`, and a `split` indicator). Coverage is engineered
# upstream: rows that fail the source paper's data-availability rules are
# dropped before the panel is published, so the columns themselves arrive
# dense - but the *firm-month* footprint shrinks substantially relative to a
# raw CRSP–Compustat join.

# %%
firm_char = load_dataset_safely(DATASET_REGISTRY["firm_characteristics"]["loader"])

if firm_char is not None:
    config = DATASET_REGISTRY["firm_characteristics"]
    firm_char = filter_from_start(firm_char, "timestamp", FIRM_CHARACTERISTICS_START_DATE)

    n_dates = firm_char["timestamp"].n_unique()
    print(f"Loaded {len(firm_char):,} rows, {n_dates} months, {len(firm_char.columns)} columns")

    idx_check = check_index_integrity(firm_char, config["time_col"], config["symbol_col"])
    print(f"Index integrity: {'PASSED' if idx_check['passed'] else 'FAILED'}")
    print(f"  {idx_check['n_symbols']:,} firms, {idx_check['unique_pairs']:,} unique firm-months")

    # Sort columns alphabetically before slicing for reproducibility
    char_cols = sorted(c for c in firm_char.columns if c not in ["timestamp", "split", "ret"])[:10]
    cov_report = coverage_report(firm_char, config["time_col"], None, char_cols)
    print(f"Coverage (sample of 10 chars): {cov_report['overall_coverage_pct']}%")
    for col, stats in cov_report["worst_columns"][:5]:
        print(f"  {col}: {stats['pct_null']:.1f}% null")

    diagnostic_results["firm_characteristics"] = {
        "index": idx_check,
        "coverage": cov_report,
        "n_characteristics": len(char_cols),
    }

# %% [markdown]
# The 10 sampled characteristics show 100% column-level coverage because the
# upstream filter only retains firm-months with valid characteristic vectors.
# The cost of that filter is in the row count: many small-cap firm-months are
# excluded entirely, which is the right framing for §7.1's discussion of
# "missing due to observed coverage rules". Downstream notebooks should not
# attempt to reconstruct dropped rows; treat the panel as the authors' own
# coverage rule applied at source.

# %% [markdown]
# ## 4. Summary Dashboard
#
# Comparative view across all datasets.


# %%
def rag_status(
    passed: bool | None, warning_threshold: float | None = None, value: float | None = None
) -> str:
    """Convert pass/fail to RAG status."""
    if passed is None:
        return "N/A"
    if passed:
        if warning_threshold is not None and value is not None and value < warning_threshold:
            return "WARN"
        return "OK"
    return "FAIL"


# %%
summary_rows = []

for name, results in diagnostic_results.items():
    row = {"Dataset": name}

    if "index" in results:
        row["Index"] = rag_status(results["index"]["passed"])
    else:
        row["Index"] = "N/A"

    if "duplicates" in results:
        row["Duplicates"] = rag_status(results["duplicates"]["passed"])
    else:
        row["Duplicates"] = "N/A"

    if "coverage" in results:
        cov_pct = results["coverage"]["overall_coverage_pct"]
        passed = cov_pct >= 95
        row["Coverage"] = f"{rag_status(passed, 99, cov_pct)} ({cov_pct}%)"
    else:
        row["Coverage"] = "N/A"

    if "outliers" in results:
        row["Outliers"] = rag_status(results["outliers"]["passed"])
    else:
        row["Outliers"] = "N/A"

    summary_rows.append(row)

summary_df = pl.DataFrame(summary_rows)
summary_df

# %% [markdown]
# All datasets pass index integrity. Coverage is high for market data (ETFs,
# crypto, futures, FX). Firm characteristics show full coverage for the sampled
# columns, though sparsity increases for less common characteristics.
# US Equities has outlier flags (price spikes) that `02_preprocessing_pipeline`
# addresses with domain filters and spike detection.

# %% [markdown]
# ## 5. Normality Testing
#
# The **Jarque-Bera test** jointly tests whether skewness and kurtosis match
# a normal distribution. Financial returns almost always reject normality
# due to fat tails and asymmetric distributions.

# %%
from ml4t.diagnostic.evaluation.distribution.tests import jarque_bera_test

if "etfs" in diagnostic_results:
    etfs_returns = (
        etfs.sort(["symbol", "timestamp"])
        .with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
        .filter(pl.col("returns").is_not_null() & pl.col("returns").is_finite())
    )

    spy_returns = etfs_returns.filter(pl.col("symbol") == "SPY")["returns"].to_numpy()
    jb_result = jarque_bera_test(spy_returns)

    print("Jarque-Bera Normality Test (SPY Daily Returns)")
    print(f"  Statistic:       {jb_result.statistic:.1f}")
    print(f"  P-value:         {jb_result.p_value:.2e}")
    print(f"  Skewness:        {jb_result.skewness:.3f}")
    print(f"  Excess Kurtosis: {jb_result.excess_kurtosis:.3f}")
    print(f"  Normal:          {jb_result.is_normal}")

# %% [markdown]
# SPY daily returns strongly reject normality ($p \approx 0$), driven by
# excess kurtosis (fat tails). This confirms that robust scaling and
# winsorization - not standard z-scoring - are appropriate for financial
# returns. See `02_preprocessing_pipeline` for winsorization in practice.

# %% [markdown]
# ### 5.1 Stationarity Quick Check
#
# Prices are non-stationary (they trend); returns are typically stationary.
# This validates using returns, not prices, as ML features and labels.
# Stationarity testing is covered formally in Chapter 9.

# %%
from ml4t.diagnostic.evaluation.stationarity import analyze_stationarity

if "etfs" in diagnostic_results:
    spy_prices = etfs.filter(pl.col("symbol") == "SPY").sort("timestamp")["close"].to_numpy()

    price_stat = analyze_stationarity(spy_prices)
    return_stat = analyze_stationarity(spy_returns)

    print(f"SPY Prices  - Consensus: {price_stat.consensus}")
    print(f"SPY Returns - Consensus: {return_stat.consensus}")

# %% [markdown]
# Prices are non-stationary (unit root), returns are stationary - the
# expected result. This validates the standard practice of using returns
# (or differences) rather than levels as model inputs. Chapter 9 covers
# formal time series analysis including ADF, KPSS, and Phillips-Perron tests.

# %% [markdown]
# ## 6. Key Findings
#
# Based on the diagnostic survey, the datasets fall into two groups:
#
# **Ready for direct use** (no structural cleaning needed):
# - **ETFs** - complete OHLCV coverage; outlier flags reflect Yahoo adjustment artifacts.
# - **Crypto perpetuals and premium** - complete coverage, 8-hour-aligned timestamps.
# - **CME futures** - session-aligned bars across 30 products with no null closes.
# - **Firm Characteristics** - column-level coverage is complete by construction;
#   downstream models inherit the source paper's coverage rule.
#
# **Require active preprocessing** (see `02_preprocessing_pipeline`):
# 1. **US Equities** - penny-stock filter (1.4% of rows below $1), extreme-return
#    handling (837 rows above 100% daily moves, every one of them an *upward* move),
#    and 2,739 price-spike bars. Corporate actions should be taken from `split_ratio`
#    rather than inferred from return size; see the split analysis in §3.2.
# 2. **FX Pairs** - of 11,344 weekend timestamps, 11,339 are legitimate Sunday
#    reopen bars (at or after 21:00 UTC) that belong in the panel; only 5
#    Saturday or Sunday-daytime bars are anomalies to filter before the
#    spot-vs-forward analytics in §7.2.
#
# The next notebook applies these cleaning steps and demonstrates split-aware
# preprocessing to prevent information leakage.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Index integrity** is fundamental - time monotonicity and uniqueness must pass,
#    and the check has to be capable of failing: sorting a frame before asking whether
#    it is sorted returns True for every input
# 2. **Column coverage is not panel coverage** - a dataset can report complete columns
#    while whole rows are absent, because a null count only sees rows that exist
# 3. **Domain violations** (negative prices/volume) indicate data quality issues
# 4. **Detect events from the column that records them** - `split_ratio` identifies
#    splits directly, while a `|return| > 1` threshold is one-sided by construction and
#    misses nearly every forward split
# 5. **Calendar alignment** matters for FX (weekends) and crypto (24/7)
#
# **Next**: See `02_preprocessing_pipeline` for cleaning and split-aware preprocessing.
# **Book**: Section 7.1 discusses why data quality determines model quality.
