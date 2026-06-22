"""Data quality and filtering utilities for data loading.

This module provides centralized functions for:
- Coverage summaries (rows, symbols, date range)
- OHLC invariant checks
- Null rate analysis
- Gap detection in time series
- Symbol subsetting for test-mode execution

Usage:
    >>> from utils.data_quality import describe_coverage, check_ohlc_invariants
    >>> coverage = describe_coverage(df, time_col="timestamp", asset_col="symbol")
    >>> invariants = check_ohlc_invariants(df)
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


def apply_max_symbols(
    data: pl.DataFrame | pl.LazyFrame,
    max_symbols: int,
    symbol_col: str = "symbol",
    seed: int = 42,
) -> pl.DataFrame | pl.LazyFrame:
    """Limit data to a random subset of symbols for fast-path testing.

    Selects a reproducible random sample of symbols using a fixed seed.
    Returns data unchanged if max_symbols <= 0 or >= total symbols.
    """
    if max_symbols <= 0:
        return data

    if isinstance(data, pl.LazyFrame):
        all_symbols = data.select(pl.col(symbol_col).unique()).collect()[symbol_col].to_list()
    else:
        all_symbols = data[symbol_col].unique().to_list()

    if max_symbols >= len(all_symbols):
        return data

    rng = random.Random(seed)
    selected = rng.sample(sorted(all_symbols), max_symbols)
    return data.filter(pl.col(symbol_col).is_in(selected))


def describe_coverage(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> dict:
    """Return coverage summary for a dataset.

    Args:
        df: DataFrame with time and asset columns
        time_col: Name of the timestamp/date column
        asset_col: Name of the asset identifier column

    Returns:
        Dictionary with rows, assets, time_min, time_max, unique_times
    """
    return {
        "rows": df.height,
        "assets": df[asset_col].n_unique() if asset_col in df.columns else 0,
        "time_min": df[time_col].min(),
        "time_max": df[time_col].max(),
        "unique_times": df[time_col].n_unique(),
    }


def print_coverage(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
    dataset_name: str = "Dataset",
) -> None:
    """Print formatted coverage summary."""
    cov = describe_coverage(df, time_col, asset_col)
    print(f"=== {dataset_name} Coverage ===")
    print(f"  Rows: {cov['rows']:,}")
    print(f"  Assets: {cov['assets']:,}")
    print(f"  Time range: {cov['time_min']} to {cov['time_max']}")
    print(f"  Unique times: {cov['unique_times']:,}")


def check_ohlc_invariants(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Check OHLC data quality invariants.

    Validates:
    - high >= low
    - high >= open
    - high >= close
    - low <= open
    - low <= close
    - volume >= 0 (if volume column exists)

    For each check, only rows where all relevant columns are non-null are
    considered. This prevents null comparisons from distorting percentages
    (important for TAQ data where trade columns may be null for no-trade bars).

    Args:
        df: DataFrame with OHLC columns
        open_col, high_col, low_col, close_col: Column names for OHLC
        volume_col: Column name for volume (optional)

    Returns:
        DataFrame with check names and valid_pct columns
    """
    results = []
    total_rows = df.height
    cols = set(df.columns)

    def _check_invariant(name: str, condition: pl.Expr, required_cols: list[str]) -> None:
        """Check an invariant on rows where all required columns are non-null."""
        # Filter to rows where all required columns are non-null
        not_null_filter = pl.all_horizontal([pl.col(c).is_not_null() for c in required_cols])
        applicable = df.filter(not_null_filter)
        n_applicable = applicable.height

        if n_applicable == 0:
            return  # Skip if no applicable rows

        valid_pct = applicable.select(condition.mean()).item() * 100
        results.append(
            {
                "check": name,
                "valid_pct": valid_pct,
                "applicable_rows": n_applicable,
                "total_rows": total_rows,
            }
        )

    # Define checks with their required columns
    if {high_col, low_col}.issubset(cols):
        _check_invariant(
            "high_gte_low",
            pl.col(high_col) >= pl.col(low_col),
            [high_col, low_col],
        )

    if {high_col, open_col}.issubset(cols):
        _check_invariant(
            "high_gte_open",
            pl.col(high_col) >= pl.col(open_col),
            [high_col, open_col],
        )

    if {high_col, close_col}.issubset(cols):
        _check_invariant(
            "high_gte_close",
            pl.col(high_col) >= pl.col(close_col),
            [high_col, close_col],
        )

    if {low_col, open_col}.issubset(cols):
        _check_invariant(
            "low_lte_open",
            pl.col(low_col) <= pl.col(open_col),
            [low_col, open_col],
        )

    if {low_col, close_col}.issubset(cols):
        _check_invariant(
            "low_lte_close",
            pl.col(low_col) <= pl.col(close_col),
            [low_col, close_col],
        )

    if volume_col in cols:
        _check_invariant(
            "volume_non_negative",
            pl.col(volume_col) >= 0,
            [volume_col],
        )

    if not results:
        return pl.DataFrame({"check": [], "valid_pct": [], "applicable_rows": [], "total_rows": []})

    return pl.DataFrame(results)


def print_ohlc_invariants(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    show_coverage: bool = False,
) -> None:
    """Print OHLC invariant check results.

    Args:
        show_coverage: If True, show how many rows each check applies to
    """
    result = check_ohlc_invariants(df, open_col, high_col, low_col, close_col, volume_col)
    print("=== OHLC Invariants ===")
    for row in result.iter_rows(named=True):
        status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
        coverage = ""
        if show_coverage and row["applicable_rows"] < row["total_rows"]:
            coverage = f" ({row['applicable_rows']:,}/{row['total_rows']:,} rows)"
        print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%{coverage}")


def null_rate(
    df: pl.DataFrame,
    cols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Calculate null rates for specified columns.

    Args:
        df: DataFrame to analyze
        cols: Columns to check (default: all columns)

    Returns:
        DataFrame with column names and null_pct
    """
    if cols is None:
        cols = df.columns
    else:
        cols = [c for c in cols if c in df.columns]

    if not cols:
        return pl.DataFrame({"column": [], "null_pct": []})

    rates = df.select([pl.col(c).is_null().mean().alias(c) for c in cols])

    return pl.DataFrame(
        {
            "column": list(rates.columns),
            "null_pct": [rates[col].item() * 100 for col in rates.columns],
        }
    )


def print_null_rates(
    df: pl.DataFrame,
    cols: Sequence[str] | None = None,
    threshold: float = 0.0,
) -> None:
    """Print null rates for columns exceeding threshold.

    Args:
        df: DataFrame to analyze
        cols: Columns to check (default: all columns)
        threshold: Only print columns with null_pct > threshold
    """
    result = null_rate(df, cols)
    result = result.filter(pl.col("null_pct") > threshold)
    print("=== Null Rates ===")
    if result.height == 0:
        print("  No nulls detected")
    else:
        for row in result.iter_rows(named=True):
            print(f"  {row['column']}: {row['null_pct']:.2f}%")


def gap_summary(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    group_col: str | None = "symbol",
    expected_delta: timedelta | None = None,
) -> pl.DataFrame:
    """Identify gaps in time series data.

    Args:
        df: DataFrame with time series data
        time_col: Name of timestamp column
        group_col: Column to group by (e.g., symbol). None for ungrouped.
        expected_delta: Expected time between rows (e.g., timedelta(hours=1))

    Returns:
        DataFrame with gap statistics per group (if grouped) or overall
    """
    df_sorted = df.sort([group_col, time_col] if group_col else [time_col])

    # Calculate time differences
    if group_col:
        df_gaps = df_sorted.with_columns(pl.col(time_col).diff().over(group_col).alias("time_diff"))
    else:
        df_gaps = df_sorted.with_columns(pl.col(time_col).diff().alias("time_diff"))

    # If expected_delta provided, filter to gaps exceeding it
    if expected_delta is not None:
        df_gaps = df_gaps.filter(
            (pl.col("time_diff") > expected_delta) | pl.col("time_diff").is_null()
        )

    # Aggregate
    if group_col:
        return (
            df_gaps.filter(pl.col("time_diff").is_not_null())
            .group_by(group_col)
            .agg(
                pl.len().alias("gap_count"),
                pl.col("time_diff").max().alias("max_gap"),
            )
            .sort(group_col)
        )
    else:
        gaps = df_gaps.filter(pl.col("time_diff").is_not_null())
        if gaps.height == 0:
            return pl.DataFrame({"gap_count": [0], "max_gap": [None]})
        return pl.DataFrame(
            {
                "gap_count": [gaps.height],
                "max_gap": [gaps["time_diff"].max()],
            }
        )


def per_asset_stats(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
    price_col: str = "close",
    volume_col: str | None = "volume",
) -> pl.DataFrame:
    """Calculate per-asset summary statistics.

    Args:
        df: DataFrame with time series data
        time_col: Timestamp column name
        asset_col: Asset identifier column name
        price_col: Price column for mean calculation
        volume_col: Volume column (optional)

    Returns:
        DataFrame with rows, start, end, avg_price per asset
    """
    aggs = [
        pl.len().alias("rows"),
        pl.col(time_col).min().alias("start"),
        pl.col(time_col).max().alias("end"),
        pl.col(price_col).mean().alias("avg_price"),
    ]

    if volume_col and volume_col in df.columns:
        aggs.append(pl.col(volume_col).mean().alias("avg_volume"))

    return df.group_by(asset_col).agg(aggs).sort(asset_col)


# ---------------------------------------------------------------------------
# Modeling pipeline quality gates
# ---------------------------------------------------------------------------


def validate_prices(
    df: pl.DataFrame,
    price_cols: Sequence[str] = ("open", "high", "low", "close"),
    asset_col: str = "symbol",
    time_col: str = "timestamp",
) -> list[str]:
    """Check price columns for negative values, infinities, and NaN.

    Returns a list of warning/error strings. Empty list = all clean.
    """
    issues: list[str] = []
    cols_present = [c for c in price_cols if c in df.columns]

    for col in cols_present:
        n_neg = df.filter(pl.col(col) < 0).height
        n_inf = df.filter(pl.col(col).is_infinite()).height
        n_nan = df.filter(pl.col(col).is_nan()).height

        if n_neg > 0:
            # Show which assets have negative prices
            neg_assets = df.filter(pl.col(col) < 0).select(asset_col).unique().to_series().to_list()
            issues.append(
                f"CRITICAL: {col} has {n_neg} negative values "
                f"(assets: {neg_assets[:5]}{'...' if len(neg_assets) > 5 else ''})"
            )
        if n_inf > 0:
            issues.append(f"CRITICAL: {col} has {n_inf} infinite values")
        if n_nan > 0:
            issues.append(f"WARNING: {col} has {n_nan} NaN values")

    return issues


def validate_labels(
    df: pl.DataFrame,
    label_col: str,
    max_abs_return: float = 0.5,
) -> list[str]:
    """Check forward return labels for data quality issues.

    Args:
        df: DataFrame containing the label column
        label_col: Name of the forward return column
        max_abs_return: Maximum plausible absolute return (e.g., 0.5 = 50%)

    Returns list of warning/error strings.
    """
    issues: list[str] = []
    vals = df[label_col].drop_nulls()

    n_inf = vals.filter(vals.is_infinite()).len()
    n_nan = vals.filter(vals.is_nan()).len()
    n_extreme = vals.filter(vals.abs() > max_abs_return).len()
    n_total = vals.len()

    if n_inf > 0:
        issues.append(f"CRITICAL: {label_col} has {n_inf} infinite values")
    if n_nan > 0:
        issues.append(f"CRITICAL: {label_col} has {n_nan} NaN values")
    if n_extreme > 0:
        pct = n_extreme / n_total * 100
        issues.append(
            f"WARNING: {label_col} has {n_extreme} values with |ret| > {max_abs_return:.0%} "
            f"({pct:.2f}% of {n_total:,} rows)"
        )

    return issues


def validate_features(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    max_abs_value: float = 1e6,
) -> list[str]:
    """Check feature columns for infinities, all-null, and extreme values.

    Args:
        df: DataFrame containing feature columns
        feature_cols: List of feature column names to validate
        max_abs_value: Threshold for flagging extreme values

    Returns list of warning/error strings.
    """
    issues: list[str] = []
    n_rows = df.height

    inf_cols = []
    null_cols = []
    extreme_cols = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        series = df[col]
        n_null = series.null_count()
        non_null = series.drop_nulls()

        if n_null == n_rows:
            null_cols.append(col)
            continue

        if non_null.len() > 0:
            n_inf = non_null.filter(non_null.is_infinite()).len()
            if n_inf > 0:
                inf_cols.append((col, n_inf))

            n_extreme = non_null.filter(non_null.abs() > max_abs_value).len()
            if n_extreme > 0:
                extreme_cols.append((col, n_extreme))

    if inf_cols:
        details = ", ".join(f"{c}({n})" for c, n in inf_cols[:10])
        issues.append(f"CRITICAL: {len(inf_cols)} features have infinite values: {details}")

    if null_cols:
        issues.append(
            f"WARNING: {len(null_cols)} features are entirely null: "
            f"{null_cols[:10]}{'...' if len(null_cols) > 10 else ''}"
        )

    if extreme_cols:
        details = ", ".join(f"{c}({n})" for c, n in extreme_cols[:10])
        issues.append(
            f"WARNING: {len(extreme_cols)} features have values |x| > {max_abs_value:.0e}: {details}"
        )

    return issues


def validate_modeling_inputs(
    features_df: pl.DataFrame,
    label_df: pl.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    join_cols: Sequence[str] = ("timestamp", "symbol"),
    price_cols: Sequence[str] = (),
    asset_col: str = "symbol",
    max_abs_return: float = 0.5,
    max_abs_feature: float = 1e6,
    fail_on_critical: bool = True,
) -> dict:
    """Run all data quality checks before modeling.

    This is the gate between data preparation (labels + features) and
    model training. Call this at the start of evaluation notebooks.

    Args:
        features_df: Feature DataFrame
        label_df: Label DataFrame with forward returns
        feature_cols: Feature column names to validate
        label_col: Forward return column name
        join_cols: Columns used to join features and labels
        price_cols: Price columns to check (if present in features_df)
        asset_col: Asset identifier column name
        max_abs_return: Max plausible absolute return for labels
        max_abs_feature: Max plausible absolute feature value
        fail_on_critical: If True, raise ValueError on CRITICAL issues

    Returns:
        Dict with 'issues' (list of strings), 'n_critical', 'n_warning'

    Raises:
        ValueError: If fail_on_critical=True and any CRITICAL issues found
    """
    all_issues: list[str] = []

    # 1. Price checks (if price columns present)
    if price_cols:
        all_issues.extend(validate_prices(features_df, price_cols, asset_col=asset_col))

    # 2. Label checks
    all_issues.extend(validate_labels(label_df, label_col, max_abs_return))

    # 3. Feature checks
    all_issues.extend(validate_features(features_df, feature_cols, max_abs_feature))

    # Summarize
    n_critical = sum(1 for i in all_issues if i.startswith("CRITICAL"))
    n_warning = sum(1 for i in all_issues if i.startswith("WARNING"))

    # Print results
    if all_issues:
        print(f"Data Quality Gate: {n_critical} CRITICAL, {n_warning} WARNING")
        for issue in all_issues:
            marker = "[X]" if issue.startswith("CRITICAL") else "[!]"
            print(f"  {marker} {issue}")
    else:
        print("Data Quality Gate: ALL CLEAR")

    result = {
        "issues": all_issues,
        "n_critical": n_critical,
        "n_warning": n_warning,
    }

    if fail_on_critical and n_critical > 0:
        raise ValueError(
            f"Data quality gate FAILED: {n_critical} critical issues. "
            f"Fix upstream data before modeling."
        )

    return result
