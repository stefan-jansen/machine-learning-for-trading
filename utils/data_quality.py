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

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
from ml4t.diagnostic.splitters.calendar import TradingCalendar

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def top_entities(
    data: pl.DataFrame | pl.LazyFrame,
    max_entities: int,
    entity_col: str = "symbol",
) -> list:
    """The ``max_entities`` entities with the most rows, ties broken by name.

    **This is the one rule for reducing a panel's entity axis**, and every reduction
    in the test and fixture path has to reach it, whether from a loader or from a
    modelling helper. Two callers reducing the same panel to the same size have to
    get the same universe or they are not measuring the same study: a symbol only
    one side chose carries null features on the other, which runs clean and answers
    wrongly.

    Measured on nasdaq100_microstructure's CI fixture before the rules were unified:
    ``02_labels`` and ``03_financial_features`` reduced through the loader to
    {AAPL, AMD, CMCSA, CSCO, SIRI} - a seeded random sample - while
    ``04_model_based_features`` took the five most-observed symbols,
    {AAPL, AMD, AMZN, FB, TSLA}. Three of the five symbols the labels and financial
    features covered therefore had no temporal features at all.

    Row counts tie readily on these panels - five of the twelve fixture symbols sit
    at exactly 136,140 bars - and a tie broken by frame order is not stable across
    runs or across callers, so the entity name is the secondary key.

    Production runs pass 0 and never reach this.
    """
    counts = (
        data.lazy()
        .group_by(entity_col)
        .len()
        .sort(["len", entity_col], descending=[True, False])
        .head(max_entities)
        .collect()
    )
    return counts[entity_col].to_list()


def apply_max_symbols(
    data: pl.DataFrame | pl.LazyFrame,
    max_symbols: int,
    symbol_col: str = "symbol",
) -> pl.DataFrame | pl.LazyFrame:
    """Limit data to the ``max_symbols`` most-observed symbols, for fast-path testing.

    The loader-side entry point to :func:`top_entities`; ``utils.modeling`` reaches
    the same rule from the modelling side. It used to be a seeded random sample of
    the sorted symbol list, which disagreed with every consumer that reduced by
    observation count and moved whenever the underlying symbol set changed.

    Returns data unchanged if max_symbols <= 0.
    """
    if max_symbols <= 0:
        return data

    selected = top_entities(data, max_symbols, symbol_col)
    # implode: is_in against a bare Series of the same dtype is deprecated in polars
    # as ambiguous, and membership in the value set is what is meant.
    return data.filter(pl.col(symbol_col).is_in(pl.Series(symbol_col, selected).implode()))


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


def absent_calendar_sessions(
    session_dates: Iterable[date],
    *,
    calendar: str,
    known_absent: Iterable[date] = (),
) -> list[date]:
    """Sessions the exchange held between the first and last of *session_dates*, that are not
    in *session_dates* and are not declared in *known_absent*.

    A panel is normally checked the other way round: each date it carries is asked whether the
    exchange was open, and the dates that fail are dropped as stray prints. That direction
    cannot see this one. A session the exchange held and the archive never printed leaves no
    row to test, so nothing raises, every query succeeds, and one day's rows are simply gone.
    `us_equities_panel`'s single missing session was found by two counts of an unrelated
    quantity differing by one, which is the only way a defect of this shape surfaces on its own.

    It matters because every rolling window downstream reads its input in order and treats
    consecutive elements as consecutive sessions. A variance recursion, a fractional-difference
    convolution and a rolling average all price the gap across a missing session as one day's
    move.

    *known_absent* is the declaration, not a suppression: a caller states the sessions it has
    established are missing upstream, and anything else is returned for the caller to refuse.

    A date counts as a session when it settles itself, which is the rule
    :meth:`TradingCalendar.get_sessions` applies and the same one a stray-print filter uses -
    so the two directions cannot disagree about what a session is. Running it over every
    calendar day of the span enumerates what the exchange held, rather than classifying only
    the dates the archive happens to carry.

    Args:
        session_dates: The panel's session index. Order and duplicates do not matter.
        calendar: Exchange calendar name, as ``config/setup.yaml``'s ``evaluation.calendar``
            gives it.
        known_absent: Sessions already established as missing upstream.

    Returns:
        The undeclared absent sessions, earliest first. Empty when the panel is complete.
    """
    present = {d.date() if isinstance(d, datetime) else d for d in session_dates}
    if not present:
        return []

    span = pd.date_range(str(min(present)), str(max(present)), freq="D", tz="UTC")
    settling = TradingCalendar(calendar).get_sessions(pd.DatetimeIndex(span))
    held = set(pd.DatetimeIndex(settling.to_numpy()).date) & set(span.date)
    declared = {d.date() if isinstance(d, datetime) else d for d in known_absent}
    return sorted(held - present - declared)


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
    allow_missing: bool = False,
) -> list[str]:
    """Check feature columns for infinities, absent values, and extreme values.

    A NaN counts as absent here, not as a number. Polars evaluates ``NaN > x`` as
    True, so a feature carrying the warm-up head every rolling window leaves would
    otherwise be reported as holding values above ``max_abs_value``: a 252-session
    warm-up over 30 products reported 7,560 extreme values for a Shannon entropy
    bounded well below ten. Reading it the other way round also matters - a column
    that is entirely NaN carries no value at all, and was previously reported as
    neither absent nor extreme.

    A column named in ``feature_cols`` that ``df`` does not carry raises. A check
    that cannot find what it is checking has not passed, and reporting success is
    the one thing it must not do: called with one frame's column names against a
    different frame, this skipped all 22 columns it was given and reported the
    panel clean. An empty ``feature_cols`` fails for the same reason. Where a
    caller genuinely holds a superset - a column list spanning several artifacts,
    checked one artifact at a time - pass ``allow_missing=True`` and the absent
    names are reported as a warning instead.

    Args:
        df: DataFrame containing feature columns
        feature_cols: List of feature column names to validate
        max_abs_value: Threshold for flagging extreme values
        allow_missing: Report columns absent from ``df`` as a warning rather than
            raising. The default refuses them.

    Returns list of warning/error strings.

    Raises:
        ValueError: If ``feature_cols`` is empty, or names a column ``df`` does
            not carry and ``allow_missing`` is False.
    """
    issues: list[str] = []

    if not feature_cols:
        raise ValueError(
            "validate_features was given no columns to check. A gate over nothing "
            "reports success without reading a value; pass the columns the frame "
            "carries, or do not call the gate."
        )

    missing = [col for col in feature_cols if col not in df.columns]
    if missing and not allow_missing:
        raise ValueError(
            f"validate_features cannot find {len(missing)} of the {len(feature_cols)} "
            f"columns it was asked to check: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. The frame carries "
            f"{len(df.columns)} columns. Pass the frame these names come from, or "
            f"allow_missing=True if the list deliberately spans several frames."
        )
    if missing:
        issues.append(
            f"WARNING: {len(missing)} of {len(feature_cols)} columns are not in the frame "
            f"and were not checked: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    inf_cols = []
    absent_cols = []
    extreme_cols = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        present = df[col].drop_nulls()
        is_float = present.dtype.is_float()
        if is_float:
            present = present.filter(present.is_not_nan())

        if present.len() == 0:
            absent_cols.append(col)
            continue

        if is_float:
            n_inf = present.filter(present.is_infinite()).len()
            if n_inf > 0:
                inf_cols.append((col, n_inf))
            # The three conditions are reported separately, so an infinity is not
            # also counted among the finite values that ran large.
            present = present.filter(present.is_finite())

        n_extreme = present.filter(present.abs() > max_abs_value).len()
        if n_extreme > 0:
            extreme_cols.append((col, n_extreme))

    if inf_cols:
        details = ", ".join(f"{c}({n})" for c, n in inf_cols[:10])
        issues.append(f"CRITICAL: {len(inf_cols)} features have infinite values: {details}")

    if absent_cols:
        issues.append(
            f"WARNING: {len(absent_cols)} features carry no value, null or NaN throughout: "
            f"{absent_cols[:10]}{'...' if len(absent_cols) > 10 else ''}"
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
    allow_missing_features: bool = False,
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
        allow_missing_features: Passed to ``validate_features``. The default
            refuses a feature column ``features_df`` does not carry, because the
            gate would otherwise skip it and still report the panel clean.

    Returns:
        Dict with 'issues' (list of strings), 'n_critical', 'n_warning'

    Raises:
        ValueError: If fail_on_critical=True and any CRITICAL issues found, or if
            ``feature_cols`` names a column ``features_df`` does not carry
    """
    all_issues: list[str] = []

    # 1. Price checks (if price columns present)
    if price_cols:
        all_issues.extend(validate_prices(features_df, price_cols, asset_col=asset_col))

    # 2. Label checks
    all_issues.extend(validate_labels(label_df, label_col, max_abs_return))

    # 3. Feature checks
    all_issues.extend(
        validate_features(
            features_df,
            feature_cols,
            max_abs_feature,
            allow_missing=allow_missing_features,
        )
    )

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
