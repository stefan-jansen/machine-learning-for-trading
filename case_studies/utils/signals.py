"""Signal conversion and weight construction for backtesting.

Converts ML model predictions (probabilities or scores) into trading signals
and portfolio weights. Used by Ch16-20 notebooks.

Signal conversion approaches:
1. Fixed threshold: signal when score exceeds cutoff
2. Rolling percentile: signal when score exceeds recent distribution quantile
3. Cross-sectional percentile: signal for top-N% of assets at each rebalance

Weight construction:
- Equal-weight top-K: rank and select, uniform allocation
- Score-weighted top-K: rank and select, weight proportional to score
- Inverse volatility: equal-weight placeholder (full impl in Ch17)
"""

from __future__ import annotations

from typing import Literal

import polars as pl

# ---------------------------------------------------------------------------
# Signal conversion
# ---------------------------------------------------------------------------


def fixed_threshold_signal(
    predictions: pl.DataFrame,
    threshold: float = 0.5,
    score_col: str = "y_score",
    signal_type: Literal["long_only", "long_short"] = "long_only",
) -> pl.DataFrame:
    """Convert predictions to signals using a fixed threshold.

    For classification predictions, the score is typically a probability [0, 1].
    For regression predictions, the score may need normalization first.

    Args:
        predictions: DataFrame with at least [timestamp, symbol, y_score]
        threshold: Score threshold for entry signal
        score_col: Column containing prediction scores
        signal_type: "long_only" (signal=1 when above threshold) or
                     "long_short" (signal=1 above, signal=-1 below mirror threshold)

    Returns:
        DataFrame with added 'signal' column (-1, 0, or 1)
    """
    if signal_type == "long_only":
        return predictions.with_columns(
            signal=pl.when(pl.col(score_col) > threshold).then(1).otherwise(0).cast(pl.Int8)
        )
    else:  # long_short
        lower_threshold = 1.0 - threshold
        return predictions.with_columns(
            signal=pl.when(pl.col(score_col) > threshold)
            .then(1)
            .when(pl.col(score_col) < lower_threshold)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )


def rolling_percentile_signal(
    predictions: pl.DataFrame,
    window: int = 63,
    percentile: float = 90.0,
    score_col: str = "y_score",
    time_col: str = "timestamp",
    asset_col: str = "symbol",
    signal_type: Literal["long_only", "long_short"] = "long_only",
) -> pl.DataFrame:
    """Convert predictions to signals using rolling percentile threshold.

    Computes a rolling percentile of recent scores per asset and generates
    entry signals when the current score exceeds this adaptive threshold.

    Args:
        predictions: DataFrame with at least [timestamp, symbol, y_score]
        window: Rolling window size (e.g., 63 for ~3 months of daily data)
        percentile: Percentile threshold (e.g., 90 for top 10%)
        score_col: Column containing prediction scores
        time_col: Column containing timestamps
        asset_col: Column containing asset identifiers
        signal_type: "long_only" or "long_short"

    Returns:
        DataFrame with added 'signal' column and 'rolling_threshold' column
    """
    df = predictions.sort(time_col)

    df = df.with_columns(
        rolling_threshold=pl.col(score_col)
        .rolling_quantile(quantile=percentile / 100.0, window_size=window)
        .over(asset_col)
    )

    if signal_type == "long_only":
        df = df.with_columns(
            signal=pl.when(pl.col(score_col) > pl.col("rolling_threshold"))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
        )
    else:  # long_short
        lower_percentile = 100.0 - percentile
        df = df.with_columns(
            rolling_lower_threshold=pl.col(score_col)
            .rolling_quantile(quantile=lower_percentile / 100.0, window_size=window)
            .over(asset_col)
        )
        df = df.with_columns(
            signal=pl.when(pl.col(score_col) > pl.col("rolling_threshold"))
            .then(1)
            .when(pl.col(score_col) < pl.col("rolling_lower_threshold"))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )

    return df


def cross_sectional_percentile_signal(
    predictions: pl.DataFrame,
    percentile: float = 90.0,
    score_col: str = "y_score",
    time_col: str = "timestamp",
    signal_type: Literal["long_only", "long_short"] = "long_only",
) -> pl.DataFrame:
    """Convert predictions to signals using cross-sectional percentile.

    At each timestamp, selects assets in the top N% by score. Controls
    position count regardless of absolute score levels.

    Args:
        predictions: DataFrame with at least [timestamp, symbol, y_score]
        percentile: Percentile cutoff (e.g., 90 for top 10% of assets)
        score_col: Column containing prediction scores
        time_col: Column containing timestamps
        signal_type: "long_only" or "long_short"

    Returns:
        DataFrame with added 'signal' column and 'cs_threshold' column
    """
    df = predictions.with_columns(
        cs_threshold=pl.col(score_col).quantile(percentile / 100.0).over(time_col)
    )

    if signal_type == "long_only":
        df = df.with_columns(
            signal=pl.when(pl.col(score_col) >= pl.col("cs_threshold"))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
        )
    else:  # long_short
        lower_percentile = 100.0 - percentile
        df = df.with_columns(
            cs_lower_threshold=pl.col(score_col).quantile(lower_percentile / 100.0).over(time_col)
        )
        df = df.with_columns(
            signal=pl.when(pl.col(score_col) >= pl.col("cs_threshold"))
            .then(1)
            .when(pl.col(score_col) <= pl.col("cs_lower_threshold"))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )

    return df


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------


def build_target_weights(
    predictions: pl.DataFrame,
    method: Literal[
        "equal_weight_top_k",
        "score_weighted_top_k",
        "inverse_vol",
    ] = "equal_weight_top_k",
    top_k: int = 10,
    long_short: bool = False,
    score_col: str = "y_score",
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Convert predictions to portfolio target weights.

    Args:
        predictions: DataFrame with at least [timestamp, asset, y_score]
        method: Weight construction method
        top_k: Number of assets to select per rebalance
        long_short: If True, go long top_k and short bottom_k
        score_col: Column with prediction scores
        time_col: Timestamp column
        asset_col: Asset identifier column

    Returns:
        DataFrame with [timestamp, asset, weight] — weights sum to ~1.0 per timestamp
    """
    df = predictions.sort(time_col)

    # Rank assets cross-sectionally at each timestamp
    df = df.with_columns(
        cs_rank=pl.col(score_col).rank(method="ordinal", descending=True).over(time_col),
        n_assets=pl.col(score_col).count().over(time_col),
    )

    # Effective top_k (can't select more assets than available)
    df = df.with_columns(
        eff_k=pl.min_horizontal(pl.lit(top_k), pl.col("n_assets")),
    )

    if method == "equal_weight_top_k":
        if long_short:
            df = df.with_columns(
                weight=pl.when(pl.col("cs_rank") <= pl.col("eff_k"))
                .then(1.0 / pl.col("eff_k"))
                .when(pl.col("cs_rank") > pl.col("n_assets") - pl.col("eff_k"))
                .then(-1.0 / pl.col("eff_k"))
                .otherwise(0.0)
            )
        else:
            df = df.with_columns(
                weight=pl.when(pl.col("cs_rank") <= pl.col("eff_k"))
                .then(1.0 / pl.col("eff_k"))
                .otherwise(0.0)
            )

    elif method == "score_weighted_top_k":
        # When the top-K absolute-score sum is 0 at a timestamp (all top-K
        # predictions exactly zero), score-proportional weighting would
        # divide by zero. Fall back to equal-weight within the top-K for
        # those timestamps so the rebalance is well-defined.
        if long_short:
            top = df.filter(pl.col("cs_rank") <= pl.col("eff_k"))
            bottom = df.filter(pl.col("cs_rank") > pl.col("n_assets") - pl.col("eff_k"))

            top_denom = pl.col(score_col).abs().sum().over(time_col)
            top = top.with_columns(
                weight=pl.when(top_denom > 0)
                .then(pl.col(score_col).abs() / top_denom)
                .otherwise(1.0 / pl.col("eff_k"))
            )
            bottom_denom = pl.col(score_col).abs().sum().over(time_col)
            bottom = bottom.with_columns(
                weight=pl.when(bottom_denom > 0)
                .then(-pl.col(score_col).abs() / bottom_denom)
                .otherwise(-1.0 / pl.col("eff_k"))
            )
            mid = df.filter(
                (pl.col("cs_rank") > pl.col("eff_k"))
                & (pl.col("cs_rank") <= pl.col("n_assets") - pl.col("eff_k"))
            ).with_columns(weight=pl.lit(0.0))

            df = pl.concat([top, mid, bottom], how="diagonal_relaxed")
        else:
            top = df.filter(pl.col("cs_rank") <= pl.col("eff_k"))
            top_denom = pl.col(score_col).abs().sum().over(time_col)
            top = top.with_columns(
                weight=pl.when(top_denom > 0)
                .then(pl.col(score_col).abs() / top_denom)
                .otherwise(1.0 / pl.col("eff_k"))
            )
            rest = df.filter(pl.col("cs_rank") > pl.col("eff_k")).with_columns(weight=pl.lit(0.0))
            df = pl.concat([top, rest], how="diagonal_relaxed")

    elif method == "inverse_vol":
        # Placeholder — requires historical returns; full impl in Ch17
        df = df.with_columns(
            weight=pl.when(pl.col("cs_rank") <= pl.col("eff_k"))
            .then(1.0 / pl.col("eff_k"))
            .otherwise(0.0)
        )

    # Clean up helper columns
    result = df.select([time_col, asset_col, "weight"]).filter(pl.col("weight") != 0.0)
    return result.sort(time_col, asset_col)


# ---------------------------------------------------------------------------
# Config-driven dispatcher
# ---------------------------------------------------------------------------


def _signals_to_equal_weights(
    df: pl.DataFrame,
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Convert a signal column ({-1, 0, 1}) to equal weights within each group.

    Long signals get +1/N_long, short signals get -1/N_short, zero signals excluded.
    """
    # Count longs and shorts per timestamp
    df = df.with_columns(
        n_long=pl.col("signal").filter(pl.col("signal") > 0).count().over(time_col),
        n_short=pl.col("signal").filter(pl.col("signal") < 0).count().over(time_col),
    )

    df = df.with_columns(
        weight=pl.when(pl.col("signal") > 0)
        .then(1.0 / pl.col("n_long"))
        .when(pl.col("signal") < 0)
        .then(-1.0 / pl.col("n_short"))
        .otherwise(0.0)
    )

    return (
        df.select([time_col, asset_col, "weight"])
        .filter(pl.col("weight") != 0.0)
        .sort(time_col, asset_col)
    )


def per_symbol_rolling_percentile_signal(
    predictions: pl.DataFrame,
    long_q: float = 0.80,
    lookback_days: int = 20,
    bars_per_day: int = 390,
    score_col: str = "y_score",
    time_col: str = "timestamp",
    asset_col: str = "symbol",
    signal_type: Literal["long_only", "long_short"] = "long_only",
    stay_q: float | None = None,
) -> pl.DataFrame:
    """Per-symbol time-series rolling-percentile entry — ranks within own history.

    At the first bar of each session, computes the trailing rolling quantile
    of `y_score` over the past `lookback_days × bars_per_day` rows per symbol
    (causal: shifted by 1 before rolling). The day's threshold is held
    constant for that session's later bars via forward-fill within (symbol, date).

    A bar enters when `y_score` crosses its symbol-specific session threshold:
      long_only: signal=+1 if y_score > p_long
      long_short: signal=+1 if y_score > p_long, -1 if y_score < (1 - long_q)

    The short tail is the symmetric complement of `long_q` (so `long_q=0.85`
    sets `p_short` at the 0.15 quantile). The function asserts `long_q > 0.5`
    in `long_short` mode to keep `p_long > p_short`.

    Warm-up: `min_samples=W // 2` means roughly the first `lookback_days / 2`
    sessions per symbol have a null rolling quantile and therefore a null
    threshold; the signal is coerced to 0 for those bars (no entry).

    When ``stay_q`` is provided (must be < ``long_q``), a second rolling
    quantile is computed at that lower level using identical windowing and
    daily anchoring, exposed as a ``stay_thresh`` column on the output. The
    stay threshold is used by ``slot_strategy.build_persistent_slot_weights_hybrid``
    for signal-based slot exits; the entry signal column is unchanged.

    Mirrors the polars aggregator at
    `agents/.agents/work/nasdaq100_v3/scripts/sweep_daily_thresh_v2.py::add_daily_pct`,
    which is the canonical reference for the nasdaq100 v3 strategy.
    """
    if signal_type == "long_short" and long_q <= 0.5:
        msg = (
            f"per_symbol_rolling_percentile_signal long_short requires long_q > 0.5 "
            f"to keep p_long > p_short; got long_q={long_q}"
        )
        raise ValueError(msg)
    if stay_q is not None and stay_q >= long_q:
        msg = (
            f"per_symbol_rolling_percentile_signal stay_q must be < long_q so the "
            f"stay threshold sits below the entry threshold; got stay_q={stay_q}, "
            f"long_q={long_q}"
        )
        raise ValueError(msg)
    W = int(lookback_days * bars_per_day)
    df = predictions.sort([asset_col, time_col]).with_columns(
        _date=pl.col(time_col).dt.date(),
    )
    y_lag = pl.col(score_col).shift(1).over(asset_col)
    df = df.with_columns(
        _raw_p_long=y_lag.rolling_quantile(
            quantile=long_q,
            window_size=W,
            min_samples=W // 2,
        ).over(asset_col),
    )
    if signal_type == "long_short":
        df = df.with_columns(
            _raw_p_short=y_lag.rolling_quantile(
                quantile=1 - long_q,
                window_size=W,
                min_samples=W // 2,
            ).over(asset_col),
        )
    if stay_q is not None:
        df = df.with_columns(
            _raw_p_stay=y_lag.rolling_quantile(
                quantile=stay_q,
                window_size=W,
                min_samples=W // 2,
            ).over(asset_col),
        )
    df = (
        df.with_columns(
            _is_first=(pl.col(time_col) == pl.col(time_col).min().over([asset_col, "_date"])),
        )
        .with_columns(
            _p_long_seed=pl.when(pl.col("_is_first")).then(pl.col("_raw_p_long")).otherwise(None),
        )
        .with_columns(
            p_long=pl.col("_p_long_seed").forward_fill().over([asset_col, "_date"]),
        )
    )
    if signal_type == "long_short":
        df = (
            df.with_columns(
                _p_short_seed=pl.when(pl.col("_is_first"))
                .then(pl.col("_raw_p_short"))
                .otherwise(None),
            )
            .with_columns(
                p_short=pl.col("_p_short_seed").forward_fill().over([asset_col, "_date"]),
            )
            .with_columns(
                signal=pl.when(
                    pl.col("p_long").is_not_null() & (pl.col(score_col) > pl.col("p_long"))
                )
                .then(pl.lit(1).cast(pl.Int8))
                .when(pl.col("p_short").is_not_null() & (pl.col(score_col) < pl.col("p_short")))
                .then(pl.lit(-1).cast(pl.Int8))
                .otherwise(pl.lit(0).cast(pl.Int8))
            )
        )
        drop_cols = [
            "_date",
            "_raw_p_long",
            "_raw_p_short",
            "_is_first",
            "_p_long_seed",
            "_p_short_seed",
            "p_long",
            "p_short",
        ]
    else:
        df = df.with_columns(
            signal=pl.when(pl.col("p_long").is_not_null() & (pl.col(score_col) > pl.col("p_long")))
            .then(pl.lit(1).cast(pl.Int8))
            .otherwise(pl.lit(0).cast(pl.Int8))
        )
        drop_cols = ["_date", "_raw_p_long", "_is_first", "_p_long_seed", "p_long"]
    if stay_q is not None:
        df = df.with_columns(
            _p_stay_seed=pl.when(pl.col("_is_first")).then(pl.col("_raw_p_stay")).otherwise(None),
        ).with_columns(
            stay_thresh=pl.col("_p_stay_seed").forward_fill().over([asset_col, "_date"]),
        )
        drop_cols = [*drop_cols, "_raw_p_stay", "_p_stay_seed"]
    return df.drop(drop_cols)


def _decile_long_short(
    predictions: pl.DataFrame,
    n_quantiles: int = 10,
    score_col: str = "y_score",
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Academic factor portfolio: long top decile, short bottom decile.

    Args:
        predictions: DataFrame with [timestamp, asset, y_score]
        n_quantiles: Number of quantile bins (10=decile, 5=quintile)
        score_col: Score column
        time_col: Timestamp column
        asset_col: Asset column

    Returns:
        DataFrame with [timestamp, asset, weight]
    """
    df = predictions.sort(time_col)

    # Cross-sectional quantile rank per timestamp
    df = df.with_columns(
        cs_rank=pl.col(score_col).rank(method="ordinal", descending=True).over(time_col),
        n_assets=pl.col(score_col).count().over(time_col),
    )

    # Determine top and bottom quantile thresholds
    df = df.with_columns(
        top_cutoff=(pl.col("n_assets") / n_quantiles).floor().cast(pl.Int64).clip(lower_bound=1),
    )

    # Top quantile = long, bottom quantile = short
    df = df.with_columns(
        signal=pl.when(pl.col("cs_rank") <= pl.col("top_cutoff"))
        .then(pl.lit(1).cast(pl.Int8))
        .when(pl.col("cs_rank") > pl.col("n_assets") - pl.col("top_cutoff"))
        .then(pl.lit(-1).cast(pl.Int8))
        .otherwise(pl.lit(0).cast(pl.Int8))
    )

    return _signals_to_equal_weights(df, time_col, asset_col)


def build_target_weights_from_config(
    predictions: pl.DataFrame,
    config: dict,
    score_col: str = "y_score",
    time_col: str = "timestamp",
    asset_col: str = "symbol",
) -> pl.DataFrame:
    """Config-dict dispatcher for signal conversion and weight construction.

    Dispatches to existing methods based on config["method"]:
    - "equal_weight_top_k": top_k assets, equal weight
    - "score_weighted_top_k": top_k assets, score-proportional weight
    - "cross_sectional_percentile": percentile-based selection, equal weight
    - "per_symbol_rolling_percentile": per-symbol time-series rolling quantile,
        daily-anchored; long_q + lookback_days + bars_per_day
    - "fixed_threshold": threshold-based selection, equal weight
    - "decile_long_short": top/bottom decile, equal weight (academic factor)
    - "quintile_long_short": top/bottom quintile, equal weight

    Config dict keys:
        method (str): One of the methods above
        top_k (int): For top-k methods
        long_short (bool): For top-k methods
        percentile (float): For cross-sectional percentile (e.g., 90.0)
        threshold (float): For fixed threshold
        n_quantiles (int): For decile/quintile methods (default 10)

    Returns:
        DataFrame with [timestamp, asset, weight]
    """
    method = config["method"]
    long_short = config.get("long_short", False)
    direction = str(config.get("direction", "long_only")).strip().lower()

    def _apply_direction(weights: pl.DataFrame) -> pl.DataFrame:
        if direction == "long_only":
            return weights
        if direction == "short_only":
            return weights.with_columns((-pl.col("weight")).alias("weight"))
        msg = f"Unknown signal direction: {direction}"
        raise ValueError(msg)

    if method in ("equal_weight_top_k", "score_weighted_top_k", "inverse_vol"):
        return _apply_direction(
            build_target_weights(
                predictions,
                method=method,
                top_k=config.get("top_k", 10),
                long_short=long_short,
                score_col=score_col,
                time_col=time_col,
                asset_col=asset_col,
            )
        )

    elif method == "cross_sectional_percentile":
        percentile = config.get("percentile", 90.0)
        signal_type = "long_short" if long_short else "long_only"
        df_with_signal = cross_sectional_percentile_signal(
            predictions,
            percentile=percentile,
            score_col=score_col,
            time_col=time_col,
            signal_type=signal_type,
        )
        return _apply_direction(_signals_to_equal_weights(df_with_signal, time_col, asset_col))

    elif method == "per_symbol_rolling_percentile":
        long_q = float(config.get("long_q", 0.80))
        lookback_days = int(config.get("lookback_days", 20))
        bars_per_day = int(config.get("bars_per_day", 390))
        signal_type = "long_short" if long_short else "long_only"
        df_with_signal = per_symbol_rolling_percentile_signal(
            predictions,
            long_q=long_q,
            lookback_days=lookback_days,
            bars_per_day=bars_per_day,
            score_col=score_col,
            time_col=time_col,
            asset_col=asset_col,
            signal_type=signal_type,
        )
        return _apply_direction(_signals_to_equal_weights(df_with_signal, time_col, asset_col))

    elif method == "fixed_threshold":
        threshold = config.get("threshold", 0.0)
        signal_type = "long_short" if long_short else "long_only"
        df_with_signal = fixed_threshold_signal(
            predictions,
            threshold=threshold,
            score_col=score_col,
            signal_type=signal_type,
        )
        return _apply_direction(_signals_to_equal_weights(df_with_signal, time_col, asset_col))

    elif method in ("decile_long_short", "quintile_long_short"):
        n_q = config.get("n_quantiles", 10 if method == "decile_long_short" else 5)
        return _apply_direction(
            _decile_long_short(
                predictions,
                n_quantiles=n_q,
                score_col=score_col,
                time_col=time_col,
                asset_col=asset_col,
            )
        )

    else:
        msg = f"Unknown signal method: {method}"
        raise ValueError(msg)
