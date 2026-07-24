"""Financial-only feature pipeline for the ETFs deployment loop.

Mirrors the financial-feature subset of `case_studies/etfs/03_financial_features.py`
without the model-based path (HMM regimes, GARCH conditional volatility) and
without the QC/EDA cells. The source pipeline produces 57 columns; this module
intentionally drops HMM regimes and GARCH conditional volatility (expensive to
refit per-symbol on every data update, with limited IC contribution), leaving
the financial feature columns plus three regime indicators for the live
deployment.

Public API:
    compute_financial_features(prices, yield_curve) -> pl.DataFrame

The function expects the same canonical OHLCV schema produced by
`data.load_etfs()` and yield-curve frame built from FRED `dgs10` / `dgs2`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from ml4t.engineer.features.momentum import adx, aroon, cci, macd, rsi, stochastic
from ml4t.engineer.features.regime import choppiness_index, hurst_exponent
from ml4t.engineer.features.trend import ema, sma
from ml4t.engineer.features.volatility import natr
from ml4t.engineer.features.volume import obv

MOMENTUM_HORIZONS = [5, 10, 21, 42, 63, 126, 189, 252]
VOLATILITY_HORIZONS = [21, 63, 126, 252]
RANK_FEATURES = ["ret_126d", "sharpe_126d", "vol_63d"]
REGIME_THRESHOLD = 0.005

EXCLUDE_COLS = {
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
}


def build_yield_curve(macro_data: pl.DataFrame) -> pl.DataFrame:
    """Build the yield-curve slope frame the regime indicator expects.

    Args:
        macro_data: FRED macro frame with `timestamp`, `dgs10`, `dgs2`.

    Returns:
        DataFrame with columns: timestamp, yield_10y, yield_2y, slope.
    """
    return (
        macro_data.select(["timestamp", "dgs10", "dgs2"])
        .with_columns(
            (pl.col("dgs10") / 100).alias("yield_10y"),
            (pl.col("dgs2") / 100).alias("yield_2y"),
            ((pl.col("dgs10") - pl.col("dgs2")) / 100).alias("slope"),
        )
        .select(["timestamp", "yield_10y", "yield_2y", "slope"])
        .drop_nulls()
        .sort("timestamp")
    )


def _compute_momentum(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["symbol", "timestamp"])
    df = df.with_columns(
        [
            (
                pl.col("close") / pl.col("close").shift(h).over("symbol").clip(lower_bound=1e-8) - 1
            ).alias(f"ret_{h}d")
            for h in MOMENTUM_HORIZONS
        ]
    )
    df = df.with_columns(
        (pl.col("ret_252d") - pl.col("ret_21d")).alias("skip_recent_12_1"),
        (pl.col("ret_126d") - pl.col("ret_21d")).alias("skip_recent_6_1"),
    )
    df = df.with_columns(pl.col("close").log().diff().over("symbol").alias("log_return"))
    df = df.with_columns(
        [
            pl.col("log_return").rolling_std(h).over("symbol").mul(np.sqrt(252)).alias(f"vol_{h}d")
            for h in VOLATILITY_HORIZONS
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("log_return").rolling_sum(h).over("symbol")
                / pl.col("log_return").rolling_std(h).over("symbol")
                * np.sqrt(252 / h)
            ).alias(f"sharpe_{h}d")
            for h in MOMENTUM_HORIZONS
        ]
    )
    df = df.with_columns(
        (pl.col("ret_21d") - pl.col("ret_63d")).alias("mom_accel_short"),
        (pl.col("ret_63d") - pl.col("ret_126d")).alias("mom_accel_medium"),
        (pl.col("ret_126d") - pl.col("ret_252d")).alias("mom_accel_long"),
    )
    return df.with_columns(
        (pl.col("vol_21d") / pl.col("vol_63d")).alias("vol_ratio_short"),
        (pl.col("vol_63d") / pl.col("vol_126d")).alias("vol_ratio_medium"),
    )


def _compute_technicals(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        rsi("close", period=7).over("symbol").alias("rsi_7"),
        rsi("close", period=14).over("symbol").alias("rsi_14"),
        macd("close", fast_period=12, slow_period=26).over("symbol").alias("macd_line"),
        adx("high", "low", "close", period=14).over("symbol").alias("adx_14"),
        cci("high", "low", "close", period=14).over("symbol").alias("cci_14"),
        cci("high", "low", "close", period=21).over("symbol").alias("cci_21"),
        stochastic("high", "low", "close", fastk_period=14).over("symbol").alias("stoch_k"),
    )
    df = (
        df.with_columns(aroon("high", "low", timeperiod=25).over("symbol").alias("_aroon_struct"))
        .with_columns(
            (
                pl.col("_aroon_struct").struct.field("up")
                - pl.col("_aroon_struct").struct.field("down")
            ).alias("aroon_diff")
        )
        .drop("_aroon_struct")
    )
    df = df.with_columns(
        (pl.col("close") / sma("close", period=50).over("symbol")).alias("sma_ratio_50"),
        (pl.col("close") / sma("close", period=200).over("symbol")).alias("sma_ratio_200"),
        (pl.col("close") / ema("close", period=26).over("symbol")).alias("ema_ratio_26"),
    )
    df = df.with_columns(
        pl.col("close").rolling_mean(20).over("symbol").alias("_bb_mid"),
        pl.col("close").rolling_std(20).over("symbol").alias("_bb_std"),
    )
    df = df.with_columns(
        (
            (pl.col("close") - (pl.col("_bb_mid") - 2 * pl.col("_bb_std")))
            / (4 * pl.col("_bb_std"))
        ).alias("bb_pctb_20")
    ).drop(["_bb_mid", "_bb_std"])
    return df.with_columns(natr("high", "low", "close", period=14).over("symbol").alias("natr_14"))


def _compute_regime_volume(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        choppiness_index("high", "low", "close", period=14).over("symbol").alias("chop_14"),
        hurst_exponent("close", period=100).over("symbol").alias("hurst_100"),
    )
    df = df.with_columns(
        (
            (pl.col("close") - pl.col("close").rolling_max(63).over("symbol"))
            / pl.col("close").rolling_max(63).over("symbol").clip(lower_bound=1e-8)
        ).alias("max_dd_63d"),
        (
            (pl.col("close") - pl.col("close").rolling_max(126).over("symbol"))
            / pl.col("close").rolling_max(126).over("symbol").clip(lower_bound=1e-8)
        ).alias("max_dd_126d"),
    )
    df = df.with_columns(
        (
            pl.col("volume")
            / pl.col("volume").rolling_mean(21).over("symbol").clip(lower_bound=1e-8)
        ).alias("vol_ratio_21d_raw"),
        (
            pl.col("volume")
            / pl.col("volume").rolling_mean(63).over("symbol").clip(lower_bound=1e-8)
        ).alias("vol_ratio_63d_raw"),
    )
    df = df.with_columns(
        pl.col("vol_ratio_21d_raw")
        .clip(
            pl.col("vol_ratio_21d_raw").quantile(0.01).over("timestamp"),
            pl.col("vol_ratio_21d_raw").quantile(0.99).over("timestamp"),
        )
        .alias("vol_ratio_21d"),
        pl.col("vol_ratio_63d_raw")
        .clip(
            pl.col("vol_ratio_63d_raw").quantile(0.01).over("timestamp"),
            pl.col("vol_ratio_63d_raw").quantile(0.99).over("timestamp"),
        )
        .alias("vol_ratio_63d"),
    ).drop(["vol_ratio_21d_raw", "vol_ratio_63d_raw"])
    df = df.with_columns(obv("close", "volume").over("symbol").alias("_obv"))
    df = df.with_columns(
        (
            (pl.col("_obv") - pl.col("_obv").rolling_mean(63).over("symbol"))
            / pl.col("_obv").rolling_std(63).over("symbol")
        ).alias("obv_zscore_63d"),
    ).drop("_obv")
    df = df.with_columns(
        (pl.col("log_return") > 0)
        .cast(pl.Float64)
        .rolling_mean(63)
        .over("symbol")
        .alias("pct_positive_63d"),
    )
    return df.with_columns(
        (
            pl.col("close") / pl.col("close").rolling_max(252).over("symbol").clip(lower_bound=1e-8)
        ).alias("dist_52w_high"),
        (
            pl.col("close") / pl.col("close").rolling_min(252).over("symbol").clip(lower_bound=1e-8)
        ).alias("dist_52w_low"),
    )


def _compute_cross_asset(df: pl.DataFrame) -> pl.DataFrame:
    # Hard-fail when either anchor symbol is missing rather than silently
    # populating ``corr_spy_tlt_63d`` with NULLs (an empty inner-join would
    # do exactly that, which the model has never seen during training).
    available = set(df["symbol"].unique().to_list())
    missing = [s for s in ("SPY", "TLT") if s not in available]
    if missing:
        raise ValueError(
            f"_compute_cross_asset requires SPY and TLT in `prices`; missing: {missing}"
        )
    spy_ret = (
        df.filter(pl.col("symbol") == "SPY")
        .select(["timestamp", "log_return"])
        .rename({"log_return": "_spy_ret"})
    )
    tlt_ret = (
        df.filter(pl.col("symbol") == "TLT")
        .select(["timestamp", "log_return"])
        .rename({"log_return": "_tlt_ret"})
    )
    corr_df = (
        spy_ret.join(tlt_ret, on="timestamp", how="inner")
        .sort("timestamp")
        .with_columns(
            pl.rolling_corr(pl.col("_spy_ret"), pl.col("_tlt_ret"), window_size=63).alias(
                "corr_spy_tlt_63d"
            )
        )
        .select(["timestamp", "corr_spy_tlt_63d"])
    )
    return df.join(corr_df, on="timestamp", how="left")


def _add_cross_sectional_ranks(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col(c).rank().over("timestamp") / pl.col(c).count().over("timestamp")).alias(
                f"{c}_rank"
            )
            for c in RANK_FEATURES
        ]
    )


def _add_regime_indicators(
    df: pl.DataFrame, yield_curve: pl.DataFrame, threshold: float = REGIME_THRESHOLD
) -> pl.DataFrame:
    # Sort BEFORE the rolling computation: rolling_mean / rolling_std use the
    # incoming row order, and the trailing .sort() only orders the output —
    # not the rolling-window inputs. A caller that hands us an unsorted frame
    # would otherwise compute yield_curve_zscore on shuffled values.
    regime_df = (
        yield_curve.sort("timestamp")
        .with_columns(
            pl.when(pl.col("slope") > threshold).then(1).otherwise(0).alias("regime"),
            pl.col("slope").alias("yield_curve_slope"),
            (
                (pl.col("slope") - pl.col("slope").rolling_mean(252))
                / pl.col("slope").rolling_std(252)
            ).alias("yield_curve_zscore"),
        )
        .select(["timestamp", "regime", "yield_curve_slope", "yield_curve_zscore"])
    )
    return df.sort("timestamp").join_asof(regime_df, on="timestamp", strategy="backward")


def compute_financial_features(prices: pl.DataFrame, yield_curve: pl.DataFrame) -> pl.DataFrame:
    """Compute the financial-only feature subset for the ETFs deployment loop.

    Args:
        prices: OHLCV frame with columns timestamp, symbol, open, high, low, close, volume.
        yield_curve: Yield-curve frame from `build_yield_curve(load_macro())`.

    Returns:
        DataFrame with timestamp, symbol, the financial feature columns
        (momentum, technicals, regime-volume, cross-asset, cross-sectional ranks)
        plus three regime indicators (regime, yield_curve_slope,
        yield_curve_zscore). The exact count is pinned by ``feature_columns()``.
    """
    features = (
        prices.pipe(_compute_momentum)
        .pipe(_compute_technicals)
        .pipe(_compute_regime_volume)
        .pipe(_compute_cross_asset)
    )
    features = _add_cross_sectional_ranks(features)
    features = _add_regime_indicators(features, yield_curve)
    return features.sort(["symbol", "timestamp"])


def feature_columns(features: pl.DataFrame) -> list[str]:
    """Return the deployment-model feature column list (deterministic order)."""
    return sorted(c for c in features.columns if c not in EXCLUDE_COLS)
