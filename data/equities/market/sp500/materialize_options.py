"""Materialize SP500 options data into daily summaries.

Raw SP500 options = 347M rows / 11GB on disk / ~30GB in RAM.
Notebooks never need the raw chains for pipeline work — they need daily
per-symbol aggregates. This script produces two materialized files:

1. options_surface_daily.parquet — IV surface summary (ATM IV at 3 tenors,
   25-delta skew, term structure, quality metrics). Used by
   sp500_equity_option_analytics case study.

2. options_straddles_daily.parquet — 30D ATM straddle selection (call/put
   legs, straddle metrics, Greeks). Used by sp500_options case study.

Both files: ~500 symbols × 1250 days ≈ 600K rows each.

Run from repo root:
    uv run python data/equities/market/sp500/materialize_options.py

After materialization, notebooks use:
    from data import load_sp500_options_surface, load_sp500_options_straddles
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).parent / "options"
OUT_DIR = Path(__file__).parent

SURFACE_OUT = OUT_DIR / "options_surface_daily.parquet"
STRADDLES_OUT = OUT_DIR / "options_straddles_daily.parquet"

YEARS = [2017, 2018, 2019, 2020, 2021]

# ---------------------------------------------------------------------------
# Surface summary parameters (from sp500_equity_option_analytics/03_features)
# ---------------------------------------------------------------------------
DTE_BUCKETS = {"7d": (5, 10), "30d": (25, 35), "90d": (80, 110)}
DELTA_TARGETS = {"atm": 0.50, "25d": 0.25}

# Straddle parameters (matches sp500_options case study selection criteria)
STRADDLE_DTE_WINDOW = (25, 35)
STRADDLE_TARGET_DELTA = 0.50
STRADDLE_DELTA_TOL = 0.15
STRADDLE_MIN_BID = 0.01
STRADDLE_MAX_REL_SPREAD = 0.30


# ===================================================================
# Surface summary extraction
# ===================================================================
def _select_surface_point(
    df: pl.DataFrame,
    dte_bucket: tuple[int, int],
    delta_target: float,
    call_put: str | None = None,
) -> pl.DataFrame:
    """Select contract closest to target delta within DTE bucket."""
    filtered = df.filter(
        (pl.col("days_to_maturity").is_between(dte_bucket[0], dte_bucket[1]))
        & pl.col("delta").is_not_null()
        & pl.col("implied_vol").is_not_null()
    )
    if call_put is not None:
        filtered = filtered.filter(pl.col("call_put") == call_put)

    filtered = filtered.with_columns(
        (pl.col("delta").abs() - delta_target).abs().alias("delta_distance")
    )
    return (
        filtered.with_columns(
            pl.col("delta_distance").rank("ordinal").over(["date", "symbol"]).alias("_rank")
        )
        .filter(pl.col("_rank") == 1)
        .drop("_rank")
        .select(
            [
                "date",
                "symbol",
                "implied_vol",
                "days_to_maturity",
                "delta",
                "bid",
                "ask",
                "mid_price",
                "iv_convergence",
            ]
        )
    )


def compute_surface_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute daily surface summary per symbol.

    Extracts: iv_7_atm, iv_30_atm, iv_90_atm, 25-delta skew, term structure,
    spread, and quality metrics.
    """
    # ATM IV at 30D (average of call and put)
    atm_30_call = _select_surface_point(df, DTE_BUCKETS["30d"], DELTA_TARGETS["atm"], "C")
    atm_30_put = _select_surface_point(df, DTE_BUCKETS["30d"], DELTA_TARGETS["atm"], "P")

    iv_30_atm = (
        atm_30_call.select(["date", "symbol", "implied_vol"])
        .rename({"implied_vol": "iv_30_atm_call"})
        .join(
            atm_30_put.select(["date", "symbol", "implied_vol"]).rename(
                {"implied_vol": "iv_30_atm_put"}
            ),
            on=["date", "symbol"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            ((pl.col("iv_30_atm_call") + pl.col("iv_30_atm_put")) / 2)
            .fill_null(pl.coalesce(pl.col("iv_30_atm_call"), pl.col("iv_30_atm_put")))
            .alias("iv_30_atm")
        )
        .select(["date", "symbol", "iv_30_atm"])
    )

    # ATM IV at 7D and 90D for term structure
    atm_7 = _select_surface_point(df, DTE_BUCKETS["7d"], DELTA_TARGETS["atm"])
    atm_90 = _select_surface_point(df, DTE_BUCKETS["90d"], DELTA_TARGETS["atm"])

    iv_term = (
        atm_7.select(["date", "symbol", "implied_vol"])
        .rename({"implied_vol": "iv_7_atm"})
        .join(
            atm_90.select(["date", "symbol", "implied_vol"]).rename({"implied_vol": "iv_90_atm"}),
            on=["date", "symbol"],
            how="full",
            coalesce=True,
        )
    )

    # 25-delta skew (put vs call at 30D)
    put_25d = _select_surface_point(df, DTE_BUCKETS["30d"], DELTA_TARGETS["25d"], "P")
    call_25d = _select_surface_point(df, DTE_BUCKETS["30d"], DELTA_TARGETS["25d"], "C")

    iv_skew = (
        put_25d.select(["date", "symbol", "implied_vol"])
        .rename({"implied_vol": "iv_30_put_25d"})
        .join(
            call_25d.select(["date", "symbol", "implied_vol"]).rename(
                {"implied_vol": "iv_30_call_25d"}
            ),
            on=["date", "symbol"],
            how="full",
            coalesce=True,
        )
        .with_columns((pl.col("iv_30_put_25d") - pl.col("iv_30_call_25d")).alias("skew_rr_30_25d"))
    )

    # Quality: relative bid-ask spread for ATM 30D
    atm_30_quality = atm_30_call.with_columns(
        ((pl.col("ask") - pl.col("bid")) / pl.col("mid_price").clip(lower_bound=0.01)).alias(
            "spread_atm_30"
        )
    ).select(["date", "symbol", "spread_atm_30"])

    # Convergence quality share
    qc_call = atm_30_call.select(
        [
            "date",
            "symbol",
            (pl.col("iv_convergence") == "Converged").cast(pl.Int32).alias("_converged"),
        ]
    )
    qc_put = atm_30_put.select(
        [
            "date",
            "symbol",
            (pl.col("iv_convergence") == "Converged").cast(pl.Int32).alias("_converged"),
        ]
    )
    qc_share = (
        pl.concat([qc_call, qc_put])
        .group_by(["date", "symbol"])
        .agg(pl.col("_converged").mean().alias("qc_converged_share"))
    )

    # Term structure measures
    surface = (
        iv_30_atm.join(iv_term, on=["date", "symbol"], how="full", coalesce=True)
        .join(iv_skew, on=["date", "symbol"], how="full", coalesce=True)
        .join(atm_30_quality, on=["date", "symbol"], how="full", coalesce=True)
        .join(qc_share, on=["date", "symbol"], how="full", coalesce=True)
    )
    surface = surface.with_columns(
        (pl.col("iv_30_atm") - pl.col("iv_7_atm")).alias("term_slope_near_atm"),
        (pl.col("iv_90_atm") - pl.col("iv_30_atm")).alias("term_slope_far_atm"),
        (pl.col("iv_90_atm") / pl.col("iv_7_atm").clip(lower_bound=0.01)).alias("term_ratio_atm"),
        ((pl.col("iv_7_atm") + pl.col("iv_90_atm")) / 2 - pl.col("iv_30_atm")).alias(
            "term_convexity"
        ),
        (pl.col("skew_rr_30_25d") / pl.col("iv_30_atm").clip(lower_bound=0.01)).alias(
            "skew_to_atm_ratio"
        ),
    )
    return surface.sort(["date", "symbol"])


# ===================================================================
# Straddle extraction
# ===================================================================
def compute_straddles(df: pl.DataFrame) -> pl.DataFrame:
    """Extract 30D ATM matched-strike straddles from raw option chains.

    Joins calls and puts on (date, symbol, strike, expiration) so both legs
    share the same contract. Scores each matched pair by delta-neutrality
    (|call_delta + put_delta| → 0) and selects the best pair per (symbol, date).

    Returns one row per (symbol, date) with call/put leg details and
    straddle aggregate metrics.
    """
    # Pre-filter to ATM-ish, liquid, converged options in the DTE window
    opts = df.with_columns(
        pl.col("delta").abs().alias("abs_delta"),
        ((pl.col("ask") - pl.col("bid")) / pl.col("mid_price").clip(lower_bound=0.01)).alias(
            "rel_spread"
        ),
    )
    filtered = opts.filter(
        pl.col("days_to_maturity").is_between(STRADDLE_DTE_WINDOW[0], STRADDLE_DTE_WINDOW[1])
        & (pl.col("bid") >= STRADDLE_MIN_BID)
        & (pl.col("ask") >= STRADDLE_MIN_BID)
        & (pl.col("rel_spread") <= STRADDLE_MAX_REL_SPREAD)
        & (pl.col("iv_convergence") == "Converged")
        & (
            pl.col("abs_delta").is_between(
                STRADDLE_TARGET_DELTA - STRADDLE_DELTA_TOL,
                STRADDLE_TARGET_DELTA + STRADDLE_DELTA_TOL,
            )
        )
    )

    # Separate calls and puts, rename to avoid collisions on join
    calls = filtered.filter(pl.col("call_put") == "C").select(
        [
            "date",
            "symbol",
            "strike",
            "expiration",
            "days_to_maturity",
            "underlying_price",
            pl.col("bid").alias("call_bid"),
            pl.col("ask").alias("call_ask"),
            pl.col("mid_price").alias("call_mid"),
            pl.col("implied_vol").alias("call_iv"),
            pl.col("delta").alias("call_delta"),
            pl.col("abs_delta").alias("call_abs_delta"),
            pl.col("gamma").alias("call_gamma"),
            pl.col("theta").alias("call_theta"),
            pl.col("vega").alias("call_vega"),
            pl.col("rel_spread").alias("call_rel_spread"),
            pl.col("iv_convergence").alias("call_convergence"),
        ]
    )

    puts = filtered.filter(pl.col("call_put") == "P").select(
        [
            "date",
            "symbol",
            "strike",
            "expiration",
            pl.col("bid").alias("put_bid"),
            pl.col("ask").alias("put_ask"),
            pl.col("mid_price").alias("put_mid"),
            pl.col("implied_vol").alias("put_iv"),
            pl.col("delta").alias("put_delta"),
            pl.col("abs_delta").alias("put_abs_delta"),
            pl.col("gamma").alias("put_gamma"),
            pl.col("theta").alias("put_theta"),
            pl.col("vega").alias("put_vega"),
            pl.col("rel_spread").alias("put_rel_spread"),
            pl.col("iv_convergence").alias("put_convergence"),
        ]
    )

    # Matched-strike join: same (date, symbol, strike, expiration) for both legs
    pairs = calls.join(puts, on=["date", "symbol", "strike", "expiration"], how="inner")

    # Score by delta-neutrality: |call_delta + put_delta| → 0 is best
    pairs = pairs.with_columns(
        (pl.col("call_delta") + pl.col("put_delta")).abs().alias("_pair_delta_imbalance")
    )

    # Select best pair per (date, symbol) — most delta-neutral
    straddles = (
        pairs.with_columns(
            pl.col("_pair_delta_imbalance").rank("ordinal").over(["date", "symbol"]).alias("_rank")
        )
        .filter(pl.col("_rank") == 1)
        .drop(["_rank", "_pair_delta_imbalance"])
    )

    # Add put_strike / put_expiration aliases for backward compatibility
    # (they're identical to strike/expiration since we matched)
    straddles = straddles.with_columns(
        pl.col("strike").alias("put_strike"),
        pl.col("expiration").alias("put_expiration"),
    )

    # Straddle aggregate metrics
    straddles = straddles.with_columns(
        pl.lit("straddle_30d_atm").alias("instrument_id"),
        (pl.col("call_mid") + pl.col("put_mid")).alias("instr_mid"),
        (pl.col("call_bid") + pl.col("put_bid")).alias("instr_bid"),
        (pl.col("call_ask") + pl.col("put_ask")).alias("instr_ask"),
        ((pl.col("call_iv") + pl.col("put_iv")) / 2).alias("iv_atm"),
        (pl.col("call_delta") + pl.col("put_delta")).alias("instr_delta"),
        (pl.col("call_gamma") + pl.col("put_gamma")).alias("instr_gamma"),
        (pl.col("call_theta") + pl.col("put_theta")).alias("instr_theta"),
        (pl.col("call_vega") + pl.col("put_vega")).alias("instr_vega"),
        pl.col("days_to_maturity").alias("instr_dte"),
    )
    straddles = straddles.with_columns(
        (pl.col("instr_ask") - pl.col("instr_bid")).alias("instr_spread"),
        ((pl.col("instr_ask") - pl.col("instr_bid")) / pl.col("instr_mid")).alias(
            "instr_rel_spread"
        ),
        (pl.col("instr_mid") / pl.col("underlying_price")).alias("instr_pct_of_S"),
        pl.when(
            (pl.col("call_convergence") == "Converged") & (pl.col("put_convergence") == "Converged")
        )
        .then(0)
        .otherwise(1)
        .alias("qc_any_estimated_iv"),
    )
    return straddles.sort(["symbol", "date"])


# ===================================================================
# Main: year-by-year processing
# ===================================================================
def main():
    print("=" * 60)
    print("SP500 Options Materialization")
    print(f"Raw data: {RAW_DIR}")
    print(f"Years: {YEARS}")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw options directory not found: {RAW_DIR}")

    surface_parts = []
    straddle_parts = []

    for year in YEARS:
        t0 = time.time()
        pattern = f"year={year}/*.parquet"
        files = list(RAW_DIR.glob(pattern))
        if not files:
            print(f"\n[{year}] No files found for pattern {pattern}, skipping")
            continue

        print(f"\n[{year}] Loading {len(files)} partitions...")

        # Load one year at a time
        df = pl.read_parquet(
            RAW_DIR / pattern,
            hive_partitioning=True,
        )
        n_raw = len(df)
        print(f"  Raw: {n_raw:,} rows, {df['symbol'].n_unique()} symbols")

        # Surface summary
        surface = compute_surface_summary(df)
        print(f"  Surface: {len(surface):,} symbol-days")
        surface_parts.append(surface)

        # Straddles
        straddles = compute_straddles(df)
        print(f"  Straddles: {len(straddles):,} symbol-days")
        straddle_parts.append(straddles)

        del df
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")

    # Combine and save
    if surface_parts:
        surface_all = pl.concat(surface_parts).sort(["date", "symbol"])
        surface_all.write_parquet(SURFACE_OUT, compression="zstd", compression_level=22)
        print(f"\nSurface summary: {SURFACE_OUT}")
        print(f"  {len(surface_all):,} rows, {surface_all['symbol'].n_unique()} symbols")
        print(f"  Columns: {surface_all.columns}")
        print(f"  File size: {SURFACE_OUT.stat().st_size / 1024 / 1024:.1f} MB")

    if straddle_parts:
        straddles_all = pl.concat(straddle_parts).sort(["symbol", "date"])
        straddles_all.write_parquet(STRADDLES_OUT, compression="zstd", compression_level=22)
        print(f"\nStraddles: {STRADDLES_OUT}")
        print(f"  {len(straddles_all):,} rows, {straddles_all['symbol'].n_unique()} symbols")
        print(f"  Columns: {straddles_all.columns}")
        print(f"  File size: {STRADDLES_OUT.stat().st_size / 1024 / 1024:.1f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
