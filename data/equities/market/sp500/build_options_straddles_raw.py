"""Build ``sp500/options_straddles_raw/`` — source chains for the
sp500_options case study.

Two-pass extraction:

1. Identify every ``(symbol, strike, expiration)`` that passes the 30D ATM
   candidate filter (DTE ∈ [25, 35], |delta| ∈ [0.35, 0.65], Converged IV,
   bid ≥ 0.01, relative spread ≤ 0.30) at any point in 2017-2021. Filter
   parameters match ``compute_straddles()`` in ``materialize_options.py`` so
   the derived ``options_straddles_daily.parquet`` is byte-identical when
   regenerated from this slice.
2. Emit all daily observations for every candidate contract (both legs,
   from first listing through expiration) — the full lifecycle needed for
   same-contract exit prices and daily delta hedging.

Output: ``options_straddles_raw/year=YYYY.parquet`` (hive-partitioned).

Run from repo root:

    uv run python data/equities/market/sp500/build_options_straddles_raw.py
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).parent / "options"
OUT_DIR = Path(__file__).parent / "options_straddles_raw"

YEARS = [2017, 2018, 2019, 2020, 2021]

# Must match compute_straddles() in materialize_options.py so the derived
# options_straddles_daily.parquet is byte-identical when regenerated from
# this slim set.
STRADDLE_DTE_WINDOW = (25, 35)
STRADDLE_TARGET_DELTA = 0.50
STRADDLE_DELTA_TOL = 0.15
STRADDLE_MIN_BID = 0.01
STRADDLE_MAX_REL_SPREAD = 0.30


def identify_candidate_contracts(df: pl.DataFrame) -> pl.DataFrame:
    """Return the unique `(symbol, strike, expiration)` triples that pass
    the ATM straddle candidate filter at any point in the year.
    """
    rel_spread = (pl.col("ask") - pl.col("bid")) / pl.col("mid_price").clip(lower_bound=0.01)
    abs_delta = pl.col("delta").abs()

    candidates = (
        df.filter(
            pl.col("days_to_maturity").is_between(*STRADDLE_DTE_WINDOW)
            & (pl.col("bid") >= STRADDLE_MIN_BID)
            & (pl.col("ask") >= STRADDLE_MIN_BID)
            & (rel_spread <= STRADDLE_MAX_REL_SPREAD)
            & (pl.col("iv_convergence") == "Converged")
            & abs_delta.is_between(
                STRADDLE_TARGET_DELTA - STRADDLE_DELTA_TOL,
                STRADDLE_TARGET_DELTA + STRADDLE_DELTA_TOL,
            )
        )
        .select(["symbol", "strike", "expiration"])
        .unique()
    )
    return candidates


def main() -> None:
    if not RAW_DIR.exists():
        msg = f"Raw options directory not found: {RAW_DIR}"
        raise FileNotFoundError(msg)

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Building SP500 options straddles raw slice (ATM-band, lifecycle-preserving)")
    print(f"  Source: {RAW_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Years: {YEARS}")
    print(
        f"  Filter: DTE ∈ [{STRADDLE_DTE_WINDOW[0]}, {STRADDLE_DTE_WINDOW[1]}], "
        f"|delta| ∈ [{STRADDLE_TARGET_DELTA - STRADDLE_DELTA_TOL:.2f}, "
        f"{STRADDLE_TARGET_DELTA + STRADDLE_DELTA_TOL:.2f}], Converged IV"
    )
    print("=" * 60)

    total_raw = 0
    total_kept = 0
    total_size_mb = 0.0
    overall_start = time.time()

    for year in YEARS:
        t0 = time.time()
        pattern = f"year={year}/*.parquet"
        files = list(RAW_DIR.glob(pattern))
        if not files:
            print(f"\n[{year}] No files — skipping")
            continue

        print(f"\n[{year}] Loading {len(files)} partitions …")
        df = pl.read_parquet(RAW_DIR / pattern, hive_partitioning=True)
        n_raw = len(df)
        total_raw += n_raw
        print(f"  Raw: {n_raw:,} rows, {df['symbol'].n_unique()} symbols")

        candidates = identify_candidate_contracts(df)
        n_candidates = len(candidates)
        print(f"  Candidate contracts (year-local): {n_candidates:,}")

        # Also include candidates identified in adjacent years whose expirations
        # fall within this year — a contract can enter the 25-35 DTE window in
        # one year and be held across a year boundary. Safer to process all
        # candidates globally in one final join, so do that below.
        candidates.write_parquet(OUT_DIR / f"_candidates_{year}.parquet")
        del df, candidates
        gc.collect()
        print(f"  Year {year} candidate scan: {time.time() - t0:.0f}s")

    # Global candidate union (so cross-year lifecycles are preserved)
    print("\nBuilding global candidate union …")
    candidate_files = [OUT_DIR / f"_candidates_{y}.parquet" for y in YEARS]
    candidate_files = [p for p in candidate_files if p.exists()]
    all_candidates = (
        pl.concat([pl.read_parquet(p) for p in candidate_files])
        .unique()
        .sort(["symbol", "expiration", "strike"])
    )
    print(f"  Total unique candidate contracts: {len(all_candidates):,}")

    # Pass 2: second sweep, keeping every daily observation of any candidate
    # contract (both legs, entry through expiration).
    for year in YEARS:
        pattern = f"year={year}/*.parquet"
        files = list(RAW_DIR.glob(pattern))
        if not files:
            continue

        t0 = time.time()
        print(f"\n[{year}] Filtering to lifecycle observations …")
        df = pl.read_parquet(RAW_DIR / pattern, hive_partitioning=True)
        kept = df.join(all_candidates, on=["symbol", "strike", "expiration"], how="semi").sort(
            ["symbol", "date", "expiration", "call_put", "strike"]
        )

        out_path = OUT_DIR / f"year={year}.parquet"
        kept.write_parquet(out_path, compression="zstd", compression_level=22, statistics=True)
        size_mb = out_path.stat().st_size / 1024 / 1024
        total_kept += len(kept)
        total_size_mb += size_mb
        print(
            f"  {year}: {len(kept):,} rows kept ({len(kept) / len(df):.1%}), "
            f"{size_mb:.1f} MB, {time.time() - t0:.0f}s"
        )
        del df, kept
        gc.collect()

    # Clean up interim candidate files
    for p in candidate_files:
        p.unlink()

    print()
    print("=" * 60)
    print(f"Total raw rows: {total_raw:,}")
    print(f"Total kept rows: {total_kept:,} ({total_kept / total_raw:.1%})")
    print(f"Total output size: {total_size_mb:.1f} MB")
    print(f"Overall elapsed: {time.time() - overall_start:.0f}s")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
