"""Build ``sp500/options_eda/year=YYYY.parquet`` — source files for the
Chapter 2 option-chain, Greeks, continuous-maturity, and Chapter 8
cross-instrument feature notebooks.

- Symbols: AAPL, MSFT, GOOGL, AMZN, JPM, BA, XOM, KO (8 S&P 500
  constituents across sectors and IV regimes)
- Dates: 2019-2020, partitioned by year so readers can download one year
  at a time
- Schema: full raw AlgoSeek option chain (all Greeks + IV diagnostics)

Run from repo root:

    uv run python data/equities/market/sp500/build_options_eda.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from utils.downloading import resolve_data_dir


def sp500_data_dir(data_path: Path | None = None) -> Path:
    """Where the loaders read this dataset from.

    Not ``Path(__file__).parent``: the converter writes under ``$ML4T_DATA_PATH``,
    which a reader may point outside the repository, and a build script anchored
    to its own directory would then look in the wrong place and leave its output
    somewhere the loaders never read.
    """
    return resolve_data_dir(data_path) / "equities" / "market" / "sp500"


EDA_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BA", "XOM", "KO"]
EDA_YEARS = [2019, 2020]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH or repo/data)",
    )
    args = parser.parse_args()

    base = sp500_data_dir(args.data_path)
    RAW_DIR = base / "options"
    OUT_DIR = base / "options_eda"

    if not RAW_DIR.exists():
        msg = f"Raw options directory not found: {RAW_DIR}"
        raise FileNotFoundError(msg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building SP500 options EDA subset")
    print(f"  Source: {RAW_DIR}")
    print(f"  Output: {OUT_DIR}/year=YYYY.parquet")
    print(f"  Symbols: {EDA_SYMBOLS}")
    print(f"  Years: {EDA_YEARS}")
    print("=" * 60)

    overall = time.time()
    total_rows = 0
    total_mb = 0.0
    for year in EDA_YEARS:
        t0 = time.time()
        # A reader may have converted only part of the archive. Skipping a year
        # with no chains beats a polars schema error naming a glob;
        # build_options_straddles_raw.py already does the same.
        if not list(RAW_DIR.glob(f"year={year}/*.parquet")):
            print(f"\n[{year}] No converted chains — skipping")
            continue
        df = (
            pl.scan_parquet(RAW_DIR / f"year={year}/*.parquet", hive_partitioning=True)
            .filter(pl.col("symbol").is_in(EDA_SYMBOLS))
            .collect()
            .sort(["symbol", "date", "expiration", "call_put", "strike"])
        )
        out_path = OUT_DIR / f"year={year}.parquet"
        df.write_parquet(out_path, compression="zstd", compression_level=22, statistics=True)
        size_mb = out_path.stat().st_size / 1024 / 1024
        total_rows += len(df)
        total_mb += size_mb
        print(
            f"  {year}: {len(df):,} rows, {df['symbol'].n_unique()} symbols, "
            f"{size_mb:.1f} MB, {time.time() - t0:.0f}s"
        )

    print()
    print(f"Total: {total_rows:,} rows, {total_mb:.1f} MB, {time.time() - overall:.0f}s")


if __name__ == "__main__":
    main()
