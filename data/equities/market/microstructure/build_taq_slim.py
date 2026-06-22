"""Build ``microstructure/trade_and_quotes_slim/`` — AAPL tick data for
2020-03-16 (the crash day used by both Ch3 microstructure notebooks) and
2020-03-13 (the preceding Friday, kept for pre-stress comparison).

Output matches the layout of the full TAQ dataset
(``symbol=AAPL/date=YYYYMMDD.parquet``) so the canonical loader is unchanged.

The AlgoSeek parquets are re-encoded with zstd level 22 so the reader
download is as small as possible without changing schema or row order.

Run from repo root:

    uv run python data/equities/market/microstructure/build_taq_slim.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

SRC_DIR = Path(__file__).parent / "trade_and_quotes"
OUT_DIR = Path(__file__).parent / "trade_and_quotes_slim"

KEEP = [
    ("AAPL", "20200313"),
    ("AAPL", "20200316"),
]


def main() -> None:
    if not SRC_DIR.exists():
        msg = f"Source TAQ directory not found: {SRC_DIR}"
        raise FileNotFoundError(msg)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    total_mb = 0.0
    for symbol, date in KEEP:
        src = SRC_DIR / f"symbol={symbol}" / f"date={date}.parquet"
        if not src.exists():
            msg = f"Source file missing: {src}"
            raise FileNotFoundError(msg)
        dst = OUT_DIR / f"symbol={symbol}" / f"date={date}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pl.read_parquet(src)
        df.write_parquet(dst, compression="zstd", compression_level=22)

        mb = dst.stat().st_size / 1024 / 1024
        total_mb += mb
        print(f"  {symbol} {date}: {mb:.1f} MB ({len(df):,} rows)")

    print()
    print(f"Total: {total_mb:.1f} MB -> {OUT_DIR}")


if __name__ == "__main__":
    main()
