#!/usr/bin/env python3
"""Download CFTC Commitment of Traders (COT) data.

CFTC publishes weekly COT reports (Tuesday snapshot, released Friday) showing
futures positioning broken down by trader type (dealers, asset managers,
leveraged money for financial futures; commercials, managed money for
commodities). The data is free and useful for sentiment/positioning features.

This downloader uses ``ml4t.data.cot.COTFetcher`` — which wraps the
``cot_reports`` library — to fetch per-product panels and writes one parquet
per product to ``$ML4T_DATA_PATH/futures/positioning/cot/{product}.parquet``. The
``load_cot()`` loader in ``data/futures/loader.py`` consumes these files.

Output layout under ``$ML4T_DATA_PATH/futures/positioning/cot/``::

    {PRODUCT}.parquet    one parquet per product code (e.g., ES.parquet)

Schema (columns vary by report type but include):

    product              exchange product code (ES, CL, GC, …)
    report_type          CFTC report that produced the row
    report_date          Tuesday snapshot date
    open_interest        total open interest
    <trader>_long        long positions per trader category
    <trader>_short       short positions per trader category
    <trader>_net         computed long − short per category

Usage::

    # Default: all products in PRODUCT_MAPPINGS, 2020–current year
    python data/futures/positioning/cot_download.py

    # Restrict to a subset
    python data/futures/positioning/cot_download.py --products ES,NQ,CL,GC

    # Wider year range
    python data/futures/positioning/cot_download.py --start-year 2010 --end-year 2024

    # Override output root
    python data/futures/positioning/cot_download.py --data-path /tmp/ml4t-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml4t.data.cot import PRODUCT_MAPPINGS, COTConfig, COTFetcher

from utils.downloading import resolve_data_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download CFTC Commitment of Traders data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--products",
        type=str,
        default=None,
        help=(
            "Comma-separated product codes (default: all in PRODUCT_MAPPINGS). "
            f"Available: {', '.join(sorted(PRODUCT_MAPPINGS.keys()))}"
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="First calendar year to fetch (default: 2020)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last calendar year to fetch (default: current year)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override output root (default: $ML4T_DATA_PATH)",
    )
    args = parser.parse_args()

    if args.products:
        products = [p.strip().upper() for p in args.products.split(",") if p.strip()]
        unknown = [p for p in products if p not in PRODUCT_MAPPINGS]
        if unknown:
            print(f"ERROR: unknown product code(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(PRODUCT_MAPPINGS.keys()))}")
            return 1
    else:
        products = sorted(PRODUCT_MAPPINGS.keys())

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "futures" / "positioning" / "cot"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = COTConfig(
        products=products,
        start_year=args.start_year,
        end_year=args.end_year,
        storage_path=output_dir,
    )
    fetcher = COTFetcher(config)

    print()
    print(f"Output:       {output_dir}")
    print(f"Years:        {config.start_year}–{config.end_year}")
    print(f"Products:     {len(products)} ({', '.join(products)})")
    print()

    written = 0
    empty = 0
    failed: list[str] = []
    for i, product in enumerate(products, 1):
        print(f"  [{i}/{len(products)}] {product}…", end="", flush=True)
        try:
            df = fetcher.fetch_product(product)
        except Exception as e:
            failed.append(product)
            print(f" FAILED ({e})")
            continue

        if df.is_empty():
            empty += 1
            print(" no rows returned")
            continue

        out_path = output_dir / f"{product}.parquet"
        df.write_parquet(out_path)
        written += 1
        print(f" {len(df):,} rows → {out_path.name}")

    print()
    print(f"Wrote:        {written} parquet(s)")
    if empty:
        print(f"Empty:        {empty} product(s) returned no rows")
    if failed:
        print(f"Failed:       {len(failed)} — {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
