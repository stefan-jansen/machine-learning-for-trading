#!/usr/bin/env python3
"""
Download Wiki Prices dataset from NASDAQ Data Link.

Wiki Prices is a frozen historical US equities dataset (1962-2018) with 3,199 stocks.
Survivorship-bias free - includes delisted companies.

Usage:
    python downloads/nasdaq_wiki_prices.py              # Download to default location
    python downloads/nasdaq_wiki_prices.py --dry-run   # Show what would be downloaded
    python downloads/nasdaq_wiki_prices.py --data-path ~/my-data/  # Custom location

Requirements:
    - Free NASDAQ Data Link API key (https://data.nasdaq.com/sign-up)
    - Set QUANDL_API_KEY or NASDAQ_DATA_LINK_API_KEY environment variable
"""

import os
import sys
from pathlib import Path

from utils.downloading import (
    create_base_parser,
    load_dotenv,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    resolve_data_dir,
)


def main():
    parser = create_base_parser("Download Wiki Prices dataset from NASDAQ Data Link")
    parser.add_argument(
        "--api-key",
        type=str,
        help="NASDAQ Data Link API key (or set QUANDL_API_KEY env var)",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        print_dry_run_notice()

    from ml4t.data.providers.wiki_prices import WikiPricesProvider

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "equities" / "market" / "us_equities"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "us_equities.parquet"

    print_section("US EQUITIES DOWNLOAD (Wiki Prices)")
    print("Dataset: Quandl WIKI Prices (US Equities)")
    print("Coverage: 1962-01-02 to 2018-03-27 (frozen)")
    print("Symbols: 3,199 US companies")
    print("Size: ~650 MB (Parquet)")
    print(f"Output: {output_path}")

    # Dry run: show what would happen
    if args.dry_run:
        print_download_summary(
            {
                "dataset": "Wiki Prices (NASDAQ Data Link)",
                "date_range": "1962-01-02 to 2018-03-27",
                "symbols": "~3,199",
                "estimated_size_mb": 650,
                "output_file": str(output_path),
            },
            dry_run=True,
        )
        return

    # Check for API key
    api_key = args.api_key or os.getenv("QUANDL_API_KEY") or os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not api_key:
        print("\nERROR: No API key found")
        print()
        print("Get a free API key at: https://data.nasdaq.com/sign-up")
        print()
        print("Then either:")
        print("  1. Set QUANDL_API_KEY environment variable")
        print("  2. Add QUANDL_API_KEY=your_key to .env")
        print("  3. Pass --api-key your_key")
        sys.exit(1)

    # Check if already exists
    if output_path.exists() and not args.force:
        print(f"\nFile already exists: {output_path}")
        print("Use --force to re-download")
        return

    print("\nDownloading... (this may take several minutes)\n")

    try:
        downloaded_path = WikiPricesProvider.download(
            output_path=output_dir,
            api_key=api_key,
        )

        # Rename to canonical name if needed
        if downloaded_path.name != "us_equities.parquet":
            final_path = output_dir / "us_equities.parquet"
            downloaded_path.rename(final_path)
            downloaded_path = final_path

        # Print stats
        provider = WikiPricesProvider(parquet_path=downloaded_path)
        stats = provider.get_dataset_stats()

        print_download_summary(
            {
                "rows": stats["total_rows"],
                "symbols": stats["total_symbols"],
                "date_range": f"{stats['date_range'][0]} to {stats['date_range'][1]}",
                "file_size_mb": stats["file_size_mb"],
                "output_file": str(downloaded_path),
            }
        )

    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Download failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
