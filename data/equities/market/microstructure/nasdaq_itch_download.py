#!/usr/bin/env python3
"""
Download NASDAQ ITCH sample data for market microstructure analysis.

Downloads tick-level order book data from NASDAQ's public server.
Files are ~4-6 GB compressed each.

No API key required - data is freely available from:
https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/

Usage:
    python scripts/download_itch_sample.py --list      # List available files
    python scripts/download_itch_sample.py             # Download default (2020-01-30)
    python scripts/download_itch_sample.py --date 01302019  # Download specific date

Note: Files are 4-6 GB each and require parsing (see Chapter 4 notebooks).
"""

import argparse
import sys
from pathlib import Path

from utils.downloading import load_dotenv, resolve_data_dir


def main():
    parser = argparse.ArgumentParser(description="Download NASDAQ ITCH sample data")
    parser.add_argument("--list", action="store_true", help="List available sample files")
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default="01302020",
        help="Date to download (MMDDYYYY format, default: 01302020)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH or repo/data)",
    )
    args = parser.parse_args()

    load_dotenv()

    from ml4t.data.providers.nasdaq_itch import ITCHSampleProvider

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "equities" / "market" / "microstructure" / "nasdaq_itch" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NASDAQ ITCH SAMPLE DATA DOWNLOAD")
    print("=" * 60)
    print()
    print("Source: NASDAQ TotalView-ITCH 5.0")
    print("URL: https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/")
    print()
    print("WARNING: Files are 4-6 GB each!")
    print("   Requires parsing before use (see Chapter 4 notebooks)")
    print()

    # Initialize provider
    provider = ITCHSampleProvider(download_path=output_dir)

    # List available files
    if args.list:
        print("Available sample files:")
        print()
        files = provider.list_available_files()
        for f in files:
            print(f"  {f['name']}")
            print(f"    Date: {f['date']}")
            print(f"    Size: ~{f['size_gb']:.2f} GB")
            print()
        print("Use --date MMDDYYYY to download a specific file")
        return

    print(f"Output: {output_dir}")
    print(f"File: {args.date}.NASDAQ_ITCH50.gz")
    print()

    # Download
    print(f"Downloading ITCH sample for {args.date}...")
    print("(This may take 30-60 minutes depending on connection speed)")
    print()

    try:
        output_path = provider.download(
            date_or_filename=args.date,
        )
        print()
        print(f"Success! Data saved to: {output_path}")
        print()
        print("Next steps:")
        print("  1. Parse the binary data using the Rust parser or Python")
        print("  2. See Chapter 4 notebooks for parsing examples")
        print("  3. Parsed messages will be saved as parquet files")

    except Exception as e:
        print(f"\nERROR: Download failed - {e}")
        print()
        print("Troubleshooting:")
        print("  - Check your internet connection")
        print("  - NASDAQ may have rate limits - try again later")
        print("  - File may no longer be available on NASDAQ server")
        sys.exit(1)


if __name__ == "__main__":
    main()
