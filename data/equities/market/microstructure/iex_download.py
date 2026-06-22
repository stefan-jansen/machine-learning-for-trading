#!/usr/bin/env python3
"""
Download IEX HIST market data for microstructure analysis.

IEX provides free historical market data on a T+1 basis with 12 months rolling history.
No API key required - data is freely available with attribution.

Available feeds:
- TOPS: Top of Book (Best Bid/Offer) - smaller files, BBO analysis only
- DEEP: Depth of Book - full price-level depth, required for LOB reconstruction

File sizes: ~5-10 GB/day for liquid trading days, ~150-500 MB for low-volume days

Usage:
    python iex_hist.py --list                    # List available dates
    python iex_hist.py --smallest                # Download smallest available file
    python iex_hist.py --date 20241220           # Download specific date
    python iex_hist.py --date 20241220 --deep    # Download DEEP (full depth)

Attribution (Required):
    Data provided for free by IEX. By accessing or using IEX Historical Data,
    you agree to the IEX Historical Data Terms of Use:
    https://www.iexexchange.io/legal/hist-data-terms
"""

import argparse
import sys
from pathlib import Path

import requests

from utils.downloading import (
    create_base_parser,
    load_dotenv,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    resolve_data_dir,
)

# IEX HIST API endpoint (discovered via browser inspection)
# Note: This undocumented endpoint may change. If it fails, download manually from:
# https://iextrading.com/trading/market-data/#hist-download
HIST_API_URL = "https://iextrading.com/api/1.0/hist"

# Required headers to access the API
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Referer": "https://iextrading.com/trading/market-data/",
}


def get_available_dates() -> dict:
    """
    Fetch the catalog of all available HIST files.

    Returns
    -------
    dict mapping date strings to list of available feeds
    """
    response = requests.get(HIST_API_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def find_smallest_file(feed: str = "TOPS", max_size_mb: int = 500) -> dict | None:
    """
    Find the smallest available file for testing.

    Parameters
    ----------
    feed : str
        'TOPS' or 'DEEP'
    max_size_mb : int
        Maximum file size in MB

    Returns
    -------
    dict with date, size_mb, and link, or None if no suitable file found
    """
    catalog = get_available_dates()

    candidates = []
    for date, entries in catalog.items():
        for entry in entries:
            if entry["feed"] == feed:
                size_mb = int(entry["size"]) / 1e6
                if 50 < size_mb < max_size_mb:  # Skip empty/tiny files
                    candidates.append({"date": date, "size_mb": size_mb})

    if not candidates:
        return None

    return min(candidates, key=lambda x: x["size_mb"])


def download_hist_file(
    date: str,
    feed: str = "TOPS",
    data_dir: Path = None,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    """
    Download IEX HIST pcap file for a given date.

    Parameters
    ----------
    date : str
        Date in YYYYMMDD format
    feed : str
        'DEEP' for full depth, 'TOPS' for top-of-book (smaller files)
    data_dir : Path
        Directory to save downloaded files
    force : bool
        Re-download even if file exists
    dry_run : bool
        Show what would be downloaded without downloading

    Returns
    -------
    Path to downloaded file, or None if dry_run
    """
    feed = feed.upper()
    if feed not in ["TOPS", "DEEP"]:
        raise ValueError("feed must be 'TOPS' or 'DEEP'")

    # Fetch catalog to get download link
    catalog = get_available_dates()

    if date not in catalog:
        available = sorted(catalog.keys())
        print(f"ERROR: No data available for {date}")
        print(f"Available date range: {available[0]} to {available[-1]}")
        sys.exit(1)

    # Find matching feed entry
    entries = catalog[date]
    target = next((e for e in entries if e["feed"] == feed), None)

    if not target:
        available = [e["feed"] for e in entries]
        print(f"ERROR: No {feed} feed for {date}. Available: {available}")
        sys.exit(1)

    # Construct local filename
    filename = f"{date}_IEXTP1_{feed}{target['version']}.pcap.gz"
    local_path = data_dir / filename

    size_mb = int(target["size"]) / 1e6

    if local_path.exists() and not force:
        print(f"File already exists: {local_path}")
        print("Use --force to re-download")
        return local_path

    if dry_run:
        print(f"Would download: {filename}")
        print(f"  Size: {size_mb:.0f} MB")
        print(f"  Destination: {local_path}")
        return None

    # Download with progress
    print(f"Downloading {feed} for {date} ({size_mb:.0f} MB)...")

    response = requests.get(target["link"], headers=HEADERS, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded / total_size * 100
                    print(f"\r  Progress: {downloaded / 1e6:.1f}MB ({pct:.0f}%)", end="")

    print(f"\n  Saved to: {local_path}")
    return local_path


def main():
    parser = create_base_parser("Download IEX HIST market data")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dates and file sizes",
    )
    parser.add_argument(
        "--smallest",
        action="store_true",
        help="Download smallest available file (good for testing)",
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        help="Date to download (YYYYMMDD format)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Download DEEP (full depth) instead of TOPS (BBO only)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=500,
        help="Maximum file size in MB for --smallest (default: 500)",
    )
    args = parser.parse_args()

    load_dotenv()

    print_section("IEX HIST DATA DOWNLOAD")
    print()
    print("Source: IEX Exchange (Free)")
    print("URL: https://iextrading.com/trading/market-data/")
    print()
    print("Attribution: Data provided for free by IEX.")
    print("Terms: https://www.iexexchange.io/legal/hist-data-terms")
    print()

    # Determine feed type
    feed = "DEEP" if args.deep else "TOPS"
    print(f"Feed: {feed}")

    if args.dry_run:
        print_dry_run_notice()

    # List available dates
    if args.list:
        print("\nFetching catalog...")
        catalog = get_available_dates()
        dates = sorted(catalog.keys())

        print(f"\nAvailable dates: {len(dates)}")
        print(f"Range: {dates[0]} to {dates[-1]}")
        print()

        # Show recent dates with sizes
        print("Recent files:")
        for date in dates[-10:]:
            entries = catalog[date]
            for entry in entries:
                size_mb = int(entry["size"]) / 1e6
                print(f"  {date} {entry['feed']:4s} v{entry['version']} - {size_mb:,.0f} MB")

        return

    # Find smallest file
    if args.smallest:
        print(f"\nFinding smallest {feed} file (max {args.max_size_mb} MB)...")
        smallest = find_smallest_file(feed, args.max_size_mb)

        if not smallest:
            print(f"ERROR: No {feed} files found under {args.max_size_mb} MB")
            sys.exit(1)

        print(f"Found: {smallest['date']} ({smallest['size_mb']:.0f} MB)")
        args.date = smallest["date"]

    # Download specific date
    if not args.date:
        print("\nERROR: Specify --date, --smallest, or --list")
        parser.print_help()
        sys.exit(1)

    # Resolve data directory
    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "equities" / "market" / "microstructure" / "iex" / feed.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")

    # Download
    result = download_hist_file(
        date=args.date,
        feed=feed,
        data_dir=output_dir,
        force=args.force,
        dry_run=args.dry_run,
    )

    if result and not args.dry_run:
        print_download_summary(
            {
                "date": args.date,
                "feed": feed,
                "file": result.name,
                "size_mb": result.stat().st_size / 1e6,
            }
        )

        print("\nNext steps:")
        print("  1. Parse the pcap file using iex_parser library")
        print("  2. See Chapter 3 notebook: iex_lob_reconstruction")
        print("  3. Use load_iex_hist() from data")


if __name__ == "__main__":
    main()
