#!/usr/bin/env python3
"""
Download Fama-French factor data from Ken French's Data Library.

No API key required - data is freely available from:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Usage:
    uv run python data/factors/ff_download.py               # Download book essentials
    uv run python data/factors/ff_download.py --all         # Download all datasets at all frequencies
    uv run python data/factors/ff_download.py --dataset ff5 --frequency daily
"""

import argparse
import sys
from itertools import product
from pathlib import Path

from utils.downloading import load_dotenv, resolve_data_dir

# Datasets the book + case studies actually consume.
# Kept explicit so a vanilla `python data/factors/ff_download.py` produces
# every file the chapter notebooks and case study analytics expect.
BOOK_ESSENTIALS = [
    ("ff3", "daily"),
    ("ff3", "monthly"),
    ("ff5", "daily"),
    ("ff5", "monthly"),
    ("mom", "daily"),
    ("mom", "monthly"),
]


def main():
    parser = argparse.ArgumentParser(description="Download Fama-French factor data")
    parser.add_argument(
        "--all", action="store_true", help="Download all available datasets (70+ datasets)"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, help="Download specific dataset (e.g., ff3, ff5, mom)"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--frequency",
        "-f",
        type=str,
        default=None,
        choices=["monthly", "daily", "weekly"],
        help="Data frequency. Default: book essentials cover both daily and monthly.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH or repo/data)",
    )
    args = parser.parse_args()

    load_dotenv()

    from ml4t.data.providers.fama_french import FF_CATEGORIES, FamaFrenchProvider

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "factors" / "fama-french"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FAMA-FRENCH FACTOR DATA DOWNLOAD")
    print("=" * 60)
    print()
    print("Source: Ken French Data Library")
    print("URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
    print(f"Output: {output_dir}")
    print()

    # Initialize provider with caching to output directory
    provider = FamaFrenchProvider(cache_path=output_dir, use_cache=True)

    # List datasets
    if args.list:
        print("Available dataset categories:")
        for category, datasets in FF_CATEGORIES.items():
            print(f"\n  {category}:")
            for ds in datasets[:5]:  # Show first 5
                print(f"    - {ds}")
            if len(datasets) > 5:
                print(f"    ... and {len(datasets) - 5} more")
        print()
        print("Use --dataset <name> to download specific dataset")
        print("Use --all to download everything")
        return

    # Build (dataset, frequency) job list.
    if args.dataset:
        frequencies = [args.frequency] if args.frequency else ["daily", "monthly"]
        jobs = [(args.dataset, freq) for freq in frequencies]
    elif args.all:
        all_datasets = [d for ds_list in FF_CATEGORIES.values() for d in ds_list]
        frequencies = [args.frequency] if args.frequency else ["daily", "monthly"]
        jobs = list(product(all_datasets, frequencies))
    else:
        # Default: book essentials. Honour --frequency if user asked for one.
        if args.frequency:
            unique_datasets = list(dict.fromkeys(ds for ds, _ in BOOK_ESSENTIALS))
            jobs = [(d, args.frequency) for d in unique_datasets]
        else:
            jobs = list(BOOK_ESSENTIALS)

    print(f"Downloading {len(jobs)} (dataset, frequency) pairs:")
    print()

    success_count = 0
    for dataset, frequency in jobs:
        print(f"  {dataset} [{frequency}]...", end=" ", flush=True)
        try:
            df = provider.fetch(dataset, frequency=frequency)
            print(f"OK ({len(df)} rows)")
            success_count += 1
        except Exception as e:
            print(f"ERROR ({e})")

    print()
    print(f"Downloaded {success_count}/{len(jobs)} pairs")
    print(f"Data saved to: {output_dir}")
    print()

    # List saved files
    parquet_files = list(output_dir.glob("*.parquet"))
    if parquet_files:
        print(f"Files created ({len(parquet_files)}):")
        for f in sorted(parquet_files)[:10]:
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")
        if len(parquet_files) > 10:
            print(f"  ... and {len(parquet_files) - 10} more")

    if success_count < len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
