#!/usr/bin/env python3
"""
Download FNSPID (Financial News and Stock Price Integration Dataset) from HuggingFace.

FNSPID contains 15.7M financial news records with stock tickers and dates for
4,775 S&P 500 companies (1999-2023), enabling text-to-market signal research.

Dataset: https://huggingface.co/datasets/Zihan1004/FNSPID
GitHub:  https://github.com/Zdong104/FNSPID_Financial_News_Dataset

Usage:
    python fnspid_download.py                    # Download 1M sample (default)
    python fnspid_download.py --sample 500000    # Sample 500K articles
    python fnspid_download.py --sample 0         # Full dataset (~15.7M articles)
    python fnspid_download.py --dry-run          # Show what would be downloaded

Requirements:
    - datasets library: pip install datasets
    - Internet connection for HuggingFace download
"""

import argparse
import sys
from pathlib import Path

# Add data/ to path for imports
from utils.downloading import atomic_write_parquet, print_section, resolve_data_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download FNSPID financial news dataset from HuggingFace"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1_000_000,
        help="Sample N articles (default: 1,000,000; use 0 for full dataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without doing it",
    )
    args = parser.parse_args()

    import polars as pl
    from datasets import load_dataset

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "alternative" / "news" / "fnspid"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Name file based on sample size
    if args.sample == 0:
        output_file = output_dir / "fnspid_full.parquet"
    else:
        output_file = output_dir / f"fnspid_{args.sample // 1000}k.parquet"

    print_section("FNSPID DOWNLOAD")
    print()
    print("Dataset: Financial News and Stock Price Integration Dataset")
    print("Source:  HuggingFace (Zihan1004/FNSPID)")
    print("Content: 15.7M financial news records, 4,775 S&P 500 companies")
    print("Period:  1999-2023")
    print()
    print(f"Output:  {output_file}")
    if args.sample > 0:
        print(f"Sample:  {args.sample:,} articles (seed={args.seed})")
    else:
        print("Sample:  Full dataset (~15.7M articles)")
    print()

    if args.dry_run:
        print("[DRY RUN] Would download from HuggingFace")
        print("[DRY RUN] No files created")
        return

    print("Downloading from HuggingFace (this may take several minutes)...")
    print("Using streaming mode for reliability...")
    print()

    try:
        # Use streaming mode - more reliable for this dataset
        ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

        # Collect records up to sample size (or all if sample=0)
        records = []
        limit = args.sample if args.sample > 0 else float("inf")

        for i, record in enumerate(ds):
            if i >= limit:
                break
            records.append(record)
            if i % 100_000 == 0 and i > 0:
                print(f"  Loaded {i:,} records...")

        print(f"Downloaded {len(records):,} articles")

        # Convert to Polars
        df = pl.DataFrame(records)

        # Save with atomic write
        atomic_write_parquet(df, output_file)

        print()
        print(f"Success! Data saved to: {output_file}")
        print()

        # Print stats
        print(f"Total articles: {len(df):,}")
        print(f"Columns: {df.columns}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

        # Show sample
        print("\nSample headlines:")
        for row in df.head(3).iter_rows(named=True):
            # Find text column
            text_col = next(
                (
                    c
                    for c in df.columns
                    if "headline" in c.lower() or "title" in c.lower() or "text" in c.lower()
                ),
                df.columns[0],
            )
            text = str(row.get(text_col, ""))[:80]
            print(f"  - {text}...")

    except Exception as e:
        print(f"ERROR: Download failed - {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try again later (HuggingFace may be busy)")
        print("  3. Check https://huggingface.co/datasets/Zihan1004/FNSPID")
        sys.exit(1)


if __name__ == "__main__":
    main()
