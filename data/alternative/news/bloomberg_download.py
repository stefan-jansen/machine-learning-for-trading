#!/usr/bin/env python3
"""
Download Bloomberg Financial News dataset from HuggingFace.

446K financial news articles with full article text, headlines, and real timestamps
from Bloomberg (2006-2013). Apache 2.0 license.

Dataset: https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News

Usage:
    python bloomberg_download.py                    # Download full dataset (~446K articles)
    python bloomberg_download.py --dry-run          # Show what would be downloaded

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
        description="Download Bloomberg Financial News dataset from HuggingFace"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH)",
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
    output_dir = data_path / "alternative" / "news" / "bloomberg"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bloomberg_news.parquet"

    print_section("BLOOMBERG FINANCIAL NEWS DOWNLOAD")
    print()
    print("Dataset: Bloomberg Financial News")
    print("Source:  HuggingFace (danidanou/Bloomberg_Financial_News)")
    print("License: Apache 2.0")
    print("Content: ~446K financial news articles with full text")
    print("Period:  2006-2013")
    print()
    print(f"Output:  {output_file}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would download from HuggingFace")
        print("[DRY RUN] No files created")
        return

    if output_file.exists():
        existing = pl.read_parquet(output_file)
        print(f"File already exists: {output_file}")
        print(f"  Rows: {len(existing):,}")
        print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        print("To re-download, delete the existing file first.")
        return

    print("Downloading from HuggingFace...")
    print()

    try:
        from huggingface_hub import hf_hub_download

        # Download the parquet file directly (more reliable than datasets streaming)
        print("Downloading parquet file via huggingface_hub...")
        parquet_path = hf_hub_download(
            repo_id="danidanou/Bloomberg_Financial_News",
            filename="bloomberg_financial_data.parquet.gzip",
            repo_type="dataset",
            local_dir=str(output_dir),
        )
        print(f"Downloaded to: {parquet_path}")

        # Read and normalize
        df = pl.read_parquet(parquet_path)
        print(f"Read {len(df):,} articles")

        # Normalize column names to lowercase
        df = df.rename({col: col.lower() for col in df.columns})

        # Parse date column if string
        if df["date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("date").str.to_datetime().alias("date"))

        # Drop null articles (should be very few)
        n_before = len(df)
        df = df.filter(pl.col("article").is_not_null() & (pl.col("article").str.len_chars() > 10))
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"Dropped {n_dropped:,} articles with null/empty text")

        # Save with atomic write
        atomic_write_parquet(df, output_file)

        print()
        print(f"Saved to: {output_file}")
        print()
        print(f"Total articles: {len(df):,}")
        print(f"Columns: {df.columns}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        # Show sample
        print("\nSample headlines:")
        for row in df.head(3).iter_rows(named=True):
            text = str(row.get("headline", ""))[:80]
            print(f"  - {text}")

    except Exception as e:
        print(f"ERROR: Download failed - {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try again later (HuggingFace may be busy)")
        print("  3. Check https://huggingface.co/datasets/danidanou/Bloomberg_Financial_News")
        sys.exit(1)


if __name__ == "__main__":
    main()
