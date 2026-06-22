#!/usr/bin/env python3
"""
Download AQR factor data from AQR Research Library.

No API key required - data is freely available from:
https://www.aqr.com/Insights/Datasets

Usage:
    python scripts/download_aqr_factors.py           # Download core datasets
    python scripts/download_aqr_factors.py --all     # Download all datasets
    python scripts/download_aqr_factors.py --dataset qmj_factors  # Download specific
"""

import argparse
import sys
from pathlib import Path

from utils.downloading import load_dotenv, resolve_data_dir


def main():
    parser = argparse.ArgumentParser(description="Download AQR factor data")
    parser.add_argument(
        "--all", action="store_true", help="Download all available datasets (16 datasets)"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Download specific dataset (e.g., qmj_factors, bab_factors)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH or repo/data)",
    )
    args = parser.parse_args()

    load_dotenv()

    import openpyxl  # noqa: F401  # required for Excel parsing
    from ml4t.data.providers.aqr import AQR_CATEGORIES, AQRFactorProvider

    data_path = resolve_data_dir(args.data_path)
    output_dir = data_path / "factors" / "aqr"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AQR FACTOR DATA DOWNLOAD")
    print("=" * 60)
    print()
    print("Source: AQR Capital Management Research Library")
    print("URL: https://www.aqr.com/Insights/Datasets")
    print(f"Output: {output_dir}")
    print()

    # List datasets
    if args.list:
        print("Available dataset categories:")
        for category, datasets in AQR_CATEGORIES.items():
            print(f"\n  {category}:")
            for ds in datasets:
                print(f"    - {ds}")
        print()
        print("Use --dataset <name> to download specific dataset")
        print("Use --all to download everything")
        return

    # Determine which datasets to download
    if args.dataset:
        datasets = [args.dataset]
    elif args.all:
        # All datasets from all categories
        datasets = []
        for ds_list in AQR_CATEGORIES.values():
            datasets.extend(ds_list)
    else:
        # Default: download first dataset from each category
        datasets = [ds_list[0] for ds_list in AQR_CATEGORIES.values()]

    print(f"Downloading {len(datasets)} datasets to {output_dir}:")
    print()

    try:
        # Use the classmethod download
        output_path = AQRFactorProvider.download(
            output_path=output_dir,
            datasets=datasets,
            include_optional=args.all,
        )
        print()
        print(f"Success! Data saved to: {output_path}")

    except Exception as e:
        print(f"\nERROR: Download failed - {e}")
        print()
        print("Trying individual downloads...")
        print()

        # Fallback: try individual downloads
        provider = AQRFactorProvider(data_path=output_dir)

        success_count = 0
        for dataset in datasets:
            print(f"  {dataset}...", end=" ", flush=True)
            try:
                df = provider.fetch(dataset)
                # Save to parquet
                output_file = output_dir / f"{dataset}.parquet"
                df.write_parquet(output_file)
                print(f"OK ({len(df)} rows)")
                success_count += 1
            except Exception as e:
                print(f"ERROR ({e})")

        print()
        print(f"Downloaded {success_count}/{len(datasets)} datasets")

    # Verify files were created
    parquet_files = list(output_dir.glob("*.parquet"))
    if parquet_files:
        print()
        print(f"Files created ({len(parquet_files)}):")
        for f in sorted(parquet_files):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")
    else:
        print()
        print("=" * 60)
        print("DOWNLOAD FAILED: No parquet files were created!")
        print("=" * 60)
        print()
        print("The download completed but no data was saved.")
        print("This usually means the Excel files could not be parsed.")
        print()
        print("Check that openpyxl is installed: pip install openpyxl")
        print()
        print("For help, see: data/README.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
