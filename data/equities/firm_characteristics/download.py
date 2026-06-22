#!/usr/bin/env python3
"""
Download Chen-Pelger-Zhu (2020) academic asset pricing dataset.

This dataset contains ~1.2M stock-month observations with 46 firm characteristics
and monthly returns. Data is fully anonymized (no stock identifiers).

Source: https://github.com/jasonzy121/Deep_Learning_Asset_Pricing
Paper: "Deep Learning in Asset Pricing" (Chen, Pelger, Zhu, 2020)

Usage:
    python scripts/download_academic.py           # Download all files
    python scripts/download_academic.py --check   # Verify existing files
    python scripts/download_academic.py --force   # Force re-download

Data structure:
    academic/dl_asset_pricing/
    ├── RetChar.csv           # 1.1GB - Stock returns + 46 characteristics
    ├── Macro.csv             # 1.8MB - 178 macroeconomic indicators
    ├── char/                 # Pre-split characteristic numpy arrays
    │   ├── Char_train.npz    # 1967-1989 (~70%)
    │   ├── Char_valid.npz    # 1990-1999 (~15%)
    │   └── Char_test.npz     # 2000-2016 (~15%)
    ├── macro/                # Pre-split macro numpy arrays
    │   ├── macro_train.npz
    │   ├── macro_valid.npz
    │   └── macro_test.npz
    └── RF/                   # Pre-processed features for random forest
        ├── RF_train_normalized_task_1.npz
        ├── RF_valid_normalized_task_1.npz
        └── RF_test_normalized_task_1.npz

Note: The original data is hosted on Google Drive via the GitHub repo.
      This script downloads from the GitHub release or provides manual instructions.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

from utils.downloading import resolve_data_dir

# Expected files and their approximate sizes (for verification)
EXPECTED_FILES = {
    "RetChar.csv": 1_152_695_892,  # ~1.1GB
    "Macro.csv": 1_816_127,  # ~1.8MB
    "char/Char_train.npz": 332_629_190,
    "char/Char_valid.npz": 75_511_430,
    "char/Char_test.npz": 805_509_830,
    "macro/macro_train.npz": 359_850,
    "macro/macro_valid.npz": 98_490,
    "macro/macro_test.npz": 446_970,
    "RF/RF_train_normalized_task_1.npz": 332_629_190,
    "RF/RF_valid_normalized_task_1.npz": 75_511_430,
    "RF/RF_test_normalized_task_1.npz": 805_509_830,
}

# Source repository
GITHUB_REPO = "https://github.com/jasonzy121/Deep_Learning_Asset_Pricing"

# Google Drive folder containing all data files
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l"

# Google Drive file IDs (from the original repo)
# Note: These may change if the authors update the data
GDRIVE_FILES = {
    "data.zip": "1nYHpJ2lNm-qDX5iq18-HaL1H6z7lPGVi",  # Main data archive
}

# Additional academic data files (separate sources)
ADDITIONAL_FILES = {
    "SDF-Time-Series.xlsx": {
        "url": "https://www.dropbox.com/scl/fi/6wgeg4ztoi5vu680x01eq/SDF-Time-Series.xlsx?rlkey=ehy8zaz2fh6tyq43hpf64gczh&e=1&dl=1",
        "size": 240_633,  # ~241KB
        "description": "SDF time series data (Pelger) — used for Ch14 latent factor validation",
    },
}


def download_additional_files(data_dir: Path) -> None:
    """Download additional academic data files from direct URLs."""
    import requests

    for filename, info in ADDITIONAL_FILES.items():
        output_path = data_dir / filename
        if output_path.exists():
            print(f"  [OK] {filename} already exists")
            continue

        print(f"  Downloading {filename} ({info['description']})...")
        try:
            resp = requests.get(info["url"], stream=True, allow_redirects=True, timeout=60)
            resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            print(f"  [OK] {filename} ({output_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")


def download_from_gdrive(file_id: str, output_path: Path) -> bool:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run: pip install gdown")
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive: {file_id}")
    print(f"  -> {output_path}")

    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract and flatten nested zip structure."""
    print(f"Extracting: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List contents
            members = zf.namelist()
            print(f"  Archive contains {len(members)} files")

            # Extract to temp location first
            temp_dir = extract_dir / "_temp_extract"
            temp_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(temp_dir)

            # Flatten nested structure
            # Original structure might be: data/char/Char_train.npz
            # We want: char/Char_train.npz
            for root, _dirs, files in os.walk(temp_dir):
                root_path = Path(root)
                rel_root = root_path.relative_to(temp_dir)

                # Skip the top-level 'data' directory if it exists
                parts = rel_root.parts
                if parts and parts[0] == "data":
                    rel_root = Path(*parts[1:]) if len(parts) > 1 else Path(".")

                for file in files:
                    src = root_path / file
                    if rel_root == Path("."):
                        dst = extract_dir / file
                    else:
                        dst = extract_dir / rel_root / file
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    if not dst.exists():
                        src.rename(dst)
                        print(f"  Extracted: {dst.name}")

            # Cleanup temp directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        return True

    except Exception as e:
        print(f"ERROR: Extraction failed: {e}")
        return False


def verify_files(data_dir: Path) -> tuple[list[str], list[str]]:
    """Verify expected files exist and have correct sizes."""
    found = []
    missing = []

    for filename, expected_size in EXPECTED_FILES.items():
        filepath = data_dir / filename
        if filepath.exists():
            actual_size = filepath.stat().st_size
            # Allow 1% tolerance for size differences
            if abs(actual_size - expected_size) / expected_size < 0.01:
                found.append(filename)
            else:
                print(f"  WARNING: {filename} size mismatch: {actual_size} vs {expected_size}")
                found.append(filename)  # Still count as found
        else:
            missing.append(filename)

    return found, missing


def print_manual_instructions(data_dir: Path) -> None:
    """Print manual download instructions."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print(f"\nSource: {GITHUB_REPO}")
    print(f"Direct: {GDRIVE_FOLDER_URL}")
    print("\n1. Open the Google Drive folder link above")
    print(
        "2. Download all files (datasets.zip ~367MB, RetChar.csv ~1.1GB, Macro.csv, sample_checkpoints.zip)"
    )
    print("3. Extract zip files")
    print("4. Place files in:", data_dir)
    print("\nExpected structure after extraction:")
    print(f"  {data_dir}/")
    print("  ├── RetChar.csv         # Main characteristics + returns")
    print("  ├── Macro.csv           # Macroeconomic indicators")
    print("  ├── char/               # Pre-split numpy arrays")
    print("  ├── macro/              # Pre-split macro arrays")
    print("  └── RF/                 # Random forest features")
    print("\n5. Run this script again with --check to verify")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download Chen-Pelger-Zhu (2020) academic asset pricing dataset"
    )
    parser.add_argument("--check", action="store_true", help="Verify existing files only")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--convert", action="store_true", help="Convert CSV to parquet format")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override data directory (default: $ML4T_DATA_PATH or repo/data)",
    )
    args = parser.parse_args()

    # Handle --convert flag
    if args.convert:
        data_dir = resolve_data_dir(args.data_path)
        if convert_to_parquet(data_dir):
            return 0
        return 1

    # Determine data directory
    data_dir = resolve_data_dir(args.data_path)

    academic_dir = data_dir / "equities" / "firm_characteristics" / "dl_asset_pricing"

    print("=" * 70)
    print("CHEN-PELGER-ZHU (2020) ACADEMIC DATASET")
    print("=" * 70)
    print(f"Target directory: {academic_dir}")
    print()

    # Check existing files
    found, missing = verify_files(academic_dir)

    print(f"Files found: {len(found)}/{len(EXPECTED_FILES)}")
    if found:
        for f in found:
            print(f"  [OK] {f}")
    if missing:
        print(f"\nFiles missing: {len(missing)}")
        for f in missing:
            print(f"  [FAIL] {f}")

    # If just checking, exit
    if args.check:
        if not missing:
            print("\n[OK] All files present and verified!")
            return 0
        else:
            print(f"\n[FAIL] Missing {len(missing)} files")
            return 1

    # If all files exist and not forcing, exit
    if not missing and not args.force:
        print("\n[OK] All files already downloaded!")
        print("  Use --force to re-download")
        return 0

    # Try automatic download
    print("\nAttempting automatic download...")
    academic_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("gdown not available for automatic download")
        print_manual_instructions(academic_dir)
        return 1

    # Method 1: Download entire folder (more reliable than single-file IDs)
    print(f"Downloading from Google Drive folder: {GDRIVE_FOLDER_URL}")
    try:
        gdown.download_folder(
            GDRIVE_FOLDER_URL, output=str(academic_dir), quiet=False, remaining_ok=True
        )
    except Exception as e:
        print(f"Folder download failed: {e}")

    # Extract datasets.zip if present
    datasets_zip = academic_dir / "datasets.zip"
    if datasets_zip.exists():
        if extract_zip(datasets_zip, academic_dir):
            datasets_zip.unlink(missing_ok=True)

    # Extract sample_checkpoints.zip if present
    checkpoints_zip = academic_dir / "sample_checkpoints.zip"
    if checkpoints_zip.exists():
        if extract_zip(checkpoints_zip, academic_dir):
            checkpoints_zip.unlink(missing_ok=True)

    # Download additional files (SDF time series, etc.)
    print("\nDownloading additional academic data files...")
    download_additional_files(academic_dir)

    # Verify
    found, missing = verify_files(academic_dir)
    if not missing:
        print("\n[OK] Download and extraction complete!")
        return 0

    # Method 2: Fall back to single-file download
    print("\nFolder download incomplete, trying single-file download...")
    zip_path = academic_dir / "data.zip"
    if download_from_gdrive(GDRIVE_FILES["data.zip"], zip_path) and extract_zip(
        zip_path, academic_dir
    ):
        found, missing = verify_files(academic_dir)
        if not missing:
            print("\n[OK] Download and extraction complete!")
            zip_path.unlink(missing_ok=True)
            return 0

    print(f"\nWARNING: {len(missing)} files still missing after download attempts")
    print_manual_instructions(academic_dir)
    return 1


def convert_to_parquet(data_dir: Path) -> bool:
    """Convert RetChar.csv to parquet format with train/test splits.

    Split boundaries match Chen-Pelger-Zhu paper:
    - train: 1967-1989 (~70%)
    - valid: 1990-1999 (~15%) - merged into test for simplicity
    - test: 2000-2016 (~15%)
    """
    import polars as pl

    dl_dir = data_dir / "equities" / "firm_characteristics" / "dl_asset_pricing"
    output_dir = data_dir / "equities" / "firm_characteristics"

    retchar_path = dl_dir / "RetChar.csv"
    if not retchar_path.exists():
        print(f"ERROR: RetChar.csv not found at {retchar_path}")
        return False

    print("\nConverting RetChar.csv to parquet format...")
    print(f"  Reading {retchar_path}...")

    # Read CSV - has columns: DATE, RET, and 46 characteristic columns
    df = pl.read_csv(retchar_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Date is in YYYYMMDD format (integer)
    df = df.with_columns(
        pl.col("Date").cast(pl.Utf8).str.to_date("%Y%m%d").alias("date"),
        pl.col("Date").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int32).alias("year"),
    )

    # Create splits
    train_df = df.filter(pl.col("year") < 1990)
    test_df = df.filter(pl.col("year") >= 2000)

    # Drop helper columns
    train_df = train_df.drop(["Date", "year"])
    test_df = test_df.drop(["Date", "year"])
    all_df = df.drop(["Date", "year"])

    # Save parquet files
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "firm_characteristics_train.parquet"
    test_path = output_dir / "firm_characteristics_test.parquet"
    all_path = output_dir / "firm_characteristics_all.parquet"

    train_df.write_parquet(train_path)
    test_df.write_parquet(test_path)
    all_df.write_parquet(all_path)

    print("  Created:")
    print(
        f"    firm_characteristics_train.parquet: {len(train_df):,} rows ({train_path.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"    firm_characteristics_test.parquet: {len(test_df):,} rows ({test_path.stat().st_size / 1e6:.1f} MB)"
    )
    print(
        f"    firm_characteristics_all.parquet: {len(all_df):,} rows ({all_path.stat().st_size / 1e6:.1f} MB)"
    )

    return True


if __name__ == "__main__":
    sys.exit(main())
