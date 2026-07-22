#!/usr/bin/env python3
"""
Download Chen-Pelger-Zhu (2020) academic asset pricing dataset.

This dataset contains ~1.2M stock-month observations with 46 firm characteristics
and monthly returns. The released tensors retain persistent anonymous firm axes
within each published train, validation, and test block.

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
    │   ├── Char_train.npz    # 1967-1986
    │   ├── Char_valid.npz    # 1987-1991
    │   └── Char_test.npz     # 1992-2016
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
        "description": "SDF time series data (Pelger) - used for Ch14 latent factor validation",
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

            # Flatten nested structure. The published archives wrap their
            # contents in a single top-level directory (e.g. datasets/char/...
            # or data/char/...); we want char/... directly under extract_dir.
            for root, _dirs, files in os.walk(temp_dir):
                root_path = Path(root)
                rel_root = root_path.relative_to(temp_dir)
                parts = rel_root.parts

                # Skip macOS archive cruft (__MACOSX/... resource forks)
                if parts and parts[0] == "__MACOSX":
                    continue

                # Strip a redundant top-level wrapper directory
                if parts and parts[0] in ("data", "datasets"):
                    rel_root = Path(*parts[1:]) if len(parts) > 1 else Path(".")

                for file in files:
                    if file == ".DS_Store":
                        continue
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


def _parquet_has_persistent_symbols(path: Path) -> bool:
    """Return whether an existing parquet satisfies the identity-preserving schema."""
    if not path.exists():
        return False
    import polars as pl

    try:
        names = set(pl.scan_parquet(path).collect_schema().names())
    except Exception:
        return False
    return {"symbol", "timestamp", "split"}.issubset(names)


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

    # If all source files exist and not forcing, skip download but ensure the
    # parquet outputs exist (convert only if missing; the CSV read is ~1.1 GB).
    if not missing and not args.force:
        print("\n[OK] All source files already downloaded!")
        print("  Use --force to re-download")
        all_parquet = (
            data_dir / "equities" / "firm_characteristics" / "firm_characteristics_all.parquet"
        )
        if _parquet_has_persistent_symbols(all_parquet):
            return 0
        if all_parquet.exists():
            print("  Existing parquet predates persistent-symbol recovery; regenerating.")
        return 0 if convert_to_parquet(data_dir) else 1

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
    print("  ~1.5 GB across 4 files (RetChar.csv ~1.1 GB); per-file progress below.")
    try:
        # NOTE: no remaining_ok kwarg; it was removed in gdown 6.x and passing it
        # raises TypeError, which silently aborts the (working) folder download.
        gdown.download_folder(GDRIVE_FOLDER_URL, output=str(academic_dir), quiet=False)
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

    # Verify, then convert RetChar.csv -> parquet splits
    found, missing = verify_files(academic_dir)
    if not missing:
        print("\n[OK] Download and extraction complete!")
        return 0 if convert_to_parquet(data_dir) else 1

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
            return 0 if convert_to_parquet(data_dir) else 1

    print(f"\nWARNING: {len(missing)} files still missing after download attempts")
    print_manual_instructions(academic_dir)
    return 1


def _characteristic_frame(npz_path: Path, split: str, symbol_offset: int):
    """Flatten one published tensor while preserving its anonymous firm axis."""
    import numpy as np
    import polars as pl

    with np.load(npz_path) as archive:
        dates = archive["date"]
        variables = [str(name) for name in archive["variable"]]
        data = archive["data"]

    if data.ndim != 3 or data.shape[0] != len(dates) or data.shape[2] != len(variables):
        raise ValueError(f"Unexpected characteristic tensor shape in {npz_path}: {data.shape}")
    if not variables or variables[0] != "ret":
        raise ValueError(f"Expected 'ret' as the first variable in {npz_path}: {variables[:3]}")

    valid = data[:, :, 0] != -99.99
    date_index, firm_index = np.nonzero(valid)
    values = data[valid]
    frame = pl.DataFrame(values, schema=variables, orient="row")
    return frame.with_columns(
        pl.Series("symbol", symbol_offset + firm_index, dtype=pl.UInt32),
        pl.Series("timestamp", dates[date_index].astype(str)).str.to_date("%Y%m%d"),
        pl.lit(split).alias("split"),
    ).select("symbol", "timestamp", *variables, "split")


def convert_to_parquet(data_dir: Path) -> bool:
    """Convert the published tensors to canonical, identity-preserving Parquet files.

    The CSV omits firm identifiers, but each NPZ block has a fixed anonymous firm
    axis. Axis positions are persistent within a block. The archive publishes no
    mapping between blocks, so offsets keep their identifier namespaces disjoint.
    """
    import polars as pl

    dl_dir = data_dir / "equities" / "firm_characteristics" / "dl_asset_pricing"
    char_dir = dl_dir / "char"
    output_dir = data_dir / "equities" / "firm_characteristics"
    split_specs = (
        ("train", char_dir / "Char_train.npz", 0),
        ("valid", char_dir / "Char_valid.npz", 1_000_000),
        ("test", char_dir / "Char_test.npz", 2_000_000),
    )

    missing = [path for _, path, _ in split_specs if not path.exists()]
    if missing:
        print("ERROR: Required characteristic tensors are missing:")
        for path in missing:
            print(f"  {path}")
        return False

    print("\nConverting identity-preserving characteristic tensors to parquet...")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: list[Path] = []
    split_counts: dict[str, int] = {}
    for split, npz_path, symbol_offset in split_specs:
        print(f"  Reading {npz_path}...")
        frame = _characteristic_frame(npz_path, split, symbol_offset)
        path = output_dir / f"firm_characteristics_{split}.parquet"
        frame.write_parquet(path)
        split_paths.append(path)
        split_counts[split] = len(frame)
        print(f"    {split}: {len(frame):,} rows ({path.stat().st_size / 1e6:.1f} MB)")

    all_path = output_dir / "firm_characteristics_all.parquet"
    pl.concat([pl.scan_parquet(path) for path in split_paths]).sink_parquet(all_path)
    all_count = sum(split_counts.values())
    print(f"    all: {all_count:,} rows ({all_path.stat().st_size / 1e6:.1f} MB)")
    return True


if __name__ == "__main__":
    sys.exit(main())
