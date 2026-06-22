"""Shared utilities for ML4T data download scripts.

Provides:
- Standardized argument parsing with --dry-run, --data-path, --force, --verbose
- Standardized path resolution via utils.config
- YAML config helpers for dataset download scripts
- Import checking with helpful errors
- Consistent output formatting
- Atomic file writes
- Download summary reporting
- DataBento cost acknowledgment
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_data_dir(cli_arg: Path | None = None) -> Path:
    """Resolve data directory with standardized precedence.

    Priority:
    1. CLI argument (highest)
    2. ML4T_DATA_PATH environment variable (from utils.config)
    3. <repo>/data (default)
    """
    if cli_arg is not None:
        path = Path(cli_arg).expanduser().resolve()
        print(f"Using data path (CLI): {path}")
        return path

    try:
        from utils.config import ML4T_DATA_PATH

        print(f"Using data path (ML4T_DATA_PATH): {ML4T_DATA_PATH}")
        return ML4T_DATA_PATH
    except (ImportError, FileNotFoundError):
        pass

    env_dir = os.environ.get("ML4T_DATA_PATH")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        print(f"Using data path (env): {path}")
        return path

    default_path = _repo_root / "data"
    print(f"Using data path (default): {default_path}")
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def load_dotenv(env_file: Path | None = None):
    """Load environment variables from .env file."""
    if env_file is None:
        env_file = _repo_root / ".env"

    if not env_file.exists():
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if value and not os.getenv(key):
                os.environ[key] = value


def require_env(var: str, hint: str | None = None) -> str:
    """Get required environment variable or exit with helpful message."""
    value = os.getenv(var)
    if not value:
        print(f"ERROR: {var} not set")
        if hint:
            print(f"       {hint}")
        print(f"       Add to .env file: {var}=your-value")
        sys.exit(1)
    assert value is not None
    return value


def check_import(module: str, install_hint: str):
    """Check if module can be imported, exit with helpful message if not."""
    try:
        __import__(module)
    except ImportError as e:
        print(f"ERROR: {module} not available")
        print(f"       Run: {install_hint}")
        print(f"       ({e})")
        sys.exit(1)


# ---------------------------------------------------------------------------
# YAML config helpers (from book_config)
# ---------------------------------------------------------------------------


def load_section(config_path: str | Path, section: str) -> dict[str, Any]:
    """Load a top-level YAML section from a config file."""
    import yaml

    path = Path(config_path).expanduser()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return raw.get(section, {})


def resolve_storage_path(data_root: Path, configured_path: str | None, fallback: str) -> Path:
    """Resolve storage path relative to the selected ML4T data root."""
    raw_path = configured_path or fallback
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else data_root / path


def flatten_group_values(groups: dict[str, Any], values_key: str) -> list[str]:
    """Flatten grouped config values like symbols or pairs into a unique ordered list."""
    values: list[str] = []
    seen: set[str] = set()

    for group in groups.values():
        if not isinstance(group, dict):
            continue
        for value in group.get(values_key, []):
            if value not in seen:
                values.append(value)
                seen.add(value)

    return values


def save_dataset_profile(
    df,
    data_path: str | Path,
    *,
    source: str,
    timestamp_col: str = "timestamp",
    symbol_col: str | None = "symbol",
) -> Path:
    """Generate and save a dataset profile next to a data file."""
    from ml4t.data.storage.data_profile import generate_profile, get_profile_path, save_profile

    path = Path(data_path)
    profile = generate_profile(
        df, source=source, timestamp_col=timestamp_col, symbol_col=symbol_col
    )
    profile_path = get_profile_path(path)
    save_profile(profile, profile_path)
    return profile_path


# ---------------------------------------------------------------------------
# DataBento
# ---------------------------------------------------------------------------


def patch_databento_symbology():
    """Patch databento 0.72.0 bug where insert_metadata expects 'asset' key."""
    try:
        import databento.common.symbology as sym
    except ImportError:
        return

    if getattr(sym.InstrumentMap.insert_metadata, "_ml4t_patched", False):
        return

    _orig = sym.InstrumentMap.insert_metadata

    def _patched(self, metadata):
        mappings = metadata.mappings
        if mappings:
            for _symbol_in, entries in mappings.items():
                for entry in entries:
                    if "asset" not in entry and "symbol" in entry:
                        entry["asset"] = entry["symbol"]

        class _PatchedMeta:
            """Thin wrapper that returns fixed mappings."""

            def __init__(self, orig, fixed_mappings):
                self._orig = orig
                self._fixed = fixed_mappings

            @property
            def mappings(self):
                return self._fixed

            def __getattr__(self, name):
                return getattr(self._orig, name)

        return _orig(self, _PatchedMeta(metadata, mappings))

    _patched._ml4t_patched = True
    sym.InstrumentMap.insert_metadata = _patched


DATABENTO_WARNING = """
================================================================================
                    DATABENTO API - PAID SERVICE WARNING
================================================================================

This download uses the DataBento API which is a PAID service.

IMPORTANT INFORMATION:
  - DataBento requires registration with a credit card on file
  - As of February 2026, DataBento offers a $125 sign-up credit for new accounts
  - This credit is sufficient to cover the ML4T book datasets:
    * CME Futures (continuous contracts): ~$75
    * MBO Tick Data (3 symbols, 10 days): ~$10-15

WARNINGS:
  - If you have already used your sign-up credit on other downloads,
    this download WILL BE CHARGED to your credit card
  - DataBento may change or remove the sign-up credit at any time
  - Cost estimates are approximate; actual costs may vary
  - YOU are responsible for managing your DataBento account and costs

ESTIMATED COST FOR THIS DOWNLOAD: ${estimated_cost:.2f}

================================================================================
"""


def databento_acknowledge(estimated_cost: float, force: bool = False) -> bool:
    """Display DataBento cost warning and require explicit acknowledgment."""
    print(DATABENTO_WARNING.format(estimated_cost=estimated_cost))

    if force:
        print("--force flag set: Proceeding without interactive confirmation.")
        print("By using --force, you acknowledge the above warnings.")
        return True

    print("To proceed with this download, type exactly: I UNDERSTAND")
    print("To cancel, press Ctrl+C or type anything else.")
    print()

    try:
        response = input("Your response: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nDownload cancelled.")
        return False

    if response == "I UNDERSTAND":
        print("\nAcknowledgment received. Proceeding with download...")
        return True
    else:
        print(f"\nResponse '{response}' does not match 'I UNDERSTAND'.")
        print("Download cancelled for your protection.")
        return False


def databento_estimate_only_notice(estimated_cost: float) -> None:
    """Print cost estimate without prompting for download."""
    print("\n" + "=" * 70)
    print("DATABENTO COST ESTIMATE (No download - estimate only)")
    print("=" * 70)
    print(f"\n  Estimated cost: ${estimated_cost:.2f}")
    print()
    print("  Note: As of Feb 2026, new DataBento accounts receive $125 credit.")
    print("        If your credit is exhausted, this amount will be charged.")
    print()
    print("  To proceed with download, run without --estimate-only flag.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_section(title: str, char: str = "=", width: int = 60):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(title)
    print(char * width)


def atomic_write_parquet(df, path: Path):
    """Write Polars DataFrame to parquet with atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = path.parent / f".{path.name}.tmp"

    try:
        df.write_parquet(tmp_file)
        tmp_file.replace(path)
    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()
        raise e


def get_repo_root() -> Path:
    """Get repository root directory."""
    return _repo_root


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with standard ML4T download flags."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data storage location (default: $ML4T_DATA_PATH or repo/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser


def print_download_summary(stats: dict, dry_run: bool = False) -> None:
    """Print standardized download summary."""
    prefix = "[DRY RUN] " if dry_run else ""
    print_section(f"{prefix}SUMMARY")
    for key, value in stats.items():
        display_key = key.replace("_", " ").title()
        if isinstance(value, int) and value > 1000:
            print(f"  {display_key}: {value:,}")
        elif isinstance(value, float):
            print(f"  {display_key}: {value:.2f}")
        else:
            print(f"  {display_key}: {value}")


def print_dry_run_notice() -> None:
    """Print notice that this is a dry run."""
    print("\n" + "=" * 60)
    print("DRY RUN - No data will be downloaded")
    print("Remove --dry-run to actually download")
    print("=" * 60 + "\n")
