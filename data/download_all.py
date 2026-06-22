#!/usr/bin/env python3
"""
Download all ML4T datasets.

Usage:
    python download_all.py           # Download core datasets (free)
    python download_all.py --all     # Download everything (incl. paid)
    python download_all.py --update  # Update all datasets to present
    python download_all.py --estimate-only  # Show cost estimates

Datasets:
  FREE (no API key):
    - ETF Universe (Yahoo Finance) - Case study primary
    - Crypto Premium Index (Binance Public) - Case study primary
    - FX Pairs (Yahoo Finance)
    - Fama-French Factors (Ken French Library)
    - AQR Factors (AQR Research)

  FREE (requires free API key):
    - Treasury Yields (FRED)
    - US Equities (NASDAQ Data Link) - FROZEN, ends 2018
    - Yahoo S&P 500 (for survivorship bias demo)

  PAID:
    - CME Futures (Databento - $125 free credit)
    - NASDAQ ITCH (5-6 GB download)

Update Mode:
    The --update flag extends all updateable datasets from their configured
    end date (e.g., 2025-12-31) to the present. Use this in 2026+ to get
    the latest data for strategies.

    Updateable: ETFs, Crypto, Macro, FX, FF Factors, AQR Factors
    Frozen: US Equities (ends 2018), AlgoSeek (licensed snapshots)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.downloading import load_dotenv, resolve_data_dir

# Script name to path mapping (new directory structure)
DOWNLOAD_SCRIPTS = {
    # Asset-class market data
    "etfs.py": "etfs/market/download.py",
    "crypto.py": "crypto/market/download.py",
    "cme_futures.py": "futures/market/download.py",
    "fx_pairs.py": "fx/market/download.py",
    "us_equities.py": "equities/market/us_equities/download.py",
    "mbo_data.py": "equities/market/microstructure/mbo_download.py",
    "nasdaq_itch.py": "equities/market/microstructure/nasdaq_itch_download.py",
    "iex_hist.py": "equities/market/microstructure/iex_download.py",
    # Positioning
    "cot.py": "futures/positioning/cot_download.py",
    "institutional_13f.py": "equities/positioning/13f_download.py",
    "sec_form4.py": "equities/positioning/form4_download.py",
    # Fundamentals (SEC filings + XBRL)
    "sec_filings.py": "equities/fundamentals/filings_download.py",
    "sec_xbrl.py": "equities/fundamentals/xbrl_download.py",
    # Standalone packaged datasets
    "firm_characteristics.py": "equities/firm_characteristics/download.py",
    # Cross-asset macro / factors
    "macro.py": "macro/download.py",
    "ff_factors.py": "factors/ff_download.py",
    "aqr_factors.py": "factors/aqr_download.py",
    # Prediction markets
    "prediction_markets.py": "prediction_markets/download.py",
    # Alternative (cross-asset third-party)
    "fnspid.py": "alternative/news/fnspid_download.py",
    "bloomberg_news.py": "alternative/news/bloomberg_download.py",
    "onchain.py": "crypto/onchain/download.py",
}


def run_download_script(script_name: str, extra_args: list | None = None) -> bool:
    """Run a download script from the appropriate dataset directory."""
    # Map old script name to new path
    relative_path = DOWNLOAD_SCRIPTS.get(script_name, script_name)
    script_path = Path(__file__).parent / relative_path
    if not script_path.exists():
        print(f"  Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode == 0


def download_etfs(data_path: Path, force: bool = False):
    """Download ETF data from Yahoo Finance (free)."""
    print("\n" + "=" * 60)
    print("ETF UNIVERSE (Free - Yahoo Finance)")
    print("=" * 60)

    extra_args = ["--data-path", str(data_path)]
    if force:
        extra_args.append("--force")
    return run_download_script("etfs.py", extra_args)


def download_crypto(data_path: Path, force: bool = False):
    """Download crypto perpetuals and premium index from Binance Public (free)."""
    print("\n" + "=" * 60)
    print("CRYPTO PERPS + PREMIUM (Free - Binance Public)")
    print("=" * 60)

    extra_args = ["--data-path", str(data_path)]
    if force:
        extra_args.append("--force")
    return run_download_script("crypto.py", extra_args)


def download_macro(data_path: Path, _force: bool = False):
    """Download macro indicators from FRED."""
    print("\n" + "=" * 60)
    print("MACRO INDICATORS (FRED - requires free API key)")
    print("=" * 60)

    return run_download_script("macro.py", ["--data-path", str(data_path)])


def download_fx(data_path: Path):
    """Download FX data from OANDA."""
    print("\n" + "=" * 60)
    print("FX PAIRS (OANDA - requires free API key)")
    print("=" * 60)

    return run_download_script("fx_pairs.py", ["--data-path", str(data_path)])


def download_ff_factors(data_path: Path):
    """Download Fama-French factors (free, no API key)."""
    print("\n" + "=" * 60)
    print("FAMA-FRENCH FACTORS (Free - Ken French Library)")
    print("=" * 60)

    return run_download_script("ff_factors.py", ["--data-path", str(data_path)])


def download_aqr_factors(data_path: Path):
    """Download AQR factors (free, no API key)."""
    print("\n" + "=" * 60)
    print("AQR FACTORS (Free - AQR Research)")
    print("=" * 60)

    return run_download_script("aqr_factors.py", ["--data-path", str(data_path)])


def download_firm_characteristics(data_path: Path):
    """Download Chen-Pelger-Zhu firm characteristics (free, from GitHub)."""
    print("\n" + "=" * 60)
    print("FIRM CHARACTERISTICS (Free - GitHub)")
    print("=" * 60)

    return run_download_script(
        "firm_characteristics.py", ["--data-path", str(data_path), "--convert"]
    )


def download_us_equities(data_path: Path):
    """Download US Equities from NASDAQ Data Link (free API key required)."""
    print("\n" + "=" * 60)
    print("US EQUITIES (NASDAQ Data Link - requires free API key)")
    print("=" * 60)

    return run_download_script("us_equities.py", ["--output", str(data_path / "equities")])


def download_yahoo_sp500(data_path: Path):
    """Download Yahoo S&P 500 for survivorship bias demo."""
    print("\n" + "=" * 60)
    print("YAHOO S&P 500 (For survivorship bias demonstration)")
    print("=" * 60)

    return run_download_script("etfs.py", ["--sp500-only", "--data-path", str(data_path)])


def download_futures(data_path: Path, estimate_only: bool = False):
    """Download futures data from Databento (paid) and consolidate."""
    print("\n" + "=" * 60)
    print("CME FUTURES (Databento - requires API key, $125 free credit)")
    print("=" * 60)

    previous_data_path = os.environ.get("ML4T_DATA_PATH")
    os.environ["ML4T_DATA_PATH"] = str(data_path)
    try:
        extra_args = ["--max-cost", "125"]
        if estimate_only:
            extra_args.append("--estimate-only")
        return run_download_script("cme_futures.py", extra_args)
    finally:
        if previous_data_path is None:
            os.environ.pop("ML4T_DATA_PATH", None)
        else:
            os.environ["ML4T_DATA_PATH"] = previous_data_path


def download_prediction_markets(data_path: Path):
    """Download prediction market data from Kalshi + Polymarket (free)."""
    print("\n" + "=" * 60)
    print("PREDICTION MARKETS (Free - Kalshi + Polymarket)")
    print("=" * 60)

    return run_download_script("prediction_markets.py", ["--data-path", str(data_path)])


def download_cot(data_path: Path):
    """Download CFTC Commitment of Traders (free, public, no API key)."""
    print("\n" + "=" * 60)
    print("CFTC COT (Free - Commitment of Traders)")
    print("=" * 60)

    return run_download_script("cot.py", ["--data-path", str(data_path)])


def download_itch(data_path: Path):
    """Download NASDAQ ITCH sample data (5-6 GB)."""
    print("\n" + "=" * 60)
    print("NASDAQ ITCH SAMPLE (5-6 GB - for Chapter 4 microstructure)")
    print("=" * 60)

    return run_download_script("nasdaq_itch.py", ["--data-path", str(data_path)])


def update_datasets(data_path: Path) -> dict:
    """
    Update all updateable datasets to the present date.

    Extends datasets beyond their configured end date (e.g., 2025-12-31).
    Skips frozen datasets (US Equities ends 2018, AlgoSeek is licensed snapshots).

    Returns:
        Dictionary of dataset names to success status
    """
    from datetime import date

    today = date.today().isoformat()
    print("\n" + "=" * 60)
    print(f"UPDATE MODE - Extending datasets to {today}")
    print("=" * 60)
    print("\nUpdateable datasets:")
    print("  - ETF Universe (Yahoo Finance)")
    print("  - Crypto Premium (Binance Public)")
    print("  - Macro/Treasury (FRED)")
    print("  - FX Pairs (OANDA/Yahoo)")
    print("  - Fama-French Factors")
    print("  - AQR Factors")
    print("  - CFTC Commitment of Traders")
    print("\nFrozen datasets (skipped):")
    print("  - US Equities (ends 2018)")
    print("  - AlgoSeek data (licensed snapshots)")
    print()

    results = {}

    # Update ETFs
    print("\n[1/7] Updating ETF Universe...")
    args_list = ["--data-path", str(data_path), "--update"]
    results["ETFs"] = run_download_script("etfs.py", args_list)

    # Update Crypto
    print("\n[2/7] Updating Crypto Premium...")
    args_list = ["--data-path", str(data_path), "--update"]
    results["Crypto"] = run_download_script("crypto.py", args_list)

    # Update Macro (FRED)
    print("\n[3/7] Updating Macro (FRED)...")
    args_list = ["--data-path", str(data_path), "--update"]
    results["Macro"] = run_download_script("macro.py", args_list)

    # Update FX
    print("\n[4/7] Updating FX Pairs...")
    args_list = ["--data-path", str(data_path), "--update"]
    results["FX"] = run_download_script("fx_pairs.py", args_list)

    # Update Fama-French
    print("\n[5/7] Updating Fama-French Factors...")
    args_list = ["--data-path", str(data_path)]
    results["Fama-French"] = run_download_script("ff_factors.py", args_list)

    # Update AQR
    print("\n[6/7] Updating AQR Factors...")
    args_list = ["--data-path", str(data_path)]
    results["AQR"] = run_download_script("aqr_factors.py", args_list)

    # Update CoT (re-fetches through current year)
    print("\n[7/7] Updating CFTC Commitment of Traders...")
    args_list = ["--data-path", str(data_path)]
    results["CFTC COT"] = run_download_script("cot.py", args_list)

    return results


def main():
    parser = argparse.ArgumentParser(description="Download all ML4T datasets")
    parser.add_argument(
        "--all", action="store_true", help="Download all datasets including paid/large ones"
    )
    parser.add_argument(
        "--free-only", action="store_true", help="Only download free datasets (no API keys needed)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update all datasets to present (extend beyond configured end date)",
    )
    parser.add_argument(
        "--estimate-only", action="store_true", help="Show cost estimates for paid datasets"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if data exists"
    )
    # Default to repo's data directory (this script lives in code/data/)
    repo_root = Path(__file__).parent.parent
    parser.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "data",
        help="Data storage location (default: <repo>/data/)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Paths - ML4T_DATA_PATH takes precedence (canonical env var)
    data_path = resolve_data_dir(args.data_path)

    print("=" * 60)
    print("ML4T DATA DOWNLOAD")
    print("=" * 60)
    print(f"Data path: {data_path}")

    # Determine mode
    if args.update:
        mode = "update"
    elif args.all:
        mode = "all"
    elif args.free_only:
        mode = "free-only"
    else:
        mode = "core"
    print(f"Mode:        {mode}")

    # Create data directory
    data_path.mkdir(parents=True, exist_ok=True)

    # Handle update mode separately
    if args.update:
        results = update_datasets(data_path)

        # Summary
        print("\n" + "=" * 60)
        print("UPDATE SUMMARY")
        print("=" * 60)
        for name, success in results.items():
            status = "[OK]" if success else "[FAIL]"
            print(f"  {status} {name}")

        total_success = sum(results.values())
        total = len(results)
        print(f"\nUpdated: {total_success}/{total} datasets")

        if total_success < total:
            print("\nNote: Some datasets may require API keys:")
            print("  FRED_API_KEY   - for Macro data")
            print("  OANDA_API_KEY  - for FX data (optional, uses Yahoo fallback)")
        return

    results = {}

    # === CORE DATASETS (always download) ===
    print("\n" + "=" * 60)
    print("CORE DATASETS (Case Studies)")
    print("=" * 60)
    results["ETFs"] = download_etfs(data_path, args.force)
    results["Crypto"] = download_crypto(data_path, args.force)
    results["Prediction Markets"] = download_prediction_markets(data_path)
    results["CFTC COT"] = download_cot(data_path)

    # === FREE DATASETS (no API key) ===
    print("\n" + "=" * 60)
    print("FACTOR DATA (Free, no API key)")
    print("=" * 60)
    results["Fama-French"] = download_ff_factors(data_path)
    results["AQR"] = download_aqr_factors(data_path)
    results["Firm Characteristics"] = download_firm_characteristics(data_path)

    # === FREE WITH API KEY ===
    if not args.free_only:
        print("\n" + "=" * 60)
        print("DATASETS REQUIRING FREE API KEY")
        print("=" * 60)
        results["Macro (FRED)"] = download_macro(data_path, args.force)
        results["FX"] = download_fx(data_path)

        if args.all:
            results["US Equities"] = download_us_equities(data_path)
            results["Yahoo S&P500"] = download_yahoo_sp500(data_path)

    # === PAID / LARGE DATASETS ===
    if args.all:
        print("\n" + "=" * 60)
        print("PAID / LARGE DATASETS")
        print("=" * 60)
        results["Futures"] = download_futures(data_path, args.estimate_only)

        # ITCH is large (5-6GB) so only download if explicitly requested
        print("\nNote: ITCH sample data (5-6 GB) not included in --all")
        print("      Run separately: python equities/nasdaq_itch_download.py")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    total_success = sum(results.values())
    total = len(results)
    print(f"\nCompleted: {total_success}/{total} datasets")

    if total_success < total:
        print("\nTo fix failures:")
        print("  1. Ensure ml4t-data is installed: pip install ml4t-data")
        print("  2. Set required API keys in .env file (see .env.example)")
        print("  3. Re-run this script")

    print("\nAdditional datasets available:")
    print(
        "  python data/equities/market/us_equities/download.py          # Historical equities (1962-2018)"
    )
    print(
        "  python data/equities/market/microstructure/nasdaq_itch_download.py  # Tick data (5-6 GB)"
    )
    print(
        "  data/equities/market/microstructure/MBO_DOWNLOAD.md          # MBO tick data (Databento, manual)"
    )


if __name__ == "__main__":
    main()
