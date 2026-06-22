#!/usr/bin/env python3
"""
Download Market-By-Order (MBO) tick data from DataBento (API alternative).

The recommended path for readers is the manual Download Center workflow
described in MBO_DOWNLOAD.md (same directory). This script is kept as an
alternative for users who already have a DATABENTO_API_KEY and prefer a
programmatic download.

Source: DataBento API (XNAS.ITCH dataset)
Target: $ML4T_DATA_PATH/equities/market/microstructure/market_by_order/{SYMBOL}/{DATE}.parquet

MBO data provides individual order messages (add, cancel, fill, modify, trade)
from the NASDAQ ITCH feed. This is the most granular market data available
and is used for order book reconstruction and microstructure analysis.

Coverage shipped with the book: NVDA for 10 trading days (2024-11-04 to
2024-11-15). These are also the script defaults.

Usage:
    # ALWAYS estimate cost first
    uv run python data/equities/market/microstructure/mbo_download.py --estimate-only

    # Download defaults (NVDA, 10 trading days ending 2024-11-15)
    uv run python data/equities/market/microstructure/mbo_download.py

    # Override symbols / dates
    uv run python data/equities/market/microstructure/mbo_download.py --symbols NVDA --dates 20241104 20241105
    uv run python data/equities/market/microstructure/mbo_download.py --symbols NVDA --start 2024-11-04 --end 2024-11-15

    # List existing local data
    uv run python data/equities/market/microstructure/mbo_download.py --status

Cost:
    MBO data costs approximately $0.45-0.50 per symbol per day.
    The default 10-day / 1-symbol slice costs about $5, well under the
    Databento $125 free-credit limit for new accounts.

Requires: DATABENTO_API_KEY environment variable.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from utils.downloading import (
    databento_acknowledge,
    databento_estimate_only_notice,
    load_dotenv,
    patch_databento_symbology,
    print_section,
    resolve_data_dir,
)

# Default configuration — only NVDA is used in Ch3 notebooks
DEFAULT_SYMBOLS = ["NVDA"]
DEFAULT_DAYS = 10  # Number of trading days to download


def get_trading_dates(start_date: str, end_date: str) -> list[str]:
    """Generate list of trading dates (weekdays) between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        # Skip weekends
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def estimate_cost(symbols: list[str], dates: list[str]) -> dict:
    """Estimate download cost."""
    # MBO data costs approximately $0.45-0.50 per symbol-day
    cost_per_symbol_day = 0.50
    total_symbol_days = len(symbols) * len(dates)
    estimated_cost = total_symbol_days * cost_per_symbol_day

    return {
        "symbols": symbols,
        "dates": dates,
        "num_symbols": len(symbols),
        "num_days": len(dates),
        "total_symbol_days": total_symbol_days,
        "estimated_cost_usd": estimated_cost,
        "cost_per_symbol_day_usd": cost_per_symbol_day,
    }


def download_mbo_data(
    symbols: list[str],
    dates: list[str],
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Download MBO data from DataBento.

    Returns:
        Dictionary with download results
    """
    import databento as db

    # Fix databento 0.72.0 symbology bug (entry["asset"] → entry["symbol"])
    patch_databento_symbology()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY environment variable not set")
        print("       Get API key at: https://databento.com")
        sys.exit(1)

    client = db.Historical(api_key)

    results = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "total_rows": 0,
        "files": [],
        "errors": [],
    }

    for symbol in symbols:
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        for date in dates:
            output_file = symbol_dir / f"{date}.parquet"

            if output_file.exists() and not force:
                print(f"  Skipping {symbol}/{date} (exists)")
                results["skipped"] += 1
                continue

            try:
                print(f"  Downloading {symbol}/{date}...", flush=True)

                # Convert date format for API
                date_str = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

                # Request data
                data = client.timeseries.get_range(
                    dataset="XNAS.ITCH",
                    schema="mbo",
                    symbols=[symbol],
                    start=f"{date_str}T00:00:00",
                    end=f"{date_str}T23:59:59",
                )

                # Save directly to parquet via databento
                data.to_parquet(output_file)

                import polars as pl

                n_records = pl.scan_parquet(output_file).select(pl.len()).collect().item()

                if n_records > 0:
                    results["downloaded"] += 1
                    results["total_rows"] += n_records
                    results["files"].append(str(output_file))
                    print(f"    Saved {n_records:,} records to {output_file.name}")
                else:
                    output_file.unlink(missing_ok=True)
                    print(f"    No data for {symbol}/{date}")
                    results["skipped"] += 1

            except Exception as e:
                print(f"    ERROR: {e}")
                results["failed"] += 1
                results["errors"].append(f"{symbol}/{date}: {e}")

    return results


def show_status(output_dir: Path) -> None:
    """Show status of existing MBO data."""
    if not output_dir.exists():
        print("No MBO data downloaded yet.")
        print(f"Expected location: {output_dir}")
        return

    import polars as pl

    symbols = [d.name for d in output_dir.iterdir() if d.is_dir()]
    if not symbols:
        print("No MBO data downloaded yet.")
        return

    print(f"MBO data directory: {output_dir}")
    print()

    for symbol in sorted(symbols):
        symbol_dir = output_dir / symbol
        files = sorted(symbol_dir.glob("*.parquet"))

        if not files:
            continue

        # Get total rows and date range
        total_rows = 0
        dates = []
        for f in files:
            try:
                df = pl.read_parquet(f)
                total_rows += len(df)
                dates.append(f.stem)
            except Exception:
                pass

        if dates:
            print(f"{symbol}:")
            print(f"  Files: {len(files)}")
            print(f"  Rows: {total_rows:,}")
            print(f"  Dates: {min(dates)} to {max(dates)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download DataBento MBO tick data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # ALWAYS estimate cost first
    uv run python data/equities/market/microstructure/mbo_download.py --estimate-only

    # Download defaults (NVDA, 10 trading days ending 2024-11-15)
    uv run python data/equities/market/microstructure/mbo_download.py

    # Override symbols or dates
    uv run python data/equities/market/microstructure/mbo_download.py --symbols NVDA --dates 20241104 20241105
    uv run python data/equities/market/microstructure/mbo_download.py --symbols NVDA --start 2024-11-04 --end 2024-11-15

    # Show existing local data
    uv run python data/equities/market/microstructure/mbo_download.py --status

Cost:
    MBO data costs approximately $0.50 per symbol per day.
    Default 10-day NVDA slice is about $5. The manual workflow in
    MBO_DOWNLOAD.md is usually easier for this one-off slice.
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to download (default: {DEFAULT_SYMBOLS})",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="Specific dates to download (YYYYMMDD format)",
    )
    parser.add_argument(
        "--start",
        help="Start date for date range (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end",
        help="End date for date range (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of trading days to download if no dates specified (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Show cost estimate without downloading",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of existing data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override data directory (default: $ML4T_DATA_PATH)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Resolve paths
    data_dir = resolve_data_dir(args.data_path)
    output_dir = data_dir / "equities" / "market" / "microstructure" / "market_by_order"

    # Handle --status
    if args.status:
        print_section("MBO DATA STATUS")
        show_status(output_dir)
        return

    # Determine dates
    if args.dates:
        dates = args.dates
    elif args.start and args.end:
        dates = get_trading_dates(args.start, args.end)
    else:
        # Default: last N trading days from a recent date
        # Using November 2024 as default
        end = datetime(2024, 11, 15)
        start = end - timedelta(days=args.days * 2)  # Account for weekends
        dates = get_trading_dates(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        dates = dates[-args.days :]  # Take last N days

    symbols = args.symbols

    print_section("DATABENTO MBO DATA DOWNLOAD")
    print(f"Data directory: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Symbols: {symbols}")
    print(f"Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print()

    # Cost estimate
    estimate = estimate_cost(symbols, dates)
    estimated_cost = estimate["estimated_cost_usd"]

    print("Cost Estimate:")
    print(f"  Symbols: {estimate['num_symbols']}")
    print(f"  Days: {estimate['num_days']}")
    print(f"  Total symbol-days: {estimate['total_symbol_days']}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print()

    # If --estimate-only, show detailed notice and exit
    if args.estimate_only:
        databento_estimate_only_notice(estimated_cost)
        return

    # Require explicit acknowledgment before paid download
    if not databento_acknowledge(estimated_cost, force=args.force):
        print("Download cancelled.")
        return

    # Download
    print_section("DOWNLOADING")
    results = download_mbo_data(symbols, dates, output_dir, force=args.force)

    # Summary
    print_section("SUMMARY")
    print(f"Downloaded: {results['downloaded']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"Total rows: {results['total_rows']:,}")

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"][:5]:
            print(f"  {error}")

    print(f"\nData stored in: {output_dir}")


if __name__ == "__main__":
    main()
