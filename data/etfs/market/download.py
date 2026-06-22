#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

from utils.downloading import (
    create_base_parser,
    load_dotenv,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    resolve_data_dir,
    resolve_storage_path,
)


def write_dictionary(manager) -> Path:
    rows: list[dict[str, str]] = []
    for group, info in manager.config.tickers.items():
        description = info.get("description", "")
        for symbol in info.get("symbols", []):
            rows.append({"symbol": symbol, "group": group, "description": description})

    dictionary = pl.DataFrame(rows).sort(["group", "symbol"])
    output_path = manager.config.storage_path / "etf_universe_dictionary.parquet"
    dictionary.write_parquet(output_path)
    return output_path


def main() -> None:
    parser = create_base_parser("Download ETF data from Yahoo Finance")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to ETF config",
    )
    parser.add_argument("--symbol", "-s", type=str, help="Download a single ETF symbol")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Extend the configured end date to today and append new data",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        print_dry_run_notice()

    from ml4t.data.etfs import ETFDataManager

    data_root = resolve_data_dir(args.data_path)
    manager = ETFDataManager.from_config(args.config)
    manager.config.storage_path = resolve_storage_path(
        data_root, str(manager.config.storage_path), "etfs"
    )

    if args.symbol:
        manager.config.tickers = {
            "adhoc": {"description": "Ad hoc symbol", "symbols": [args.symbol]}
        }

    if args.update:
        manager.config.end = date.today().isoformat()

    output_path = manager.config.storage_path / "etf_universe.parquet"

    print_section("ETF DATA DOWNLOAD (Yahoo Finance)")
    print(f"Config: {args.config}")
    print(f"Output: {output_path}")
    print(f"Date range: {manager.config.start} to {manager.config.end}")
    print(f"Symbols: {len(manager.config.get_all_symbols())}")

    if args.dry_run:
        print_download_summary(
            {
                "symbols": len(manager.config.get_all_symbols()),
                "start_date": manager.config.start,
                "end_date": manager.config.end,
                "output_file": str(output_path),
            },
            dry_run=True,
        )
        return

    manager.config.storage_path.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        stats = manager.update()
        action = "updated"
    else:
        stats = manager.download_all(force=args.force)
        action = "downloaded"

    dictionary_path = write_dictionary(manager)
    metadata_path = manager.config.storage_path / "etf_universe_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["dictionary_file"] = str(dictionary_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    df = pl.read_parquet(output_path)
    print_download_summary(
        {
            "action": action,
            "rows": len(df),
            "symbols": df["symbol"].n_unique(),
            "partitions": len(stats),
            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            "output_file": str(output_path),
            "dictionary_file": str(dictionary_path),
        }
    )


if __name__ == "__main__":
    main()
