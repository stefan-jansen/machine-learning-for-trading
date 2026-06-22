#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from utils.downloading import (
    atomic_write_parquet,
    create_base_parser,
    flatten_group_values,
    load_dotenv,
    load_section,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    resolve_data_dir,
    resolve_storage_path,
    save_dataset_profile,
)


def write_dictionary(storage_path: Path, groups: dict[str, dict[str, object]]) -> Path:
    rows: list[dict[str, str]] = []
    for group, info in groups.items():
        description = str(info.get("description", ""))
        for pair in info.get("pairs", []):
            normalized = f"{pair[:3]}_{pair[3:]}"
            rows.append({"pair": normalized, "group": group, "description": description})

    output_path = storage_path / "fx_dictionary.parquet"
    pl.DataFrame(rows).sort(["group", "pair"]).write_parquet(output_path)
    return output_path


def save_partitioned(df: pl.DataFrame, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for symbol in df["symbol"].unique().sort().to_list():
        output = root / f"symbol={symbol}" / "data.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(df.filter(pl.col("symbol") == symbol), output)


def combine_existing(output_path: Path, new_df: pl.DataFrame) -> pl.DataFrame:
    if not output_path.exists():
        return new_df.sort(["symbol", "timestamp"])

    existing = pl.read_parquet(output_path).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms"))
    )
    incoming = new_df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    return (
        pl.concat([existing, incoming], how="vertical_relaxed")
        .unique(subset=["symbol", "timestamp"], keep="last", maintain_order=True)
        .sort(["symbol", "timestamp"])
    )


def get_update_start(output_path: Path, end_date: str, frequency: str) -> str | None:
    if not output_path.exists():
        return None

    last_ts = pl.read_parquet(output_path).select(pl.col("timestamp").max()).item()
    if last_ts is None:
        return None

    delta = timedelta(days=1) if frequency == "daily" else timedelta(hours=4)
    start_date = (last_ts + delta).date().isoformat()
    return None if start_date > end_date else start_date


def clamp_date_range(df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    return df.filter(
        pl.col("timestamp")
        .dt.date()
        .is_between(pl.lit(start_date).str.to_date(), pl.lit(end_date).str.to_date(), closed="both")
    )


def main() -> None:
    parser = create_base_parser("Download FX data from OANDA")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to FX config",
    )
    parser.add_argument("--pair", "-p", type=str, help="Download a single pair (e.g. EURUSD)")
    parser.add_argument(
        "--frequency",
        "-f",
        type=str,
        default="4h",
        choices=["daily", "4h"],
        help="Data frequency",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Extend the configured end date to today and append new data",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        print_dry_run_notice()

    api_key = os.getenv("OANDA_API_KEY")
    if not api_key:
        print("ERROR: OANDA_API_KEY not set")
        sys.exit(1)

    from ml4t.data.providers.oanda import OandaProvider

    config = load_section(args.config, "fx")
    data_root = resolve_data_dir(args.data_path)
    storage_path = resolve_storage_path(data_root, config.get("storage_path"), "fx")
    outputs = config.get("outputs", {})
    file_template = str(outputs.get("file_template", "{frequency}.parquet"))
    partition_template = str(outputs.get("partition_dir_template", "ohlcv_{frequency}"))
    start_date = str(config.get("start", "2011-01-01"))
    end_date = date.today().isoformat() if args.update else str(config.get("end", "2025-12-31"))

    groups = config.get("pairs", {})
    raw_pairs = [args.pair.upper()] if args.pair else flatten_group_values(groups, "pairs")
    pairs = [pair if "_" in pair else f"{pair[:3]}_{pair[3:]}" for pair in raw_pairs]
    output_path = storage_path / file_template.format(frequency=args.frequency)

    print_section("FX DATA DOWNLOAD (OANDA)")
    print(f"Config: {args.config}")
    print(f"Output: {output_path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Pairs: {len(pairs)}")

    if args.dry_run:
        print_download_summary(
            {
                "pairs": len(pairs),
                "frequency": args.frequency,
                "start_date": start_date,
                "end_date": end_date,
                "output_file": str(output_path),
            },
            dry_run=True,
        )
        return

    storage_path.mkdir(parents=True, exist_ok=True)
    dictionary_path = write_dictionary(storage_path, groups)
    provider = OandaProvider(api_key=api_key)

    request_start = start_date
    if args.update and not args.force:
        incremental_start = get_update_start(output_path, end_date, args.frequency)
        if incremental_start is None:
            df = pl.read_parquet(output_path) if output_path.exists() else pl.DataFrame()
            failed: list[str] = []
        else:
            request_start = incremental_start
            print(f"\nAppending {args.frequency} bars from {request_start}...")
            frames: list[pl.DataFrame] = []
            failed = []
            for pair in pairs:
                print(f"  {pair}...", end=" ", flush=True)
                try:
                    frame = provider.fetch_ohlcv(pair, request_start, end_date, args.frequency)
                    if "symbol" not in frame.columns:
                        frame = frame.with_columns(pl.lit(pair).alias("symbol"))
                    else:
                        frame = frame.with_columns(pl.lit(pair).alias("symbol"))
                    frames.append(frame)
                    print(f"OK ({len(frame):,} rows)")
                except Exception as exc:
                    print(f"ERROR ({exc})")
                    failed.append(pair)
            new_df = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
            df = (
                combine_existing(output_path, new_df)
                if not new_df.is_empty()
                else pl.read_parquet(output_path)
            )
    else:
        print(f"\nDownloading {args.frequency} bars...")
        frames = []
        failed = []
        for pair in pairs:
            print(f"  {pair}...", end=" ", flush=True)
            try:
                frame = provider.fetch_ohlcv(pair, start_date, end_date, args.frequency)
                frame = frame.with_columns(pl.lit(pair).alias("symbol"))
                frames.append(frame)
                print(f"OK ({len(frame):,} rows)")
            except Exception as exc:
                print(f"ERROR ({exc})")
                failed.append(pair)

        if not frames:
            print("ERROR: no FX data downloaded")
            sys.exit(1)

        new_df = pl.concat(frames, how="vertical_relaxed")
        df = (
            combine_existing(output_path, new_df)
            if output_path.exists() and not args.force
            else new_df
        )

    if df.is_empty():
        print("ERROR: no FX data available")
        sys.exit(1)

    df = clamp_date_range(df, start_date, end_date)
    atomic_write_parquet(df, output_path)
    save_partitioned(df, storage_path / partition_template.format(frequency=args.frequency))
    profile_path = save_dataset_profile(df, output_path, source="BookFXDownloader")

    print_download_summary(
        {
            "rows": len(df),
            "pairs": df["symbol"].n_unique(),
            "failed_pairs": len(failed),
            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            "output_file": str(output_path),
            "profile_file": str(profile_path),
            "dictionary_file": str(dictionary_path),
        }
    )


if __name__ == "__main__":
    main()
