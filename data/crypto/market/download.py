#!/usr/bin/env python3

from __future__ import annotations

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
        for symbol in info.get("symbols", []):
            rows.append({"symbol": str(symbol), "group": group, "description": description})

    output_path = storage_path / "crypto_dictionary.parquet"
    pl.DataFrame(rows).sort(["group", "symbol"]).write_parquet(output_path)
    return output_path


def save_partitioned(df: pl.DataFrame, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for symbol in df["symbol"].unique().sort().to_list():
        symbol_path = root / f"symbol={symbol}" / "data.parquet"
        symbol_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(df.filter(pl.col("symbol") == symbol), symbol_path)


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


def get_update_start(output_path: Path, end_date: str, interval_hours: int) -> str | None:
    if not output_path.exists():
        return None

    last_ts = pl.read_parquet(output_path).select(pl.col("timestamp").max()).item()
    if last_ts is None:
        return None

    start_date = (last_ts + timedelta(hours=interval_hours)).date().isoformat()
    return None if start_date > end_date else start_date


def clamp_date_range(df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    return df.filter(
        pl.col("timestamp")
        .dt.date()
        .is_between(pl.lit(start_date).str.to_date(), pl.lit(end_date).str.to_date(), closed="both")
    )


def download_perps(
    provider,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pl.DataFrame, list[str]]:
    # Use parallel download if available (3-10x faster)
    if hasattr(provider, "fetch_ohlcv_multi_parallel"):
        print(f"  Fetching {len(symbols)} symbols in parallel...", flush=True)
        df = provider.fetch_ohlcv_multi_parallel(
            symbols=symbols,
            start=start_date,
            end=end_date,
            frequency="hourly",
            max_concurrent=5,
        )
        if df.is_empty():
            return pl.DataFrame(), symbols
        fetched = set(df["symbol"].unique().to_list())
        failed = [s for s in symbols if s not in fetched]
        print(f"  OK ({len(df):,} rows, {len(fetched)} symbols, {len(failed)} failed)")
        return df, failed

    # Fallback: sequential download
    frames: list[pl.DataFrame] = []
    failed: list[str] = []
    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            df = provider.fetch_ohlcv(symbol, start=start_date, end=end_date, frequency="hourly")
            if df.is_empty():
                print("EMPTY")
                failed.append(symbol)
                continue
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            frames.append(df)
            print(f"OK ({len(df):,} rows)")
        except Exception as exc:
            print(f"ERROR ({exc})")
            failed.append(symbol)

    return (pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame(), failed)


def download_premium(
    provider,
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str,
) -> tuple[pl.DataFrame, list[str]]:
    # Use parallel multi-symbol download (3-10x faster)
    print(f"  Fetching {len(symbols)} symbols in parallel...", flush=True)
    df = provider.fetch_premium_index_multi_parallel(
        symbols=symbols,
        start=start_date,
        end=end_date,
        interval=interval,
        max_concurrent=5,
    )
    if df.is_empty():
        return pl.DataFrame(), symbols

    fetched = set(df["symbol"].unique().to_list())
    failed = [s for s in symbols if s not in fetched]
    print(f"  OK ({len(df):,} rows, {len(fetched)} symbols, {len(failed)} failed)")
    return df, failed


def main() -> None:
    parser = create_base_parser("Download crypto perpetual futures and premium index data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to crypto config",
    )
    parser.add_argument("--symbol", "-s", type=str, help="Download a single symbol")
    parser.add_argument("--premium", action="store_true", help="Download premium index only")
    parser.add_argument("--perps", action="store_true", help="Download perpetual OHLCV only")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Extend the configured end date to today and append new rows",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.dry_run:
        print_dry_run_notice()

    from ml4t.data.providers.binance_public import BinancePublicProvider

    config = load_section(args.config, "crypto")
    data_root = resolve_data_dir(args.data_path)
    storage_path = resolve_storage_path(data_root, config.get("storage_path"), "crypto")

    symbol_groups = config.get("symbols", {})
    symbols = (
        [args.symbol.upper()] if args.symbol else flatten_group_values(symbol_groups, "symbols")
    )
    end_date = date.today().isoformat() if args.update else str(config.get("end", "2025-12-31"))
    download_perps_flag = args.perps or not args.premium
    download_premium_flag = args.premium or not args.perps

    outputs = config.get("outputs", {})
    perps_cfg = config.get("perps", {})
    perps_start = str(perps_cfg.get("start", config.get("start", "2020-01-01")))
    perps_end = date.today().isoformat() if args.update else str(perps_cfg.get("end", end_date))
    premium_start = str(config.get("start", "2020-01-01"))
    premium_end = end_date
    premium_interval = str(config.get("interval", "8h"))
    premium_file = str(outputs.get("premium_file", "premium_index_8h.parquet"))
    perps_template = str(outputs.get("perps_file_template", "perps_{frequency}.parquet"))

    print_section("CRYPTO DATA DOWNLOAD (Binance Public)")
    print(f"Config: {args.config}")
    print(f"Output: {storage_path}")
    print(f"Symbols: {len(symbols)}")
    print(f"Premium window: {premium_start} to {premium_end}")
    print(f"Perps window: {perps_start} to {perps_end}")

    if args.dry_run:
        print_download_summary(
            {
                "symbols": len(symbols),
                "premium": download_premium_flag,
                "perps": download_perps_flag,
                "premium_file": str(storage_path / premium_file),
                "perps_file": str(storage_path / perps_template.format(frequency="1h")),
            },
            dry_run=True,
        )
        return

    storage_path.mkdir(parents=True, exist_ok=True)
    dictionary_path = write_dictionary(storage_path, symbol_groups)
    summary: dict[str, object] = {"dictionary_file": str(dictionary_path)}

    if download_perps_flag:
        perps_output = storage_path / perps_template.format(frequency="1h")
        provider = BinancePublicProvider(market=str(perps_cfg.get("market", "futures")))
        start_date = perps_start
        if args.update and not args.force:
            incremental_start = get_update_start(perps_output, perps_end, interval_hours=1)
            if incremental_start is None:
                print("\nPerpetual OHLCV already up to date.")
                perps_df = (
                    pl.read_parquet(perps_output) if perps_output.exists() else pl.DataFrame()
                )
                perps_failed: list[str] = []
            else:
                start_date = incremental_start
                print(f"\nAppending perpetual OHLCV from {start_date}...")
                new_df, perps_failed = download_perps(provider, symbols, start_date, perps_end)
                perps_df = (
                    combine_existing(perps_output, new_df)
                    if not new_df.is_empty()
                    else pl.read_parquet(perps_output)
                )
        else:
            print("\nDownloading perpetual OHLCV...")
            new_df, perps_failed = download_perps(provider, symbols, start_date, perps_end)
            if new_df.is_empty():
                print("ERROR: no perpetual OHLCV data downloaded")
                sys.exit(1)
            perps_df = (
                combine_existing(perps_output, new_df)
                if perps_output.exists() and not args.force
                else new_df.sort(["symbol", "timestamp"])
            )

        if not perps_df.is_empty():
            perps_df = clamp_date_range(perps_df, perps_start, perps_end)
            if not args.symbol:
                perps_df = perps_df.filter(pl.col("symbol").is_in(symbols))
            atomic_write_parquet(perps_df, perps_output)
            save_partitioned(perps_df, storage_path / "ohlcv_1h")
            profile_path = save_dataset_profile(
                perps_df, perps_output, source="BookCryptoDownloader", timestamp_col="timestamp"
            )
            summary["perps_rows"] = len(perps_df)
            summary["perps_symbols"] = perps_df["symbol"].n_unique()
            summary["perps_failed"] = len(perps_failed)
            summary["perps_output"] = str(perps_output)
            summary["perps_profile"] = str(profile_path)

    if download_premium_flag:
        premium_output = storage_path / premium_file
        provider = BinancePublicProvider(market=str(config.get("market", "futures")))
        start_date = premium_start
        if args.update and not args.force:
            incremental_start = get_update_start(premium_output, premium_end, interval_hours=8)
            if incremental_start is None:
                print("\nPremium index already up to date.")
                premium_df = (
                    pl.read_parquet(premium_output) if premium_output.exists() else pl.DataFrame()
                )
                premium_failed: list[str] = []
            else:
                start_date = incremental_start
                print(f"\nAppending premium index from {start_date}...")
                new_df, premium_failed = download_premium(
                    provider, symbols, start_date, premium_end, premium_interval
                )
                premium_df = (
                    combine_existing(premium_output, new_df)
                    if not new_df.is_empty()
                    else pl.read_parquet(premium_output)
                )
        else:
            print("\nDownloading premium index...")
            new_df, premium_failed = download_premium(
                provider, symbols, start_date, premium_end, premium_interval
            )
            if new_df.is_empty():
                print("ERROR: no premium index data downloaded")
                sys.exit(1)
            premium_df = (
                combine_existing(premium_output, new_df)
                if premium_output.exists() and not args.force
                else new_df.sort(["symbol", "timestamp"])
            )

        if not premium_df.is_empty():
            premium_df = clamp_date_range(premium_df, premium_start, premium_end)
            if not args.symbol:
                premium_df = premium_df.filter(pl.col("symbol").is_in(symbols))
            atomic_write_parquet(premium_df, premium_output)
            save_partitioned(premium_df, storage_path / "premium_index")
            profile_path = save_dataset_profile(
                premium_df, premium_output, source="BookCryptoDownloader", timestamp_col="timestamp"
            )
            summary["premium_rows"] = len(premium_df)
            summary["premium_symbols"] = premium_df["symbol"].n_unique()
            summary["premium_failed"] = len(premium_failed)
            summary["premium_output"] = str(premium_output)
            summary["premium_profile"] = str(profile_path)

    print_download_summary(summary)


if __name__ == "__main__":
    main()
