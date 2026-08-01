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


def missing_symbols(
    df: pl.DataFrame, symbols: list[str], failed: list[str], *, updating: bool
) -> list[str]:
    """Which requested symbols this run did not deliver.

    On a plain run the answer comes from the merged dataset, not from the symbols
    that failed in this request: combine_existing() folds a retry into what a
    previous run already wrote, so a symbol that fails on the retry is still
    present, and reporting the request's failures would fail a download that is in
    fact complete.

    --update inverts that. Every symbol is already on disk by construction, so
    presence proves nothing about whether the window was extended - there the
    request's failures are exactly what did not arrive.
    """
    present = set() if df.is_empty() else set(df["symbol"].unique().to_list())
    absent = {s for s in symbols if s not in present}
    if updating:
        absent |= {s for s in failed if s in symbols}
    return sorted(absent)


def _empty_response_note(output_path: Path, force: bool) -> str:
    """What happens next when nothing arrived."""
    if force:
        return "--force replaces rather than merges, so there is nothing to fall back to."
    if not output_path.exists():
        return "Nothing has been downloaded before either."
    return "Falling back to what is already on disk."


def get_update_start(
    output_path: Path,
    end_date: str,
    interval_hours: int,
    symbols: list[str] | None = None,
    configured_start: str | None = None,
) -> str | None:
    """Where an incremental update has to start so no symbol is left short.

    The *earliest* per-symbol last timestamp, not the dataset-wide maximum. Those
    differ exactly when an earlier update was partial: the symbols that succeeded
    carry the dataset maximum, so starting there would step over the gap left by
    the ones that failed and never fill it, while the run reported success. The
    cost of starting earlier is refetching rows for symbols that are already
    current, which combine_existing() dedups on (symbol, timestamp).

    A symbol missing from the file entirely has no history at all, not a recent
    gap, so the window has to reopen at *configured_start* — otherwise the symbol
    arrives with only the tail, and its presence then reads as success.
    """
    if not output_path.exists():
        return None

    per_symbol = pl.read_parquet(output_path).group_by("symbol").agg(pl.col("timestamp").max())
    if per_symbol.is_empty():
        return None

    if symbols and configured_start:
        present = set(per_symbol["symbol"].to_list())
        if any(s not in present for s in symbols):
            return None if configured_start > end_date else configured_start

    last_ts = per_symbol["timestamp"].min()
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even if symbols are missing (keeps what arrived; re-run to retry them)",
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
    # What is absent from the merged dataset, which is what the exit status is
    # about. Not the same as the failed lists below, which are per-request.
    perps_missing: list[str] = []
    premium_missing: list[str] = []
    perps_failed: list[str] = []
    premium_failed: list[str] = []

    if download_perps_flag:
        perps_output = storage_path / perps_template.format(frequency="1h")
        provider = BinancePublicProvider(market=str(perps_cfg.get("market", "futures")))
        start_date = perps_start
        if args.update and not args.force:
            incremental_start = get_update_start(perps_output, perps_end, 1, symbols, perps_start)
            if incremental_start is None:
                print("\nPerpetual OHLCV already up to date.")
                perps_df = (
                    pl.read_parquet(perps_output) if perps_output.exists() else pl.DataFrame()
                )
                perps_failed = []
            else:
                start_date = incremental_start
                print(f"\nAppending perpetual OHLCV from {start_date}...")
                new_df, perps_failed = download_perps(provider, symbols, start_date, perps_end)
                perps_df = (
                    combine_existing(perps_output, new_df)
                    if not new_df.is_empty()
                    else pl.read_parquet(perps_output)
                )  # the update branch only runs with an existing file
        else:
            print("\nDownloading perpetual OHLCV...")
            new_df, perps_failed = download_perps(provider, symbols, start_date, perps_end)
            if new_df.is_empty():
                # Not necessarily a failure: a fully rate-limited retry against a
                # complete dataset arrives here. Fall through to the merged-dataset
                # check rather than exiting, so the status reflects what is on disk
                # and --allow-partial still applies. The provider returns a frame
                # with no columns at all, so it can be neither combined nor sorted.
                # --force means replace, not merge, so there is nothing to fall
                # back to: a forced refresh that returned nothing has failed, and
                # keeping the old rows would report stale data as the result.
                print(
                    f"No perpetual OHLCV rows returned. {_empty_response_note(perps_output, args.force)}"
                )
                perps_df = (
                    pl.read_parquet(perps_output)
                    if perps_output.exists() and not args.force
                    else new_df
                )
            else:
                perps_df = (
                    combine_existing(perps_output, new_df)
                    if perps_output.exists() and not args.force
                    else new_df.sort(["symbol", "timestamp"])
                )

        if not perps_df.is_empty():
            perps_df = clamp_date_range(perps_df, perps_start, perps_end)
            if not args.symbol:
                perps_df = perps_df.filter(pl.col("symbol").is_in(symbols))
            perps_missing = missing_symbols(perps_df, symbols, perps_failed, updating=args.update)
            atomic_write_parquet(perps_df, perps_output)
            save_partitioned(perps_df, storage_path / "ohlcv_1h")
            profile_path = save_dataset_profile(
                perps_df, perps_output, source="BookCryptoDownloader", timestamp_col="timestamp"
            )
            summary["perps_rows"] = len(perps_df)
            summary["perps_symbols"] = perps_df["symbol"].n_unique()
            summary["perps_missing"] = len(perps_missing)
            summary["perps_output"] = str(perps_output)
            summary["perps_profile"] = str(profile_path)
        else:
            perps_missing = list(symbols)

    if download_premium_flag:
        premium_output = storage_path / premium_file
        provider = BinancePublicProvider(market=str(config.get("market", "futures")))
        start_date = premium_start
        if args.update and not args.force:
            incremental_start = get_update_start(
                premium_output, premium_end, 8, symbols, premium_start
            )
            if incremental_start is None:
                print("\nPremium index already up to date.")
                premium_df = (
                    pl.read_parquet(premium_output) if premium_output.exists() else pl.DataFrame()
                )
                premium_failed = []
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
                print(
                    f"No premium index rows returned. {_empty_response_note(premium_output, args.force)}"
                )
                premium_df = (
                    pl.read_parquet(premium_output)
                    if premium_output.exists() and not args.force
                    else new_df
                )
            else:
                premium_df = (
                    combine_existing(premium_output, new_df)
                    if premium_output.exists() and not args.force
                    else new_df.sort(["symbol", "timestamp"])
                )

        if not premium_df.is_empty():
            premium_df = clamp_date_range(premium_df, premium_start, premium_end)
            if not args.symbol:
                premium_df = premium_df.filter(pl.col("symbol").is_in(symbols))
            premium_missing = missing_symbols(
                premium_df, symbols, premium_failed, updating=args.update
            )
            atomic_write_parquet(premium_df, premium_output)
            save_partitioned(premium_df, storage_path / "premium_index")
            profile_path = save_dataset_profile(
                premium_df, premium_output, source="BookCryptoDownloader", timestamp_col="timestamp"
            )
            summary["premium_rows"] = len(premium_df)
            summary["premium_symbols"] = premium_df["symbol"].n_unique()
            summary["premium_missing"] = len(premium_missing)
            summary["premium_output"] = str(premium_output)
            summary["premium_profile"] = str(profile_path)
        else:
            premium_missing = list(symbols)

    print_download_summary(summary)

    # Exit non-zero when a requested symbol is absent from what is now on disk.
    # Binance rate-limits this download — roughly 700 calls over 10-15 minutes —
    # so coming back short is the expected failure, and exiting 0 makes it
    # indistinguishable from a complete download to download_all.py, to CI, and to
    # a reader who checks the status rather than reading the summary.
    if perps_missing or premium_missing:
        for label, missing in (
            ("perpetual OHLCV", perps_missing),
            ("premium index", premium_missing),
        ):
            if missing:
                print(f"\n{len(missing)} {label} symbol(s) missing: {', '.join(sorted(missing))}")
        # Re-running fetches every configured symbol again and merges the result
        # into what is already on disk, so a symbol that failed once can arrive on
        # the second run. It is a retry, not a resume; --update is the incremental
        # path, and it extends the window rather than filling gaps in it.
        if args.allow_partial:
            print("\nPartial download accepted (--allow-partial). Re-run to try the rest.")
        else:
            print("\nPartial download. Re-run to try the missing symbols again,")
            print("or pass --allow-partial to accept what arrived.")
            sys.exit(1)


if __name__ == "__main__":
    main()
