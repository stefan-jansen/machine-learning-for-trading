#!/usr/bin/env python3

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from utils.downloading import (
    atomic_write_parquet,
    create_base_parser,
    load_dotenv,
    load_section,
    print_download_summary,
    print_dry_run_notice,
    print_section,
    require_env,
    resolve_data_dir,
    resolve_storage_path,
    save_dataset_profile,
)


def collect_series(config: dict[str, object]) -> dict[str, dict[str, str]]:
    descriptions = config.get("descriptions", {})
    series_cfg = config.get("series", {})
    series_map: dict[str, dict[str, str]] = {}
    for frequency, info in series_cfg.items():
        for symbol in info.get("symbols", []):
            series_map[str(symbol)] = {
                "frequency": str(frequency),
                "description": str(descriptions.get(symbol, "")),
                "group": str(frequency),
            }
    return series_map


def build_metadata(
    config: dict[str, object], series_map: dict[str, dict[str, str]]
) -> pl.DataFrame:
    rows = [
        {
            "series": symbol.lower(),
            "source_id": symbol,
            "native_frequency": meta["frequency"],
            "group": meta["group"],
            "description": meta["description"],
            "kind": "observed",
            "formula": None,
        }
        for symbol, meta in sorted(series_map.items())
    ]
    for derived in config.get("derived", []):
        rows.append(
            {
                "series": str(derived["name"]),
                "source_id": None,
                "native_frequency": "derived_daily",
                "group": "derived",
                "description": str(derived.get("description", "")),
                "kind": "derived",
                "formula": str(derived.get("formula", "")),
            }
        )
    return pl.DataFrame(rows)


def combine_aligned(existing_path: Path, new_df: pl.DataFrame) -> pl.DataFrame:
    if not existing_path.exists():
        return new_df.sort("date")

    existing = pl.read_parquet(existing_path)
    combined = (
        pl.concat([existing, new_df], how="diagonal_relaxed")
        .unique(subset=["date"], keep="last", maintain_order=True)
        .sort("date")
    )
    for col in combined.columns:
        if col != "date":
            combined = combined.with_columns(pl.col(col).forward_fill())
    return combined


def combine_raw(existing_path: Path, new_df: pl.DataFrame) -> pl.DataFrame:
    if not existing_path.exists():
        return new_df.sort(["series", "date"])

    existing = pl.read_parquet(existing_path)
    return (
        pl.concat([existing, new_df], how="vertical_relaxed")
        .unique(subset=["series", "date"], keep="last", maintain_order=True)
        .sort(["series", "date"])
    )


def compute_derived(df: pl.DataFrame, derived_defs: list[dict[str, str]]) -> pl.DataFrame:
    result = df
    for series_def in derived_defs:
        name = str(series_def.get("name", ""))
        formula = str(series_def.get("formula", ""))
        if not name or not formula or "-" not in formula:
            continue
        left, right = [part.strip().lower() for part in formula.split("-", 1)]
        if left in result.columns and right in result.columns:
            result = result.with_columns((pl.col(left) - pl.col(right)).alias(name))
    return result


def main() -> None:
    parser = create_base_parser("Download macro data from FRED")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to macro config",
    )
    parser.add_argument("--series", "-s", type=str, help="Download a single FRED series only")
    parser.add_argument("--list", action="store_true", help="List configured series")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Extend the configured end date to today and append new rows",
    )
    args = parser.parse_args()

    load_dotenv()

    config = load_section(args.config, "macro")
    series_map = collect_series(config)

    if args.list:
        print("Configured macro series:")
        for symbol, meta in series_map.items():
            print(f"  {symbol:10s} ({meta['frequency']:9s}) {meta['description']}")
        return

    if args.dry_run:
        print_dry_run_notice()

    api_key = require_env(
        "FRED_API_KEY", "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
    )

    from ml4t.data.providers.fred import FREDProvider

    data_root = resolve_data_dir(args.data_path)
    storage_path = resolve_storage_path(data_root, config.get("storage_path"), "macro")
    outputs = config.get("outputs", {})
    aligned_path = storage_path / str(outputs.get("aligned_file", "fred_macro.parquet"))
    raw_path = storage_path / str(outputs.get("raw_file", "fred_macro_raw.parquet"))
    metadata_path = storage_path / str(outputs.get("metadata_file", "fred_macro_metadata.parquet"))
    start_date = str(config.get("start", "2000-01-01"))
    end_date = date.today().isoformat() if args.update else str(config.get("end", "2025-12-31"))

    if args.series:
        series_id = args.series.upper()
        if series_id not in series_map:
            print(f"ERROR: unknown series {series_id}")
            sys.exit(1)
        requested = {series_id: series_map[series_id]}
    else:
        requested = series_map

    print_section("MACRO DATA DOWNLOAD (FRED)")
    print(f"Config: {args.config}")
    print(f"Output: {aligned_path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Series: {len(requested)}")

    if args.dry_run:
        print_download_summary(
            {
                "series": len(requested),
                "start_date": start_date,
                "end_date": end_date,
                "output_file": str(aligned_path),
                "raw_file": str(raw_path),
                "metadata_file": str(metadata_path),
            },
            dry_run=True,
        )
        return

    storage_path.mkdir(parents=True, exist_ok=True)
    provider = FREDProvider(api_key=api_key)
    request_start = start_date
    if args.update and aligned_path.exists():
        last_date = pl.read_parquet(aligned_path).select(pl.col("date").max()).item()
        if last_date is not None:
            request_start = (last_date + timedelta(days=1)).isoformat()

    daily_frames: list[pl.DataFrame] = []
    raw_frames: list[pl.DataFrame] = []
    failed: list[str] = []

    print("\nDownloading configured series...\n")
    for series_id, meta in requested.items():
        frequency = meta["frequency"]
        description = meta["description"]
        print(f"  {series_id} ({frequency})...", end=" ", flush=True)
        try:
            frame = provider.fetch_ohlcv(
                series_id,
                start=request_start,
                end=end_date,
                frequency=frequency,
            )
            if frame.is_empty():
                print("EMPTY")
                failed.append(series_id)
                continue

            series_col = series_id.lower()
            series_frame = frame.select(
                [
                    pl.col("timestamp").cast(pl.Date).alias("date"),
                    pl.col("close").alias(series_col),
                ]
            )
            daily_frames.append(series_frame)
            raw_frames.append(
                series_frame.rename({series_col: "value"}).with_columns(
                    pl.lit(series_col).alias("series")
                )
            )
            print(f"OK ({len(frame):,} rows) [{description}]")
        except Exception as exc:
            print(f"ERROR ({exc})")
            failed.append(series_id)

    provider.close()

    if not daily_frames:
        print("ERROR: no macro series downloaded")
        sys.exit(1)

    dates = pl.date_range(
        datetime.strptime(request_start, "%Y-%m-%d"),
        datetime.strptime(end_date, "%Y-%m-%d"),
        eager=True,
    )
    aligned = pl.DataFrame({"date": dates})
    for series_frame in daily_frames:
        series_name = next(col for col in series_frame.columns if col != "date")
        aligned = aligned.join(series_frame, on="date", how="left")
        aligned = aligned.with_columns(pl.col(series_name).forward_fill())

    aligned = compute_derived(aligned, list(config.get("derived", [])))
    raw_df = pl.concat(raw_frames, how="vertical_relaxed").sort(["series", "date"])

    aligned = (
        combine_aligned(aligned_path, aligned) if args.update and aligned_path.exists() else aligned
    )
    raw_df = combine_raw(raw_path, raw_df) if args.update and raw_path.exists() else raw_df
    metadata_df = build_metadata(config, series_map)

    atomic_write_parquet(aligned, aligned_path)
    atomic_write_parquet(raw_df, raw_path)
    atomic_write_parquet(metadata_df, metadata_path)
    profile_path = save_dataset_profile(
        aligned, aligned_path, source="BookMacroDownloader", timestamp_col="date", symbol_col=None
    )

    print_download_summary(
        {
            "rows": len(aligned),
            "columns": len(aligned.columns),
            "failed_series": len(failed),
            "date_range": f"{aligned['date'].min()} to {aligned['date'].max()}",
            "output_file": str(aligned_path),
            "raw_file": str(raw_path),
            "metadata_file": str(metadata_path),
            "profile_file": str(profile_path),
        }
    )


if __name__ == "__main__":
    main()
