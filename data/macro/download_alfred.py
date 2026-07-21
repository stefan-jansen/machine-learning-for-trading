#!/usr/bin/env python3
"""Materialize ALFRED initial-release market-state observations."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

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
)

FRED_API = "https://api.stlouisfed.org/fred"
USER_AGENT = (
    "ml4t-alfred-initial-release/1.0 "
    "(https://github.com/stefan-jansen/machine-learning-for-trading)"
)


def _json_request(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{FRED_API}/{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def fetch_vintage_dates(series_id: str, api_key: str) -> list[str]:
    """Return every ALFRED vintage date for a series."""
    dates: list[str] = []
    offset = 0
    while True:
        payload = _json_request(
            "series/vintagedates",
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": 10000,
                "offset": offset,
                "sort_order": "asc",
            },
        )
        page = payload.get("vintage_dates", [])
        dates.extend(page)
        offset += len(page)
        if not page or offset >= int(payload.get("count", len(dates))):
            return dates


def fetch_initial_release_series(series_id: str, api_key: str) -> pl.DataFrame:
    """Fetch the first value ALFRED recorded for each observation date."""
    vintage_dates = fetch_vintage_dates(series_id, api_key)
    if not vintage_dates:
        raise RuntimeError(f"ALFRED returned no vintage dates for {series_id}")

    rows: list[dict[str, Any]] = []
    for start in range(0, len(vintage_dates), 1500):
        chunk = vintage_dates[start : start + 1500]
        print(
            f"    vintages {chunk[0]} through {chunk[-1]}",
            flush=True,
        )
        offset = 0
        while True:
            payload = _json_request(
                "series/observations",
                {
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "realtime_start": chunk[0],
                    "realtime_end": chunk[-1],
                    "output_type": 4,
                    "limit": 100000,
                    "offset": offset,
                    "sort_order": "asc",
                },
            )
            page = payload.get("observations", [])
            rows.extend(
                {
                    "series": series_id.lower(),
                    "timestamp": row["date"],
                    "vintage_date": row["realtime_start"],
                    "value": float(row["value"]),
                }
                for row in page
                if row.get("value") not in {None, "."}
            )
            offset += len(page)
            if not page or offset >= int(payload.get("count", offset)):
                break

    if not rows:
        raise RuntimeError(f"ALFRED returned no initial-release observations for {series_id}")
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("timestamp").str.to_date(),
            pl.col("vintage_date").str.to_date(),
        )
        .unique(subset=["series", "timestamp"], keep="first", maintain_order=True)
        .sort(["series", "timestamp"])
    )


def build_aligned_panel(
    raw: pl.DataFrame,
    series_ids: list[str],
    derived: list[dict[str, str]],
) -> pl.DataFrame:
    """Build a complete daily panel without backfilling before source coverage."""
    source_columns = [series_id.lower() for series_id in series_ids]
    wide = raw.pivot(
        on="series",
        index="timestamp",
        values="value",
        aggregate_function="first",
    ).sort("timestamp")
    missing = [name for name in source_columns if name not in wide.columns]
    if missing:
        raise RuntimeError(f"Initial-release panel is missing source series: {missing}")

    calendar = pl.DataFrame(
        {
            "timestamp": pl.date_range(
                wide["timestamp"].min(),
                wide["timestamp"].max(),
                interval="1d",
                eager=True,
            )
        }
    )
    panel = (
        calendar.join(wide.select(["timestamp", *source_columns]), on="timestamp", how="left")
        .with_columns(pl.col(source_columns).forward_fill())
        .drop_nulls(source_columns)
    )
    for definition in derived:
        name = str(definition["name"])
        left, right = [part.strip().lower() for part in str(definition["formula"]).split("-", 1)]
        panel = panel.with_columns((pl.col(left) - pl.col(right)).alias(name))
    return panel


def main() -> None:
    parser = create_base_parser("Materialize ALFRED initial-release market-state data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to macro config",
    )
    args = parser.parse_args()
    load_dotenv()

    config = load_section(args.config, "macro")
    alfred = config["alfred_initial_release"]
    series_ids = [str(series_id) for series_id in alfred["symbols"]]
    outputs = alfred["outputs"]
    data_root = resolve_data_dir(args.data_path)
    storage_path = resolve_storage_path(data_root, config.get("storage_path"), "macro")
    aligned_path = storage_path / str(outputs["aligned_file"])
    raw_path = storage_path / str(outputs["raw_file"])

    print_section("MACRO DATA DOWNLOAD (ALFRED INITIAL RELEASES)")
    print(f"Config: {args.config}")
    print(f"Output: {aligned_path}")
    print(f"Series: {len(series_ids)}")
    if args.dry_run:
        print_dry_run_notice()
        print_download_summary(
            {
                "series": len(series_ids),
                "output_file": str(aligned_path),
                "raw_file": str(raw_path),
            },
            dry_run=True,
        )
        return

    api_key = require_env(
        "FRED_API_KEY", "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    frames: list[pl.DataFrame] = []
    for series_id in series_ids:
        print(f"  {series_id}...", flush=True)
        frame = fetch_initial_release_series(series_id, api_key)
        frames.append(frame)
        print(
            f"    OK {frame.height:,} observations "
            f"({frame['timestamp'].min()} through {frame['timestamp'].max()})",
            flush=True,
        )

    raw = pl.concat(frames, how="vertical").sort(["series", "timestamp"])
    aligned = build_aligned_panel(raw, series_ids, list(alfred.get("derived", [])))
    storage_path.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(raw, raw_path)
    atomic_write_parquet(aligned, aligned_path)
    print_download_summary(
        {
            "rows": aligned.height,
            "columns": len(aligned.columns),
            "date_range": f"{aligned['timestamp'].min()} to {aligned['timestamp'].max()}",
            "output_file": str(aligned_path),
            "raw_file": str(raw_path),
        }
    )


if __name__ == "__main__":
    main()
