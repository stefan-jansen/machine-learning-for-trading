"""Official Binance USD-M perpetual funding-rate data for the crypto case study."""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import yaml

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH
from utils.paths import get_case_study_dir

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
FUNDING_PATH = ML4T_DATA_PATH / "crypto" / "market" / "funding_rate.parquet"


def _month_keys(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    keys = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        keys.append(f"{year}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return keys


def _fetch_month(symbol: str, month: str) -> tuple[pl.DataFrame | None, bool]:
    url = f"{BASE_URL}/{symbol}/{symbol}-fundingRate-{month}.zip"
    response = httpx.get(url, timeout=30, follow_redirects=True)
    if response.status_code == 404:
        return None, True
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {url}, found {names}")
        frame = pl.read_csv(io.BytesIO(archive.read(names[0]))).with_columns(
            pl.lit(symbol).alias("symbol")
        )
    required = {"calc_time", "funding_interval_hours", "last_funding_rate", "symbol"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Funding archive schema changed for {url}: missing {sorted(missing)}")
    return frame, False


def _normalize_funding(frames: list[pl.DataFrame]) -> pl.DataFrame:
    funding = (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(
            pl.from_epoch("calc_time", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .cast(pl.Datetime("ms", "UTC"))
            .dt.truncate("1h")
            .alias("timestamp"),
            pl.col("funding_interval_hours").cast(pl.Int16),
            pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
        )
        .select("timestamp", "symbol", "funding_interval_hours", "funding_rate")
        .unique(["timestamp", "symbol"], maintain_order=False)
        .sort("timestamp", "symbol")
    )
    if funding.is_empty():
        raise ValueError("Funding archive produced no rows")
    if funding.select(pl.struct("timestamp", "symbol").is_duplicated().any()).item():
        raise ValueError("Funding archive contains duplicate timestamp-symbol keys")
    invalid = funding.filter(
        pl.col("funding_rate").is_nan()
        | pl.col("funding_rate").is_infinite()
        | (pl.col("funding_interval_hours") <= 0)
        | (pl.col("funding_interval_hours") > 8)
    )
    if invalid.height:
        raise ValueError(f"Funding archive contains {invalid.height} invalid rows")
    return funding


def download_funding_rates(
    symbols: list[str],
    start_date: str,
    end_date: str,
    *,
    output_path: Path = FUNDING_PATH,
    max_workers: int = 12,
) -> pl.DataFrame:
    """Download monthly official funding settlements and atomically cache them."""
    tasks = [(symbol, month) for symbol in symbols for month in _month_keys(start_date, end_date)]
    frames: list[pl.DataFrame] = []
    missing = 0
    print(f"Downloading {len(tasks):,} Binance funding-rate archives", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_month, symbol, month): (symbol, month) for symbol, month in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            frame, not_listed = future.result()
            if frame is not None:
                frames.append(frame)
            elif not_listed:
                missing += 1
            if index % 100 == 0 or index == len(tasks):
                print(
                    f"  completed {index:,}/{len(tasks):,}; files={len(frames):,}; "
                    f"unavailable={missing:,}",
                    flush=True,
                )
    funding = _normalize_funding(frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    funding.write_parquet(temporary_path)
    temporary_path.replace(output_path)
    print(
        f"Saved {len(funding):,} settlements for {funding['symbol'].n_unique()} symbols "
        f"to {output_path}",
        flush=True,
    )
    return funding


def load_funding_rates(
    *,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    path: Path = FUNDING_PATH,
) -> pl.DataFrame:
    """Load official funding settlements with optional symbol and date filters."""
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Binance USD-M Perpetual Funding Rates",
            path=path,
            download_script="case_studies/crypto_perps_funding/funding_data.py",
            readme="case_studies/crypto_perps_funding/README.md",
        )
    funding = pl.read_parquet(path)
    if symbols:
        funding = funding.filter(pl.col("symbol").is_in(symbols))
    if start_date:
        funding = funding.filter(pl.col("timestamp").dt.date() >= pl.lit(start_date).str.to_date())
    if end_date:
        funding = funding.filter(pl.col("timestamp").dt.date() <= pl.lit(end_date).str.to_date())
    return funding.sort("timestamp", "symbol")


def main() -> None:
    case_dir = get_case_study_dir("crypto_perps_funding")
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
    download_funding_rates(
        list(setup["universe"]["symbols"]),
        "2020-01-01",
        "2025-12-31",
    )
    print(f"Completed at {datetime.now(UTC).isoformat()}", flush=True)


if __name__ == "__main__":
    main()
