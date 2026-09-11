"""Timezone and session normalization for MNQ bar data."""

from __future__ import annotations

from datetime import time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a project dependency
    pd = None


NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
BAR_INTERVAL = timedelta(minutes=5)
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
OUTPUT_COLUMNS = (
    "timestamp",
    "timestamp_ny",
    "session_date",
    "session_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_closed",
)


def _as_polars(frame: pl.DataFrame | "pd.DataFrame") -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    if pd is not None and isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    raise TypeError("frame must be a polars.DataFrame or pandas.DataFrame")


def _normalize_timestamp(frame: pl.DataFrame) -> pl.DataFrame:
    timestamp = frame.schema["timestamp"]
    if timestamp == pl.Utf8:
        try:
            frame = frame.with_columns(pl.col("timestamp").str.to_datetime(strict=True))
        except (pl.exceptions.ComputeError, ValueError) as exc:
            raise ValueError("timestamp contains invalid datetime values") from exc
    elif timestamp != pl.Datetime:
        try:
            frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=True))
        except (pl.exceptions.ComputeError, ValueError) as exc:
            raise ValueError("timestamp contains invalid datetime values") from exc

    # Naive source timestamps are interpreted as UTC; aware timestamps retain
    # their instant while being represented in a single canonical timezone.
    frame = frame.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
        if frame.schema["timestamp"].time_zone is None
        else pl.col("timestamp").dt.convert_time_zone("UTC")
    )
    return frame


def _validate_five_minute_contract(frame: pl.DataFrame) -> None:
    timestamps = frame.get_column("timestamp").to_list()
    if not timestamps:
        return
    if any(
        value.second != 0
        or value.microsecond != 0
        or value.minute % 5 != 0
        for value in timestamps
    ):
        raise ValueError("MNQ timestamps must align to exact 5-minute boundaries")
    if len(timestamps) < 2:
        return
    deltas = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    if any(delta <= timedelta(0) for delta in deltas):
        raise ValueError("MNQ timestamps must be strictly increasing")
    if any(delta != BAR_INTERVAL for delta in deltas):
        raise ValueError("MNQ input contains irregular, non-5-minute cadence")


def assign_new_york_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    """Add New York timestamp, session date, and CME session classification."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")
    if "timestamp" not in frame.columns:
        raise ValueError("required input column missing: timestamp")

    frame = _normalize_timestamp(frame)
    frame = frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("timestamp_ny")
    )
    local_time = pl.col("timestamp_ny").dt.time()
    session_type = (
        pl.when((local_time >= time(9, 30)) & (local_time < time(16)))
        .then(pl.lit("rth"))
        .when((local_time >= time(18)) | (local_time < time(9, 30)))
        .then(pl.lit("overnight"))
        .otherwise(pl.lit("closed"))
        .alias("session_type")
    )
    session_date = (
        pl.when(local_time >= time(18))
        .then(pl.col("timestamp_ny").dt.date() + pl.duration(days=1))
        .otherwise(pl.col("timestamp_ny").dt.date())
        .alias("session_date")
    )
    return frame.with_columns(
        session_date,
        session_type,
    )


def normalize_mnq_bars(frame: pl.DataFrame | "pd.DataFrame") -> pl.DataFrame:
    """Normalize MNQ OHLCV bars into the canonical UTC/New York contract.

    Rows explicitly marked ``bar_closed=False`` are discarded so downstream
    signal code cannot accidentally consume an in-progress bar.
    """
    result = _as_polars(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"required input columns missing: {', '.join(missing)}")
    if "bar_closed" not in result.columns:
        raise ValueError("required input column missing: bar_closed")

    result = _normalize_timestamp(result)
    if result.schema["bar_closed"] != pl.Boolean:
        raise ValueError("bar_closed must contain only explicit boolean values")
    if result.get_column("bar_closed").null_count():
        raise ValueError("bar_closed must contain only explicit boolean values")
    _validate_five_minute_contract(result)
    result = result.filter(pl.col("bar_closed"))

    result = assign_new_york_sessions(result)
    return result.select(OUTPUT_COLUMNS)
