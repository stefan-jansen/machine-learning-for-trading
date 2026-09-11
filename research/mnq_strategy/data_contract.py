"""Timezone and session normalization for MNQ bar data."""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import polars as pl

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a project dependency
    pd = None


NEW_YORK = ZoneInfo("America/New_York")
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
        frame = frame.with_columns(pl.col("timestamp").str.to_datetime())
    elif timestamp != pl.Datetime:
        frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime))

    # Naive source timestamps are interpreted as UTC; aware timestamps retain
    # their instant while being represented in a single canonical timezone.
    frame = frame.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
        if frame.schema["timestamp"].time_zone is None
        else pl.col("timestamp").dt.convert_time_zone("UTC")
    )
    return frame


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
    return frame.with_columns(
        pl.col("timestamp_ny").dt.date().alias("session_date"),
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

    result = _normalize_timestamp(result)
    if "bar_closed" not in result.columns:
        result = result.with_columns(pl.lit(True).alias("bar_closed"))
    else:
        result = result.with_columns(pl.col("bar_closed").cast(pl.Boolean))
        result = result.filter(pl.col("bar_closed"))

    result = assign_new_york_sessions(result)
    return result.select(OUTPUT_COLUMNS)
