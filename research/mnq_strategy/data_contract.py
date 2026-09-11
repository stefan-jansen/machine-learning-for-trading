"""Timezone and session normalization for MNQ bar data."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
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


def _as_polars(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.clone()
    if pd is not None and isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    raise TypeError("frame must be a polars.DataFrame or pandas.DataFrame")


def _coerce_datetime_column(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    timestamp = frame.schema[column]
    if timestamp == pl.Utf8:
        try:
            return frame.with_columns(pl.col(column).str.to_datetime(strict=True).alias(column))
        except (pl.exceptions.ComputeError, ValueError) as exc:
            raise ValueError(f"{column} contains invalid datetime values") from exc
    if not isinstance(timestamp, pl.Datetime):
        try:
            return frame.with_columns(pl.col(column).cast(pl.Datetime, strict=True).alias(column))
        except (pl.exceptions.ComputeError, ValueError) as exc:
            raise ValueError(f"{column} contains invalid datetime values") from exc
    return frame


def _normalize_timestamp(frame: pl.DataFrame) -> pl.DataFrame:
    frame = _coerce_datetime_column(frame, "timestamp")

    # Naive source timestamps are interpreted as UTC; aware timestamps retain
    # their instant while being represented in a single canonical timezone.
    frame = frame.with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
        if frame.schema["timestamp"].time_zone is None
        else pl.col("timestamp").dt.convert_time_zone("UTC")
    )
    return frame


def normalize_timestamp_columns(
    frame: pl.DataFrame,
    timezone: str = "America/New_York",
) -> pl.DataFrame:
    """Return one canonical UTC timestamp and an aware local timestamp.

    ``timestamp`` is a source timestamp and naive values mean UTC.  A
    supplied ``timestamp_ny`` is already a local representation, so a naive
    value is rejected rather than silently assigning it a second meaning.
    When both columns are supplied they must describe the same instant.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")
    if "timestamp" not in frame.columns and "timestamp_ny" not in frame.columns:
        raise ValueError("required input column missing: timestamp or timestamp_ny")

    result = frame.clone()
    has_raw_timestamp = "timestamp" in result.columns
    has_local_timestamp = "timestamp_ny" in result.columns
    if has_raw_timestamp:
        result = _normalize_timestamp(result)

    if has_local_timestamp:
        result = _coerce_datetime_column(result, "timestamp_ny")
        local_dtype = result.schema["timestamp_ny"]
        if local_dtype.time_zone is None:
            raise ValueError("timestamp_ny must be timezone-aware; naive values are ambiguous")
        provided_local = pl.col("timestamp_ny").dt.convert_time_zone(timezone)
    else:
        provided_local = pl.col("timestamp").dt.convert_time_zone(timezone)

    if has_raw_timestamp:
        expected_local = pl.col("timestamp").dt.convert_time_zone(timezone)
        if has_local_timestamp:
            expected_values = result.select(expected_local.alias("_expected")).get_column(
                "_expected"
            )
            provided_values = result.select(provided_local.alias("_provided")).get_column(
                "_provided"
            )
            if any(
                expected != provided for expected, provided in zip(expected_values, provided_values)
            ):
                raise ValueError("timestamp and timestamp_ny must describe the same instant")
        result = result.with_columns(expected_local.alias("timestamp_ny"))
    else:
        result = result.with_columns(provided_local.alias("timestamp_ny"))
        result = result.with_columns(
            pl.col("timestamp_ny").dt.convert_time_zone("UTC").alias("timestamp")
        )
    return result


def cme_session_date(timestamp: datetime) -> date:
    """Return the CME session date for an aware New York timestamp."""
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    local_timestamp = timestamp.astimezone(NEW_YORK)
    return local_timestamp.date() + timedelta(days=int(local_timestamp.time() >= time(18)))


def add_cme_session_dates(frame: pl.DataFrame) -> pl.DataFrame:
    """Overwrite ``session_date`` using the canonical 18:00 rollover rule."""
    if "timestamp_ny" not in frame.columns:
        raise ValueError("required input column missing: timestamp_ny")
    timestamp_dtype = frame.schema["timestamp_ny"]
    if not isinstance(timestamp_dtype, pl.Datetime) or timestamp_dtype.time_zone is None:
        raise ValueError("timestamp_ny must be timezone-aware")
    local_time = pl.col("timestamp_ny").dt.time()
    return frame.with_columns(
        pl.when(local_time >= time(18))
        .then(pl.col("timestamp_ny").dt.date() + pl.duration(days=1))
        .otherwise(pl.col("timestamp_ny").dt.date())
        .alias("session_date")
    )


def validate_five_minute_cadence(
    frame: pl.DataFrame,
    timestamp_column: str = "timestamp",
) -> None:
    """Reject misaligned bars and irregular gaps within one CME session."""
    if timestamp_column not in frame.columns:
        raise ValueError(f"required input column missing: {timestamp_column}")
    timestamps = frame.get_column(timestamp_column).to_list()
    if not timestamps:
        return
    if any(
        value.second != 0 or value.microsecond != 0 or value.minute % 5 != 0 for value in timestamps
    ):
        raise ValueError("MNQ timestamps must align to exact 5-minute boundaries")
    if len(timestamps) < 2:
        return
    deltas = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    if any(delta <= timedelta(0) for delta in deltas):
        raise ValueError("MNQ timestamps must be strictly increasing")
    if "session_date" in frame.columns:
        session_dates = frame.get_column("session_date").to_list()
    else:
        session_dates = [cme_session_date(value) for value in timestamps]
    local_times = [value.astimezone(NEW_YORK).time() for value in timestamps]

    def session_type(local_time: time) -> str:
        if time(9, 30) <= local_time < time(16):
            return "rth"
        if time(18) <= local_time or local_time < time(9, 30):
            return "overnight"
        return "closed"

    session_types = [session_type(value) for value in local_times]
    if any(
        delta != BAR_INTERVAL
        and session_dates[index] == session_dates[index + 1]
        and session_types[index] == session_types[index + 1]
        and session_types[index] != "closed"
        for index, delta in enumerate(deltas)
    ):
        raise ValueError("MNQ input contains irregular, non-5-minute cadence")


def _validate_five_minute_contract(frame: pl.DataFrame) -> None:
    validate_five_minute_cadence(frame)


def assign_new_york_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    """Add New York timestamp, session date, and CME session classification."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a polars.DataFrame")
    if "timestamp" not in frame.columns:
        raise ValueError("required input column missing: timestamp")

    frame = normalize_timestamp_columns(frame)
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
    return add_cme_session_dates(frame.with_columns(session_type))


def normalize_mnq_bars(frame: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
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
