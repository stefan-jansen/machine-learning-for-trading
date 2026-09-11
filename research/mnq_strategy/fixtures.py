"""Small deterministic Polars fixtures for MNQ strategy tests and notebooks."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

NEW_YORK = ZoneInfo("America/New_York")


def _as_new_york(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=NEW_YORK)
    return timestamp.astimezone(NEW_YORK)


def _session_type(timestamp: datetime) -> str:
    local_time = timestamp.time()
    if time(9, 30) <= local_time < time(16):
        return "rth"
    if local_time >= time(18) or local_time < time(9, 30):
        return "overnight"
    return "closed"


def _canonical_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a fresh frame with stable signal columns and timestamp dtypes."""
    frame = pl.DataFrame(rows)
    required_defaults: dict[str, Any] = {
        "signal": False,
        "direction": None,
        "signal_type": None,
        "entry_time": None,
        "entry_window_start": None,
        "entry_window_end": None,
    }
    for column, default in required_defaults.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(default).alias(column))

    timestamp_ny_dtype = pl.Datetime(time_zone="America/New_York")
    frame = frame.with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_zone="UTC")),
        pl.col("timestamp_ny").cast(timestamp_ny_dtype),
        pl.col("session_date").cast(pl.Date),
        pl.col("bar_closed").cast(pl.Boolean),
        pl.col("signal").cast(pl.Boolean),
        pl.col("direction").cast(pl.Utf8),
        pl.col("signal_type").cast(pl.Utf8),
        pl.col("entry_time").cast(timestamp_ny_dtype),
        pl.col("entry_window_start").cast(timestamp_ny_dtype),
        pl.col("entry_window_end").cast(timestamp_ny_dtype),
    )
    return frame


def _row(
    timestamp_ny: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    volume: int = 100,
    session_type: str | None = None,
    signal: bool = False,
    direction: str | None = None,
    signal_type: str | None = None,
    entry_time: datetime | None = None,
    entry_window_start: datetime | None = None,
    entry_window_end: datetime | None = None,
) -> dict[str, Any]:
    local_timestamp = _as_new_york(timestamp_ny)
    session_date = local_timestamp.date()
    if local_timestamp.time() >= time(18):
        session_date += timedelta(days=1)
    return {
        "timestamp": local_timestamp.astimezone(UTC),
        "timestamp_ny": local_timestamp,
        "session_date": session_date,
        "session_type": session_type or _session_type(local_timestamp),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "bar_closed": True,
        "signal": signal,
        "direction": direction,
        "signal_type": signal_type,
        "entry_time": _as_new_york(entry_time) if entry_time is not None else None,
        "entry_window_start": (
            _as_new_york(entry_window_start) if entry_window_start is not None else None
        ),
        "entry_window_end": (
            _as_new_york(entry_window_end) if entry_window_end is not None else None
        ),
    }


def make_confirmed_signal_fixture() -> pl.DataFrame:
    """Return a closed confirmation followed by a 100.25 next-bar open."""
    confirmation = datetime(2024, 1, 8, 10, 0)
    entry = confirmation + timedelta(minutes=5)
    rows = [
        _row(
            confirmation,
            99.75,
            100.50,
            99.50,
            100.25,
            signal=True,
            direction="long",
            signal_type="10am",
            entry_time=entry,
            entry_window_start=entry,
            entry_window_end=entry,
        ),
        _row(entry, 100.25, 100.75, 100.00, 100.50, signal=False),
        _row(entry + timedelta(minutes=5), 100.50, 101.00, 100.25, 100.75),
    ]
    return _canonical_frame(rows)


def make_overlapping_signals_fixture() -> pl.DataFrame:
    """Return two signals whose five-minute eligible entry windows overlap."""
    first_signal = datetime(2024, 2, 5, 9, 45)
    second_signal = datetime(2024, 2, 5, 9, 55)
    first_entry = first_signal + timedelta(minutes=5)
    second_entry = second_signal + timedelta(minutes=5)
    first_window_end = first_entry + timedelta(minutes=15)
    second_window_end = second_entry + timedelta(minutes=10)
    rows = [
        _row(
            first_signal,
            100.00,
            100.75,
            99.75,
            100.50,
            signal=True,
            direction="long",
            signal_type="rejection",
            entry_time=first_entry,
            entry_window_start=first_entry,
            entry_window_end=first_window_end,
        ),
        _row(first_entry, 100.75, 101.00, 100.50, 100.75),
        _row(
            second_signal,
            100.50,
            101.00,
            100.25,
            100.75,
            signal=True,
            direction="short",
            signal_type="lvn_break_retest",
            entry_time=second_entry,
            entry_window_start=second_entry,
            entry_window_end=second_window_end,
        ),
        _row(second_entry, 100.75, 101.00, 100.25, 100.50),
        _row(second_entry + timedelta(minutes=10), 100.50, 100.75, 100.00, 100.25),
    ]
    return _canonical_frame(rows)


def make_multi_month_fixture() -> pl.DataFrame:
    """Return chronological 2024 train/test bars with a holdout signal."""
    rows: list[dict[str, Any]] = []
    months = (1, 3, 5, 7, 9, 11)
    for index, month in enumerate(months):
        signal_time = datetime(2024, month, 8, 10, 0)
        is_test_signal = month >= 7
        entry_time = signal_time + timedelta(minutes=5)
        rows.extend(
            [
                _row(
                    signal_time,
                    100.0 + index,
                    100.75 + index,
                    99.75 + index,
                    100.50 + index,
                    signal=is_test_signal,
                    direction="long" if is_test_signal else None,
                    signal_type="10am" if is_test_signal else None,
                    entry_time=entry_time if is_test_signal else None,
                    entry_window_start=entry_time if is_test_signal else None,
                    entry_window_end=entry_time + timedelta(minutes=5) if is_test_signal else None,
                ),
                _row(
                    entry_time,
                    100.50 + index,
                    101.25 + index,
                    100.25 + index,
                    101.0 + index,
                ),
            ]
        )
    return _canonical_frame(rows)


def make_session_transition_fixture() -> pl.DataFrame:
    """Return bars spanning RTH, maintenance, and overnight boundaries."""
    return _canonical_frame(
        [
            _row(datetime(2024, 1, 8, 9, 30), 100, 101, 99, 100),
            _row(datetime(2024, 1, 8, 15, 55), 100, 101, 99, 100),
            _row(datetime(2024, 1, 8, 16, 0), 100, 101, 99, 100, session_type="closed"),
            _row(datetime(2024, 1, 8, 18, 0), 100, 101, 99, 100, session_type="overnight"),
        ]
    )


def make_profile_attachment_fixture() -> pl.DataFrame:
    """Return two RTH sessions suitable for previous-profile attachment tests."""
    rows = [
        _row(datetime(2024, 1, 8, 15, 55), 100, 100.5, 99.5, 100.25, volume=200),
        _row(datetime(2024, 1, 9, 9, 30), 101, 101.5, 100.5, 101.25, volume=100),
    ]
    return _canonical_frame(rows)


def make_rejection_fixture() -> pl.DataFrame:
    """Return a midnight level and a bullish closed rejection bar."""
    return _canonical_frame(
        [
            _row(datetime(2024, 1, 8, 0, 0), 100, 100.5, 99.5, 100.25, session_type="overnight"),
            _row(datetime(2024, 1, 8, 9, 55), 103, 103.5, 99.5, 103.5),
        ]
    )


def make_lvn_retest_fixture() -> pl.DataFrame:
    """Return a compact LVN break, later retest, and confirmation sequence."""
    zone = {"low": 100.25, "high": 100.75}
    return _canonical_frame(
        [
            _row(datetime(2024, 1, 9, 9, 30), 100, 101, 99.75, 101, volume=100),
            _row(datetime(2024, 1, 9, 9, 35), 101, 101.25, 100.5, 101, volume=100),
            _row(datetime(2024, 1, 9, 9, 40), 101, 101.25, 100.5, 100.5, volume=100),
            _row(datetime(2024, 1, 9, 9, 45), 100.5, 101.5, 100.5, 101.25, volume=100),
        ]
    ).with_columns(pl.lit([zone] * 4).alias("previous_lvn_zones"))


def make_10am_confirmation_fixture() -> pl.DataFrame:
    """Return six small bars plus a closed 10:00 confirmation bar."""
    rows = [
        _row(
            datetime(2024, 1, 10, 9, 30) + timedelta(minutes=index * 5),
            100 + index * 0.05,
            100.25 + index * 0.05,
            99.75 + index * 0.05,
            100.1 + index * 0.05,
            volume=50,
        )
        for index in range(6)
    ]
    rows.append(_row(datetime(2024, 1, 10, 10, 0), 100, 102, 99.8, 101.8, volume=300))
    return _canonical_frame(rows)


def make_stop_target_fixture() -> pl.DataFrame:
    """Return bars that expose both a configured stop and target path."""
    return _canonical_frame(
        [
            _row(
                datetime(2024, 1, 11, 10, 0),
                100,
                100.5,
                99.5,
                100.25,
                signal=True,
                direction="long",
                signal_type="10am",
                entry_time=datetime(2024, 1, 11, 10, 5),
                entry_window_start=datetime(2024, 1, 11, 10, 5),
                entry_window_end=datetime(2024, 1, 11, 10, 10),
            ),
            _row(datetime(2024, 1, 11, 10, 5), 100.25, 120.25, 80.25, 110),
        ]
    )


def make_cost_fixture() -> pl.DataFrame:
    """Return a deterministic signal and its matching entry bar."""
    return make_confirmed_signal_fixture()


def make_daily_guard_fixture() -> pl.DataFrame:
    """Return three trades with two consecutive losses for guard tests."""
    rows = []
    for index, pnl in enumerate((-200.0, -200.0, 100.0)):
        signal_time = datetime(2024, 1, 12, 10, index * 10)
        entry_time = signal_time + timedelta(minutes=5)
        rows.extend(
            [
                _row(
                    signal_time,
                    100,
                    100.25,
                    99.75,
                    100,
                    signal=True,
                    direction="long",
                    signal_type="10am",
                    entry_time=entry_time,
                    entry_window_start=entry_time,
                    entry_window_end=entry_time,
                ),
                _row(entry_time, 100, 100.25, 99.75, 100, volume=100),
            ]
        )
    frame = _canonical_frame(rows)
    return frame.with_columns(
        pl.col("session_date").alias("trade_date"),
        pl.Series("net_pnl", [pnl for pnl in (-200.0, -200.0, 100.0) for _ in range(2)]),
    )


__all__ = [
    "make_10am_confirmation_fixture",
    "make_confirmed_signal_fixture",
    "make_cost_fixture",
    "make_daily_guard_fixture",
    "make_lvn_retest_fixture",
    "make_multi_month_fixture",
    "make_overlapping_signals_fixture",
    "make_profile_attachment_fixture",
    "make_rejection_fixture",
    "make_session_transition_fixture",
    "make_stop_target_fixture",
]
