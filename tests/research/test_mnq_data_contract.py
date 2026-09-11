from datetime import datetime, timezone

import polars as pl

from research.mnq_strategy.data_contract import (
    assign_new_york_sessions,
    normalize_mnq_bars,
)


def make_session_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1200, 800],
        }
    )


def test_normalize_uses_new_york_and_rejects_intrabar_rows():
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
                datetime(2026, 1, 5, 14, 31, tzinfo=timezone.utc),
            ],
            "open": [100.0, 100.5],
            "high": [101.0, 101.5],
            "low": [99.0, 100.0],
            "close": [100.5, 101.0],
            "volume": [1200, 1000],
            "bar_closed": [True, False],
        }
    )
    result = normalize_mnq_bars(frame)
    assert result["timestamp_ny"][0].strftime("%Y-%m-%d %H:%M") == "2026-01-05 09:30"
    assert result["bar_closed"].to_list() == [True]


def test_assigns_rth_and_overnight_sessions():
    result = assign_new_york_sessions(make_session_fixture())
    assert result["session_type"].to_list() == ["rth", "overnight"]
