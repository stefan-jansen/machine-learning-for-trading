"""A daily-returns frame joins another regardless of the dtype its parquet carried.

`daily_returns.parquet` is written with `Date` by monthly-rebalance aggregations, `Datetime[ms]`
by some engine paths and `Datetime[us]` by others. Polars refuses to join across them, so a
caller holding two of these frames - `17_risk_management`'s paired overlay comparison is one,
and `us_equities_panel/18_risk_management` is another - failed with a comparison error that
depended on which two backtests it happened to be given.

The normalization existed, inside `_align_variants_on_timestamp`, and reached only callers that
routed through it. These tests pin it to the loader, which is where every caller is covered.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from case_studies.utils.uncertainty import (
    _normalized_timestamp,
    load_daily_returns_with_timestamp,
)

_DAYS = [dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)]
_RETS = [0.001, -0.002, 0.003]


def _write(path, timestamps, dtype, ret_col="daily_return"):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": timestamps, ret_col: _RETS}).with_columns(
        pl.col("timestamp").cast(dtype)
    ).write_parquet(path)


def _returns_path(root, case_study, backtest_hash):
    # `get_case_study_dir` resolves to `$ML4T_OUTPUT_DIR/<case_study>` when the variable is set,
    # so the fixture writes where the loader will look.
    return root / case_study / "run_log" / "backtest" / backtest_hash / "daily_returns.parquet"


@pytest.fixture
def study_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "dtype",
    [pl.Date, pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")],
    ids=["date", "ms", "us", "ns"],
)
def test_loader_returns_microsecond_datetime_whatever_the_parquet_holds(study_root, dtype):
    path = _returns_path(study_root, "unit_case_study", "hash_a")
    _write(path, _DAYS, dtype)

    frame = load_daily_returns_with_timestamp("unit_case_study", "hash_a")

    assert frame is not None
    assert frame.schema["timestamp"] == pl.Datetime("us")
    assert frame.height == len(_DAYS)


def test_two_frames_written_with_different_dtypes_join_on_timestamp(study_root):
    """The failure this fixes: an inner join across `Date` and `Datetime[ms]` matched nothing."""
    _write(_returns_path(study_root, "unit_case_study", "baseline"), _DAYS, pl.Date)
    _write(_returns_path(study_root, "unit_case_study", "challenger"), _DAYS, pl.Datetime("ms"))

    baseline = load_daily_returns_with_timestamp("unit_case_study", "baseline")
    challenger = load_daily_returns_with_timestamp("unit_case_study", "challenger")
    joined = baseline.rename({"ret": "baseline_ret"}).join(
        challenger.rename({"ret": "challenger_ret"}), on="timestamp", how="inner"
    )

    assert joined.height == len(_DAYS), "every session should align, not a subset"


def test_a_timezone_aware_stamp_is_made_naive():
    """These are calendar-day rebalance stamps, so a tz would make the join a pure key mismatch."""
    aware = pl.Datetime("us", time_zone="America/New_York")

    frame = pl.DataFrame({"timestamp": _DAYS}).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")).dt.replace_time_zone("America/New_York")
    )
    normalized = frame.select(_normalized_timestamp(aware).alias("timestamp"))

    assert normalized.schema["timestamp"] == pl.Datetime("us")
