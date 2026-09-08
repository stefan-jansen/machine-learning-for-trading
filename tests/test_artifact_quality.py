"""What the stage-output profile has to catch that a null check does not."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from case_studies.utils.artifact_quality import (
    coverage_against,
    flag_columns,
    profile_columns,
    quality_report,
)


def _panel(symbols: list[str], n: int, *, value: float = 0.01) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [s for s in symbols for _ in range(n)],
            "timestamp": list(range(n)) * len(symbols),
            "fwd_ret": [value] * (n * len(symbols)),
        }
    )


def test_a_complete_frame_that_is_a_thousandth_of_the_universe_reports_the_shortfall():
    """The failure the whole module exists for: no nulls, and almost nothing there.

    A stage emitting 10 of 10,000 owed rows passes every value-level check, because
    each of the 10 values is fine. Coverage is the only thing that sees it.
    """
    expected = _panel([f"S{i:03d}" for i in range(100)], 100)
    produced = _panel(["S000"], 10)

    profile = profile_columns(produced, key_columns=["symbol", "timestamp"])
    assert profile.filter(pl.col("column") == "fwd_ret")["n_null"].item() == 0

    coverage = coverage_against(produced, expected, keys=["symbol", "timestamp"])["summary"]
    assert coverage["expected"].item() == 10_000
    assert coverage["produced"].item() == 10
    assert coverage["missing"].item() == 9_990
    assert coverage["coverage"].item() == 0.001


def test_missing_keys_separate_an_absent_symbol_from_an_interior_hole():
    """A percentage cannot tell a universe restriction from a warm-up; the keys can."""
    expected = _panel(["AAA", "BBB"], 10)
    # AAA is absent entirely; BBB is missing its first three timestamps.
    produced = expected.filter((pl.col("symbol") == "BBB") & (pl.col("timestamp") >= 3))

    missing = coverage_against(produced, expected, keys=["symbol", "timestamp"])["missing"]
    per_symbol = dict(missing.group_by("symbol").len().sort("symbol").iter_rows())
    assert per_symbol == {"AAA": 10, "BBB": 3}
    assert set(produced["symbol"].unique()) == {"BBB"}


def test_a_column_produced_but_never_owed_is_reported_rather_than_ignored():
    expected = _panel(["AAA"], 3)
    produced = _panel(["AAA", "ZZZ"], 3)

    coverage = coverage_against(produced, expected, keys=["symbol", "timestamp"])["summary"]
    assert coverage["missing"].item() == 0
    assert coverage["unexpected"].item() == 3


def test_nulls_and_infinities_are_counted_apart():
    """An absent observation and an overflowed computation have different causes."""
    frame = pl.DataFrame({"symbol": ["A"] * 4, "x": [1.0, None, float("inf"), float("nan")]})
    row = profile_columns(frame, key_columns=["symbol"]).row(0, named=True)
    assert row["n_null"] == 1
    assert row["n_nonfinite"] == 2
    assert row["n_distinct"] == 1


def test_a_legitimately_zero_heavy_column_is_not_flagged_and_a_constant_one_is():
    """Half zeros is what a direction label looks like; every row equal is not."""
    frame = pl.DataFrame(
        {
            "direction": [0.0, 1.0] * 50,
            "always_one": [1.0] * 100,
        }
    )
    flags = flag_columns(profile_columns(frame))
    flagged = dict(flags.select("column", "why").iter_rows())
    assert "direction" not in flagged
    assert "constant over every row" in flagged["always_one"]


def test_an_infinity_is_flagged_even_when_every_threshold_is_relaxed():
    frame = pl.DataFrame({"x": [1.0, 2.0, float("inf")]})
    flags = flag_columns(
        profile_columns(frame),
        rules={"null_share": 1.0, "zero_share": 1.0, "tail_ratio": 1e9},
    )
    assert flags.height == 1
    assert "non-finite" in flags["why"].item()


def test_quality_report_carries_coverage_only_when_a_universe_is_declared():
    frame = _panel(["AAA"], 5)
    assert "coverage_summary" not in quality_report(
        frame, name="x", key_columns=["symbol", "timestamp"]
    )
    with_universe = quality_report(
        frame,
        name="x",
        key_columns=["symbol", "timestamp"],
        expected=_panel(["AAA", "BBB"], 5),
    )
    assert with_universe["coverage_summary"]["coverage"].item() == 0.5


def test_a_key_dtype_mismatch_is_named_rather_than_silently_making_everything_missing():
    """Measured against sp500_equity_option_analytics: the label artifact keys
    ``timestamp`` as Date and the prediction set as Datetime, and an anti-join across
    the two would report every declared row missing."""
    expected = pl.DataFrame({"symbol": ["A"], "timestamp": [date(2020, 1, 1)]})
    produced = pl.DataFrame({"symbol": ["A"], "timestamp": [datetime(2020, 1, 1)]})

    with pytest.raises(ValueError, match="key dtypes differ"):
        coverage_against(produced, expected, keys=["symbol", "timestamp"])
