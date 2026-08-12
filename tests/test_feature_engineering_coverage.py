"""A NaN is not a value, in the warm-up audit or in the coverage figure.

Polars evaluates ``is_not_null()`` as True for NaN, and the library feature calls
do not agree on which one a warm-up head is written with: polars' own
``rolling_mean`` emits null, ``ml4t.engineer.features.momentum.rsi`` emits NaN.
Both stage-03 helpers read presence, so both were wrong for the NaN half of the
corpus - loudly in one direction and silently in the other.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from case_studies.utils.feature_engineering import family_coverage, warmup_audit


def _panel(values: list[float | None], symbol: str = "AAA") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(values),
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, len(values)), "1d", eager=True
            ),
            "rsi_14d": values,
        }
    )


def test_the_library_still_writes_a_nan_warm_up_head() -> None:
    """The premise. If this fails, the two helpers below are guarding nothing."""
    from ml4t.engineer.features.momentum import rsi

    df = pl.DataFrame({"close": np.linspace(100, 120, 30) + np.sin(np.arange(30))})
    head = df.with_columns(rsi("close", period=14).alias("r"))["r"].to_list()[:3]
    assert all(np.isnan(v) for v in head), f"expected a NaN head, got {head}"


def test_a_nan_warm_up_does_not_fail_a_correctly_warmed_column() -> None:
    """The loud direction, measured on fx_pairs' rsi_14d.

    Fourteen NaN bars then a value at bar 15, against a declared warmup of 14.
    Read as populated at bar 1, that raises the look-ahead assertion - a report
    that reads exactly like a leak on a column that has none.
    """
    census = warmup_audit(
        _panel([float("nan")] * 14 + [55.0, 56.0, 57.0]),
        {"rsi_14d": 14},
        entity="symbol",
    )
    assert census["first populated bar"].to_list() == [15]


def test_a_column_populated_before_its_window_allows_still_raises() -> None:
    """The check keeps its teeth: this is what it exists to catch."""
    with pytest.raises(AssertionError, match="fewer bars than their window spans"):
        warmup_audit(_panel([50.0] * 17), {"rsi_14d": 14}, entity="symbol")


def test_an_all_nan_column_is_reported_as_null_everywhere() -> None:
    """Read as covered, a column holding nothing passed the audit."""
    with pytest.raises(AssertionError, match="null everywhere"):
        warmup_audit(_panel([float("nan")] * 17), {"rsi_14d": 14}, entity="symbol")


def test_a_null_warm_up_behaves_the_same_as_a_nan_one() -> None:
    """Which one the library emitted must not change the answer."""
    nan_head = warmup_audit(
        _panel([float("nan")] * 14 + [55.0, 56.0, 57.0]), {"rsi_14d": 14}, entity="symbol"
    )
    null_head = warmup_audit(
        _panel([None] * 14 + [55.0, 56.0, 57.0]), {"rsi_14d": 14}, entity="symbol"
    )
    assert nan_head["first populated bar"].to_list() == null_head["first populated bar"].to_list()


def test_coverage_draws_a_nan_stretch_as_empty_not_as_dense() -> None:
    """The silent direction: F1 drew a stretch holding no values as fully covered."""
    frame = _panel([float("nan")] * 3 + [55.0, 56.0, 57.0])
    coverage = family_coverage(frame, {"rsi_14d": "momentum"})
    assert coverage["momentum"].to_list() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_coverage_leaves_an_integer_column_alone() -> None:
    """is_not_nan is undefined on an integer series, so the dtype decides."""
    frame = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 3), "1d", eager=True
            ),
            "n_quotes": [1, None, 3],
        }
    )
    assert family_coverage(frame, {"n_quotes": "breadth"})["breadth"].to_list() == [1.0, 0.0, 1.0]
