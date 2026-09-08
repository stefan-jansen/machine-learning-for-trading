"""What the stage-output profile has to catch that a null check does not."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from case_studies.utils.artifact_quality import (
    coverage_against,
    explain_missing,
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


def _span(symbol: str, first: int, last: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [symbol] * (last - first + 1), "timestamp": list(range(first, last + 1))}
    )


def test_a_burn_in_and_an_interior_hole_are_the_same_percentage_and_different_findings():
    """The failure the classifier exists for, measured against seoa's two feature stages.

    ``04_model_based_features`` and ``03_financial_features`` both reach 76% of the label
    universe. One pays a declared 252-session estimation burn-in at the start of every
    security; the other loses whole years from the middle of a security's quoted range.
    Coverage cannot tell them apart and ``explain_missing`` has to.
    """
    expected = pl.concat([_span("AAA", 0, 99), _span("BBB", 0, 99)])
    burn_in = pl.concat([_span("AAA", 24, 99), _span("BBB", 24, 99)])
    holes = pl.concat(
        [_span("AAA", 0, 11), _span("AAA", 36, 99), _span("BBB", 0, 11), _span("BBB", 36, 99)]
    )

    for produced in (burn_in, holes):
        coverage = coverage_against(produced, expected, keys=["symbol", "timestamp"])["summary"]
        assert coverage["coverage"].item() == 0.76

    declared = {"leading": (24, "24-session estimation burn-in")}
    clean = explain_missing(
        coverage_against(burn_in, expected, keys=["symbol", "timestamp"])["missing"],
        burn_in,
        entity="symbol",
        session="timestamp",
        expected=declared,
    )
    assert clean["summary"]["where"].to_list() == ["leading"]
    assert clean["summary"]["excess_keys"].item() == 0
    assert clean["residual"].height == 0

    broken = explain_missing(
        coverage_against(holes, expected, keys=["symbol", "timestamp"])["missing"],
        holes,
        entity="symbol",
        session="timestamp",
        expected=declared,
    )
    by_position = dict(broken["summary"].select("where", "n_keys").iter_rows())
    assert by_position == {"interior": 48}
    assert broken["summary"]["why"].item() == "not declared"


def test_an_entity_with_no_produced_row_is_absent_rather_than_leading():
    """A universe restriction is not a burn-in, and a percentage reads it as one."""
    expected = pl.concat([_span("AAA", 0, 9), _span("GONE", 0, 9)])
    produced = _span("AAA", 0, 9)

    ex = explain_missing(
        coverage_against(produced, expected, keys=["symbol", "timestamp"])["missing"],
        produced,
        entity="symbol",
        session="timestamp",
    )
    assert dict(ex["summary"].select("where", "n_keys").iter_rows()) == {"absent": 10}


def test_an_unbounded_expectation_explains_a_position_without_pretending_to_bound_it():
    """seoa's interior holes: the surface quotes no ATM IV, and no session count says how many."""
    expected = pl.concat([_span("AAA", 0, 9), _span("BBB", 0, 9)])
    produced = pl.concat([_span("AAA", 0, 9), _span("BBB", 0, 2), _span("BBB", 8, 9)])

    ex = explain_missing(
        coverage_against(produced, expected, keys=["symbol", "timestamp"])["missing"],
        produced,
        entity="symbol",
        session="timestamp",
        expected={"interior": (None, "the surface quotes nothing there")},
    )
    row = ex["summary"].row(0, named=True)
    assert row["where"] == "interior"
    assert row["n_keys"] == 5
    assert row["entities_over"] == 0
    assert ex["residual"].height == 0


def test_a_compound_entity_is_one_entity():
    """sp500_options keys (symbol, instrument_id, timestamp) and cme_futures (product, position,
    timestamp). Classifying on the first column alone would read one contract's burn-in as an
    interior hole in another's, because the two share an underlying and not a life."""
    rows = []
    for instrument, first in (("C1", 0), ("C2", 5)):
        for t in range(10):
            rows.append({"symbol": "AAA", "instrument_id": instrument, "timestamp": t})
    expected = pl.DataFrame(rows)
    produced = expected.filter(
        ((pl.col("instrument_id") == "C1") & (pl.col("timestamp") >= 2))
        | ((pl.col("instrument_id") == "C2") & (pl.col("timestamp") >= 7))
    )

    keys = ["symbol", "instrument_id", "timestamp"]
    ex = explain_missing(
        coverage_against(produced, expected, keys=keys)["missing"],
        produced,
        entity=["symbol", "instrument_id"],
        session="timestamp",
    )
    # Both contracts pay a leading burn-in; neither shows an interior hole.
    assert dict(ex["summary"].select("where", "n_keys").iter_rows()) == {"leading": 9}
    assert ex["summary"]["n_entities"].item() == 2


def test_an_entity_column_named_like_the_classification_does_not_collide():
    """cme_futures keys on (product, position) and `position` is a contract's place on the
    curve. The classification column has to be named something the panel does not already use,
    or the group_by raises DuplicateError and the whole block fails at run time."""
    rows = [{"product": "CL", "position": 0, "timestamp": t} for t in range(10)] + [
        {"product": "CL", "position": 1, "timestamp": t} for t in range(10)
    ]
    expected = pl.DataFrame(rows)
    produced = expected.filter(pl.col("timestamp") < 8)

    ex = explain_missing(
        coverage_against(produced, expected, keys=["product", "position", "timestamp"])["missing"],
        produced,
        entity=["product", "position"],
        session="timestamp",
        expected={"trailing": (2, "2-settlement horizon")},
    )
    assert "where" in ex["classified"].columns
    assert ex["summary"]["where"].to_list() == ["trailing"]
    assert ex["summary"]["excess_keys"].item() == 0
