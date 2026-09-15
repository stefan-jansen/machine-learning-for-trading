"""A forecast that ranks nothing within a decision time must be visible as such.

The defect these cover is the one `profile_columns` structurally cannot see: its
`constant` flag is `n_unique() <= 1` over the whole column, so a forecast constant
within each timestamp and drifting across them profiles as varied while ordering
nothing. Every test here is written so that a mutation of the rule reds it - the
negative controls matter as much as the positive ones, because a check that flags
everything is as useless as one that flags nothing.
"""

from __future__ import annotations

import polars as pl
import pytest

from case_studies.utils.artifact_quality import (
    cross_sectional_dispersion,
    profile_columns,
    render_cross_sectional_dispersion,
)


def _panel(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "timestamp": [r[1] for r in rows],
            "prediction": [r[2] for r in rows],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())


def _flat_but_drifting() -> pl.DataFrame:
    """One value per timestamp, a different value each timestamp."""
    return _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("BBB", "2024-01-02T00:00:00", 0.10),
            ("CCC", "2024-01-02T00:00:00", 0.10),
            ("AAA", "2024-01-03T00:00:00", 0.20),
            ("BBB", "2024-01-03T00:00:00", 0.20),
            ("CCC", "2024-01-03T00:00:00", 0.20),
        ]
    )


def test_the_column_profile_cannot_see_it_and_that_is_why_this_exists():
    """The negative control on the OTHER tool. If this ever fails, delete this module."""
    frame = _flat_but_drifting()
    profile = profile_columns(frame, key_columns=["symbol", "timestamp"])
    row = profile.filter(pl.col("column") == "prediction")
    assert row.get_column("constant").item() is False
    assert row.get_column("n_distinct").item() == 2

    report = cross_sectional_dispersion(frame, value="prediction")
    assert report["summary"]["n_degenerate"] == 2
    assert report["summary"]["share_degenerate"] == 1.0


def test_a_forecast_that_orders_its_cross_section_is_not_flagged():
    frame = _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("BBB", "2024-01-02T00:00:00", -0.04),
            ("CCC", "2024-01-02T00:00:00", 0.33),
            ("AAA", "2024-01-03T00:00:00", 0.21),
            ("BBB", "2024-01-03T00:00:00", 0.05),
            ("CCC", "2024-01-03T00:00:00", -0.11),
        ]
    )
    report = cross_sectional_dispersion(frame, value="prediction")
    assert report["summary"]["n_degenerate"] == 0
    assert report["summary"]["share_degenerate"] == 0.0
    assert not report["per_session"].get_column("degenerate").any()


def test_float_noise_below_eps_counts_as_ranking_nothing():
    """The 2e-16 case: not exactly constant, and still ordering nothing.

    Measured across all nine registries: denormal rows carry ic_std 1.95e-17 to
    2.02e-17 while the smallest genuine value is 1.69e-03.
    """
    frame = _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("BBB", "2024-01-02T00:00:00", 0.10 + 2e-17),
            ("CCC", "2024-01-02T00:00:00", 0.10 - 1e-17),
        ]
    )
    report = cross_sectional_dispersion(frame, value="prediction")
    assert report["per_session"].get_column("n_distinct").item() == 3, "not exactly constant"
    assert report["summary"]["n_degenerate"] == 1


def test_a_genuine_but_small_spread_is_not_flagged():
    """The negative control on eps. A tight forecast still ranks, and must survive."""
    frame = _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("BBB", "2024-01-02T00:00:00", 0.10 + 1.69e-03),
            ("CCC", "2024-01-02T00:00:00", 0.10 - 1.69e-03),
        ]
    )
    report = cross_sectional_dispersion(frame, value="prediction")
    assert report["summary"]["n_degenerate"] == 0


def test_a_single_entity_session_is_not_a_failure_to_rank():
    """A universe that shrinks to one name has nothing to rank and is not a defect."""
    frame = _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("AAA", "2024-01-03T00:00:00", 0.20),
            ("BBB", "2024-01-03T00:00:00", 0.05),
        ]
    )
    report = cross_sectional_dispersion(frame, value="prediction")
    summary = report["summary"]
    assert summary["n_sessions"] == 2
    assert summary["n_rankable_sessions"] == 1, "the single-entity session is excluded"
    assert summary["n_degenerate"] == 0
    assert summary["share_degenerate"] == 0.0


def test_the_share_is_over_rankable_sessions_not_all_sessions():
    """Otherwise a panel padded with single-entity sessions dilutes its own defect."""
    rows = [("AAA", "2024-01-02T00:00:00", 0.1), ("BBB", "2024-01-02T00:00:00", 0.1)]
    rows += [("AAA", f"2024-01-{d:02d}T00:00:00", 0.3) for d in range(3, 13)]
    report = cross_sectional_dispersion(_panel(rows), value="prediction")
    summary = report["summary"]
    assert summary["n_sessions"] == 11
    assert summary["n_rankable_sessions"] == 1
    assert summary["share_degenerate"] == 1.0, "1 of 1 rankable, not 1 of 11"


def test_nulls_are_dropped_before_the_count_not_treated_as_a_value():
    frame = _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("BBB", "2024-01-02T00:00:00", 0.10),
            ("CCC", "2024-01-02T00:00:00", 0.10),
        ]
    ).with_columns(
        pl.when(pl.col("symbol") == "CCC")
        .then(None)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    report = cross_sectional_dispersion(frame, value="prediction")
    session = report["per_session"]
    assert session.get_column("n_entities").item() == 2, "the null row is not an entity here"
    assert report["summary"]["n_degenerate"] == 1


def test_an_absent_column_raises_rather_than_reporting_a_clean_panel():
    """The one place this module does raise. A typo must not read as a pass."""
    with pytest.raises(KeyError, match="prediciton"):
        cross_sectional_dispersion(_flat_but_drifting(), value="prediciton")


def test_eps_is_the_registry_degeneracy_constant_not_a_second_opinion():
    from case_studies.utils.notebook_contracts import _DEGENERATE_IC_EPS

    report = cross_sectional_dispersion(_flat_but_drifting(), value="prediction")
    assert report["summary"]["eps"] == _DEGENERATE_IC_EPS


def test_the_renderer_states_the_verdict_and_names_the_offenders(capsys):
    render_cross_sectional_dispersion(
        cross_sectional_dispersion(_flat_but_drifting(), value="prediction")
    )
    out = capsys.readouterr().out
    assert "2 of 2 rankable session(s) rank nothing (100.00%)" in out
    assert "2024-01-02" in out, "the offending session is named, not just counted"


def test_the_renderer_says_so_when_nothing_is_wrong(capsys):
    frame = _panel(
        [
            ("AAA", "2024-01-02T00:00:00", 0.10),
            ("BBB", "2024-01-02T00:00:00", -0.04),
        ]
    )
    render_cross_sectional_dispersion(cross_sectional_dispersion(frame, value="prediction"))
    out = capsys.readouterr().out
    assert "every rankable session orders its cross-section" in out
