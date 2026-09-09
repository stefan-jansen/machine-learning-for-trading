"""A family can score every session and still score half the universe.

The three conditions ``coverage.py`` states are all about the time axis, so a result
that drops symbols satisfies every one of them. This file is the cross-section: the
same declaration, read as ``(entity, session)`` pairs rather than as sessions.

The fixture below is deliberately the one ``test_coverage_gate.py`` already uses - its
label artifact carries three symbols and its prediction frame carries two, and the
session gate passes on it. That is the blind spot stated as a fixture rather than as a
claim.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.coverage import (
    CoverageError,
    check_prediction_coverage,
    check_prediction_cross_section,
    declared_cross_section,
)

LABEL = "fwd_ret_1d"
FOLDS = [
    {"fold": 0, "val_start": dt.date(2020, 1, 6), "val_end": dt.date(2020, 1, 10)},
    {"fold": 1, "val_start": dt.date(2020, 1, 13), "val_end": dt.date(2020, 1, 17)},
]
SESSIONS = [dt.datetime(2020, 1, d, 16, 0) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
UNIVERSE = ("AAA", "BBB", "CCC")


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, LABEL: 0.01 * i}
            for i, ts in enumerate(SESSIONS)
            for sym in UNIVERSE
        ]
    ).write_parquet(labels / f"{LABEL}.parquet")
    return tmp_path


@pytest.fixture(autouse=True)
def declared(monkeypatch):
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda cs, label: list(FOLDS))
    monkeypatch.setattr(cv_window, "_holdout_window", lambda cs: None)


def _predictions(symbols=UNIVERSE, sessions=SESSIONS) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, "fold_id": 0 if ts.day <= 10 else 1, "prediction": 0.5}
            for ts in sessions
            for sym in symbols
        ]
    )


def test_a_dropped_symbol_passes_the_session_gate_and_fails_the_cross_section(case_dir):
    partial = _predictions(symbols=("AAA", "BBB"))

    assert check_prediction_coverage(partial, "cs", LABEL, case_dir=case_dir).complete

    report = check_prediction_cross_section(partial, "cs", LABEL, case_dir=case_dir)
    assert not report.complete
    assert report.expected == 30
    assert report.delivered == 20
    assert report.coverage == pytest.approx(2 / 3)
    assert report.never_scored == ("CCC",)
    assert report.partially_scored == ()


def test_a_complete_cross_section_is_complete(case_dir):
    report = check_prediction_cross_section(_predictions(), "cs", LABEL, case_dir=case_dir)
    assert report.complete
    assert report.coverage == 1.0
    assert report.never_scored == ()


def test_an_absent_entity_and_a_warm_up_hole_are_reported_apart(case_dir):
    """Same percentage, different defects: a universe restriction and a burn-in."""
    frame = _predictions(symbols=("AAA", "BBB")).filter(
        ~((pl.col("symbol") == "BBB") & (pl.col("timestamp").dt.day() < 9))
    )
    report = check_prediction_cross_section(frame, "cs", LABEL, case_dir=case_dir)
    assert report.never_scored == ("CCC",)
    assert report.partially_scored == ("BBB",)


def test_the_shortfall_a_stage_inherited_is_not_charged_to_it(case_dir):
    """The seoa shape: every family misses the pairs the feature panel never offered.

    Charged against the label, a family that delivered everything it was handed reads at
    two thirds and a gate refuses it. Charged against the panel, it reads at 100% and the
    gate is free to refuse the families that actually lost rows.
    """
    panel = pl.DataFrame(
        [{"timestamp": ts, "symbol": sym} for ts in SESSIONS for sym in ("AAA", "BBB")]
    )
    report = check_prediction_cross_section(
        _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    assert report.coverage == pytest.approx(2 / 3)
    assert report.achievable == 20
    assert report.delivered_achievable == 20
    assert report.accountable_coverage == 1.0
    assert "of the 20 its input panel offered" in report.summary()


def test_a_family_that_loses_rows_the_panel_offered_is_still_charged(case_dir):
    panel = pl.DataFrame([{"timestamp": ts, "symbol": sym} for ts in SESSIONS for sym in UNIVERSE])
    report = check_prediction_cross_section(
        _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, input_panel=panel
    )
    assert report.accountable_coverage == pytest.approx(2 / 3)
    with pytest.raises(CoverageError, match="is below"):
        report.raise_if_below(0.98)


def test_minimum_raises_at_the_call_site(case_dir):
    with pytest.raises(CoverageError, match="66.7% is below 98.0%"):
        check_prediction_cross_section(
            _predictions(symbols=("AAA", "BBB")), "cs", LABEL, case_dir=case_dir, minimum=0.98
        )


def test_no_minimum_returns_the_report_rather_than_judging_it(case_dir):
    """The default is a measurement. Only a caller naming a threshold gets a refusal."""
    report = check_prediction_cross_section(
        _predictions(symbols=("AAA",)), "cs", LABEL, case_dir=case_dir
    )
    assert report.coverage == pytest.approx(1 / 3)


def test_the_declared_grid_carries_its_fold(case_dir):
    grid = declared_cross_section("cs", LABEL, case_dir=case_dir)
    assert grid.height == 30
    assert dict(grid.group_by("fold").len().sort("fold").iter_rows()) == {0: 15, 1: 15}


def test_an_entity_column_under_the_other_canonical_name_is_found(case_dir):
    """cme_futures keys its labels by `product` and its prediction panels by `symbol`."""
    labels = case_dir / "labels" / f"{LABEL}.parquet"
    pl.read_parquet(labels).rename({"symbol": "product"}).write_parquet(labels)
    report = check_prediction_cross_section(_predictions(), "cs", LABEL, case_dir=case_dir)
    assert report.complete


def test_a_frame_with_no_entity_column_cannot_read_as_a_pass(case_dir):
    frame = _predictions().drop("symbol")
    with pytest.raises(CoverageError, match="no entity column"):
        check_prediction_cross_section(frame, "cs", LABEL, case_dir=case_dir)


class TestTheFoldAxisComesFromTheRun:
    """A run can declare fewer folds than the configuration lists.

    `declared_sessions` reads the fold windows out of `setup.yaml`, which lists every
    configured fold whatever the run was asked to do, so a run that fitted a subset reads
    at the ratio of the two counts however complete it is - one fold of two here is 50%,
    and a member that far short is dropped from the pool that ranks it. The shape that
    produces it is `splits[:MAX_FOLDS]`, which ml4t/agent-workspace#1076 finds in three
    case studies.

    The narrowing is on the fold axis alone. Within the folds a run declares, the symbol
    axis is still the panel's, so a family that lost names is still charged for them.
    """

    FOLD_0 = [ts for ts in SESSIONS if ts.day <= 10]

    def test_without_the_run_s_folds_a_one_fold_run_reads_half(self, case_dir):
        report = check_prediction_cross_section(
            _predictions(sessions=self.FOLD_0),
            "cs",
            LABEL,
            case_dir=case_dir,
            source="deep_learning/lstm_h64",
        )
        assert report.expected == len(UNIVERSE) * len(SESSIONS)
        assert report.accountable_coverage == pytest.approx(0.5)

    def test_declaring_fold_0_makes_the_same_run_whole(self, case_dir):
        report = check_prediction_cross_section(
            _predictions(sessions=self.FOLD_0),
            "cs",
            LABEL,
            case_dir=case_dir,
            folds=[0],
            source="deep_learning/lstm_h64",
        )
        assert report.expected == len(UNIVERSE) * len(self.FOLD_0)
        assert report.accountable_coverage == pytest.approx(1.0)
        assert report.complete

    def test_a_symbol_dropped_inside_a_declared_fold_is_still_charged(self, case_dir):
        """The narrowing must not excuse the shortfall it exists to measure."""
        report = check_prediction_cross_section(
            _predictions(symbols=("AAA", "BBB"), sessions=self.FOLD_0),
            "cs",
            LABEL,
            case_dir=case_dir,
            folds=[0],
            source="deep_learning/lstm_h64",
        )
        assert report.accountable_coverage == pytest.approx(2 / 3)
        assert report.never_scored == ("CCC",)

    def test_declaring_a_fold_the_configuration_does_not_have_is_refused(self, case_dir):
        with pytest.raises(CoverageError, match="declares fold"):
            check_prediction_cross_section(
                _predictions(sessions=self.FOLD_0),
                "cs",
                LABEL,
                case_dir=case_dir,
                folds=[7],
            )
