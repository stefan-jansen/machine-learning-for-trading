"""The canonical eligibility bar must range over every prediction, not one stage's.

`full_coverage_prediction_sql` takes `MAX(ic_n_days)` over all of `(split, family, label)`
with no stage restriction. The canonical path replaces that comparison, so it has to keep
the same bar: computing the max after filtering to the stage being resolved lowers it
whenever the family's full-coverage prediction was never backtested at that stage, and a
partial-coverage prediction then qualifies.
"""

import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry import queries

COVERAGE = {"full": 500, "short": 480}


@pytest.fixture
def case_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One family, two predictions: only the shorter one carries a backtest."""
    run_log = tmp_path / "run_log"
    run_log.mkdir(parents=True)
    db = sqlite3.connect(run_log / "registry.db")
    db.executescript("""
        CREATE TABLE training_runs (
            training_hash TEXT PRIMARY KEY, family TEXT, label TEXT, config_name TEXT
        );
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
        );
        CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
        CREATE TABLE prediction_metrics (prediction_hash TEXT PRIMARY KEY);
    """)
    db.execute("INSERT INTO training_runs VALUES ('T1', 'gbm', 'fwd_ret_5d', 'leaves_7')")
    db.executemany(
        "INSERT INTO prediction_sets VALUES (?, 'T1', 'validation')",
        [("full",), ("short",)],
    )
    db.executemany(
        "INSERT INTO prediction_metrics VALUES (?)",
        [("full",), ("short",)],
    )
    db.commit()
    db.close()

    monkeypatch.setattr(
        queries,
        "canonical_coverage_days",
        lambda cs, label, split, prediction_hash, cdir: COVERAGE.get(prediction_hash),
    )
    return tmp_path


def test_the_bar_is_the_family_best_across_every_prediction(case_dir: Path) -> None:
    bar = queries._canonical_family_coverage_bar("fixture", "fwd_ret_5d", "validation", case_dir)

    assert bar == {"gbm": 500}, (
        "the bar must be the full-coverage prediction's 500 even though only the "
        "480-day one has a backtest at the stage being resolved"
    )


def test_a_prediction_that_cannot_be_evaluated_does_not_set_the_bar(
    case_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`canonical_coverage_days` returns None for "cannot evaluate", which must not
    be read as zero coverage and must not become the family maximum either."""
    monkeypatch.setattr(
        queries,
        "canonical_coverage_days",
        lambda cs, label, split, prediction_hash, cdir: None if prediction_hash == "full" else 480,
    )

    bar = queries._canonical_family_coverage_bar("fixture", "fwd_ret_5d", "validation", case_dir)

    assert bar == {"gbm": 480}


def test_a_prediction_without_metrics_does_not_set_the_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prediction_set committed before its metrics finished computing must not set
    the family bar - full_coverage_prediction_sql requires a prediction_metrics row
    on the raw path (it joins pm), so the canonical bar must too."""
    run_log = tmp_path / "run_log"
    run_log.mkdir(parents=True)
    db = sqlite3.connect(run_log / "registry.db")
    db.executescript("""
        CREATE TABLE training_runs (
            training_hash TEXT PRIMARY KEY, family TEXT, label TEXT, config_name TEXT
        );
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
        );
        CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL);
        CREATE TABLE prediction_metrics (prediction_hash TEXT PRIMARY KEY);
    """)
    db.execute("INSERT INTO training_runs VALUES ('T1', 'gbm', 'fwd_ret_5d', 'leaves_7')")
    db.executemany(
        "INSERT INTO prediction_sets VALUES (?, 'T1', 'validation')",
        [("full",), ("short",)],
    )
    # Only 'short' has a prediction_metrics row - 'full' is metricless.
    db.execute("INSERT INTO prediction_metrics VALUES ('short')")
    db.commit()
    db.close()

    monkeypatch.setattr(
        queries,
        "canonical_coverage_days",
        lambda cs, label, split, prediction_hash, cdir: COVERAGE.get(prediction_hash),
    )

    bar = queries._canonical_family_coverage_bar("fixture", "fwd_ret_5d", "validation", tmp_path)

    assert bar == {"gbm": 480}, (
        "the metricless 'full' prediction (coverage 500) must not set the bar - "
        "only 'short' (coverage 480, has a prediction_metrics row) is eligible"
    )


def test_an_empty_universe_yields_no_bar(tmp_path: Path) -> None:
    (tmp_path / "run_log").mkdir()

    assert queries._canonical_family_coverage_bar("fixture", "l", "validation", tmp_path) == {}


def test_intraday_timestamps_count_one_decision_date_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A minute-bar case study's `timestamp` is a Datetime, so deduplicating before
    the cast to Date counts every bar as its own decision date and the count stops
    being comparable to the daily `ic_n_days` it stands in for.

    The frame carries a realized return and an entity column because the count is
    of *scorable* dates, and a date is scorable only where at least `min_obs`
    entities hold a finite prediction and a finite return, varying across the
    cross-section - a rank correlation is undefined where either side is constant.
    Eight names per bar, each with its own value, puts every date above the floor
    with a defined coefficient, so what this measures is the dedup and nothing else.
    """
    import datetime as dt

    import polars as pl

    from case_studies.utils import cv_window, notebook_contracts

    pred_dir = tmp_path / "run_log" / "predictions" / "H"
    pred_dir.mkdir(parents=True)
    bars = [dt.datetime(2020, 1, day, hour) for day in (6, 7, 8) for hour in (9, 10, 11, 12)]
    rows = [(bar, f"S{i}") for bar in bars for i in range(8)]
    pl.DataFrame(
        {
            "timestamp": [t for t, _ in rows],
            "symbol": [s for _, s in rows],
            "prediction": [float(s[1:]) for _, s in rows],
            "actual": [float(s[1:]) * 0.5 for _, s in rows],
        }
    ).write_parquet(pred_dir / "predictions.parquet")
    monkeypatch.setattr(
        cv_window,
        "canonical_window",
        lambda cs, label, split: (dt.date(2020, 1, 6), dt.date(2020, 1, 8)),
    )

    days = notebook_contracts.canonical_coverage_days(
        "fixture", "label", "validation", "H", tmp_path
    )

    assert days == 3, f"three calendar days across twelve bars, got {days}"
