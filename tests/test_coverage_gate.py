"""The coverage gate compares against the declaration, not against the other results.

The three guards that existed before it are all relative - ``full_coverage_prediction_sql``
keeps rows whose ``ic_n_days`` ties the family maximum, the period counts compare one
backtest's observation count with another's, and ``rank_returns_on_common_support``
intersects what two results share. None of them can see a failure that moves every peer
the same way, which is exactly what a stale fold, a boundary that does not line up, or a
join upstream of the backtest produces.

These tests are therefore about the absolute cases: a fold that is declared and absent, a
fold that is present and undeclared, an interior session with no row, and - the one that
matters most - a check that cannot run must not read as a pass.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

import case_studies.utils.coverage as coverage
from case_studies.utils.coverage import (
    CoverageError,
    check_backtest_input_coverage,
    check_prediction_coverage,
    declared_sessions,
)

LABEL = "fwd_ret_1d"
FOLDS = [
    {"fold": 1, "val_start": dt.date(2020, 1, 6), "val_end": dt.date(2020, 1, 10)},
    {"fold": 0, "val_start": dt.date(2020, 1, 13), "val_end": dt.date(2020, 1, 17)},
]
# Two five-session weeks. The weekend between them is not in either fold and is not a
# gap: nothing is declared there, and the axis has no sessions there either.
SESSIONS = [dt.datetime(2020, 1, d, 16, 0) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, LABEL: 0.01 * i}
            for i, ts in enumerate(SESSIONS)
            for sym in ("AAA", "BBB", "CCC")
        ]
    ).write_parquet(labels / f"{LABEL}.parquet")
    return tmp_path


@pytest.fixture(autouse=True)
def declared(monkeypatch):
    """Declare the folds above, and no holdout, so the seal is a no-op here."""
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda cs, label: list(FOLDS))
    monkeypatch.setattr(cv_window, "_holdout_window", lambda cs: None)


def _frame(sessions=SESSIONS, folds=None) -> pl.DataFrame:
    fold_of = folds or (lambda ts: 1 if ts.day <= 10 else 0)
    return pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, "fold_id": fold_of(ts), "prediction": 0.5}
            for ts in sessions
            for sym in ("AAA", "BBB")
        ]
    )


def test_declared_sessions_are_the_axis_inside_each_fold(case_dir):
    sessions = declared_sessions("cs", LABEL, case_dir=case_dir)
    assert sorted(sessions) == [0, 1]
    assert len(sessions[1]) == 5
    assert len(sessions[0]) == 5
    assert sessions[1][0].day == 6
    assert sessions[0][-1].day == 17


def test_complete_frame_passes(case_dir):
    report = check_prediction_coverage(_frame(), "cs", LABEL, case_dir=case_dir)
    assert report.complete
    assert report.expected_sessions == 10
    assert report.observed_sessions == 10


def test_a_dropped_fold_is_caught(case_dir):
    """The crypto case: a join upstream removes every timestamp of one fold."""
    survivors = [ts for ts in SESSIONS if ts.day > 10]
    with pytest.raises(CoverageError) as excinfo:
        check_backtest_input_coverage(_frame(survivors), "cs", LABEL, case_dir=case_dir)
    message = str(excinfo.value)
    assert "missing_fold" in message
    assert "5 of 5 declared sessions absent" in message


def test_a_single_missing_interior_session_is_caught(case_dir):
    survivors = [ts for ts in SESSIONS if ts.day != 8]
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(_frame(survivors), "cs", LABEL, case_dir=case_dir)
    assert "1 of 5 declared sessions absent" in str(excinfo.value)


def test_a_stale_fold_id_is_caught(case_dir):
    """A fold the current setup.yaml does not declare is a leftover, not a bonus."""
    frame = _frame(folds=lambda ts: 7 if ts.day == 6 else (1 if ts.day <= 10 else 0))
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert "undeclared_fold" in str(excinfo.value)


def test_timestamps_outside_their_declared_fold_are_caught(case_dir):
    """Fold 1's rows carrying fold 0's dates is a boundary that did not line up."""
    frame = _frame(folds=lambda ts: 1)
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    message = str(excinfo.value)
    assert "out_of_window" in message
    assert "missing_fold" in message


def test_report_without_raising_still_reports_the_gaps(case_dir):
    survivors = [ts for ts in SESSIONS if ts.day != 8]
    report = check_prediction_coverage(
        _frame(survivors), "cs", LABEL, case_dir=case_dir, raise_on_gap=False
    )
    assert not report.complete
    assert [gap.kind for gap in report.gaps] == ["missing_sessions"]
    assert report.gaps[0].n == 1


def test_an_empty_frame_raises_rather_than_passing_vacuously(case_dir):
    with pytest.raises(CoverageError, match="must not read as a pass"):
        check_prediction_coverage(_frame([]), "cs", LABEL, case_dir=case_dir)


def test_a_missing_label_artifact_raises_rather_than_passing(tmp_path):
    with pytest.raises(CoverageError, match="does not exist"):
        check_prediction_coverage(_frame(), "cs", LABEL, case_dir=tmp_path)


def test_undeclared_folds_raise_rather_than_passing(case_dir, monkeypatch):
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "fold_boundaries", lambda cs, label: None)
    with pytest.raises(CoverageError, match="no CV fold boundaries"):
        check_prediction_coverage(_frame(), "cs", LABEL, case_dir=case_dir)


def test_a_frame_with_no_time_column_raises(case_dir):
    frame = pl.DataFrame({"symbol": ["AAA"], "prediction": [0.1]})
    with pytest.raises(CoverageError, match="no time column"):
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)


def test_sessions_no_fold_declares_are_reported(case_dir, monkeypatch):
    """A hole between two folds that the axis does have sessions in."""
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(
        cv_window,
        "fold_boundaries",
        lambda cs, label: [
            {"fold": 1, "val_start": dt.date(2020, 1, 6), "val_end": dt.date(2020, 1, 8)},
            {"fold": 0, "val_start": dt.date(2020, 1, 13), "val_end": dt.date(2020, 1, 17)},
        ],
    )
    frame = _frame([ts for ts in SESSIONS if ts.day not in (9, 10)])
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert "belong to no declared fold" in str(excinfo.value)


def test_the_seal_shortens_the_last_fold_when_the_horizon_is_intraday(case_dir, monkeypatch):
    """An 8-hourly label cannot be declared at date granularity without over-declaring.

    The last session before a holdout that starts on 2020-01-20 has its 8h outcome at
    2020-01-20 00:00, which is inside the holdout. The seal removes it; a date-granular
    ``val_end`` of 2020-01-17 cannot.
    """
    import case_studies.utils.cv_window as cv_window
    import utils.artifact_specs as artifact_specs

    intraday = [dt.datetime(2020, 1, d, h) for d in (16, 17) for h in (0, 8, 16)]
    labels = case_dir / "labels"
    pl.DataFrame(
        [{"timestamp": ts, "symbol": "AAA", LABEL: 0.01} for ts in intraday]
    ).write_parquet(labels / f"{LABEL}.parquet")

    monkeypatch.setattr(
        cv_window,
        "fold_boundaries",
        lambda cs, label: [
            {"fold": 0, "val_start": dt.date(2020, 1, 16), "val_end": dt.date(2020, 1, 17)}
        ],
    )
    monkeypatch.setattr(
        cv_window, "_holdout_window", lambda cs: (dt.date(2020, 1, 18), dt.date(2020, 1, 31))
    )
    monkeypatch.setattr(artifact_specs, "resolve_label_horizon", lambda cs, label, setup: "8H")
    monkeypatch.setattr(artifact_specs, "resolve_market_semantics", lambda cs, setup: {})
    monkeypatch.setattr(cv_window, "_load_setup_yaml", lambda cs: {})

    sessions = declared_sessions("cs", LABEL, case_dir=case_dir)[0]
    assert sessions[-1] == dt.datetime(2020, 1, 17, 8), sessions
    assert dt.datetime(2020, 1, 17, 16) not in sessions


# --- The three defects the first version shipped with, each measured on a real disagreement.
#
# All three passed the original suite because its fixture wrote naive timestamps and a
# fold_id column on both sides, so the two artifacts agreed by construction. In this repo
# they do not: crypto_perps_funding's label axis is Datetime('ms','UTC') while the
# prediction diagnostics are naive, and the linear/GBM prediction folds carry `fold`
# rather than `fold_id`. A fixture that agrees with itself measures nothing.


@pytest.fixture
def tz_case_dir(tmp_path: Path) -> Path:
    """A tz-aware UTC label axis, as crypto_perps_funding actually writes it."""
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, LABEL: 0.01 * i}
            for i, ts in enumerate(SESSIONS)
            for sym in ("AAA", "BBB", "CCC")
        ]
    ).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    ).write_parquet(labels / f"{LABEL}.parquet")
    return tmp_path


def test_a_tz_aware_label_axis_still_matches_a_naive_prediction_frame(tz_case_dir):
    """The exact disagreement on disk: tz-aware labels, tz-naive predictions.

    Compared as raw objects a tz-aware value never equals a naive one, so a complete
    prediction set reported 100% of sessions missing - on the case study the module's
    own docstring cites as its motivating failure.
    """
    report = check_prediction_coverage(
        _frame(), "cs", LABEL, case_dir=tz_case_dir, raise_on_gap=False
    )
    assert report.complete, report.summary()
    assert report.expected_sessions == len(SESSIONS)


def test_a_tz_aware_frame_matches_a_naive_axis_too(case_dir):
    """The reverse direction, which fails the same way."""
    frame = _frame().with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC")
    )
    report = check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir, raise_on_gap=False)
    assert report.complete, report.summary()


def test_a_date_axis_matches_a_datetime_frame(tmp_path):
    """A Date label column against a Datetime frame, the daily-equities shape."""
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    days = [dt.date(2020, 1, d) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
    pl.DataFrame([{"timestamp": d, "symbol": "AAA", LABEL: 0.01} for d in days]).write_parquet(
        labels / f"{LABEL}.parquet"
    )
    frame = pl.DataFrame(
        [
            {
                "timestamp": dt.datetime(d.year, d.month, d.day),
                "symbol": "AAA",
                "fold_id": 1 if d.day <= 10 else 0,
                "prediction": 0.5,
            }
            for d in days
        ]
    )
    report = check_prediction_coverage(frame, "cs", LABEL, case_dir=tmp_path, raise_on_gap=False)
    assert report.complete, report.summary()


def test_the_fold_checks_run_on_a_fold_schema_frame(case_dir):
    """`fold`, not `fold_id`, is what the linear and GBM path writes.

    A fold_id-only lookup skipped condition (1) in silence here, so a stale fold on the
    artifacts the docstring sends callers to reported as a clean pass.
    """
    frame = _frame().rename({"fold_id": "fold"}).with_columns(pl.lit(7).alias("fold"))
    with pytest.raises(CoverageError, match="undeclared_fold|stale fold"):
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)


def test_a_frame_with_no_fold_column_raises_rather_than_passing(case_dir):
    """The module's own header rule, applied to the module."""
    frame = _frame().drop("fold_id")
    with pytest.raises(CoverageError, match="no fold column"):
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)


def test_fold_column_none_asks_for_session_coverage_only(case_dir):
    """The explicit opt-out is a decision the caller states, not a silent skip."""
    frame = _frame().drop("fold_id")
    report = check_prediction_coverage(
        frame, "cs", LABEL, case_dir=case_dir, fold_column=None, raise_on_gap=False
    )
    assert report.complete, report.summary()


def test_sessions_outside_every_declared_window_are_a_gap(case_dir):
    """`observed_sessions` used to count rows the declaration never asked for."""
    extra = dt.datetime(2020, 2, 3, 16, 0)
    frame = pl.concat(
        [
            _frame(),
            pl.DataFrame([{"timestamp": extra, "symbol": "AAA", "fold_id": 0, "prediction": 0.5}]),
        ]
    )
    with pytest.raises(CoverageError, match="belong to no declared|outside the declared"):
        check_backtest_input_coverage(frame, "cs", LABEL, case_dir=case_dir)
