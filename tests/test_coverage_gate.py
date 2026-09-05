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


def test_a_frame_with_no_prediction_column_is_not_a_prediction_set(case_dir):
    """`check_prediction_coverage` read a row as a prediction, so a bare session axis passed."""
    frame = _frame().drop("prediction")
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert "no prediction column" in str(excinfo.value)


def test_a_backtest_input_frame_still_needs_no_score_column(case_dir):
    """The requirement sits on the prediction entry point; `_coverage` is shared."""
    report = check_backtest_input_coverage(
        _frame().drop("prediction"), "cs", LABEL, case_dir=case_dir
    )
    assert report.complete


def test_all_null_predictions_do_not_read_as_coverage(case_dir):
    frame = _frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("prediction"))
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert "no predictions" in str(excinfo.value)


def test_non_finite_predictions_do_not_read_as_coverage(case_dir):
    """NaN and infinity are non-null and rank against nothing; they are not decisions."""
    for value in (float("nan"), float("inf"), float("-inf")):
        frame = _frame().with_columns(pl.lit(value, dtype=pl.Float64).alias("prediction"))
        with pytest.raises(CoverageError) as excinfo:
            check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
        assert "no finite value" in str(excinfo.value)


def test_a_session_whose_predictions_are_all_nan_counts_as_missing(case_dir):
    frame = _frame().with_columns(
        pl.when(pl.col("timestamp").dt.day() == 8)
        .then(float("nan"))
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert "1 of 5 declared sessions absent" in str(excinfo.value)


def test_an_integer_score_column_needs_only_to_be_non_null(case_dir):
    """`is_finite` is undefined off a float column, so the condition there is non-null."""
    frame = _frame().with_columns(pl.col("prediction").cast(pl.Int64).alias("prediction"))
    assert check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir).complete


def test_a_session_whose_predictions_are_all_null_counts_as_missing(case_dir):
    frame = _frame().with_columns(
        pl.when(pl.col("timestamp").dt.day() == 8)
        .then(None)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    with pytest.raises(CoverageError) as excinfo:
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert "1 of 5 declared sessions absent" in str(excinfo.value)


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


# A frame checked against the wrong label reports a gap that is not there. Measured in
# `crypto_perps_funding`: the 8-hour label declares 2,189 validation sessions and the
# 24-hour label 2,187, because a decision at 2023-12-31 00:00 realizes inside the
# holdout under a 24-hour horizon and the seal purges it. Checking a 24-hour artifact
# against the primary 8-hour label therefore reports exactly two missing sessions on a
# correct result - and a gate that cries wolf on correct artifacts gets switched off.


def test_a_frame_carrying_another_label_is_refused(case_dir):
    frame = _frame().with_columns(pl.lit("fwd_ret_5d").alias("label"))
    with pytest.raises(CoverageError, match="not 'fwd_ret_1d'"):
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)


def test_a_frame_carrying_the_same_label_passes(case_dir):
    frame = _frame().with_columns(pl.lit(LABEL).alias("label"))
    report = check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)
    assert report.complete


def test_a_frame_with_no_label_column_is_checked_as_before(case_dir):
    # The cross-check is skipped rather than asserted on absent evidence.
    report = check_prediction_coverage(_frame(), "cs", LABEL, case_dir=case_dir)
    assert report.complete


def test_a_frame_mixing_labels_is_refused(case_dir):
    frame = _frame().with_columns(
        pl.when(pl.col("timestamp").dt.day() <= 10)
        .then(pl.lit(LABEL))
        .otherwise(pl.lit("fwd_ret_5d"))
        .alias("label")
    )
    with pytest.raises(CoverageError, match="fwd_ret_5d"):
        check_prediction_coverage(frame, "cs", LABEL, case_dir=case_dir)


# The panel a model actually reads. Two sessions of the label axis are missing from it,
# standing in for an input feed that was out while the forward return - computed from
# prices alone - carried on.
PANEL = pl.Series("timestamp", [ts for ts in SESSIONS if ts.day not in (8, 15)])


def test_a_model_is_not_asked_to_predict_where_the_panel_gave_it_nothing(case_dir):
    """The label artifact declares sessions the feature panel does not have.

    Without `decision_axis` the gate reports the model incomplete for not predicting on
    days it was blind, which is a fact about the feed and not about the model.
    """
    predictions = _frame([ts for ts in SESSIONS if ts.day not in (8, 15)])

    with pytest.raises(CoverageError, match="declared sessions absent"):
        check_prediction_coverage(predictions, "cs", LABEL, case_dir=case_dir)

    report = check_prediction_coverage(
        predictions, "cs", LABEL, case_dir=case_dir, decision_axis=PANEL
    )
    assert report.expected_sessions == 8
    assert report.observed_sessions == 8


def test_the_decision_axis_narrows_the_declaration_and_never_widens_it(case_dir):
    """A timestamp the panel holds and the label artifact does not is still not a session."""
    intruder = dt.datetime(2020, 1, 11, 16, 0)
    assert intruder not in SESSIONS
    wider = pl.Series("timestamp", [*SESSIONS, intruder])

    sessions = declared_sessions("cs", LABEL, case_dir=case_dir, decision_axis=wider)

    assert sum(len(v) for v in sessions.values()) == len(SESSIONS)
    assert all(intruder not in fold for fold in sessions.values())


def test_a_decision_axis_disjoint_from_the_label_artifact_is_refused(case_dir):
    """Not silently zero sessions: an axis that shares nothing is a wrong axis."""
    elsewhere = pl.Series("timestamp", [dt.datetime(2021, 6, d, 16, 0) for d in (1, 2, 3)])

    with pytest.raises(CoverageError, match="share no timestamp"):
        declared_sessions("cs", LABEL, case_dir=case_dir, decision_axis=elsewhere)


def test_the_gap_between_folds_is_read_on_the_decision_axis_too(case_dir, monkeypatch):
    """An outage between two folds is not a fold that failed to account for its sessions.

    The weekend between the two folds carries no sessions, so the between-fold check is silent
    on this fixture. Declaring a session there and withholding it from the panel is what the
    check has to see through: the label artifact has a row nothing could have decided at.
    """
    weekend = dt.datetime(2020, 1, 11, 16, 0)
    labels = case_dir / "labels"
    pl.DataFrame(
        [
            {"timestamp": ts, "symbol": sym, LABEL: 0.01 * i}
            for i, ts in enumerate([*SESSIONS, weekend])
            for sym in ("AAA", "BBB", "CCC")
        ]
    ).write_parquet(labels / f"{LABEL}.parquet")

    with pytest.raises(CoverageError, match="belong to no declared fold"):
        check_prediction_coverage(_frame(), "cs", LABEL, case_dir=case_dir)

    report = check_prediction_coverage(
        _frame(),
        "cs",
        LABEL,
        case_dir=case_dir,
        decision_axis=pl.Series("timestamp", SESSIONS),
    )
    assert report.expected_sessions == len(SESSIONS)
