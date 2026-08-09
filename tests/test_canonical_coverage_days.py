"""A date counts toward canonical coverage only where its cross-section is scorable.

This count stands in for ``ic_n_days``, which is the number of days
``cross_sectional_ic_series`` returned a coefficient for, and that function nulls a
day unless at least ``min_obs`` entities carry a finite prediction and a finite
realized return. Counting rows present instead let a prediction set with a row every
day but two usable names on some of them read as full coverage while the raw path
correctly discounted it - and the two counts are compared against each other in
``full_coverage_prediction_sql``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

import case_studies.utils.notebook_contracts as contracts
from case_studies.utils.notebook_contracts import IC_MIN_OBS, canonical_coverage_days

WINDOW = (dt.date(2020, 1, 1), dt.date(2020, 1, 31))


def _write(
    case_dir: Path,
    rows: list[dict],
    prediction_hash: str = "abc",
    prediction_col: str = "prediction",
    actual_col: str = "actual",
    entity_col: str = "symbol",
) -> None:
    out = case_dir / "run_log" / "predictions" / prediction_hash
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "timestamp": r["timestamp"],
                entity_col: r["entity"],
                prediction_col: r["prediction"],
                actual_col: r["actual"],
            }
            for r in rows
        ]
    ).write_parquet(out / "predictions.parquet")


def _rows(day: int, n: int, prediction=None, actual=None) -> list[dict]:
    """A cross-section that varies across names, because a rank correlation needs it.

    ``prediction`` and ``actual`` override the per-name value with a constant, which
    is how a test asks for the undefined-correlation case.
    """
    return [
        {
            "timestamp": dt.datetime(2020, 1, day, 16, 0),
            "entity": f"S{i}",
            "prediction": float(i) if prediction is None else prediction,
            "actual": float(i) * 0.5 if actual is None else actual,
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _fixed_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(contracts, "canonical_window", None, raising=False)
    import case_studies.utils.cv_window as cv_window

    monkeypatch.setattr(cv_window, "canonical_window", lambda *a, **k: WINDOW)


def _count(case_dir: Path, **kwargs) -> int | None:
    return canonical_coverage_days("cs", "fwd_ret_5d", "validation", "abc", case_dir=case_dir)


def test_a_date_whose_cross_section_is_too_thin_does_not_count(tmp_path: Path) -> None:
    """Three names on the 2nd. The IC series returns no coefficient for that day."""
    rows = _rows(2, IC_MIN_OBS - 2) + _rows(3, 40) + _rows(6, 40)
    _write(tmp_path, rows)
    assert _count(tmp_path) == 2


def test_a_date_at_exactly_min_obs_counts(tmp_path: Path) -> None:
    """The rule is >= min_obs, so the boundary date is scored, not dropped."""
    _write(tmp_path, _rows(2, IC_MIN_OBS) + _rows(3, 40))
    assert _count(tmp_path) == 2


def test_rows_present_but_unusable_do_not_make_a_date_scorable(tmp_path: Path) -> None:
    """Forty rows on the 2nd, all NaN. Present is not the same as scorable."""
    rows = _rows(2, 40, prediction=float("nan")) + _rows(3, 40)
    _write(tmp_path, rows)
    assert _count(tmp_path) == 1


def test_an_infinite_prediction_is_not_a_value(tmp_path: Path) -> None:
    rows = _rows(2, 40, actual=float("inf")) + _rows(3, 40)
    _write(tmp_path, rows)
    assert _count(tmp_path) == 1


def test_a_repeated_entity_does_not_widen_the_cross_section(tmp_path: Path) -> None:
    """The cross-section is a set of names. Four names duplicated is still four."""
    duplicated = _rows(2, IC_MIN_OBS - 1) * 3
    _write(tmp_path, duplicated + _rows(3, 40))
    assert _count(tmp_path) == 1


def test_a_constant_cross_section_has_no_correlation_to_count(tmp_path: Path) -> None:
    """Forty names on the 2nd, every one scored the same. Spearman is undefined.

    Breadth alone would count the date, and the IC series returns null for it, so
    the two counts this stands in for would disagree by exactly that date.
    """
    _write(tmp_path, _rows(2, 40, prediction=0.7) + _rows(3, 40))
    assert _count(tmp_path) == 1


def test_a_constant_realized_return_is_the_same_case(tmp_path: Path) -> None:
    _write(tmp_path, _rows(2, 40, actual=0.0) + _rows(3, 40))
    assert _count(tmp_path) == 1


def test_the_continuous_return_decides_a_classification_prediction_set(tmp_path: Path) -> None:
    """A classification label stores the class target and the return it is scored on.

    `registry/store.py` writes the continuous return as `eval_actual` beside the
    class column. IC is computed against the return, so a date whose classes vary
    while its returns do not produces no coefficient - and resolving `actual` first
    would count it.
    """
    out = tmp_path / "run_log" / "predictions" / "abc"
    out.mkdir(parents=True)
    rows = _rows(2, 40) + _rows(3, 40)
    pl.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in rows],
            "symbol": [r["entity"] for r in rows],
            "prediction": [r["prediction"] for r in rows],
            # The class target varies on both dates ...
            "actual": [float(i % 2) for i in range(len(rows))],
            # ... while the return the IC is taken against is flat on the 2nd.
            "eval_actual": [0.0 if r["timestamp"].day == 2 else r["actual"] for r in rows],
        }
    ).write_parquet(out / "predictions.parquet")

    assert _count(tmp_path) == 1


def test_dates_outside_the_window_are_excluded(tmp_path: Path) -> None:
    outside = [{**r, "timestamp": dt.datetime(2019, 12, 31, 16, 0)} for r in _rows(3, 40)]
    _write(tmp_path, _rows(3, 40) + outside)
    assert _count(tmp_path) == 1


def test_intraday_timestamps_collapse_to_one_decision_date(tmp_path: Path) -> None:
    """Every minute of a session is one date, not many."""
    minutes = [
        {**r, "timestamp": dt.datetime(2020, 1, 3, 9, 30 + m)}
        for m in range(20)
        for r in _rows(3, 40)
    ]
    _write(tmp_path, minutes)
    assert _count(tmp_path) == 1


def test_the_eoa_column_names_resolve(tmp_path: Path) -> None:
    """The predictions schema is not uniform: eoa writes y_score / y_true."""
    _write(tmp_path, _rows(3, 40), prediction_col="y_score", actual_col="y_true")
    assert _count(tmp_path) == 1


def test_the_cme_entity_column_resolves(tmp_path: Path) -> None:
    """cme_futures keys on product, so a symbol-only lookup would count rows."""
    _write(tmp_path, _rows(3, IC_MIN_OBS - 1) * 3, entity_col="product")
    assert _count(tmp_path) == 0


def test_an_unrecognised_schema_is_cannot_evaluate_not_zero(tmp_path: Path) -> None:
    """Callers distinguish None from 0; a schema we cannot read must not read as empty."""
    out = tmp_path / "run_log" / "predictions" / "abc"
    out.mkdir(parents=True)
    pl.DataFrame({"timestamp": [dt.datetime(2020, 1, 3)], "mystery": [1.0]}).write_parquet(
        out / "predictions.parquet"
    )
    assert _count(tmp_path) is None


def test_a_missing_parquet_is_cannot_evaluate(tmp_path: Path) -> None:
    assert _count(tmp_path) is None
