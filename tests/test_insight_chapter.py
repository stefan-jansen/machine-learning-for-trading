"""Behavioral tests for cross-case-study insight diagnostics."""

from __future__ import annotations

import json

import polars as pl
import pytest

from case_studies.utils import insight_chapter
from case_studies.utils.conformal import (
    holdout_conformal_embargo_steps,
    walk_forward_conformal_coverage,
)


def test_compare_ic_uses_only_shared_intraday_timestamps() -> None:
    left = pl.DataFrame(
        {
            "date": ["2026-01-02 09:30", "2026-01-02 10:00", "2026-01-03 09:30"],
            "ic": [0.9, 0.2, 0.4],
        }
    ).with_columns(pl.col("date").str.to_datetime().cast(pl.Datetime("ms")))
    right = pl.DataFrame(
        {
            "date": ["2026-01-02 10:00", "2026-01-03 09:30", "2026-01-03 10:00"],
            "ic": [0.1, 0.3, -0.9],
        }
    ).with_columns(pl.col("date").str.to_datetime().cast(pl.Datetime("us")))

    result = insight_chapter.compare_ic_on_shared_timestamps(left, right)

    assert result == {
        "left_ic": pytest.approx(0.3),
        "right_ic": pytest.approx(0.2),
        "n_timestamps": 2,
    }


def _write_prediction_panel(
    prediction_dir, residuals: dict[str, list[float]], *, fold_ids: list[int] | None = None
):
    """One row per (day, symbol), with `y_true - y_score` exactly as `residuals` says."""
    lengths = {len(values) for values in residuals.values()}
    assert len(lengths) == 1
    steps = lengths.pop()
    days = [f"2020-{1 + step // 28:02d}-{1 + step % 28:02d}" for step in range(steps)]
    scores = [float(step) / steps for step in range(steps)]
    folds = fold_ids or [1 if step < steps // 2 else 0 for step in range(steps)]
    pl.DataFrame(
        {
            "timestamp": [day for day in days for _ in residuals],
            "symbol": [symbol for _ in days for symbol in residuals],
            "y_true": [
                scores[step] + residuals[symbol][step]
                for step in range(steps)
                for symbol in residuals
            ],
            "y_score": [scores[step] for step in range(steps) for _ in residuals],
            "fold_id": [folds[step] for step in range(steps) for _ in residuals],
        }
    ).write_parquet(prediction_dir / "predictions.parquet")


def _selected(prediction_hash: str, spec: dict, *, label: str = "fwd_ret_5d") -> dict:
    return {
        "case_study": "probe",
        "family": "gbm",
        "config_name": "probe-config",
        "label": label,
        "prediction_hash": prediction_hash,
        "spec_json": json.dumps(spec),
    }


_TWO_FOLD_SPEC = {"computation": {"expected_prediction_keys": {"n_folds": 2}}}


def test_selected_prediction_conformal_coverage_measures_the_sizing_widths(
    tmp_path, monkeypatch
) -> None:
    """The chapter reports the estimator `conformal_weighted` allocates with.

    Asserted against `walk_forward_conformal_coverage` on the same artifact, because what this
    pins is that the two are one measurement: the chapter used to run a second estimator -
    pooled across symbols, fixed on the earliest fold, unembargoed - and print its coverage as
    the strategy's.
    """
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    _write_prediction_panel(prediction_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    result = insight_chapter.conformal_coverage_for_selected_prediction(
        _selected("prediction-a", _TWO_FOLD_SPEC), levels=(0.80,), embargo_steps=1
    )
    expected = walk_forward_conformal_coverage(
        pl.read_parquet(prediction_dir / "predictions.parquet"), levels=(0.80,), embargo_steps=1
    )

    assert result.height == 1
    assert result.row(0, named=True) == {
        "case_study": "probe",
        "family": "gbm",
        "config_name": "probe-config",
        "prediction_hash": "prediction-a",
        **expected[0],
    }


def test_selected_prediction_conformal_coverage_defaults_to_the_reviewed_horizon(
    tmp_path, monkeypatch
) -> None:
    """The row's own label decides the embargo, so the figure and the widths cannot disagree
    about how far a residual reaches. `label` is required of the selected row for that reason.
    """
    case_dir = tmp_path / "case_studies" / "etfs"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    _write_prediction_panel(prediction_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    row = _selected("prediction-a", _TWO_FOLD_SPEC)
    row["case_study"] = "etfs"
    defaulted = insight_chapter.conformal_coverage_for_selected_prediction(row, levels=(0.80,))
    explicit = insight_chapter.conformal_coverage_for_selected_prediction(
        row, levels=(0.80,), embargo_steps=holdout_conformal_embargo_steps("etfs", "fwd_ret_5d")
    )
    assert defaulted.equals(explicit)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="missing conformal fields"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            {key: value for key, value in row.items() if key != "label"}, levels=(0.80,)
        )


def test_selected_prediction_conformal_coverage_rejects_all_null_declared_fold(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "symbol": ["AAA"] * 80,
            "y_true": [0.1] * 40 + [None] * 40,
            "y_score": [0.0] * 40 + [None] * 40,
            "fold_id": [0] * 40 + [1] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match=r"observed \[0\]"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            _selected("prediction-a", _TWO_FOLD_SPEC), levels=(0.80,), embargo_steps=1
        )


def test_selected_prediction_conformal_coverage_rejects_non_finite_rows(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "symbol": ["AAA"] * 80,
            "y_true": [0.1] * 80,
            "y_score": [0.0] * 79 + [float("inf")],
            "fold_id": [0] * 40 + [1] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="non-finite y_score"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            _selected("prediction-a", _TWO_FOLD_SPEC), levels=(0.80,), embargo_steps=1
        )


def test_selected_prediction_conformal_coverage_reads_the_legacy_spec_shape(
    tmp_path, monkeypatch
) -> None:
    """Two spec shapes are live, and the older one is still written.

    `build_training_spec` puts `n_folds` at the top level and emits no `computation`
    key at all; `run_dl_cv` still uses it and LEGACY_IDENTITY_VERSION is still
    supported. Reading only the identity-v3 location answered 0 for every such row, so
    a row declaring five folds raised "requires at least two declared folds" and
    12_case_study_insights, which tolerates only "fewer than 30 rows", aborted the
    chapter rather than degrading.
    """
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-legacy"
    prediction_dir.mkdir(parents=True)
    _write_prediction_panel(prediction_dir, {"CALM": [0.1] * 80, "WILD": [10.0] * 80})
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    legacy = insight_chapter.conformal_coverage_for_selected_prediction(
        _selected("prediction-legacy", {"family": "deep_learning", "n_folds": 2}),
        levels=(0.80,),
        embargo_steps=1,
    )

    assert legacy["nominal_level"].to_list() == [0.80]


def test_selected_prediction_conformal_coverage_still_rejects_a_single_fold(
    tmp_path, monkeypatch
) -> None:
    """The fallback must not turn the real one-fold refusal into a pass."""
    case_dir = tmp_path / "case_studies" / "probe"
    (case_dir / "run_log" / "predictions" / "prediction-one").mkdir(parents=True)
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="at least two declared folds"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            _selected("prediction-one", {"family": "deep_learning", "n_folds": 1}),
            levels=(0.80,),
            embargo_steps=1,
        )
