"""Behavioral tests for cross-case-study insight diagnostics."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from case_studies.utils import insight_chapter


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


def test_selected_prediction_conformal_coverage_uses_chronology_and_exact_rank(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    calibration = [0.1] * 33 + [5.0] * 7
    evaluation = [1.0] * 40
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "y_true": calibration + evaluation,
            "y_score": [0.0] * 80,
            "fold_id": [1] * 40 + [0] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    result = insight_chapter.conformal_coverage_for_selected_prediction(
        {
            "case_study": "probe",
            "family": "gbm",
            "config_name": "probe-config",
            "prediction_hash": "prediction-a",
            "spec_json": json.dumps({"n_folds": 2}),
        },
        levels=(0.80,),
    )

    calibration_scale = pl.Series(calibration).std()
    expected_quantile = sorted(calibration)[int(np.ceil(41 * 0.80)) - 1]
    implied_quantile = result["mean_interval_width_frac_std"][0] * calibration_scale / 2.0
    assert expected_quantile == 0.1
    assert implied_quantile == expected_quantile
    assert result["empirical_coverage"].to_list() == [0.0]


def test_selected_prediction_conformal_coverage_uses_calibration_scale(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    calibration = [0.1] * 33 + [5.0] * 7
    evaluation = [40.0, -40.0] * 20
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "y_true": calibration + evaluation,
            "y_score": [0.0] * 80,
            "fold_id": [1] * 40 + [0] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    result = insight_chapter.conformal_coverage_for_selected_prediction(
        {
            "case_study": "probe",
            "family": "gbm",
            "config_name": "probe-config",
            "prediction_hash": "prediction-a",
            "spec_json": json.dumps({"n_folds": 2}),
        },
        levels=(0.80,),
    )

    quantile = 0.1
    width = result["mean_interval_width_frac_std"][0]
    assert width == 2.0 * quantile / pl.Series(calibration).std()
    assert width != 2.0 * quantile / pl.Series(calibration + evaluation).std()


def test_selected_prediction_conformal_coverage_rejects_all_null_declared_fold(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_studies" / "probe"
    prediction_dir = case_dir / "run_log" / "predictions" / "prediction-a"
    prediction_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * 40 + ["2020-01-02"] * 40,
            "y_true": [0.1] * 40 + [None] * 40,
            "y_score": [0.0] * 40 + [None] * 40,
            "fold_id": [0] * 40 + [1] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match=r"observed \[0\]"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            {
                "case_study": "probe",
                "family": "gbm",
                "config_name": "probe-config",
                "prediction_hash": "prediction-a",
                "spec_json": json.dumps({"n_folds": 2}),
            },
            levels=(0.80,),
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
            "y_true": [0.1] * 80,
            "y_score": [0.0] * 79 + [float("inf")],
            "fold_id": [0] * 40 + [1] * 40,
        }
    ).write_parquet(prediction_dir / "predictions.parquet")
    monkeypatch.setattr(insight_chapter, "get_case_study_dir", lambda _case_study: case_dir)

    with pytest.raises(insight_chapter.RegistrySelectionError, match="non-finite y_score"):
        insight_chapter.conformal_coverage_for_selected_prediction(
            {
                "case_study": "probe",
                "family": "gbm",
                "config_name": "probe-config",
                "prediction_hash": "prediction-a",
                "spec_json": json.dumps({"n_folds": 2}),
            },
            levels=(0.80,),
        )
