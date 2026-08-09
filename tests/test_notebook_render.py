"""Regression tests for notebook rendering diagnostics."""

import sqlite3

import numpy as np
import polars as pl

from case_studies.utils import notebook_render


def test_conformal_diagnostic_excludes_partial_and_orders_folds_chronologically(
    tmp_path, monkeypatch
):
    """Coverage uses the full-history leader and the earliest validation fold."""
    run_log = tmp_path / "run_log"
    pred_dir = run_log / "predictions"
    pred_dir.mkdir(parents=True)
    db_path = run_log / "registry.db"

    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY,
                family TEXT,
                config_name TEXT,
                label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY,
                training_hash TEXT,
                split TEXT
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY,
                ic_mean_daily REAL,
                ic_n_days REAL
            );
            """
        )
        db.executemany(
            "INSERT INTO training_runs VALUES (?, 'gbm', ?, 'fwd_ret_5d')",
            [("train_partial", "partial"), ("train_full", "full")],
        )
        db.executemany(
            "INSERT INTO prediction_sets VALUES (?, ?, 'validation')",
            [("pred_partial", "train_partial"), ("pred_full", "train_full")],
        )
        db.executemany(
            "INSERT INTO prediction_metrics VALUES (?, ?, ?)",
            [("pred_partial", 0.20, 1.0), ("pred_full", 0.05, 2.0)],
        )

    for prediction_hash in ("pred_partial", "pred_full"):
        path = pred_dir / prediction_hash
        path.mkdir()
        n = 80
        scores = np.linspace(-1.0, 1.0, n)
        residuals = np.concatenate([np.full(n // 2, 0.1), np.full(n // 2, 0.5)])
        pl.DataFrame(
            {
                "timestamp": ["2020-01-02"] * (n // 2) + ["2019-01-02"] * (n // 2),
                "y_true": scores + residuals,
                "y_score": scores,
                "fold_id": np.repeat([0, 1], n // 2),
            }
        ).write_parquet(path / "predictions.parquet")

    monkeypatch.setattr(notebook_render, "registry_path", lambda _case_study: db_path)
    result = notebook_render.conformal_coverage_diagnostic("test", label="fwd_ret_5d")

    assert result["config_name"].unique().to_list() == ["full"]
    assert result["empirical_coverage"].to_list() == [1.0, 1.0, 1.0]


def _single_config_registry(tmp_path):
    """A registry with one full-history validation config, ready for predictions."""
    run_log = tmp_path / "run_log"
    pred_dir = run_log / "predictions" / "pred_only"
    pred_dir.mkdir(parents=True)
    db_path = run_log / "registry.db"
    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE training_runs (
                training_hash TEXT PRIMARY KEY, family TEXT, config_name TEXT, label TEXT
            );
            CREATE TABLE prediction_sets (
                prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
            );
            CREATE TABLE prediction_metrics (
                prediction_hash TEXT PRIMARY KEY, ic_mean_daily REAL, ic_n_days REAL
            );
            """
        )
        db.execute("INSERT INTO training_runs VALUES ('t', 'gbm', 'only', 'fwd_ret_5d')")
        db.execute("INSERT INTO prediction_sets VALUES ('pred_only', 't', 'validation')")
        db.execute("INSERT INTO prediction_metrics VALUES ('pred_only', 0.1, 2.0)")
    return db_path, pred_dir


def _write_predictions(pred_dir, calibration_residuals, test_residuals):
    """Calibration fold is the earlier one and is deliberately not fold_id 0."""
    n_cal, n_test = len(calibration_residuals), len(test_residuals)
    pl.DataFrame(
        {
            "timestamp": ["2019-01-02"] * n_cal + ["2020-01-02"] * n_test,
            "y_true": list(calibration_residuals) + list(test_residuals),
            "y_score": [0.0] * (n_cal + n_test),
            "fold_id": [1] * n_cal + [0] * n_test,
        }
    ).write_parquet(pred_dir / "predictions.parquet")


def test_conformal_quantile_is_the_exact_order_statistic(tmp_path, monkeypatch):
    """The calibration residual must be selected by rank, not via a probability.

    Split conformal calls for the ceil((n+1)*level)-th smallest residual: with
    40 calibration residuals at the 80% level that is the 33rd, i.e. 0.1 here.
    The calibration set is built so the three plausible answers are all
    distinct, which is what makes this test able to fail:

        rank 33 (correct)                    0.1
        np.quantile(k/n, method="higher")    5.0    - one rank too high
        np.quantile(k/n) linear              0.9575 - a value no residual attains
    """
    db_path, pred_dir = _single_config_registry(tmp_path)
    calibration = [0.1] * 33 + [5.0] * 7
    _write_predictions(pred_dir, calibration, [1.0] * 40)

    monkeypatch.setattr(notebook_render, "registry_path", lambda _cs: db_path)
    result = notebook_render.conformal_coverage_diagnostic(
        "test", label="fwd_ret_5d", levels=(0.80,)
    )

    scale = pl.Series(calibration).std()
    implied_quantile = result["mean_interval_width_frac_std"][0] * scale / 2.0

    n = len(calibration)
    expected = sorted(calibration)[int(np.ceil((n + 1) * 0.80)) - 1]
    assert expected == 0.1
    assert implied_quantile == expected

    # The two wrong branches this replaced would both be visible here.
    assert implied_quantile != 5.0
    assert implied_quantile != float(np.quantile(np.array(calibration), 33 / n))

    # Every test residual is 1.0, above the 0.1 quantile, so nothing is covered.
    assert result["empirical_coverage"].to_list() == [0.0]


def test_conformal_reports_an_unbounded_interval_when_the_level_is_unattainable(
    tmp_path, monkeypatch
):
    """A level the calibration set cannot certify must not be clamped.

    With 40 calibration residuals the 99% rank is ceil(41*0.99) = 41, which no
    residual attains: the conformal interval is unbounded. Reporting the largest
    residual instead would under-cover while still claiming 99%.
    """
    db_path, pred_dir = _single_config_registry(tmp_path)
    calibration = [0.1] * 33 + [5.0] * 7
    _write_predictions(pred_dir, calibration, [1.0] * 40)

    monkeypatch.setattr(notebook_render, "registry_path", lambda _cs: db_path)
    result = notebook_render.conformal_coverage_diagnostic(
        "test", label="fwd_ret_5d", levels=(0.99,)
    )

    assert result["mean_interval_width_frac_std"][0] == float("inf")
    assert result["empirical_coverage"][0] == 1.0

    # The largest residual would have been the clamped answer, and is not used.
    scale = pl.Series(calibration).std()
    assert result["mean_interval_width_frac_std"][0] != 2.0 * max(calibration) / scale


def test_conformal_width_is_scaled_by_the_calibration_window_alone(tmp_path, monkeypatch):
    """The reported width must not be normalized by evaluation-fold outcomes.

    The calibration fold and the evaluation fold are given deliberately
    different return scales, so normalizing over the whole panel produces a
    different number than normalizing over the calibration window.
    """
    db_path, pred_dir = _single_config_registry(tmp_path)
    calibration = [0.1] * 33 + [5.0] * 7
    test_residuals = [40.0, -40.0] * 20  # far wider spread than the calibration fold
    _write_predictions(pred_dir, calibration, test_residuals)

    monkeypatch.setattr(notebook_render, "registry_path", lambda _cs: db_path)
    result = notebook_render.conformal_coverage_diagnostic(
        "test", label="fwd_ret_5d", levels=(0.80,)
    )

    calibration_scale = pl.Series(calibration).std()
    whole_panel_scale = pl.Series(calibration + test_residuals).std()
    width = result["mean_interval_width_frac_std"][0]
    q_hat = sorted(calibration)[int(np.ceil((len(calibration) + 1) * 0.80)) - 1]

    assert width == 2.0 * q_hat / calibration_scale
    assert width != 2.0 * q_hat / whole_panel_scale
