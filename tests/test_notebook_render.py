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

    monkeypatch.setattr(notebook_render, "_registry_path", lambda _case_study: db_path)
    result = notebook_render.conformal_coverage_diagnostic("test", label="fwd_ret_5d")

    assert result["config_name"].unique().to_list() == ["full"]
    assert result["empirical_coverage"].to_list() == [1.0, 1.0, 1.0]
