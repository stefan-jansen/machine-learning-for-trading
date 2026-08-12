import sqlite3

import numpy as np
import polars as pl
import pytest

from case_studies.utils.registry import (
    clear_prediction_sets,
    register_prediction_set,
    register_training_run,
)


def test_clear_prediction_sets_removes_metrics_and_artifact(tmp_path):
    spec = {
        "family": "tabular_dl",
        "label": "fwd_ret_1d",
        "config_name": "tabm_test",
        "params": {},
        "seed": 42,
    }
    training_hash = register_training_run("test", spec, case_dir=tmp_path)
    prediction_hash = register_prediction_set(
        "test",
        training_hash,
        checkpoint_value=1,
        split="validation",
        predictions=pl.DataFrame(
            {
                "timestamp": ["2020-01-01"],
                "symbol": ["A"],
                "fold_id": [0],
                "y_true": [0.1],
                "y_score": [0.2],
            }
        ),
        metrics={"ic_mean": 0.1},
        label="fwd_ret_1d",
        case_dir=tmp_path,
    )

    removed = clear_prediction_sets("test", training_hash, split="validation", case_dir=tmp_path)

    assert removed == {"prediction_sets": 1, "backtest_runs": 0}
    with sqlite3.connect(tmp_path / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM prediction_metrics").fetchone()[0] == 0
    assert not (tmp_path / "run_log" / "predictions" / prediction_hash).exists()


def test_diverged_fold_is_rejected_before_registry_or_artifact_write(tmp_path) -> None:
    spec = {
        "family": "deep_learning",
        "label": "fwd_ret_15m",
        "config_name": "nlinear",
        "params": {},
        "seed": 42,
    }
    training_hash = register_training_run("test", spec, case_dir=tmp_path)
    stable_n = 1_000
    diverged_n = 20
    predictions = pl.DataFrame(
        {
            "timestamp": list(range(stable_n)) + list(range(diverged_n)),
            "symbol": [f"S{i % 10}" for i in range(stable_n + diverged_n)],
            "fold_id": [0] * stable_n + [1] * diverged_n,
            "y_true": [float(i % 11) for i in range(stable_n)]
            + [float(i % 7) / 1_000 for i in range(diverged_n)],
            "y_score": [float(i % 11) for i in range(stable_n)]
            + [float(i % 7) * 100 for i in range(diverged_n)],
        }
    )

    with pytest.raises(ValueError, match=r"fold 1.*dispersion ratio"):
        register_prediction_set(
            "test",
            training_hash,
            split="validation",
            predictions=predictions,
            label="fwd_ret_15m",
            case_dir=tmp_path,
        )

    with sqlite3.connect(tmp_path / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM prediction_metrics").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM fold_metrics").fetchone()[0] == 0
    predictions_dir = tmp_path / "run_log" / "predictions"
    assert not predictions_dir.exists() or not any(predictions_dir.iterdir())


def test_wide_but_finite_prediction_dispersion_is_registered(tmp_path) -> None:
    spec = {
        "family": "linear",
        "label": "fwd_ret_1d",
        "config_name": "ridge",
        "params": {},
        "seed": 42,
    }
    training_hash = register_training_run("test", spec, case_dir=tmp_path)
    actual = np.linspace(-0.01, 0.01, 40)
    predictions = pl.DataFrame(
        {
            "timestamp": list(range(40)),
            "symbol": [f"S{i % 10}" for i in range(40)],
            "fold_id": [0] * 40,
            "y_true": actual,
            "y_score": actual * 50,
        }
    )

    prediction_hash = register_prediction_set(
        "test",
        training_hash,
        split="validation",
        predictions=predictions,
        label="fwd_ret_1d",
        case_dir=tmp_path,
    )

    assert (tmp_path / "run_log" / "predictions" / prediction_hash / "predictions.parquet").exists()


@pytest.mark.parametrize(
    "invalid_scores",
    ([0.2, float("inf")], [float("nan"), float("nan")]),
    ids=("partly-infinite", "entirely-non-finite"),
)
def test_non_finite_fold_is_rejected_before_registry_or_artifact_write(
    tmp_path, invalid_scores
) -> None:
    spec = {
        "family": "deep_learning",
        "label": "fwd_ret_1d",
        "config_name": "nlinear",
        "params": {},
        "seed": 42,
    }
    training_hash = register_training_run("test", spec, case_dir=tmp_path)
    predictions = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "symbol": ["A", "B", "A", "B"],
            "fold_id": [0, 0, 1, 1],
            "y_true": [0.1, 0.2, 0.1, 0.2],
            "y_score": [0.1, 0.2, *invalid_scores],
        }
    )

    with pytest.raises(ValueError, match=r"non-finite fold.*fold 1"):
        register_prediction_set(
            "test",
            training_hash,
            split="validation",
            predictions=predictions,
            label="fwd_ret_1d",
            case_dir=tmp_path,
        )

    with sqlite3.connect(tmp_path / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 0
    predictions_dir = tmp_path / "run_log" / "predictions"
    assert not predictions_dir.exists() or not any(predictions_dir.iterdir())
