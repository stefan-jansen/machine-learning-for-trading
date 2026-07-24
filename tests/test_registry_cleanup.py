import sqlite3

import polars as pl

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
