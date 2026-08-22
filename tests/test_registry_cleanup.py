import sqlite3
from pathlib import Path

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
    (
        [0.2, float("inf")],
        [0.2, float("nan")],
        [0.2, None],
        [float("nan"), float("nan")],
    ),
    ids=("partly-infinite", "partly-nan", "partly-null", "entirely-non-finite"),
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


def _seed_prediction_store(case_dir: Path, rows: list[tuple[str, int | None]]) -> None:
    """Write a minimal training_runs/prediction_sets pair, one row per (hash, identity).

    ``identity`` of None writes NULL, which is what a row created before the field
    existed carries.
    """
    db_path = case_dir / "run_log" / "registry.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = case_dir / "run_log" / "predictions"
    db = sqlite3.connect(db_path)
    db.executescript(
        """
        CREATE TABLE training_runs (
            training_hash TEXT PRIMARY KEY, family TEXT NOT NULL, label TEXT NOT NULL,
            config_name TEXT, spec_json TEXT, created_at TEXT NOT NULL,
            git_commit TEXT, entry_point TEXT, identity_version INTEGER
        );
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY,
            training_hash TEXT NOT NULL REFERENCES training_runs(training_hash),
            checkpoint_value INTEGER, checkpoint_kind TEXT,
            split TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE prediction_metrics (
            prediction_hash TEXT PRIMARY KEY, computed_at TEXT, ic_mean REAL
        );
        CREATE TABLE fold_metrics (
            prediction_hash TEXT NOT NULL, fold INTEGER NOT NULL, ic REAL
        );
        """
    )
    for t_hash, identity in rows:
        p_hash = f"pred_{t_hash}"
        db.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (t_hash, "linear", "fwd_ret_1m", t_hash, "{}", "2026-01-01", "c", "e", identity),
        )
        db.execute(
            "INSERT INTO prediction_sets VALUES (?,?,?,?,?,?)",
            (p_hash, t_hash, 100, "final", "validation", "2026-01-01"),
        )
        db.execute("INSERT INTO prediction_metrics VALUES (?,?,?)", (p_hash, "2026-01-01", 0.01))
        target = pred_dir / p_hash
        target.mkdir(parents=True, exist_ok=True)
        (target / "predictions.parquet").write_bytes(b"")
    db.commit()
    db.close()


@pytest.mark.parametrize(
    ("rows", "expected", "why"),
    [
        ([("a", None), ("b", None)], 2, "one generation, all pre-field"),
        ([("a", 3), ("b", 3)], 2, "one generation, all current"),
        ([("a", 3), ("b", None)], 1, "two generations, only the current one survives"),
    ],
)
def test_the_identity_filter_applies_only_when_the_store_spans_generations(
    tmp_path: Path, rows: list[tuple[str, int | None]], expected: int, why: str
) -> None:
    """A store holding one generation is not something a query can select across.

    The filter exists to stop a backtest comparing models fitted under different identity
    rules. Applied unconditionally it instead empties every registry whose rows predate the
    field, which is every seeded fixture and every reader's existing run log: three
    case-study CI jobs went from green to "No predictions found" in the backtest downstream
    of the model stage, and no reader would have got a different answer.
    """
    from case_studies.utils.registry import queries

    case_dir = tmp_path / "somecase"
    _seed_prediction_store(case_dir, rows)
    df = queries.load_prediction_index("somecase", case_dir=case_dir)
    assert df.height == expected, why
