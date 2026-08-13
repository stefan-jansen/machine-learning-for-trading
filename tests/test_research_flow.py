from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CandidateSet, Result, Strategy, Study
from tests.test_research_registry import _predictions, _training_spec
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-05", "2024-01-05"],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 99.5],
            "volume": [1_000, 1_000],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


def test_fake_prediction_to_backtest_flow_survives_restart(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    training = study.results.register_training(_training_spec())
    predictions = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=predictions,
        expected_keys=predictions.select("symbol", "timestamp", "fold_id"),
    )
    direct = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    )
    first = direct.run(prices=_prices())

    reopened_study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    reopened_prediction = Result.open(reopened_study, prediction.hash)
    notebook_request = {
        "prediction": reopened_prediction,
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    notebook_style = Strategy.from_request(reopened_study, notebook_request)
    second = notebook_style.run(prices=_prices())

    assert reopened_prediction.hash == prediction.hash
    assert direct.identity(prices=_prices()) == notebook_style.identity(prices=_prices())
    assert first.hash == second.hash


def test_strategy_identity_covers_prices_costs_and_rejects_unknown_fields(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    base = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    )
    changed_costs = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        costs={"commission_bps": 5.0, "slippage_bps": 2.0},
        execution_mode="vectorized",
    )
    changed_prices = _prices().with_columns((pl.col("close") + 1).alias("close"))

    assert base.identity(prices=_prices()) != changed_costs.identity(prices=_prices())
    assert base.identity(prices=_prices()) != base.identity(prices=changed_prices)
    with pytest.raises(ValueError, match="unsupported"):
        study.strategy(
            prediction=prediction,
            signal={"method": "equal_weight_top_k", "top_k": 1},
            typo=True,
        )


def test_preview_prediction_to_backtest_flow_stays_isolated(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    training = study.results.register_training(
        _training_spec(
            execution_tier="preview",
            preview_reductions={"folds": [0], "max_rows": 2},
        ),
        execution_tier="preview",
    )
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    preview_backtest = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=_prices())

    canonical_db = study.root / "run_log" / "registry.db"
    with sqlite3.connect(canonical_db) as db:
        backtest_table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'backtest_runs'"
        ).fetchone()
        assert backtest_table is None
    assert preview_backtest.execution_tier == "preview"
    assert Result.open(study, preview_backtest.hash, include_preview=True).complete
    with pytest.raises(KeyError):
        Result.open(study, preview_backtest.hash)


def test_lock_transition_failure_is_atomic(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    training = study.results.register_training(_training_spec())
    frame = _predictions()
    prediction = study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    )
    backtest = study.strategy(
        prediction=prediction,
        signal={"method": "equal_weight_top_k", "top_k": 1},
        execution_mode="vectorized",
    ).run(prices=_prices())
    candidates = CandidateSet.create(study, "selection", [backtest])
    assert candidates.best_validation_sharpe().hash == backtest.hash
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
    )

    with pytest.raises(ValueError, match="complete canonical holdout"):
        study.lifecycle.record_holdout(
            lock.hash,
            holdout_training_hash=training.hash,
            holdout_prediction_hash=prediction.hash,
            holdout_backtest_hash=backtest.hash,
        )

    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0] == 0
