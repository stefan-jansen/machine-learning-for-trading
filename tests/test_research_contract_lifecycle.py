from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import (
    CandidateSet,
    DecisionArtifact,
    StateTransitionPolicy,
    Study,
)
from tests.test_research_contract_catalog import _publish, _resolved_spec
from tests.test_research_flow import _patch_holdout_prices, _prices
from tests.test_research_registry import _predictions
from tests.test_research_workspace import _seed_release


@pytest.fixture(autouse=True)
def _restore_output_root():
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)
    from case_studies.research import workspace

    workspace._ACTIVE_OUTPUT_ROOT = None
    workspace._clear_root_sensitive_caches()


def _study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _decisions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": [date(2024, 1, 5), date(2024, 1, 5)],
            "position": [1.0, 0.0],
        }
    )


def _prediction(study: Study) -> str:
    training = study.results.register_training(_resolved_spec())
    frame = _predictions()
    return study.results.publish_predictions(
        training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=frame.select("symbol", "timestamp", "fold_id"),
    ).hash


def test_stateful_decision_artifact_requires_policy_before_write(tmp_path: Path) -> None:
    study = _study(tmp_path)

    with pytest.raises(ValueError, match="state transition policy"):
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=["prediction-a"],
            parameters={"entry_threshold": 0.5},
        )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM decision_artifacts").fetchone() == (0,)
    assert not (study.root / "run_log" / "decisions").exists()


def test_typed_decision_artifact_survives_restart_with_state_policy(tmp_path: Path) -> None:
    study = _study(tmp_path)
    policy = StateTransitionPolicy(fold_boundary="liquidate", temporal_gap="reset")
    prediction_hash = _prediction(study)
    published = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={"entry_threshold": 0.5},
        state_transition_policy=policy,
    )
    reopened_study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=study.release_root
    )
    reopened = DecisionArtifact.open(reopened_study, published.hash)

    assert reopened.kind == "target_positions"
    assert reopened.spec["state_transition_policy"] == {
        "fold_boundary": "liquidate",
        "temporal_gap": "reset",
    }
    assert reopened.load().equals(_decisions())


def test_identical_decision_publishers_and_orphan_recovery_preserve_artifact(
    tmp_path: Path,
) -> None:
    study = _study(tmp_path)
    prediction_hash = _prediction(study)
    policy = StateTransitionPolicy(fold_boundary="reset", temporal_gap="reset")

    def publish() -> DecisionArtifact:
        return DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=[prediction_hash],
            parameters={"threshold": 0.5},
            state_transition_policy=policy,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        published = list(pool.map(lambda _: publish(), range(2)))

    assert published[0].hash == published[1].hash
    assert published[0].path.is_file()
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "DELETE FROM decision_artifacts WHERE decision_hash = ?",
            (published[0].hash,),
        )
        db.commit()

    recovered = publish()
    assert recovered.load().equals(_decisions())


def test_released_decision_artifact_opens_without_workspace_copy(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    release_case = release / "case_studies" / "etfs"
    prediction_hash = _publish(release_case, spec=_resolved_spec())
    writable_release = Study("etfs", release_case, release, release.parent, False, {})
    published = DecisionArtifact.publish(
        writable_release,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={},
        state_transition_policy=StateTransitionPolicy("reset", "reset"),
    )
    workspace = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    reopened = DecisionArtifact.open(workspace, published.hash)

    assert reopened.root == release_case
    assert reopened.load().equals(_decisions())


def test_canonical_decision_promotion_requires_replay_evidence(tmp_path: Path) -> None:
    study = _study(tmp_path)
    policy = StateTransitionPolicy(fold_boundary="reset", temporal_gap="reset")
    prediction_hash = _prediction(study)

    with pytest.raises(ValueError, match="missing evidence"):
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=[prediction_hash],
            parameters={},
            source_identity={"module": "research.custom"},
            state_transition_policy=policy,
            canonical=True,
        )

    with pytest.raises(ValueError, match="exact prediction_hashes"):
        DecisionArtifact.publish(
            study,
            kind="target_positions",
            decisions=_decisions(),
            prediction_hashes=[prediction_hash],
            parameters={},
            source_identity={
                "module": "case_studies.research.decisions",
                "source_digest": "source-a",
                "declared_inputs": {"prediction_hashes": ["undisclosed-other-input"]},
                "determinism": {"seed": 42},
                "clean_replay_digest": "not-reached",
            },
            state_transition_policy=policy,
            canonical=True,
        )

    exploratory = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={},
        state_transition_policy=policy,
    )
    evidence = {
        "module": "case_studies.research.decisions",
        "source_digest": "source-a",
        "declared_inputs": {"prediction_hashes": [prediction_hash]},
        "determinism": {"seed": 42},
        "clean_replay_digest": exploratory.spec["artifact_digest"],
    }
    canonical = DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=_decisions(),
        prediction_hashes=[prediction_hash],
        parameters={},
        source_identity=evidence,
        state_transition_policy=policy,
        canonical=True,
    )

    assert not exploratory.canonical
    assert canonical.canonical
    assert canonical.hash != exploratory.hash


def test_holdout_stages_then_transitions_in_one_atomic_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path)
    frame = _predictions()
    expected = frame.select("symbol", "timestamp", "fold_id")
    validation_spec = _resolved_spec()
    validation_training = study.results.register_training(validation_spec)
    validation_prediction = study.results.publish_predictions(
        validation_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="validation",
        predictions=frame,
        expected_keys=expected,
    )
    request = {
        "signal": {"method": "equal_weight_top_k", "top_k": 1},
        "execution_mode": "vectorized",
    }
    validation_backtest = study.strategy(prediction=validation_prediction, **request).run(
        prices=_prices()
    )
    candidates = CandidateSet.create(study, "selection", [validation_backtest])
    holdout_spec = _resolved_spec()
    holdout_spec["computation"]["cv"] = {
        "identity": "holdout-cv",
        "request": {"phase": "holdout", "train_end": "2024-01-04"},
    }
    holdout_prices = _prices().with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    _patch_holdout_prices(monkeypatch, holdout_prices)
    lock = study.lifecycle.lock(
        candidate_set_hash=candidates.hash,
        selected_backtest_hash=validation_backtest.hash,
        selection_evidence={"metric": "validation_backtest_sharpe"},
        holdout_training_spec=holdout_spec,
    )
    assert (
        study.lifecycle.lock(
            candidate_set_hash=candidates.hash,
            selected_backtest_hash=validation_backtest.hash,
            selection_evidence={"metric": "validation_backtest_sharpe"},
            holdout_training_spec=holdout_spec,
        ).hash
        == lock.hash
    )
    holdout_training = study.results.register_training(holdout_spec)
    holdout_frame = frame.with_columns(pl.lit(date(2024, 1, 11)).alias("timestamp"))
    holdout_prediction = study.results.publish_predictions(
        holdout_training,
        checkpoint_kind="final",
        checkpoint_value=None,
        split="holdout",
        predictions=holdout_frame,
        expected_keys=holdout_frame.select("symbol", "timestamp", "fold_id"),
    )
    holdout_backtest = study.strategy(prediction=holdout_prediction, **request).run()

    staged = study.lifecycle.stage_holdout(
        lock.hash,
        holdout_training_hash=holdout_training.hash,
        holdout_prediction_hash=holdout_prediction.hash,
        holdout_backtest_hash=holdout_backtest.hash,
    )
    assert staged.state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_staging").fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (0,)

    from case_studies.research import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "load_backtest_prices_for",
        lambda *args, **kwargs: holdout_prices.with_columns((pl.col("close") + 1).alias("close")),
    )
    with pytest.raises(ValueError, match="exact complete canonical lineage"):
        study.lifecycle.finalize_holdout(lock.hash)
    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    monkeypatch.setattr(
        lifecycle,
        "load_backtest_prices_for",
        lambda *args, **kwargs: holdout_prices,
    )

    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        db.execute(
            "CREATE TRIGGER force_holdout_failure BEFORE UPDATE OF state ON research_locks "
            "WHEN NEW.state = 'HOLDOUT_EVALUATED' BEGIN "
            "SELECT RAISE(ABORT, 'forced holdout failure'); END"
        )
        db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="forced holdout failure"):
        study.lifecycle.finalize_holdout(lock.hash)
    assert study.lifecycle.open(lock.hash).state == "LOCKED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (0,)
        db.execute("DROP TRIGGER force_holdout_failure")
        db.commit()

    evaluated = study.lifecycle.finalize_holdout(lock.hash)
    assert evaluated.state == "HOLDOUT_EVALUATED"
    with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
        assert db.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone() == (1,)
