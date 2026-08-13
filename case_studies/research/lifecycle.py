from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from case_studies.utils.registry.specs import canonical_json, compute_hash
from case_studies.utils.registry.store import _open_registry, _utc_now

from .comparison import CandidateSet
from .contracts import LifecycleState
from .results import BacktestResult, PredictionResult, Result, TrainingResult

if TYPE_CHECKING:
    from .workspace import Study


@dataclass(frozen=True)
class ResearchLock:
    study: Study
    hash: str
    state: str
    record: dict[str, Any]


class Lifecycle:
    def __init__(self, study: Study) -> None:
        self.study = study

    @property
    def state(self) -> str:
        db_path = self.study.root / "run_log" / "registry.db"
        if not db_path.exists():
            return LifecycleState.DEVELOPMENT.value
        with sqlite3.connect(db_path) as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'research_locks'"
            ).fetchone()
            if exists is None:
                return LifecycleState.DEVELOPMENT.value
            row = db.execute("SELECT state FROM research_locks LIMIT 1").fetchone()
        return row[0] if row else LifecycleState.DEVELOPMENT.value

    def lock(
        self,
        *,
        candidate_set_hash: str,
        selected_backtest_hash: str,
        selection_evidence: dict[str, Any],
    ) -> ResearchLock:
        self.study.require_writable()
        self.study.activate()
        candidates = CandidateSet.open(self.study, candidate_set_hash)
        if candidates.member_kind != "backtest" or selected_backtest_hash not in candidates.members:
            raise ValueError("selected backtest must be an exact member of the candidate set")
        if candidates.best_validation_sharpe().hash != selected_backtest_hash:
            raise ValueError("research lock must select highest validation backtest Sharpe")
        selected = Result.open(self.study, selected_backtest_hash)
        if not isinstance(selected, BacktestResult) or not selected.complete:
            raise ValueError("lock requires a complete backtest")
        if selected.execution_tier != "canonical":
            raise ValueError("preview ancestry cannot enter a lock")
        backtest_record = selected.registry_record()
        prediction = Result.open(self.study, backtest_record["prediction_hash"])
        assert isinstance(prediction, PredictionResult)
        prediction_record = prediction.registry_record()
        training = Result.open(self.study, prediction_record["training_hash"])
        assert isinstance(training, TrainingResult)
        training_record = training.registry_record()
        lock_record = {
            "candidate_set_hash": candidates.hash,
            "selection_evidence": selection_evidence,
            "label": training.spec().get("label"),
            "label_artifact": training.spec().get("label_artifact"),
            "feature_artifacts": training.spec().get("feature_artifacts"),
            "cv": training.spec().get("cv"),
            "training_hash": training.hash,
            "prediction_hash": prediction.hash,
            "checkpoint_kind": prediction_record["checkpoint_kind"],
            "checkpoint_value": prediction_record["checkpoint_value"],
            "validation_backtest_hash": selected.hash,
            "strategy_spec": selected.spec(),
            "source_identity": training_record.get("git_commit"),
            "runtime_provenance": json.loads(training_record.get("runtime_json") or "{}"),
        }
        lock_hash = compute_hash(canonical_json(lock_record))
        db = _open_registry(self.study.root)
        try:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM research_locks LIMIT 1").fetchone() is not None:
                raise ValueError("lifecycle can lock only from DEVELOPMENT")
            db.execute(
                "INSERT INTO research_locks (lock_hash, lock_json, state, created_at) "
                "VALUES (?,?,?,?)",
                (
                    lock_hash,
                    canonical_json(lock_record),
                    LifecycleState.LOCKED.value,
                    _utc_now(),
                ),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return ResearchLock(self.study, lock_hash, LifecycleState.LOCKED.value, lock_record)

    def open(self, lock_hash: str) -> ResearchLock:
        with sqlite3.connect(self.study.root / "run_log" / "registry.db") as db:
            row = db.execute(
                "SELECT lock_json, state FROM research_locks WHERE lock_hash = ?", (lock_hash,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown research lock {lock_hash!r}")
        return ResearchLock(self.study, lock_hash, row[1], json.loads(row[0]))

    def record_holdout(
        self,
        lock_hash: str,
        *,
        holdout_training_hash: str,
        holdout_prediction_hash: str,
        holdout_backtest_hash: str,
    ) -> ResearchLock:
        self.study.require_writable()
        self.study.activate()
        lock = self.open(lock_hash)
        if lock.state != LifecycleState.LOCKED.value:
            raise ValueError("holdout evaluation requires a LOCKED research lock")
        training = Result.open(self.study, holdout_training_hash)
        prediction = Result.open(self.study, holdout_prediction_hash)
        backtest = Result.open(self.study, holdout_backtest_hash)
        valid = (
            isinstance(training, TrainingResult)
            and training.complete
            and training.execution_tier == "canonical"
            and isinstance(prediction, PredictionResult)
            and prediction.complete
            and prediction.registry_record()["split"] == "holdout"
            and prediction.registry_record()["training_hash"] == training.hash
            and isinstance(backtest, BacktestResult)
            and backtest.complete
            and backtest.registry_record()["prediction_hash"] == prediction.hash
        )
        if not valid:
            raise ValueError("holdout transition requires one complete canonical holdout lineage")

        db = _open_registry(self.study.root)
        try:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT state FROM research_locks WHERE lock_hash = ?", (lock_hash,)
            ).fetchone()
            if state != (LifecycleState.LOCKED.value,):
                raise ValueError("research lock is not available for holdout evaluation")
            db.execute(
                "INSERT INTO holdout_evaluations "
                "(lock_hash, holdout_training_hash, holdout_prediction_hash, "
                "holdout_backtest_hash, evaluated_at) VALUES (?,?,?,?,?)",
                (
                    lock_hash,
                    training.hash,
                    prediction.hash,
                    backtest.hash,
                    _utc_now(),
                ),
            )
            updated = db.execute(
                "UPDATE research_locks SET state = ? WHERE lock_hash = ? AND state = ?",
                (
                    LifecycleState.HOLDOUT_EVALUATED.value,
                    lock_hash,
                    LifecycleState.LOCKED.value,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("research lock transition lost an atomicity race")
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return self.open(lock_hash)
