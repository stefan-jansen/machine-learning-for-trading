from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.backtest_runner import resolve_cost_feasible_universe
from case_studies.utils.registry.specs import (
    canonical_json,
    compute_hash,
    project_training_identity,
    training_hash_from_spec,
)
from case_studies.utils.registry.store import _open_registry, _utc_now

from .comparison import CandidateSet
from .contracts import LifecycleState
from .results import BacktestResult, PredictionResult, Result, TrainingResult
from .strategy import strategy_warmup_periods

if TYPE_CHECKING:
    from .workspace import Study


@dataclass(frozen=True)
class ResearchLock:
    study: Study
    hash: str
    state: str
    record: dict[str, Any]


def _locked_training_spec(validation_spec: dict[str, Any], holdout_spec: dict[str, Any]) -> dict:
    projected_validation = project_training_identity(validation_spec)
    projected_holdout = project_training_identity(holdout_spec)
    if projected_holdout.get("execution_tier") != "canonical":
        raise ValueError("holdout retraining must use the canonical execution tier")
    validation_computation = projected_validation.get("computation", projected_validation)
    holdout_computation = projected_holdout.get("computation", projected_holdout)
    validation_cv = validation_computation.pop("cv", None)
    holdout_cv = holdout_computation.pop("cv", None)
    if holdout_cv is None or holdout_cv == validation_cv:
        raise ValueError("holdout retraining requires an explicit, distinct CV interval")
    if projected_holdout != projected_validation:
        raise ValueError("holdout retraining may differ from selected training only in CV interval")
    return project_training_identity(holdout_spec)


def _locked_strategy_projection(spec: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(spec)
    projected.pop("_runtime_backtest_config", None)
    metadata = projected.get("backtest_config", {}).get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("prediction_hash", None)
    input_identity = projected.get("input_identity")
    if isinstance(input_identity, dict):
        input_identity.pop("prices", None)
    signal = projected.get("strategy", {}).get("signal")
    if isinstance(signal, dict) and signal.get("universe_filter") == "cost_feasible":
        if isinstance(input_identity, dict):
            input_identity.pop("universe", None)
        signal.pop("universe_split", None)
        signal.pop("universe_symbols", None)
    return projected


def _strategy_roster_is_resolved(
    spec: dict[str, Any],
    case_study: str,
    prediction_hash: str,
) -> bool:
    signal = spec.get("strategy", {}).get("signal", {})
    if signal.get("universe_filter") != "cost_feasible":
        return True
    split, symbols = resolve_cost_feasible_universe(case_study, prediction_hash)
    expected_digest = compute_hash(canonical_json({"split": split, "symbols": symbols}))
    return (
        signal.get("universe_split") == split
        and signal.get("universe_symbols") == symbols
        and spec.get("input_identity", {}).get("universe") == expected_digest
    )


def _candidate_prices_are_canonical(candidates: CandidateSet) -> bool:
    price_digests: dict[tuple[str, int], str] = {}
    for member_hash in candidates.members:
        member = Result.open(candidates.study, member_hash)
        if not isinstance(member, BacktestResult):
            return False
        training_spec = member.lineage()["training_spec"]
        label = training_spec.get("label")
        if not isinstance(label, str) or not label:
            return False
        warmup = strategy_warmup_periods(member.spec())
        cache_key = (label, warmup)
        if cache_key not in price_digests:
            prices = load_backtest_prices_for(
                candidates.study.case_study,
                label,
                split="validation",
                warmup_periods=warmup,
            )
            price_digests[cache_key] = value_digest(prices)
        if member.spec().get("input_identity", {}).get("prices") != price_digests[cache_key]:
            return False
    return True


class Lifecycle:
    def __init__(self, study: Study) -> None:
        self.study = study

    @property
    def state(self) -> str:
        db_path = self.study.root / "run_log" / "registry.db"
        if not db_path.exists():
            return LifecycleState.DEVELOPMENT.value
        with closing(sqlite3.connect(db_path)) as db:
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
        holdout_training_spec: dict[str, Any],
    ) -> ResearchLock:
        self.study.require_writable()
        self.study.activate()
        candidates = CandidateSet.open(self.study, candidate_set_hash)
        if candidates.member_kind != "backtest" or selected_backtest_hash not in candidates.members:
            raise ValueError("selected backtest must be an exact member of the candidate set")
        if not _candidate_prices_are_canonical(candidates):
            raise ValueError("research lock requires canonical validation prices for every member")
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
        if not _strategy_roster_is_resolved(
            selected.spec(), self.study.case_study, prediction.hash
        ):
            raise ValueError("selected strategy does not contain its resolved validation roster")
        prediction_record = prediction.registry_record()
        training = Result.open(self.study, prediction_record["training_hash"])
        assert isinstance(training, TrainingResult)
        training_record = training.registry_record()
        training_spec = training.spec()
        training_computation = training_spec.get("computation", training_spec)
        locked_holdout_spec = _locked_training_spec(training_spec, holdout_training_spec)
        holdout_training_hash = training_hash_from_spec(locked_holdout_spec)
        lock_record = {
            "candidate_set_hash": candidates.hash,
            "selection_evidence": selection_evidence,
            "label": training_spec.get("label"),
            "label_artifact": training_computation.get("label_artifact"),
            "feature_artifacts": training_computation.get("feature_artifacts"),
            "cv": training_computation.get("cv"),
            "training_hash": training.hash,
            "holdout_training_hash": holdout_training_hash,
            "holdout_training_spec": locked_holdout_spec,
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
            existing_lock = db.execute(
                "SELECT lock_hash, state FROM research_locks LIMIT 1"
            ).fetchone()
            if existing_lock is not None:
                if existing_lock[0] != lock_hash:
                    raise ValueError("lifecycle already contains a different research lock")
                return ResearchLock(self.study, lock_hash, existing_lock[1], lock_record)
            db.execute("BEGIN IMMEDIATE")
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
        with closing(sqlite3.connect(self.study.root / "run_log" / "registry.db")) as db:
            row = db.execute(
                "SELECT lock_json, state FROM research_locks WHERE lock_hash = ?", (lock_hash,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown research lock {lock_hash!r}")
        return ResearchLock(self.study, lock_hash, row[1], json.loads(row[0]))

    def _validated_holdout_lineage(
        self,
        lock_hash: str,
        *,
        holdout_training_hash: str,
        holdout_prediction_hash: str,
        holdout_backtest_hash: str,
    ) -> tuple[ResearchLock, TrainingResult, PredictionResult, BacktestResult, str]:
        lock = self.open(lock_hash)
        if lock.state != LifecycleState.LOCKED.value:
            raise ValueError("holdout evaluation requires a LOCKED research lock")
        training = Result.open(self.study, holdout_training_hash)
        prediction = Result.open(self.study, holdout_prediction_hash)
        backtest = Result.open(self.study, holdout_backtest_hash)
        canonical_holdout_prices = load_backtest_prices_for(
            self.study.case_study,
            str(lock.record["label"]),
            split="holdout",
            warmup_periods=strategy_warmup_periods(lock.record["strategy_spec"]),
        )
        canonical_price_digest = value_digest(canonical_holdout_prices)
        valid = (
            isinstance(training, TrainingResult)
            and training.complete
            and training.execution_tier == "canonical"
            and training.hash == lock.record["holdout_training_hash"]
            and project_training_identity(training.spec()) == lock.record["holdout_training_spec"]
            and isinstance(prediction, PredictionResult)
            and prediction.complete
            and prediction.execution_tier == "canonical"
            and prediction.registry_record()["split"] == "holdout"
            and prediction.registry_record()["training_hash"] == training.hash
            and prediction.registry_record()["checkpoint_kind"] == lock.record["checkpoint_kind"]
            and prediction.registry_record()["checkpoint_value"] == lock.record["checkpoint_value"]
            and isinstance(backtest, BacktestResult)
            and backtest.complete
            and backtest.execution_tier == "canonical"
            and backtest.registry_record()["prediction_hash"] == prediction.hash
            and backtest.spec().get("input_identity", {}).get("prices") == canonical_price_digest
            and _strategy_roster_is_resolved(
                backtest.spec(), self.study.case_study, prediction.hash
            )
            and _locked_strategy_projection(backtest.spec())
            == _locked_strategy_projection(lock.record["strategy_spec"])
        )
        if not valid:
            raise ValueError(
                "holdout transition requires the exact complete canonical lineage in the lock"
            )
        assert isinstance(training, TrainingResult)
        assert isinstance(prediction, PredictionResult)
        assert isinstance(backtest, BacktestResult)
        lineage = {
            "lock_hash": lock_hash,
            "holdout_training_hash": training.hash,
            "holdout_prediction_hash": prediction.hash,
            "holdout_backtest_hash": backtest.hash,
        }
        return lock, training, prediction, backtest, compute_hash(canonical_json(lineage))

    def stage_holdout(
        self,
        lock_hash: str,
        *,
        holdout_training_hash: str,
        holdout_prediction_hash: str,
        holdout_backtest_hash: str,
    ) -> ResearchLock:
        self.study.require_writable()
        self.study.activate()
        lock, training, prediction, backtest, lineage_digest = self._validated_holdout_lineage(
            lock_hash,
            holdout_training_hash=holdout_training_hash,
            holdout_prediction_hash=holdout_prediction_hash,
            holdout_backtest_hash=holdout_backtest_hash,
        )
        db = _open_registry(self.study.root)
        try:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT holdout_training_hash, holdout_prediction_hash, "
                "holdout_backtest_hash, lineage_digest FROM holdout_staging WHERE lock_hash = ?",
                (lock_hash,),
            ).fetchone()
            expected = (training.hash, prediction.hash, backtest.hash, lineage_digest)
            if existing is not None and existing != expected:
                raise ValueError("immutable staged holdout lineage conflict")
            if existing is None:
                db.execute(
                    "INSERT INTO holdout_staging "
                    "(lock_hash, holdout_training_hash, holdout_prediction_hash, "
                    "holdout_backtest_hash, lineage_digest, staged_at) VALUES (?,?,?,?,?,?)",
                    (lock_hash, *expected[:3], lineage_digest, _utc_now()),
                )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return lock

    def finalize_holdout(self, lock_hash: str) -> ResearchLock:
        self.study.require_writable()
        self.study.activate()
        with closing(sqlite3.connect(self.study.root / "run_log" / "registry.db")) as db:
            staged = db.execute(
                "SELECT holdout_training_hash, holdout_prediction_hash, "
                "holdout_backtest_hash, lineage_digest FROM holdout_staging WHERE lock_hash = ?",
                (lock_hash,),
            ).fetchone()
        if staged is None:
            raise ValueError("holdout artifacts must be staged before finalization")
        _, training, prediction, backtest, lineage_digest = self._validated_holdout_lineage(
            lock_hash,
            holdout_training_hash=staged[0],
            holdout_prediction_hash=staged[1],
            holdout_backtest_hash=staged[2],
        )
        if lineage_digest != staged[3]:
            raise ValueError("staged holdout lineage digest does not validate")

        db = _open_registry(self.study.root)
        try:
            db.execute("BEGIN IMMEDIATE")
            current_staged = db.execute(
                "SELECT holdout_training_hash, holdout_prediction_hash, "
                "holdout_backtest_hash, lineage_digest FROM holdout_staging WHERE lock_hash = ?",
                (lock_hash,),
            ).fetchone()
            if current_staged != staged:
                raise ValueError("staged holdout lineage changed before finalization")
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

    def record_holdout(
        self,
        lock_hash: str,
        *,
        holdout_training_hash: str,
        holdout_prediction_hash: str,
        holdout_backtest_hash: str,
    ) -> ResearchLock:
        self.stage_holdout(
            lock_hash,
            holdout_training_hash=holdout_training_hash,
            holdout_prediction_hash=holdout_prediction_hash,
            holdout_backtest_hash=holdout_backtest_hash,
        )
        return self.finalize_holdout(lock_hash)
