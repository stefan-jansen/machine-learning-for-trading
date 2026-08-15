from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.registry.specs import (
    canonical_json,
    canonical_value,
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

    def reopen(self) -> ResearchLock:
        reopened = self.study.lifecycle.open(self.hash)
        if reopened.record != self.record:
            raise ValueError("research lock object differs from its immutable registry record")
        return reopened


def _canonical_fold_value(value: Any) -> str:
    return json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"))


def _fold_keyed_parameters(computation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    model = computation.get("model")
    if isinstance(model, dict) and isinstance(model.get("effective_params_by_fold"), dict):
        found.append(("model parameters", model["effective_params_by_fold"]))
    task = computation.get("task")
    if isinstance(task, dict):
        imbalance = task.get("imbalance")
        if isinstance(imbalance, dict) and isinstance(
            imbalance.get("effective_class_weights_by_fold"), dict
        ):
            found.append(("class weights", imbalance["effective_class_weights_by_fold"]))
    return found


def _require_consistent_fold_parameters(
    validation_computation: dict[str, Any],
    holdout_computation: dict[str, Any],
) -> None:
    """Reject a holdout spec whose per-fold parameters contradict the selected training.

    These dicts are keyed by fold id, so the holdout's single entry can never equal the
    validation's several and the comparison below has to drop them. Dropping them outright
    would let a holdout spec declare any parameters at all - a different alpha locks and
    then finalizes as the selected candidate's holdout result.

    Only the homogeneous case is decidable here. When every validation fold resolved to the
    same values, nothing about that configuration is derived from a fold, so the holdout
    fold must resolve to those values too. When the validation folds disagree, the values
    demonstrably depend on the fold's training data and the holdout's legitimately differs;
    which keys may move is family-specific and is checked by each adapter's
    ``reconstruct_locked_request``.
    """
    validation = dict(_fold_keyed_parameters(validation_computation))
    for name, holdout_by_fold in _fold_keyed_parameters(holdout_computation):
        validation_by_fold = validation.get(name)
        if not validation_by_fold or len(holdout_by_fold) != 1:
            continue
        # Not canonical_json: a fold entry is a dict of parameters for one family and a
        # list of class weights for another, and canonical_json rejects anything but a
        # dict. Both sides go through canonical_value so a legacy spec, which
        # project_training_identity deep-copies rather than canonicalizing, does not
        # compare unequal to its canonicalized counterpart over a tuple or a float.
        distinct = {_canonical_fold_value(value): value for value in validation_by_fold.values()}
        if len(distinct) != 1:
            continue
        actual = next(iter(holdout_by_fold.values()))
        expected_key, expected = next(iter(distinct.items()))
        if _canonical_fold_value(actual) != expected_key:
            raise ValueError(
                f"locked holdout {name} differ from the selected training, which resolved "
                f"identically on every validation fold: {actual!r} != {expected!r}"
            )


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
    if "computation" in projected_holdout:
        _require_consistent_fold_parameters(validation_computation, holdout_computation)
        for computation in (validation_computation, holdout_computation):
            computation.pop("expected_prediction_keys", None)
            model = computation.get("model")
            if isinstance(model, dict):
                model.pop("effective_params_by_fold", None)
            task = computation.get("task")
            if isinstance(task, dict):
                imbalance = task.get("imbalance")
                if isinstance(imbalance, dict):
                    imbalance.pop("effective_class_weights_by_fold", None)
            macro = computation.get("macro_context")
            if isinstance(macro, dict):
                macro.pop("resolved_fold_digest", None)
    if projected_holdout != projected_validation:
        raise ValueError(
            "holdout retraining may differ from selected training only in CV interval "
            "and its derived parameters or eligibility"
        )
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
        input_identity.pop("funding_rates", None)
    decision = projected.get("decision_artifact")
    if isinstance(decision, dict):
        decision.pop("hash", None)
        decision.pop("artifact_digest", None)
        source_identity = decision.get("source_identity")
        if isinstance(source_identity, dict):
            declared_inputs = source_identity.get("declared_inputs")
            if isinstance(declared_inputs, dict):
                declared_inputs.pop("prediction_hashes", None)
                declared_inputs.pop("prices", None)
            source_identity.pop("clean_replay_digest", None)
    return projected


def _valid_holdout_decision(
    study: Study,
    locked_spec: dict[str, Any],
    holdout_spec: dict[str, Any],
    prediction_hash: str,
) -> bool:
    locked = locked_spec.get("decision_artifact")
    holdout = holdout_spec.get("decision_artifact")
    if locked is None:
        return holdout is None
    if not isinstance(holdout, dict) or not holdout.get("canonical"):
        return False
    from .decisions import DecisionArtifact

    try:
        artifact = DecisionArtifact.open(study, str(holdout["hash"]))
        artifact.load()
    except (KeyError, OSError, ValueError):
        return False
    valid = (
        artifact.hash == holdout.get("hash")
        and artifact.kind == holdout.get("kind")
        and artifact.canonical
        and artifact.spec["prediction_hashes"] == [prediction_hash]
        and artifact.spec["artifact_digest"] == holdout.get("artifact_digest")
        and artifact.spec["source_identity"] == holdout.get("source_identity")
        and artifact.spec["state_transition_policy"] == holdout.get("state_transition_policy")
    )
    for name in ("decision_keys", "parameters"):
        if name in holdout:
            valid = valid and artifact.spec.get(name) == holdout[name]
    return valid


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
        serialized_lock = canonical_json(lock_record)
        lock_record = json.loads(serialized_lock)
        lock_hash = compute_hash(serialized_lock)
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
                    serialized_lock,
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
        record = json.loads(row[0])
        if compute_hash(canonical_json(record)) != lock_hash:
            raise ValueError(f"research lock digest mismatch for {lock_hash!r}")
        return ResearchLock(self.study, lock_hash, row[1], record)

    def holdout_lineage(self, lock_hash: str) -> dict[str, str]:
        lock = self.open(lock_hash)
        if lock.state != LifecycleState.HOLDOUT_EVALUATED.value:
            raise ValueError("holdout lineage is available only after evaluation")
        with closing(sqlite3.connect(self.study.root / "run_log" / "registry.db")) as db:
            row = db.execute(
                "SELECT holdout_training_hash, holdout_prediction_hash, "
                "holdout_backtest_hash, fitted_state_digest "
                "FROM holdout_evaluations WHERE lock_hash = ?",
                (lock_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("evaluated research lock has no finalized holdout lineage")
        lineage = {
            "lock_hash": lock_hash,
            "holdout_training_hash": row[0],
            "holdout_prediction_hash": row[1],
            "holdout_backtest_hash": row[2],
        }
        if row[3]:
            lineage["fitted_state_digest"] = row[3]
        return lineage

    def _validated_holdout_lineage(
        self,
        lock_hash: str,
        *,
        holdout_training_hash: str,
        holdout_prediction_hash: str,
        holdout_backtest_hash: str,
    ) -> tuple[ResearchLock, TrainingResult, PredictionResult, BacktestResult]:
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
        backtest_spec = backtest.spec() if isinstance(backtest, BacktestResult) else {}
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
            and backtest_spec.get("input_identity", {}).get("prices") == canonical_price_digest
            and _valid_holdout_decision(
                self.study,
                lock.record["strategy_spec"],
                backtest_spec,
                prediction.hash,
            )
            and _locked_strategy_projection(backtest_spec)
            == _locked_strategy_projection(lock.record["strategy_spec"])
        )
        if not valid:
            raise ValueError(
                "holdout transition requires the exact complete canonical lineage in the lock"
            )
        assert isinstance(training, TrainingResult)
        assert isinstance(prediction, PredictionResult)
        assert isinstance(backtest, BacktestResult)
        return lock, training, prediction, backtest

    def stage_holdout(
        self,
        lock_hash: str,
        *,
        holdout_training_hash: str,
        holdout_prediction_hash: str,
        holdout_backtest_hash: str,
        fitted_state_digest: str | None = None,
    ) -> ResearchLock:
        self.study.require_writable()
        self.study.activate()
        lock, training, prediction, backtest = self._validated_holdout_lineage(
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
                "holdout_backtest_hash, fitted_state_digest, lineage_digest "
                "FROM holdout_staging WHERE lock_hash = ?",
                (lock_hash,),
            ).fetchone()
            if fitted_state_digest is not None and not fitted_state_digest:
                raise ValueError("fitted-state digest cannot be empty")
            lineage = {
                "lock_hash": lock_hash,
                "holdout_training_hash": training.hash,
                "holdout_prediction_hash": prediction.hash,
                "holdout_backtest_hash": backtest.hash,
            }
            if fitted_state_digest is not None:
                lineage["fitted_state_digest"] = fitted_state_digest
            lineage_digest = compute_hash(canonical_json(lineage))
            expected = (
                training.hash,
                prediction.hash,
                backtest.hash,
                fitted_state_digest,
                lineage_digest,
            )
            if existing is not None and existing != expected:
                raise ValueError("immutable staged holdout lineage conflict")
            if existing is None:
                db.execute(
                    "INSERT INTO holdout_staging "
                    "(lock_hash, holdout_training_hash, holdout_prediction_hash, "
                    "holdout_backtest_hash, fitted_state_digest, lineage_digest, staged_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (lock_hash, *expected, _utc_now()),
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
                "holdout_backtest_hash, fitted_state_digest, lineage_digest "
                "FROM holdout_staging WHERE lock_hash = ?",
                (lock_hash,),
            ).fetchone()
        if staged is None:
            raise ValueError("holdout artifacts must be staged before finalization")
        _, training, prediction, backtest = self._validated_holdout_lineage(
            lock_hash,
            holdout_training_hash=staged[0],
            holdout_prediction_hash=staged[1],
            holdout_backtest_hash=staged[2],
        )
        lineage = {
            "lock_hash": lock_hash,
            "holdout_training_hash": training.hash,
            "holdout_prediction_hash": prediction.hash,
            "holdout_backtest_hash": backtest.hash,
        }
        if staged[3] is not None:
            lineage["fitted_state_digest"] = staged[3]
        lineage_digest = compute_hash(canonical_json(lineage))
        if lineage_digest != staged[4]:
            raise ValueError("staged holdout lineage digest does not validate")

        db = _open_registry(self.study.root)
        try:
            db.execute("BEGIN IMMEDIATE")
            current_staged = db.execute(
                "SELECT holdout_training_hash, holdout_prediction_hash, "
                "holdout_backtest_hash, fitted_state_digest, lineage_digest "
                "FROM holdout_staging WHERE lock_hash = ?",
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
                "holdout_backtest_hash, fitted_state_digest, evaluated_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    lock_hash,
                    training.hash,
                    prediction.hash,
                    backtest.hash,
                    staged[3],
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
