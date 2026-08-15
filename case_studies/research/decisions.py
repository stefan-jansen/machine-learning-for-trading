from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.registry.specs import (
    IDENTITY_VERSION,
    canonical_json,
    canonical_value,
    compute_hash,
)
from case_studies.utils.registry.store import _open_registry, _utc_now

if TYPE_CHECKING:
    from .workspace import Study

from .results import PredictionResult, Result

Transition = Literal["continue", "reset", "liquidate"]


@dataclass(frozen=True)
class StateTransitionPolicy:
    fold_boundary: Transition
    temporal_gap: Transition

    def __post_init__(self) -> None:
        allowed = {"continue", "reset", "liquidate"}
        if self.fold_boundary not in allowed or self.temporal_gap not in allowed:
            raise ValueError(f"state transitions must be one of {sorted(allowed)}")


def _decision_entity_key(decisions: pl.DataFrame) -> str:
    entity_keys = [column for column in ("symbol", "product") if column in decisions.columns]
    if len(entity_keys) != 1:
        raise ValueError("decision artifacts require exactly one entity key: symbol or product")
    return entity_keys[0]


def _validate_frame(kind: str, decisions: pl.DataFrame) -> tuple[str, ...]:
    value_column = {
        "target_weights": "weight",
        "target_positions": "position",
        "orders": "quantity",
        "short_straddles": "weight",
    }.get(kind)
    if value_column is None:
        raise ValueError("unsupported decision artifact kind")
    keys = (_decision_entity_key(decisions), "timestamp")
    required = {*keys, value_column}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"decision artifact is missing columns: {sorted(missing)}")
    selected = decisions.select(*keys, value_column)
    if selected.null_count().row(0) != (0, 0, 0):
        raise ValueError("decision artifacts cannot contain null keys or values")
    if selected.n_unique(list(keys)) != selected.height:
        raise ValueError("decision artifact keys must be unique")
    values = selected.get_column(value_column).cast(pl.Float64)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("decision artifact values must be finite")
    fold_columns = [column for column in ("fold", "fold_id") if column in decisions.columns]
    if len(fold_columns) > 1:
        raise ValueError("decision artifacts cannot contain both fold and fold_id")
    if fold_columns and decisions.get_column(fold_columns[0]).null_count():
        raise ValueError("decision artifact fold values cannot be null")
    if kind == "short_straddles":
        contract_columns = {
            "strike",
            "expiration",
            "entry_date",
            "entry_straddle_mid",
            "entry_call_mid",
            "entry_call_bid",
            "entry_call_ask",
            "entry_put_mid",
            "entry_put_bid",
            "entry_put_ask",
        }
        missing_contract = contract_columns - set(decisions.columns)
        if missing_contract:
            raise ValueError(
                f"short-straddle decisions are missing columns: {sorted(missing_contract)}"
            )
        if decisions.select(*sorted(contract_columns)).null_count().sum_horizontal().item():
            raise ValueError("short-straddle decisions cannot contain null contract fields")
        invalid_dates = decisions.filter(
            (pl.col("entry_date").cast(pl.Date) <= pl.col("timestamp").cast(pl.Date))
            | (pl.col("expiration").cast(pl.Date) < pl.col("entry_date").cast(pl.Date))
        )
        if not invalid_dates.is_empty():
            raise ValueError("short-straddle entry and expiration dates are invalid")
        if decisions.filter(
            (pl.col("strike") <= 0)
            | (pl.col("entry_straddle_mid") <= 0)
            | (pl.col("entry_call_mid") <= 0)
            | (pl.col("entry_put_mid") <= 0)
            | (pl.col("entry_call_bid") < 0)
            | (pl.col("entry_put_bid") < 0)
            | (pl.col("entry_call_ask") < pl.col("entry_call_bid"))
            | (pl.col("entry_put_ask") < pl.col("entry_put_bid"))
            | (pl.col("weight") <= 0)
        ).height:
            raise ValueError("short-straddle decisions contain invalid quotes, strike, or weight")
        weight_sums = decisions.group_by("timestamp").agg(pl.col("weight").sum().alias("weight"))
        if weight_sums.filter((pl.col("weight") - 1.0).abs() > 1e-10).height:
            raise ValueError("short-straddle weights must sum to one per decision timestamp")
    return keys


def _validate_promotion(
    source_identity: dict[str, Any],
    prediction_hashes: tuple[str, ...],
) -> None:
    required = {
        "module",
        "source_digest",
        "declared_inputs",
        "determinism",
        "clean_replay_digest",
    }
    missing = required - set(source_identity)
    if missing:
        raise ValueError(f"canonical decision promotion is missing evidence: {sorted(missing)}")
    if not source_identity["module"] or not source_identity["source_digest"]:
        raise ValueError("canonical decision promotion requires stable importable source identity")
    if find_spec(str(source_identity["module"])) is None:
        raise ValueError("canonical decision promotion source module is not importable")
    if not source_identity["declared_inputs"]:
        raise ValueError("canonical decision promotion requires declared immutable inputs")
    declared = source_identity["declared_inputs"]
    if not isinstance(declared, dict) or declared.get("prediction_hashes") != list(
        prediction_hashes
    ):
        raise ValueError(
            "canonical decision promotion must disclose the exact prediction_hashes input"
        )
    determinism = source_identity["determinism"]
    if not isinstance(determinism, dict) or not (
        determinism.get("deterministic") or determinism.get("seed") is not None
    ):
        raise ValueError("canonical decision promotion requires deterministic or seeded execution")


@dataclass(frozen=True)
class DecisionArtifact:
    study: Study
    hash: str
    kind: str
    spec: dict[str, Any]
    canonical: bool
    root: Path

    @property
    def path(self) -> Path:
        return self.root / "run_log" / "decisions" / self.hash / "decisions.parquet"

    @classmethod
    def publish(
        cls,
        study: Study,
        *,
        kind: str,
        decisions: pl.DataFrame,
        prediction_hashes: tuple[str, ...] | list[str],
        parameters: dict[str, Any],
        source_identity: dict[str, Any] | None = None,
        state_transition_policy: StateTransitionPolicy | None = None,
        canonical: bool = False,
    ) -> DecisionArtifact:
        study.require_writable()
        if not isinstance(decisions, pl.DataFrame):
            raise TypeError("decision publication requires a Polars DataFrame")
        key_columns = _validate_frame(kind, decisions)
        if kind in {"target_positions", "orders"} and state_transition_policy is None:
            raise ValueError(f"{kind} decisions require a state transition policy")
        lineage = tuple(dict.fromkeys(prediction_hashes))
        if not lineage or len(lineage) != len(prediction_hashes):
            raise ValueError("decision prediction lineage must be non-empty and unique")
        prediction_results = []
        for prediction_hash in lineage:
            result = Result.open(study, prediction_hash, include_preview=not canonical)
            if not isinstance(result, PredictionResult):
                raise ValueError(f"decision lineage {prediction_hash!r} is not a prediction")
            if canonical and (
                result.identity_version != IDENTITY_VERSION
                or not result.complete
                or result.execution_tier != "canonical"
            ):
                raise ValueError("canonical decision lineage requires current complete predictions")
            prediction_results.append(result)
        tiers = {result.execution_tier for result in prediction_results}
        if len(tiers) != 1:
            raise ValueError("decision lineage cannot mix canonical and preview predictions")
        execution_tier = tiers.pop()
        normalized_source = canonical_value(source_identity or {})
        if canonical:
            _validate_promotion(normalized_source, lineage)
        normalized_decisions = decisions.sort(list(key_columns))
        spec = {
            "schema_version": 1,
            "kind": kind,
            "prediction_hashes": list(lineage),
            "parameters": canonical_value(parameters),
            "source_identity": normalized_source,
            "state_transition_policy": (
                asdict(state_transition_policy) if state_transition_policy is not None else None
            ),
            "decision_keys": list(key_columns),
            "artifact_digest": value_digest(normalized_decisions),
            "canonical": canonical,
            "execution_tier": execution_tier,
        }
        if canonical and normalized_source["clean_replay_digest"] != spec["artifact_digest"]:
            raise ValueError("canonical decision clean replay does not match the artifact digest")
        decision_hash = compute_hash(canonical_json(spec))
        storage_root = study.storage_root(execution_tier)
        artifact_dir = storage_root / "run_log" / "decisions" / decision_hash
        artifact = artifact_dir / "decisions.parquet"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary = artifact_dir / f".decisions.{uuid.uuid4().hex}.tmp"
        normalized_decisions.write_parquet(temporary)
        db = _open_registry(storage_root)
        created_artifact = False
        try:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT decision_kind, spec_json, artifact_digest, canonical "
                "FROM decision_artifacts WHERE decision_hash = ?",
                (decision_hash,),
            ).fetchone()
            expected = (kind, canonical_json(spec), spec["artifact_digest"], int(canonical))
            if existing is not None and existing != expected:
                raise ValueError(f"immutable decision identity conflict for {decision_hash}")
            if (
                artifact.is_file()
                and value_digest(pl.read_parquet(artifact)) != spec["artifact_digest"]
            ):
                raise ValueError(f"immutable decision artifact conflict for {decision_hash}")
            if existing is None:
                db.execute(
                    "INSERT INTO decision_artifacts "
                    "(decision_hash, decision_kind, spec_json, artifact_digest, canonical, "
                    "created_at) VALUES (?,?,?,?,?,?)",
                    (decision_hash, *expected, _utc_now()),
                )
            if not artifact.is_file():
                os.replace(temporary, artifact)
                created_artifact = True
            db.commit()
        except Exception:
            db.rollback()
            if created_artifact:
                artifact.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            db.close()
        return cls(study, decision_hash, kind, spec, canonical, storage_root)

    @classmethod
    def open(cls, study: Study, decision_hash: str) -> DecisionArtifact:
        roots = [study.root]
        if study.release_case_root != study.root:
            roots.append(study.release_case_root)
        if study.output_root is not None:
            roots.append(study.output_root / ".preview" / study.case_study)
        for root in roots:
            db_path = root / "run_log" / "registry.db"
            if not db_path.is_file():
                continue
            with sqlite3.connect(db_path) as db:
                table = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'decision_artifacts'"
                ).fetchone()
                if table is None:
                    continue
                row = db.execute(
                    "SELECT decision_kind, spec_json, canonical FROM decision_artifacts "
                    "WHERE decision_hash = ?",
                    (decision_hash,),
                ).fetchone()
            if row is not None:
                return cls(
                    study,
                    decision_hash,
                    row[0],
                    json.loads(row[1]),
                    bool(row[2]),
                    root,
                )
        raise KeyError(f"unknown decision artifact {decision_hash!r}")

    def load(self) -> pl.DataFrame:
        frame = pl.read_parquet(self.path)
        if value_digest(frame) != self.spec["artifact_digest"]:
            raise ValueError(f"decision artifact digest mismatch for {self.hash}")
        return frame
