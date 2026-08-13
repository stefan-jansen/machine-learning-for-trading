from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from case_studies.utils.registry import register_prediction_set, register_training_run

from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .workspace import Study


def _record(db: sqlite3.Connection, query: str, params: tuple) -> dict[str, Any] | None:
    cursor = db.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(zip((column[0] for column in cursor.description), row, strict=True))


@dataclass(frozen=True)
class Result:
    study: Study
    hash: str
    kind: str
    execution_tier: str
    identity_version: int | None

    @classmethod
    def open(
        cls,
        study: Study,
        result_hash: str,
        *,
        include_preview: bool = False,
    ) -> Result:
        roots = [(study.root, ExecutionTier.CANONICAL.value)]
        if include_preview and not study.read_only and study.output_root is not None:
            roots.append(
                (study.output_root / ".preview" / study.case_study, ExecutionTier.PREVIEW.value)
            )
        for root, namespace in roots:
            db_path = root / "run_log" / "registry.db"
            if not db_path.exists():
                continue
            with sqlite3.connect(db_path) as db:
                tables = {
                    row[0]
                    for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if "training_runs" not in tables:
                    continue
                training = _record(
                    db,
                    "SELECT identity_version, execution_tier FROM training_runs "
                    "WHERE training_hash = ?",
                    (result_hash,),
                )
                if training is not None:
                    tier = training["execution_tier"] or namespace
                    return TrainingResult(
                        study, result_hash, "training", tier, training["identity_version"]
                    )
                prediction = _record(
                    db,
                    """
                    SELECT t.identity_version, t.execution_tier
                    FROM prediction_sets p
                    JOIN training_runs t ON t.training_hash = p.training_hash
                    WHERE p.prediction_hash = ?
                    """,
                    (result_hash,),
                )
                if prediction is not None:
                    tier = prediction["execution_tier"] or namespace
                    return PredictionResult(
                        study, result_hash, "prediction", tier, prediction["identity_version"]
                    )
                backtest = _record(
                    db,
                    """
                    SELECT t.identity_version, t.execution_tier
                    FROM backtest_runs b
                    JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
                    JOIN training_runs t ON t.training_hash = p.training_hash
                    WHERE b.backtest_hash = ?
                    """,
                    (result_hash,),
                )
                if backtest is not None:
                    tier = backtest["execution_tier"] or namespace
                    return BacktestResult(
                        study, result_hash, "backtest", tier, backtest["identity_version"]
                    )
        raise KeyError(f"Unknown result hash {result_hash!r}")

    @property
    def root(self) -> Path:
        return self.study.storage_root(self.execution_tier)

    @property
    def complete(self) -> bool:
        return False

    def registry_record(self) -> dict[str, Any]:
        table, key = {
            "training": ("training_runs", "training_hash"),
            "prediction": ("prediction_sets", "prediction_hash"),
            "backtest": ("backtest_runs", "backtest_hash"),
        }[self.kind]
        with sqlite3.connect(self.root / "run_log" / "registry.db") as db:
            record = _record(db, f"SELECT * FROM {table} WHERE {key} = ?", (self.hash,))
        assert record is not None
        return record

    def spec(self) -> dict[str, Any]:
        record = self.registry_record()
        if self.kind == "prediction":
            return {
                "training_hash": record["training_hash"],
                "checkpoint_kind": record["checkpoint_kind"],
                "checkpoint_value": record["checkpoint_value"],
                "split": record["split"],
            }
        return json.loads(record.get("spec_json") or "{}")

    def artifacts(self) -> tuple[Path, ...]:
        directory = (
            self.root
            / "run_log"
            / {
                "training": "training",
                "prediction": "predictions",
                "backtest": "backtest",
            }[self.kind]
            / self.hash
        )
        if not directory.exists():
            return ()
        return tuple(sorted(path for path in directory.rglob("*") if path.is_file()))

    def lineage(self) -> dict[str, Any]:
        if self.kind == "training":
            return {"training_hash": self.hash, "training_spec": self.spec()}
        record = self.registry_record()
        if self.kind == "prediction":
            training = Result.open(
                self.study,
                record["training_hash"],
                include_preview=self.execution_tier == ExecutionTier.PREVIEW.value,
            )
            return {
                "training_hash": training.hash,
                "training_spec": training.spec(),
                "prediction_hash": self.hash,
            }
        prediction = Result.open(
            self.study,
            record["prediction_hash"],
            include_preview=self.execution_tier == ExecutionTier.PREVIEW.value,
        )
        return {
            **prediction.lineage(),
            "backtest_hash": self.hash,
            "strategy_spec": self.spec(),
        }

    def protocol(self) -> dict[str, Any]:
        lineage = self.lineage()
        training = lineage["training_spec"]
        split = None
        if self.kind == "prediction":
            split = self.registry_record()["split"]
        elif self.kind == "backtest":
            prediction = Result.open(
                self.study,
                self.registry_record()["prediction_hash"],
                include_preview=self.execution_tier == ExecutionTier.PREVIEW.value,
            )
            split = prediction.registry_record()["split"]
        return {
            "label_artifact": training.get("label_artifact"),
            "feature_artifacts": training.get("feature_artifacts"),
            "cv": training.get("cv"),
            "split": split,
            "execution_tier": self.execution_tier,
        }


@dataclass(frozen=True)
class TrainingResult(Result):
    @property
    def complete(self) -> bool:
        return self.identity_version == 2 and bool(self.spec())


@dataclass(frozen=True)
class PredictionResult(Result):
    def coverage(self) -> dict[str, Any] | None:
        with sqlite3.connect(self.root / "run_log" / "registry.db") as db:
            return _record(
                db,
                "SELECT * FROM prediction_coverage WHERE prediction_hash = ?",
                (self.hash,),
            )

    @property
    def complete(self) -> bool:
        coverage = self.coverage()
        prediction_file = self.root / "run_log" / "predictions" / self.hash / "predictions.parquet"
        return bool(coverage and coverage["status"] == "complete" and prediction_file.is_file())

    def load(self):
        import polars as pl

        path = self.root / "run_log" / "predictions" / self.hash / "predictions.parquet"
        return pl.read_parquet(path)


@dataclass(frozen=True)
class BacktestResult(Result):
    @property
    def complete(self) -> bool:
        record = self.registry_record()
        prediction = Result.open(
            self.study,
            record["prediction_hash"],
            include_preview=self.execution_tier == ExecutionTier.PREVIEW.value,
        )
        if not isinstance(prediction, PredictionResult) or not prediction.complete:
            return False
        returns = self.root / "run_log" / "backtest" / self.hash / "daily_returns.parquet"
        with sqlite3.connect(self.root / "run_log" / "registry.db") as db:
            metrics = db.execute(
                "SELECT 1 FROM backtest_metrics WHERE backtest_hash = ?", (self.hash,)
            ).fetchone()
        return returns.is_file() and metrics is not None


class ResultsCatalog:
    def __init__(self, study: Study) -> None:
        self.study = study

    def register_training(
        self,
        spec: dict[str, Any],
        *,
        execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL,
        runtime_provenance: dict[str, Any] | None = None,
    ) -> TrainingResult:
        self.study.require_writable()
        tier = ExecutionTier(execution_tier)
        resolved = dict(spec)
        resolved.setdefault("identity_version", 2)
        resolved.setdefault("execution_tier", tier.value)
        if resolved["identity_version"] != 2 or resolved["execution_tier"] != tier.value:
            raise ValueError(
                "training spec identity version or execution tier conflicts with request"
            )
        if tier is ExecutionTier.PREVIEW and not resolved.get("preview_reductions"):
            raise ValueError("preview training specs must identity-cover every preview reduction")
        if tier is ExecutionTier.CANONICAL and resolved.get("preview_reductions"):
            raise ValueError("canonical training specs cannot contain preview reductions")
        case_dir = self.study.activate(tier)
        training_hash = register_training_run(
            self.study.case_study,
            resolved,
            case_dir=case_dir,
            runtime_provenance=runtime_provenance,
        )
        result = Result.open(
            self.study,
            training_hash,
            include_preview=tier is ExecutionTier.PREVIEW,
        )
        assert isinstance(result, TrainingResult)
        return result

    def publish_predictions(
        self,
        training: TrainingResult,
        *,
        checkpoint_kind: str,
        checkpoint_value: int | None,
        split: str,
        predictions,
        expected_keys,
        allow_partial: bool = False,
    ) -> PredictionResult:
        self.study.require_writable()
        if training.study != self.study or training.kind != "training":
            raise ValueError("training result belongs to another study")
        tier = ExecutionTier(training.execution_tier)
        case_dir = self.study.activate(tier)
        prediction_hash = register_prediction_set(
            self.study.case_study,
            training.hash,
            checkpoint_kind=checkpoint_kind,
            checkpoint_value=checkpoint_value,
            split=split,
            predictions=predictions,
            expected_keys=expected_keys,
            allow_partial=allow_partial,
            case_dir=case_dir,
        )
        result = Result.open(
            self.study,
            prediction_hash,
            include_preview=tier is ExecutionTier.PREVIEW,
        )
        assert isinstance(result, PredictionResult)
        return result

    def open(self, result_hash: str, *, include_preview: bool = False) -> Result:
        return Result.open(self.study, result_hash, include_preview=include_preview)
