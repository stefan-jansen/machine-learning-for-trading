from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.registry import register_prediction_set, register_training_run
from case_studies.utils.registry.specs import (
    IDENTITY_VERSION,
    SUPPORTED_IDENTITY_VERSIONS,
)

from .contracts import ExecutionTier

# Digest verification reads the artifact off disk, and `complete` is evaluated in loops
# over whole populations - CandidateSet.members and OfficialPopulation both re-check
# every member, and members is a property, so an unmemoized check re-reads two parquet
# files per member on every access. Published artifacts are immutable, and the key
# carries size and nanosecond mtime, so a file that is replaced misses the cache.
_VERIFIED_ARTIFACT_DIGESTS: dict[tuple[str, int, int], str] = {}


def _verified_digest(path: Path, load) -> str:
    from case_studies.utils.artifact_digest import value_digest

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    digest = _VERIFIED_ARTIFACT_DIGESTS.get(key)
    if digest is None:
        digest = value_digest(load())
        _VERIFIED_ARTIFACT_DIGESTS[key] = digest
    return digest


if TYPE_CHECKING:
    from .workspace import Study


def _record(db: sqlite3.Connection, query: str, params: tuple) -> dict[str, Any] | None:
    cursor = db.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(zip((column[0] for column in cursor.description), row, strict=True))


def _columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}


def _training_identity_projection(db: sqlite3.Connection, alias: str = "") -> str:
    columns = _columns(db, "training_runs")
    prefix = f"{alias}." if alias else ""
    identity = (
        f"{prefix}identity_version AS identity_version"
        if "identity_version" in columns
        else "NULL AS identity_version"
    )
    tier = (
        f"{prefix}execution_tier AS execution_tier"
        if "execution_tier" in columns
        else "NULL AS execution_tier"
    )
    return f"{identity}, {tier}"


def _stored_source(
    db: sqlite3.Connection,
    result_hash: str,
    result_kind: str,
    default: str,
) -> tuple[str, Path | None]:
    tables = {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if "overlay_references" not in tables:
        return default, None
    row = db.execute(
        "SELECT source_root FROM overlay_references WHERE result_hash = ? AND result_kind = ?",
        (result_hash, result_kind),
    ).fetchone()
    return ("released", Path(row[0]).resolve()) if row is not None else (default, None)


@dataclass(frozen=True)
class Result:
    study: Study
    hash: str
    kind: str
    execution_tier: str
    identity_version: int | None
    origin: str = "workspace"
    source_root: Path | None = None

    @classmethod
    def open(
        cls,
        study: Study,
        result_hash: str,
        *,
        include_preview: bool = False,
    ) -> Result:
        roots = []
        if not study.read_only:
            roots.append((study.root, ExecutionTier.CANONICAL.value, "workspace"))
        roots.append((study.release_case_root, ExecutionTier.CANONICAL.value, "released"))
        if include_preview and not study.read_only and study.output_root is not None:
            roots.append(
                (
                    study.output_root / ".preview" / study.case_study,
                    ExecutionTier.PREVIEW.value,
                    "workspace",
                )
            )
        for root, namespace, origin in roots:
            db_path = root / "run_log" / "registry.db"
            if not db_path.exists():
                continue
            with closing(sqlite3.connect(db_path)) as db:
                tables = {
                    row[0]
                    for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if "training_runs" not in tables:
                    continue
                identity_projection = _training_identity_projection(db)
                training = _record(
                    db,
                    f"SELECT {identity_projection} FROM training_runs WHERE training_hash = ?",
                    (result_hash,),
                )
                if training is not None:
                    tier = training["execution_tier"] or namespace
                    result_origin, source_root = _stored_source(db, result_hash, "training", origin)
                    return TrainingResult(
                        study,
                        result_hash,
                        "training",
                        tier,
                        training["identity_version"],
                        result_origin,
                        source_root,
                    )
                prediction = None
                if "prediction_sets" in tables:
                    identity_projection = _training_identity_projection(db, "t")
                    prediction = _record(
                        db,
                        f"""
                        SELECT {identity_projection}
                        FROM prediction_sets p
                        JOIN training_runs t ON t.training_hash = p.training_hash
                        WHERE p.prediction_hash = ?
                        """,
                        (result_hash,),
                    )
                if prediction is not None:
                    tier = prediction["execution_tier"] or namespace
                    result_origin, source_root = _stored_source(
                        db, result_hash, "prediction", origin
                    )
                    return PredictionResult(
                        study,
                        result_hash,
                        "prediction",
                        tier,
                        prediction["identity_version"],
                        result_origin,
                        source_root,
                    )
                backtest = None
                if {"prediction_sets", "backtest_runs"} <= tables:
                    identity_projection = _training_identity_projection(db, "t")
                    backtest = _record(
                        db,
                        f"""
                        SELECT {identity_projection}
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
                        study,
                        result_hash,
                        "backtest",
                        tier,
                        backtest["identity_version"],
                        origin,
                    )
        raise KeyError(f"Unknown result hash {result_hash!r}")

    @property
    def root(self) -> Path:
        if self.source_root is not None:
            return self.source_root
        if self.origin == "released":
            return self.study.release_case_root
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
        with closing(sqlite3.connect(self.root / "run_log" / "registry.db")) as db:
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
        computation = training.get("computation", training)
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
            "label_artifact": computation.get("label_artifact"),
            "feature_artifacts": computation.get("feature_artifacts"),
            "cv": computation.get("cv"),
            "split": split,
            "execution_tier": self.execution_tier,
        }


@dataclass(frozen=True)
class TrainingResult(Result):
    def fitted_states(self) -> list[Any]:
        """The per-fold fitted state this run stored, in fold order.

        A run writes what its family needs to reproduce a prediction without refitting, and the
        shape is the family's own: the linear runner stores a mapping with `model`,
        `preprocessor` and `feature_names`. This returns those objects unchanged rather than
        interpreting them, because a caller that asks for fitted state already knows which family
        it asked about. It is the supported way to read them - the layout under
        `run_log/training/<hash>/models/` is an implementation detail, and a notebook that opens
        those files itself is asserting something the registry has not been asked to confirm.
        """
        import joblib

        models = self.root / "run_log" / "training" / self.hash / "models"
        # Fold order, not filename order: a lexicographic sort puts fold_10 before fold_2, and
        # `us_equities_panel` declares sixteen splits.
        paths = sorted(models.glob("fold_*.joblib"), key=lambda path: int(path.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(
                f"training run {self.hash} stored no fitted state under {models}"
            )
        return [joblib.load(path) for path in paths]

    @property
    def complete(self) -> bool:
        if self.identity_version not in SUPPORTED_IDENTITY_VERSIONS or not self.spec():
            return False
        spec_path = self.root / "run_log" / "training" / self.hash / "spec.json"
        try:
            if json.loads(spec_path.read_text()) != self.spec():
                return False
        except (OSError, json.JSONDecodeError):
            return False
        with closing(sqlite3.connect(self.root / "run_log" / "registry.db")) as db:
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            completed_attempt = (
                db.execute(
                    "SELECT 1 FROM execution_attempts "
                    "WHERE scientific_identity = ? AND status = 'completed' LIMIT 1",
                    (self.hash,),
                ).fetchone()
                if "execution_attempts" in tables
                else None
            )
            prediction_hashes = (
                [
                    row[0]
                    for row in db.execute(
                        "SELECT prediction_hash FROM prediction_sets WHERE training_hash = ?",
                        (self.hash,),
                    ).fetchall()
                ]
                if "prediction_sets" in tables
                else []
            )
        if completed_attempt is not None:
            return True
        return any(
            isinstance(
                result := Result.open(
                    self.study,
                    prediction_hash,
                    include_preview=self.execution_tier == ExecutionTier.PREVIEW.value,
                ),
                PredictionResult,
            )
            and result.complete
            for prediction_hash in prediction_hashes
        )


@dataclass(frozen=True)
class PredictionResult(Result):
    def coverage(self) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.root / "run_log" / "registry.db")) as db:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'prediction_coverage'"
            ).fetchone()
            if exists is None:
                return None
            return _record(
                db,
                "SELECT * FROM prediction_coverage WHERE prediction_hash = ?",
                (self.hash,),
            )

    @property
    def complete(self) -> bool:
        from case_studies.utils.artifact_digest import value_digest

        coverage = self.coverage()
        prediction_file = self.root / "run_log" / "predictions" / self.hash / "predictions.parquet"
        if (
            self.identity_version not in SUPPORTED_IDENTITY_VERSIONS
            or not coverage
            or coverage["status"] != "complete"
            or not prediction_file.is_file()
        ):
            return False
        # artifact_digest arrived as a nullable column on an existing table, so rows
        # registered before it exists carry NULL and there is nothing to compare
        # against. register_prediction_set reads that NULL as "legacy, backfill it"
        # rather than as a conflict; completeness has to agree, or every prediction in
        # a pre-existing registry reports incomplete and Lifecycle.lock refuses a
        # backtest whose artifacts are all present. Rows written since always carry a
        # digest and are held to it.
        recorded_digest = coverage.get("artifact_digest")
        if recorded_digest:
            try:
                if _verified_digest(prediction_file, self.load) != recorded_digest:
                    return False
            except (OSError, ValueError, pl.exceptions.PolarsError):
                return False
        with closing(sqlite3.connect(self.root / "run_log" / "registry.db")) as db:
            headline = db.execute(
                "SELECT 1 FROM prediction_metrics WHERE prediction_hash = ?", (self.hash,)
            ).fetchone()
            fold_count = db.execute(
                "SELECT COUNT(*) FROM fold_metrics WHERE prediction_hash = ?", (self.hash,)
            ).fetchone()[0]
        return headline is not None and fold_count == coverage["n_folds_expected"]

    def load(self):
        import polars as pl

        path = self.root / "run_log" / "predictions" / self.hash / "predictions.parquet"
        return pl.read_parquet(path)


@dataclass(frozen=True)
class BacktestResult(Result):
    @property
    def complete(self) -> bool:
        from case_studies.utils.artifact_digest import value_digest

        record = self.registry_record()
        spec_path = self.root / "run_log" / "backtest" / self.hash / "spec.json"
        try:
            stored_spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        stored_spec.pop("_runtime_backtest_config", None)
        if stored_spec != self.spec():
            return False
        prediction = Result.open(
            self.study,
            record["prediction_hash"],
            include_preview=self.execution_tier == ExecutionTier.PREVIEW.value,
        )
        if not isinstance(prediction, PredictionResult) or not prediction.complete:
            return False
        returns = self.root / "run_log" / "backtest" / self.hash / "daily_returns.parquet"
        with closing(sqlite3.connect(self.root / "run_log" / "registry.db")) as db:
            metrics = db.execute(
                "SELECT 1 FROM backtest_metrics WHERE backtest_hash = ?", (self.hash,)
            ).fetchone()
        # As with prediction_coverage.artifact_digest, artifact_digests_json is NULL on
        # every backtest_runs row that predates the column. Treat that as "nothing
        # recorded to verify" and fall back to requiring the returns file, rather than
        # reporting every pre-existing backtest incomplete.
        recorded_digests = record.get("artifact_digests_json")
        if not recorded_digests:
            return metrics is not None and returns.is_file()
        try:
            artifact_digests = json.loads(recorded_digests)
        except (json.JSONDecodeError, TypeError):
            return False
        if (
            not isinstance(artifact_digests, dict)
            or "daily_returns.parquet" not in artifact_digests
        ):
            return False
        for filename, expected_digest in artifact_digests.items():
            path = returns.parent / filename
            try:
                if not path.is_file():
                    return False
                if _verified_digest(path, partial(pl.read_parquet, path)) != expected_digest:
                    return False
            except (OSError, ValueError, pl.exceptions.PolarsError):
                return False
        return metrics is not None


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
        resolved.setdefault("identity_version", IDENTITY_VERSION)
        resolved.setdefault("execution_tier", tier.value)
        if (
            resolved["identity_version"] not in SUPPORTED_IDENTITY_VERSIONS
            or resolved["execution_tier"] != tier.value
        ):
            raise ValueError(
                "training spec identity version or execution tier conflicts with request"
            )
        if resolved["identity_version"] == IDENTITY_VERSION:
            from .identity import ResolvedSpec

            ResolvedSpec.from_dict(resolved)
        computation = resolved.get("computation", resolved)
        if tier is ExecutionTier.PREVIEW and not computation.get("preview_reductions"):
            raise ValueError("preview training specs must identity-cover every preview reduction")
        if tier is ExecutionTier.CANONICAL and computation.get("preview_reductions"):
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
        metrics: dict[str, float | dict] | None = None,
        task_type: str = "regression",
        class_values: list | None = None,
        eval_col: str | None = None,
        label: str | None = None,
    ) -> PredictionResult:
        self.study.require_writable()
        if training.study != self.study or training.kind != "training":
            raise ValueError("training result belongs to another study")
        tier = ExecutionTier(training.execution_tier)
        case_dir = self.study.activate(tier)
        from .cv import EligibilityManifest

        if isinstance(expected_keys, EligibilityManifest):
            expected_keys = expected_keys.eligible_keys
        prediction_hash = register_prediction_set(
            self.study.case_study,
            training.hash,
            checkpoint_kind=checkpoint_kind,
            checkpoint_value=checkpoint_value,
            split=split,
            predictions=predictions,
            expected_keys=expected_keys,
            allow_partial=allow_partial,
            metrics=metrics,
            task_type=task_type,
            class_values=class_values,
            eval_col=eval_col,
            label=label,
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
