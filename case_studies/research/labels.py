from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from case_studies.utils.artifact_digest import digest_sidecar, sidecar_path, value_digest
from utils.artifact_specs import load_label_spec, resolve_label_horizon, resolve_storage_path

from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .workspace import Study


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    task_type: Literal["regression", "classification"]
    horizon: str
    continuous_eval_label: str | None = None

    def __post_init__(self) -> None:
        if not self.name or any(char in self.name for char in "/\\"):
            raise ValueError("label name must be a non-empty path-safe name")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be regression or classification")
        if not self.horizon:
            raise ValueError("label horizon is required")
        if self.task_type == "classification" and not self.continuous_eval_label:
            raise ValueError("classification labels require continuous_eval_label")


@dataclass(frozen=True)
class LabelRef:
    definition: LabelDefinition
    path: Path
    digest: str

    @property
    def name(self) -> str:
        return self.definition.name

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return pl.read_parquet(self.path)


class LabelCatalog:
    def __init__(self, study: Study) -> None:
        self.study = study

    def _path(self, name: str) -> Path:
        return self.study.root / "labels" / f"{name}.parquet"

    def list(self) -> tuple[LabelRef, ...]:
        self.study.activate()
        setup_path = self.study.root / "config" / "setup.yaml"
        names: set[str] = set()
        if setup_path.exists():
            import yaml

            setup = yaml.safe_load(setup_path.read_text()) or {}
            labels = setup.get("labels") or {}
            if labels.get("primary"):
                names.add(str(labels["primary"]))
            names.update(str(name) for name in labels.get("variants") or [])
        label_dir = self.study.root / "labels"
        if label_dir.exists():
            names.update(path.stem for path in label_dir.glob("*.parquet"))
        return tuple(self.get(name) for name in sorted(names))

    def get(
        self,
        name: str,
        *,
        execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL,
    ) -> LabelRef:
        self.study.activate(execution_tier)
        path = self._path(name)
        metadata_path = sidecar_path(path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            raw_definition = metadata.get("definition")
            if raw_definition:
                if not path.is_file():
                    raise FileNotFoundError(f"label sidecar has no artifact: {path}")
                actual_digest = value_digest(pl.read_parquet(path))
                if actual_digest != metadata.get("digest"):
                    raise ValueError(f"label artifact digest does not match its sidecar: {path}")
                return LabelRef(
                    definition=LabelDefinition(**raw_definition),
                    path=path,
                    digest=actual_digest,
                )

        import yaml

        setup_path = self.study.root / "config" / "setup.yaml"
        setup = yaml.safe_load(setup_path.read_text()) if setup_path.exists() else {}
        labels = (setup or {}).get("labels") or {}
        configured = {labels.get("primary"), *(labels.get("variants") or [])}
        if name not in configured:
            raise KeyError(f"Unknown label {name!r}")
        continuous = (labels.get("classification_eval_label") or {}).get(name)
        task_type = "classification" if continuous else "regression"
        definition = LabelDefinition(
            name=name,
            task_type=task_type,
            horizon=resolve_label_horizon(self.study.case_study, name, setup) or "unknown",
            continuous_eval_label=continuous,
        )
        spec = load_label_spec(self.study.case_study, name)
        path = resolve_storage_path(self.study.case_study, spec, f"labels/{name}.parquet")
        if not path.is_file():
            raise FileNotFoundError(f"label artifact is missing: {path}")
        resolved_metadata_path = sidecar_path(path)
        if resolved_metadata_path.exists():
            metadata = json.loads(resolved_metadata_path.read_text())
            digest = value_digest(pl.read_parquet(path))
            if metadata.get("digest") != digest:
                raise ValueError(f"label artifact digest does not match its sidecar: {path}")
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return LabelRef(definition=definition, path=path, digest=digest)

    def publish(self, definition: LabelDefinition, frame: pl.DataFrame) -> LabelRef:
        self.study.require_writable()
        self.study.activate()
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("label publication requires a Polars DataFrame")
        required = {"symbol", "timestamp", definition.name}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"label artifact missing columns: {sorted(missing)}")
        if frame.select("symbol", "timestamp").null_count().row(0) != (0, 0):
            raise ValueError("label keys cannot contain nulls")
        if frame.n_unique(["symbol", "timestamp"]) != frame.height:
            raise ValueError("label keys must be unique")
        if not frame.get_column(definition.name).dtype.is_numeric():
            raise ValueError("label values must be numeric")
        if definition.task_type == "classification":
            eval_label = definition.continuous_eval_label
            assert eval_label is not None
            if eval_label not in frame.columns:
                raise ValueError(f"continuous evaluation target {eval_label!r} is missing")
            if not frame.get_column(eval_label).dtype.is_numeric():
                raise ValueError("continuous evaluation target must be numeric")

        path = self._path(definition.name)
        metadata_path = sidecar_path(path)
        record = digest_sidecar(
            frame,
            keys=("symbol", "timestamp"),
            written_by="case_studies.research.LabelCatalog.publish",
        )
        record["definition"] = asdict(definition)
        if path.exists() or metadata_path.exists():
            if path.exists() and metadata_path.exists():
                existing = json.loads(metadata_path.read_text())
                if existing == record:
                    return LabelRef(definition, path, str(record["digest"]))
            raise FileExistsError(f"immutable label artifact already exists: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        parquet_tmp = path.with_name(f".{path.name}.{token}.tmp")
        metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
        try:
            frame.write_parquet(parquet_tmp)
            metadata_tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            os.replace(metadata_tmp, metadata_path)
            try:
                os.replace(parquet_tmp, path)
            except Exception:
                metadata_path.unlink(missing_ok=True)
                raise
        finally:
            parquet_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
        return LabelRef(definition, path, str(record["digest"]))
