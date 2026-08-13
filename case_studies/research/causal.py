from __future__ import annotations

import json
import math
import sqlite3
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.causal import _walk_forward_indices, run_dml_analysis
from case_studies.utils.registry.registration import register_causal_run
from case_studies.utils.registry.specs import (
    canonical_json,
    training_hash_from_spec,
)

from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .workspace import Study


_NUISANCE_MODEL_SPEC = {
    "outcome": {
        "class": "HistGradientBoostingRegressor",
        "params": {"max_depth": 3, "max_iter": 50, "random_state": 42},
    },
    "treatment": {
        "class": "HistGradientBoostingRegressor",
        "params": {"max_depth": 3, "max_iter": 50, "random_state": 42},
    },
}


@dataclass(frozen=True)
class CausalResult:
    study: Study
    hash: str
    execution_tier: str
    identity_version: int | None

    @classmethod
    def open(
        cls,
        study: Study,
        causal_hash: str,
        *,
        include_preview: bool = False,
    ) -> CausalResult:
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
                table = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'causal_runs'"
                ).fetchone()
                if table is None:
                    continue
                row = db.execute(
                    "SELECT spec_json FROM causal_runs WHERE causal_hash = ?",
                    (causal_hash,),
                ).fetchone()
            if row is not None:
                spec = json.loads(row[0] or "{}")
                return cls(
                    study=study,
                    hash=causal_hash,
                    execution_tier=spec.get("execution_tier") or namespace,
                    identity_version=spec.get("identity_version"),
                )
        raise KeyError(f"Unknown causal result hash {causal_hash!r}")

    @property
    def root(self) -> Path:
        return self.study.storage_root(self.execution_tier)

    def registry_record(self) -> dict[str, Any]:
        with sqlite3.connect(self.root / "run_log" / "registry.db") as db:
            cursor = db.execute(
                "SELECT * FROM causal_runs WHERE causal_hash = ?",
                (self.hash,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"Unknown causal result hash {self.hash!r}")
            return dict(zip((column[0] for column in cursor.description), row, strict=True))

    def spec(self) -> dict[str, Any]:
        return json.loads(self.registry_record().get("spec_json") or "{}")

    @property
    def complete(self) -> bool:
        record = self.registry_record()
        values = (record.get("dml_effect"), record.get("dml_se_hac"), record.get("n_obs"))
        return (
            self.identity_version == 2
            and bool(self.spec())
            and all(value is not None and math.isfinite(value) for value in values)
            and int(record["n_obs"]) > 0
        )

    def protocol(self) -> dict[str, Any]:
        spec = self.spec()
        return {
            "input_identity": spec.get("input_identity"),
            "cv": spec.get("cv"),
            "execution_tier": self.execution_tier,
        }


@dataclass(frozen=True)
class CausalRequest:
    study: Study
    label: str
    treatment: str
    confounders: tuple[str, ...]
    n_folds: int
    embargo: int
    observation_frequency: str
    horizon: int
    block_size: int
    n_placebo: int
    seed: int
    time_col: str
    entity_col: str
    development_end: str
    source_identity: dict[str, Any]
    runtime_identity: dict[str, Any]
    notebook: str
    hac_maxlags: int | None = None
    execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL
    preview_reductions: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "confounders", tuple(self.confounders))
        object.__setattr__(self, "source_identity", deepcopy(self.source_identity))
        object.__setattr__(self, "runtime_identity", deepcopy(self.runtime_identity))
        object.__setattr__(
            self,
            "preview_reductions",
            deepcopy(self.preview_reductions),
        )
        tier = ExecutionTier(self.execution_tier)
        object.__setattr__(self, "execution_tier", tier)
        if not self.confounders:
            raise ValueError("causal request requires at least one confounder")
        if min(self.n_folds, self.embargo, self.horizon, self.block_size) < 1:
            raise ValueError("causal fold, embargo, horizon, and block counts must be positive")
        if self.embargo < self.horizon:
            raise ValueError("causal embargo must cover the complete outcome horizon")
        if self.hac_maxlags is not None and self.hac_maxlags < self.horizon - 1:
            raise ValueError("causal HAC bandwidth must cover horizon minus one")
        if self.n_placebo < 10:
            raise ValueError("causal refutation requires at least 10 placebo replications")
        if not self.source_identity or not self.runtime_identity:
            raise ValueError("causal request requires source_identity and runtime_identity")
        if tier is ExecutionTier.PREVIEW and not self.preview_reductions:
            raise ValueError("preview causal requests must identity-cover every reduction")
        if tier is ExecutionTier.CANONICAL and self.preview_reductions:
            raise ValueError("canonical causal requests cannot contain preview reductions")

    def _prepare(self, data: pd.DataFrame | pl.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame = data.to_pandas() if isinstance(data, pl.DataFrame) else data.copy()
        required = [
            self.time_col,
            self.entity_col,
            self.treatment,
            self.label,
            *self.confounders,
        ]
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise KeyError(f"causal input is missing required columns: {missing}")
        frame[self.time_col] = pd.to_datetime(frame[self.time_col])
        development_end = pd.Timestamp(self.development_end)
        development = frame.loc[frame[self.time_col] < development_end, required].copy()
        input_digest = value_digest(pl.from_pandas(development))
        analysis = development.dropna(subset=required).sort_values(
            [self.time_col, self.entity_col],
            kind="stable",
        )
        reductions = self.preview_reductions or {}
        unknown = set(reductions) - {"max_decision_times", "max_entities"}
        if unknown:
            raise ValueError(f"unsupported causal preview reductions: {sorted(unknown)}")
        max_entities = reductions.get("max_entities")
        if max_entities is not None:
            if int(max_entities) < 1:
                raise ValueError("max_entities must be positive")
            entities = sorted(analysis[self.entity_col].unique())[: int(max_entities)]
            analysis = analysis.loc[analysis[self.entity_col].isin(entities)]
        max_times = reductions.get("max_decision_times")
        if max_times is not None:
            if int(max_times) < 1:
                raise ValueError("max_decision_times must be positive")
            decision_times = analysis[self.time_col].drop_duplicates().iloc[: int(max_times)]
            analysis = analysis.loc[analysis[self.time_col].isin(decision_times)]
        if analysis.empty:
            raise ValueError("causal request produced an empty complete-case analysis frame")
        if analysis.duplicated([self.time_col, self.entity_col]).any():
            raise ValueError("causal input contains duplicate decision-time and entity keys")
        numeric = analysis[[self.treatment, self.label, *self.confounders]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("causal analysis columns must be finite after complete-case filtering")
        analysis_frame = pl.from_pandas(analysis)
        identity = {
            "source": deepcopy(self.source_identity),
            "development_frame": input_digest,
            "analysis_frame": value_digest(analysis_frame),
            "n_development_rows": len(development),
            "n_analysis_rows": len(analysis),
        }
        return analysis, identity

    def _resolve_folds(self, analysis: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
        groups = analysis[self.time_col].to_numpy()
        folds = _walk_forward_indices(
            len(analysis),
            self.n_folds,
            self.embargo,
            groups=groups,
        )
        resolved = []
        test_periods = []
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            if len(train_idx) < 50 or len(test_idx) < 10:
                raise ValueError(
                    f"causal fold {fold_id} is too small: "
                    f"n_train={len(train_idx)}, n_test={len(test_idx)}"
                )
            train_times = groups[train_idx]
            test_times = groups[test_idx]
            test_periods.extend(pd.unique(test_times))
            resolved.append(
                {
                    "fold": fold_id,
                    "train_start": str(pd.Timestamp(train_times[0])),
                    "train_end": str(pd.Timestamp(train_times[-1])),
                    "test_start": str(pd.Timestamp(test_times[0])),
                    "test_end": str(pd.Timestamp(test_times[-1])),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )
        n_periods = len(np.unique(np.asarray(test_periods)))
        resolved_hac = self.hac_maxlags
        if resolved_hac is None:
            resolved_hac = max(self.horizon - 1, max(1, int(n_periods ** (1 / 3))))
            resolved_hac = min(resolved_hac, max(1, n_periods // 2))
        return resolved, resolved_hac

    def resolve(self, data: pd.DataFrame | pl.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
        analysis, input_identity = self._prepare(data)
        folds, resolved_hac = self._resolve_folds(analysis)
        spec = {
            "identity_version": 2,
            "execution_tier": self.execution_tier.value,
            "family": "causal_dml",
            "label": self.label,
            "input_identity": input_identity,
            "cv": {
                "method": "expanding_walk_forward",
                "n_folds": self.n_folds,
                "embargo": self.embargo,
                "development_end": self.development_end,
                "folds": folds,
            },
            "causal": {
                "treatment": self.treatment,
                "confounders": list(self.confounders),
                "time_col": self.time_col,
                "entity_col": self.entity_col,
                "observation_frequency": self.observation_frequency,
                "horizon": self.horizon,
                "block_size": self.block_size,
                "n_placebo": self.n_placebo,
                "hac_maxlags": resolved_hac,
                "nuisance_models": deepcopy(_NUISANCE_MODEL_SPEC),
            },
            "runtime_identity": deepcopy(self.runtime_identity),
            "seed": self.seed,
        }
        if self.preview_reductions:
            spec["preview_reductions"] = deepcopy(self.preview_reductions)
        return spec, analysis

    def run(self, data: pd.DataFrame | pl.DataFrame) -> CausalResult:
        self.study.require_writable()
        spec, analysis = self.resolve(data)
        case_dir = self.study.activate(self.execution_tier)
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        computed = run_dml_analysis(
            analysis,
            treatment_col=self.treatment,
            outcome_col=self.label,
            confounder_cols=list(self.confounders),
            n_folds=self.n_folds,
            embargo=self.embargo,
            n_placebo=self.n_placebo,
            block_size=self.block_size,
            seed=self.seed,
            hac_maxlags=int(spec["causal"]["hac_maxlags"]),
            horizon=self.horizon,
            time_col=self.time_col,
            entity_col=self.entity_col,
        )
        dml = computed["dml_result"]
        if dml.get("covariance_type") == "hc0_fallback":
            raise RuntimeError(
                "causal run did not produce the required robust covariance: "
                f"{dml.get('covariance_error', 'unknown failure')}"
            )
        refutation = computed.get("refutation") or {}
        required_values = [
            dml.get("theta"),
            dml.get("se_hac"),
            dml.get("n_obs"),
            computed.get("p_value_hac"),
            computed.get("naive_effect"),
            computed.get("confounding_bias_pct"),
            refutation.get("empirical_p"),
        ]
        if not all(value is not None and np.isfinite(value) for value in required_values):
            raise RuntimeError("causal run produced incomplete or non-finite core results")
        causal_hash = training_hash_from_spec(spec)
        register_causal_run(
            self.study.case_study,
            causal_hash,
            label=self.label,
            treatment=self.treatment,
            confounders_json=canonical_json(list(self.confounders)),
            embargo=self.embargo,
            n_folds=self.n_folds,
            n_obs=int(dml["n_obs"]),
            dml_effect=float(dml["theta"]),
            dml_se_hac=float(dml["se_hac"]),
            p_value_hac=float(computed["p_value_hac"]),
            naive_effect=float(computed["naive_effect"]),
            confounding_bias_pct=float(computed["confounding_bias_pct"]),
            refutation_p=float(refutation["empirical_p"]),
            spec_json=canonical_json(spec),
            notebook=self.notebook,
            started_at=started_at,
            elapsed_s=time.perf_counter() - started,
            case_dir=case_dir,
        )
        result = CausalResult.open(
            self.study,
            causal_hash,
            include_preview=self.execution_tier is ExecutionTier.PREVIEW,
        )
        return result
