from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from case_studies.utils.registry.specs import (
    IDENTITY_VERSION,
    SUPPORTED_IDENTITY_VERSIONS,
    training_hash_from_spec,
)

from .adapters import get_adapter, registered_adapters
from .contracts import ExecutionTier

if TYPE_CHECKING:
    from .workspace import Study


@dataclass(frozen=True)
class CausalResult:
    study: Study
    hash: str
    spec: dict[str, Any]
    metrics: dict[str, Any]
    execution_tier: str

    @classmethod
    def one(
        cls,
        study: Study,
        *,
        label: str,
        execution_tier: str = "canonical",
    ) -> CausalResult:
        """Resolve one causal result by declared label and execution tier."""
        tier = ExecutionTier(execution_tier)
        root = study.storage_root(tier)
        db_path = root / "run_log" / "registry.db"
        if not db_path.is_file():
            raise ValueError(f"causal selection for {label!r} resolved to 0 identities")
        with sqlite3.connect(db_path) as db:
            rows = db.execute(
                "SELECT causal_hash, spec_json FROM causal_runs WHERE label = ? ORDER BY causal_hash",
                (label,),
            ).fetchall()
        current = [
            causal_hash
            for causal_hash, spec_json in rows
            if json.loads(spec_json or "{}").get("identity_version") == IDENTITY_VERSION
            and json.loads(spec_json or "{}").get("execution_tier", tier.value) == tier.value
        ]
        if len(current) != 1:
            raise ValueError(
                f"causal selection for {label!r} resolved to {len(current)} identities"
            )
        return cls.open(
            study,
            current[0],
            include_preview=tier is ExecutionTier.PREVIEW,
        )

    @classmethod
    def open(
        cls,
        study: Study,
        causal_hash: str,
        *,
        include_preview: bool = False,
    ) -> CausalResult:
        roots = [(study.root, ExecutionTier.CANONICAL.value)]
        if include_preview and study.output_root is not None:
            roots.insert(
                0,
                (
                    study.output_root / ".preview" / study.case_study,
                    ExecutionTier.PREVIEW.value,
                ),
            )
        for root, namespace in roots:
            db_path = root / "run_log" / "registry.db"
            if not db_path.is_file():
                continue
            with sqlite3.connect(db_path) as db:
                row = db.execute(
                    "SELECT n_obs, dml_effect, dml_se_hac, p_value_hac, naive_effect, "
                    "confounding_bias_pct, refutation_p, spec_json "
                    "FROM causal_runs WHERE causal_hash = ?",
                    (causal_hash,),
                ).fetchone()
            if row is None:
                continue
            spec = json.loads(row[7])
            tier = str(spec.get("execution_tier", namespace))
            return cls(
                study=study,
                hash=causal_hash,
                spec=spec,
                metrics={
                    "n_obs": row[0],
                    "dml_effect": row[1],
                    "dml_se_hac": row[2],
                    "p_value_hac": row[3],
                    "naive_effect": row[4],
                    "confounding_bias_pct": row[5],
                    "refutation_p": row[6],
                },
                execution_tier=tier,
            )
        raise KeyError(f"unknown causal result {causal_hash!r}")

    @property
    def complete(self) -> bool:
        return (
            self.spec.get("identity_version") in SUPPORTED_IDENTITY_VERSIONS
            and self.metrics.get("n_obs", 0) > 0
            and self.metrics.get("dml_effect") is not None
            and self.metrics.get("dml_se_hac") is not None
        )


@dataclass(frozen=True)
class ResolvedCausalRequest:
    study: Study
    method: str
    spec: dict[str, Any]
    _context: Any

    @property
    def identity(self) -> str:
        return training_hash_from_spec(self.spec)

    def run(self) -> CausalResult:
        module = get_adapter("causal", self.method)
        runner = getattr(module, "run_resolved_causal_request", None)
        if runner is None:
            raise NotImplementedError(f"{self.method!r} has no shared causal runner")
        result = runner(self.study, self.spec, self._context)
        if not isinstance(result, CausalResult):
            raise TypeError(
                f"{self.method!r} runner returned {type(result).__name__}, not CausalResult"
            )
        return result


@dataclass(frozen=True)
class CausalRequest:
    study: Study
    method: str
    label: str
    config_name: str
    overrides: dict[str, Any]
    execution_tier: ExecutionTier
    preview_reductions: dict[str, Any]

    @classmethod
    def from_request(cls, study: Study, request: dict[str, Any]) -> CausalRequest:
        supported = {
            "method",
            "label",
            "config_name",
            "overrides",
            "execution_tier",
            "preview_reductions",
        }
        unknown = set(request) - supported
        if unknown:
            raise ValueError(f"unsupported causal request fields: {sorted(unknown)}")
        missing = {"method", "label"} - set(request)
        if missing:
            raise ValueError(f"causal request is missing fields: {sorted(missing)}")
        method = str(request["method"])
        available = {binding.name for binding in registered_adapters("causal")}
        if method not in available:
            raise ValueError(f"unsupported causal method {method!r}")
        tier = ExecutionTier(request.get("execution_tier", ExecutionTier.CANONICAL))
        reductions = dict(request.get("preview_reductions") or {})
        if tier is ExecutionTier.PREVIEW and not reductions:
            raise ValueError("preview causal requests must declare every reduction")
        if tier is ExecutionTier.CANONICAL and reductions:
            raise ValueError("canonical causal requests cannot declare preview reductions")
        return cls(
            study=study,
            method=method,
            label=str(request["label"]),
            config_name=str(request.get("config_name", method)),
            overrides=dict(request.get("overrides") or {}),
            execution_tier=tier,
            preview_reductions=reductions,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "config_name": self.config_name,
            "overrides": dict(self.overrides),
            "execution_tier": self.execution_tier.value,
            "preview_reductions": dict(self.preview_reductions),
        }

    def resolve(self) -> ResolvedCausalRequest:
        module = get_adapter("causal", self.method)
        resolver = getattr(module, "resolve_causal_request", None)
        if resolver is None:
            raise NotImplementedError(f"{self.method!r} has no shared causal resolver")
        spec, context = resolver(self.study, self.as_dict())
        if spec.get("identity_version") != IDENTITY_VERSION:
            raise ValueError("causal resolver did not produce the current identity version")
        if spec.get("resolved_spec_schema") != "ml4t.resolved-spec/v1":
            raise ValueError("causal resolver did not produce the resolved-spec schema")
        if spec.get("execution_tier") != self.execution_tier.value:
            raise ValueError("causal resolver changed the execution tier")
        return ResolvedCausalRequest(self.study, self.method, spec, context)

    def run(self) -> CausalResult:
        return self.resolve().run()
