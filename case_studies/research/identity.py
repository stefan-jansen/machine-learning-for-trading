from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from case_studies.utils.registry.specs import (
    IDENTITY_VERSION,
    canonical_value,
    training_hash_from_spec,
)


@dataclass(frozen=True)
class ResolvedSpec:
    """Versioned resolved computation and its non-identity operational provenance."""

    family: str
    label: str
    seed: int
    computation: dict[str, Any]
    provenance: dict[str, Any]
    config_name: str | None = None
    execution_tier: str = "canonical"

    def __post_init__(self) -> None:
        if not self.family or not self.label:
            raise ValueError("resolved specifications require family and label")
        if self.execution_tier not in {"canonical", "preview"}:
            raise ValueError("execution_tier must be canonical or preview")
        computation = canonical_value(self.computation)
        provenance = canonical_value(self.provenance)
        if not isinstance(computation, dict) or not computation:
            raise ValueError("resolved computation must be a non-empty dictionary")
        if not isinstance(provenance, dict):
            raise TypeError("resolved provenance must be a dictionary")
        object.__setattr__(self, "computation", computation)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def create(
        cls,
        *,
        family: str,
        label: str,
        seed: int,
        computation: dict[str, Any],
        provenance: dict[str, Any],
        config_name: str | None = None,
        execution_tier: str = "canonical",
    ) -> ResolvedSpec:
        return cls(family, label, seed, computation, provenance, config_name, execution_tier)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResolvedSpec:
        if value.get("identity_version") != IDENTITY_VERSION:
            raise ValueError(f"resolved specification requires identity_version={IDENTITY_VERSION}")
        if value.get("resolved_spec_schema") != "ml4t.resolved-spec/v1":
            raise ValueError("unsupported resolved specification schema")
        return cls(
            family=str(value["family"]),
            label=str(value["label"]),
            seed=int(value["seed"]),
            computation=deepcopy(value["computation"]),
            provenance=deepcopy(value.get("provenance") or {}),
            config_name=value.get("config_name"),
            execution_tier=str(value.get("execution_tier", "canonical")),
        )

    @property
    def identity(self) -> str:
        return training_hash_from_spec(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity_version": IDENTITY_VERSION,
            "resolved_spec_schema": "ml4t.resolved-spec/v1",
            "family": self.family,
            "label": self.label,
            "seed": self.seed,
            "config_name": self.config_name,
            "execution_tier": self.execution_tier,
            "computation": deepcopy(self.computation),
            "provenance": deepcopy(self.provenance),
        }
