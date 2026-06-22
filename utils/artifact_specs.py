"""Compatibility loaders for case-study artifact sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from utils.paths import get_case_study_dir

try:
    from ml4t.backtest.spec_io import load_spec as _load_shared_spec
except ImportError:  # pragma: no cover - depends on install state
    _load_shared_spec = None


def _artifact_root(case_study_id: str) -> Path:
    return get_case_study_dir(case_study_id, create=False) / "config" / "artifacts"


def _to_mapping(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        return dict(spec)
    if hasattr(spec, "to_dict"):
        return dict(spec.to_dict())
    raise TypeError(f"Unsupported artifact spec type: {type(spec).__name__}")


def _load_spec(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if _load_shared_spec is not None:
        return _to_mapping(_load_shared_spec(path))
    with path.open() as f:
        data = yaml.safe_load(f)
    return dict(data) if data else None


@cache
def _load_market_data_spec_cached(case_study_id: str) -> dict[str, Any] | None:
    return _load_spec(_artifact_root(case_study_id) / "market_data.yaml")


@cache
def _load_label_spec_cached(case_study_id: str, label: str) -> dict[str, Any] | None:
    return _load_spec(_artifact_root(case_study_id) / "labels" / f"{label}.yaml")


@cache
def _load_feature_spec_cached(case_study_id: str, feature_set: str) -> dict[str, Any] | None:
    return _load_spec(_artifact_root(case_study_id) / "features" / f"{feature_set}.yaml")


def load_market_data_spec(case_study_id: str) -> dict[str, Any] | None:
    spec = _load_market_data_spec_cached(case_study_id)
    return deepcopy(spec) if spec is not None else None


def load_label_spec(case_study_id: str, label: str) -> dict[str, Any] | None:
    spec = _load_label_spec_cached(case_study_id, label)
    return deepcopy(spec) if spec is not None else None


def load_feature_spec(case_study_id: str, feature_set: str) -> dict[str, Any] | None:
    spec = _load_feature_spec_cached(case_study_id, feature_set)
    return deepcopy(spec) if spec is not None else None


def resolve_storage_path(
    case_study_id: str,
    spec: dict[str, Any] | None,
    default_relative_path: str,
) -> Path:
    if spec is None:
        return get_case_study_dir(case_study_id, create=False) / default_relative_path
    storage = spec.get("storage", {})
    rel_path = storage.get("path", default_relative_path)
    return get_case_study_dir(case_study_id, create=False) / str(rel_path)


def resolve_label_buffer(
    case_study_id: str,
    label: str,
    setup: Mapping[str, Any] | None = None,
) -> str | None:
    label_spec = load_label_spec(case_study_id, label)
    if label_spec is not None:
        definition = label_spec.get("definition", {})
        if definition.get("buffer"):
            return str(definition["buffer"])

    labels = (setup or {}).get("labels", {})
    if labels.get("primary") == label and labels.get("buffer"):
        return str(labels["buffer"])
    variant_buffers = labels.get("variant_buffers", {})
    if label in variant_buffers:
        return str(variant_buffers[label])
    return None


def resolve_market_semantics(
    case_study_id: str,
    setup: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    market_spec = load_market_data_spec(case_study_id) or {}
    semantics = market_spec.get("semantics", {})
    evaluation = (setup or {}).get("evaluation", {})
    return {
        "calendar": semantics.get("calendar") or evaluation.get("calendar"),
        "timezone": semantics.get("timezone"),
        "data_frequency": semantics.get("data_frequency"),
        "timestamp_semantics": semantics.get("timestamp_semantics"),
        "session_start_time": semantics.get("session_start_time"),
        "bar_type": semantics.get("bar_type"),
    }


def resolve_market_runtime(case_study_id: str) -> dict[str, Any]:
    market_spec = load_market_data_spec(case_study_id) or {}
    runtime = market_spec.get("runtime", {})
    return dict(runtime) if isinstance(runtime, Mapping) else {}


__all__ = [
    "load_feature_spec",
    "load_label_spec",
    "load_market_data_spec",
    "resolve_label_buffer",
    "resolve_market_runtime",
    "resolve_market_semantics",
    "resolve_storage_path",
]
