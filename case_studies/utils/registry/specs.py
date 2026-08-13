"""Specification and hashing helpers for the experiment registry."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default seed — must match utils.modeling.RANDOM_SEED.
# Duplicated here to avoid importing the full modeling stack (torch, etc.)
# into a lightweight registry module.
DEFAULT_SEED = 42

# Required fields in every training spec.  ``seed`` is enforced so that
# two runs with different seeds always produce different hashes.
_REQUIRED_SPEC_FIELDS = {"family", "label", "seed"}

IDENTITY_VERSION = 3
LEGACY_IDENTITY_VERSION = 2
SUPPORTED_IDENTITY_VERSIONS = {LEGACY_IDENTITY_VERSION, IDENTITY_VERSION}
_V2_PROVENANCE_FIELDS = {
    "chapter",
    "config_name",
    "created_at",
    "display_name",
    "entry_point",
    "friendly_name",
    "git_commit",
    "notebook_path",
    "preset_path",
    "runtime",
    "runtime_provenance",
}

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

HASH_LENGTH = 12


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    """Return the supported JSON value for an identity-bearing object."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"canonical values must be finite at {path}")
        return 0.0 if value == 0 else value
    if isinstance(value, Enum):
        return _canonical_value(value.value, path=path)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"canonical mappings require string keys at {path}")
        return {key: _canonical_value(value[key], path=f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return _canonical_value(converted, path=path)
    raise TypeError(f"unsupported canonical value {type(value).__name__} at {path}")


def canonical_value(value: Any) -> Any:
    """Normalize and round-trip-check a supported identity-bearing value."""
    normalized = _canonical_value(value)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if decoded != normalized:
        raise ValueError("canonical value failed JSON round-trip validation")
    return normalized


def canonical_json(d: dict) -> str:
    """Deterministic JSON serialization for hashing.

    Sorted keys, no whitespace, consistent float/None handling.
    """
    if not isinstance(d, dict):
        raise TypeError("canonical_json requires a dictionary")
    return json.dumps(
        canonical_value(d),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compute_hash(content: str, length: int = HASH_LENGTH) -> str:
    """SHA-256 of *content*, truncated to *length* hex chars."""
    return hashlib.sha256(content.encode()).hexdigest()[:length]


def _validate_spec(spec: dict) -> dict:
    """Ensure spec contains all required fields; inject default seed if missing."""
    missing = _REQUIRED_SPEC_FIELDS - spec.keys()
    if missing == {"seed"}:
        logger.warning(
            "spec missing 'seed' — injecting default %d. Pass seed explicitly for reproducibility.",
            DEFAULT_SEED,
        )
        spec = {**spec, "seed": DEFAULT_SEED}
    elif missing:
        raise ValueError(f"spec missing required fields: {missing}")
    return spec


def training_hash_from_spec(spec: dict) -> str:
    """Compute training_hash from a spec dict.

    Validates that ``seed`` is present (injects default if missing).
    """
    spec = _validate_spec(spec)
    if spec.get("identity_version") in SUPPORTED_IDENTITY_VERSIONS:
        return compute_hash(canonical_json(project_training_identity(spec)))
    return compute_hash(canonical_json(spec))


def project_training_identity(spec: dict) -> dict:
    """Return the versioned semantic training identity projection."""
    version = spec.get("identity_version")
    if version == IDENTITY_VERSION:
        if spec.get("resolved_spec_schema") != "ml4t.resolved-spec/v1":
            raise ValueError("version-3 identity requires ml4t.resolved-spec/v1")
        computation = canonical_value(spec.get("computation"))
        if not isinstance(computation, dict):
            raise TypeError("resolved computation must be a dictionary")
        return {
            "identity_version": IDENTITY_VERSION,
            "resolved_spec_schema": "ml4t.resolved-spec/v1",
            "family": str(spec["family"]),
            "label": str(spec["label"]),
            "seed": int(spec["seed"]),
            "execution_tier": str(spec["execution_tier"]),
            "computation": computation,
        }
    if version != LEGACY_IDENTITY_VERSION:
        raise ValueError(
            f"identity projection requires identity_version in {sorted(SUPPORTED_IDENTITY_VERSIONS)}"
        )
    projected = {
        key: copy.deepcopy(value) for key, value in spec.items() if key not in _V2_PROVENANCE_FIELDS
    }
    projected["identity_version"] = LEGACY_IDENTITY_VERSION
    return projected


def prediction_hash_from_parts(
    training_hash: str,
    checkpoint_value: int | None,
    split: str,
    *,
    checkpoint_kind: str | None = None,
    identity_version: int | None = None,
) -> str:
    """Compute prediction_hash from its defining components."""
    if identity_version in SUPPORTED_IDENTITY_VERSIONS:
        if not checkpoint_kind:
            raise ValueError("versioned prediction identity requires checkpoint_kind")
        return compute_hash(
            canonical_json(
                {
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_value": checkpoint_value,
                    "identity_version": identity_version,
                    "split": split,
                    "training_hash": training_hash,
                }
            )
        )
    cp = str(checkpoint_value) if checkpoint_value is not None else "final"
    return compute_hash(f"{training_hash}|{cp}|{split}")


# Provenance keys under ``backtest_config.metadata`` that must NOT enter the
# backtest hash: they record where/how a run was launched, not the strategy's
# identity. ``preset_path`` is an absolute filesystem path — ``/home/.../base.yaml``
# on the host vs ``/app/.../base.yaml`` in Docker — so including it made the hash
# non-portable: re-running a sweep from a different path recomputed every hash and
# silently duplicated the registry. Stripping it here keeps ``spec_json`` (stored
# separately) fully intact for provenance while making the hash path-independent.
_HASH_EXCLUDED_METADATA = ("preset_path",)


def _hashable_strategy_spec(strategy_spec: dict) -> dict:
    """Copy of *strategy_spec* with non-portable provenance stripped for hashing."""
    spec = copy.deepcopy(strategy_spec)
    spec.pop("_runtime_backtest_config", None)
    strategy = spec.get("strategy")
    if isinstance(strategy, dict):
        signal = strategy.get("signal")
        if isinstance(signal, dict):
            # ``long_only`` was historically implicit. Remove the explicit
            # default so direction-less legacy specs and current specs retain
            # one semantic identity. Non-default directions remain hash inputs.
            direction = str(signal.get("direction", "long_only")).strip().lower()
            if direction == "long_only":
                signal.pop("direction", None)
    backtest_config = spec.get("backtest_config")
    if isinstance(backtest_config, dict):
        metadata = backtest_config.get("metadata")
        if isinstance(metadata, dict):
            for key in _HASH_EXCLUDED_METADATA:
                metadata.pop(key, None)
    return spec


def backtest_hash_from_parts(
    prediction_hash: str,
    strategy_spec: dict,
    *,
    identity_version: int | None = None,
) -> str:
    """Compute backtest_hash from prediction_hash + strategy spec.

    Non-portable provenance (see ``_HASH_EXCLUDED_METADATA``) is stripped before
    hashing so the same strategy hashes identically regardless of where it runs.
    """
    hashable = _hashable_strategy_spec(strategy_spec)
    resolved_version = identity_version or strategy_spec.get("identity_version")
    if resolved_version in SUPPORTED_IDENTITY_VERSIONS:
        return compute_hash(
            canonical_json(
                {
                    "identity_version": resolved_version,
                    "prediction_hash": prediction_hash,
                    "strategy": hashable,
                }
            )
        )
    return compute_hash(f"{prediction_hash}|{canonical_json(hashable)}")


# ---------------------------------------------------------------------------
# Preset Loader
# ---------------------------------------------------------------------------

_CONFIG_DIR: Path | None = None


def _get_config_dir() -> Path:
    """Resolve shared config directory (lazy, cached)."""
    global _CONFIG_DIR
    output_root = os.environ.get("ML4T_OUTPUT_DIR")
    if output_root:
        output_config = Path(output_root) / "config"
        if output_config.is_dir():
            return output_config
    if _CONFIG_DIR is None:
        from utils.paths import REPO_ROOT

        _CONFIG_DIR = REPO_ROOT / "case_studies" / "config"
    return _CONFIG_DIR


def load_preset(family: str, config_name: str) -> dict:
    """Load a model preset YAML file.

    Searches all ``case_studies/config/{model_type}/`` subdirectories for
    ``{config_name}.yaml``.

    Returns dict with keys: family, config_name, library, params,
    and optionally checkpoint_interval, max_iterations.

    Raises FileNotFoundError if the preset doesn't exist.
    """
    import yaml

    from utils.modeling import _enrich_config

    config_dir = _get_config_dir()
    matches = list(config_dir.glob(f"*/{config_name}.yaml"))
    if not matches:
        raise FileNotFoundError(f"No preset found: {config_name}.yaml in {config_dir}/*/")
    with open(matches[0]) as f:
        preset = yaml.safe_load(f)
    return _enrich_config(preset, matches[0])


def build_training_spec(
    family: str,
    config_name: str,
    label: str,
    *,
    n_folds: int,
    feature_sets: list[str] | None = None,
    n_epochs: int | None = None,
    max_bin: int | None = None,
    num_class: int | None = None,
    checkpoint_interval: int | None = None,
    seed: int = DEFAULT_SEED,
    causal_params: dict | None = None,
    extra_params: dict | None = None,
    train_sample_frac: float = 1.0,
    input_lineage: dict | None = None,
) -> dict:
    """Build a complete training spec from a preset + case-study context.

    The resulting spec is deterministically hashable and matches the
    rich format stored in existing registry DBs.

    Parameters
    ----------
    family : str
        Model family (gbm, linear, deep_learning, tabular_dl, latent_factors, causal_dml).
    config_name : str
        Config name matching a preset file (e.g. "leaves_15_huber").
    label : str
        Target label (e.g. "fwd_ret_21d").
    n_folds : int
        Number of CV folds.
    feature_sets : list[str], optional
        Feature set names. Default: ["financial", "model_based"].
    n_epochs : int, optional
        Override for DL/TabM/Latent n_epochs (preset default used if None).
    max_bin : int, optional
        LightGBM max_bin (63 for GPU, 255 for CPU). Added to GBM params.
    num_class : int, optional
        Number of classes for multiclass GBM.
    checkpoint_interval : int, optional
        Override for checkpoint interval (preset default used if None).
    seed : int
        Random seed.
    causal_params : dict, optional
        Case-study-specific causal DML params (treatment, confounders, embargo).
    extra_params : dict, optional
        Additional params to merge into the params dict.
    input_lineage : dict, optional
        Legacy top-level model-input identity used by certified case-study
        registries. New callers should place ``input_data_spec`` in
        ``extra_params``.
    """
    preset = load_preset(family, config_name)

    if feature_sets is None:
        feature_sets = ["financial", "model_based"]

    # Start with preset params
    params = dict(preset.get("params", {}))

    # Build spec common fields
    spec: dict = {
        "config_name": config_name,
        "family": family,
        "feature_sets": feature_sets,
        "label": label,
        "library": preset.get("library", ""),
        "n_folds": n_folds,
        "seed": seed,
    }
    if input_lineage is not None:
        spec["input_lineage"] = copy.deepcopy(input_lineage)

    # Family-specific fields
    if family == "gbm":
        spec["max_iterations"] = preset.get("max_iterations", 500)
        spec["checkpoint_interval"] = checkpoint_interval or preset.get("checkpoint_interval", 50)
        if preset.get("huber_alpha_scale") is not None:
            params["huber_alpha_scale"] = preset["huber_alpha_scale"]
        if max_bin is not None:
            params["max_bin"] = max_bin
        if num_class is not None and num_class > 2:
            params["num_class"] = num_class

    elif family in ("deep_learning", "tabular_dl"):
        cp = checkpoint_interval or preset.get("checkpoint_interval", 5)
        spec["checkpoint_interval"] = cp
        if n_epochs is not None:
            spec["n_epochs"] = n_epochs

    elif family == "latent_factors":
        if n_epochs is not None:
            spec["n_epochs"] = n_epochs
        for field in (
            "checkpoint_interval",
            "checkpoint_epochs",
            "n_epochs_unc",
            "n_epochs_moment",
            "n_epochs_cond",
            "burn_in_epochs",
            "beta_n_epochs",
            "beta_checkpoint_interval",
            "beta_checkpoint_epochs",
            "beta_default_checkpoint",
        ):
            value = preset.get(field)
            if value not in (None, (), []):
                spec[field] = value

    elif family == "causal_dml":
        if causal_params:
            params.update(causal_params)

    # Merge extra params
    if extra_params:
        params.update(extra_params)

    spec["params"] = params

    if 0.0 < train_sample_frac < 1.0:
        spec["train_sample_frac"] = float(train_sample_frac)

    return spec
