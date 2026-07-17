"""Specification and hashing helpers for the experiment registry."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default seed — must match utils.modeling.RANDOM_SEED.
# Duplicated here to avoid importing the full modeling stack (torch, etc.)
# into a lightweight registry module.
DEFAULT_SEED = 42

# Required fields in every training spec.  ``seed`` is enforced so that
# two runs with different seeds always produce different hashes.
_REQUIRED_SPEC_FIELDS = {"family", "label", "seed"}

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

HASH_LENGTH = 12


def canonical_json(d: dict) -> str:
    """Deterministic JSON serialization for hashing.

    Sorted keys, no whitespace, consistent float/None handling.
    """
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


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
    return compute_hash(canonical_json(spec))


def prediction_hash_from_parts(
    training_hash: str,
    checkpoint_value: int | None,
    split: str,
) -> str:
    """Compute prediction_hash from its defining components."""
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
) -> str:
    """Compute backtest_hash from prediction_hash + strategy spec.

    Non-portable provenance (see ``_HASH_EXCLUDED_METADATA``) is stripped before
    hashing so the same strategy hashes identically regardless of where it runs.
    """
    return compute_hash(
        f"{prediction_hash}|{canonical_json(_hashable_strategy_spec(strategy_spec))}"
    )


# ---------------------------------------------------------------------------
# Preset Loader
# ---------------------------------------------------------------------------

_CONFIG_DIR: Path | None = None


def _get_config_dir() -> Path:
    """Resolve shared config directory (lazy, cached)."""
    global _CONFIG_DIR
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

    # Family-specific fields
    if family == "gbm":
        spec["max_iterations"] = preset.get("max_iterations", 500)
        spec["checkpoint_interval"] = checkpoint_interval or preset.get("checkpoint_interval", 50)
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
