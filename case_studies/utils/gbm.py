"""Shared GBM pipeline infrastructure for Ch12 notebooks and case study templates.

Provides:
- load_gbm_config(): Canonical params for a named capacity preset
- make_model_params(): Transparent library-specific parameter mapping
- create_model(): Factory for unfitted sklearn-compatible GBM regressors

Usage:
    from case_studies.utils.gbm import load_gbm_config, make_model_params, create_model

    config = load_gbm_config("medium")
    params = make_model_params(config, "lightgbm", "cpu")
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import lightgbm before ml4t.diagnostic, which transitively loads
# scikit-learn. Both scikit-learn and LightGBM ship their own OpenMP runtime
# and the first one loaded wins for the whole process; on macOS ARM64 the
# loser's first multithreaded fit dies inside
# `__kmp_suspend_initialize_thread`, taking the kernel with it and printing no
# traceback. Every function below re-imports lightgbm locally for reading;
# this one exists only to lose no race, for every case study that imports this
# module. `import x` sorts ahead of `from x import y`, so isort keeps it here.
import lightgbm  # noqa: F401
import numpy as np
import polars as pl

# Import torch before ml4t.diagnostic. ml4t.diagnostic transitively loads the
# `cuda` Python package, which dlopens the older system `libcudart.so.12`
# (12.0.146) and wins the symbol resolution; subsequent torch imports then
# fail with `undefined symbol: cudaGetDriverEntryPointByVersion`. Loading
# torch first ensures its bundled CUDA runtime wins. Same pattern as in
# `case_studies/utils/latent_factors/__init__.py` and `model_analysis.py`.
import torch  # noqa: F401
import yaml
from ml4t.diagnostic.metrics import cross_sectional_ic

from case_studies.research.contracts import ExecutionTier
from case_studies.research.cv import require_fold_scoped_temporal_compatibility
from case_studies.research.identity import ResolvedSpec
from case_studies.research.models import ModelRun
from case_studies.research.recovery import ExecutionAttempt, ExecutionLedger
from case_studies.research.results import PredictionResult, Result, TrainingResult
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.derived_params import quantize_derived
from case_studies.utils.folds import (
    FOLD_PREPARATION_VERSION,
    prepare_gbm_folds_from_mds,
    training_labels_for_split,
)
from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec
from case_studies.utils.registry.specs import canonical_json
from case_studies.utils.runtime import cpu_seconds, resource_measurement
from utils.modeling import RANDOM_SEED, seed_everything

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


_GBM_PREVIEW_FIELDS = {
    "checkpoint_interval",
    "folds",
    "max_iterations",
    "max_symbols",
    "train_sample_frac",
}
_GBM_REQUEST_FIELDS = {
    "checkpoint_interval",
    "device",
    "huber_alpha_scale",
    "max_bin",
    "max_iterations",
    "num_threads",
}


@dataclass(frozen=True)
class GBMContext:
    folds: tuple[dict[str, Any], ...]
    fold_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    label_col: str
    eval_label_col: str | None
    date_col: str
    entity_col: str
    task_type: str
    class_values: tuple[Any, ...]
    expected_keys: pl.DataFrame
    runtime_provenance: dict[str, Any]
    device: str
    num_threads: int
    prediction_split: str = "validation"
    published_checkpoints: tuple[int, ...] | None = None


@dataclass
class _GBMBatchCandidate:
    index: int
    request: dict[str, Any]
    config: dict[str, Any]
    effective_params: dict[str, dict[str, Any]]
    device: str
    max_bin: int
    num_threads: int
    spec: dict[str, Any] | None = None
    context: GBMContext | None = None
    training: TrainingResult | None = None
    ledger: ExecutionLedger | None = None
    attempt: ExecutionAttempt | None = None
    frames: list[pl.DataFrame] = field(default_factory=list)
    reused_folds: list[int] = field(default_factory=list)
    fitted_folds: list[int] = field(default_factory=list)
    fit_elapsed_s: float = 0.0
    started_at_s: float = 0.0
    started_cpu_s: float = 0.0
    result: ModelRun | None = None
    error: Exception | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Capacity presets for the Ch12 cross-library benchmark. These are library-neutral
# canonical names (see PARAM_NAMES below), deliberately separate from the case-study
# LightGBM presets in `case_studies/config/lgb/`, which are LightGBM-native and drive
# the registered sweep.
PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "n_trees": 200,
        "max_depth": 4,
        "lr": 0.10,
        "l2": 1.0,
        "subsample": 0.8,
        "colsample": 0.8,
        "min_leaf": 20,
    },
    "medium": {
        "n_trees": 500,
        "max_depth": 6,
        "lr": 0.05,
        "l2": 1.0,
        "subsample": 0.8,
        "colsample": 0.8,
        "min_leaf": 20,
    },
    "heavy": {
        "n_trees": 1000,
        "max_depth": 8,
        "lr": 0.01,
        "l2": 1.0,
        "subsample": 0.8,
        "colsample": 0.8,
        "min_leaf": 20,
    },
    "default": {
        "n_trees": 500,
    },
}


def load_gbm_config(preset: str = "medium") -> dict[str, Any]:
    """Load canonical GBM parameters for a capacity preset.

    Parameters
    ----------
    preset : str
        Preset name ("light", "medium", "heavy", "default").

    Returns
    -------
    dict
        Canonical parameters (n_trees, max_depth, lr, l2, ...).
    """
    if preset in PRESETS:
        return dict(PRESETS[preset])

    raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")


# ---------------------------------------------------------------------------
# Parameter Translation (transparent name mapping)
# ---------------------------------------------------------------------------

PARAM_NAMES: dict[str, dict[str, str]] = {
    "xgboost": {
        "n_trees": "n_estimators",
        "lr": "learning_rate",
        "l1": "reg_alpha",
        "l2": "reg_lambda",
        "colsample": "colsample_bytree",
        "min_leaf": "min_child_weight",
    },
    "lightgbm": {
        "n_trees": "n_estimators",
        "lr": "learning_rate",
        "l1": "reg_alpha",
        "l2": "reg_lambda",
        "colsample": "colsample_bytree",
        "min_leaf": "min_child_samples",
    },
    "catboost": {
        "n_trees": "iterations",
        "lr": "learning_rate",
        "l1": "model_size_reg",
        "l2": "l2_leaf_reg",
        "max_depth": "depth",
        "min_leaf": "min_data_in_leaf",
    },
    "sklearn_hgb": {
        "n_trees": "max_iter",
        "lr": "learning_rate",
        "l2": "l2_regularization",
        "min_leaf": "min_samples_leaf",
        "colsample": "max_features",
    },
}

# Canonical params that have no equivalent in certain libraries
_SKIP_PARAMS: dict[str, set[str]] = {
    "sklearn_hgb": {"subsample"},
    "catboost": {"colsample"},
}

# Cached GPU device per library: "cuda" or None (CPU only)
# OpenCL ("gpu") is NEVER used — it is slower and produces misleading benchmarks.
_BEST_GPU: dict[str, str | None] = {}

DEFAULT_GBM_CPU_THREADS = 8
# What every case study's setup.yaml declares under modeling.gbm.max_bin. It is LightGBM's own
# default; the 63 that preceded it was inherited from a device branch rather than declared, and
# the populations fitted under it are superseded.
GBM_DEFAULT_MAX_BIN = 255

# Declared behaviour of this runner. Bump when a change here would change a fitted result: the
# libraries it dispatches to, how a parameter is derived, the fitting procedure, the checkpoint
# schedule, or what is predicted. Do not bump for logging, comments, refactoring or anything a run
# merely records.
GBM_RUNNER_VERSION = 1

# What a fold is cast to before it reaches the booster. No imputation and no scaling: a tree splits
# on the ordering of a feature and routes a missing value down its own branch, so both would only
# fabricate observations. Bump when that casting changes.
GBM_PREPROCESSING_ID = "lightgbm-native-float32/v1"


def _best_gpu_device(library: str) -> str | None:
    """Return "cuda" if library supports CUDA on this system, else None.

    Only CUDA is accepted. OpenCL (device="gpu") is explicitly excluded —
    it is orders of magnitude slower and produces misleading benchmark results.
    """
    if library not in _BEST_GPU:
        import numpy as _np

        _X = _np.random.randn(10, 2).astype(_np.float32)
        _y = _np.random.randn(10).astype(_np.float32)

        try:
            if library == "lightgbm":
                import lightgbm as lgb

                lgb.LGBMRegressor(n_estimators=2, device="cuda", verbose=-1).fit(_X, _y)
            elif library == "xgboost":
                import xgboost as xgb

                xgb.XGBRegressor(
                    n_estimators=2, device="cuda", tree_method="hist", verbosity=0
                ).fit(_X, _y)
            _BEST_GPU[library] = "cuda"
        except Exception:
            _BEST_GPU[library] = None
    return _BEST_GPU[library]


def lightgbm_runtime_params(
    device: str,
    *,
    num_threads: int = DEFAULT_GBM_CPU_THREADS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Return explicit LightGBM execution provenance.

    CPU is the reproducible reader default. A GPU request fails when the
    installed LightGBM build cannot provide CUDA instead of silently training
    a different CPU model. These values configure execution and are recorded
    next to the portable training identity; they do not enter its hash.
    """
    normalized = device.lower()
    if num_threads < 1:
        raise ValueError("num_threads must be at least 1")
    if normalized == "cpu":
        return {
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": int(num_threads),
            "seed": int(seed),
            "data_random_seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "drop_seed": int(seed),
            "extra_seed": int(seed),
            "objective_seed": int(seed),
        }
    if normalized in ("cuda", "gpu"):
        gpu_device = _best_gpu_device("lightgbm")
        if gpu_device is None:
            raise RuntimeError(
                "LightGBM CUDA was requested but is unavailable. "
                "Use device='cpu' or install a CUDA-enabled LightGBM build."
            )
        return {
            "device_type": gpu_device,
            "num_threads": int(num_threads),
            "seed": int(seed),
            "data_random_seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "drop_seed": int(seed),
            "extra_seed": int(seed),
            "objective_seed": int(seed),
        }
    raise ValueError(f"Unsupported LightGBM device: {device!r}")


def resolve_gbm_device(requested: str | None, configured: str = "cpu") -> str:
    """Use an explicit runtime override when supplied, otherwise use the configured backend."""
    device = str(requested or configured).lower()
    if device == "gpu":
        device = "cuda"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported LightGBM device: {device!r}")
    return device


def resolve_gbm_execution_config(config: dict[str, Any]) -> tuple[str, int, int]:
    """Resolve a declared GBM backend without deriving model parameters from hardware."""
    device = resolve_gbm_device(None, str(config.get("device", "cpu")))

    if "max_bin" not in config:
        raise ValueError("modeling.gbm.max_bin must be declared explicitly")
    max_bin = int(config["max_bin"])
    if max_bin < 2:
        raise ValueError("modeling.gbm.max_bin must be at least 2")

    num_threads = int(config.get("num_threads", DEFAULT_GBM_CPU_THREADS))
    if num_threads < 1:
        raise ValueError("modeling.gbm.num_threads must be at least 1")
    return device, max_bin, num_threads


def gbm_checkpoint_iterations(config: dict[str, Any]) -> tuple[int, ...]:
    """Return the exact checkpoint surface implied by one GBM config."""
    n_iterations = int(config.get("max_iterations", 500))
    checkpoint_interval = int(config.get("checkpoint_interval", 50))
    if n_iterations < 1 or checkpoint_interval < 1:
        raise ValueError("max_iterations and checkpoint_interval must be positive")
    checkpoints = list(range(checkpoint_interval, n_iterations + 1, checkpoint_interval))
    if not checkpoints or checkpoints[-1] != n_iterations:
        checkpoints.append(n_iterations)
    return tuple(checkpoints)


def build_gbm_training_spec(
    config: dict[str, Any],
    *,
    label_col: str,
    n_folds: int,
    max_bin: int,
    feature_names: list[str],
    splits: list[dict[str, Any]],
    eval_label_col: str | None,
    task_type: str,
    class_values: list | None,
    seed: int,
    train_sample_frac: float = 1.0,
) -> dict[str, Any]:
    """Build the portable declared identity used for GBM lookup and registration."""
    from case_studies.utils.registry import build_training_spec

    identity_params = {
        "class_values": list(class_values) if class_values is not None else None,
        "eval_label_col": eval_label_col,
        "feature_names": list(feature_names),
        "splits": [
            {
                key: str(split[key]) if key != "fold" else int(split[key])
                for key in ("fold", "train_start", "train_end", "val_start", "val_end")
            }
            for split in splits
        ],
        "task_type": task_type,
    }
    return build_training_spec(
        config.get("family", "gbm"),
        config["config_name"],
        label_col,
        n_folds=n_folds,
        max_bin=max_bin,
        checkpoint_interval=config.get("checkpoint_interval", 50),
        seed=seed,
        extra_params=identity_params,
        train_sample_frac=train_sample_frac,
    )


# Library-specific defaults (not in canonical config)
_LIB_DEFAULTS: dict[str, dict[str, Any]] = {
    "xgboost": {"tree_method": "hist", "random_state": RANDOM_SEED, "verbosity": 0, "n_jobs": -1},
    "lightgbm": {"random_state": RANDOM_SEED, "verbose": -1, "n_jobs": -1},
    "catboost": {
        "bootstrap_type": "Bernoulli",
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": -1,
    },
    "sklearn_hgb": {"random_state": RANDOM_SEED},
}

# Canonical objective → library-specific mapping
_OBJECTIVE_MAP: dict[str, dict[str, str]] = {
    "lightgbm": {
        "mse": "regression",
        "mae": "regression_l1",
        "huber": "huber",
        "binary": "binary",
        "multiclass": "multiclass",
    },
    "xgboost": {
        "mse": "reg:squarederror",
        "mae": "reg:absoluteerror",
        "huber": "reg:pseudohubererror",
        "binary": "binary:logistic",
        "multiclass": "multi:softprob",
    },
    "catboost": {
        "mse": "RMSE",
        "mae": "MAE",
        "huber": "Huber",
        "binary": "Logloss",
        "multiclass": "MultiClass",
    },
}


def make_model_params(
    canonical: dict[str, Any],
    library: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Map canonical params to library-specific kwargs.

    GPU overrides come from the config's ``gpu`` section (visible in YAML),
    not from hidden internal logic.

    Parameters
    ----------
    canonical : dict
        Canonical params (n_trees, max_depth, lr, ...).
    library : str
        One of "xgboost", "lightgbm", "catboost", "sklearn_hgb".
    device : str
        "cpu" or "gpu".

    Returns
    -------
    dict
        Library-specific kwargs ready for model constructor.
    """
    if library not in PARAM_NAMES:
        raise ValueError(f"Unknown library: {library}. Use xgboost/lightgbm/catboost/sklearn_hgb.")

    name_map = PARAM_NAMES[library]
    lib_params: dict[str, Any] = {}

    # Map canonical names to library names
    skip = _SKIP_PARAMS.get(library, set())
    for k, v in canonical.items():
        if k in ("gpu", "objective"):
            continue  # Handled separately
        if k in skip:
            continue  # No equivalent in this library
        lib_name = name_map.get(k, k)
        lib_params[lib_name] = v

    # LightGBM: num_leaves — explicit value wins over max_depth derivation
    if library == "lightgbm":
        if "num_leaves" in canonical:
            lib_params["num_leaves"] = canonical["num_leaves"]
        elif "max_depth" in canonical:
            lib_params["num_leaves"] = 2 ** canonical["max_depth"] - 1

    # Objective mapping (canonical → library-specific)
    if "objective" in canonical and library in _OBJECTIVE_MAP:
        lib_params["objective"] = _OBJECTIVE_MAP[library].get(
            canonical["objective"], canonical["objective"]
        )

    # Library defaults
    lib_params.update(_LIB_DEFAULTS.get(library, {}))

    # GPU: device params + config overrides (visible in YAML)
    # Accept both "gpu" and "cuda" — both mean "use CUDA" (OpenCL is never used)
    if device in ("gpu", "cuda"):
        if library in ("xgboost", "lightgbm"):
            gpu_dev = _best_gpu_device(library)
            if gpu_dev:
                lib_params["device"] = gpu_dev
            else:
                raise RuntimeError(
                    f"{library} GPU requested but CUDA is not available. "
                    f"Run with device='cpu' or install {library} with CUDA support."
                )
        elif library == "catboost":
            lib_params["task_type"] = "GPU"
            lib_params["devices"] = "0"

        gpu_overrides = canonical.get("gpu", {}).get(library, {})
        lib_params.update(gpu_overrides)

    return lib_params


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


def create_model(
    library: str,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    gpu_adjustments: bool = True,
    task_type: str = "regression",
):
    """Create an unfitted sklearn-compatible GBM model.

    Parameters
    ----------
    library : str
        One of "xgboost", "lightgbm", "catboost", "sklearn_hgb".
    params : dict, optional
        Canonical params. Defaults to "medium" preset.
    device : str
        "cpu" or "gpu".
    gpu_adjustments : bool
        If True and device="gpu", applies GPU-specific params from config.
    task_type : str
        "regression" or "classification".

    Returns
    -------
    Unfitted sklearn-compatible model (regressor or classifier).
    """
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    if params is None:
        params = load_gbm_config("medium")

    effective_device = device if gpu_adjustments else "cpu"
    lib_params = make_model_params(params, library, effective_device)

    if task_type == "classification":
        if library == "sklearn_hgb":
            return HistGradientBoostingClassifier(**lib_params)
        if library == "xgboost":
            return xgb.XGBClassifier(**lib_params)
        if library == "lightgbm":
            return lgb.LGBMClassifier(**lib_params)
        if library == "catboost":
            return cb.CatBoostClassifier(**lib_params)
    else:
        if library == "sklearn_hgb":
            return HistGradientBoostingRegressor(**lib_params)
        if library == "xgboost":
            return xgb.XGBRegressor(**lib_params)
        if library == "lightgbm":
            return lgb.LGBMRegressor(**lib_params)
        if library == "catboost":
            return cb.CatBoostRegressor(**lib_params)

    raise ValueError(f"Unknown library: {library}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Checkpoint Prediction
# ---------------------------------------------------------------------------


def _predict_at_checkpoint(model, X: np.ndarray, n_trees: int, library: str) -> np.ndarray:
    """Predict using only the first `n_trees` trees from a trained model.

    Supports partial-iteration prediction for LightGBM, XGBoost, and CatBoost.
    For sklearn HistGradientBoosting, returns full prediction (no checkpoint support).
    """
    if library == "lightgbm":
        return model.predict(X, num_iteration=n_trees)
    elif library == "xgboost":
        return model.predict(X, iteration_range=(0, n_trees))
    elif library == "catboost":
        return model.predict(X, ntree_end=n_trees)
    else:  # sklearn_hgb — no checkpoint support
        return model.predict(X)


def _extract_feature_importance(
    model, feature_names: list[str], library: str, top_n: int = 10
) -> list[tuple[str, float]]:
    """Extract top-N feature importances from a fitted model."""
    try:
        importances = model.feature_importances_
        if importances is None or len(importances) == 0:
            return []
        pairs = sorted(
            zip(feature_names, importances, strict=False), key=lambda x: abs(x[1]), reverse=True
        )
        return pairs[:top_n]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Config-driven GBM training (public API for notebooks)
# ---------------------------------------------------------------------------


def prepare_gbm_folds(
    dataset_pd,
    splits: list[dict[str, Any]],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str = "symbol",
    task_type: str = "regression",
    class_values: list | None = None,
    temporal_by_fold=None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    train_sample_frac: float = 1.0,
    eval_label_col: str | None = None,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Prepare CV fold data for GBM training.

    Unlike linear folds, GBM folds:
    - Use float32 (LightGBM native precision)
    - No imputation or scaling (GBM handles NaN natively)
    - Include remapped labels for classification (0-indexed for LightGBM)

    Parameters
    ----------
    dataset_pd : pandas DataFrame
        Full dataset.
    splits : list[dict]
        Walk-forward splits.
    feature_names : list[str]
        Feature column names.
    label_col, date_col, entity_col : str
        Column names.
    task_type : str
        "regression" or "classification".
    class_values : list, optional
        Sorted unique class values for classification.
    temporal_by_fold : pd.DataFrame, optional
        Per-fold temporal features with a 'fold' column.
    temporal_keys : list[str], optional
        Join keys for temporal features.
    temporal_feature_names : list[str], optional
        Temporal feature column names to replace per fold.
    train_sample_frac : float, optional
        Fraction of training rows to keep per fold (1.0 = keep all).
        Walk-forward CV structure is preserved (date ranges unchanged);
        only the within-fold row density is reduced. Validation set is
        NEVER sampled — OOS IC is always computed on the full val slice.
        Seed is tied to fold_id for reproducibility. Use < 1.0 for
        memory/compute-constrained runs on large datasets (e.g.,
        nasdaq100 minute bars). Default 1.0.
    eval_label_col : str, optional
        Continuous return used for classification IC. The discrete label
        remains the fitting target and is retained as ``y_val``.
    seed : int
        Base seed for optional within-fold training subsampling.

    Returns
    -------
    list[dict]
        Each dict has: fold, X_train, y_train, y_train_lgb, X_val, y_val,
        y_val_lgb, dates, entities, n_train, n_val.
    """
    from utils.modeling import replace_temporal_columns

    dates_series = dataset_pd[date_col]
    entity_series = dataset_pd.get(entity_col)
    is_classification = task_type == "classification" and class_values
    has_fold_temporal = temporal_by_fold is not None and temporal_keys and temporal_feature_names

    folds = []
    for split in splits:
        fold_id = split["fold"]
        train_mask = (dates_series >= split["train_start"]) & (dates_series <= split["train_end"])
        val_start = split.get("val_start", split.get("test_start"))
        val_end = split.get("val_end", split.get("test_end"))
        val_mask = (dates_series >= val_start) & (dates_series <= val_end)

        if has_fold_temporal:
            assert temporal_by_fold is not None
            assert temporal_keys is not None
            assert temporal_feature_names is not None
            train_rows = replace_temporal_columns(
                dataset_pd,
                train_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            val_rows = replace_temporal_columns(
                dataset_pd,
                val_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            X_train = train_rows[feature_names].values.astype(np.float32)
            y_train = train_rows[label_col].values.astype(np.float32)
            X_val = val_rows[feature_names].values.astype(np.float32)
            y_val = val_rows[label_col].values.astype(np.float32)
            y_eval = val_rows[eval_label_col].values.astype(np.float32) if eval_label_col else None
            val_dates = val_rows[date_col].values
            del train_rows, val_rows
        else:
            X_train = dataset_pd.loc[train_mask, feature_names].values.astype(np.float32)
            y_train = dataset_pd.loc[train_mask, label_col].values.astype(np.float32)
            X_val = dataset_pd.loc[val_mask, feature_names].values.astype(np.float32)
            y_val = dataset_pd.loc[val_mask, label_col].values.astype(np.float32)
            y_eval = (
                dataset_pd.loc[val_mask, eval_label_col].values.astype(np.float32)
                if eval_label_col
                else None
            )
            val_dates = dataset_pd.loc[val_mask, date_col].values

        # Drop NaN labels
        tv = ~np.isnan(y_train)
        vv = ~np.isnan(y_val)
        X_train, y_train = X_train[tv], y_train[tv]
        X_val, y_val = X_val[vv], y_val[vv]
        if y_eval is not None:
            y_eval = y_eval[vv]
        val_dates = val_dates[vv]
        val_entities = (
            dataset_pd.loc[val_mask, entity_col].values[vv] if entity_series is not None else None
        )

        # Optional train subsample (never touch val — OOS IC uses full val slice).
        # Seed is tied to fold_id for reproducibility.
        if 0.0 < train_sample_frac < 1.0 and len(X_train) > 0:
            n_keep = max(1, int(len(X_train) * train_sample_frac))
            rng = np.random.default_rng(seed + fold_id)
            keep_idx = rng.choice(len(X_train), size=n_keep, replace=False)
            keep_idx.sort()  # preserve row order
            X_train = X_train[keep_idx]
            y_train = y_train[keep_idx]

        # Classification: remap labels to 0-indexed for LightGBM
        if is_classification:
            assert class_values is not None
            y_train_lgb, _ = _remap_labels_for_lgb(y_train.astype(int), class_values)
            y_val_lgb, _ = _remap_labels_for_lgb(y_val.astype(int), class_values)
        else:
            y_train_lgb = y_train
            y_val_lgb = y_val

        folds.append(
            {
                "fold": split["fold"],
                "X_train": X_train,
                "y_train": y_train,
                "y_train_lgb": y_train_lgb,
                "X_val": X_val,
                "y_val": y_val,
                "y_val_lgb": y_val_lgb,
                "y_eval": y_eval,
                "dates": val_dates,
                "entities": val_entities,
                "n_train": len(X_train),
                "n_val": len(X_val),
            }
        )

    return folds


def _checkpoint_metrics_from_predictions(
    predictions: list[dict[str, Any]],
    checkpoints: list[int] | tuple[int, ...],
) -> dict[int, dict[str, float]]:
    """Score each checkpoint on one complete per-decision-time IC series."""
    metrics: dict[int, dict[str, float]] = {}
    for checkpoint in checkpoints:
        frames = []
        for entry in predictions:
            if entry["n_trees"] != checkpoint:
                continue
            target = entry["y_eval"] if entry.get("y_eval") is not None else entry["y_true"]
            frames.append(
                pl.DataFrame(
                    {
                        "timestamp": entry["dates"],
                        "symbol": entry["entities"],
                        "y_true": target,
                        "y_pred": entry["y_pred"],
                    }
                )
            )
        if not frames:
            raise ValueError(f"Checkpoint {checkpoint} has no validation predictions")
        complete = pl.concat(frames)
        metric = cross_sectional_ic(
            complete,
            complete,
            pred_col="y_pred",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            min_obs=5,
        )
        metrics[int(checkpoint)] = {
            "ic_mean": float(metric["ic_mean"]),
            "ic_std": float(metric.get("ic_std", 0.0)),
        }
    return metrics


def _learning_curves_from_predictions(
    config_name: str,
    predictions: list[dict[str, Any]],
    checkpoints: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    metrics = _checkpoint_metrics_from_predictions(predictions, checkpoints)
    return [
        {
            "config": config_name,
            "iteration": checkpoint,
            "ic_mean": metrics[checkpoint]["ic_mean"],
            "ic_std": metrics[checkpoint]["ic_std"],
        }
        for checkpoint in checkpoints
    ]


def load_cached_gbm_config(
    *,
    case_study: str,
    training_spec: dict[str, Any],
    config_name: str,
    prediction_split: str,
    date_col: str,
    entity_col: str,
    eval_col: str | None,
    expected_iterations: tuple[int, ...],
    expected_keys: pl.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one complete GBM config or reject an incomplete cache."""
    from case_studies.utils.registry import (
        get_training_dir,
        load_prediction_metrics,
        load_prediction_sets,
        prediction_dir,
        training_hash_from_spec,
    )

    training_hash = training_hash_from_spec(training_spec)
    prediction_sets = load_prediction_sets(
        case_study,
        training_hash=training_hash,
        split=prediction_split,
    )
    required_metadata = {"prediction_hash", "checkpoint_value", "checkpoint_kind"}
    missing_metadata = required_metadata - set(prediction_sets.columns)
    if missing_metadata:
        raise ValueError(f"Cached {config_name} metadata is missing {sorted(missing_metadata)}")
    if prediction_sets.height != 1:
        raise ValueError(f"Cached {config_name} must contain exactly one prediction set")
    row = prediction_sets.row(0, named=True)
    if row["checkpoint_kind"] != "iteration" or row["checkpoint_value"] is None:
        raise ValueError(f"Cached {config_name} has invalid iteration checkpoint metadata")

    curves_path = get_training_dir(case_study, training_spec) / "learning_curves.parquet"
    if not curves_path.exists():
        raise ValueError(f"Cached {config_name} is missing learning curves")
    curves = pl.read_parquet(curves_path)
    required_curve_cols = {"config", "iteration", "ic_mean", "ic_std"}
    missing_curve_cols = required_curve_cols - set(curves.columns)
    if missing_curve_cols:
        raise ValueError(
            f"Cached {config_name} learning curves are missing {sorted(missing_curve_cols)}"
        )
    if curves.height != len(expected_iterations):
        raise ValueError(
            f"Cached {config_name} learning curves have incomplete checkpoint coverage"
        )
    if curves.select(pl.col(list(required_curve_cols)).null_count()).row(0) != (0,) * len(
        required_curve_cols
    ):
        raise ValueError(f"Cached {config_name} learning curves contain nulls")
    observed_iterations = tuple(sorted(int(value) for value in curves["iteration"].to_list()))
    if (
        observed_iterations != expected_iterations
        or curves["iteration"].n_unique() != curves.height
    ):
        raise ValueError(f"Cached {config_name} learning curves have invalid checkpoints")
    if set(curves["config"].unique().to_list()) != {config_name}:
        raise ValueError(f"Cached {config_name} learning curves contain another config")
    best = curves.sort("ic_mean", descending=True).row(0, named=True)
    if int(row["checkpoint_value"]) != int(best["iteration"]):
        raise ValueError(f"Cached {config_name} prediction checkpoint is not the curve leader")

    artifact_path = prediction_dir(case_study, row["prediction_hash"]) / "predictions.parquet"
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    predictions = pl.read_parquet(artifact_path)
    required_prediction_cols = {date_col, entity_col, "fold", "prediction", "actual"}
    if eval_col:
        required_prediction_cols.add(eval_col)
    missing_prediction_cols = required_prediction_cols - set(predictions.columns)
    if missing_prediction_cols:
        raise ValueError(
            f"Cached {config_name} predictions are missing {sorted(missing_prediction_cols)}"
        )
    if predictions.select(pl.col(list(required_prediction_cols)).null_count()).row(0) != (0,) * len(
        required_prediction_cols
    ):
        raise ValueError(f"Cached {config_name} predictions contain nulls")
    key_cols = [date_col, entity_col, "fold"]
    actual_keys = predictions.select(key_cols)
    if actual_keys.n_unique() != predictions.height:
        raise ValueError(f"Cached {config_name} predictions contain duplicate keys")
    if not actual_keys.sort(key_cols).equals(expected_keys.select(key_cols).sort(key_cols)):
        raise ValueError(f"Cached {config_name} prediction key or fold coverage is incomplete")

    target_col = eval_col or "actual"
    metric = cross_sectional_ic(
        predictions,
        predictions,
        pred_col="prediction",
        ret_col=target_col,
        date_col=date_col,
        entity_col=entity_col,
        min_obs=5,
    )
    registry_metrics = load_prediction_metrics(case_study, prediction_hash=row["prediction_hash"])
    if registry_metrics.height != 1:
        raise ValueError(f"Cached {config_name} has invalid prediction metrics")
    comparisons = {
        "curve mean": (float(best["ic_mean"]), float(metric["ic_mean"])),
        "curve std": (float(best["ic_std"]), float(metric.get("ic_std", 0.0))),
        "registry mean": (float(registry_metrics["ic_mean"][0]), float(metric["ic_mean"])),
        "registry std": (
            float(registry_metrics["ic_std"][0]),
            float(metric.get("ic_std", 0.0)),
        ),
    }
    mismatches = {
        name: values
        for name, values in comparisons.items()
        if not np.isclose(values[0], values[1], atol=1e-12, rtol=0.0)
    }
    if mismatches:
        raise ValueError(f"Cached {config_name} metric mismatch: {mismatches}")

    result = {
        "config_name": config_name,
        "best_iter": int(best["iteration"]),
        "best_ic": float(metric["ic_mean"]),
        "best_ic_std": float(metric.get("ic_std", 0.0)),
        "elapsed_s": 0.0,
        "learning_curves": curves.to_dicts(),
        "cached": True,
    }
    return result, curves.to_dicts()


def train_gbm_config(
    config: dict[str, Any],
    fold_data: list[dict[str, Any]],
    *,
    feature_names: list[str],
    device: str = "cpu",
    num_threads: int = DEFAULT_GBM_CPU_THREADS,
    seed: int = RANDOM_SEED,
    max_bin: int | None = None,
    entity_col: str = "symbol",
    date_col: str = "timestamp",
    task_type: str = "regression",
    class_values: list | None = None,
    save_dir: Path | None = None,
    effective_params_by_fold: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train a single GBM config across all CV folds.

    Trains to max_iterations, evaluates cross-sectional IC at checkpoints,
    and returns the best checkpoint along with predictions and learning curves.

    Parameters
    ----------
    config : dict
        Preset dict with config_name, params, max_iterations, checkpoint_interval.
    fold_data : list[dict]
        From prepare_gbm_folds().
    feature_names : list[str]
        For feature importance extraction.
    device : str
        "cpu" or "cuda"/"gpu". GPU requests fail when CUDA is unavailable.
    num_threads : int
        Fixed LightGBM CPU thread count. Included in deterministic execution.
    seed : int
        Seed applied to every LightGBM stochastic mechanism.
    max_bin : int, optional
        Override max_bin (GPU typically needs 63).
    entity_col, date_col : str
        For IC computation.
    task_type : str
        "regression" or "classification".
    class_values : list, optional
        For classification score extraction.
    save_dir : Path, optional
        Save booster files here.

    Returns
    -------
    dict with keys:
        config_name, best_iter, best_ic, elapsed_s, fold_ics (dict[int, list]),
        learning_curves (list[dict]), predictions (list[dict]), top_features.
    """
    import lightgbm as lgb

    config_name = config["config_name"]
    num_boost_round = config.get("max_iterations", 500)
    is_classification = task_type == "classification" and class_values

    # Build LightGBM params from preset
    base_params = dict(config["params"])
    base_params["metric"] = "None"
    base_params["verbosity"] = base_params.get("verbosity", -1)

    # Runtime settings are recorded as provenance by the caller. Numerical
    # model parameters such as max_bin are declared separately in the hashed
    # training spec and never inferred from this runtime backend.
    base_params.update(lightgbm_runtime_params(device, num_threads=num_threads, seed=seed))
    if max_bin is not None:
        base_params["max_bin"] = max_bin
    if (
        base_params.get("objective") == "huber"
        and "alpha" not in base_params
        and effective_params_by_fold is None
    ):
        scale = config.get("huber_alpha_scale")
        if scale is None:
            raise ValueError("Huber GBM configs must declare huber_alpha_scale or alpha")
        effective_params_by_fold = {}
        for fold in fold_data:
            params = dict(base_params)
            params["alpha"] = _scaled_huber_alpha(float(scale), fold["y_train"])
            effective_params_by_fold[str(int(fold["fold"]))] = params

    # Classification: ensure num_class for multiclass
    if is_classification and class_values and len(class_values) > 2:
        base_params["num_class"] = len(class_values)

    checkpoints = list(gbm_checkpoint_iterations(config))

    t0 = time.perf_counter()
    checkpoint_ics: dict[int, list[float]] = {cp: [] for cp in checkpoints}
    all_preds: list[dict] = []
    top_features: list[tuple[str, float]] = []
    booster_dir = save_dir / "boosters" if save_dir else None
    if booster_dir:
        booster_dir.mkdir(parents=True, exist_ok=True)

    for fd in fold_data:
        if fd["n_train"] == 0 or fd["n_val"] == 0:
            continue

        params = (
            dict(effective_params_by_fold[str(int(fd["fold"]))])
            if effective_params_by_fold is not None
            else dict(base_params)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtrain = lgb.Dataset(
                fd["X_train"],
                label=fd["y_train_lgb"],
                feature_name=feature_names,
                free_raw_data=False,
            )
            # Print progress every 50 iterations so long runs aren't silent.
            # Also print per-fold heartbeat so we see which fold is active
            # on large datasets.
            print(
                f"      fold {fd['fold']}: training "
                f"n_train={fd['n_train']:,} n_val={fd['n_val']:,} "
                f"trees={num_boost_round} num_leaves={params.get('num_leaves', '?')} "
                f"obj={params.get('objective', '?')}",
                flush=True,
            )
            _fold_t0 = time.perf_counter()
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                callbacks=[lgb.log_evaluation(period=50)],
            )
            print(
                f"      fold {fd['fold']}: done in {time.perf_counter() - _fold_t0:.0f}s",
                flush=True,
            )

        if booster_dir:
            model.save_model(str(booster_dir / f"fold_{fd['fold']}.txt"))

        # Feature importance (first fold only)
        if not top_features:
            imp = model.feature_importance(importance_type="gain")
            pairs = sorted(zip(feature_names, imp, strict=False), key=lambda x: x[1], reverse=True)
            top_features = pairs[:10]

        # Predict at all checkpoints
        for cp in checkpoints:
            raw_preds = model.predict(fd["X_val"], num_iteration=cp)
            if is_classification:
                assert class_values is not None
                preds = _extract_gbm_score(np.asarray(raw_preds), class_values, len(fd["X_val"]))
            else:
                preds = raw_preds
            ic_frame = pl.DataFrame(
                {
                    "timestamp": fd["dates"],
                    "symbol": fd["entities"],
                    "y_true": fd["y_eval"] if fd.get("y_eval") is not None else fd["y_val"],
                    "y_pred": preds,
                }
            )
            ic = cross_sectional_ic(
                ic_frame,
                ic_frame,
                pred_col="y_pred",
                ret_col="y_true",
                date_col="timestamp",
                entity_col="symbol",
                min_obs=5,
            )["ic_mean"]
            checkpoint_ics[cp].append(ic)
            all_preds.append(
                {
                    "dates": fd["dates"],
                    "entities": fd["entities"],
                    "y_true": fd["y_val"],
                    "y_eval": fd.get("y_eval"),
                    "y_pred": preds,
                    "fold": fd["fold"],
                    "n_trees": cp,
                }
            )

        del dtrain, model

    # Select on the complete per-decision-time IC series. Averaging fold means
    # would give folds equal weight even when they contain different month counts.
    checkpoint_metrics = _checkpoint_metrics_from_predictions(all_preds, checkpoints)
    best_cp = max(checkpoints, key=lambda cp: checkpoint_metrics[cp]["ic_mean"])
    best_ic = float(checkpoint_metrics[best_cp]["ic_mean"])
    best_ic_std = float(checkpoint_metrics[best_cp]["ic_std"])
    elapsed = time.perf_counter() - t0

    # Learning curves
    curves = [
        {
            "config": config_name,
            "iteration": cp,
            "ic_mean": float(checkpoint_metrics[cp]["ic_mean"]),
            "ic_std": float(checkpoint_metrics[cp]["ic_std"]),
        }
        for cp in checkpoints
    ]

    # Per-fold metrics at best checkpoint
    def _fold_ic(e: dict[str, Any]) -> float:
        ic_target = e["y_eval"] if e.get("y_eval") is not None else e["y_true"]
        frame = pl.DataFrame(
            {
                "timestamp": e["dates"],
                "symbol": e["entities"],
                "y_true": ic_target,
                "y_pred": e["y_pred"],
            }
        )
        return cross_sectional_ic(
            frame,
            frame,
            pred_col="y_pred",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            min_obs=5,
        )["ic_mean"]

    fold_metrics = [
        {
            "fold_id": e["fold"],
            "ic_mean": _fold_ic(e),
            "n_train": [fd for fd in fold_data if fd["fold"] == e["fold"]][0]["n_train"],
            "n_test": len(e["y_true"]),
        }
        for e in all_preds
        if e["n_trees"] == best_cp
    ]

    gc.collect()

    return {
        "config_name": config_name,
        "best_iter": best_cp,
        "best_ic": best_ic,
        "best_ic_std": best_ic_std,
        "elapsed_s": elapsed,
        "checkpoint_ics": checkpoint_ics,
        "checkpoint_metrics": checkpoint_metrics,
        "learning_curves": curves,
        "predictions": all_preds,
        "fold_metrics": fold_metrics,
        "top_features": top_features,
        "effective_params_by_fold": effective_params_by_fold,
    }


def _make_lgb_native_params(canonical: dict[str, Any], device: str) -> dict[str, Any]:
    """Convert canonical config to native lgb.train() params dict.

    Strips sklearn-only keys (n_estimators) and disables built-in metrics.
    """
    params = make_model_params(canonical, "lightgbm", device)
    params.pop("n_estimators", None)
    params.pop("n_jobs", None)
    params["metric"] = "None"
    params["seed"] = params.pop("random_state", RANDOM_SEED)
    # Subsampling requires bagging_freq in native API
    if params.get("subsample", 1.0) < 1.0:
        params["bagging_freq"] = 1
    return params


def _remap_labels_for_lgb(y: np.ndarray, class_values: list) -> tuple[np.ndarray, dict]:
    """Remap class labels to 0-indexed for LightGBM native API.

    E.g., {-1, 0, 1} -> {0, 1, 2}. Returns (remapped_y, mapping_dict).
    """
    sorted_vals = sorted(class_values)
    mapping = {v: i for i, v in enumerate(sorted_vals)}
    remapped = np.array([mapping[v] for v in y], dtype=np.int32)
    return remapped, mapping


def _extract_gbm_score(raw_preds: np.ndarray, class_values: list, n_samples: int) -> np.ndarray:
    """Extract continuous score from GBM classification output for IC computation.

    Binary: raw_preds is P(class=1) directly.
    Multiclass: raw_preds shape = (n_samples, n_classes) -> expected value.
    """
    sorted_vals = sorted(class_values)
    if len(sorted_vals) == 2:
        # Binary: LightGBM native returns P(class=1) directly
        return raw_preds.ravel()
    # Multiclass: raw_preds shape = (n_samples, n_classes)
    proba = raw_preds.reshape(n_samples, len(sorted_vals))
    return proba @ np.array(sorted_vals, dtype=np.float64)


def register_gbm_result(
    case_study_id: str,
    result: dict,
    cfg: dict,
    label_col: str,
    n_folds: int,
    *,
    max_bin: int | None = None,
    entry_point: str = "07_gbm",
    date_col: str = "timestamp",
    entity_col: str = "symbol",
    train_sample_frac: float = 1.0,
    prediction_split: str = "validation",
    runtime_params: dict[str, Any] | None = None,
    task_type: str = "regression",
    class_values: list | None = None,
    eval_col: str | None = None,
    training_spec: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    replace_existing: bool = False,
) -> str:
    """Register a single GBM config's result to the registry.

    Called INSIDE the training loop (per-config) so each config is persisted
    immediately after it trains. This protects against interruption losing
    all completed configs — a failure rule enforced by the memory file
    ``feedback_incremental_save_violation.md``.

    Writes training_run, prediction_set (best-iter predictions),
    learning_curves.parquet, and fold_metrics.parquet.

    Returns
    -------
    str
        The training_hash for the registered run.
    """
    import polars as pl

    from case_studies.utils.registry import (
        build_training_spec,
        clear_prediction_sets,
        get_training_dir,
        register_prediction_set,
        register_training_run,
        training_hash_from_spec,
    )

    if training_spec is None:
        spec_extra = dict(extra_params or {})
        if input_data_spec is not None:
            existing_input = spec_extra.get("input_data_spec")
            if existing_input is not None and existing_input != input_data_spec:
                raise ValueError("extra_params and input_data_spec disagree")
            spec_extra["input_data_spec"] = input_data_spec
        spec = build_training_spec(
            cfg["family"],
            cfg["config_name"],
            label_col,
            n_folds=n_folds,
            max_bin=max_bin,
            checkpoint_interval=cfg.get("checkpoint_interval", 50),
            train_sample_frac=train_sample_frac,
            extra_params=spec_extra or None,
        )
    else:
        spec = dict(training_spec)
        expected_identity = {
            "family": cfg["family"],
            "config_name": cfg["config_name"],
            "label": label_col,
            "n_folds": n_folds,
        }
        mismatches = {
            key: (spec.get(key), value)
            for key, value in expected_identity.items()
            if spec.get(key) != value
        }
        if mismatches:
            raise ValueError(f"training_spec disagrees with registration inputs: {mismatches}")
        if (
            input_data_spec is not None
            and spec.get("params", {}).get("input_data_spec") != input_data_spec
        ):
            raise ValueError("training_spec disagrees with input_data_spec")
    expected_hash = training_hash_from_spec(spec)
    if replace_existing:
        clear_prediction_sets(case_study_id, expected_hash, split=prediction_split)
    t_hash = register_training_run(
        case_study_id,
        spec=spec,
        entry_point=entry_point,
        elapsed_s=result.get("elapsed_s"),
        runtime_provenance=runtime_params,
    )
    if t_hash != expected_hash:
        raise RuntimeError(f"registered GBM hash drifted: expected {expected_hash}, got {t_hash}")

    # Best-checkpoint predictions as a DataFrame
    best_preds = [e for e in result["predictions"] if e["n_trees"] == result["best_iter"]]
    if best_preds:
        pred_rows = []
        for e in best_preds:
            n = len(e["y_pred"])
            data = {
                date_col: e["dates"],
                entity_col: e["entities"] if e["entities"] is not None else ["unknown"] * n,
                "fold": [e["fold"]] * n,
                "prediction": e["y_pred"],
                "actual": e["y_true"],
            }
            if e.get("y_eval") is not None:
                data["eval_actual"] = e["y_eval"]
            pred_rows.append(pl.DataFrame(data))
        pred_df = pl.concat(pred_rows).to_pandas()
        resolved_eval_col = eval_col or ("eval_actual" if task_type == "classification" else None)
        register_prediction_set(
            case_study_id,
            t_hash,
            split=prediction_split,
            predictions=pred_df,
            metrics={
                "ic_mean": result["best_ic"],
                "ic_std": result["best_ic_std"],
            },
            task_type=task_type,
            class_values=class_values,
            eval_col=resolved_eval_col,
            label=label_col,
            checkpoint_value=int(result["best_iter"]),
            checkpoint_kind="iteration",
        )

    # Save learning curves and fold metrics to registry training dir
    reg_dir = get_training_dir(case_study_id, spec)
    cfg_curves = list(result.get("learning_curves", []))
    if cfg_curves:
        pl.DataFrame(cfg_curves).write_parquet(reg_dir / "learning_curves.parquet")

    cfg_fold_metrics = result.get("fold_metrics", [])
    if cfg_fold_metrics:
        fm_df = pl.DataFrame(cfg_fold_metrics)
        if "config_name" not in fm_df.columns:
            fm_df = fm_df.with_columns(pl.lit(result["config_name"]).alias("config_name"))
        fm_df.write_parquet(reg_dir / "fold_metrics.parquet")

    return t_hash


def _gbm_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gbm_source_identity() -> dict[str, int | str]:
    """The behaviour of this runner, declared rather than fingerprinted.

    This used to be the SHA-256 of ``gbm.py`` and ``utils/modeling.py``. ``gbm.py`` is nearly three
    thousand lines, so every edit to any part of it - a comment, a log line, a fix to a function no
    boosted tree ever calls - invalidated every GBM result ever registered. That is unworkable
    against the rule that a fix which does not change a result must not force a refit.

    What replaces it is a declaration. ``GBM_RUNNER_VERSION`` is bumped when a change to this module
    would change a fitted result, ``FOLD_PREPARATION_VERSION`` covers the shared fold preparation
    the same way, and ``GBM_PREPROCESSING_ID`` names the cast applied to a fold.
    ``tests/test_gbm_identity.py`` pins the predictions these versions claim to describe and fails
    when they move without a bump, so the declaration is checked rather than trusted.
    """
    return {
        "gbm_runner": GBM_RUNNER_VERSION,
        "fold_preparation": FOLD_PREPARATION_VERSION,
        "preprocessing": GBM_PREPROCESSING_ID,
    }


def _gbm_runtime_identity() -> dict[str, str]:
    return {
        "lightgbm": importlib.metadata.version("lightgbm"),
        "numpy": importlib.metadata.version("numpy"),
    }


def _gbm_runtime_provenance(study: Study, device: str) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(study.release_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    lock_path = study.release_root / "uv.lock"
    record: dict[str, Any] = {
        "device": device,
        "entry_point": "case_studies.utils.gbm",
        "lock_digest": _gbm_sha256(lock_path) if lock_path.is_file() else None,
        "packages": _gbm_runtime_identity(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": commit,
    }
    if device == "cuda":
        record["cuda_runtime"] = torch.version.cuda
        record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return record


def _gbm_normalize_folds(splits: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    fields = ("fold", "train_start", "train_end", "val_start", "val_end")
    return tuple(
        {
            key: int(split[key]) if key == "fold" else str(split[key])
            for key in fields
            if split.get(key) is not None
        }
        for split in splits
    )


def _gbm_select_splits(
    mds,
    request: dict[str, Any],
    label_timeline: pl.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cv = request.get("cv")
    if cv is None:
        splits = list(mds.splits)
        normalized = _gbm_normalize_folds(splits)
        cv_record = {
            "request": {"source": "case_study_default"},
            "folds": list(normalized),
            "identity": value_digest(pl.DataFrame(list(normalized))),
        }
    else:
        resolved = cv.resolve(label_timeline, date_col=mds.date_col)
        splits = [dict(fold) for fold in resolved.normalized_folds]
        cv_record = resolved.as_dict()
    requested_folds = request["preview_reductions"].get("folds")
    if requested_folds is not None:
        selected = {int(fold) for fold in requested_folds}
        splits = [split for split in splits if int(split["fold"]) in selected]
        if {int(split["fold"]) for split in splits} != selected:
            raise ValueError("preview fold reduction refers to an unavailable fold")
        cv_record = {**cv_record, "preview_folds": sorted(selected)}
    if not splits:
        raise ValueError("GBM request resolved no cross-validation folds")
    return splits, cv_record


def _gbm_expected_keys(folds: list[dict[str, Any]], entity_col: str, date_col: str) -> pl.DataFrame:
    frames = []
    for fold in folds:
        frame = pl.DataFrame(
            {
                "symbol": fold["entities"],
                "timestamp": fold["dates"],
                "fold": [int(fold["fold"])] * int(fold["n_val"]),
            }
        )
        frames.append(frame)
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("GBM request produced duplicate expected prediction keys")
    return expected


def _validate_lightgbm_params(params: dict[str, Any]) -> None:
    import lightgbm as lgb

    aliases = lgb.basic._ConfigAliases  # noqa: SLF001
    aliases.get("num_leaves")
    valid = {alias for values in (aliases.aliases or {}).values() for alias in values}
    unknown = set(params) - valid
    if unknown:
        raise ValueError(f"unsupported LightGBM parameters: {sorted(unknown)}")


def _scaled_huber_alpha(scale: float, labels: np.ndarray) -> float:
    """Resolve LightGBM's residual-unit Huber delta from one training fold.

    Quantized for the reason in :mod:`case_studies.utils.derived_params`: the delta is computed
    from the training labels, so its last digits carry the reduction order rather than
    information, and an unrounded value gives one declared configuration two training identities.
    """
    delta = max(scale * float(np.nanstd(labels)), float(np.finfo(np.float32).eps))
    return quantize_derived(delta)


def _gbm_effective_params_by_fold(
    config: dict[str, Any],
    folds: list[dict[str, Any]],
    *,
    device: str,
    max_bin: int,
    num_threads: int,
    seed: int,
    task_type: str = "regression",
    class_values: list[Any] | tuple[Any, ...] = (),
) -> dict[str, dict[str, Any]]:
    base = dict(config["params"])
    base["metric"] = "None"
    base["verbosity"] = base.get("verbosity", -1)
    base["max_bin"] = max_bin
    base.update(lightgbm_runtime_params(device, num_threads=num_threads, seed=seed))
    if task_type == "classification" and len(class_values) > 2:
        base["num_class"] = len(class_values)
    scale = config.get("huber_alpha_scale")
    if base.get("objective") == "huber" and "alpha" not in base and scale is None:
        raise ValueError("Huber GBM configs must declare huber_alpha_scale or alpha")
    effective = {}
    for fold in folds:
        params = dict(base)
        if params.get("objective") == "huber" and "alpha" not in params:
            assert scale is not None
            params["alpha"] = _scaled_huber_alpha(float(scale), fold["y_train"])
        _validate_lightgbm_params(params)
        effective[str(int(fold["fold"]))] = params
    return effective


def _load_gbm_request_config(
    study: Study,
    label: str,
    config_name: str,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from utils.modeling import load_configs

    configs = load_configs(study.case_study, label, family="gbm")
    matches = [config for config in configs if config["config_name"] == config_name]
    if len(matches) != 1:
        raise ValueError(f"unknown GBM config {config_name!r}")
    config = {**matches[0], "params": dict(matches[0]["params"])}
    request_fields = {key: value for key, value in overrides.items() if key in _GBM_REQUEST_FIELDS}
    config.update(
        {
            key: value
            for key, value in request_fields.items()
            if key in {"checkpoint_interval", "huber_alpha_scale", "max_iterations"}
        }
    )
    config["params"].update(
        {key: value for key, value in overrides.items() if key not in _GBM_REQUEST_FIELDS}
    )
    return config, request_fields


def _gbm_expected_keys_from_dataset(mds, splits: list[dict[str, Any]]) -> pl.DataFrame:
    date_dtype = mds.dataset.schema[mds.date_col]
    label_valid = pl.col(mds.label_col).is_not_null()
    if mds.dataset.schema[mds.label_col] in {pl.Float32, pl.Float64}:
        label_valid &= pl.col(mds.label_col).is_not_nan()
    frames = []
    for split in splits:
        val_start = split.get("val_start", split.get("test_start"))
        val_end = split.get("val_end", split.get("test_end"))
        frame = (
            mds.dataset.filter(
                pl.col(mds.date_col).is_between(
                    pl.lit(val_start).cast(date_dtype, strict=False),
                    pl.lit(val_end).cast(date_dtype, strict=False),
                    closed="both",
                )
                & label_valid
            )
            .select(
                pl.col(mds.entity_cols[0]).alias("symbol"),
                pl.col(mds.date_col).alias("timestamp"),
            )
            .with_columns(pl.lit(int(split["fold"]), dtype=pl.Int64).alias("fold"))
        )
        if frame.is_empty():
            raise ValueError(f"GBM request produced no validation keys for fold {split['fold']}")
        frames.append(frame)
    expected = pl.concat(frames).sort("symbol", "timestamp", "fold")
    if expected.n_unique(["symbol", "timestamp", "fold"]) != expected.height:
        raise ValueError("GBM request produced duplicate expected prediction keys")
    return expected


def _load_gbm_batch_base(
    study: Study,
    request: dict[str, Any],
    *,
    inputs: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    from utils.modeling import load_modeling_dataset

    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _GBM_PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported GBM preview reductions: {sorted(unknown_reductions)}")
    study.require_writable()
    study.activate(tier)
    max_symbols = int(reductions.get("max_symbols", 0))
    train_sample_frac = float(reductions.get("train_sample_frac", 1.0))
    if not 0 < train_sample_frac <= 1:
        raise ValueError("train_sample_frac must be in (0, 1]")
    if inputs is None:
        label_ref = study.labels.get(request["label"], execution_tier=tier)
        mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=max_symbols)
        dataset_pd = mds.dataset.to_pandas()
    else:
        label_ref, mds, dataset_pd = inputs
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("GBM runner requires timestamp and an entity key")
    entity_col = mds.entity_cols[0]
    if entity_col not in {"product", "symbol"}:
        raise ValueError(f"GBM runner does not support entity key {entity_col!r}")
    splits, cv_record = _gbm_select_splits(
        mds,
        request,
        label_ref.load().select(mds.date_col).unique(),
    )
    if (
        request.get("cv") is not None
        and mds.temporal_by_fold is not None
        and mds.temporal_keys
        and mds.temporal_feature_names
    ):
        require_fold_scoped_temporal_compatibility(splits, mds.temporal_artifact_splits)
    return {
        "label_ref": label_ref,
        "mds": mds,
        "dataset_pd": dataset_pd,
        "splits": splits,
        "cv_record": cv_record,
        "expected": _gbm_expected_keys_from_dataset(mds, splits),
        "train_sample_frac": train_sample_frac,
    }


def _gbm_execution_settings(study: Study, request_fields: dict[str, Any]) -> tuple[str, int, int]:
    setup = yaml.safe_load((study.root / "config" / "setup.yaml").read_text()) or {}
    setup_gbm = (setup.get("modeling") or {}).get("gbm") or {}
    execution_config = {
        **setup_gbm,
        **{
            key: request_fields[key]
            for key in ("device", "max_bin", "num_threads")
            if key in request_fields
        },
    }
    return resolve_gbm_execution_config(execution_config)


def _build_gbm_resolved_request(
    study: Study,
    request: dict[str, Any],
    *,
    base: dict[str, Any],
    config: dict[str, Any],
    effective: dict[str, dict[str, Any]],
    folds: tuple[dict[str, Any], ...],
    device: str,
) -> tuple[dict[str, Any], GBMContext]:
    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    mds = base["mds"]
    label_ref = base["label_ref"]
    checkpoints = gbm_checkpoint_iterations(config)
    expected = base["expected"]
    input_lineage = mds.input_lineage
    computation = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "task": {
            "type": mds.task_type,
            "class_values": list(mds.class_values),
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
        "cv": base["cv_record"],
        "model": {
            "class": "lightgbm.Booster",
            "implementation": "lightgbm",
            "effective_params_by_fold": effective,
            "huber_alpha_scale": config.get("huber_alpha_scale"),
            "max_iterations": int(config["max_iterations"]),
        },
        "checkpoint_schedule": [
            {"kind": "iteration", "value": checkpoint} for checkpoint in checkpoints
        ],
        "expected_prediction_keys": {
            "digest": value_digest(expected, ("symbol", "timestamp", "fold")),
            "n_rows": expected.height,
            "n_folds": expected.get_column("fold").n_unique(),
        },
        "input_data_spec": input_lineage,
        "sampling": {
            "train_sample_frac": base["train_sample_frac"],
            "max_symbols": int(reductions.get("max_symbols", 0)),
        },
        "source_identity": _gbm_source_identity(),
        "runtime_identity": _gbm_runtime_identity(),
    }
    if tier is ExecutionTier.PREVIEW:
        computation["preview_reductions"] = reductions
    runtime_provenance = _gbm_runtime_provenance(study, device)
    spec = ResolvedSpec.create(
        family="gbm",
        label=label_ref.name,
        seed=RANDOM_SEED,
        computation=computation,
        provenance=runtime_provenance,
        config_name=request["config_name"],
        execution_tier=tier.value,
    ).as_dict()
    context = GBMContext(
        folds=folds,
        fold_ids=tuple(int(split["fold"]) for split in base["splits"]),
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=mds.entity_cols[0],
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        expected_keys=expected,
        runtime_provenance=runtime_provenance,
        device=device,
        num_threads=int(next(iter(effective.values()))["num_threads"]),
    )
    return spec, context


def _apply_gbm_preview_reductions(config: dict[str, Any], request: dict[str, Any]) -> None:
    if ExecutionTier(request["execution_tier"]) is not ExecutionTier.PREVIEW:
        return
    reductions = request["preview_reductions"]
    for name in ("max_iterations", "checkpoint_interval"):
        if name in reductions:
            config[name] = int(reductions[name])


def _gbm_input_compatibility_key(request: dict[str, Any]) -> tuple[str, str, int]:
    reductions = request["preview_reductions"]
    return (
        request["label"],
        request["execution_tier"],
        int(reductions.get("max_symbols", 0)),
    )


def _gbm_compatibility_key(study: Study, request: dict[str, Any]) -> str:
    _, request_fields = _load_gbm_request_config(
        study,
        request["label"],
        request["config_name"],
        request["overrides"],
    )
    device, max_bin, num_threads = _gbm_execution_settings(study, request_fields)
    cv = request.get("cv")
    reductions = request["preview_reductions"]
    return canonical_json(
        {
            "label": request["label"],
            "execution_tier": request["execution_tier"],
            "cv": asdict(cv) if cv is not None else None,
            "folds": reductions.get("folds"),
            "max_symbols": reductions.get("max_symbols", 0),
            "train_sample_frac": reductions.get("train_sample_frac", 1.0),
            "device": device,
            "max_bin": max_bin,
            "num_threads": num_threads,
            "preprocessing": GBM_PREPROCESSING_ID,
        }
    )


def resolve_model_request(study: Study, request: dict[str, Any]):
    base = _load_gbm_batch_base(study, request)
    mds = base["mds"]
    folds = prepare_gbm_folds_from_mds(
        mds, base["splits"], train_sample_frac=base["train_sample_frac"]
    )
    if len(folds) != len(base["splits"]) or any(
        not fold["n_train"] or not fold["n_val"] for fold in folds
    ):
        raise ValueError("GBM request did not prepare every declared fold")
    config, request_fields = _load_gbm_request_config(
        study,
        base["label_ref"].name,
        request["config_name"],
        request["overrides"],
    )
    _apply_gbm_preview_reductions(config, request)
    device, max_bin, num_threads = _gbm_execution_settings(study, request_fields)
    effective = _gbm_effective_params_by_fold(
        config,
        folds,
        device=device,
        max_bin=max_bin,
        num_threads=num_threads,
        seed=RANDOM_SEED,
        task_type=mds.task_type,
        class_values=mds.class_values,
    )
    return _build_gbm_resolved_request(
        study,
        request,
        base=base,
        config=config,
        effective=effective,
        folds=tuple(folds),
        device=device,
    )


def reconstruct_locked_request(
    study: Study,
    spec: dict[str, Any],
    *,
    checkpoint_kind: str,
    checkpoint_value: int | None,
):
    """Reconstruct a GBM holdout fit without loading the named preset."""
    from case_studies.research.models import (
        ResolvedModelRequest,
        locked_holdout_split,
        validate_locked_expected_keys,
    )
    from utils.modeling import load_modeling_dataset

    if checkpoint_kind != "iteration" or checkpoint_value is None:
        raise ValueError("GBM holdout requires one locked iteration checkpoint")
    study.require_writable()
    study.activate(ExecutionTier.CANONICAL)
    if spec.get("seed") != RANDOM_SEED:
        raise ValueError("locked GBM seed cannot be reproduced")
    computation = spec["computation"]
    if computation.get("sampling") != {"train_sample_frac": 1.0, "max_symbols": 0}:
        raise ValueError("locked GBM holdout requires an unreduced canonical dataset")
    label_ref = study.labels.get(spec["label"], execution_tier=ExecutionTier.CANONICAL)
    mds = load_modeling_dataset(study.case_study, label_ref.name, max_symbols=0)
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("locked GBM runner requires timestamp and an entity key")
    entity_col = mds.entity_cols[0]
    if entity_col not in {"product", "symbol"}:
        raise ValueError(f"locked GBM runner does not support entity key {entity_col!r}")
    expected_inputs = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": mds.input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "input_data_spec": mds.input_lineage,
        "source_identity": _gbm_source_identity(),
        "runtime_identity": _gbm_runtime_identity(),
        "task": {
            "type": mds.task_type,
            "class_values": list(mds.class_values),
            "continuous_eval_label": label_ref.definition.continuous_eval_label,
        },
    }
    for name, expected_value in expected_inputs.items():
        if computation.get(name) != expected_value:
            raise ValueError(f"locked GBM {name} does not match the available computation")
    schedule = computation.get("checkpoint_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("locked GBM checkpoint schedule is missing")
    declared = tuple(
        int(item["value"])
        for item in schedule
        if item.get("kind") == "iteration" and item.get("value") is not None
    )
    model = computation.get("model")
    reproduced_schedule = (
        gbm_checkpoint_iterations(
            {
                "max_iterations": int(model["max_iterations"]),
                "checkpoint_interval": declared[0],
            }
        )
        if declared and isinstance(model, dict) and model.get("max_iterations") is not None
        else ()
    )
    if (
        len(declared) != len(schedule)
        or checkpoint_value not in declared
        or declared != reproduced_schedule
    ):
        raise ValueError("locked GBM checkpoint is absent from its exact schedule")

    split = locked_holdout_split(spec, mds.dataset, mds.date_col, study.case_study)
    if mds.temporal_by_fold is not None and mds.temporal_keys and mds.temporal_feature_names:
        require_fold_scoped_temporal_compatibility([split], mds.temporal_artifact_splits)
    expected = _gbm_expected_keys_from_dataset(mds, [split])
    validate_locked_expected_keys(spec, expected)
    folds = prepare_gbm_folds(
        mds.dataset.to_pandas(),
        [split],
        mds.feature_names,
        mds.label_col,
        mds.date_col,
        entity_col,
        task_type=mds.task_type,
        class_values=mds.class_values,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
        train_sample_frac=1.0,
        eval_label_col=mds.eval_label_col,
        seed=RANDOM_SEED,
    )
    if len(folds) != 1 or not folds[0]["n_train"] or not folds[0]["n_val"]:
        raise ValueError("locked GBM holdout fold could not be prepared")
    if not isinstance(model, dict):
        raise ValueError("locked GBM model specification is missing")
    fold_id = str(split["fold"])
    effective = model.get("effective_params_by_fold")
    if not isinstance(effective, dict) or set(effective) != {fold_id}:
        raise ValueError("locked GBM model must declare parameters for the holdout fold")
    _validate_lightgbm_params(effective[fold_id])
    locked_runtime = effective[fold_id]
    device = "cpu" if locked_runtime.get("device_type") == "cpu" else "cuda"
    num_threads = int(locked_runtime.get("num_threads", 0))
    reproduced_runtime = lightgbm_runtime_params(
        device,
        num_threads=num_threads,
        seed=int(spec["seed"]),
    )
    if any(locked_runtime.get(name) != value for name, value in reproduced_runtime.items()):
        raise ValueError("locked GBM runtime parameters cannot be reproduced")
    if (
        model.get("class") != "lightgbm.Booster"
        or model.get("implementation") != "lightgbm"
        or set(model)
        != {
            "class",
            "implementation",
            "effective_params_by_fold",
            "huber_alpha_scale",
            "max_iterations",
        }
        or int(model["max_iterations"]) < int(checkpoint_value)
    ):
        raise ValueError("locked GBM model specification is unsupported")
    context = GBMContext(
        folds=(folds[0],),
        fold_ids=(int(split["fold"]),),
        feature_names=tuple(mds.feature_names),
        label_col=mds.label_col,
        eval_label_col=mds.eval_label_col,
        date_col=mds.date_col,
        entity_col=entity_col,
        task_type=mds.task_type,
        class_values=tuple(mds.class_values),
        expected_keys=expected,
        runtime_provenance=_gbm_runtime_provenance(study, device),
        device=device,
        num_threads=num_threads,
        prediction_split="holdout",
        published_checkpoints=(int(checkpoint_value),),
    )
    return ResolvedModelRequest(study, "gbm", spec, context)


def _gbm_manifest_files(model_dir: Path) -> dict[str, str]:
    manifest = model_dir / "manifest.json"
    if not manifest.is_file():
        return {}
    record = json.loads(manifest.read_text())
    return dict(record.get("files") or {})


def _valid_gbm_model_dir(model_dir: Path, context: GBMContext) -> bool:
    files = _gbm_manifest_files(model_dir)
    expected = {f"boosters/fold_{fold_id}.txt" for fold_id in context.fold_ids}
    return set(files) == expected and all(
        (model_dir / name).is_file() and _gbm_sha256(model_dir / name) == digest
        for name, digest in files.items()
    )


def _valid_learning_curves(path: Path, spec: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        curves = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError):
        return False
    required = {"config", "iteration", "ic_mean", "ic_std"}
    expected = {
        int(checkpoint["value"]) for checkpoint in spec["computation"]["checkpoint_schedule"]
    }
    return required <= set(curves.columns) and set(curves["iteration"].to_list()) == expected


def _cached_model_run(study: Study, spec: dict[str, Any], context: GBMContext):
    from case_studies.research.models import ModelRun
    from case_studies.research.results import PredictionResult, Result, TrainingResult
    from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec

    include_preview = spec["execution_tier"] == "preview"
    training_hash = training_hash_from_spec(spec)
    published = context.published_checkpoints or tuple(
        int(item["value"]) for item in spec["computation"]["checkpoint_schedule"]
    )
    try:
        training = Result.open(study, training_hash, include_preview=include_preview)
        predictions = tuple(
            Result.open(
                study,
                prediction_hash_from_parts(
                    training_hash,
                    checkpoint,
                    context.prediction_split,
                    checkpoint_kind="iteration",
                    identity_version=spec["identity_version"],
                ),
                include_preview=include_preview,
            )
            for checkpoint in published
        )
    except KeyError:
        return None
    if not isinstance(training, TrainingResult) or any(
        not isinstance(result, PredictionResult) or not result.complete for result in predictions
    ):
        return None
    prediction_results = tuple(
        result for result in predictions if isinstance(result, PredictionResult)
    )
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    curves_path = training.root / "run_log" / "training" / training.hash / "learning_curves.parquet"
    if not _valid_gbm_model_dir(model_dir, context) or not _valid_learning_curves(
        curves_path, spec
    ):
        return None
    return ModelRun(
        training=training,
        predictions=prediction_results,
        diagnostics={
            "cache_hit": True,
            "reused_folds": sorted(context.fold_ids),
            "fitted_folds": [],
        },
    )


def _gbm_prediction_frame(
    entries: list[dict[str, Any]], checkpoint: int, context: GBMContext
) -> pl.DataFrame:
    frames = []
    for entry in entries:
        if int(entry["n_trees"]) != checkpoint:
            continue
        frame = pl.DataFrame(
            {
                "symbol": entry["entities"],
                "timestamp": entry["dates"],
                "fold": [int(entry["fold"])] * len(entry["y_pred"]),
                "prediction": entry["y_pred"],
                "actual": entry["y_true"],
            }
        )
        if entry.get("y_eval") is not None:
            frame = frame.with_columns(pl.Series("eval_actual", entry["y_eval"]))
        frames.append(
            frame.with_columns(pl.col("timestamp").cast(context.expected_keys.schema["timestamp"]))
        )
    if len(frames) != len(context.folds):
        raise ValueError(f"GBM checkpoint {checkpoint} is missing a declared fold")
    return pl.concat(frames).sort("symbol", "timestamp", "fold")


def _write_gbm_manifest(staging: Path, folds: tuple[dict[str, Any], ...]) -> None:
    expected = [staging / "boosters" / f"fold_{int(fold['fold'])}.txt" for fold in folds]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise ValueError(f"GBM fit did not persist every fold booster: {missing}")
    files = {str(path.relative_to(staging)): _gbm_sha256(path) for path in expected}
    (staging / "manifest.json").write_text(
        json.dumps({"files": files, "schema_version": 1}, indent=2, sort_keys=True) + "\n"
    )


def _predict_from_gbm_models(
    model_dir: Path,
    spec: dict[str, Any],
    context: GBMContext,
) -> dict[str, Any]:
    import lightgbm as lgb

    predictions = []
    is_classification = context.task_type == "classification" and context.class_values
    for fold in context.folds:
        model = lgb.Booster(model_file=str(model_dir / "boosters" / f"fold_{fold['fold']}.txt"))
        for checkpoint in spec["computation"]["checkpoint_schedule"]:
            value = int(checkpoint["value"])
            raw = np.asarray(model.predict(fold["X_val"], num_iteration=value))
            score = (
                _extract_gbm_score(raw, list(context.class_values), len(fold["X_val"]))
                if is_classification
                else raw
            )
            predictions.append(
                {
                    "dates": fold["dates"],
                    "entities": fold["entities"],
                    "y_true": fold["y_val"],
                    "y_eval": fold.get("y_eval"),
                    "y_pred": score,
                    "fold": fold["fold"],
                    "n_trees": value,
                }
            )
    checkpoints = [
        int(checkpoint["value"]) for checkpoint in spec["computation"]["checkpoint_schedule"]
    ]
    return {
        "learning_curves": _learning_curves_from_predictions(
            str(spec.get("config_name") or f"locked-{training_hash_from_spec(spec)}"),
            predictions,
            checkpoints,
        ),
        "predictions": predictions,
    }


def _write_learning_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pl.DataFrame(rows).write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gbm_sampled_training_labels(base: dict[str, Any], split: dict[str, Any]) -> np.ndarray:
    """The training labels Huber's threshold is derived from, as the fold will actually see them.

    This used to re-select the rows itself, in pandas, casting to float32. It agreed with fold
    preparation until preparation stopped casting labels, and then the planner and the resolver
    derived two different thresholds for one declared configuration and gave it two identities.
    Both now come from the one selection, so they cannot drift apart again.
    """
    labels = training_labels_for_split(
        base["mds"], split, train_sample_frac=base["train_sample_frac"], seed=RANDOM_SEED
    )
    if not len(labels):
        raise ValueError(f"GBM request has no training labels for fold {split['fold']}")
    return labels


def _gbm_effective_params_for_splits(
    config: dict[str, Any],
    base: dict[str, Any],
    *,
    device: str,
    max_bin: int,
    num_threads: int,
    task_type: str,
    class_values: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    params = config["params"]
    fold_dependent = (
        params.get("objective") == "huber"
        and "alpha" not in params
        and config.get("huber_alpha_scale") is not None
    )
    folds = [
        {
            "fold": int(split["fold"]),
            **({"y_train": _gbm_sampled_training_labels(base, split)} if fold_dependent else {}),
        }
        for split in base["splits"]
    ]
    return _gbm_effective_params_by_fold(
        config,
        folds,
        device=device,
        max_bin=max_bin,
        num_threads=num_threads,
        seed=RANDOM_SEED,
        task_type=task_type,
        class_values=class_values,
    )


def _prepare_gbm_batch_fold(base: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    folds = prepare_gbm_folds_from_mds(
        base["mds"], [split], train_sample_frac=base["train_sample_frac"]
    )
    if len(folds) != 1 or int(folds[0]["fold"]) != int(split["fold"]):
        raise ValueError(f"GBM request could not prepare fold {split['fold']}")
    if not folds[0]["n_train"] or not folds[0]["n_val"]:
        raise ValueError(f"GBM request prepared fold {split['fold']} empty")
    return folds[0]


def _gbm_fold_settings(candidate: _GBMBatchCandidate, fold_id: int) -> dict[str, Any]:
    assert candidate.spec is not None
    return {
        "effective_params": candidate.spec["computation"]["model"]["effective_params_by_fold"][
            str(fold_id)
        ],
        "checkpoint_schedule": candidate.spec["computation"]["checkpoint_schedule"],
    }


def _gbm_fold_prediction_shard(entries: list[dict[str, Any]], context: GBMContext) -> pl.DataFrame:
    frames = []
    timestamp_dtype = context.expected_keys.schema["timestamp"]
    for entry in entries:
        frame = pl.DataFrame(
            {
                "symbol": entry["entities"],
                "timestamp": entry["dates"],
                "fold": [int(entry["fold"])] * len(entry["y_pred"]),
                "checkpoint": [int(entry["n_trees"])] * len(entry["y_pred"]),
                "prediction": entry["y_pred"],
                "actual": entry["y_true"],
            }
        ).with_columns(pl.col("timestamp").cast(timestamp_dtype))
        if entry.get("y_eval") is not None:
            frame = frame.with_columns(pl.Series("eval_actual", entry["y_eval"]))
        frames.append(frame)
    if not frames:
        raise ValueError("GBM fold fit produced no checkpoint predictions")
    return pl.concat(frames).sort("checkpoint", "symbol", "timestamp", "fold")


def _fit_or_reuse_gbm_fold(
    candidate: _GBMBatchCandidate,
    fold: dict[str, Any],
) -> tuple[pl.DataFrame, bool, float]:
    assert candidate.spec is not None
    assert candidate.context is not None
    assert candidate.training is not None
    assert candidate.ledger is not None
    fold_id = int(fold["fold"])
    training_dir = candidate.training.root / "run_log" / "training" / candidate.training.hash
    artifact = training_dir / "models" / "boosters" / f"fold_{fold_id}.txt"
    shard = training_dir / "prediction_folds" / f"fold_{fold_id}.parquet"
    settings = _gbm_fold_settings(candidate, fold_id)
    if candidate.ledger.reusable_fold(
        training_hash=candidate.training.hash,
        candidate_identity=candidate.training.hash,
        fold_id=fold_id,
        fitted_state=artifact,
        prediction_shard=shard,
        resolved_settings=settings,
    ):
        return pl.read_parquet(shard), True, 0.0

    started = time.perf_counter()
    staging = training_dir / f".fold_{fold_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard_staging = staging / "predictions.parquet"
    try:
        result = train_gbm_config(
            candidate.config,
            [fold],
            feature_names=list(candidate.context.feature_names),
            device=candidate.device,
            num_threads=candidate.num_threads,
            max_bin=candidate.max_bin,
            entity_col=candidate.context.entity_col,
            date_col=candidate.context.date_col,
            task_type=candidate.context.task_type,
            class_values=list(candidate.context.class_values) or None,
            save_dir=staging,
            effective_params_by_fold={
                str(fold_id): settings["effective_params"],
            },
        )
        booster = staging / "boosters" / f"fold_{fold_id}.txt"
        if not booster.is_file():
            raise ValueError(f"GBM fit did not persist fold {fold_id} booster")
        frame = _gbm_fold_prediction_shard(result["predictions"], candidate.context)
        expected_checkpoints = {
            int(item["value"]) for item in candidate.spec["computation"]["checkpoint_schedule"]
        }
        if set(frame.get_column("checkpoint").unique()) != expected_checkpoints:
            raise ValueError(f"GBM fold {fold_id} did not produce every checkpoint")
        expected_rows = candidate.context.expected_keys.filter(pl.col("fold") == fold_id).height
        if frame.height != expected_rows * len(expected_checkpoints):
            raise ValueError(f"GBM fold {fold_id} prediction coverage is incomplete")
        frame.write_parquet(shard_staging)
        os.replace(booster, artifact)
        os.replace(shard_staging, shard)
        candidate.ledger.complete_fold(
            training_hash=candidate.training.hash,
            candidate_identity=candidate.training.hash,
            fold_id=fold_id,
            fitted_state=artifact,
            prediction_shard=shard,
            resolved_settings=settings,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return frame, False, time.perf_counter() - started


def _write_gbm_training_manifest(training: TrainingResult, fold_ids: tuple[int, ...]) -> None:
    model_dir = training.root / "run_log" / "training" / training.hash / "models"
    expected = [model_dir / "boosters" / f"fold_{fold_id}.txt" for fold_id in fold_ids]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise ValueError(f"GBM fit did not persist every fold booster: {missing}")
    files = {str(path.relative_to(model_dir)): _gbm_sha256(path) for path in expected}
    manifest = model_dir / "manifest.json"
    temporary = model_dir / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps({"files": files, "schema_version": 1}, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)


def _gbm_curves_from_shards(
    config_name: str,
    predictions: pl.DataFrame,
    checkpoints: tuple[int, ...],
) -> list[dict[str, Any]]:
    target = "eval_actual" if "eval_actual" in predictions.columns else "actual"
    curves = []
    for checkpoint in checkpoints:
        frame = predictions.filter(pl.col("checkpoint") == checkpoint)
        metric = cross_sectional_ic(
            frame,
            frame,
            pred_col="prediction",
            ret_col=target,
            date_col="timestamp",
            entity_col="symbol",
            min_obs=5,
        )
        curves.append(
            {
                "config": config_name,
                "iteration": checkpoint,
                "ic_mean": float(metric["ic_mean"]),
                "ic_std": float(metric.get("ic_std", 0.0)),
            }
        )
    return curves


def _write_gbm_runtime_fields(path: Path, **fields: float) -> None:
    if not path.exists():
        return
    runtime = json.loads(path.read_text())
    runtime.update(fields)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_gbm_batch_candidate(
    study: Study,
    candidate: _GBMBatchCandidate,
    base: dict[str, Any],
) -> None:
    placeholder_folds = tuple({"fold": int(split["fold"])} for split in base["splits"])
    spec, context = _build_gbm_resolved_request(
        study,
        candidate.request,
        base=base,
        config=candidate.config,
        effective=candidate.effective_params,
        folds=placeholder_folds,
        device=candidate.device,
    )
    candidate.spec = spec
    candidate.context = context
    cached = _cached_model_run(study, spec, context)
    if cached is not None:
        candidate.result = cached
        return
    candidate.started_at_s = time.perf_counter()
    candidate.started_cpu_s = cpu_seconds()
    candidate.training = study.results.register_training(
        spec,
        execution_tier=spec["execution_tier"],
        runtime_provenance=context.runtime_provenance,
    )
    candidate.ledger = ExecutionLedger(study, candidate.training.root)
    candidate.attempt = candidate.ledger.start(candidate.training.hash)


def _fail_gbm_batch_candidate(candidate: _GBMBatchCandidate, error: Exception) -> None:
    if candidate.error is not None:
        return
    candidate.error = error
    if candidate.attempt is not None:
        candidate.attempt.finish(
            "failed",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "reused_folds": candidate.reused_folds,
                "fitted_folds": candidate.fitted_folds,
            },
        )
        candidate.attempt = None


def _reuse_gbm_batch_fold(candidate: _GBMBatchCandidate, fold_id: int) -> bool:
    if candidate.result is not None or candidate.error is not None:
        return True
    assert candidate.training is not None
    assert candidate.ledger is not None
    training_dir = candidate.training.root / "run_log" / "training" / candidate.training.hash
    artifact = training_dir / "models" / "boosters" / f"fold_{fold_id}.txt"
    shard = training_dir / "prediction_folds" / f"fold_{fold_id}.parquet"
    if not candidate.ledger.reusable_fold(
        training_hash=candidate.training.hash,
        candidate_identity=candidate.training.hash,
        fold_id=fold_id,
        fitted_state=artifact,
        prediction_shard=shard,
        resolved_settings=_gbm_fold_settings(candidate, fold_id),
    ):
        return False
    candidate.frames.append(pl.read_parquet(shard))
    candidate.reused_folds.append(fold_id)
    return True


def _run_gbm_batch_fold(candidate: _GBMBatchCandidate, fold: dict[str, Any]) -> None:
    if candidate.result is not None or candidate.error is not None:
        return
    fold_id = int(fold["fold"])
    if fold_id in candidate.reused_folds or fold_id in candidate.fitted_folds:
        return
    try:
        frame, reused, elapsed = _fit_or_reuse_gbm_fold(candidate, fold)
    except Exception as exc:
        _fail_gbm_batch_candidate(candidate, exc)
        return
    candidate.frames.append(frame)
    candidate.fit_elapsed_s += elapsed
    (candidate.reused_folds if reused else candidate.fitted_folds).append(fold_id)


def _record_gbm_runtime(
    study: Study,
    training: TrainingResult,
    *,
    elapsed_s: float,
    cpu_s: float | None = None,
    fit_s: float | None = None,
) -> None:
    """Record what this training run cost, against its registry row.

    Wall time on its own cannot tell a run that saturated the machine from one that spent the
    time waiting, so CPU seconds and the ratio between them go with it, and peak resident memory
    decides whether two notebooks can share the machine. Every GBM row had these NULL, which is
    what made the boosting families unschedulable from recorded cost.
    """
    from case_studies.utils.registry.registration import record_training_runtime

    record_training_runtime(
        study.case_study,
        training.hash,
        case_dir=training.root,
        measured=resource_measurement(elapsed_s=elapsed_s, cpu_s=cpu_s, fit_s=fit_s),
    )


def _finish_gbm_batch_candidate(study: Study, candidate: _GBMBatchCandidate) -> None:
    if candidate.result is not None or candidate.error is not None:
        return
    assert candidate.spec is not None
    assert candidate.context is not None
    assert candidate.training is not None
    assert candidate.attempt is not None
    try:
        if len(candidate.frames) != len(candidate.context.fold_ids):
            raise RuntimeError(
                f"GBM candidate produced {len(candidate.frames)} of "
                f"{len(candidate.context.fold_ids)} fold shards"
            )
        _write_gbm_training_manifest(candidate.training, candidate.context.fold_ids)
        predictions = pl.concat(candidate.frames).sort("checkpoint", "symbol", "timestamp", "fold")
        prediction_results = []
        checkpoints = tuple(
            int(item["value"]) for item in candidate.spec["computation"]["checkpoint_schedule"]
        )
        for checkpoint in checkpoints:
            frame = predictions.filter(pl.col("checkpoint") == checkpoint).drop("checkpoint")
            prediction_results.append(
                study.results.publish_predictions(
                    candidate.training,
                    checkpoint_kind="iteration",
                    checkpoint_value=checkpoint,
                    split="validation",
                    predictions=frame,
                    expected_keys=candidate.context.expected_keys,
                    task_type=candidate.context.task_type,
                    class_values=list(candidate.context.class_values) or None,
                    eval_col="eval_actual" if candidate.context.eval_label_col else None,
                    label=candidate.spec["label"],
                )
            )
        curves_path = (
            candidate.training.root
            / "run_log"
            / "training"
            / candidate.training.hash
            / "learning_curves.parquet"
        )
        curves = _gbm_curves_from_shards(candidate.spec["config_name"], predictions, checkpoints)
        _write_learning_curves(curves_path, curves)
        diagnostics = {
            "cache_hit": False,
            "reused_folds": candidate.reused_folds,
            "fitted_folds": candidate.fitted_folds,
        }
        candidate.attempt.finish("completed", diagnostics)
        candidate.attempt = None
        runtime_path = curves_path.with_name("runtime.json")
        elapsed_s = time.perf_counter() - candidate.started_at_s
        _write_gbm_runtime_fields(runtime_path, elapsed_s=elapsed_s)
        _record_gbm_runtime(
            study,
            candidate.training,
            elapsed_s=elapsed_s,
            cpu_s=cpu_seconds() - candidate.started_cpu_s,
            fit_s=candidate.fit_elapsed_s,
        )
        candidate.result = ModelRun(
            candidate.training,
            tuple(prediction_results),
            diagnostics,
        )
    except Exception as exc:
        _fail_gbm_batch_candidate(candidate, exc)


def _run_gbm_batch_group(
    study: Study,
    indexed_requests: list[tuple[int, dict[str, Any]]],
    compatibility_key: str,
    base: dict[str, Any],
    *,
    report_batch: bool,
    planned_candidates: dict[
        int,
        tuple[dict[str, Any], dict[str, dict[str, Any]], str, int, int],
    ]
    | None = None,
) -> list[_GBMBatchCandidate]:
    mds = base["mds"]
    candidates = []
    for index, request in indexed_requests:
        if planned_candidates is None:
            config, request_fields = _load_gbm_request_config(
                study,
                base["label_ref"].name,
                request["config_name"],
                request["overrides"],
            )
            _apply_gbm_preview_reductions(config, request)
            device, max_bin, num_threads = _gbm_execution_settings(study, request_fields)
            effective = _gbm_effective_params_for_splits(
                config,
                base,
                device=device,
                max_bin=max_bin,
                num_threads=num_threads,
                task_type=mds.task_type,
                class_values=tuple(mds.class_values),
            )
        else:
            config, effective, device, max_bin, num_threads = planned_candidates[index]
        candidate = _GBMBatchCandidate(
            index=index,
            request=request,
            config=config,
            effective_params=effective,
            device=device,
            max_bin=max_bin,
            num_threads=num_threads,
        )
        candidates.append(candidate)
        _resolve_gbm_batch_candidate(study, candidate, base)

    preparation_elapsed_s = 0.0
    preparation_count = 0
    execution_needed = any(
        candidate.result is None and candidate.error is None for candidate in candidates
    )
    if execution_needed:
        for split in base["splits"]:
            fold_id = int(split["fold"])
            pending = [
                candidate
                for candidate in candidates
                if not _reuse_gbm_batch_fold(candidate, fold_id)
            ]
            if not pending:
                continue
            started = time.perf_counter()
            try:
                fold = _prepare_gbm_batch_fold(base, split)
            except Exception as exc:
                for candidate in candidates:
                    _fail_gbm_batch_candidate(candidate, exc)
                break
            preparation_elapsed_s += time.perf_counter() - started
            preparation_count += 1
            for candidate in pending:
                _run_gbm_batch_fold(candidate, fold)
            del fold
            gc.collect()

    for candidate in candidates:
        _finish_gbm_batch_candidate(study, candidate)

    group_digest = hashlib.sha256(compatibility_key.encode()).hexdigest()[:12]
    fit_elapsed_s = sum(candidate.fit_elapsed_s for candidate in candidates)
    measured_s = preparation_elapsed_s + fit_elapsed_s
    for candidate in candidates:
        if candidate.result is None or not report_batch:
            continue
        candidate.result.diagnostics.update(
            {
                "execution_order": "fold_major",
                "compatibility_group": group_digest,
                "compatibility_group_size": len(candidates),
                "base_fold_preparations": preparation_count,
                "base_fold_preparation_s": preparation_elapsed_s,
                "candidate_fit_s": candidate.fit_elapsed_s,
                "preparation_fraction": (preparation_elapsed_s / measured_s if measured_s else 0.0),
                "disk_fold_cache": False,
            }
        )
    return candidates


def plan_model_requests(
    study: Study,
    requests: list[dict[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[Any, ...]]:
    """Resolve a request batch once without fitting or writing result rows."""
    if not requests:
        raise ValueError("GBM batch planner requires at least one request")
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(requests):
        groups.setdefault(_gbm_compatibility_key(study, request), []).append((index, request))

    ordered: list[dict[str, Any] | None] = [None] * len(requests)
    planned_groups = []
    input_cache: dict[tuple[str, str, int], tuple[Any, Any, Any]] = {}
    for key, indexed_requests in groups.items():
        input_key = _gbm_input_compatibility_key(indexed_requests[0][1])
        base = _load_gbm_batch_base(
            study,
            indexed_requests[0][1],
            inputs=input_cache.get(input_key),
        )
        input_cache.setdefault(
            input_key,
            (base["label_ref"], base["mds"], base["dataset_pd"]),
        )
        mds = base["mds"]
        placeholder_folds = tuple({"fold": int(split["fold"])} for split in base["splits"])
        planned_candidates = {}
        for index, request in indexed_requests:
            config, request_fields = _load_gbm_request_config(
                study,
                base["label_ref"].name,
                request["config_name"],
                request["overrides"],
            )
            _apply_gbm_preview_reductions(config, request)
            device, max_bin, num_threads = _gbm_execution_settings(study, request_fields)
            effective = _gbm_effective_params_for_splits(
                config,
                base,
                device=device,
                max_bin=max_bin,
                num_threads=num_threads,
                task_type=mds.task_type,
                class_values=tuple(mds.class_values),
            )
            spec, _ = _build_gbm_resolved_request(
                study,
                request,
                base=base,
                config=config,
                effective=effective,
                folds=placeholder_folds,
                device=device,
            )
            ordered[index] = spec
            planned_candidates[index] = (config, effective, device, max_bin, num_threads)
        planned_groups.append((key, indexed_requests, base, planned_candidates))
    if any(spec is None for spec in ordered):
        raise RuntimeError("GBM batch planner did not resolve every request")
    return tuple(spec for spec in ordered if spec is not None), tuple(planned_groups)


def run_model_plan(study: Study, payload: tuple[Any, ...]) -> tuple[ModelRun, ...]:
    ordered: list[ModelRun | None] = [
        None for _ in range(sum(len(indexed) for _, indexed, _, _ in payload))
    ]
    failures = []
    for key, indexed_requests, base, planned_candidates in payload:
        try:
            candidates = _run_gbm_batch_group(
                study,
                indexed_requests,
                key,
                base,
                report_batch=len(ordered) > 1,
                planned_candidates=planned_candidates,
            )
        except Exception as error:
            failures.append(error)
            continue
        for candidate in candidates:
            if candidate.error is not None:
                failures.append(candidate.error)
            elif candidate.result is not None:
                ordered[candidate.index] = candidate.result
    if failures:
        raise failures[0]
    if any(result is None for result in ordered):
        raise RuntimeError("GBM planned batch did not produce every requested result")
    return tuple(result for result in ordered if result is not None)


def run_model_requests(study: Study, requests: list[dict[str, Any]]) -> tuple[ModelRun, ...]:
    if not requests:
        raise ValueError("GBM batch runner requires at least one request")
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(requests):
        groups.setdefault(_gbm_compatibility_key(study, request), []).append((index, request))

    ordered: list[ModelRun | None] = [None] * len(requests)
    failures = []
    input_cache: dict[tuple[str, str, int], tuple[Any, Any, Any]] = {}
    for key, indexed_requests in groups.items():
        input_key = _gbm_input_compatibility_key(indexed_requests[0][1])
        base = _load_gbm_batch_base(
            study,
            indexed_requests[0][1],
            inputs=input_cache.get(input_key),
        )
        input_cache.setdefault(
            input_key,
            (base["label_ref"], base["mds"], base["dataset_pd"]),
        )
        candidates = _run_gbm_batch_group(
            study,
            indexed_requests,
            key,
            base,
            report_batch=len(requests) > 1,
        )
        for candidate in candidates:
            if candidate.error is not None:
                failures.append(candidate.error)
            elif candidate.result is not None:
                ordered[candidate.index] = candidate.result
    if failures:
        raise failures[0]
    if any(result is None for result in ordered):
        raise RuntimeError("GBM batch did not produce every requested result")
    return tuple(result for result in ordered if result is not None)


def run_resolved_request(study: Study, spec: dict[str, Any], context: GBMContext):
    from case_studies.research.models import ModelRun

    cached = _cached_model_run(study, spec, context)
    if cached is not None:
        return cached
    started = time.perf_counter()
    started_cpu = cpu_seconds()
    computation = spec["computation"]
    training = study.results.register_training(
        spec,
        execution_tier=spec["execution_tier"],
        runtime_provenance=context.runtime_provenance,
    )
    train_dir = training.root / "run_log" / "training" / training.hash
    model_dir = train_dir / "models"
    if model_dir.exists():
        if not _valid_gbm_model_dir(model_dir, context):
            raise ValueError(f"partial fitted-state directory requires inspection: {model_dir}")
        result = _predict_from_gbm_models(model_dir, spec, context)
    else:
        staging = train_dir / f".models.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            result = train_gbm_config(
                {
                    "config_name": str(
                        spec.get("config_name") or f"locked-{training_hash_from_spec(spec)}"
                    ),
                    "max_iterations": computation["model"]["max_iterations"],
                    "checkpoint_interval": computation["checkpoint_schedule"][0]["value"],
                    "params": {},
                },
                list(context.folds),
                feature_names=list(context.feature_names),
                device=context.device,
                num_threads=context.num_threads,
                entity_col=context.entity_col,
                date_col=context.date_col,
                task_type=context.task_type,
                class_values=list(context.class_values) or None,
                save_dir=staging,
                effective_params_by_fold=computation["model"]["effective_params_by_fold"],
            )
            _write_gbm_manifest(staging, context.folds)
            os.replace(staging, model_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    prediction_results = []
    published = context.published_checkpoints or tuple(
        int(item["value"]) for item in computation["checkpoint_schedule"]
    )
    for value in published:
        frame = _gbm_prediction_frame(result["predictions"], value, context)
        prediction_results.append(
            study.results.publish_predictions(
                training,
                checkpoint_kind="iteration",
                checkpoint_value=value,
                split=context.prediction_split,
                predictions=frame,
                expected_keys=context.expected_keys,
                task_type=context.task_type,
                class_values=list(context.class_values) or None,
                eval_col="eval_actual" if context.eval_label_col else None,
                label=spec["label"],
            )
        )
    curves_path = train_dir / "learning_curves.parquet"
    if not curves_path.exists() and result["learning_curves"]:
        _write_learning_curves(curves_path, result["learning_curves"])
    elapsed_s = time.perf_counter() - started
    runtime_path = train_dir / "runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        runtime["elapsed_s"] = elapsed_s
        temporary = runtime_path.with_name(f".{runtime_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, runtime_path)
    # The artifact above is not queryable, and the schedule reads the column. Both are written:
    # a resolved request runs through here rather than through the batch path, so recording it
    # only there left every row this path produced with a NULL elapsed_s.
    _record_gbm_runtime(study, training, elapsed_s=elapsed_s, cpu_s=cpu_seconds() - started_cpu)
    return ModelRun(training=training, predictions=tuple(prediction_results))


def validate_locked_run(
    study: Study,
    spec: dict[str, Any],
    context: GBMContext,
    run: ModelRun,
) -> str:
    """Validate the selected prediction and every persisted GBM booster digest."""
    if run.training.hash != training_hash_from_spec(spec) or len(run.predictions) != 1:
        raise ValueError("locked GBM run has the wrong training or prediction identity")
    prediction = run.predictions[0]
    record = prediction.registry_record()
    selected = context.published_checkpoints
    if selected is None or len(selected) != 1:
        raise ValueError("locked GBM context must select exactly one checkpoint")
    if (
        record["split"],
        record["checkpoint_kind"],
        record["checkpoint_value"],
    ) != (context.prediction_split, "iteration", selected[0]):
        raise ValueError("locked GBM run published the wrong checkpoint")
    published = prediction.load().sort("symbol", "timestamp", "fold")
    model_dir = run.training.root / "run_log" / "training" / run.training.hash / "models"
    if not _valid_gbm_model_dir(model_dir, context):
        raise ValueError("locked GBM fitted-state manifest does not validate")
    reconstructed = _gbm_prediction_frame(
        _predict_from_gbm_models(model_dir, spec, context)["predictions"],
        selected[0],
        context,
    )
    key_columns = ["symbol", "timestamp", "fold"]
    value_columns = ["prediction", "actual"]
    if "eval_actual" in published.columns or "eval_actual" in reconstructed.columns:
        if "eval_actual" not in published.columns or "eval_actual" not in reconstructed.columns:
            raise ValueError("locked GBM fitted state changed the prediction schema")
        value_columns.append("eval_actual")
    if not reconstructed.select(key_columns).equals(
        published.select(key_columns)
    ) or not np.allclose(
        reconstructed.select(value_columns).to_numpy(),
        published.select(value_columns).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    ):
        raise ValueError("locked GBM fitted state does not reproduce published predictions")
    manifest = json.loads((model_dir / "manifest.json").read_text())
    return hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
