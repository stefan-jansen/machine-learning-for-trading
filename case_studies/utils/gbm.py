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
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Import torch before ml4t.diagnostic. ml4t.diagnostic transitively loads the
# `cuda` Python package, which dlopens the older system `libcudart.so.12`
# (12.0.146) and wins the symbol resolution; subsequent torch imports then
# fail with `undefined symbol: cudaGetDriverEntryPointByVersion`. Loading
# torch first ensures its bundled CUDA runtime wins. Same pattern as in
# `case_studies/utils/latent_factors/__init__.py` and `model_analysis.py`.
import torch  # noqa: F401
from ml4t.diagnostic.metrics import cross_sectional_ic

from utils.modeling import RANDOM_SEED, seed_everything

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
GBM_DEFAULT_MAX_BIN = 63


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
    if normalized == "cpu":
        if num_threads < 1:
            raise ValueError("num_threads must be at least 1")
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
        return {"device_type": gpu_device}
    raise ValueError(f"Unsupported LightGBM device: {device!r}")


def resolve_gbm_execution_config(config: dict[str, Any]) -> tuple[str, int, int]:
    """Resolve a declared GBM backend without deriving model parameters from hardware."""
    device = str(config.get("device", "cpu")).lower()
    if device == "gpu":
        device = "cuda"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported LightGBM device: {device!r}")

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
    from utils.modeling import _replace_temporal_columns

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
            train_rows = _replace_temporal_columns(
                dataset_pd,
                train_mask,
                temporal_by_fold,
                temporal_keys,
                temporal_feature_names,
                fold_id,
            )
            val_rows = _replace_temporal_columns(
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
    params = dict(config["params"])
    params["metric"] = "None"
    params["verbosity"] = params.get("verbosity", -1)

    # Runtime settings are recorded as provenance by the caller. Numerical
    # model parameters such as max_bin are declared separately in the hashed
    # training spec and never inferred from this runtime backend.
    params.update(lightgbm_runtime_params(device, num_threads=num_threads, seed=seed))
    if max_bin is not None:
        params["max_bin"] = max_bin

    # Classification: ensure num_class for multiclass
    if is_classification and class_values and len(class_values) > 2:
        params["num_class"] = len(class_values)

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
                preds = _extract_gbm_score(raw_preds, class_values, len(fd["X_val"]))
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
