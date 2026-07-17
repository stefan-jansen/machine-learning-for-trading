"""Unified experiment registry with content-addressed storage.

Three-level entity model::

    training_run → prediction_set → backtest_run

Each level is identified by a deterministic hash of its spec (the
identity-defining configuration).  The DB is a lean queryable index;
the filesystem ``spec.json`` is the source of truth.

Usage::

    from case_studies.utils.registry import (
        build_training_spec,
        load_preset,
        register_training_run,
        register_prediction_set,
        register_prediction_metrics,
        register_backtest_run,
        register_backtest_metrics,
        load_training_runs,
        load_prediction_sets,
        load_prediction_metrics,
    )

    # Build spec from preset + context
    spec = build_training_spec("gbm", "leaves_15_huber", "fwd_ret_21d",
                               n_folds=8, max_bin=63)

    # Register a training run
    training_hash = register_training_run("etfs", spec=spec)

    # Register predictions at a checkpoint
    prediction_hash = register_prediction_set(
        "etfs", training_hash,
        checkpoint_value=150, checkpoint_kind="tree_limit",
        predictions=predictions_df,
    )

    # Register metrics for those predictions
    register_prediction_metrics("etfs", prediction_hash, {
        "ic_mean": 0.031,
        "ic_std": 0.015,
    })
"""

# --- completeness ---
from .completeness import (
    BacktestRunStatus,
    TrainingRunStatus,
    backtest_run_status,
    skip_backtest_if_complete,
    skip_training_if_complete,
    training_run_status,
)

# --- metrics ---
from .metrics import (
    compute_backtest_fold_metrics,
    compute_classification_metrics_from_predictions,
    compute_fold_metrics_from_predictions,
    compute_prediction_fold_metrics,
    compute_regression_vs_binary_auc,
)

# --- queries ---
from .queries import (
    backfill_stages,
    backtest_dir,
    load_all_prediction_metrics,
    load_all_training_runs,
    load_backtest_fold_metrics,
    load_backtest_metrics,
    load_backtest_runs,
    load_existing_backtest_hashes,
    load_paired_metrics,
    load_prediction_index,
    load_prediction_metrics,
    load_prediction_sets,
    load_training_runs,
    model_source,
    prediction_dir,
    read_backtest_spec,
    read_predictions,
    read_training_spec,
    resolve_best_backtest_runs,
    resolve_best_predictions,
    training_dir,
)

# --- registration ---
from .registration import (
    register_backtest_fold_metrics,
    register_backtest_metrics,
    register_backtest_run,
    register_cohort_metrics,
    register_epoch_checkpoint,
    register_fold_metrics,
    register_paired_metrics,
    register_prediction_metrics,
    register_prediction_set,
    register_training_run,
)

# --- specs ---
from .specs import (
    DEFAULT_SEED,
    HASH_LENGTH,
    backtest_hash_from_parts,
    build_training_spec,
    canonical_json,
    compute_hash,
    load_preset,
    prediction_hash_from_parts,
    training_hash_from_spec,
)

# --- store ---
from .store import (
    REGISTRY_SCHEMA_SQL,
    VALID_STAGES,
    get_training_dir,
)

__all__ = [
    # specs
    "DEFAULT_SEED",
    "HASH_LENGTH",
    "canonical_json",
    "compute_hash",
    "training_hash_from_spec",
    "prediction_hash_from_parts",
    "backtest_hash_from_parts",
    "load_preset",
    "build_training_spec",
    # store
    "REGISTRY_SCHEMA_SQL",
    "VALID_STAGES",
    "get_training_dir",
    # registration
    "register_training_run",
    "register_epoch_checkpoint",
    "register_prediction_set",
    "register_prediction_metrics",
    "register_fold_metrics",
    "register_backtest_run",
    "register_backtest_metrics",
    "register_backtest_fold_metrics",
    "register_paired_metrics",
    "register_cohort_metrics",
    # completeness
    "TrainingRunStatus",
    "BacktestRunStatus",
    "training_run_status",
    "backtest_run_status",
    "skip_training_if_complete",
    "skip_backtest_if_complete",
    # metrics
    "compute_prediction_fold_metrics",
    "compute_backtest_fold_metrics",
    "compute_fold_metrics_from_predictions",
    "compute_classification_metrics_from_predictions",
    "compute_regression_vs_binary_auc",
    # queries
    "load_training_runs",
    "load_prediction_sets",
    "load_prediction_metrics",
    "load_backtest_runs",
    "load_backtest_metrics",
    "load_backtest_fold_metrics",
    "load_all_training_runs",
    "load_all_prediction_metrics",
    "load_existing_backtest_hashes",
    "load_paired_metrics",
    "load_prediction_index",
    "read_training_spec",
    "read_backtest_spec",
    "read_predictions",
    "training_dir",
    "prediction_dir",
    "backtest_dir",
    "model_source",
    "resolve_best_predictions",
    "resolve_best_backtest_runs",
    "backfill_stages",
]
