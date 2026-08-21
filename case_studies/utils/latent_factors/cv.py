"""Walk-forward orchestration for latent factor models."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml
from ml4t.diagnostic.metrics import cross_sectional_ic
from threadpoolctl import threadpool_limits

from case_studies.utils.backtest_loaders import get_rebalance_step, thin_to_rebalance_dates
from case_studies.utils.latent_factors.cae import run_cae_fold
from case_studies.utils.latent_factors.ipca import run_ipca_fold
from case_studies.utils.latent_factors.library_bridge import configure_latent_torch_runtime
from case_studies.utils.latent_factors.panel import (
    prepare_panel_data,
    prepare_ragged_panel_data,
    rank_normalize_cross_section,
)
from case_studies.utils.latent_factors.pca import run_pca_fold
from case_studies.utils.latent_factors.sae import run_sae_fold
from case_studies.utils.latent_factors.sdf import run_sdf_fold
from utils.modeling import RANDOM_SEED, seed_everything

_MODEL_RUNNERS = {
    "pca": run_pca_fold,
    "ipca": run_ipca_fold,
    "cae": run_cae_fold,
    "sdf": run_sdf_fold,
    "sae": run_sae_fold,
}

TEMPORAL_FEATURE_ASSEMBLY = "fold_scoped_v1"

# Runner arguments the caller supplies directly, which a preset must not overwrite.
# Every latent-factor preset declares n_factors, so before this set covered it the
# N_FACTORS notebook parameter was a no-op in every configured notebook.
_CALLER_SUPPLIED_RUNNER_ARGS = frozenset({"device", "n_factors"})
_EPOCH_MODELS = frozenset({"cae", "sae"})


def merge_preset_into_runner_kwargs(
    kwargs: dict[str, Any],
    *,
    preset: dict[str, Any],
    allowed: set[str],
    model_name: str,
) -> dict[str, Any]:
    """Fold a model's preset into runner kwargs the caller already filled in.

    Mutates and returns ``kwargs``. A preset value is applied only when the runner
    accepts that argument and the caller did not supply it.
    """
    explicit = set(_CALLER_SUPPLIED_RUNNER_ARGS)
    if model_name in _EPOCH_MODELS:
        explicit.add("n_epochs")
    kwargs.update(
        {key: value for key, value in preset.items() if key in allowed and key not in explicit}
    )
    return kwargs


def _numpy_serializer(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_fold_extras(path: Path, fold_extras: list[dict]) -> None:
    path.write_text(json.dumps(fold_extras, default=_numpy_serializer, indent=1))


def load_fold_extras(case_study_id: str, training_hash: str) -> list[dict] | None:
    from utils.paths import get_case_study_dir

    extras_path = (
        get_case_study_dir(case_study_id)
        / "run_log"
        / "training"
        / training_hash
        / "fold_extras.json"
    )
    if not extras_path.exists():
        return None
    return json.loads(extras_path.read_text())


def _expected_latent_checkpoints(
    model_name: str,
    *,
    n_epochs: int,
    model_kwargs: dict[str, Any],
    include_internal_aliases: bool = False,
) -> tuple[int, ...]:
    """Resolve physical checkpoints and, when requested, fitted-state aliases."""
    from case_studies.utils.latent_factors.common import resolve_checkpoint_epochs

    if model_name in {"pca", "ipca"}:
        return (0,)
    if model_name in {"cae", "sae"}:
        physical = resolve_checkpoint_epochs(
            n_epochs,
            checkpoint_interval=model_kwargs.get("checkpoint_interval", 5),
            checkpoint_epochs=model_kwargs.get("checkpoint_epochs"),
        )
        if model_name == "cae" and include_internal_aliases:
            return tuple(sorted({0, *physical}))
        return tuple(physical)
    if model_name == "sdf":
        n_epochs_unc = int(model_kwargs.get("n_epochs_unc", 256))
        n_epochs_cond = int(model_kwargs.get("n_epochs_cond", 1024))
        physical = resolve_checkpoint_epochs(
            max(n_epochs_unc, n_epochs_cond),
            checkpoint_interval=model_kwargs.get("checkpoint_interval"),
            checkpoint_epochs=model_kwargs.get("checkpoint_epochs"),
        )
        labels: set[int] = {-3, -2, -1, 0} if include_internal_aliases else set()
        labels.update(epoch for epoch in physical if epoch <= n_epochs_unc)
        labels.update(n_epochs_unc + epoch for epoch in physical if epoch <= n_epochs_cond)
        return tuple(sorted(labels))
    raise ValueError(f"Unsupported latent-factor model: {model_name!r}")


def _build_expected_latent_training_spec(
    *,
    model_name: str,
    label_col: str,
    n_factors: int,
    n_epochs: int,
    model_kwargs: dict[str, Any],
    feature_names: list[str],
    splits: list[dict[str, Any]],
    task_type: str,
    class_values: list | None,
    eval_label_col: str | None,
    input_digest: str,
    macro_digest: str | None,
    runtime_spec: dict[str, Any],
    temporal_feature_assembly: str | None = None,
    temporal_feature_digest: str | None = None,
    macro_context_spec: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Build the exact identity expected from the registration path."""
    from case_studies.utils.registry import build_training_spec

    try:
        spec = build_training_spec(
            "latent_factors",
            model_name,
            label_col,
            n_folds=len(splits),
            n_epochs=n_epochs,
        )
    except FileNotFoundError:
        spec = {
            "config_name": model_name,
            "family": "latent_factors",
            "feature_sets": ["financial", "model_based"],
            "label": label_col,
            "library": "pytorch",
            "n_epochs": n_epochs,
            "params": {"n_factors": n_factors},
            "seed": 42,
        }

    checkpoints = _expected_latent_checkpoints(
        model_name,
        n_epochs=n_epochs,
        model_kwargs=model_kwargs,
    )
    fold_extras: list[dict[str, Any]] = []
    if model_name in {"cae", "sae", "sdf"}:
        expected_extra: dict[str, Any] = {"checkpoint_epochs": list(checkpoints)}
        if model_name == "sdf":
            expected_extra["output_mode"] = model_kwargs.get("output_mode", "beta_network")
            expected_extra["expected_return_mapper"] = model_kwargs.get(
                "expected_return_mapper", "linear"
            )
        fold_extras = [expected_extra]
    expected = _apply_latent_factor_runtime_spec(
        spec=spec,
        n_factors=n_factors,
        n_epochs=n_epochs,
        model_kwargs=model_kwargs,
        fold_extras=fold_extras,
        feature_names=feature_names,
        splits=splits,
        task_type=task_type,
        class_values=class_values,
        eval_label_col=eval_label_col,
        input_digest=input_digest,
        macro_digest=macro_digest,
        runtime_spec=runtime_spec,
        temporal_feature_assembly=temporal_feature_assembly,
        temporal_feature_digest=temporal_feature_digest,
        macro_context_spec=macro_context_spec,
        input_data_spec=input_data_spec,
    )
    return expected, checkpoints


def _expected_latent_prediction_keys(
    dataset: pl.DataFrame,
    *,
    splits: list[dict[str, Any]],
    label_col: str,
    eval_label_col: str | None,
    date_col: str,
    entity_col: str,
) -> pl.DataFrame:
    """Return the exact validation key surface implied by the dated input."""
    frames: list[pl.DataFrame] = []
    for split in splits:
        frame = dataset.filter(
            (pl.col(date_col) >= _to_naive_timestamp(split["val_start"]))
            & (pl.col(date_col) <= _to_naive_timestamp(split["val_end"]))
            & pl.col(label_col).is_finite()
        )
        if eval_label_col:
            frame = frame.filter(pl.col(eval_label_col).is_finite())
        frames.append(
            frame.select(date_col, entity_col).with_columns(
                pl.lit(int(split["fold"]), dtype=pl.Int64).alias("fold_id")
            )
        )
    if not frames:
        return pl.DataFrame()
    return _normalize_prediction_keys(pl.concat(frames).unique(), date_col, entity_col)


def _normalize_prediction_keys(
    frame: pl.DataFrame,
    date_col: str,
    entity_col: str,
) -> pl.DataFrame:
    key_cols = [date_col, entity_col, "fold_id"]
    return (
        frame.select(key_cols)
        .with_columns(
            pl.col(date_col).cast(pl.Datetime("us")),
            pl.col(entity_col).cast(pl.String),
            pl.col("fold_id").cast(pl.Int64),
        )
        .sort(key_cols)
    )


def _load_registered_latent_factor(
    case_study_id: str,
    *,
    model_name: str,
    training_spec: dict[str, Any],
    prediction_split: str,
    expected_checkpoints: tuple[int, ...],
    expected_keys: pl.DataFrame,
    date_col: str,
    entity_col: str,
    eval_label_col: str | None,
) -> tuple[str, pl.DataFrame, pl.DataFrame] | None:
    """Replay one exact, complete registry cohort and validate its metrics."""
    from case_studies.utils import registry

    training_hash = registry.training_hash_from_spec(training_spec)
    runs = registry.load_training_runs(
        case_study_id,
        family="latent_factors",
        label=training_spec["label"],
    )
    if runs.height == 0 or "training_hash" not in runs.columns:
        return None
    matching = runs.filter(pl.col("training_hash") == training_hash)
    if matching.height == 0:
        return None
    if matching.height != 1 or json.loads(matching["spec_json"][0]) != training_spec:
        raise ValueError(f"Cached {model_name} training identity is inconsistent")

    prediction_sets = registry.load_prediction_sets(
        case_study_id,
        training_hash=training_hash,
        split=prediction_split,
    )
    required_metadata = {"prediction_hash", "checkpoint_value", "checkpoint_kind"}
    missing_metadata = required_metadata - set(prediction_sets.columns)
    if missing_metadata:
        raise ValueError(
            f"Cached {model_name} checkpoint metadata is missing {sorted(missing_metadata)}"
        )
    if prediction_sets.height != len(expected_checkpoints):
        raise ValueError(
            f"Cached {model_name} checkpoint count {prediction_sets.height} does not match "
            f"expected {len(expected_checkpoints)}"
        )
    if prediction_sets["checkpoint_value"].null_count():
        raise ValueError(f"Cached {model_name} has a null checkpoint value")
    if prediction_sets.filter(pl.col("checkpoint_kind") != "epoch").height:
        raise ValueError(f"Cached {model_name} contains a non-epoch checkpoint")
    observed = tuple(sorted(int(value) for value in prediction_sets["checkpoint_value"]))
    if len(observed) != len(set(observed)) or observed != expected_checkpoints:
        raise ValueError(
            f"Cached {model_name} checkpoints {observed} do not match "
            f"expected {expected_checkpoints}"
        )

    key_cols = [date_col, entity_col, "fold_id"]
    metrics_rows: list[dict[str, Any]] = []
    frames: list[pl.DataFrame] = []
    for row in prediction_sets.sort("checkpoint_value").iter_rows(named=True):
        epoch = int(row["checkpoint_value"])
        path = (
            registry.prediction_dir(case_study_id, row["prediction_hash"]) / "predictions.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        predictions = pl.read_parquet(path)
        required_cols = {date_col, entity_col, "fold_id", "y_true", "y_score", "epoch"}
        if eval_label_col:
            required_cols.add("eval_actual")
        missing_cols = required_cols - set(predictions.columns)
        if missing_cols:
            raise ValueError(
                f"Cached {model_name} checkpoint {epoch} schema is missing {sorted(missing_cols)}"
            )
        if predictions.select(pl.col(list(required_cols)).null_count()).row(0) != (0,) * len(
            required_cols
        ):
            raise ValueError(f"Cached {model_name} checkpoint {epoch} contains nulls")
        if predictions["epoch"].n_unique() != 1 or int(predictions["epoch"][0]) != epoch:
            raise ValueError(f"Cached {model_name} checkpoint {epoch} has contradictory epoch data")
        actual_keys = _normalize_prediction_keys(predictions, date_col, entity_col)
        if actual_keys.n_unique() != predictions.height:
            raise ValueError(f"Cached {model_name} checkpoint {epoch} has duplicate keys")
        if not actual_keys.equals(expected_keys):
            raise ValueError(
                f"Cached {model_name} checkpoint {epoch} key or fold coverage is incomplete"
            )

        actual_col = "eval_actual" if eval_label_col else "y_true"
        metric = cross_sectional_ic(
            predictions,
            predictions,
            pred_col="y_score",
            ret_col=actual_col,
            date_col=date_col,
            entity_col=entity_col,
            method="spearman",
            min_obs=5,
        )
        registry_metrics = registry.load_prediction_metrics(
            case_study_id,
            prediction_hash=row["prediction_hash"],
        )
        required_daily_metrics = {"ic_mean_daily", "ic_std_daily"}
        if registry_metrics.height != 1 or not required_daily_metrics <= set(
            registry_metrics.columns
        ):
            raise ValueError(
                f"Cached {model_name} checkpoint {epoch} has invalid daily registry metrics"
            )
        comparisons = {
            "daily mean": (registry_metrics["ic_mean_daily"][0], float(metric["ic_mean"])),
            "daily std": (registry_metrics["ic_std_daily"][0], float(metric["ic_std"])),
        }
        mismatches = {
            name: values
            for name, values in comparisons.items()
            if values[0] is None
            or not np.isclose(float(values[0]), values[1], atol=1e-12, rtol=0.0)
        }
        if mismatches:
            raise ValueError(
                f"Cached {model_name} checkpoint {epoch} daily metric mismatch: {mismatches}"
            )

        for fold_id in expected_keys["fold_id"].unique().sort():
            fold_metric = cross_sectional_ic(
                predictions.filter(pl.col("fold_id") == fold_id),
                predictions.filter(pl.col("fold_id") == fold_id),
                pred_col="y_score",
                ret_col=actual_col,
                date_col=date_col,
                entity_col=entity_col,
                method="spearman",
                min_obs=5,
            )
            metrics_rows.append(
                {
                    "epoch": epoch,
                    "fold_id": int(fold_id),
                    "ic_mean": float(fold_metric["ic_mean"]),
                }
            )
        frames.append(predictions)

    return training_hash, pl.DataFrame(metrics_rows), pl.concat(frames)


def run_latent_factor_cv(
    panel_data: dict | None,
    splits: list[dict[str, Any]],
    *,
    models: list[str],
    n_factors: int = 5,
    n_epochs: int = 50,
    model_kwargs: dict[str, dict[str, Any]] | None = None,
    save_dir: Path | None = None,
    use_cache: bool = True,
    force_retrain: bool = False,
    random_state: int | None = None,
    dataset: pl.DataFrame | None = None,
    feature_names: list[str] | None = None,
    label_col: str | None = None,
    date_col: str = "timestamp",
    entity_col: str = "symbol",
    case_study_id: str | None = None,
    notebook: str = "latent_factors",
    eval_label_col: str | None = None,
    task_type: str = "regression",
    class_values: list | None = None,
    prediction_split: str = "validation",
    macro_panel: pl.DataFrame | None = None,
    macro_context_spec: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
    persistent_entities: bool = True,
    checkpoint_selection_policy: str | None = None,
    reporting_epoch: int | None = None,
    score_dates: str = "auto",
    score_cadence: str | None = None,
    score_rebalance_step: int | None = None,
    device: str = "cpu",
    num_threads: int = 8,
    deterministic_algorithms: bool = True,
    temporal_by_fold: pd.DataFrame | None = None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
    fold_workers: int = 1,
    checkpoint_surface: str = "physical",
) -> dict[str, Any]:
    """Run walk-forward latent factor CV from the raw dated dataset."""
    del panel_data
    if dataset is None or feature_names is None or label_col is None:
        raise ValueError(
            "run_latent_factor_cv requires dataset, feature_names, and label_col. "
            "Pre-built latent-factor panels are no longer supported."
        )
    if isinstance(fold_workers, bool) or not isinstance(fold_workers, int) or fold_workers < 1:
        raise ValueError("fold_workers must be a positive integer")
    if fold_workers > 1 and models != ["ipca"]:
        raise ValueError("parallel fold execution is currently supported only for IPCA-only runs")
    if checkpoint_surface not in {"fitted_state", "physical"}:
        raise ValueError("checkpoint_surface must be 'physical' or 'fitted_state'")

    model_kwargs = model_kwargs or {}
    runtime_spec = configure_latent_torch_runtime(
        device,
        seed=random_state if random_state is not None else RANDOM_SEED,
        num_threads=num_threads,
        deterministic_algorithms=deterministic_algorithms,
    )
    input_digest = _latent_input_digest(
        dataset,
        feature_names=feature_names,
        label_col=label_col,
        eval_label_col=eval_label_col,
        date_col=date_col,
        entity_col=entity_col,
        splits=splits,
    )
    macro_digest = (
        _frame_digest(_filter_dataset_for_splits(macro_panel, splits=splits, date_col=date_col))
        if macro_panel is not None
        else None
    )
    expected_prediction_keys = _expected_latent_prediction_keys(
        dataset,
        splits=splits,
        label_col=label_col,
        eval_label_col=eval_label_col,
        date_col=date_col,
        entity_col=entity_col,
    )
    has_fold_temporal = bool(
        temporal_by_fold is not None and temporal_keys and temporal_feature_names
    )
    temporal_feature_assembly = TEMPORAL_FEATURE_ASSEMBLY if has_fold_temporal else None
    temporal_feature_digest = (
        _frame_digest(
            _temporal_digest_frame(temporal_by_fold, temporal_keys, temporal_feature_names)
        )
        if has_fold_temporal
        else None
    )
    metric_policy = _resolve_metric_policy(
        case_study_id=case_study_id,
        label_col=label_col,
        checkpoint_selection_policy=checkpoint_selection_policy,
        reporting_epoch=reporting_epoch,
        score_dates=score_dates,
        score_cadence=score_cadence,
        score_rebalance_step=score_rebalance_step,
    )
    model_results: list[dict[str, Any]] = []
    all_predictions: dict[str, pl.DataFrame] = {}
    fold_metrics: dict[str, pl.DataFrame] = {}
    all_extras: dict[str, list[dict]] = {}

    if save_dir is not None:
        log_path = save_dir / "latent_factors.log"
    elif case_study_id:
        from utils.paths import get_case_study_dir

        log_path = get_case_study_dir(case_study_id) / "run_log" / "latent_factors.log"
    else:
        log_path = Path("/tmp/latent_factors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")  # noqa: SIM115

    def log(message: str) -> None:
        line = f"[{datetime.now(UTC):%H:%M:%S}] {message}"
        log_file.write(line + "\n")
        log_file.flush()
        print(message)

    from utils.paths import display_path

    log(f"Latent factor CV: {len(models)} models × {len(splits)} folds")
    log(f"Log file: {display_path(log_path)}")
    log(
        "Scoring: "
        f"dates={metric_policy['score_dates']} "
        f"cadence={metric_policy['score_cadence'] or '-'} "
        f"step={metric_policy['score_rebalance_step']} "
        f"checkpoint_selection={metric_policy['checkpoint_selection_policy']} "
        f"reporting_epoch={metric_policy['reporting_epoch'] if metric_policy['reporting_epoch'] is not None else 'last'}"
    )

    seed_everything(random_state if random_state is not None else RANDOM_SEED)

    active_models: list[str] = []
    state: dict[str, dict[str, Any]] = {}
    started_at: dict[str, str] = {}
    started_clock: dict[str, float] = {}
    model_dirs: dict[str, Path | None] = {}

    for model_name in models:
        if model_name not in _MODEL_RUNNERS:
            log(f"  WARNING: unknown model {model_name!r}, skipping")
            continue
        if model_name == "pca" and not persistent_entities:
            raise ValueError("PCA requires persistent entity IDs for the current dataset")

        model_dir = save_dir / model_name if save_dir is not None else None
        model_dirs[model_name] = model_dir
        model_temporal_feature_assembly = temporal_feature_assembly if model_name != "pca" else None
        model_temporal_feature_digest = temporal_feature_digest if model_name != "pca" else None
        model_macro_context = macro_context_spec if model_name == "sdf" else None
        if use_cache and not force_retrain and case_study_id:
            training_spec, expected_checkpoints = _build_expected_latent_training_spec(
                model_name=model_name,
                label_col=label_col,
                n_factors=n_factors,
                n_epochs=n_epochs,
                model_kwargs=model_kwargs.get(model_name, {}),
                feature_names=feature_names,
                splits=splits,
                task_type=task_type,
                class_values=class_values,
                eval_label_col=eval_label_col,
                input_digest=input_digest,
                macro_digest=macro_digest,
                runtime_spec=runtime_spec,
                temporal_feature_assembly=model_temporal_feature_assembly,
                temporal_feature_digest=model_temporal_feature_digest,
                macro_context_spec=model_macro_context,
                input_data_spec=input_data_spec,
            )
            registered = _load_registered_latent_factor(
                case_study_id,
                model_name=model_name,
                training_spec=training_spec,
                prediction_split=prediction_split,
                expected_checkpoints=expected_checkpoints,
                expected_keys=expected_prediction_keys,
                date_col=date_col,
                entity_col=entity_col,
                eval_label_col=eval_label_col,
            )
            if registered is not None:
                training_hash, metrics_df, preds_df = registered
                best_epoch, mean_ic = _select_reporting_epoch(
                    metrics_df,
                    checkpoint_selection_policy=metric_policy["checkpoint_selection_policy"],
                    reporting_epoch=metric_policy["reporting_epoch"],
                )
                model_results.append(
                    {
                        "model_name": model_name,
                        "mean_ic": round(mean_ic, 4),
                        "best_epoch": best_epoch,
                        "n_folds": int(metrics_df["fold_id"].n_unique()),
                        "elapsed_s": 0.0,
                        "started_at": None,
                    }
                )
                all_predictions[model_name] = preds_df
                fold_metrics[model_name] = metrics_df
                all_extras[model_name] = load_fold_extras(case_study_id, training_hash) or []
                log(f"  {model_name}: loaded exact registry cohort (IC={mean_ic:+.4f})")
                continue
        if (
            use_cache
            and not force_retrain
            and model_dir is not None
            and model_macro_context is None
            and (model_dir / "predictions.parquet").exists()
            and (model_dir / "fold_metrics.parquet").exists()
        ):
            preds_df = pl.read_parquet(model_dir / "predictions.parquet")
            metrics_df = pl.read_parquet(model_dir / "fold_metrics.parquet")
            expected_cache_checkpoints = set(
                _expected_latent_checkpoints(
                    model_name,
                    n_epochs=n_epochs,
                    model_kwargs=model_kwargs.get(model_name, {}),
                    include_internal_aliases=checkpoint_surface == "fitted_state",
                )
            )
            expected_cache_surface = {
                (int(split["fold"]), checkpoint)
                for split in splits
                for checkpoint in expected_cache_checkpoints
            }
            cached_prediction_surface = set(
                preds_df.select("fold_id", "epoch").unique().iter_rows()
            )
            cached_metric_surface = set(metrics_df.select("fold_id", "epoch").unique().iter_rows())
            if (
                cached_prediction_surface != expected_cache_surface
                or cached_metric_surface != expected_cache_surface
            ):
                log(f"  {model_name}: cache checkpoint surface mismatch, retraining")
            else:
                best_epoch, mean_ic = _select_reporting_epoch(
                    metrics_df,
                    checkpoint_selection_policy=metric_policy["checkpoint_selection_policy"],
                    reporting_epoch=metric_policy["reporting_epoch"],
                )
                model_results.append(
                    {
                        "model_name": model_name,
                        "mean_ic": round(mean_ic, 4),
                        "best_epoch": best_epoch,
                        "n_folds": int(metrics_df["fold_id"].n_unique())
                        if metrics_df.height > 0
                        else 0,
                        "elapsed_s": 0.0,
                        "started_at": None,
                    }
                )
                all_predictions[model_name] = preds_df
                fold_metrics[model_name] = metrics_df
                all_extras[model_name] = []
                log(f"  {model_name}: loaded cache (best IC={mean_ic:+.4f})")
                continue

        active_models.append(model_name)
        started_at[model_name] = datetime.now(UTC).isoformat()
        started_clock[model_name] = time.perf_counter()
        state[model_name] = {
            "fold_ics": [],
            "pred_frames": [],
            "fold_extras": [],
        }
        log(f"  {model_name} (K={n_factors}):")

    need_pca_inputs = "pca" in active_models
    need_ragged_inputs = any(model_name != "pca" for model_name in active_models)

    def runner_kwargs(
        model_name: str,
        model_input: dict[str, Any],
        fold_id: int,
    ) -> dict[str, Any]:
        runner = _MODEL_RUNNERS[model_name]
        kwargs: dict[str, Any] = {"n_factors": n_factors}
        if model_name in {"cae", "sae"}:
            kwargs["n_epochs"] = n_epochs
        if model_name in {"cae", "sae", "sdf"}:
            kwargs["log_fn"] = log
            kwargs["device"] = runtime_spec["device"]
        if model_name in {"cae", "sae"}:
            kwargs["task_type"] = task_type
            if (
                task_type == "classification"
                and model_input.get("factor_returns_train") is not None
            ):
                kwargs["factor_returns_train"] = model_input["factor_returns_train"]
        if model_name == "sdf" and model_input.get("macro_train") is not None:
            kwargs["macro_train"] = model_input["macro_train"]
            kwargs["macro_val"] = model_input["macro_val"]
        model_dir = model_dirs[model_name]
        if model_dir is not None and "artifact_dir" in inspect.signature(runner).parameters:
            kwargs["artifact_dir"] = model_dir / "artifacts" / f"fold_{fold_id}"
        if model_name in model_kwargs:
            merge_preset_into_runner_kwargs(
                kwargs,
                preset=model_kwargs[model_name],
                allowed=set(inspect.signature(runner).parameters),
                model_name=model_name,
            )
        return kwargs

    def fit_fold(
        model_name: str,
        model_input: dict[str, Any],
        fold_id: int,
    ) -> tuple[dict[int, np.ndarray], dict[str, Any], float]:
        fold_started = time.perf_counter()
        result = _MODEL_RUNNERS[model_name](
            model_input["chars_train"],
            model_input["returns_train"],
            model_input["chars_val"],
            model_input["returns_val"],
            **runner_kwargs(model_name, model_input, fold_id),
        )
        if isinstance(result[0], dict):
            checkpoint_preds, extra = result
        else:
            predictions_arr, extra = result
            checkpoint_preds = {0: predictions_arr}
        return checkpoint_preds, extra, time.perf_counter() - fold_started

    def record_fold(
        *,
        split: dict[str, Any],
        model_name: str,
        model_input: dict[str, Any],
        checkpoint_preds: dict[int, np.ndarray],
        extra: dict[str, Any],
        fold_elapsed: float,
    ) -> None:
        state[model_name]["fold_extras"].append({"fold_id": split["fold"], **extra})
        if checkpoint_surface == "physical":
            physical = set(
                _expected_latent_checkpoints(
                    model_name,
                    n_epochs=n_epochs,
                    model_kwargs=model_kwargs.get(model_name, {}),
                )
            )
            checkpoint_preds = {
                epoch: predictions
                for epoch, predictions in checkpoint_preds.items()
                if epoch in physical
            }
            if not checkpoint_preds:
                raise ValueError(f"{model_name} produced no physical checkpoints")
        checkpoint_ics: dict[int, float] = {}
        for epoch, predictions in checkpoint_preds.items():
            frame = _build_prediction_frame(
                predictions=predictions,
                returns_val=model_input["returns_val"],
                eval_returns_val=model_input.get("eval_returns_val"),
                val_dates=model_input["val_dates"],
                val_entities=model_input["val_entities"],
                fold_id=split["fold"],
                model_name=model_name,
                epoch=epoch,
            )
            scored_frame = _score_prediction_frame(
                frame,
                score_dates=metric_policy["score_dates"],
                score_cadence=metric_policy["score_cadence"],
                score_rebalance_step=metric_policy["score_rebalance_step"],
            )
            ic, n_scored_dates = _compute_frame_ic(scored_frame)
            checkpoint_ics[epoch] = ic
            state[model_name]["fold_ics"].append(
                {
                    "fold_id": split["fold"],
                    "epoch": epoch,
                    "ic_mean": round(ic, 4),
                    "n_train": model_input["n_train_periods"],
                    "n_test": model_input["n_val_periods"],
                    "n_scored_dates": n_scored_dates,
                }
            )
            if frame is not None:
                state[model_name]["pred_frames"].append(frame)
        best_epoch, reported_ic = _select_epoch_from_values(
            checkpoint_ics,
            checkpoint_selection_policy=metric_policy["checkpoint_selection_policy"],
            reporting_epoch=metric_policy["reporting_epoch"],
        )
        log(
            f"      fold {split['fold']}: reported_epoch={best_epoch}, "
            f"IC={reported_ic:+.4f}, {fold_elapsed:.1f}s"
        )
        _write_incremental_fold(
            model_dir=model_dirs[model_name],
            fold_id=split["fold"],
            predictions=checkpoint_preds,
            model_input=model_input,
            model_name=model_name,
        )

    if fold_workers > 1 and active_models:
        prepared_folds: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for split in splits:
            fold_inputs = _prepare_fold_inputs(
                dataset=dataset,
                split=split,
                feature_names=feature_names,
                label_col=label_col,
                date_col=date_col,
                entity_col=entity_col,
                eval_label_col=eval_label_col,
                macro_panel=macro_panel,
                need_pca_inputs=False,
                need_ragged_inputs=True,
                temporal_by_fold=temporal_by_fold,
                temporal_keys=temporal_keys,
                temporal_feature_names=temporal_feature_names,
            )
            if fold_inputs is None:
                log(f"    Fold {split['fold']}: skipped (insufficient train/validation dates)")
                continue
            model_input = fold_inputs["ragged"]
            log(
                f"    Fold {split['fold']}: ragged "
                f"train={model_input['n_train_periods']}, "
                f"val={model_input['n_val_periods']}, "
                f"max_N={model_input['chars_train'].shape[1]}"
            )
            prepared_folds.append((split, model_input))
        if not prepared_folds:
            raise RuntimeError("IPCA parallel execution produced no eligible folds")
        prepared_folds.sort(key=lambda item: int(item[0]["fold"]))
        worker_count = min(fold_workers, len(prepared_folds))
        log(f"    IPCA parallel execution: {worker_count} workers, BLAS threads=1")
        completed: dict[int, tuple[dict[int, np.ndarray], dict[str, Any], float]] = {}
        with (
            threadpool_limits(limits=1, user_api="blas"),
            ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ipca-fold") as pool,
        ):
            futures = {
                pool.submit(fit_fold, "ipca", model_input, int(split["fold"])): int(split["fold"])
                for split, model_input in prepared_folds
            }
            for future in as_completed(futures):
                fold_id = futures[future]
                checkpoint_preds, extra, fold_elapsed = future.result()
                completed[fold_id] = (checkpoint_preds, extra, fold_elapsed)
                log(
                    f"      fold {fold_id}: fit complete, "
                    f"iterations={extra.get('iterations', '?')}, "
                    f"converged={extra.get('converged', False)}, {fold_elapsed:.1f}s"
                )
        ordered_extras = [
            {"fold_id": int(split["fold"]), **completed[int(split["fold"])][1]}
            for split, _ in prepared_folds
        ]
        _require_ipca_convergence("ipca", ordered_extras)
        for split, model_input in prepared_folds:
            checkpoint_preds, extra, fold_elapsed = completed[int(split["fold"])]
            record_fold(
                split=split,
                model_name="ipca",
                model_input=model_input,
                checkpoint_preds=checkpoint_preds,
                extra=extra,
                fold_elapsed=fold_elapsed,
            )
    else:
        for split in splits:
            if not active_models:
                break
            seed_everything(RANDOM_SEED + int(split["fold"]))
            fold_inputs = _prepare_fold_inputs(
                dataset=dataset,
                split=split,
                feature_names=feature_names,
                label_col=label_col,
                date_col=date_col,
                entity_col=entity_col,
                eval_label_col=eval_label_col,
                macro_panel=macro_panel,
                need_pca_inputs=need_pca_inputs,
                need_ragged_inputs=need_ragged_inputs,
                temporal_by_fold=temporal_by_fold,
                temporal_keys=temporal_keys,
                temporal_feature_names=temporal_feature_names,
            )
            if fold_inputs is None:
                log(f"    Fold {split['fold']}: skipped (insufficient train/validation dates)")
                continue
            display_input = fold_inputs["ragged"] or fold_inputs["pca"]
            input_kind = "ragged" if fold_inputs["ragged"] is not None else "persistent"
            log(
                f"    Fold {split['fold']}: {input_kind} "
                f"train={display_input['n_train_periods']}, "
                f"val={display_input['n_val_periods']}, "
                f"max_N={display_input['chars_train'].shape[1]}"
            )
            for model_name in active_models:
                model_input = fold_inputs["pca"] if model_name == "pca" else fold_inputs["ragged"]
                checkpoint_preds, extra, fold_elapsed = fit_fold(
                    model_name,
                    model_input,
                    int(split["fold"]),
                )
                record_fold(
                    split=split,
                    model_name=model_name,
                    model_input=model_input,
                    checkpoint_preds=checkpoint_preds,
                    extra=extra,
                    fold_elapsed=fold_elapsed,
                )

    for model_name in active_models:
        fold_ics_df = pl.DataFrame(state[model_name]["fold_ics"])
        preds_df = (
            pl.concat(state[model_name]["pred_frames"])
            if state[model_name]["pred_frames"]
            else pl.DataFrame()
        )
        best_epoch, mean_ic = _select_reporting_epoch(
            fold_ics_df,
            checkpoint_selection_policy=metric_policy["checkpoint_selection_policy"],
            reporting_epoch=metric_policy["reporting_epoch"],
        )
        elapsed = time.perf_counter() - started_clock[model_name]

        model_results.append(
            {
                "model_name": model_name,
                "mean_ic": round(mean_ic, 4),
                "best_epoch": best_epoch,
                "n_folds": int(fold_ics_df["fold_id"].n_unique()) if fold_ics_df.height > 0 else 0,
                "elapsed_s": round(elapsed, 1),
                "started_at": started_at[model_name],
            }
        )

        all_predictions[model_name] = preds_df
        fold_metrics[model_name] = fold_ics_df
        all_extras[model_name] = state[model_name]["fold_extras"]

        _require_ipca_convergence(model_name, state[model_name]["fold_extras"])

        model_dir = model_dirs[model_name]
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            preds_df.write_parquet(model_dir / "predictions.parquet")
            fold_ics_df.write_parquet(model_dir / "fold_metrics.parquet")

        if case_study_id and preds_df.height > 0:
            model_temporal_feature_assembly = (
                temporal_feature_assembly if model_name != "pca" else None
            )
            model_temporal_feature_digest = temporal_feature_digest if model_name != "pca" else None
            model_macro_context = macro_context_spec if model_name == "sdf" else None
            training_hash = _register_model_predictions(
                case_study_id=case_study_id,
                model_name=model_name,
                label_col=label_col,
                n_epochs=n_epochs,
                n_factors=n_factors,
                notebook=notebook,
                prediction_split=prediction_split,
                task_type=task_type,
                class_values=class_values,
                eval_label_col=eval_label_col,
                started_at=started_at[model_name],
                elapsed=elapsed,
                model_kwargs=model_kwargs.get(model_name, {}),
                fold_extras=state[model_name]["fold_extras"],
                fold_ics_df=fold_ics_df,
                preds_df=preds_df,
                feature_names=feature_names,
                splits=splits,
                input_digest=input_digest,
                macro_digest=macro_digest,
                runtime_spec=runtime_spec,
                temporal_feature_assembly=model_temporal_feature_assembly,
                temporal_feature_digest=model_temporal_feature_digest,
                macro_context_spec=model_macro_context,
                input_data_spec=input_data_spec,
            )
            if state[model_name]["fold_extras"]:
                from utils.paths import get_case_study_dir

                extras_dir = (
                    get_case_study_dir(case_study_id) / "run_log" / "training" / training_hash
                )
                extras_dir.mkdir(parents=True, exist_ok=True)
                _save_fold_extras(extras_dir / "fold_extras.json", state[model_name]["fold_extras"])

        log(f"    -> best epoch={best_epoch}, IC={mean_ic:+.4f} ({elapsed:.1f}s)")
        gc.collect()

    if model_results:
        model_results.sort(key=lambda row: row["mean_ic"], reverse=True)
        best = model_results[0]
    else:
        best = {"model_name": "none", "mean_ic": 0.0}

    log(f"  Best: {best['model_name']} (IC={best['mean_ic']:+.4f})")
    log_file.close()

    return {
        "model_results": model_results,
        "best_model": best["model_name"],
        "best_ic": best["mean_ic"],
        "all_predictions": all_predictions,
        "fold_metrics": fold_metrics,
        "fold_extras": all_extras,
    }


def _frame_digest(frame: pl.DataFrame) -> str:
    """Return a stable digest of an ordered Polars frame and its schema."""
    columns = list(frame.columns)
    key_cols = [column for column in ("fold", "timestamp", "symbol") if column in columns]
    ordered = frame.sort(key_cols) if key_cols else frame
    row_hashes = ordered.hash_rows(seed=42).to_numpy()
    schema = [(column, str(ordered.schema[column])) for column in columns]
    digest = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _latent_input_digest(
    dataset: pl.DataFrame,
    *,
    feature_names: list[str],
    label_col: str,
    eval_label_col: str | None,
    date_col: str,
    entity_col: str,
    splits: list[dict[str, Any]],
) -> str:
    columns = [date_col, entity_col, *feature_names, label_col]
    if eval_label_col:
        columns.append(eval_label_col)
    dataset = _filter_dataset_for_splits(dataset, splits=splits, date_col=date_col)
    return _frame_digest(dataset.select(columns))


def _filter_dataset_for_splits(
    dataset: pl.DataFrame,
    *,
    splits: list[dict[str, Any]],
    date_col: str,
) -> pl.DataFrame:
    """Keep only rows used by at least one declared train or validation window."""
    filter_col = pl.col(date_col)
    if (
        hasattr(dataset[date_col].dtype, "time_zone")
        and dataset[date_col].dtype.time_zone is not None
    ):
        filter_col = filter_col.dt.replace_time_zone(None)

    used = pl.lit(False)
    for split in splits:
        train = filter_col.is_between(
            _to_naive_timestamp(split["train_start"]),
            _to_naive_timestamp(split["train_end"]),
            closed="both",
        )
        validation = filter_col.is_between(
            _to_naive_timestamp(split["val_start"]),
            _to_naive_timestamp(split["val_end"]),
            closed="both",
        )
        used = used | train | validation
    return dataset.filter(used)


def _require_ipca_convergence(
    model_name: str,
    fold_extras: list[dict[str, Any]],
) -> None:
    if model_name != "ipca":
        return
    failed = [int(extra["fold_id"]) for extra in fold_extras if not extra.get("converged", False)]
    if failed:
        raise RuntimeError(
            "IPCA did not converge for folds "
            f"{failed}; refusing to register predictions from an incomplete ALS fit"
        )


def _prepare_fold_inputs(
    *,
    dataset: pl.DataFrame,
    split: dict[str, Any],
    feature_names: list[str],
    label_col: str,
    date_col: str,
    entity_col: str,
    eval_label_col: str | None,
    macro_panel: pl.DataFrame | None,
    need_pca_inputs: bool,
    need_ragged_inputs: bool = True,
    temporal_by_fold: pd.DataFrame | None = None,
    temporal_keys: list[str] | None = None,
    temporal_feature_names: list[str] | None = None,
) -> dict[str, Any] | None:
    if temporal_by_fold is not None and temporal_keys and temporal_feature_names:
        dataset = _replace_fold_temporal_features(
            dataset=dataset,
            temporal_by_fold=temporal_by_fold,
            temporal_keys=temporal_keys,
            temporal_feature_names=temporal_feature_names,
            fold_id=int(split["fold"]),
        )
    fold_dataset = _filter_dataset_window(
        dataset,
        date_col=date_col,
        start=split["train_start"],
        end=split["val_end"],
    )
    ragged_inputs = None
    if need_ragged_inputs:
        ragged_dataset = fold_dataset
        if macro_panel is not None:
            macro_start = macro_panel.select(pl.col(date_col).min()).item()
            ragged_dataset = ragged_dataset.filter(pl.col(date_col) >= macro_start)
        ragged_panel = prepare_ragged_panel_data(
            ragged_dataset,
            feature_names=feature_names,
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            eval_label_col=eval_label_col,
            macro_panel=macro_panel,
        )
        ragged_panel["chars"] = rank_normalize_cross_section(ragged_panel["chars"])

        ragged_train_mask = _date_mask(
            ragged_panel["dates"], split["train_start"], split["train_end"]
        )
        ragged_val_mask = _date_mask(ragged_panel["dates"], split["val_start"], split["val_end"])
        ragged_inputs = {
            "chars_train": ragged_panel["chars"][ragged_train_mask],
            "returns_train": ragged_panel["returns"][ragged_train_mask],
            "chars_val": ragged_panel["chars"][ragged_val_mask],
            "returns_val": ragged_panel["returns"][ragged_val_mask],
            "factor_returns_train": (
                ragged_panel["eval_returns"][ragged_train_mask]
                if ragged_panel.get("eval_returns") is not None
                else None
            ),
            "eval_returns_val": (
                ragged_panel["eval_returns"][ragged_val_mask]
                if ragged_panel.get("eval_returns") is not None
                else None
            ),
            "val_dates": ragged_panel["dates"][ragged_val_mask],
            "val_entities": ragged_panel["entities"][ragged_val_mask],
            "macro_train": (
                ragged_panel["macro"][ragged_train_mask]
                if ragged_panel.get("macro") is not None
                else None
            ),
            "macro_val": (
                ragged_panel["macro"][ragged_val_mask]
                if ragged_panel.get("macro") is not None
                else None
            ),
            "n_train_periods": int(ragged_train_mask.sum()),
            "n_val_periods": int(ragged_val_mask.sum()),
        }

    persistent_inputs = None
    if need_pca_inputs:
        train_dataset = _filter_dataset_window(
            fold_dataset,
            date_col=date_col,
            start=split["train_start"],
            end=split["train_end"],
        )
        persistent_panel = prepare_panel_data(
            fold_dataset,
            feature_names=feature_names,
            label_col=label_col,
            date_col=date_col,
            entity_col=entity_col,
            eligibility_dataset=train_dataset,
            eval_label_col=eval_label_col,
        )
        persistent_train_mask = _date_mask(
            persistent_panel["dates"],
            split["train_start"],
            split["train_end"],
        )
        persistent_val_mask = _date_mask(
            persistent_panel["dates"],
            split["val_start"],
            split["val_end"],
        )
        persistent_inputs = {
            "chars_train": persistent_panel["chars"][persistent_train_mask],
            "returns_train": persistent_panel["returns"][persistent_train_mask],
            "chars_val": persistent_panel["chars"][persistent_val_mask],
            "returns_val": persistent_panel["returns"][persistent_val_mask],
            "eval_returns_val": (
                persistent_panel["eval_returns"][persistent_val_mask]
                if persistent_panel.get("eval_returns") is not None
                else None
            ),
            "val_dates": persistent_panel["dates"][persistent_val_mask],
            "val_entities": np.broadcast_to(
                persistent_panel["entities"][None, :],
                (int(persistent_val_mask.sum()), len(persistent_panel["entities"])),
            ).copy(),
            "macro_train": None,
            "macro_val": None,
            "n_train_periods": int(persistent_train_mask.sum()),
            "n_val_periods": int(persistent_val_mask.sum()),
        }

    available_input = ragged_inputs or persistent_inputs
    if available_input is None:
        raise ValueError("At least one latent-factor input representation is required")
    if available_input["n_train_periods"] < 10 or available_input["n_val_periods"] < 3:
        return None

    return {"ragged": ragged_inputs, "pca": persistent_inputs}


def _replace_fold_temporal_features(
    *,
    dataset: pl.DataFrame,
    temporal_by_fold: pd.DataFrame,
    temporal_keys: list[str],
    temporal_feature_names: list[str],
    fold_id: int,
) -> pl.DataFrame:
    """Replace the schema-placeholder columns with one fold's learned features."""
    from utils.modeling import fold_temporal_frame

    fold_temporal = fold_temporal_frame(
        temporal_by_fold, fold_id, temporal_keys=temporal_keys, schema=dataset.schema
    )
    if fold_temporal.is_empty():
        raise ValueError(f"No temporal features found for fold {fold_id}")
    fold_temporal = fold_temporal.unique(subset=temporal_keys, keep="last")

    missing = sorted(set(temporal_feature_names) - set(fold_temporal.columns))
    if missing:
        raise ValueError(f"Fold {fold_id} temporal data is missing features: {missing}")
    return dataset.drop(temporal_feature_names, strict=False).join(
        fold_temporal.select([*temporal_keys, *temporal_feature_names]),
        on=temporal_keys,
        how="left",
    )


def _filter_dataset_window(
    dataset: pl.DataFrame,
    *,
    date_col: str,
    start: Any,
    end: Any,
) -> pl.DataFrame:
    start_ts = _to_naive_timestamp(start)
    end_ts = _to_naive_timestamp(end)
    filter_col = pl.col(date_col)
    if (
        hasattr(dataset[date_col].dtype, "time_zone")
        and dataset[date_col].dtype.time_zone is not None
    ):
        filter_col = filter_col.dt.replace_time_zone(None)
    return dataset.filter((filter_col >= start_ts) & (filter_col <= end_ts))


def _date_mask(dates: np.ndarray, start: Any, end: Any) -> np.ndarray:
    start_dt = np.datetime64(_to_naive_timestamp(start))
    end_dt = np.datetime64(_to_naive_timestamp(end))
    dates_arr = np.asarray(dates, dtype="datetime64[ns]")
    return (dates_arr >= start_dt) & (dates_arr <= end_dt)


def _to_naive_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _temporal_digest_frame(
    temporal_by_fold: Any,
    temporal_keys: list[str],
    temporal_feature_names: list[str],
) -> pl.DataFrame:
    """The columns the temporal digest covers, projected out of whatever form is held.

    The one consumer that spans every fold rather than selecting one, so it is also the only
    place the whole artifact is read - and it reads the hashed columns alone, not the table.
    """
    columns = ["fold", *temporal_keys, *temporal_feature_names]
    if isinstance(temporal_by_fold, pl.LazyFrame):
        return temporal_by_fold.select(columns).collect()
    if isinstance(temporal_by_fold, pl.DataFrame):
        return temporal_by_fold.select(columns)
    return pl.from_pandas(temporal_by_fold.loc[:, columns])


def _resolve_metric_policy(
    *,
    case_study_id: str | None,
    label_col: str | None,
    checkpoint_selection_policy: str | None,
    reporting_epoch: int | None,
    score_dates: str,
    score_cadence: str | None,
    score_rebalance_step: int | None,
) -> dict[str, Any]:
    from utils import CASE_STUDIES_DIR

    setup: dict[str, Any] = {}
    lf_setup: dict[str, Any] = {}
    if case_study_id:
        setup = yaml.safe_load(
            (CASE_STUDIES_DIR / case_study_id / "config" / "setup.yaml").read_text()
        )
        lf_setup = (setup.get("modeling") or {}).get("latent_factors") or {}

    selection_policy = checkpoint_selection_policy or lf_setup.get(
        "checkpoint_selection_policy",
        "fixed",
    )
    if selection_policy not in {"fixed", "validation_ic"}:
        raise ValueError(
            "checkpoint_selection_policy must be 'fixed' or 'validation_ic'; "
            f"got {selection_policy!r}"
        )

    resolved_reporting_epoch = reporting_epoch
    if resolved_reporting_epoch is None and lf_setup.get("reporting_epoch") is not None:
        resolved_reporting_epoch = int(lf_setup["reporting_epoch"])

    score_mode = score_dates
    if score_mode == "auto":
        # IC is defined at every prediction timestamp, then averaged. Portfolio
        # rebalance cadence belongs to the downstream backtest and must not
        # thin the model-evaluation series or checkpoint-selection metric.
        score_mode = lf_setup.get("score_dates") or "all"
    if score_mode not in {"all", "rebalance"}:
        raise ValueError(f"score_dates must be 'auto', 'all', or 'rebalance'; got {score_dates!r}")

    resolved_cadence = score_cadence or lf_setup.get("score_cadence")
    resolved_step = score_rebalance_step
    if resolved_step is None and lf_setup.get("score_rebalance_step") is not None:
        resolved_step = int(lf_setup["score_rebalance_step"])

    if score_mode == "rebalance":
        if resolved_cadence is None:
            resolved_cadence = (setup.get("decision") or {}).get("cadence")
        if resolved_step is None:
            resolved_step = (
                get_rebalance_step(case_study_id, label_col) if case_study_id and label_col else 1
            )
        if not resolved_cadence:
            raise ValueError(
                "score_dates='rebalance' requires a cadence; pass score_cadence or declare "
                "decision.cadence in setup.yaml."
            )
    else:
        resolved_step = 1 if resolved_step is None else int(resolved_step)
        resolved_cadence = resolved_cadence or ""

    return {
        "checkpoint_selection_policy": selection_policy,
        "reporting_epoch": resolved_reporting_epoch,
        "score_dates": score_mode,
        "score_cadence": resolved_cadence,
        "score_rebalance_step": int(resolved_step),
    }


def _score_prediction_frame(
    frame: pl.DataFrame | None,
    *,
    score_dates: str,
    score_cadence: str,
    score_rebalance_step: int,
) -> pl.DataFrame | None:
    if frame is None or frame.height == 0 or score_dates == "all":
        return frame

    return thin_to_rebalance_dates(
        frame,
        cadence=score_cadence,
        step=score_rebalance_step,
        time_col="timestamp",
    )


def _compute_frame_ic(frame: pl.DataFrame | None) -> tuple[float, int]:
    if frame is None or frame.height == 0:
        return 0.0, 0

    target_col = "eval_actual" if "eval_actual" in frame.columns else "y_true"
    scored = frame.filter(pl.col(target_col).is_finite() & pl.col("y_score").is_finite())
    if scored.height == 0:
        return 0.0, 0

    predictions = scored.select(
        pl.col("timestamp").alias("date"),
        pl.col("symbol").alias("entity"),
        pl.col("y_score").alias("prediction"),
    )
    returns = scored.select(
        pl.col("timestamp").alias("date"),
        pl.col("symbol").alias("entity"),
        pl.col(target_col).alias("forward_return"),
    )
    ic_result = cross_sectional_ic(
        predictions=predictions,
        returns=returns,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="date",
        entity_col="entity",
        method="spearman",
        min_obs=3,
    )
    n_periods = int(ic_result["n_periods"]) if ic_result["n_periods"] > 0 else 0
    ic_mean = float(ic_result["ic_mean"]) if n_periods > 0 else 0.0
    return ic_mean, n_periods


def _build_prediction_frame(
    *,
    predictions: np.ndarray,
    returns_val: np.ndarray,
    eval_returns_val: np.ndarray | None,
    val_dates: np.ndarray,
    val_entities: np.ndarray,
    fold_id: int,
    model_name: str,
    epoch: int,
) -> pl.DataFrame | None:
    frames: list[pl.DataFrame] = []
    for date_idx in range(predictions.shape[0]):
        valid = np.isfinite(returns_val[date_idx]) & np.isfinite(predictions[date_idx])
        if eval_returns_val is not None:
            valid &= np.isfinite(eval_returns_val[date_idx])
        if not valid.any():
            continue
        timestamp = _normalize_timestamp_value(val_dates[date_idx])
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": [timestamp] * int(valid.sum()),
                    "symbol": [val_entities[date_idx, idx] for idx in np.nonzero(valid)[0]],
                    "y_true": returns_val[date_idx, valid].astype(np.float64).tolist(),
                    "y_score": predictions[date_idx, valid].astype(np.float64).tolist(),
                    "fold_id": [fold_id] * int(valid.sum()),
                    "config_name": [model_name] * int(valid.sum()),
                    "epoch": [epoch] * int(valid.sum()),
                    **(
                        {
                            "eval_actual": eval_returns_val[date_idx, valid]
                            .astype(np.float64)
                            .tolist()
                        }
                        if eval_returns_val is not None
                        else {}
                    ),
                }
            )
        )
    if not frames:
        return None
    return pl.concat(frames)


def _normalize_timestamp_value(value: Any) -> Any:
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").item()
    if isinstance(value, datetime):
        return value
    return value


def _write_incremental_fold(
    *,
    model_dir: Path | None,
    fold_id: int,
    predictions: dict[int, np.ndarray],
    model_input: dict[str, Any],
    model_name: str,
) -> None:
    if model_dir is None:
        return
    incremental_dir = model_dir / "_incremental"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    for epoch, preds in predictions.items():
        frame = _build_prediction_frame(
            predictions=preds,
            returns_val=model_input["returns_val"],
            eval_returns_val=model_input.get("eval_returns_val"),
            val_dates=model_input["val_dates"],
            val_entities=model_input["val_entities"],
            fold_id=fold_id,
            model_name=model_name,
            epoch=epoch,
        )
        if frame is not None:
            frames.append(frame)
    if frames:
        pl.concat(frames).write_parquet(incremental_dir / f"fold{fold_id}.parquet")


def _select_epoch_from_values(
    checkpoint_ics: dict[int, float],
    *,
    checkpoint_selection_policy: str,
    reporting_epoch: int | None,
) -> tuple[int, float]:
    if not checkpoint_ics:
        return 0, 0.0

    if checkpoint_selection_policy == "validation_ic":
        epoch, ic = max(checkpoint_ics.items(), key=lambda item: (item[1], -item[0]))
        return int(epoch), float(ic)

    epoch = max(checkpoint_ics) if reporting_epoch is None else int(reporting_epoch)
    if epoch not in checkpoint_ics:
        raise ValueError(
            f"Configured reporting_epoch={epoch} was not emitted; available epochs: "
            f"{sorted(checkpoint_ics)}"
        )
    return epoch, float(checkpoint_ics[epoch])


def _select_reporting_epoch(
    metrics_df: pl.DataFrame,
    *,
    checkpoint_selection_policy: str,
    reporting_epoch: int | None,
) -> tuple[int, float]:
    if metrics_df.height == 0:
        return 0, 0.0

    summary = (
        metrics_df.group_by("epoch").agg(pl.col("ic_mean").mean().alias("mean_ic")).sort("epoch")
    )
    checkpoint_ics = {
        int(epoch): float(mean_ic)
        for epoch, mean_ic in zip(
            summary["epoch"].to_list(), summary["mean_ic"].to_list(), strict=True
        )
    }
    return _select_epoch_from_values(
        checkpoint_ics,
        checkpoint_selection_policy=checkpoint_selection_policy,
        reporting_epoch=reporting_epoch,
    )


def _register_model_predictions(
    *,
    case_study_id: str,
    model_name: str,
    label_col: str,
    n_epochs: int,
    n_factors: int,
    notebook: str,
    prediction_split: str,
    task_type: str,
    class_values: list | None,
    eval_label_col: str | None,
    started_at: str,
    elapsed: float,
    model_kwargs: dict[str, Any],
    fold_extras: list[dict[str, Any]],
    fold_ics_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    feature_names: list[str],
    splits: list[dict[str, Any]],
    input_digest: str,
    macro_digest: str | None,
    runtime_spec: dict[str, Any],
    temporal_feature_assembly: str | None = None,
    temporal_feature_digest: str | None = None,
    macro_context_spec: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
) -> str:
    from case_studies.utils.registry import (
        build_training_spec,
        register_prediction_set,
        register_training_run,
    )

    n_folds = int(fold_ics_df["fold_id"].n_unique()) if fold_ics_df.height > 0 else 0
    registered_checkpoints = _expected_latent_checkpoints(
        model_name,
        n_epochs=n_epochs,
        model_kwargs=model_kwargs,
    )
    registered_extras = (
        [{**fold_extras[0], "checkpoint_epochs": list(registered_checkpoints)}]
        if fold_extras
        else []
    )
    try:
        spec = build_training_spec(
            "latent_factors",
            model_name,
            label_col,
            n_folds=n_folds,
            n_epochs=n_epochs,
        )
    except FileNotFoundError:
        spec = {
            "config_name": model_name,
            "family": "latent_factors",
            "feature_sets": ["financial", "model_based"],
            "label": label_col,
            "library": "pytorch",
            "n_epochs": n_epochs,
            "n_folds": n_folds,
            "params": {"n_factors": n_factors},
            "seed": 42,
        }

    spec = _apply_latent_factor_runtime_spec(
        spec=spec,
        n_factors=n_factors,
        n_epochs=n_epochs,
        model_kwargs=model_kwargs,
        fold_extras=registered_extras,
        feature_names=feature_names,
        splits=splits,
        task_type=task_type,
        class_values=class_values,
        eval_label_col=eval_label_col,
        input_digest=input_digest,
        macro_digest=macro_digest,
        runtime_spec=runtime_spec,
        temporal_feature_assembly=temporal_feature_assembly,
        temporal_feature_digest=temporal_feature_digest,
        macro_context_spec=macro_context_spec,
        input_data_spec=input_data_spec,
    )

    training_hash = register_training_run(
        case_study_id,
        spec=spec,
        entry_point=notebook,
        started_at=started_at,
        elapsed_s=elapsed,
    )
    eval_col = "eval_actual" if eval_label_col else None

    for epoch in registered_checkpoints:
        epoch_preds = preds_df.filter(pl.col("epoch") == epoch)
        if epoch_preds.height == 0:
            raise ValueError(f"Missing registered checkpoint {epoch} for {model_name}")
        epoch_metrics = fold_ics_df.filter(pl.col("epoch") == epoch)
        ic_mean = float(epoch_metrics["ic_mean"].mean()) if epoch_metrics.height > 0 else 0.0
        register_prediction_set(
            case_study_id,
            training_hash,
            checkpoint_value=int(epoch),
            checkpoint_kind="epoch",
            split=prediction_split,
            predictions=epoch_preds,
            task_type=task_type,
            class_values=class_values,
            eval_col=eval_col,
            metrics={"ic_mean": ic_mean},
        )
    return training_hash


def _apply_latent_factor_runtime_spec(
    *,
    spec: dict[str, Any],
    n_factors: int,
    n_epochs: int,
    model_kwargs: dict[str, Any],
    fold_extras: list[dict[str, Any]],
    feature_names: list[str],
    splits: list[dict[str, Any]],
    task_type: str,
    class_values: list | None,
    eval_label_col: str | None,
    input_digest: str,
    macro_digest: str | None,
    runtime_spec: dict[str, Any],
    temporal_feature_assembly: str | None = None,
    temporal_feature_digest: str | None = None,
    macro_context_spec: dict[str, Any] | None = None,
    input_data_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(spec)
    params = dict(resolved.get("params", {}))
    # The K the fit ran at, not the K the preset declares. `resolved["params"]`
    # comes from the preset, which always sets n_factors, so a setdefault here
    # never took effect: a K=2 fit registered under a spec reading n_factors 5.
    # Now that the caller's K wins at the runner, K would no longer separate
    # training hashes, and a K=2 request would load or overwrite the K=5 cohort.
    params["n_factors"] = n_factors
    params["feature_names"] = list(feature_names)
    params["splits"] = [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in split.items()
        }
        for split in splits
    ]
    params["task_type"] = task_type
    params["class_values"] = list(class_values) if class_values is not None else None
    params["eval_label_col"] = eval_label_col
    params["input_digest"] = input_digest
    params["macro_digest"] = macro_digest
    params["runtime"] = dict(runtime_spec)

    # IPCA solver controls are part of the configured training identity. If
    # omitted, changing the ALS budget or tolerances reuses the historical
    # training hash and overwrites its prediction artifact in place.
    solver_fields = ("max_iter", "tol", "factor_ridge", "gamma_ridge")
    first_extra = fold_extras[0] if fold_extras else {}
    for field in solver_fields:
        if field in model_kwargs:
            params[field] = model_kwargs[field]
        elif field in first_extra:
            params[field] = first_extra[field]

    runtime_fields: dict[str, Any] = {}
    if n_epochs:
        runtime_fields["n_epochs"] = n_epochs
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
        "output_mode",
        "expected_return_mapper",
    ):
        if field in model_kwargs and model_kwargs[field] not in (None, (), []):
            runtime_fields[field] = model_kwargs[field]

    if fold_extras:
        for field in (
            "checkpoint_epochs",
            "n_epochs_unc",
            "n_epochs_moment",
            "n_epochs_cond",
            "beta_n_epochs",
            "beta_checkpoint_epochs",
            "beta_default_checkpoint",
            "output_mode",
            "expected_return_mapper",
        ):
            if field in first_extra and first_extra[field] not in (None, (), []):
                runtime_fields[field] = first_extra[field]

    resolved.update(runtime_fields)
    if (temporal_feature_assembly is None) != (temporal_feature_digest is None):
        raise ValueError("Fold-scoped temporal identity requires both assembly and input digest")
    if temporal_feature_assembly is not None:
        resolved["temporal_feature_assembly"] = temporal_feature_assembly
        resolved["temporal_feature_digest"] = temporal_feature_digest
    if macro_context_spec is not None:
        resolved["macro_context"] = macro_context_spec
    if input_data_spec is not None:
        resolved["input_data"] = input_data_spec
    resolved["params"] = params
    return resolved


def _macro_context_matches(
    registered_spec: dict[str, Any],
    expected_macro_context: dict[str, Any] | None,
) -> bool:
    """Require exact macro identity when the caller declares one."""
    if expected_macro_context is None:
        return True
    return registered_spec.get("macro_context") == expected_macro_context
