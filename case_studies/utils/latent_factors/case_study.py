"""Shared case-study helpers for latent-factor notebook runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import yaml

from case_studies.utils.latent_factors import EXPENSIVE_MODELS, run_latent_factor_cv
from case_studies.utils.latent_factors.library_bridge import preferred_latent_device
from case_studies.utils.latent_factors.macro_context import load_configured_macro_context
from data import load_macro
from utils.artifact_specs import load_feature_spec, load_label_spec, resolve_storage_path
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LatentFactorCaseStudyContext:
    case_study_id: str
    case_dir: Any
    setup: dict[str, Any]
    primary_label: str
    variant_labels: list[str]
    model_kwargs: dict[str, dict[str, Any]]
    setup_model_kwargs: dict[str, dict[str, Any]]
    persistent_entities: bool
    macro_panel: pl.DataFrame | None
    macro_context_spec: dict[str, Any] | None
    input_data_spec: dict[str, Any]
    dataset: pl.DataFrame
    feature_names: list[str]
    task_type: str
    class_values: list[Any]
    eval_label_col: str | None
    date_col: str
    entity_col: str
    splits: list[dict[str, Any]]
    temporal_by_fold: pd.DataFrame | None
    temporal_keys: list[str]
    temporal_feature_names: list[str]
    temporal_artifact_splits: list[dict[str, Any]]
    max_symbols: int = 0
    max_folds: int = 0
    device: str = "cpu"
    num_threads: int = 8
    deterministic_algorithms: bool = True

    def model_kwargs_for_label(self, label: str) -> dict[str, dict[str, Any]]:
        """Return preset+setup-merged model_kwargs for a specific label.

        The primary-label kwargs are pre-merged in ``model_kwargs``. Variant
        labels can declare different preset menus / hyperparameters under
        ``case_studies/config/{model}/{label}.yaml``, so re-resolve per label
        and re-apply the global setup overrides.
        """
        if label == self.primary_label:
            return self.model_kwargs
        preset = _load_preset_model_kwargs(self.case_study_id, label)
        return _merge_model_kwargs(preset, self.setup_model_kwargs)


def load_case_study_context(
    case_study_id: str,
    *,
    primary_label: str = "",
    max_symbols: int = 0,
    max_folds: int = 0,
    max_variant_labels: int = -1,
    use_macro: bool = True,
) -> LatentFactorCaseStudyContext:
    case_dir = get_case_study_dir(case_study_id)
    setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())

    resolved_primary = primary_label or setup["labels"]["primary"]
    variant_labels = list(setup.get("labels", {}).get("variants", []))
    if max_variant_labels >= 0:
        variant_labels = variant_labels[:max_variant_labels]

    lf_setup = setup.get("modeling", {}).get("latent_factors", {})
    preset_kwargs = _load_preset_model_kwargs(case_study_id, resolved_primary)
    setup_kwargs = _normalize_model_kwargs(lf_setup.get("model_kwargs", {}))
    model_kwargs = _merge_model_kwargs(preset_kwargs, setup_kwargs)
    persistent_entities = bool(lf_setup.get("persistent_entities", True))
    device = str(lf_setup.get("device", preferred_latent_device()))
    num_threads = int(lf_setup.get("num_threads", 8))
    deterministic_algorithms = bool(lf_setup.get("deterministic_algorithms", True))
    macro_context_config = lf_setup.get("macro_context")
    if use_macro and macro_context_config:
        macro_panel, macro_context_spec = load_configured_macro_context(macro_context_config)
    elif use_macro:
        macro_panel = _load_macro_panel(lf_setup)
        macro_context_spec = {
            "policy": "load_macro_fallback",
            "version": "v1",
            "series": [column for column in macro_panel.columns if column != "timestamp"],
            "availability_lag_days": int(lf_setup.get("macro_availability_lag_days", 0)),
            "alignment": "backward_asof",
        }
    else:
        macro_panel, macro_context_spec = (
            None,
            {
                "policy": "disabled",
                "version": "v1",
                "series": [],
                "availability_lag_days": 0,
                "alignment": "none",
                "input_digest": None,
            },
        )

    modeling_dataset = load_modeling_dataset(
        case_study_id, resolved_primary, max_symbols=max_symbols
    )
    input_data_spec = training_input_identity(
        case_study_id,
        resolved_primary,
        eval_label=modeling_dataset.eval_label_col,
    )
    splits = modeling_dataset.splits[:max_folds] if max_folds else modeling_dataset.splits

    return LatentFactorCaseStudyContext(
        case_study_id=case_study_id,
        case_dir=case_dir,
        setup=setup,
        primary_label=resolved_primary,
        variant_labels=variant_labels,
        model_kwargs=model_kwargs,
        setup_model_kwargs=setup_kwargs,
        persistent_entities=persistent_entities,
        macro_panel=macro_panel,
        macro_context_spec=macro_context_spec,
        input_data_spec=input_data_spec,
        dataset=modeling_dataset.dataset,
        feature_names=modeling_dataset.feature_names,
        task_type=modeling_dataset.task_type,
        class_values=list(modeling_dataset.class_values or []),
        eval_label_col=modeling_dataset.eval_label_col,
        date_col=modeling_dataset.date_col,
        entity_col=modeling_dataset.entity_cols[0] if modeling_dataset.entity_cols else "symbol",
        splits=splits,
        temporal_by_fold=modeling_dataset.temporal_by_fold,
        temporal_keys=modeling_dataset.temporal_keys,
        temporal_feature_names=modeling_dataset.temporal_feature_names,
        temporal_artifact_splits=modeling_dataset.temporal_artifact_splits,
        max_symbols=max_symbols,
        max_folds=max_folds,
        device=device,
        num_threads=num_threads,
        deterministic_algorithms=deterministic_algorithms,
    )


def _load_macro_panel(lf_setup: dict[str, Any]) -> pl.DataFrame:
    """Load the configured point-in-time macro surface."""
    configured_series = lf_setup.get("macro_series")
    if configured_series is not None:
        if not isinstance(configured_series, list) or not configured_series:
            raise ValueError("modeling.latent_factors.macro_series must be a non-empty list")
        if not all(isinstance(name, str) and name for name in configured_series):
            raise ValueError("Every latent-factor macro series must be a non-empty string")

    macro_panel = load_macro(series=configured_series)
    if configured_series is not None:
        missing = sorted(set(configured_series) - set(macro_panel.columns))
        if missing:
            raise ValueError(f"Configured latent-factor macro series are unavailable: {missing}")

    availability_lag_days = int(lf_setup.get("macro_availability_lag_days", 0))
    if availability_lag_days < 0:
        raise ValueError("macro_availability_lag_days cannot be negative")
    if availability_lag_days:
        macro_panel = macro_panel.with_columns(
            pl.col("timestamp") + pl.duration(days=availability_lag_days)
        )
    return macro_panel


def configured_models(
    context: LatentFactorCaseStudyContext,
    *,
    include_expensive: bool = True,
) -> list[str]:
    configs = load_configs(context.case_study_id, context.primary_label, "latent_factors")
    models = [config["config_name"] for config in configs]
    if not context.persistent_entities:
        models = [model_name for model_name in models if model_name != "pca"]
    if not include_expensive:
        models = [model_name for model_name in models if model_name not in EXPENSIVE_MODELS]
    return models


def run_case_study_model(
    context: LatentFactorCaseStudyContext,
    *,
    model_name: str,
    notebook: str = "latent_factors",
    n_factors: int,
    n_epochs: int,
    use_cache: bool,
    force_retrain: bool,
    macro_panel: pl.DataFrame | None = None,
    reporting_epoch: int | None = None,
    fold_workers: int = 1,
) -> dict[str, Any]:
    return run_latent_factor_cv(
        panel_data=None,
        splits=context.splits,
        models=[model_name],
        n_factors=n_factors,
        n_epochs=n_epochs,
        model_kwargs=context.model_kwargs,
        case_study_id=context.case_study_id,
        label_col=context.primary_label,
        eval_label_col=context.eval_label_col,
        task_type=context.task_type,
        class_values=context.class_values or None,
        notebook=notebook,
        save_dir=None,
        use_cache=use_cache,
        force_retrain=force_retrain,
        dataset=context.dataset,
        feature_names=context.feature_names,
        date_col=context.date_col,
        entity_col=context.entity_col,
        macro_panel=context.macro_panel if macro_panel is None else macro_panel,
        macro_context_spec=context.macro_context_spec,
        input_data_spec=context.input_data_spec,
        persistent_entities=context.persistent_entities,
        device=context.device,
        num_threads=context.num_threads,
        deterministic_algorithms=context.deterministic_algorithms,
        temporal_by_fold=context.temporal_by_fold,
        temporal_keys=context.temporal_keys,
        temporal_feature_names=context.temporal_feature_names,
        reporting_epoch=reporting_epoch,
        fold_workers=fold_workers,
    )


def run_case_study_variants(
    context: LatentFactorCaseStudyContext,
    *,
    model_name: str,
    notebook: str = "latent_factors",
    n_factors: int,
    n_epochs: int,
    use_cache: bool,
    force_retrain: bool,
    reporting_epoch: int | None = None,
    fold_workers: int = 1,
) -> dict[str, dict[str, Any]]:
    from utils.modeling import ConfigError

    results: dict[str, dict[str, Any]] = {}
    for variant_label in context.variant_labels:
        # Skip variants whose training menu has no latent_factors family
        # (e.g., classification variants that intentionally exclude LF/DL).
        try:
            load_configs(context.case_study_id, variant_label, "latent_factors")
        except ConfigError:
            logger.info(
                "Skipping variant %r: no latent_factors family in training menu",
                variant_label,
            )
            continue
        modeling_dataset = load_modeling_dataset(
            context.case_study_id, variant_label, max_symbols=context.max_symbols
        )
        input_data_spec = training_input_identity(
            context.case_study_id,
            variant_label,
            eval_label=modeling_dataset.eval_label_col,
        )
        # Honour the same max_folds truncation the primary run uses, so a
        # `max_folds=2` smoke test does not silently retrain variants on the
        # full split set.
        variant_splits = (
            modeling_dataset.splits[: context.max_folds]
            if context.max_folds
            else modeling_dataset.splits
        )
        results[variant_label] = run_latent_factor_cv(
            panel_data=None,
            splits=variant_splits,
            models=[model_name],
            n_factors=n_factors,
            n_epochs=n_epochs,
            # Re-resolve per-label preset kwargs: variants can declare
            # different model menus or hyperparameters under
            # `case_studies/config/{family}/{label}.yaml`, and reusing the
            # primary-label kwargs would silently override them.
            model_kwargs=context.model_kwargs_for_label(variant_label),
            case_study_id=context.case_study_id,
            label_col=variant_label,
            eval_label_col=modeling_dataset.eval_label_col,
            task_type=modeling_dataset.task_type,
            class_values=list(modeling_dataset.class_values or []) or None,
            notebook=notebook,
            save_dir=None,
            use_cache=use_cache,
            force_retrain=force_retrain,
            dataset=modeling_dataset.dataset,
            feature_names=modeling_dataset.feature_names,
            date_col=modeling_dataset.date_col,
            entity_col=modeling_dataset.entity_cols[0]
            if modeling_dataset.entity_cols
            else "symbol",
            # Forward macro_panel so SDF variants train with macro inputs;
            # without this the variant SDF runs would silently degrade to a
            # macro-less fit while the primary uses the full panel.
            macro_panel=context.macro_panel,
            macro_context_spec=context.macro_context_spec,
            input_data_spec=input_data_spec,
            persistent_entities=context.persistent_entities,
            device=context.device,
            num_threads=context.num_threads,
            deterministic_algorithms=context.deterministic_algorithms,
            temporal_by_fold=modeling_dataset.temporal_by_fold,
            temporal_keys=modeling_dataset.temporal_keys,
            temporal_feature_names=modeling_dataset.temporal_feature_names,
            reporting_epoch=reporting_epoch,
            fold_workers=fold_workers,
        )
    return results


def training_input_identity(
    case_study_id: str,
    label: str,
    *,
    eval_label: str | None = None,
) -> dict[str, Any]:
    """Return portable content identity for every materialized modeling input."""
    case_dir = get_case_study_dir(case_study_id)
    inputs = {
        "financial": resolve_storage_path(
            case_study_id,
            load_feature_spec(case_study_id, "financial"),
            "features/financial.parquet",
        ),
        "label": resolve_storage_path(
            case_study_id,
            load_label_spec(case_study_id, label),
            f"labels/{label}.parquet",
        ),
        "setup": case_dir / "config" / "setup.yaml",
    }
    temporal_path = resolve_storage_path(
        case_study_id,
        load_feature_spec(case_study_id, "model_based"),
        "features/model_based.parquet",
    )
    if temporal_path.exists():
        inputs["model_based"] = temporal_path
    if eval_label is not None and eval_label != label:
        inputs["evaluation_label"] = resolve_storage_path(
            case_study_id,
            load_label_spec(case_study_id, eval_label),
            f"labels/{eval_label}.parquet",
        )

    missing = [role for role, path in inputs.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training inputs for identity: {sorted(missing)}")

    files = [
        {"role": role, "sha256": f"sha256:{_sha256_file(path)}"}
        for role, path in sorted(inputs.items())
    ]
    aggregate = sha256(
        "\n".join(f"{item['role']}={item['sha256']}" for item in files).encode()
    ).hexdigest()
    return {"version": "v1", "files": files, "input_digest": f"sha256:{aggregate}"}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_model_kwargs(model_kwargs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized = {model_name: dict(params) for model_name, params in model_kwargs.items()}
    for params in normalized.values():
        if "hidden_units" in params and isinstance(params["hidden_units"], list):
            params["hidden_units"] = tuple(params["hidden_units"])
    return normalized


def _load_preset_model_kwargs(case_study_id: str, label: str) -> dict[str, dict[str, Any]]:
    configs = load_configs(case_study_id, label, "latent_factors")
    preset_kwargs: dict[str, dict[str, Any]] = {}
    metadata_keys = {"config_name", "family", "library", "model_class"}
    for config in configs:
        params = dict(config.get("params", {}))
        for key, value in config.items():
            if key in metadata_keys or key == "params":
                continue
            params[key] = value
        preset_kwargs[config["config_name"]] = params
    return _normalize_model_kwargs(preset_kwargs)


def _merge_model_kwargs(
    base: dict[str, dict[str, Any]],
    override: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {model_name: dict(params) for model_name, params in base.items()}
    for model_name, params in override.items():
        merged.setdefault(model_name, {})
        merged[model_name].update(params)
    return merged


# Deprecated private aliases. Thirty notebook cells import these names with their leading
# underscore, which is what made them look private in the first place; each is a
# cross-module interface and is now public. The aliases keep those callers working until
# each notebook is re-executed with the public name, and go when the last one moves.
_training_input_identity = training_input_identity
