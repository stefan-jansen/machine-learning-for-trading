"""Shared case-study helpers for latent-factor notebook runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl
import yaml

from case_studies.utils.latent_factors import EXPENSIVE_MODELS, run_latent_factor_cv
from data import load_macro
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
    dataset: pl.DataFrame
    feature_names: list[str]
    task_type: str
    class_values: list[Any]
    eval_label_col: str | None
    date_col: str
    entity_col: str
    splits: list[dict[str, Any]]
    max_symbols: int = 0
    max_folds: int = 0

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
    macro_panel = load_macro() if use_macro else None

    modeling_dataset = load_modeling_dataset(
        case_study_id, resolved_primary, max_symbols=max_symbols
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
        dataset=modeling_dataset.dataset,
        feature_names=modeling_dataset.feature_names,
        task_type=modeling_dataset.task_type,
        class_values=list(modeling_dataset.class_values or []),
        eval_label_col=modeling_dataset.eval_label_col,
        date_col=modeling_dataset.date_col,
        entity_col=modeling_dataset.entity_cols[0] if modeling_dataset.entity_cols else "symbol",
        splits=splits,
        max_symbols=max_symbols,
        max_folds=max_folds,
    )


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
        persistent_entities=context.persistent_entities,
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
            persistent_entities=context.persistent_entities,
        )
    return results


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
