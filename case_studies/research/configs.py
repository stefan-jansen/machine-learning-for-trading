"""The declared model population, as a visible catalog.

A case study declares which models it fits in its training menus,
``config/training/<label>.yaml``. This module turns those menus into a Polars frame so a notebook
can *show* the population it is about to fit rather than describe it in prose, and build the
requests from the same frame the reader just saw.

Three case studies had each written a private copy of this before it existed here, and the copies
had drifted: two of the three silently shrank the population when a configuration name was
mistyped. Selection is strict here for that reason.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import polars as pl
import yaml

from case_studies.utils.registry.specs import training_hash_from_spec
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

from .contracts import ExecutionTier
from .model_planning import ModelPlan
from .models import ModelRequest, ResolvedModelRequest
from .workspace import Study

CATALOG_COLUMNS = ("family", "label", "config_name", "model_class", "params")
REQUEST_COLUMNS = ("family", "label", "config_name")


def _format_params(params: dict | None) -> str:
    if not params:
        return "defaults"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _setup_labels(study: Study) -> dict:
    setup_path = get_case_study_dir(study.case_study) / "config" / "setup.yaml"
    return (yaml.safe_load(setup_path.read_text()) or {}).get("labels") or {}


def primary_label(study: Study) -> str:
    """The label the case study trades, as ``config/setup.yaml`` declares it.

    Notebooks that fit every declared label need one of them to order a comparison by, and the
    primary is the horizon the strategy chapters trade. Reading it here rather than re-parsing
    ``setup.yaml`` in each notebook keeps the answer the same everywhere it is asked.
    """
    name = _setup_labels(study).get("primary")
    if not name:
        raise ValueError(f"{study.case_study} declares no labels.primary in setup.yaml")
    return str(name)


def sweep_labels(study: Study) -> tuple[str, ...]:
    """The labels the sweep fits: ``labels.primary`` then ``labels.variants``, in that order."""
    labels = _setup_labels(study)
    primary = labels.get("primary")
    if not primary:
        raise ValueError(f"{study.case_study} declares no labels.primary in setup.yaml")
    ordered = [str(primary)]
    ordered += [str(name) for name in (labels.get("variants") or []) if str(name) != primary]
    return tuple(ordered)


def declared_labels(study: Study, family: str) -> tuple[str, ...]:
    """The sweep labels whose training menu declares ``family``, in menu-file order.

    Two files have a say and they mean different things. ``config/setup.yaml`` says which labels
    the sweep fits; a training menu says what to fit *for* a label. Reading the menu directory
    alone conflated them: ``sp500_options`` keeps full menus for four fixed-horizon labels that
    ``02_labels`` writes for ``03_financial_features``, ``05_evaluation`` and ``90_ic_diagnostic``
    to read, and that ``setup.yaml`` dropped from the sweep on 2026-05-17. A notebook fitting
    every declared label would have fitted 140 linear configurations instead of 28, and published
    four out-of-sweep labels' predictions into the population ``12_backtest`` selects over.

    The other eight case studies declare exactly the labels their menus do, so the membership is
    unchanged there. The order stays menu-file order rather than moving to ``setup.yaml`` order:
    a population's hash is computed over its members as an ordered list, so re-ordering would
    give every already-published population a new identity and make its next run demand a
    ``supersedes``. A sweep label whose menu does not declare ``family`` is skipped rather than
    raising - not every label declares every family.
    """
    menu_dir = get_case_study_dir(study.case_study) / "config" / "training"
    if not menu_dir.is_dir():
        raise FileNotFoundError(f"{study.case_study} has no training menus: {menu_dir}")
    in_sweep = set(sweep_labels(study))
    labels = []
    for path in sorted(menu_dir.glob("*.yaml")):
        if path.stem not in in_sweep:
            continue
        menu = yaml.safe_load(path.read_text()) or {}
        if menu.get(family):
            labels.append(path.stem)
    if not labels:
        raise ValueError(f"no sweep label of {study.case_study} declares {family!r} in {menu_dir}")
    return tuple(labels)


def narrows_declared_catalog(study: Study, family: str, configs: pl.DataFrame) -> bool:
    """Whether ``configs`` is less than the complete declared ``family`` catalog.

    A run that narrows the member set declares a different population from the canonical one and
    must publish under its own name. Comparing row counts is not enough: ``sp500_options`` keeps
    four out-of-sweep menus with exactly the 28 linear and 15 GBM configurations the canonical
    menu has, so ``LABELS=["fwd_ret_5d"]`` would match on count while declaring an entirely
    different set of members. The comparison is therefore over ``(label, config_name)`` pairs.
    """
    declared = load_model_configs(study, family)
    return set(zip(configs["label"], configs["config_name"], strict=True)) != set(
        zip(declared["label"], declared["config_name"], strict=True)
    )


def load_model_configs(
    study: Study,
    family: str,
    *,
    labels: Iterable[str] | None = None,
    config_names: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return the declared ``family`` population for ``study`` as visible Polars rows.

    ``labels`` defaults to every label whose training menu declares the family, so the population
    follows the menus instead of a constant that has to be kept in step with them.

    ``config_names`` selects a subset and raises on any name the requested labels do not declare.
    A mistyped configuration must fail rather than quietly fit a smaller population.
    """
    requested = tuple(labels) if labels is not None else declared_labels(study, family)
    selected = set(config_names) if config_names is not None else None

    rows: list[dict] = []
    declared: set[str] = set()
    for label in requested:
        for config in load_configs(study.case_study, label, family):
            name = str(config["config_name"])
            declared.add(name)
            if selected is not None and name not in selected:
                continue
            rows.append(
                {
                    "family": family,
                    "label": label,
                    "config_name": name,
                    "model_class": str(config.get("model_class", "")),
                    "params": _format_params(config.get("params")),
                }
            )

    if selected is not None:
        unknown = sorted(selected - declared)
        if unknown:
            raise ValueError(
                f"{family!r} configurations not declared by labels {list(requested)}: {unknown}"
            )
    if not rows:
        raise ValueError(f"no declared {family!r} requests for labels {list(requested)}")
    return pl.DataFrame(rows).unique(subset=REQUEST_COLUMNS, maintain_order=True)


def model_requests(
    study: Study,
    catalog: pl.DataFrame,
    *,
    execution_tier: str | ExecutionTier = ExecutionTier.CANONICAL,
    overrides: dict | None = None,
    preview_reductions: dict | None = None,
) -> tuple[ModelRequest, ...]:
    """Build one unresolved request per catalog row.

    Only the identity columns reach the request; the display columns exist for the reader.
    """
    missing = set(REQUEST_COLUMNS) - set(catalog.columns)
    if missing:
        raise ValueError(f"model catalog is missing {sorted(missing)}")
    tier = ExecutionTier(execution_tier)
    return tuple(
        study.model(
            **row,
            execution_tier=tier,
            overrides=dict(overrides or {}),
            preview_reductions=dict(preview_reductions or {}),
        )
        for row in catalog.select(*REQUEST_COLUMNS).iter_rows(named=True)
    )


def planned_model_plan(plan: ModelPlan) -> pl.DataFrame:
    """Show what each planned request will compute, without holding what it will compute it from.

    This is :func:`resolved_model_plan` for a notebook that plans rather than resolves, which is
    what a large panel must do: resolving every request to build the table holds every
    configuration's prepared folds at once. Everything here is read out of the planned
    specification, so the table costs one JSON parse per configuration.

    It has no `eligible_entities` column, and that is the one difference. An entity count needs the
    eligibility keys themselves, which is exactly the memory the planning path avoids. The check
    that column existed for - a request that silently narrowed its universe - is carried by
    `eligible_rows` and the eligibility digest, which move whenever the universe does.
    """
    rows = []
    for spec_json in dict.fromkeys(member.spec_json for member in plan.members):
        spec = json.loads(spec_json)
        computation = spec.get("computation", spec)
        expected = computation["expected_prediction_keys"]
        folds = computation["cv"]["folds"]
        rows.append(
            {
                "family": spec["family"],
                "label": spec["label"],
                "config_name": spec.get("config_name"),
                "task": (computation.get("task") or {}).get("type", "regression"),
                "feature_count": len(computation.get("feature_names") or []),
                "eligible_rows": expected["n_rows"],
                "folds": expected["n_folds"],
                "validation_start": min(fold["val_start"] for fold in folds),
                "validation_end": max(fold["val_end"] for fold in folds),
                "checkpoints": len(computation["checkpoint_schedule"]),
                "execution_tier": spec["execution_tier"],
                "training_hash": training_hash_from_spec(spec),
            }
        )
    return pl.DataFrame(rows).sort("label", "family", "config_name")


def resolved_model_plan(resolved_requests: Iterable[ResolvedModelRequest]) -> pl.DataFrame:
    """Show what each resolved request will actually compute, before it computes it.

    The declared catalog says which models were asked for; this says which data, folds,
    checkpoints and validation window each one resolved to. A request that silently narrowed its
    universe or its folds is visible here and nowhere else.
    """
    rows = []
    for request in resolved_requests:
        computation = request.spec.get("computation", request.spec)
        expected = request._context.expected_keys
        entity = next(
            (column for column in ("symbol", "product") if column in expected.columns), None
        )
        fold = next((column for column in ("fold", "fold_id") if column in expected.columns), None)
        if entity is None or fold is None:
            raise ValueError("resolved model eligibility has no entity or fold key")
        timestamps = expected.get_column("timestamp")
        rows.append(
            {
                "family": request.family,
                "label": request.spec["label"],
                "config_name": request.spec.get("config_name"),
                "task": (computation.get("task") or {}).get("type", "regression"),
                "feature_count": len(computation.get("feature_names") or []),
                "eligible_entities": expected.get_column(entity).n_unique(),
                "eligible_rows": expected.height,
                "folds": expected.get_column(fold).n_unique(),
                "validation_start": timestamps.min(),
                "validation_end": timestamps.max(),
                "checkpoints": len(computation["checkpoint_schedule"]),
                "execution_tier": request.spec["execution_tier"],
                "training_hash": request.identity,
            }
        )
    return pl.DataFrame(rows).sort("label", "family", "config_name")
