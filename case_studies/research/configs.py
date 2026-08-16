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

from collections.abc import Iterable

import polars as pl
import yaml

from utils.modeling import load_configs
from utils.paths import get_case_study_dir

from .contracts import ExecutionTier
from .models import ModelRequest, ResolvedModelRequest
from .workspace import Study

CATALOG_COLUMNS = ("family", "label", "config_name", "model_class", "params")
REQUEST_COLUMNS = ("family", "label", "config_name")


def _format_params(params: dict | None) -> str:
    if not params:
        return "defaults"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def declared_labels(study: Study, family: str) -> tuple[str, ...]:
    """Every label whose training menu declares ``family``, in menu-file order."""
    menu_dir = get_case_study_dir(study.case_study) / "config" / "training"
    if not menu_dir.is_dir():
        raise FileNotFoundError(f"{study.case_study} has no training menus: {menu_dir}")
    labels = []
    for path in sorted(menu_dir.glob("*.yaml")):
        menu = yaml.safe_load(path.read_text()) or {}
        if menu.get(family):
            labels.append(path.stem)
    if not labels:
        raise ValueError(f"no training menu in {menu_dir} declares {family!r}")
    return tuple(labels)


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
