"""Reader-facing workflow helpers for the crypto perpetual-funding case study."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from case_studies.research import (
    DecisionArtifact,
    ModelPlan,
    OfficialPopulation,
    StateTransitionPolicy,
    Study,
    plan_models,
)
from case_studies.research.contracts import ExecutionTier
from case_studies.research.execution import ModelExecution
from case_studies.research.models import ModelRequest
from utils.modeling import load_configs
from utils.paths import REPO_ROOT

CASE_STUDY = "crypto_perps_funding"
ALL_LABELS = ("fwd_ret_8h", "fwd_ret_24h", "fwd_dir_8h", "fwd_dir_8h_3c")
REGRESSION_LABELS = ("fwd_ret_8h", "fwd_ret_24h")
OFFICIAL_POPULATION = "crypto-validation-predictions-v1"

CLASSIFICATION_LABELS = tuple(label for label in ALL_LABELS if label not in REGRESSION_LABELS)


def open_study(*, execution_tier: str, workspace: str | Path | None = None) -> Study:
    """Open canonical regeneration or an isolated reader preview."""
    if execution_tier == "canonical":
        if workspace is None:
            return Study.regenerate(CASE_STUDY, release_root=REPO_ROOT)
    elif execution_tier == "preview":
        workspace = workspace or os.environ.get("ML4T_OUTPUT_DIR")
        if workspace is None:
            raise ValueError("preview execution requires an explicit workspace")
    else:
        raise ValueError("execution_tier must be canonical or preview")
    return Study.open(
        CASE_STUDY,
        workspace=Path(workspace).expanduser().resolve(),
        release_root=REPO_ROOT,
    )


def model_request_catalog(
    family: str,
    *,
    labels: Iterable[str] = ALL_LABELS,
    config_prefix: str | tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Return the declared label/config population as a Polars catalog.

    A ``config_prefix`` that matches nothing raises. It used to be applied with a bare
    ``startswith``, so a prefix naming no declared configuration quietly produced a smaller
    population instead of an error - and only a population that came out *entirely* empty was
    caught. A prefix that missed for some labels and hit for others therefore fitted a subset and
    reported nothing, which is the strictness `load_model_configs` adopted for `config_names`.
    """
    prefixes = (
        (config_prefix,)
        if isinstance(config_prefix, str)
        else tuple(config_prefix)
        if config_prefix is not None
        else ()
    )
    matched: set[str] = set()
    rows = []
    for label in labels:
        for config in load_configs(CASE_STUDY, label, family):
            name = str(config["config_name"])
            hit = next((p for p in prefixes if name.startswith(p)), None)
            if config_prefix is not None and hit is None:
                continue
            if hit is not None:
                matched.add(hit)
            rows.append(
                {
                    "family": family,
                    "label": label,
                    "config_name": name,
                }
            )
    unmatched = sorted(set(prefixes) - matched)
    if unmatched:
        raise ValueError(
            f"{family!r} configuration prefixes match nothing declared by "
            f"labels {list(labels)}: {unmatched}"
        )
    if not rows:
        raise ValueError(f"no declared requests for {family!r}")
    return pl.DataFrame(rows).unique(maintain_order=True)


def model_requests(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> tuple[ModelRequest, ...]:
    """Build every visible catalog row into a shared, still-unresolved model request."""
    required = {"family", "label", "config_name"}
    missing = required - set(request_catalog.columns)
    if missing:
        raise ValueError(f"model request catalog is missing {sorted(missing)}")
    return tuple(
        study.model(
            **row,
            execution_tier=execution_tier,
            overrides=dict(overrides or {}),
            preview_reductions=dict(preview_reductions or {}),
        )
        for row in request_catalog.select(*sorted(required)).iter_rows(named=True)
    )


def declared_menu() -> set[tuple[str, str, str]]:
    """Every `(family, label, config_name)` the published training menus declare."""
    return {
        (family, label, str(config["config_name"]))
        for family in ("linear", "gbm", "tabular_dl", "deep_learning")
        for label in ALL_LABELS
        for config in load_configs(CASE_STUDY, label, family)
    }


def unsupported_requests() -> set[tuple[str, str, str]]:
    """Declared members the shared runners cannot execute as written.

    The training menus list sequence configs for the direction labels, but the sequence runner
    resolves a regression task unconditionally and raises "sequence runner currently supports
    regression labels only" (`case_studies/utils/deep_learning.py`). Deriving the exclusion from
    that rule rather than listing members keeps it true when a menu changes, and keeps the
    omission visible instead of hidden inside a label filter.
    """
    return {
        member
        for member in declared_menu()
        if member[0] == "deep_learning" and member[1] in CLASSIFICATION_LABELS
    }


def official_model_requests(study: Study) -> tuple[ModelRequest, ...]:
    """Build the complete canonical model population this case study declares."""
    declared = (
        (model_request_catalog("linear"), {}),
        (model_request_catalog("gbm"), {}),
        (
            model_request_catalog("tabular_dl", config_prefix="tabm"),
            {"class_weight": "balanced", "device": "cuda"},
        ),
        (
            model_request_catalog("deep_learning", labels=REGRESSION_LABELS),
            {"device": "cuda"},
        ),
    )
    requests = tuple(
        request
        for request_catalog, overrides in declared
        for request in model_requests(
            study,
            request_catalog,
            execution_tier="canonical",
            overrides=overrides,
        )
    )
    covered = {(request.family, request.label, request.config_name) for request in requests}
    omitted = declared_menu() - covered - unsupported_requests()
    if omitted:
        raise ValueError(
            f"the official crypto model population omits declared configurations: {sorted(omitted)}"
        )
    return requests


def plan_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> ModelPlan:
    """Resolve every training and checkpoint identity in the catalog without fitting."""
    return plan_models(
        study,
        requests=model_requests(
            study,
            request_catalog,
            execution_tier=execution_tier,
            overrides=overrides,
            preview_reductions=preview_reductions,
        ),
    )


def plan_official_models(study: Study) -> ModelPlan:
    """Resolve every canonical training and checkpoint identity without fitting."""
    return plan_models(study, requests=official_model_requests(study))


def plan_specs(plan: ModelPlan) -> tuple[dict[str, Any], ...]:
    """Return the resolved computation behind every declared checkpoint, in plan order."""
    return tuple(json.loads(member.spec_json) for member in plan.members)


def declared_contracts(plan: ModelPlan) -> pl.DataFrame:
    """Project a frozen plan to one visible row per declared checkpoint."""
    rows = []
    for member, spec in zip(plan.members, plan_specs(plan), strict=True):
        computation = spec.get("computation", spec)
        task = computation.get("task") or {}
        rows.append(
            {
                "family": member.family,
                "label": member.label,
                "config_name": member.config_name,
                "task": task.get("type"),
                "continuous_eval_label": task.get("continuous_eval_label"),
                "checkpoint_kind": member.checkpoint_kind,
                "checkpoint_value": member.checkpoint_value,
                "eligible_rows": computation["expected_prediction_keys"]["n_rows"],
                "training_hash": member.training_hash,
                "prediction_hash": member.prediction_hash,
            }
        )
    # Declared, not inferred. continuous_eval_label is null for regression rows and a label
    # name for classification ones, and the plan emits every regression config before any
    # classification config: gbm puts its first string at row 300, past the window polars
    # infers from, and the frame fails to build. linear and tabular_dl happen to place theirs
    # inside it, so they build -- on row order rather than on anything declared.
    return pl.DataFrame(
        rows,
        schema={
            "family": pl.String,
            "label": pl.String,
            "config_name": pl.String,
            "task": pl.String,
            "continuous_eval_label": pl.String,
            "checkpoint_kind": pl.String,
            # null for families that checkpoint once (linear), an iteration count for those
            # that checkpoint through training (gbm, tabular_dl).
            "checkpoint_value": pl.Int64,
            "eligible_rows": pl.Int64,
            "training_hash": pl.String,
            "prediction_hash": pl.String,
        },
    )


def freeze_official_model_population(study: Study) -> OfficialPopulation:
    """Record every canonical model checkpoint before the first fit starts."""
    return plan_official_models(study).create_population(name=OFFICIAL_POPULATION)


def run_model_plan(plan: ModelPlan, *, population_name: str | None = None) -> ModelExecution:
    """Freeze the planned checkpoint population, then execute exactly that population."""
    canonical = plan.execution_tier is ExecutionTier.CANONICAL
    population = None
    if canonical:
        if not population_name:
            raise ValueError("canonical model execution requires an official population name")
        population = plan.create_population(name=population_name)
    elif population_name is not None:
        raise ValueError("preview model execution cannot create an official population")
    execution = plan.run()
    if execution.catalog_rows.height != len(plan.expected_prediction_hashes):
        raise RuntimeError("model execution did not return every checkpoint catalog row")
    if execution.catalog_rows.filter(~pl.col("complete")).height:
        raise RuntimeError("model execution returned incomplete prediction rows")
    if population is not None:
        population.require_complete()
    return execution


def run_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    population_name: str | None = None,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> ModelExecution:
    """Plan a complete declared population, freeze it, then execute it."""
    return run_model_plan(
        plan_model_catalog(
            study,
            request_catalog,
            execution_tier=execution_tier,
            overrides=overrides,
            preview_reductions=preview_reductions,
        ),
        population_name=population_name,
    )


def target_positions(
    predictions: pl.DataFrame,
    *,
    long_count: int = 1,
    short_count: int = 1,
) -> pl.DataFrame:
    """Map scores to deterministic long/short target positions at each decision time."""
    if long_count < 1 or short_count < 1:
        raise ValueError("long_count and short_count must be positive")
    required = {"symbol", "timestamp", "fold", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame is missing canonical columns: {sorted(missing)}")
    ranked = predictions.with_columns(
        pl.col("prediction").rank("ordinal").over("timestamp").alias("ascending_rank"),
        pl.col("prediction")
        .rank("ordinal", descending=True)
        .over("timestamp")
        .alias("descending_rank"),
    )
    result = (
        ranked.with_columns(
            pl.when(pl.col("descending_rank") <= long_count)
            .then(1.0)
            .when(pl.col("ascending_rank") <= short_count)
            .then(-1.0)
            .otherwise(0.0)
            .alias("position")
        )
        .select("symbol", "timestamp", "position", "fold")
        .sort("timestamp", "symbol")
    )
    return result


def publish_exploratory_positions(
    study: Study,
    prediction_hash: str,
    predictions: pl.DataFrame,
    *,
    long_count: int = 1,
    short_count: int = 1,
    cadence: str = "8h",
) -> DecisionArtifact:
    """Publish an immediately backtestable, non-canonical Python decision."""
    return DecisionArtifact.publish(
        study,
        kind="target_positions",
        decisions=target_positions(
            predictions,
            long_count=long_count,
            short_count=short_count,
        ),
        prediction_hashes=[prediction_hash],
        parameters={"long_count": long_count, "short_count": short_count, "cadence": cadence},
        state_transition_policy=StateTransitionPolicy(
            fold_boundary="liquidate",
            temporal_gap="reset",
        ),
        canonical=False,
    )


def target_positions_source_digest() -> str:
    """Return the digest used when this decision generator is promoted after replay."""
    return hashlib.sha256(inspect.getsource(target_positions).encode()).hexdigest()
