"""Reader-facing research workflow for US firm characteristics."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from case_studies.research import ModelRequest, OfficialPopulation, ResolvedModelRequest, Study
from case_studies.utils.registry import prediction_hash_from_parts
from utils.modeling import load_configs
from utils.paths import REPO_ROOT

CASE_STUDY = "us_firm_characteristics"
PREVIEW_DIR_NAME = ".preview"
PREDICTIVE_FAMILIES = ("linear", "gbm", "tabular_dl", "latent_factors")
# Runtime a model needs and its family's declaration does not give it. `setup.yaml` puts the
# latent-factor family on CUDA, which is right for the three neural members; IPCA is alternating
# least squares over the panel and has no GPU implementation, so it runs on bounded CPU workers.
# Both keys are inside the hashed computation rather than provenance beside it, so this decides
# what the result is. `08a_ipca` reads the same mapping - one home for the declaration.
MODEL_RUNTIME_OVERRIDES = {
    ("latent_factors", "ipca"): {"device": "cpu", "fold_workers": 4},
}


def open_study(*, execution_tier: str, workspace: str | Path | None = None) -> Study:
    """Open a writable canonical or preview workspace without changing the release.

    A preview workspace is activated here rather than left to the individual requests, so
    that everything the session writes lands under the isolated preview root instead of
    only the results that happen to carry the tier.
    """
    if execution_tier not in {"canonical", "preview"}:
        raise ValueError("execution_tier must be canonical or preview")
    output_root = os.environ.get("ML4T_OUTPUT_DIR") or workspace
    if output_root is None:
        raise ValueError("execution requires an explicit writable workspace")
    resolved_root = Path(output_root).expanduser()
    if not resolved_root.is_absolute():
        resolved_root = REPO_ROOT / resolved_root
    # Activating a preview rewrites ML4T_OUTPUT_DIR to the preview root, so a second call
    # in the same kernel would otherwise nest one preview root inside another.
    if resolved_root.name == PREVIEW_DIR_NAME:
        resolved_root = resolved_root.parent
    study = Study.open(CASE_STUDY, workspace=resolved_root)
    study.activate(execution_tier)
    return study


def declared_labels() -> tuple[str, ...]:
    """Return the complete label set in its published order."""
    setup_path = REPO_ROOT / "case_studies" / CASE_STUDY / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())
    labels = setup["labels"]
    return (str(labels["primary"]), *(str(label) for label in labels.get("variants", [])))


def model_request_catalog(
    family: str,
    *,
    labels: Iterable[str] | None = None,
    config_names: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return the declared model population as a Polars request catalog."""
    selected_names = set(config_names) if config_names is not None else None
    rows = []
    for label in declared_labels() if labels is None else tuple(labels):
        for config in load_configs(CASE_STUDY, label, family):
            name = str(config["config_name"])
            if selected_names is None or name in selected_names:
                rows.append({"family": family, "label": label, "config_name": name})
    if not rows:
        raise ValueError(f"no declared requests for {family!r}")
    return pl.DataFrame(rows).unique(maintain_order=True)


def causal_estimand_labels(study: Study, labels: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the declared labels whose target the DML estimator can actually fit.

    The nuisance and outcome models are regressors, so a classification label produces a
    number that resolves without meaning anything. The split is derived from each label's
    own declared task rather than from a list, so adding a label cannot silently widen or
    narrow the estimand set.
    """
    requested = tuple(labels) if labels is not None else declared_labels()
    tasks = {label: study.labels.get(label).definition.task_type for label in requested}
    continuous = tuple(label for label, task in tasks.items() if task == "regression")
    excluded = tuple(label for label, task in tasks.items() if task != "regression")
    if not continuous:
        raise ValueError(f"no continuous label among {requested}; DML has no estimand to fit")
    if excluded:
        print(
            "Excluded from causal estimation, the target is not continuous: "
            + ", ".join(f"{label} ({tasks[label]})" for label in excluded)
        )
    return continuous


def causal_request_catalog(
    study: Study,
    *,
    labels: Iterable[str] | None = None,
    config_names: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return the declared DML estimands as a Polars request catalog."""
    selected_names = set(config_names) if config_names is not None else None
    rows = []
    for label in causal_estimand_labels(study, labels):
        for config in load_configs(CASE_STUDY, label, "causal_dml"):
            name = str(config["config_name"])
            if selected_names is None or name in selected_names:
                rows.append({"method": "dml", "label": label, "config_name": name})
    if not rows:
        raise ValueError("no declared causal requests")
    return pl.DataFrame(rows).unique(maintain_order=True)


def causal_requests(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
):
    """Construct visible causal requests for every declared estimand."""
    required = {"method", "label", "config_name"}
    missing = required - set(request_catalog.columns)
    if missing:
        raise ValueError(f"causal request catalog is missing {sorted(missing)}")
    return tuple(
        study.causal(
            **row,
            execution_tier=execution_tier,
            overrides=dict(overrides or {}),
            preview_reductions=dict(preview_reductions or {}),
        )
        for row in request_catalog.select("method", "label", "config_name").iter_rows(named=True)
    )


def model_requests(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
):
    """Construct visible requests for every row in a declared catalog."""
    required = {"family", "label", "config_name"}
    missing = required - set(request_catalog.columns)
    if missing:
        raise ValueError(f"model request catalog is missing {sorted(missing)}")
    return tuple(
        study.model(
            **row,
            execution_tier=execution_tier,
            overrides={
                **MODEL_RUNTIME_OVERRIDES.get((row["family"], row["config_name"]), {}),
                **dict(overrides or {}),
            },
            preview_reductions=dict(preview_reductions or {}),
        )
        for row in request_catalog.select("family", "label", "config_name").iter_rows(named=True)
    )


def resolved_request_table(requests: Iterable[Any]) -> pl.DataFrame:
    """Resolve requests and expose their complete identity before execution."""
    rows = []
    for request in requests:
        resolved = request.resolve() if isinstance(request, ModelRequest) else request
        if not isinstance(resolved, ResolvedModelRequest):
            raise TypeError("expected model requests or resolved model requests")
        computation = resolved.spec["computation"]
        runtime = computation.get("runtime") or {}
        rows.append(
            {
                "family": resolved.family,
                "label": resolved.spec["label"],
                "config_name": resolved.spec["config_name"],
                "training_hash": resolved.identity,
                "checkpoints": len(computation["checkpoint_schedule"]),
                "device": runtime.get("device"),
                "fold_workers": runtime.get("fold_workers"),
                "execution_tier": resolved.spec["execution_tier"],
            }
        )
    return pl.DataFrame(rows)


def expected_prediction_hashes(requests: Iterable[Any]) -> tuple[str, ...]:
    """Project each request and checkpoint to its validation prediction identity."""
    hashes = []
    for request in requests:
        resolved = request.resolve() if isinstance(request, ModelRequest) else request
        if not isinstance(resolved, ResolvedModelRequest):
            raise TypeError("expected model requests or resolved model requests")
        for checkpoint in resolved.spec["computation"]["checkpoint_schedule"]:
            hashes.append(
                prediction_hash_from_parts(
                    resolved.identity,
                    checkpoint["value"],
                    "validation",
                    checkpoint_kind=checkpoint["kind"],
                    identity_version=resolved.spec["identity_version"],
                )
            )
    if len(hashes) != len(set(hashes)):
        raise ValueError("declared request population contains duplicate prediction identities")
    return tuple(hashes)


def snapshot_model_population(study: Study, *, name: str) -> OfficialPopulation:
    """Freeze the complete canonical prediction population before model execution."""
    requests = tuple(
        request
        for family in PREDICTIVE_FAMILIES
        for request in model_requests(
            study,
            model_request_catalog(family),
            execution_tier="canonical",
        )
    )
    return OfficialPopulation.create(
        study,
        name=name,
        member_kind="prediction",
        members=expected_prediction_hashes(requests),
    )


def completed_prediction_rows(study: Study, prediction_hashes: Iterable[str]) -> pl.DataFrame:
    """Return complete current rows for an executed request population."""
    hashes = tuple(prediction_hashes)
    rows = study.predictions.table(include_preview=True).filter(
        pl.col("prediction_hash").is_in(hashes)
    )
    if rows.height != len(hashes) or rows.filter(~pl.col("complete")).height:
        raise RuntimeError("execution did not publish every complete declared checkpoint")
    return rows.sort("family", "label", "config_name", "checkpoint_value")
