"""Reader-facing workflow helpers for the crypto perpetual-funding case study."""

from __future__ import annotations

import hashlib
import inspect
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from case_studies.research import (
    DecisionArtifact,
    StateTransitionPolicy,
    Study,
    run_models,
)
from case_studies.research.execution import ModelExecution
from case_studies.utils.registry import prediction_hash_from_parts
from utils.modeling import load_configs
from utils.paths import REPO_ROOT

CASE_STUDY = "crypto_perps_funding"
ALL_LABELS = ("fwd_ret_8h", "fwd_ret_24h", "fwd_dir_8h", "fwd_dir_8h_3c")
REGRESSION_LABELS = ("fwd_ret_8h", "fwd_ret_24h")


def open_study(*, execution_tier: str, workspace: str | Path | None = None) -> Study:
    """Open canonical regeneration or an isolated reader preview."""
    if execution_tier == "canonical":
        return Study.regenerate(CASE_STUDY, release_root=REPO_ROOT)
    if execution_tier != "preview":
        raise ValueError("execution_tier must be canonical or preview")
    if workspace is None:
        raise ValueError("preview execution requires an explicit workspace")
    workspace = Path(workspace).expanduser().resolve()
    try:
        return Study.open(CASE_STUDY, workspace=workspace, release_root=REPO_ROOT)
    except ValueError as error:
        generated = tuple(
            REPO_ROOT / "case_studies" / CASE_STUDY / name
            for name in ("features", "labels", "run_log")
        )
        if "artifact bundle" not in str(error) or not all(path.is_symlink() for path in generated):
            raise
        workspace.mkdir(parents=True, exist_ok=True)
        shared_config = workspace / "config"
        if not shared_config.exists():
            shared_config.symlink_to(
                REPO_ROOT / "case_studies" / "config", target_is_directory=True
            )
        return Study(
            case_study=CASE_STUDY,
            root=REPO_ROOT / "case_studies" / CASE_STUDY,
            release_root=REPO_ROOT,
            output_root=workspace,
            read_only=False,
            manifest={
                "schema_version": 1,
                "case_study": CASE_STUDY,
                "baseline_source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
                "preview_only": True,
            },
        )


def model_request_catalog(
    family: str,
    *,
    labels: Iterable[str] = ALL_LABELS,
    config_prefix: str | None = None,
) -> pl.DataFrame:
    """Return the declared label/config population as a Polars catalog."""
    rows = []
    for label in labels:
        for config in load_configs(CASE_STUDY, label, family):
            name = str(config["config_name"])
            if config_prefix is None or name.startswith(config_prefix):
                rows.append(
                    {
                        "family": family,
                        "label": label,
                        "config_name": name,
                    }
                )
    if not rows:
        raise ValueError(f"no declared requests for {family!r}")
    return pl.DataFrame(rows).unique(maintain_order=True)


def resolve_model_requests(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
):
    """Resolve every visible catalog row through the shared family boundary."""
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
        ).resolve()
        for row in request_catalog.select(*sorted(required)).iter_rows(named=True)
    )


def run_model_catalog(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> ModelExecution:
    """Resolve and execute a complete declared population."""
    resolved = resolve_model_requests(
        study,
        request_catalog,
        execution_tier=execution_tier,
        overrides=overrides,
        preview_reductions=preview_reductions,
    )
    execution = run_models(study, requests=resolved)
    if execution.catalog_rows.height != sum(len(run.predictions) for run in execution.runs):
        raise RuntimeError("model execution did not return every checkpoint catalog row")
    if execution.catalog_rows.filter(~pl.col("complete")).height:
        raise RuntimeError("model execution returned incomplete prediction rows")
    return execution


def expected_prediction_hashes(resolved_requests) -> tuple[str, ...]:
    """Project every declared checkpoint to its immutable validation prediction identity."""
    hashes = []
    for request in resolved_requests:
        computation = request.spec.get("computation", request.spec)
        checkpoints = computation["checkpoint_schedule"]
        for checkpoint in checkpoints:
            hashes.append(
                prediction_hash_from_parts(
                    request.identity,
                    checkpoint["value"],
                    "validation",
                    checkpoint_kind=checkpoint["kind"],
                    identity_version=request.spec["identity_version"],
                )
            )
    if len(hashes) != len(set(hashes)):
        raise ValueError("declared request population contains duplicate prediction identities")
    return tuple(hashes)


def target_positions(
    predictions: pl.DataFrame,
    *,
    long_count: int = 1,
    short_count: int = 1,
) -> pl.DataFrame:
    """Map scores to deterministic long/short target positions at each decision time."""
    if long_count < 1 or short_count < 1:
        raise ValueError("long_count and short_count must be positive")
    ranked = predictions.with_columns(
        pl.col("prediction").rank("ordinal").over("timestamp").alias("ascending_rank"),
        pl.col("prediction")
        .rank("ordinal", descending=True)
        .over("timestamp")
        .alias("descending_rank"),
    )
    fold_columns = [column for column in ("fold", "fold_id") if column in ranked.columns]
    if len(fold_columns) > 1:
        raise ValueError("predictions cannot contain both fold and fold_id")
    selected_fold = fold_columns[:1]
    result = (
        ranked.with_columns(
            pl.when(pl.col("descending_rank") <= long_count)
            .then(1.0)
            .when(pl.col("ascending_rank") <= short_count)
            .then(-1.0)
            .otherwise(0.0)
            .alias("position")
        )
        .select("symbol", "timestamp", "position", *selected_fold)
        .sort("timestamp", "symbol")
    )
    return result.rename({"fold_id": "fold"}) if selected_fold == ["fold_id"] else result


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
