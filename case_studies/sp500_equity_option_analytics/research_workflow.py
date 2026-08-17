"""Reader-facing model workflow for S&P 500 equity-option analytics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from case_studies.research import (
    ExecutionTier,
    ModelExecution,
    OfficialPopulation,
    ResolvedModelRequest,
    Study,
    run_models,
)
from case_studies.research.adapters import registered_adapters
from case_studies.utils.registry import prediction_hash_from_parts
from utils.modeling import load_configs
from utils.paths import REPO_ROOT

CASE_STUDY = "sp500_equity_option_analytics"


def published_labels() -> tuple[str, ...]:
    """Return the primary label followed by the configured variants."""
    setup = yaml.safe_load(
        (REPO_ROOT / "case_studies" / CASE_STUDY / "config" / "setup.yaml").read_text()
    )
    labels = setup["labels"]
    return (str(labels["primary"]), *(str(label) for label in labels.get("variants", [])))


def open_study(*, execution_tier: str, workspace: str | Path | None = None) -> Study:
    """Open canonical regeneration or an isolated preview workspace."""
    tier = ExecutionTier(execution_tier)
    if tier is ExecutionTier.CANONICAL:
        return Study.regenerate(CASE_STUDY, release_root=REPO_ROOT)
    if workspace is None:
        raise ValueError("preview execution requires an explicit workspace")
    return Study.open(
        CASE_STUDY,
        workspace=Path(workspace).expanduser().resolve(),
        release_root=REPO_ROOT,
    )


def model_request_catalog(
    family: str,
    *,
    labels: Iterable[str] | None = None,
    config_names: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Return every configured label and model requested by one notebook.

    The two selection arguments treat an empty collection differently, on purpose.
    This case study's six model notebooks leave `LABELS = []` in the parameters cell
    to mean every published label, and each converts it with `LABELS or None` before
    calling - so `labels=None` is what actually arrives, and widening an empty list
    here is for consistency with that convention rather than something a notebook
    depends on. ``config_names`` is only ever a literal in notebook code, never a
    reader parameter, so an empty list there cannot be that convention and is
    refused instead of silently widened to the whole menu.

    The convention is this case study's, not the corpus's: `cme_futures` and
    `crypto_perps_funding` are the only siblings with this function and both declare
    `labels: Iterable[str] = ALL_LABELS` with no empty-means-all behaviour.
    """
    selected_labels = tuple(labels or published_labels())
    unknown = sorted(set(selected_labels) - set(published_labels()))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    selected_names = set(config_names) if config_names is not None else None
    if selected_names is not None and not selected_names:
        raise ValueError(
            "config_names is empty; omit it to request every declared configuration. "
            "An empty labels list does mean every published label - see the docstring."
        )
    rows = []
    declared_names: set[str] = set()
    declaring_labels = []
    case_dir = REPO_ROOT / "case_studies" / CASE_STUDY
    for label in selected_labels:
        menu = yaml.safe_load((case_dir / "config" / "training" / f"{label}.yaml").read_text())
        if not menu.get(family):
            continue
        declaring_labels.append(label)
        for config in load_configs(CASE_STUDY, label, family):
            name = str(config["config_name"])
            declared_names.add(name)
            if selected_names is None or name in selected_names:
                rows.append({"family": family, "label": label, "config_name": name})
    # Two reachable ways this comes back empty, and each names its own cause: no
    # selected label declares the family, or some do and the caller named a
    # configuration none of them declares. Reporting either as the other sends a
    # reader to the wrong place - the first version of this function blamed the
    # family for a misnamed configuration, and the fix for that blamed the
    # configuration for a label declaring no such family. A label whose menu lists
    # the family is skipped rather than refused when it lists a different family,
    # which is what lets the latent-factor notebooks pass every published label.
    if not declaring_labels:
        raise ValueError(f"no declared requests for {family!r}")
    if selected_names is not None:
        missing = sorted(selected_names - declared_names)
        if missing:
            raise ValueError(
                f"requested configurations are not declared for {family!r} on "
                f"{declaring_labels}: {missing}"
            )
    if not rows:
        # Unreachable while load_configs raises on an empty menu entry rather than
        # returning one. Kept as a consistency check, with its own cause named.
        raise ValueError(
            f"{family!r} is declared for {declaring_labels} but resolved no configurations"
        )
    return pl.DataFrame(rows).unique(maintain_order=True)


def configured_model_menu() -> pl.DataFrame:
    """Return every predictive model the published YAML menus declare, across every label.

    A family is predictive when a shared model adapter answers to its name, so
    `causal_dml` is absent by that rule rather than by a list that could go stale.
    """
    predictive = {binding.name for binding in registered_adapters("model")}
    case_dir = REPO_ROOT / "case_studies" / CASE_STUDY
    rows = []
    for label in published_labels():
        menu = yaml.safe_load((case_dir / "config" / "training" / f"{label}.yaml").read_text())
        for family in menu:
            if family not in predictive:
                continue
            for config in load_configs(CASE_STUDY, label, family):
                rows.append(
                    {"family": family, "label": label, "config_name": str(config["config_name"])}
                )
    if not rows:
        raise ValueError("no predictive model is declared for any published label")
    return pl.DataFrame(rows).unique(maintain_order=True)


def require_declared_menu_coverage(
    catalog: pl.DataFrame,
    *,
    unfitted: dict[tuple[str, str], str],
) -> pl.DataFrame:
    """Fail unless the population covers every declared model on every label that declares it.

    Comparing counts passes a population of the right size built on the wrong
    labels, so this compares `(family, label, config_name)` and names what is
    absent. `unfitted` maps a `(family, config_name)` the case study declares but
    no notebook fits to the reason; it applies to every label declaring that
    model rather than enumerating the triples, and an entry that excludes nothing
    is itself an error - a stale exclusion hides the next real gap.
    """
    identity = ["family", "label", "config_name"]
    declared = configured_model_menu()
    keys = pl.DataFrame(
        {
            "family": [family for family, _ in unfitted],
            "config_name": [config for _, config in unfitted],
            "reason": list(unfitted.values()),
        },
        schema={"family": pl.String, "config_name": pl.String, "reason": pl.String},
    )
    excluded = declared.join(keys, on=["family", "config_name"], how="inner")
    stale = sorted(set(unfitted) - set(excluded.select("family", "config_name").iter_rows()))
    if stale:
        raise ValueError(f"declared-but-unfitted entries match no configured model: {stale}")

    declared_rows = set(declared.select(identity).iter_rows())
    excluded_rows = set(excluded.select(identity).iter_rows())
    produced = set(catalog.select(identity).unique().iter_rows())
    missing = sorted(declared_rows - produced - excluded_rows)
    if missing:
        raise RuntimeError(
            f"the official population omits declared models that nothing excludes: {missing}"
        )
    undeclared = sorted(produced - declared_rows)
    if undeclared:
        raise RuntimeError(f"the official population holds models no menu declares: {undeclared}")
    return excluded.sort(identity)


def resolve_model_requests(
    study: Study,
    request_catalog: pl.DataFrame,
    *,
    execution_tier: str,
    overrides: dict[str, Any] | None = None,
    preview_reductions: dict[str, Any] | None = None,
) -> tuple[ResolvedModelRequest, ...]:
    """Resolve visible catalog rows through the shared family boundary."""
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
        for row in request_catalog.select("family", "label", "config_name").iter_rows(named=True)
    )


def require_complete_canonical_requests(
    request_catalog: pl.DataFrame,
    *,
    family: str,
    execution_tier: str,
    config_names: Iterable[str] | None = None,
) -> None:
    """Reject a partial declared request surface from canonical execution."""
    if ExecutionTier(execution_tier) is ExecutionTier.PREVIEW:
        return
    complete = model_request_catalog(family, config_names=config_names)
    identity = ["family", "label", "config_name"]
    requested_rows = set(request_catalog.select(identity).iter_rows())
    complete_rows = set(complete.select(identity).iter_rows())
    if requested_rows != complete_rows:
        missing = sorted(complete_rows - requested_rows)
        extra = sorted(requested_rows - complete_rows)
        raise ValueError(
            "canonical execution requires the complete declared request surface: "
            f"missing={missing}, extra={extra}"
        )


def expected_prediction_hashes(
    resolved_requests: Iterable[ResolvedModelRequest],
) -> tuple[str, ...]:
    """Project the resolved checkpoint population to immutable prediction identities."""
    hashes = []
    for request in resolved_requests:
        computation = request.spec.get("computation", request.spec)
        for checkpoint in computation["checkpoint_schedule"]:
            hashes.append(
                prediction_hash_from_parts(
                    request.identity,
                    checkpoint["value"],
                    "validation",
                    checkpoint_kind=checkpoint["kind"],
                    identity_version=request.spec["identity_version"],
                )
            )
    if not hashes or len(hashes) != len(set(hashes)):
        raise ValueError("resolved requests produced an empty or duplicate prediction population")
    return tuple(hashes)


def resolved_model_plan(requests: Iterable[ResolvedModelRequest]) -> pl.DataFrame:
    """Show the data, folds, checkpoints, and eligibility of each resolved request."""
    rows = []
    for request in requests:
        computation = request.spec.get("computation", request.spec)
        expected = request._context.expected_keys
        fold = next((column for column in ("fold", "fold_id") if column in expected.columns), None)
        if fold is None or "symbol" not in expected.columns:
            raise ValueError("resolved prediction eligibility lacks symbol or fold identity")
        timestamps = expected.get_column("timestamp")
        rows.append(
            {
                "family": request.family,
                "label": request.spec["label"],
                "config_name": request.spec["config_name"],
                "features": len(computation.get("feature_names") or []),
                "eligible_symbols": expected.get_column("symbol").n_unique(),
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


def run_model_population(
    study: Study,
    requests: Iterable[ResolvedModelRequest],
    *,
    population_name: str,
) -> tuple[ModelExecution, OfficialPopulation | None]:
    """Snapshot a canonical population before fitting, then require exact completion."""
    resolved = tuple(requests)
    if not resolved:
        raise ValueError("model population requires at least one resolved request")
    tiers = {ExecutionTier(request.spec["execution_tier"]) for request in resolved}
    if len(tiers) != 1:
        raise ValueError("one model population cannot mix canonical and preview requests")
    tier = tiers.pop()
    expected = expected_prediction_hashes(resolved)
    population = None
    if tier is ExecutionTier.CANONICAL:
        population = OfficialPopulation.create(
            study,
            name=population_name,
            member_kind="prediction",
            members=expected,
        )

    execution = run_models(study, requests=resolved)
    actual = tuple(prediction.hash for run in execution.runs for prediction in run.predictions)
    if len(actual) != len(expected) or set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"model population mismatch: missing={missing}, extra={extra}")
    if execution.catalog_rows.height != len(expected):
        raise RuntimeError("model execution did not return every checkpoint catalog row")
    if execution.catalog_rows.filter(~pl.col("complete")).height:
        raise RuntimeError("model execution returned incomplete prediction rows")
    if population is not None:
        population.require_complete()
    return execution, population
