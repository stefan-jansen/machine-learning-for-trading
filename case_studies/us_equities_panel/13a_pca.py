# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Principal Components - US Equities Panel
#
# This notebook generates walk-forward validation predictions from return-panel principal
# components. Readers choose the labels, factor count, parameter overrides, and execution tier.
# Shared latent-factor code owns fold preparation, train-only fitting, fitted-state persistence,
# prediction reconstruction, exact coverage, metrics, and registry writes.
#
# The implementation is in `case_studies/utils/latent_factors/pca.py` and its shared research
# adapter. Readers can modify the factor construction in ordinary Python while preserving the same
# request, result, and catalog boundary.
#
# **Learning objectives**
#
# - Configure label-specific PCA requests with train-only fold fitting.
# - Trace fitted components, checkpoint identities, and reconstructed predictions.
# - Validate exact coverage before publishing compatible label sets.
#
# **Book reference**: Chapter 13
#
# **Prerequisites**: `05_evaluation.py` and the finalized label and feature artifacts.

# %%
"""Generate PCA validation predictions through the shared research interface."""

import os
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import Study, plan_models
from utils.modeling import load_configs
from utils.paths import REPO_ROOT, get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
LABELS = []
N_FACTORS = 5
OVERRIDES = {}
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
FOLD_IDS = []
PREVIEW_N_FACTORS = 0

# %% [markdown]
# ## Configure the experiment
#
# `LABELS = []` runs the primary label and every configured variant. Set it to a visible subset for
# a targeted experiment. `N_FACTORS` and `OVERRIDES` are resolved into the complete model
# specification. Canonical execution uses the complete panel and fold protocol. Reduced checks use
# the preview tier, declare every reduction below, and remain outside official result populations.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
published_labels = [setup["labels"]["primary"], *setup["labels"].get("variants", [])]
selected_labels = list(LABELS) if LABELS else published_labels
unknown_labels = sorted(set(selected_labels) - set(published_labels))
if unknown_labels:
    raise ValueError(f"Unknown labels: {unknown_labels}")
if len(selected_labels) != len(set(selected_labels)):
    raise ValueError("LABELS contains duplicates")

for label in selected_labels:
    configured = {
        config["config_name"]
        for config in load_configs(CASE_STUDY_ID, label, family="latent_factors")
    }
    if "pca" not in configured:
        raise ValueError(f"PCA is not configured for {label}")

label_menu = pl.DataFrame(
    {
        "label": published_labels,
        "selected": [label in selected_labels for label in published_labels],
        "n_factors": [N_FACTORS] * len(published_labels),
    }
)
label_menu

# %%
preview_reductions = {}
if MAX_SYMBOLS:
    preview_reductions["max_symbols"] = int(MAX_SYMBOLS)
if FOLD_IDS:
    preview_reductions["folds"] = [int(fold) for fold in FOLD_IDS]
if PREVIEW_N_FACTORS:
    preview_reductions["n_factors"] = int(PREVIEW_N_FACTORS)

if EXECUTION_TIER == "canonical":
    if preview_reductions:
        raise ValueError("Canonical execution cannot declare preview reductions")
    study = Study.regenerate(CASE_STUDY_ID, release_root=REPO_ROOT)
elif EXECUTION_TIER == "preview":
    if not preview_reductions:
        raise ValueError("Preview execution requires at least one declared reduction")
    study = Study.open(
        CASE_STUDY_ID,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
        release_root=REPO_ROOT,
    )
else:
    raise ValueError("EXECUTION_TIER must be 'canonical' or 'preview'")

# %% [markdown]
# ## Build the label requests
#
# Each selected label receives a separate PCA request under the shared factor-count specification.

# %%
requests = tuple(
    study.model(
        family="latent_factors",
        label=label,
        config_name="pca",
        overrides={"n_factors": int(N_FACTORS), **OVERRIDES},
        execution_tier=EXECUTION_TIER,
        preview_reductions=preview_reductions,
    )
    for label in selected_labels
)

request_table = pl.DataFrame(
    {
        "family": [request.family for request in requests],
        "label": [request.label for request in requests],
        "config_name": [request.config_name for request in requests],
        "overrides": [str(request.overrides) for request in requests],
        "execution_tier": [request.execution_tier.value for request in requests],
        "preview_reductions": [str(request.preview_reductions) for request in requests],
    }
)
request_table

# %% [markdown]
# ## Plan and execute the selected labels
#
# The planner resolves every label-specific training and checkpoint identity before fitting and
# writes the canonical checkpoint population first. Each fold then fits PCA on its training return
# panel only. The runner persists the fitted components
# and reconstructs each registered prediction set from those artifacts before accepting cached
# work.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-pca-checkpoints-v1",
    )

planned_population = pl.DataFrame(
    {
        "label": [member.label for member in plan.members],
        "config_name": [member.config_name for member in plan.members],
        "checkpoint_kind": [member.checkpoint_kind for member in plan.members],
        "checkpoint_value": [member.checkpoint_value for member in plan.members],
        "training_hash": [member.training_hash for member in plan.members],
        "prediction_hash": [member.prediction_hash for member in plan.members],
    }
)
planned_population

# %%
execution = plan.run()

# %% [markdown]
# ## Inspect the resolved computation

# %%
resolved_rows = []
for run in execution.runs:
    spec = run.training.spec()
    computation = spec["computation"]
    resolved_rows.append(
        {
            "label": spec["label"],
            "features": len(computation["feature_names"]),
            "folds": computation["expected_prediction_keys"]["n_folds"],
            "eligible_rows": computation["expected_prediction_keys"]["n_rows"],
            "n_factors": computation["model"]["n_factors"],
            "device": computation["runtime"]["device"],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("label")
resolved_table

# %% [markdown]
# ## Validate and inspect the handoff
#
# Each catalog row is a complete validation prediction set with exact training, label, fold, and
# fitted-state lineage. Downstream notebooks select these rows with Polars rather than copying
# hashes, while the hashes remain visible for exact provenance.

# %% tags=["results"]
catalog_columns = [
    "family",
    "config_name",
    "label",
    "split",
    "checkpoint_kind",
    "checkpoint_value",
    "execution_tier",
    "complete",
    "ic_mean",
    "training_hash",
    "prediction_hash",
]
catalog_rows = execution.catalog_rows.select(
    column for column in catalog_columns if column in execution.catalog_rows.columns
).sort("label", "checkpoint_value", "prediction_hash")
catalog_rows

# %%
coverage_rows = []
for run in execution.runs:
    if not run.training.complete:
        raise RuntimeError(f"Incomplete training result: {run.training.hash}")
    for prediction in run.predictions:
        coverage = prediction.coverage()
        if not prediction.complete or coverage is None or coverage["status"] != "complete":
            raise RuntimeError(f"Incomplete prediction result: {prediction.hash}")
        coverage_rows.append(
            {
                "label": run.training.spec()["label"],
                "training_hash": run.training.hash,
                "prediction_hash": prediction.hash,
                "coverage_status": coverage["status"],
                "expected_rows": coverage["n_expected"],
                "actual_rows": coverage["n_actual"],
                "training_artifacts": len(run.training.artifacts()),
                "prediction_artifacts": len(prediction.artifacts()),
            }
        )

coverage_table = pl.DataFrame(coverage_rows).sort("label", "prediction_hash")
if official_population is not None:
    official_population.require_complete()
coverage_table

# %%
execution_diagnostics = pl.DataFrame(execution.diagnostics)
execution_diagnostics

# %% [markdown]
# ## Freeze the compatible result sets
#
# A canonical default run freezes one complete PCA result set for each configured label. Each set
# is also small enough for raw diagnostic comparisons. Preview and customized canonical requests
# do not publish official sets.

# %% tags=["results"]
set_rows = []
is_published_population = (
    EXECUTION_TIER == "canonical"
    and selected_labels == published_labels
    and N_FACTORS == 5
    and not OVERRIDES
)
if is_published_population:
    for selected_label in selected_labels:
        label_name = selected_label.replace("_", "-")
        result_set = study.predictions.freeze(
            execution.catalog_rows.filter(pl.col("label") == selected_label),
            name=f"us-equities-{label_name}-pca-v1",
        )
        set_rows.append(
            {
                "role": "backtest and diagnostic population",
                "set_name": result_set.name,
                "members": len(result_set.members),
            }
        )
compatible_sets = pl.DataFrame(
    set_rows,
    schema={"role": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `15_model_analysis.py` reopens the named label sets for descriptive analysis. `16_backtest.py`
# passes every catalog row directly to the shared backtest runner. Predictive metrics do not choose
# a configuration or checkpoint.

# %% [markdown]
# ## Key takeaways and limitations
#
# - PCA is fitted separately inside each training fold and then applied to that fold's validation
#   observations.
# - Fitted components and reconstructed predictions are digest-validated before reuse.
# - The factor model represents linear directions of variation in the training return panel; it
#   does not identify a structural economic factor.
# - Label-specific sets preserve compatible comparisons for analysis and backtesting.
