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
# # TabM Models - US Equities Panel
#
# This notebook generates walk-forward validation predictions for the published TabM
# configurations and every declared epoch checkpoint. Readers choose the label, configurations,
# parameter overrides, and execution tier. Shared code owns panel preparation, fold preprocessing,
# fitting, checkpoint persistence, restart, prediction coverage, metrics, and registry writes.
#
# Compatible configurations run together so the large panel is prepared once per fold.
# Each configuration and checkpoint still has its own immutable identity. The implementation is in
# `case_studies/utils/tabular_dl.py` for readers who want to change the architecture or add another
# tabular model family.
#
# **Learning objectives**
#
# - Configure TabM candidates and epoch checkpoints through the shared request boundary.
# - Explain compatible batching, per-candidate persistence, and completed-fold restart.
# - Validate fitted-state artifacts, prediction coverage, and catalog identities.
#
# **Book reference**: Chapter 12, Section 12.3 (Deep Learning Alternatives)
#
# **Prerequisites**: `03_financial_features.py`, `04_model_based_features.py`, and
# `05_evaluation.py`.

# %%
"""Generate TabM validation predictions through the shared research interface."""

import os
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import Study, plan_models
from utils.modeling import load_configs
from utils.paths import REPO_ROOT, get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = ""
CONFIG_NAMES = []
COMMON_OVERRIDES = {}
CONFIG_OVERRIDES = {}
DIAGNOSTIC_CONFIG_NAMES = ["tabm_s"]
DEVICE = "cuda"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
MAX_FOLDS = 0
PREVIEW_N_EPOCHS = 0
PREVIEW_CHECKPOINT_INTERVAL = 0

# %% [markdown]
# ## Configure the experiment
#
# `CONFIG_NAMES = []` runs the complete published TabM menu. Set it to a subset such as
# `['tabm_s']` for a targeted experiment. `COMMON_OVERRIDES` changes validated TabM or runner
# parameters for every selected configuration. `CONFIG_OVERRIDES` adds changes for one named
# configuration and takes precedence. The resolved training specification records all published
# defaults and overrides. `DIAGNOSTIC_CONFIG_NAMES` declares the bounded subset used for raw
# prediction comparisons in the analysis notebook.
#
# Canonical requests use CUDA, the complete panel, every fold, and the published epoch schedule.
# Reduced checks must use the preview tier and declare every reduction below. Preview identities
# and artifacts are isolated from official comparisons and holdout decisions.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
label = PRIMARY_LABEL or setup["labels"]["primary"]

published_configs = load_configs(CASE_STUDY_ID, label, family="tabular_dl")
published_names = [str(config["config_name"]) for config in published_configs]
selected_names = list(CONFIG_NAMES) if CONFIG_NAMES else published_names
unknown_names = sorted(set(selected_names) - set(published_names))
unknown_overrides = sorted(set(CONFIG_OVERRIDES) - set(selected_names))
if unknown_names:
    raise ValueError(f"Unknown TabM configurations: {unknown_names}")
if unknown_overrides:
    raise ValueError(f"Overrides supplied for unselected configurations: {unknown_overrides}")
if len(selected_names) != len(set(selected_names)):
    raise ValueError("CONFIG_NAMES contains duplicates")
unknown_diagnostics = sorted(set(DIAGNOSTIC_CONFIG_NAMES) - set(selected_names))
if not DIAGNOSTIC_CONFIG_NAMES or unknown_diagnostics:
    raise ValueError(f"Invalid diagnostic configurations: {unknown_diagnostics}")

menu = pl.DataFrame(
    {
        "config_name": [config["config_name"] for config in published_configs],
        "library": [config["library"] for config in published_configs],
        "published_params": [str(config.get("params") or {}) for config in published_configs],
        "n_epochs": [config.get("n_epochs") for config in published_configs],
        "checkpoint_interval": [config.get("checkpoint_interval") for config in published_configs],
        "selected": [config["config_name"] in selected_names for config in published_configs],
    }
)
menu

# %%
preview_reductions = {}
if MAX_SYMBOLS:
    preview_reductions["max_symbols"] = int(MAX_SYMBOLS)
if MAX_FOLDS:
    preview_reductions["folds"] = list(range(int(MAX_FOLDS)))
if PREVIEW_N_EPOCHS:
    preview_reductions["n_epochs"] = int(PREVIEW_N_EPOCHS)
if PREVIEW_CHECKPOINT_INTERVAL:
    preview_reductions["checkpoint_interval"] = int(PREVIEW_CHECKPOINT_INTERVAL)

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
# ## Build the model requests
#
# Each selected configuration becomes one CUDA request after common and configuration-specific
# overrides are combined.

# %%
requests = []
for config_name in selected_names:
    overrides = {
        "device": DEVICE,
        **COMMON_OVERRIDES,
        **dict(CONFIG_OVERRIDES.get(config_name, {})),
    }
    requests.append(
        study.model(
            family="tabular_dl",
            label=label,
            config_name=config_name,
            overrides=overrides,
            execution_tier=EXECUTION_TIER,
            preview_reductions=preview_reductions,
        )
    )
requests = tuple(requests)

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
# ## Plan and execute the selected configurations
#
# The planner resolves every training and epoch-checkpoint identity before fitting and writes the
# canonical checkpoint population first. The plan then prepares one fold for all compatible
# resident candidates. Each
# declared epoch checkpoint stores fitted preprocessing, model weights, predictions, and coverage
# evidence before the fold is released. A retry validates completed candidate-fold checkpoints and
# recomputes only missing or corrupt work.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-tabular-dl-checkpoints-v1",
    )

planned_population = pl.DataFrame(
    {
        "family": [member.family for member in plan.members],
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
#
# The resolved specifications below contain the actual feature, label, fold, task, runtime, model,
# preprocessing, and checkpoint settings used by the runner. This includes defaults that were not
# repeated in the notebook parameters.

# %%
resolved_rows = []
for run in execution.runs:
    spec = run.training.spec()
    computation = spec["computation"]
    model = computation["model"]
    resolved_rows.append(
        {
            "config_name": spec["config_name"],
            "task": computation["task"]["type"],
            "features": len(computation["feature_names"]),
            "folds": computation["expected_prediction_keys"]["n_folds"],
            "device": computation["numerics"]["device"],
            "n_epochs": model["params"]["n_epochs"],
            "batch_size": model["params"]["batch_size"],
            "checkpoints": [item["value"] for item in computation["checkpoint_schedule"]],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("config_name")
resolved_table

# %% [markdown]
# ## Validate and inspect the handoff
#
# Each catalog row is one complete validation prediction set for one training identity and epoch.
# Downstream notebooks filter these rows with ordinary Polars expressions and pass the selected
# table directly to backtesting. The hashes remain visible for exact provenance and artifact reads.

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
).sort("config_name", "checkpoint_value", "prediction_hash")
catalog_rows

# %%
coverage_rows = []
for run in execution.runs:
    if not run.training.complete:
        raise RuntimeError(f"Incomplete training result: {run.training.hash}")
    for prediction in run.predictions:
        record = prediction.registry_record()
        coverage = prediction.coverage()
        if not prediction.complete or coverage is None or coverage["status"] != "complete":
            raise RuntimeError(f"Incomplete prediction result: {prediction.hash}")
        coverage_rows.append(
            {
                "config_name": run.training.spec()["config_name"],
                "checkpoint": record["checkpoint_value"],
                "training_hash": run.training.hash,
                "prediction_hash": prediction.hash,
                "coverage_status": coverage["status"],
                "expected_rows": coverage["n_expected"],
                "actual_rows": coverage["n_actual"],
                "training_artifacts": len(run.training.artifacts()),
                "prediction_artifacts": len(prediction.artifacts()),
            }
        )

coverage_table = pl.DataFrame(coverage_rows).sort("config_name", "checkpoint")
if official_population is not None:
    official_population.require_complete()
coverage_table

# %%
execution_diagnostics = pl.DataFrame(execution.diagnostics)
execution_diagnostics

# %% [markdown]
# ## Freeze the compatible result sets
#
# A canonical default CUDA run freezes every returned prediction row under a stable family/label
# name. The separately named diagnostic subset is bounded by the visible configuration list above.
# Preview and customized canonical requests retain their result rows without publishing an
# official set.

# %% tags=["results"]
set_rows = []
is_published_population = (
    EXECUTION_TIER == "canonical"
    and selected_names == published_names
    and not COMMON_OVERRIDES
    and not CONFIG_OVERRIDES
    and DEVICE == "cuda"
)
if is_published_population:
    label_name = label.replace("_", "-")
    full_set = study.predictions.freeze(
        execution.catalog_rows,
        name=f"us-equities-{label_name}-tabular-dl-v1",
    )
    diagnostic_set = study.predictions.freeze(
        execution.catalog_rows.filter(pl.col("config_name").is_in(DIAGNOSTIC_CONFIG_NAMES)),
        name=f"us-equities-{label_name}-tabular-dl-diagnostics-v1",
    )
    set_rows = [
        {
            "role": "backtest population",
            "set_name": full_set.name,
            "members": len(full_set.members),
        },
        {
            "role": "bounded diagnostics",
            "set_name": diagnostic_set.name,
            "members": len(diagnostic_set.members),
        },
    ]
compatible_sets = pl.DataFrame(
    set_rows,
    schema={"role": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `15_model_analysis.py` reopens the named compatible and diagnostic sets. `16_backtest.py` passes
# every full-set catalog row directly to the shared backtest runner. Model metrics do not choose a
# configuration or checkpoint.

# %% [markdown]
# ## Key takeaways and limitations
#
# - TabM candidates share compatible fold preparation while each candidate and epoch retains its own
#   durable fitted state.
# - Restart validates completed candidate-fold artifacts before deciding what to recompute.
# - Preview reductions are identity-bearing and remain outside the canonical population.
# - The architecture learns interactions from the declared tabular features; its validation results
#   do not establish stability under a changed feature distribution.
