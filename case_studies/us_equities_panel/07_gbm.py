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
# # Gradient Boosting - US Equities Panel
#
# This notebook generates walk-forward validation predictions for the published LightGBM
# configurations and every declared iteration checkpoint. The notebook exposes the model menu,
# overrides, and execution tier. Shared code owns fold preparation, training, fitted-model
# persistence, restart, prediction coverage, metrics, and registry writes.
#
# The same request can run one configuration, a selected subset, or the complete menu. The batch
# runner prepares each compatible fold once and retains separate immutable identities for every
# configuration and checkpoint. The implementation is in `case_studies/utils/gbm.py` for readers
# who want to add a model family or change its training logic.
#
# **Learning objectives**
#
# - Configure a complete boosting menu and its iteration checkpoints.
# - Explain when candidates may reuse fold preparation and when they require separate work.
# - Validate fitted boosters, checkpoint coverage, and catalog identities.
#
# **Book reference**: Chapter 12, Section 12.2 (GBM Libraries)
#
# **Prerequisites**: `03_financial_features.py`, `04_model_based_features.py`, and
# `05_evaluation.py`.

# %%
"""Generate GBM validation predictions through the shared research interface."""

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
DIAGNOSTIC_CONFIG_NAMES = ["default_mse"]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
TRAIN_SAMPLE_FRAC = 1.0
MAX_FOLDS = 0
PREVIEW_MAX_ITERATIONS = 0
PREVIEW_CHECKPOINT_INTERVAL = 0

# %% [markdown]
# ## Configure the experiment
#
# `CONFIG_NAMES = []` runs the complete published GBM menu. A subset makes a targeted extension or
# diagnostic visible without changing orchestration code. `COMMON_OVERRIDES` applies validated
# LightGBM or runner parameters to every selected configuration. `CONFIG_OVERRIDES` adds changes for
# one named configuration; those values take precedence.
#
# Canonical requests use the complete declared folds and data. Reduced checks must use the preview
# tier, declare every reduction below, and write to an isolated workspace. Preview results cannot
# enter official comparisons, candidate sets, locks, or holdout evaluation.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
label = PRIMARY_LABEL or setup["labels"]["primary"]

published_configs = load_configs(CASE_STUDY_ID, label, family="gbm")
published_names = [str(config["config_name"]) for config in published_configs]
selected_names = list(CONFIG_NAMES) if CONFIG_NAMES else published_names
unknown_names = sorted(set(selected_names) - set(published_names))
unknown_overrides = sorted(set(CONFIG_OVERRIDES) - set(selected_names))
if unknown_names:
    raise ValueError(f"Unknown GBM configurations: {unknown_names}")
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
        # The 15 configurations are five capacity profiles crossed with three objectives, and
        # those two fields are what tell them apart. `library` is lightgbm on every row and
        # `model_class` is not a key a GBM preset has at all.
        "objective": [
            (config.get("params") or {}).get("objective") for config in published_configs
        ],
        "num_leaves": [
            (config.get("params") or {}).get("num_leaves") for config in published_configs
        ],
        "published_params": [str(config.get("params") or {}) for config in published_configs],
        "max_iterations": [config.get("max_iterations") for config in published_configs],
        "checkpoint_interval": [config.get("checkpoint_interval") for config in published_configs],
        "selected": [config["config_name"] in selected_names for config in published_configs],
    }
)
menu

# %%
preview_reductions = {}
if MAX_SYMBOLS:
    preview_reductions["max_symbols"] = int(MAX_SYMBOLS)
if TRAIN_SAMPLE_FRAC != 1.0:
    preview_reductions["train_sample_frac"] = float(TRAIN_SAMPLE_FRAC)
if MAX_FOLDS:
    preview_reductions["folds"] = list(range(int(MAX_FOLDS)))
if PREVIEW_MAX_ITERATIONS:
    preview_reductions["max_iterations"] = int(PREVIEW_MAX_ITERATIONS)
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
# Each selected configuration becomes one request after common and configuration-specific overrides
# are combined.

# %%
requests = []
for config_name in selected_names:
    overrides = {**COMMON_OVERRIDES, **dict(CONFIG_OVERRIDES.get(config_name, {}))}
    requests.append(
        study.model(
            family="gbm",
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
# The planner resolves every training and iteration-checkpoint identity before fitting and writes
# the canonical checkpoint population first. The runner then trains all compatible candidates while
# one prepared fold is resident. It writes every
# declared booster checkpoint and prediction shard before releasing the fold. A retry validates and
# reuses complete candidate-fold work instead of restarting the full grid.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-gbm-checkpoints-v1",
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
# These rows show the feature artifacts, feature count, folds, effective device, checkpoint
# schedule, and immutable training identity. Defaults that were not repeated in the parameter cell
# remain visible in the stored specification.

# %%
resolved_rows = []
for run in execution.runs:
    spec = run.training.spec()
    computation = spec["computation"]
    feature_artifacts = computation["feature_artifacts"]
    artifact_names = (
        sorted(feature_artifacts)
        if isinstance(feature_artifacts, dict)
        else [str(item) for item in feature_artifacts]
    )
    resolved_rows.append(
        {
            "config_name": spec["config_name"],
            "task": computation["task"]["type"],
            "features": len(computation["feature_names"]),
            "feature_artifacts": artifact_names,
            "folds": computation["expected_prediction_keys"]["n_folds"],
            "device": spec["provenance"]["device"],
            "checkpoints": [item["value"] for item in computation["checkpoint_schedule"]],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("config_name")
resolved_table

# %% [markdown]
# ## Validate and inspect the handoff
#
# Each catalog row represents one complete validation prediction set for one training identity and
# checkpoint. Downstream notebooks filter these rows with ordinary Polars expressions and pass the
# selected table directly to backtesting. Multiple selected rows remain independent candidates.

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
    "ic_n_days",
    "training_hash",
    "prediction_hash",
]
catalog_rows = execution.catalog_rows.select(
    column for column in catalog_columns if column in execution.catalog_rows.columns
).sort("config_name", "checkpoint_value", "prediction_hash")
full_days = int(catalog_rows.get_column("ic_n_days").max())
catalog_rows = catalog_rows.with_columns(full_coverage=pl.col("ic_n_days") == full_days)
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
# A canonical default run freezes every returned prediction row under a stable family/label name.
# The separately named diagnostic subset is bounded by the visible configuration list above. Preview
# and customized canonical requests retain their result rows without publishing an official set.

# %% tags=["results"]
set_rows = []
is_published_population = (
    EXECUTION_TIER == "canonical"
    and selected_names == published_names
    and not COMMON_OVERRIDES
    and not CONFIG_OVERRIDES
)
if is_published_population:
    label_name = label.replace("_", "-")
    full_set = study.predictions.freeze(
        execution.catalog_rows,
        name=f"us-equities-{label_name}-gbm-v1",
    )
    diagnostic_set = study.predictions.freeze(
        execution.catalog_rows.filter(pl.col("config_name").is_in(DIAGNOSTIC_CONFIG_NAMES)),
        name=f"us-equities-{label_name}-gbm-diagnostics-v1",
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
# - Each iteration checkpoint is an immutable prediction result attached to one fitted training
#   identity.
# - Fold-major execution shares compatible preparation without sharing candidate-specific model
#   state.
# - The full validation population remains available to analysis and backtesting without an IC-based
#   selection step.
# - Tree ensembles represent nonlinear interactions within the configured depth, leaf, and sampling
#   settings; behavior outside the observed feature support is not identified.
