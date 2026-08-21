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
# # Linear Models - US Equities Panel
#
# This notebook generates walk-forward validation predictions for the published linear-model
# configurations. The visible decisions are the label, configurations, parameter overrides, and
# execution tier. Shared code owns fold construction, preprocessing, fitting, recovery, coverage
# checks, and registry writes.
#
# The normal path is intentionally short:
#
# 1. Open the case-study workspace.
# 2. Select configurations from the published menu.
# 3. Run them through the shared fold-major implementation.
# 4. Inspect the returned catalog rows, coverage, lineage, and artifacts.
#
# To study or extend the implementation, open `case_studies/utils/linear.py`. A new estimator or
# preprocessing idea remains ordinary Python and can implement the same request/result contract.
#
# **Learning objectives**
#
# - Express a complete linear-model experiment through visible request parameters.
# - Distinguish compatible fold preparation from estimator-specific fitting.
# - Validate checkpoint identities, prediction coverage, and downstream catalog rows.
#
# **Book reference**: Chapter 11, Section 11.2 (Regularized Linear Models)
#
# **Prerequisites**: `03_financial_features.py`, `04_model_based_features.py`, and
# `05_evaluation.py`.

# %%
"""Generate linear-model validation predictions through the shared research interface."""

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
CONFIG_OVERRIDES = {}
DIAGNOSTIC_CONFIG_NAMES = ["ridge_a1.0"]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
TRAIN_SAMPLE_FRAC = 1.0
MAX_FOLDS = 0

# %% [markdown]
# ## Configure the experiment
#
# `CONFIG_NAMES = []` runs the complete published linear menu. Set it to a visible subset such as
# `['ridge_a1.0', 'lasso_a0.001']` for a targeted experiment. `CONFIG_OVERRIDES` changes only the
# named estimator parameters, for example `{'ridge_a1.0': {'alpha': 2}}`; the resolved training
# specification records every effective default and override. `DIAGNOSTIC_CONFIG_NAMES` declares
# the bounded subset used for raw prediction comparisons in the analysis notebook.
#
# Canonical execution uses the complete data and fold protocol. For a reduced pipeline check, set
# `EXECUTION_TIER = 'preview'` and declare at least one reduction. Preview identities and artifacts
# are isolated from official comparisons and holdout decisions.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
label = PRIMARY_LABEL or setup["labels"]["primary"]

published_configs = load_configs(CASE_STUDY_ID, label, family="linear")
published_names = [str(config["config_name"]) for config in published_configs]
selected_names = list(CONFIG_NAMES) if CONFIG_NAMES else published_names
unknown_names = sorted(set(selected_names) - set(published_names))
unknown_overrides = sorted(set(CONFIG_OVERRIDES) - set(selected_names))
if unknown_names:
    raise ValueError(f"Unknown linear configurations: {unknown_names}")
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
        "model_class": [config["model_class"] for config in published_configs],
        "published_params": [str(config.get("params") or {}) for config in published_configs],
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
# Each selected configuration becomes one visible request with its own estimator overrides.

# %%
requests = tuple(
    study.model(
        family="linear",
        label=label,
        config_name=config_name,
        overrides=dict(CONFIG_OVERRIDES.get(config_name, {})),
        execution_tier=EXECUTION_TIER,
        preview_reductions=preview_reductions,
    )
    for config_name in selected_names
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
# ## Plan and execute the selected configurations
#
# `plan_models` resolves every training and checkpoint identity before fitting. The canonical
# checkpoint population is written first, so a failed configuration remains visible as missing.
# The plan materializes the panel once for the compatible request batch. Execution then prepares
# one compatible fold at a time for fitting.
# Each configuration still receives its own immutable training and prediction identities. Completed
# folds are committed incrementally, so a retry reuses valid work and recomputes only incomplete
# folds.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-linear-checkpoints-v1",
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
# These rows show the feature artifacts, feature count, folds, task, estimator, and immutable
# training identity used by each request. Defaults that were not repeated in the parameter cell are
# part of the stored specification.

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
            "estimator": computation["model"]["class"],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("config_name")
resolved_table

# %% [markdown]
# ## Validate and inspect the handoff
#
# The returned Polars rows are the downstream interface. A strategy notebook can filter these rows
# by human-readable fields and pass the selection directly to `run_backtests`; readers do not need
# to copy registry hashes. The hashes remain visible for exact provenance and artifact inspection.
#
# **Read `ic_n_days` before `ic_mean`, or the table will mislead you.** The information
# coefficient is an average of per-date rank correlations, and a date only contributes one if the
# model's predictions vary across the panel that day. An L1 penalty large enough to zero every
# coefficient predicts the same value for every stock, and a constant has no rank correlation with
# anything - so those dates drop out and the configuration's IC is an average over the dates where
# it happened to stay non-degenerate. That is not the same measurement as its neighbours', and
# comparing the two compares different samples. `full_coverage` marks the rows measured on every
# validation date; the rest are reporting on a subset the model selected for itself.

# %% tags=["results"]
catalog_columns = [
    "family",
    "config_name",
    "label",
    "split",
    "checkpoint_kind",
    "execution_tier",
    "complete",
    "ic_mean",
    "ic_n_days",
    "training_hash",
    "prediction_hash",
]
catalog_rows = execution.catalog_rows.select(
    column for column in catalog_columns if column in execution.catalog_rows.columns
).sort("config_name", "prediction_hash")
full_days = int(catalog_rows.get_column("ic_n_days").max())
catalog_rows = catalog_rows.with_columns(full_coverage=pl.col("ic_n_days") == full_days)
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
                "config_name": run.training.spec()["config_name"],
                "training_hash": run.training.hash,
                "prediction_hash": prediction.hash,
                "coverage_status": coverage["status"],
                "expected_rows": coverage["n_expected"],
                "actual_rows": coverage["n_actual"],
                "training_artifacts": len(run.training.artifacts()),
                "prediction_artifacts": len(prediction.artifacts()),
            }
        )

coverage_table = pl.DataFrame(coverage_rows).sort("config_name", "prediction_hash")
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
    EXECUTION_TIER == "canonical" and selected_names == published_names and not CONFIG_OVERRIDES
)
if is_published_population:
    label_name = label.replace("_", "-")
    full_set = study.predictions.freeze(
        execution.catalog_rows,
        name=f"us-equities-{label_name}-linear-v1",
    )
    diagnostic_set = study.predictions.freeze(
        execution.catalog_rows.filter(pl.col("config_name").is_in(DIAGNOSTIC_CONFIG_NAMES)),
        name=f"us-equities-{label_name}-linear-diagnostics-v1",
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
# - The request records the label, estimator configuration, data reductions, and execution tier.
# - Compatible estimators share fold preparation while retaining separate fitted states and result
#   identities.
# - The named validation population supports later analysis and backtesting without selecting on
#   predictive metrics.
# - Linear models restrict the conditional mean to the declared transformed feature basis;
#   nonlinear structure requires a different family request.
