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
# # Linear Models - FX Pairs
#
# This notebook fits the published linear-model menu to cross-sectional FX return labels. Each
# configuration uses the same finalized labels, features, and walk-forward boundaries. The shared
# runner prepares every fold from those artifacts, fits train-only preprocessing, persists each
# fold model, and publishes validation predictions only when their keys exactly match the expected
# validation rows.
#
# **Learning objectives**
#
# - Load a published model menu and turn each entry into a visible model request.
# - Run compatible linear configurations through one fold-major implementation.
# - Check model persistence, fold coverage, and the catalog handoff to backtesting.
#
# **Book reference**: Chapter 11, Section 11.2
#
# **Prerequisites**: `02_labels`, `03_financial_features`, and `04_model_based_features`.

# %%
"""Fit and catalog the published linear FX configurations."""

import polars as pl
import yaml

from case_studies.research import ExecutionTier, Study, plan_models
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
FORCE_RETRAIN = False
PREDICTION_SPLIT = "validation"
TRAIN_SAMPLE_FRAC = 1.0
MAX_FOLDS = 0
SUPERSEDES_POPULATION = ""

# %% [markdown]
# ## Select the learning tasks
#
# The canonical population spans every configured horizon, because the allocation and cost stages
# select per label. An empty override therefore runs all of them; set `PRIMARY_LABEL` to one
# horizon to reproduce just that label's population. Reductions select preview execution
# automatically, which keeps the resulting identities out of official comparisons.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
labels = (
    [PRIMARY_LABEL]
    if PRIMARY_LABEL
    else [setup["labels"]["primary"], *setup["labels"].get("variants", [])]
)

if PREDICTION_SPLIT != "validation":
    raise ValueError("model selection uses validation predictions; holdout runs start from a lock")
if FORCE_RETRAIN:
    raise ValueError("identical requests reuse valid folds; change the request to create new work")
if not 0 < TRAIN_SAMPLE_FRAC <= 1:
    raise ValueError("TRAIN_SAMPLE_FRAC must be in (0, 1]")

reductions = {
    **({"folds": list(range(MAX_FOLDS))} if MAX_FOLDS else {}),
    **({"max_symbols": MAX_SYMBOLS} if MAX_SYMBOLS else {}),
    **({"train_sample_frac": TRAIN_SAMPLE_FRAC} if TRAIN_SAMPLE_FRAC < 1 else {}),
}
tier = ExecutionTier.PREVIEW if reductions else ExecutionTier.CANONICAL
study = Study.regenerate(CASE_STUDY_ID)

print(f"Labels: {', '.join(labels)}")
print(f"Execution tier: {tier.value}")
print(f"Configured folds: {MAX_FOLDS if MAX_FOLDS else 'all'}")
print(f"Configured symbols: {MAX_SYMBOLS if MAX_SYMBOLS else 'all'}")

# %% [markdown]
# ## Build visible model requests
#
# The YAML menu defines the published configurations. Every request below records the complete
# resolved estimator, preprocessing, cross-validation boundaries, input digests, and runtime
# identity before fitting starts.

# %%
menu = [
    (label, config)
    for label in labels
    for config in load_configs(CASE_STUDY_ID, label, family="linear")
]
requests = [
    study.model(
        family="linear",
        label=label,
        config_name=config["config_name"],
        execution_tier=tier,
        preview_reductions=reductions,
        overrides={},
    )
    for label, config in menu
]

request_table = pl.DataFrame(
    {
        "family": [request.family for request in requests],
        "config_name": [request.config_name for request in requests],
        "label": [request.label for request in requests],
        "execution_tier": [request.execution_tier.value for request in requests],
    }
)
request_table

# %% [markdown]
# ## Resolve every identity before fitting
#
# Planning resolves the complete estimator, preprocessing, fold boundaries, and checkpoint schedule
# of each request without fitting anything. The resulting training and prediction identities are the
# population this notebook owes, so a configuration that later fails is a missing member rather than
# a shorter result.

# %% tags=["results"]
plan = plan_models(study, requests=requests)
plan_table = pl.DataFrame(
    {
        "label": [member.label for member in plan.members],
        "config_name": [member.config_name for member in plan.members],
        "checkpoint_kind": [member.checkpoint_kind for member in plan.members],
        "training_hash": [member.training_hash for member in plan.members],
        "prediction_hash": [member.prediction_hash for member in plan.members],
    }
)
if len(plan.expected_training_hashes) != len(requests):
    raise RuntimeError("each linear configuration must plan exactly one training identity")

configured = {(label, config["config_name"]) for label, config in menu}
planned = {(member.label, member.config_name) for member in plan.members}
if planned != configured:
    raise RuntimeError(
        "the plan does not match the configured linear menu; "
        f"missing {sorted(configured - planned)}, unexpected {sorted(planned - configured)}"
    )
plan_table

# %% [markdown]
# ## Record the official population, then fit
#
# The population is written before the first fit, so an interrupted or failed run leaves a member
# that `require_complete` reports as missing. Compatible requests share base-fold preparation, and
# each configuration keeps its own fitted state, training identity, and validation prediction
# identity.

# %% [markdown]
# ### Replacing an earlier population
#
# The population is written before the first fit, so an interrupted run leaves a snapshot behind
# with no artifacts under it. Re-running is silent while the identities are unchanged, because an
# identical snapshot resolves to the same hash. Once the identities move, the registry refuses to
# overwrite the earlier snapshot and names it. Set `SUPERSEDES_POPULATION` to that hash to declare
# which snapshot this one replaces. It is deliberately a value a reader supplies, because the
# lineage is part of what the new population is and not a formality to be filled in automatically.
# The value is coerced to text because a population hash is a string, and an all-digit hash supplied
# through papermill's typed `-p` arrives as an integer that would never match the stored value.

# %% tags=["results"]
population = (
    plan.create_population(
        name=f"{CASE_STUDY_ID}:{'+'.join(labels)}:linear",
        supersedes=str(SUPERSEDES_POPULATION) or None,
    )
    if tier is ExecutionTier.CANONICAL
    else None
)

execution = plan.run()
if len(execution.runs) != len(requests):
    raise RuntimeError("the linear runner did not return every requested configuration")

catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("partial prediction sets cannot pass to backtesting")
if catalog.get_column("prediction_hash").n_unique() != catalog.height:
    raise RuntimeError("prediction identities must be unique")

catalog.select(
    "family",
    "config_name",
    "label",
    "checkpoint_kind",
    "complete",
    "ic_mean",
    "ic_t",
    "training_hash",
    "prediction_hash",
)

# %% [markdown]
# ## Verify restart and downstream handoff
#
# Reopening the study reconstructs the same catalog from the registry. Backtest notebooks receive
# selected catalog rows directly, so readers do not copy hashes between stages.

# %% tags=["results"]
reopened = Study.regenerate(CASE_STUDY_ID)
reloaded = reopened.predictions.table(include_preview=tier is ExecutionTier.PREVIEW).filter(
    pl.col("prediction_hash").is_in(catalog.get_column("prediction_hash").to_list())
)
if set(reloaded.get_column("prediction_hash")) != set(catalog.get_column("prediction_hash")):
    raise RuntimeError("the catalog did not recover every linear prediction after restart")

if population is not None:
    population.require_complete()
    print(f"Official prediction population: {population.hash}")
else:
    print("Preview rows are available only with include_preview=True and are not official members.")

# %% [markdown]
# ## Key takeaways
#
# - One request records one linear configuration on one label, and its complete computation.
# - Planning resolves every identity first, so the population is fixed before any model fits.
# - Compatible configurations share fold preparation without sharing fitted preprocessing or models.
# - Only complete validation prediction rows pass to the backtest stage.
