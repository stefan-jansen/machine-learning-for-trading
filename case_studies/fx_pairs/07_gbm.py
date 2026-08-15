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
# # Gradient Boosting - FX Pairs
#
# This notebook fits the published LightGBM menu to the same FX learning task used by the linear
# models. The model's tree count is part of the configuration: every declared checkpoint is
# persisted, predicted, checked for exact validation coverage, and exposed to backtesting. Rank
# correlation describes predictions but does not select a checkpoint.
#
# **Learning objectives**
#
# - Resolve loss, tree capacity, and hardware settings before fitting.
# - Persist every declared tree checkpoint from each fold's complete booster.
# - Pass the full checkpoint population to the prediction catalog.
#
# **Book reference**: Chapter 12, Section 12.2
#
# **Prerequisites**: `02_labels`, `03_financial_features`, and `04_model_based_features`.

# %%
"""Fit and catalog the published gradient-boosting FX configurations."""

import polars as pl
import yaml

from case_studies.research import ExecutionTier, Study, plan_models
from case_studies.utils.gbm import gbm_checkpoint_iterations
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
MAX_FOLDS = 0
FORCE_RETRAIN = False
PREDICTION_SPLIT = "validation"
TRAIN_SAMPLE_FRAC = 1.0

# %% [markdown]
# ## Select the task and execution tier
#
# The setup file chooses GPU execution for canonical runs. Any fold, symbol, iteration, or sampling
# reduction belongs to a preview identity and cannot enter an official population.

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
    raise ValueError("valid boosters are replayed by identity; change the request to refit")
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
print(f"LightGBM device: {setup['modeling']['gbm']['device']}")

# %% [markdown]
# ## Build the declared request population
#
# Huber presets declare a scale that the runner resolves against each training fold. This makes the
# intended robust loss active at the return scale instead of reproducing squared error.

# %%
menu = [
    (label, config)
    for label in labels
    for config in load_configs(CASE_STUDY_ID, label, family="gbm")
]
requests = [
    study.model(
        family="gbm",
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
        "label": [label for label, _ in menu],
        "config_name": [config["config_name"] for _, config in menu],
        "objective": [config["params"]["objective"] for _, config in menu],
        "max_iterations": [config["max_iterations"] for _, config in menu],
        "checkpoint_interval": [config["checkpoint_interval"] for _, config in menu],
        "declared_checkpoints": [len(gbm_checkpoint_iterations(config)) for _, config in menu],
    }
)
request_table

# %% [markdown]
# ## Declare every checkpoint before fitting
#
# Each tree checkpoint is its own downstream configuration, so the population is the full cross
# product of configurations and declared checkpoints. Planning resolves those identities without
# fitting a booster, which is what lets a later failure show up as a missing member.

# %% tags=["results"]
plan = plan_models(study, requests=requests)

configured = {(label, config["config_name"]) for label, config in menu}
planned = {(member.label, member.config_name) for member in plan.members}
if planned != configured:
    raise RuntimeError(
        "the plan does not match the configured GBM menu; "
        f"missing {sorted(configured - planned)}, unexpected {sorted(planned - configured)}"
    )

expected_checkpoints = sum(request_table.get_column("declared_checkpoints"))
if len(plan.expected_prediction_hashes) != expected_checkpoints:
    raise RuntimeError(
        f"the menu declares {expected_checkpoints} checkpoints, "
        f"the plan resolved {len(plan.expected_prediction_hashes)}"
    )

pl.DataFrame(
    {
        "label": [member.label for member in plan.members],
        "config_name": [member.config_name for member in plan.members],
        "checkpoint_kind": [member.checkpoint_kind for member in plan.members],
        "checkpoint_value": [member.checkpoint_value for member in plan.members],
        "prediction_hash": [member.prediction_hash for member in plan.members],
    }
)

# %% [markdown]
# ## Record the official population, then fit or replay every booster
#
# A complete fold model contains all tree checkpoints. The shared runner can therefore reload a
# valid booster and regenerate any declared checkpoint without retraining.

# %% tags=["results"]
population = (
    plan.create_population(name=f"{CASE_STUDY_ID}:{'+'.join(labels)}:gbm")
    if tier is ExecutionTier.CANONICAL
    else None
)

execution = plan.run()
if len(execution.runs) != len(requests):
    raise RuntimeError("the GBM runner did not return every requested configuration")

catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("partial checkpoint predictions cannot pass to backtesting")
if catalog.select("label", "config_name", "checkpoint_value").n_unique() != catalog.height:
    raise RuntimeError("each configuration and tree checkpoint must identify one prediction set")

catalog.select(
    "label",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "complete",
    "ic_mean",
    "ic_t",
    "training_hash",
    "prediction_hash",
)

# %% [markdown]
# ## Verify fitted-state replay and catalog recovery
#
# A second identical request must reuse the persisted boosters and recover the same checkpoint
# identities. The catalog remains the only handoff the backtest notebook needs.

# %% tags=["results"]
replayed = plan.run()
replayed_hashes = set(replayed.catalog_rows.get_column("prediction_hash"))
if replayed_hashes != set(catalog.get_column("prediction_hash")):
    raise RuntimeError("booster replay changed the declared checkpoint population")
if any(not diagnostic.get("cache_hit") for diagnostic in replayed.diagnostics):
    raise RuntimeError("an identical GBM request did not reuse its complete fitted state")

if population is not None:
    population.require_complete()
    print(f"Official prediction population: {population.hash}")
else:
    print("Preview checkpoints are isolated from official comparison and selection.")

# %% [markdown]
# ## Key takeaways
#
# - The requested loss and runtime settings are recorded before any model fits.
# - Every declared tree checkpoint is a separate downstream configuration.
# - Stored boosters reproduce checkpoint predictions without another training run.
