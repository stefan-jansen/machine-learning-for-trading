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
# # Tabular Deep Learning - FX Pairs
#
# TabM applies a small neural network to each decision row without turning the history into a
# sequence. This notebook submits the published capacity choices to the shared TabM runner. The
# runner fits preprocessing inside each training fold, saves every declared weight checkpoint, and
# publishes a separate complete validation prediction set for every checkpoint.
#
# **Learning objectives**
#
# - Express neural-network capacity and checkpoint schedules as visible requests.
# - Verify that every fold and epoch checkpoint has reloadable fitted state.
# - Continue from complete prediction rows without selecting a checkpoint by rank correlation.
#
# **Book reference**: Chapter 12, Section 12.3
#
# **Prerequisites**: `02_labels`, `03_financial_features`, and `04_model_based_features`.

# %%
"""Fit and catalog the published TabM FX configurations."""

import polars as pl
import torch
import yaml

from case_studies.research import ExecutionTier, Study, plan_models
from utils.modeling import load_configs
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
MAX_FOLDS = 0
FORCE_RETRAIN = False
PREDICTION_SPLIT = "validation"
N_EPOCHS = 0
BATCH_SIZE = 0
DEVICE = ""
SEED = 42

# %% [markdown]
# ## Select the task and execution tier
#
# Canonical execution uses every configured fold, symbol, epoch, and batch setting. Supplying a
# reduction creates a preview identity in an isolated registry. A preview proves the path but cannot
# join the official model population.

# %%
set_global_seeds(SEED)
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
    raise ValueError("valid checkpoints are reloaded by identity; change the request to refit")

reductions = {
    **({"folds": list(range(MAX_FOLDS))} if MAX_FOLDS else {}),
    **({"max_symbols": MAX_SYMBOLS} if MAX_SYMBOLS else {}),
    **({"n_epochs": N_EPOCHS} if N_EPOCHS else {}),
}
tier = ExecutionTier.PREVIEW if reductions else ExecutionTier.CANONICAL
study = Study.regenerate(CASE_STUDY_ID)

print(f"Labels: {', '.join(labels)}")
print(f"Execution tier: {tier.value}")
# An empty DEVICE resolves to what the machine has. The runners refuse "cuda" on a host without
# it rather than falling back silently - which is the right contract for a run whose results get
# registered - so a hardcoded "cuda" default made the notebook unrunnable for any reader without
# an NVIDIA card, and unrunnable on a CPU CI runner. Resolving here keeps the refusal for anyone
# who asks for "cuda" explicitly, and prints what was chosen so a run never leaves it implicit.
device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## Build the published requests
#
# The YAML menu supplies the architecture settings and production checkpoint schedules. The
# parameter cell can reduce epochs or change batch size for a preview without changing the menu.

# %%
overrides = {
    "device": device,
    **({"batch_size": BATCH_SIZE} if BATCH_SIZE else {}),
}
menu = [
    (label, config)
    for label in labels
    for config in load_configs(CASE_STUDY_ID, label, family="tabular_dl")
]
requests = [
    study.model(
        family="tabular_dl",
        label=label,
        config_name=config["config_name"],
        execution_tier=tier,
        preview_reductions=reductions,
        overrides=overrides,
    )
    for label, config in menu
]

pl.DataFrame(
    {
        "config_name": [request.config_name for request in requests],
        "label": [request.label for request in requests],
        "device": [device] * len(requests),
        "execution_tier": [request.execution_tier.value for request in requests],
    }
)

# %% [markdown]
# ## Declare every epoch checkpoint before training
#
# The declared epoch schedule, not the run that follows, decides how many downstream configurations
# this notebook owes. Planning resolves each one without training, so a failed member is visible as
# a gap in the population rather than a shorter catalog.

# %% tags=["results"]
plan = plan_models(study, requests=requests)
if len(plan.expected_training_hashes) != len(requests):
    raise RuntimeError("each TabM configuration must plan exactly one training identity")

configured = {(label, config["config_name"]) for label, config in menu}
planned = {(member.label, member.config_name) for member in plan.members}
if planned != configured:
    raise RuntimeError(
        "the plan does not match the configured TabM menu; "
        f"missing {sorted(configured - planned)}, unexpected {sorted(planned - configured)}"
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
# ## Record the official population, then fit or reload every capacity choice
#
# Compatible TabM requests share base-fold materialization. Candidate-specific scaling, random
# state, weights, and prediction identities remain separate. Any failed member stops the cell.

# %% tags=["results"]
population = (
    plan.create_population(name=f"{CASE_STUDY_ID}:{'+'.join(labels)}:tabular_dl")
    if tier is ExecutionTier.CANONICAL
    else None
)

execution = plan.run()
if len(execution.runs) != len(requests):
    raise RuntimeError("the TabM runner did not return every requested configuration")

catalog = execution.catalog_rows.sort("label", "config_name", "checkpoint_value")
if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("partial TabM checkpoints cannot pass to backtesting")
if catalog.select("label", "config_name", "checkpoint_value").n_unique() != catalog.height:
    raise RuntimeError("each configuration and epoch checkpoint must identify one prediction set")
if catalog.get_column("checkpoint_value").null_count():
    raise RuntimeError("every TabM prediction must name its epoch checkpoint")

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
# ## Reload the checkpoint population
#
# Repeating the same request validates the saved checkpoint manifests and returns the same catalog
# identities. No empty cached summary or single IC-chosen checkpoint is substituted.

# %% tags=["results"]
replayed = plan.run()
if set(replayed.catalog_rows.get_column("prediction_hash")) != set(
    catalog.get_column("prediction_hash")
):
    raise RuntimeError("TabM checkpoint reload changed the prediction population")

if population is not None:
    population.require_complete()
    print(f"Official prediction population: {population.hash}")
else:
    print("Preview checkpoints remain outside official comparison and holdout selection.")

# %% [markdown]
# ## Key takeaways
#
# - Train-only preprocessing and checkpoint persistence belong to the shared TabM computation.
# - Every declared epoch remains available to the backtest stage.
# - Rank correlation is a diagnostic field in the catalog, not a checkpoint-selection rule.
