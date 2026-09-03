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

from case_studies.research import (
    ExecutionTier,
    declared_labels,
    narrows_declared_catalog,
    open_study,
    plan_models,
    population_supersedes,
    sweep_labels,
)
from utils.modeling import load_configs
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
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = "7896f6bcaf7e"
# The tier is a parameter, not something inferred from whether a reduction happens to be set.
# Inferring it meant a run could be reduced and still open the case study's own artifacts in
# place, which is the production path; a reader under test then wrote where the published run
# writes. WORKSPACE is the other half: a preview has nowhere else to put its results.
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None

# %% [markdown]
# ## Select the task and execution tier
#
# Canonical execution uses every configured fold, symbol, epoch, and batch setting. A preview
# declares its reductions, takes an isolated workspace, and creates a preview identity there. A
# preview proves the path but cannot join the official model population.
#
# The reductions are read before the study is opened, because which study to open is decided by
# the tier and the two have to agree: a preview that reduces nothing is a canonical run wearing
# the wrong tier, and a canonical run carrying reductions would publish a narrowed population
# under the canonical name.

# %%
set_global_seeds(SEED)
REDUCTION_PARAMETERS = {
    "folds": list(range(MAX_FOLDS)) if MAX_FOLDS else None,
    "max_symbols": MAX_SYMBOLS or None,
    "n_epochs": N_EPOCHS or None,
}
reductions = {key: value for key, value in REDUCTION_PARAMETERS.items() if value is not None}
tier = ExecutionTier(EXECUTION_TIER)
if tier is ExecutionTier.PREVIEW and not reductions:
    raise ValueError("preview execution must declare at least one reduction")
if tier is ExecutionTier.CANONICAL and reductions:
    raise ValueError(f"canonical execution cannot carry reductions: {sorted(reductions)}")
study = open_study(CASE_STUDY_ID, execution_tier=tier, workspace=WORKSPACE or None)

# Which labels this notebook fits is a question for the training menus, not for the sweep list:
# `setup.yaml` says which labels the case study carries, a menu says what to fit for one of them,
# and a label in the sweep whose menu declares no `tabular_dl:` section owes nothing here. The two
# agree in this case study today, so restating the sweep list produced the right answer by
# coincidence and would have kept producing it silently after a menu changed. The order stays
# `setup.yaml`'s rather than `declared_labels`' menu-file order because the population is named
# after its labels and hashed over its members as an ordered list, so re-ordering would give the
# published population a new identity and demand a supersedes for a run that fits the same models.
fits_tabm = set(declared_labels(study, "tabular_dl"))
labels = (
    [PRIMARY_LABEL]
    if PRIMARY_LABEL
    else [label for label in sweep_labels(study) if label in fits_tabm]
)

if PREDICTION_SPLIT != "validation":
    raise ValueError("model selection uses validation predictions; holdout runs start from a lock")
if FORCE_RETRAIN:
    raise ValueError("valid checkpoints are reloaded by identity; change the request to refit")


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

# `PRIMARY_LABEL` narrows what is fitted, and a narrowed run declares a different set of members
# than the canonical population does. A population is immutable once written, so such a run must
# publish under its own name. The comparison is over `(label, config_name)` pairs rather than a
# row count, and it says so here rather than several cells later in a message about hashes.
if (
    narrows_declared_catalog(
        study,
        "tabular_dl",
        pl.DataFrame(
            {
                "label": [label for label, _ in menu],
                "config_name": [config["config_name"] for _, config in menu],
            }
        ),
    )
    and not POPULATION_NAME
):
    raise ValueError(
        f"this run declares {len(menu)} label-configuration pairs, which is not the complete "
        "declared catalog, so it cannot publish the canonical population; pass POPULATION_NAME "
        "to give it its own"
    )
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
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A population is the set of
# prediction identities it publishes, so anything that moves a training identity produces a
# different population under the same name, and the registry refuses to write it without being
# told which snapshot it supersedes. That lineage is the only record of which generation is which,
# and what moved the identities here was a change to the family's own source file rather than to
# anything the notebook declares.
#
# `population_supersedes` decides whether the declared hash may be offered. It is offered when the
# name already carries the generation this declaration produced, so a re-run resolves to the
# population it published, and when the declaration names the generation in force, so a refit
# publishes the next one. It is withheld everywhere else - on a reader's clean clone, where
# `run_log/` is gitignored and the registry has no generation at all; under a caller's own
# `POPULATION_NAME`; and in a preview, whose isolated registry holds nothing under this name.

# %% tags=["results"]
population_name = POPULATION_NAME or f"{CASE_STUDY_ID}:{'+'.join(labels)}:tabular_dl"
population = (
    plan.create_population(
        name=population_name,
        supersedes=population_supersedes(
            study, name=population_name, declared=SUPERSEDES_POPULATION
        ),
    )
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
