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
# # Temporal Convolutional Network - FX Pairs
#
# A temporal convolutional network reads a fixed number of consecutive daily observations for each
# currency pair. A missing daily observation invalidates every lookback window that crosses it. The
# shared sequence runner derives that eligible endpoint grid, primes each validation fold only with
# observable earlier rows, saves epoch checkpoints, and publishes predictions against the same grid.
#
# **Learning objectives**
#
# - Define one sequence-model request without rebuilding windows in the notebook.
# - Inspect the cadence-aware eligibility and checkpoint identities recorded by the runner.
# - Reload fitted weights and pass complete predictions through the shared catalog.
#
# **Book reference**: Chapter 13, Sections 13.2 and 13.4
#
# **Prerequisites**: `02_labels`, `03_financial_features`, and `04_model_based_features`.

# %%
"""Fit and catalog the published TCN FX configuration."""

import json

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
LOOKBACK = 0
BATCH_SIZE = 0
DEVICE = ""
SEED = 42

# %% [markdown]
# ## Plan the sequence request
#
# Fold and symbol reductions create a preview. Epochs, lookback length, and batch size are visible
# model overrides: changing any of them creates a different training identity. Planning resolves
# that identity, the eligible validation keys, and every declared epoch checkpoint before any
# training starts.

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
}
tier = ExecutionTier.PREVIEW if reductions else ExecutionTier.CANONICAL
# An empty DEVICE resolves to what the machine has. The runners refuse "cuda" on a host without
# it rather than falling back silently - which is the right contract for a run whose results get
# registered - so a hardcoded "cuda" default made the notebook unrunnable for any reader without
# an NVIDIA card, and unrunnable on a CPU CI runner. Resolving here keeps the refusal for anyone
# who asks for "cuda" explicitly; the resolved value is printed with the rest of the numerics
# below, so a run never leaves it implicit.
device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
overrides = {
    "device": device,
    **({"n_epochs": N_EPOCHS} if N_EPOCHS else {}),
    **({"batch_size": BATCH_SIZE} if BATCH_SIZE else {}),
    **({"lookback": LOOKBACK} if LOOKBACK else {}),
}
study = Study.regenerate(CASE_STUDY_ID)
ARCHITECTURE = "tcn"
menu = {
    label: [
        config["config_name"]
        for config in load_configs(CASE_STUDY_ID, label, family="deep_learning")
    ]
    for label in labels
}
uncovered = {label: sorted(set(names) - {ARCHITECTURE}) for label, names in menu.items()}
for label, names in menu.items():
    if ARCHITECTURE not in names:
        raise RuntimeError(
            f"{ARCHITECTURE} is not in the configured deep_learning menu for {label}: {names}"
        )

requests = [
    study.model(
        family="deep_learning",
        label=label,
        config_name=ARCHITECTURE,
        execution_tier=tier,
        preview_reductions=reductions,
        overrides=overrides,
    )
    for label in labels
]
plan = plan_models(study, requests=requests)

# This notebook owes one architecture on every configured label. The rest of the family menu is
# named here rather than left implicit, because a population that is short a configured model is
# otherwise indistinguishable from a complete one.
configured = {(label, ARCHITECTURE) for label in labels}
planned = {(member.label, member.config_name) for member in plan.members}
if planned != configured:
    raise RuntimeError(
        f"the plan does not match this notebook's declared coverage; "
        f"missing {sorted(configured - planned)}, unexpected {sorted(planned - configured)}"
    )
specs = {member.label: json.loads(member.spec_json) for member in plan.members}
computations = {label: spec.get("computation", spec) for label, spec in specs.items()}
computation = computations[labels[0]]

print(f"Labels: {', '.join(labels)}")
print(f"Execution tier: {tier.value}")
print(f"Device: {computation['numerics']['device']}")
print(f"Lookback: {computation['preprocessing']['lookback']} consecutive daily observations")
for horizon, values in computations.items():
    print(f"Eligible validation rows, {horizon}: {values['expected_prediction_keys']['n_rows']:,}")
for horizon, names in uncovered.items():
    print(
        f"Configured deep_learning models this notebook does not run, {horizon}: {names or 'none'}"
    )

# %% [markdown]
# ## Inspect the declared checkpoints and gap policy
#
# The resolved request records exact validation keys and the rule that excludes windows crossing a
# missing expected day. Checkpoint values below are training epochs, not IC-selected summaries.

# %%
checkpoint_schedule = pl.DataFrame(computation["checkpoint_schedule"])
input_summary = pl.DataFrame(
    {
        "label": list(computations),
        "gap_policy": [c["preprocessing"]["gap_policy"] for c in computations.values()],
        "validation_folds": [
            c["expected_prediction_keys"]["n_folds"] for c in computations.values()
        ],
        "validation_rows": [c["expected_prediction_keys"]["n_rows"] for c in computations.values()],
        "key_digest": [c["expected_prediction_keys"]["digest"] for c in computations.values()],
    }
)
input_summary
checkpoint_schedule

# %% [markdown]
# ## Record the official population, then fit or reload the TCN
#
# The same resolved request is used by the notebook and direct Python callers. Publication fails if
# any fold is missing, any prediction is non-finite, or the prediction keys differ from eligibility.

# %% tags=["results"]
if len(plan.expected_prediction_hashes) != checkpoint_schedule.height * len(labels):
    raise RuntimeError("the plan does not cover every declared epoch checkpoint on every label")
population = (
    plan.create_population(name=f"{CASE_STUDY_ID}:{'+'.join(labels)}:tcn")
    if tier is ExecutionTier.CANONICAL
    else None
)

execution = plan.run()
catalog = execution.catalog_rows.sort("label", "checkpoint_value")
if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("partial TCN checkpoints cannot pass to backtesting")
for label in labels:
    published = catalog.filter(pl.col("label") == label).get_column("checkpoint_value").to_list()
    if published != checkpoint_schedule["value"].to_list():
        raise RuntimeError(f"catalog checkpoints for {label} differ from the resolved request")

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
# ## Reload the fitted state
#
# An identical call validates the saved weights and returns the same prediction identities. The
# comparison with other model families belongs in `12_model_analysis` after every family completes.

# %% tags=["results"]
replayed = plan.run()
if set(replayed.catalog_rows.get_column("prediction_hash")) != set(
    catalog.get_column("prediction_hash")
):
    raise RuntimeError("TCN checkpoint reload changed the prediction population")

if population is not None:
    population.require_complete()
    print(f"Official prediction population: {population.hash}")
else:
    print("Preview sequence checkpoints remain outside official comparisons.")

# %% [markdown]
# ## Key takeaways
#
# - Eligibility is defined by consecutive observations at the declared daily cadence.
# - Validation priming uses earlier observable rows without admitting training targets.
# - Every saved epoch checkpoint remains available to the backtest stage.
