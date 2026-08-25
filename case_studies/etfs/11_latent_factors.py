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
# # ETFs: what the latent-factor family is, and who publishes what
#
# The three modelling notebooks before this one predicted the return from the feature row
# directly. [`06_linear`](06_linear.ipynb) gave each column a coefficient,
# [`07_gbm`](07_gbm.ipynb) split on them, [`08_tabular_dl`](08_tabular_dl.ipynb) mixed them in a
# hidden layer, and they differ only in the shape of the function they may write down.
#
# The latent-factor family starts somewhere else. It supposes the hundred funds move together
# along a handful of common directions, and treats a feature as evidence about **how exposed a
# fund is to them** rather than about its return. What gets estimated is a map from features to
# exposures, shared by every fund and every date, so it is fitted on the whole panel instead of one
# cross-section at a time.
#
# Five members are declared, and they are **one baseline and two pairs, not five points on one
# axis**:
#
# | notebook | model | what it assumes |
# |---|---|---|
# | [`11a_pca`](11a_pca.ipynb) | principal components | the return panel alone; reads no features |
# | [`11b_ipca`](11b_ipca.ipynb) | instrumented PCA | exposures are a **linear** function of the features |
# | [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) | conditional autoencoder | same structure, the map is a **network** |
# | [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) | stochastic discount factor | no two-stage split: prices the cross-section directly |
# | [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) | supervised autoencoder | no two-stage split: predicts the return directly, keeping only the bottleneck |
#
# `11a` is the bar rather than a rung: it conditions on nothing, so what the four conditioned
# members beat it by is what conditioning bought. It is available here only because an ETF is the
# same fund throughout the sample, which [`11a_pca`](11a_pca.ipynb) sets out - a panel whose
# members enter and leave cannot support it, and the runner refuses it there.
#
# The first pair differs in the shape of one function, which is what makes those two worth reading
# against each other. The second pair breaks the two-stage shape from opposite ends - one because
# it prices, one because it predicts - and `11e` is the family's own control: it keeps the
# low-dimensional bottleneck and drops the factor interpretation, so what it does not achieve is
# what the structure is worth.
#
# **Learning objectives**
#
# - Say what the family asserts that the direct predictors do not.
# - Read a family whose members are split across notebooks as one declared population.
# - Check that the notebooks that exist cover the menu that is declared.
# - Say why five members that share a menu cannot share a population.
#
# **Book reference**: Chapter 14, Sections 14.5 to 14.7 (bridging economics and statistics,
# the conditional autoencoder, and the stochastic discount factor and supervised autoencoder).
# Chapter 13, Section 13.3 covers the return-panel PCA that `11a` publishes.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) for the feature matrix, and
# [`05_evaluation`](05_evaluation.ipynb) for the walk-forward folds.
#
# **What it writes**: nothing. This notebook fits no model, registers no run and opens no holdout.
# The five notebooks it points at each publish their own population, and
# [`13_model_analysis`](13_model_analysis.ipynb) is where they are compared against the other
# families.

# %%
"""Index and coverage check for the ETF latent-factor family."""

import ast
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import declared_labels, load_model_configs, open_study
from utils.paths import REPO_ROOT

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %%
study = open_study("etfs", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. The declared menu
#
# `config/training/{label}.yaml` lists the family's members for each label, and every declared
# label declares the same five. That is the population the five execution notebooks are between
# them responsible for.

# %%
declared_labels(study, "latent_factors")

# %%
menu = load_model_configs(study, "latent_factors")
menu

# %% [markdown]
# ## 2. Which notebook claims which member
#
# The menu says what must be produced; it does not say by whom. The family is split across five
# notebooks, each publishing one model under its own population name, so the mapping between the
# two lives in the notebooks and is read back here rather than restated.
#
# **A member the menu declares and no notebook claims publishes nothing, and nothing else would
# catch it**: each execution notebook checks the labels it covers against its own declared rows, so
# none of them can see a model that no notebook requests at all. That is what this cell is for.

# %%
# The repository, not `get_case_study_dir`. That helper answers "where does this case study read
# and write its data", which `ML4T_OUTPUT_DIR` redirects to an isolated root - correct for labels,
# features and the run log, and wrong here: the notebooks are source, they live where the source
# lives, and under a redirect the glob below would find none of them and report every declared
# member as unclaimed.
NOTEBOOK_DIR = REPO_ROOT / "case_studies" / "etfs"


def claimed_model(path: Path) -> str:
    """Return the model a latent-factor execution notebook publishes.

    Read from the notebook's own `MODEL_NAME` binding rather than from a list kept here. A list
    would be a second declaration of the same fact, and the failure it invites is the one this
    cell exists to detect: it would keep agreeing with itself after a notebook changed.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODEL_NAME" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path.name} binds no MODEL_NAME")


notebooks = sorted(NOTEBOOK_DIR.glob("11[a-z]_*.py"))
if not notebooks:
    raise RuntimeError(f"no latent-factor execution notebooks under {NOTEBOOK_DIR}")
claims = pl.DataFrame(
    {
        "notebook": [path.stem for path in notebooks],
        "config_name": [claimed_model(path) for path in notebooks],
    }
).sort("config_name")
claims

# %%
declared_models = set(menu.get_column("config_name"))
claimed = claims.get_column("config_name").to_list()
if len(claimed) != len(set(claimed)):
    raise RuntimeError(f"two notebooks claim the same latent-factor model: {sorted(claimed)}")
if set(claimed) != declared_models:
    raise RuntimeError(
        "the latent-factor notebooks do not cover the declared menu; "
        f"unclaimed {sorted(declared_models - set(claimed))}, "
        f"undeclared {sorted(set(claimed) - declared_models)}"
    )
print(f"{len(declared_models)} declared members, each claimed by exactly one notebook")

# %% [markdown]
# ## 3. What each one costs to publish
#
# The number of prediction sets a member contributes is its labels times its checkpoints, and the
# checkpoints come from how the estimator is trained rather than from a shared setting. PCA and
# IPCA solve to completion and have one state per fold; the two autoencoders train for a declared
# epoch budget and save at a declared interval; the stochastic discount factor trains in phases and
# saves at a declared list of cumulative epochs.
#
# The schedule below is read out of each estimator's own configuration, so it is the declaration
# and not a copy of one. The authoritative count is the `checkpoints` column in each notebook's own
# resolved plan, which is derived from these fields and printed before that notebook fits anything.

# %%
CONFIG_DIR = REPO_ROOT / "case_studies" / "config"
SCHEDULE_KEYS = ("n_epochs", "checkpoint_interval", "checkpoint_epochs")


def declared_schedule(name: str) -> str:
    """Summarise the checkpoint schedule an estimator's configuration declares."""
    config = yaml.safe_load((CONFIG_DIR / name / f"{name}.yaml").read_text())
    declared = {key: config[key] for key in SCHEDULE_KEYS if key in config}
    return ", ".join(f"{key}={value}" for key, value in declared.items()) or "none declared"


labels = declared_labels(study, "latent_factors")
pl.DataFrame(
    {
        "config_name": sorted(declared_models),
        "labels": [len(labels)] * len(declared_models),
        "schedule declared": [declared_schedule(name) for name in sorted(declared_models)],
        "population": [f"etfs-{name}-validation-v1" for name in sorted(declared_models)],
    }
)

# %% [markdown]
# This is why the family cannot publish one population. A population is an immutable list of
# prediction identities, and the five members are fitted by five notebooks at different times, so
# one shared name would mean the first to run either blocks the others or publishes a snapshot
# missing them.

# %% [markdown]
# ## 4. What to notice
#
# **A family split across notebooks needs its coverage checked somewhere, and this is that place.**
# Each execution notebook can tell that it fitted every label it declared; none of them can tell
# that a sixth member exists in the menu with no notebook behind it. The check in section 2 reads
# the claim out of each notebook's source rather than from a list maintained here, so adding a
# member to the menu without adding a notebook fails, and so does adding a notebook that duplicates
# another's model.
#
# **The five members are not a ranking.** They are a baseline and two pairs, and the interesting
# comparisons are within a pair - linear map against network map in the first, pricing against
# predicting in the second - plus the cross-pair one between `11c` and `11e`, which share a network
# and a bottleneck and differ only in whether the factor structure is imposed. Every one of those
# is also read against `11a`, which conditions on nothing. A single ordering over all five would
# hide all of it.
#
# **Nothing here compares results, and that is deliberate.** Reading the five populations against
# each other, and against the linear, boosted and tabular families, is
# [`13_model_analysis`](13_model_analysis.ipynb)'s job, with the whole population in front of it
# and the selection rule stated. A comparison made here would be made before
# [`12_causal_dml`](12_causal_dml.ipynb) has run, and selection on validation backtest Sharpe
# happens in [`14_backtest`](14_backtest.ipynb) rather than on any ranking shown earlier.

# %% [markdown]
# **Next**: [`11a_pca`](11a_pca.ipynb) publishes the unconditional baseline and is the one to read
# first, because the four that follow are all described by what they add to it.
