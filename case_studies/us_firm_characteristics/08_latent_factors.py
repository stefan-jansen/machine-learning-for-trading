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
# # Firm characteristics: what the latent-factor family is, and who publishes what
#
# The three notebooks before this one predicted the return from the characteristics directly. The
# latent-factor family starts somewhere else: it supposes the cross-section is driven by a small
# number of common factors, and treats a characteristic as evidence about **how exposed a firm is
# to them** rather than about the return itself. What gets estimated is a map from characteristics
# to exposures, shared by every firm and every month, so it is fitted on the whole panel instead of
# one cross-section at a time.
#
# Four members of that family are declared here, and they are **two pairs rather than four points
# on one axis**:
#
# | notebook | model | what it assumes |
# |---|---|---|
# | [`08a_ipca`](08a_ipca.ipynb) | instrumented PCA | exposures are a **linear** function of the characteristics |
# | [`08b_conditional_autoencoder`](08b_conditional_autoencoder.ipynb) | conditional autoencoder | same structure, the map is a **network** |
# | [`08c_stochastic_discount_factor`](08c_stochastic_discount_factor.ipynb) | stochastic discount factor | no two-stage split: prices the cross-section directly |
# | [`08d_supervised_autoencoder`](08d_supervised_autoencoder.ipynb) | supervised autoencoder | no two-stage split: predicts the return directly, keeping only the bottleneck |
#
# The first pair differs in the shape of one function, which is what makes those two worth reading
# against each other. The second pair breaks the two-stage shape from opposite ends - one because
# it prices, one because it predicts - and `08d` is in particular the family's own control: it
# keeps the low-dimensional bottleneck and drops the factor interpretation, so what it does not
# achieve is what the structure is worth.
#
# **Ordinary PCA is not among them**, and its absence is declared rather than accidental. PCA finds
# the directions of the return panel that explain the most variance, which involves the
# characteristics nowhere, and it needs a firm to be the same firm across the whole sample. This
# release publishes anonymous identifiers that persist only inside each tensor block, so that
# second requirement cannot be met. `config/training/{label}.yaml` therefore declares no `pca`, and
# IPCA is the linear characteristic-sorted baseline in its place.
#
# **Learning objectives**
#
# - Say what the family asserts that the direct predictors do not.
# - Read a family whose members are split across notebooks as one declared population.
# - Check that the notebooks that exist cover the menu that is declared.
#
# **Book reference**: Chapter 14, Sections 14.5 to 14.7 (advanced conditional-factor models, the
# conditional autoencoder, and the stochastic discount factor and supervised autoencoder).
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_evaluation`](04_evaluation.ipynb).
#
# **What it writes**: nothing. This notebook fits no model and opens no holdout. The four notebooks
# it points at each publish their own population, and
# [`10_model_analysis`](10_model_analysis.ipynb) is where they are compared against the other
# families.

# %%
"""Index and coverage check for the US firm characteristics latent-factor family."""

import ast
from pathlib import Path

import polars as pl

from case_studies.research import declared_labels, load_model_configs, open_study
from utils.paths import REPO_ROOT

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %%
study = open_study(
    "us_firm_characteristics", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)

# %% [markdown]
# ## 1. The declared menu
#
# `config/training/{label}.yaml` lists the family's members for each label, and every declared
# label declares the same four. That is the population the four execution notebooks are between
# them responsible for.

# %%
declared_labels(study, "latent_factors")

# %%
menu = load_model_configs(study, "latent_factors")
menu

# %% [markdown]
# ## 2. Which notebook claims which member
#
# The menu says what must be produced; it does not say by whom. The family is split across four
# notebooks, each publishing one model under its own population name, so the mapping between the
# two lives in the notebooks and is read back here rather than restated.
#
# **A member the menu declares and no notebook claims publishes nothing, and nothing else would
# catch it**: each execution notebook checks the labels it covers against its own declared rows, so
# none of them can see a model that no notebook requests at all. That is what this cell is for.
#
# `NOTEBOOK_DIR` below is the repository, not `get_case_study_dir`. That helper answers "where does
# this case study read and write its data", which `ML4T_OUTPUT_DIR` redirects to an isolated root -
# correct for labels, features and the run log, and wrong here: the notebooks are source, they live
# where the source lives, and under a redirect the glob found none of them and reported every
# declared member as unclaimed.

# %%
NOTEBOOK_DIR = REPO_ROOT / "case_studies" / "us_firm_characteristics"


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


notebooks = sorted(NOTEBOOK_DIR.glob("08[a-z]_*.py"))
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
# checkpoints come from how the estimator is trained rather than from a shared setting. IPCA runs
# alternating least squares to convergence and has one state per fold; the two autoencoders train
# for a declared epoch budget and save at a declared interval; the stochastic discount factor
# trains in phases and saves at a declared list of cumulative epochs. Each notebook's plan shows
# its own count before it fits anything.
#
# This is why the family cannot publish one population. A population is an immutable list of
# prediction identities, and the four members are fitted by four notebooks at different times, so
# one shared name would mean the first to run either blocks the others or publishes a snapshot
# missing them.

# %%
labels = declared_labels(study, "latent_factors")
pl.DataFrame(
    {
        "config_name": sorted(declared_models),
        "labels": [len(labels)] * len(declared_models),
        "population": [
            f"us_firm_characteristics-{name}-validation-v1" for name in sorted(declared_models)
        ],
    }
)

# %% [markdown]
# ## 4. What to notice
#
# **A family split across notebooks needs its coverage checked somewhere, and this is that place.**
# Each execution notebook can tell that it fitted every label it declared; none of them can tell
# that a fifth member exists in the menu with no notebook behind it. The check in section 2 reads
# the claim out of each notebook's source rather than from a list maintained here, so adding a
# member to the menu without adding a notebook fails, and so does adding a notebook that duplicates
# another's model.
#
# **The four members are not a ladder.** They are two pairs, and the interesting comparisons are
# within a pair - linear map against network map in the first, pricing against predicting in the
# second - plus the cross-pair one between `08b` and `08d`, which share a network and a bottleneck
# and differ only in whether the factor structure is imposed. A single ranking over all four would
# hide every one of those.
#
# **Nothing here compares results, and that is deliberate.** Reading the four populations against
# each other, and against the linear, boosted and tabular families, is
# [`10_model_analysis`](10_model_analysis.ipynb)'s job, with the whole population in front of it
# and the selection rule stated. A comparison made here would be made on a subset and would be made
# before [`09_causal_dml`](09_causal_dml.ipynb) has run.

# %% [markdown]
# **Next**: [`08a_ipca`](08a_ipca.ipynb) publishes the linear member and is the one to read first,
# because the three that follow are all described by what they change about it.
