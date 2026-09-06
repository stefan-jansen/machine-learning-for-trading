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
# # US equities panel: the few movements the whole panel shares
#
# Most of what three thousand stocks do on a given day is not three thousand different things.
# One movement runs through nearly all of them, a few more run through overlapping groups, and
# what is left over is specific to each name. **Principal component analysis** finds those shared
# movements without being told anything about the stocks: it looks only at how the returns moved
# together and extracts the directions that account for the most of that common variation, in
# order.
#
# Each extracted direction is a **factor**. Each stock gets a **loading** on each factor, saying
# how much of that movement it takes. A prediction for a stock is then rebuilt from the factors
# and that stock's loadings, so what the model can say about a stock is exactly what the stock has
# in common with the rest of the panel - and nothing that is specific to it.
#
# **That is a compression, and the discarding is the point.** A handful of factors stand in for the
# whole cross-section, so this model is deliberately blind to what separates two stocks that
# load the same way. The notebooks before it are where stock-specific information lives; this
# one asks how much is left once you keep only what is shared.
#
# **Everything is fitted on training rows only.** Factors and loadings are both estimated inside
# each fold's training window and then applied to the validation rows, because a factor extracted
# from the whole sample would have been computed partly from the returns it is later asked to
# predict.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a factor and a loading are in terms of a panel of returns, without reference to an
#   algorithm.
# - Explain what a few-factor reconstruction of a stock's return can and cannot contain.
# - Say why the factors have to be extracted inside a fold's training window, and what a
#   whole-sample extraction would have used.
# - State what the declared factor count does and does not establish about the right number.
#
# **Book reference**: Chapter 13.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# [`02_labels`](02_labels.ipynb) the labels, and [`05_evaluation`](05_evaluation.ipynb) has
# established the walk-forward folds.
#
# **What it writes**: one training run and one complete validation prediction set per label, in
# `run_log/registry.db` and under `run_log/training/` and `run_log/predictions/`, frozen under a
# name per label. [`13_latent_factors`](13_latent_factors.ipynb) indexes them,
# [`15_model_analysis`](15_model_analysis.ipynb) compares them against the other families, and
# [`16_backtest`](16_backtest.ipynb) backtests every one. **Selection happens there, not here.**

# %%
"""Generate PCA validation predictions through the shared research interface."""

import os
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import open_study, plan_models
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
LABELS = []
OVERRIDES = {}
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
FOLD_IDS = []
PREVIEW_N_FACTORS = 0

# %% [markdown]
# ## 1. Which labels, and how many factors
#
# What each setting a run may pass decides:
#
# - **`LABELS`** empty fits the primary label and every declared variant. A subset fits only those.
# - **The factor count** is how many common movements are extracted. It is read from the preset at
#   `case_studies/config/pca/pca.yaml` rather than set here, so there is one place to change it
#   and one place to look. It is declared rather than searched, and that is a design choice with
#   consequences: too few and distinct common movements are forced into one direction, too many
#   and the later ones are fitting noise that will not repeat out of sample. Nothing in this case
#   study tunes it, so no result here is evidence that the declared count is right - only evidence
#   of what it gives.
# - **`OVERRIDES`** changes a resolved model parameter. An override moves the training identity, so
#   an overridden run registers beside the published one rather than replacing it.
# - **`EXECUTION_TIER`** is `canonical` or `preview`. A canonical run fits the whole panel on every
#   fold. A preview run declares its reductions and carries them in the identity, so its results
#   can never be compared against canonical ones or reach a holdout decision.
# - **`FOLD_IDS`** and **`MAX_SYMBOLS`** are the reductions a preview declares.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
published_labels = [setup["labels"]["primary"], *setup["labels"].get("variants", [])]
selected_labels = list(LABELS) if LABELS else published_labels
unknown_labels = sorted(set(selected_labels) - set(published_labels))
if unknown_labels:
    raise ValueError(f"Unknown labels: {unknown_labels}")
if len(selected_labels) != len(set(selected_labels)):
    raise ValueError("LABELS contains duplicates")

declared_factors = set()
for label in selected_labels:
    configured = {
        config["config_name"]: config
        for config in load_configs(CASE_STUDY_ID, label, family="latent_factors")
    }
    if "pca" not in configured:
        raise ValueError(f"PCA is not configured for {label}")
    declared_factors.add(int(configured["pca"]["params"]["n_factors"]))
# One declaration, read rather than restated. A count typed here as well would be a second
# declaration free to disagree with the preset, and the preset would then decide nothing.
if len(declared_factors) != 1:
    raise ValueError(f"the selected labels declare different factor counts: {declared_factors}")
N_FACTORS = declared_factors.pop()

label_menu = pl.DataFrame(
    {
        "label": published_labels,
        "selected": [label in selected_labels for label in published_labels],
        "n_factors": [N_FACTORS] * len(published_labels),
    }
)
label_menu

# %%
preview_reductions = {}
if MAX_SYMBOLS:
    preview_reductions["max_symbols"] = int(MAX_SYMBOLS)
if FOLD_IDS:
    preview_reductions["folds"] = [int(fold) for fold in FOLD_IDS]
if PREVIEW_N_FACTORS:
    preview_reductions["n_factors"] = int(PREVIEW_N_FACTORS)

# Both tiers resolve the study through `open_study`. It reads the labels and features in place and
# redirects only writes, so a preview run scores the same inputs a canonical one does and cannot
# publish over it.
if EXECUTION_TIER == "canonical":
    if preview_reductions:
        raise ValueError("Canonical execution cannot declare preview reductions")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if not preview_reductions:
        raise ValueError("Preview execution requires at least one declared reduction")
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError("EXECUTION_TIER must be 'canonical' or 'preview'")

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# One request per label. A **request** is the declaration bound to a label and an execution
# tier, with its overrides resolved; it holds no data, so the table below can be read before
# anything is loaded.
#
# The labels get separate requests rather than one shared fit because a label defines which
# rows are scorable and over what horizon. Fitting once and scoring three ways would give the
# three labels a common estimate built partly from rows that only one of them can see.

# %%
requests = tuple(
    study.model(
        family="latent_factors",
        label=label,
        config_name="pca",
        # The preset supplies the factor count; a value passed here would win over it, which is
        # what OVERRIDES is for and what makes such a run unpublishable.
        overrides=dict(OVERRIDES),
        execution_tier=EXECUTION_TIER,
        preview_reductions=preview_reductions,
        notebook="13a_pca",
    )
    for label in selected_labels
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
# ## 3. Planning, then fitting
#
# **Planning resolves every identity before any fitting starts**, and the list of them is written
# down as a population the run then has to fill completely - so a run that came out short reads as
# a failure rather than as a smaller experiment.
#
# The planner resolves every label-specific training and checkpoint identity before fitting and
# writes the canonical checkpoint population first. Each fold then fits PCA on its training return
# panel only. The runner persists the fitted components
# and reconstructs each registered prediction set from those artifacts before accepting cached
# work.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-pca-checkpoints-v1",
    )

planned_population = pl.DataFrame(
    {
        "label": [member.label for member in plan.members],
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
# ## 4. What was actually fitted
#
# The fully resolved specification, including every default nothing above restated: the factor
# count, the feature count, the fold count, and the cross-validation identity. This is the
# record a result is checked against, and it is what the training hash is computed from - so
# two rows with the same hash were fitted under the same declaration, and two with different
# hashes were not.

# %%
resolved_rows = []
for run in execution.runs:
    spec = run.training.spec()
    computation = spec["computation"]
    resolved_rows.append(
        {
            "label": spec["label"],
            "features": len(computation["feature_names"]),
            "folds": computation["expected_prediction_keys"]["n_folds"],
            "eligible_rows": computation["expected_prediction_keys"]["n_rows"],
            "n_factors": computation["model"]["n_factors"],
            "device": computation["runtime"]["device"],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("label")
resolved_table

# %% [markdown]
# ## 5. What came out
#
# One row per label, each a complete set of validation predictions carrying the hash of the
# training run behind it and of the predictions themselves, so any row traces back to the
# fitted state that produced it.
#
# Coverage is checked exactly rather than approximately: a factor model reconstructs a
# prediction for every stock-date its loadings cover, so a shortfall means a stock or a
# session the reconstruction could not reach, and that is a fact about the fit rather than a
# rounding difference.

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
).sort("label", "checkpoint_value", "prediction_hash")
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
                "label": run.training.spec()["label"],
                "training_hash": run.training.hash,
                "prediction_hash": prediction.hash,
                "coverage_status": coverage["status"],
                "expected_rows": coverage["n_expected"],
                "actual_rows": coverage["n_actual"],
                "training_artifacts": len(run.training.artifacts()),
                "prediction_artifacts": len(prediction.artifacts()),
            }
        )

coverage_table = pl.DataFrame(coverage_rows).sort("label", "prediction_hash")
if official_population is not None:
    official_population.require_complete()
coverage_table

# %%
execution_diagnostics = pl.DataFrame(execution.diagnostics)
execution_diagnostics

# %% [markdown]
# ## 6. Naming the sets the later notebooks open
#
# One frozen set per label, under a name the later notebooks open by. Only an unnarrowed
# canonical run publishes one: a name must not mean two different member sets at two different
# times, so a run that overrode a parameter, narrowed the labels or ran under the preview tier
# keeps its rows and publishes no name.
#
# These sets are small enough to be compared prediction by prediction, which is why
# [`15_model_analysis`](15_model_analysis.ipynb) does not need a separate bounded subset for
# them the way it does for the larger grids.

# %% tags=["results"]
set_rows = []
is_published_population = (
    EXECUTION_TIER == "canonical" and selected_labels == published_labels and not OVERRIDES
)
if is_published_population:
    for selected_label in selected_labels:
        label_name = selected_label.replace("_", "-")
        result_set = study.predictions.freeze(
            execution.catalog_rows.filter(pl.col("label") == selected_label),
            name=f"us-equities-{label_name}-pca-v1",
        )
        set_rows.append(
            {
                "role": "backtest and diagnostic population",
                "set_name": result_set.name,
                "members": len(result_set.members),
            }
        )
compatible_sets = pl.DataFrame(
    set_rows,
    schema={"role": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `15_model_analysis.py` reopens the named label sets for descriptive analysis. `16_backtest.py`
# passes every catalog row directly to the shared backtest runner. Predictive metrics do not choose
# a configuration or checkpoint.

# %% [markdown]
# ## What to notice
#
# **A factor has no name.** It is a direction in the returns, ordered by how much common variation
# it accounts for. The first one usually looks like the market because the market is what most
# stocks share, but nothing in the method labels it, and reading an economic story into the second
# or third is an interpretation this notebook does not support.
#
# **What this model cannot say is as informative as what it can.** A prediction is built only from
# what a stock shares with the panel, so where it ranks the cross-section well, the ranking is
# coming from common movement rather than from anything specific to a name.
#
# **The loadings belong to the stocks that were there.** A loading is fitted per stock, so a stock
# with no training history in a fold has none, and a stock whose character changes over a decade
# keeps the one it was fitted with. [`13b_ipca`](13b_ipca.ipynb) is the answer to both, and the
# comparison between the two is what the pair is for.
#
# **Known limitations.** The factor count is declared rather than searched, so nothing here says
# the declared one is right. Everything is measured on validation folds read many times over by the
# time a case study reaches this notebook, and ranking accuracy is not strategy performance.
#
# **Next**: [`13b_ipca`](13b_ipca.ipynb) makes a stock's loading a function of what the stock is.
