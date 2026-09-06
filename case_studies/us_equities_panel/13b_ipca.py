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
# # US equities panel: loadings that depend on what a stock is
#
# [`13a_pca`](13a_pca.ipynb) gave every stock a **loading** on each common movement - a number
# saying how much of that movement the stock takes - and fitted that number to the stock. Two
# limitations follow directly, and neither is technical.
#
# **A number fitted to a stock cannot answer for a stock it never saw.** On a panel where names
# list and delist across three decades, that is a large part of the universe.
#
# **A number fitted to a stock cannot move when the stock does.** A company that was small and
# cheap ten years ago and is large and expensive now keeps the loading its whole training window
# implied.
#
# **Instrumented principal component analysis makes the loading a function of the stock's
# observable characteristics instead.** Rather than fitting a number per stock, it fits a mapping
# from characteristics to loadings, shared across the whole panel. So a stock that was never seen
# still has a loading, computed from its characteristics; and a stock whose characteristics change
# has its loading change with them, without anything being refitted.
#
# **What that buys is paid for with an assumption.** The mapping is the same for every stock and
# is held fixed across the training window, so the method assumes the relation between what a
# stock is and how it moves is stable and universal. Where that fails - a characteristic that
# means something different in one decade than another - the conditioned loadings are wrong in a
# way the unconditioned ones are not.
#
# **Everything is fitted on training rows only**, the mapping included, for the reason
# [`13a_pca`](13a_pca.ipynb) gives: a relation estimated from the whole sample would have been
# estimated partly from the returns it is later asked to predict.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - State the two limitations of a per-stock loading, and say why each is a problem on a panel
#   whose membership and whose companies both change.
# - Explain what it means for a loading to be a function of characteristics rather than a number
#   attached to a stock.
# - Say what the conditioned model assumes in exchange, and describe a situation where that
#   assumption would be uncomfortable.
# - Say why the mapping has to be estimated inside the training window.
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
"""Generate IPCA validation predictions through the shared research interface."""

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
PREVIEW_MAX_ITER = 0

# %% [markdown]
# ## 1. Which labels, and how many factors
#
# What each setting a run may pass decides:
#
# - **`LABELS`** empty fits the primary label and every declared variant. A subset fits only those.
# - **The factor count** is how many common movements are extracted. It is read from the preset at
#   `case_studies/config/ipca/ipca.yaml` rather than set here, so there is one place to change it
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
    if "ipca" not in configured:
        raise ValueError(f"IPCA is not configured for {label}")
    declared_factors.add(int(configured["ipca"]["params"]["n_factors"]))
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
if PREVIEW_MAX_ITER:
    preview_reductions["max_iter"] = int(PREVIEW_MAX_ITER)

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
        config_name="ipca",
        # The preset supplies the factor count; a value passed here would win over it, which is
        # what OVERRIDES is for and what makes such a run unpublishable.
        overrides=dict(OVERRIDES),
        execution_tier=EXECUTION_TIER,
        preview_reductions=preview_reductions,
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
# The fit itself is iterative: factors and the characteristic mapping are estimated in alternation
# until they stop moving. Whether that happened is recorded per fold, and it is a statement about
# the stability of the estimate rather than about whether the mapping it found is any good.
#
# The planner resolves every label-specific training and checkpoint identity before fitting and
# writes the canonical checkpoint population first. Each fold then fits IPCA on its training panel
# only. The runner validates convergence, persists the
# fitted factor state, and reconstructs each registered prediction set from those artifacts before
# accepting cached work.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-ipca-checkpoints-v1",
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
            "max_iter": computation["model"]["params"]["max_iter"],
            "fold_workers": computation["runtime"]["fold_workers"],
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
            name=f"us-equities-{label_name}-ipca-v1",
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
# **Read this against [`13a_pca`](13a_pca.ipynb) rather than on its own.** The two extract the same
# number of factors from the same panel over the same folds, and differ in whether a loading is
# fitted to a stock or computed from what the stock is. That difference is the only thing a
# comparison between them measures.
#
# **The characteristics are doing the work, so which ones are declared matters.** A characteristic
# absent from the feature set cannot inform a loading, and one that is present but stale informs it
# wrongly. This is the model in the case study most sensitive to what stage 03 chose to compute.
#
# **A convergence check is not a goodness check.** The fit alternates between the factors and the
# mapping until they stop moving, and the runner refuses a fold that never got there rather than
# reporting it - so every result you see converged. That says the estimate is stable, not that the
# mapping it found is right.
#
# **Known limitations.** The factor count is declared rather than searched. The mapping from
# characteristics to loadings is assumed common across stocks and fixed within a training window,
# which is the assumption the whole method rests on and which nothing here tests. And everything is
# measured on validation folds read many times over by the time a case study reaches this notebook.
