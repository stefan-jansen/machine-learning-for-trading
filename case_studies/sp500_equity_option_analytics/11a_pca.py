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
# # Option analytics: the factors the returns have, before anything is conditioned on
#
# Every model up to here has predicted the return from the features directly - a weighted sum in
# [`06_linear`](06_linear.ipynb), an ensemble of trees in [`07_gbm`](07_gbm.ipynb), a neural
# ensemble in [`08_tabular_dl`](08_tabular_dl.ipynb), a sequence in
# [`09_dl_lstm`](09_dl_lstm.ipynb). Each asked what a stock's own option surface says about its own
# next five days.
#
# The latent-factor family asks something else first. Suppose the cross-section moves together:
# most of what happens to any one stock in a week is a few common movements it is exposed to,
# rather than something particular to it. Then the object worth estimating is those movements and
# each stock's exposure to them, and a forecast is what the exposures imply.
#
# **PCA is the unconditioned member of that family, and it is deliberately the least informed
# one.** It reads the panel of realized returns and nothing else: no option surface, no
# characteristic, no target. It finds the directions along which the cross-section has historically
# moved most, projects each stock onto them, and forecasts from the training window's factor
# means. The features this case study is built on - implied volatility, skew, the term structure,
# the variance risk premium - reach it nowhere.
#
# That is the point of fitting it. It is the control the four conditioned members are read
# against: [`11b_ipca`](11b_ipca.ipynb) makes the exposures a linear function of the features,
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) makes that function a network,
# and [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) and
# [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) drop the two-stage shape
# altogether. Whatever any of them achieves is worth only the distance between it and this.
#
# **A model with no epochs still has a checkpoint.** `config/pca/pca.yaml` declares
# `checkpoint_interval: 0`, because a principal-component decomposition is solved rather than
# trained: a fold produces one factorization and one set of predictions. That is why the plan below
# shows a single checkpoint where [`08_tabular_dl`](08_tabular_dl.ipynb) shows eight, and it is a
# property of the estimator rather than a reduced setting.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Distinguish predicting a return from estimating exposures that a factor return is applied to.
# - Say what PCA is unable to use, and why that makes it the control rather than a weak competitor.
# - Say why a model that is solved rather than trained still publishes a checkpoint.
# - Read a population published for every declared label rather than for the traded one.
#
# **Book reference**: Chapter 14, Section 14.2 (extracting factors from the return panel).
# Chapter 6, Section 6.7 (Search accounting and run logging) introduces the run log this notebook
# writes into.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds, and
# [`11_latent_factors`](11_latent_factors.ipynb) introduces the family and says which of its five
# members each notebook publishes.
#
# **What it writes**: one training run per label and one complete validation prediction set per
# label, in `run_log/registry.db` and under `run_log/training/` and `run_log/predictions/`, grouped
# under a population named for this model. The family splits across five notebooks, so each
# publishes its own population rather than one shared one.
# [`13_model_analysis`](13_model_analysis.ipynb) compares them against the other families and
# [`14_backtest`](14_backtest.ipynb) selects on validation backtest Sharpe. **Selection happens
# there, not here.**

# %%
"""Fit the declared option-analytics PCA population on the walk-forward folds."""

import plotly.graph_objects as go
import polars as pl

from case_studies.research import (
    declared_labels,
    load_model_configs,
    model_requests,
    open_study,
    plan_models,
    planned_model_plan,
    primary_label,
    run_model_population,
)
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = ""

MODEL_NAME = "pca"

# %%
study = open_study(
    "sp500_equity_option_analytics",
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="11a_pca",
)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and the cell below reads
# which those are rather than this sentence asserting a count that would go stale the moment a
# sixth label declared the family. What the names mean is the part prose has to supply:
# `fwd_ret_5d` is the stock's total return over the five trading days after the decision date,
# `fwd_ret_10d` the same over ten, and `fwd_ret_risk_adj_5d` the five-day return divided by a
# measure of its own dispersion. A `fwd_dir_*` classification label declaring only linear and
# gradient boosting is absent here rather than dropped.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/pca/pca.yaml` declares two things and nothing else. `n_factors` is how many
# leading directions to keep, and it is the one number that decides how much of the panel's
# co-movement the model is allowed to use. `checkpoint_interval: 0` says there is no intermediate
# state to publish, for the reason given above.
#
# Both reach the fit from this preset, and two things can override them. The case study may declare
# per-model values under `modeling.latent_factors.model_kwargs` in `config/setup.yaml`, which win
# where they are given; this one declares entries for `ipca` and `sdf` only, so PCA's values are
# the preset's. A reduced run may then override either through `PREVIEW_REDUCTIONS`, where the
# reduction becomes part of the preview identity. The order is setup, then preset, then a built-in
# default that is only reached when neither declares the value.

# %%
configs = load_model_configs(
    study,
    "latent_factors",
    labels=LABELS or None,
    config_names=[MODEL_NAME],
)
configs

# %% [markdown]
# `LABELS` narrows what is fitted, and a narrowed run declares a different set of members than the
# canonical population does. A population is immutable once written, so such a run must publish
# under its own name: on a fresh workspace it would otherwise register an incomplete snapshot under
# the canonical one, and where the full population already exists the registry refuses it.
#
# The comparison is against this model's own declared rows rather than the whole `latent_factors`
# catalog. The family is split across five notebooks and each publishes one model, so the complete
# catalog is five times what this one publishes, and comparing against it would report every
# canonical run as narrowed.

# %%
declared = load_model_configs(study, "latent_factors", config_names=[MODEL_NAME])
if set(configs.get_column("label")) != set(declared.get_column("label")) and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the {declared.height} declared labels, so it cannot "
        "publish the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ### Where this one runs, and why it is not the family's device
#
# `config/setup.yaml` declares the latent-factor family on CUDA, which is right for the three
# neural members. PCA is not one of them: `run_pca_fold` is a numpy and scipy decomposition and
# takes no device at all. The device nonetheless sits inside the hashed computation rather than
# beside it as provenance, so leaving the family's value in place would record a computation that
# did not happen and would make this population's identity depend on whether the machine had a GPU.
# It is declared here instead, and a run on another device has to publish under its own name.

# %%
PUBLISHED_DEVICE = "cpu"
overrides = {"device": PUBLISHED_DEVICE}
print(f"training device: {PUBLISHED_DEVICE}")

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry names an estimator and a factor count. It does not say which feature columns exist
# today, where the walk-forward folds fall, or which symbol-date pairs have both a return and a
# label. **Planning** works all of that out - and every training and prediction identity with it -
# without fitting anything. Four things to check:
#
# - **`feature_count` and `eligible_rows` agree across the rows.** A row that differs is a label
#   measured on a different sample from its neighbours. They differ *between* labels, because a
#   ten-day forward window runs out earlier than a five-day one.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   [`05_evaluation`](05_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample**, with none of the
#   held-out 2021 tail visible.
# - **`checkpoints` is 1**, for the reason the configuration gives above.
#
# `feature_count` is worth a second look here rather than a glance. It counts the columns the
# request carries, and PCA uses none of them: the decomposition is of the return panel. The column
# that matters to this model is the return itself, and the count above is a property of the request
# shared with its siblings rather than of the fit.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    overrides=overrides,
    preview_reductions=PREVIEW_REDUCTIONS,
)
plan = plan_models(study, requests=requests)

planned_model_plan(plan).select(
    "label",
    "config_name",
    "task",
    "feature_count",
    "eligible_rows",
    "folds",
    "checkpoints",
    "validation_start",
    "validation_end",
)

# %% [markdown]
# The plan is also where a short population is visible. A run that fitted fewer labels than the
# menu declares would still print an IC table and still register its rows; what it would not do is
# announce the gap.

# %%
planned_pairs = {(member.label, member.config_name) for member in plan.members}
requested_pairs = set(zip(configs["label"], configs["config_name"], strict=True))
if planned_pairs != requested_pairs:
    raise RuntimeError(
        "the plan does not match the loaded PCA menu; "
        f"missing {sorted(requested_pairs - planned_pairs)}, "
        f"unexpected {sorted(planned_pairs - requested_pairs)}"
    )
print(f"{len(requested_pairs)} label-configuration pairs")
print(f"{len(plan.expected_training_hashes)} training identities")
print(f"{len(plan.expected_prediction_hashes)} validation prediction sets")

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` runs every planned request. For one request it walks the folds, and on
# each one:
#
# 1. takes the returns of the stocks inside that fold's training window, arranged as a panel of
#    dates by stocks,
# 2. finds the `n_factors` directions along which that panel varies most, which are the eigenvectors
#    of its covariance matrix in descending order of eigenvalue, and each stock's loading on them,
# 3. forecasts each validation date from those loadings and the training window's mean factor
#    returns.
#
# Step 3 is what makes this a forecast rather than a description. The loadings and the factor means
# come from the training window only, and the validation dates contribute nothing to either - which
# matters more here than for a supervised model, because a decomposition fitted on the whole sample
# would place every stock using returns it had not yet earned.
#
# **This notebook passes the plan rather than resolved requests, and on this family that changes
# what the plan table can show rather than how the fitting is ordered.** The linear, boosted and
# TabM adapters each provide a batch runner that prepares one fold set for several configurations;
# the latent-factor adapter provides none, so every request is fitted on its own whichever path is
# taken. There would be nothing to share in any case: this notebook publishes one model against
# three labels, and each label is a different set of rows. What passing the plan costs is
# `eligible_entities`, which needs the eligibility keys themselves; `eligible_rows` above moves
# whenever the universe does, so the check that column existed for is still made.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, written down before the first fit. Afterwards every member must exist and be
# complete, which is what makes the downstream comparison well defined.


# %%
def catalog_labels(execution) -> int:
    """Number of distinct labels the execution published, read from its catalog rows."""
    return execution.catalog_rows.get_column("label").n_unique()


population_name = POPULATION_NAME or f"sp500_equity_option_analytics-{MODEL_NAME}-validation-v1"
execution, population = run_model_population(
    study, plan, population_name=population_name, supersedes=SUPERSEDES_POPULATION or None
)

# No per-fold counts: this family's runner reports none, and two zeros would read as a failed fit.
print(f"{len(execution.runs)} training runs across {catalog_labels(execution)} labels")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# `reused` is not zero on a second run. Every identity is re-derived from the inputs, the registry
# already holds the matching rows, and the runner returns the stored result rather than fitting
# again - so re-running this notebook unchanged costs the time it takes to read the data.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which holds
# its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("sp500_equity_option_analytics", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "latent_factors", labels=["fwd_ret_5d"], config_names=["pca"]
# )
# requests = model_requests(study, configs, overrides={"device": "cpu"})
# plan = plan_models(study, requests=requests)
# execution, population = run_model_population(study, plan, population_name="my-pca-v1")
# ```
#
# To change the number of factors for your own run, edit `case_studies/config/pca/pca.yaml`, or
# declare `n_factors` under `modeling.latent_factors.model_kwargs.pca` in `config/setup.yaml` to
# override the preset for this case study alone. Note which of those you want: the preset is shared
# by every case study that declares `pca`, so editing it moves the training identity of all of
# them. Either way the result registers as a new row beside the old one rather than replacing it.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per label. The **information coefficient** is the rank correlation, on one validation
# date, between the stocks ordered by the model's prediction and the stocks ordered by the return
# they went on to earn.
#
# `ic_mean` aggregates that **over folds, not over days**: each fold's own mean IC is computed and
# those are averaged with equal weight (`latent_factors/cv.py`, and
# `registry/metrics.py` states the convention). With folds of unequal length the fold mean and the
# pooled daily mean are different numbers, and this column is the first.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it decides which rows
# are comparable with each other. `auc_scored_against` says what the AUC column was scored against:
# a regression row has no classes of its own and is scored as a ranking signal against a declared
# direction sibling, so `fwd_ret_5d` is scored against `fwd_dir_5d` and `fwd_ret_10d` against
# `fwd_dir_10d`. `fwd_ret_risk_adj_5d` declares no sibling and carries no AUC; null there means not
# computed, not zero.
#
# The published catalog is checked against the population planned before fitting rather than against
# its own row count, because a run that lost a member would otherwise report a shorter table and
# nothing else.

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "config_name",
    "label",
    "task",
    "complete",
    "checkpoint_kind",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_n_days",
    "auc_mean_daily",
    pl.col("direction_label").alias("auc_scored_against"),
    "n_folds",
    "training_hash",
    "prediction_hash",
).sort("label")

if set(catalog.get_column("prediction_hash")) != set(plan.expected_prediction_hashes):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("a partial PCA prediction set cannot pass to backtesting")
if catalog.select("label", "config_name", "checkpoint_value").n_unique() != catalog.height:
    raise RuntimeError("each label and checkpoint must identify one prediction set")

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders by whichever
# label it did fit rather than by one that is not there.
ordered_labels = [label for label in [primary] if label in present] + [
    label for label in present if label != primary
]
print(f"{catalog.height} candidate models across {len(ordered_labels)} labels")
catalog.select(
    "label",
    "task",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_n_days",
    "auc_mean_daily",
    "auc_scored_against",
)

# %% [markdown]
# ### One decomposition, three targets
#
# The returns decomposed, the folds and the factor count are identical across these rows. What
# changes is only which forward return the factor-implied forecast is scored against, which makes
# this the cleanest comparison in the notebook: the model does not know what it is being scored on.
#
# `ic_t` is the t-statistic across those fold means, not a Newey-West statistic on the daily
# series - the registry keeps the HAC-corrected version separately as `ic_t_hac`, and that is the
# inferential one. Either way it is a diagnostic and not a selection rule: the series is short,
# overlapping multi-day returns make successive days dependent, and the folds have been read many
# times over by the time a case study reaches this notebook.

# %% tags=["results"]
by_label = catalog.select(
    "label",
    "task",
    ic_mean=pl.col("ic_mean"),
    ic_t=pl.col("ic_t"),
    scored_dates=pl.col("ic_n_days"),
    # `ic_n_days` counts the days behind `ic_mean_daily`, the pooled daily statistic - not the
    # folds behind `ic_mean`, which is what the rows are ordered by. So this column is a
    # comparability guarantee about one statistic attached to a ranking on another. It is kept
    # because unequal day counts are still the thing that makes two rows incomparable, and
    # flagged because the two are not the same measurement: `registry/metrics.py:203` averages
    # folds for `ic_mean`, and `:234-250` computes the daily family together. Reading the
    # ordering on `ic_mean_daily` would need `PredictionCatalog` to carry it.
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max(),
).sort("ic_mean", descending=True)
by_label

# %% [markdown]
# ### The same decomposition against each target
#
# One bar per label, on one axis, with the primary target first. The bars are the quantity the
# downstream comparison uses; the axis is shared so the distance between them is readable rather
# than each panel filling itself.

# %%
panel = catalog.with_columns(
    rank=pl.col("label").replace_strict(
        {label: index for index, label in enumerate(ordered_labels)}, return_dtype=pl.Int32
    )
).sort("rank")
fig = go.Figure(
    go.Bar(
        x=panel.get_column("label").to_list(),
        y=panel.get_column("ic_mean").to_list(),
        marker_color=[
            COLORS["blue"] if label == primary else COLORS["slate"]
            for label in panel.get_column("label")
        ],
        showlegend=False,
    )
)
fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.update_yaxes(title_text="Mean IC (validation)")
fig.update_xaxes(title_text="Label, primary target first")
fig.update_layout(
    title="What the return factors rank depends on which forward return is asked for",
    height=420,
    width=800,
    margin=dict(t=90),
)
# Which side of zero each bar sits on is a fact about the frame, so the alt text reads it rather
# than asserting a shape the next run may not reproduce.
sides = "; ".join(
    f"{row['label']} {'above' if row['ic_mean'] > 0 else 'below'} zero"
    for row in panel.select("label", "ic_mean").iter_rows(named=True)
)
show_plotly_with_alt(
    fig,
    "Bar chart of mean validation information coefficient, one bar per label on one axis, with the "
    "primary target in dark navy first and the two variants in slate, and a dashed zero line. "
    f"Counted from the frame: {sides}.",
)

# %% [markdown]
# ## 5. What to notice
#
# **The estimator is the same and only the target moves, which is what makes these rows
# comparable.** They share the return panel, the folds, the dates scored and the factor count.
# `fwd_ret_10d` is the same construction over twice the horizon, and `fwd_ret_risk_adj_5d` is
# `fwd_ret_5d` divided by a measure of its own dispersion - so a gap between those two rows is a
# statement about scaling by width and nothing else.
#
# **PCA cannot see the features this case study is about, and that is what it is for.** No implied
# volatility, no skew, no term structure, no variance risk premium reaches the decomposition. Read
# every conditioned member of this family against it: [`11b_ipca`](11b_ipca.ipynb) adds a linear map
# from the features to the exposures and nothing else, so the distance between the two is what
# conditioning on the surface buys, measured rather than assumed.
#
# **A model with no epochs still has a checkpoint, and the checkpoint is not a formality.** The
# registry keys a prediction set on `(training identity, checkpoint)`, so a family whose members
# publish one, five and ten checkpoints each needs one convention rather than three. PCA publishes
# at zero because the decomposition is solved: there is no intermediate state a reader could have
# stopped at, and inventing one would offer a choice the estimator does not have.
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) is the contrast, where the
# stopping point is real and is published.
#
# **What a factor model buys over predicting the return directly is a constraint, not more
# capacity.** The supervised models had one weight per feature and one cross-section at a time to
# find them in. A factor model says the cross-section is a few common movements plus noise, and
# estimates far fewer numbers against far more data. That is the entire argument for the family, and
# it is also its limitation: where the constraint is wrong it is wrong everywhere at once rather
# than merely tight.
#
# **None of this selects anything.** IC measures whether predictions rank stocks correctly, not
# whether a strategy trading them makes money after costs and turnover, and every label's
# prediction set stays in the published population for that reason. Selection is on validation
# backtest Sharpe in [`14_backtest`](14_backtest.ipynb).
#
# **Known limitations.** Two folds, one of them validating on 2020, a year in which the
# cross-section co-moved unlike any other in the sample - which bears on a factor model more
# directly than on a supervised one, because co-movement is the thing being estimated. The number
# of factors is declared rather than chosen, and nothing here tests whether a different count would
# order the cross-section better; that would be a search, and a search over validation IC is what
# this notebook is arranged to avoid. The IC carries no adjustment for the serial dependence
# overlapping multi-day returns create, so it is a ranking diagnostic rather than a test. And PCA
# needs a stock to be the same stock across the training window, which this case study's
# `persistent_entities` declaration asserts and which a heavily reconstituted universe would not
# support.
#
# **Next**: [`11b_ipca`](11b_ipca.ipynb) keeps this structure - exposures times factor returns - and
# makes the exposures a linear function of each stock's own option-surface features, so the
# difference between the two notebooks is one map and nothing else.
