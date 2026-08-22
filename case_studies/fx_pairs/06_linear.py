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
# # FX pairs: regularizing when the cross-section itself is dependent
#
# The 20 pairs in this universe are not 20 independent assets. They are quotes between eight
# currencies, so `EUR_USD`, `GBP_USD` and `EUR_GBP` are related by an accounting identity: hold
# the first two in the right proportions and you have the third. A dollar move shows up in every
# pair with USD on one side, in the same direction, at the same moment.
#
# That is a different problem from a collinear feature set, and it is worth separating the two.
# **Collinear features** are columns of the design matrix that carry nearly the same information,
# and this universe has those too - seven return windows, eight volatility windows, seven rolling
# Sharpe ratios, and eight cross-sectional ranks computed from them. **A dependent cross-section**
# is something else: the rows are linked, so ranking the pairs against each other is ranking
# overlapping bets rather than distinct ones. A model that discovers "the dollar is strong" scores
# well on the ranking and has found one fact, not twenty.
#
# **Regularization** - adding a penalty on coefficient size to the fitting objective - addresses
# the first problem directly and the second not at all. This notebook runs the penalty sweep and
# the results are read with that distinction in mind.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read the set of models a case study has declared for a label, and say which estimator and
#   which hyperparameters each declared name resolves to.
# - Bind those declarations to the data on disk and check, before anything is fitted, that every
#   configuration will be measured on the same pairs, the same dates and the same folds.
# - Fit a population of models on walk-forward folds and publish one complete set of validation
#   predictions per configuration.
# - Tell apart the two things a penalty can do to a correlated feature set - shrink correlated
#   coefficients towards each other, or select a few features and zero the rest - and read from
#   the results which one this data rewards.
# - Distinguish collinearity among the features from dependence among the assets being ranked,
#   and say which one a penalty can fix.
# - Run configurations of your own into a private copy of the run log, and have them compared on
#   the same footing as the ones shipped here.
#
# **Book reference**: Chapter 11 (The ML Pipeline). Chapter 6, Section 6.7 (Search accounting and
# run logging) introduces the run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# and [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds.
#
# **What it writes**: one training run and one complete validation prediction set per
# configuration, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population.
# [`13_backtest`](13_backtest.ipynb) reads that population, runs every member against the
# equal-weight baseline, and selects on validation backtest Sharpe. **Selection happens there,
# not here.** This notebook ranks configurations by information coefficient to show what
# regularization does to a correlated feature set; that ranking decides nothing.

# %%
"""Fit the declared FX linear-model population on the walk-forward validation folds."""

import re

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.research import (
    declared_labels,
    load_model_configs,
    model_requests,
    open_study,
    primary_label,
    resolved_model_plan,
    run_model_population,
)
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
CONFIG_NAMES: list[str] = []
POPULATION_NAME = ""

# %%
study = open_study("fx_pairs", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which label, and which models
#
# A label is the thing being predicted. This case study defines three in `config/setup.yaml`:
# `fwd_ret_1d`, the return over the trading day after the decision date, and `fwd_ret_5d` and
# `fwd_ret_21d` over longer horizons. The one-day label is the primary one - the horizon the
# strategy chapters trade - and the other two are variants, kept so the effect of the prediction
# horizon can be examined separately.
#
# **This notebook fits every declared label.** Each has its own training menu at
# `config/training/{label}.yaml`, listing family by family the named configurations to fit for
# that label, and the notebook fits the union of them. A label with no menu file has nothing
# declared and nothing to fit; the labels below are the ones that declare linear models.
#
# Fitting all three in one run is what makes the horizon comparable. The same penalty grid on
# three different labels, with the features, the folds and the pairs held fixed, isolates what
# the prediction horizon does; running one label at a time leaves that comparison to be
# assembled by hand from three runs, and until it is assembled the variants are declared and
# never fitted. `LABELS` restricts the run to a subset when you want one, and defaults to
# everything the menus declare.

# %%
declared_labels(study, "linear")


# %% [markdown]
# Each name in the menu resolves to a preset file in the shared directory
# `case_studies/config/{model_type}/`, which holds that configuration's hyperparameters. The
# frame below is the menu for every label above, with each name resolved to the estimator class
# it names and the arguments that class is constructed with. To change what runs, edit the menu
# or the presets rather than this notebook.
#
# The grid covers the two shapes a penalty can take:
#
# - **Ridge** penalizes the sum of squared coefficients. It shrinks correlated coefficients
#   towards each other and keeps every feature, at a strength set by `alpha`. The grid steps
#   `alpha` by powers of ten across ten orders of magnitude, because the useful value depends on
#   the scale and the collinearity of the design matrix and neither is known in advance.
# - **Lasso** penalizes the sum of absolute coefficients, which drives some of them exactly to
#   zero: it selects features rather than shrinking them. **ElasticNet** mixes the two.
#
# Lasso and ElasticNet are parameterized here by `alpha_frac` rather than a raw penalty. For any
# fold there is a threshold penalty $\alpha_{\max}$ - the smallest one that zeros every
# coefficient - which is computed from that fold's own data. `alpha_frac` is the fraction of it
# to apply, so one declared `alpha_frac` means the same thing on every fold, while a fixed raw
# penalty would mean something different on each.

# %%
configs = load_model_configs(
    study,
    "linear",
    labels=LABELS or None,
    config_names=CONFIG_NAMES or None,
)
configs

# %% [markdown]
# `LABELS` and `CONFIG_NAMES` both narrow what is fitted, and a narrowed run declares a different
# set of members than the canonical population does. A population is immutable once written, so
# such a run must publish under its own name: on a fresh workspace it would otherwise register an
# incomplete snapshot under the canonical one, and where the full population already exists the
# registry refuses it. Comparing the loaded rows against the complete declared catalog catches
# either knob, and says so here rather than several cells later in a message about hashes.

# %%
if configs.height < load_model_configs(study, "linear").height and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the declared configurations, so it cannot publish "
        "the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry says which estimator to fit. It does not say which feature columns exist today,
# where the walk-forward folds fall, or which pair-date pairs have both a feature row and a
# label. **Resolving** a request is the step that goes and finds all of that: it reads the label
# and feature files, computes the fold boundaries from the walk-forward parameters in
# `config/setup.yaml`, works out the exact set of rows each fit is expected to predict, and turns
# any data-dependent hyperparameter into the number it will actually use - each fold's own
# $\alpha_{\max}$ times `alpha_frac`, in the case of Lasso.
#
# Resolving reads the inputs and fits nothing, so the plan below can be inspected before any
# computation starts. The three things to check in it:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row.** They are
#   the width of the design matrix, the number of currency pairs, and the number of pair-date
#   pairs to be predicted. Every configuration here reads the same feature matrix, so a row that
#   differs is a configuration being measured on a different sample from its neighbours, and its
#   results are not comparable with theirs.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   `05_evaluation` established.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   must not appear here: it is scored once, at the end of the case study, and any of it visible
#   in this window would mean it had been used to choose something.
#
# Each row also carries a `training_hash`: the identity of that computation, derived from
# everything that can change its result. [`RUN_LOG.md`](../RUN_LOG.md#identity) sets out what goes
# into one and what follows from it.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
resolved = tuple(request.resolve() for request in requests)

plan = resolved_model_plan(resolved)
plan.select(
    "config_name",
    "feature_count",
    "eligible_entities",
    "eligible_rows",
    "folds",
    "validation_start",
    "validation_end",
)

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` fits every resolved request. For one request it walks the folds, and on
# each one:
#
# 1. takes the rows inside that fold's training window,
# 2. fills missing feature values with the training window's median for that column, then
#    standardizes each column to zero mean and unit variance - both fitted on training rows only
#    and then applied to the validation rows, so nothing from the validation window reaches the
#    fit,
# 3. fits the estimator with that fold's resolved parameters,
# 4. predicts the fold's validation rows.
#
# The fold predictions are concatenated into one series covering the whole validation period,
# which is what a walk-forward prediction set is: each date predicted by a model that saw only
# data before it. The run then writes a `training_runs` row and the fitted coefficients, a
# `prediction_sets` row and the predictions themselves, and the metrics computed from them. It
# does this per configuration rather than once at the end, so an interruption costs the
# configuration in flight and nothing else.
#
# Every case study that fits linear models calls this same runner, which is what makes their
# results comparable. Unlike gradient boosting or a neural network, a linear model has no
# intermediate states worth scoring: there is one fit and therefore one checkpoint per
# configuration, and no learning curve to plot.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# is going to produce. The list is computed from the resolved specifications before the first fit
# and written down, and afterwards every member must exist and be complete. It is why a
# configuration that raises fails the whole call rather than publishing a population one member
# short.
#
# **One population covers every label**, because one run fits every label and the population is
# what that run declares. A population is immutable once written: registering a different set of
# members under a name that already exists is refused unless the caller names the snapshot it
# supersedes. A notebook that fitted one label per run under a single name would therefore
# publish the first label and be refused for the second, which is what happened before this
# notebook fitted them together. Everything that finished stays registered, and re-running fits
# only what is missing.

# %%
population_name = POPULATION_NAME or "fx_pairs-linear-validation-v1"
execution, population = run_model_population(study, resolved, population_name=population_name)

fitted = sum(len(item["fitted_folds"]) for item in execution.diagnostics)
reused = sum(len(item["reused_folds"]) for item in execution.diagnostics)
print(f"{len(execution.runs)} configurations: {fitted} folds fitted, {reused} reused")
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# `reused` is not zero on a second run. Every identity is re-derived from the inputs, the
# registry already holds the matching rows, and the runner returns the stored result rather than
# fitting again - so re-running this notebook unchanged costs the time it takes to read the data.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which
# holds its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("fx_pairs", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "linear", labels=["fwd_ret_1d"], config_names=["ols", "ridge_a1.0", "ridge_a3.0"]
# )
# requests = model_requests(study, configs)
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-linear-v1")
# ```
#
# `CONFIG_NAMES` fits a subset of what the menu already declares; a name the menu does not
# declare raises rather than quietly fitting fewer models than you asked for. To fit something
# new, add a preset at `case_studies/config/ridge/ridge_a3.0.yaml` and list `ridge_a3.0` under
# `linear:` in the label's menu. Editing an existing preset changes that configuration's
# identity, so its result registers as a new row beside the old one instead of replacing it.
#
# Give the run its own `population_name`: a name refers to one set of members permanently, and
# reusing it for a different set raises. Everything downstream reads the registry rather than the
# notebook, so predictions produced this way are selected and backtested on the same footing as
# the ones shipped here, inside your workspace.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest, including how to
# rehearse on a reduced universe first.

# %% [markdown]
# ## 4. What came out
#
# One row per configuration, read back from the registry. `ic_mean` is the **information
# coefficient**: on each validation date, rank the pairs by the model's prediction, rank them by
# the return they went on to earn, correlate the two rankings, and average that daily correlation
# over the validation period. It measures whether the model ranks pairs correctly, on a scale
# where zero is no relationship, positive means the ranking points the right way, and negative
# means it points the wrong way.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it is not a
# footnote. A model whose coefficients collapse to one or two features predicts nearly the same
# value for every pair on some dates; a constant has no rank correlation with anything, so those
# dates contribute nothing. Its `ic_mean` would then be an average over fewer dates than its
# neighbours', chosen by where it happened to stay non-degenerate, and comparing it with theirs
# would compare two different samples. `full_coverage` marks the configurations measured on all
# of them - the column is there so you can see whether that happened, not because it always does.
#
# **Coverage is judged within a label, not across them.** The three horizons do not have the same
# number of scorable validation dates to begin with: a 21-day label runs out of forward window
# earlier than a one-day label does, so it has fewer dates before any model is fitted. Comparing
# every configuration against one global maximum would mark the entire 21-day grid incomplete for
# a reason that has nothing to do with the models. The reference is each label's own maximum, and
# `full_coverage` then means what it says: measured on every date that label offers.

# %% tags=["results"]
catalog = (
    execution.catalog_rows.select(
        "config_name",
        "label",
        "complete",
        "ic_mean",
        "ic_std",
        "ic_n_days",
        "n_folds",
        "training_hash",
        "prediction_hash",
    )
    .sort(["label", "ic_mean"], descending=[False, True])
    .join(
        configs.select("config_name", "label", "model_class", "params"),
        on=["config_name", "label"],
        how="left",
    )
)

if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("linear execution returned a partial prediction set")

catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)
catalog.select(
    "label",
    "config_name",
    "model_class",
    "params",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "full_coverage",
)

# %% [markdown]
# The configurations left out of the charts below, with the number of dates each was measured on
# against its label's maximum. An empty frame means every configuration scored every date its
# label offered.

# %% tags=["results"]
catalog.with_columns(label_dates=pl.col("ic_n_days").max().over("label")).filter(
    ~pl.col("full_coverage")
).select("label", "config_name", "model_class", "ic_mean", "ic_n_days", "label_dates")

# %% [markdown]
# ### What each horizon reached
#
# One row per label, over the configurations that horizon actually charted. It is the frame the
# horizon comparison below is read from, so the numbers quoted there are these and not a
# different subset: `best_ic` is the best full-coverage result at that horizon, which is not the
# same as the best result at that horizon when a partial-coverage configuration scores higher on
# fewer dates.

# %% tags=["results"]
horizons = (
    catalog.filter("full_coverage")
    .group_by("label")
    .agg(
        charted=pl.len(),
        scorable_dates=pl.col("ic_n_days").max(),
        best_config=pl.col("config_name").sort_by("ic_mean", descending=True).first(),
        best_ic=pl.col("ic_mean").max(),
        worst_ic=pl.col("ic_mean").min(),
        above_zero=(pl.col("ic_mean") > 0).sum(),
    )
    .sort("label")
)
horizons

# %% [markdown]
# ### How the penalty grid ranks
#
# One panel per label, so the same grid can be read at each horizon. Only the configurations
# measured on all of their own label's validation dates are charted; any partial-coverage ones
# are in the table above with `full_coverage` false, and are left out here because their IC would
# be an average over a different set of dates. The zero line is the reference that matters: a bar
# below it is a model whose ranking pointed the wrong way out of sample.
#
# **Each panel has its own vertical scale**, because the horizons do not produce ICs of the same
# size and a shared scale would flatten the one-day panel to a line. Read a panel for the shape
# of its ranking, and the numbers rather than the bar heights when comparing panels.
#
# **The configurations are in the same order in all three panels**, that order being their
# ranking on the primary label. Sorting each panel independently would put a different
# configuration at each left edge and make the panels look alike whatever the numbers did. Held
# in one order, a panel that slopes down from left to right is a horizon that ranks the grid the
# way the primary label does, and a panel that does not is one where the penalty that works at
# one horizon does not work at another.


# %%
def compact(params: str) -> str:
    """Render declared parameters for a label: `alpha=1000000.0` reads as `alpha=1e+06`."""
    return re.sub(r"\d+\.?\d*(?:[eE][+-]?\d+)?", lambda m: f"{float(m.group()):g}", params)


primary = primary_label(study)
full = catalog.filter("full_coverage")
charted = sorted(set(full.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders the panels
# by whichever label it did fit rather than by one that is not there.
panel_labels = [label for label in [primary] if label in charted] + [
    label for label in charted if label != primary
]
order_label = panel_labels[0]
config_order = (
    full.filter(pl.col("label") == order_label)
    .sort("ic_mean", descending=True)
    .get_column("config_name")
    .to_list()
)
leader = full.sort("ic_mean", descending=True).row(0, named=True)

fig_ic = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    panel = full.filter(pl.col("label") == label)
    fig_ic.add_trace(
        go.Bar(
            x=panel.get_column("config_name").to_list(),
            y=panel.get_column("ic_mean").to_list(),
            marker_color=[
                COLORS["amber"]
                if (name, label) == (leader["config_name"], leader["label"])
                else COLORS["blue"]
                for name in panel.get_column("config_name")
            ],
        ),
        row=row,
        col=1,
    )
    fig_ic.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
    fig_ic.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig_ic.update_xaxes(
    categoryorder="array",
    categoryarray=config_order,
    tickangle=-45,
    title_text=f"Configuration (ordered by validation IC on {order_label})",
    row=len(panel_labels),
    col=1,
)
fig_ic.update_layout(
    title="Validation IC across the full-coverage penalty grid, by label",
    height=320 * len(panel_labels),
    width=1100,
    showlegend=False,
    margin=dict(t=90),
)
show_plotly_with_alt(
    fig_ic,
    f"Stacked bar charts of mean validation information coefficient across the linear penalty "
    f"grid, one panel per label and the configurations in the same order in each, that "
    f"order being their ranking on {order_label}. Each panel carries a dashed zero line. The "
    f"highest "
    f"bar anywhere is {leader['config_name']} ({compact(leader['params'])}) on {leader['label']} "
    f"at IC {leader['ic_mean']:+.3f}, highlighted in amber.",
)

# %% [markdown]
# ### Whether the ranking transfers between horizons
#
# The panels are held in one order so the question can be asked, and this answers it with a
# number rather than an impression: the rank correlation between two labels' orderings of the
# configurations they both charted. One would mean the horizons agree on the whole grid, zero
# that the ordering at one horizon says nothing about the other.
#
# Each pair is built from its own two labels. One frame pivoted across all three and then
# dropped would intersect every pair down to the configurations charted everywhere, so a pair
# that shares its whole grid would still be reported on the smaller set, and `configurations`
# would print the same number for all three pairs whatever their coverage was.


# %% tags=["results"]
def rank_transfer(label_a: str, label_b: str) -> dict:
    """Rank correlation between two labels' orderings, over what both of them charted."""
    pair = (
        full.filter(pl.col("label").is_in([label_a, label_b]))
        .select("label", "config_name", "ic_mean")
        .pivot(on="label", index="config_name", values="ic_mean")
        .drop_nulls()
    )
    return {
        "label_a": label_a,
        "label_b": label_b,
        "configurations": pair.height,
        "rank_correlation": pair.select(
            pl.corr(pl.col(label_a).rank(), pl.col(label_b).rank())
        ).item(),
    }


transfer = pl.DataFrame(
    [rank_transfer(a, b) for index, a in enumerate(panel_labels) for b in panel_labels[index + 1 :]]
)
transfer

# %% [markdown]
# ### What shrinkage does on its own
#
# The bar chart mixes three estimators. Tracing IC across the Ridge penalty alone isolates the
# effect of shrinkage, with the estimator, the features and the folds all held fixed and only
# `alpha` moving. The alpha is read from each configuration's declared parameters rather than
# parsed out of its name, so the curve plots what was fitted.
#
# One curve per label puts the three horizons on one pair of axes, which is the comparison
# fitting them together exists to make: whether the penalty that helps most is the same strength
# at every horizon, and whether the curve has the same shape at all.

# %%
ridge = (
    catalog.filter(pl.col("model_class") == "Ridge")
    .with_columns(
        alpha=pl.col("params").str.extract(r"alpha=([0-9.eE+-]+)").cast(pl.Float64),
    )
    .drop_nulls("alpha")
    .sort("label", "alpha")
)
if ridge.height:
    ridge_labels = [label for label in panel_labels if label in set(ridge.get_column("label"))]
    curve_colors = [COLORS["blue"], COLORS["copper"], COLORS["amber"]]
    fig_alpha = go.Figure()
    peaks = {}
    for index, label in enumerate(ridge_labels):
        series = ridge.filter(pl.col("label") == label)
        log_alpha = np.log10(series.get_column("alpha").to_numpy())
        ridge_ic = series.get_column("ic_mean").to_numpy()
        peak = int(np.argmax(ridge_ic))
        peaks[label] = (float(log_alpha[peak]), float(ridge_ic[peak]))
        fig_alpha.add_trace(
            go.Scatter(
                x=log_alpha,
                y=ridge_ic,
                mode="lines+markers",
                name=label,
                line=dict(color=curve_colors[index % len(curve_colors)], width=2),
                marker=dict(size=7, color=curve_colors[index % len(curve_colors)]),
            )
        )
        fig_alpha.add_trace(
            go.Scatter(
                x=[log_alpha[peak]],
                y=[ridge_ic[peak]],
                mode="markers",
                marker=dict(
                    size=15,
                    color=curve_colors[index % len(curve_colors)],
                    line=dict(width=2, color=COLORS["neutral"]),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig_alpha.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
    fig_alpha.update_layout(
        title="Ridge IC against penalty strength, over ten orders of magnitude, by label",
        height=520,
        width=900,
        margin=dict(t=70),
        legend=dict(title_text="Label"),
    )
    fig_alpha.update_xaxes(title_text="log₁₀(α)  (Ridge penalty strength)", zeroline=False)
    fig_alpha.update_yaxes(title_text="Mean cross-sectional IC (validation)")
    peak_text = ", ".join(
        f"{label} at 1e{int(round(peaks[label][0]))} (IC {peaks[label][1]:+.3f})"
        for label in ridge_labels
    )
    show_plotly_with_alt(
        fig_alpha,
        "Line chart of mean validation information coefficient against the base-ten logarithm of "
        "the Ridge penalty, one line per label, against a dashed zero line. All three lines sit "
        "below zero at weak penalties, cross it within an order of magnitude of each other and "
        "peak near the strong end of the grid before easing back. The peak on each line is ringed: "
        f"{peak_text}.",
    )
else:
    print(
        "No declared label declares Ridge configurations, so there is no penalty sweep to trace. "
        "Which estimators this section can show is decided by the menus at "
        "config/training/*.yaml."
    )

# %% [markdown]
# ## 5. What to notice
#
# **Shrinkage moves this fit from the wrong side of zero to the right side of it, at every
# horizon.** The unregularized fit and the weakly penalized Ridge configurations rank the
# cross-section backwards on all three labels; each curve crosses zero in the middle of the
# penalty range and reaches its maximum well up the grid. That shape has a clear reading. The
# design matrix lets an unpenalized fit chase relationships in the training window that reverse
# out of it, and a negative out-of-sample IC is what a reversed relationship looks like. Each
# order of magnitude of penalty removes more of that. Three labels agreeing on the shape is
# stronger evidence for that mechanism than one label showing it, because the three do not share
# a target: only the features and the folds are common to them.
#
# **The best result a horizon reaches rises several-fold with the horizon.** Read `best_ic` down
# the horizons frame: the best full-coverage configuration at 21 days reaches several times what
# the best one at one day reaches, from the same features, the same folds and the same universe.
# `above_zero` moves the same way. The one-day horizon is the one the strategy chapters trade, so
# the case study leads with its weakest label. That is worth knowing before reading any later
# number, and it is a property of the horizon rather than of anything chosen here.
#
# It does not follow that the horizon matters more than the penalty. On the frame's own numbers
# the 21-day grid spans a wider range from `worst_ic` to `best_ic` than separates the best 21-day
# configuration from the best one-day configuration. Both choices move the result by more than
# the result itself is worth, which is the reading to take rather than a ranking between them.
#
# The comparison is between full-coverage configurations on both sides. At five days the two most
# aggressive L1 settings post a higher IC than any charted configuration does, on the smaller
# number of dates the partial-coverage frame above reports, and taking that number as the
# horizon's result would compare a selected subsample against two full ones.
#
# **The horizons agree on the direction and disagree on the detail.** The first panel descends
# by construction, since it defines the order; the point is what the other two do under that
# order, and they are jagged rather than flat or descending. The rank correlations put the
# agreement between any two horizons well above zero and well below one, and `best_config` in the
# horizons frame names a different configuration at each horizon. What survives the change of
# horizon is the broad direction - heavy penalties beat light ones everywhere - and what does not
# is the fine ordering. Reading a specific configuration as "the best" therefore says which label
# it was best on.
#
# **Small and positive is the honest description at the traded horizon.** Nothing in the one-day
# grid reaches an information coefficient of the size a working equity cross-sectional signal
# posts, and the spread between the top of that grid and the bottom is itself a few thousandths.
# A notebook that reported only the top row would be describing the least bad member of a narrow
# set; the comparison that matters is against zero, and the margin above it is thin.
#
# **Coverage is uniform within each label but not identical across them.** `scorable_dates` in
# the horizons frame differs by label, because a longer forward window runs out earlier. Within
# the one-day and 21-day grids every configuration scored every date available to it; `charted`
# equals the menu size for those two and is smaller for five days. The configurations missing
# there are the most aggressive L1 settings, which zero all but a couple of columns and predict a
# near-constant value on some dates. A constant has no rank correlation with anything, so those
# dates drop out of the average. They are marked `full_coverage` false and left out of the chart
# rather than compared on a smaller sample.
#
# **A penalty cannot fix a dependent cross-section.** The `alpha` sweep addresses collinearity
# among the columns of the design matrix. It does nothing about the fact that the rows being
# ranked are 20 quotes among eight currencies: a prediction that ranks every USD pair the same way
# on a given day is one bet expressed 20 times, and a rank correlation counts it 20 times. That is
# a property of the universe rather than of the model, it is not visible in an IC, and it is why
# `13_backtest` puts these predictions through a portfolio rather than reading the ranking as a
# result.
#
# **None of this selects anything.** IC measures whether predictions rank pairs correctly, not
# whether a strategy trading them makes money after costs and turnover. Those are different
# questions and a configuration can win the first while losing the second: a signal that ranks
# well but reorders the portfolio at every rebalance can lose its edge to trading costs.
# Selection is on validation backtest Sharpe over the population this notebook just published,
# and it happens in [`13_backtest`](13_backtest.ipynb).
#
# **Known limitations.** The IC is an average of daily rank correlations. At the one-day horizon
# the overlap problem does not arise, but the five-day and 21-day labels do overlap: consecutive
# decision dates share most of their forward window, so their daily correlations are not
# independent draws and the IC of those two labels is a less precise number than its one-day
# counterpart at the same nominal date count. In the other direction, the dependence among the
# pairs inflates the effective sample relative to what a naive standard error assumes, at every
# horizon. The grid is a one-dimensional sweep of penalty strength at fixed features and fixed
# folds, so it says nothing about interactions between the penalty and either. And every number
# here is measured on the validation folds, which have been read many times over by the time a
# case study reaches this notebook.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) asks whether gradient boosting finds structure a linear model
# cannot represent at all - in particular whether the carry and momentum columns interact, which a
# linear model can only see if someone multiplies them together first.
