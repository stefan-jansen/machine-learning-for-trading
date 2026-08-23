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
# # CME futures: regularizing a design matrix built in blocks
#
# The 30 products in this universe are described by 69 features that do not form one
# undifferentiated set. They arrive in families. Eleven columns measure **carry** - the return
# earned by holding a futures position as it converges towards spot - as a level, a percentile, a
# sector-relative rank, a z-score over two windows, and its interaction with momentum. Eight
# measure a rolling Sharpe ratio, seven momentum, seven volatility. Within a family the columns
# are near-copies of one another by construction; across families they are not.
#
# That block structure changes what a penalty has to do. A design matrix whose columns are all
# variations on one signal has a single direction worth keeping, and shrinkage finds it. This one
# has several, and the question is whether a penalty strong enough to collapse each family leaves
# the differences between families intact. **Regularization** - adding a penalty on coefficient
# size to the fitting objective - is the instrument, and how much of it to apply is the experiment
# this notebook runs.
#
# The cross-section is also unlike an equity universe. Ranking these products means ranking corn
# against ten-year notes against crude, whose returns differ in volatility by an order of
# magnitude and whose drivers have little in common. A cross-sectional information coefficient
# over that universe is a weaker instrument than one over a set of comparable assets, and the
# results below should be read with that in mind.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read the set of models a case study has declared for a label, and say which estimator and
#   which hyperparameters each declared name resolves to.
# - Bind those declarations to the data on disk and check, before anything is fitted, that every
#   configuration will be measured on the same products, the same dates and the same folds.
# - Fit a population of models on walk-forward folds and publish one complete set of validation
#   predictions per configuration.
# - Tell apart the two things a penalty can do to a blocked feature set - shrink the members of a
#   family towards each other, or select one member and zero the rest - and read from the results
#   which one this data rewards.
# - Recognise when a high information coefficient is an artifact of a model that scored fewer
#   dates than its neighbours, and use prediction coverage to rule it out.
# - Run the same grid on two prediction horizons and read what changes between them.
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
# regularization does to a blocked feature set; that ranking decides nothing.

# %%
"""Fit the declared CME futures linear-model population on the walk-forward validation folds."""

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
study = open_study("cme_futures", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which label, and which models
#
# A label is the thing being predicted. This case study defines two in `config/setup.yaml`:
# `fwd_ret_5d`, the total return over the five trading days after the decision date, and
# `fwd_ret_21d` over 21 days. The five-day label is the primary one - the horizon the strategy
# chapters trade - and the 21-day label is a variant, kept so the effect of the prediction horizon
# can be examined separately.
#
# **This notebook fits every declared label.** Each has its own training menu at
# `config/training/{label}.yaml`, listing family by family the named configurations to fit for
# that label, and the notebook fits the union of them. A label with no menu file has nothing
# declared and nothing to fit; the labels below are the ones that declare linear models.
#
# Fitting both in one run is what puts the horizons side by side at all. It does not make them
# one controlled experiment: each label carries its own purge buffer in `config/setup.yaml`, `5D`
# against `21D`, so the fold boundaries and the eligible samples differ too, and the comparison
# below is between two label-specific protocols rather than one protocol at two horizons. Running
# one label at a time leaves even that comparison to be assembled by hand from two runs, and
# until it is assembled the variant is declared and never fitted. `LABELS` restricts the run to a
# subset when you want one, and defaults to everything the menus declare.

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
# where the walk-forward folds fall, or which product-date pairs have both a feature row and a
# label. **Resolving** a request is the step that goes and finds all of that: it reads the label
# and feature files, computes the fold boundaries from the walk-forward parameters in
# `config/setup.yaml`, works out the exact set of rows each fit is expected to predict, and turns
# any data-dependent hyperparameter into the number it will actually use - each fold's own
# $\alpha_{\max}$ times `alpha_frac`, in the case of Lasso.
#
# Resolving reads the inputs and fits nothing, so the plan below can be inspected before any
# computation starts. The three things to check in it:
#
# **Read the checks within a label, not across the whole frame.** Each label has its own purge
# buffer in `config/setup.yaml` - `5D` for the primary and `21D` for the variant - so the two
# horizons resolve to different fold boundaries and different eligible samples. `label` is the
# first column for that reason: rows from different labels are expected to differ, and it is a
# difference *inside* one label that means something.
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row of the same
#   label.** They are the width of the design matrix, the number of products, and the number of
#   product-date pairs to be predicted. Every configuration on one label reads the same feature
#   matrix, so a row that differs from its own label's neighbours is a configuration being
#   measured on a different sample from theirs, and its results are not comparable with theirs.
#   Between labels they differ by construction: the longer buffer removes more rows.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   `05_evaluation` established. This one does hold across labels.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   must not appear here: it is scored once, at the end of the case study, and any of it visible
#   in this window would mean it had been used to choose something. `validation_end` falls earlier
#   for the 21-day label than for the five-day one, because a longer forward window has to stop
#   earlier to keep its outcome inside the development period.
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
    "label",
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
population_name = POPULATION_NAME or "cme_futures-linear-validation-v1"
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
# study = open_study("cme_futures", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "linear", labels=["fwd_ret_5d"], config_names=["ols", "ridge_a1.0", "ridge_a3.0"]
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
# coefficient**: on each validation date, rank the products by the model's prediction, rank them
# by the return they went on to earn, correlate the two rankings, and average that daily
# correlation over the validation period. It measures whether the model ranks products correctly,
# on a scale where zero is no relationship, positive means the ranking points the right way, and
# negative means it points the wrong way.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it is not a
# footnote. A model whose coefficients collapse to one or two features predicts nearly the same
# value for every product on some dates; a constant has no rank correlation with anything, so
# those dates contribute nothing. Its `ic_mean` is then an average over fewer dates than its
# neighbours', chosen by where it happened to stay non-degenerate, and comparing it with theirs
# compares two different samples. `full_coverage` marks the configurations measured on all of
# them.
#
# **Coverage is judged within a label, not across them.** The two horizons do not have the same
# number of scorable validation dates to begin with: a 21-day label runs out of forward window
# earlier than a five-day label does, so it has fewer dates before any model is fitted. Comparing
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
# size and a shared scale would flatten the smaller panel to a line. Read a panel for the shape
# of its ranking, and the numbers rather than the bar heights when comparing panels.
#
# **The configurations are in the same order in both panels**, that order being their ranking on
# the primary label. Sorting each panel independently would put a different configuration at each
# left edge and make the panels look alike whatever the numbers did. Held in one order, a panel
# that slopes down from left to right is a horizon that ranks the grid the way the primary label
# does, and a panel that does not is one where the penalty that works at one horizon does not
# work at another.


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
    "Stacked bar charts of mean validation information coefficient across the linear penalty "
    "grid, one panel per label and the configurations in the same order in each, that order "
    f"being their ranking on {order_label}. Each panel carries a dashed zero line. The highest "
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
# One curve per label puts both horizons on one pair of axes, which is the comparison fitting
# them together exists to make: whether the penalty that helps most is the same strength at each
# horizon, and whether the curve has the same shape at all.

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
        "the Ridge penalty, one line per label, against a dashed zero line. Both lines sit below "
        "zero across the grid, flat at weak penalties and rising towards zero as the penalty "
        f"strengthens. The peak on each line is ringed: {peak_text}.",
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
# **Almost nothing in the grid clears zero, and what does is L1.** Read `above_zero` against
# `charted` in the horizons frame: a handful at each horizon, out of two dozen charted.
# Every one of them is a Lasso or an ElasticNet at `alpha_frac=0.5` or `alpha_frac=0.7` - a
# penalty strong enough to zero most of the columns outright. No Ridge configuration clears
# zero at either horizon; filter the catalog on `model_class` to see it. The comparison that
# matters is against zero rather than against the rest of the grid, and on that comparison most
# of this grid fails.
#
# **Shrinkage and selection do different things here, and only one of them helps.** The Ridge
# curve has the same shape at both horizons: flat and negative at weak penalties, rising towards
# zero as the penalty strengthens. The five-day curve peaks at `alpha=1e6`, one step short of the
# grid's edge; the 21-day curve peaks at `1e7`, which *is* the edge, so nothing here rules out
# its optimum lying beyond the strongest penalty declared. That shape
# reads cleanly. The 69 columns let a weakly penalized fit chase relationships in the training
# window that reverse out of it, and a negative out-of-sample IC is what a reversed relationship
# looks like; each order of magnitude of penalty removes more of that, and the value it converges
# on is zero, because a sufficiently penalized linear model predicts a constant and a constant
# has no rank correlation with anything. Shrinking every family towards the others is therefore
# damage limitation. Discarding most of them outright is the only setting that ends up on the
# right side of zero, which says the blocked design matrix has fewer usable directions than it
# has families.
#
# **The charted gap between the horizons is large, and most of it is the coverage filter.** Read
# `best_ic` down the horizons frame and 21 days is an order of magnitude above five. That
# reading does not survive inspection. The five-day configurations that would be nearest the
# top - the `alpha_frac=0.7` pair, and `alpha_frac=0.85` - are the ones the coverage filter
# removes, and their raw ICs are in the partial-coverage frame above; the `alpha_frac=0.7` pair
# is full-coverage at 21 days and is exactly what leads there. Compared like against like the
# gap is a small multiple, and it is a comparison between two configurations that are not
# measured on the same set of dates.
#
# **What the gap is not is a clean property of the horizon.** The two labels do not share a
# protocol: their purge buffers are `5D` and `21D`, which is why the plan frame shows different
# eligible rows and a different `validation_end`, so the features, folds and sample are not held
# fixed across the comparison. What can be said without qualification is narrower and still
# useful: **whatever the linear family finds here, it finds more of at 21 days than at five, and
# only through L1.** The 21-day Ridge curve is the worse of the two, so shrinkage is not carrying
# any of it. `07_gbm` fits the same two labels and settles how far this reading generalises.
#
# **The horizons agree on the ordering while disagreeing on the level.** `rank_correlation` in
# the `transfer` frame is high over the configurations both panels charted, while `best_config`
# names a different one at each. So the broad direction transfers: strong L1, then weak L1,
# then Ridge, then no penalty, in that order at both horizons. What does not transfer is how
# far up the scale that ordering reaches. This is the one horizon comparison the coverage
# filter does not distort, because it is computed over exactly the configurations both panels
# charted.
#
# **Read the ranking with the coverage column or it will mislead you, and read it per label.**
# `alpha_frac=0.85` is missing from both panels: it zeros all but a couple of features on some
# folds, predicts a near-constant value on those dates, and contributes no correlation there,
# which the partial-coverage frame shows as an `ic_n_days` well short of its label's full
# count. Its raw IC at 21 days would place it near the top of that panel if it were charted, on
# a sample it selected by going degenerate everywhere else, and a metric averaged over a set
# the model itself chose is not a metric. **At five days `alpha_frac=0.7` goes the same way**
# and is excluded on the same dates. The same pair is full-coverage at 21 days and leads that
# panel. So the exclusion is asymmetric between the horizons, and the next paragraph is where
# that matters.
#
# **A cross-sectional ranking may still be the wrong question for this universe.** The features
# here describe each product against its own history - carry against its own z-score, momentum
# against its own trailing window - and the label is that product's own forward return. Ranking
# those predictions across corn, notes and crude asks the model to make one number comparable
# across assets whose return distributions are not. The literature on futures returns is largely
# a time-series literature for that reason. That a handful of configurations do clear zero says
# the question is not hopeless as posed; it does not make it the right question.
#
# **None of this selects anything.** IC measures whether predictions rank products correctly, not
# whether a strategy trading them makes money after costs and turnover. Those are different
# questions and a configuration can win the first while losing the second: a signal built on two
# surviving features that reorders the portfolio at every rebalance can lose its edge to trading
# costs. Selection is on validation backtest Sharpe over the population this notebook just
# published, and it happens in [`13_backtest`](13_backtest.ipynb).
#
# **Known limitations.** The IC here is an average of daily rank correlations with no adjustment
# for serial dependence, and both labels overlap: consecutive decision dates share most of their
# forward window, so their daily correlations are not independent draws and the 21-day number is
# the less precise of the two at the same nominal date count. In the other direction, 30 products
# is a narrow cross-section for a daily rank correlation, which is why `ic_std` is an order of
# magnitude larger than any `ic_mean` in the table. `05_evaluation` does the inference for
# individual features. The grid is a one-dimensional sweep of penalty strength at fixed features
# and fixed folds, so it says nothing about interactions between the penalty and either. And
# every number here is measured on the validation folds, which have been read many times over by
# the time a case study reaches this notebook.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) asks whether gradient boosting finds structure a linear model
# cannot represent at all. The feature set already contains two hand-built interaction terms,
# `carry_mom_composite` and `carry_mom_interaction`, which exist because a linear model can only
# see an interaction if someone multiplies the columns first; a tree ensemble does not need them
# named in advance. It fits both labels too, so the two questions this notebook leaves open carry
# over: whether anything clears zero at the traded horizon, and whether the 21-day advantage is a
# property of the horizon or of what L1 happened to keep.
