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
# # Option analytics: can what the options market prices rank the stock
#
# Every other case study in this book builds its features from the history of the thing it is
# trying to predict - past returns, past volatility, past volume. This one does not. Its features
# come from the options written on each stock, and an option price is not a record of what
# happened. It is what someone was willing to pay today for a claim on what happens next. Implied
# volatility, the skew between puts and calls, the slope of the term structure, the gap between
# implied and realized variance: each is a statement the market is making about the future.
#
# That makes the question here sharper than "do these features work". The features are already
# forecasts, made by people with money at stake. If a forecast that good does not rank the
# cross-section, the interesting part is understanding what it is a forecast *of*.
#
# The design matrix has the redundancy every case study in this book has - implied volatility
# appears at three tenors and again as a rank, a percentile and two z-scores; the variance risk
# premium appears as a level, a rank and a z-score - so the penalty sweep has the same job here as
# elsewhere. **Regularization** adds a penalty on coefficient size to the fitting objective, and
# how much to apply is an empirical question this notebook answers by trying ten orders of
# magnitude of it.
#
# Two folds, and one of them validates on 2020. The usable history of this option analytics
# dataset is short, so `05_evaluation` set a walk-forward schedule whose validation windows are
# 2019 and 2020. Whatever the models find or fail to find, they are being asked about a period
# containing the fastest volatility shock in the sample.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read the set of models a case study has declared for a label, and say which estimator and
#   which hyperparameters each declared name resolves to.
# - Bind those declarations to the data on disk and check, before anything is fitted, that every
#   configuration will be measured on the same stocks, the same dates and the same folds.
# - Fit a population of models on walk-forward folds and publish one complete set of validation
#   predictions per configuration.
# - Say what a feature set of option-implied quantities is a forecast of, and why that is not the
#   same thing as the quantity this notebook asks it to rank.
# - Read a grid in which no configuration clears zero, and say what that is and is not evidence
#   for.
# - Run configurations of your own into a private copy of the run log, and have them compared on
#   the same footing as the ones shipped here.
#
# **Book reference**: Chapter 11 (The ML Pipeline). Chapter 6, Section 6.7 (Search accounting and
# run logging) introduces the run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# and [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds and screened
# the individual features.
#
# **What it writes**: one training run and one complete validation prediction set per
# configuration, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population.
# [`14_backtest`](14_backtest.ipynb) reads that population, runs every member against the
# equal-weight baseline, and selects on validation backtest Sharpe. **Selection happens there,
# not here.** This notebook ranks configurations by information coefficient to show what
# regularization does to this feature set; that ranking decides nothing.

# %%
"""Fit the declared option-analytics linear-model population on the walk-forward folds."""

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
SUPERSEDES_POPULATION: str = ""

# %%
study = open_study(
    "sp500_equity_option_analytics", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)

# %% [markdown]
# ## 1. Which label, and which models
#
# A label is the thing being predicted. This case study defines five in `config/setup.yaml`.
# `fwd_ret_5d`, the stock's total return over the five trading days after the decision date, is
# the primary one - the horizon the strategy chapters trade. `fwd_ret_10d` is the same idea over
# a longer horizon, `fwd_ret_risk_adj_5d` divides the move by a volatility estimate, and
# `fwd_dir_5d` and `fwd_dir_10d` are classification variants that ask for the direction rather
# than the size.
#
# **This notebook fits every declared label in one run.** Each label has its own training menu at
# `config/training/{label}.yaml`, listing family by family the named configurations to fit for
# that label. A label with no menu file has nothing declared and nothing to fit; the ones below
# are those that declare linear models. `LABELS` narrows the run to a subset, which is a
# diagnostic rather than the canonical population.

# %%
declared_labels(study, "linear")

# %% [markdown]
# Each name in the menu resolves to a preset file in the shared directory
# `case_studies/config/{model_type}/`, which holds that configuration's hyperparameters. The
# frame below is the menu for every declared label, with each name resolved to the estimator class it names
# and the arguments that class is constructed with. To change what runs, edit the menu or the
# presets rather than this notebook.
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
# where the walk-forward folds fall, or which stock-date pairs have both a feature row and a
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
#   the width of the design matrix, the number of stocks with a usable option surface, and the
#   number of stock-date pairs to be predicted. Every configuration here reads the same feature
#   matrix, so a row that differs is a configuration being measured on a different sample from
#   its neighbours, and its results are not comparable with theirs.
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
# and written down, and afterwards every member must exist and be complete. That is what makes
# the downstream comparison well defined - `14_backtest` backtests this population, not whatever
# predictions happen to be in the registry - and it is why a configuration that raises fails the
# whole call rather than publishing a population one member short. Everything that finished stays
# registered, and re-running fits only what is missing.

# %%
population_name = POPULATION_NAME or "sp500_equity_option_analytics-linear-validation-v1"
execution, population = run_model_population(
    study, resolved, population_name=population_name, supersedes=SUPERSEDES_POPULATION or None
)

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
# study = open_study("sp500_equity_option_analytics", workspace="~/ml4t-experiments")
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
# One row per configuration and label, read back from the registry. `ic_mean` is the **information
# coefficient**: on each validation date, rank the stocks by the model's prediction, rank them by
# the return they went on to earn, correlate the two rankings, and average that daily correlation
# over the validation period. It measures whether the model ranks stocks correctly, on a scale
# where zero is no relationship, positive means the ranking points the right way, and negative
# means it points the wrong way.
#
# The catalog is joined to the menu on **both** `label` and `config_name`. A configuration name is
# unique within a label's menu and not across them: `ridge_a1.0` is declared by every regression
# label here, so joining on the name alone would multiply each result row by the number of labels
# that declare it and fill the table with copies carrying identical ICs.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and coverage is judged
# against **each label's own** maximum. The labels do not offer the same number of dates to begin
# with - a ten-day forward window runs out earlier than a five-day one - so a single global
# maximum would mark a whole label incomplete for a reason that has nothing to do with any model.
# Within a label, the check is doing what it was built for: a model whose coefficients collapse to
# one or two features predicts nearly the same value for every stock on some dates, a constant has
# no rank correlation with anything, and its IC is then an average over a sample it selected
# itself. `full_coverage` marks the configurations measured on all of their label's dates.

# %% tags=["results"]
catalog = (
    execution.catalog_rows.select(
        "config_name",
        "label",
        "task",
        "complete",
        "ic_mean",
        "ic_std",
        "ic_n_days",
        "auc_mean_daily",
        "n_folds",
        "training_hash",
        "prediction_hash",
    )
    .sort(["label", "ic_mean"], descending=[False, True])
    .join(
        configs.select("label", "config_name", "model_class", "params"),
        on=["label", "config_name"],
        how="left",
    )
)

if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("linear execution returned a partial prediction set")

catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
# The primary label leads when it was fitted. A subset run that leaves it out orders the panels by
# whichever label it did fit rather than by one that is not there.
panel_labels = [label for label in [primary] if label in present] + [
    label for label in present if label != primary
]
print(f"{catalog.height} candidate models across {len(panel_labels)} labels")
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
# ### What the grid does on each label
#
# The frame below is the comparison this notebook exists to make, and it is only available because
# every declared label was fitted in one run. The features are the same, the folds are the same,
# and the estimators are the same; the only thing that changes down the rows is what is being
# predicted.
#
# Read `n_positive` against `configurations`. A label where the whole grid sits on one side of zero
# is saying something about the label; a label where the grid straddles zero is saying that the
# spread across configurations is larger than anything the features contribute.
#
# The two `fwd_dir_*` labels are the classification form of the same forward window, and carry
# `auc_mean_daily` as well: the within-date reading of how well the predicted probability
# separates the classes. It is null on the regression labels, which have no classes to separate.

# %% tags=["results"]
by_label = (
    catalog.filter("full_coverage")
    .group_by("label")
    .agg(
        task=pl.col("task").first(),
        configurations=pl.len(),
        scored_dates=pl.col("ic_n_days").max(),
        best_ic=pl.col("ic_mean").max(),
        worst_ic=pl.col("ic_mean").min(),
        n_positive=(pl.col("ic_mean") > 0).sum(),
        best_auc_daily=pl.col("auc_mean_daily").max(),
    )
    .sort("best_ic", descending=True)
)
by_label

# %% [markdown]
# ### How the penalty grid ranks
#
# One panel per label, sharing a vertical scale so the labels are compared rather than each one
# rescaled to fill its own panel. Only configurations measured on all of their label's dates are
# charted. The zero line is the reference that matters: a bar below it is a model whose ranking
# pointed the wrong way out of sample.
#
# The configurations are held in one order across the panels - their ranking on the primary label
# - so a panel that does not descend is a label that orders the grid differently.


# %%
def compact(params: str) -> str:
    """Render declared parameters for a label: `alpha=1000000.0` reads as `alpha=1e+06`."""
    return re.sub(r"\d+\.?\d*(?:[eE][+-]?\d+)?", lambda m: f"{float(m.group()):g}", params)


full = catalog.filter("full_coverage")
config_order = (
    full.filter(pl.col("label") == panel_labels[0])
    .sort("ic_mean", descending=True)
    .get_column("config_name")
    .to_list()
)

fig_ic = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    shared_yaxes=True,
    vertical_spacing=0.04,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    panel = full.filter(pl.col("label") == label)
    # A label whose menu declares different configurations - the classification ones do - keeps
    # its own members and appends them after the shared order rather than being dropped.
    order = [name for name in config_order if name in set(panel.get_column("config_name"))]
    order += [name for name in panel.get_column("config_name").to_list() if name not in order]
    panel = panel.with_columns(
        rank=pl.col("config_name").replace_strict(
            {name: index for index, name in enumerate(order)}, return_dtype=pl.Int32
        )
    ).sort("rank")
    best = panel.get_column("ic_mean").max()
    fig_ic.add_trace(
        go.Bar(
            x=panel.get_column("config_name").to_list(),
            y=panel.get_column("ic_mean").to_list(),
            marker_color=[
                COLORS["amber"] if value == best else COLORS["blue"]
                for value in panel.get_column("ic_mean")
            ],
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig_ic.add_hline(
        y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1
    )
    fig_ic.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig_ic.update_xaxes(
    title_text=f"Configuration, ordered by rank on {panel_labels[0]}",
    tickangle=-45,
    row=len(panel_labels),
    col=1,
)
fig_ic.update_layout(
    title="Which side of zero the grid sits on depends on the label, not the penalty",
    height=260 * len(panel_labels),
    width=1100,
    margin=dict(t=90),
)
# Which labels clear zero is a fact about the frame, so the alt text reads it rather than
# asserting it. Describing every bar as negative was true of the one label this notebook used to
# fit and is false of the set it fits now.
side_text = "; ".join(
    f"{row['label']} has {row['n_positive']} of {row['configurations']} above zero"
    for row in by_label.sort("label").iter_rows(named=True)
)
show_plotly_with_alt(
    fig_ic,
    "Bar charts of mean validation information coefficient for every full-coverage linear "
    "configuration, one panel per declared label sharing a vertical scale, each panel's highest "
    "bar in amber and the rest in dark navy, with a dashed zero line across each. The bars are "
    "held in the primary label's ranking order in every panel, so a panel that does not descend "
    f"is a label that orders the grid differently. Counted from the frame: {side_text}. The "
    "spread within any one panel is a few thousandths, far smaller than the difference between "
    "panels.",
)

# %% [markdown]
# ### What shrinkage does on its own
#
# The bar chart mixes three estimators. Tracing IC across the Ridge penalty alone isolates the
# effect of shrinkage, with the estimator, the features and the folds all held fixed and only
# `alpha` moving. The alpha is read from each configuration's declared parameters rather than
# parsed out of its name, so the curve plots what was fitted. One line per label, because the
# question is whether the penalty behaves the same way regardless of what is being predicted.

# %%
ridge = (
    catalog.filter(pl.col("model_class") == "Ridge")
    .with_columns(alpha=pl.col("params").str.extract(r"alpha=([0-9.eE+-]+)").cast(pl.Float64))
    .drop_nulls("alpha")
    .sort("label", "alpha")
)
if ridge.height:
    line_colors = [COLORS["blue"], COLORS["copper"], COLORS["slate"], COLORS["amber"]]
    fig_alpha = go.Figure()
    for index, label in enumerate(panel_labels):
        series = ridge.filter(pl.col("label") == label)
        if not series.height:
            continue
        log_alpha = np.log10(series.get_column("alpha").to_numpy())
        values = series.get_column("ic_mean").to_numpy()
        peak = int(np.argmax(values))
        color = line_colors[index % len(line_colors)]
        fig_alpha.add_trace(
            go.Scatter(
                x=log_alpha,
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color),
            )
        )
        fig_alpha.add_trace(
            go.Scatter(
                x=[log_alpha[peak]],
                y=[values[peak]],
                mode="markers",
                marker=dict(size=13, color=COLORS["amber"], symbol="circle-open", line_width=3),
                showlegend=False,
            )
        )
    fig_alpha.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
    fig_alpha.update_layout(
        title="Ridge IC against penalty strength, over ten orders of magnitude",
        height=520,
        width=950,
        margin=dict(t=70),
        legend=dict(title_text="Label"),
    )
    fig_alpha.update_xaxes(title_text="log₁₀(α)  (Ridge penalty strength)", zeroline=False)
    fig_alpha.update_yaxes(title_text="Mean cross-sectional IC (validation)")
    show_plotly_with_alt(
        fig_alpha,
        "Line chart of mean validation information coefficient against the base-ten logarithm of "
        "the Ridge penalty, one line per label over ten orders of magnitude, each line's highest "
        "point ringed in amber, against a dashed zero line. The lines are separated from each "
        "other by more than any of them moves across the penalty range, so the label a line "
        "belongs to matters more than where on the line it sits.",
    )
else:
    print(
        "No declared label carries a Ridge configuration, so there is no penalty sweep to "
        "trace. Which estimators this section can show is decided by the per-label menus at "
        "config/training/{label}.yaml."
    )

# %% [markdown]
# ## 5. What to notice
#
# **What is being predicted decides the result here, and the penalty does not.** The panels are
# separated from each other by more than the grid moves inside any one of them: a sweep over ten
# orders of magnitude of Ridge penalty moves the metric less than swapping the label does. When
# that is the shape of the evidence, tuning the penalty is not the experiment worth running. The
# top row of a grid sorted across labels reports which label was easiest, dressed as a model
# comparison, so the table below is grouped by label and never sorted across them.
#
# **These features are a forecast of dispersion, not of direction, and the labels test that
# directly.** That is a property of what an option price is. Implied volatility says how wide the
# market expects the distribution to be; skew says how asymmetric; the term structure says how
# that changes with horizon; the variance risk premium says how much the market charges for
# bearing it. None of them is a statement about the *mean* of the distribution, which is what a
# forward return is. A risk-adjusted return divides that mean by a measure of width, so a feature
# set that forecasts width well and direction badly should rank the risk-adjusted label better
# than the raw one. Read the `by_label` frame with that expectation in hand: it is a prediction
# the mechanism makes about which row leads, and it is the reason to fit more than one label
# rather than a decoration on having done so. `05_evaluation` screens these features one at a time
# and reaches the same place before any model is fitted.
#
# **Two folds, and one of them is 2020.** Half the validation evidence comes from a year in which
# implied volatility across every name moved together and by more than in the rest of the sample
# combined. A cross-sectional ranking asks which stock will out-return which, and that question
# gets harder when one factor is moving everything. Nothing above separates a weak feature set
# from an unrepresentative window, and with two folds nothing can. This applies to every panel,
# including whichever one leads.
#
# **What this does not rule out.** It says nothing about whether these features forecast the
# *volatility* of these stocks, which is the quantity they are actually about and which the risk
# chapters use them for. It says nothing about a strategy that trades the options rather than the
# underlying, which is the question the `sp500_options` case study asks. And a linear model can
# only represent a weighted sum of these columns, so it cannot express something like "skew
# matters when the term structure is inverted" unless someone builds that column first.
#
# **None of this selects anything.** IC measures whether predictions rank stocks correctly, not
# whether a strategy trading them makes money after costs and turnover. Selection is on validation
# backtest Sharpe over the population this notebook just published, and it happens in
# [`14_backtest`](14_backtest.ipynb).
#
# **Known limitations.** The IC here is an average of daily rank correlations with no adjustment
# for the serial dependence that overlapping multi-day returns create, so it is a ranking
# diagnostic rather than a test. The grid is a one-dimensional sweep of penalty strength at fixed
# features and fixed folds. And every number here is measured on the validation folds, which have
# been read many times over by the time a case study reaches this notebook.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) asks whether gradient boosting finds structure a linear model
# cannot represent at all. The interaction named above is the concrete version of that question,
# and it is worth being clear in advance what a better result there would mean: with this many
# features and two folds, a tree ensemble that clears zero where the linear grid did not has
# either found a real interaction or fitted the 2020 window, and telling those apart is what
# [`13_model_analysis`](13_model_analysis.ipynb) is for.
