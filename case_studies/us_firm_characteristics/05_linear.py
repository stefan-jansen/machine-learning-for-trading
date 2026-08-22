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
# # Firm characteristics: strong features, and a target that decides the fit
#
# The features in this case study are not invented for the book. They are the cross-section of
# firm characteristics the asset pricing literature has spent thirty years assembling: value in
# several forms (`BEME`, `E2P`, `S2P`, `CF2P`), profitability (`ROA`, `ROE`, `OP`, `PROF`),
# investment and accruals (`Investment`, `NOA`, `OA`), size (`LME`), momentum at several formation
# windows (`r2_1`, `r12_2`, `r12_7`, `r36_13`), reversal, and a set of risk and liquidity measures
# (`IdioVol`, `Resid_Var`, `Variance`, `Spread`, `Beta`). Each arrives with published evidence that
# it orders the cross-section of monthly stock returns.
#
# That makes this the one case study where the features are not in question, and it lets the
# notebook ask a cleaner question than "do these predict". `04_evaluation` has already screened
# them one at a time and found what the literature would lead you to expect. **The question here is
# what happens when you fit a linear model to all of them at once, and it turns out to be a
# question about the target rather than about the features.**
#
# A monthly cross-section of individual US stocks is not a well-behaved regression target. Most
# firms move a few percent; a small number of them multiply. `02_labels` measures the resulting
# kurtosis directly and is where the case study's winsorized variant comes from. The consequence
# for this notebook is specific and worth stating before any results appear: a linear model fits by
# **minimizing squared error**, so a firm that returned many multiples of its price contributes to
# the objective in proportion to the square of that return, while the **information coefficient**
# this notebook reports is a rank correlation that treats it as one observation among thousands.
# Those are different targets, and section 6 reads the results in that light.
#
# The design matrix also has the redundancy every case study here has - four momentum windows,
# three variance measures, six composites and four explicit interaction terms built from columns
# already present - so the penalty sweep has its usual job. **Regularization** adds a penalty on
# coefficient size to the fitting objective, and how much to apply is an empirical question this
# notebook answers by trying ten orders of magnitude of it.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read the set of models a case study has declared for a label, and say which estimator and
#   which hyperparameters each declared name resolves to.
# - Bind those declarations to the data on disk and check, before anything is fitted, that every
#   configuration will be measured on the same firms, the same months and the same folds.
# - Fit a population of models on walk-forward folds and publish one complete set of validation
#   predictions per configuration.
# - Say why a squared-error fit and a rank correlation can disagree about the same model, and
#   recognise the distributional shape that makes them disagree.
# - Tell apart a result that is about the features from one that is about the target.
# - Run configurations of your own into a private copy of the run log, and have them compared on
#   the same footing as the ones shipped here.
#
# **Book reference**: Chapter 11 (The ML Pipeline). Chapter 6, Section 6.7 (Search accounting and
# run logging) introduces the run log this notebook writes to.
#
# **Prerequisites**: [`02_labels`](02_labels.ipynb) has constructed the monthly returns and the
# winsorized variant, [`03_financial_features`](03_financial_features.ipynb) has written the
# characteristics, and [`04_evaluation`](04_evaluation.ipynb) has established the walk-forward
# folds and screened the characteristics individually.
#
# **What it writes**: one training run and one complete validation prediction set per
# configuration, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population.
# [`11_backtest`](11_backtest.ipynb) reads that population, runs every member against the
# equal-weight baseline, and selects on validation backtest Sharpe. **Selection happens there,
# not here.** This notebook ranks configurations by information coefficient to show what
# regularization does to this feature set; that ranking decides nothing.

# %%
"""Fit the declared firm-characteristics linear-model population on the walk-forward folds."""

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
    "us_firm_characteristics", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None
)

# %% [markdown]
# ## 1. Which label, and which models
#
# A label is the thing being predicted. This case study defines three in `config/setup.yaml`.
# `fwd_ret_1m`, the firm's total return over the month after the decision date, is the primary
# one - the horizon the strategy chapters trade. `fwd_ret_1m_win` is the same return with each
# month's cross-section clipped at its own tails, and `fwd_class_1m` is a classification variant
# that asks which bucket the return falls in rather than how large it is. `02_labels` builds all
# three and shows how far apart the raw and clipped distributions are in the tails.
#
# **This notebook fits all three in one run.** Each carries its own training menu at
# `config/training/{label}.yaml`, listing family by family the named configurations to fit for
# that label, and the three grids enter one population together. None of them is privileged in
# code: the comparison below reads the primary label first because that is the horizon the
# strategy chapters trade, not because the other two were fitted separately.

# %%
declared_labels(study, "linear")

# %%
primary = primary_label(study)
primary

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
# where the walk-forward folds fall, or which firm-month pairs have both a feature row and a
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
#   the width of the design matrix, the number of firms that appear anywhere in the sample, and
#   the number of firm-month pairs to be predicted. Every configuration here reads the same
#   feature matrix, so a row that differs is a configuration being measured on a different sample
#   from its neighbours, and its results are not comparable with theirs. `eligible_entities`
#   counts firms over the whole period rather than in any one month: the universe turns over as
#   firms list and delist, and a typical month holds a fraction of that total.
# - **`folds` is the same everywhere**, and equals the number of walk-forward splits
#   `04_evaluation` established.
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
# which is what a walk-forward prediction set is: each month predicted by a model that saw only
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
# the downstream comparison well defined - `11_backtest` backtests this population, not whatever
# predictions happen to be in the registry - and it is why a configuration that raises fails the
# whole call rather than publishing a population one member short. Everything that finished stays
# registered, and re-running fits only what is missing.

# %%
population_name = POPULATION_NAME or "us_firm_characteristics-linear-validation-v1"
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
# study = open_study("us_firm_characteristics", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "linear", labels=["fwd_ret_1m"], config_names=["ols", "ridge_a1.0", "ridge_a3.0"]
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
# ## 4. What the targets look like
#
# Before the model results, the targets themselves, because their shape is what the results below
# turn on. Each observation is one firm's total return over one month. The count axis is
# logarithmic, because a linear one wide enough to show the right tail would compress everything
# else into a single bar.
#
# The raw distribution is not symmetric and it is not thin-tailed. A firm can lose at most
# everything, so there is a floor at a return of minus one; there is no corresponding ceiling, and
# the largest observations in this sample are multiples of the starting price. `02_labels`
# quantifies the kurtosis and builds `fwd_ret_1m_win` in response to it, clipping each month's
# cross-section at its own tails.
#
# What matters here is the arithmetic of a squared-error fit: an observation ten times further
# from the prediction than another contributes a hundred times as much to what the fit is
# minimizing. **Both targets are fitted in this run**, on the same folds with the same grid, so
# the effect of that clipping is measured below rather than asserted.

# %% tags=["results"]
regression_labels = [
    label
    for label in declared_labels(study, "linear")
    if not label.startswith("fwd_class") and (LABELS == [] or label in LABELS)
]
label_series = {
    label: study.labels.get(label, execution_tier=EXECUTION_TIER)
    .load()
    .get_column(label)
    .drop_nulls()
    .to_numpy()
    for label in regression_labels
}

summary = pl.DataFrame(
    [
        {
            "label": label,
            "observations": int(values.size),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std()),
            "share_above_1": float((values > 1).mean()),
        }
        for label, values in label_series.items()
    ]
)
summary

# %%
fig_label = go.Figure()
series_colors = [COLORS["blue"], COLORS["copper"], COLORS["slate"]]
for index, (label, values) in enumerate(label_series.items()):
    fig_label.add_trace(
        go.Histogram(
            x=values,
            nbinsx=140,
            name=label,
            marker_color=series_colors[index % len(series_colors)],
            opacity=0.6,
        )
    )
fig_label.add_vline(x=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig_label.update_layout(
    title="Clipping each month's tails is what separates the two return targets",
    height=460,
    width=950,
    barmode="overlay",
    margin=dict(t=70),
    bargap=0.02,
    legend=dict(title_text="Label"),
)
fig_label.update_xaxes(title_text="Monthly total return")
fig_label.update_yaxes(title_text="Firm-months", type="log")
show_plotly_with_alt(
    fig_label,
    "Overlaid histograms of the regression targets on a logarithmic count axis, with a dashed "
    "line at zero. The raw monthly return has a hard edge at minus one and a long thin tail of "
    "large positive returns extending far to the right; the winsorized variant is the same "
    "distribution with that tail cut back to the clipping point, so the two are indistinguishable "
    "through the bulk and differ only at the extremes.",
)

# %% [markdown]
# ## 5. What the models produced
#
# One row per configuration and label, read back from the registry. `ic_mean` is the **information
# coefficient**: in each validation month, rank the firms by the model's prediction, rank them by
# the return they went on to earn, correlate the two rankings, and average that monthly
# correlation over the validation period. It measures whether the model ranks firms correctly, on
# a scale where zero is no relationship, positive means the ranking points the right way, and
# negative means it points the wrong way.
#
# The catalog is joined to the menu on **both** `label` and `config_name`. A configuration name is
# unique within a label's menu and not across them - `ridge_a1.0` is declared by every label here
# - so joining on the name alone would multiply each result row by the number of labels declaring
# it and fill the table with copies carrying identical ICs.
#
# `ic_n_days` is how many validation months produced a defined correlation, and coverage is judged
# against **each label's own** maximum, so a label with fewer scorable months is not marked
# incomplete for a reason unrelated to any model. Within a label the check does what it was built
# for: a model whose coefficients collapse to one or two features predicts nearly the same value
# for every firm in some months, a constant has no rank correlation with anything, and its IC is
# then an average over a sample it selected itself.

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
# ### The same grid, on each target
#
# This frame is the experiment. The features are the same, the folds are the same and the twenty-
# eight configurations are the same; the only thing that changes down the rows is what is being
# predicted. `n_positive` against `configurations` is the column to read: a label where the whole
# grid sits on one side of zero is telling you about the target, not about the penalty.
#
# `fwd_class_1m` is the classification form of the same forward month and carries
# `auc_mean_daily` as well - the within-month reading of how well the predicted probability
# separates the classes. It is null on the regression labels, which have no classes to separate.
# `ic_mean` is defined for all three, which is what puts them on one axis.

# %% tags=["results"]
by_label = (
    catalog.filter("full_coverage")
    .group_by("label")
    .agg(
        task=pl.col("task").first(),
        configurations=pl.len(),
        scored_months=pl.col("ic_n_days").max(),
        best_ic=pl.col("ic_mean").max(),
        worst_ic=pl.col("ic_mean").min(),
        n_positive=(pl.col("ic_mean") > 0).sum(),
        best_auc_monthly=pl.col("auc_mean_daily").max(),
    )
    .sort("best_ic", descending=True)
)
by_label

# %% [markdown]
# ### How the penalty grid ranks
#
# One panel per label, sharing a vertical scale so the targets are compared rather than each one
# rescaled to fill its own panel. Only configurations measured on all of their label's months are
# charted. The zero line is the reference that matters: a bar below it is a model whose ranking
# pointed the wrong way out of sample.
#
# The configurations are held in one order across the panels - their ranking on the primary label
# - so a panel that does not descend is a target that orders the grid differently.


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
    vertical_spacing=0.05,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    panel = full.filter(pl.col("label") == label)
    # A label whose menu declares different configurations - the classification one does - keeps
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
    title="The target decides which side of zero the grid sits on, not the penalty",
    height=280 * len(panel_labels),
    width=1100,
    margin=dict(t=90),
)
# Which side of zero each panel sits on is a fact about the frame, so the alt text reads it.
# Describing every bar as negative was true of the one label this notebook used to fit.
side_text = "; ".join(
    f"{row['label']} has {row['n_positive']} of {row['configurations']} above zero"
    for row in by_label.sort("label").iter_rows(named=True)
)
show_plotly_with_alt(
    fig_ic,
    "Bar charts of mean validation information coefficient for every full-coverage linear "
    "configuration, one panel per declared label sharing a vertical scale, each panel's highest "
    "bar in amber and the rest in dark navy, with a dashed zero line across each. The bars are "
    "held in the primary label's ranking order in every panel. Counted from the frame: "
    f"{side_text}. The spread within any one panel is small next to the distance between panels.",
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
    fig_alpha = go.Figure()
    for index, label in enumerate(panel_labels):
        series = ridge.filter(pl.col("label") == label)
        if not series.height:
            continue
        log_alpha = np.log10(series.get_column("alpha").to_numpy())
        values = series.get_column("ic_mean").to_numpy()
        peak = int(np.argmax(values))
        color = series_colors[index % len(series_colors)]
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
        "other by more than any of them moves across the penalty range, and each falls away at "
        "the strongest penalties.",
    )
else:
    print(
        "No declared label carries a Ridge configuration, so there is no penalty sweep to "
        "trace. Which estimators this section can show is decided by the per-label menus at "
        "config/training/{label}.yaml."
    )

# %% [markdown]
# ## 6. What to notice
#
# **The target decides the sign, and the penalty does not.** Read `by_label` first. The same
# twenty-eight configurations, on the same folds and the same characteristics, land on different
# sides of zero depending only on which forward-return column they were fitted against. A sweep
# over ten orders of magnitude of Ridge penalty moves the metric less than that choice does.
#
# **The two things being compared are not the same thing, and that is the mechanism.** The fit
# minimizes squared error in return space, where a firm that tripled sits hundreds of times
# further from any prediction than a typical firm and therefore dominates what the coefficients
# are chosen to do. The metric is a rank correlation in each month's cross-section, where that
# same firm is one name near the top of a list of a couple of thousand. A procedure tuned almost
# entirely by the extreme right tail is being graded on an ordering that barely notices it.
# Nothing about that is a defect in either choice; they are answers to different questions.
#
# **`fwd_ret_1m_win` is that mechanism's control, and this run is where it stops being an
# assertion.** `02_labels` clips each month's cross-section at its own tails precisely because a
# monthly cross-section of individual stocks is not something a squared-error procedure handles
# well. Fitting the same grid against the clipped column changes one variable, so the difference
# between those two rows of `by_label` is the size of the effect described above. The histogram in
# section 4 shows how little of the data the clipping touches; the frame shows how much it moves.
#
# **Read this against `04_evaluation`.** That notebook screened these same characteristics one at
# a time on these same folds and found the cross-sectional orderings the literature describes. The
# signal is in the columns. A linear fit that fails to recover it against one target and recovers
# it against a clipped version of the same target is a statement about what happens in between,
# not about the features.
#
# **The strongest shrinkage hurts rather than helps, on every target.** The Ridge curves are flat
# across most of their range and fall at the largest penalties. That is the opposite of the
# collinearity story in the other case studies, where shrinkage helps. Here the fit is not being
# defeated by correlated columns, so removing coefficient magnitude removes what little the fit
# has without fixing the reason it is misaligned.
#
# **None of this selects anything.** IC measures whether predictions rank firms correctly, not
# whether a strategy trading them makes money after costs and turnover. A monthly rebalance across
# thousands of names is where turnover costs bite hardest, and the smallest firms - the ones with
# the widest `Spread` in this feature set - are where a paper ranking is most likely to be
# untradable. Selection is on validation backtest Sharpe over the population this notebook just
# published, and it happens in [`11_backtest`](11_backtest.ipynb).
#
# **Known limitations.** The IC is an average of monthly rank correlations with no adjustment for
# the dependence across months that persistent characteristics induce, so it is a ranking
# diagnostic rather than a test; `04_evaluation` does that inference for individual
# characteristics. The grid is a one-dimensional sweep of penalty strength at fixed features and
# fixed folds. And every number here is measured on the validation folds, which have been read
# many times over by the time a case study reaches this notebook.
#
# **Next**: [`06_gbm`](06_gbm.ipynb) fits gradient boosting to the same targets. It minimizes the
# same squared error by default, so the comparison to watch is whether the gap between the raw and
# the clipped target is still there under a model that splits rather than solves - and
# [`10_model_analysis`](10_model_analysis.ipynb) is where that question is settled rather than
# guessed at.
