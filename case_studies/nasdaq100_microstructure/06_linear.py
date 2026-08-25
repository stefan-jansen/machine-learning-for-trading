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
# # NASDAQ-100 microstructure: what a linear map of order flow is worth
#
# This is the first model fitted in the case study, and deliberately the simplest one that can
# use the whole feature set: a linear map from the microstructure features observed at one
# minute to the return over the next fifteen.
#
# Fitting it first is not a formality. A linear model cannot represent an interaction between
# two features, so whatever it achieves is what the features provide on their own. Every later
# model in this case study is more expressive and more expensive, and what each one adds is only
# readable against this number.
#
# The features are built from overlapping views of the same order flow - several spread
# measures, several imbalance measures, and the same quantities at several lookbacks - so many
# columns carry almost the same information. A design matrix like that is **collinear**, and
# many different coefficient vectors fit the training window about equally well. Ordinary least
# squares has no way to choose between them and will spend a large positive coefficient on one
# feature against a large negative one on a near-copy. **Regularization** is the fix: penalize
# coefficient size, so a solution that spreads weight across correlated features is preferred to
# one that plays them off against each other. How much penalty is an empirical question, and the
# grid below is the experiment that answers it.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Read the set of models this case study declares for a label, and say which estimator and
#   which hyperparameters each declared name resolves to.
# - Bind those declarations to the data on disk and check, before anything is fitted, that every
#   configuration will be measured on the same symbols, the same timestamps and the same folds.
# - Fit a population of models on walk-forward folds and publish one complete set of validation
#   predictions per configuration.
# - Tell apart the two things a penalty can do to a collinear feature set - shrink correlated
#   coefficients towards each other, or select a few and zero the rest - and read from the
#   results which one order-flow data rewards.
# - Recognise when an information coefficient is an artifact of a model that scored fewer
#   decision times than its neighbours, and use coverage to rule it out.
# - Run configurations of your own into a private copy of the run log, and have them compared on
#   the same footing as the ones shipped here.
#
# **Docker image**: `ml4t`
#
# **Book reference**: Chapter 11, Section 11.2 (Regularized Linear Models). Chapter 6,
# Section 6.7 (Search accounting and run logging) introduces the run log this notebook
# writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# and [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds.
#
# **What it writes**: one training run and one complete validation prediction set per
# configuration, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a named population.
# [`14_backtest`](14_backtest.ipynb) reads that population, runs every member against the
# equal-weight baseline, and selects on validation backtest Sharpe. **Selection happens there,
# not here.** This notebook ranks configurations by information coefficient to show what
# regularization does to a collinear feature set; that ranking decides nothing.

# %%
"""Fit the declared NASDAQ-100 microstructure linear population on the validation folds."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from case_studies.research import (
    declared_labels,
    load_model_configs,
    model_requests,
    narrows_declared_catalog,
    open_study,
    plan_models,
    primary_label,
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
    "nasdaq100_microstructure",
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="06_linear",
)

# %% [markdown]
# ## 1. Which label, and which models
#
# A label is the thing being predicted. This case study defines four in `config/setup.yaml`.
# `fwd_ret_15m`, the return over the fifteen minutes after the decision minute, is the primary
# one - the horizon the strategy chapters trade. `fwd_ret_5m` and `fwd_ret_60m` are the same
# construction at shorter and longer horizons, kept so the effect of the prediction horizon can
# be examined separately, and `fwd_dir_15m` is the classification form of the primary label.
#
# **`LABELS = []` fits all of them in one run**, which is what the empty list means: not "no
# labels" but "every label whose training menu declares this family". Each label has its own
# menu at `config/training/{label}.yaml`, which lists, family by family, the named
# configurations to fit for that label, so the grid is not necessarily the same on each - the
# classification label declares fewer estimators than the regression ones do.

# %%
declared_labels(study, "linear")

# %% [markdown]
# Each name in the menu resolves to a preset file in the shared directory
# `case_studies/config/{model_type}/`, which holds that configuration's hyperparameters. The
# frame below is the menu for every declared label, with each name resolved to the estimator
# class it names and the arguments that class is constructed with. To change what runs, edit the menu or the
# presets rather than this notebook.
#
# The grid covers the two shapes a penalty can take:
#
# - **Ridge** penalizes the sum of squared coefficients. It shrinks correlated coefficients
#   towards each other and keeps every feature, at a strength set by `alpha`. The grid steps
#   `alpha` by powers of ten across eleven orders of magnitude, because the useful value depends
#   on the scale and the collinearity of the design matrix and neither is known in advance.
# - **Lasso** penalizes the sum of absolute coefficients, which drives some of them exactly to
#   zero: it selects features rather than shrinking them. **ElasticNet** mixes the two.
#
# Lasso and ElasticNet are parameterized here by `alpha_frac` rather than a raw penalty. For any
# fold there is a threshold penalty $\alpha_{\max}$ - the smallest one that zeros every
# coefficient - computed from that fold's own data. `alpha_frac` is the fraction of it to apply,
# so one declared `alpha_frac` means the same thing on every fold, while a fixed raw penalty
# would mean something different on each.

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
# member set from the canonical population. A population is immutable once written, so such a
# run must publish under its own name; the guard catches either knob here rather than several
# cells later in a message about hashes.
#
# The comparison is over `(label, config_name)` pairs rather than a row count, because a count
# can match while the members differ.

# %%
if narrows_declared_catalog(study, "linear", configs) and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the declared configurations, so it cannot publish "
        "the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry says which estimator to fit. It does not say which feature columns exist today,
# where the walk-forward folds fall, or which symbol-timestamp pairs have both a feature row and
# a label. **Resolving** a request is the step that goes and finds all of that: it reads the
# label and feature files, computes the fold boundaries from the walk-forward parameters in
# `config/setup.yaml`, works out the exact set of rows each fit is expected to predict, and
# turns any data-dependent hyperparameter into the number it will actually use - each fold's own
# $\alpha_{\max}$ times `alpha_frac`, in the case of Lasso.
#
# **Planning** derives all of that without holding it. `plan_models` works out every training
# and prediction identity from the declarations and placeholder folds, so the whole panel of
# configurations is priced before a single fit, and nothing but the plan is alive when it
# finishes. Resolving each request is the same arithmetic at ruinous cost here: a resolved
# request carries its prepared folds, and this case study is a minute panel, so one resolved
# request per configuration holds the same standardized design matrix sixteen times over. Execution then walks folds on the outside and configurations on the inside, so one fold
# set is live at a time however many configurations were declared.
#
# The plan is the population, written down before anything is fitted. Two things to check in it:
#
# - **One row per configuration and checkpoint**, matching the catalog above. A missing row is a
#   configuration that will not be fitted, and a run that later produces a different set than it
#   declared here fails rather than quietly publishing what it happened to produce.
# - **`training_hash` differs wherever the configurations differ, and repeats wherever they do
#   not.** Two configurations that resolve to the same computation share an identity and are
#   fitted once; two that differ must not collide.
#
# The `training_hash` is the identity of that computation, derived from everything that can
# change its result. [`RUN_LOG.md`](../RUN_LOG.md#identity) sets out what goes into one and what
# follows from it.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
)
plan = plan_models(study, requests=requests)

pl.DataFrame(
    {
        "config_name": [member.config_name for member in plan.members],
        "checkpoint_kind": [member.checkpoint_kind for member in plan.members],
        "checkpoint_value": [member.checkpoint_value for member in plan.members],
        "training_hash": [member.training_hash for member in plan.members],
        "prediction_hash": [member.prediction_hash for member in plan.members],
    }
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
# which is what a walk-forward prediction set is: each timestamp predicted by a model that saw
# only data before it. The run then writes a `training_runs` row and the fitted coefficients, a
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
population_name = POPULATION_NAME or "nasdaq100_microstructure-linear-validation-v1"
execution, population = run_model_population(
    study, plan, population_name=population_name, supersedes=SUPERSEDES_POPULATION or None
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
# study = open_study("nasdaq100_microstructure", workspace="~/ml4t-experiments")
# configs = load_model_configs(
#     study, "linear", labels=["fwd_ret_15m"], config_names=["ols", "ridge_a1.0", "ridge_a10.0"]
# )
# requests = model_requests(study, configs)
# plan = plan_models(study, requests=requests)
# execution, population = run_model_population(study, plan, population_name="my-linear-v1")
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
# the ones shipped here.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest, including how to
# rehearse on a reduced universe first.

# %% [markdown]
# ## 4. What came out
#
# One row per configuration, read back from the registry. `ic_mean` is the **information
# coefficient**: at each decision time, rank the constituents by the model's prediction, rank
# them by the return they went on to earn over the next fifteen minutes, correlate the two
# rankings, and average that correlation over the validation period. It measures whether the
# model ranks names correctly, on a scale where zero is no relationship. Intraday values are
# smaller than the daily-horizon ICs elsewhere in the book: a fifteen-minute return is mostly
# noise, and a few thousandths of consistent rank correlation is a real effect at this horizon.
#
# **`ic_n_days` does not count days here.** The column is named for the daily case studies, where
# one decision date produces one cross-section. This case study decides every fifteen minutes,
# so the stored count is a count of *decision times* - tens of thousands of them across the
# validation period, not hundreds. It is still the right thing to compare configurations on,
# because every configuration is counted the same way, but do not read it as a number of trading
# days.
#
# What the count is for: a model whose coefficients collapse to one or two features predicts
# nearly the same value for every constituent at some timestamps, and a constant has no rank
# correlation with anything, so those timestamps contribute nothing. Its `ic_mean` is then an
# average over fewer decision times than its neighbours', chosen by where it happened to stay
# non-degenerate, and comparing it with theirs compares two different samples. `full_coverage`
# marks the configurations measured on all of them. A cross-section also has to be wide enough
# to rank at all: a timestamp scores only where at least five constituents carry both a finite
# prediction and a finite realized return.

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
        "n_folds",
        "training_hash",
        "prediction_hash",
    )
    .sort(["label", "ic_mean"], descending=[False, True])
    # On both keys: the three return horizons declare the same sixteen configuration names, so
    # joining on the name alone would attach each label's estimator to every label's row and
    # triple the frame.
    .join(
        configs.select("label", "config_name", "model_class", "params"),
        on=["label", "config_name"],
        how="left",
    )
)

if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("linear execution returned a partial prediction set")

# Per label, not across them: a five-minute horizon and a sixty-minute one do not score the
# same number of decision times, so one global maximum would mark every configuration of the
# shorter-calendar labels incomplete and drop them from every chart that filters on this.
catalog = catalog.with_columns(
    full_coverage=pl.col("ic_n_days") == pl.col("ic_n_days").max().over("label")
)
for row in (
    catalog.group_by("label")
    .agg(full_times=pl.col("ic_n_days").max())
    .sort("label")
    .iter_rows(named=True)
):
    print(f"{row['label']}: full coverage is {row['full_times']:,} scored decision times")

primary = primary_label(study)
present = sorted(set(catalog.get_column("label")))
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
# The frame below is the comparison this notebook can make only because every declared label was
# fitted in one run. The features are the same and the folds are the same throughout, and the
# three return horizons share one menu of sixteen configurations, so down those three rows the
# only thing that changes is how far ahead the model is asked to look. `fwd_dir_15m` declares its
# own thirteen classifiers, because a ridge regression has nothing to say about a binary outcome;
# read that row against `fwd_ret_15m`, which is the return it was cut from, rather than as a
# fourth member of one sweep. `configurations` is what tells them apart.
#
# `ic_mean` is defined for every row, which is what puts all four on one axis: for the classifier
# it correlates the predicted probability with the realized return, exactly as it does for a
# predicted return. Read `n_positive` against `configurations`: a label where the whole grid sits
# on one side of zero is saying something about the horizon, and a label whose grid straddles
# zero is saying that the spread across configurations is larger than anything the features
# contribute at that horizon.

# %% tags=["results"]
by_label = (
    catalog.filter("full_coverage")
    .group_by("label")
    .agg(
        task=pl.col("task").first(),
        configurations=pl.len(),
        scored_times=pl.col("ic_n_days").max(),
        best_ic=pl.col("ic_mean").max(),
        worst_ic=pl.col("ic_mean").min(),
        n_positive=(pl.col("ic_mean") > 0).sum(),
    )
    .sort("best_ic", descending=True)
)
by_label

# %% [markdown]
# ### How the penalty grid ranks
#
# One panel per label on one shared vertical scale, so the horizons are compared rather than each
# one rescaled to fill its own panel. Only the configurations measured at all of their own label's
# scorable decision times are charted; the partial-coverage ones are in the table above with
# `full_coverage` false, and are left out here because their IC is an average over a different set
# of timestamps.
#
# The configurations are held in one order across the panels - their ranking on the primary label
# - so a panel that does not descend is a horizon that orders the grid differently.


# %%
full = catalog.filter("full_coverage")
config_order = (
    full.filter(pl.col("label") == panel_labels[0])
    .sort("ic_mean", descending=True)
    .get_column("config_name")
    .to_list()
)

# `shared_yaxes` matches axes across columns, so with one column it does nothing and each panel
# would be rescaled to fill itself. Matching every row to the first is what puts the horizons on
# one vertical scale, which is what stacking them is for.
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
    # The classification label declares its own configurations; they keep them and are appended
    # after the shared order rather than dropped from the figure.
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
    if row > 1:
        fig_ic.update_yaxes(matches="y", row=row, col=1)
fig_ic.update_xaxes(
    title_text=f"Configuration, ordered by rank on {panel_labels[0]}",
    tickangle=-45,
    row=len(panel_labels),
    col=1,
)
fig_ic.update_layout(
    title="Which side of zero the grid sits on depends on the horizon, not the penalty",
    height=280 * len(panel_labels),
    width=1100,
    margin=dict(t=90),
)
# Which labels clear zero is a fact about the frame, so the alt text reads it rather than
# asserting it. A sentence about "the" leading configuration was true of the one label this
# notebook used to fit and is false of the set it fits now.
side_text = "; ".join(
    f"{row['label']} has {row['n_positive']} of {row['configurations']} above zero"
    for row in by_label.sort("label").iter_rows(named=True)
)
show_plotly_with_alt(
    fig_ic,
    "Bar charts of mean validation information coefficient for every full-coverage linear "
    "configuration, one panel per declared label on one shared vertical scale, each panel's "
    "highest bar in amber and the rest in dark navy, with a dashed zero line across each. The "
    "bars are held in the primary label's ranking order in every panel, so a panel that does not "
    f"descend is a horizon that orders the grid differently. Counted from the frame: {side_text}.",
)

# %% [markdown]
# ### What shrinkage does on its own
#
# The bar chart mixes three estimators. Tracing IC across the Ridge penalty alone isolates the
# effect of shrinkage, with the estimator, the features and the folds all held fixed and only
# `alpha` moving. The alpha is read from each configuration's declared parameters rather than
# parsed out of its name, so the curve plots what was fitted. One line per label, because the
# question is whether the penalty behaves the same way at every horizon.

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
    # One colour per label, and amber is not among them: the peak marker is amber, so a line drawn
    # in it would swallow its own ring. Wrapping a short palette would give two labels the same
    # colour and two identical legend swatches, so this raises instead.
    line_colors = [
        COLORS["blue"],
        COLORS["copper"],
        COLORS["positive"],
        COLORS["negative"],
        COLORS["slate"],
        COLORS["recede"],
    ]
    if len(panel_labels) > len(line_colors):
        raise ValueError(
            f"{len(panel_labels)} labels declared but only {len(line_colors)} distinct line "
            "colours; add colours rather than letting two labels share one"
        )
    fig_alpha = go.Figure()
    peaks: list[str] = []
    for index, label in enumerate(panel_labels):
        series = ridge.filter(pl.col("label") == label)
        if not series.height:
            continue
        log_alpha = np.log10(series.get_column("alpha").to_numpy())
        values = series.get_column("ic_mean").to_numpy()
        peak = int(np.argmax(values))
        color = line_colors[index]
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
        peaks.append(f"{label} peaks at 1e{int(round(log_alpha[peak]))}")
    fig_alpha.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
    fig_alpha.update_layout(
        title="Ridge IC against penalty strength, over eleven orders of magnitude",
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
        "the Ridge penalty, one line per label over eleven orders of magnitude, each line's "
        "highest point ringed in amber, against a dashed zero line, and no two lines sharing a "
        f"colour. Read off the frame: {'; '.join(peaks)}.",
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
# **Where a Ridge curve turns tells you how collinear the design matrix is.** Each line is flat
# while the penalty is too weak to bind, rises as shrinkage starts collapsing groups of
# near-duplicate order-flow features onto their common direction, and falls once the penalty is
# strong enough to erode the signal along with the noise. The distance from a peak back to
# unregularized OLS is the part of the signal that multicollinearity was burying. On a feature
# set that was close to orthogonal the same curve would be nearly flat, and that comparison is
# worth making on your own data before spending a grid on it.
#
# **One feature set, three horizons, and the penalty that suits them need not be the same.** The
# alt text under the sweep reads each line's peak off the frame. The design matrix does not
# change between the horizons - the same order-flow features, the same folds, the same sixteen
# configurations - so a peak that moves is telling you how much of the collinear structure is
# worth keeping at that horizon, not that one horizon has a different feature set. Fitting one
# label and reporting its optimal alpha as the case study's would hide that.
#
# **Read the ranking with the coverage column or it will mislead you.** The most aggressive L1
# settings can post a high raw IC while not being comparable to the rest: they zero all but a
# couple of features on several folds, predict a near-constant value at those timestamps, and
# contribute no correlation there. Read without `ic_n_days` such a table says hard feature
# selection wins; read with it, the same table says those configurations failed on part of the
# sample. The general lesson is that a metric averaged over a set the model itself selected is
# not a metric.
#
# **The absolute level is small at every horizon here, and that is the reading an intraday
# forecast deserves.** A daily-horizon equity signal reports IC in the hundredths; over minutes
# a few thousandths is what a working signal looks like, because almost all of a five- or
# fifteen-minute return is noise. The `by_label` frame is where to check whether the level rises
# with the horizon, which is the direction the noise argument predicts. Whether any of it is
# tradable is not answerable from IC at all - at these horizons the position turns over
# constantly, so costs decide it, and costs are not in this notebook.
#
# **None of this selects anything.** IC measures whether predictions rank names correctly, not
# whether a strategy trading them makes money after costs and turnover. Those are different
# questions and a configuration can win the first while losing the second. Selection is on
# validation backtest Sharpe over the population this notebook just published, and it happens in
# [`14_backtest`](14_backtest.ipynb).
#
# **Known limitations.** The IC here is an average of per-timestamp rank correlations with no
# adjustment for the serial dependence that overlapping intraday returns create, and that
# dependence is heavier at sixty minutes than at five, so it is a ranking diagnostic rather than
# a test and it is not equally conservative across the panels. The grid is a one-dimensional sweep of penalty strength
# at fixed features and fixed folds, so it says nothing about interactions between the penalty
# and either. And every number here is measured on the validation folds, which have been read
# many times over by the time a case study reaches this notebook.
#
# **Next**: [`07_gbm`](07_gbm.ipynb) asks whether gradient boosting can find interactions in the
# order flow that a linear map cannot represent at all.
