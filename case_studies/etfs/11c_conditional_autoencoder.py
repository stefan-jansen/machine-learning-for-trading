# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -papermill
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
# # ETFs: the same factor structure with a network in place of the linear map
#
# [`11b_ipca`](11b_ipca.ipynb) made a fund's exposure a **linear** function of its feature row.
# That is a strong claim: doubling a fund's momentum doubles its exposure to the momentum-like
# factor, and two features can only combine by adding. The conditional autoencoder keeps everything
# else about that structure - features to exposures, exposures times factor returns, both fitted on
# the training window alone - and replaces the linear map with a small neural network.
#
# **So the difference between these two notebooks is the shape of one function and nothing else.**
# That is what makes them worth reading against each other. If the network ranks the cross-section
# better, the relationship between a feature and the exposure it implies is not linear; if it does
# not, the extra capacity bought nothing and the linear constraint was not what was holding IPCA
# back. Neither outcome is decided here - [`13_model_analysis`](13_model_analysis.ipynb) compares
# the populations with every family in front of it.
#
# **The network is trained rather than solved, which changes what this notebook publishes.** IPCA
# runs alternating least squares to convergence and has one fitted state per fold. A network is
# trained for a declared number of passes over the training rows - **epochs** - and has a
# meaningful state at every one of them. `case_studies/config/cae/cae.yaml` declares 50 epochs
# saved every 5, so each label produces ten scoreable models rather than one, each registered
# separately.
#
# **All ten are published, and that is deliberate.** An earlier version of this notebook fixed a
# reporting checkpoint in advance, showed that the best validation checkpoint was a different one,
# and reported the fixed one anyway. That is the right instinct about the trap - picking the epoch
# with the best validation IC is choosing after seeing the answer - but the wrong remedy, because
# it throws away nine models the case study paid to train and leaves the choice to be made again,
# undocumented, downstream. Publishing the whole schedule lets
# [`14_backtest`](14_backtest.ipynb) select on validation backtest Sharpe with every candidate
# visible and the rule stated. **Selection happens there, not here.**
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a conditional autoencoder relaxes relative to instrumented PCA, and what it keeps.
# - Read an epoch schedule out of a declared configuration and say how many prediction sets a
#   label will owe.
# - Read a curve of out-of-sample ranking accuracy against training epoch, and tell a model still
#   learning from one that has begun fitting its training window.
# - Say why the training device is part of a model's identity rather than a note about how it ran.
#
# **Book reference**: Chapter 14, Section 14.6 (Nonlinear conditional factor models). Chapter 6,
# Section 6.7 (Search accounting and run logging) introduces the run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) for the feature matrix,
# [`05_evaluation`](05_evaluation.ipynb) for the walk-forward folds, and
# [`11b_ipca`](11b_ipca.ipynb), which is this estimator with the network replaced by a linear map.
#
# **What it writes**: one training run per label and one complete validation prediction set per
# label and epoch checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a population named for this model. The family splits across
# five notebooks, so each publishes its own population rather than one shared one.

# %%
"""Fit the declared ETF conditional-autoencoder population on the walk-forward folds."""

# Load PyTorch's bundled CUDA runtime before other ML4T libraries.
import torch  # noqa: F401

# isort: split
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
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = ""
DEVICE: str = ""

MODEL_NAME = "cae"

# %%
study = open_study("etfs", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and both do: `fwd_ret_21d`,
# the total return over the 21 trading days after the decision date, and `fwd_ret_5d`, the same
# thing over five. The network's exposures are fitted against the label, so these are two separate
# fits rather than one fit scored twice. `LABELS` restricts the run to a subset when you want one.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/cae/cae.yaml` declares the estimator and its schedule. `n_factors` is the
# width of the bottleneck - how many exposures the network maps a feature row onto. `n_epochs` and
# `checkpoint_interval` decide how many prediction sets each label owes: 50 epochs saved every 5 is
# ten. They are declared beside the architecture rather than passed in here, because a run that
# quietly trained for fewer would publish a different population under the same name.

# %%
configs = load_model_configs(
    study,
    "latent_factors",
    labels=LABELS or None,
    config_names=[MODEL_NAME],
)
configs.select("label", "config_name", "params")

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
#
# The device is checked in the same cell. A network trained on a GPU and the same network trained
# on a CPU accumulate their sums in different orders and reach different weights, so the device is
# part of what the fitted model is and is recorded inside the computation's identity rather than
# beside it. `PUBLISHED_DEVICE` is the device this population was fitted on; the runner refuses to
# substitute a CPU for a requested GPU rather than publishing a different model under the published
# name, so on a machine with no NVIDIA card this notebook stops at the next cell. Set
# `DEVICE="cpu"` and pass a `POPULATION_NAME` to fit the same schedule there.
#
# This is where the family stops being uniform. [`11a_pca`](11a_pca.ipynb) and
# [`11b_ipca`](11b_ipca.ipynb) declare CPU because they have no GPU implementation at all; the
# three neural members declare CUDA. One family, two runtimes, and both inside the identity.

# %%
PUBLISHED_DEVICE = "cuda"
device = DEVICE or PUBLISHED_DEVICE
print(f"training device: {device}")

declared = load_model_configs(study, "latent_factors", config_names=[MODEL_NAME])
narrowed = set(configs.get_column("label")) != set(declared.get_column("label"))
if (narrowed or device != PUBLISHED_DEVICE) and not POPULATION_NAME:
    raise ValueError(
        f"this run fits {configs.height} of the {declared.height} declared labels on device "
        f"{device!r}, which is not the complete declared catalog on {PUBLISHED_DEVICE!r}, so it "
        "cannot publish the canonical population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# A menu entry says which estimator to fit. It does not say which fund-date pairs have both a
# feature row and a label, or where the walk-forward folds fall. **Resolving** a request goes and
# finds that: it reads the label and feature files, computes the fold boundaries from the
# walk-forward parameters in `config/setup.yaml`, and works out the exact set of rows each fit is
# expected to predict. It fits nothing, so the plan can be read before any training starts.
#
# Four things to check in it:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across the rows.** They are
#   the width of the network's input, the number of funds in the cross-section, and the fund-date
#   pairs to be predicted. A row that differs is a label measured on a different sample.
# - **`folds` is the same on both rows**, and equals the number of walk-forward splits
#   [`05_evaluation`](05_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   is scored once, at the end of the case study; any of it visible here would mean it had been
#   used to choose something.
# - **`checkpoints` is 10**, the epoch schedule declared above. Multiply it by the number of rows
#   to get the number of candidate models this notebook is about to create.
#
# Each row also carries a `training_hash`: the identity of that computation, derived from
# everything that can change its result. [`RUN_LOG.md`](../RUN_LOG.md#identity) sets out what goes
# into one and what follows from it.

# %%
requests = model_requests(
    study,
    configs,
    execution_tier=EXECUTION_TIER,
    overrides={"device": device},
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
    "checkpoints",
    "validation_start",
    "validation_end",
)

# %% [markdown]
# The plan is also where a short population is visible. A run that fitted fewer labels than the
# menu declares would still print an IC table and still register its rows; what it would not do is
# announce the gap.

# %%
planned_pairs = set(zip(plan["label"], plan["config_name"], strict=True))
requested_pairs = set(zip(configs["label"], configs["config_name"], strict=True))
if planned_pairs != requested_pairs:
    raise RuntimeError(
        "the plan does not match the loaded CAE menu; "
        f"missing {sorted(requested_pairs - planned_pairs)}, "
        f"unexpected {sorted(planned_pairs - requested_pairs)}"
    )
print(f"{len(requested_pairs)} label-configuration pairs")

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` runs every resolved request. For one request it walks the folds, and on
# each one:
#
# 1. takes the fund-dates inside that fold's training window,
# 2. trains the two networks together - one mapping a fund's feature row to its exposures, one
#    producing each date's factor returns from that date's cross-section - by gradient descent on
#    the error between the reconstructed and realised returns, writing the weights to disk every
#    five epochs,
# 3. applies each saved set of weights to the validation dates' feature rows to get exposures
#    there, and multiplies them by the training-window average factor returns to get a predicted
#    return.
#
# Step 3 is what makes one training run produce ten results. The fold predictions are concatenated
# into one series per checkpoint covering the whole validation period, and each becomes its own
# registered prediction set with its own identity.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, computed from the resolved specifications and written down before the first fit.
# Afterwards every member must exist and be complete, which is what makes the downstream comparison
# well defined - [`14_backtest`](14_backtest.ipynb) backtests this population, not whatever
# predictions happen to be in the registry - and it is why a configuration that raises fails the
# whole call rather than publishing a population one member short.
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A population is the set of
# prediction identities, so anything that moves a training identity - a changed epoch schedule as
# much as a changed label menu - produces a different set under the same name, and the registry
# refuses to write it without being told which snapshot it supersedes. It is empty here because
# this notebook, run as it stands, reproduces the members already published under that name rather
# than changing them, and reproducing a published list is not a replacement. Fill it in when you
# have changed something that moves an identity and want the new set to take the name; the error
# raised on the attempt tells you which hash to name. A reduced-scale run passes it empty whatever
# the default is: a population produced under a reduction is thrown away with the workspace it was
# written to, so it has no lineage to extend.

# %%
population_name = POPULATION_NAME or f"etfs-{MODEL_NAME}-validation-v1"
execution, population = run_model_population(
    study,
    resolved,
    population_name=population_name,
    supersedes=SUPERSEDES_POPULATION or None,
)

published_sets = sum(len(run.predictions) for run in execution.runs)
print(f"{len(execution.runs)} configurations, {published_sets} prediction sets")
print(f"population {population.name}: {len(population.members)} members")

# %% [markdown]
# A second run of this notebook trains nothing. Every identity is re-derived from the inputs, the
# registry already holds the matching rows and the saved weights, and the runner returns the stored
# result rather than training again - so re-running it unchanged costs the time it takes to read
# the data. The latent-factor runner reports no per-fold fitted-or-reused breakdown, unlike the
# linear and boosted families, so the counts above are of what the population holds rather than of
# what this particular run computed.
#
# ### Running configurations of your own
#
# The published run log is read-only. To add runs, open the study against a workspace, which holds
# its own registry and artifacts and reads the same labels and features:
#
# ```python
# study = open_study("etfs", workspace="~/ml4t-experiments")
# configs = load_model_configs(study, "latent_factors", labels=["fwd_ret_5d"], config_names=["cae"])
# requests = model_requests(study, configs, overrides={"device": "cuda"})
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-cae-v1")
# ```
#
# To change the architecture or the schedule, edit `case_studies/config/cae/cae.yaml`. That changes
# this configuration's identity, so its result registers as a new row beside the old one rather
# than replacing it - and that includes `n_epochs` and `checkpoint_interval`, which decide how many
# members the population has. Give the run its own `population_name`: a name refers to one set of
# members permanently, and reusing it for a different set raises.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per label and epoch checkpoint, read back from the registry. `ic_mean` is the
# **information coefficient**: on each validation date, rank the funds by the model's prediction,
# rank them by the return they went on to earn, correlate the two rankings, and average that daily
# correlation over the validation period.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it decides which
# rows are comparable with each other. A network that has settled into predicting nearly the same
# value for every fund on a date gives that date no spread to rank, and the date drops out of the
# average - so that checkpoint's `ic_mean` is an average over a different sample from its
# neighbours'. `full_coverage` marks the rows measured on every date their own label offers, and
# the curves below are restricted to those. **Coverage is judged within a label**, because a
# 21-day forward return runs out of window earlier than a five-day one and one global maximum would
# mark the whole 21-day grid incomplete for a reason that has nothing to do with the models.
#
# The published catalog is checked against the population planned before fitting rather than
# against its own row count, because a run that lost a member would otherwise report a shorter
# table and nothing else.

# %% tags=["results"]
catalog = execution.catalog_rows.select(
    "label",
    "config_name",
    "complete",
    "checkpoint_kind",
    "checkpoint_value",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "n_folds",
    "training_hash",
    "prediction_hash",
).sort("label", "checkpoint_value")

if set(catalog.get_column("prediction_hash")) != set(population.members):
    raise RuntimeError("the published catalog differs from the population planned before fitting")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("a partial CAE prediction set cannot pass to backtesting")

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
print(f"at {catalog.n_unique('checkpoint_value')} checkpoints each")
catalog.select("label", "checkpoint_value", "ic_mean", "ic_std", "ic_n_days", "full_coverage").head(
    12
)

# %% [markdown]
# ### What more training does
#
# Each line traces one label's out-of-sample IC as epochs are added. This is the figure the
# checkpoint dimension exists to produce, and it separates two things a single end-of-training
# number cannot.
#
# A line that rises and then falls has an interior optimum: the network was still learning, then
# began fitting the training window at the expense of the validation folds. A line that wanders
# around zero without trend never had anything to learn, and its highest point is wherever the
# noise happened to peak. Both produce a respectable-looking maximum, which is why the maximum is
# not what a checkpoint is judged on and why all ten are published rather than the best one.
#
# One panel per label, each with its own vertical scale, because a horizon with something to learn
# and one without would be averaged into a single indistinct band if they shared axes.

# %%
curves = catalog.filter("full_coverage").sort("label", "checkpoint_value")
fig = make_subplots(
    rows=len(panel_labels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=[
        f"{label} ({'primary' if label == primary else 'variant'})" for label in panel_labels
    ],
)
for row, label in enumerate(panel_labels, start=1):
    series = curves.filter(pl.col("label") == label)
    fig.add_trace(
        go.Scatter(
            x=series.get_column("checkpoint_value").to_list(),
            y=series.get_column("ic_mean").to_list(),
            mode="lines+markers",
            name=label,
            showlegend=False,
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=6, color=COLORS["blue"]),
        ),
        row=row,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"], row=row, col=1)
    fig.update_yaxes(title_text="Mean IC (validation)", row=row, col=1)
fig.update_xaxes(title_text="Training epoch", row=len(panel_labels), col=1)
fig.update_layout(
    title="Out-of-sample ranking accuracy against training epoch",
    height=320 * len(panel_labels),
    width=800,
    margin=dict(t=90),
)
# Read off the frame rather than asserted, so the alt text stays true of the next run.
_spans = "; ".join(
    f"{label} from {curves.filter(pl.col('label') == label).get_column('ic_mean').min():.3f} "
    f"to {curves.filter(pl.col('label') == label).get_column('ic_mean').max():.3f}"
    for label in panel_labels
)
show_plotly_with_alt(
    fig,
    "Line chart of mean validation information coefficient against training epoch, one panel per "
    "label, each with its own vertical scale and a dashed zero line. Counted from the frame: "
    f"{_spans}.",
)

# %% [markdown]
# ### What one checkpoint's number averages
#
# Each row of the catalog is one number over the whole validation period, and the eight folds
# behind it are trained on different windows and scored on different years. The table below
# recomputes each fold's mean IC from the published predictions of one checkpoint, so the
# disagreement between folds measuring the same model can be compared against the differences
# between checkpoints in the curve above.
#
# The checkpoint shown is the last of the schedule. That is where training stopped, not a
# selection: reading these eight numbers at the checkpoint with the highest `ic_mean` and
# reporting them as the model's would be the choice this notebook publishes all ten checkpoints
# to avoid.

# %%
published = {result.hash: result for run in execution.runs for result in run.predictions}
last_checkpoint = catalog.get_column("checkpoint_value").max()


def _fold_ic(prediction_hash: str) -> pl.DataFrame:
    """Mean daily rank correlation of prediction against realised return, within each fold."""
    return (
        published[prediction_hash]
        .load()
        .group_by("fold", "timestamp")
        .agg(pl.corr("prediction", "actual", method="spearman").alias("ic"))
        .group_by("fold")
        .agg(
            pl.col("ic").mean().alias("fold_ic"),
            pl.len().alias("validation_dates"),
        )
        .sort("fold")
    )


by_fold = {
    row["label"]: _fold_ic(row["prediction_hash"])
    for row in catalog.filter(pl.col("checkpoint_value") == last_checkpoint).iter_rows(named=True)
}
for label in panel_labels:
    ic = by_fold[label].get_column("fold_ic")
    print(
        f"{label} at epoch {last_checkpoint}: mean {ic.mean():+.4f}, "
        f"standard deviation across folds {ic.std():.4f}, "
        f"{(ic < 0).sum()} of {ic.len()} folds negative"
    )

by_fold[panel_labels[0]]

# %% [markdown]
# ## 5. What to notice
#
# **The network orders the cross-section better than the linear map does, on both labels and at
# every checkpoint it published.** At the end of the declared schedule, 50 epochs, the mean
# validation IC is **+0.072**
# on the primary label and **+0.046** on the five-day variant, against **+0.041** and **+0.035**
# for [`11b_ipca`](11b_ipca.ipynb) and **-0.033** and **+0.010** for
# [`11a_pca`](11a_pca.ipynb). The comparison holds across the whole checkpoint dimension rather
# than at one point: the primary label's ten checkpoints run from +0.058 to +0.080, all of them
# above IPCA's +0.041. On the five-day label the margin is not uniform - the earliest checkpoint,
# +0.035 at five epochs, is indistinguishable from IPCA's +0.035, and the nine later ones sit
# between +0.040 and +0.052.
#
# **The checkpoint curve is not evidence for an epoch.** On the primary label the ten checkpoints
# span 0.022 of IC end to end. The eight fold ICs behind the last of them - +0.018, -0.023,
# +0.073, +0.169, +0.160, -0.079, +0.149, +0.110 - have a standard deviation across folds of 0.092,
# four times that span. The highest checkpoint, +0.080 at 45 epochs, differs from the one before it
# by 0.019, which is a fifth of the disagreement between folds measuring the same checkpoint.
# Nothing in this table distinguishes an epoch worth stopping at from an epoch that happened to
# score well on eight folds, which is the reason all ten checkpoints are published rather than one
# and the reason nothing downstream may select among them on validation IC.
#
# **No checkpoint collapsed, and that had to be checked rather than assumed.** `ic_n_days` is 1,995
# on every primary-label row and 2,011 on every five-day row - the full count each label offers -
# so `full_coverage` is true for all twenty published models. A network that had settled into
# predicting nearly the same value for every fund would have dropped dates out of its average and
# left one row comparing against a different sample from its neighbours'. The training schedule
# here never went there, on either label, at any of the ten checkpoints.
#
# **What this establishes for the rest of the family.** Three members now differ in one thing each,
# fitted on the same panel over the same folds. PCA reads no features. IPCA reads the 71 feature
# columns through a linear map. This notebook reads them through a network. Conditioning at all is
# worth about +0.074 of IC on the primary label; replacing the linear map with the network is worth
# about a further +0.031. The second gap is real and smaller than the first, and it is bought with
# an architecture, a training schedule, ten checkpoints and a GPU, against a solver that takes an
# iteration cap and a tolerance. [`13_model_analysis`](13_model_analysis.ipynb) is where that
# trade is priced against every other family in front of it.
#
# **Known limitations.** The architecture, the bottleneck width and the schedule are declared in
# `case_studies/config/cae/cae.yaml` rather than searched, so nothing here says this network is the
# right one - only what this one did. Fifty epochs is where training stops, not where anything
# established it should stop. Two of the eight folds are negative on the primary label and one on
# the five-day, and the fold-to-fold dispersion is essentially unchanged from IPCA's
# (0.092 against 0.099), so the improvement is in the average rather than in how much any
# single fold can be relied on. The panel narrows from 96 funds in the
# first fold to 90 in the last. The published result is a CUDA result, and a reader who refits on
# CPU gets a different identity registered beside it rather than the same numbers. And every number
# here is measured on validation folds that have been read many times over by the time a case study
# reaches this notebook.

# %% [markdown]
# **Next**: [`11d_stochastic_discount_factor`](11d_stochastic_discount_factor.ipynb) leaves the
# two-stage shape behind entirely - it prices the cross-section rather than splitting the problem
# into exposures and factor returns - and
# [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) keeps this notebook's network
# and bottleneck while dropping the factor interpretation, which makes it the family's own control.
