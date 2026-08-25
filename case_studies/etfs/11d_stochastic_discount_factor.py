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
# # ETFs: a model that prices the cross-section instead of predicting it
#
# The three members before this one share a shape. [`11a_pca`](11a_pca.ipynb),
# [`11b_ipca`](11b_ipca.ipynb) and
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) all split the problem in two:
# estimate what each fund is exposed to, estimate what those exposures earned, multiply. They
# differ in how the first half is written down - not at all, a linear map, a network - and in
# nothing else.
#
# **The stochastic discount factor does not split it.** It asks for a single series that prices
# every fund at once: a weighting of the cross-section whose covariance with each fund's return
# accounts for that fund's average return. That is a statement about *no arbitrage* rather than
# about forecasting, and it is fitted by driving pricing errors toward zero rather than by
# minimising a forecast error. The two networks it trains - one producing the discount-factor
# weights from the cross-section, one producing each fund's exposure to it - are trained against
# that pricing objective.
#
# A predicted return still comes out, which is why this notebook sits beside the others and is
# scored the same way. But the number being optimised is not the number being reported, and that
# gap is the thing to hold on to while reading the result.
#
# **It is also the only member that reads macro data.** The other four see the fund panel and the
# feature matrix. This one additionally takes eleven macro series as context for the
# discount factor, which raises a question none of the others have to answer: what was actually
# knowable on the decision date. Section 2 is where that contract is read back out of the resolved
# specification rather than asserted here.
#
# **The checkpoint schedule is phased, and all of it is published.** Training runs an
# unconditional stage and then a conditional one, and `case_studies/config/sdf/sdf.yaml` declares
# saves inside each. An earlier version of this notebook fixed the last of them in advance,
# showed that a different checkpoint scored better on validation, and reported the fixed one
# anyway. That is the right instinct about the trap - picking the checkpoint with the best
# validation IC is choosing after seeing the answer - but the wrong remedy, because it discards
# models the case study paid to train and leaves the choice to be made again, undocumented,
# downstream. Publishing the whole schedule lets [`14_backtest`](14_backtest.ipynb) select on
# validation backtest Sharpe with every candidate visible and the rule stated. **Selection happens
# there, not here.**
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a pricing objective asks for that a forecasting objective does not.
# - Read a phased training schedule and work out how many prediction sets a label will owe.
# - Read a point-in-time data contract out of a resolved specification and say what it rules out.
# - Say why the macro context belongs inside the model's identity rather than beside it.
#
# **Book reference**: Chapter 14, Section 14.7 (The stochastic discount factor and the supervised
# autoencoder models). Chapter 6, Section 6.7 (Search accounting and run logging) introduces the
# run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) for the feature matrix,
# [`05_evaluation`](05_evaluation.ipynb) for the walk-forward folds, and
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb), which is the two-stage model
# this one abandons.
#
# **What it writes**: one training run per label and one complete validation prediction set per
# label and checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a population named for this model. The family splits across
# five notebooks, so each publishes its own population rather than one shared one.

# %%
"""Fit the declared ETF stochastic-discount-factor population on the walk-forward folds."""

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

MODEL_NAME = "sdf"

# %%
study = open_study("etfs", execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and both do: `fwd_ret_21d`,
# the total return over the 21 trading days after the decision date, and `fwd_ret_5d`, the same
# thing over five. The pricing objective is written against the label's returns, so these are two
# separate fits rather than one fit scored twice. `LABELS` restricts the run to a subset when you
# want one.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/sdf/sdf.yaml` declares the estimator and its schedule, and the schedule has
# more structure here than in the two autoencoders. `n_epochs_unc` trains the unconditional stage,
# `n_epochs_cond` the conditional one that follows it, and `checkpoint_epochs` lists the saves
# inside a stage. The resolved plan in section 2 turns those into the actual list of checkpoints,
# which is what decides how many prediction sets each label owes - the count is not the length of
# the declared list, because a save inside the unconditional stage and the same offset inside the
# conditional one are two different states of the model.

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
# The device is checked in the same cell. Two networks trained on a GPU and the same two trained on
# a CPU accumulate their sums in different orders and reach different weights, so the device is
# part of what the fitted model is and is recorded inside the computation's identity rather than
# beside it. `PUBLISHED_DEVICE` is the device this population was fitted on; the runner refuses to
# substitute a CPU for a requested GPU rather than publishing a different model under the published
# name, so on a machine with no NVIDIA card this notebook stops at the next cell. Set
# `DEVICE="cpu"` and pass a `POPULATION_NAME` to fit the same schedule there.

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
# finds that: it reads the label, feature and macro files, computes the fold boundaries from the
# walk-forward parameters in `config/setup.yaml`, and works out the exact set of rows each fit is
# expected to predict. It fits nothing, so the plan can be read before any training starts.
#
# Four things to check in it:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across the rows.** They are
#   the width of the networks' input, the number of funds in the cross-section, and the fund-date
#   pairs to be predicted. A row that differs is a label measured on a different sample.
# - **`folds` is the same on both rows**, and equals the number of walk-forward splits
#   [`05_evaluation`](05_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   is scored once, at the end of the case study; any of it visible here would mean it had been
#   used to choose something.
# - **`checkpoints`** is the resolved schedule length, and it is the number to multiply by the row
#   count to get the candidate models this notebook is about to create.
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
        "the plan does not match the loaded SDF menu; "
        f"missing {sorted(requested_pairs - planned_pairs)}, "
        f"unexpected {sorted(planned_pairs - requested_pairs)}"
    )
print(f"{len(requested_pairs)} label-configuration pairs")

# %% [markdown]
# ### The macro contract, read back from what was resolved
#
# The discount factor takes macro context, and the point-in-time question that raises is not
# rhetorical: a macro series as it reads today is not what it read on a decision date years ago,
# because the agencies that publish these numbers revise them. Training on the revised series would
# let the model condition on information nobody had.
#
# The contract below is read out of the resolved specification rather than restated here, so it is
# what actually entered the identity:
#
# - **`source`** says the values are ALFRED *initial releases* - each series as it was first
#   published, not as it now reads.
# - **`availability_lag_days`** and **`alignment`** say a value dated *t* becomes usable on *t+1*
#   and is then carried backward, so a same-day close cannot reach a same-day decision.
# - **`coverage_start`** is where the panel begins, which is set by the last series to start rather
#   than the first.
# - **`resolved_fold_digest`** is the content of the macro values each fold actually saw. It is
#   inside the training identity, so a macro panel that changed underneath would register as a
#   different model rather than overwriting this one.
#
# **What this does not buy.** The financial feature matrix still contains yield-curve variables
# built from finalized FRED series, so the run as a whole is not a vintage-clean live estimate. The
# macro contract covers the macro context and nothing else, and the feature matrix's exact content
# is pinned by the modeling-input digest rather than by a release policy.

# %%
macro = resolved[0].spec["computation"]["macro_context"]
if macro is None:
    raise RuntimeError("the SDF request resolved without a macro context")

pl.DataFrame(
    {
        "field": ["source", "policy", "version", "availability_lag_days", "alignment"]
        + ["coverage_start", "series", "resolved_fold_digest"],
        "value": [
            str(macro["source"]),
            str(macro["policy"]),
            str(macro["version"]),
            str(macro["availability_lag_days"]),
            str(macro["alignment"]),
            str(macro["coverage_start"]),
            str(len(macro["series"])),
            str(macro["resolved_fold_digest"]),
        ],
    }
)

# %% [markdown]
# ## 3. Fitting the population
#
# `run_model_population` runs every resolved request. For one request it walks the folds, and on
# each one:
#
# 1. takes the fund-dates inside that fold's training window, together with the macro values that
#    were available on those dates,
# 2. trains the unconditional stage, then the conditional one that adds the macro context, by
#    gradient descent on the pricing errors, writing the weights to disk at each declared save,
# 3. applies each saved set of weights to the validation dates to get each fund's exposure to the
#    discount factor there, and turns that into a predicted return.
#
# Step 3 is what makes one training run produce several results. The fold predictions are
# concatenated into one series per checkpoint covering the whole validation period, and each
# becomes its own registered prediction set with its own identity.
#
# **What the call publishes is a population**: a named, immutable list of the prediction sets it
# will produce, computed from the resolved specifications and written down before the first fit.
# Afterwards every member must exist and be complete, which is what makes the downstream comparison
# well defined - [`14_backtest`](14_backtest.ipynb) backtests this population, not whatever
# predictions happen to be in the registry - and it is why a configuration that raises fails the
# whole call rather than publishing a population one member short.
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces, and is empty because this is
# the first generation published under this name. A population is the set of prediction identities,
# so anything that moves a training identity - a changed macro panel as much as a changed epoch
# schedule - produces a different population under the same name, and the registry refuses to write
# it without being told which snapshot it supersedes. A reduced-scale run passes it empty whatever
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
# its own registry and artifacts and reads the same labels, features and macro panel:
#
# ```python
# study = open_study("etfs", workspace="~/ml4t-experiments")
# configs = load_model_configs(study, "latent_factors", labels=["fwd_ret_5d"], config_names=["sdf"])
# requests = model_requests(study, configs, overrides={"device": "cuda"})
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-sdf-v1")
# ```
#
# To change the architecture or the schedule, edit `case_studies/config/sdf/sdf.yaml`. That changes
# this configuration's identity, so its result registers as a new row beside the old one rather
# than replacing it - and that includes the phase lengths and `checkpoint_epochs`, which decide how
# many members the population has. Give the run its own `population_name`: a name refers to one set
# of members permanently, and reusing it for a different set raises.
# [`RUN_LOG.md`](../RUN_LOG.md#running-your-own-configurations) covers the rest.

# %% [markdown]
# ## 4. What came out
#
# One row per label and checkpoint, read back from the registry. `ic_mean` is the **information
# coefficient**: on each validation date, rank the funds by the model's prediction, rank them by
# the return they went on to earn, correlate the two rankings, and average that daily correlation
# over the validation period.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it decides which
# rows are comparable with each other. A model that has settled into assigning nearly the same
# exposure to every fund on a date gives that date no spread to rank, and the date drops out of the
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
    raise RuntimeError("a partial SDF prediction set cannot pass to backtesting")

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
catalog.select("label", "checkpoint_value", "ic_mean", "ic_std", "ic_n_days", "full_coverage")

# %% [markdown]
# ### What more training does
#
# Each line traces one label's out-of-sample IC across the phased schedule. The horizontal axis is
# cumulative epochs, so the first point is the end of the unconditional stage and the rest are
# saves inside the conditional one - the step from the first to the second is where the macro
# context enters, and it is the only step on the axis that changes what the model is allowed to
# see rather than only how long it has trained.
#
# A line that rises and then falls has an interior optimum: the model was still learning, then
# began fitting the training window at the expense of the validation folds. A line that wanders
# around zero without trend never had anything to learn, and its highest point is wherever the
# noise happened to peak. Both produce a respectable-looking maximum, which is why the maximum is
# not what a checkpoint is judged on and why the whole schedule is published rather than the best
# of it.
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
fig.update_xaxes(title_text="Cumulative training epoch", row=len(panel_labels), col=1)
fig.update_layout(
    title="Out-of-sample ranking accuracy across the phased schedule",
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
    "Line chart of mean validation information coefficient against cumulative training epoch, one "
    "panel per label, each with its own vertical scale and a dashed zero line. Counted from the "
    f"frame: {_spans}.",
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
# selection: reading these eight numbers at the checkpoint with the highest `ic_mean` and reporting
# them as the model's would be the choice this notebook publishes the whole schedule to avoid.

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
        f"{label} at cumulative epoch {last_checkpoint}: mean {ic.mean():+.4f}, "
        f"standard deviation across folds {ic.std():.4f}, "
        f"{(ic < 0).sum()} of {ic.len()} folds negative"
    )

by_fold[panel_labels[0]]

# %% [markdown]
# ## 5. What to notice
#
# **The macro step is the whole story on the primary label, and it runs the other way on the
# five-day one.** Checkpoint 256 is where the unconditional stage ends, and on `fwd_ret_21d` it
# scores **-0.0070** - the model orders the cross-section slightly worse than a coin. The four
# conditional saves that follow run **+0.0132, +0.0223, +0.0379, +0.0370**. Every bit of ranking
# accuracy this notebook publishes on its primary label appears after the macro context enters, at
# the one step on the axis that changes what the model is allowed to see rather than only how long
# it has trained. On `fwd_ret_5d` the same step is a loss: 256 scores **+0.0394**, the highest of
# its five, and the four conditional checkpoints sit **below** it at +0.0091, +0.0152, +0.0233 and
# +0.0238. Conditioning on eleven macro series helps the 21-day horizon and hurts the 5-day one,
# which is not a result that survives being averaged into one number for the model.
#
# **Pricing the cross-section did not beat splitting the problem in two.** Against the members
# fitted on the same panel over the same folds, this one's best primary-label checkpoint
# (**+0.0379**) sits below [`11b_ipca`](11b_ipca.ipynb)'s single **+0.0413** and well below
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb)'s **+0.0722** at the end of its
# schedule. On the five-day label its best (+0.0394, from the unconditional stage) is near IPCA's
# +0.0349 and under the autoencoder's +0.0457. It clears [`11a_pca`](11a_pca.ipynb)'s -0.0326 on
# the primary label, so conditioning still bought something over conditioning on nothing - just
# less than either two-stage member bought, at four times the training. That is the trade this
# member exists to price, and here the answer is negative.
#
# **The checkpoint curve is not evidence for an epoch, and the arithmetic says so plainly.** The
# five primary-label checkpoints span 0.0449 of IC end to end. The eight fold ICs behind the last of
# them - +0.005, +0.027, +0.165, +0.159, +0.077, +0.021, -0.140, -0.017 - have a standard deviation
# across folds of **0.0989**, more than twice that span. The apparent best checkpoint, 1024, beats
# 1280 by 0.0009, a fiftieth of the disagreement between folds measuring the same model. Nothing
# here distinguishes an epoch worth stopping at from an epoch that happened to score well on eight
# folds, which is why all five are published and why nothing downstream may select among them on
# validation IC. The dispersion is also no better than the neighbours': 0.0989 here against 0.092
# for the conditional autoencoder and 0.099 for IPCA.
#
# **No checkpoint collapsed, and that had to be checked rather than assumed.** `ic_n_days` is 1,995
# on every primary-label row and 2,011 on every five-day row - the full count each label offers - so
# `full_coverage` holds for all ten published models. A model that had settled into pricing every
# fund alike on a date would have dropped that date out of its average and left one row measured on
# a different sample from its neighbours'.
#
# **Known limitations.** The architecture, the factor count and the two-phase schedule are declared
# in `case_studies/config/sdf/sdf.yaml` rather than searched, so nothing here says this is the right
# SDF - only what this one did. The macro contract is point-in-time by construction, and its
# coverage starts **2010-11-23** while the panel starts in 2006, so the earliest training windows
# carry less macro history than the later ones and the conditional stage is not learning from the
# same amount of context in every fold. Two of eight folds are negative on the primary label and
# three on the five-day. The panel narrows from 96 funds in the first fold to 90 in the last. The
# published result is a CUDA result, and a reader who refits on CPU gets a different identity
# registered beside it rather than these numbers. And every number here is measured on validation
# folds that have been read many times over by the time a case study reaches this notebook.

# %% [markdown]
# **Next**: [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) is the family's own
# control - it keeps [`11c`](11c_conditional_autoencoder.ipynb)'s network and bottleneck while
# dropping the factor interpretation, so what it does not achieve is what the structure was worth.
# [`13_model_analysis`](13_model_analysis.ipynb) is where all five are read against the other
# families.
