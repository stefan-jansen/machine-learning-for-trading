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
# # ETFs: a conditioned network with no factor structure
#
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb) maps a fund's feature row to a
# small set of exposures, estimates a factor return for each date from the cross-section, and
# multiplies the two to get a forecast. The supervised autoencoder keeps the first half of that and
# drops the second: a network compresses the feature row to a low-dimensional representation, and a
# head predicts the forward return from that representation directly rather than through a factor
# return.
#
# **That makes this the one member of the family that asserts no factor structure.** Every other
# member says the cross-section is driven by a few common movements and that a fund's return is its
# exposure to them. This one keeps a low-dimensional representation and says nothing about what it
# represents.
#
# **It is not a controlled comparison with `11c`, and the notebook does not read it as one.** The
# two differ in the factor structure and in the network as well.
# `case_studies/utils/latent_factors/cae.py` maps the feature row through `hidden_units=(32,)` to
# five factors at a learning rate of 1e-3;
# [`sae.py`](../utils/latent_factors/sae.py) maps it through `(896, 448, 448, 256)` to a 96-unit
# bottleneck at 1e-4, with Gaussian input noise of 0.035 and a second loss term. Capacity, depth,
# width, the learning rate and the objective all move together, so a difference in their results is
# not attributable to any one of them. What the pair gives is a member of the family fitted without
# the factor assumption, which is worth having and is not the same thing as a measurement of what
# the assumption is worth.
#
# **The label enters the fit here, which it does not everywhere in the family.** The reconstruction
# term trains the encoder to compress the feature row; the supervised term trains the head against
# the fold's forward returns. Both are backpropagated together. So the two rows this notebook fits
# are two different models rather than one model scored against two returns, and that is true of
# `11c` as well but for a different reason: there the label enters through the returns matrix the
# factor stage reproduces, here it enters through the objective directly.
#
# **The network is trained rather than solved, which decides what this notebook publishes.**
# `case_studies/config/sae/sae.yaml` declares 50 epochs saved every 5, the same schedule
# `cae.yaml` declares, so each label produces ten scoreable models rather than one and each is
# registered separately. All ten are published. An earlier version of this notebook fixed a
# reporting checkpoint in advance and reported that one, which is the right instinct about the trap
# - reading off the epoch with the best validation IC is choosing after seeing the answer - and the
# wrong remedy, because it discards nine models the case study paid to train and leaves the choice
# to be made again, undocumented, downstream. Publishing the whole schedule lets
# [`14_backtest`](14_backtest.ipynb) select on validation backtest Sharpe with every candidate
# visible and the rule stated. **Selection happens there, not here.**
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Say what a supervised autoencoder keeps from a conditional autoencoder and what it drops.
# - Say what a member that imposes no factor structure still shares with the rest of the family.
# - Read a curve of out-of-sample ranking accuracy against training epoch, and tell a model still
#   learning from one that has begun fitting its training window.
# - Say why the training device is part of a model's identity rather than a note about how it ran.
#
# **Book reference**: Chapter 14, Section 14.7 (The stochastic discount factor and the supervised
# autoencoder models). Chapter 6, Section 6.7 (Search accounting and run logging) introduces the
# run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) for the feature matrix,
# [`05_evaluation`](05_evaluation.ipynb) for the walk-forward folds,
# [`11_latent_factors`](11_latent_factors.ipynb), which introduces the family, and
# [`11c_conditional_autoencoder`](11c_conditional_autoencoder.ipynb), the family's other neural
# member, which imposes the factor structure this one drops.
#
# **What it writes**: one training run per label and one complete validation prediction set per
# label and epoch checkpoint, in `run_log/registry.db` and under `run_log/training/` and
# `run_log/predictions/`, grouped under a population named for this model. The family splits across
# five notebooks, so each publishes its own population rather than one shared one.
# [`13_model_analysis`](13_model_analysis.ipynb) compares them against the other families and
# [`14_backtest`](14_backtest.ipynb) backtests every member and selects on validation backtest
# Sharpe. **Selection happens there, not here.**

# %%
"""Fit the declared ETF supervised-autoencoder population on the walk-forward folds."""

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

MODEL_NAME = "sae"

# %%
study = open_study(
    "etfs",
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="11e_supervised_autoencoder",
)

# %% [markdown]
# ## 1. Which labels, and what the configuration says
#
# Every label whose training menu declares `latent_factors:` is fitted, and both do: `fwd_ret_21d`,
# the total return over the 21 trading days after the decision date, and `fwd_ret_5d`, the same
# thing over five. The label is inside the objective here rather than only in the scoring, so the
# two rows below are two different fits with two different sets of weights. `LABELS` restricts the
# run to a subset when you want one.

# %%
declared_labels(study, "latent_factors")

# %% [markdown]
# `case_studies/config/sae/sae.yaml` declares `n_epochs: 50`, `checkpoint_interval: 5` and
# `n_factors: 5`, which is the same schedule `cae.yaml` declares and is why the two are read at the
# same points. `n_factors` is the one field that does not carry over: `run_sae_fold` deletes the
# argument and takes the representation's width from `bottleneck_dim`, which is 96. The training
# log's `sae (K=5)` should not be read as a factor count, and the parity with `cae.yaml` on that
# field is an artifact of a shared runner signature rather than a shared setting.
#
# One value the configuration does not name reaches the fit anyway. `run_sae_fold` passes
# `batch_size=10_000`, the same default its sibling `run_cae_fold` carries, because
# `SAEConfig.batch_size` is `None` when nothing passes one and the library reads that as a single
# batch over the whole training window. That is what the runner's declared version 2 in
# `case_studies/utils/latent_factors/versions.py` records: results fitted before it are a different
# computation and must not be reused. A case study that wants a different value declares it under
# `modeling.latent_factors.model_kwargs.sae` in `config/setup.yaml`, where it is hashed into the
# training identity with every other model argument. This one declares none and takes the default.

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
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree with `11c`'s plan.** They are
#   the width of the network's input, the number of funds in the cross-section, and the fund-date
#   pairs to be predicted. The two notebooks read the same panel, and that much really is held
#   level even though the architectures are not.
# - **`folds` is the same on both rows**, and equals the number of walk-forward splits
#   [`05_evaluation`](05_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   is scored once, at the end of the case study; any of it visible here would mean it had been
#   used to choose something.
# - **`checkpoints` is 10**, the epoch schedule declared above, and the same number `11c` publishes.
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
        "the plan does not match the loaded SAE menu; "
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
# 2. trains one network on two objectives at once - reconstructing the feature row from a
#    96-unit bottleneck, and predicting that row's forward return from the same bottleneck - by
#    gradient descent on the sum of the two errors, writing the weights to disk every five epochs,
# 3. applies each saved set of weights to the validation dates' feature rows and reads the
#    prediction head's output as the forecast.
#
# Step 3 is where the structural difference from [`11c`](11c_conditional_autoencoder.ipynb) sits:
# there the saved weights produce exposures, which are multiplied by training-window factor returns
# to get a forecast. Here the forecast comes off the head, and nothing between the bottleneck and
# the number is interpreted as a factor. It is not the only difference between the two, as the
# introduction sets out.
#
# Step 3 is also what makes one training run produce ten results. The fold predictions are
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
# `SUPERSEDES_POPULATION` names the population hash this run replaces. A population is the set of
# prediction identities, so anything that moves a training identity - a changed epoch schedule as
# much as a changed label menu - produces a different set under the same name, and the registry
# refuses to write it without being told which snapshot it supersedes. It is empty here because
# this notebook publishes `etfs-sae-validation-v1` for the first time: the runs this member
# registered before it moved onto the research boundary carried no computation identity at all and
# were listed by no population, so there is no snapshot to replace. Fill it in when you have
# changed something that moves an identity and want the new set to take the name; the error raised
# on the attempt tells you which hash to name. A reduced-scale run passes it empty whatever the
# default is: a population produced under a reduction is thrown away with the workspace it was
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
# configs = load_model_configs(study, "latent_factors", labels=["fwd_ret_5d"], config_names=["sae"])
# requests = model_requests(study, configs, overrides={"device": "cuda"})
# resolved = tuple(request.resolve() for request in requests)
# execution, population = run_model_population(study, resolved, population_name="my-sae-v1")
# ```
#
# To change the architecture or the schedule, edit `case_studies/config/sae/sae.yaml`. Changing
# `cae.yaml` to match keeps the two read at the same points on their epoch schedules, which is the
# only axis the pair holds level; it does not make them a controlled comparison, for the reasons
# the introduction gives. Note which change you want: the preset is shared by every case study that
# declares `sae`, so editing it moves the training identity of all of them, while a value under
# `modeling.latent_factors.model_kwargs.sae` in `config/setup.yaml` moves this case study alone.
# Either way the result registers as a new row beside the old one rather than replacing it. Give
# the run its own `population_name`: a name refers to one set of members permanently, and reusing
# it for a different set raises.
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
    raise RuntimeError("a partial SAE prediction set cannot pass to backtesting")

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
# Each line traces one label's out-of-sample IC as epochs are added, on the same axes
# [`11c`](11c_conditional_autoencoder.ipynb) draws for the same schedule.
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
    # See 11b: polars returns None for the mean of an empty column and for the standard
    # deviation of a single row, and a narrowed population produces both.
    if ic.len() == 0:
        print(f"{label} at epoch {last_checkpoint}: no fold scored, so there is no IC")
        continue
    spread = (
        f"standard deviation across folds {ic.std():.4f}"
        if ic.len() > 1
        else "one fold only, so no spread across folds"
    )
    print(
        f"{label} at epoch {last_checkpoint}: mean {ic.mean():+.4f}, "
        f"{spread}, "
        f"{(ic < 0).sum()} of {ic.len()} folds negative"
    )

by_fold[panel_labels[0]]

# %% [markdown]
# ## 5. What to notice
#
# **Where this member lands.** At the end of the declared schedule, the family's mean validation
# IC reads:
#
# | | `fwd_ret_21d` | `fwd_ret_5d` |
# |---|---|---|
# | [`11c`](11c_conditional_autoencoder.ipynb) cae | +0.072 | +0.046 |
# | [`11b_ipca`](11b_ipca.ipynb) ipca | +0.041 | +0.035 |
# | [`11d`](11d_stochastic_discount_factor.ipynb) sdf | +0.037 | +0.024 |
# | [`11a_pca`](11a_pca.ipynb) pca | -0.033 | +0.010 |
# | this notebook, sae | +0.029 | -0.002 |
#
# So this member is fourth of five on the primary label and fifth on the five-day one. The three
# **conditioned** factor models are above it on both. The one member below it on the primary label,
# `11a_pca`, is the unconditional baseline that reads none of the 71 feature columns - and on the
# five-day label that baseline is above it. A network with 96 units of representation and four
# hidden layers does not beat a singular value decomposition of the return panel on both horizons,
# which is the sort of result the baseline is in the family to produce.
#
# **What that does and does not license.** The three members above it on both labels assert a
# factor structure and this one does not, which is the difference worth naming. It is not the only
# difference: as the introduction sets out, this network is deeper, wider, differently regularised
# and trained at a tenth of `11c`'s learning rate. So the ordering is a fact about these five
# fitted models on this panel, and it is consistent with the factor structure earning its place. It
# is not a measurement of how much the structure is worth, and no arithmetic on the gaps recovers
# one. Isolating that would need a run holding the architecture fixed and switching only the
# objective, which no configuration in `case_studies/config/` currently declares.
#
# **The two labels behave differently, and the curve is where that shows.** On the primary label
# the ten checkpoints run from -0.000 at five epochs to +0.029 at fifty, and the highest of them is
# the last. That is the observed maximum and not evidence of continued learning: the curve falls
# from ten epochs to fifteen and again from forty to forty-five, and epoch 50 sits 0.0003 above
# epoch 40 against a fold-to-fold standard deviation of 0.089. Nothing here establishes an optimal
# stopping point or that another fifty epochs would add anything. On the five-day label the highest
# checkpoint is +0.007 at ten epochs and every checkpoint from fifteen on is negative, ending at
# -0.002. One label produced a positive average and one did not, under one declared schedule, which
# is why the schedule is declared per estimator rather than per label and why both are published
# rather than summarised.
#
# **The checkpoint curve is not evidence for an epoch.** On the primary label the ten checkpoints
# span 0.029 of IC end to end. The eight fold ICs behind the last of them - -0.045, -0.060, +0.128,
# +0.053, +0.101, -0.037, +0.149, -0.059 - have a standard deviation across folds of 0.089, three
# times that span, and four of the eight are negative. Nothing in this table distinguishes an epoch
# worth stopping at from an epoch that happened to score well on eight folds, which is why all ten
# checkpoints are published rather than one and why nothing downstream may select among them on
# validation IC.
#
# **No checkpoint's coverage fell behind the others.** `ic_n_days` is 1,995 on every primary-label
# row and 2,011 on every five-day row, so `full_coverage` is true for all twenty published models.
# What that establishes is that no checkpoint dropped dates its siblings kept, which is the failure
# it is there to catch: a network settling into predicting nearly the same value for every fund on
# a date gives that date no spread to rank, and the date leaves that row's average alone. It does
# not establish that the count equals every date the label offers, because the comparison is
# within this run rather than against the resolved validation sample.
#
# **Known limitations.** The architecture, the bottleneck width and the schedule are declared in
# `case_studies/config/sae/sae.yaml` and the runner's own defaults rather than searched, so nothing
# here says this network is the right one - only what this one did. Fifty epochs is where training
# stops, not where anything established it should stop, and the primary label's checkpoint curve
# is non-monotonic rather than a schedule cut short. The fold-to-fold dispersion is 0.089 against
# a mean of 0.029, so the
# primary-label result is in the average rather than in how much any single fold can be relied on.
# The panel narrows from 96 funds in the first fold to 90 in the last. The published result is a
# CUDA result, and a reader who refits on CPU gets a different identity registered beside it rather
# than the same numbers. And every number here is measured on validation folds that have been read
# many times over by the time a case study reaches this notebook.

# %% [markdown]
# **Next**: [`11_latent_factors`](11_latent_factors.ipynb) is where the five members' coverage of
# the declared menu is checked, [`12_causal_dml`](12_causal_dml.ipynb) asks whether the momentum
# signal has a stable causal interpretation, and
# [`13_model_analysis`](13_model_analysis.ipynb) compares this family against every other one with
# the whole population in front of it.
