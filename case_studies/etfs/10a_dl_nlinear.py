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
# # NLinear as a Global ETF Sequence Model
#
# NLinear is the smallest thing that still counts as a sequence model: subtract the last observed
# level from the lookback window and fit one linear map from what remains to the forecast. It has
# no gate, no recurrence and no mixing layer, so it cannot represent an interaction that
# [`09_dl_lstm`](09_dl_lstm.ipynb) or [`10_dl_tsmixer`](10_dl_tsmixer.ipynb) can. That is what
# makes it worth fitting: the two larger architectures are only worth their cost if they beat it,
# and until it is on the same folds and the same eligible rows there is nothing saying they do.
#
# **Learning objectives**
# - Declare a sequence population and resolve it against the data before any fitting happens
# - Tune the epoch checkpoint on the validation folds without reading the holdout, under a
#   full-coverage guard that excludes checkpoints whose predictions collapsed on some folds
# - Place the simplest sequence model against the linear and gradient-boosting leaders on the
#   same cross-section
#
# **Book reference**: Chapter 13, Section 13.8 (Case Study Results)
#
# **Prerequisites**: [`06_linear`](06_linear.ipynb) and [`07_gbm`](07_gbm.ipynb), which fit the
# populations this one is compared against.

# %%
"""Fit the declared ETF NLinear population on the walk-forward folds."""

import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display

from case_studies.research import (
    OfficialPopulation,
    Result,
    load_model_configs,
    model_requests,
    open_study,
    primary_label,
    resolved_model_plan,
    run_model_population,
    split_unpublished_members,
)
from case_studies.utils.analytics import load_best_ic_per_family, load_model_ic
from case_studies.utils.model_analysis import common_sample_daily_ic, load_predictions
from case_studies.utils.notebook_contracts import prediction_members_in_force
from case_studies.utils.registry import load_prediction_metrics
from utils.style import COLORS

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
LABELS: list[str] = []
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
CONFIG_NAMES: list[str] = []
POPULATION_NAME = ""
SUPERSEDES_POPULATION: str = ""
DEVICE: str = ""

# %%
study = open_study(
    CASE_STUDY_ID,
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="10a_dl_nlinear",
)
PRIMARY_LABEL = primary_label(study)

# %% [markdown]
# ## 1. Which models, and on what
#
# The training menu declares three architectures for this case study, and this notebook fits the
# third: [`09_dl_lstm`](09_dl_lstm.ipynb) fits the recurrent member,
# [`10_dl_tsmixer`](10_dl_tsmixer.ipynb) the mixer, and `nlinear` went declared and unfitted until
# this notebook existed. Each names its own population rather than the family's, because each
# publishes a different slice of it.
#
# The reconciliation in section 5 is what keeps that a fact rather than a comment. This is the
# last sequence notebook to run, so once it has fitted, the registry holds whatever the family
# fitted and the menu can be checked against it. Nothing else reconciles a declaration against
# what was executed - which is how a declared-and-never-fitted configuration stayed invisible for
# as long as it did - so the gap is printed every run rather than asserted once here.

# %%
SEQUENCE_CONFIGS = ("nlinear",)
declared = load_model_configs(study, "deep_learning", config_names=list(SEQUENCE_CONFIGS))
configs = load_model_configs(
    study,
    "deep_learning",
    labels=LABELS or None,
    config_names=CONFIG_NAMES or list(SEQUENCE_CONFIGS),
)
configs

# %% [markdown]
# `LABELS` and `CONFIG_NAMES` narrow the run below this notebook's own slice, and a narrowed run
# declares a different set of members than the published population does. A population is immutable
# once written, so such a run has to publish under its own name: on a fresh workspace it would
# otherwise freeze an incomplete snapshot under the published one, and where that population
# already exists the registry refuses the write outright. The comparison is over
# label-configuration pairs rather than row counts, because two different subsets can have the
# same height.
#
# The device is checked in the same place and for the same reason. A network trained on a GPU and
# the same network trained on a CPU accumulate their sums in a different order and reach different
# weights, so the device is part of what the fitted model *is*. It sits inside the hashed identity
# rather than beside it, and the runner refuses to substitute a CPU for a requested GPU rather than
# publish a different model under the published name. On a machine with no NVIDIA card this
# notebook therefore stops at the next cell; set `DEVICE="cpu"` and pass a `POPULATION_NAME` to fit
# the same grid there.

# %%
PUBLISHED_DEVICE = "cuda"
device = DEVICE or PUBLISHED_DEVICE
print(f"training device: {device}")

narrows = set(zip(configs["label"], configs["config_name"], strict=True)) != set(
    zip(declared["label"], declared["config_name"], strict=True)
)
if (narrows or device != PUBLISHED_DEVICE) and not POPULATION_NAME:
    raise ValueError(
        f"this run declares {configs.height} label-configuration pairs on device {device!r}, "
        f"which is not this notebook's declared slice on {PUBLISHED_DEVICE!r}, so it cannot "
        "publish the sequence population; pass POPULATION_NAME to give it its own"
    )

# %% [markdown]
# ## 2. The baselines this is measured against
#
# Read back from the registry rather than re-fitted here: each is its family's full-coverage
# validation leader, reported as the daily-pooled IC that `load_model_ic` returns for every
# family alike.
#
# Full coverage is decided within a family, not across families, so two families can each be
# complete by their own measure and be scored over different windows with nothing saying so.
# Equal date *counts* would not settle it either: a sequence model drops the warm-up rows a flat
# model keeps, so the same date can carry a different cross-section in each family. Section 7
# therefore intersects the three prediction sets on their exact `(symbol, timestamp)` keys and
# recomputes the IC on what they share, rather than taking the comparability on trust.

# %%
prior_baselines = {}
baseline_hashes = {}
# `load_best_ic_per_family` reads the metrics catalog, which carries no lineage: when
# `06_linear` or `07_gbm` refits, the generation it replaced stays behind scored and complete,
# and the family leader read back here can be the retired one. The comparison in section 7 would
# then measure this network against a baseline its own publisher no longer stands behind.
#
# The question is membership, not retirement. `split_retired_members` asks whether a publisher
# moved past an identity, which admits every row no population ever listed - an experimental fit,
# a one-off, a row written before its notebook declared a population. Nobody retired those, so
# they pass an exclusion test and can outrank the published leader.
# `split_unpublished_members` asks the population lineage what is listed at all.
#
# The inventory it is asked about is the complete one. `load_model_ic` filters to the maximum
# coverage within each family and label by default, and a retired generation scored over a
# shorter window is dropped by that filter before the split ever sees it - so its hash never
# reaches `exclude_prediction_hashes`, and `load_best_ic_per_family`, applying its own coverage
# bar, can hand that row back as the leader. Loading without the filter is what makes the
# exclusion set complete.
#
# The exclusion goes in before the per-family maximum, not after it. Filtering the returned frame
# would drop a family outright whenever its highest-IC row happened to be excluded - the live
# runner-up is already gone by then - and section 7 looks the baselines up by name, so the family
# would not fall back, it would raise. Excluded rows are named rather than counted, so a baseline
# that moves is visible as a refit upstream.
_candidates = load_model_ic(
    ["linear", "gbm"], case_studies=[CASE_STUDY_ID], require_full_coverage=False
)
_retired = split_unpublished_members(study, _candidates).retired
if not _retired.is_empty():
    print("Not listed by any current population, excluded before the family leaders are taken:")
    for _row in _retired.iter_rows(named=True):
        print(f"  {_row['family']}/{_row['config_name']}: {_row['prediction_hash']}")
_baselines = load_best_ic_per_family(
    ["linear", "gbm"],
    case_studies=[CASE_STUDY_ID],
    exclude_prediction_hashes=_retired["prediction_hash"].to_list(),
)
for row in _baselines.iter_rows(named=True):
    # The full-coverage linear leader is a Ridge configuration; name it plainly so the comparison
    # below resolves it.
    name = {"linear": "Ridge (Ch11)", "gbm": "GBM (Ch12)"}.get(row["family"])
    if name:
        prior_baselines[name] = row["ic_mean"]
        baseline_hashes[name] = row.get("prediction_hash")

if prior_baselines:
    for name, ic in prior_baselines.items():
        print(f"  {name}: IC={ic:+.4f}" if ic is not None else f"  {name}: IC=n/a")
else:
    print("  No prior results registered - run 06_linear and 07_gbm first")

# %% [markdown]
# ## 3. Binding the declaration to the data
#
# A menu entry says which network to fit. It does not say which feature columns exist today, where
# the walk-forward folds fall, which fund-date pairs have both a feature row and a label, or - for
# a sequence model - which of those pairs have a full unbroken window of prior observations behind
# them. **Resolving** the request goes and finds all of it.
#
# Resolving reads the inputs and fits nothing, so the plan can be read before any GPU time is
# spent. Four things to check in it:
#
# - **`feature_count`, `eligible_entities` and `eligible_rows` agree across every row of a label.**
#   They are the width of the design matrix, the number of funds, and the number of fund-date pairs
#   to be predicted. A row that differs is a configuration measured on a different sample from its
#   neighbours, and its result is not comparable with theirs.
# - **`eligible_rows` is below what a flat-feature family reports on the same label.** A sequence
#   prediction needs a gap-free window behind it, so what drops out is a fund too new to have
#   accumulated one, or a stretch where the calendar has a hole inside the window. That is why a
#   sequence result and a tabular result on the same label are measurements on different samples,
#   which `full_coverage` marks within this family and
#   [`13_model_analysis`](13_model_analysis.ipynb) has to account for across families.
# - **`folds` is the same everywhere** and equals the number of walk-forward splits
#   [`05_evaluation`](05_evaluation.ipynb) established.
# - **`validation_start` and `validation_end` bracket the development sample.** The held-out tail
#   must not appear here: any of it visible in this window would mean it had been used to choose
#   something.
#
# Each row also carries a `training_hash`, the identity of that computation, derived from
# everything that can change its result - the device and the lookback included.

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
# ## 4. Fitting the population
#
# `run_model_population` fits every resolved request: for one request it walks the folds, saves a
# checkpoint at each declared epoch interval, scores each checkpoint on that fold's validation
# dates, and registers every checkpoint as its own prediction set. What the call publishes is a
# **population** - a named, immutable list of the prediction sets it produced - which is what
# [`14_backtest`](14_backtest.ipynb) resolves rather than a query it composes itself.
#
# **There is one identity builder, and the runner owns it.** The previous version of this notebook
# built its own lookup specification to decide whether a configuration was already fitted, and that
# specification had to agree field for field with the one the runner registered under. It stopped
# agreeing when the device became identity-bearing, and the failure was not a wasted cache lookup:
# the model trained, registered under the fuller identity, and the notebook then reported its own
# checkpoints incomplete. Nothing here derives an identity any more, so nothing here can disagree.
#
# **A second run fits nothing.** Every identity is re-derived from the inputs, the registry already
# holds the matching rows and the saved weights, and `reused` in the line below counts what came
# back from store rather than from a GPU.
#
# `SUPERSEDES_POPULATION` names the population hash this run replaces. It is empty here because
# this run publishes a name nothing has published before; a later refit under changed code produces
# a different member set, and the registry refuses the write until it is told which snapshot it
# retires.

# %%
population_name = POPULATION_NAME or "etfs-nlinear-validation-v1"
execution, population = run_model_population(
    study,
    resolved,
    population_name=population_name,
    supersedes=SUPERSEDES_POPULATION or None,
)

reused = sum(1 for item in execution.diagnostics if item.get("reused"))
print(
    f"{len(execution.runs)} configurations: {len(execution.runs) - reused} trained, {reused} read"
)
print(f"population {population.name}: {len(population.members)} prediction sets")

# %% [markdown]
# ## 5. What came out, and which checkpoints are comparable
#
# One row per epoch checkpoint. The **information coefficient** is the daily one: on each
# validation date, rank the funds by the model's prediction, rank them by the return they went on
# to earn, correlate the two rankings, and average that daily correlation over the validation
# period. Zero is no relationship.
#
# **The daily IC is read from `prediction_metrics` rather than from the catalog, and that is not a
# stylistic choice.** The research catalog exposes `ic_mean` and `ic_n_days` side by side, but they
# do not describe the same quantity: `ic_mean` is the mean over folds, while `ic_n_days` counts the
# dates behind `ic_mean_daily`, which the catalog does not carry. Pairing the two that the catalog
# offers reports a coverage count for a statistic it was not computed from. Until the catalog
# carries the daily IC, this cell reads it where it lives.
#
# `ic_n_days` is how many validation dates produced a defined correlation, and it decides which
# rows are comparable with each other. A network that has settled into predicting nearly the same
# value for every fund on a date gives that date no spread to rank, and the date drops out of the
# average - which makes a collapsed checkpoint look better by being scored on fewer, easier days.
# `full_coverage` marks the checkpoints measured on every date the label offers, and selection
# happens only among those.

# %%
_daily = pl.concat(
    [
        load_prediction_metrics(CASE_STUDY_ID, prediction_hash=prediction_hash).select(
            "prediction_hash", "ic_mean_daily", "ic_n_days"
        )
        for prediction_hash in execution.catalog_rows["prediction_hash"].to_list()
    ]
)
coverage = (
    execution.catalog_rows.select(
        "config_name", "label", "complete", "checkpoint_value", "n_folds", "prediction_hash"
    )
    .join(_daily, on="prediction_hash", how="inner")
    .join(
        plan.select("config_name", "label", pl.col("eligible_dates").alias("expected_days")),
        on=["config_name", "label"],
        how="left",
    )
    # Against the dates the resolved eligibility says this configuration should have scored, not
    # against the best its own grid managed. A grid where every checkpoint loses the same dates
    # has a maximum that is itself short, and calling that full coverage would publish truncated
    # coverage as complete.
    .with_columns((pl.col("ic_n_days") == pl.col("expected_days")).alias("full_coverage"))
    .sort("label", "checkpoint_value")
)
if coverage.is_empty():
    raise RuntimeError(f"no registered checkpoints for {CASE_STUDY_ID}")
if not coverage["complete"].all():
    raise RuntimeError("an incomplete checkpoint is registered; the population cannot be read")

# Coverage is measured over every label the run fitted, because the population spans them all and
# the cell below republishes it. The table this section shows is the primary label's slice of it.
checkpoints = coverage.filter(pl.col("label") == PRIMARY_LABEL)
if checkpoints.is_empty():
    raise RuntimeError(f"no registered checkpoints for {CASE_STUDY_ID}/{PRIMARY_LABEL}")

# %% [markdown]
# Every full-coverage checkpoint is published as a candidate; this notebook chooses none of them.
# Scanning the grid for the highest validation IC is the sequence-model analogue of early
# stopping, and it is read here as a diagnostic - it says where in training the signal peaked, and
# it anchors the learning curve and the per-fold table below. It is not the pipeline's selection:
# IC ranks nothing in this pipeline, and a configuration is chosen on validation backtest Sharpe
# in the evaluation stage, from the whole grid published here. The holdout is not touched.

# %% tags=["results"]
eligible = checkpoints.filter("full_coverage")
if eligible.is_empty():
    _short = checkpoints.select("checkpoint_value", "ic_n_days", "expected_days")
    raise RuntimeError(
        "no checkpoint scored on every date its resolved eligibility declares, so there is "
        f"nothing to publish as a full-coverage candidate:\n{_short}"
    )
peak_row = eligible.sort("ic_mean_daily", descending=True).row(0, named=True)
PEAK_EPOCH = int(peak_row["checkpoint_value"])
PEAK_IC = float(peak_row["ic_mean_daily"])
CONFIG_NAME = peak_row["config_name"]
PEAK_PHASH = peak_row["prediction_hash"]
FULL_DAYS = float(peak_row["ic_n_days"])
PUBLISHED_CHECKPOINTS = int(eligible.height)
partial_epochs = (
    checkpoints.filter(~pl.col("full_coverage"))["checkpoint_value"].cast(pl.Int64).to_list()
)

print(f"Config: {CONFIG_NAME}   full-coverage validation days = {int(FULL_DAYS)}")
print(checkpoints.select("checkpoint_value", "ic_mean_daily", "ic_n_days", "full_coverage"))
print(f"\nPublished as candidates: {PUBLISHED_CHECKPOINTS} full-coverage checkpoints")
print(f"Highest validation IC: epoch {PEAK_EPOCH} (IC={PEAK_IC:+.4f})")
if partial_epochs:
    print(f"Excluded before selection, partial coverage: epochs {partial_epochs}")

# %% [markdown]
# **The published population is narrowed to what covered its dates.** A population is declared
# before the first fit, so it necessarily lists every checkpoint the run intended, including one
# that ends up scored on fewer dates than its own resolved eligibility declares. Coverage is only
# knowable afterwards, and [`14_backtest`](14_backtest.ipynb) sweeps whatever the current
# population lists - so leaving a partial checkpoint in it would carry a collapsed epoch,
# flattered by the days it could not score, into validation-Sharpe selection. Republishing the
# full-coverage subset under the same name supersedes the declared generation, which is the
# lineage record of what left the candidate set and why.
#
# When every checkpoint covered its dates the member list is unchanged, and the registry returns
# the published snapshot rather than writing a second generation. That is the case here, and it is
# what makes a re-run of this notebook a no-op rather than a new lineage entry.

# %%
_full_coverage = set(coverage.filter("full_coverage")["prediction_hash"].to_list())
_declared = tuple(population.members)
_selectable = [member for member in _declared if member in _full_coverage]
if not _selectable:
    raise RuntimeError("no checkpoint covered its dates; there is nothing to publish")
if len(_selectable) != len(_declared):
    population = OfficialPopulation.create(
        study,
        name=population.name,
        member_kind="prediction",
        members=_selectable,
        supersedes=population.hash,
    )
print(
    f"population {population.name}: {len(_selectable)} of {len(_declared)} declared prediction "
    "sets are selectable"
)

# %% [markdown]
# **What the menu declares, against what the registry holds.** The check runs here rather than
# before the fit for two reasons. This notebook is itself one of the entries, so asking before
# section 4 reports NLinear as never fitted on the run that is fitting it. And a row in
# `training_runs` is not a fitted configuration: an interrupted run leaves one behind, and a run
# narrowed to one label leaves one that says nothing about the other. The pair a menu entry
# declares - a configuration and a label - is fitted when a prediction set for it is registered
# and listed by a population currently in force, which is the same bar
# [`14_backtest`](14_backtest.ipynb) sweeps at.

# %%
_menu_pairs = set(
    load_model_configs(study, "deep_learning").select("config_name", "label").unique().iter_rows()
)
_registered = load_model_ic(
    ["deep_learning"], case_studies=[CASE_STUDY_ID], require_full_coverage=False
)
_in_force, _in_force_notes = prediction_members_in_force(study)
if _in_force is not None:
    _registered = _registered.filter(pl.col("prediction_hash").is_in(list(_in_force)))
for _note in _in_force_notes:
    print(f"  note: {_note}")
_fitted_pairs = set(_registered.select("config_name", "label").unique().iter_rows())
_unfitted = sorted(_menu_pairs - _fitted_pairs)
print(f"deep_learning menu: {len(_menu_pairs)} configuration-label pairs")
print(f"published and scored in this registry: {len(_menu_pairs & _fitted_pairs)}")
print(
    "declared and never fitted: "
    + (", ".join(f"{config}/{label}" for config, label in _unfitted) or "none")
)

# %% [markdown]
# ## 6. The learning curve
#
# Validation IC at each epoch checkpoint. The peak is amber; any checkpoint left unpublished
# for partial coverage is copper. A curve that rises and then falls is the model fitting and then
# overfitting the validation folds; a flat one says the epoch count was never the binding choice.

# %%
_epochs = checkpoints["checkpoint_value"].cast(pl.Int64).to_list()
_ics = checkpoints["ic_mean_daily"].to_list()
_days = checkpoints["ic_n_days"].to_list()
_colors = [
    COLORS["amber"]
    if epoch == PEAK_EPOCH
    else (COLORS["copper"] if epoch in partial_epochs else COLORS["blue"])
    for epoch in _epochs
]
fig_lc = go.Figure(
    go.Scatter(
        x=_epochs,
        y=_ics,
        mode="lines+markers+text",
        line=dict(color=COLORS["silver_muted"], width=1.5),
        marker=dict(
            color=_colors,
            size=[16 if epoch == PEAK_EPOCH else 11 for epoch in _epochs],
            line=dict(color="white", width=1),
        ),
        text=[
            f"{days:.0f}d" if epoch == PEAK_EPOCH or days < FULL_DAYS else ""
            for epoch, days in zip(_epochs, _days, strict=True)
        ],
        textposition="top center",
        showlegend=False,
    )
)
fig_lc.add_vline(x=PEAK_EPOCH, line=dict(color=COLORS["amber"], width=1, dash="dot"))
fig_lc.update_layout(
    title=(
        f"Validation IC by checkpoint; peak at epoch {PEAK_EPOCH}"
        + (f", partial epochs {partial_epochs} excluded" if partial_epochs else "")
    ),
    height=500,
    width=1000,
    margin=dict(t=70),
    title_font=dict(size=15),
)
fig_lc.update_xaxes(title_text="Training epoch (checkpoint)")
fig_lc.update_yaxes(title_text="Mean cross-sectional IC (validation)")
fig_lc.show()

# %% [markdown]
# ## 7. Against the flat-feature baselines
#
# The peak checkpoint against the linear and gradient-boosting leaders. This is the comparison the
# notebook exists to make: whether reading the history as a sequence adds anything over handing
# the same history to a model as columns.
#
# Each family's stored IC is computed over its own sample, and those samples are not the same:
# sequence eligibility drops the warm-up rows a flat model keeps, so a shared date can carry a
# different cross-section in each. Matching date counts would not settle it. The three prediction
# sets are therefore intersected on their exact `(symbol, timestamp)` keys and the daily rank IC
# is recomputed on the rows all three share - a smaller sample than any family's own, and the
# only one on which they are a single comparison.

# %%
_missing = [name for name, phash in baseline_hashes.items() if not phash]
if _missing:
    raise RuntimeError(
        f"no prediction_hash registered for {_missing}, so their rows cannot be intersected "
        "with this notebook's; re-run the stage that published them"
    )

_frames = {
    name: load_predictions(CASE_STUDY_ID, prediction_hash=phash)
    for name, phash in baseline_hashes.items()
}
_frames[f"NLinear ({CONFIG_NAME})"] = load_predictions(CASE_STUDY_ID, prediction_hash=PEAK_PHASH)
_empty = [name for name, frame in _frames.items() if frame.is_empty()]
if _empty:
    raise RuntimeError(f"no prediction rows on disk for {_empty}; the comparison cannot be made")

COMMON_IC, COMMON_DAYS, COMMON_ROWS = common_sample_daily_ic(_frames)
if not COMMON_IC:
    raise RuntimeError(
        "the three prediction sets share no (symbol, timestamp) rows, so there is no sample "
        "on which they can be compared"
    )
print(f"Common sample: {COMMON_ROWS:,} rows on {COMMON_DAYS:,} dates")
for _name, _ic in COMMON_IC.items():
    print(f"  {_name}: IC={_ic:+.4f} on the shared rows")

# Charted in a fixed order, but only the members the population actually carries. A narrowed
# run - a smoke configuration, or a fit restricted to one family - publishes no linear or GBM
# leader, and a label list written as a literal turns that into a KeyError on a presentational
# cell rather than into a comparison drawn over what is there.
_SELF = f"NLinear ({CONFIG_NAME})"
_labels = [name for name in ["Ridge (Ch11)", "GBM (Ch12)", _SELF] if name in COMMON_IC]
_missing = [name for name in ["Ridge (Ch11)", "GBM (Ch12)"] if name not in COMMON_IC]
if _missing:
    print(
        f"No {' or '.join(_missing)} baseline in this population, so the chart compares "
        f"{len(_labels)} of 3 families. The cross-chapter comparison needs a canonical run."
    )
_vals = [COMMON_IC[name] for name in _labels]
fig_cmp = go.Figure(
    go.Bar(
        x=_labels,
        y=_vals,
        marker=dict(
            color=[
                {"Ridge (Ch11)": COLORS["slate"], "GBM (Ch12)": COLORS["copper"]}.get(
                    name, COLORS["blue"]
                )
                for name in _labels
            ],
            # The amber outline marks this notebook's own model, wherever it lands once the
            # absent families are dropped.
            line=dict(color=COLORS["amber"], width=[3 if n == _SELF else 0 for n in _labels]),
        ),
        text=[f"{value:+.4f}" for value in _vals],
        textposition="outside",
        cliponaxis=False,
        showlegend=False,
    )
)
fig_cmp.update_layout(
    title="Peak-checkpoint IC by family, measured on the same validation dates",
    height=500,
    width=1000,
    margin=dict(t=70),
    title_font=dict(size=15),
)
fig_cmp.update_xaxes(title_text="Model (validation leader per family)")
fig_cmp.update_yaxes(title_text="Peak-checkpoint cross-sectional IC (validation)")
fig_cmp.show()

# %% [markdown]
# ## 8. Is the average stable across time?
#
# The headline IC above is the mean over every validation date, pooled across folds, and a mean
# can be positive while whole stretches of it are not. The per-fold breakdown re-cuts the same
# observations by fold, which is what separates a model with a weak but consistent edge from one
# whose mean is carried by a single fold. The two numbers are not the same statistic: each fold
# contributes a different number of dates to the pooled mean, so the fold ICs below do not
# average to it.

# %% tags=["results"]
folds = (
    Result.open(study, PEAK_PHASH, include_preview=EXECUTION_TIER == "preview")
    .folds()
    .select("fold_id", "ic", "n_entities")
    .sort("fold_id")
)
print(f"Per-fold validation IC ({CONFIG_NAME} @ epoch {PEAK_EPOCH}):")
print(folds)
NEGATIVE_FOLDS = int(folds.filter(pl.col("ic") < 0).height)
print(f"\n  Folds with negative IC: {NEGATIVE_FOLDS} of {folds.height}")
print(f"  Peak-IC prediction_hash: {PEAK_PHASH}")

# %% [markdown]
# ## 9. What to notice

# %%
_coverage_text = (
    f"All {checkpoints.height} checkpoints cover {FULL_DAYS:.0f} validation dates."
    if not partial_epochs
    else f"Checkpoints at epochs {partial_epochs} were not published, on partial coverage."
)
# The baselines named in prose come from the population rather than from a literal, for the
# same reason the chart's labels do: a sentence naming a model the run did not fit is a
# sentence that goes stale silently.
_present = [
    f"{name.split(' (')[0]} **{COMMON_IC[name]:+.3f}**" for name in _labels if name != _SELF
]
_baseline_clause = (
    "against " + " and ".join(_present)
    if _present
    else "with no flat-feature baseline in this population to compare against"
)

display(
    Markdown(
        f"""
- **The simplest sequence model is measured against the flat-feature families on rows all three
  actually share** - {COMMON_ROWS:,} of them across {COMMON_DAYS:,} dates, intersected on
  `(symbol, timestamp)` and recomputed rather than assumed comparable. On that common sample the
  peak checkpoint scores **{COMMON_IC[_SELF]:+.3f}**, {_baseline_clause}. Its own
  mean daily IC over the whole validation sample is **{PEAK_IC:+.3f}** at epoch
  **{PEAK_EPOCH}**.
- **Coverage is comparable across the checkpoint choice.** {_coverage_text} The guard stays
  necessary either way: a collapsed checkpoint scores on fewer dates and can look better for it.
- **The average is not the whole story.** The peak checkpoint is negative in
  **{NEGATIVE_FOLDS} of {folds.height}** validation folds. The holdout is not read here; it is
  evaluated once, after the development-stage selection is fixed.

**Next**: [`11_latent_factors`](11_latent_factors.ipynb) tests whether explicit factor
  structure organizes the cross-section more effectively.
**Book**: Chapter 13, Section 13.8 (Case Study Results).
"""
    )
)
