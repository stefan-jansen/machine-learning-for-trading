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
# # US equities panel: from a ranking to a book, at equal weight
#
# Everything before this notebook produced **predictions**: a number per stock per session saying
# how that stock is expected to rank. Nothing has yet held a position. This is where a ranking
# becomes a portfolio, and it is the first point in the case study where the answer can be
# measured in money rather than in correlation.
#
# **The rule is deliberately the plainest one available.** On each rebalancing date, sort the
# stocks by the model's prediction, go long the top *k* and short the bottom *k*, and put the same
# amount of money in every position. No weighting by conviction, no risk targeting, no overlay -
# those come in the two notebooks after this one, and each is measured against what this produces.
#
# **Why equal weight is the right baseline rather than a weak one.** A sizing rule can flatter a
# ranking or bury it, so measuring a model under a clever allocator confounds two things. Equal
# weight adds the least: whatever separates two models here is the ranking, because nothing else
# differs. It is also the hardest baseline to beat by accident, which is what makes the comparison
# in [`17_portfolio_management`](17_portfolio_management.ipynb) worth making.
#
# **Every member of every model population is backtested, not a shortlist.** A model that ranked
# poorly on information coefficient is still run, because ranking accuracy and strategy
# performance are different questions - a model can order the cross-section well and trade so much
# that nothing survives turnover, or rank indifferently and hold a book that does. Selecting on
# the ranking measure before backtesting would decide the second question with the answer to the
# first.
#
# **This is where selection genuinely begins.** Every notebook so far has said "selection happens
# in `16_backtest`". This notebook produces the validation backtests it happens on; the choice
# itself is made in the strategy notebooks that follow, over the population frozen here.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Describe how a cross-sectional ranking becomes a long-short book, and name every decision that
#   turns has to make.
# - Say why the plainest sizing rule is the right one for a baseline, and what a comparison under
#   a sophisticated allocator would confound.
# - Say why the whole model population is backtested rather than a shortlist chosen on ranking
#   accuracy.
# - Read a validation Sharpe across a grid of models and top-*k* choices and say what varies along
#   each axis.
#
# **Book reference**: Chapter 16, Sections 16.4 to 16.8.
#
# **Prerequisites**: the model notebooks - [`06_linear`](06_linear.ipynb) through
# [`13b_ipca`](13b_ipca.ipynb) - have frozen the named prediction sets listed below, and
# [`15_model_analysis`](15_model_analysis.ipynb) has confirmed each is complete.
#
# **What it writes**: one validation backtest per prediction set member and top-*k* choice, in
# `run_log/registry.db`, frozen as one named baseline set per label.
# [`17_portfolio_management`](17_portfolio_management.ipynb) varies the sizing on what wins here,
# [`18_risk_management`](18_risk_management.ipynb) lays overlays on top, and
# [`19_costs`](19_costs.ipynb) charges the result for trading.

# %%
"""Generate the complete US-equities equal-weight validation baseline."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from case_studies.research import (
    CandidateSet,
    OfficialPopulation,
    Study,
    open_study,
    plan_backtests,
    run_backtests,
)
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.sweep_config import get_entry_schemes_for
from utils.paths import REPO_ROOT
from utils.style import COLORS, show_with_alt, zero_line

# %% [markdown]
# ### Which prediction sets enter the pool
#
# Every model notebook froze its results under a name per label and family, and the list below is
# the pool this baseline ranks. A name missing from it is a family that never competes, so the
# list is the declaration of what the strategy chain is allowed to choose from.
#
# **The weekly experiment is deliberately absent.** [`12_dl_weekly`](12_dl_weekly.ipynb) fits
# `fwd_ret_5d` on a Friday-only grid, and its predictions carry the same label as the daily models
# below. Ranking them together would compare series scored on different sets of decision dates -
# one on every session, one on about a fifth of them - and read the difference as a difference
# between models. It runs and registers; it is not a candidate here.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PREDICTION_SET_NAMES = [
    "us-equities-fwd-ret-1d-linear-v1",
    "us-equities-fwd-ret-5d-linear-v1",
    "us-equities-fwd-ret-21d-linear-v1",
    "us-equities-fwd-ret-1d-gbm-v1",
    "us-equities-fwd-ret-5d-gbm-v1",
    "us-equities-fwd-ret-21d-gbm-v1",
    "us-equities-fwd-ret-1d-tabular-dl-v1",
    "us-equities-fwd-ret-1d-nlinear-v1",
    "us-equities-fwd-ret-1d-lstm-v1",
    "us-equities-fwd-ret-1d-tsmixer-v1",
    "us-equities-fwd-ret-1d-pca-v1",
    "us-equities-fwd-ret-1d-ipca-v1",
    "us-equities-fwd-ret-5d-pca-v1",
    "us-equities-fwd-ret-5d-ipca-v1",
    "us-equities-fwd-ret-21d-pca-v1",
    "us-equities-fwd-ret-21d-ipca-v1",
]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_FAMILIES = []
PREVIEW_CONFIG_NAMES = []
PREVIEW_MAX_PREDICTIONS = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## 2. Every model, no shortlist
#
# The named sets above are opened and their membership checked, and every member goes on to be
# backtested. Nothing here filters on a predictive metric, for the reason the preamble gives: a
# model that ranks the cross-section well can still trade too much to survive turnover, and one
# that ranks indifferently can hold a book that works. Deciding the second question with the answer
# to the first is the mistake the whole design avoids.

# %%
preview_filters = bool(PREVIEW_LABELS or PREVIEW_FAMILIES or PREVIEW_CONFIG_NAMES)
# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if preview_filters or PREVIEW_MAX_PREDICTIONS or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not PREDICTION_SET_NAMES or len(PREDICTION_SET_NAMES) != len(set(PREDICTION_SET_NAMES)):
        raise ValueError("Canonical execution requires unique named prediction sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if not preview_filters or PREVIEW_MAX_PREDICTIONS < 1 or MAX_SYMBOLS < 1:
        raise ValueError(
            "Preview execution requires a catalog filter, prediction limit, and symbol limit"
        )
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

# %% [markdown]
# ## 3. Which rows can be backtested at all
#
# A prediction row is eligible when it is complete, scored on the validation split, and produced
# under this run's execution tier. A row failing any of those is not a weaker candidate; it is a
# row that would make the population mean something other than what it says, so the run refuses
# rather than dropping it silently.

# %%
prediction_catalog = study.predictions.table(include_preview=True)
if EXECUTION_TIER == "canonical":
    prediction_sets = tuple(CandidateSet.one(study, name=name) for name in PREDICTION_SET_NAMES)
    if any(result_set.member_kind != "prediction" for result_set in prediction_sets):
        raise ValueError("Every declared input set must contain predictions")
    prediction_members = tuple(
        member for result_set in prediction_sets for member in result_set.members
    )
    if len(prediction_members) != len(set(prediction_members)):
        raise ValueError("Declared prediction sets overlap")
    predictions = prediction_catalog.filter(pl.col("prediction_hash").is_in(prediction_members))
    if predictions.height != len(prediction_members):
        raise ValueError("The prediction catalog does not contain every declared member")
else:
    predictions = prediction_catalog.filter(pl.col("execution_tier") == "preview")
    if PREVIEW_LABELS:
        predictions = predictions.filter(pl.col("label").is_in(PREVIEW_LABELS))
    if PREVIEW_FAMILIES:
        predictions = predictions.filter(pl.col("family").is_in(PREVIEW_FAMILIES))
    if PREVIEW_CONFIG_NAMES:
        predictions = predictions.filter(pl.col("config_name").is_in(PREVIEW_CONFIG_NAMES))
    predictions = predictions.sort("prediction_hash").head(PREVIEW_MAX_PREDICTIONS)

ineligible = predictions.filter(
    (pl.col("split") != "validation")
    | (pl.col("execution_tier") != EXECUTION_TIER)
    | ~pl.col("complete")
)
if predictions.is_empty() or not ineligible.is_empty():
    raise ValueError(
        "The baseline requires complete validation predictions from one execution tier"
    )

# %% [markdown]
# The selected rows expose the model family, configuration, label, checkpoint, and exact result
# identities that enter the baseline.

# %% tags=["results"]
prediction_population = predictions.select(
    "family",
    "config_name",
    "label",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
).sort("label", "family", "config_name", "checkpoint_value", "prediction_hash")
prediction_population

# %% [markdown]
# ## 4. Planning the backtests before running any
#
# Each prediction row crossed with each declared top-*k* becomes one planned backtest, and every
# identity is written down before the first one runs. That is what makes a run that comes up short
# read as a failure rather than as a smaller experiment, and it is what lets an interrupted run
# resume by filling what is missing.
#
# **The top-*k* grid is a strategy decision, not a tuning knob.** Holding 20 names a side and
# holding 50 are different strategies with different concentration and different turnover, and
# both are backtested rather than one being chosen in advance.

# %%
backtest_config = get_backtest_config(CASE_STUDY_ID)
planned_requests = []
plan_rows = []


def plan_baseline_member(label, prices, scheme, prediction_row):
    selected_prediction = predictions.filter(
        pl.col("prediction_hash") == prediction_row["prediction_hash"]
    )
    signal = {key: value for key, value in scheme.items() if key != "name"}
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        prices=prices,
        chapter="ch16",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One prediction and signal request must plan one backtest")
    expected_hash = plan.expected_hashes[0]
    request = {
        "label": label,
        "prediction_hash": prediction_row["prediction_hash"],
        "selection": selected_prediction,
        "signal_name": scheme["name"],
        "signal": signal,
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "family": prediction_row["family"],
        "config_name": prediction_row["config_name"],
        "checkpoint_kind": prediction_row["checkpoint_kind"],
        "checkpoint_value": prediction_row["checkpoint_value"],
        "signal": scheme["name"],
        "prediction_hash": prediction_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
for label in predictions.get_column("label").unique().sort().to_list():
    label_predictions = predictions.filter(pl.col("label") == label)
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
    )
    n_assets = prices.get_column("symbol").n_unique()
    schemes = get_entry_schemes_for(
        CASE_STUDY_ID,
        label,
        n_assets,
        long_short=backtest_config.long_short,
    )
    if not schemes or any(scheme["method"] != "equal_weight_top_k" for scheme in schemes):
        raise ValueError(f"{label} does not declare only equal-weight baseline schemes")
    for prediction_row in label_predictions.iter_rows(named=True):
        for scheme in schemes:
            request, row = plan_baseline_member(label, prices, scheme, prediction_row)
            planned_requests.append(request)
            plan_rows.append(row)
    del prices

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "family", "config_name", "checkpoint_value", "signal", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The baseline plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-baseline-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## 5. Running them
#
# Each planned backtest is independent, so a failure costs that member and leaves the rest usable,
# and the declared membership stays intact for a re-run to fill.

# %%
execution_rows = []
failure_rows = []


def execute_baseline_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        prices=prices,
        chapter="ch16",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Baseline execution changed its planned identity")
    return {
        "label": request["label"],
        "prediction_hash": request["prediction_hash"],
        "signal": request["signal_name"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in predictions.get_column("label").unique().sort().to_list():
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
    )
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            execution_rows.append(execute_baseline_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "prediction_hash": request["prediction_hash"],
                    "signal": request["signal_name"],
                    "backtest_hash": request["expected_hash"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    del prices

# %% tags=["results"]
execution_diagnostics = pl.DataFrame(
    execution_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "signal": pl.String,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "signal": pl.String,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Baseline population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## 6. Naming the baseline sets
#
# One frozen set per label, under a name the later notebooks open by. Only an unnarrowed canonical
# run publishes one, because a name must not mean two different member sets at two different times.

# %% tags=["results"]
set_rows = []
completed = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed.height != planned_population.height
    or completed.filter(~pl.col("complete")).height
    or completed.filter(pl.col("stage") != "signal").height
    or completed.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The baseline catalog is incomplete or mis-staged")
if EXECUTION_TIER == "canonical":
    for label in completed.get_column("label").unique().sort().to_list():
        label_name = label.replace("_", "-")
        # No comparison_contract, matching cme_futures/research_workflow.py:811, which builds the
        # same per-label pool across the full funnel and declares nothing. An empty contract makes
        # every protocol field required-constant, which is the guard: if two members disagree on
        # `cv` they measured their Sharpe on different folds and ranking them is not a comparison,
        # and this field is the only thing checking that. Latent-factor members will refuse on
        # `feature_artifacts` when they enter this pool - latent builds it from a different object
        # than the other five families (latent_factors/case_study.py:337-383, carrying the label
        # digest and setup.yaml bytes). That refusal is a known adapter defect surfacing, not a
        # property to declare around; report it rather than adding the field here.
        result_set = study.backtests.freeze(
            completed.filter(pl.col("label") == label),
            name=f"us-equities-{label_name}-baseline-v1",
        )
        set_rows.append(
            {"label": label, "set_name": result_set.name, "members": len(result_set.members)}
        )

compatible_sets = pl.DataFrame(
    set_rows,
    schema={"label": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# ## 7. What came out
#
# One row per model, checkpoint and top-*k*. Read across the two axes separately: down a model at
# fixed top-*k* is what the ranking was worth, and across top-*k* at fixed model is what
# concentration did to it.
#
# **These are gross of trading costs.** Every number here is what the book earned before anything
# was charged for turning it over, and on a three-thousand-name panel rebalanced daily that is a
# large omission by design - [`19_costs`](19_costs.ipynb) is where it is put back.

# %% tags=["results"]
groups = completed.select("label", "family").unique().sort("label", "family").rows(named=True)
fig, ax = plt.subplots(figsize=(9, 5))
for position, group in enumerate(groups):
    values = completed.filter(
        (pl.col("label") == group["label"]) & (pl.col("family") == group["family"])
    ).get_column("sharpe")
    jitter = (values.arg_sort().to_numpy() % 9 - 4) / 40
    ax.scatter(
        position + jitter,
        values.to_numpy(),
        alpha=0.45,
        s=18,
        color=COLORS["blue"],
        edgecolors="none",
    )
zero_line(ax)
ax.set_xticks(
    range(len(groups)),
    [f"{group['label']}\n{group['family']}" for group in groups],
)
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylabel("Validation Sharpe")
ax.set_title("Equal-weight baseline Sharpe by label and model family")
fig.tight_layout()
show_with_alt(
    fig,
    "A strip plot with one column per label and model family. Each point is one complete "
    "equal-weight validation backtest, placed at its Sharpe ratio on the vertical axis and "
    "spread horizontally within its column so overlapping points stay visible. A dashed "
    "horizontal reference line marks zero.",
)

# %% [markdown]
# The allocation notebook consumes these complete named baseline sets. Selection remains based on
# validation backtest Sharpe, with the checkpoint retained as part of the selected configuration.

# %% [markdown]
# ## What to notice
#
# **A ranking and a strategy are not the same thing, and this is where they separate.** Every model
# was scored on how well it ordered the cross-section. What it earns depends on that order *and*
# on how often the order changes, because every change is a trade. A model can rank well and turn
# its book over so fast that nothing survives, and the previous notebooks had no way to see it.
#
# **Equal weight is what makes the comparison about the models.** Nothing here estimates anything
# from the data beyond the predictions themselves, so a difference between two rows is a difference
# between two rankings. Every sizing rule added later has to earn its place against that.
#
# **Top-*k* is a strategy axis, not a tuning axis.** Twenty names a side and fifty are different
# strategies - different concentration, different turnover, different capacity - and both are
# backtested rather than one being chosen up front.
#
# **Everything here is gross.** No commission, no spread, no borrow cost on the short leg. On a
# daily-rebalanced long-short book across three thousand names that omission is large and is put
# back in [`19_costs`](19_costs.ipynb); until then no number on this page is what anyone would have
# received.
#
# **Known limitations.** Validation folds have been read many times over by the time a case study
# reaches this notebook, so a validation Sharpe is a selection statistic rather than an estimate of
# future performance - the holdout, opened once at the end, is what speaks to that. The book is
# rebalanced on a fixed schedule with no capacity constraint, so a position is taken at whatever
# size the rule implies whether or not the stock trades enough to absorb it.
#
# **Next**: [`17_portfolio_management`](17_portfolio_management.ipynb) keeps these names and
# changes how the money is spread across them.
