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
# # ETF allocation: how much is the signal, and how much is the weighting
#
# [`14_backtest`](14_backtest.ipynb) traded every prediction set one way: pick the top k funds by
# predicted score and hold them in equal weight. That rule makes one decision - **which** funds to
# hold - and declines the other, which is **how much** of each. This notebook makes the second
# decision six different ways on the same predictions and measures what it is worth.
#
# The reason to ask is that the two decisions are not obviously comparable in size. A weighting
# scheme can only redistribute capital among funds the signal already chose; it cannot put money
# into a fund the ranking left out. So the question is not whether a better allocator helps, it is
# whether the help is on the scale of the difference between model families, or an order of
# magnitude below it.
#
# **The concentration is swept alongside the allocator, because the two interact.** At a top-5
# selection there is little for an allocator to do and the schemes converge on each other; at
# top-20 the basket holds funds with genuinely different volatilities and the schemes separate.
# Reading either axis alone would attribute the interaction to whichever one was varied.
#
# **Learning objectives**
#
# - Say what a portfolio allocator can and cannot change about a strategy built on a ranking.
# - Compare equal-weight, score-weighted, inverse-volatility, risk-parity, mean-variance and
#   hierarchical risk parity on one set of predictions.
# - Read an allocator comparison against the spread across prediction sources, rather than on its
#   own scale.
# - Say why the concentration level and the allocator have to be swept together.
#
# **Book reference**: Chapter 17, Sections 17.2 to 17.8.
#
# **Prerequisites**: [`14_backtest`](14_backtest.ipynb), whose signal-stage results select the
# prediction sets this sweep starts from.
#
# **What it writes**: one row in `backtest_runs` per prediction set, concentration level and
# allocator, at `stage='allocation'`. [`16_risk_management`](16_risk_management.ipynb) overlays
# risk rules on the leading combinations from here, and [`17_costs`](17_costs.ipynb) then prices
# the winner of both.

# %%
"""Sweep portfolio allocators and concentration levels over the leading ETF predictions."""

import json
import time
import warnings

import plotly.graph_objects as go
import polars as pl

from case_studies.research import open_study, split_retired_members
from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import build_backtest_spec, strategy_view
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import (
    load_existing_backtest_hashes,
    load_prediction_index,
    read_predictions,
    resolve_best_backtest_runs,
    resolve_best_predictions,
)
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.style import COLORS, show_plotly_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
LABEL = ""
MAX_SYMBOLS = 0
# None means every live prediction set; an int caps the shortlist.
TOP_N_PREDICTIONS = None
# Both names stay bound here although nothing below reads them: that is what makes the harness
# force preview and supply a workspace (`tests/pm_helpers.py:954`). Without them the canonical
# branch regenerates in place, which needs symlinks a CI checkout does not have.
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %% [markdown]
# ## 1. Which predictions the sweep starts from
#
# The prediction sets carried forward are the leaders **by signal-stage Sharpe**, not by
# information coefficient. That is the selection rule the case study states, and
# [`14_backtest`](14_backtest.ipynb) is where the two orderings were shown to disagree: a ranking
# that correlates well with returns across the whole cross-section is not the same thing as a
# ranking whose head is worth holding.
#
# `checkpoints_per_config` caps how many checkpoints of a single configuration may advance. Without
# it a family that saves eight checkpoints of one network would fill the shortlist with eight
# near-identical variants of one model, and the allocator comparison below would be measured almost
# entirely on that one prediction.

# %%
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_PREDICTIONS is None:
    TOP_N_PREDICTIONS = get_top_n_predictions(CASE_STUDY_ID, "allocation")
if not LABEL:
    LABEL = bt_config.primary_label
CHECKPOINTS_PER_CONFIG = get_checkpoints_per_config(CASE_STUDY_ID)
print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")

# %% [markdown]
# **The population the ranking runs over.** A model notebook that refits publishes a second
# generation under the same population name, and the generation it replaced stays in the registry:
# complete, current under a schema version that has not moved, and carrying whatever backtests the
# previous sweep registered for it. Ranking over both lets an identity its own publisher has
# retired take a slot from a live one, which is not a reporting error - the retired row is what
# every stage after this one then builds on. `split_retired_members` reads the population lineage,
# and the surviving members are what every reader below is scoped to.

# %%
LIVE_PREDICTIONS = (
    split_retired_members(
        open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None),
        load_prediction_index(CASE_STUDY_ID, label=LABEL, split="validation"),
    )
    .live["prediction_hash"]
    .to_list()
)
if not LIVE_PREDICTIONS:
    raise RuntimeError(
        f"no live prediction sets for {CASE_STUDY_ID}/{LABEL}/validation; run 14_backtest first"
    )
print(f"Live prediction sets: {len(LIVE_PREDICTIONS):,}")

# %%
top_preds = resolve_best_predictions(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    stage="signal",
    top_n=TOP_N_PREDICTIONS,
    checkpoints_per_config=CHECKPOINTS_PER_CONFIG,
    prediction_hashes=set(LIVE_PREDICTIONS),
)
if top_preds.is_empty():
    raise RuntimeError(
        "no signal-stage backtests are registered, so there is nothing to advance; "
        "run 14_backtest first"
    )
print(f"{len(top_preds)} prediction sets advance, ranked by equal-weight top-k Sharpe:")
top_preds.select("source", "sharpe")

# %% [markdown]
# ## 2. The sweep
#
# Every advancing prediction, crossed with every concentration level and every allocator. As in the
# signal stage this is orchestration over one `run_backtest` call: the allocator travels inside the
# strategy specification, so it lands in the backtest hash and a combination already registered is
# not recomputed.

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
n_assets = prices["symbol"].n_unique()
TOP_K_VALUES = get_top_k_values_for(CASE_STUDY_ID, LABEL, n_assets)
ALLOC_CONFIGS = get_allocators(CASE_STUDY_ID)
n_total = len(top_preds) * len(TOP_K_VALUES) * len(ALLOC_CONFIGS)

print(f"Prices: {len(prices):,} rows, {n_assets} tradeable funds")
print(f"Concentration grid: top_k in {TOP_K_VALUES}")
print(f"Allocators ({len(ALLOC_CONFIGS)}): {', '.join(a['method'] for a in ALLOC_CONFIGS)}")
print(
    f"Grid: {len(top_preds)} predictions x {len(TOP_K_VALUES)} concentrations x "
    f"{len(ALLOC_CONFIGS)} allocators = {n_total} backtests"
)

# %% [markdown]
# Mean-variance optimization is the one allocator whose cost is not bounded by the grid: it inverts
# a covariance matrix per rebalance, and on a wide basket that is where the sweep's time goes. The
# loop measures its first fit, projects the remainder, and drops it from the rest of the grid if
# that projection exceeds the budget - reporting that it did, because an allocator silently absent
# from the comparison below would read as one that was tried and lost.

# %% [markdown]
# Two facts, and a reader needs both: what the **stage** holds before this run, and what **this
# execution** did. `run_backtest` returns a cached result and a fresh fit through the same call,
# so a warm re-run would otherwise report a completed sweep in no time at all - a wrong number
# that looks exactly like a right one - while reporting only what this run computed would make
# that same re-run look like an empty stage.

# %%
BUDGET_SECONDS = 3600
MVO_METHODS = ("mvo", "mvo_ledoit_wolf")

n_done = 0
served = 0
failures = []
skip_mvo = False
sweep_start = time.monotonic()
registered_before = load_existing_backtest_hashes(CASE_STUDY_ID, stage="allocation")
print(f"Allocation-stage backtests already registered: {len(registered_before):,}")

for top_k in TOP_K_VALUES:
    print(f"\n--- top_k = {top_k} ---")
    for pred_row in top_preds.iter_rows(named=True):
        pred_hash = pred_row["prediction_hash"]
        source = pred_row["source"]
        predictions = read_predictions(CASE_STUDY_ID, pred_hash)

        for alloc in ALLOC_CONFIGS:
            alloc_name = alloc["method"]
            if skip_mvo and alloc_name in MVO_METHODS:
                continue
            n_done += 1

            spec = build_backtest_spec(
                CASE_STUDY_ID,
                bt_config,
                prices=prices,
                prediction_hash=pred_hash,
                initial_cash=bt_config.initial_cash,
                chapter="ch17",
                signal={
                    "method": "equal_weight_top_k",
                    "top_k": top_k,
                    "long_short": bt_config.long_short,
                },
                allocation={**alloc, "top_k": top_k, "long_short": bt_config.long_short},
            )

            started = time.monotonic()
            try:
                result = run_backtest(
                    CASE_STUDY_ID,
                    pred_hash,
                    spec,
                    prices=prices,
                    predictions=predictions,
                    label=LABEL,
                    register=True,
                    initial_cash=bt_config.initial_cash,
                    calendar=bt_config.calendar,
                )
            except Exception as error:
                failures.append(
                    {
                        "top_k": top_k,
                        "source": source,
                        "allocator": alloc_name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            elapsed = time.monotonic() - started
            if result.backtest_hash in registered_before:
                served += 1

            if alloc_name in MVO_METHODS and not skip_mvo:
                remaining = len(top_preds) * len(TOP_K_VALUES) - 1
                projected = (time.monotonic() - sweep_start) + elapsed * remaining
                if projected > BUDGET_SECONDS:
                    skip_mvo = True
                    print(
                        f"    dropping {alloc_name} from the rest of the grid: "
                        f"projected {projected / 60:.0f} minutes against a "
                        f"{BUDGET_SECONDS / 60:.0f} minute budget"
                    )

            print(
                f"  [{n_done}/{n_total}] k={top_k} {source} x {alloc_name}: "
                f"Sharpe={result.metrics.get('sharpe', 0):+.3f}"
            )

print(
    f"\nSweep complete in {(time.monotonic() - sweep_start) / 60:.1f} minutes: "
    f"{n_done - served - len(failures)} computed, {served} served from the registry, "
    f"{len(failures)} failed"
)

# What this sweep actually advanced, which is narrower than "every live prediction". The
# allocation stage holds rows from earlier sweeps too, and a live prediction this run did not
# advance still has historical allocation rows. Reporting over those would let a prediction the
# current sweep never touched set the source spreads, the concentration table and the leaders.
ADVANCED_PREDICTIONS = top_preds["prediction_hash"].to_list()

# An allocator dropped part-way through the grid has results for some cells and not others, so
# its average is taken over an easier subset than a complete allocator's. Comparing the two
# would read the truncation as a property of the method.
INCOMPLETE_ALLOCATORS = sorted(MVO_METHODS) if skip_mvo else []
if INCOMPLETE_ALLOCATORS:
    print(
        f"Excluded from the allocator comparison, incomplete grid: "
        f"{', '.join(INCOMPLETE_ALLOCATORS)}"
    )
if failures:
    failure_frame = pl.DataFrame(failures)
    print(f"{failure_frame.height} backtests raised. Distinct causes:")
    print(failure_frame.group_by("error").len().sort("len", descending=True))
else:
    print("no backtest raised")

# %% [markdown]
# ## 3. What the allocator bought
#
# From here the notebook queries the registry rather than the sweep it just ran, so the tables
# describe the whole registered allocation stage and not the part this session happened to compute.
#
# The comparison is restricted to this notebook's label and to the prediction sets that advanced.
# Both restrictions matter: the registry accumulates rows from every label the case study has ever
# swept, and from earlier sweeps whose shortlist differed, and folding those in would rank
# allocators on a population that was never compared against itself.

# %%
explorer = BacktestExplorer(CASE_STUDY_ID)
alloc_comparison = explorer.compare_allocators(
    label=LABEL,
    prediction_hashes=ADVANCED_PREDICTIONS,
)
if INCOMPLETE_ALLOCATORS:
    alloc_comparison = alloc_comparison.filter(~pl.col("allocator").is_in(INCOMPLETE_ALLOCATORS))
if alloc_comparison.is_empty():
    raise RuntimeError("the allocation stage registered no comparable rows")

# %% [markdown]
# ### Against what the allocator changed, and what the prediction changed
#
# The bar chart is the mean Sharpe per allocator. On its own it says which weighting scheme did
# best on average, which is the less interesting half. The number printed beneath it is the one to
# read it against: the spread across allocators next to the spread across the prediction sets they
# were applied to. If the second is the larger, then which model produced the ranking decided more
# than how the ranking was weighted, and no allocator rescues a prediction that had nothing in it.

# %% tags=["results"]
alloc_comparison

# %%
ordered = alloc_comparison.sort("avg_sharpe")
fig = go.Figure(
    go.Bar(
        x=ordered["avg_sharpe"].to_list(),
        y=ordered["allocator"].to_list(),
        orientation="h",
        marker_color=COLORS["blue"],
        showlegend=False,
    )
)
fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=COLORS["neutral"])
fig.update_xaxes(title_text="Mean Sharpe ratio across the allocation sweep")
fig.update_yaxes(title_text="Allocator")
fig.update_layout(
    title="Mean Sharpe by weighting scheme",
    height=380,
    width=800,
    margin=dict(t=90),
)
_span = ordered["avg_sharpe"]
show_plotly_with_alt(
    fig,
    "Horizontal bar chart of the mean Sharpe ratio of every allocation-stage backtest, one bar per "
    f"weighting scheme, with a dashed line at zero. Counted from the frame: {ordered.height} "
    f"allocators, mean Sharpe from {_span.min():+.3f} to {_span.max():+.3f}.",
)

# %% [markdown]
# The whole allocation stage for this label, with the concentration and the weighting scheme read
# back out of each row's own stored specification. They are not columns of the registry: they are
# part of what the backtest hash was taken over, so reading them from the spec is reading the same
# declaration the run was identified by rather than a second copy of it.
#
# It takes two readers, because neither carries both halves. `resolve_best_backtest_runs` returns
# the stored specification and projects the model away; `BacktestExplorer.best` returns the model
# and drops the specification. They share `backtest_hash`, which is what the join is on.


# %%
def _strategy_field(spec_json: str, section: str, field: str):
    """Read one declared field out of a registered backtest specification."""
    return strategy_view(json.loads(spec_json)).get(section, {}).get(field)


specs = resolve_best_backtest_runs(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    stage="allocation",
    top_n=100000,
    prediction_hashes=set(ADVANCED_PREDICTIONS),
)
if specs.is_empty():
    raise RuntimeError("the allocation stage registered no rows for this label")
allocation_rows = (
    specs.with_columns(
        pl.Series(
            "allocator",
            [_strategy_field(spec, "allocation", "method") for spec in specs["spec_json"]],
        ),
        pl.Series(
            "top_k",
            [_strategy_field(spec, "signal", "top_k") for spec in specs["spec_json"]],
            dtype=pl.Int64,
        ),
    )
    .drop("spec_json", "sharpe")
    .join(
        explorer.best(
            stage="allocation",
            top_n=100000,
            label=LABEL,
            prediction_hashes=ADVANCED_PREDICTIONS,
        ).select("backtest_hash", "source", "sharpe"),
        on="backtest_hash",
        how="inner",
    )
)
# Dropped here rather than at each reader. An allocator that ran part of the grid is excluded
# from the allocator comparison above for a reason that holds everywhere: its average is taken
# over an easier subset. Filtering only the concentration table left its partial rows moving the
# prediction-source spread below - which is then compared against a span computed on the filtered
# frame - and standing among the leading combinations at the end.
if INCOMPLETE_ALLOCATORS:
    allocation_rows = allocation_rows.filter(~pl.col("allocator").is_in(INCOMPLETE_ALLOCATORS))
if allocation_rows.is_empty():
    raise RuntimeError("no allocation row carries both a stored specification and a source")

# %% [markdown]
# Beneath the spread comparison, the same rows crossed the other way: one row per concentration
# level, one column per allocator. A weighting scheme has nothing to redistribute when the basket
# is small, so the columns should converge at the tightest concentration and separate as the basket
# widens - and where they do not, the allocator is responding to something other than how much the
# holdings differ from each other.

# %% tags=["results"]
by_source = allocation_rows.group_by("source").agg(pl.col("sharpe").mean().alias("avg_sharpe"))
allocator_span = _span.max() - _span.min()
source_span = by_source["avg_sharpe"].max() - by_source["avg_sharpe"].min()
print(f"spread in mean Sharpe across {ordered.height} allocators:         {allocator_span:.3f}")
print(f"spread in mean Sharpe across {by_source.height} prediction sources: {source_span:.3f}")
print(
    "the prediction moved performance more than the allocator did"
    if source_span > allocator_span
    else "the allocator moved performance more than the prediction did"
)

concentration = (
    allocation_rows.drop_nulls(["top_k", "allocator"])
    .group_by("top_k", "allocator")
    .agg(pl.col("sharpe").mean().alias("avg_sharpe"))
    .sort("top_k", "allocator")
)
if concentration.is_empty():
    raise RuntimeError("no allocation row declares both a concentration and an allocator")
print("\nMean Sharpe by concentration and allocator:")
print(concentration.pivot(on="allocator", index="top_k", values="avg_sharpe"))

# %% [markdown]
# ### The leading combinations

# %% tags=["results"]
# Ranked over the whole advanced allocation stage, then narrowed to the allocators whose grid
# completed - so a truncated allocator cannot lead the table on the easier subset it finished.
# The ranking is taken deep and cut to ten afterwards rather than asked for ten and filtered,
# which would return fewer than ten rows whenever an excluded row was inside the top ten.
#
# The sort is re-applied after the join because a semi-join carries no ordering guarantee: it
# returns the left rows that matched, in whatever order the join produced them, so `head(10)`
# on the unsorted result would be ten eligible rows rather than the ten leading ones.
_leaders = explorer.best(
    stage="allocation", top_n=100000, label=LABEL, prediction_hashes=ADVANCED_PREDICTIONS
)
if INCOMPLETE_ALLOCATORS:
    _leaders = _leaders.join(
        allocation_rows.select("backtest_hash"), on="backtest_hash", how="semi"
    ).sort("sharpe", descending=True, nulls_last=True)
_leaders.head(10).select("source", "signal_method", "sharpe", "cagr", "max_drawdown")

# %% [markdown]
# ## 4. What to notice
#
# **An allocator cannot choose an asset.** Every scheme here weights the same top-k basket, chosen
# by the same predicted ranking, rebalanced on the same calendar. What separates two rows of the
# comparison above is entirely how capital was split among funds the signal had already picked, so
# whatever the allocator is worth is bounded above by how much the funds in that basket differ from
# each other.
#
# **The concentration is a portfolio decision made before the allocator sees anything.** Holding
# five funds instead of twenty changes which funds are in the basket, how much of the ranking's
# information is used, and how much any weighting scheme can move the result. It is swept here
# rather than fixed because the answer to "does the allocator matter" is different at each level.
#
# **Mean-variance is the one scheme whose cost scales with the basket.** Estimating a covariance
# matrix from a rolling window and inverting it is where the sweep's time goes, and the estimation
# error in that matrix is also why the theoretical advantage over a simpler scheme often does not
# appear. The budget check above is an operational guard, not a judgement about the method.
#
# **Known limitations.** Every Sharpe here is gross of the cost differences between schemes: a
# scheme that rebalances weights within a stable basket trades more than one that does not, and
# nothing in this stage charges it for that. [`17_costs`](17_costs.ipynb) is where that is priced.
# The volatility windows are declared rather than tuned. And this is measured on validation folds
# throughout; the holdout is not consulted.

# %% [markdown]
# **Next**: [`16_risk_management`](16_risk_management.ipynb) overlays position and portfolio risk
# rules on the leading combinations here, and [`17_costs`](17_costs.ipynb) then walks a cost grid
# over whichever of the two the measurement favours.
