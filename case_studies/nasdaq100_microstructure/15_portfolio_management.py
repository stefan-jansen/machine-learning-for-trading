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
# # NASDAQ-100 Microstructure: Portfolio: Allocator Sweep
#
# **Chapter 17 — Portfolio Construction**
#
# This notebook sweeps **top predictions × TOP_K concentration × allocators** on
# the **full universe** to ask a focused question: can portfolio construction
# rescue the every-bar strategy that Chapter 16 (Act 1) showed is cost-defeated?
# Each combination re-sizes the top signal-stage predictions with `equal_weight`,
# `score_weighted`, or `inverse_vol`, rebalancing every 15-minute bar across all
# 114 names.
#
# The answer, established below, is that it cannot: every allocator lands deep in
# negative territory. Allocator choice and concentration are **second-order** to
# the turnover problem — the binding constraint is how often the strategy trades,
# not how it weights what it holds. The cost-feasible carrier (Chapter 16,
# Section 4) addresses turnover at the *signal* stage through the slot mechanism,
# which is itself the position-sizing rule and so does not pass through this
# allocator sweep. This notebook therefore documents why standard allocation is
# the wrong lever here, motivating the cost-feasibility screen and the cadence
# analysis in Chapter 18.
#
# Sections 1–2 generate the full-universe allocation backtests (write to
# registry). Section 3 queries the registry via `BacktestExplorer` and can be
# re-run independently.
#
# **Learning Objectives:**
# 1. Sweep top signal-stage predictions × concentration levels × allocation methods
# 2. Compare equal-weight, score-weighted, and inverse-vol sizing under intraday costs
# 3. Show that allocator choice is second-order to trade frequency at 15-minute cadence
#
# **Book Reference:** Chapter 17, Sections 17.2–17.8
#
# **Prerequisites:** Completed Ch16 backtest with results in `registry.db`.

# %%
"""NASDAQ-100 Microstructure: Portfolio: Allocator Sweep."""

import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import build_backtest_spec
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.notebook_contracts import excluded_families
from case_studies.utils.registry import read_predictions, resolve_best_predictions
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_expensive_allocators_skip,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
LABEL = ""
MAX_SYMBOLS = 0
# Default value comes from setup.yaml::backtest.sweep.expensive_allocators_skip
# after CASE_STUDY_ID is set; papermill can still override at injection time.
SKIP_EXPENSIVE_ALLOC = None
TOP_N_PREDICTIONS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_PREDICTIONS is None:
    TOP_N_PREDICTIONS = get_top_n_predictions(CASE_STUDY_ID, "allocation")
CHECKPOINTS_PER_CONFIG = get_checkpoints_per_config(CASE_STUDY_ID)
if SKIP_EXPENSIVE_ALLOC is None:
    SKIP_EXPENSIVE_ALLOC = get_expensive_allocators_skip(CASE_STUDY_ID)
if not LABEL:
    LABEL = bt_config.primary_label

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")
if excluded_families(CASE_STUDY_ID):
    print(
        "Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY_ID)))} pending corrected reruns"
    )

# %% [markdown]
# ## 1. Load Top Predictions from Signal Stage
#
# We select from the full-universe signal-stage Sharpe ranking, which is led by
# GBM slot configurations. The allocation sweep below re-runs these predictions
# with `equal_weight_top_k` selection rebalancing every bar — deliberately
# stripping out the slot mechanism's turnover control so the allocator
# comparison is run on the naive every-bar baseline.
#
# That stripping is what makes this notebook answer a question rather than repeat
# one. The slot mechanism allocates a fixed weight per slot and holds it for a
# swept `hold_bars`, so in the sweep configuration's own words the slots ARE the
# allocation: the mechanism decides what is held, how much of it, and for how
# long, in one rule. Feeding it through an allocator sweep would vary the
# weighting on top of a turnover control already doing that work, and any
# difference between allocators would be read against a baseline that had
# already addressed the problem. Running the sweep on the naive every-bar
# baseline instead isolates the weighting decision, which is the only decision
# an allocator makes.

# %%
top_preds = resolve_best_predictions(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    stage="signal",
    top_n=TOP_N_PREDICTIONS,
    checkpoints_per_config=CHECKPOINTS_PER_CONFIG,
)
print(f"Top {len(top_preds)} prediction sources by equal-weight baseline Sharpe:")
print(top_preds.select(["source", "sharpe"]))

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)
n_assets = prices["symbol"].n_unique()
print(f"Prices: {len(prices):,} rows, {n_assets} assets")

# %% [markdown]
# ## 2. Allocation Sweep
#
# For each (prediction × TOP_K × allocator), call `run_backtest()` with the
# allocation config added to the strategy spec. The spec hash automatically
# differentiates these from signal-stage backtests.
#
# **The book is long-short, and that shapes every allocator below.**
# `get_backtest_config("nasdaq100_microstructure").long_short` is `True`, and the
# notebook passes it into both the signal and the allocation spec. So each
# decision time selects **two** sets of `TOP_K` names - the top of the ranking
# long and the bottom short - and every allocator here normalises the two legs
# **separately**: the long weights sum to +1 and the short weights to -1. A
# statement below about "the selection" is a statement about one leg.
#
# **The three allocators swept here, and what each one decides.** All three are
# handed the same two ranked tails and differ only in the weight they put on each
# name:
#
# - **`equal_weight`** is a pass-through: the allocation branch returns the
#   weights it was handed, unchanged. The equal sizing it is named for was
#   already done one stage earlier, by the `equal_weight_top_k` selection this
#   sweep re-runs the predictions under, which gives every name in a leg the same
#   share of that leg. So this arm is the null hypothesis in the literal sense -
#   it is the book with no allocator applied to it, and the name ranked first and
#   the name ranked `TOP_K`-th are held in the same size.
# - **`score_weighted`** sizes each name in proportion to the magnitude of its
#   predicted score, so the ranking becomes a magnitude rather than an order.
#   That is a stronger claim about the model than `equal_weight` makes - it
#   asserts the predicted values are calibrated well enough that twice the score
#   deserves twice the capital, not merely that the order is right. Weights are
#   absolute scores normalised within each leg, with an equal-weight fallback
#   applied per leg at any decision time where that leg's scores sum to zero.
# - **`inverse_vol`** sizes each name in inverse proportion to its own recent
#   realized volatility, so a quiet name carries more capital than a volatile
#   one. It uses no property of the prediction at all beyond membership; the
#   quantity it reads is a property of the price series. `allocator_lookback` in
#   `config/setup.yaml` sets the window that volatility is estimated over, and it
#   is counted in rows of the price frame rather than in decision bars: the frame
#   is one-minute, and `_compute_rolling_vol` rolls over its rows without
#   resampling, so 520 is **520 minutes - about 1.3 sessions**, not the month a
#   fifteen-minute reading of it would give. It is the one setting that decides
#   how quickly a name's size responds to its own volatility, and at 1.3 sessions
#   it responds fast.
#
# The three therefore span a real axis: one uses none of the signal's magnitude,
# one uses all of it, and one substitutes a price property for it.
#
# Two further allocators exist and need something the three above do not: a full
# covariance or correlation matrix between the held names, rather than a per-name
# quantity. `mvo_ledoit_wolf` estimates one under Ledoit-Wolf shrinkage and `hrp`
# clusters on a rolling correlation window. Estimated from the same short-horizon
# returns being traded, that matrix has far more entries to fill than the return
# series has independent observations to fill them from, so it is noisy in a way
# a per-name volatility is not, and it is expensive to recompute over the full
# history.
#
# `risk_parity` is grouped with them by name and does not belong there:
# `compute_risk_parity_weights` weights by inverse volatility raised to 1.5 and
# reads no covariance at all, using that exponent as a proxy for the empirical
# relation between volatility and correlation. It is a per-name quantity like
# `inverse_vol`, and cheap for the same reason. For both reasons this case study does not
# declare them at all - `backtest.sweep.allocators` in `config/setup.yaml` names
# the three above and nothing else - so `SKIP_EXPENSIVE_ALLOC` below filters a
# list they are already absent from. It is there for the case studies that do
# declare them, and it is a no-op here.
#
# **`TOP_K` is the other axis, and it decides concentration rather than
# weighting.** It is a count per leg, so the book holds up to `2 * TOP_K` names.
# A small `TOP_K` bets each leg on the extreme of the ranking, where the model is
# most confident and least diversified; a large one walks both legs in towards
# the middle of the cross-section, where the ranking barely separates names.
#
# It is swept alongside the allocators because the two interact, though not in a
# direction that can be asserted in advance. `score_weighted` differs from
# `equal_weight` by however much the selected scores are dispersed **within a
# leg**, and that is not a function of how many names the leg holds: five names
# on nearly equal scores are weighted nearly equally, and widening to twenty by
# admitting lower-scored names spreads them out. Which way `TOP_K` moves the gap
# is a question for the results below.
#
# The comparison to draw from this grid is between allocators under one trading
# rule, not between allocators and anything else. Every configuration here
# rebalances at every decision time across the whole universe, so they all carry
# the same turnover, and a weighting scheme redistributes capital across
# positions without changing how often those positions turn over.

# %%
TOP_K_VALUES = get_top_k_values_for(CASE_STUDY_ID, LABEL, n_assets)
print(f"TOP_K grid: {TOP_K_VALUES} (universe: {n_assets} assets)")

_ALL_ALLOC_CONFIGS = get_allocators(CASE_STUDY_ID)

_EXPENSIVE = {"mvo_ledoit_wolf", "hrp"}
if SKIP_EXPENSIVE_ALLOC:
    ALLOC_CONFIGS = [a for a in _ALL_ALLOC_CONFIGS if a["method"] not in _EXPENSIVE]
    print(
        f"Skipping expensive allocators ({', '.join(_EXPENSIVE)}) — "
        f"covariance estimation on 1.3M 15-min bars is prohibitive"
    )
else:
    ALLOC_CONFIGS = _ALL_ALLOC_CONFIGS

n_total = len(top_preds) * len(TOP_K_VALUES) * len(ALLOC_CONFIGS)
print(
    f"Total backtests: {len(top_preds)} preds × {len(TOP_K_VALUES)} top_k × "
    f"{len(ALLOC_CONFIGS)} allocs = {n_total}"
)

# %% [markdown]
# ### Run allocation backtest for a single configuration
#
# Helper that builds a backtest spec and runs it. Returns (sharpe, elapsed)
# on success, or None on failure. Tracks MVO budget to drop slow allocators.


# %%
def run_alloc_backtest(pred_hash, source, top_k, alloc, predictions, state):
    """Run one allocation backtest and update sweep state counters."""
    alloc_name = alloc["method"]
    if state["skip_mvo"] and alloc_name in ("mvo", "mvo_ledoit_wolf"):
        return

    state["n_done"] += 1
    n_done = state["n_done"]

    spec = build_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        prices=prices,
        prediction_hash=pred_hash,
        initial_cash=bt_config.initial_cash,
        chapter="ch17",
        label=LABEL,
        signal={"method": "equal_weight_top_k", "top_k": top_k, "long_short": bt_config.long_short},
        allocation={**alloc, "top_k": top_k, "long_short": bt_config.long_short},
    )

    is_mvo = alloc_name in ("mvo", "mvo_ledoit_wolf")
    t0 = time.monotonic()
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
        elapsed = time.monotonic() - t0

        if is_mvo and not state["skip_mvo"]:
            projected = (time.monotonic() - state["sweep_start"]) + elapsed * (
                len(top_preds) * len(TOP_K_VALUES) - 1
            )
            if projected > BUDGET_SECONDS:
                print(f"    >> Dropping MVO — projected {projected / 60:.0f}m exceeds budget")
                state["skip_mvo"] = True

        sharpe = result.metrics.get("sharpe", 0)
        print(f"  [{n_done}/{n_total}] k={top_k} {source} × {alloc_name}: Sharpe={sharpe:.3f}")
    except Exception as e:
        state["n_failed"] += 1
        print(f"  [{n_done}/{n_total}] k={top_k} {source} × {alloc_name}: FAILED — {e}")


# %%
BUDGET_SECONDS = 3600
state = {"n_done": 0, "n_failed": 0, "skip_mvo": False, "sweep_start": time.monotonic()}

for top_k in TOP_K_VALUES:
    print(f"\n--- TOP_K = {top_k} ---")
    for pred_row in top_preds.iter_rows(named=True):
        pred_hash = pred_row["prediction_hash"]
        source = pred_row["source"]
        predictions = read_predictions(CASE_STUDY_ID, pred_hash)
        for alloc in ALLOC_CONFIGS:
            run_alloc_backtest(pred_hash, source, top_k, alloc, predictions, state)

elapsed_min = (time.monotonic() - state["sweep_start"]) / 60
print(
    f"\nSweep completed in {elapsed_min:.1f} minutes ({state['n_done']} done, {state['n_failed']} failed)"
)

# %% [markdown]
# ## 3. Allocation Analysis
#
# This section is **read-only** — it queries the registry via `BacktestExplorer`
# and can be re-run independently without re-running the sweep.
#
# Key question: does allocator choice change the outcome on the full universe,
# or do all allocators share the same fate? For high-turnover intraday strategies
# the holding period dominates the weighting scheme, so the spread between
# allocators is small relative to the gap between every-bar rebalancing and the
# slot mechanism — a point the cost notebook makes quantitative.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %% [markdown]
# ### Allocator Comparison
#
# Mean Sharpe by allocator across all TOP_K values and predictions. Every bar is
# deep in negative territory; the bars differ in degree, not sign. The ordering
# (score-weighted least bad, inverse-vol worst) is second-order to the shared
# cause — every-bar turnover across the full universe — confirming that
# allocation cannot rescue a cost-defeated trading frequency.

# %%
alloc_comparison = explorer.compare_allocators()
print(alloc_comparison)

# %%
import matplotlib.pyplot as plt

if not alloc_comparison.is_empty():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(alloc_comparison["allocator"].to_list(), alloc_comparison["avg_sharpe"].to_list())
    ax.set_xlabel("Average Sharpe")
    ax.set_title(f"{CASE_STUDY_ID}: Mean Sharpe by Allocator")
    fig.tight_layout()
    fig.show()

# %% [markdown]
# ### The strongest allocation combinations
#
# The highest-scoring combinations of prediction, position count and allocator.
# They pass to the cost notebook as the input surface for the rebalancing-cadence
# sweep, which varies how often the same ordering is traded. Nothing is selected
# here.

# %%
top10 = explorer.best(stage="allocation", top_n=10)
print(top10.select("source", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## Key Takeaways
#
# 1. **An allocator decides sizes, not turnover.** Every configuration in this
#    grid rebalances at the same times, so the differences between them are
#    differences in how capital is spread across positions. When the cost of
#    turning positions over dominates, redistributing weight across them cannot
#    recover it, and the allocator comparison is measuring a second-order effect.
#
# 2. **Covariance-based allocation needs an estimate the data may not support.**
#    Mean-variance and hierarchical risk parity require a correlation matrix
#    between assets. Estimated from short-horizon returns, that matrix is noisy,
#    and a weighting scheme built on a noisy correlation is not more principled
#    than an equal weighting - it is differently wrong and more expensive.
#
# 3. **Position count is a concentration decision.** Holding more names spreads
#    the same capital more thinly, which reduces the contribution of any single
#    position, correct or not, and increases the number of positions paying
#    costs. It moves the outcome without addressing what drives it.
#
# 4. **Allocation sits downstream of the cost problem.** Position sizing can only
#    distribute whatever the signal and the trading rule leave behind. The lever
#    that acts on the cost itself is how often the strategy trades, which is what
#    the cost notebook sweeps.
#
# **Known limitations**: Every result here is computed on the whole universe at
# one rebalancing frequency, so it describes allocation under those conditions
# only. Covariance-based allocators are excluded by default and their inclusion
# is a configuration change, not a change to the comparison being made.
