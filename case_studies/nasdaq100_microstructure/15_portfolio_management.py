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
# Covariance-based allocators (MVO, HRP, risk-parity) must estimate
# short-horizon correlations from 15-minute returns — a noisier signal than
# daily, and prohibitively slow over 1.3M bars — so they are skipped by default
# for this case study. The comparison that matters here is not which allocator
# wins but how far below zero all of them sit: every-bar rebalancing across the
# full universe pays turnover cost no weighting scheme can offset.

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
# ### Top 10 Combinations
#
# Least-negative (prediction × TOP_K × allocator) triples by allocation-stage
# Sharpe. Even the best of these is well below zero — they pass to the cost
# notebook (Ch18) not as viable strategies but as the input surface for the
# cadence sweep, which shows how much of the gap closes when trade frequency
# drops.

# %%
top10 = explorer.best(stage="allocation", top_n=10)
print(top10.select("source", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## Key Takeaways
#
# 1. On the full universe at 15-minute cadence, every allocator is deeply
#    negative. The spread between best and worst allocator is second-order to
#    the spread between every-bar rebalancing and the slot mechanism: weighting
#    scheme is the wrong lever when turnover is the binding constraint.
# 2. Covariance-based allocators (MVO, HRP) are skipped by default: estimating
#    correlations from 1.3M 15-minute bars is prohibitively slow and noisier
#    than daily. Set `SKIP_EXPENSIVE_ALLOC = False` to include them — they do
#    not change the conclusion.
# 3. Larger TOP_K spreads cost drag across more positions but cannot offset
#    every-bar turnover; concentration is a marginal adjustment within a
#    uniformly loss-making regime.
# 4. The portfolio-construction lesson here is diagnostic: allocation sits
#    downstream of the cost problem. The cost-feasible carrier (Ch16 §4) solves
#    that problem upstream, at the signal stage, by controlling turnover through
#    the slot mechanism rather than through position weights.
#
# **Next**: The costs notebook (Ch18) compares the full universe against the
# cost-feasible screen and sweeps cadence × per-share cost to locate the viable
# implementation regime.
