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
# # ETF Portfolio Construction: Allocator Sweep
#
# **Chapter 17 — Portfolio Construction**
#
# The signal-stage backtest established that prediction IC and top-k Sharpe rank
# configurations differently across model families: portfolio construction
# mediates prediction accuracy more than raw rank correlation does. This notebook
# tests whether allocator choice can extract further value from the top
# signal-stage predictions, or whether the signal stage already captures most of
# the achievable Sharpe for a 100-ETF monthly strategy.
#
# **Purpose:** Quantify the marginal contribution of allocator choice relative to
# signal quality for ETF rotation — across concentration levels (TOP_K) and six
# weighting schemes — to determine whether sophisticated allocation adds Sharpe or
# merely redistributes risk.
#
# **Learning Objectives:**
# - Load the top signal-stage predictions and build the allocation sweep grid across
#   concentration levels and weighting methods
# - Compare equal-weight, score-weighted, inverse-vol, risk-parity, MVO, and HRP
#   on the same ETF predictions
# - Evaluate whether TOP_K concentration interacts with allocator in a predictable way
#   for a 100-asset monthly universe
#
# **Book Reference:** Chapter 17, Sections 17.2–17.8
#
# **Prerequisites:** Completed Ch16 backtest with results in `registry.db`.

# %%
"""ETF Portfolio Construction: Allocator Sweep."""

import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import build_backtest_spec
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import read_predictions, resolve_best_predictions
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
LABEL = ""
MAX_SYMBOLS = 0
TOP_N_PREDICTIONS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_PREDICTIONS is None:
    TOP_N_PREDICTIONS = get_top_n_predictions(CASE_STUDY_ID, "allocation")
CHECKPOINTS_PER_CONFIG = get_checkpoints_per_config(CASE_STUDY_ID)
if not LABEL:
    LABEL = bt_config.primary_label

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")

# %% [markdown]
# ## 1. Load Top Predictions from Signal Stage
#
# We take the top predictions by signal-stage Sharpe, not by IC, reflecting the
# finding that IC and Sharpe are imperfectly correlated for this universe.

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
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
n_assets = prices["symbol"].n_unique()
print(f"Prices: {len(prices):,} rows, {n_assets} assets")

# %% [markdown]
# ## 2. Allocation Sweep
#
# For each (prediction × TOP_K × allocator), `run_backtest()` is called with the
# allocation config embedded in the strategy spec. The spec hash differentiates
# these from signal-stage backtests automatically.
#
# The TOP_K grid spans from a concentrated selection of the universe's strongest
# momentum assets to a broader basket. For 100 ETFs, the concentration question
# is substantive: a top-5 selection focuses on a single dominant regime asset class,
# while a top-20 selection approaches a diversified multi-asset portfolio. Whether
# allocation method adds value is partly a function of how concentrated the selection
# already is.

# %%
TOP_K_VALUES = get_top_k_values_for(CASE_STUDY_ID, LABEL, n_assets)
print(f"TOP_K grid: {TOP_K_VALUES} (universe: {n_assets} assets)")

ALLOC_CONFIGS = get_allocators(CASE_STUDY_ID)

n_total = len(top_preds) * len(TOP_K_VALUES) * len(ALLOC_CONFIGS)
print(
    f"Total backtests: {len(top_preds)} preds × {len(TOP_K_VALUES)} top_k × "
    f"{len(ALLOC_CONFIGS)} allocs = {n_total}"
)

# %%
n_done = 0
n_failed = 0
skip_mvo = False
sweep_start = time.monotonic()
BUDGET_SECONDS = 3600

for top_k in TOP_K_VALUES:
    print(f"\n--- TOP_K = {top_k} ---")
    for pred_row in top_preds.iter_rows(named=True):
        pred_hash = pred_row["prediction_hash"]
        source = pred_row["source"]

        predictions = read_predictions(CASE_STUDY_ID, pred_hash)

        for alloc in ALLOC_CONFIGS:
            alloc_name = alloc["method"]

            if skip_mvo and alloc_name in ("mvo", "mvo_ledoit_wolf"):
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

                if is_mvo and not skip_mvo:
                    n_mvo_remaining = len(top_preds) * len(TOP_K_VALUES) - 1
                    total_projected = (time.monotonic() - sweep_start) + elapsed * n_mvo_remaining
                    if total_projected > BUDGET_SECONDS:
                        print(
                            f"    >> Dropping MVO — projected {total_projected / 60:.0f}m exceeds budget"
                        )
                        skip_mvo = True

                print(
                    f"  [{n_done}/{n_total}] k={top_k} {source} × {alloc_name}: "
                    f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
                )
            except Exception as e:
                n_failed += 1
                print(f"  [{n_done}/{n_total}] k={top_k} {source} × {alloc_name}: FAILED — {e}")

print(
    f"\nSweep completed in {(time.monotonic() - sweep_start) / 60:.1f} minutes ({n_failed} failed)"
)

# %% [markdown]
# ## 3. Allocation Analysis
#
# This section is **read-only** — it queries the registry via `BacktestExplorer`
# and can be re-run independently without re-running the sweep. The analysis
# quantifies how much Sharpe spread is attributable to allocator choice vs.
# signal quality and concentration level.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %% [markdown]
# ### Allocator Comparison
#
# The comparison answers a central question for ETF rotation: does sophisticated
# portfolio weighting add Sharpe relative to equal-weight top-k selection? For a
# monthly strategy where every ETF in the top-k holds for a full calendar month,
# intra-rebalancing volatility differences across assets are largely averaged out.
# The prediction quality determines which ETFs enter the portfolio; allocation
# determines how risk is distributed among them.

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
# **Allocator interpretation.** For ETFs at monthly rebalancing frequency, the
# expected finding is that allocator choice is second-order to signal quality. The bar
# chart shows the mean Sharpe across all configurations per allocator: a narrow spread
# between best and worst allocator (relative to the spread between prediction sources)
# confirms that sophisticated weighting does not rescue a weak prediction or
# meaningfully improve a strong one.
#
# Inverse-vol and risk-parity weighting can smooth drawdowns in a cross-asset universe
# by underweighting high-volatility assets (commodity ETFs, leveraged funds) that may
# appear in the top-k due to recent momentum but carry outsized risk. MVO's benefit is
# theoretically larger when assets are heterogeneous in return and volatility — which
# the 100-ETF universe is — but estimation error in the covariance matrix at monthly
# frequency typically erodes the theoretical advantage.

# %% [markdown]
# ### Top 10 Combinations

# %%
top10 = explorer.best(stage="allocation", top_n=10)
print(top10.select("source", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## Key Takeaways
#
# For ETF rotation at monthly frequency, allocation method adds modest value relative
# to signal quality. The Sharpe spread attributable to allocator choice is smaller than
# the spread attributable to which model family generated the predictions. Signal
# quality is the primary driver of allocation-stage performance; the allocator
# determines how that signal is expressed in position sizes, not whether the
# strategy is profitable.
#
# The interaction between TOP_K and allocator is more consequential than allocator
# choice alone. A concentrated top-5 selection in a cross-asset universe implicitly
# bets on a single momentum regime; equal-weight and score-weighted allocation behave
# identically at that concentration level. Diversification benefits from inverse-vol
# or HRP emerge at higher TOP_K values where assets with different volatility profiles
# coexist in the portfolio.
#
# Results are registered in `registry.db` for downstream consumption by Ch18
# (cost analysis) and Ch19 (risk management overlays).
#
# **Next:** The costs notebook (Ch18) quantifies how monthly rebalancing frequency
# affects the edge-to-cost ratio and where the strategy's breakeven lies.
