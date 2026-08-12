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
# # US Firm Characteristics: Portfolio: Allocator Sweep
#
# **Chapter 17 — Portfolio Construction**
#
# The current validation price panel contains roughly 3,700 stocks. The
# equal-weight baseline sends ten top-ranked `fwd_ret_1m` GBM configurations
# into this allocation comparison. The leading baseline Sharpe is 2.63, so an
# allocator must improve an already diversified long-short portfolio.
#
# This notebook sweeps **10 predictions x 4 concentration levels x 2 allocators**
# to find the best prediction and sizing combination.
#
# Sections 1–2 generate new allocation backtests (write to registry). Section 3
# queries the registry via `BacktestExplorer` and can be re-run independently.
#
# **Learning Objectives:**
# 1. Sweep top equal-weight baseline predictions across concentration levels
# 2. Compare score-weighted and conformal-weighted allocation
# 3. Show how portfolio concentration (TOP_K) interacts with allocation method
#    in a deep stock universe
#
# **Book Reference:** Chapter 17, Sections 17.2–17.8
#
# **Prerequisites:** Completed Ch16 backtest with results in `registry.db`.

# %%
"""US Firm Characteristics: Portfolio: Allocator Sweep."""

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
CASE_STUDY_ID = "us_firm_characteristics"
LABEL = ""
MAX_SYMBOLS = 0
SKIP_EXPENSIVE_ALLOC = False
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
# ## 1. Load Top Predictions from the Equal-Weight Baseline
#
# The top predictions are sorted by equal-weight baseline Sharpe for the primary
# `fwd_ret_1m` label. All ten advancing configurations are GBM variants, led by
# `leaves_7_mse` at validation Sharpe 2.63.

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
# For each (prediction × TOP_K × allocator), call `run_backtest()` with the
# allocation config added to the strategy spec. The spec hash automatically
# differentiates these from the equal-weight baseline backtests.
#
# With roughly 3,700 stocks, the TOP_K grid explores meaningful concentration levels:
# small TOP_K is highly concentrated in the strongest signals; large TOP_K
# diversifies broadly but includes weaker predictions. The two configured
# alternatives test score magnitude and walk-forward conformal uncertainty;
# moment-based allocators are not part of this current surface.

# %%
TOP_K_VALUES = get_top_k_values_for(CASE_STUDY_ID, LABEL, n_assets)
print(f"TOP_K grid: {TOP_K_VALUES} (universe: {n_assets} assets)")

_ALL_ALLOC_CONFIGS = get_allocators(CASE_STUDY_ID)

_EXPENSIVE = {"mvo_ledoit_wolf", "hrp"}
if SKIP_EXPENSIVE_ALLOC:
    ALLOC_CONFIGS = [a for a in _ALL_ALLOC_CONFIGS if a["method"] not in _EXPENSIVE]
    print(f"Skipping expensive allocators: {', '.join(sorted(_EXPENSIVE))}")
else:
    ALLOC_CONFIGS = _ALL_ALLOC_CONFIGS

n_total = len(top_preds) * len(TOP_K_VALUES) * len(ALLOC_CONFIGS)
print(
    f"Total backtests: {len(top_preds)} preds × {len(TOP_K_VALUES)} top_k × "
    f"{len(ALLOC_CONFIGS)} allocs = {n_total}"
)

# %%
n_done = 0
n_failed = 0
skip_expensive = SKIP_EXPENSIVE_ALLOC
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

            if skip_expensive and alloc_name in _EXPENSIVE:
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

                if is_mvo and not skip_expensive:
                    n_mvo_remaining = len(top_preds) * len(TOP_K_VALUES) - 1
                    total_projected = (time.monotonic() - sweep_start) + elapsed * n_mvo_remaining
                    if total_projected > BUDGET_SECONDS:
                        print(
                            f"    >> Dropping expensive allocators — projected "
                            f"{total_projected / 60:.0f}m exceeds budget"
                        )
                        skip_expensive = True

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
# and can be re-run independently without re-running the sweep.
#
# The key question: does portfolio construction add value on top of the
# validation signal? The baseline leader enters at Sharpe 2.63. The best
# allocation result, conformal weighting at TOP_K 50, reaches 2.59 and therefore
# does not overtake the equal-weight baseline. Allocation choice is secondary
# to the prediction and concentration choices on this surface.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %% [markdown]
# ### Allocator Comparison
#
# Average Sharpe by allocator across all TOP_K levels and prediction sources.
# Conformal weighting averages Sharpe 1.866 across its 40 runs, while score
# weighting averages 1.855. Their best results are 2.592 and 2.572, respectively,
# so the aggregate difference is small.

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
# Best (prediction × TOP_K × allocator) triplets by allocation-stage Sharpe.
# The combination that passes to Ch18 for cost sensitivity testing should have
# the best Sharpe, low drawdown, and a TOP_K that reflects the natural
# diversification capacity of the roughly 3,700-stock validation universe.

# %%
top10 = explorer.best(stage="allocation", top_n=10)
print(top10.select("source", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## Key Takeaways
#
# 1. All 80 current allocation combinations complete without failure.
# 2. TOP_K 50 produces the leading allocation results for both configured
#    allocators, so broader diversification helps this monthly panel.
# 3. Conformal weighting leads the allocation stage with `gbm/leaves_7_mse`
#    at iteration 500 and TOP_K 50: Sharpe 2.592 [2.105, 3.212].
# 4. The allocation leader remains below its equal-weight baseline Sharpe of
#    2.632. The cross-stage rank-1 therefore remains the baseline result.
#
# **Next:** The costs notebook (Ch18) tests how durable this Sharpe is under
# realistic transaction costs for the allocation-stage leader.
