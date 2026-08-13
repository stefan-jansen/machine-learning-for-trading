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
# # US Equities Panel: Portfolio: Allocator Sweep
#
# **Chapter 17 — Portfolio Construction**
#
# The broadest universe in the book — 3,200 stocks — offers the most
# diversification potential of any case study, but daily rebalancing makes
# that potential difficult to exploit. Allocation method can in principle
# reduce turnover (covariance-aware methods rebalance less than score-weighted
# ones) or reduce drawdowns (risk-parity, HRP), but the dominant constraint
# here is cost: any Sharpe gain from smarter allocation is small relative to
# the gain from reducing rebalancing frequency.
#
# This notebook sweeps **top predictions × TOP_K concentration × 6 allocators**
# to find the best prediction+sizing combination before the cost stage.
#
# Sections 1–2 generate new allocation backtests (write to registry). Section 3
# queries the registry via `BacktestExplorer` and can be re-run independently.
#
# **Learning Objectives:**
# 1. Sweep top signal-stage predictions × concentration levels × allocation methods
# 2. Compare equal-weight, score-weighted, inverse-vol, risk-parity, MVO, and HRP
# 3. Show how portfolio concentration (TOP_K) interacts with allocation method
#    for a 3,200-stock universe where diversification is nearly unconstrained
#
# **Book Reference:** Chapter 17, Sections 17.2–17.8
#
# **Prerequisites:** Completed Ch16 backtest with results in `registry.db`.

# %%
"""US Equities Panel: Portfolio: Allocator Sweep."""

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
CASE_STUDY_ID = "us_equities_panel"
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
# ## 1. Load Top Predictions from Signal Stage
#
# The signal-stage ranking is GBM-dominated for this case study. We take the
# top signal-stage Sharpe predictions as the input candidates. All expected
# to be GBM configurations — the allocation sweep tests whether concentration
# adjustments or covariance-aware sizing can add incremental value on top of
# the GBM signal.

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
# differentiates these from signal-stage backtests.
#
# For the US equities panel, the TOP_K grid spans from concentrated to
# diversified portfolios. With 3,200 assets, even TOP_K = 100 is a 3% coverage
# ratio — far more diversified than narrower universes. The question is whether
# holding more positions (lower TOP_K %) reduces the drawdowns caused by
# individual stock adverse moves, or whether GBM signal quality is already
# strong enough that equal-weight concentration produces a higher Sharpe.

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
# For the US equities panel, the allocation analysis is primarily a secondary
# diagnostic. The dominant finding from this case study lives in Ch18: daily
# rebalancing cost fragility. Here we check whether any allocator materially
# reduces turnover (and therefore extends the cost tolerance window), or whether
# all methods produce similarly high-turnover strategies under daily cadence.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %% [markdown]
# ### Allocator Comparison
#
# Expected finding for this case study: allocator choice is second-order.
# The Sharpe spread across allocators will be narrow relative to the spread
# between the 0-bps gross Sharpe and the 50-bps near-breakeven Sharpe quantified
# in Ch18. Covariance-aware methods (HRP, MVO) may modestly reduce drawdowns
# but are unlikely to change the cost-fragility conclusion.

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
# These combinations — best prediction × concentration × allocator — form the
# input set for the cost sweep in Ch18. Note that all top combinations will
# reflect the same GBM prediction source; the variation here is concentration
# and sizing method, not model family.

# %%
top10 = explorer.best(stage="allocation", top_n=10)
print(top10.select("source", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## Key Takeaways
#
# 1. For the US equities panel, allocator choice is second-order: GBM signal
#    quality is the primary driver, and no allocation method materially changes the
#    risk-return profile relative to the cost sensitivity that follows in Ch18.
# 2. The 3,200-stock universe enables near-full diversification even at large
#    TOP_K values, but daily rebalancing across a broad universe generates
#    high turnover that offsets diversification benefits under realistic frictions.
# 3. Covariance-aware methods (HRP, risk-parity) may produce modestly smoother
#    equity curves, but their turnover characteristics under daily cadence are
#    not materially better than equal-weight for this case study.
# 4. The top allocation-stage combos all share the same GBM prediction source —
#    the spread across allocators is narrow, confirming that cost management
#    (Ch18) is the critical next step, not further allocation optimization.
#
# **Next**: The costs notebook (Ch18) sweeps the cost grid on the top combos
# and quantifies the breakeven level — the binding constraint for this strategy.
