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
# # US Firm Characteristics: Allocator Sweep
#
# **Chapter 17 - Portfolio Construction**
#
# The backtest notebook weighted every selected name equally. That is a choice, not
# an absence of one: equal weighting throws away the model's own ranking inside the
# selected set, on the argument that the ranking is too noisy to size with. This
# notebook tests that argument by sizing positions three ways and comparing what
# each earned.
#
# An **allocator** turns a set of selected names into weights. Two are declared for
# this case study. **Score weighting** sizes each position by the model's own
# predicted score, so a name the model is more confident about gets more capital.
# **Conformal weighting** sizes by an interval rather than a point estimate: it
# calibrates, on data the model did not fit, how wide each prediction's error
# distribution is, and gives less capital to names whose predictions have been less
# reliable. The two disagree exactly where a large score comes with a wide interval.
#
# The sweep crosses those allocators with the concentration levels the backtest
# notebook already swept, so the comparison holds concentration fixed within each
# cell rather than confounding it with the sizing rule.
#
# Sections 1-2 write allocation backtests to the registry. Section 3 is read-only:
# it queries the registry through `BacktestExplorer` and can be re-run without
# re-running the sweep.
#
# **Book Reference:** Chapter 17, Sections 17.2-17.8
#
# **Prerequisites:** the Chapter 16 backtest notebook, whose registered baselines
# decide which predictions advance to here.

# %%
"""US Firm Characteristics: Portfolio: Allocator Sweep."""

import sqlite3
import time
import warnings
from collections import Counter

import polars as pl

from utils.style import COLORS, add_message_title

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
# Wall-clock the allocation sweep may spend before the covariance-estimating
# allocators are dropped from the remainder. Declared here rather than buried beside
# the loop, because it decides which runs the notebook publishes.
BUDGET_SECONDS = 3600

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
# ## 1. Which predictions advance
#
# Allocation is swept over the predictions that ranked highest at the equal-weight
# baseline rather than over all of them, because the sweep is multiplicative:
# every prediction carried forward is multiplied by the concentration grid and
# again by the allocator menu. That selection is itself a source of the overfitting
# the strategy analysis notebook has to correct for, and it is recorded here as a
# step rather than treated as neutral.

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
# Each cell of the grid is one prediction, one concentration and one allocator, run
# through the same `run_backtest()` the baseline used. The allocation config enters
# the strategy spec, so the spec hash separates these runs from the equal-weight
# baselines rather than overwriting them, and both stages stay readable side by side.
#
# The concentration grid is printed below with the universe it was resolved against.
# A small value concentrates capital in the names the model ranked highest; a large
# one spreads it across names the model ranked lower, so the grid trades conviction
# against diversification.
#
# The declared menu is score weighting and conformal weighting. The moment-based
# allocators - mean-variance optimisation and hierarchical risk parity - are absent
# because they estimate a covariance matrix from a rolling window, and this panel
# rebalances monthly over a window too short for that estimate to exist. `setup.yaml`
# records that reasoning next to the menu.

# %%
TOP_K_VALUES = get_top_k_values_for(CASE_STUDY_ID, LABEL, n_assets)
print(f"TOP_K grid: {TOP_K_VALUES} (universe: {n_assets} assets)")

_ALL_ALLOC_CONFIGS = get_allocators(CASE_STUDY_ID)

# One definition of expensive, used by both the skip flag and the budget check. It
# had been two that disagreed: the skip set was {mvo_ledoit_wolf, hrp} while the
# budget check fired on {mvo, mvo_ledoit_wolf}, so `hrp` could never trip the budget
# and plain `mvo` could trip it and then not be skipped. What makes an allocator
# expensive here is estimating a covariance matrix per rebalance, which is what these
# three do and what score and conformal weighting do not.
_EXPENSIVE = {"mvo", "mvo_ledoit_wolf", "hrp"}
if SKIP_EXPENSIVE_ALLOC:
    ALLOC_CONFIGS = [a for a in _ALL_ALLOC_CONFIGS if a["method"] not in _EXPENSIVE]
    print(f"Skipping expensive allocators: {', '.join(sorted(_EXPENSIVE))}")
else:
    ALLOC_CONFIGS = _ALL_ALLOC_CONFIGS

n_total = len(top_preds) * len(TOP_K_VALUES) * len(ALLOC_CONFIGS)
print(
    f"Total backtests: {len(top_preds)} preds x {len(TOP_K_VALUES)} top_k x "
    f"{len(ALLOC_CONFIGS)} allocs = {n_total}"
)

# %%
n_done = 0
n_failed = 0
failures: Counter[str] = Counter()
skip_expensive = SKIP_EXPENSIVE_ALLOC
sweep_start = time.monotonic()
n_expensive_total = (
    len(top_preds) * len(TOP_K_VALUES) * sum(1 for a in ALLOC_CONFIGS if a["method"] in _EXPENSIVE)
)
n_expensive_done = 0

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

                # Project from what is actually left, not from a constant. This counted
                # (prediction, top_k) pairs and recomputed the same figure after every
                # expensive run, so it ignored how many had already finished and
                # undercounted by the number of expensive allocators in the menu.
                if alloc_name in _EXPENSIVE and not skip_expensive:
                    n_expensive_done += 1
                    remaining = n_expensive_total - n_expensive_done
                    projected = (time.monotonic() - sweep_start) + elapsed * remaining
                    if projected > BUDGET_SECONDS:
                        print(
                            f"    >> dropping the expensive allocators: {remaining} left at "
                            f"{elapsed:.1f}s each projects {projected / 60:.0f}m, over budget"
                        )
                        skip_expensive = True

                print(
                    f"  [{n_done}/{n_total}] k={top_k} {source} x {alloc_name}: "
                    f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
                )
            except Exception as error:
                n_failed += 1
                failures[f"{type(error).__name__}: {error}"] += 1
                print(
                    f"  [{n_done}/{n_total}] k={top_k} {source} x {alloc_name}: "
                    f"FAILED - {type(error).__name__}: {error}"
                )

print(
    f"\nSweep completed in {(time.monotonic() - sweep_start) / 60:.1f} minutes ({n_failed} failed)"
)
for reason, count in failures.most_common():
    print(f"  {count:>4} x {reason[:150]}")

# %% [markdown]
# ## 3. Allocation Analysis
#
# This section is **read-only**: it queries the registry through `BacktestExplorer`
# and can be re-run without re-running the sweep.
#
# The question the sweep was run to answer is whether sizing by the model's own
# scores earns more than sizing every selected name equally. There are two ways for
# the answer to be no, and they mean different things. If the allocators land close
# to the baseline, the ranking inside the selected set carries little information
# beyond membership, and equal weighting was the right default. If they land well
# below it, sizing by score actively concentrates capital into the noisiest part of
# the ranking.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %% [markdown]
# ### By allocator
#
# Sharpe averaged over every concentration level and prediction, one row per
# allocator. Averaging is the point: a single high cell says which combination
# happened to land highest, and the average says whether the sizing rule helped
# across the grid. A difference between the two rows that is small next to the
# spread within either row is not evidence that one rule beat the other.

# %% tags=["results"]
alloc_comparison = explorer.compare_allocators()
print(alloc_comparison)

# %%
import matplotlib.pyplot as plt

if not alloc_comparison.is_empty():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        alloc_comparison["allocator"].to_list(),
        alloc_comparison["avg_sharpe"].to_list(),
        color=COLORS["blue"],
    )
    ax.set_xlabel("Sharpe, averaged over concentration and prediction")
    add_message_title(
        ax,
        "Sizing rule moves the average less than the grid around it",
        subtitle="Validation months only, before trading costs",
    )
    fig.tight_layout()
    fig.show()

# %% [markdown]
# ### The upper tail of the allocation grid
#
# The ten highest allocation-stage Sharpes, with the concentration and allocator
# behind each. Read it against the table above rather than on its own: this is the
# tail of a grid, so the top row is the largest of eighty draws and is inflated by
# that count. What is worth reading here is whether one allocator or one
# concentration fills the tail, which would be a pattern, or whether the tail is
# mixed, which would say the grid found no reliable ordering.

# %% tags=["results"]
top10 = explorer.best(stage="allocation", top_n=10)

# `best` reports `signal.method`, which is one string across the whole grid, and
# drops both the allocator and the concentration - the two dimensions this notebook
# varies. Both are in the same spec, so they are joined back here. Shared-code fix
# filed as ml4t/agent-workspace#910.
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as conn:
    grid = (
        pl.DataFrame(
            conn.execute(
                "SELECT backtest_hash, spec_json FROM backtest_runs WHERE stage = 'allocation'"
            ).fetchall(),
            schema=["backtest_hash", "spec_json"],
            orient="row",
        )
        .with_columns(
            allocator=pl.col("spec_json").str.json_path_match("$.strategy.allocation.method"),
            names_per_side=pl.col("spec_json")
            .str.json_path_match("$.strategy.signal.top_k")
            .cast(pl.Int64),
        )
        .drop("spec_json")
    )

top10 = top10.join(grid, on="backtest_hash", how="left")
print(top10.select("source", "allocator", "names_per_side", "sharpe", "cagr", "max_drawdown"))

# %% [markdown]
# ## What this notebook establishes, and what it does not
#
# The sweep answers a narrow question: holding the prediction and the concentration
# fixed, does sizing positions by the model's score or by a conformal interval earn
# more than sizing them equally? The comparison is clean in the sense that every
# allocator saw the same selected names at the same concentration.
#
# It is not clean in a second sense, and that carries forward. The predictions that
# entered were the ones that ranked highest at the equal-weight baseline, so this
# stage inherits that selection and adds a grid of its own on top. Both counts feed
# the trial count the strategy analysis notebook has to deflate by, and neither
# number is visible in a Sharpe read off the table above.
#
# Nothing here has been charged trading costs. That matters more for this stage than
# for the last one, because an allocator that changes weights every month trades more
# than one that does not, and the sweep does not price that difference. The costs
# notebook does, which is why it comes next rather than after a decision has been
# taken here.
#
# **Next:** the costs notebook charges the sweep at realistic commission and
# slippage, and asks which of these results survive it.
