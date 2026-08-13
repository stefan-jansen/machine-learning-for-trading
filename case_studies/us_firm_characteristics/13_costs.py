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
# # US Firm Characteristics: Costs
#
# **Chapter 18 — Transaction Costs and Execution**
#
# US firm characteristics has a favorable nominal cost profile along the
# protocol grid: monthly rebalancing runs at ~1/21 the turnover of daily
# strategies, and the roughly 3,700-stock validation universe keeps individual
# positions small. The cross-stage rank-1 is the equal-weight baseline
# `gbm/leaves_7_mse` at iteration 500 and TOP_K 50, Sharpe 2.63 [2.07, 3.24].
# This notebook sweeps the full registered cost grid from that winner.
#
# Sections 1–2 generate cost-sensitivity backtests (write to registry).
# Section 3 queries the registry via `BacktestExplorer` for analysis.
#
# **Learning Objectives:**
# 1. Run a cost grid sweep on the best baseline-or-allocation result
# 2. Compare net Sharpe decay across the registered cost range
# 3. Quantify what execution quality is required to deploy this strategy
#
# **Book Reference:** Chapter 18, Sections 18.2–18.5
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""US Firm Characteristics: Costs."""

import json
import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    set_backtest_costs_bps,
    strategy_view,
)
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import get_cost_grid_bps, get_top_n_predictions
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
LABEL = ""
MAX_SYMBOLS = 0
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "cost_sensitivity")
if not LABEL:
    LABEL = bt_config.primary_label

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")

COST_GRID_BPS = get_cost_grid_bps(CASE_STUDY_ID)

# %% [markdown]
# ## 1. Load the Best Pre-Cost Run
#
# Cost analysis starts from the top validation run across the equal-weight
# baseline and allocation stages. This preserves the established greedy funnel
# when an allocator does not improve on its baseline parent.


# %%
def _resolve_pre_cost_runs(case_study: str, label: str, *, split: str, top_n: int) -> pl.DataFrame:
    candidates = [
        resolve_best_backtest_runs(
            case_study,
            label,
            split=split,
            stage=stage,
            top_n=top_n,
        )
        for stage in ("signal", "allocation")
    ]
    candidates = [frame for frame in candidates if not frame.is_empty()]
    if not candidates:
        return pl.DataFrame()
    return (
        pl.concat(candidates)
        .sort("sharpe", descending=True)
        .unique("backtest_hash", maintain_order=True)
        .head(top_n)
    )


top_combos = _resolve_pre_cost_runs(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    top_n=TOP_N_COMBOS,
)

if top_combos.is_empty():
    print("No baseline or allocation results found. Run the upstream notebooks first.")
else:
    for row in top_combos.iter_rows(named=True):
        spec = json.loads(row["spec_json"])
        alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
        print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} assets")

# %% [markdown]
# ## 2. Cost Grid Sweep
#
# For each top combo, re-run the backtest at different cost levels. Because
# this strategy rebalances monthly, the per-period cost exposure is modest:
# a 30 bps cost on a monthly rebalance is equivalent to about 1.43 bps per
# trading day. The sweep tests whether Sharpe decays gradually rather than
# collapsing across the declared range.

# %%
n_total = len(top_combos) * len(COST_GRID_BPS) if not top_combos.is_empty() else 0
n_done = 0
t0 = time.time()

for combo_row in top_combos.iter_rows(named=True):
    pred_hash = combo_row["prediction_hash"]
    base_spec = ensure_backtest_spec(
        CASE_STUDY_ID,
        bt_config,
        json.loads(combo_row["spec_json"]),
        prices=prices,
        prediction_hash=pred_hash,
        initial_cash=bt_config.initial_cash,
    )
    alloc_method = strategy_view(base_spec).get("allocation", {}).get("method", "equal_weight")

    predictions = read_predictions(CASE_STUDY_ID, pred_hash)

    for cost_bps in COST_GRID_BPS:
        n_done += 1

        spec = set_backtest_costs_bps(
            clone_backtest_spec(base_spec),
            commission_bps=cost_bps / 2,
            slippage_bps=cost_bps / 2,
        )
        spec["chapter"] = "ch18"

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

            if cost_bps % 10 == 0:
                print(
                    f"  [{n_done}/{n_total}] {alloc_method} @ {cost_bps}bps: "
                    f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
                )
        except Exception as e:
            print(f"  [{n_done}/{n_total}] {alloc_method} @ {cost_bps}bps: FAILED — {e}")

elapsed = time.time() - t0
print(f"\nCost sweep complete: {n_done} backtests in {elapsed:.0f}s")

# %% [markdown]
# ## 3. Cost Sensitivity Analysis
#
# This section is **read-only** — queries the registry for cost-sensitivity
# results and computes breakeven levels.
#
# The current curve declines monotonically from Sharpe 2.684 at 0 bps to
# 2.559 at 30 bps and 2.476 at 50 bps. The registered grid therefore does not
# reach breakeven, while still making the cost sensitivity visible.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %%
cost_df = explorer.cost_sensitivity()

if not cost_df.is_empty():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for alloc in cost_df["allocator"].unique().sort().to_list():
        subset = cost_df.filter(pl.col("allocator") == alloc).sort("cost_bps")
        ax.plot(subset["cost_bps"].to_list(), subset["sharpe"].to_list(), marker="o", label=alloc)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Total Cost (bps per leg)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Sharpe Decay Under Transaction Costs")
    ax.legend()
    fig.tight_layout()
    fig.show()
else:
    print("No cost sensitivity data in registry")

# %% [markdown]
# ## Key Takeaways
#
# 1. The established funnel selects the best result across the equal-weight
#    baseline and allocation stages. Here that is the baseline, not the best
#    allocation row.
# 2. All 11 cost variants complete without failure, and Sharpe remains above
#    2.47 through the maximum registered cost of 50 bps.
# 3. The decline from 2.684 at 0 bps to 2.476 at 50 bps is gradual and
#    monotonic; the grid does not identify a breakeven point.
# 4. These are validation results. The cost sweep does not access or select on
#    the sealed holdout.
#
# **Next:** The risk management notebook (Ch19) tests whether risk overlays
# add value on top of the strong validation signal.
