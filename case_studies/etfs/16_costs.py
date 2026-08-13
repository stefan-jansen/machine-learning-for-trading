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
# # ETF Transaction Cost Analysis
#
# **Chapter 18 — Transaction Costs and Execution**
#
# ETF rotation at monthly frequency is the most cost-favorable configuration in this
# book's nine case studies. Monthly rebalancing means a fully new portfolio is
# constructed approximately twelve times per year, keeping turnover far lower than
# weekly or daily strategies. This notebook quantifies the cost advantage precisely:
# how wide is the gap between the strategy's gross allocation-stage Sharpe and its
# breakeven cost level, and where does the net Sharpe land at realistic ETF
# transaction costs (2–5 bps per leg for liquid US-listed ETFs)?
#
# **Purpose:** Sweep a cost grid on the top ETF allocation configurations to measure
# the Sharpe decay curve, identify the breakeven cost level, and confirm that monthly
# rebalancing makes this strategy viable at standard institutional execution costs.
#
# **Learning Objectives:**
# - Sweep a cost grid on the top allocation-stage configurations and register results
# - Plot the net Sharpe decay curve and identify the breakeven cost level for ETF rotation
# - Compare cost robustness across allocators and predict which combination preserves
#   the most Sharpe at realistic cost levels
#
# **Book Reference:** Chapter 18, Sections 18.2–18.5
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""ETF Transaction Cost Analysis."""

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
    set_backtest_costs_per_share,
    strategy_view,
)
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import (
    get_cost_grid_bps,
    get_cost_grid_half_spread_usd,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
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

COST_GRID_BPS = get_cost_grid_bps(CASE_STUDY_ID)
COST_GRID_HALF_SPREAD_USD = get_cost_grid_half_spread_usd(CASE_STUDY_ID)
# IBKR Pro Tiered commission for the per-share companion regime, sourced
# from setup.yaml::costs.per_share (single source of truth across notebooks
# and the planner). Unlike us_eq / sp500_eoa (where per_share is exploratory
# and tolerates omission via get_per_share_commission's default), per_share is
# the ETFs headline regime, so the key is mandatory: a missing key must fail
# loudly here rather than silently fall back to a default.
import yaml  # noqa: E402

with open(CASE_DIR / "config" / "setup.yaml") as _f:
    PER_SHARE_COMMISSION = float(yaml.safe_load(_f)["costs"]["per_share"])

print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}")

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# We select the top allocation-stage combinations by Sharpe as the baseline for cost
# testing. These represent the best realized performance before cost adjustment.
# The cost sweep will show how much of this Sharpe survives at different cost levels.

# %%
top_combos = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=TOP_N_COMBOS
)

if top_combos.is_empty():
    print("No allocation-stage results found. Run the portfolio management notebook first.")
else:
    for row in top_combos.iter_rows(named=True):
        spec = json.loads(row["spec_json"])
        alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
        print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique()} assets")

# %% [markdown]
# ## 2a. Bps Regime — Cost Grid Sweep
#
# The declared headline regime for ETFs is `per_share_plus_spread`
# (see `setup.yaml::costs.model` and the cost-regime-choice subsection in
# `01_feasibility_analysis.py`), but we run the bps grid alongside it as a
# regime comparison. The bps grid spans from near-zero (frictionless
# baseline) through levels that exceed realistic execution costs for
# US-listed ETFs. Liquid mega-ETFs (SPY, QQQ) trade at 1–2 bps spread;
# institutional-size blocks in less liquid ETFs reach 5–10 bps. Monthly
# rebalancing distributes each position's transaction cost over a full
# month of returns, so even 20–30 bps round-trip costs leave substantial
# headroom.
#
# **Uniform cost-sensitivity sweep, not faithful production cost
# structure.** This sweep walks a single flat bps rate across all assets;
# production backtests in the signal/allocation/risk_overlay stages use
# the tiered per-asset half-spread map from `setup.yaml::costs.asset_spreads`
# (SPY/QQQ/etc. at 0.5¢, sector XL* at 1¢, default 2¢). The cost-sensitivity
# sweep is *one-axis sensitivity to a uniform cost level*, not a faithful
# per-row scaling of the headline cost model.

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
print(f"\nBps sweep complete: {n_done} backtests in {elapsed:.0f}s")

# %% [markdown]
# ## 2b. Per-Share + Spread Regime — Headline-Aligned Cost Sweep
#
# The case study's headline cost model is `per_share_plus_spread`, so the
# per-share half-spread grid is the regime-aligned cost sensitivity for
# ETFs (the bps panel above is the cross-regime comparator). We walk
# `setup.yaml::backtest.sweep.cost_grid_half_spread_usd` (0¢, 0.5¢, 1¢,
# 2.5¢, 5¢, 10¢ half-spread per share) at a fixed IBKR Pro Tiered
# commission of $0.0035/share. As with §2a, this sweep uses a uniform
# default half-spread for all assets rather than the tiered per-asset map
# from production — read the curve as sensitivity to the universe-wide
# default, not as a re-pricing of the live cost structure.

# %%
n_total_ps = len(top_combos) * len(COST_GRID_HALF_SPREAD_USD) if not top_combos.is_empty() else 0
n_done_ps = 0
t1 = time.time()

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

    for half_spread_usd in COST_GRID_HALF_SPREAD_USD:
        n_done_ps += 1

        spec = set_backtest_costs_per_share(
            clone_backtest_spec(base_spec),
            per_share=PER_SHARE_COMMISSION,
            default_half_spread_usd=half_spread_usd,
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
            print(
                f"  [{n_done_ps}/{n_total_ps}] ps  {alloc_method} @ "
                f"{half_spread_usd * 100:.1f}¢ half-spread: "
                f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
            )
        except Exception as e:
            print(
                f"  [{n_done_ps}/{n_total_ps}] ps  {alloc_method} @ "
                f"{half_spread_usd * 100:.1f}¢: FAILED — {e}"
            )

elapsed_ps = time.time() - t1
print(f"\nPer-share sweep complete: {n_done_ps} backtests in {elapsed_ps:.0f}s")

# %% [markdown]
# ## 3. Cost Sensitivity Analysis — Two Regimes Side by Side
#
# This section is **read-only** — queries the registry for cost-sensitivity
# results across both regimes and renders them as paired panels. The
# per-share panel is the regime-aligned headline; the bps panel is the
# cross-regime comparator. The panels share the Sharpe y-axis but use
# different x-axis units; the absolute decay curves should not be compared
# point-by-point, because a 1¢ half-spread maps to ~1 bps half on a $500
# stock and ~5 bps half on a $20 stock. Read each panel for its slope and
# breakeven, and read the *gap between regimes* as evidence of how much
# the cost-model convention matters for this universe.

# %%
import sqlite3

import matplotlib.pyplot as plt

REGISTRY_DB = CASE_DIR / "run_log" / "registry.db"


def load_cost_rows(commission_model: str) -> pl.DataFrame:
    """Load cost_sensitivity rows for a given commission model.

    Returns the cost knob (bps for ``percentage``, USD half-spread for
    ``per_share``) plus realized Sharpe / max_drawdown / allocator.
    """
    conn = sqlite3.connect(str(REGISTRY_DB))
    df = pl.read_database(
        """
        SELECT
            b.spec_json,
            bm.sharpe,
            bm.max_drawdown
        FROM backtest_runs b
        JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
        WHERE b.stage = 'cost_sensitivity'
          AND bm.sharpe IS NOT NULL
          AND (bm.num_trades IS NULL OR bm.num_trades > 0)
        """,
        connection=conn,
    )
    conn.close()
    if df.is_empty():
        return df

    rows = []
    for spec_str, sharpe, max_dd in zip(
        df["spec_json"].to_list(),
        df["sharpe"].to_list(),
        df["max_drawdown"].to_list(),
        strict=False,
    ):
        spec = json.loads(spec_str)
        commission = spec.get("backtest_config", {}).get("commission", {})
        if commission.get("model") != commission_model:
            continue
        slippage = spec.get("backtest_config", {}).get("slippage", {})
        if commission_model == "percentage":
            cost_value = round(
                (commission.get("rate", 0.0) + slippage.get("rate", 0.0)) * 10_000.0, 4
            )
        else:  # per_share
            cost_value = round(slippage.get("spread", 0.0), 6)
        alloc = spec.get("strategy", {}).get("allocation", {}).get("method", "equal_weight")
        rows.append(
            {
                "cost_value": cost_value,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "allocator": alloc,
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("cost_value")


bps_df = load_cost_rows("percentage")
ps_df = load_cost_rows("per_share")

print(f"bps regime rows:        {len(bps_df):,}")
print(f"per-share regime rows:  {len(ps_df):,}")

# %%
fig, (ax_bps, ax_ps) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

if not bps_df.is_empty():
    for alloc in bps_df["allocator"].unique().sort().to_list():
        subset = bps_df.filter(pl.col("allocator") == alloc).sort("cost_value")
        ax_bps.plot(
            subset["cost_value"].to_list(),
            subset["sharpe"].to_list(),
            marker="o",
            label=alloc,
        )
    ax_bps.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_bps.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax_bps.set_xlabel("Total Cost (bps per leg)")
    ax_bps.set_ylabel("Net Sharpe Ratio")
    ax_bps.set_title("Bps Regime (cross-regime comparator)")
    ax_bps.legend(fontsize=8)
else:
    ax_bps.text(0.5, 0.5, "No bps rows", ha="center", va="center", transform=ax_bps.transAxes)

if not ps_df.is_empty():
    for alloc in ps_df["allocator"].unique().sort().to_list():
        subset = ps_df.filter(pl.col("allocator") == alloc).sort("cost_value")
        ax_ps.plot(
            [v * 100 for v in subset["cost_value"].to_list()],  # USD → cents
            subset["sharpe"].to_list(),
            marker="s",
            label=alloc,
        )
    ax_ps.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_ps.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax_ps.set_xlabel(f"Half-Spread (¢ per share, +${PER_SHARE_COMMISSION}/sh commission)")
    ax_ps.set_title("Per-Share + Spread Regime (headline)")
    ax_ps.legend(fontsize=8)
else:
    ax_ps.text(0.5, 0.5, "No per-share rows", ha="center", va="center", transform=ax_ps.transAxes)

fig.suptitle("ETF Rotation — Cost Sensitivity Across Two Regimes")
fig.tight_layout()
fig.show()

# %% [markdown]
# **Cost sensitivity interpretation.** The decay curve quantifies what the monthly
# cadence buys in cost tolerance. A daily strategy with the same gross Sharpe would
# incur transaction costs 20–25 times per month per position; a monthly strategy
# incurs them once. The result is that the cost threshold at which ETF rotation
# becomes unprofitable is an order of magnitude higher than for comparable daily
# strategies.
#
# The Sharpe reference lines at 0.0 and 0.5 show two interpretively meaningful
# thresholds: where the strategy turns unprofitable (breakeven), and where it falls
# below the Sharpe that justifies execution overhead for institutional strategies.
# For ETF rotation, both thresholds should appear well above 10 bps — the realistic
# upper bound for US-listed ETF execution costs at moderate size.
#
# Allocator differences in the cost curve are typically small for a monthly strategy,
# since cost exposure is proportional to position turnover, and all allocators
# rebalance on the same monthly calendar. A higher-concentration allocator (equal-weight
# top-k with small k) may have slightly lower turnover if fewer positions change between
# months; risk-parity and HRP may generate more turnover from weight rebalancing within
# a stable top-k selection.

# %% [markdown]
# ## Key Takeaways
#
# Monthly rebalancing is ETF rotation's structural cost advantage. With realistic
# ETF execution costs of 2–5 bps per leg, the net Sharpe decay from the
# allocation-stage baseline is minimal. This is in contrast to daily or weekly
# strategies where the same gross Sharpe would leave a much narrower margin above
# costs.
#
# The key metric from this notebook is the edge-to-cost ratio: how much Sharpe is
# available per basis point of cost incurred. For ETF rotation, this ratio is high
# because the monthly holding period amortizes transaction costs over a long return
# interval.
#
# This cost profile confirms that the strategy generates economically meaningful
# net returns after realistic execution costs, with a breakeven level that provides
# substantial margin against cost inflation.
#
# **Next:** The risk management notebook (Ch19) tests whether position-level risk
# overlays (stop-loss, trailing stop, time exit) improve the drawdown profile without
# eroding the Sharpe gains this cost analysis confirms.
