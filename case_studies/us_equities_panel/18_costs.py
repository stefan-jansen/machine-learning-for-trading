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
# # US Equities Panel: Costs
#
# **Chapter 18 — Transaction Costs and Execution**
#
# Cost sensitivity is a central diagnostic for the US equities panel, not
# a secondary one. Daily rebalancing across 3,200 stocks generates
# consistently high turnover, and the locked cost-sensitivity envelope is
# steep: zero-cost gross Sharpe is 3.98 (the mechanical gross-return
# ceiling), and Sharpe at the 10 bps post-decimalization midpoint is 2.10.
# The edge-to-cost ratio against the kill-condition floor of 1.2x is
# recorded as `evidence_partial` in the spine narrative_facts — the curve
# is flat enough at low frictions to clear the floor, but steep enough at
# realistic per-leg costs that execution quality is the binding operational
# constraint, not the gross signal.
#
# This notebook tests cost sensitivity on the **top allocation-stage combos**
# under two cost regimes:
#
# 1. **Bps regime (headline)** — total cost expressed in basis points of
#    notional traded. The case study's setup.yaml uses this convention with
#    an era-aware range (5-15 bps post-decimalization, 15-30 bps pre-2001).
# 2. **Per-share + spread regime (exploratory)** — IBKR Pro Tiered $0.0035
#    per share commission plus a flat half-spread in dollars per share. This
#    regime is presented for comparison: it imposes more realistic
#    nominal-dollar costs on liquid high-priced names, but applies current
#    dollars to historically split-adjusted prices and assumes the spread
#    floor is independent of liquidity. The two regimes disagree most for
#    low-priced and pre-decimalization names.
#
# Sections 1, 2a, and 2b generate cost-sensitivity backtests (write to
# registry). Section 3 queries the registry directly and plots both regimes
# side by side.
#
# **Learning Objectives:**
# 1. Run a cost grid sweep on top allocation combos to find breakeven
# 2. Compare two cost regimes (bps vs per-share+spread) and read the points
#    where they disagree
# 3. Identify the viable cost range and its implications for execution
#    requirements
#
# **Book Reference:** Chapter 18, Sections 18.2–18.5
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""US Equities Panel: Costs."""

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
    get_per_share_commission,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
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
COST_GRID_HALF_SPREAD_USD = get_cost_grid_half_spread_usd(CASE_STUDY_ID)
# IBKR Pro Tiered commission for the per-share companion regime. us_eq
# headline regime is bps; per-share is exploratory, so the YAML may not
# declare costs.per_share — get_per_share_commission falls back to the
# IBKR Pro Tiered top tier ($0.0035/share) as a documented default.
PER_SHARE_COMMISSION = get_per_share_commission(CASE_STUDY_ID)

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# The top allocation-stage combos are all GBM-prediction-based configurations.
# The cost sweep applies the same cost grid to each, producing decay curves
# that confirm whether the steep fragility is uniform across allocators or
# concentrated in specific sizing methods.

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
# For each top combo, re-run the backtest across `COST_GRID_BPS`.
#
# For the US equities panel, the shape of the cost-sensitivity envelope is
# convex and steep: small cost increases in the 0–20 bps range produce
# large Sharpe drops because daily rebalancing means every basis point of
# cost is incurred frequently. In the locked registry the zero-cost gross
# Sharpe is 3.98 and the 10 bps post-decimalization midpoint reads 2.10;
# the curve continues to fall toward and across zero in the high-cost end
# of the grid, so retail-style execution costs erode the alpha entirely
# while institutional execution at the lower end of the post-decimalization
# era retains a positive but narrowing margin.

# %%
n_total_bps = len(top_combos) * len(COST_GRID_BPS) if not top_combos.is_empty() else 0
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
                    f"  [{n_done}/{n_total_bps}] bps  {alloc_method} @ {cost_bps}bps: "
                    f"Sharpe={result.metrics.get('sharpe', 0):.3f}"
                )
        except Exception as e:
            print(f"  [{n_done}/{n_total_bps}] bps  {alloc_method} @ {cost_bps}bps: FAILED — {e}")

elapsed = time.time() - t0
print(f"\nBps sweep complete: {n_done} backtests in {elapsed:.0f}s")

# %% [markdown]
# ## 2b. Per-Share + Spread Regime — Exploratory Cost Sweep
#
# The bps regime treats friction as a fraction of notional, which understates
# the cost of low-priced names (where a fixed-dollar half-spread is many
# basis points) and overstates the cost of high-priced ones (where the same
# half-spread is sub-basis-point). The per-share + spread regime expresses
# friction in dollars per share — closer to how brokers quote and how
# microstructure works — but it has its own weakness: applying current
# dollars to historically split-adjusted prices conflates the adjustment
# factor with realized friction. For the US equities panel this is most
# acute pre-2001 (fractional ticks) and on names that have undergone large
# split adjustments.
#
# We walk a half-spread grid in cents per share at a fixed IBKR Pro Tiered
# commission ($0.0035/share). The point of this sweep is not to replace the
# bps headline but to expose where the two regimes disagree.
#
# **Uniform half-spread, not faithful per-row scaling of production
# costs.** This sweep passes a single `default_half_spread_usd` value to
# every backtest (no per-asset `asset_spreads` map). Read the curve as
# sensitivity to the universe-wide default, not as a re-pricing of the
# headline cost model.

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
# results across both regimes and renders them as paired panels.
#
# The bps panel anchors the headline cost-sensitivity envelope: gross
# Sharpe 3.98 at 0 bps falls to 2.10 at 10 bps and crosses zero in the
# upper grid. The per-share panel shows what happens when we re-price the
# same strategies in dollars per share at a fixed broker commission. The
# panels share Sharpe on the y-axis but use different x-axis units; the
# absolute decay curves should not be compared point-by-point because a
# 2.5¢ half-spread maps to ~12 bps half on a $20 stock and ~0.5 bps half
# on a $500 stock. Read each panel for its slope and breakeven, and read
# the *gap between regimes* as evidence of how much the headline cost
# convention matters for this universe.

# %%
import sqlite3

import matplotlib.pyplot as plt

REGISTRY_DB = CASE_DIR / "run_log" / "registry.db"


def load_cost_rows(commission_model: str) -> pl.DataFrame:
    """Load cost_sensitivity rows for a given commission model, computing the
    cost knob (bps for percentage, half-spread USD for per_share)."""
    conn = sqlite3.connect(str(REGISTRY_DB))
    df = pl.read_database(
        """
        SELECT
            b.spec_json,
            bm.sharpe,
            bm.max_drawdown,
            bm.num_trades
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
    ax_bps.set_title("Bps Regime (headline)")
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
    ax_ps.set_title("Per-Share + Spread Regime (exploratory)")
    ax_ps.legend(fontsize=8)
else:
    ax_ps.text(0.5, 0.5, "No per-share rows", ha="center", va="center", transform=ax_ps.transAxes)

fig.suptitle("US Equities Panel — Cost Sensitivity Across Two Regimes")
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Key Takeaways
#
# 1. The US equities panel has one of the steepest cost-sensitivity
#    envelopes in the book under the bps headline: gross Sharpe 3.98 at
#    0 bps falls to 2.10 at the 10 bps post-decimalization midpoint and
#    continues toward and through zero in the upper end of the cost grid.
#    The slope is what readers should anchor on, not the gross point.
# 2. The edge-to-cost ratio against the kill-condition floor of 1.2x is
#    recorded as `evidence_partial` in the spine narrative_facts: the
#    curve clears the floor at the lower end of the post-decimalization
#    cost range and fails it at the higher end. The envelope is therefore
#    consistent with institutional-quality execution and inconsistent with
#    retail-quality execution.
# 3. The per-share + spread regime is shown for comparison, not as an
#    alternative headline. Two assumptions limit it for this universe:
#    (a) it applies a single dollar half-spread to all symbols regardless
#    of liquidity (a $500 active name and a $25 thin name pay the same
#    flat cost per share), and (b) it applies current-dollar friction to
#    historically split-adjusted prices, which inflates the bps-equivalent
#    on names that have undergone large adjustments. The bps regime has
#    its own distortions but is closer to the era-aware spread ranges
#    documented in setup.yaml.
# 4. Daily rebalancing across the broad universe is the binding constraint.
#    The highest-IC GBM 1d signal IC of 0.023 [0.020, 0.026] clears zero on a
#    16-fold panel, but the delivery mechanism (daily turnover across
#    ~3,200 names) consumes most of the gross alpha under realistic
#    frictions in either cost regime.
# 5. This case study contrasts sharply with lower-frequency strategies in
#    the book: the same per-leg cost levels that clip monthly-rebalancing
#    Sharpe modestly reduce daily-rebalancing Sharpe materially. Cadence
#    is the dominant cost lever; position sizing is second-order.
#
# **Next**: The risk management notebook (Ch19) tests risk overlays on the top
# combos. Given the cost fragility, risk overlays that reduce turnover (time
# exits, drawdown breakers that pause trading) are the most relevant to evaluate.
