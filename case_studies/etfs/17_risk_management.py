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
# # ETF Risk Management: Engine-Level Risk Rules
#
# **Chapter 19 — Risk Management**
#
# The ETF rotation strategy has allocation-stage Sharpe above +0.7 and a cost
# breakeven well above realistic ETF execution costs. Risk management for this
# strategy is therefore an optimization problem — can overlays improve the drawdown
# profile without eroding the Sharpe that cost analysis confirmed?
# — rather than a rescue operation. Monthly holding periods create a specific challenge
# for position-level rules: a stop-loss or trailing stop that triggers mid-month forces
# an early exit and potentially leaves capital idle until the next rebalance, which may
# hurt performance more than the loss it prevented.
#
# **Purpose:** Test position-level risk controls — stop-loss, trailing stop, time exit —
# on the top ETF rotation configurations to determine which overlays improve the
# risk-adjusted profile of a monthly momentum strategy and which degrade it.
#
# **Learning Objectives:**
# - Apply position-level rules (stop-loss, trailing stop, time exit) to monthly ETF
#   positions and measure their effect on drawdown and Sharpe
# - Calibrate trailing stops via the MAE (maximum adverse excursion) distribution of
#   in-sample positions and contrast MAE-calibrated thresholds with fixed-percent stops
# - Interpret which risk overlay types are structurally compatible with monthly
#   rebalancing frequency and cross-asset momentum signals
#
# **Book Reference:** Chapter 19, Sections 19.3–19.6
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""ETF Risk Management: Engine-Level Risk Rules."""

import json
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    strategy_view,
)
from case_studies.utils.backtest_runner import precompute_weights, run_backtest
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import (
    calibrate_trailing_stops,
    get_portfolio_risk_controls,
    get_position_risk_controls,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
LABEL = ""
MAX_SYMBOLS = 0
MAX_RISK_VARIANTS = 0  # 0 = all; >0 limits position + portfolio controls each
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
if not LABEL:
    LABEL = bt_config.primary_label

from case_studies.utils.backtest_loaders import VECTORIZED_CASE_STUDIES

IS_VECTORIZED = CASE_STUDY_ID in VECTORIZED_CASE_STUDIES
MODE_LABEL = "vectorized" if IS_VECTORIZED else "engine"
print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}, mode: {MODE_LABEL}")

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# The risk sweep operates on the same top allocation-stage configurations used in the
# cost analysis. Each combo represents a (prediction, allocator, TOP_K) combination
# whose net Sharpe at realistic costs we already know. Risk overlays will be evaluated
# relative to these allocation-stage baselines.

# %%
top_combos = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=TOP_N_COMBOS
)

if top_combos.is_empty():
    msg = "No allocation-stage results found. Run the portfolio management notebook first."
    raise RuntimeError(msg)

# Summarize the top allocation-stage backtests we'll be re-using as the
# baseline for the risk overlay sweep below. Specs are normalized to v2 by
# ensure_backtest_spec() inside run_backtest(), so no pre-check needed here.
for row in top_combos.iter_rows(named=True):
    spec = json.loads(row["spec_json"])
    alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
    print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)

# %% [markdown]
# ### MAE/MFE-Calibrated Trailing Stops
#
# Trailing stop thresholds calibrated from Maximum Adverse Excursion (MAE) and
# Maximum Favorable Excursion (MFE) distributions set thresholds that reflect the
# actual intra-month volatility of ETF positions. Thresholds set too tight relative
# to typical intra-month noise will trigger frequently on normal price variation and
# produce excessive turnover. Thresholds set to the 5th percentile of MAE reflect
# only positions that moved unusually far against the momentum signal.

# %%
_position_grid = get_position_risk_controls(CASE_STUDY_ID)
if not IS_VECTORIZED and "close" in prices.columns:
    calibrated = calibrate_trailing_stops(prices)
    if calibrated:
        existing_thresholds = {rc.get("threshold", 0) for rc in _position_grid}
        new_calibrated = [c for c in calibrated if c["threshold"] not in existing_thresholds]
        position_controls = _position_grid + new_calibrated
        print(f"MAE/MFE calibration added {len(new_calibrated)} thresholds")
    else:
        position_controls = _position_grid
        print("MAE/MFE calibration returned no results; using standard grid")
else:
    position_controls = _position_grid
    print("Skipping MAE/MFE calibration (vectorized or no close column)")

portfolio_controls = get_portfolio_risk_controls(CASE_STUDY_ID)
if MAX_RISK_VARIANTS > 0:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
    portfolio_controls = portfolio_controls[:MAX_RISK_VARIANTS]
    print(f"Risk variants limited to {MAX_RISK_VARIANTS} each")

# %% [markdown]
# ## 2. Risk Overlay Sweep
#
# For each top combo, we run one backtest per risk control configuration — no
# baseline is needed here since the allocation-stage backtests already serve as the
# reference. Position-level rules execute inside the engine on each bar; portfolio-level
# limits check aggregate drawdown or loss thresholds and can pause or flatten the book.
#
# For a monthly ETF strategy, the structural risk question is whether position-level
# rules interact badly with the rebalancing cadence. A stop-loss that exits a position
# on day 15 of a 20-trading-day month leaves capital undeployed until month-end, at
# which point the next rebalance would have exited the position anyway. Whether the
# early exit helps or hurts depends on what the position does in the remaining days.

# %%
n_done = 0

for combo_idx, combo_row in enumerate(top_combos.iter_rows(named=True)):
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

    # Precompute allocation weights ONCE per combo — avoids re-running
    # expensive MVO/HRP for every risk variant (167s → 0s per variant)
    import time

    t0 = time.time()
    combo_weights = precompute_weights(
        predictions,
        base_spec,
        prices,
        label=LABEL,
        case_study=CASE_STUDY_ID,
        prediction_hash=pred_hash,
    )
    print(
        f"  Combo {combo_idx + 1}/{len(top_combos)}: {alloc_method} — "
        f"weights precomputed in {time.time() - t0:.0f}s"
    )

    # Position-level risk rules (engine only)
    if not IS_VECTORIZED:
        for rc in position_controls:
            spec_risk = clone_backtest_spec(base_spec)
            spec_risk["chapter"] = "ch19"
            if rc["type"] == "time_exit":
                spec_risk["strategy"]["risk"] = {
                    "name": rc["name"],
                    "position_rules": [{"type": rc["type"], "bars": rc["bars"]}],
                }
            else:
                spec_risk["strategy"]["risk"] = {
                    "name": rc["name"],
                    "position_rules": [{"type": rc["type"], "threshold": rc["threshold"]}],
                }

            try:
                result = run_backtest(
                    CASE_STUDY_ID,
                    pred_hash,
                    spec_risk,
                    prices=prices,
                    predictions=predictions,
                    label=LABEL,
                    register=True,
                    initial_cash=bt_config.initial_cash,
                    calendar=bt_config.calendar,
                    precomputed_weights=combo_weights,
                )
                n_done += 1
                print(
                    f"    {rc['name']}: Sharpe={result.metrics.get('sharpe', 0):.3f}, "
                    f"MaxDD={result.metrics.get('max_drawdown', 0):.2%}"
                )
            except Exception as e:
                print(f"    {rc['name']}: FAILED — {e}")

    # Portfolio-level risk limits
    for rc in portfolio_controls:
        spec_risk = clone_backtest_spec(base_spec)
        spec_risk["chapter"] = "ch19"
        spec_risk["strategy"]["risk"] = {
            "name": rc["name"],
            "portfolio_limits": [{"type": rc["type"], "threshold": rc["threshold"]}],
        }

        try:
            result = run_backtest(
                CASE_STUDY_ID,
                pred_hash,
                spec_risk,
                prices=prices,
                predictions=predictions,
                label=LABEL,
                register=True,
                initial_cash=bt_config.initial_cash,
                calendar=bt_config.calendar,
                precomputed_weights=combo_weights,
            )
            n_done += 1
            print(
                f"    {rc['name']}: Sharpe={result.metrics.get('sharpe', 0):.3f}, "
                f"MaxDD={result.metrics.get('max_drawdown', 0):.2%}"
            )
        except Exception as e:
            print(f"    {rc['name']}: FAILED — {e}")

print(f"\nRisk sweep complete: {n_done} backtests")

# %% [markdown]
# ## 3. Risk Impact Analysis
#
# This section is **read-only** — queries the registry for risk overlay results.
# For each overlay, `sharpe_delta` measures the change relative to the allocation-stage
# baseline (positive = improvement, negative = degradation). For a high-Sharpe monthly
# strategy, the threshold for an overlay to be worth adopting is a positive Sharpe delta
# with meaningful drawdown reduction — not just drawdown reduction at the cost of Sharpe.

# %%
from case_studies.utils.backtest_explorer import BacktestExplorer

explorer = BacktestExplorer(CASE_STUDY_ID)

# %%
risk_df = explorer.risk_impact()

if not risk_df.is_empty():
    # Best by risk type
    for risk_type in risk_df["risk_type"].unique().sort().to_list():
        subset = risk_df.filter(pl.col("risk_type") == risk_type).sort("sharpe", descending=True)
        best = subset.head(1)
        print(f"  Best {risk_type}: {best['risk_name'][0]} → Sharpe={best['sharpe'][0]:.3f}")

    print(f"\nAll risk overlays ({len(risk_df)}):")
    print(
        risk_df.select("risk_name", "risk_type", "sharpe", "max_drawdown", "sharpe_delta")
        .sort("sharpe", descending=True)
        .head(15)
    )
else:
    print("No risk overlay data in registry")

# %% [markdown]
# **Risk overlay interpretation.** Within the position-level overlay family, the
# MAE-calibrated trailing stops outperform fixed-percent stops by a clear margin.
# Calibrating the trailing threshold to the 25th percentile of the in-sample MAE
# distribution at a 20-bar horizon (`trailing_mae_p25_h20`) produces an overlay that
# fires only on positions that have moved unusually far from entry by historical
# standards, leaving typical-volatility positions to resolve at month-end. Stop-loss
# rules at the same percentage level are uniformly worse because they cap the upside
# distribution while still firing on the negative tail.
#
# The reason is structural. A position-level stop-loss that exits at a 5% drawdown
# triggers throughout the month, interrupting the monthly holding period before the
# rebalance resolves the position naturally. Tight trailing stops (2–3%) are
# particularly costly for monthly ETF positions because they fire on intra-month
# noise: an ETF with a 15% annualized volatility has an expected daily move of roughly
# 1%, so a 2% trailing stop triggers on roughly half of all positions within their
# first week. MAE-calibrated stops shift the threshold to a percentile of the
# historical adverse-excursion distribution rather than a fixed price level — they
# trigger only on positions whose drawdown exceeds the in-sample typical range.

# %% [markdown]
# ## Key Takeaways
#
# Risk overlays for ETF rotation at monthly frequency produce a clear pattern within
# the position-level family: MAE-calibrated trailing stops at a 20-bar horizon are
# the strongest single overlay; tight fixed-percent stops and short time exits hurt
# both Sharpe and drawdown. The monthly holding period creates a mismatch between
# the rebalancing cadence and the trigger frequency of tight stop rules — most are
# designed for daily or intraday strategies where stop-and-flip is a coherent
# response to adverse price action.
#
# The practical implication is that risk management for this strategy benefits from
# overlays that filter for the unusual rather than the merely uncomfortable. An
# MAE-calibrated trailing stop captures the rare positions that decay far past
# typical in-sample drawdown without aborting the more common positions whose
# month-long evolution is what produces the strategy's edge.
#
# With a cost breakeven well above realistic ETF friction and an MAE-calibrated
# trailing overlay as the primary risk control, the ETF rotation strategy has a
# favorable signal-to-risk profile for a monthly strategy.
#
# **Next:** Ch20 synthesis aggregates the full pipeline results — from model analysis
# through backtest, allocation, costs, and risk — across all nine case studies to
# identify cross-cutting patterns in what makes ML strategies implementable.
