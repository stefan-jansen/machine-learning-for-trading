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
# # NASDAQ-100 Microstructure: Risk Controls
#
# **Chapter 19 — Risk Management**
#
# This notebook tests **position-level risk controls** on the top
# allocation-stage combos via the ml4t-backtest engine with 15-minute OHLCV
# bars. Three rule families are swept: stop-loss, trailing stop (including
# MAE-calibrated variants), and time-exit. Each rule exits an individual
# position without halting the rest of the book, so the strategy continues
# trading after a single name's exit.
#
# Portfolio-level kill switches (max-drawdown breaker, daily-loss limit) are
# treated as governance instruments in Ch19 §19.8 and are NOT swept as
# selectable hyperparameters here. Their permanent-halt semantics produce
# zero-std Sharpe artifacts in ranking — see the engine library's
# `MaxDrawdownLimit` / `DailyLossLimit` classes (still available for
# governance use) and the §19.8 demo notebook.
#
# Sections 1–2 generate risk-overlay backtests (write to registry).
# Section 3 queries the registry via `BacktestExplorer` for analysis.
#
# **Learning Objectives:**
# 1. Apply position-level controls (stop-loss, trailing stop, time exit) in
#    the engine backtest context
# 2. Measure how each overlay modifies the equity curve and drawdown profile
# 3. Identify threshold values that improve risk-adjusted returns without
#    excessively reducing time in market
#
# **Book Reference:** Chapter 19, Sections 19.3–19.6
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""NASDAQ-100 Microstructure: Risk Controls."""

import json
import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    warmup_periods_for,
)
from case_studies.utils.backtest_presets import (
    clone_backtest_spec,
    ensure_backtest_spec,
    strategy_view,
)
from case_studies.utils.backtest_runner import precompute_weights, run_backtest
from case_studies.utils.notebook_contracts import excluded_families
from case_studies.utils.registry import read_predictions, resolve_best_backtest_runs
from case_studies.utils.sweep_config import (
    calibrate_trailing_stops,
    get_portfolio_risk_controls,
    get_position_risk_controls,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
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
if excluded_families(CASE_STUDY_ID):
    print(
        "Active-model filter: excluding "
        f"{', '.join(sorted(excluded_families(CASE_STUDY_ID)))} pending corrected reruns"
    )

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# We load the best full-universe slot-mechanism backtests as the baseline.
# These reach positive validation Sharpe on the full universe — but, as Act 1
# established (Ch16 §3), the full-universe slot winners are a validation
# coin-flip that collapses out of sample, so a high in-sample Sharpe here is a
# selection artifact, not a tradeable result. The point of this notebook is the
# *mechanics* of position-level overlays — how stop-loss, trailing-stop, and
# time-exit rules reshape the drawdown profile — not to crown a configuration.
# The case study's tradeable strategy is the cost-feasible carrier (Ch16 §4),
# carried into the synthesis chapter.

# %%
top_combos = resolve_best_backtest_runs(
    CASE_STUDY_ID, LABEL, split="validation", stage="allocation", top_n=TOP_N_COMBOS
)

if top_combos.is_empty():
    msg = "No allocation-stage results found. Run the portfolio management notebook first."
    raise RuntimeError(msg)

for row in top_combos.iter_rows(named=True):
    spec = json.loads(row["spec_json"])
    alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
    print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    warmup_periods=warmup_periods_for(CASE_STUDY_ID),
    max_symbols=MAX_SYMBOLS,
)

# %% [markdown]
# ### MAE/MFE-Calibrated Trailing Stops
#
# Trailing stops are calibrated from historical maximum adverse excursion (MAE)
# and maximum favorable excursion (MFE) distributions. The engine supports
# per-position stop mechanisms, so calibrated thresholds are applied directly
# in the risk overlay sweep below.

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
    print("Skipping MAE/MFE calibration (no close column in prices)")

portfolio_controls = get_portfolio_risk_controls(CASE_STUDY_ID)
if MAX_RISK_VARIANTS > 0:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
    portfolio_controls = portfolio_controls[:MAX_RISK_VARIANTS]
    print(f"Risk variants limited to {MAX_RISK_VARIANTS} each")

# %% [markdown]
# ## 2. Risk Overlay Sweep
#
# For each top combo, run one baseline (no risk rules) then one backtest
# per position-level risk control variant:
#
# - **Trailing stop** — exits individual positions when price retraces a
#   percentage from its high-water mark since entry. Includes MAE-calibrated
#   variants derived from the prices loaded above.
# - **Stop-loss** — exits when a position's unrealized loss exceeds a threshold.
# - **Time exit** — closes a position after a fixed number of bars.
#
# Each rule exits one position at a time; the strategy continues holding
# everything else and re-enters fresh names on the next rebalance. At
# 15-minute cadence, every exit incurs round-trip cost — tighter thresholds
# trigger more often and compound that drag.

# %% [markdown]
# ### Run a single risk-overlay backtest
#
# Helper that applies a position-level risk rule (stop-loss, trailing stop,
# or time exit) to a base spec and runs the backtest.


# %%
def run_risk_backtest(rc, base_spec, pred_hash, predictions, combo_weights, level="position"):
    """Run one risk-overlay backtest. Returns True on success."""
    spec_risk = clone_backtest_spec(base_spec)
    spec_risk["chapter"] = "ch19"

    if level == "position":
        rule_key = "bars" if rc["type"] == "time_exit" else "threshold"
        spec_risk["strategy"]["risk"] = {
            "name": rc["name"],
            "position_rules": [{"type": rc["type"], rule_key: rc[rule_key]}],
        }
    else:
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
        print(
            f"    {rc['name']}: Sharpe={result.metrics.get('sharpe', 0):.3f}, "
            f"MaxDD={result.metrics.get('max_drawdown', 0):.2%}"
        )
        return True
    except Exception as e:
        print(f"    {rc['name']}: FAILED — {e}")
        return False


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

    t0 = time.time()
    combo_weights = precompute_weights(
        predictions, base_spec, prices, label=LABEL, case_study=CASE_STUDY_ID
    )
    print(
        f"  Combo {combo_idx + 1}/{len(top_combos)}: {alloc_method} — "
        f"weights precomputed in {time.time() - t0:.0f}s"
    )

    if not IS_VECTORIZED:
        for rc in position_controls:
            n_done += run_risk_backtest(
                rc, base_spec, pred_hash, predictions, combo_weights, "position"
            )

    for rc in portfolio_controls:
        n_done += run_risk_backtest(
            rc, base_spec, pred_hash, predictions, combo_weights, "portfolio"
        )

print(f"\nRisk sweep complete: {n_done} backtests")

# %% [markdown]
# ## 3. Risk Impact Analysis
#
# This section is **read-only** — queries the registry for risk overlay
# results and computes impact relative to the allocation-stage baseline.
#
# The key metric is `sharpe_delta`: the change in Sharpe from adding the
# overlay. For intraday strategies, overly tight thresholds can trigger
# frequently and reduce time-in-market enough to hurt Sharpe even when
# they successfully cut drawdowns. The analysis identifies which thresholds
# improve the risk/return trade-off and which cut too aggressively.

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
# ## Key Takeaways
#
# At 15-minute cadence the dominant risk is cost drag, not tail events. Every
# position-level overlay trades one round-trip cost band for a tighter risk
# profile — the question is whether the cost is worth the drawdown reduction.
#
# The clearest pattern is the threshold gradient: longer time-exits (20–40 bars)
# and looser stop / trailing thresholds (≥15%) sit at the top of the overlay
# Sharpe distribution because they minimize cost-induced churn, while tight
# thresholds (1–5%) trigger constantly at 15-minute cadence and stack the
# largest cost drag (the 1% trailing stop is the worst overlay tested). That
# ordering is the transferable lesson — overlay aggressiveness trades drawdown
# reduction against turnover cost.
#
# The validation Sharpes here are computed on full-universe slot configs and
# look strong, but they are Act-1 numbers: the full-universe slot winners do
# not survive out of sample (Ch16 §3, Ch20). The case study's tradeable result
# is the cost-feasible carrier (slot ensemble, validation +1.13 / holdout
# +0.53), not any configuration ranked in this sweep.
#
# Portfolio-level kill switches (max-drawdown, daily-loss) are NOT swept here:
# their permanent-halt semantics produce zero-std Sharpe artifacts in ranking.
# They remain available as governance instruments via the engine's
# `MaxDrawdownLimit` / `DailyLossLimit` classes — Ch19 §19.8 demonstrates that
# usage. The lesson the case study narrates is that intraday signal economics,
# not overlay choice, is the binding constraint here.
#
# **Next**: Ch20 synthesis aggregates results from Ch16–19 across all case studies.
