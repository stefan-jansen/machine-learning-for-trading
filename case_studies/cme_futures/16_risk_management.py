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
# # CME Futures: Risk: Engine-Level Risk Rules
#
# **Chapter 19 — Risk Management**
#
# Commodity futures carry strategies face two distinct risk regimes. During
# normal markets, carry persists — contango and backwardation patterns are
# stable, sector rotation is gradual, and mean-reversion in roll yields
# provides a natural risk buffer. During regime shifts — such as the
# 2024–2025 transition to broad contango dominance — carry signals invert
# sharply across correlated sectors simultaneously, producing drawdowns that
# position-level stops and trailing exits are designed to limit.
#
# This notebook tests **position-level risk controls** on the top
# allocation-stage combos. Stop-losses, trailing stops, and time exits execute
# inside the backtest engine — the results reflect the real execution path.
# Portfolio-level limits (max-drawdown breakers, daily-loss caps) are
# intentionally not part of the sweep: their permanent-halt semantics
# inflate Sharpe through a zero-std artifact on rarely-active strategies
# and are framed in Chapter 19 as governance kill-switches rather than
# sweepable overlays.
#
# Sections 1–2 generate risk-overlay backtests (write to registry).
# Section 3 queries the registry via `BacktestExplorer` for analysis.
#
# **Learning Objectives:**
# 1. Measure how position-level rules (stop-loss, trailing stop, time exit) change
#    trade count, drawdown, and Sharpe in a weekly commodity futures context
# 2. Distinguish stop-driven exits from cadence-driven exits on a carry
#    signal whose natural decay horizon is the weekly rebalance step
# 3. Assess whether position-level risk overlays help or hurt a
#    fundamentally carry-driven strategy
#
# **Data caveat**: Results use ratio back-adjusted continuous contracts
# (multiplicative adjustment at roll points). See [`13_backtest`](13_backtest.ipynb) preamble for details.
#
# **Book Reference:** Chapter 19, Sections 19.3–19.6
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""CME Futures: Risk: Engine-Level Risk Rules."""

import json
import time
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    load_contract_specs_from_yaml,
    warmup_periods_for,
)
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
CASE_STUDY_ID = "cme_futures"
LABEL = ""
MAX_SYMBOLS = 0
MAX_RISK_VARIANTS = 0  # 0 = all; >0 limits position + portfolio controls each
TOP_N_COMBOS = None

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)
if TOP_N_COMBOS is None:
    TOP_N_COMBOS = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
CONTRACT_SPECS = load_contract_specs_from_yaml()
if not LABEL:
    LABEL = bt_config.primary_label

from case_studies.utils.backtest_loaders import VECTORIZED_CASE_STUDIES

IS_VECTORIZED = CASE_STUDY_ID in VECTORIZED_CASE_STUDIES
MODE_LABEL = "vectorized" if IS_VECTORIZED else "engine"
print(f"Case study: {CASE_STUDY_ID}, label: {LABEL}, mode: {MODE_LABEL}")

# %% [markdown]
# ## 1. Load Top Combos from Allocation Stage
#
# The allocation lineage carried forward is the highest-Sharpe GBM
# `leaves_7_huber` on `fwd_ret_5d` (signal-stage Sharpe 0.47 climbing to
# allocation-stage 0.84) with `score_weighted` top-k=5 long-short sizing.
# Risk overlay analysis tests whether position-level stop-losses,
# trailing stops, and time exits improve the risk-return profile on a
# strategy whose primary vulnerability is carry regime reversal — a
# simultaneous shift to contango dominance across commodity sectors
# that would invalidate the cross-sectional carry signal.

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
# For commodity futures, MAE/MFE calibration is particularly valuable because
# carry-driven positions exhibit asymmetric trade paths. A position caught in
# a sector-wide regime reversal typically shows a large adverse excursion early
# in the hold period — MAE-calibrated stops can identify this pattern and exit
# before the full drawdown materializes.

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
# For each top combo, run one baseline (no risk rules) then one backtest
# per position-level risk control. Stop-loss, trailing-stop, and
# time-exit rules execute inside the engine and surface as new
# `risk_overlay`-stage rows in the registry.
#
# Two mechanisms are of particular interest for commodity carry strategies:
# - **Stop-losses**: Carry positions that reverse sharply often signal a
#   roll yield regime change. Early exit can prevent riding a full reversal.
# - **Trailing stops**: A position that has run profitably and reverses
#   captures the asymmetric carry payoff while limiting give-back when
#   roll yield regime shifts begin to assert.

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
    t0 = time.time()
    combo_weights = precompute_weights(
        predictions, base_spec, prices, label=LABEL, case_study=CASE_STUDY_ID
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
                    contract_specs=CONTRACT_SPECS,
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
                contract_specs=CONTRACT_SPECS,
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
# This section is **read-only** — queries the registry for risk overlay
# results and computes impact relative to the allocation-stage baseline.
#
# For a carry strategy, risk overlays face a structural tension: carry profits
# from holding positions through temporary adverse moves (roll yield
# mean-reversion). Tight stops will cut positions just before roll yield
# reasserts, reducing Sharpe. Loose stops add little protection when sector
# reversals are fast. The optimal threshold, if any, will likely sit in a
# wide middle range rather than at the extremes.

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
# 1. Commodity carry strategies exhibit a structural tension with tight
#    position-level stop-losses: roll yield mean-reversion means positions
#    that look adverse early often recover. Tight stops destroy value by
#    cutting before the carry premium materializes; only stops calibrated
#    to large adverse excursions (regime reversals, not normal volatility)
#    should improve the risk-return profile. The MAE/MFE-calibrated
#    trailing-stop grid is built to surface this band.
# 2. The cross-stage rank-1 overlay is a **3.3% trailing stop** on the
#    `score_weighted top_k=5` allocator on GBM `leaves_7_huber` scores:
#    validation Sharpe 1.264 [+0.345, +2.087] vs the 0.88 0-bps cost-stage
#    baseline (paired diff +0.375 [−0.22, +1.04], p=0.236, straddles zero).
#    The lift is operationally meaningful in point but not statistically
#    resolved at 95% on the 1,290-day validation backtest window (the
#    strategy-stage window, matched to the daily IC pool in §1).
# 3. Time exits are particularly well-suited to the weekly cadence:
#    exiting positions after a fixed number of bars aligns with the
#    carry signal's natural decay horizon and prevents accumulating
#    roll costs in unfavorable contango environments. The time-exit
#    overlays sit alongside the trailing-stop cohort in the registry.
# 4. The 2024–2025 holdout window for the top-Sharpe lineage gives a
#    paired-bootstrap Sharpe diff of −0.103 [−1.68, +1.41] (p=0.90)
#    against the validation window — the point estimate decays by
#    ≈0.1 Sharpe units (validation 1.264 → holdout 1.142), but the CI
#    straddles zero and decay is not statistically resolved. The strategy
#    vs equal-weight holdout benchmark Sharpe diff is +0.364 [−1.43, +2.05]
#    (p=0.69), also straddling zero. Position-level overlays designed in-sample
#    cannot anticipate the magnitude of a multi-sector regime shift,
#    but the holdout window itself does not deliver one.
#
# Both Ch20 kill-condition gates pass: the validation Sharpe lower bound
# sits above zero, and the holdout strategy CI does not exclude zero
# negatively. The carry-premium evidence is consistent with a
# CI-credible Sharpe and a holdout whose Sharpe diff against the
# equal-weight CME universe straddles zero at conventional thresholds.
# A live strategy requires active monitoring of sector-level
# contango/backwardation balances and explicit regime indicators;
# portfolio-level limits would be wired separately as governance
# kill-switches at the operator layer.
#
# **Next**: Ch20 synthesis aggregates results from Ch16–19 across all case
# studies and contextualizes CME futures within the full book strategy set.
