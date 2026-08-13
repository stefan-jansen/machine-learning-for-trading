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
# # US Equities Panel: Risk: Engine-Level Risk Rules
#
# **Chapter 19 — Risk Management**
#
# For the US equities panel, the risk-overlay question is different from
# most other case studies. The 3,200-stock universe already provides
# inherent diversification, so portfolio-level concentration limits are
# not the relevant lever. The motivating concern is regime sensitivity:
# the cross-stage validation rank-1 — gbm `leaves_31_huber` on the
# fwd_ret_5d horizon, score-weighted top_k=20 — pairs a validation
# Sharpe of 2.028 [1.464, 2.549] with a holdout Sharpe that resolves on
# the negative side of zero. Position-level rules (stop-loss, trailing
# stop, time exit) modify trade duration and risk-shed behavior on
# losing positions; whether any of them materially blunts the
# validation-to-holdout decay is an empirical question this notebook
# answers in the negative.
#
# This notebook tests **position-level risk controls** on the top
# allocation-stage combos. Stop-losses, trailing stops, and time exits
# execute inside the backtest engine — the results reflect the real
# execution path. Portfolio-level controls (max-drawdown breakers,
# daily-loss limits) were removed from the sweep across all case
# studies on 2026-05-17 because their permanent-halt semantics produced
# a degenerate Sharpe artifact (mean of one active day divided by the
# standard deviation of zeros); `get_portfolio_risk_controls` returns
# an empty list and the portfolio-rule loop below executes zero
# variants.
#
# Sections 1–2 generate risk-overlay backtests (write to registry).
# Section 3 queries the registry via `BacktestExplorer` for analysis.
#
# **Learning Objectives:**
# 1. Measure how position-level rules (stop-loss, trailing stop, time exit) change
#    trade count, drawdown, and Sharpe for a high-turnover daily strategy
# 2. Locate the best-performing position-level overlay for this lineage and
#    quantify how much (if any) of the validation→holdout decay it absorbs
# 3. Recognize that risk overlays applied in-sample cannot retroactively close
#    an out-of-sample Sharpe gap whose source is a signal-regime interaction
#
# **Book Reference:** Chapter 19, Sections 19.3–19.6
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""US Equities Panel: Risk: Engine-Level Risk Rules."""

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
CASE_STUDY_ID = "us_equities_panel"
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
# The allocation-stage top combos are all GBM-based configurations. We apply
# risk overlays to measure how each rule type modifies the risk-return profile.
# For daily-cadence strategies, the baseline drawdown profile will show frequent
# small drawdowns rather than occasional large ones — position-level stops are
# calibrated to this environment.

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
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)

# %% [markdown]
# ### MAE/MFE-Calibrated Trailing Stops

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
# For each top combo, run one position-level overlay per rule in the
# grid. Position-level rules (stop_loss, trailing_stop, time_exit)
# execute inside the engine. The portfolio-level loop below runs zero
# variants because `get_portfolio_risk_controls` returns an empty list
# after the 2026-05-17 setup.yaml purge.
#
# For the US equities panel, watch one pattern that is specific to
# daily-cadence broad-universe strategies:
#
# - **Stop-losses that increase turnover**: tight position-level stops fire
#   frequently under daily rebalancing, adding trades and worsening the already
#   tight cost tolerance. A stop that "protects" Sharpe at 0 bps may make the
#   strategy inoperable at 15 bps. The MAE/MFE-calibrated trailing stops
#   below probe whether thresholds tuned to the empirical loss distribution
#   improve on the round-number grid.

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
# This section is **read-only** — queries the registry for risk overlay
# results and computes impact relative to the allocation-stage baseline.
#
# For the US equities panel, the key comparison is not simply "which rule
# has the best Sharpe" but "which rule improves the cost-adjusted profile."
# A risk overlay that adds turnover is net-negative here even if its gross
# Sharpe is higher. Time-based exits, which cap holding duration rather
# than firing on price moves, change trade count modestly and are the
# candidates with the smallest cost penalty.

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
# 1. Position-level rules (stop-loss, trailing stop) increase trade count
#    for a daily-cadence strategy, exacerbating the cost fragility identified
#    in Ch18. Any gross Sharpe improvement from stops must be evaluated net
#    of the additional turnover costs they generate.
# 2. Among the position-level overlays swept here, the time-exit family
#    (cap holding duration at 10 / 20 / 40 bars) carries the smallest
#    turnover penalty and is the only family that lifts the lineage's
#    validation Sharpe relative to the no-overlay allocation row; the
#    cross-stage validation rank-1 carrier is `time_exit_40` on top of
#    the score-weighted top_k=20 allocation. Trailing-stop and
#    hard-stop variants all sit below the unadorned allocation Sharpe.
# 3. The validation-to-holdout deterioration is a signal-regime
#    interaction, not a risk-management failure. Risk overlays applied
#    in-sample cannot retroactively close an out-of-sample Sharpe gap
#    whose source is the signal-regime interaction itself; the §6 spine
#    reading quantifies the gap.
# 4. The US equities panel pairs a 16-fold validation record that resolves
#    above zero with a holdout closure that resolves on the negative side
#    under index-paired resampling — illustrating that fold count and fold
#    stability are necessary but not sufficient for generalization. Regime
#    coverage of the training set is a separate axis of evidence.
#
# **Next**: Ch20 synthesis aggregates results from Ch16–19 across all case
# studies. The US equities panel contributes a strong validation reading
# paired with a holdout closure that resolves on the negative side under
# index-paired resampling — a pairing that anchors the cross-CS holdout-
# decay table in the synthesis chapter.
