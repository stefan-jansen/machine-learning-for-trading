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
# # US Firm Characteristics: Risk Overlay Applicability
#
# **Chapter 19 — Risk Management**
#
# The cross-stage rank-1 is the equal-weight baseline `gbm/leaves_7_mse` at
# iteration 500 and TOP_K 50, validation Sharpe 2.63 [2.07, 3.24]. This case
# study uses the vectorized forward-return backtest path. That path evaluates
# weights at the label horizon and cannot represent intra-period stop events.
#
# The setup retains 14 position-level controls as the engine-path catalog, but
# this notebook must not pretend to execute them on monthly outcome labels.
# Portfolio-level controls are intentionally absent. The result is an explicit
# applicability boundary with no registered risk-overlay variants.
#
# Sections 1-2 establish the parent and execution boundary. Section 3 confirms
# that no risk-overlay rows were registered.
#
# **Learning Objectives:**
# 1. Select the correct parent across baseline and allocation stages
# 2. Identify whether the declared backtest mode can represent intra-period rules
# 3. Connect the risk overlay reading back to Ch19 §19.8 governance framing
#
# **Book Reference:** Chapter 19, Sections 19.3–19.6, 19.8
#
# **Prerequisites:** Completed Ch17 allocation sweep with results in `registry.db`.

# %%
"""US Firm Characteristics: Risk: Engine-Level Risk Rules."""

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
CASE_STUDY_ID = "us_firm_characteristics"
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
# ## 1. Load the Best Pre-Risk Run
#
# Risk analysis starts from the top validation run across the equal-weight
# baseline and allocation stages. This preserves the established greedy funnel
# when an allocator does not improve on its baseline parent.


# %%
def _resolve_pre_risk_runs(case_study: str, label: str, *, split: str, top_n: int) -> pl.DataFrame:
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


top_combos = _resolve_pre_risk_runs(
    CASE_STUDY_ID,
    LABEL,
    split="validation",
    top_n=TOP_N_COMBOS,
)

if top_combos.is_empty():
    msg = "No baseline or allocation results found. Run the upstream notebooks first."
    raise RuntimeError(msg)

for row in top_combos.iter_rows(named=True):
    spec = json.loads(row["spec_json"])
    alloc = strategy_view(spec).get("allocation", {}).get("method", "equal_weight")
    print(f"  Sharpe={row['sharpe']:.3f}  alloc={alloc}  bt_hash={row['backtest_hash'][:8]}")

# %%
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="validation", max_symbols=MAX_SYMBOLS)

# %% [markdown]
# ### MAE/MFE-Calibrated Trailing Stops
#
# MAE/MFE calibration requires an engine path with intra-period prices. The
# vectorized monthly-outcome path skips this calibration and leaves the
# configured position-control catalog unexecuted.

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
# Portfolio-limit overlays were purged 2026-05-17; this CS sweeps position-level
# overlays only. Fail loudly if a portfolio overlay is ever re-introduced into
# setup.yaml so it cannot silently re-file overlay backtests against the spine.
assert not portfolio_controls, (
    f"Unexpected portfolio risk controls for {CASE_STUDY_ID}: {portfolio_controls}. "
    "Portfolio-limit overlays were removed; only position-level overlays are swept."
)
if MAX_RISK_VARIANTS > 0:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
    portfolio_controls = portfolio_controls[:MAX_RISK_VARIANTS]
    print(f"Risk variants limited to {MAX_RISK_VARIANTS} each")

# %% [markdown]
# ## 2. Risk Overlay Sweep
#
# Engine-mode case studies run one backtest per position-level control here.
# US Firm Characteristics is vectorized, so the position loop is deliberately
# inactive; the asserted-empty portfolio-control list also produces no run.

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
# results and computes impact relative to the selected pre-risk baseline.
#
# An empty result is the expected outcome for this execution mode. It records
# that no risk rule was evaluated rather than assigning a fabricated Sharpe or
# drawdown effect to an inapplicable overlay.

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
# 1. The established cross-stage selector carries the equal-weight baseline
#    `15356ec80a3e`, Sharpe 2.632, into the risk boundary.
# 2. The vectorized monthly-outcome path cannot model intra-period stop-loss,
#    trailing-stop, or time-exit events. Running those controls would require
#    an engine-level return path, not a parameter toggle.
# 3. Portfolio-level limits are intentionally absent. They remain governance
#    controls rather than validation variants competing on Sharpe.
# 4. No risk-overlay row is registered and the sealed holdout is not accessed.
#
# **Next:** Ch20 synthesis aggregates results from Ch16–19 across all case studies,
# placing this case study's first-pass performance in context against the full suite.
