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
# allocation-stage combos via the ml4t-backtest engine. Three rule families are
# swept: stop-loss, trailing stop (including MAE-calibrated variants), and
# time-exit. Each rule exits an individual position without halting the rest of
# the book, so the strategy continues trading after a single name's exit.
#
# **Two clocks run here, and every threshold below is stated in the faster one.**
# The strategy *decides* every fifteen minutes - `decision.cadence_by_label` in
# `config/setup.yaml`, which is the horizon `fwd_ret_15m` measures. The engine
# *watches* every minute: `config/backtest/base.yaml` declares
# `calendar.data_frequency: 1m`, and the broker advances a position's `bars_held`
# once per price bar it processes. A risk rule is evaluated on the watch clock,
# not the decision clock, and that is the whole reason these controls can express
# anything the signal does not already say. A stop that could only fire at the
# next rebalance would fire exactly when the strategy was going to reconsider the
# position anyway.
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
# Zero means all controls; a positive value limits position and portfolio
# controls each.
MAX_RISK_VARIANTS = 0
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
# ## 1. Load the allocation-stage combinations
#
# The overlays below are applied to the highest-scoring full-universe
# slot-mechanism backtests from the allocation stage. Those serve as a fixed
# base so the overlay is the only thing that varies between rows.
#
# What this notebook establishes is the mechanics of position-level overlays -
# how a stop-loss, a trailing stop and a time-based exit each reshape the
# distribution of outcomes. It does not select a configuration, and the level of
# the underlying results is not the subject: two overlays are compared against
# each other on one base, not against the rest of the pipeline.

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
# ### MAE-Calibrated Trailing Stops
#
# The thresholds declared in `setup.yaml` are round numbers, chosen before
# anything was measured. The quantity that would justify one is a property of the
# price paths themselves: the **maximum adverse excursion** is the worst drawdown
# reached over a holding horizon. A stop placed inside the body of that
# distribution cuts positions that would have recovered; one placed outside its
# tail never fires at all.
#
# `calibrate_trailing_stops` converts MAE percentiles - the 10th and the 25th, at
# 10-, 20- and 40-bar horizons - into thresholds named `trailing_mae_p10_h20` and
# so on, and the cell below **adds** them to the declared grid rather than
# replacing it. Both are then swept, which is what lets the round numbers and the
# calibrated ones be compared on the same base instead of one being asserted to
# be better.
#
# Maximum favorable excursion is the mirror quantity - the best unrealized gain
# over the same horizon - and it is worth knowing about because it is what a
# take-profit would be calibrated from. It does not enter this notebook: the
# calibration reads the adverse side only.
#
# It is skipped for the vectorized path and wherever the loaded frame carries no
# `close` column, because an excursion is a statement about the path a position
# took between entry and exit and neither of those can see the path.

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
# The book is long-short - `get_backtest_config("nasdaq100_microstructure").long_short`
# is `True` - so each rule below has two sides, and the engine implements both.
#
# - **Stop-loss** — places a level a fixed percentage away from the position's
#   base price on the losing side and exits when the bar's range touches it:
#   below the base and triggered by the bar's low for a long, above it and
#   triggered by the bar's high for a short. The base is the fill price by
#   default and the signal price where `stop_level_basis` says so, and the
#   trigger is intrabar rather than on the close, so a bar that pierces the level
#   and recovers still exits. It is a statement about absolute loss against that
#   base and it does not care what the position did before: a long that rose four
#   percent and gave it all back is not down against its base, and a stop-loss
#   does not see it.
# - **Trailing stop** — exits when price retreats by the threshold from the best
#   level reached *since entry*: the highest price for a long, the lowest for a
#   short. It is the same instrument measured against a moving reference, so it
#   converts an unrealized gain into a floor and it does see the long that gave
#   four percent back. The water mark is the previous bar's by default, which is
#   what `TrailStopTiming` selects. The declared grid runs 1% to 20%,
#   wider than the stop-loss grid's 3% to 15%, because the quantity it measures
#   is a retreat from a high-water mark rather than a drawdown from entry and the
#   two are not on the same scale.
# - **Time exit** — closes a position after a fixed number of watch bars, which
#   on this case study's one-minute feed means minutes. The declared grid is 10,
#   20 and 40 of them, and it straddles the decision cadence deliberately. A time
#   exit is a CAP on holding duration and never an extension: the strategy goes
#   on rebalancing, and a name dropped from the targets is closed whether or not
#   its cap has been reached. So `time_exit_10` is the only one of the three that
#   binds on every position it is applied to - it closes five minutes before the
#   fifteen-minute outcome the position was entered on has even resolved. The
#   other two bind only on a position the signal would otherwise have carried
#   through one further decision, or through several.
#
# Each rule exits one position at a time; the strategy continues holding
# everything else and re-enters fresh names on the next rebalance. Every
# triggered exit is a round trip the signal did not ask for, so it is paid for
# in spread and impact whether or not the loss it avoided was real. That is the
# trade every row of the sweep is making, and it is why a rule that cuts
# drawdown can still cost Sharpe.

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
# The key metric is `sharpe_delta`: the change in Sharpe from adding the overlay,
# measured against the no-overlay Sharpe of the allocation the overlay was applied
# to, matched on prediction and allocator. That baseline is what makes it a delta
# rather than a ranking - each combination carries its own allocation method and
# its own signal, and comparing an overlaid run against another combination's
# baseline would attribute the difference between two strategies to the risk rule.
#
# Two columns move in opposite directions and both are reported. `max_drawdown`
# is what the overlay is for, and a rule that does not reduce it is not doing its
# job. `sharpe` is what it costs, through the round trips the exits add and
# through the time out of the market between an exit and the next decision. For
# an intraday strategy the second term is the one that bites: a tight threshold
# on a one-minute watch clock can trigger on ordinary fifteen-minute fluctuation,
# and each trigger buys a small reduction in drawdown at the price of a full
# round trip.
#
# What transfers to another strategy is the gradient across thresholds - whether
# the trade-off improves monotonically, turns at some threshold, or never
# improves at all. The level of any single row does not.

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
# Each position-level overlay exits a position on a condition other than the
# signal: a stop-loss on a loss threshold, a trailing stop on a retreat from the
# best level reached, a time exit after a fixed holding period. Every one of them
# buys a tighter loss distribution by trading more, because each triggered exit
# is a round trip that the signal did not ask for.
#
# That makes the threshold the whole of the design. A tight threshold triggers
# often, so it pays for its protection frequently, and at a short rebalancing
# interval it can trigger on ordinary fluctuation rather than on the loss it was
# meant to catch. A loose threshold rarely triggers, costs little and protects
# little. The gradient across thresholds is what transfers to another strategy;
# the level of any single row does not.
#
# Portfolio-level kill switches - a maximum drawdown limit, a daily loss limit -
# are deliberately not swept here. They halt trading permanently once breached,
# so every configuration that trips one produces a truncated return series whose
# summary statistics are not comparable with those of a configuration that ran
# to the end. They remain available through the engine's `MaxDrawdownLimit` and
# `DailyLossLimit` classes as governance instruments rather than as parameters
# to optimise.
#
# **Next**: Ch20 synthesis aggregates results from Ch16–19 across all case studies.
