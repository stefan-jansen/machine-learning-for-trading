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
# # NASDAQ-100 Microstructure: Holdout Backtest
#
# **Chapter 20 - Out-of-sample evaluation**
#
# [`18_holdout_predictions`](18_holdout_predictions.ipynb) refitted the selected
# configuration on the history before the holdout window and wrote its predictions over it.
# This notebook trades them, with the concentration, the allocator, the risk overlay and the
# cost assumption the rest of the case study used, and registers the result.
#
# Nothing is chosen here. The predictions, the allocator, the concentration, the rebalance
# cadence, the overlay and the charge all arrive fixed from earlier notebooks, and the only
# thing this notebook decides is that they are applied unchanged. That is the whole design: a
# holdout result is worth something exactly to the extent that no decision was made after
# seeing it, and every knob left open here would be a decision.
#
# The comparison to validation is printed but not interpreted. The validation figure is the
# maximum of a ranking over more than a thousand backtests and the holdout figure is one
# measurement over a much shorter window; what can be said about the gap between them is
# [`20_strategy_analysis`](20_strategy_analysis.ipynb)'s subject, with the intervals to say
# it.
#
# **Prerequisites:** [`18_holdout_predictions`](18_holdout_predictions.ipynb).
#
# **Scope:** one backtest. No selection, no comparison beyond a printed pair.

# %%
"""NASDAQ-100 Microstructure: Holdout Backtest."""

import json
import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import ensure_backtest_spec, strategy_view
from case_studies.utils.backtest_runner import resolved_allow_short_selling, run_backtest
from case_studies.utils.conformal import (
    compute_holdout_conformal_widths,
    ensure_conformal_calibration_identity,
    holdout_conformal_embargo_steps,
)
from case_studies.utils.registry import (
    backtest_run_status,
    read_predictions,
    training_hash_from_spec,
)
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
MAX_SYMBOLS = 0

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)


def _registered_holdout_backtests(case_dir, prediction_hash):
    """The backtest hashes already registered against one holdout prediction set."""
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as conn:
        rows = conn.execute(
            "SELECT backtest_hash FROM backtest_runs WHERE prediction_hash = ? "
            "ORDER BY backtest_hash",
            (prediction_hash,),
        ).fetchall()
    return [backtest_hash for (backtest_hash,) in rows]


# %% [markdown]
# ## 1. The configuration, and the predictions it produced on the holdout
#
# The carrier is resolved the same way [`17_costs`](17_costs.ipynb) and
# [`18_holdout_predictions`](18_holdout_predictions.ipynb) resolve it, so all three run the
# same configuration by construction rather than by a hash copied between them.
#
# Which holdout prediction set belongs to it is derived rather than searched for. Re-deriving
# the holdout training specification reproduces the training identity 18 registered - the
# derivation is deterministic and the identity covers it - so the prediction set is looked up
# by that identity and the carrier's checkpoint. A search over holdout prediction sets would
# have to guess which one belonged to this configuration.

# %%
carrier = resolve_solvent_carrier(CASE_STUDY_ID)
LABEL = carrier["label"]
validation_prediction_record = study.results.open(carrier["val_prediction_hash"]).registry_record()

holdout_spec = build_holdout_training_spec(
    study,
    study.results.open(carrier["training_hash"]).spec(),
    timeline=(
        pl.read_parquet(study.root / "labels" / f"{LABEL}.parquet")
        .get_column("timestamp")
        .unique()
        .sort()
        .to_list()
    ),
    case_study=CASE_STUDY_ID,
)
holdout_training_hash = training_hash_from_spec(holdout_spec)

with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as conn:
    match = conn.execute(
        """
        SELECT prediction_hash FROM prediction_sets
        WHERE split = 'holdout' AND training_hash = ?
          AND checkpoint_kind IS ? AND checkpoint_value IS ?
        """,
        (
            holdout_training_hash,
            validation_prediction_record["checkpoint_kind"],
            validation_prediction_record["checkpoint_value"],
        ),
    ).fetchone()
if match is None:
    raise RuntimeError(
        f"No holdout prediction set for training {holdout_training_hash}. Run "
        "18_holdout_predictions first; this notebook does not fit."
    )
HOLDOUT_PREDICTION_HASH = match[0]

print(f"Carrier:            {carrier['val_backtest_hash']}  {carrier['config_name']} ({LABEL})")
print(f"Holdout training:   {holdout_training_hash}")
print(f"Holdout prediction: {HOLDOUT_PREDICTION_HASH}")

# %% [markdown]
# ## 2. Calibration, where the allocator needs it
#
# An allocator that sizes by a conformal width is calibrated from errors the model has already
# made, and on the holdout there are none to use: an error is usable only once the return it
# measures has been realised, and every holdout return realises inside the window being
# evaluated. Such a carrier takes its widths from the validation residuals of the validation
# prediction set, which is what the allocator would have had standing at the start of the
# window.
#
# The embargo matters for this case study's label and would not for every one. A residual
# observed at `t` measures a return realising over `(t, t+15min]`, so the last residuals of the
# validation span reach into the holdout window, and calibrating on them would size holdout
# positions with holdout price information. The reach is minutes rather than the ETF study's
# three weeks, and it is the same defect at any width: the residuals that cross the boundary are
# the ones nearest it, which are also the ones a calibration weights most. The step count comes
# from the reviewed table in `conformal.py`, which records the label horizon.
#
# The branch is here rather than assumed away because the carrier can change. It does not fire
# for the one this case study currently reports, which sizes by inverse volatility over a
# declared window and needs no calibration - so the line below prints that rather than staying
# silent, which is what tells a reader the branch was evaluated.
#
# The widths themselves are NOT written here. Writing them replaces the artifact an already
# registered run was sized by, and the replacement guard in section 3 can still refuse this run
# afterwards - which would leave the registered holdout pointing at a calibration that no
# longer existed. The write is below the guard.

# %% tags=["results"]
allocation = strategy_view(json.loads(carrier["spec_json"])).get("allocation") or {}
NEEDS_CALIBRATION = allocation.get("method") == "conformal_weighted"
embargo_steps = holdout_conformal_embargo_steps(CASE_STUDY_ID, LABEL) if NEEDS_CALIBRATION else 0
if NEEDS_CALIBRATION:
    print(f"Conformal carrier: embargo {embargo_steps} observation(s), widths written below.")
else:
    print(f"Allocator {allocation.get('method', 'equal_weight')!r} needs no calibration.")

# %% [markdown]
# ## 3. The backtest
#
# The strategy specification is the carrier's own, re-pointed at the holdout prediction set and
# the holdout price window. Nothing else about it changes - the concentration, the allocator,
# the risk overlay and the commission and slippage levels are the ones `setup.yaml` declares
# and every validation number in this case study was net of.
#
# The run registers under `stage='holdout'`, which the registry derives from the prediction
# set's split rather than from anything asserted here - and that derivation takes precedence
# over the risk block the carrier carries, which would otherwise file this as another risk
# overlay.
#
# **The window carries one backtest**, for the same reason 18 lets it carry one prediction
# generation. 18's guard is on the model - the training identity and the checkpoint - and it
# cannot see this one: a changed allocator, overlay, cost level or calibration produces the
# same holdout predictions and a different result from them. Like 18's, this guard refuses
# rather than offering a replacement, because deleting a result that has been seen does not
# unsee it.
#
# The test is the backtest hash, not a field-by-field comparison. Every input that changes the
# result is in that hash by construction, and a guard naming fields instead has to be right
# about all of them. The hash is resolved before anything runs, so nothing is evaluated on the
# holdout before the question is answered, and it comes from `backtest_run_status` - the call
# the runner itself makes - because asking it is the only way to be sure the guard and the
# runner agree about identity. The run asserts they still agree afterwards, because a guard
# that had quietly stopped predicting the hash would let everything through while looking
# correct.

# %% tags=["results"]
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="holdout", max_symbols=MAX_SYMBOLS)
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique():,} symbols")
print(
    f"Predictions: {predictions.height:,} rows, "
    f"{predictions['timestamp'].n_unique():,} decision timestamps, "
    f"{predictions['timestamp'].dt.date().n_unique():,} sessions"
)

spec = ensure_backtest_spec(
    CASE_STUDY_ID,
    bt_config,
    json.loads(carrier["spec_json"]),
    prices=prices,
    prediction_hash=HOLDOUT_PREDICTION_HASH,
    initial_cash=bt_config.initial_cash,
)
spec["chapter"] = "ch20"
# The embargo goes into the specification before anything hashes it. The widths are an input to
# this backtest and the embargo decides them, so two embargoes are two results and must not
# share an identity.
if NEEDS_CALIBRATION:
    spec = ensure_conformal_calibration_identity(spec, holdout_embargo_steps=embargo_steps)

spec["backtest_config"]["account"]["allow_short_selling"] = resolved_allow_short_selling(spec, None)
prospective_hash = backtest_run_status(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH, spec).backtest_hash
superseded_backtests = sorted(
    set(_registered_holdout_backtests(CASE_DIR, HOLDOUT_PREDICTION_HASH)) - {prospective_hash}
)
if superseded_backtests:
    raise RuntimeError(
        "the holdout window already carries a backtest of a different configuration: "
        + ", ".join(superseded_backtests)
        + f". This run would register {prospective_hash} and has not run. Same rule as "
        "18_holdout_predictions: discarding the earlier result would not undo having observed "
        "it, so there is no switch here. Leave the selection where it was, or retire the "
        "earlier evaluation through the registry's lifecycle."
    )

# The guard has passed, so this run will register and the widths it is sized by are the ones
# that belong beside this prediction set.
if NEEDS_CALIBRATION:
    widths = compute_holdout_conformal_widths(
        CASE_STUDY_ID,
        carrier["val_prediction_hash"],
        HOLDOUT_PREDICTION_HASH,
        alpha=float(allocation.get("alpha", 0.2)),
        min_calibration_n=int(allocation["min_calibration_n"]),
        embargo_steps=embargo_steps,
        write=True,
    )
    print(
        f"Conformal widths: {widths.height:,} rows over "
        f"{widths['symbol'].n_unique():,} names, embargo {embargo_steps} observation(s)"
    )

result = run_backtest(
    CASE_STUDY_ID,
    HOLDOUT_PREDICTION_HASH,
    spec,
    prices=prices,
    predictions=predictions,
    label=LABEL,
    register=True,
    initial_cash=bt_config.initial_cash,
    calendar=bt_config.calendar,
)
if result.backtest_hash != prospective_hash:
    raise RuntimeError(
        f"the guard predicted {prospective_hash} and the runner registered "
        f"{result.backtest_hash}. The guard decides what may run on the holdout, so a guard "
        "that no longer reproduces the runner's identity is not a smaller problem than the one "
        "it was written for."
    )
print(f"Holdout backtest: {result.backtest_hash}")

# The stage is checked rather than trusted. This carrier comes from the risk stage and its spec
# carries a risk block, and stage inference reads the prediction's split before that block - so
# a holdout run files as `holdout`. If that order ever changes, the whole out-of-sample result
# lands in `risk_overlay` and `20_strategy_analysis` finds no holdout at all, which is a failure
# four notebooks away from its cause.
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as conn:
    registered_stage = conn.execute(
        "SELECT stage FROM backtest_runs WHERE backtest_hash = ?", (result.backtest_hash,)
    ).fetchone()[0]
if registered_stage != "holdout":
    raise RuntimeError(
        f"the holdout backtest registered under stage={registered_stage!r} rather than "
        "'holdout'; the split-based inference in registry.store._infer_stage did not take "
        "precedence over this carrier's risk block"
    )

# %% [markdown]
# ## 4. What it came out at
#
# The two numbers below are one strategy measured on two disjoint periods, and the gap between
# them is not an estimate of decay. The validation figure is the maximum of a ranking over more
# than a thousand backtests, so it carries the selection; the holdout figure is one measurement
# over a much shorter window, so it carries that window's sampling error. Both push the pair
# apart on their own, before any real change in the strategy's edge.
# [`20_strategy_analysis`](20_strategy_analysis.ipynb) is where they are given intervals and a
# paired comparison.

# %% tags=["results"]
metrics = result.metrics
# The carrier's own registered Sharpe, not the resolver's. `resolve_solvent_carrier` reports the
# common-support figure, which re-ranks candidates on the timestamps every one of them covers;
# that is the right number for choosing between candidates and the wrong one to set beside a
# holdout measured over its own full window. Both are printed, so neither has to be inferred
# from the other.
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as conn:
    carrier_sharpe, carrier_periods, carrier_trades = conn.execute(
        "SELECT sharpe, n_periods, num_trades FROM backtest_metrics WHERE backtest_hash = ?",
        (carrier["val_backtest_hash"],),
    ).fetchone()

# `n_periods` is on the DAILY grid, not the decision grid. The backtester aggregates to daily
# returns before it computes anything, and `evaluation.periods_per_year` (252) annualizes that
# grid - which is why a six-month intraday holdout reports a few hundred periods rather than
# tens of thousands. Reading it as decision slots is the error that made the committed
# NASDAQ-100 benchmark understate itself roughly fivefold before #362 regenerated it.
print(f"Validation Sharpe over {int(carrier_periods):,} sessions: {carrier_sharpe:.3f}")
print(f"  the same run re-ranked on common support: {carrier['val_sharpe']:.3f}")
print(
    f"Holdout Sharpe over {int(metrics['n_periods']):,} sessions:    "
    f"{metrics.get('sharpe', float('nan')):.3f}"
)
print(
    f"Holdout: CAGR {metrics.get('cagr', float('nan')):.1%}, "
    f"max drawdown {metrics.get('max_drawdown', float('nan')):.2%}, "
    f"win rate {metrics.get('win_rate', float('nan')):.0%}"
)
# This case study runs the bar-by-bar engine - its carrier declares a trailing stop, which a
# vectorized weight-times-return path cannot express - so trade counts are recorded and can be
# compared. A holdout that rebalanced far less than the validation run at the same cadence
# would say the basket stopped changing, which is a different thing from a lower Sharpe.
print(
    f"Trades: {int(metrics.get('num_trades', 0)):,} on the holdout, "
    f"{int(carrier_trades):,} on validation"
)

# %% [markdown]
# ## What this notebook establishes, and what it does not
#
# It establishes a return series for the selected configuration over a period no choice in this
# case study was made on. That is the only thing a holdout can give, and it is worth less than
# it looks: the window is short beside the validation span it is being compared with, which is
# too few observations to separate a strategy that decayed from one that had an ordinary year.
#
# It does not establish that this configuration was the right one to carry here. The selection
# that brought it was made on validation, over a pool large enough that its maximum is
# optimistic by construction, and this notebook inherits that pool without correcting for it.
# The deflation is [`20_strategy_analysis`](20_strategy_analysis.ipynb)'s.
#
# Re-running this notebook against the same carrier is free and idempotent - the backtest hash
# is unchanged and the registered run is served back. A different carrier is refused, for the
# reason 18 gives.
#
# **Next:** [`20_strategy_analysis`](20_strategy_analysis.ipynb).
