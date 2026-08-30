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
# # US Firm Characteristics: Holdout Backtest
#
# **Chapter 20 - Out-of-sample evaluation**
#
# [`15_holdout_predictions`](15_holdout_predictions.ipynb) refitted the selected
# configuration on the history before the holdout window and wrote its predictions over
# it. This notebook trades them, with the sizing and the cost assumption the rest of the
# case study used, and registers the result.
#
# Nothing is chosen here. The predictions, the allocator, the concentration, the rebalance
# cadence and the charge all arrive fixed from earlier notebooks, and the only thing this
# notebook decides is that they are applied unchanged. That is the whole design: a holdout
# result is worth something exactly to the extent that no decision was made after seeing
# it, and every knob left open here would be a decision.
#
# The comparison to validation is printed but not interpreted. One year of monthly
# decisions is twelve observations, and what can be said about a Sharpe estimated from
# twelve observations is [`17_strategy_analysis`](17_strategy_analysis.ipynb)'s subject,
# with the intervals to say it.
#
# **Prerequisites:** [`15_holdout_predictions`](15_holdout_predictions.ipynb).
#
# **Scope:** one backtest. No selection, no comparison beyond a printed pair.

# %%
"""US Firm Characteristics: Holdout Backtest."""

import json
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import ensure_backtest_spec, strategy_view
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.conformal import (
    compute_holdout_conformal_widths,
    holdout_conformal_embargo_steps,
)
from case_studies.utils.registry import read_predictions
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
MAX_SYMBOLS = 0

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
bt_config = get_backtest_config(CASE_STUDY_ID)

# %% [markdown]
# ## 1. The configuration, and the predictions it produced on the holdout
#
# The carrier is resolved the same way [`14_costs`](14_costs.ipynb) and
# [`15_holdout_predictions`](15_holdout_predictions.ipynb) resolve it, so all three run
# the same configuration by construction rather than by a hash copied between them.
#
# Which holdout prediction set belongs to it is derived rather than searched for. Re-deriving
# the holdout training specification reproduces the training identity 15 registered - the
# derivation is deterministic and the identity covers it - so the prediction set is looked up
# by that identity and the carrier's checkpoint. A search over holdout prediction sets would
# have to guess which one belonged to this configuration, and this case study's registry holds
# an older one that does not.

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

# %%
import sqlite3

from case_studies.utils.registry import training_hash_from_spec

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
        "15_holdout_predictions first; this notebook does not fit."
    )
HOLDOUT_PREDICTION_HASH = match[0]

print(f"Carrier:            {carrier['val_backtest_hash']}  {carrier['config_name']} ({LABEL})")
print(f"Holdout training:   {holdout_training_hash}")
print(f"Holdout prediction: {HOLDOUT_PREDICTION_HASH}")

# %% [markdown]
# ## 2. Calibrating the allocator on validation residuals only
#
# This carrier sizes positions by a conformal width, and a width is calibrated from the
# errors the model has already made. On the holdout there are none to use: an error is
# only usable once the return it measures has been realised, and every holdout return
# realises inside the window being evaluated. So the widths come from the validation
# residuals of the validation prediction set, which is what the allocator would have had
# standing at the start of the window.
#
# The last validation residuals are dropped rather than used. A residual observed at the
# end of the validation span measures a return that realises one month later, and one
# month later is inside the holdout window - so keeping it would size holdout positions
# with holdout price information. The number of observations dropped is the label's
# horizon on this panel's own grid, one monthly step, read from the reviewed table rather
# than assumed here.

# %% tags=["results"]
allocation = strategy_view(json.loads(carrier["spec_json"])).get("allocation") or {}
if allocation.get("method") == "conformal_weighted":
    embargo_steps = holdout_conformal_embargo_steps(CASE_STUDY_ID, LABEL)
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
    print(f"  calibration_n: median {widths['calibration_n'].median():.0f}")
else:
    print(f"Allocator {allocation.get('method', 'equal_weight')!r} needs no calibration.")

# %% [markdown]
# ## 3. The backtest
#
# The strategy specification is the carrier's own, re-pointed at the holdout prediction
# set and the holdout price window. Nothing else about it changes - the commission and
# slippage are the levels `setup.yaml` declares, the same ones every validation number in
# this case study was net of, and the same ones sitting inside the swept grid in
# [`14_costs`](14_costs.ipynb).
#
# The run registers under `stage='holdout'`, which the registry derives from the
# prediction set's split rather than from anything asserted here.

# %% tags=["results"]
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="holdout", max_symbols=MAX_SYMBOLS)
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique():,} assets")
print(f"Predictions: {predictions.height:,} rows, {predictions['timestamp'].n_unique()} dates")

spec = ensure_backtest_spec(
    CASE_STUDY_ID,
    bt_config,
    json.loads(carrier["spec_json"]),
    prices=prices,
    prediction_hash=HOLDOUT_PREDICTION_HASH,
    initial_cash=bt_config.initial_cash,
)
spec["chapter"] = "ch20"

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
print(f"Holdout backtest: {result.backtest_hash}")

# %% [markdown]
# ## 4. What it came out at
#
# The two numbers below are one strategy measured on two disjoint periods, and the gap
# between them is not an estimate of decay. The validation figure is the maximum of a
# ranking over more than a thousand backtests, so it carries the selection; the holdout
# figure is one measurement of twelve monthly returns, so it carries the sampling error of
# twelve observations. Both facts push the pair apart on their own, before any real change
# in the strategy's edge. [`17_strategy_analysis`](17_strategy_analysis.ipynb) is where
# they are given intervals and a paired comparison.

# %% tags=["results"]
metrics = result.metrics
# The carrier's own registered Sharpe, not the resolver's. `resolve_solvent_carrier` reports
# the common-support figure, which re-ranks the conformal field on the timestamps every
# candidate covers; that is the right number for choosing between candidates and the wrong
# one to set beside a holdout measured over its own full window. Both are printed, so
# neither has to be inferred from the other.
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as conn:
    carrier_sharpe, carrier_periods = conn.execute(
        "SELECT sharpe, n_periods FROM backtest_metrics WHERE backtest_hash = ?",
        (carrier["val_backtest_hash"],),
    ).fetchone()

print(f"Validation Sharpe over its {int(carrier_periods)} months:  {carrier_sharpe:.3f}")
print(f"  the same run re-ranked on common support: {carrier['val_sharpe']:.3f}")
print(
    f"Holdout Sharpe over {int(metrics['n_periods'])} months:        "
    f"{metrics.get('sharpe', float('nan')):.3f}"
)
print(
    f"Holdout: CAGR {metrics.get('cagr', float('nan')):.1%}, "
    f"max drawdown {metrics.get('max_drawdown', float('nan')):.2%}, "
    f"win rate {metrics.get('win_rate', float('nan')):.0%}"
)
# No trade or turnover figure is reported. The vectorized rebalance path this case study
# runs does not record one - `num_trades` is NULL for every backtest in this registry,
# holdout and validation alike - and a zero standing in for an unrecorded count reads as a
# strategy that never traded.

# %% [markdown]
# ## What this notebook establishes, and what it does not
#
# It establishes a return series for the selected configuration over a period no choice in
# this case study was made on. That is the only thing a holdout can give, and it is worth
# less than it looks: one year of monthly rebalances is twelve observations, which is too
# few to separate a strategy that decayed from one that had an ordinary year.
#
# It does not establish that this configuration was the right one to carry here. The
# selection that brought it was made on validation, over a pool large enough that its
# maximum is optimistic by construction, and this notebook inherits that pool without
# correcting for it. The deflation is [`17_strategy_analysis`](17_strategy_analysis.ipynb)'s.
#
# The holdout stays re-runnable. If the selection changes, this generation is deleted and
# another is produced; it is not a resource that has been spent.
#
# **Next:** [`17_strategy_analysis`](17_strategy_analysis.ipynb).
