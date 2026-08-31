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
# # Crypto Perpetuals Funding: Holdout Backtest
#
# **Chapter 20 - Out-of-sample evaluation**
#
# [`17_holdout_predictions`](17_holdout_predictions.ipynb) refitted the selected
# configuration on history ending before 2024 and wrote its predictions over 2024-25. This
# notebook trades them with the sizing and the overlay that configuration carries, and
# registers the result under `stage='holdout'`.
#
# **Prerequisites:** [`17_holdout_predictions`](17_holdout_predictions.ipynb). This notebook
# does not fit; if the prediction set it needs is absent, it raises rather than producing
# one.
#
# **Scope:** one backtest. The comparison against validation, and the correction that
# comparison needs, are [`19_strategy_analysis`](19_strategy_analysis.ipynb)'s.

# %%
"""Crypto Perpetuals Funding: Holdout Backtest."""

import json
import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.crypto_perps_funding.funding_data import funding_rates_for_prices
from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.backtest_presets import (
    ensure_backtest_spec,
    strategy_view,
)
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
CASE_STUDY_ID = "crypto_perps_funding"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
MAX_SYMBOLS = 0
# Whether a holdout backtest of a DIFFERENT strategy may be superseded by this run. Off by
# default, and the same switch `17_holdout_predictions` uses for the model side.
REPLACE_HOLDOUT = False

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
    return [{"backtest_hash": backtest_hash} for (backtest_hash,) in rows]


def _delete_holdout_backtest(case_dir, backtest_hash):
    """Remove one registered holdout backtest and the rows derived from it.

    Same rule as `17_holdout_predictions`' replacement of a superseded generation: a holdout
    result that has been observed and then left readable beside its replacement is still a
    number someone can quote.
    """
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as conn:
        conn.execute(
            "DELETE FROM backtest_paired_metrics WHERE challenger_hash = ? OR benchmark_hash = ?",
            (backtest_hash, backtest_hash),
        )
        conn.execute("DELETE FROM backtest_metrics WHERE backtest_hash = ?", (backtest_hash,))
        conn.execute("DELETE FROM backtest_runs WHERE backtest_hash = ?", (backtest_hash,))


# %% [markdown]
# ## 1. The configuration, and the predictions it produced on the holdout
#
# The carrier is resolved the same way [`16_costs`](16_costs.ipynb) and
# [`17_holdout_predictions`](17_holdout_predictions.ipynb) resolve it, so all three run the
# same configuration by construction rather than by a hash copied between them.
#
# Which holdout prediction set belongs to it is derived rather than searched for.
# Re-deriving the holdout training specification reproduces the training identity 17
# registered - the derivation is deterministic and the identity covers it - so the prediction
# set is looked up by that identity and the carrier's checkpoint. A search over holdout
# prediction sets would have to guess which one belonged to this configuration.

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
        "17_holdout_predictions first; this notebook does not fit."
    )
HOLDOUT_PREDICTION_HASH = match[0]

print(f"Carrier:            {carrier['val_backtest_hash']}  {carrier['config_name']} ({LABEL})")
print(f"Holdout training:   {holdout_training_hash}")
print(f"Holdout prediction: {HOLDOUT_PREDICTION_HASH}")

# %% [markdown]
# ## 2. What the allocator needs before the window opens
#
# This carrier allocates by `mvo_ledoit_wolf`, which is a moment estimator: it reads a
# rolling window of underlying prices and produces weights from their covariance. That window
# does not restart because the evaluation period does. Loading the holdout slice alone would
# leave the first rebalance with no history to estimate from, and the loader would fall back
# to a median-imputed warmup - so the first weeks of the holdout would be traded on weights
# that describe nothing, and the resulting Sharpe would be a measurement of the fallback.
#
# So the prices are loaded with the same lookback the allocator was given on validation.
# `setup.yaml` declares `allocator_lookback: 240`, which is 240 eight-hourly bars, about
# eighty days. The loader leaves the start of the window unconstrained by that much and still
# caps the end at the canonical window end; the extra prefix is consumed by the rolling
# window and does not enter return aggregation, because the engine only aggregates over the
# rebalance timestamps the predictions carry.
#
# The conformal branch below is inert for this carrier and is kept because the carrier is
# resolved rather than fixed. `conformal_weighted` sizes from residuals the model has already
# made, and on the holdout there are none to use - every holdout return realises inside the
# window being evaluated - so a conformal carrier would calibrate from validation residuals
# with an embargo covering the label horizon. This one allocates from price moments and needs
# no calibration at all.

# %% tags=["results"]
strategy = strategy_view(json.loads(carrier["spec_json"]))
allocation = strategy.get("allocation") or {}
warmup = strategy_warmup_periods({"allocation": allocation} if allocation else {})
NEEDS_CALIBRATION = allocation.get("method") == "conformal_weighted"
embargo_steps = holdout_conformal_embargo_steps(CASE_STUDY_ID, LABEL) if NEEDS_CALIBRATION else 0
print(f"Allocator {allocation.get('method', 'equal_weight')!r}, warmup {warmup} bars")
if NEEDS_CALIBRATION:
    print(f"  conformal carrier: embargo {embargo_steps} observation(s), widths written below.")
else:
    print("  needs no calibration; sized from price moments over the warmup window.")

# %% [markdown]
# ## 3. The backtest
#
# The strategy specification is the carrier's own, re-pointed at the holdout prediction set
# and the holdout price window. Nothing else about it changes - the signal, the allocator and
# the `time_exit_20` overlay are carried across, and the commission and slippage are the
# levels `setup.yaml` declares, the same ones every validation number in this case study was
# net of and the same ones sitting inside the swept grid in [`16_costs`](16_costs.ipynb).
#
# The run registers under `stage='holdout'`, which the registry derives from the prediction
# set's split rather than from anything asserted here.
#
# The window carries one backtest at a time, for the same reason 17 lets it carry one
# prediction generation at a time. 17's guard is on the model - the training identity and the
# checkpoint - and it cannot see this one: a changed allocator, overlay or cost level
# produces the same holdout predictions and a different result from them.
#
# Two fields are rebuilt rather than carried. `input_identity` records the digests of the
# data a run actually read - here the price panel and the official funding settlements - and
# the carrier's copy describes the validation window. Cloning it produces a record that names
# inputs the run never touched, which is exactly what a consumer checking a price digest
# against the canonical one would refuse. They are derived from the holdout frames instead.
#
# The funding settlements are also passed to the runner, not merely digested. This case study
# is about a cashflow that accrues on holding rather than trading, every validation number in
# it is net of the funding the position actually paid or received, and a holdout run that
# omitted it would compare a strategy without its central economics against one with it.
#
# The test is the backtest hash, not a field-by-field comparison. Every input that changes
# the result is in that hash by construction, and a guard naming fields instead has to be
# right about all of them. The hash is resolved before anything runs, so nothing is evaluated
# on the holdout before the question is answered, and it comes from `backtest_run_status` -
# the call the runner itself makes - so the guard and the runner cannot disagree about
# identity. The run asserts they still agree afterwards, because a guard that had quietly
# stopped predicting the hash would let everything through while looking correct.

# %% tags=["results"]
prices = load_backtest_prices_for(
    CASE_STUDY_ID,
    LABEL,
    split="holdout",
    warmup_periods=warmup,
    max_symbols=MAX_SYMBOLS,
)
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
funding_rates = funding_rates_for_prices(prices)
print(f"Prices: {len(prices):,} rows, {prices['symbol'].n_unique():,} assets")
print(
    f"Funding: {funding_rates.height:,} settlements over "
    f"{funding_rates['symbol'].n_unique():,} names"
)
print(f"Predictions: {predictions.height:,} rows, {predictions['timestamp'].n_unique():,} stamps")

spec = ensure_backtest_spec(
    CASE_STUDY_ID,
    bt_config,
    json.loads(carrier["spec_json"]),
    prices=prices,
    prediction_hash=HOLDOUT_PREDICTION_HASH,
    initial_cash=bt_config.initial_cash,
)
spec["chapter"] = "ch20"
spec["input_identity"] = {
    **spec.get("input_identity", {}),
    "prices": value_digest(prices),
    "funding_rates": value_digest(funding_rates),
}
if NEEDS_CALIBRATION:
    spec = ensure_conformal_calibration_identity(spec, holdout_embargo_steps=embargo_steps)

spec["backtest_config"]["account"]["allow_short_selling"] = resolved_allow_short_selling(spec, None)
prospective_hash = backtest_run_status(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH, spec).backtest_hash
superseded_backtests = sorted(
    {
        row["backtest_hash"]
        for row in _registered_holdout_backtests(CASE_DIR, HOLDOUT_PREDICTION_HASH)
    }
    - {prospective_hash}
)
if superseded_backtests and not REPLACE_HOLDOUT:
    raise RuntimeError(
        "the holdout window already carries a backtest of a different configuration: "
        + ", ".join(superseded_backtests)
        + f". This run would register {prospective_hash} and has not run. Set "
        "REPLACE_HOLDOUT=True to discard the earlier one, or leave the selection where it was."
    )
for backtest_hash in superseded_backtests:
    print(f"REPLACING holdout backtest {backtest_hash}")
    _delete_holdout_backtest(CASE_DIR, backtest_hash)

# The guard has passed, so this run will register and any artifact it is sized by belongs
# beside this prediction set. Writing before the guard would replace the calibration a
# registered run was sized by, and the guard could still refuse afterwards.
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
    funding_rates=funding_rates,
    register=True,
    initial_cash=bt_config.initial_cash,
    calendar=bt_config.calendar,
)
if result.backtest_hash != prospective_hash:
    raise RuntimeError(
        f"the guard predicted {prospective_hash} and the runner registered "
        f"{result.backtest_hash}. The guard decides what may run on the holdout, so a guard "
        "that no longer reproduces the runner's identity is not a smaller problem than the "
        "one it was written for."
    )
print(f"Holdout backtest: {result.backtest_hash}")

# %% [markdown]
# ## 4. What it came out at
#
# The two numbers below are one strategy measured on two disjoint periods, and the gap
# between them is not an estimate of decay. The validation figure is the maximum of a ranking
# over thousands of backtests, so it carries the selection; the holdout figure is one
# measurement, so it carries its own sampling error. Both push the pair apart before any real
# change in the strategy's edge.
# [`19_strategy_analysis`](19_strategy_analysis.ipynb) is where they are given intervals and
# the selection correction.

# %% tags=["results"]
metrics = result.metrics
# The carrier's own registered Sharpe, not the resolver's. `resolve_solvent_carrier` reports
# the common-support figure, which re-ranks candidates on the timestamps every one of them
# covers; that is the right number for choosing between candidates and the wrong one to set
# beside a holdout measured over its own full window. Both are printed, so neither has to be
# inferred from the other.
with sqlite3.connect(str(CASE_DIR / "run_log" / "registry.db")) as conn:
    carrier_sharpe, carrier_periods = conn.execute(
        "SELECT sharpe, n_periods FROM backtest_metrics WHERE backtest_hash = ?",
        (carrier["val_backtest_hash"],),
    ).fetchone()

print(f"Validation Sharpe over its {int(carrier_periods):,} periods: {carrier_sharpe:.3f}")
print(f"  the same run re-ranked on common support: {carrier['val_sharpe']:.3f}")
print(
    f"Holdout Sharpe over {int(metrics['n_periods']):,} periods:      "
    f"{metrics.get('sharpe', float('nan')):.3f}"
)
print(
    f"Holdout: CAGR {metrics.get('cagr', float('nan')):.1%}, "
    f"max drawdown {metrics.get('max_drawdown', float('nan')):.2%}, "
    f"win rate {metrics.get('win_rate', float('nan')):.0%}"
)

# %% [markdown]
# ## What this notebook establishes, and what it does not
#
# It establishes a return series for the selected configuration over a period no choice in
# this case study was made on. That is the only thing a holdout can give.
#
# It does not establish that this configuration was the right one to carry here. The
# selection that brought it was made on validation, over a pool large enough that its maximum
# is optimistic by construction, and this notebook inherits that pool without correcting for
# it. The deflation is [`19_strategy_analysis`](19_strategy_analysis.ipynb)'s.
#
# The holdout stays re-runnable. If the selection changes, this generation is deleted and
# another is produced; it is not a resource that has been spent.
#
# **Next:** [`19_strategy_analysis`](19_strategy_analysis.ipynb).
