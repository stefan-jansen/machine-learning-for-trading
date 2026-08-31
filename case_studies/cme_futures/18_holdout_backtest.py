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
# # CME Futures: Holdout Backtest
#
# **Chapter 20 - Out-of-sample evaluation**
#
# [`17_holdout_predictions`](17_holdout_predictions.ipynb) refitted the selected
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
# The comparison to validation is printed but not interpreted. Two years of weekly
# decisions is on the order of a hundred observations - more than a monthly panel gives,
# and still few enough that the interval around a Sharpe estimated from them is wide.
# Saying what can be concluded from it is
# [`19_strategy_analysis`](19_strategy_analysis.ipynb)'s subject, with the intervals to
# say it.
#
# **Prerequisites:** [`17_holdout_predictions`](17_holdout_predictions.ipynb).
#
# **Scope:** one backtest. No selection, no comparison beyond a printed pair.

# %%
"""CME Futures: Holdout Backtest."""

import dataclasses
import json
import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
    load_contract_specs_from_yaml,
    load_futures_market_contract,
)
from case_studies.utils.backtest_presets import (
    ensure_backtest_spec,
    serializable_backtest_spec,
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
    canonical_json,
    compute_hash,
    read_predictions,
)
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "cme_futures"
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
    return [{"backtest_hash": backtest_hash} for (backtest_hash,) in rows]


# %% [markdown]
# ## 1. The configuration, and the predictions it produced on the holdout
#
# The carrier is resolved the same way [`16_costs`](16_costs.ipynb) and
# [`17_holdout_predictions`](17_holdout_predictions.ipynb) resolve it, so all three run
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
        "17_holdout_predictions first; this notebook does not fit."
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
# No validation observation is dropped at the boundary, and the reason is the label rather
# than a choice. The embargo exists because a residual observed at `t` measures a return
# realising over `(t, t+h]`, so with `h > 0` the last residuals of the validation span
# reach into the holdout window and would size holdout positions with holdout price
# information. This panel declares `h = 0D` - each row is dated by the month the return
# was earned - so the outcome is already realised at the observation and nothing reaches
# forward. The step count comes from the reviewed table in `conformal.py`, which records
# the label horizon; it carried 1 for these labels, which discarded the last month of
# calibration against a leak the label cannot have.
#
# The embargo is derived here because the backtest identity below is built from it. The
# widths themselves are NOT written here: writing them replaces the artifact the already
# registered run was sized by, and the replacement guard in section 3 can still refuse this
# run afterwards. That order left the registered holdout pointing at a calibration that no
# longer existed, so the write moved below the guard and nothing is overwritten until this
# run is cleared to register.

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
# The strategy specification is the carrier's own, re-pointed at the holdout prediction
# set and the holdout price window. Nothing else about it changes - the commission and
# slippage are the levels `setup.yaml` declares, the same ones every validation number in
# this case study was net of, and the same ones sitting inside the swept grid in
# [`16_costs`](16_costs.ipynb).
#
# The run registers under `stage='holdout'`, which the registry derives from the
# prediction set's split rather than from anything asserted here.
#
# One thing the hash does not cover: a conformal carrier reads its widths from an artifact
# beside the prediction set, and the backtest identity covers the allocator's declared
# parameters but not the calibration those widths were built from. Change the embargo and
# the hash does not move, so a registered run would be served back against inputs that no
# longer exist - and the registry refuses the overwrite rather than accepting either, which
# is how that state announces itself. Re-calibrating this case study's holdout therefore
# means deleting the registered run first, the same rule section 3 of
# [`17_holdout_predictions`](17_holdout_predictions.ipynb) applies to a superseded
# generation.

# %% tags=["results"]
prices = load_backtest_prices_for(CASE_STUDY_ID, LABEL, split="holdout", max_symbols=MAX_SYMBOLS)
predictions = read_predictions(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH)
# `13_backtest` records that reader-facing rows use `product` while the shared boundary
# converts to the engine's `symbol` key, so a price frame can arrive carrying either.
# Named from the frame rather than assumed, so this line cannot quietly report nothing.
_ENTITY_COL = next(c for c in ("product", "symbol") if c in prices.columns)
print(f"Prices: {len(prices):,} rows, {prices[_ENTITY_COL].n_unique():,} {_ENTITY_COL}s")
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
# Futures need their contract specifications, and this is the one place in the tail where
# they have to be restored by hand. `ensure_backtest_spec` is idempotent on an
# already-canonical spec: it deep-copies and refreshes the prediction hash and nothing
# else. That is correct for every other case study, which is why `etfs/19_holdout_backtest`
# does no more than this. cme is the exception - `research/strategy.py` loads contract
# specs for `cme_futures` alone, and it does so on the path that builds a spec from
# scratch, which a clone never enters.
#
# Two of the four entries are functions of the price frame and so belong to the holdout,
# not to the run this spec was cloned from. `futures_market` is loaded for the products
# actually priced, and if the holdout window prices a different set than validation did,
# the cloned value describes contracts this run does not trade. The specs themselves come
# from a static YAML and the entity contract is a fixed key mapping, so those two carry
# over unchanged - they are rewritten here anyway rather than relied on, because a spec
# assembled half from the clone and half from the holdout is the harder thing to check.
#
# Without this the engine receives no multipliers, tick sizes or margin schedules while
# the spec's hash goes on claiming it did. The result would not be a failure; it would be
# a holdout P&L in the wrong units, compared against validation numbers that had them.
contract_specs = load_contract_specs_from_yaml()
serialized_contract_specs = {
    symbol: dataclasses.asdict(contract_spec) for symbol, contract_spec in contract_specs.items()
}
futures_market = load_futures_market_contract(
    prices.get_column("symbol").unique().sort().to_list()
    if "symbol" in prices.columns
    else prices.get_column("product").unique().sort().to_list()
)
identity = spec.setdefault("input_identity", {})
identity["contract_specs"] = compute_hash(canonical_json(serialized_contract_specs))
identity["futures_market"] = compute_hash(canonical_json(futures_market))
# `prices` is the third entry and the same kind of mistake as the other two: cloned from the
# validation run, it is the digest of the validation price frame while this backtest consumes
# the holdout one. The record would say the run read prices it did not read, and
# `us_equities_panel/20_strategy_analysis.py` shows the shape of the consumer that checks
# exactly this.
#
# It is digested on the engine-keyed frame, which for cme is the reader frame with `product`
# renamed to `symbol` - the rename `research/strategy.py::_engine_prices` performs before
# `_build_spec` digests it, and the reason the digest cannot be taken off the frame as loaded.
engine_prices = prices.rename({"product": "symbol"}) if "product" in prices.columns else prices
identity["prices"] = value_digest(engine_prices)
spec["futures_market"] = futures_market
spec["entity_contract"] = {
    "reader_key": "product",
    "engine_key": "symbol",
    "mapping": "one_to_one_at_backtest_boundary",
}
# The embargo goes into the specification here, before anything hashes it. The widths are
# an input to this backtest and the embargo decides them, so two embargoes are two results
# and must not share an identity - which they did: changing it left the hash where it was
# and the registry refused to overwrite the registered run rather than accept either.
# Recorded by the notebook rather than inside `run_backtest`, because callers elsewhere
# construct and hash their own resolved specifications and compare the runner's answer to
# them; a runner that added a key after that would make those comparisons fail.
if NEEDS_CALIBRATION:
    spec = ensure_conformal_calibration_identity(spec, holdout_embargo_steps=embargo_steps)

# The window carries one backtest at a time, for the same reason `15` lets it carry one
# prediction generation at a time. `15`'s guard is on the model - the training identity and
# the checkpoint - and it cannot see this one: a changed allocator, overlay, cost level or
# calibration produces the same holdout predictions and a different result from them.
#
# The test is the backtest hash, not a field-by-field comparison. Every input that changes
# the result is in that hash by construction, and a guard naming fields instead has to be
# right about all of them - it was written first as a `strategy` comparison and missed the
# cost configuration and the calibration identity, both of which sit outside that block.
#
# The hash is resolved before anything runs, so nothing is evaluated on the holdout before
# the question is answered. It comes from `backtest_run_status`, which is the call the
# runner itself makes to decide whether a spec is already registered - asking it is the
# only way to be sure the guard and the runner agree about identity, and reconstructing the
# hash from parts here did not: it predicted f23ff90cf518 against the runner's b2acfd5420c8.
# The run asserts the two still agree afterwards, because a guard that had quietly stopped
# predicting the hash would let everything through while looking correct.
spec["backtest_config"]["account"]["allow_short_selling"] = resolved_allow_short_selling(spec, None)
prospective_hash = backtest_run_status(CASE_STUDY_ID, HOLDOUT_PREDICTION_HASH, spec).backtest_hash
superseded_backtests = sorted(
    {
        row["backtest_hash"]
        for row in _registered_holdout_backtests(CASE_DIR, HOLDOUT_PREDICTION_HASH)
    }
    - {prospective_hash}
)
if superseded_backtests:
    raise RuntimeError(
        "the holdout window already carries a backtest of a different configuration: "
        + ", ".join(superseded_backtests)
        + f". This run would register {prospective_hash} and has not run. Same rule as "
        "17_holdout_predictions: discarding the earlier result would not undo having "
        "observed it, so there is no switch here. Leave the selection where it was, or "
        "retire the earlier evaluation through the registry's lifecycle."
    )

# The guard has passed, so this run will register and the widths it is sized by are the
# ones that belong beside this prediction set.
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
        f"{widths[next(c for c in ('product', 'symbol') if c in widths.columns)].n_unique():,} "
        f"assets, embargo {embargo_steps} observation(s)"
    )
    print(f"  calibration_n: median {widths['calibration_n'].median():.0f}")

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
    contract_specs=contract_specs,
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
# between them is not an estimate of decay. The validation figure is the maximum of a
# ranking over more than a thousand backtests, so it carries the selection; the holdout
# figure is one measurement of twelve monthly returns, so it carries the sampling error of
# twelve observations. Both facts push the pair apart on their own, before any real change
# in the strategy's edge. [`19_strategy_analysis`](19_strategy_analysis.ipynb) is where
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
# correcting for it. The deflation is [`19_strategy_analysis`](19_strategy_analysis.ipynb)'s.
#
# The holdout stays re-runnable. If the selection changes, this generation is deleted and
# another is produced; it is not a resource that has been spent.
#
# **Next:** [`19_strategy_analysis`](19_strategy_analysis.ipynb).
