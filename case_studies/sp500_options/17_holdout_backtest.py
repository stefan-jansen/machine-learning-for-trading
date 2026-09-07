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
# # S&P 500 Options: Holdout Backtest
#
# **Chapter 20 - Out-of-sample evaluation**
#
# [`16_holdout_predictions`](16_holdout_predictions.ipynb) refitted the selected
# configuration on the history before the holdout window and wrote its predictions over it.
# This notebook writes straddles from them, with the concentration, the allocator, the entry
# schedule, the hedge rule and the cost assumption the rest of the case study used, and
# registers the result.
#
# Nothing is chosen here. The predictions, the allocator, the concentration, the weekly entry
# schedule, the delta-hedge threshold and the per-leg charges all arrive fixed from earlier
# notebooks, and the only thing this notebook decides is that they are applied unchanged. That
# is the whole design: a holdout result is worth something exactly to the extent that no
# decision was made after seeing it, and every knob left open here would be a decision.
#
# The comparison to validation is printed but not interpreted. The validation figure is the
# maximum of a ranking over the whole sweep and the holdout figure is one measurement over a
# single year; what can be said about the gap between them is
# [`18_strategy_analysis`](18_strategy_analysis.ipynb)'s subject, with the intervals to say
# it.
#
# **Prerequisites:** [`16_holdout_predictions`](16_holdout_predictions.ipynb).
#
# **Scope:** one backtest. No selection, no comparison beyond a printed pair.

# %%
"""S&P 500 Options: Holdout Backtest."""

import json
import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import supersedes_for_run
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.sp500_options.research_workflow import (
    open_study,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.backtest_presets import strategy_view
from case_studies.utils.registry import training_hash_from_spec
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_options"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
# The generation of the holdout population this run retires. A population is immutable under
# its name, so a re-run whose member has moved has to name the one it replaces; the refusal
# prints the current hash. Empty is correct only for a name this registry has never held.
# It moves whenever `16_holdout_predictions` refits a different configuration, which is what
# an upstream correction does - the notebook carried no declaration at all, so the first such
# correction left it unable to register the backtest it had just run.
SUPERSEDES_HOLDOUT_POPULATION: str = ""

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
REGISTRY = CASE_DIR / "run_log" / "registry.db"

# %% [markdown]
# ## 1. The configuration, and the predictions it produced on the holdout
#
# The carrier is resolved the same way [`15_costs`](15_costs.ipynb) and
# [`16_holdout_predictions`](16_holdout_predictions.ipynb) resolve it, so all three run the
# same configuration by construction rather than by a hash copied between them.
#
# Which holdout prediction set belongs to it is derived rather than searched for. Re-deriving
# the holdout training specification reproduces the training identity 16 registered - the
# derivation is deterministic and the identity covers it - so the prediction set is looked up
# by that identity and the carrier's checkpoint. A search over holdout prediction sets would
# have to guess which one belonged to this configuration, and this registry has held holdout
# prediction sets that belonged to no refit at all.

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

with sqlite3.connect(str(REGISTRY)) as conn:
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
        "16_holdout_predictions first; this notebook does not fit."
    )
HOLDOUT_PREDICTION_HASH = match[0]

print(f"Carrier:            {carrier['val_backtest_hash']}  {carrier['config_name']} ({LABEL})")
print(f"Holdout training:   {holdout_training_hash}")
print(f"Holdout prediction: {HOLDOUT_PREDICTION_HASH}")

# %% [markdown]
# ## 2. The request, and what may run on the window
#
# The strategy is the carrier's own, read off its registered specification and re-pointed at
# the holdout prediction set. `signal` carries the entry schedule, the liquid-universe filter
# and the concentration; `allocation` carries the sizing. Neither is retyped here, so a
# rebuilt sweep cannot leave this notebook running last quarter's strategy.
#
# **The window carries one backtest.** 16's guard is on the model - the training identity and
# the checkpoint - and it cannot see this one: a changed allocator, concentration or cost
# level produces the same holdout predictions and a different result from them. Two things
# enforce it, and they catch different mistakes. The check below refuses a holdout-stage
# backtest that belongs to some other prediction set, which is a second *configuration*
# measured on the window. The official population this run publishes refuses a second
# *strategy* over the same predictions: the name already exists with a different member list,
# and `create` will not overwrite it without being told which generation it replaces, which
# this notebook does not say.
#
# Both refuse rather than offering a replacement switch. Deleting a result that has been seen
# does not unsee it, and the selection bias it introduces is not removed by removing the rows.

# %%
with sqlite3.connect(str(REGISTRY)) as conn:
    foreign_holdout_backtests = [
        backtest_hash
        for (backtest_hash,) in conn.execute(
            """
            SELECT b.backtest_hash FROM backtest_runs b
            JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
            WHERE p.split = 'holdout' AND b.prediction_hash != ?
            ORDER BY b.backtest_hash
            """,
            (HOLDOUT_PREDICTION_HASH,),
        ).fetchall()
    ]
if foreign_holdout_backtests:
    raise RuntimeError(
        "the holdout window already carries a backtest of a different configuration: "
        + ", ".join(foreign_holdout_backtests)
        + ". This run has not started. Same rule as 16_holdout_predictions: discarding the "
        "earlier result would not undo having observed it, so there is no switch here. Leave "
        "the selection where it was, or retire the earlier evaluation through the registry's "
        "lifecycle."
    )

carrier_strategy = strategy_view(json.loads(carrier["spec_json"]))
SIGNAL = carrier_strategy["signal"]
ALLOCATION = carrier_strategy.get("allocation")
HOLDOUT_POPULATION = f"{CASE_STUDY_ID}-holdout-{LABEL}"
print(f"Signal:     {SIGNAL}")
print(f"Allocation: {ALLOCATION}")

# %% [markdown]
# ## 3. The backtest
#
# The run goes through `run_official_backtest_requests`, the same call the validation sweeps
# use, with `split="holdout"` naming the price window. It is a parameter rather than a second
# code path on purpose: the entry schedule, the delta hedge and the settlement at intrinsic
# value are identical in both windows and only the prices differ, so a holdout-specific copy of
# that function would be a copy that can drift from the one every validation number came from.
#
# What that call does here is what it does everywhere in this case study. It resolves the short
# straddles the predictions imply, replays that resolution in a fresh interpreter and refuses
# to continue if the two disagree, publishes the decision artifact, computes the backtest
# identity before executing, and registers the run under it.
#
# The run registers under `stage='holdout'`, which the registry derives from the prediction
# set's split rather than from anything asserted here. That is checked below rather than
# trusted: if the inference ever stopped taking precedence, the whole out-of-sample result
# would land in `allocation` and 18 would find no holdout at all, which is a failure two
# notebooks away from its cause.

# %% tags=["results"]
requests = strategy_request_frame(
    [
        {
            "request_name": f"holdout-{carrier['config_name']}",
            "prediction_hash": HOLDOUT_PREDICTION_HASH,
            "label": LABEL,
            "signal": SIGNAL,
            "allocation": ALLOCATION,
            "chapter": "ch20",
        }
    ]
)
execution = run_official_backtest_requests(
    study,
    requests,
    population_name=HOLDOUT_POPULATION,
    split="holdout",
    supersedes=supersedes_for_run(
        study,
        population_name=HOLDOUT_POPULATION,
        declared=SUPERSEDES_HOLDOUT_POPULATION or None,
        execution_tier=EXECUTION_TIER,
    ),
)
result = execution.results[0]
print(f"Holdout backtest: {result.hash}")
execution.catalog_rows

# %% [markdown]
# The registered run is read back rather than described from the request. The strategy view of
# what was registered has to equal the carrier's, or this is a different strategy wearing the
# carrier's name - the specification is rebuilt from `signal` and `allocation` here, and the
# rebalance cadence and cost levels it fills in around them come from `setup.yaml`, so the
# comparison is what establishes that they filled in the same way they did on validation.

# %% tags=["results"]
with sqlite3.connect(str(REGISTRY)) as conn:
    registered_stage, registered_spec_json = conn.execute(
        "SELECT stage, spec_json FROM backtest_runs WHERE backtest_hash = ?", (result.hash,)
    ).fetchone()
if registered_stage != "holdout":
    raise RuntimeError(
        f"the holdout backtest registered under stage={registered_stage!r} rather than "
        "'holdout'; the split-based inference in registry.store._infer_stage did not take "
        "precedence over this carrier's allocation block"
    )
registered_strategy = strategy_view(json.loads(registered_spec_json))
if registered_strategy != carrier_strategy:
    raise RuntimeError(
        "the holdout backtest did not register the carrier's strategy. Carrier: "
        f"{carrier_strategy}. Registered: {registered_strategy}."
    )
print(f"stage={registered_stage}, strategy matches the carrier")
print(f"Official population {HOLDOUT_POPULATION}: {execution.population.hash}")

# %% [markdown]
# ## 4. What it came out at
#
# The two numbers below are one strategy measured on two disjoint periods, and the gap between
# them is not an estimate of decay. The validation figure is the maximum of a ranking over the
# whole sweep, so it carries the selection; the holdout figure is one measurement over a single
# year, so it carries that window's sampling error. Both push the pair apart on their own,
# before any real change in the strategy's edge - and this carrier's validation Sharpe is
# already negative, so what the holdout can confirm or disturb is a negative result rather than
# a positive one.
#
# [`18_strategy_analysis`](18_strategy_analysis.ipynb) is where they are given intervals and a
# paired comparison.

# %% tags=["results"]
METRIC_COLUMNS = ("sharpe", "cagr", "max_drawdown", "win_rate", "n_periods")
with sqlite3.connect(str(REGISTRY)) as conn:
    columns = ", ".join(METRIC_COLUMNS)
    carrier_metrics, holdout_metrics = (
        dict(
            zip(
                METRIC_COLUMNS,
                conn.execute(
                    f"SELECT {columns} FROM backtest_metrics WHERE backtest_hash = ?",
                    (backtest_hash,),
                ).fetchone(),
                strict=True,
            )
        )
        for backtest_hash in (carrier["val_backtest_hash"], result.hash)
    )

# The carrier's own registered Sharpe, not the resolver's. `resolve_solvent_carrier` reports the
# common-support figure, which re-ranks candidates on the timestamps every one of them covers;
# that is the right number for choosing between candidates and the wrong one to set beside a
# holdout measured over its own full window. Both are printed, so neither has to be inferred
# from the other.
#
# `num_trades` is deliberately absent. The HTM cohort engine settles each straddle at expiration
# rather than rebalancing a book, so it registers no trade count for either run, and printing a
# zero would read as "it stopped trading" rather than "this engine does not count trades".
for name, metrics in (("Validation", carrier_metrics), ("Holdout   ", holdout_metrics)):
    print(
        f"{name} over {int(metrics['n_periods']):,} sessions: Sharpe "
        f"{metrics['sharpe']:.3f}, CAGR {metrics['cagr']:.1%}, "
        f"max drawdown {metrics['max_drawdown']:.2%}, win rate {metrics['win_rate']:.0%}"
    )
print(f"  the validation run re-ranked on common support: {carrier['val_sharpe']:.3f}")

# %% [markdown]
# ## What this notebook establishes, and what it does not
#
# It establishes a return series for the selected configuration over a period no choice in this
# case study was made on. That is the only thing a holdout can give, and it is worth less than
# it looks: one year of weekly straddle cohorts is a small number of independent observations,
# which is too few to separate a strategy that decayed from one that had an ordinary year.
#
# It does not establish that this configuration was the right one to carry here. The selection
# that brought it was made on validation, over a pool large enough that its maximum is
# optimistic by construction, and this notebook inherits that pool without correcting for it.
# The deflation is [`18_strategy_analysis`](18_strategy_analysis.ipynb)'s.
#
# Re-running this notebook against the same carrier is free and idempotent - the backtest
# identity is unchanged and the registered run is served back. A different carrier is refused,
# for the reason 16 gives.
#
# **Next:** [`18_strategy_analysis`](18_strategy_analysis.ipynb).
