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
# # S&P 500 Options: The Risk-Overlay Boundary
#
# In the other case studies this stage adds a risk overlay: a rule sitting on top of the
# allocator's weights that caps a position, scales the book down after a drawdown, or targets a
# volatility. The overlay is expressed as a target-weight transformation, which works because in
# those case studies a position is a quantity of one instrument and its risk moves with that
# quantity.
#
# A short straddle does not have that shape. Its exposure is two option legs whose sensitivity
# changes as the underlying moves and as expiration approaches, plus a hedge position in the
# underlying that is rebalanced daily against exactly that drift. Scaling a target weight would
# leave the leg pairing, the settlement and the hedge accounting untouched, so the resulting
# position would be a different instrument from the one whose risk was being managed. The controls
# that do govern this strategy - the delta-hedge threshold, the settlement convention, the entry
# cost model, how many weekly cohorts run at once - are part of the strategy specification itself
# and were set in `12_backtest` and `14_costs`.
#
# So this case study declares no risk-overlay variants, and this notebook is where that is
# checked rather than assumed. It resolves the candidate set that came out of
# `13_portfolio_management`, shows that the configured risk request set is empty, demonstrates that
# a risk request would be refused if one were configured, and confirms it wrote nothing.
#
# **Learning objectives**
#
# - Recognise when a generic portfolio control cannot be applied to an instrument, and say what
#   about the instrument makes it inapplicable.
# - Read a stage that deliberately produces no results, and check that claim against the registry
#   rather than against the notebook's own narration.
#
# **Book reference**: Chapter 19
#
# **Prerequisites**: the finalized candidate set published by `13_portfolio_management`.

# %%
"""Validate the empty S&P 500 options risk-overlay request boundary."""

import polars as pl

from case_studies.research import CandidateSet
from case_studies.sp500_options.research_workflow import (
    open_study,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import (
    get_portfolio_risk_controls,
    get_position_risk_controls,
)

CASE_STUDY = "sp500_options"
STRATEGY_CANDIDATES = "sp500-options-strategy-candidates-v1"

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %% [markdown]
# ## The candidate set that passes through
#
# Every member is required to be a complete backtest before the set is allowed to move on, so a
# partial result cannot reach selection by being carried through a stage that does nothing.

# %%
if EXECUTION_TIER != "canonical":
    raise ValueError("risk-boundary validation requires the canonical candidate set")
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
candidates = CandidateSet.one(study, name=STRATEGY_CANDIDATES)
if candidates.member_kind != "backtest":
    raise TypeError("the finalized strategy candidate set must contain backtests")
backtests_before = study.backtests.table()
members = backtests_before.filter(pl.col("backtest_hash").is_in(candidates.members))
if members.height != len(candidates.members) or members.filter(~pl.col("complete")).height:
    raise RuntimeError("the finalized strategy candidate set is incomplete")

# %% tags=["results"]
pl.DataFrame(
    {
        "candidate_set": [candidates.name],
        "candidate_set_hash": [candidates.hash],
        "member_count": [len(candidates.members)],
        "stages": [", ".join(sorted(members.get_column("stage").unique().to_list()))],
    }
)

# %% [markdown]
# ## The configured risk requests
#
# Position-scope controls act on one holding, portfolio-scope controls act on the book. Both lists
# come from `config/setup.yaml`, and both are empty for this case study. Reading them rather than
# writing the emptiness into the notebook is what makes this a check: adding a control to the
# configuration makes the cell below raise instead of quietly running an overlay the option path
# cannot represent.

# %%
risk_rows = [
    {"scope": "position", **request} for request in get_position_risk_controls(CASE_STUDY)
] + [{"scope": "portfolio", **request} for request in get_portfolio_risk_controls(CASE_STUDY)]
risk_requests = (
    pl.DataFrame(risk_rows)
    if risk_rows
    else pl.DataFrame(schema={"scope": pl.String, "name": pl.String, "method": pl.String})
)
if not risk_requests.is_empty():
    raise RuntimeError("risk variants require an implemented typed options path before execution")
risk_requests

# %% [markdown]
# ## What happens to a risk request that is submitted anyway
#
# The refusal lives in the execution path, not in this notebook, so it holds for a reader who
# writes their own request as well. The cell below builds one against the highest-Sharpe candidate
# and confirms it is rejected before anything is fitted or written.

# %%
probe_member = members.sort("sharpe", "backtest_hash", descending=[True, False]).row(0, named=True)
probe = strategy_request_frame(
    [
        {
            "request_name": "risk-overlay-probe",
            "prediction_hash": probe_member["prediction_hash"],
            "label": probe_member["label"],
            "signal": {"method": "equal_weight_top_k", "top_k": 5, "universe_filter": "liquid"},
            "allocation": None,
            "risk": {"name": "position_cap", "method": "max_weight", "max_weight": 0.1},
            "costs": None,
            "chapter": "ch19",
        }
    ]
)
try:
    run_official_backtest_requests(study, probe, population_name=None)
except ValueError as refusal:
    print(f"risk request refused: {refusal}")
else:
    raise RuntimeError("the option execution path accepted a risk overlay it cannot represent")

# %% [markdown]
# ## Nothing was written
#
# The registry is read back and compared against the snapshot taken before the probe. This is the
# claim the stage makes, so it is checked against the store rather than against a counter this
# notebook keeps.

# %% tags=["results"]
backtests_after = study.backtests.table()
if backtests_after.height != backtests_before.height:
    raise RuntimeError("the empty risk boundary wrote a backtest result")
if set(backtests_after.get_column("backtest_hash")) != set(
    backtests_before.get_column("backtest_hash")
):
    raise RuntimeError("the empty risk boundary changed the published backtest set")
pl.DataFrame(
    {
        "check": ["configured risk requests", "backtests before", "backtests after"],
        "value": [
            str(risk_requests.height),
            str(backtests_before.height),
            str(backtests_after.height),
        ],
    }
)

# %% [markdown]
# ## Key takeaways
#
# - A portfolio control is defined against a representation of a position. When the instrument's
#   risk lives in a leg structure and a rebalancing hedge rather than in a quantity, a
#   target-weight overlay changes the position without changing its risk.
# - A stage that produces nothing still has to prove it, and the proof is the store's contents
#   before and after, not a statement in the notebook.
# - Refusing an unsupported request in the shared execution path, rather than in the notebook, is
#   what makes the boundary hold for a reader's own requests too.
#
# **Known limitations**: this says nothing about whether risk controls on a short-volatility book
# are a good idea, only that the generic target-weight form cannot express them here. Implementing
# them would mean an option-aware overlay acting on cohort membership, contract selection or the
# hedge rule, and that is a change to the strategy specification rather than a stage on top of it.
