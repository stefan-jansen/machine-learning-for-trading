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
# A short straddle does not have that shape here. Scaling the number of contracts would in fact
# scale the legs, the hedge, the costs and the dollar Greeks together, so the objection is not that
# option risk is independent of quantity. It is that the option engine never sees a quantity. It
# holds five weekly cohorts at a fixed fifth of capital each and normalizes the weights inside a
# cohort to sum to one, so an overlay that scales those weights down is renormalized straight back
# up: there is no cash position for the book to move into. The execution path therefore refuses a
# risk block rather than accept one it would silently discard. The controls that do govern this
# strategy - the delta-hedge threshold, the settlement convention, the entry cost model, how many
# weekly cohorts run at once - are fields of the strategy specification itself, fixed in
# `12_backtest`; `15_costs` afterwards varies one of them to measure what the result depends on.
#
# So this case study declares no risk-overlay variants, and this notebook is where that is
# checked rather than assumed. It resolves the candidate set that came out of
# `13_portfolio_management`, shows that the configured risk request set is empty, demonstrates that
# a risk request would be refused if one were configured, and confirms it wrote nothing.
#
# This is the third of the four backtest stages, and the last one that could add a run to the
# candidate pool. It registers none, so the pool `15_costs` prices and `16_strategy_analysis`
# reports is the one `13_portfolio_management` left. Costs runs after this notebook rather than
# beside it so that the last stage to select is the last stage to run.
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
# **Prerequisites**: the finalized candidate set published by
# [`13_portfolio_management`](13_portfolio_management.ipynb), and through it
# [`12_backtest`](12_backtest.ipynb).

# %%
"""Validate the empty S&P 500 options risk-overlay request boundary."""

import polars as pl

from case_studies.research import CandidateSet, Result
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
#
# The probe opens that candidate and copies its strategy verbatim, so the risk block is the only
# thing about the request that is new. Substituting a signal of the notebook's own would make the
# refusal a statement about that substitute rather than about a candidate the pipeline produced.

# %%
probe_member = members.sort("sharpe", "backtest_hash", descending=[True, False]).row(0, named=True)
probe_strategy = Result.open(study, probe_member["backtest_hash"]).spec()["strategy"]
probe = strategy_request_frame(
    [
        {
            "request_name": "risk-overlay-probe",
            "prediction_hash": probe_member["prediction_hash"],
            "label": probe_member["label"],
            "signal": probe_strategy["signal"],
            "allocation": probe_strategy.get("allocation"),
            "risk": {"name": "position_cap", "method": "max_weight", "max_weight": 0.1},
            "costs": probe_strategy.get("costs"),
            "chapter": "ch19",
        }
    ]
)
try:
    run_official_backtest_requests(study, probe, population_name=None)
except ValueError as refusal:
    # The refusal has to name the risk overlay. The request also carries the candidate's costs
    # block, which this path refuses separately, so accepting any ValueError would let a refusal
    # about costs be reported as the risk boundary holding.
    if "risk overlay" not in str(refusal):
        raise RuntimeError(
            f"the request was refused for something other than risk: {refusal}"
        ) from refusal
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
# - A portfolio control is defined against a representation of a position. When the engine holds a
#   fully invested book of normalized cohort weights, there is no quantity for a target-weight
#   overlay to act on, whatever the instrument.
# - A stage that produces nothing still has to prove it, and the proof is the store's contents
#   before and after, not a statement in the notebook.
# - Refusing an unsupported request in the shared execution path, rather than in the notebook, is
#   what makes the boundary hold for a reader's own requests too.
#
# **Known limitations**: this says nothing about whether risk controls on a short-volatility book
# are a good idea, only that the generic target-weight form cannot express them here. Implementing
# them would mean an option-aware overlay acting on cohort membership, contract selection or the
# hedge rule, and that is a change to the strategy specification rather than a stage on top of it.
