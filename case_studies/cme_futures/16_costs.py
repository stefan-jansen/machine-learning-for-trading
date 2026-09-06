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
# # CME Futures: Transaction-Cost Sensitivity
#
# This notebook prices one configuration: the single carrier the case study ships, resolved
# across labels from the immutable union of signal, allocation and risk-overlay results, with
# its risk overlay carried rather than cleared. It then applies the declared all-in cost grid
# to that fixed configuration. Commission and slippage each receive half of the grid value.
#
# One curve, not one per horizon. The carrier sits on a single label, so the other label
# produces no cost rows, and that is the intended state rather than a missing run.
#
# Cost sensitivity is not a selection stage. Its rows are excluded from the final selection pool.
# Contract multipliers, tick sizes, margin rates, front-contract position, roll adjustment, and
# product identity remain unchanged across the grid.

# %% [markdown]
# ## Why a strategy is priced rather than costed
#
# Every backtest before this one traded for free. That is not an oversight being corrected
# here - it is the only way the earlier comparisons could mean anything. A cost assumption
# applied during selection silently picks the strategy that best suits *that assumption*, and
# since nobody knows the true figure in advance, the result would be a configuration tuned to a
# guess. Selecting frictionlessly and pricing afterwards keeps the two decisions separate and
# visible.
#
# So this notebook does not ask "what does the strategy earn after costs?" as though there were
# one answer. It asks a more useful question: **at what cost level does the edge disappear?**
# The grid is swept because that breakeven point, not any single net Sharpe, is what a reader
# can compare against what they would actually pay.
#
# ### What the number covers
#
# Each grid value is an all-in per-trade cost in basis points, split evenly between commission
# and slippage. Those are different things and only the first is knowable in advance:
#
# - **Commission** is the broker and exchange fee. It is a published number, it is small for
#   liquid futures, and it does not depend on what the strategy does.
# - **Slippage** is the gap between the price that triggered the decision and the price actually
#   received. It depends on order size against available depth, on how quickly the position is
#   built, and on whether the trade is demanding liquidity or supplying it. It is the larger and
#   far less predictable half, and it is the reason a single figure has to be swept rather than
#   assumed.
#
# The even split between the two is a convention rather than a measurement, and it is worth
# naming as one. Nothing here estimates the commission and slippage components separately, so
# the split changes no result: the backtest charges the total. It exists so that the grid value
# reads as an all-in figure a reader can compare against their own all-in figure, rather than as
# a commission a reader might then add slippage to and double-count.
#
# A flat rate across thirty products is a deliberate simplification, and a reader sizing real
# positions should know which way it errs. Depth differs by orders of magnitude across this
# universe - the equity index contracts absorb size that would move a livestock or lumber
# contract several ticks - so one rate is optimistic for the thin products and pessimistic for
# the deep ones. It is the right choice here because the alternative is a per-product model
# whose assumptions would be doing more work than the measurement.
#
# ### Why turnover is what the cost multiplies
#
# Cost is charged per unit traded, not per unit held, so what a grid value buys is entirely
# decided by how much the strategy rebalances. Two configurations with identical gross returns
# can sit far apart on these curves if one of them churns.
#
# That gives this stage a second use beyond the headline breakeven. A signal whose edge survives
# realistic costs is one whose ranking is stable enough that yesterday's positions are mostly
# still today's; a signal that dies at a few basis points was earning its gross return by
# trading constantly. The first is a strategy and the second is a description of a data
# artifact, and no gross Sharpe distinguishes them.
#
# ### Why the risk overlay is carried rather than cleared
#
# The carrier is priced with its risk overlay in place, because the overlay changes the
# positions and therefore changes what is traded. Pricing a bare signal and then adding an
# overlay afterwards would cost a strategy nobody runs. This matters more here than it would
# elsewhere: an overlay that reduces position size in volatile periods also reduces turnover in
# exactly the periods where slippage is worst, so stripping it out would misstate the cost in a
# direction that flatters the strategy.

# %%
"""Run CME futures transaction-cost sensitivity on fixed configurations."""

import json

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    open_study,
    product_universe_table,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.research.population import supersedes_for_run
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from case_studies.utils.sweep_config import get_cost_grid_bps

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []

# The cost population is immutable under its name, so a run whose members have moved has to say
# which generation it retires. Anything upstream that changes a backtest identity moves them - a
# corrected label, a changed accounting field, a re-run after a registry reset - and
# `OfficialPopulation.create` refuses to write a different member list under a name that already
# exists. Declared as a literal so that running the committed notebook as it stands recomputes
# the population on record. Empty for a first snapshot.
COST_POPULATION = "cme_futures-cost-validation-v1"
SUPERSEDES_COST_POPULATION: str = ""

# %% [markdown]
# ## Fixed inputs
#
# There is one configuration to price, and the shared carrier selector below decides which
# label it is on. `PREVIEW_LABELS` is still validated rather than ignored: a preview run
# that names a label the case study does not declare is a mistake worth stopping, even
# though nothing here loops over the set.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS:
        raise ValueError("canonical execution cannot declare preview reductions")
elif EXECUTION_TIER == "preview":
    if WORKSPACE is None or not PREVIEW_LABELS:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_LABELS")
    unknown = sorted(set(PREVIEW_LABELS) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"preview labels this case study does not declare: {unknown}")
else:
    raise ValueError(f"unsupported execution tier: {EXECUTION_TIER!r}")
universe = product_universe_table()
universe

# %%
cost_grid = get_cost_grid_bps("cme_futures")
if not cost_grid:
    raise ValueError("the configured cost grid is empty")

# The configuration this case study reports, resolved once by the shared selector rather
# than ranked again here. `resolve_solvent_carrier` goes through
# `resolve_canonical_rank1_lineage`, which ranks across the signal, allocation and
# risk-overlay stages together, re-ranks conformal candidates on exact common timestamp
# support and applies LABEL_RESTRICTIONS, UNIVERSE_RESTRICTIONS and CARRIER_PINS. A plain
# Sharpe ranking beside it does none of those, and where the two disagree this notebook
# prices a strategy the chapter does not report while `19_strategy_analysis` finds no cost
# rows for the carrier it selected. It also refuses a carrier whose equity reached zero,
# whose Sharpe would be computed on a balance that no longer exists.
#
# This replaces a per-label loop that ran a cost grid for each label off the pre-overlay
# allocation results. That was wrong twice over: there is one strategy, not one per label,
# and it read the stage before risk management, so the ladder priced the pre-overlay winner
# rather than the configuration that is actually shipped.
carrier = resolve_solvent_carrier("cme_futures")
strategy = json.loads(carrier["spec_json"])["strategy"]
selected_label = carrier["label"]
prediction_hash = carrier["val_prediction_hash"]
print(
    f"Pricing the canonical validation rank-1: {carrier['val_backtest_hash']} "
    f"({carrier['family']}/{carrier['config_name']}) on {selected_label}, "
    f"from the {carrier['val_stage']} stage, validation Sharpe {carrier['val_sharpe']:.3f}, "
    f"max drawdown {carrier['max_drawdown']:.3f}."
)
print(
    f"  It carries {'a' if strategy.get('risk') else 'no'} risk overlay, and that block "
    "travels with it below rather than being cleared."
)

request_rows = []
for total_cost_bps in cost_grid:
    request_rows.append(
        {
            "request_name": f"{carrier['val_backtest_hash']}-cost-{total_cost_bps:g}",
            "prediction_hash": prediction_hash,
            "label": selected_label,
            "signal": strategy["signal"],
            "allocation": strategy.get("allocation"),
            # Carried, not cleared. The configuration being priced is the one that came out
            # of risk management, and dropping its overlay here would price a strategy
            # nobody selected.
            "risk": strategy.get("risk"),
            "costs": {
                "commission_bps": total_cost_bps / 2,
                "slippage_bps": total_cost_bps / 2,
            },
            "chapter": "ch18",
        }
    )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "costs")

# %% [markdown]
# ## Execute the declared grid
#
# Expected identities are snapshotted before execution. An empty input, failed grid member, missing
# sidecar, or incomplete lineage fails the notebook instead of reporting a smaller population.
#
# Failing rather than reporting a smaller population is the important half of that sentence. A
# cost curve with a missing point still plots, and it plots as a curve that bends somewhere it
# does not really bend - so the one output a reader would take a breakeven from is exactly the
# one that hides an incomplete run. There is no partial answer to this question worth having.
#
# The results are written as a named population for the same reason the baseline sweep is, and
# `SUPERSEDES_COST_POPULATION` above is how a re-run says which generation it retires. A cost
# curve is quoted in the chapter, so the set of backtests behind a quoted number has to stay
# recoverable after the numbers move.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name=COST_POPULATION if EXECUTION_TIER == "canonical" else None,
    supersedes=supersedes_for_run(
        study,
        population_name=COST_POPULATION,
        declared=SUPERSEDES_COST_POPULATION or None,
        execution_tier=EXECUTION_TIER,
    ),
)

# %% [markdown]
# `source` says whether each member was computed by this run or served from the registry because
# an identical identity was already recorded. A re-run of a registered sweep is entirely `reused`
# and completes in seconds; without the column that is indistinguishable from having computed
# every row.

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# `19_strategy_analysis` may describe the cost curve, but these backtests do not participate in
# configuration selection.
#
# ## Reading the curve
#
# The quantity to take from this notebook is the cost level at which the net Sharpe reaches
# zero, not the net Sharpe at any particular level. That breakeven is a property of the
# strategy; the figure a given reader pays is a property of their broker, their size and their
# execution, and only they know it.
#
# Two failure modes to keep apart when reading it. A strategy that breaks even far above any
# plausible cost is genuinely robust to friction. A strategy that breaks even just above the
# plausible range is not "profitable but tight" - it is a strategy whose result depends on an
# assumption nobody can pin down, and the honest description of it says so rather than quoting
# the frictionless number with a footnote.
#
# What this stage cannot tell you is the cost of trading at a size the market notices. Every
# figure here is linear in turnover: doubling the position doubles the cost. Real slippage grows
# faster than that once an order is large against available depth, so the curve is a lower bound
# on what scale costs, and the point where it crosses zero is optimistic for anyone trading
# size. Nothing in this case study measures market impact, and reading the breakeven as a
# capacity limit would be reading it for something it does not contain.
