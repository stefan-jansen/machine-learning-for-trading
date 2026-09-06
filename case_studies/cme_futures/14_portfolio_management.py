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
# # CME Futures: Portfolio Allocation
#
# The baseline stage ranks complete configurations by equal-weight validation backtest Sharpe. For
# each label, this notebook retains the strongest checkpoint and signal concentration for each of
# the configured number of distinct model configurations, then evaluates the declared alternative
# position sizing methods. Equal weight is not among them: it is the baseline itself, and because
# `stage` is not part of `backtest_hash`, running it again here produces a row hashing
# identically to its baseline parent, so one of the two is silently lost. Measured in this
# case study's own pre-rebuild store: 48 rows stamped `stage='signal'` while carrying
# `allocation.method='equal_weight'`, and no allocation-stage equal-weight rows at all.
#
# All allocator lookbacks come from the case-study configuration. The official population is fixed
# before execution; machine speed and caught failures cannot change which allocators run.

# %% [markdown]
# ## What allocation decides, and why it is not a detail
#
# The baseline held every selected product in the same size. That is a real choice, not the
# absence of one, and it says something specific: that the signal's ranking carries information
# about *which* products to hold and nothing about *how much* of each.
#
# An allocator disputes that. It sizes positions by some property the ranking does not capture -
# how volatile a product has been, how it co-moves with the others, how confident the model was.
# The claim is that two products the signal ranks equally are not equally worth the same dollar
# risk.
#
# For futures this is a larger effect than it would be in equities, and the reason is in the
# instruments. A gold contract and a natural-gas contract with equal notional exposure carry
# entirely different risk, because their volatilities differ by a wide margin and they do not
# move together. An equal-weight book of futures is therefore not a neutral book: it is one
# whose realized risk is dominated by whichever contracts happen to be most volatile, and its
# overall behaviour can be driven by two or three positions out of thirty regardless of what the
# signal said.
#
# ### What the alternatives are actually doing
#
# The declared methods differ in how much they estimate, and that is the axis to read them on.
# Sizing inversely to a product's own volatility uses one number per product and no relationship
# between them - it equalizes risk contribution under the assumption that the products are
# independent, which they are not, but it is robust because a single volatility is estimated
# accurately from little data. Methods that use the covariance between products can in principle
# do better, because diversification is a property of the relationships rather than of any one
# series. In practice they must estimate far more quantities from the same history, and a
# covariance matrix estimated from a short window is dominated by noise that the optimizer then
# treats as signal - which is why the more sophisticated method is not reliably the better one
# and why they are compared here rather than assumed.
#
# ### Why the lookbacks are declared in configuration
#
# Every allocator reads a history to estimate from, and the length of that history is a free
# parameter that materially changes the result: a short window tracks a regime change quickly
# and is noisy, a long one is stable and stale. Choosing it by validation Sharpe would be fitting
# the allocator to the same path it is then assessed on, and the improvement would be
# indistinguishable from a real one. The lookbacks come from `config/setup.yaml` for the same
# reason the risk-overlay parameters do.
#
# ### Why equal weight is excluded here rather than re-run
#
# Equal weight is the baseline, and the paragraph above the parameter cell records what happens
# if it is run again in this stage: because `stage` is not part of `backtest_hash`, the row
# hashes identically to its baseline parent and one of the two is silently lost. Measured in
# this case study's own pre-rebuild store, 48 rows carried `stage='signal'` while declaring
# `allocation.method='equal_weight'`, and there were no allocation-stage equal-weight rows at
# all. The comparison a reader wants - allocator against equal weight - is made against the
# baseline population, not by recomputing it here.
#
# ### These candidates stay eligible
#
# Allocation results are part of the final selection pool, so an allocator that genuinely
# improves validation Sharpe can be what the case study ships. One consequence to carry forward:
# every allocator except equal weight re-sizes positions as its estimates move, which is
# turnover the baseline did not have. `16_costs` is where that is priced, and an allocator's
# advantage here is a gross number.

# %%
"""Run the declared CME futures allocation population."""

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    create_label_candidate_sets,
    open_study,
    product_universe_table,
    run_official_backtest_requests,
    shortlist_signal_configurations,
    strategy_request_frame,
)
from case_studies.research.population import supersedes_for_run
from case_studies.utils.sweep_config import get_allocators, get_top_n_predictions

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []
PREVIEW_MAX_BASELINE_ROWS = 0

# The allocation population is immutable under its name, so a run whose members have moved has
# to say which generation it retires. Anything upstream that changes a backtest identity moves
# them - a corrected label, a changed accounting field, a re-run after a registry reset - and
# `OfficialPopulation.create` refuses to write a different member list under a name that already
# exists. Declared as a literal so that running the committed notebook as it stands recomputes
# the population on record. Empty for a first snapshot.
ALLOCATION_POPULATION = "cme_futures-allocation-validation-v1"
SUPERSEDES_ALLOCATION_POPULATION: str = ""

# %% [markdown]
# ## Select signal configurations by validation Sharpe
#
# The shortlist is deterministic. It scans the immutable signal candidate set in descending Sharpe
# order with the backtest identity as tie-break, and keeps one exact checkpoint and strategy per
# distinct `(family, config_name)` pair.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS or PREVIEW_MAX_BASELINE_ROWS:
        raise ValueError("canonical execution cannot declare preview reductions")
    labels = ALL_LABELS
elif EXECUTION_TIER == "preview":
    if WORKSPACE is None or not PREVIEW_LABELS or PREVIEW_MAX_BASELINE_ROWS < 1:
        raise ValueError(
            "preview execution requires WORKSPACE, PREVIEW_LABELS and PREVIEW_MAX_BASELINE_ROWS"
        )
    unknown = sorted(set(PREVIEW_LABELS) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"preview labels this case study does not declare: {unknown}")
    labels = tuple(PREVIEW_LABELS)
else:
    raise ValueError(f"unsupported execution tier: {EXECUTION_TIER!r}")
universe = product_universe_table()
universe

# %% [markdown]
# ## How many baseline configurations the position sizing methods run on
#
# **Why a shortlist rather than every baseline row.** Allocation is applied to the strongest
# configurations rather than to all of them, and the reason is that the question is about the
# allocator, not about the configuration underneath it. Running every allocator against every
# baseline row would multiply the population by the number of methods and answer a question
# nobody asked, while making the effective number of trials far larger - which the deflated
# Sharpe downstream then has to divide by. Keeping the strongest checkpoint and concentration
# per configuration asks the allocator's question against the strategies that would otherwise
# have shipped.
#
# Canonical takes the shortlist size from `setup.yaml`, which is the declared width of the
# allocation stage. A preview cannot: it backtests a bounded slice of the baseline stage, so the
# canonical width names more distinct configurations than its pool contains and
# `shortlist_signal_configurations` refuses - correctly, since silently returning fewer is the
# quiet shrinking that strictness exists to prevent. The preview therefore declares its own
# width, and is held to it just as strictly.

# %%
shortlist_size = (
    get_top_n_predictions("cme_futures", "allocation")
    if EXECUTION_TIER == "canonical"
    else PREVIEW_MAX_BASELINE_ROWS
)
allocators = get_allocators("cme_futures")
if not allocators:
    raise ValueError("the configured allocator population is empty")
if any(allocation.get("method") == "equal_weight" for allocation in allocators):
    raise ValueError(
        "equal_weight is the baseline stage, not an allocator: `stage` is not part of "
        "`backtest_hash`, so an equal-weight reweight hashes identically to its baseline "
        "parent and one of the two rows is lost. Remove it from the configured menu."
    )

request_rows = []
for label in labels:
    for baseline in shortlist_signal_configurations(
        study,
        label=label,
        limit=shortlist_size,
        execution_tier=EXECUTION_TIER,
    ):
        prediction_hash = baseline.registry_record()["prediction_hash"]
        signal = baseline.spec()["strategy"]["signal"]
        for allocation in allocators:
            method = allocation["method"]
            request_rows.append(
                {
                    "request_name": f"{baseline.hash}-{method}",
                    "prediction_hash": prediction_hash,
                    "label": label,
                    "signal": signal,
                    "allocation": allocation,
                    "risk": None,
                    "costs": None,
                    "chapter": "ch17",
                }
            )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "signal", "allocation")

# %% [markdown]
# ## Execute and freeze allocation candidates
#
# Moment-based allocators receive only price history before each decision. Product-keyed typed
# decisions retain the selected prediction, roll audit, expiry reference, and allocation settings.
#
# "Only price history before each decision" is the causality condition for this stage, and it is
# the one an allocator makes easy to violate: a covariance or volatility estimate computed over
# the whole sample would size every position using dispersion the market had not yet shown, and
# the resulting book would look well-balanced for reasons unavailable at the time. The estimate
# is rebuilt at each decision from history strictly before it.
#
# The results freeze as a named population, and `SUPERSEDES_ALLOCATION_POPULATION` in the
# parameter cell names the generation a re-run retires. The retired snapshot stays in the
# registry, so a Sharpe quoted from it remains traceable to the set it was computed over.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name=ALLOCATION_POPULATION if EXECUTION_TIER == "canonical" else None,
    supersedes=supersedes_for_run(
        study,
        population_name=ALLOCATION_POPULATION,
        declared=SUPERSEDES_ALLOCATION_POPULATION or None,
        execution_tier=EXECUTION_TIER,
    ),
)
candidate_sets = (
    create_label_candidate_sets(study, execution, stage="allocation")
    if EXECUTION_TIER == "canonical"
    else {}
)

# %% [markdown]
# `source` says whether each member was computed by this run or served from the registry because
# an identical identity was already recorded. A re-run of a registered sweep is entirely `reused`
# and completes in seconds; without the column that is indistinguishable from having computed
# every row.

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# The next two execution notebooks select the highest validation Sharpe from the union of signal and
# allocation results for each label. Cost sensitivity is diagnostic; risk overlays remain eligible
# for final selection.
