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
# # CME Futures: Risk Overlays
#
# For each return horizon, this notebook selects the highest validation Sharpe from the immutable
# union of signal and allocation results, then applies every position-level risk rule declared in
# the case-study configuration. Stop-loss, trailing-stop, and time-exit parameters are fixed before
# the validation backtest. They are not calibrated from the same validation price path they assess.
#
# Risk rules execute inside the existing futures engine after product-keyed target decisions cross
# the typed boundary. Every declared rule must finish, and the resulting per-label candidate sets
# remain eligible for final validation selection.

# %% [markdown]
# ## What a risk overlay is, and why it is a separate stage
#
# The stages before this one decided *what to hold*: a signal ranked the products, an allocation
# rule decided how much of each. A risk overlay decides *when to stop holding it* - it sits on
# top of an existing set of positions and closes them on a condition the signal never
# considered.
#
# The three rules here are the standard family. A **stop-loss** exits when a position has lost
# more than a set amount from entry. A **trailing stop** exits when it has given back a set
# amount from its best level, so it protects an unrealized gain rather than only the entry
# price. A **time exit** closes after a fixed holding period whatever the position is doing, on
# the reasoning that a signal with a horizon has nothing to say beyond it.
#
# ### The asymmetry these introduce, which is the point and the danger
#
# A signal is symmetric about its own prediction: it is as willing to be wrong in one direction
# as the other. A stop is not. It truncates the loss side of the distribution and leaves the
# gain side alone, and that is why it appeals.
#
# What it also does is convert an unrealized loss into a realized one at the worst available
# moment, and give up any recovery that would have followed. For a mean-reverting signal - which
# describes carry, the signal this case study trades - that is a direct conflict: the position
# is exited precisely when the thing the signal is betting on has become most attractive. A stop
# on a mean-reverting strategy is not a free reduction in risk. It is a change to the strategy,
# and it can easily be a change for the worse.
#
# That is the whole reason this is measured rather than assumed. Risk management is the part of
# a strategy where intuition is least reliable and where "obviously prudent" is applied without
# testing more often than anywhere else in the pipeline.
#
# ### Why the parameters are fixed before the backtest, and not after
#
# The stop distances and holding periods come from `config/setup.yaml` and are fixed before the
# validation backtest runs. They are deliberately **not** calibrated on the price path they are
# then assessed against.
#
# The reason is that this stage is unusually easy to cheat at without noticing. Choosing a stop
# level by trying several and keeping the one with the best validation Sharpe would find the
# level that best avoided the particular drawdowns that particular history happened to contain,
# and would report the result as a risk improvement. Nothing about it would generalize, and
# nothing in the output frame would show what happened - the returns would simply look better.
# Fixing the parameters in configuration is what makes the comparison between overlay and no
# overlay a real one.
#
# ### Why every declared rule must finish
#
# A rule that failed and was skipped would leave a candidate set that silently means "the rules
# that happened to work", and the selection downstream would then choose from it as though it
# were the declared set. Failing the notebook is the only outcome that keeps the population
# equal to the configuration.
#
# ### These candidates stay eligible
#
# Unlike the cost sweep, risk-overlay results are part of the final selection pool.
# `19_strategy_analysis` selects over the union of signal, allocation and risk-overlay
# backtests, so an overlay that genuinely improves validation Sharpe can be what the case study
# ships - and one that does not is visible as such next to the configuration it was applied to.

# %%
"""Run the declared CME futures risk-overlay population."""

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    create_label_candidate_sets,
    open_study,
    pre_overlay_results,
    product_universe_table,
    rank_by_validation_sharpe,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.research.population import supersedes_for_run
from case_studies.utils.sweep_config import get_position_risk_controls

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []

# The risk population is immutable under its name, so a run whose members have moved has to say
# which generation it retires. Anything upstream that changes a backtest identity moves them - a
# corrected label, a changed accounting field, a re-run after a registry reset - and
# `OfficialPopulation.create` refuses to write a different member list under a name that already
# exists. Declared as a literal so that running the committed notebook as it stands recomputes
# the population on record. Empty for a first snapshot.
RISK_POPULATION = "cme_futures-risk-validation-v1"
SUPERSEDES_RISK_POPULATION: str = ""

# %% [markdown]
# ## Fixed per-label inputs and risk rules
#
# No candidate cap or runtime-dependent skip is allowed. The configured list is the population.
#
# **What the overlay is applied to.** An overlay needs an existing strategy to sit on, and there
# is one per label: the highest validation Sharpe from the immutable union of the signal and
# allocation stages. Taking the best of the two stages rather than the signal alone matters,
# because an overlay applied to a weaker parent would be measuring the overlay against a
# strategy the case study would not have shipped anyway.
#
# **Why per label rather than one overall.** Each return horizon is a different prediction
# problem and its best configuration is chosen within its own horizon. Picking one parent across
# all labels would let the strongest horizon's configuration stand in for horizons it was never
# fitted for, and the overlay comparison would then be confounded by which label the parent came
# from. Every configured rule runs against every label's own parent, so the comparison within a
# label is like for like.
#
# **Why the configured list is the population, with no cap.** A runtime cap would make the set
# depend on how long the run took, which means a re-run could select from a different set and
# nothing would record that it had. The rules are declared in configuration precisely so the
# population is a property of the configuration rather than of the execution.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS:
        raise ValueError("canonical execution cannot declare preview reductions")
    labels = ALL_LABELS
elif EXECUTION_TIER == "preview":
    if WORKSPACE is None or not PREVIEW_LABELS:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_LABELS")
    unknown = sorted(set(PREVIEW_LABELS) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"preview labels this case study does not declare: {unknown}")
    labels = tuple(PREVIEW_LABELS)
else:
    raise ValueError(f"unsupported execution tier: {EXECUTION_TIER!r}")
universe = product_universe_table()
universe

# %%
risk_controls = get_position_risk_controls("cme_futures")
if not risk_controls:
    raise ValueError("the configured position-risk population is empty")

request_rows = []
for label in labels:
    selected = rank_by_validation_sharpe(
        study, pre_overlay_results(study, label=label, execution_tier=EXECUTION_TIER)
    )[0]
    strategy = selected.spec()["strategy"]
    prediction_hash = selected.registry_record()["prediction_hash"]
    for control in risk_controls:
        rule = {key: value for key, value in control.items() if key != "name"}
        request_rows.append(
            {
                "request_name": f"{selected.hash}-risk-{control['name']}",
                "prediction_hash": prediction_hash,
                "label": label,
                "signal": strategy["signal"],
                "allocation": strategy.get("allocation"),
                "risk": {"position_rules": [rule]},
                "costs": None,
                "chapter": "ch19",
            }
        )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "risk")

# %% [markdown]
# ## Execute and freeze risk candidates
#
# Each request carries the fitted prediction checkpoint, product decisions, fold-transition policy,
# contract and roll inputs, and one risk rule. Missing members fail before the candidate set exists.
#
# One request is one rule applied to one parent, so a rule's effect is read against its own
# parent rather than against the field. Two rules that both improve Sharpe are not therefore
# combinable: they may exit on the same moves, and their joint effect is not the sum of their
# separate ones. Nothing here estimates that, and a reader stacking rules on the strength of
# this table would be assuming an additivity it does not measure.
#
# The results are frozen as a named population, and `SUPERSEDES_RISK_POPULATION` in the
# parameter cell is how a re-run names the generation it retires. A retired snapshot stays in
# the registry rather than being deleted, so a Sharpe quoted from an earlier generation remains
# traceable to the population it was computed over.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name=RISK_POPULATION if EXECUTION_TIER == "canonical" else None,
    supersedes=supersedes_for_run(
        study,
        population_name=RISK_POPULATION,
        declared=SUPERSEDES_RISK_POPULATION or None,
        execution_tier=EXECUTION_TIER,
    ),
)
candidate_sets = (
    create_label_candidate_sets(study, execution, stage="risk")
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
# Final selection in `19_strategy_analysis` uses the union of signal, allocation, and risk-overlay
# results. Cost-sensitivity rows are excluded.
#
# One consequence to carry into `16_costs`: an overlay only ever adds trades. Every stop that
# fires is an exit that the signal did not ask for, and often a re-entry afterwards. So an
# overlay that improves Sharpe here can still be the worse strategy once friction is priced, and
# the two notebooks have to be read together rather than in sequence. This is also why the
# carrier is priced with its overlay in place rather than bare.
