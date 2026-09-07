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
# # Transaction-Cost Sensitivity - FX Pairs
#
# This notebook selects one validation strategy per label from the equal-weight, allocation and
# risk-overlay populations, then changes only its percentage transaction costs. The sensitivity
# grid does not participate in later model or strategy selection.
#
# Costs are swept last because a cost curve is only informative about the strategy that would
# actually be traded, and that strategy is not settled until the risk controls have been measured.
# Sweeping before the overlay charges the grid against a configuration the next notebook may
# discard.
#
# This is a perturbation analysis, not a choice. It asks what happens to a settled strategy if the
# cost model is wrong, which is a different question from which strategy to trade. The two must
# stay separate for a mechanical reason: a strategy allowed to compete on its own cost assumption
# would win by having costs assumed away, and the ranking would report the most optimistic
# assumption rather than the best strategy.
#
# **Why basis points here.** FX spreads are quoted in pips, a fraction of the rate itself, so a
# proportional charge is how the friction is actually expressed - there is no share or contract
# to bill per unit. Case studies where nominal prices are stable, or where spreads are measured
# from quote data, use per-share instead; applying a flat per-unit charge to a rate would assume
# spread scales with the level, which it does not. The configured grid runs from 0 to 50 basis
# points per traded leg, against a real quoted band of roughly 1 to 3 for major pairs and 3 to 8
# for crosses, so most of the grid sits deliberately past anything plausible.
#
# `config/setup.yaml` names the cost components as spread and swap points. That list is a
# taxonomy, read by the Chapter 18 teaching notebooks to describe what the friction consists of;
# the backtest charges a single aggregate rate per traded leg. So this curve perturbs the
# aggregate, and the overnight financing cost of carrying a position is described rather than
# separately priced.
#
# **Learning objectives**
#
# - Select from an immutable validation candidate set by backtest Sharpe.
# - Preserve model, checkpoint, signal, allocation, and execution identities across a cost curve.
# - Keep cost sensitivity outside the official selection cohort.
# - Read a cost curve as a statement about turnover.
#
# **Book reference**: Chapter 18
#
# **Prerequisite**: `15_risk_management`.

# %%
"""Run one cost-sensitivity curve per FX prediction label."""

from copy import deepcopy
from typing import Any

import polars as pl
import yaml

from case_studies.research import (
    BacktestResult,
    CandidateSet,
    OfficialPopulation,
    PredictionResult,
    Result,
    candidate_set_supersedes,
    open_study,
    plan_backtests,
    population_supersedes,
    research_name,
    run_backtests,
    superseded_members,
)
from case_studies.utils.sweep_config import (
    get_allocators,
    get_cost_grid_bps,
    get_top_k_values_for,
)
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
LABEL = ""
SPLIT = "validation"
TOP_K = 0
TOP_N_PREDICTIONS = None
MAX_COST_POINTS = 0
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
POPULATION_NAME = ""
SUPERSEDES_COST_BACKTESTS: str = "fa1d30ceb0f8"
# The same rule the populations follow: a candidate set is immutable under its name, so a rebuilt
# upstream generation must name the set it replaces. Keyed by the full set name, which is what the
# refusal prints. `15_risk_management` states the reasoning once.
SUPERSEDES_CANDIDATE_SETS: dict[str, str] = {
    "fx_pairs:fwd_ret_1d:pre-cost-strategies": "7da3c99d1e71",
    "fx_pairs:fwd_ret_5d:pre-cost-strategies": "cd5141886b53",
    "fx_pairs:fwd_ret_21d:pre-cost-strategies": "16892238f4f8",
}

# %% [markdown]
# ## Select one strategy for each label
#
# Production selection considers the signal, allocation and risk-overlay populations together -
# the same three stages the canonical validation rank-1 is selected over in
# `case_studies/utils/strategy_analysis.py:resolve_canonical_rank1_lineage`. All three are
# candidates because each stage is an alternative to the one before it rather than an improvement
# on it by construction. Naming only the overlays would charge the cost grid against a risk
# control even where every control measured worse than leaving the position rule alone; naming
# only signal and allocation would sweep a strategy the risk notebook has already improved on.
# Which stage the parent came from is printed below rather than assumed, and the gap between the
# best overlay and the best un-overlaid configuration is printed with it: a negative gap is the
# measurement that the controls did not help on this label.
#
# Cost variants are descendants of that choice and cannot improve their own chance of selection.
# Preview mode uses a deterministic allocation request from the reduced catalog and remains
# outside candidate sets.
#
# Getting the parent wrong has a quiet failure mode. The cost curve would be computed correctly,
# the population would freeze and validate, and every number would be right - about a strategy
# the chapter does not report. Nothing raises, because a cost sweep over the wrong parent is a
# perfectly valid sweep. That is why the stage the parent came from is printed rather than
# assumed, and why the selection here is made over the same three stages, in the same way, as
# `resolve_canonical_rank1_lineage` selects the strategy the chapter goes on to describe.

# %% tags=["results"]
set_global_seeds(SEED)
universe_symbols = yaml.safe_load(
    (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
)["universe"]["symbols"]
n_assets = len(universe_symbols)
if SPLIT != "validation":
    raise ValueError("cost sensitivity uses validation backtests")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible cost request")

study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
# The execution tier decides which registry namespace this run reads and writes;
# the reduction knobs decide only how much of it is covered. Inferring the tier
# from the knobs conflated the two, so any reduced run went looking for preview
# predictions - and a reduced run over a canonical upstream, which is what the
# test suite exercises, then resolved no rows at all.
include_preview = EXECUTION_TIER == "preview"

# The tier decides the namespace, so a canonical run may legitimately be narrowed -
# but a narrowed run declares a different set of members than the canonical
# population does, and a population is immutable once written. Such a run must
# publish under its own name rather than register a partial snapshot of the cost sweep
# under the canonical one.
if (
    (TOP_K or TOP_N_PREDICTIONS is not None or MAX_COST_POINTS or LABEL)
    and not include_preview
    and not POPULATION_NAME
):
    raise ValueError(
        "this run narrows the cost sweep, so it cannot publish the canonical "
        "population; pass POPULATION_NAME to give it its own"
    )
catalog = study.predictions.table(include_preview=include_preview).filter(
    (pl.col("identity_status") == "current")
    & (pl.col("split") == SPLIT)
    & pl.col("complete")
    & (pl.col("execution_tier") == ("preview" if include_preview else "canonical"))
)
# `identity_status` is the schema version a row was written under, not a statement about which
# generation its producer publishes. A model notebook that refits leaves the generation it
# replaced in the registry, complete and current, so this filter alone would carry a retired
# prediction set into the sweep. `superseded_members` reads the lineage instead - see
# `13_backtest`, which drops the same set before it freezes the baseline population.
# `SUPERSEDES_COST_BACKTESTS` names the snapshot this run replaces under the name it publishes,
# offered through `population_supersedes` on the same rule. It is empty until that name has a
# first generation; after that, an upstream refit changes this population's member list and
# the registry refuses the write without it. `13_backtest` states the reasoning once.
retired = superseded_members(study, member_kind="prediction")
if retired:
    catalog = catalog.filter(~pl.col("prediction_hash").is_in(list(retired)))
if LABEL:
    catalog = catalog.filter(pl.col("label") == LABEL)
if TOP_N_PREDICTIONS is not None:
    catalog = catalog.sort("label", "family", "config_name", "checkpoint_value").head(
        TOP_N_PREDICTIONS
    )
if catalog.is_empty():
    raise RuntimeError("cost sensitivity resolved no complete prediction rows")


def _open_backtests(population: OfficialPopulation) -> list[BacktestResult]:
    population.require_complete()
    opened = [Result.open(study, value) for value in population.members]
    if any(not isinstance(result, BacktestResult) for result in opened):
        raise TypeError(f"population {population.name!r} contains a non-backtest result")
    return [result for result in opened if isinstance(result, BacktestResult)]


def _label(result: BacktestResult) -> str:
    return str(result.lineage()["training_spec"]["label"])


def _registered_preview_allocations() -> pl.DataFrame:
    """The allocation backtests an upstream preview registered.

    One predicate, read once. Stating it twice - once to decide which labels are covered
    and once to pick a label's leader - makes the two required to agree forever: tighten
    one and the other admits a label whose backtests it then rejects, which is exactly the
    "no preview allocation backtests are registered" failure this exists to prevent.
    """
    return study.backtests.table(include_preview=True).filter(
        (pl.col("stage") == "allocation")
        & (pl.col("execution_tier") == "preview")
        & (pl.col("identity_status") == "current")
        & pl.col("complete")
    )


def _preview_leader(rows: pl.DataFrame, registered_allocations: pl.DataFrame) -> BacktestResult:
    """One allocation backtest for this label, read from what 14 registered.

    Rebuilding the identity restated two of the upstream run's choices - its `top_k` and
    which allocator came first - from this notebook's own defaults. A preview reduces
    each notebook independently, so those are guesses about another run's parameters,
    and a guess that is wrong looks for a hash nothing wrote instead of reporting the
    disagreement.
    """
    registered = registered_allocations.filter(
        pl.col("prediction_hash").is_in(rows.get_column("prediction_hash").implode())
    )
    if registered.is_empty():
        raise RuntimeError(
            "no preview allocation backtests are registered for this prediction "
            "catalog; run 14_portfolio_management at the same reduction first"
        )
    row = registered.sort("family", "config_name", "checkpoint_value", "backtest_hash").row(
        0, named=True
    )
    result = Result.open(study, row["backtest_hash"], include_preview=True)
    if not isinstance(result, BacktestResult) or not result.complete:
        raise RuntimeError("the deterministic preview allocation result is not complete")
    return result


selected_by_label: dict[str, BacktestResult] = {}
candidate_sets: dict[str, CandidateSet] = {}
if include_preview:
    # The labels come from what the upstream preview registered, the same rule the
    # canonical branch below follows. Enumerating this notebook's own catalog asks
    # _preview_leader for labels 14_portfolio_management was configured not to allocate,
    # and it raises "no preview allocation backtests are registered" for each - reporting
    # a reduction the run was told to make as a missing upstream.
    registered_allocations = _registered_preview_allocations()
    covered = catalog.filter(
        pl.col("prediction_hash").is_in(
            registered_allocations.get_column("prediction_hash").implode()
        )
    )
    if covered.is_empty():
        raise RuntimeError(
            "no preview allocation backtests cover this prediction catalog; "
            "run 14_portfolio_management at the same reduction first"
        )
    for label in sorted(covered.get_column("label").unique()):
        selected_by_label[label] = _preview_leader(
            covered.filter(pl.col("label") == label), registered_allocations
        )
else:
    baselines = _open_backtests(
        OfficialPopulation.one(
            study,
            name=research_name(CASE_STUDY_ID, "equal-weight-baselines", scope=POPULATION_NAME),
        )
    )
    allocations = _open_backtests(
        OfficialPopulation.one(
            study,
            name=research_name(CASE_STUDY_ID, "allocation-backtests", scope=POPULATION_NAME),
        )
    )
    risk_overlays = _open_backtests(
        OfficialPopulation.one(
            study,
            name=research_name(CASE_STUDY_ID, "risk-overlay-backtests", scope=POPULATION_NAME),
        )
    )
    # The labels come from the upstream populations this run resolved, not from this
    # notebook's own catalog. A narrowed upstream covers fewer labels than the catalog
    # holds, and rebuilding the list locally reproduces the upstream narrowing by
    # convention. Unscoped, the run publishes canonical names and the two must agree.
    upstream = [*baselines, *allocations, *risk_overlays]
    upstream_labels = sorted({_label(result) for result in upstream})
    if not upstream_labels:
        raise RuntimeError("the upstream populations carry no labels")
    if not POPULATION_NAME and upstream_labels != sorted(catalog.get_column("label").unique()):
        raise RuntimeError(
            "the canonical upstream populations do not cover every label in the catalog: "
            f"upstream {upstream_labels}, "
            f"catalog {sorted(catalog.get_column('label').unique())}"
        )
    for label in upstream_labels:
        members = [result for result in upstream if _label(result) == label]
        _set_name = research_name(
            CASE_STUDY_ID, f"{label}:pre-cost-strategies", scope=POPULATION_NAME
        )
        candidates = CandidateSet.create(
            study,
            name=_set_name,
            members=members,
            supersedes=candidate_set_supersedes(
                study, name=_set_name, declared=SUPERSEDES_CANDIDATE_SETS.get(_set_name)
            ),
        )
        candidate_sets[label] = candidates
        leader = candidates.best_validation_sharpe()
        if not isinstance(leader, BacktestResult):
            raise TypeError("strategy selection did not return a backtest")
        selected_by_label[label] = leader


def _overlay_gap(label: str) -> dict[str, object]:
    """What the risk controls were worth on *label*, in validation Sharpe.

    The parent is chosen across all three stages, so "the overlay won" and "the overlay
    helped" are the same statement only when an un-overlaid configuration was in the running.
    Both sides are reported: the best risk overlay, the best signal-or-allocation strategy it
    was measured against, and the difference. A negative difference is the finding that the
    controls cost more than they saved on this label, and the selected parent is then
    un-overlaid.
    """
    members = candidate_sets[label].members
    rows = study.backtests.table().filter(
        pl.col("backtest_hash").is_in(list(members))
        & (pl.col("split") == "validation")
        & pl.col("sharpe").is_not_null()
    )
    overlaid = rows.filter(pl.col("stage") == "risk_overlay")
    un_overlaid = rows.filter(pl.col("stage").is_in(["signal", "allocation"]))
    best_overlaid = overlaid.get_column("sharpe").max() if overlaid.height else None
    best_un_overlaid = un_overlaid.get_column("sharpe").max() if un_overlaid.height else None
    gap = (
        best_overlaid - best_un_overlaid
        if best_overlaid is not None and best_un_overlaid is not None
        else None
    )
    return {
        "best_overlaid_sharpe": best_overlaid,
        "best_un_overlaid_sharpe": best_un_overlaid,
        "overlay_gap": gap,
    }


pl.DataFrame(
    [
        {
            "label": label,
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
            "stage": result.registry_record()["stage"],
            **(_overlay_gap(label) if label in candidate_sets else {}),
        }
        for label, result in selected_by_label.items()
    ]
)

# %% [markdown]
# ## Plan and freeze exact cost siblings
#
# The configured cost grid is expressed as total basis points per traded leg. Commission and
# slippage each receive half. The identity audit removes only the cost fields and the chapter label;
# every remaining field must match the selected validation strategy. Production freezes the full
# sensitivity set before the first backtest is written.
#
# The even split between commission and slippage is a modelling convention, not a measurement.
# Nothing in the data says the two halves of the friction are equal; the split exists so that a
# single configured rate can populate two fields the backtest engine charges separately. Read the
# total, not the halves.
#
# What the curve measures, once it exists, is turnover. Cost enters the return series through
# `|delta w|` at each rebalance, so a strategy's sensitivity to the assumed rate is set by how
# much of the book it moves and how often, not by how good its predictions are. Two strategies
# with the same gross Sharpe can have breakevens that differ by an order of magnitude, and the
# whole reason to plot a curve rather than report one number is that the difference is invisible
# at any single rate. The quantity to read off is where the curve crosses zero and how far that
# sits from the quoted band above - a strategy that survives to 40 basis points on pairs that
# trade at 3 has room; one that dies at 4 is reporting an edge that is really a spread.

# %% tags=["results"]
cost_grid = get_cost_grid_bps(CASE_STUDY_ID)
if MAX_COST_POINTS:
    cost_grid = cost_grid[:MAX_COST_POINTS]
if not cost_grid:
    raise RuntimeError("the cost grid is empty")


def _catalog_row(result: BacktestResult) -> pl.DataFrame:
    prediction_hash = result.registry_record()["prediction_hash"]
    row = catalog.filter(pl.col("prediction_hash") == prediction_hash)
    if row.height != 1:
        raise RuntimeError(f"prediction {prediction_hash} resolved to {row.height} catalog rows")
    return row


def _strategy_arguments(result: BacktestResult) -> dict[str, Any]:
    strategy = result.spec()["strategy"]
    return {
        "signal": deepcopy(strategy["signal"]),
        "allocation": deepcopy(strategy.get("allocation")),
        "risk": deepcopy(strategy.get("risk")),
        "execution_mode": strategy.get("rebalance", {}).get("mode"),
    }


def _non_cost_projection(spec: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(spec)
    projected.pop("chapter", None)
    projected.pop("_runtime_backtest_config", None)
    config = projected.get("backtest_config", {})
    config.pop("commission", None)
    config.pop("slippage", None)
    metadata = config.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("chapter", None)
    return projected


cost_jobs = []
for label, selected in selected_by_label.items():
    arguments = _strategy_arguments(selected)
    for total_bps in cost_grid:
        costs = {
            "commission_bps": total_bps / 2.0,
            "slippage_bps": total_bps / 2.0,
        }
        plan = plan_backtests(
            study,
            predictions=_catalog_row(selected),
            signal=arguments["signal"],
            allocation=arguments["allocation"],
            risk=arguments["risk"],
            costs=costs,
            chapter="ch18",
            execution_mode=arguments["execution_mode"],
        )
        if len(plan.members) != 1:
            raise RuntimeError("a cost plan must contain exactly one backtest")
        cost_jobs.append(
            {
                "label": label,
                "selected": selected,
                "arguments": arguments,
                "total_bps": total_bps,
                "costs": costs,
                "backtest_hash": plan.expected_hashes[0],
            }
        )

planned_hashes = [job["backtest_hash"] for job in cost_jobs]
if len(planned_hashes) != len(set(planned_hashes)):
    raise RuntimeError("two planned cost requests collapse to the same identity")

cost_population = None
if not include_preview:
    costs_name = research_name(CASE_STUDY_ID, "cost-sensitivity-backtests", scope=POPULATION_NAME)
    cost_population = OfficialPopulation.create(
        study,
        name=costs_name,
        member_kind="backtest",
        members=planned_hashes,
        supersedes=population_supersedes(
            study, name=costs_name, declared=SUPERSEDES_COST_BACKTESTS
        ),
    )
    print(f"Frozen expected cost population: {cost_population.hash}")

# %% [markdown]
# ## Execute the frozen cost grid, and validate its membership without making it selectable
#
# The population is validated in the cell that fills it: the expected set was written down before
# the first member ran, and `require_complete` is what turns that declaration into a published
# result. Publishing it does not make it selectable - a cost sensitivity is a curve through a
# parameter the strategy does not choose, and later selection reads the allocation population.
#
# "Registered but not selectable" is a distinction worth being concrete about, because both parts
# are deliberate. These rows are written to the registry, complete and current, exactly like every
# other backtest: the curve is a published result that a reader can look up and re-derive. What
# makes them unselectable is that the downstream stages read named populations rather than
# querying the registry for whatever is complete, so a cost sibling is never a member of a set
# anything ranks. The separation lives in which population a stage reads, not in a flag on the
# row - which is why a stage that queried the registry directly would silently acquire eleven
# copies of one strategy, each at a different assumed rate, and would rank them.

# %% tags=["results"]
# A sweep that recomputes everything and a sweep that recomputes nothing print the same summary
# unless the two are counted apart. `run_backtests` serves an identity that is already registered
# and complete instead of running it again, which is what makes a re-run affordable and what makes
# a bare member count say nothing about whether this run did any work.
#
# The runner already knows which it did and says so per member in `execution.diagnostics`, as
# `status` "reused" or "completed". Comparing against the registered hashes instead would be
# wrong in both directions: a registered-but-partial backtest is in that set, gets recomputed and
# would report as reused, and a preview re-run reads a table that excludes preview rows by default
# and would report every reused member as computed.
run_status: list[str] = []
cost_results: list[BacktestResult] = []
cost_rows = []
for job in cost_jobs:
    selected = job["selected"]
    arguments = job["arguments"]
    execution = run_backtests(
        study,
        predictions=_catalog_row(selected),
        signal=arguments["signal"],
        allocation=arguments["allocation"],
        risk=arguments["risk"],
        costs=job["costs"],
        chapter="ch18",
        execution_mode=arguments["execution_mode"],
    )
    if len(execution.results) != 1:
        raise RuntimeError("a cost request must produce exactly one backtest")
    result = execution.results[0]
    if result.hash != job["backtest_hash"]:
        raise RuntimeError("a completed cost identity differs from the frozen plan")
    if _non_cost_projection(result.spec()) != _non_cost_projection(selected.spec()):
        raise RuntimeError("a cost sibling changed a non-cost strategy field")
    if result.registry_record()["stage"] != "cost_sensitivity" or not result.complete:
        raise RuntimeError("a cost result is incomplete or misclassified")
    cost_results.append(result)
    run_status.extend(entry["status"] for entry in execution.diagnostics)
    cost_rows.append(
        {
            "label": job["label"],
            "total_cost_bps": job["total_bps"],
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
        }
    )

served = run_status.count("reused")
print(
    f"Cost siblings: {len(cost_results) - served} computed, {served} served from the registry, "
    f"{len(cost_results)} in the population"
)

if not include_preview:
    if cost_population is None:
        raise RuntimeError("the canonical cost population was not frozen before execution")
    cost_population.require_complete()
    print(f"Official cost-sensitivity population: {cost_population.hash}")
else:
    print("Preview cost curves remain outside official populations and candidate sets.")

pl.DataFrame(cost_rows).sort("label", "total_cost_bps")

# %% [markdown]
# ## Key takeaways
#
# - Each label has one validation-selected parent strategy, chosen across all three upstream
#   stages so the curve prices the strategy the chapter reports rather than a sibling of it.
# - Cost siblings preserve every non-cost identity field.
# - Cost sensitivity is frozen for completeness but excluded from later selection. A strategy that
#   could compete on its cost assumption would win by assuming costs away.
# - Basis points are the FX regime because spreads are quoted as a fraction of the rate. The
#   declared components are a taxonomy; the backtest charges one aggregate rate per traded leg.
# - The curve is a turnover measurement. Read where it crosses zero and compare that to the
#   quoted band, not the Sharpe at any single rate.
#
# The breakeven this produces is still a validation-period number, and turnover is not stable
# across regimes: a strategy that trades more in volatile periods pays more exactly when spreads
# are widest, and a curve computed at a constant rate cannot show that. The curve bounds the
# question rather than settling it.
