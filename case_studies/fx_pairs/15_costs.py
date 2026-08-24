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
# This notebook selects one validation strategy per label from the equal-weight and allocation
# populations, then changes only its percentage transaction costs. The sensitivity grid does not
# participate in later model or strategy selection.
#
# **Learning objectives**
#
# - Select from an immutable validation candidate set by backtest Sharpe.
# - Preserve model, checkpoint, signal, allocation, and execution identities across a cost curve.
# - Keep cost sensitivity outside the official selection cohort.
#
# **Book reference**: Chapter 18
#
# **Prerequisite**: `14_portfolio_management`.

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
    open_study,
    plan_backtests,
    run_backtests,
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

# %% [markdown]
# ## Select one strategy for each label
#
# Production selection considers only the signal and allocation populations. Cost variants are
# descendants of that choice and cannot improve their own chance of selection. Preview mode uses a
# deterministic allocation request from the reduced catalog and remains outside candidate sets.

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
    (TOP_K or TOP_N_PREDICTIONS is not None or MAX_COST_POINTS)
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


def _preview_leader(rows: pl.DataFrame) -> BacktestResult:
    """One allocation backtest for this label, read from what 14 registered.

    Rebuilding the identity restated two of the upstream run's choices - its `top_k` and
    which allocator came first - from this notebook's own defaults. A preview reduces
    each notebook independently, so those are guesses about another run's parameters,
    and a guess that is wrong looks for a hash nothing wrote instead of reporting the
    disagreement.
    """
    registered = study.backtests.table(include_preview=True).filter(
        (pl.col("stage") == "allocation")
        & (pl.col("execution_tier") == "preview")
        & pl.col("prediction_hash").is_in(rows.get_column("prediction_hash"))
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
    for label in sorted(catalog.get_column("label").unique()):
        selected_by_label[label] = _preview_leader(catalog.filter(pl.col("label") == label))
else:
    baselines = _open_backtests(
        OfficialPopulation.one(study, name=f"{CASE_STUDY_ID}:equal-weight-baselines")
    )
    allocations = _open_backtests(
        OfficialPopulation.one(study, name=f"{CASE_STUDY_ID}:allocation-backtests")
    )
    for label in sorted(catalog.get_column("label").unique()):
        members = [result for result in [*baselines, *allocations] if _label(result) == label]
        candidates = CandidateSet.create(
            study,
            name=f"{CASE_STUDY_ID}:{label}:pre-cost-strategies",
            members=members,
        )
        candidate_sets[label] = candidates
        leader = candidates.best_validation_sharpe()
        if not isinstance(leader, BacktestResult):
            raise TypeError("strategy selection did not return a backtest")
        selected_by_label[label] = leader

pl.DataFrame(
    [
        {
            "label": label,
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
            "stage": result.registry_record()["stage"],
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
    cost_population = OfficialPopulation.create(
        study,
        name=POPULATION_NAME or f"{CASE_STUDY_ID}:cost-sensitivity-backtests",
        member_kind="backtest",
        members=planned_hashes,
    )
    print(f"Frozen expected cost population: {cost_population.hash}")

# %% [markdown]
# ## Execute the frozen cost grid

# %% tags=["results"]
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
    cost_rows.append(
        {
            "label": job["label"],
            "total_cost_bps": job["total_bps"],
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
        }
    )

pl.DataFrame(cost_rows).sort("label", "total_cost_bps")

# %% [markdown]
# ## Validate sensitivity membership without making it selectable

# %% tags=["results"]
if not include_preview:
    if cost_population is None:
        raise RuntimeError("the canonical cost population was not frozen before execution")
    cost_population.require_complete()
    print(f"Official cost-sensitivity population: {cost_population.hash}")
else:
    print("Preview cost curves remain outside official populations and candidate sets.")

# %% [markdown]
# ## Key takeaways
#
# - Each label has one validation-selected parent strategy.
# - Cost siblings preserve every non-cost identity field.
# - Cost sensitivity is frozen for completeness but excluded from later selection.
