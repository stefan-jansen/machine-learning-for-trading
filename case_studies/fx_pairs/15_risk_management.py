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
# # Position Risk Controls - FX Pairs
#
# This notebook selects one validation strategy per label from the equal-weight and allocation
# populations, then changes only its predeclared position-risk rule. Cost-sensitivity results are
# excluded from selection, so an optimistic cost assumption cannot advance a strategy.
#
# **Learning objectives**
#
# - Select one parent strategy from an immutable validation cohort.
# - Compare declared position controls while preserving upstream identity.
# - Freeze the complete risk population before final strategy selection.
#
# **Book reference**: Chapter 19
#
# **Prerequisite**: `14_portfolio_management`.

# %%
"""Run the declared FX position-risk controls for each label."""

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
    get_portfolio_risk_controls,
    get_position_risk_controls,
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
MAX_RISK_VARIANTS = 0
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
POPULATION_NAME = ""
SUPERSEDES_RISK_BACKTESTS: str = "cd421f7757e0"
# A candidate set is immutable under its name, exactly as a population is, so a rebuilt upstream
# generation has to name the set it replaces. Keyed by the full set name because that is what the
# refusal prints: pasting back the name it names is the obvious thing to try, and it has to work.
# Resolved through `candidate_set_supersedes` rather than passed straight to `create`, because a
# reader's clean clone has no generation to supersede and `create` refuses a first version that
# claims to replace one.
SUPERSEDES_CANDIDATE_SETS: dict[str, str] = {
    "fx_pairs:fwd_ret_1d:pre-risk-strategies": "208fc4bbc14c",
    "fx_pairs:fwd_ret_5d:pre-risk-strategies": "ceea3ffc2dd3",
    "fx_pairs:fwd_ret_21d:pre-risk-strategies": "23087a1081bf",
    "fx_pairs:holdout-candidates": "4aea5c6c1218",
}

# %% [markdown]
# ## Select the parent strategy
#
# Production uses the same signal-plus-allocation candidate cohort as the cost notebook. Preview
# mode resolves one deterministic allocation result from the reduced prediction catalog.

# %% tags=["results"]
set_global_seeds(SEED)
universe_symbols = yaml.safe_load(
    (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
)["universe"]["symbols"]
n_assets = len(universe_symbols)
if SPLIT != "validation":
    raise ValueError("risk comparison uses validation backtests")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible risk request")

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
# publish under its own name rather than register a partial snapshot of the risk overlay sweep
# under the canonical one.
if (
    (TOP_K or TOP_N_PREDICTIONS is not None or MAX_RISK_VARIANTS or LABEL)
    and not include_preview
    and not POPULATION_NAME
):
    raise ValueError(
        "this run narrows the risk overlay sweep, so it cannot publish the canonical "
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
# `SUPERSEDES_RISK_BACKTESTS` names the snapshot this run replaces under the name it publishes,
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
    raise RuntimeError("risk comparison resolved no complete prediction rows")


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
    # The labels come from the upstream populations this run resolved, not from this
    # notebook's own catalog. A narrowed upstream covers fewer labels than the catalog
    # holds, and rebuilding the list locally reproduces the upstream narrowing by
    # convention. Unscoped, the run publishes canonical names and the two must agree.
    upstream = [*baselines, *allocations]
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
            CASE_STUDY_ID, f"{label}:pre-risk-strategies", scope=POPULATION_NAME
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
# ## Plan and freeze exact risk siblings
#
# Every position rule is declared in `config/setup.yaml`. Portfolio-level controls are absent for
# this case study. The identity audit removes only the risk block and chapter label; the prediction,
# signal, allocation, configured costs, and execution contract must remain unchanged. Production
# freezes every expected risk identity before the first backtest is written.

# %% tags=["results"]
position_controls = get_position_risk_controls(CASE_STUDY_ID)
if get_portfolio_risk_controls(CASE_STUDY_ID):
    raise RuntimeError("FX declares position controls only")
if MAX_RISK_VARIANTS:
    position_controls = position_controls[:MAX_RISK_VARIANTS]
if not position_controls:
    raise RuntimeError("the position-risk grid is empty")


def _risk_payload(control: dict[str, Any]) -> dict[str, Any]:
    if control["type"] == "time_exit":
        rule = {"type": control["type"], "bars": int(control["bars"])}
    else:
        rule = {"type": control["type"], "threshold": float(control["threshold"])}
    return {"name": control["name"], "position_rules": [rule]}


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
        "execution_mode": strategy.get("rebalance", {}).get("mode"),
    }


def _non_risk_projection(spec: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(spec)
    projected.pop("chapter", None)
    projected.pop("_runtime_backtest_config", None)
    projected.get("strategy", {}).pop("risk", None)
    metadata = projected.get("backtest_config", {}).get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("chapter", None)
    return projected


risk_jobs = []
for label, selected in selected_by_label.items():
    arguments = _strategy_arguments(selected)
    for control in position_controls:
        risk = _risk_payload(control)
        plan = plan_backtests(
            study,
            predictions=_catalog_row(selected),
            signal=arguments["signal"],
            allocation=arguments["allocation"],
            risk=risk,
            chapter="19",
            execution_mode=arguments["execution_mode"],
        )
        if len(plan.members) != 1:
            raise RuntimeError("a risk plan must contain exactly one backtest")
        risk_jobs.append(
            {
                "label": label,
                "selected": selected,
                "arguments": arguments,
                "risk": risk,
                "risk_name": control["name"],
                "backtest_hash": plan.expected_hashes[0],
            }
        )

planned_hashes = [job["backtest_hash"] for job in risk_jobs]
if len(planned_hashes) != len(set(planned_hashes)):
    raise RuntimeError("two planned risk requests collapse to the same identity")

risk_population = None
if not include_preview:
    risks_name = research_name(CASE_STUDY_ID, "risk-overlay-backtests", scope=POPULATION_NAME)
    risk_population = OfficialPopulation.create(
        study,
        name=risks_name,
        member_kind="backtest",
        members=planned_hashes,
        supersedes=population_supersedes(
            study, name=risks_name, declared=SUPERSEDES_RISK_BACKTESTS
        ),
    )
    print(f"Frozen expected risk population: {risk_population.hash}")

# %% [markdown]
# ## Execute the frozen risk grid, and validate what was frozen
#
# The population is validated in the cell that fills it: the expected set was written down before
# the first member ran, and `require_complete` is what turns that declaration into a published
# result.

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
risk_results: list[BacktestResult] = []
risk_rows = []
for job in risk_jobs:
    selected = job["selected"]
    arguments = job["arguments"]
    execution = run_backtests(
        study,
        predictions=_catalog_row(selected),
        signal=arguments["signal"],
        allocation=arguments["allocation"],
        risk=job["risk"],
        chapter="19",
        execution_mode=arguments["execution_mode"],
    )
    if len(execution.results) != 1:
        raise RuntimeError("a risk request must produce exactly one backtest")
    result = execution.results[0]
    if result.hash != job["backtest_hash"]:
        raise RuntimeError("a completed risk identity differs from the frozen plan")
    if _non_risk_projection(result.spec()) != _non_risk_projection(selected.spec()):
        raise RuntimeError("a risk sibling changed a non-risk strategy field")
    if result.registry_record()["stage"] != "risk_overlay" or not result.complete:
        raise RuntimeError("a risk result is incomplete or misclassified")
    risk_results.append(result)
    run_status.extend(entry["status"] for entry in execution.diagnostics)
    risk_rows.append(
        {
            "label": job["label"],
            "risk_name": job["risk_name"],
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
        }
    )

served = run_status.count("reused")
print(
    f"Risk overlays: {len(risk_results) - served} computed, {served} served from the registry, "
    f"{len(risk_results)} in the population"
)

if not include_preview:
    if risk_population is None:
        raise RuntimeError("the canonical risk population was not frozen before execution")
    risk_population.require_complete()
    print(f"Official risk population: {risk_population.hash}")

pl.DataFrame(risk_rows).sort("label", "risk_name")

# %% [markdown]
# ## Freeze the set the holdout will choose from
#
# Everything this case study has backtested on validation goes in: the equal-weight baselines, the
# allocation variants, and the risk overlays. The comparison contract names the fields every member
# must agree on, so a candidate fitted against different labels, features or folds cannot silently
# join a set the holdout will pick from. Only identities are printed here; what the selection is
# worth is `19_strategy_analysis`'s question.

# %%
if not include_preview:
    _holdout_set_name = research_name(CASE_STUDY_ID, "holdout-candidates", scope=POPULATION_NAME)
    holdout_candidates = CandidateSet.create(
        study,
        name=_holdout_set_name,
        members=[*baselines, *allocations, *risk_results],
        comparison_contract={"comparable_fields": ["label_artifact", "feature_artifacts", "cv"]},
        supersedes=candidate_set_supersedes(
            study,
            name=_holdout_set_name,
            declared=SUPERSEDES_CANDIDATE_SETS.get(_holdout_set_name),
        ),
    )
    selected = holdout_candidates.best_validation_sharpe()
    print(f"Frozen holdout candidate set: {holdout_candidates.hash}")
    print(f"Validation-selected backtest: {selected.hash}")
else:
    print("Preview risk results remain outside official populations and candidate sets.")

# %% [markdown]
# ## Key takeaways
#
# - Risk variants descend from the same signal-plus-allocation selection cohort as cost variants.
# - Each comparison changes one declared position-risk rule.
# - The final candidate set contains signal, allocation, and risk results across every label.
