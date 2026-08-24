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
# # Portfolio Allocation - FX Pairs
#
# This notebook compares allocation methods after the equal-weight baseline. Production advances
# the ten model configurations with highest validation backtest Sharpe for each label, then retains
# every checkpoint belonging to those configurations. Preview mode uses a deterministic reduced
# catalog selection and never writes an official population or candidate set.
#
# **Learning objectives**
#
# - Select configurations from an immutable equal-weight candidate set.
# - Preserve checkpoint identity when configurations advance to allocation.
# - Change allocation while holding prediction, signal, costs, and execution fixed.
#
# **Book reference**: Chapter 17
#
# **Prerequisite**: `13_backtest`.

# %%
"""Run the FX allocation sweep from the frozen equal-weight population."""

from collections.abc import Iterable
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
    research_name,
    run_backtests,
)
from case_studies.utils.sweep_config import (
    get_allocators,
    get_top_k_values_for,
    get_top_n_predictions,
)
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
LABEL = ""
SPLIT = "validation"
TOP_N_CONFIGS = 0
TOP_K = 0
TOP_N_PREDICTIONS = None
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
POPULATION_NAME = ""

# %% [markdown]
# ## Resolve the equal-weight inputs
#
# Canonical execution reads the exact population frozen by the baseline notebook. Its label-specific
# candidate sets provide the only performance ranking used here. The selected unit is a model
# configuration. After a configuration advances, all of its complete checkpoints advance.

# %% tags=["results"]
set_global_seeds(SEED)
universe_symbols = yaml.safe_load(
    (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
)["universe"]["symbols"]
n_assets = len(universe_symbols)
max_sleeve = n_assets // 2
if SPLIT != "validation":
    raise ValueError("allocation selection uses validation backtests")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible allocation request")

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
# publish under its own name rather than register a partial snapshot of the allocation sweep
# under the canonical one.
if (
    (TOP_K or TOP_N_PREDICTIONS is not None or TOP_N_CONFIGS or LABEL)
    and not include_preview
    and not POPULATION_NAME
):
    raise ValueError(
        "this run narrows the allocation sweep, so it cannot publish the canonical "
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
if catalog.is_empty():
    raise RuntimeError("allocation resolved no complete prediction rows")


def _result_config(result: BacktestResult) -> tuple[str, str, str]:
    training = result.lineage()["training_spec"]
    return str(training["label"]), str(training["family"]), str(training["config_name"])


def _open_backtests(hashes: Iterable[str]) -> list[BacktestResult]:
    results = [Result.open(study, value, include_preview=include_preview) for value in hashes]
    if any(not isinstance(result, BacktestResult) for result in results):
        raise TypeError("the equal-weight population contains a non-backtest result")
    return [result for result in results if isinstance(result, BacktestResult)]


def _preview_baselines(rows: pl.DataFrame) -> list[BacktestResult]:
    """The baselines 13_backtest registered, read rather than reconstructed.

    Rebuilding the identity here meant restating the upstream run's `top_k`, and the two
    notebooks reduce independently: a preview gives 13_backtest its own `TOP_K` and says
    nothing to this one, so the reconstruction was a guess about someone else's
    parameters. A wrong guess does not report a disagreement - it computes a hash that
    was never written and fails looking for it, or worse, finds an unrelated run. The
    canonical branch below already reads its members from the published population; this
    is the same read against the preview registry.
    """
    registered = study.backtests.table(include_preview=True).filter(
        (pl.col("stage") == "signal")
        & (pl.col("execution_tier") == "preview")
        & pl.col("prediction_hash").is_in(rows.get_column("prediction_hash"))
        & (pl.col("identity_status") == "current")
        & pl.col("complete")
    )
    if registered.is_empty():
        raise RuntimeError(
            "no preview equal-weight baselines are registered for this prediction "
            "catalog; run 13_backtest at the same reduction before this notebook"
        )
    return _open_backtests(registered.get_column("backtest_hash").unique().sort())


if include_preview:
    baseline_results = _preview_baselines(catalog)
else:
    baseline_population = OfficialPopulation.one(
        study,
        name=research_name(CASE_STUDY_ID, "equal-weight-baselines", scope=POPULATION_NAME),
    )
    baseline_population.require_complete()
    baseline_results = _open_backtests(baseline_population.members)

if any(result.registry_record()["stage"] != "signal" for result in baseline_results):
    raise RuntimeError("the allocation input contains a non-baseline backtest")

# %% [markdown]
# ## Advance configurations without dropping checkpoints
#
# Production ranks the complete baseline results for each label. Once the first result for a model
# configuration appears, that configuration occupies one of the declared slots. The catalog filter
# then restores every checkpoint for each selected configuration, rather than advancing only the
# checkpoint that happened to rank highest.

# %% tags=["results"]
top_n = TOP_N_CONFIGS or get_top_n_predictions(CASE_STUDY_ID, "allocation")
selected_configs: dict[str, set[tuple[str, str]]] = {}
candidate_sets: dict[str, CandidateSet] = {}

# The labels come from the baselines this run resolved, not from this notebook's own
# catalog. A narrowed upstream covers fewer labels than the catalog holds, and rebuilding
# the label list locally means reproducing 13_backtest's narrowing by convention - the
# same guess that reading the registered population exists to avoid. When the run is not
# scoped it is publishing canonical names, and then the two must agree exactly.
baseline_labels = sorted({_result_config(result)[0] for result in baseline_results})
if not baseline_labels:
    raise RuntimeError("the equal-weight baselines carry no labels")
if not POPULATION_NAME and baseline_labels != sorted(catalog.get_column("label").unique()):
    raise RuntimeError(
        "the canonical baseline population does not cover every label in the catalog: "
        f"baselines {baseline_labels}, catalog {sorted(catalog.get_column('label').unique())}"
    )

for label in baseline_labels:
    label_results = [result for result in baseline_results if _result_config(result)[0] == label]
    if include_preview:
        ordered_configs = sorted({_result_config(result)[1:] for result in label_results})
    else:
        candidates = CandidateSet.create(
            study,
            name=research_name(
                CASE_STUDY_ID, f"{label}:equal-weight-candidates", scope=POPULATION_NAME
            ),
            members=label_results,
        )
        candidate_sets[label] = candidates
        ordered_configs = []
        for result in candidates.ranked_validation_sharpe():
            if not isinstance(result, BacktestResult):
                raise TypeError("validation-Sharpe ranking returned a non-backtest result")
            config = _result_config(result)[1:]
            if config not in ordered_configs:
                ordered_configs.append(config)
    selected_configs[label] = set(ordered_configs[:top_n])
    if len(selected_configs[label]) != min(top_n, len(set(ordered_configs))):
        raise RuntimeError(f"configuration selection for {label} is incomplete")

selected_rows = []
for label, configs in selected_configs.items():
    label_rows = catalog.filter(pl.col("label") == label)
    for family, config_name in sorted(configs):
        members = label_rows.filter(
            (pl.col("family") == family) & (pl.col("config_name") == config_name)
        )
        if members.is_empty():
            raise RuntimeError(
                f"selected configuration disappeared: {label}/{family}/{config_name}"
            )
        selected_rows.append(members)

selected = pl.concat(selected_rows).unique(subset=["prediction_hash"], maintain_order=True)
if selected.get_column("prediction_hash").n_unique() != selected.height:
    raise RuntimeError("allocation input contains duplicate prediction identities")
selected.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "prediction_hash",
).sort("label", "family", "config_name", "checkpoint_value")

# %% [markdown]
# ## Plan and freeze the allocation grid
#
# Each request changes only the allocator and `top_k`. The complete selected prediction rows pass
# directly to the shared backtest boundary. Production freezes every expected identity before the
# first allocation result is written.

# %%
allocators = get_allocators(CASE_STUDY_ID)
if not allocators or any(config.get("method") == "equal_weight" for config in allocators):
    raise RuntimeError("allocation methods must be non-empty and exclude the equal-weight baseline")

jobs: list[dict[str, Any]] = []
for label in sorted(selected.get_column("label").unique()):
    rows = selected.filter(pl.col("label") == label)
    top_k_values = [TOP_K] if TOP_K else get_top_k_values_for(CASE_STUDY_ID, label, n_assets)
    oversized = sorted({value for value in top_k_values if value > max_sleeve})
    if oversized:
        raise RuntimeError(
            f"top_k {oversized} exceed the {max_sleeve}-pair sleeve ceiling for a long-short "
            f"account on {n_assets} pairs; the engine clamps them to {max_sleeve}, so each would "
            "register a duplicate weight series under a distinct identity"
        )
    for top_k in top_k_values:
        for allocation in allocators:
            jobs.append(
                {
                    "label": label,
                    "top_k": top_k,
                    "allocation": allocation,
                    "predictions": rows,
                    "expected": rows.height,
                }
            )

pl.DataFrame(
    [
        {
            "label": job["label"],
            "top_k": job["top_k"],
            "allocator": job["allocation"]["method"],
            "prediction_sets": job["expected"],
        }
        for job in jobs
    ]
)

# %% tags=["results"]
planned_hashes = []
for job in jobs:
    plan = plan_backtests(
        study,
        predictions=job["predictions"],
        signal={"method": "equal_weight_top_k", "top_k": job["top_k"]},
        allocation=job["allocation"],
        chapter="17",
    )
    if len(plan.members) != job["expected"]:
        raise RuntimeError("an allocation plan omitted a selected prediction")
    planned_hashes.extend(plan.expected_hashes)
if len(planned_hashes) != len(set(planned_hashes)):
    raise RuntimeError("two planned allocation requests collapse to the same identity")

allocation_population = None
if not include_preview:
    allocation_population = OfficialPopulation.create(
        study,
        name=research_name(CASE_STUDY_ID, "allocation-backtests", scope=POPULATION_NAME),
        member_kind="backtest",
        members=planned_hashes,
    )
    print(f"Frozen expected allocation population: {allocation_population.hash}")

# %% [markdown]
# ## Execute the frozen allocation grid

# %% tags=["results"]
allocation_results = []
for job in jobs:
    execution = run_backtests(
        study,
        predictions=job["predictions"],
        signal={"method": "equal_weight_top_k", "top_k": job["top_k"]},
        allocation=job["allocation"],
        chapter="17",
    )
    if len(execution.results) != job["expected"]:
        raise RuntimeError("an allocation member disappeared during execution")
    allocation_results.extend(execution.results)

expected_count = sum(job["expected"] for job in jobs)
if len(allocation_results) != expected_count:
    raise RuntimeError(
        f"expected {expected_count} allocation runs, found {len(allocation_results)}"
    )
if {result.hash for result in allocation_results} != set(planned_hashes):
    raise RuntimeError("completed allocation identities differ from the frozen plan")
if any(
    not result.complete or result.registry_record()["stage"] != "allocation"
    for result in allocation_results
):
    raise RuntimeError("the allocation population is incomplete or misclassified")


def _non_allocation_projection(spec: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(spec)
    projected.pop("chapter", None)
    projected.pop("_runtime_backtest_config", None)
    projected.get("strategy", {}).pop("allocation", None)
    metadata = projected.get("backtest_config", {}).get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("chapter", None)
    return projected


for result in allocation_results:
    prediction_hash = result.registry_record()["prediction_hash"]
    signal = result.spec()["strategy"]["signal"]
    siblings = [
        baseline
        for baseline in baseline_results
        if baseline.registry_record()["prediction_hash"] == prediction_hash
        and baseline.spec()["strategy"]["signal"] == signal
    ]
    if len(siblings) != 1:
        raise RuntimeError(
            f"allocation {result.hash} resolved to {len(siblings)} equal-weight siblings"
        )
    if _non_allocation_projection(result.spec()) != _non_allocation_projection(siblings[0].spec()):
        raise RuntimeError("an allocation result changed a non-allocation strategy field")

# %% [markdown]
# ## Validate the frozen allocation population

# %% tags=["results"]
if not include_preview:
    if allocation_population is None:
        raise RuntimeError("the canonical allocation population was not frozen before execution")
    allocation_population.require_complete()
    print(f"Official allocation population: {allocation_population.hash}")
else:
    print("Preview allocation results remain outside official populations and candidate sets.")

# %% [markdown]
# ## Key takeaways
#
# - Validation Sharpe ranks an immutable equal-weight candidate set for each label.
# - Configuration selection retains every complete checkpoint for each advancing configuration.
# - Preview reductions exercise the same backtest engine without entering production populations.
