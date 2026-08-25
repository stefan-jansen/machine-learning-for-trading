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
# # Equal-Weight Backtest - FX Pairs
#
# This notebook establishes the strategy baseline. At each decision time it ranks the currency-pair
# scores, takes equal-sized long and short sleeves, and runs the existing FX backtest engine. Every
# complete model configuration and checkpoint is evaluated. The input is a Polars catalog selection,
# so no prediction hash is copied into orchestration code.
#
# **Learning objectives**
#
# - Freeze the exact prediction population before running a baseline sweep.
# - Send selected catalog rows to the canonical FX engine.
# - Verify that every declared prediction and portfolio-size combination produces one backtest.
#
# **Book reference**: Chapter 16
#
# **Prerequisite**: `12_model_analysis`.

# %%
"""Run the complete equal-weight FX validation backtest population."""

import polars as pl
import yaml

from case_studies.research import (
    OfficialPopulation,
    open_study,
    plan_backtests,
    population_supersedes,
    research_name,
    run_backtests,
    superseded_members,
)
from case_studies.utils.sweep_config import get_top_k_values_for
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
LABEL = ""
SPLIT = "validation"
TOP_K = 0
SEED = 42
RUN_SWEEP = True
FORCE_REBACKTEST = False
TOP_N_PREDICTIONS = None
POPULATION_NAME = ""
SUPERSEDES_EQUAL_WEIGHT_BASELINES: str = ""
SUPERSEDES_VALIDATION_PREDICTIONS: str = ""

# %% [markdown]
# ## Select the exact prediction population
#
# Canonical production includes every complete validation prediction the model notebooks currently
# publish. Label, configuration, or population limits are preview controls and cannot define the
# official baseline.
#
# **"Currently publish" is a question about lineage, and the catalog cannot answer it.** A row's
# `identity_status` is derived from the schema version it was written under, so it says the registry
# still understands the row - not that the row is the one its producer stands behind. The two agree
# until a model notebook refits. Then it publishes a second generation of its population under the
# same name, the first generation's prediction sets stay in the registry complete and current, and a
# sweep selecting on the catalog alone runs over both. It would not fail; it would report every
# member complete, over twice the population, and freeze the retired half into the baseline the rest
# of the case study is measured against.
#
# `superseded_members` asks the registry which identities a later generation retired and no
# generation in force still lists, which is exactly the set to drop. The `tabular_dl` and
# `deep_learning` populations each have a retired generation here, because a training identity
# covers the runner's own source file and both runners changed after their first fits were
# registered. The count of what that excludes is printed rather than left implicit.
#
# **A published population can need a second generation too.** The two names this notebook
# publishes are lists of identities, and the exclusion above changes both of them the moment a
# model notebook refits: the prediction population loses the retired members, and every baseline
# backtest resolved from them goes with it. `OfficialPopulation.create` refuses a changed list
# under an existing name without being told which snapshot it replaces, so each name has its own
# declaration and each is offered through `population_supersedes` on the same rule the model
# notebooks use. Both are empty here because neither name has published a first generation yet;
# after the first canonical run, a refit upstream is answered by filling in the hash that run
# printed.

# %% tags=["results"]
set_global_seeds(SEED)
if SPLIT != "validation":
    raise ValueError("the baseline sweep uses validation predictions")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible baseline request")

study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
# The execution tier decides which registry namespace this run reads and writes;
# the reduction knobs decide only how much of it is covered. Inferring the tier
# from the knobs conflated the two, so any reduced run went looking for preview
# predictions - and a reduced run over a canonical upstream, which is what the
# test suite exercises, then resolved no rows at all.
include_preview = EXECUTION_TIER == "preview"
catalog = study.predictions.table(include_preview=include_preview).filter(
    (pl.col("identity_status") == "current") & (pl.col("split") == SPLIT) & pl.col("complete")
)
if include_preview:
    catalog = catalog.filter(pl.col("execution_tier") == "preview")
else:
    catalog = catalog.filter(pl.col("execution_tier") == "canonical")
retired = superseded_members(study, member_kind="prediction")
if retired:
    offered = catalog.height
    catalog = catalog.filter(~pl.col("prediction_hash").is_in(list(retired)))
    print(f"Retired by a later generation, excluded: {offered - catalog.height} of {offered}")
if LABEL:
    catalog = catalog.filter(pl.col("label") == LABEL)
if TOP_N_PREDICTIONS is not None:
    catalog = catalog.sort("label", "family", "config_name", "checkpoint_value").head(
        TOP_N_PREDICTIONS
    )

if catalog.is_empty():
    raise RuntimeError("the baseline request resolved no complete prediction rows")

# TOP_K and TOP_N_PREDICTIONS both narrow what is backtested, and a narrowed run declares
# a different set of members than the canonical population does. A population is immutable
# once written, so such a run must publish under its own name rather than register a
# partial snapshot under the canonical one. The tier is a separate question: a canonical
# run may legitimately be narrowed, it just may not claim to be the whole population.
if (
    (TOP_K or TOP_N_PREDICTIONS is not None or LABEL)
    and not include_preview
    and not POPULATION_NAME
):
    raise ValueError(
        "this run narrows the baseline sweep, so it cannot publish the canonical "
        "population; pass POPULATION_NAME to give it its own"
    )
if catalog.get_column("prediction_hash").n_unique() != catalog.height:
    raise RuntimeError("the baseline population contains duplicate prediction identities")

if not include_preview:
    predictions_name = research_name(CASE_STUDY_ID, "validation-predictions", scope=POPULATION_NAME)
    prediction_population = OfficialPopulation.create(
        study,
        name=predictions_name,
        member_kind="prediction",
        members=catalog.get_column("prediction_hash").to_list(),
        supersedes=population_supersedes(
            study, name=predictions_name, declared=SUPERSEDES_VALIDATION_PREDICTIONS
        ),
    )
    prediction_population.require_complete()
    print(f"Frozen prediction population: {prediction_population.hash}")
else:
    print("Preview selection is isolated from the official prediction population.")

catalog.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "prediction_hash",
)

# %% [markdown]
# ## Build the baseline strategy grid
#
# `top_k` is the number of pairs in each sleeve. This account allows short selling, so the engine
# holds `top_k` long and `top_k` short and clamps each sleeve to half the universe to keep them
# disjoint. Two consequences worth stating exactly, because both shape how the grid reads:
#
# - At `top_k` below half the universe, ranking decides *which* pairs are held.
# - At `top_k` equal to half the universe, every pair is held at every rebalance and ranking decides
#   only the *side*. The grid then varies concentration and sign exposure, not membership.
#
# A `top_k` above half the universe cannot express anything the half-universe value does not: the
# engine clamps it to the same sleeves, so it would register a distinct identity carrying an
# identical weight series. The grid is refused rather than allowed to contain that duplicate.

# %%
universe_symbols = yaml.safe_load(
    (get_case_study_dir(CASE_STUDY_ID) / "config" / "setup.yaml").read_text()
)["universe"]["symbols"]
n_assets = len(universe_symbols)
max_sleeve = n_assets // 2

jobs = []
for label in sorted(catalog.get_column("label").unique()):
    selected = catalog.filter(pl.col("label") == label)
    top_k_values = [TOP_K] if TOP_K else get_top_k_values_for(CASE_STUDY_ID, label, n_assets)
    duplicates = sorted({value for value in top_k_values if value > max_sleeve})
    if duplicates:
        raise RuntimeError(
            f"top_k {duplicates} exceed the {max_sleeve}-pair sleeve ceiling for a long-short "
            f"account on {n_assets} pairs; the engine would clamp them to {max_sleeve} and register "
            "a duplicate weight series under a distinct identity"
        )
    for top_k in top_k_values:
        jobs.append(
            {
                "label": label,
                "top_k": top_k,
                "predictions": selected,
                "expected": selected.height,
            }
        )

pl.DataFrame(
    [
        {"label": job["label"], "top_k": job["top_k"], "prediction_sets": job["expected"]}
        for job in jobs
    ]
)

# %% [markdown]
# ## Freeze the expected baseline population
#
# Planning resolves every backtest identity without running or writing it. Production freezes that
# complete expected set before the first member executes, so a failed member remains visible.

# %% tags=["results"]
planned_hashes = []
for job in jobs:
    plan = plan_backtests(
        study,
        predictions=job["predictions"],
        signal={"method": "equal_weight_top_k", "top_k": job["top_k"]},
        chapter="16",
    )
    if len(plan.members) != job["expected"]:
        raise RuntimeError("a baseline plan omitted a selected prediction")
    planned_hashes.extend(plan.expected_hashes)

if len(planned_hashes) != len(set(planned_hashes)):
    raise RuntimeError("two planned baseline jobs collapse to the same backtest identity")

baseline_population = None
if not include_preview:
    baselines_name = research_name(CASE_STUDY_ID, "equal-weight-baselines", scope=POPULATION_NAME)
    baseline_population = OfficialPopulation.create(
        study,
        name=baselines_name,
        member_kind="backtest",
        members=planned_hashes,
        supersedes=population_supersedes(
            study, name=baselines_name, declared=SUPERSEDES_EQUAL_WEIGHT_BASELINES
        ),
    )
    print(f"Frozen expected baseline population: {baseline_population.hash}")
else:
    print("Preview backtests remain outside official populations and selection.")

# %% [markdown]
# ## Run every catalog row through the FX engine, then validate what was frozen
#
# Each selected row produces an independent backtest. The loop has no exception-and-continue path:
# one failed member leaves the predeclared population incomplete and stops publication.
#
# The population is validated in the same cell that fills it, because the two are one act: the
# expected set was written down before the first member ran, and `require_complete` is the only
# thing that turns it from a declaration into a published result. It can pass only when every
# planned model, checkpoint and portfolio size completed.

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
backtests = []
for job in jobs:
    execution = run_backtests(
        study,
        predictions=job["predictions"],
        signal={"method": "equal_weight_top_k", "top_k": job["top_k"]},
        chapter="16",
    )
    if len(execution.results) != job["expected"]:
        raise RuntimeError("a baseline member disappeared during execution")
    backtests.extend(execution.results)
    run_status.extend(entry["status"] for entry in execution.diagnostics)

expected_count = sum(job["expected"] for job in jobs)
if len(backtests) != expected_count:
    raise RuntimeError(f"expected {expected_count} baseline runs, found {len(backtests)}")
if {result.hash for result in backtests} != set(planned_hashes):
    raise RuntimeError("completed baseline identities differ from the frozen plan")
if any(not result.complete for result in backtests):
    raise RuntimeError("the baseline population contains an incomplete backtest")

backtest_rows = pl.DataFrame(
    [
        {
            "backtest_hash": result.hash,
            "prediction_hash": result.registry_record()["prediction_hash"],
            "stage": result.registry_record()["stage"],
            "complete": result.complete,
        }
        for result in backtests
    ]
)
if set(backtest_rows.get_column("stage")) != {"signal"}:
    raise RuntimeError("equal-weight baseline runs must register with stage='signal'")

served = run_status.count("reused")
print(
    f"Equal-weight baselines: {len(backtests) - served} computed, {served} served from the registry, "
    f"{len(backtests)} in the population"
)

if not include_preview:
    if baseline_population is None:
        raise RuntimeError("the canonical baseline population was not frozen before execution")
    baseline_population.require_complete()
    print(f"Official equal-weight population: {baseline_population.hash}")

backtest_rows

# %% [markdown]
# ## Key takeaways
#
# - The baseline evaluates every complete model configuration and checkpoint.
# - Catalog rows pass directly to the existing FX backtest engine.
# - Immutable population membership makes missing or failed jobs visible.
