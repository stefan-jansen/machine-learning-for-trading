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
    run_backtests,
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

# %% [markdown]
# ## Select the exact prediction population
#
# Canonical production includes every current, complete validation prediction. Label, configuration,
# or population limits are preview controls and cannot define the official baseline.

# %% tags=["results"]
set_global_seeds(SEED)
if SPLIT != "validation":
    raise ValueError("the baseline sweep uses validation predictions")
if FORCE_REBACKTEST:
    raise ValueError("identical complete backtests are reused by identity")
if not RUN_SWEEP:
    raise ValueError("set RUN_SWEEP=True to execute the visible baseline request")

study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
include_preview = bool(TOP_K or TOP_N_PREDICTIONS)
catalog = study.predictions.table(include_preview=include_preview).filter(
    (pl.col("identity_status") == "current") & (pl.col("split") == SPLIT) & pl.col("complete")
)
if include_preview:
    catalog = catalog.filter(pl.col("execution_tier") == "preview")
else:
    catalog = catalog.filter(pl.col("execution_tier") == "canonical")
if LABEL:
    catalog = catalog.filter(pl.col("label") == LABEL)
if TOP_N_PREDICTIONS is not None:
    catalog = catalog.sort("label", "family", "config_name", "checkpoint_value").head(
        TOP_N_PREDICTIONS
    )

if catalog.is_empty():
    raise RuntimeError("the baseline request resolved no complete prediction rows")
if catalog.get_column("prediction_hash").n_unique() != catalog.height:
    raise RuntimeError("the baseline population contains duplicate prediction identities")

if not include_preview:
    prediction_population = OfficialPopulation.create(
        study,
        name=f"{CASE_STUDY_ID}:validation-predictions",
        member_kind="prediction",
        members=catalog.get_column("prediction_hash").to_list(),
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
    baseline_population = OfficialPopulation.create(
        study,
        name=f"{CASE_STUDY_ID}:equal-weight-baselines",
        member_kind="backtest",
        members=planned_hashes,
    )
    print(f"Frozen expected baseline population: {baseline_population.hash}")
else:
    print("Preview backtests remain outside official populations and selection.")

# %% [markdown]
# ## Run every catalog row through the FX engine
#
# Each selected row produces an independent backtest. The loop has no exception-and-continue path:
# one failed member leaves the predeclared population incomplete and stops publication.

# %% tags=["results"]
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
backtest_rows

# %% [markdown]
# ## Validate the frozen baseline population
#
# This check can pass only when every planned model, checkpoint, and portfolio-size member completed.

# %% tags=["results"]
if not include_preview:
    if baseline_population is None:
        raise RuntimeError("the canonical baseline population was not frozen before execution")
    baseline_population.require_complete()
    print(f"Official equal-weight population: {baseline_population.hash}")

# %% [markdown]
# ## Key takeaways
#
# - The baseline evaluates every complete model configuration and checkpoint.
# - Catalog rows pass directly to the existing FX backtest engine.
# - Immutable population membership makes missing or failed jobs visible.
