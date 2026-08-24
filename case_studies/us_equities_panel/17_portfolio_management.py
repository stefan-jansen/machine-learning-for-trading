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
# # US Equities Panel: Alternative Position Sizing
#
# The equal-weight baseline supplies a complete immutable population. Within each label, this
# notebook ranks baseline validation Sharpe, retains one checkpoint and signal decision per
# distinct model configuration, and applies every declared alternative sizing method. Equal weight
# is excluded because it is the baseline and would produce the same backtest identity.
#
# **Learning objectives**
#
# - Derive the allocation shortlist from complete equal-weight validation results.
# - Hold model, checkpoint, and signal decisions fixed while changing position sizing.
# - Plan, execute, and validate the complete alternative-sizing population.
#
# **Book reference**: Chapter 17, Sections 17.2-17.8
#
# **Prerequisites**: `16_backtest.py` publishes the compatible equal-weight baseline sets.

# %%
"""Generate the US-equities allocation-stage validation population."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from case_studies.research import (
    CandidateSet,
    OfficialPopulation,
    Study,
    open_study,
    plan_backtests,
    run_backtests,
)
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_top_n_predictions,
)
from utils.paths import REPO_ROOT

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
BASELINE_SET_NAMES = [
    "us-equities-fwd-ret-1d-baseline-v1",
    "us-equities-fwd-ret-5d-baseline-v1",
    "us-equities-fwd-ret-21d-baseline-v1",
]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_MAX_BASELINE_ROWS = 0
PREVIEW_MAX_ALLOCATORS = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## Open and validate the baseline population
#
# Canonical execution resolves the named sets produced by the baseline notebook. A reduced proof
# selects preview baseline rows by visible labels and explicit row, allocator, and symbol limits.

# %%
# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS or PREVIEW_MAX_BASELINE_ROWS or PREVIEW_MAX_ALLOCATORS or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not BASELINE_SET_NAMES or len(BASELINE_SET_NAMES) != len(set(BASELINE_SET_NAMES)):
        raise ValueError("Canonical execution requires unique named baseline sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if (
        not PREVIEW_LABELS
        or PREVIEW_MAX_BASELINE_ROWS < 1
        or PREVIEW_MAX_ALLOCATORS < 1
        or MAX_SYMBOLS < 1
    ):
        raise ValueError(
            "Preview execution requires labels and explicit row, allocator, and symbol limits"
        )
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

# %% [markdown]
# ## Select eligible baseline rows
#
# Canonical rows come from the named immutable sets. Preview rows use the visible label and row
# limits declared above.

# %%
backtest_catalog = study.backtests.table(include_preview=True)
if EXECUTION_TIER == "canonical":
    baseline_sets = tuple(CandidateSet.one(study, name=name) for name in BASELINE_SET_NAMES)
    if any(result_set.member_kind != "backtest" for result_set in baseline_sets):
        raise ValueError("Every declared baseline set must contain backtests")
    baseline_members = tuple(
        member for result_set in baseline_sets for member in result_set.members
    )
    if len(baseline_members) != len(set(baseline_members)):
        raise ValueError("Declared baseline sets overlap")
    baseline = backtest_catalog.filter(pl.col("backtest_hash").is_in(baseline_members))
    if baseline.height != len(baseline_members):
        raise ValueError("The backtest catalog does not contain every baseline member")
else:
    baseline = (
        backtest_catalog.filter(
            (pl.col("execution_tier") == "preview")
            & (pl.col("stage") == "signal")
            & pl.col("label").is_in(PREVIEW_LABELS)
        )
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(PREVIEW_MAX_BASELINE_ROWS)
    )

ineligible = baseline.filter(
    (pl.col("split") != "validation")
    | (pl.col("execution_tier") != EXECUTION_TIER)
    | (pl.col("stage") != "signal")
    | ~pl.col("complete")
    | pl.col("sharpe").is_null()
    | ~pl.col("sharpe").is_finite()
)
if baseline.is_empty() or not ineligible.is_empty():
    raise ValueError("Allocation requires complete finite equal-weight validation rows")

# %% [markdown]
# ## Select the declared baseline survivors
#
# `top_n` counts distinct `(family, config_name)` pairs within each label. Sorting first retains the
# exact prediction checkpoint and equal-weight signal decision associated with the highest baseline
# validation Sharpe for that model configuration.

# %% tags=["results"]
top_n = get_top_n_predictions(CASE_STUDY_ID, "allocation")
checkpoints_per_config = get_checkpoints_per_config(CASE_STUDY_ID)
if checkpoints_per_config != 1:
    raise ValueError(
        "backtest.sweep.checkpoints_per_config is "
        f"{checkpoints_per_config}; this notebook advances one checkpoint per "
        "model configuration"
    )
shortlist_parts = []
for label in baseline.get_column("label").unique().sort().to_list():
    ranked = baseline.filter(pl.col("label") == label).sort(
        "sharpe", "backtest_hash", descending=[True, False]
    )
    shortlist_parts.append(
        ranked.unique(
            subset=["family", "config_name"],
            keep="first",
            maintain_order=True,
        ).head(top_n)
    )
shortlist = pl.concat(shortlist_parts).sort("label", "sharpe", descending=[False, True])
if shortlist.is_empty():
    raise RuntimeError("The equal-weight baseline produced no allocation survivors")

shortlist.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "prediction_hash",
    "backtest_hash",
    "sharpe",
)

# %% [markdown]
# ## Plan every alternative sizing member
#
# The selected baseline row supplies the prediction and signal decision. The only new decision is
# position sizing. Every planned identity is snapshotted before execution so an unsuccessful member
# cannot disappear from the allocation population.

# %%
allocators = [
    config for config in get_allocators(CASE_STUDY_ID) if config["method"] != "equal_weight"
]
if EXECUTION_TIER == "preview":
    allocators = allocators[:PREVIEW_MAX_ALLOCATORS]
if not allocators or any(config["method"] == "equal_weight" for config in allocators):
    raise ValueError("Allocation requires at least one non-baseline sizing method")

prediction_catalog = study.predictions.table(include_preview=True)
backtest_config = get_backtest_config(CASE_STUDY_ID)
planned_requests = []
plan_rows = []


# %%
def plan_allocation_member(label, prices, allocation, baseline_row):
    selected_prediction = prediction_catalog.filter(
        pl.col("prediction_hash") == baseline_row["prediction_hash"]
    )
    if selected_prediction.height != 1:
        raise ValueError("A baseline survivor must resolve one prediction catalog row")
    baseline_spec = json.loads(baseline_row["spec_json"])
    signal = dict(baseline_spec["strategy"]["signal"])
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        allocation=allocation,
        prices=prices,
        chapter="ch17",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One allocation request must plan one backtest")
    expected_hash = plan.expected_hashes[0]
    request = {
        "label": label,
        "selection": selected_prediction,
        "signal": signal,
        "allocation": allocation,
        "prediction_hash": baseline_row["prediction_hash"],
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "family": baseline_row["family"],
        "config_name": baseline_row["config_name"],
        "checkpoint_kind": baseline_row["checkpoint_kind"],
        "checkpoint_value": baseline_row["checkpoint_value"],
        "allocation": allocation["method"],
        "prediction_hash": baseline_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
for label in shortlist.get_column("label").unique().sort().to_list():
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
    )
    for baseline_row in shortlist.filter(pl.col("label") == label).iter_rows(named=True):
        for allocation in allocators:
            request, row = plan_allocation_member(label, prices, allocation, baseline_row)
            planned_requests.append(request)
            plan_rows.append(row)
    del prices

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "family", "config_name", "checkpoint_value", "allocation", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The allocation plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-allocation-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## Execute the planned members
#
# Each selected prediction row is passed directly to the shared runner. Completed siblings remain
# reusable if another sizing member fails, and the notebook raises after attempting the full pass.

# %%
execution_rows = []
failure_rows = []


def execute_allocation_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        allocation=request["allocation"],
        prices=prices,
        chapter="ch17",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Allocation execution changed its planned identity")
    return {
        "label": request["label"],
        "prediction_hash": request["prediction_hash"],
        "allocation": request["allocation"]["method"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in shortlist.get_column("label").unique().sort().to_list():
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
    )
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            execution_rows.append(execute_allocation_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "prediction_hash": request["prediction_hash"],
                    "allocation": request["allocation"]["method"],
                    "backtest_hash": request["expected_hash"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    del prices

# %% tags=["results"]
execution_diagnostics = pl.DataFrame(
    execution_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "allocation": pl.String,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "allocation": pl.String,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Allocation population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## Freeze the reader-facing allocation sets
#
# Each label gets one immutable set of alternative sizing results. Costs and risk controls derive
# their inputs from the union of the matching baseline and allocation sets.

# %% tags=["results"]
set_rows = []
completed = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed.height != planned_population.height
    or completed.filter(~pl.col("complete")).height
    or completed.filter(pl.col("stage") != "allocation").height
    or completed.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The allocation catalog is incomplete or mis-staged")
if EXECUTION_TIER == "canonical":
    for label in completed.get_column("label").unique().sort().to_list():
        label_name = label.replace("_", "-")
        result_set = study.backtests.freeze(
            completed.filter(pl.col("label") == label),
            name=f"us-equities-{label_name}-allocation-v1",
        )
        set_rows.append(
            {"label": label, "set_name": result_set.name, "members": len(result_set.members)}
        )

compatible_sets = pl.DataFrame(
    set_rows,
    schema={"label": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# ## Inspect the alternative-sizing population
#
# Each point is one complete allocation-stage validation backtest. The chart retains every planned
# result and groups them only by the allocator named in the request.

# %% tags=["results"]
allocation_results = planned_population.select("label", "allocation", "backtest_hash").join(
    completed.select("backtest_hash", "sharpe"),
    on="backtest_hash",
    how="inner",
    validate="1:1",
)
if allocation_results.height != planned_population.height:
    raise RuntimeError("The plotted allocation population differs from the planned population")

fig, ax = plt.subplots(figsize=(10, 5))
for label in allocation_results.get_column("label").unique().sort().to_list():
    label_rows = allocation_results.filter(pl.col("label") == label)
    ax.scatter(
        label_rows["allocation"],
        label_rows["sharpe"],
        alpha=0.5,
        s=20,
        label=label,
    )
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Allocator")
ax.set_ylabel("Validation Sharpe")
ax.set_title("Alternative-sizing validation Sharpe by allocator")
ax.tick_params(axis="x", rotation=25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.show()

# %% [markdown]
# The cost and risk notebooks reopen these names together with the matching equal-weight baseline.
# No model is retrained, and the selected checkpoint remains part of every downstream identity.

# %% [markdown]
# ## Key takeaways and limitations
#
# - Allocation requests change position sizing while retaining the selected model, checkpoint, and
#   signal rule.
# - The shortlist uses validation backtest Sharpe within each label and distinct model
#   configuration.
# - The population snapshot preserves every planned allocator result across failure and restart.
# - These comparisons remain validation evidence; costs, risk controls, and the locked holdout are
#   evaluated separately.
