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
# # US Equities Panel: Transaction-Cost Sensitivity
#
# Cost sensitivity is not a selection step. For each label, this notebook selects the highest
# validation-Sharpe configuration from the complete equal-weight and alternative-sizing populations,
# then applies every declared cost value to that fixed configuration. The resulting cost rows cannot
# replace the configuration chosen from the selection-eligible stages.
#
# **Learning objectives**
#
# - Fix one selection-eligible strategy configuration per label before changing costs.
# - Compare percentage and per-share cost specifications without changing strategy identity.
# - Plan, execute, and validate the complete cost-sensitivity population.
#
# **Book reference**: Chapter 18, Sections 18.2-18.5
#
# **Prerequisites**: `16_backtest.py` and `17_portfolio_management.py` publish the strategy sets
# consumed here.

# %%
"""Generate the US-equities cost-sensitivity validation population."""

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
from case_studies.utils.backtest_loaders import load_backtest_prices_for, warmup_periods_for
from case_studies.utils.sweep_config import (
    get_cost_grid_bps,
    get_cost_grid_half_spread_usd,
    get_per_share_commission,
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
ALLOCATION_SET_NAMES = [
    "us-equities-fwd-ret-1d-allocation-v1",
    "us-equities-fwd-ret-5d-allocation-v1",
    "us-equities-fwd-ret-21d-allocation-v1",
]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_MAX_SOURCE_ROWS = 0
PREVIEW_MAX_COST_VALUES = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## Open the selection-eligible strategy rows
#
# Canonical execution reopens the named equal-weight and allocation sets. A reduced proof selects
# preview rows by visible labels and explicit source-row, cost-value, and symbol limits.

# %%
declared_set_names = [*BASELINE_SET_NAMES, *ALLOCATION_SET_NAMES]
# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS or PREVIEW_MAX_SOURCE_ROWS or PREVIEW_MAX_COST_VALUES or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not declared_set_names or len(declared_set_names) != len(set(declared_set_names)):
        raise ValueError("Canonical execution requires unique named strategy sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if (
        not PREVIEW_LABELS
        or PREVIEW_MAX_SOURCE_ROWS < 1
        or PREVIEW_MAX_COST_VALUES < 1
        or MAX_SYMBOLS < 1
    ):
        raise ValueError(
            "Preview execution requires labels and explicit row, cost, and symbol limits"
        )
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

# %% [markdown]
# ## Select eligible source rows
#
# Canonical rows come from the named baseline and allocation sets. Preview rows use the visible
# label and row limits declared above.

# %%
backtest_catalog = study.backtests.table(include_preview=True)
if EXECUTION_TIER == "canonical":
    declared_sets = tuple(CandidateSet.one(study, name=name) for name in declared_set_names)
    if any(result_set.member_kind != "backtest" for result_set in declared_sets):
        raise ValueError("Every declared input set must contain backtests")
    source_members = tuple(member for result_set in declared_sets for member in result_set.members)
    if len(source_members) != len(set(source_members)):
        raise ValueError("Declared baseline and allocation sets overlap")
    eligible = backtest_catalog.filter(pl.col("backtest_hash").is_in(source_members))
    if eligible.height != len(source_members):
        raise ValueError("The backtest catalog does not contain every declared strategy member")
else:
    eligible = (
        backtest_catalog.filter(
            (pl.col("execution_tier") == "preview")
            & pl.col("stage").is_in(["signal", "allocation"])
            & pl.col("label").is_in(PREVIEW_LABELS)
        )
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(PREVIEW_MAX_SOURCE_ROWS)
    )

ineligible = eligible.filter(
    (pl.col("split") != "validation")
    | (pl.col("execution_tier") != EXECUTION_TIER)
    | ~pl.col("stage").is_in(["signal", "allocation"])
    | ~pl.col("complete")
    | pl.col("sharpe").is_null()
    | ~pl.col("sharpe").is_finite()
)
if eligible.is_empty() or not ineligible.is_empty():
    raise ValueError("Cost analysis requires complete finite selection-eligible validation rows")

# %% [markdown]
# ## Fix one source configuration per label
#
# The configured shortlist size for cost sensitivity is applied within each label. The exact model
# checkpoint, signal, and allocation decisions remain fixed across every cost value.

# %% tags=["results"]
# Prices are loaded with the warmup prefix the declared allocators need. get_allocators injects
# vol_window 63 for inverse_vol, risk_parity and hrp, and lookback 126 for mvo_ledoit_wolf, and
# warmup_periods_for resolves the maximum of those from setup.yaml (126 here). Loading with the
# default warmup_periods=0 leaves an allocator estimating a 126-day covariance from whatever bars
# happen to precede the first decision date - an estimate over a truncated history that completes
# and reports rather than failing.
WARMUP_PERIODS = warmup_periods_for(CASE_STUDY_ID)

top_n = get_top_n_predictions(CASE_STUDY_ID, "cost_sensitivity")
selected_parts = []
for label in eligible.get_column("label").unique().sort().to_list():
    selected_parts.append(
        eligible.filter(pl.col("label") == label)
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(top_n)
    )
selected_sources = pl.concat(selected_parts).sort("label", "backtest_hash")
if selected_sources.is_empty():
    raise RuntimeError("No cost-sensitivity source configuration was selected")

selected_sources.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "stage",
    "prediction_hash",
    "backtest_hash",
    "sharpe",
)

# %% [markdown]
# ## Declare and plan both cost regimes
#
# The canonical bps regime splits total per-leg cost evenly between commission and slippage. The
# exploratory per-share regime uses the configured commission and a uniform half-spread. Every
# planned identity is snapshotted before the first cost backtest runs.

# %%
bps_values = get_cost_grid_bps(CASE_STUDY_ID)
half_spread_values = get_cost_grid_half_spread_usd(CASE_STUDY_ID)
if EXECUTION_TIER == "preview":
    bps_values = bps_values[:PREVIEW_MAX_COST_VALUES]
    half_spread_values = half_spread_values[:PREVIEW_MAX_COST_VALUES]
if not bps_values or not half_spread_values:
    raise ValueError("Both configured cost regimes require at least one value")

per_share_commission = get_per_share_commission(CASE_STUDY_ID)

# %%
cost_requests = [
    {
        "regime": "bps",
        "cost_value": total_bps,
        "costs": {
            "model": "percentage",
            "commission_bps": total_bps / 2,
            "slippage_bps": total_bps / 2,
        },
    }
    for total_bps in bps_values
]
cost_requests.extend(
    {
        "regime": "per_share",
        "cost_value": half_spread,
        "costs": {
            "model": "per_share_plus_spread",
            "per_share": per_share_commission,
            "minimum": 0.0,
            "default_half_spread_usd": half_spread,
            "spread_convention": "half_spread",
        },
    }
    for half_spread in half_spread_values
)

prediction_catalog = study.predictions.table(include_preview=True)
planned_requests = []
plan_rows = []


# %%
def cost_member_records(
    label, source_row, selection, signal, allocation, cost_request, expected_hash
):
    request = {
        "label": label,
        "selection": selection,
        "signal": signal,
        "allocation": allocation,
        "costs": cost_request["costs"],
        "regime": cost_request["regime"],
        "cost_value": cost_request["cost_value"],
        "prediction_hash": source_row["prediction_hash"],
        "source_backtest_hash": source_row["backtest_hash"],
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "source_stage": source_row["stage"],
        "source_backtest_hash": source_row["backtest_hash"],
        "regime": cost_request["regime"],
        "cost_value": cost_request["cost_value"],
        "prediction_hash": source_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
def plan_cost_member(label, prices, cost_request, source_row):
    selected_prediction = prediction_catalog.filter(
        pl.col("prediction_hash") == source_row["prediction_hash"]
    )
    if selected_prediction.height != 1:
        raise ValueError("A cost source must resolve one prediction catalog row")
    source_spec = json.loads(source_row["spec_json"])
    signal = dict(source_spec["strategy"]["signal"])
    allocation = source_spec["strategy"].get("allocation")
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        allocation=allocation,
        costs=cost_request["costs"],
        prices=prices,
        chapter="ch18",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One cost request must plan one backtest")
    return cost_member_records(
        label,
        source_row,
        selected_prediction,
        signal,
        allocation,
        cost_request,
        plan.expected_hashes[0],
    )


# %%
for label in selected_sources.get_column("label").unique().sort().to_list():
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
        warmup_periods=WARMUP_PERIODS,
    )
    for source_row in selected_sources.filter(pl.col("label") == label).iter_rows(named=True):
        for cost_request in cost_requests:
            request, row = plan_cost_member(label, prices, cost_request, source_row)
            planned_requests.append(request)
            plan_rows.append(row)
    del prices

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "regime", "cost_value", "source_backtest_hash", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The cost plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-cost-sensitivity-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## Execute the planned cost members
#
# Each source prediction row is passed directly to the shared runner. Failures do not erase the
# official snapshot or prevent independent siblings from completing and becoming reusable.

# %%
execution_rows = []
failure_rows = []


def execute_cost_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        allocation=request["allocation"],
        costs=request["costs"],
        prices=prices,
        chapter="ch18",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Cost execution changed its planned identity")
    return {
        "label": request["label"],
        "source_backtest_hash": request["source_backtest_hash"],
        "regime": request["regime"],
        "cost_value": request["cost_value"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in selected_sources.get_column("label").unique().sort().to_list():
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
        warmup_periods=WARMUP_PERIODS,
    )
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            execution_rows.append(execute_cost_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "source_backtest_hash": request["source_backtest_hash"],
                    "regime": request["regime"],
                    "cost_value": request["cost_value"],
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
        "source_backtest_hash": pl.String,
        "regime": pl.String,
        "cost_value": pl.Float64,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "source_backtest_hash": pl.String,
        "regime": pl.String,
        "cost_value": pl.Float64,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Cost population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## Freeze the reader-facing cost sets
#
# Each label gets one immutable cost-sensitivity set containing both declared regimes. The strategy
# analysis notebook may describe these rows, but they remain outside the official selection pool.

# %% tags=["results"]
set_rows = []
completed = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed.height != planned_population.height
    or completed.filter(~pl.col("complete")).height
    or completed.filter(pl.col("stage") != "cost_sensitivity").height
    or completed.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The cost catalog is incomplete or mis-staged")
if EXECUTION_TIER == "canonical":
    for label in completed.get_column("label").unique().sort().to_list():
        label_name = label.replace("_", "-")
        # No comparison_contract, matching cme_futures/research_workflow.py:811, which builds the
        # same per-label pool across the full funnel and declares nothing. An empty contract makes
        # every protocol field required-constant, which is the guard: if two members disagree on
        # `cv` they measured their Sharpe on different folds and ranking them is not a comparison,
        # and this field is the only thing checking that. Latent-factor members will refuse on
        # `feature_artifacts` when they enter this pool - latent builds it from a different object
        # than the other five families (latent_factors/case_study.py:337-383, carrying the label
        # digest and setup.yaml bytes). That refusal is a known adapter defect surfacing, not a
        # property to declare around; report it rather than adding the field here.
        result_set = study.backtests.freeze(
            completed.filter(pl.col("label") == label),
            name=f"us-equities-{label_name}-cost-sensitivity-v1",
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
# ## Inspect both cost regimes
#
# Each curve keeps its source model, checkpoint, signal, and allocation fixed. Only the declared
# transaction-cost value changes along the horizontal axis.

# %% tags=["results"]
cost_results = planned_population.select("label", "regime", "cost_value", "backtest_hash").join(
    completed.select("backtest_hash", "sharpe"),
    on="backtest_hash",
    how="inner",
    validate="1:1",
)
if cost_results.height != planned_population.height:
    raise RuntimeError("The plotted cost population differs from the planned population")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
regime_labels = {
    "bps": "Total cost (bps per leg)",
    "per_share": "Half-spread (USD per share)",
}
for ax, regime in zip(axes, ("bps", "per_share"), strict=True):
    regime_rows = cost_results.filter(pl.col("regime") == regime)
    for label in regime_rows.get_column("label").unique().sort().to_list():
        curve = regime_rows.filter(pl.col("label") == label).sort("cost_value")
        ax.plot(curve["cost_value"], curve["sharpe"], marker="o", label=label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(regime_labels[regime])
    ax.set_title(regime.replace("_", " ").title())
    ax.legend(fontsize=8)
axes[0].set_ylabel("Validation Sharpe")
fig.suptitle("Validation Sharpe across declared transaction costs")
fig.tight_layout()
fig.show()

# %% [markdown]
# The risk notebook uses the same fixed source configuration from the selection-eligible baseline
# and allocation sets. Cost rows do not affect that choice.

# %% [markdown]
# ## Key takeaways and limitations
#
# - A cost curve varies transaction-cost assumptions while retaining the source model, checkpoint,
#   signal, and allocation.
# - Cost-sensitivity rows remain outside the validation selection population.
# - The percentage and per-share regimes answer different implementation questions and retain
#   separate identities.
# - Uniform cost schedules summarize execution frictions; security-specific liquidity and realized
#   fills require additional data.
