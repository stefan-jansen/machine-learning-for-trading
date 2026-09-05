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
# # US Equities Panel: Risk Overlays
#
# For each label, this notebook selects the highest validation-Sharpe configuration from the complete
# equal-weight and alternative-sizing populations, then applies every declared risk overlay to that
# fixed configuration. It finally freezes the union of equal-weight, allocation, and risk-overlay
# rows as the immutable population used for official validation selection.
#
# **Learning objectives**
#
# - Fix one selection-eligible strategy configuration per label before applying risk controls.
# - Hold model, checkpoint, signal, and allocation decisions fixed across overlays.
# - Freeze the complete validation population used by the research lifecycle.
#
# **Book reference**: Chapter 19, Sections 19.3-19.6
#
# **Prerequisites**: `16_backtest.py` and `17_portfolio_management.py` publish the strategy sets
# consumed here.

# %%
"""Generate risk overlays and freeze the official US-equities validation set."""

import json
import os
from pathlib import Path

import polars as pl

from case_studies.research import (
    CandidateSet,
    OfficialPopulation,
    Study,
    open_study,
    plan_backtests,
    run_backtests,
)
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.backtest_loaders import load_backtest_prices_for, warmup_periods_for
from case_studies.utils.sweep_config import (
    get_portfolio_risk_controls,
    get_position_risk_controls,
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
VALIDATION_SET_NAME_TEMPLATE = "us-equities-{label}-validation-strategies-v1"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_MAX_SOURCE_ROWS = 0
PREVIEW_MAX_RISK_CONTROLS = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## Open the selection-eligible strategy rows
#
# Canonical execution reopens the named equal-weight and allocation sets. A reduced proof selects
# preview rows by visible labels and explicit source-row, risk-control, and symbol limits.

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
    if PREVIEW_LABELS or PREVIEW_MAX_SOURCE_ROWS or PREVIEW_MAX_RISK_CONTROLS or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not declared_set_names or len(declared_set_names) != len(set(declared_set_names)):
        raise ValueError("Canonical execution requires unique named strategy sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if (
        not PREVIEW_LABELS
        or PREVIEW_MAX_SOURCE_ROWS < 1
        or PREVIEW_MAX_RISK_CONTROLS < 1
        or MAX_SYMBOLS < 1
    ):
        raise ValueError(
            "Preview execution requires labels and explicit row, risk, and symbol limits"
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
    raise ValueError("Risk overlays require complete finite selection-eligible validation rows")

# %% [markdown]
# ## Fix one source configuration per label
#
# The exact model checkpoint, signal, and allocation decisions from the highest validation-Sharpe
# source row remain fixed while the risk decision changes.

# %% tags=["results"]
# Prices are cached by (label, warmup) rather than loaded once per label. Strategy._build_spec
# (research/strategy.py:389) digests exactly the frame it is handed, and strategy_warmup_periods
# (:201-211) resolves a different prefix per allocator: 0 for the non-moment methods, vol_window
# for inverse_vol / risk_parity / hrp, lookback for mvo and mvo_ledoit_wolf. Handing every member
# of a label the same 126-bar frame stamps a price digest that 20_strategy_analysis recomputes at
# the member's own warmup (20:157-169) and then rejects as "does not use canonical validation
# prices" - and lifecycle.evaluate_holdout (lifecycle.py:342-368) applies the same rule, so the
# holdout inherits it. cme_futures/research_workflow.py:674-682 caches on the same key.
_price_cache: dict[tuple[str, int], object] = {}


def prices_for(label, warmup_periods):
    key = (str(label), int(warmup_periods))
    if key not in _price_cache:
        _price_cache[key] = load_backtest_prices_for(
            CASE_STUDY_ID,
            label,
            split="validation",
            max_symbols=MAX_SYMBOLS,
            warmup_periods=int(warmup_periods),
        )
    return _price_cache[key]


top_n = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
selected_parts = []
for label in eligible.get_column("label").unique().sort().to_list():
    selected_parts.append(
        eligible.filter(pl.col("label") == label)
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(top_n)
    )
selected_sources = pl.concat(selected_parts).sort("label", "backtest_hash")
if selected_sources.is_empty():
    raise RuntimeError("No risk-overlay source configuration was selected")

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
# ## Declare and plan every risk overlay
#
# Position and portfolio controls come from the case-study configuration. Every planned identity is
# snapshotted before the first overlay runs, so unsuccessful controls remain visible in the official
# population and completed siblings can be reused on restart.

# %%
risk_requests = []
for control in get_position_risk_controls(CASE_STUDY_ID):
    if control["type"] == "time_exit":
        rule = {"type": control["type"], "bars": control["bars"]}
    else:
        rule = {"type": control["type"], "threshold": control["threshold"]}
    risk_requests.append(
        {
            "name": control["name"],
            "spec": {"name": control["name"], "position_rules": [rule]},
        }
    )
for control in get_portfolio_risk_controls(CASE_STUDY_ID):
    risk_requests.append(
        {
            "name": control["name"],
            "spec": {
                "name": control["name"],
                "portfolio_limits": [{"type": control["type"], "threshold": control["threshold"]}],
            },
        }
    )
if EXECUTION_TIER == "preview":
    risk_requests = risk_requests[:PREVIEW_MAX_RISK_CONTROLS]
if not risk_requests or len({request["name"] for request in risk_requests}) != len(risk_requests):
    raise ValueError("Risk controls must be non-empty and uniquely named")

prediction_catalog = study.predictions.table(include_preview=True)
planned_requests = []
plan_rows = []


# %%
def risk_member_records(
    label, source_row, selection, signal, allocation, risk_request, expected_hash
):
    request = {
        "label": label,
        "selection": selection,
        "signal": signal,
        "allocation": allocation,
        "risk": risk_request["spec"],
        "risk_name": risk_request["name"],
        "prediction_hash": source_row["prediction_hash"],
        "source_backtest_hash": source_row["backtest_hash"],
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "source_stage": source_row["stage"],
        "source_backtest_hash": source_row["backtest_hash"],
        "risk": risk_request["name"],
        "prediction_hash": source_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
def plan_risk_member(label, prices, risk_request, source_row):
    selected_prediction = prediction_catalog.filter(
        pl.col("prediction_hash") == source_row["prediction_hash"]
    )
    if selected_prediction.height != 1:
        raise ValueError("A risk source must resolve one prediction catalog row")
    source_spec = json.loads(source_row["spec_json"])
    signal = dict(source_spec["strategy"]["signal"])
    allocation = source_spec["strategy"].get("allocation")
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        allocation=allocation,
        risk=risk_request["spec"],
        prices=prices,
        chapter="ch19",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One risk request must plan one backtest")
    return risk_member_records(
        label,
        source_row,
        selected_prediction,
        signal,
        allocation,
        risk_request,
        plan.expected_hashes[0],
    )


# %%
for label in selected_sources.get_column("label").unique().sort().to_list():
    for source_row in selected_sources.filter(pl.col("label") == label).iter_rows(named=True):
        for risk_request in risk_requests:
            prices = prices_for(
                label,
                # The source's allocation lives in its spec_json, not as a catalog column, so
                # source_row.get("allocation") is always None and would silently warm up 0 bars
                # for every allocation-stage source.
                strategy_warmup_periods(json.loads(source_row["spec_json"])),
            )
            request, row = plan_risk_member(label, prices, risk_request, source_row)
            planned_requests.append(request)
            plan_rows.append(row)

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "risk", "source_backtest_hash", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The risk plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-risk-overlay-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## Execute the planned overlay members
#
# Each source prediction row is passed directly to the shared runner. The pass attempts every
# independent member before reporting an incomplete official population.

# %%
execution_rows = []
failure_rows = []


def execute_risk_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        allocation=request["allocation"],
        risk=request["risk"],
        prices=prices,
        chapter="ch19",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Risk execution changed its planned identity")
    return {
        "label": request["label"],
        "source_backtest_hash": request["source_backtest_hash"],
        "risk": request["risk_name"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in selected_sources.get_column("label").unique().sort().to_list():
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            prices = prices_for(
                label,
                strategy_warmup_periods({"strategy": {"allocation": request["allocation"]}}),
            )
            execution_rows.append(execute_risk_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "source_backtest_hash": request["source_backtest_hash"],
                    "risk": request["risk_name"],
                    "backtest_hash": request["expected_hash"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

# %% tags=["results"]
execution_diagnostics = pl.DataFrame(
    execution_rows,
    schema={
        "label": pl.String,
        "source_backtest_hash": pl.String,
        "risk": pl.String,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "source_backtest_hash": pl.String,
        "risk": pl.String,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Risk population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## Freeze the risk and official validation sets
#
# Both sets are frozen per label, and there is no cross-label union. `20_strategy_analysis` ranks
# validation Sharpe within one label, so a union is not something it can open - and it could not be
# created anyway: three labels means three `label_artifact` values, and `CandidateSet.create`
# refuses any field that differs without being declared comparable
# (research/comparison.py:65-72). cme_futures builds the same shape - a per-label pool with no
# contract at `research_workflow.py:811`, and a cross-label pool that declares
# `cv`, `feature_artifacts` and `label_artifact` at `:830-835` because those are exactly what
# varies across labels. This case study needs only the first.

# %% tags=["results"]
set_rows = []
completed_risk = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed_risk.height != planned_population.height
    or completed_risk.filter(~pl.col("complete")).height
    or completed_risk.filter(pl.col("stage") != "risk_overlay").height
    or completed_risk.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed_risk.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The risk catalog is incomplete or mis-staged")
if EXECUTION_TIER == "canonical":
    for label in completed_risk.get_column("label").unique().sort().to_list():
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
            completed_risk.filter(pl.col("label") == label),
            name=f"us-equities-{label_name}-risk-overlay-v1",
        )
        set_rows.append(
            {"label": label, "set_name": result_set.name, "members": len(result_set.members)}
        )

        # The selection pool this label's holdout is chosen from: its baseline and allocation
        # members plus the risk overlays just published. No contract - one label means one
        # label_artifact, and every other protocol field being required-constant is the guard.
        validation_candidates = pl.concat(
            [
                eligible.filter(pl.col("label") == label),
                completed_risk.filter(pl.col("label") == label),
            ]
        ).sort("backtest_hash")
        if (
            validation_candidates.get_column("backtest_hash").n_unique()
            != validation_candidates.height
        ):
            raise ValueError(f"Selection-eligible strategy sets overlap for {label}")
        validation_set = study.backtests.freeze(
            validation_candidates,
            name=VALIDATION_SET_NAME_TEMPLATE.format(label=label_name),
        )
        set_rows.append(
            {
                "label": label,
                "set_name": validation_set.name,
                "members": len(validation_set.members),
            }
        )


compatible_sets = pl.DataFrame(
    set_rows,
    schema={"label": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `20_strategy_analysis.py` reopens one of these per-label sets -
# `us-equities-<label>-validation-strategies-v1` - and applies the one official rule: highest
# validation backtest Sharpe with the backtest hash as deterministic tie-break, within that label.
# The holdout follows from that selection with nothing in between: retrain the selected
# configuration on everything up to the holdout start, predict the holdout window, and run the
# same backtest configuration on those predictions.

# %% [markdown]
# ## Key takeaways and limitations
#
# - Risk-overlay requests retain the source strategy decisions and change only the declared control.
# - The official population contains every complete equal-weight, allocation, and risk-overlay
#   validation result.
# - Validation Sharpe and the backtest hash define deterministic selection from that population.
# - The holdout is used once after selection and may disconfirm the validation evidence.
