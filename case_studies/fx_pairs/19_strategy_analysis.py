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
# # Strategy Analysis - FX Pairs
#
# This notebook reads back the selection made from the immutable validation candidate set and the
# holdout lineage that selection determines, and assesses both. Selection uses validation backtest
# Sharpe with the backtest identity as the deterministic tie-breaker. Cost sensitivity is excluded
# from selection, and nothing measured on the holdout can revise the choice - not because a lock
# forbids it, but because the choice was made upstream against a set that is already frozen.
#
# The holdout results are produced by `17_holdout_predictions` and `18_holdout_backtest`. This
# notebook writes nothing; it fails if either is missing rather than producing them itself, so
# reading the holdout and deciding what to run against it stay separate acts.
#
# **Learning objectives**
#
# - Reproduce validation selection from an immutable, complete candidate set.
# - Verify that the holdout lineage on record is the one the selection determines.
# - Interpret cost and risk variants through controlled sibling comparisons.
# - Assess validation and holdout performance with interval and paired evidence.
#
# **Book reference**: Chapters 16-20
#
# **Prerequisites**: `17_holdout_predictions` and `18_holdout_backtest`, and the candidate set
# `15_risk_management` freezes.

# %%
"""Read back the selected FX validation lineage and its holdout, and assess both."""

import json
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import polars as pl

from case_studies.research import (
    BacktestResult,
    CandidateSet,
    OfficialPopulation,
    PredictionResult,
    Result,
    TrainingResult,
    open_study,
)
from case_studies.research.holdout import resolve_holdout_selection
from case_studies.utils.backtest_presets import cost_view
from case_studies.utils.registry import load_backtest_metrics, load_paired_metrics
from utils.style import COLORS

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
CANDIDATE_SET_NAME = "fx_pairs:holdout-candidates"

# %% [markdown]
# ## Resolve the selection and its holdout lineage
#
# Selection is not a parameter and is not made here. `15_risk_management` froze an immutable
# candidate set; its highest validation backtest Sharpe is read out of that set, with the backtest
# identity as the deterministic tie-breaker. Cost sensitivity is excluded from the set, so a cost
# variant cannot be selected, and nothing on this page can revise the choice.
#
# The holdout lineage is derived from that selection rather than looked up in a ledger. The
# derivation is pure - the same candidate set yields the same training identity every time - so
# this notebook can state which holdout results it requires before checking whether they exist,
# and name the missing one when they do not. An earlier design recorded the lineage in a research
# lock instead and read it back from a lifecycle table. That made the holdout a one-shot
# transaction, which in turn made it impossible to fix: any correction upstream needed a retrain
# the lock forbade, and the lock could not be reissued.

# %% tags=["results"]
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
selection = resolve_holdout_selection(study, candidate_set_name=CANDIDATE_SET_NAME)

validation_set = selection.candidate_set
selected_validation = selection.validation_backtest
selected_prediction = selection.validation_prediction
selected_training = selection.validation_training
selected_record = selected_validation.registry_record()
selected_prediction_record = selected_prediction.registry_record()
selected_training_record = selected_training.registry_record()
selected_training_spec = selected_training.spec()
selected_computation = selected_training_spec.get("computation", selected_training_spec)

pl.DataFrame(
    {
        "field": [
            "candidate set",
            "candidate count",
            "selected validation",
            "derived holdout training",
        ],
        "value": [
            validation_set.hash,
            str(len(validation_set.members)),
            selected_validation.hash,
            selection.holdout_training_hash,
        ],
    }
)

# %% [markdown]
# ## The selected validation lineage
#
# Model family, configuration, label, checkpoint and the data artifacts the fit read are printed
# together, because they are what the holdout comparison holds fixed. The provenance fields are
# checked rather than displayed: a selected training run with no recorded source commit or runtime
# cannot be reproduced by a reader, and a holdout number from a run nobody can reproduce is not
# evidence of anything.

# %% tags=["results"]
for field in ("label_artifact", "feature_artifacts", "cv"):
    if not selected_computation.get(field):
        raise ValueError(f"the selected training run records no {field}")
if not selected_training_record.get("git_commit"):
    raise ValueError("the selected training run records no source commit")
if not json.loads(selected_training_record.get("runtime_json") or "{}"):
    raise ValueError("the selected training run records no runtime provenance")

selected_identity = pl.DataFrame(
    {
        "field": [
            "label",
            "family",
            "configuration",
            "checkpoint kind",
            "checkpoint value",
            "training hash",
            "prediction hash",
            "validation backtest hash",
            "source commit",
        ],
        "value": [
            str(selected_training_spec["label"]),
            str(selected_training_spec["family"]),
            str(selected_training_spec["config_name"]),
            str(selected_prediction_record["checkpoint_kind"]),
            str(selected_prediction_record["checkpoint_value"]),
            selected_training.hash,
            selected_prediction.hash,
            selected_validation.hash,
            str(selected_training_record["git_commit"]),
        ],
    }
)
selected_identity

# %% [markdown]
# ## Validation candidate evidence
#
# Every candidate remains visible below. The displayed order reproduces the selection rule; IC and
# every metric other than Sharpe remain descriptive.

# %% tags=["results"]


def _metric_row(result: BacktestResult) -> dict[str, Any]:
    metrics = load_backtest_metrics(
        CASE_STUDY_ID,
        backtest_hash=result.hash,
        case_dir=study.root,
    )
    if metrics.height != 1:
        raise ValueError(f"backtest {result.hash} has {metrics.height} metric rows")
    return metrics.row(0, named=True)


candidate_rows = []
for member_hash in validation_set.members:
    result = Result.open(study, member_hash)
    if not isinstance(result, BacktestResult) or not result.complete:
        raise ValueError(f"candidate {member_hash} is not a complete backtest")
    record = result.registry_record()
    lineage = result.lineage()
    training = lineage["training_spec"]
    prediction = Result.open(study, record["prediction_hash"])
    if not isinstance(prediction, PredictionResult):
        raise TypeError(f"candidate {member_hash} does not reference a prediction")
    prediction_record = prediction.registry_record()
    metric = _metric_row(result)
    if record["stage"] not in {"signal", "allocation", "risk_overlay"}:
        raise ValueError(f"candidate {member_hash} has an ineligible stage")
    if any(metric.get(name) is None for name in ("sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi")):
        raise ValueError(f"candidate {member_hash} lacks Sharpe interval evidence")
    candidate_rows.append(
        {
            "backtest_hash": result.hash,
            "prediction_hash": record["prediction_hash"],
            "stage": record["stage"],
            "label": training["label"],
            "family": training["family"],
            "config_name": training["config_name"],
            "checkpoint_kind": prediction_record["checkpoint_kind"],
            "checkpoint_value": prediction_record["checkpoint_value"],
            "sharpe": metric["sharpe"],
            "sharpe_ci95_lo": metric["sharpe_ci95_lo"],
            "sharpe_ci95_hi": metric["sharpe_ci95_hi"],
        }
    )

candidate_evidence = pl.DataFrame(candidate_rows).sort(
    ["sharpe", "backtest_hash"], descending=[True, False]
)
if candidate_evidence["backtest_hash"][0] != selected_validation.hash:
    raise ValueError("displayed evidence disagrees with the candidate-set selection rule")
candidate_evidence

# %% tags=["results"]
candidate_figure = px.scatter(
    candidate_evidence,
    x="sharpe",
    y="family",
    color="stage",
    facet_row="label",
    hover_data=["config_name", "backtest_hash"],
    title="Validation Sharpe distribution for the immutable FX candidate set",
    labels={"sharpe": "Validation Sharpe", "family": "Model family"},
)
candidate_figure.add_vline(x=0, line_dash="dot", line_color=COLORS["recede"])
candidate_figure.show()

# %% [markdown]
# ## Controlled cost and risk comparisons
#
# Cost sensitivity and risk overlays are read from their official populations. A sibling enters a
# comparison only when its prediction, signal, allocation, rebalance, execution, and price identity
# match the selected lineage after removing the field being varied.

# %% tags=["results"]


def _comparison_projection(
    result: BacktestResult,
    *,
    omit_costs: bool,
    omit_risk: bool,
) -> dict[str, Any]:
    projected = deepcopy(result.spec())
    projected.pop("chapter", None)
    projected.pop("_runtime_backtest_config", None)
    if omit_risk:
        projected.get("strategy", {}).pop("risk", None)
    metadata = projected.get("backtest_config", {}).get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("chapter", None)
    if omit_costs:
        config = projected.get("backtest_config", {})
        config.pop("commission", None)
        config.pop("slippage", None)
    return {
        "prediction_hash": result.registry_record()["prediction_hash"],
        "spec": projected,
    }


cost_population = OfficialPopulation.one(study, name=f"{CASE_STUDY_ID}:cost-sensitivity-backtests")
risk_population = OfficialPopulation.one(study, name=f"{CASE_STUDY_ID}:risk-overlay-backtests")
cost_population.require_complete()
risk_population.require_complete()

selected_cost_core = _comparison_projection(selected_validation, omit_costs=True, omit_risk=True)
selected_risk_core = _comparison_projection(selected_validation, omit_costs=False, omit_risk=True)

cost_rows = []
for member_hash in cost_population.members:
    result = Result.open(study, member_hash)
    if not isinstance(result, BacktestResult):
        raise TypeError("cost population contains a non-backtest result")
    if _comparison_projection(result, omit_costs=True, omit_risk=True) != selected_cost_core:
        continue
    costs = cost_view(result.spec())
    metric = _metric_row(result)
    if any(metric.get(name) is None for name in ("sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi")):
        raise ValueError(f"cost sibling {result.hash} lacks Sharpe interval evidence")
    cost_rows.append(
        {
            "total_cost_bps": costs["commission_bps"] + costs["slippage_bps"],
            "sharpe": metric["sharpe"],
            "sharpe_ci95_lo": metric["sharpe_ci95_lo"],
            "sharpe_ci95_hi": metric["sharpe_ci95_hi"],
            "backtest_hash": result.hash,
        }
    )
if not cost_rows:
    raise ValueError("no controlled cost siblings match the selected strategy")
cost_evidence = pl.DataFrame(cost_rows).sort("total_cost_bps")
cost_evidence

# %% tags=["results"]
cost_figure = px.line(
    cost_evidence,
    x="total_cost_bps",
    y="sharpe",
    markers=True,
    title="Validation Sharpe for exact cost siblings of the selected FX strategy",
    labels={"total_cost_bps": "Total cost per traded leg (basis points)", "sharpe": "Sharpe"},
)
cost_figure.add_hline(y=0, line_dash="dot", line_color=COLORS["recede"])
cost_figure.show()

# %% tags=["results"]
risk_rows = []
for member_hash in risk_population.members:
    result = Result.open(study, member_hash)
    if not isinstance(result, BacktestResult):
        raise TypeError("risk population contains a non-backtest result")
    if _comparison_projection(result, omit_costs=False, omit_risk=True) != selected_risk_core:
        continue
    metric = _metric_row(result)
    if any(metric.get(name) is None for name in ("sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi")):
        raise ValueError(f"risk sibling {result.hash} lacks Sharpe interval evidence")
    risk = result.spec()["strategy"]["risk"]
    risk_rows.append(
        {
            "risk_name": risk["name"],
            "sharpe": metric["sharpe"],
            "sharpe_ci95_lo": metric["sharpe_ci95_lo"],
            "sharpe_ci95_hi": metric["sharpe_ci95_hi"],
            "backtest_hash": result.hash,
        }
    )
if not risk_rows:
    raise ValueError("no controlled risk siblings match the selected strategy")
risk_evidence = pl.DataFrame(risk_rows).sort("sharpe", descending=True)
risk_evidence

# %% tags=["results"]
risk_figure = px.bar(
    risk_evidence.sort("sharpe"),
    x="sharpe",
    y="risk_name",
    orientation="h",
    title="Validation Sharpe for exact risk siblings of the selected FX strategy",
    labels={"sharpe": "Sharpe", "risk_name": "Position-risk rule"},
)
risk_figure.add_vline(x=0, line_dash="dot", line_color=COLORS["recede"])
risk_figure.show()

# %% [markdown]
# The cost curve is allowed to be monotone, nonmonotone, entirely positive, or entirely negative.
# The computed summary reports the observed shape without requiring a crossing. Risk rows are ordered
# descriptively and do not replace the candidate-set selection rule.

# %% tags=["results"]
cost_differences = cost_evidence.get_column("sharpe").diff().drop_nulls()
if (cost_differences < 0).all():
    cost_shape = "strictly decreasing over the declared grid"
elif (cost_differences <= 0).all():
    cost_shape = "nonincreasing over the declared grid"
else:
    cost_shape = "nonmonotone over the declared grid"

if (cost_evidence.get_column("sharpe") > 0).all():
    sign_summary = "all observed Sharpe estimates are positive"
elif (cost_evidence.get_column("sharpe") < 0).all():
    sign_summary = "all observed Sharpe estimates are negative"
else:
    sign_summary = "the observed Sharpe estimates include both signs or zero"

controlled_summary = pl.DataFrame(
    {
        "comparison": ["cost sensitivity", "position-risk controls"],
        "finding": [
            f"Sharpe is {cost_shape}; {sign_summary}.",
            (
                f"The highest point estimate belongs to {risk_evidence['risk_name'][0]}; "
                "the interval columns determine whether that ordering is resolved."
            ),
        ],
    }
)
controlled_summary

# %% [markdown]
# ## Require the holdout lineage the selection determines
#
# The holdout results are not looked up by whatever happens to carry the holdout split; they are
# required to be the ones this selection determines. `17_holdout_predictions` refits the derived
# training identity and `18_holdout_backtest` replays the selected strategy against it, so the
# lineage checked here is fully specified before anything is read: a holdout produced from a
# different configuration, a different checkpoint or a different strategy does not resolve, and
# does not silently take the place of the one that should have been produced.

# %% tags=["results"]
holdout_prediction = selection.holdout_prediction
if holdout_prediction is None:
    raise ValueError(
        f"no holdout prediction is registered for training {selection.holdout_training_hash} "
        f"at checkpoint {selection.checkpoint_kind}={selection.checkpoint_value}; "
        "run 17_holdout_predictions"
    )
holdout_backtest = selection.holdout_backtest
if holdout_backtest is None:
    raise ValueError(
        f"holdout prediction {holdout_prediction.hash} has no registered backtest; "
        "run 18_holdout_backtest"
    )
holdout_training = Result.open(study, holdout_prediction.registry_record()["training_hash"])
if not isinstance(holdout_training, TrainingResult) or not holdout_training.complete:
    raise ValueError("the holdout training result is incomplete")
if not isinstance(holdout_prediction, PredictionResult) or not holdout_prediction.complete:
    raise ValueError("the holdout prediction result is incomplete")
if not isinstance(holdout_backtest, BacktestResult) or not holdout_backtest.complete:
    raise ValueError("the holdout backtest result is incomplete")
if any(
    result.execution_tier != "canonical"
    for result in (holdout_training, holdout_prediction, holdout_backtest)
):
    raise ValueError("holdout lineage must use canonical execution")
if holdout_training.hash != selection.holdout_training_hash:
    raise ValueError("the holdout training identity differs from the derived one")
if holdout_training.spec() != selection.holdout_training_spec:
    raise ValueError("the holdout training specification differs from the derived one")
# The strategy is not re-checked here. `HoldoutSelection.holdout_backtest` resolves by the
# replay's own projection - the specification minus the fields that must differ between the two
# intervals - so a backtest whose strategy differed would not have been returned in the first
# place, and asserting it again would be an assertion that cannot fail. What is checked is the
# price identity, which the projection deliberately excludes and which nothing upstream requires.
if not holdout_backtest.spec().get("input_identity", {}).get("prices"):
    raise ValueError("the holdout backtest lacks canonical price identity")

# The holdout fold is the one thing the two runs are supposed to differ in, so it is printed
# rather than only checked: a reader can see which interval the holdout numbers below describe.
_holdout_fold = selection.holdout_training_spec["computation"]["cv"]["folds"][-1]
holdout_identity = pl.DataFrame(
    {
        "field": [
            "holdout training",
            "holdout prediction",
            "holdout backtest",
            "holdout train window",
            "holdout evaluation window",
        ],
        "value": [
            holdout_training.hash,
            holdout_prediction.hash,
            holdout_backtest.hash,
            f"{_holdout_fold['train_start']} to {_holdout_fold['train_end']}",
            f"{_holdout_fold['val_start']} to {_holdout_fold['val_end']}",
        ],
    }
)
holdout_identity

# %% [markdown]
# ## Validation and holdout evidence
#
# Point estimates and intervals are displayed by exact selected identity. Statistical comparisons use
# registered paired evidence. The holdout may disconfirm the validation result and cannot trigger
# fallback or reselection.

# %% tags=["results"]
required_metrics = {
    "sharpe",
    "sharpe_ci95_lo",
    "sharpe_ci95_hi",
    "total_return",
    "max_drawdown",
    "max_dd_ci95_lo",
    "max_dd_ci95_hi",
    "volatility",
    "avg_turnover",
    "num_trades",
}
performance_rows = []
for period, result in (("validation", selected_validation), ("holdout", holdout_backtest)):
    metrics = _metric_row(result)
    if any(
        metrics.get(name) is None or not np.isfinite(metrics[name]) for name in required_metrics
    ):
        raise ValueError(f"{period} result lacks finite performance evidence")
    performance_rows.append(
        {
            "period": period,
            "backtest_hash": result.hash,
            **{name: metrics[name] for name in required_metrics},
        }
    )
selected_performance = pl.DataFrame(performance_rows)
selected_performance

# %% tags=["results"]
performance_figure = px.bar(
    selected_performance,
    x="period",
    y="sharpe",
    error_y=selected_performance.get_column("sharpe_ci95_hi")
    - selected_performance.get_column("sharpe"),
    error_y_minus=selected_performance.get_column("sharpe")
    - selected_performance.get_column("sharpe_ci95_lo"),
    title="Selected FX strategy Sharpe in validation and holdout windows",
    labels={"period": "Window", "sharpe": "Sharpe"},
)
performance_figure.add_hline(y=0, line_dash="dot", line_color=COLORS["recede"])
performance_figure.show()

# %% tags=["results"]
holdout_pairs = load_paired_metrics(
    CASE_STUDY_ID,
    challenger_hash=holdout_backtest.hash,
    case_dir=study.root,
)
validation_pairs = load_paired_metrics(
    CASE_STUDY_ID,
    challenger_hash=selected_validation.hash,
    case_dir=study.root,
)
if holdout_pairs.is_empty() or validation_pairs.is_empty():
    raise ValueError("required paired validation or holdout evidence is missing")

validation_to_holdout = holdout_pairs.filter(
    (pl.col("benchmark_hash") == selected_validation.hash)
    & (pl.col("benchmark_kind") == "val_rank1_self")
)
holdout_to_benchmark = holdout_pairs.filter(
    pl.col("benchmark_kind") == "equal_weight_holdout_side_artifact"
)
validation_to_benchmark = validation_pairs.filter(
    pl.col("benchmark_kind") == "equal_weight_side_artifact"
)
paired_required = {
    "sharpe_diff",
    "sharpe_diff_ci95_lo",
    "sharpe_diff_ci95_hi",
    "ret_diff",
    "ret_diff_ci95_lo",
    "ret_diff_ci95_hi",
    "prob_challenger_wins",
    "p_value",
}
paired_rows = []
for comparison, frame in (
    ("holdout minus validation", validation_to_holdout),
    ("holdout minus equal weight", holdout_to_benchmark),
    ("validation minus equal weight", validation_to_benchmark),
):
    if frame.height != 1 or not paired_required <= set(frame.columns):
        raise ValueError(f"missing required paired comparison: {comparison}")
    values = frame.row(0, named=True)
    if any(values[name] is None or not np.isfinite(values[name]) for name in paired_required):
        raise ValueError(f"non-finite paired comparison: {comparison}")
    paired_rows.append(
        {
            "comparison": comparison,
            **{name: values[name] for name in paired_required},
        }
    )
paired_evidence = pl.DataFrame(paired_rows)
paired_evidence

# %% [markdown]
# ## Return and drawdown paths
#
# Each window compounds its own daily returns from initial wealth. The plot does not concatenate the
# disjoint validation and holdout calendars.

# %% tags=["results"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for period, result in (("validation", selected_validation), ("holdout", holdout_backtest)):
    paths = [path for path in result.artifacts() if path.name == "daily_returns.parquet"]
    if len(paths) != 1:
        raise ValueError(f"{result.hash} must have one daily return artifact")
    returns = pl.read_parquet(paths[0]).select("timestamp", "daily_return").sort("timestamp")
    if returns.get_column("timestamp").n_unique() != returns.height:
        raise ValueError(f"{result.hash} repeats a return timestamp")
    values = returns.get_column("daily_return").to_numpy()
    wealth = np.cumprod(1.0 + values)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    dates = returns.get_column("timestamp").to_list()
    axes[0].plot(dates, wealth, label=period)
    axes[1].plot(dates, drawdown, label=period)

axes[0].set_title("Wealth within each evaluation window")
axes[0].set_ylabel("Growth of 1.0")
axes[1].set_title("Drawdown within each evaluation window")
axes[1].set_ylabel("Drawdown")
for axis in axes:
    axis.set_xlabel("Date")
    axis.legend(frameon=False)
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Result interpretation
#
# The sentences below are computed from the registered metrics. An interval wholly above or below zero
# is reported as such; an interval spanning zero is not converted into a positive or negative claim.

# %% tags=["results"]


def _interval_read(name: str, point: float, lower: float, upper: float) -> str:
    if lower > 0:
        status = "its interval is above zero"
    elif upper < 0:
        status = "its interval is below zero"
    else:
        status = "its interval includes zero"
    return f"{name}: {point:.3f} [{lower:.3f}, {upper:.3f}]; {status}."


validation_row = selected_performance.filter(pl.col("period") == "validation").row(0, named=True)
holdout_row = selected_performance.filter(pl.col("period") == "holdout").row(0, named=True)
decay_row = paired_evidence.filter(pl.col("comparison") == "holdout minus validation").row(
    0, named=True
)

interpretation = pl.DataFrame(
    {
        "evidence": ["validation", "holdout", "holdout change"],
        "reading": [
            _interval_read(
                "Validation Sharpe",
                validation_row["sharpe"],
                validation_row["sharpe_ci95_lo"],
                validation_row["sharpe_ci95_hi"],
            ),
            _interval_read(
                "Holdout Sharpe",
                holdout_row["sharpe"],
                holdout_row["sharpe_ci95_lo"],
                holdout_row["sharpe_ci95_hi"],
            ),
            _interval_read(
                "Holdout minus validation Sharpe",
                decay_row["sharpe_diff"],
                decay_row["sharpe_diff_ci95_lo"],
                decay_row["sharpe_diff_ci95_hi"],
            ),
        ],
    }
)
interpretation

# %% [markdown]
# ## Key takeaways
#
# - The immutable candidate set determines the exact selected configuration, and the holdout
#   lineage follows from it rather than being recorded alongside it.
# - Controlled cost and risk evidence changes one strategy field at a time.
# - The holdout cannot cause reselection, because the selection is upstream of it and frozen.
# - All result-specific interpretation is produced from the registered artifacts this notebook
#   resolved, so a re-run reports the same numbers or fails.
