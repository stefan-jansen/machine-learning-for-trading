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
# # Strategy Analysis for the US Equities Panel
#
# This notebook reads the immutable validation backtest set, the research lock created from that
# set, and the lock's one holdout evaluation. The lock records the selected validation backtest,
# prediction checkpoint, training lineage, strategy specification, source identity, and runtime
# identity. Reopening those references makes the strategy assessment independent of registry row
# order and of any experiments added later.
#
# **Learning objectives**
#
# - Reproduce deterministic validation selection from one immutable backtest set.
# - Verify that the selected backtest and holdout evaluation match the complete locked lineage.
# - Interpret performance with uncertainty intervals and exact paired comparisons.
# - Compare return and drawdown paths without mixing validation and holdout observations.
# - Distinguish predictive validation evidence from the single holdout assessment.
#
# **Book reference**: Chapters 16-20 for signal evaluation, allocation, trading costs, risk, and
# strategy assessment.
#
# **Prerequisites**: the strategy execution notebooks must have published the canonical validation
# set, created its research lock, and recorded the corresponding holdout evaluation.

# %%
"""Read-only assessment of one locked validation and holdout lineage."""

import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from case_studies.research import BacktestResult, CandidateSet, LifecycleState, Study
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.artifact_digest import value_digest
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.registry import load_backtest_metrics, load_paired_metrics
from case_studies.utils.registry.specs import project_training_identity
from utils.style import COLORS

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
VALIDATION_BACKTEST_SET_HASH = ""
RESEARCH_LOCK_HASH = ""

# %% [markdown]
# ## Reopen the validation set and research lock
#
# Candidate-set members are complete canonical validation backtests with an explicit comparison
# protocol. Selection uses validation backtest Sharpe, with the backtest hash as the deterministic
# tie-breaker. The selected hash must equal the hash recorded in the research lock.

# %% tags=["results"]
if not VALIDATION_BACKTEST_SET_HASH or not RESEARCH_LOCK_HASH:
    raise ValueError("VALIDATION_BACKTEST_SET_HASH and RESEARCH_LOCK_HASH are required")

study = Study.open(CASE_STUDY_ID)
validation_set = CandidateSet.open(study, VALIDATION_BACKTEST_SET_HASH)
research_lock = study.lifecycle.open(RESEARCH_LOCK_HASH)

if validation_set.member_kind != "backtest":
    raise ValueError("strategy selection requires a backtest candidate set")
if research_lock.state != LifecycleState.HOLDOUT_EVALUATED.value:
    raise ValueError("strategy analysis requires one completed holdout evaluation")
if research_lock.record["candidate_set_hash"] != validation_set.hash:
    raise ValueError("research lock names a different validation candidate set")

# The official analysis restricts the general CandidateSet abstraction to one canonical comparison
# protocol. Each backtest must use the canonical validation price window plus its declared warmup.
IDENTITY_PROTOCOL_FIELDS = {"label_artifact", "feature_artifacts", "cv"}
comparable_fields = set(validation_set.comparison_contract.get("comparable_fields", ()))
variable_identity_fields = IDENTITY_PROTOCOL_FIELDS & comparable_fields
if variable_identity_fields:
    raise ValueError(f"validation set varies identity fields {sorted(variable_identity_fields)}")
validation_protocol = validation_set.comparison_contract.get("protocol", {})
missing_identity_fields = {
    field for field in IDENTITY_PROTOCOL_FIELDS if not validation_protocol.get(field)
}
if missing_identity_fields:
    raise ValueError(f"validation set lacks identity fields {sorted(missing_identity_fields)}")
if (
    validation_protocol.get("split") != "validation"
    or validation_protocol.get("execution_tier") != "canonical"
):
    raise ValueError("strategy analysis requires canonical validation results")

canonical_price_digests = {}
for candidate_hash in validation_set.members:
    candidate = study.results.open(candidate_hash)
    if not isinstance(candidate, BacktestResult) or not candidate.complete:
        raise ValueError(f"{candidate_hash} is not a complete backtest result")
    candidate_protocol = candidate.protocol()
    if any(
        candidate_protocol.get(field) != validation_protocol[field]
        for field in IDENTITY_PROTOCOL_FIELDS
    ):
        raise ValueError(f"{candidate_hash} differs from the validation input protocol")
    training_spec = candidate.lineage()["training_spec"]
    label = training_spec.get("label")
    if not label:
        raise ValueError(f"{candidate_hash} has no label")
    strategy_spec = candidate.spec()
    warmup_periods = strategy_warmup_periods(strategy_spec)
    price_key = (str(label), warmup_periods)
    if price_key not in canonical_price_digests:
        canonical_prices = load_backtest_prices_for(
            CASE_STUDY_ID,
            str(label),
            split="validation",
            warmup_periods=warmup_periods,
        )
        canonical_price_digests[price_key] = value_digest(canonical_prices)
    if strategy_spec.get("input_identity", {}).get("prices") != canonical_price_digests[price_key]:
        raise ValueError(f"{candidate_hash} does not use canonical validation prices")

selected_validation = validation_set.best_validation_sharpe()
if not isinstance(selected_validation, BacktestResult) or not selected_validation.complete:
    raise ValueError("selected validation backtest is incomplete")
if selected_validation.execution_tier != "canonical":
    raise ValueError("selected validation backtest is not canonical")
if selected_validation.hash != research_lock.record["validation_backtest_hash"]:
    raise ValueError("locked backtest does not match deterministic validation selection")

print(f"Validation set: {validation_set.hash} ({len(validation_set.members)} backtests)")
print(f"Selected validation backtest: {selected_validation.hash}")
print(f"Research lock: {research_lock.hash} ({research_lock.state})")

# %% [markdown]
# The selected result is reconstructed through the catalog. Its training artifacts, prediction
# checkpoint, strategy, source, and runtime identities must reproduce the lock record.

# %% tags=["results"]

selected_record = selected_validation.registry_record()
selected_prediction = study.results.open(selected_record["prediction_hash"])
selected_prediction_record = selected_prediction.registry_record()
selected_training = study.results.open(selected_prediction_record["training_hash"])
selected_training_record = selected_training.registry_record()
selected_training_spec = selected_training.spec()

if selected_prediction.hash != research_lock.record["prediction_hash"]:
    raise ValueError("locked prediction differs from the selected backtest lineage")
if selected_training.hash != research_lock.record["training_hash"]:
    raise ValueError("locked training differs from the selected backtest lineage")
if (
    selected_prediction_record["checkpoint_kind"] != research_lock.record["checkpoint_kind"]
    or selected_prediction_record["checkpoint_value"] != research_lock.record["checkpoint_value"]
):
    raise ValueError("locked checkpoint differs from the selected prediction")
if selected_validation.spec() != research_lock.record["strategy_spec"]:
    raise ValueError("locked strategy differs from the selected validation strategy")
for field in ("label_artifact", "feature_artifacts", "cv"):
    if selected_training_spec.get(field) != research_lock.record[field]:
        raise ValueError(f"locked {field} differs from selected training")
if selected_training_record.get("git_commit") != research_lock.record["source_identity"]:
    raise ValueError("locked source identity differs from selected training")
if (
    json.loads(selected_training_record.get("runtime_json") or "{}")
    != research_lock.record["runtime_provenance"]
):
    raise ValueError("locked runtime identity differs from selected training")

# %% [markdown]
# ## Validation selection evidence
#
# Every candidate remains visible in the evidence table. Sorting by Sharpe and then hash reproduces
# the selection rule, while all other metrics remain descriptive. Cost sensitivity is excluded from
# the rule by the candidate set's eligibility contract.

# %% tags=["results"]
required_selection_metrics = {"sharpe", "sharpe_ci95_lo", "sharpe_ci95_hi"}
with sqlite3.connect(study.root / "run_log" / "registry.db") as connection:
    connection.row_factory = sqlite3.Row
    candidate_rows = connection.execute(
        """
        SELECT c.ordinal, b.backtest_hash, b.prediction_hash, b.stage,
               t.training_hash, t.family, t.config_name, t.label, t.execution_tier,
               p.checkpoint_kind, p.checkpoint_value, p.split,
               pc.status AS coverage_status,
               bm.sharpe, bm.sharpe_ci95_lo, bm.sharpe_ci95_hi
        FROM candidate_set_members c
        JOIN backtest_runs b ON b.backtest_hash = c.member_hash
        JOIN backtest_metrics bm ON bm.backtest_hash = b.backtest_hash
        JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
        JOIN prediction_coverage pc ON pc.prediction_hash = p.prediction_hash
        JOIN training_runs t ON t.training_hash = p.training_hash
        WHERE c.set_hash = ?
        ORDER BY c.ordinal
        """,
        (validation_set.hash,),
    ).fetchall()

# %% [markdown]
# The joined rows must reproduce exact set membership. Canonical tier, validation split, complete
# prediction coverage, eligible strategy stage, and finite Sharpe evidence are required for every
# member.

# %% tags=["results"]

if len(candidate_rows) != len(validation_set.members):
    raise ValueError("validation set contains incomplete selection evidence")
selection_evidence = pl.DataFrame([dict(row) for row in candidate_rows])
if set(selection_evidence["backtest_hash"]) != set(validation_set.members):
    raise ValueError("selection evidence differs from candidate-set membership")
if selection_evidence.filter(
    (pl.col("execution_tier") != "canonical")
    | (pl.col("split") != "validation")
    | (pl.col("coverage_status") != "complete")
    | (~pl.col("stage").is_in(["signal", "allocation", "risk_overlay"]))
).height:
    raise ValueError("validation set contains an ineligible selection member")
if not required_selection_metrics <= set(selection_evidence.columns) or any(
    selection_evidence[name].null_count() or not selection_evidence[name].is_finite().all()
    for name in required_selection_metrics
):
    raise ValueError("validation set contains a non-finite selection metric")

selection_evidence = selection_evidence.sort(["sharpe", "backtest_hash"], descending=[True, False])
if selection_evidence["backtest_hash"][0] != selected_validation.hash:
    raise ValueError("displayed selection evidence disagrees with the candidate-set rule")
selection_evidence

# %% [markdown]
# ## Resolve the exact holdout evaluation
#
# The holdout table is keyed by the research lock. It provides one training hash, prediction hash,
# and backtest hash. These references are checked against the locked checkpoint, training request,
# and strategy specification before any holdout metric is displayed.

# %% tags=["results"]
with sqlite3.connect(study.root / "run_log" / "registry.db") as connection:
    holdout_rows = connection.execute(
        "SELECT holdout_training_hash, holdout_prediction_hash, holdout_backtest_hash "
        "FROM holdout_evaluations WHERE lock_hash = ?",
        (research_lock.hash,),
    ).fetchall()

if len(holdout_rows) != 1:
    raise ValueError("research lock must have exactly one holdout evaluation")

holdout_training_hash, holdout_prediction_hash, holdout_backtest_hash = holdout_rows[0]
holdout_training = study.results.open(holdout_training_hash)
holdout_prediction = study.results.open(holdout_prediction_hash)
holdout_backtest = study.results.open(holdout_backtest_hash)

if not isinstance(holdout_backtest, BacktestResult) or not holdout_backtest.complete:
    raise ValueError("holdout backtest is incomplete")
if holdout_backtest.execution_tier != "canonical":
    raise ValueError("holdout backtest is not canonical")
if holdout_prediction.registry_record()["split"] != "holdout":
    raise ValueError("holdout prediction has the wrong split")
if holdout_prediction.registry_record()["training_hash"] != holdout_training.hash:
    raise ValueError("holdout training and prediction lineage disagree")

# %% [markdown]
# The holdout lineage must retain the locked training request apart from its explicit holdout
# interval, the same checkpoint and strategy, and the source and runtime identities used for
# validation. Its backtest also records the canonical holdout price identity.

# %% tags=["results"]

if (
    holdout_prediction.registry_record()["checkpoint_kind"]
    != research_lock.record["checkpoint_kind"]
    or holdout_prediction.registry_record()["checkpoint_value"]
    != research_lock.record["checkpoint_value"]
):
    raise ValueError("holdout checkpoint differs from the lock")
if holdout_training.hash != research_lock.record["holdout_training_hash"]:
    raise ValueError("holdout training identity differs from the lock")
if (
    project_training_identity(holdout_training.spec())
    != research_lock.record["holdout_training_spec"]
):
    raise ValueError("holdout training specification differs from the lock")
if holdout_backtest.registry_record()["prediction_hash"] != holdout_prediction.hash:
    raise ValueError("holdout backtest and prediction lineage disagree")
if holdout_backtest.spec().get("strategy") != research_lock.record["strategy_spec"].get("strategy"):
    raise ValueError("holdout strategy differs from the locked validation strategy")
if not holdout_backtest.spec().get("input_identity", {}).get("prices"):
    raise ValueError("holdout backtest lacks canonical price identity")

holdout_training_record = holdout_training.registry_record()
if holdout_training_record.get("git_commit") != research_lock.record["source_identity"]:
    raise ValueError("holdout source identity differs from the lock")
if (
    json.loads(holdout_training_record.get("runtime_json") or "{}")
    != research_lock.record["runtime_provenance"]
):
    raise ValueError("holdout runtime identity differs from the lock")

locked_label = research_lock.record["label"]
if holdout_training.spec()["label"] != locked_label:
    raise ValueError("holdout label differs from the locked label")

print(f"Locked label: {locked_label}")
print(f"Holdout training: {holdout_training.hash}")
print(f"Holdout prediction: {holdout_prediction.hash}")
print(f"Holdout backtest: {holdout_backtest.hash}")

# %% [markdown]
# ## Validation and holdout performance
#
# Point estimates and bootstrap intervals are read by exact backtest hash. The two windows are
# displayed together for assessment, while their statistical difference comes from the registered
# independent-window comparison in the next section.

# %% tags=["results"]
required_performance_metrics = {
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

for period, result in (
    ("validation", selected_validation),
    ("holdout", holdout_backtest),
):
    metrics = load_backtest_metrics(
        CASE_STUDY_ID,
        backtest_hash=result.hash,
        case_dir=study.root,
    )
    if metrics.height != 1 or not required_performance_metrics <= set(metrics.columns):
        raise ValueError(f"missing exact performance metrics for {result.hash}")
    values = metrics.row(0, named=True)
    if any(
        values[name] is None or not np.isfinite(values[name])
        for name in required_performance_metrics
    ):
        raise ValueError(f"non-finite performance metric for {result.hash}")
    performance_rows.append(
        {
            "period": period,
            "backtest_hash": result.hash,
            **{name: values[name] for name in required_performance_metrics},
        }
    )

locked_performance = pl.DataFrame(performance_rows)
locked_performance

# %% [markdown]
# ## Required paired comparisons
#
# Validation and holdout windows are disjoint, so their difference uses independently resampled
# windows registered under `val_rank1_self`. Benchmark evidence uses the equal-weight return artifact
# for the locked label and window. Each comparison must resolve once and carry finite interval bounds.

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
paired_identity_columns = {"benchmark_hash", "benchmark_kind"}
if holdout_pairs.is_empty() or not paired_identity_columns <= set(holdout_pairs.columns):
    raise ValueError("holdout paired evidence is missing")
if validation_pairs.is_empty() or not paired_identity_columns <= set(validation_pairs.columns):
    raise ValueError("validation paired evidence is missing")

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

# %% [markdown]
# Each named comparison must resolve to one finite row. The equal-weight benchmark identifiers also
# carry the label recorded by the research lock.

# %% tags=["results"]

benchmark_prefix = f"side_ew:{CASE_STUDY_ID}:{locked_label}"
if any(
    not frame["benchmark_hash"][0].startswith(benchmark_prefix)
    for frame in (holdout_to_benchmark, validation_to_benchmark)
    if frame.height == 1
):
    raise ValueError("benchmark lineage does not match the locked label")

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
            "challenger_hash": values["challenger_hash"],
            "benchmark_hash": values["benchmark_hash"],
            **{name: values[name] for name in paired_required},
        }
    )

paired_evidence = pl.DataFrame(paired_rows)
paired_evidence

# %% [markdown]
# ## Return and drawdown paths
#
# Path diagnostics retain each window's own dates. Cumulative return compounds daily returns within
# the named window, and drawdown measures the decline from that window's running wealth peak.

# %% tags=["results"]
return_frames = {}

for period, result in (
    ("validation", selected_validation),
    ("holdout", holdout_backtest),
):
    paths = [path for path in result.artifacts() if path.name == "daily_returns.parquet"]
    if len(paths) != 1:
        raise ValueError(f"{result.hash} must have one daily return artifact")
    returns = pl.read_parquet(paths[0])
    required_columns = {"timestamp", "daily_return"}
    if not required_columns <= set(returns.columns):
        raise ValueError(f"{result.hash} return artifact has the wrong schema")
    returns = returns.select(
        pl.col("timestamp").cast(pl.Date),
        pl.col("daily_return").cast(pl.Float64),
    ).sort("timestamp")
    if returns["timestamp"].n_unique() != returns.height:
        raise ValueError(f"{result.hash} repeats a return timestamp")
    if (
        returns.select(pl.col("daily_return").is_null().any()).item()
        or returns.select((~pl.col("daily_return").is_finite()).any()).item()
    ):
        raise ValueError(f"{result.hash} has invalid daily returns")
    return_frames[period] = returns

# %% [markdown]
# The two columns below retain separate time axes for validation and holdout. The top row compounds
# returns; the bottom row shows the decline from each window's running peak.

# %% tags=["results"]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
for column, period in enumerate(("validation", "holdout")):
    returns = return_frames[period]
    values = returns["daily_return"].to_numpy()
    wealth = np.cumprod(1.0 + values)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    drawdown = wealth / running_peak - 1.0
    axes[0, column].plot(returns["timestamp"], wealth - 1.0, color=COLORS["blue"])
    axes[0, column].axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
    axes[0, column].set_title(f"{period.title()} Cumulative Return")
    axes[1, column].fill_between(
        returns["timestamp"],
        drawdown,
        0,
        color=COLORS["negative"],
        alpha=0.35,
    )
    axes[1, column].set_title(f"{period.title()} Drawdown")
axes[0, 0].set_ylabel("Cumulative return")
axes[1, 0].set_ylabel("Drawdown")
fig.suptitle("Locked Strategy Across Validation and Holdout Windows")
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Computed assessment
#
# The statements below are generated from the exact locked metrics. An interval that lies wholly
# above or below zero provides directional evidence at its registered confidence level; an interval
# spanning zero leaves the direction unresolved.

# %% tags=["results"]
for row in paired_evidence.iter_rows(named=True):
    lower = row["sharpe_diff_ci95_lo"]
    upper = row["sharpe_diff_ci95_hi"]
    if lower > 0:
        interval_read = "above zero"
    elif upper < 0:
        interval_read = "below zero"
    else:
        interval_read = "spans zero"
    print(
        f"{row['comparison']}: Sharpe difference {row['sharpe_diff']:+.3f}, "
        f"interval [{lower:+.3f}, {upper:+.3f}] {interval_read}; "
        f"challenger win probability {row['prob_challenger_wins']:.3f}."
    )

# %% [markdown]
# ## Key takeaways and limitations
#
# - The immutable backtest set defines the validation search population. Registry additions cannot
#   change the locked selection.
# - Validation Sharpe and the backtest hash determine the choice; predictive IC, cost sensitivity,
#   and holdout performance are excluded from that rule.
# - The holdout assessment follows the exact locked training, checkpoint, strategy, label, feature,
#   source, runtime, and price identities.
# - Registered paired comparisons distinguish uncertainty in a difference from the uncertainty of
#   two separate point estimates.
# - Validation and holdout path diagnostics retain their own time windows and are interpreted
#   alongside, rather than pooled across, the lifecycle boundary.
#
# This assessment covers one locked strategy lineage and its declared equal-weight benchmark. It
# does not estimate live market impact, borrow availability, or capacity beyond the cost and risk
# assumptions stored in the locked strategy specification.
