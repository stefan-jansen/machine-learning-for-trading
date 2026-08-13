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
# # Model Analysis for the US Equities Panel
#
# This notebook reads immutable validation results produced by the modelling notebooks. An
# immutable compatible set records the exact prediction hashes and the protocol fields they share.
# The full set supplies the descriptive table and passes unchanged to strategy evaluation. A second
# immutable set identifies a bounded subset for diagnostics that require loading raw predictions.
#
# **Learning objectives**
#
# - Verify that a declared set contains complete canonical validation predictions.
# - Describe predictive performance with daily cross-sectional information coefficients and
#   dependence-aware uncertainty intervals.
# - Compare fold stability and prediction similarity without dropping panel or fold keys.
# - Evaluate interval coverage with chronological split-conformal calibration.
# - Keep causal estimates separate from predictive results and strategy candidates.
#
# **Book reference**: Chapters 11-15 for model interpretation and Chapter 16 for the strategy
# handoff.
#
# **Prerequisites**: the phase-one execution notebooks must have published their canonical
# validation results and the two immutable compatible sets used below. The causal notebook provides
# any exact causal result hashes included in the separate causal section.

# %%
"""Read-only interpretation of explicit phase-one result sets."""

import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series

from case_studies.research import CandidateSet, PredictionResult, Study
from case_studies.utils.insight_chapter import conformal_coverage_for_selected_prediction
from case_studies.utils.registry import load_prediction_metrics
from utils.style import COLORS

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PREDICTION_SET_HASH = ""
DIAGNOSTIC_SET_HASH = ""
CAUSAL_RESULT_HASHES = []

# %% [markdown]
# ## Open the declared result sets
#
# A prediction result is eligible here when its persisted coverage matches the expected
# `(symbol, timestamp, fold)` grid, its execution tier is canonical, and its split is validation.
# The compatible-set contract also records the shared label, features, and cross-validation
# protocol. The diagnostic set is a subset of the full set, so its displays cannot change the
# collection sent to backtesting.

# %% tags=["results"]
if not PREDICTION_SET_HASH or not DIAGNOSTIC_SET_HASH:
    raise ValueError("PREDICTION_SET_HASH and DIAGNOSTIC_SET_HASH are required")

study = Study.open(CASE_STUDY_ID)
prediction_set = CandidateSet.open(study, PREDICTION_SET_HASH)
diagnostic_set = CandidateSet.open(study, DIAGNOSTIC_SET_HASH)

for declared_set in (prediction_set, diagnostic_set):
    if declared_set.member_kind != "prediction":
        raise ValueError(f"{declared_set.hash} contains {declared_set.member_kind} results")

if not set(diagnostic_set.members) <= set(prediction_set.members):
    raise ValueError("diagnostic results must be members of the full prediction set")

protocol = prediction_set.comparison_contract["protocol"]
diagnostic_protocol = diagnostic_set.comparison_contract["protocol"]
if protocol != diagnostic_protocol:
    raise ValueError("full and diagnostic sets must share one comparison protocol")
if protocol.get("split") != "validation" or protocol.get("execution_tier") != "canonical":
    raise ValueError("model analysis requires canonical validation results")

print(f"Prediction set: {prediction_set.hash} ({len(prediction_set.members)} results)")
print(f"Diagnostic set: {diagnostic_set.hash} ({len(diagnostic_set.members)} results)")

# %% [markdown]
# ## Validate identities and coverage
#
# A checkpoint is part of a prediction identity. The catalog below therefore keeps the training
# hash, checkpoint kind, checkpoint value, and prediction hash together. Coverage is checked before
# any metric is displayed, and distinct results must have distinct configuration-checkpoint
# identities.

# %% tags=["results"]
catalog_rows = []
prediction_results = {}
prediction_identities = set()

for prediction_hash in prediction_set.members:
    result = study.results.open(prediction_hash)
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{prediction_hash} is not a prediction result")
    if result.execution_tier != "canonical" or not result.complete:
        raise ValueError(f"{prediction_hash} is not a complete canonical result")

    record = result.registry_record()
    if record["split"] != "validation":
        raise ValueError(f"{prediction_hash} is not a validation result")
    if not record["checkpoint_kind"]:
        raise ValueError(f"{prediction_hash} has no checkpoint kind")
    coverage = result.coverage()
    if coverage is None or coverage["status"] != "complete":
        raise ValueError(f"{prediction_hash} has incomplete coverage")

    training = study.results.open(record["training_hash"])
    specification = training.spec()
    identity = (
        record["training_hash"],
        record["checkpoint_kind"],
        record["checkpoint_value"],
    )
    if identity in prediction_identities:
        raise ValueError("prediction set repeats a configuration-checkpoint identity")
    prediction_identities.add(identity)
    catalog_rows.append(
        {
            "prediction_hash": prediction_hash,
            "training_hash": record["training_hash"],
            "family": specification["family"],
            "config_name": specification["config_name"],
            "label": specification["label"],
            "checkpoint_kind": record["checkpoint_kind"],
            "checkpoint_value": record["checkpoint_value"],
            "n_predictions": coverage["n_actual"],
            "n_folds": coverage["n_folds_actual"],
        }
    )
    prediction_results[prediction_hash] = result

catalog = pl.DataFrame(catalog_rows)
if catalog["label"].n_unique() != 1:
    raise ValueError("prediction results must share one label")

catalog = catalog.sort(
    ["family", "config_name", "checkpoint_kind", "checkpoint_value", "prediction_hash"]
)
catalog

# %% [markdown]
# ## Daily predictive performance
#
# The information coefficient (IC) is the Spearman rank correlation between scores and realized
# returns within one decision date. Registry metrics pool those daily correlations across folds,
# giving every decision date equal weight. Heteroskedasticity-and-autocorrelation-consistent (HAC)
# intervals account for dependence induced by overlapping forward returns.

# %% tags=["results"]
metric_rows = []
required_metrics = {
    "ic_mean_daily",
    "ic_ci_lo",
    "ic_ci_hi",
    "ic_t_hac",
    "ic_p_hac",
    "ic_n_days",
    "ic_pct_positive",
}

for row in catalog.iter_rows(named=True):
    metrics = load_prediction_metrics(
        CASE_STUDY_ID,
        prediction_hash=row["prediction_hash"],
        case_dir=study.root,
    )
    if metrics.height != 1 or not required_metrics <= set(metrics.columns):
        raise ValueError(f"missing exact daily metrics for {row['prediction_hash']}")
    values = metrics.row(0, named=True)
    if any(values[name] is None or not np.isfinite(values[name]) for name in required_metrics):
        raise ValueError(f"non-finite daily metric for {row['prediction_hash']}")
    metric_rows.append({**row, **{name: values[name] for name in required_metrics}})

performance = pl.DataFrame(metric_rows).sort(
    ["family", "config_name", "checkpoint_kind", "checkpoint_value", "prediction_hash"]
)
performance.select(
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "ic_mean_daily",
    "ic_ci_lo",
    "ic_ci_hi",
    "ic_n_days",
    "ic_pct_positive",
)

# %% [markdown]
# The plot retains the catalog order rather than sorting by IC. Each horizontal interval shows the
# sampling uncertainty around one configuration and checkpoint; it is descriptive evidence, not a
# model-selection rule.

# %% tags=["results"]
plot_performance = performance.with_columns(
    (
        pl.col("family")
        + "/"
        + pl.col("config_name")
        + " @ "
        + pl.col("checkpoint_kind")
        + "="
        + pl.col("checkpoint_value").cast(pl.String).fill_null("final")
        + " ["
        + pl.col("prediction_hash").str.slice(0, 8)
        + "]"
    ).alias("model_id")
)

fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * plot_performance.height)))
y_position = np.arange(plot_performance.height)
mean_ic = plot_performance["ic_mean_daily"].to_numpy()
lower = plot_performance["ic_ci_lo"].to_numpy()
upper = plot_performance["ic_ci_hi"].to_numpy()
ax.errorbar(
    mean_ic,
    y_position,
    xerr=np.vstack([mean_ic - lower, upper - mean_ic]),
    fmt="o",
    color=COLORS["blue"],
    ecolor=COLORS["neutral"],
    capsize=2,
)
ax.axvline(0, color=COLORS["negative"], linewidth=0.8, linestyle="--")
ax.set_yticks(y_position, plot_performance["model_id"].to_list())
ax.set_xlabel("Daily cross-sectional IC")
ax.set_title("Daily Cross-Sectional IC with HAC Intervals")
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Fold stability for the diagnostic set
#
# Fold summaries show whether a configuration behaves similarly across validation windows. Raw
# predictions are loaded only for the explicit diagnostic members. Every artifact must contain
# finite values and unique canonical keys before its daily IC is computed.

# %% tags=["results"]
KEYS = ["symbol", "timestamp", "fold"]
diagnostic_frames = {}
daily_rows = []

for prediction_hash in diagnostic_set.members:
    result = prediction_results[prediction_hash]
    frame = result.load()
    required_columns = {*KEYS, "y_true", "y_score"}
    if not required_columns <= set(frame.columns):
        raise ValueError(f"{prediction_hash} lacks canonical prediction columns")
    if frame.select(KEYS).is_duplicated().any():
        raise ValueError(f"{prediction_hash} repeats canonical prediction keys")
    if frame.select(pl.any_horizontal(pl.col("y_true", "y_score").is_null())).to_series().any():
        raise ValueError(f"{prediction_hash} contains null predictions")
    if frame.select(pl.any_horizontal(~pl.col("y_true", "y_score").is_finite())).to_series().any():
        raise ValueError(f"{prediction_hash} contains non-finite predictions")

    meta = catalog.filter(pl.col("prediction_hash") == prediction_hash).row(0, named=True)
    model_id = f"{meta['family']}/{meta['config_name']} [{prediction_hash[:8]}]"
    diagnostic_frames[prediction_hash] = frame
    for fold in sorted(frame["fold"].unique().to_list()):
        fold_frame = frame.filter(pl.col("fold") == fold)
        daily = cross_sectional_ic_series(
            fold_frame,
            fold_frame,
            pred_col="y_score",
            ret_col="y_true",
            date_col="timestamp",
            entity_col="symbol",
            method="spearman",
            min_obs=5,
        ).drop_nulls("ic")
        daily_rows.append(
            daily.with_columns(
                pl.lit(model_id).alias("model_id"),
                pl.lit(prediction_hash).alias("prediction_hash"),
                pl.lit(fold).alias("fold"),
            )
        )

daily_ic = pl.concat(daily_rows, how="vertical_relaxed")
if daily_ic.select(["prediction_hash", "fold", "timestamp"]).is_duplicated().any():
    raise ValueError("daily IC keys are not unique")

fold_ic = (
    daily_ic.group_by("model_id", "prediction_hash", "fold")
    .agg(
        pl.col("ic").mean().alias("mean_daily_ic"),
        pl.len().alias("n_decision_dates"),
    )
    .sort("model_id", "fold")
)
fold_ic

# %% [markdown]
# ## Prediction similarity
#
# Pairwise similarity uses only observations shared by the two exact results. Joins retain
# `(symbol, timestamp, fold)`, validate one-to-one cardinality, and confirm that both artifacts
# carry the same realized return for every shared observation. Spearman correlations are computed
# within each decision date and then averaged so dates with larger cross-sections receive no extra
# weight.

# %% tags=["results"]
correlation_rows = []
diagnostic_hashes = list(diagnostic_set.members)

for left_index, left_hash in enumerate(diagnostic_hashes):
    left_meta = catalog.filter(pl.col("prediction_hash") == left_hash).row(0, named=True)
    left_id = f"{left_meta['family']}/{left_meta['config_name']} [{left_hash[:8]}]"
    for right_hash in diagnostic_hashes[left_index:]:
        right_meta = catalog.filter(pl.col("prediction_hash") == right_hash).row(0, named=True)
        right_id = f"{right_meta['family']}/{right_meta['config_name']} [{right_hash[:8]}]"
        left = diagnostic_frames[left_hash].select(
            *KEYS,
            pl.col("y_true").alias("y_true_left"),
            pl.col("y_score").alias("score_left"),
        )
        right = diagnostic_frames[right_hash].select(
            *KEYS,
            pl.col("y_true").alias("y_true_right"),
            pl.col("y_score").alias("score_right"),
        )
        paired = left.join(right, on=KEYS, how="inner", validate="1:1")
        if paired.is_empty() or paired.height > min(left.height, right.height):
            raise ValueError(f"invalid paired coverage for {left_hash} and {right_hash}")
        if not paired.select(pl.col("y_true_left").eq(pl.col("y_true_right")).all()).item():
            raise ValueError(f"realized returns disagree for {left_hash} and {right_hash}")
        daily_correlation = paired.group_by("timestamp", "fold").agg(
            pl.corr(
                pl.col("score_left").rank(method="average"),
                pl.col("score_right").rank(method="average"),
            ).alias("correlation")
        )
        correlation_rows.append(
            {
                "left": left_id,
                "right": right_id,
                "mean_daily_correlation": daily_correlation["correlation"].drop_nulls().mean(),
                "n_shared_rows": paired.height,
                "n_decision_dates": daily_correlation["correlation"].drop_nulls().len(),
            }
        )

correlations = pl.DataFrame(correlation_rows).sort("left", "right")
correlations

# %% [markdown]
# ## Chronological conformal coverage
#
# Split-conformal intervals use the oldest validation observations for calibration and evaluate
# coverage only on later observations. The finite-sample order statistic determines each interval
# width. This preserves chronology even when fold identifiers are not chronological.

# %% tags=["results"]
coverage_frames = []

for prediction_hash in diagnostic_set.members:
    meta = catalog.filter(pl.col("prediction_hash") == prediction_hash).row(0, named=True)
    training_spec = study.results.open(meta["training_hash"]).spec()
    coverage_frames.append(
        conformal_coverage_for_selected_prediction(
            {
                "case_study": CASE_STUDY_ID,
                "family": meta["family"],
                "config_name": meta["config_name"],
                "prediction_hash": prediction_hash,
                "spec_json": json.dumps(training_spec),
            }
        )
    )

conformal_coverage = pl.concat(coverage_frames, how="vertical_relaxed").sort(
    "family", "config_name", "nominal_level"
)
conformal_coverage.select(
    "family",
    "config_name",
    "prediction_hash",
    "nominal_level",
    "empirical_coverage",
    "mean_interval_width_frac_std",
)

# %% [markdown]
# ## Causal evidence
#
# Double machine learning estimates a treatment effect after using nuisance models to remove the
# variation explained by declared confounders. Its estimand and uncertainty differ from a predictive
# score, so causal hashes are read separately and never enter either prediction set. Each requested
# hash must resolve to exactly one registered causal result.

# %% tags=["results"]
causal_columns = [
    "causal_hash",
    "label",
    "treatment",
    "confounders_json",
    "dml_effect",
    "dml_se_hac",
    "p_value_hac",
    "naive_effect",
    "refutation_p",
    "n_obs",
]

if len(CAUSAL_RESULT_HASHES) != len(set(CAUSAL_RESULT_HASHES)):
    raise ValueError("causal result hashes must be unique")

if CAUSAL_RESULT_HASHES:
    placeholders = ",".join("?" for _ in CAUSAL_RESULT_HASHES)
    with sqlite3.connect(study.root / "run_log" / "registry.db") as connection:
        rows = connection.execute(
            f"SELECT {','.join(causal_columns)} FROM causal_runs "
            f"WHERE causal_hash IN ({placeholders}) ORDER BY causal_hash",
            tuple(CAUSAL_RESULT_HASHES),
        ).fetchall()
    if len(rows) != len(set(CAUSAL_RESULT_HASHES)):
        raise ValueError("every requested causal hash must resolve exactly once")
    causal_results = pl.DataFrame(rows, schema=causal_columns, orient="row").with_columns(
        pl.col("confounders_json").str.json_decode().alias("confounders")
    )
else:
    causal_results = pl.DataFrame(schema={name: pl.String for name in causal_columns})

causal_results.drop("confounders_json")

# %% [markdown]
# ## Handoff to strategy evaluation
#
# Strategy evaluation receives every member of the full prediction set. It constructs validation
# backtests for complete configurations and checkpoints; the later strategy-analysis notebook makes
# the single selection decision using validation backtest Sharpe.

# %% tags=["results"]
print(f"Prediction candidate set for backtesting: {prediction_set.hash}")
print(f"Members handed off: {len(prediction_set.members)}")
for prediction_hash in prediction_set.members:
    print(f"  {prediction_hash}")

# %% [markdown]
# ## Key takeaways and limitations
#
# - Immutable compatible sets make the comparison population explicit and keep unrelated registry
#   rows out of the analysis.
# - Daily-pooled IC describes cross-sectional ranking quality while giving each decision date equal
#   weight; HAC intervals reflect dependence from overlapping return horizons.
# - Exact three-column joins preserve symbol, decision time, and fold identity when predictions are
#   compared.
# - Chronological conformal calibration uses earlier observations to assess coverage on later ones.
# - Causal estimates answer a treatment-effect question and remain separate from prediction and
#   strategy contracts.
#
# The full table covers every declared prediction result. Shape-based diagnostics use the separately
# declared subset because loading every large prediction artifact together is unnecessary. All
# evidence in this notebook comes from validation data; the holdout remains reserved for the locked
# strategy evaluation.
