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
# immutable compatible set records exact prediction membership and the protocol fields its members
# share. The full sets supply the descriptive table and pass unchanged to strategy evaluation.
# Separate immutable sets identify bounded subsets for diagnostics that load raw predictions.
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
# **Prerequisites**: the modelling notebooks - [`06_linear`](06_linear.ipynb) through
# [`13_latent_factors`](13_latent_factors.ipynb) - must have published their canonical validation
# results and the two immutable compatible sets used below.
# [`14_causal_dml`](14_causal_dml.ipynb) provides any exact causal result hashes included in the
# separate causal section.

# %%
"""Read-only interpretation of the result sets the modelling notebooks published."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic_series

from case_studies.research import (
    CandidateSet,
    CausalResult,
    OfficialPopulation,
    PredictionResult,
    Study,
    open_study,
)
from case_studies.utils.backtest_runner import normalize_prediction_columns
from case_studies.utils.insight_chapter import conformal_coverage_for_selected_prediction
from case_studies.utils.registry import canonical_json, load_prediction_metrics
from utils.style import COLORS

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_FAMILIES = []
PREVIEW_CONFIG_NAMES = []
PREVIEW_MAX_PREDICTIONS = 0
PREVIEW_MAX_DIAGNOSTICS = 0
PREDICTION_SET_NAMES = [
    "us-equities-fwd-ret-1d-linear-v1",
    "us-equities-fwd-ret-5d-linear-v1",
    "us-equities-fwd-ret-21d-linear-v1",
    "us-equities-fwd-ret-1d-gbm-v1",
    "us-equities-fwd-ret-5d-gbm-v1",
    "us-equities-fwd-ret-21d-gbm-v1",
    "us-equities-fwd-ret-1d-tabular-dl-v1",
    "us-equities-fwd-ret-1d-nlinear-v1",
    "us-equities-fwd-ret-1d-lstm-v1",
    "us-equities-fwd-ret-1d-tsmixer-v1",
    "us-equities-fwd-ret-5d-weekly-v1",
    "us-equities-fwd-ret-1d-pca-v1",
    "us-equities-fwd-ret-1d-ipca-v1",
    "us-equities-fwd-ret-5d-pca-v1",
    "us-equities-fwd-ret-5d-ipca-v1",
    "us-equities-fwd-ret-21d-pca-v1",
    "us-equities-fwd-ret-21d-ipca-v1",
]

# %% tags=["parameters"]
OFFICIAL_POPULATION_NAMES = [
    "us-equities-linear-checkpoints-v1",
    "us-equities-gbm-checkpoints-v1",
    "us-equities-tabular-dl-checkpoints-v1",
    "us-equities-nlinear-checkpoints-v1",
    "us-equities-lstm-checkpoints-v1",
    "us-equities-tsmixer-checkpoints-v1",
    "us-equities-weekly-checkpoints-v1",
    "us-equities-pca-checkpoints-v1",
    "us-equities-ipca-checkpoints-v1",
]

# %% tags=["parameters"]
DIAGNOSTIC_SET_NAMES = [
    "us-equities-fwd-ret-1d-linear-diagnostics-v1",
    "us-equities-fwd-ret-5d-linear-diagnostics-v1",
    "us-equities-fwd-ret-21d-linear-diagnostics-v1",
    "us-equities-fwd-ret-1d-gbm-diagnostics-v1",
    "us-equities-fwd-ret-5d-gbm-diagnostics-v1",
    "us-equities-fwd-ret-21d-gbm-diagnostics-v1",
    "us-equities-fwd-ret-1d-tabular-dl-diagnostics-v1",
    "us-equities-fwd-ret-1d-nlinear-v1",
    "us-equities-fwd-ret-1d-lstm-v1",
    "us-equities-fwd-ret-1d-tsmixer-v1",
    "us-equities-fwd-ret-5d-weekly-diagnostics-v1",
    "us-equities-fwd-ret-1d-pca-v1",
    "us-equities-fwd-ret-1d-ipca-v1",
    "us-equities-fwd-ret-5d-pca-v1",
    "us-equities-fwd-ret-5d-ipca-v1",
    "us-equities-fwd-ret-21d-pca-v1",
    "us-equities-fwd-ret-21d-ipca-v1",
]
CAUSAL_LABELS = ["fwd_ret_1d"]

# %% [markdown]
# ## Open the declared result sets
#
# A prediction result is eligible here when its persisted coverage matches the expected
# `(symbol, timestamp, fold)` grid and its split and execution tier match the declared analysis.
# Canonical analysis verifies named compatible sets and their official checkpoint populations.
# A reduced preview selects a bounded catalog population by visible fields without publishing it.
# Diagnostic members remain a subset of the strategy handoff in either tier.

# %% tags=["results"]
preview_filters = bool(PREVIEW_LABELS or PREVIEW_FAMILIES or PREVIEW_CONFIG_NAMES)
if EXECUTION_TIER == "canonical":
    if preview_filters or PREVIEW_MAX_PREDICTIONS or PREVIEW_MAX_DIAGNOSTICS:
        raise ValueError("Canonical analysis cannot declare preview reductions")
    for names, field in (
        (PREDICTION_SET_NAMES, "PREDICTION_SET_NAMES"),
        (DIAGNOSTIC_SET_NAMES, "DIAGNOSTIC_SET_NAMES"),
        (OFFICIAL_POPULATION_NAMES, "OFFICIAL_POPULATION_NAMES"),
    ):
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{field} must contain unique names")
    study = Study.open(CASE_STUDY_ID)
elif EXECUTION_TIER == "preview":
    if not preview_filters or PREVIEW_MAX_PREDICTIONS < 1 or PREVIEW_MAX_DIAGNOSTICS < 1:
        raise ValueError(
            "Preview analysis requires a catalog filter and explicit prediction limits"
        )
    if PREVIEW_MAX_DIAGNOSTICS > PREVIEW_MAX_PREDICTIONS:
        raise ValueError("Preview diagnostics cannot exceed the preview prediction population")
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

include_preview = EXECUTION_TIER == "preview"
# Metrics are read from the tier's own storage, not from `study.root`. Under preview in a
# maintainer worktree `open_study` leaves `root` on the release case directory and redirects only
# writes, so `case_dir=study.root` sends every metric lookup to the released registry while the
# catalog rows come from the preview one - and every preview row reports as having no metrics.
# `storage_root` is the accessor that answers "where does this tier's registry live".
metrics_case_dir = study.storage_root(EXECUTION_TIER)
prediction_sets = ()
diagnostic_sets = ()
official_populations = ()

# %% tags=["results"]
if EXECUTION_TIER == "canonical":
    prediction_sets = tuple(CandidateSet.one(study, name=name) for name in PREDICTION_SET_NAMES)
    diagnostic_sets = tuple(CandidateSet.one(study, name=name) for name in DIAGNOSTIC_SET_NAMES)
    official_populations = tuple(
        OfficialPopulation.one(study, name=name) for name in OFFICIAL_POPULATION_NAMES
    )
    for population in official_populations:
        if population.member_kind != "prediction":
            raise ValueError(f"{population.hash} is not a prediction population")
        population.require_complete()

# %% tags=["results"]
if EXECUTION_TIER == "canonical":
    identity_protocol_fields = {"label_artifact", "feature_artifacts", "cv"}
    for declared_set in (*prediction_sets, *diagnostic_sets):
        if declared_set.member_kind != "prediction":
            raise ValueError(f"{declared_set.hash} contains {declared_set.member_kind} results")
        comparable_fields = set(declared_set.comparison_contract.get("comparable_fields", ()))
        variable_identity_fields = identity_protocol_fields & comparable_fields
        if variable_identity_fields:
            raise ValueError(
                f"{declared_set.hash} varies identity fields {sorted(variable_identity_fields)}"
            )
        declared_protocol = declared_set.comparison_contract.get("protocol", {})
        missing_identity_fields = {
            field for field in identity_protocol_fields if not declared_protocol.get(field)
        }
        if missing_identity_fields:
            raise ValueError(
                f"{declared_set.hash} lacks identity fields {sorted(missing_identity_fields)}"
            )
        if (
            declared_protocol.get("split") != "validation"
            or declared_protocol.get("execution_tier") != "canonical"
        ):
            raise ValueError(f"{declared_set.hash} is not a canonical validation set")

# %% tags=["results"]
if EXECUTION_TIER == "canonical":
    prediction_members = tuple(
        member for declared_set in prediction_sets for member in declared_set.members
    )
    diagnostic_members = tuple(
        member for declared_set in diagnostic_sets for member in declared_set.members
    )
    prediction_set_by_member = {
        member: declared_set.name
        for declared_set in prediction_sets
        for member in declared_set.members
    }
    diagnostic_set_by_member = {
        member: declared_set.name
        for declared_set in diagnostic_sets
        for member in declared_set.members
    }
    if len(prediction_members) != len(set(prediction_members)):
        raise ValueError("full prediction sets overlap")
    if len(diagnostic_members) != len(set(diagnostic_members)):
        raise ValueError("diagnostic prediction sets overlap")
    if not set(diagnostic_members) <= set(prediction_members):
        raise ValueError("diagnostic results must be members of the full prediction sets")
    official_members = tuple(
        member for population in official_populations for member in population.members
    )
    if len(official_members) != len(set(official_members)):
        raise ValueError("official checkpoint populations overlap")
    if set(official_members) != set(prediction_members):
        raise ValueError("official checkpoint population differs from the full prediction sets")

# %% tags=["results"]
if EXECUTION_TIER == "canonical":
    for diagnostic_set in diagnostic_sets:
        diagnostic_protocol = canonical_json(diagnostic_set.comparison_contract["protocol"])
        matching_full_sets = [
            full_set
            for full_set in prediction_sets
            if canonical_json(full_set.comparison_contract["protocol"]) == diagnostic_protocol
            and set(diagnostic_set.members) <= set(full_set.members)
        ]
        if len(matching_full_sets) != 1:
            raise ValueError(
                f"{diagnostic_set.hash} resolved {len(matching_full_sets)} matching full sets"
            )

# %% tags=["results"]
if EXECUTION_TIER == "canonical":
    set_rows = [
        {
            "role": role,
            "name": declared_set.name,
            "set_hash": declared_set.hash,
            "members": len(declared_set.members),
        }
        for role, declared_sets in (
            ("strategy handoff", prediction_sets),
            ("bounded diagnostics", diagnostic_sets),
        )
        for declared_set in declared_sets
    ]
else:
    preview_selection = study.predictions.table(include_preview=True).filter(
        (pl.col("execution_tier") == "preview")
        & (pl.col("split") == "validation")
        & pl.col("complete")
    )
    if PREVIEW_LABELS:
        preview_selection = preview_selection.filter(pl.col("label").is_in(PREVIEW_LABELS))
    if PREVIEW_FAMILIES:
        preview_selection = preview_selection.filter(pl.col("family").is_in(PREVIEW_FAMILIES))
    if PREVIEW_CONFIG_NAMES:
        preview_selection = preview_selection.filter(
            pl.col("config_name").is_in(PREVIEW_CONFIG_NAMES)
        )
    preview_selection = preview_selection.sort(
        "label", "family", "config_name", "checkpoint_value", "prediction_hash"
    ).head(PREVIEW_MAX_PREDICTIONS)
    if preview_selection.is_empty():
        raise ValueError("Preview analysis selection is empty")

# %% tags=["results"]
if EXECUTION_TIER == "preview":
    prediction_members = tuple(preview_selection.get_column("prediction_hash"))
    diagnostic_members = prediction_members[:PREVIEW_MAX_DIAGNOSTICS]
    preview_group_by_member = {
        row["prediction_hash"]: f"preview:{row['label']}:{row['cv_identity']}"
        for row in preview_selection.iter_rows(named=True)
    }
    prediction_set_by_member = dict(preview_group_by_member)
    diagnostic_set_by_member = {
        member: preview_group_by_member[member] for member in diagnostic_members
    }
    set_rows = [
        {
            "role": role,
            "name": name,
            "set_hash": None,
            "members": len(members),
        }
        for role, name, members in (
            ("strategy handoff", "preview-selection", prediction_members),
            ("bounded diagnostics", "preview-diagnostics", diagnostic_members),
        )
    ]

set_table = pl.DataFrame(set_rows)
set_table

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

# %% tags=["results"]
for prediction_hash in prediction_members:
    result = study.results.open(prediction_hash, include_preview=include_preview)
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{prediction_hash} is not a prediction result")
    if result.execution_tier != EXECUTION_TIER or not result.complete:
        raise ValueError(f"{prediction_hash} is not a complete {EXECUTION_TIER} result")

    record = result.registry_record()
    if record["split"] != "validation":
        raise ValueError(f"{prediction_hash} is not a validation result")
    if not record["checkpoint_kind"]:
        raise ValueError(f"{prediction_hash} has no checkpoint kind")
    coverage = result.coverage()
    if coverage is None or coverage["status"] != "complete":
        raise ValueError(f"{prediction_hash} has incomplete coverage")

    training = study.results.open(record["training_hash"], include_preview=include_preview)
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
            "compatible_set": prediction_set_by_member[prediction_hash],
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

# %% tags=["results"]
catalog = pl.DataFrame(catalog_rows).sort(
    ["label", "family", "config_name", "checkpoint_kind", "checkpoint_value", "prediction_hash"]
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
        case_dir=metrics_case_dir,
    )
    if metrics.height != 1 or not required_metrics <= set(metrics.columns):
        raise ValueError(f"missing exact daily metrics for {row['prediction_hash']}")
    values = metrics.row(0, named=True)
    if any(values[name] is None or not np.isfinite(values[name]) for name in required_metrics):
        raise ValueError(f"non-finite daily metric for {row['prediction_hash']}")
    metric_rows.append({**row, **{name: values[name] for name in required_metrics}})

performance = pl.DataFrame(metric_rows).sort(
    ["label", "family", "config_name", "checkpoint_kind", "checkpoint_value", "prediction_hash"]
)
performance.select(
    "label",
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
        pl.col("label")
        + " | "
        + pl.col("family")
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
KEYS = ["symbol", "timestamp", "fold_id"]
diagnostic_frames = {}
daily_rows = []

for prediction_hash in diagnostic_members:
    result = prediction_results[prediction_hash]
    frame = normalize_prediction_columns(result.load())
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
    model_id = f"{meta['label']} | {meta['family']}/{meta['config_name']} [{prediction_hash[:8]}]"
    diagnostic_frames[prediction_hash] = frame
    for fold_id in sorted(frame["fold_id"].unique().to_list()):
        fold_frame = frame.filter(pl.col("fold_id") == fold_id)
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
                pl.lit(fold_id).alias("fold_id"),
            )
        )

# %% tags=["results"]
daily_ic = pl.concat(daily_rows, how="vertical_relaxed")
if daily_ic.select(["prediction_hash", "fold_id", "timestamp"]).is_duplicated().any():
    raise ValueError("daily IC keys are not unique")

fold_ic = (
    daily_ic.group_by("model_id", "prediction_hash", "fold_id")
    .agg(
        pl.col("ic").mean().alias("mean_daily_ic"),
        pl.len().alias("n_decision_dates"),
    )
    .sort("model_id", "fold_id")
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
diagnostic_hashes = list(diagnostic_members)


def prediction_display_id(prediction_hash):
    meta = catalog.filter(pl.col("prediction_hash") == prediction_hash).row(0, named=True)
    return f"{meta['label']} | {meta['family']}/{meta['config_name']} [{prediction_hash[:8]}]"


# %% tags=["results"]
def summarize_prediction_pair(left_hash, right_hash):
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
    if not np.allclose(
        paired["y_true_left"].cast(pl.Float64).to_numpy(),
        paired["y_true_right"].cast(pl.Float64).to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(f"realized returns disagree for {left_hash} and {right_hash}")
    daily_correlation = paired.group_by("timestamp", "fold_id").agg(
        pl.corr(
            pl.col("score_left").rank(method="average"),
            pl.col("score_right").rank(method="average"),
        ).alias("correlation")
    )
    return {
        "left": prediction_display_id(left_hash),
        "right": prediction_display_id(right_hash),
        "mean_daily_correlation": daily_correlation["correlation"].drop_nulls().mean(),
        "n_shared_rows": paired.height,
        "n_decision_dates": daily_correlation["correlation"].drop_nulls().len(),
    }


# %% tags=["results"]
for left_index, left_hash in enumerate(diagnostic_hashes):
    for right_hash in diagnostic_hashes[left_index:]:
        if diagnostic_set_by_member[left_hash] != diagnostic_set_by_member[right_hash]:
            continue
        correlation_rows.append(summarize_prediction_pair(left_hash, right_hash))

correlations = pl.DataFrame(correlation_rows).sort("left", "right")
correlations

# %% [markdown]
# ## Coverage of the widths that size positions
#
# The width measured here is the one the `conformal_weighted` allocator sizes positions with:
# calibrated per symbol on every absolute residual known at `t - h`, where `h` is that label's
# horizon in data steps, falling back to a quantile pooled over every symbol where one has too
# few residuals of its own. A decision is covered when its absolute residual falls inside that
# half-width; the embargo is what keeps a residual that resolves after the decision out of the
# calibration behind it, whatever order the fold identifiers are in.
#
# Read it as a diagnostic of residual dispersion, not a guarantee: split conformal's
# finite-sample coverage needs exchangeable residuals, return residuals are not, and nothing in
# the allocation path reads an interval or a coverage level.

# %% tags=["results"]
coverage_frames = []

for prediction_hash in diagnostic_members:
    meta = catalog.filter(pl.col("prediction_hash") == prediction_hash).row(0, named=True)
    training_spec = study.results.open(
        meta["training_hash"], include_preview=include_preview
    ).spec()
    coverage_frames.append(
        conformal_coverage_for_selected_prediction(
            {
                "case_study": CASE_STUDY_ID,
                "family": meta["family"],
                "config_name": meta["config_name"],
                "prediction_hash": prediction_hash,
                "spec_json": json.dumps(training_spec),
            }
        ).with_columns(pl.lit(meta["label"]).alias("label"))
    )

conformal_coverage = pl.concat(coverage_frames, how="vertical_relaxed").sort(
    "label", "family", "config_name", "nominal_level"
)
conformal_coverage.select(
    "label",
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
# score, so causal results are read separately and never enter a prediction set. Each visible label
# request must resolve to exactly one complete canonical causal result.

# %% tags=["results"]
causal_columns = [
    "causal_hash",
    "label",
    "treatment",
    "confounders",
    "dml_effect",
    "dml_se_hac",
    "p_value_hac",
    "naive_effect",
    "refutation_p",
    "n_obs",
]

if len(CAUSAL_LABELS) != len(set(CAUSAL_LABELS)):
    raise ValueError("causal labels must be unique")

if CAUSAL_LABELS:
    causal_rows = []
    for label in CAUSAL_LABELS:
        result = CausalResult.one(study, label=label, execution_tier=EXECUTION_TIER)
        if not result.complete or result.execution_tier != EXECUTION_TIER:
            raise ValueError(
                f"{label} does not resolve to a complete {EXECUTION_TIER} causal result"
            )
        computation = result.spec["computation"]
        estimand = computation["estimand"]
        causal_rows.append(
            {
                "causal_hash": result.hash,
                "label": result.spec["label"],
                "treatment": estimand["treatment"],
                "confounders": estimand["confounders"],
                **{name: result.metrics[name] for name in causal_columns[4:]},
            }
        )
    causal_results = pl.DataFrame(causal_rows).sort("label", "causal_hash")
else:
    causal_results = pl.DataFrame(schema={name: pl.String for name in causal_columns})

causal_results

# %% [markdown]
# ## Handoff to strategy evaluation
#
# Strategy evaluation receives every member of the full prediction set. It constructs validation
# backtests for complete configurations and checkpoints; the later strategy-analysis notebook makes
# the single selection decision using validation backtest Sharpe.

# %% tags=["results"]
print(
    "Prediction candidate sets for backtesting: "
    f"{set_table.filter(pl.col('role') == 'strategy handoff').height}"
)
print(f"Members handed off: {len(prediction_members)}")
set_table.filter(pl.col("role") == "strategy handoff")

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
# Canonical execution covers every declared prediction result. Preview execution is explicitly
# bounded and cannot publish a candidate set or official population. Shape-based diagnostics use a
# separate subset because loading every large prediction artifact together is unnecessary. All
# evidence in this notebook comes from validation data; the holdout remains reserved for the locked
# strategy evaluation.
