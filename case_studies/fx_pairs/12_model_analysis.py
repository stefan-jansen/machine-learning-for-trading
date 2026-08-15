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
# # Model Analysis - FX Pairs
#
# This notebook reads the complete registered validation-prediction population. It compares model
# families, checkpoints, fold stability, prediction agreement, and uncertainty without choosing a
# model for deployment. Every exact model configuration continues to the equal-weight backtest;
# validation backtest Sharpe performs selection later.
#
# **Learning objectives**
#
# - Audit the complete prediction population by label, family, configuration, and checkpoint.
# - Compare predictive diagnostics without using them as selection criteria.
# - Evaluate stability and chronological conformal coverage from canonical prediction artifacts.
# - Keep causal estimates separate from predictive-family evidence.
#
# **Book reference**: Chapters 11-15
#
# **Prerequisites**: `06_linear`, `07_gbm`, `08_tabular_dl`, `09_dl_tcn`,
# `10_dl_nlinear`, and `11_causal_dml`.

# %%
"""Analyze the complete FX validation-prediction population."""

import plotly.express as px
import polars as pl
import yaml
from ml4t.diagnostic.metrics import cross_sectional_ic

import utils.style  # noqa: F401
from case_studies.research import CausalResult, Result, Study
from case_studies.research.results import PredictionResult
from case_studies.utils.conformal import split_conformal_coverage
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY = "fx_pairs"
PRIMARY_LABEL = "fwd_ret_1d"
N_BUCKETS = 5

# %% [markdown]
# ## Load the canonical population
#
# A downstream-selectable row must use the current identity schema, carry exact coverage and fold
# metrics, and have its prediction artifact available. Causal DML is deliberately absent because it
# estimates a treatment effect rather than a cross-sectional score.

# %%
case_dir = get_case_study_dir(CASE_STUDY)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
configured_labels = [setup["labels"]["primary"], *setup["labels"].get("variants", [])]
study = Study.open(CASE_STUDY)
catalog = study.predictions.table().filter(
    (pl.col("identity_status") == "current")
    & (pl.col("execution_tier") == "canonical")
    & (pl.col("split") == "validation")
)

if catalog.is_empty():
    raise RuntimeError("no current canonical validation predictions are registered")
if catalog.filter(~pl.col("complete")).height:
    raise RuntimeError("model analysis requires complete prediction sets")
if catalog.filter(~pl.col("artifact_available")).height:
    raise RuntimeError("model analysis requires every prediction artifact")
if "causal_dml" in set(catalog.get_column("family")):
    raise RuntimeError("causal DML must not enter the predictive catalog")

identity_columns = [
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
]
if catalog.select(identity_columns).n_unique() != catalog.height:
    raise RuntimeError("prediction rows do not retain one complete model identity")

# %% [markdown]
# ## Compare the assembled population against the configured menu
#
# Each notebook checks that it produced what it requested. Nothing checks that the requests covered
# what the case study configures, and a population that is internally consistent but short a
# configured model reads exactly like a complete one. This compares the assembled population to the
# menu itself.
#
# `deep_learning` is the one family split across two notebooks, `09_dl_tcn` and `10_dl_nlinear`, so
# any configured architecture outside that pair has no notebook to produce it. Those members are
# excluded by name and reason rather than passing unnoticed. The exclusions are derived from the
# menu per label, not listed, so a label that never declares a member does not gain a phantom
# exclusion for it.

# %% tags=["results"]
SEQUENCE_NOTEBOOK_ARCHITECTURES = {"tcn", "nlinear"}
EXCLUSION_REASON = "configured but no scoped fx_pairs notebook runs it; coverage decision open"

configured_members = {
    (label, family, config["config_name"])
    for label in configured_labels
    for family in ("linear", "gbm", "tabular_dl", "deep_learning")
    for config in load_configs(CASE_STUDY, label, family=family)
}
excluded_members = {
    (label, family, config_name)
    for label, family, config_name in configured_members
    if family == "deep_learning" and config_name not in SEQUENCE_NOTEBOOK_ARCHITECTURES
}
expected_members = configured_members - excluded_members
present_members = set(catalog.select("label", "family", "config_name").unique().iter_rows())

if excluded_members:
    print(f"Declared exclusions ({EXCLUSION_REASON}):")
    for member in sorted(excluded_members):
        print(f"  {member[0]} / {member[1]} / {member[2]}")

missing_members = sorted(expected_members - present_members)
unexpected_members = sorted(present_members - expected_members)
if missing_members or unexpected_members:
    raise RuntimeError(
        "the assembled population does not match the configured menu; "
        f"missing {missing_members}, unexpected {unexpected_members}"
    )
if set(catalog.get_column("label")) != set(configured_labels):
    raise RuntimeError("the canonical prediction population does not cover every configured label")

# %% tags=["results"]
population_summary = (
    catalog.group_by("label", "family")
    .agg(
        pl.col("config_name").n_unique().alias("configurations"),
        pl.len().alias("checkpoints"),
        pl.col("complete").all().alias("complete"),
        pl.col("ic_mean").is_not_null().sum().alias("diagnosed_checkpoints"),
    )
    .sort("label", "family")
)
population_summary

# %% [markdown]
# ## Compare predictive diagnostics
#
# Rank correlation is computed across pairs at each decision date and then averaged through time.
# The rows shown below are descriptive representatives for plots. They do not filter the official
# prediction population and do not choose checkpoints for backtesting.

# %% tags=["results"]
diagnosed = catalog.filter(pl.col("ic_mean").is_not_null())
representatives = (
    diagnosed.sort(
        ["label", "family", "ic_mean", "config_name", "checkpoint_value", "prediction_hash"],
        descending=[False, False, True, False, False, False],
        nulls_last=True,
    )
    .group_by("label", "family", maintain_order=True)
    .first()
    .sort("label", "family")
)
representatives.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "ic_mean",
    "ic_t",
    "prediction_hash",
)

# %% tags=["results"]
# Families whose checkpoint is `final` carry no numeric checkpoint position, so they have no x
# coordinate on this axis and plotly would drop them without saying so. They are excluded here by
# name rather than silently, and the count is stated in the title.
numbered = diagnosed.filter(pl.col("checkpoint_value").is_not_null())
unnumbered = diagnosed.filter(pl.col("checkpoint_value").is_null())
excluded_families = sorted(set(unnumbered.get_column("family"))) if unnumbered.height else []
figure = px.scatter(
    numbered,
    x="checkpoint_value",
    y="ic_mean",
    color="family",
    facet_row="label",
    hover_data=["config_name", "prediction_hash"],
    title=(
        "Validation rank correlation across numbered model checkpoints"
        + (
            f" (excludes {unnumbered.height} rows from {', '.join(excluded_families)},"
            " whose checkpoint is `final` and has no numeric position)"
            if excluded_families
            else ""
        )
    ),
    labels={"checkpoint_value": "Checkpoint", "ic_mean": "Mean daily rank correlation"},
)
figure.show()

# %% [markdown]
# ## Load canonical prediction columns
#
# Producers use `symbol`, `timestamp`, `fold`, `prediction`, and `actual`. Adding the full catalog
# identity to each frame keeps two checkpoints from collapsing into one plot label or fold count.

# %%
prediction_frames = []
for row in representatives.iter_rows(named=True):
    result = Result.open(study, row["prediction_hash"])
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{row['prediction_hash']} is not a prediction result")
    frame = result.load()
    required = {"symbol", "timestamp", "fold", "prediction", "actual"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction {result.hash} lacks canonical columns: {sorted(missing)}")
    if frame.select("symbol", "timestamp", "fold").n_unique() != frame.height:
        raise ValueError(f"prediction {result.hash} has duplicate eligible keys")
    prediction_frames.append(
        frame.with_columns(
            pl.lit(row["label"]).alias("label"),
            pl.lit(row["family"]).alias("family"),
            pl.lit(row["config_name"]).alias("config_name"),
            pl.lit(row["checkpoint_value"]).alias("checkpoint_value"),
            pl.lit(row["prediction_hash"]).alias("prediction_hash"),
        )
    )

representative_predictions = pl.concat(prediction_frames, how="diagonal_relaxed")

# %% [markdown]
# ## Fold stability
#
# Fold identifiers are labels, not dates. The table orders validation windows by their earliest
# timestamp and retains checkpoint identity in every row.

# %% tags=["results"]
fold_rows = []
for keys, frame in representative_predictions.group_by(
    "label", "family", "config_name", "checkpoint_value", "prediction_hash", "fold"
):
    stats = cross_sectional_ic(
        frame.select("symbol", "timestamp", "prediction"),
        frame.select("symbol", "timestamp", "actual"),
        pred_col="prediction",
        ret_col="actual",
        date_col="timestamp",
        entity_col="symbol",
        min_obs=5,
    )
    fold_rows.append(
        {
            "label": keys[0],
            "family": keys[1],
            "config_name": keys[2],
            "checkpoint_value": keys[3],
            "prediction_hash": keys[4],
            "fold": keys[5],
            "validation_start": frame.get_column("timestamp").min(),
            "ic_mean": stats["ic_mean"],
            "n_decision_times": stats["n_periods"],
        }
    )
fold_metrics = pl.DataFrame(fold_rows).sort("label", "validation_start", "family")
fold_metrics

# %% tags=["results"]
fold_figure = px.box(
    fold_metrics,
    x="family",
    y="ic_mean",
    color="family",
    facet_row="label",
    points="all",
    title="Validation rank correlation varies across chronological folds",
    labels={"family": "Model family", "ic_mean": "Fold mean rank correlation"},
)
fold_figure.show()

# %% [markdown]
# ## Prediction agreement and ranking shape
#
# Correlation between model scores shows whether families rank pairs similarly. Return buckets show
# how realized returns vary from the lowest to highest prediction ranks without turning that pattern
# into a deployment decision.

# %% tags=["results"]
primary = representative_predictions.filter(pl.col("label") == PRIMARY_LABEL)
# The pivot index is the canonical eligibility key alone. `actual` is the same realized return for
# every model at a given key, but the families do not agree on its float representation, so including
# it split the rows into disjoint groups - each score column populated on a different half - and every
# correlation, including the diagonal, came out NaN.
wide = primary.pivot(
    on="prediction_hash",
    index=["symbol", "timestamp", "fold"],
    values="prediction",
)
prediction_columns = [
    column for column in wide.columns if column not in {"symbol", "timestamp", "fold"}
]
if wide.height != primary.height // primary.get_column("prediction_hash").n_unique():
    raise RuntimeError("the agreement pivot did not align every representative on the same keys")
if any(wide.get_column(column).null_count() for column in prediction_columns):
    raise RuntimeError("a representative is missing scores at keys the others cover")
agreement = wide.select(prediction_columns).corr()
agreement

# %% tags=["results"]
bucket_rows = []
for keys, frame in primary.group_by(
    "family", "config_name", "checkpoint_value", "prediction_hash", "timestamp"
):
    if frame.height < N_BUCKETS:
        continue
    ranked = frame.sort("prediction").with_columns(
        ((pl.int_range(pl.len()) * N_BUCKETS) // pl.len())
        .clip(upper_bound=N_BUCKETS - 1)
        .alias("bucket")
    )
    for bucket, values in ranked.group_by("bucket"):
        bucket_rows.append(
            {
                "family": keys[0],
                "config_name": keys[1],
                "checkpoint_value": keys[2],
                "prediction_hash": keys[3],
                "timestamp": keys[4],
                "bucket": bucket[0],
                "actual": values.get_column("actual").mean(),
            }
        )
bucket_summary = (
    pl.DataFrame(bucket_rows)
    .group_by("family", "config_name", "checkpoint_value", "prediction_hash", "bucket")
    .agg(pl.col("actual").mean().alias("mean_realized_return"))
    .sort("family", "bucket")
)
bucket_summary

# %% [markdown]
# ## Chronological conformal coverage
#
# The earliest validation fold supplies both the absolute-residual calibration sample and its return
# scale. Later folds are evaluation data. The threshold is the finite-sample higher order statistic,
# so no interpolated quantile enters the interval.

# %% tags=["results"]
conformal_rows = []
for keys, frame in primary.group_by("family", "config_name", "checkpoint_value", "prediction_hash"):
    for row in split_conformal_coverage(frame):
        conformal_rows.append(
            {
                "family": keys[0],
                "config_name": keys[1],
                "checkpoint_value": keys[2],
                "prediction_hash": keys[3],
                **row,
            }
        )
conformal = pl.DataFrame(conformal_rows).sort("nominal_level", "family")
conformal

# %% [markdown]
# ## Causal evidence remains separate
#
# The causal result answers whether the configured momentum treatment has an estimated effect after
# adjustment for its declared confounders. It does not count as predictive-family coverage and does
# not enter the model or backtest population.

# %% tags=["results"]
causal = CausalResult.one(study, label=PRIMARY_LABEL)
causal_summary = pl.DataFrame([{"causal_hash": causal.hash, **causal.metrics}]).select(
    "causal_hash",
    "n_obs",
    "dml_effect",
    "dml_se_hac",
    "p_value_hac",
    "naive_effect",
    "confounding_bias_pct",
    "refutation_p",
)
causal_summary

# %% [markdown]
# ## Handoff to the equal-weight backtest
#
# Every complete catalog row, including every checkpoint, advances to `13_backtest`. The backtest
# receives these rows directly and records one result per row. Neither this notebook's descriptive
# representatives nor any IC or causal statistic changes that population.

# %%
handoff = catalog.select([*identity_columns, "cv_identity", "complete"]).sort(
    "label", "family", "config_name", "checkpoint_value"
)
handoff

# %% [markdown]
# ## Key takeaways
#
# - Model identity includes label, family, configuration, checkpoint, and validation protocol.
# - Daily rank correlation, fold stability, bucket shape, and conformal coverage are diagnostics.
# - The equal-weight validation backtest receives the complete prediction population.
# - Causal estimates remain a separate form of evidence.
