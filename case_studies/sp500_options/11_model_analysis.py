# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S&P 500 Options: Model Analysis
#
# This notebook describes the complete official validation prediction populations produced by the
# model execution notebooks. Model identity always includes family, configuration, and checkpoint.
# Every comparison requires identical expected prediction keys and the same cross-validation
# identity. Information coefficient and related diagnostics do not select a model or checkpoint.
#
# Causal DML is reported separately because it estimates a treatment effect rather than a
# cross-sectional prediction configuration.

# %%
"""Analyze complete S&P 500 options model populations."""

import plotly.graph_objects as go
import polars as pl

from case_studies.research import CausalResult, PredictionResult, Result
from case_studies.sp500_options.research_workflow import (
    official_prediction_catalog,
    open_study,
)

MODEL_POPULATIONS = (
    "sp500-options-linear-validation-v1",
    "sp500-options-gbm-validation-v1",
    "sp500-options-tabular-dl-validation-v1",
    "sp500-options-sequence-validation-v1",
)

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %% [markdown]
# ## Complete population and its eligibility groups
#
# Named immutable populations replace hash lists and registry-presence filters. The coverage table
# comes from each prediction result's registered eligibility contract.
#
# The four populations share one CV identity but not one eligibility set. A sequence model scores a
# symbol only after its lookback window is available, so it is eligible on fewer symbols and fewer
# rows than the cross-sectional families reading the same panel. The audit below reports one row per
# eligibility group rather than asserting a single one, and the diagnostics that follow are read
# within a group.

# %% tags=["results"]
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
catalog = official_prediction_catalog(study, MODEL_POPULATIONS)

coverage_rows = []
for prediction_hash in catalog.get_column("prediction_hash"):
    result = Result.open(study, prediction_hash)
    if not isinstance(result, PredictionResult):
        raise TypeError(f"{prediction_hash} is not a prediction result")
    coverage = result.coverage()
    if coverage is None or coverage["status"] != "complete":
        raise RuntimeError(f"prediction {prediction_hash} has incomplete coverage")
    coverage_rows.append(
        {
            "prediction_hash": prediction_hash,
            "expected_key_digest": coverage["expected_key_digest"],
            "n_expected": coverage["n_expected"],
            "n_actual": coverage["n_actual"],
            "n_folds": coverage["n_folds_actual"],
        }
    )
coverage = pl.DataFrame(coverage_rows)

cv_identities = catalog.get_column("cv_identity").drop_nulls()
if cv_identities.is_empty() or cv_identities.n_unique() != 1:
    raise RuntimeError("official model populations do not share one CV identity")

# %% [markdown]
# Eligibility is grouped, not assumed identical. A sequence model needs a lookback window before it
# can score a symbol, so it is eligible on strictly fewer rows than a cross-sectional model reading
# the same panel - which is a property of the model class, not a defect. Requiring one eligibility
# digest across all four populations would fail here for a correct run. What must hold is that every
# prediction sharing an eligibility contract agrees on its dimensions exactly.

# %%
coverage = coverage.join(
    catalog.select("prediction_hash", "family"), on="prediction_hash", how="left"
)
for digest, group in coverage.group_by("expected_key_digest"):
    if group.select("n_expected", "n_actual", "n_folds").n_unique() != 1:
        raise RuntimeError(
            f"predictions sharing eligibility {digest[0]} disagree on coverage dimensions"
        )
    if group.filter(pl.col("n_actual") != pl.col("n_expected")).height:
        raise RuntimeError(f"eligibility {digest[0]} has predictions short of their declaration")

population_audit = (
    coverage.group_by("expected_key_digest")
    .agg(
        pl.col("family").unique().sort().str.join(", ").alias("families"),
        pl.len().alias("predictions"),
        pl.col("n_expected").first().alias("rows_per_prediction"),
        pl.col("n_folds").first().alias("folds"),
    )
    .with_columns(pl.lit(cv_identities[0]).alias("cv_identity"))
    .sort("rows_per_prediction", descending=True)
)
if population_audit.get_column("predictions").sum() != catalog.height:
    raise RuntimeError("the eligibility audit does not account for every declared prediction")
population_audit

# %% [markdown]
# ## Predictive diagnostics
#
# The table retains each checkpoint as a separate row. It supports descriptive comparison only;
# strategy selection occurs after every row has an equal-weight validation backtest.
#
# Each IC is computed on its own prediction's eligible rows, so two rows are directly comparable
# only when the same eligibility group above covers both. A sequence checkpoint and a linear
# checkpoint are scored on different populations, and the difference between their IC values
# therefore mixes model behaviour with the population each was scored on.

# %% tags=["results"]
analysis = (
    catalog.with_columns(
        pl.when(pl.col("checkpoint_value").is_null())
        .then(pl.col("checkpoint_kind").fill_null("final"))
        .otherwise(
            pl.concat_str(
                pl.col("checkpoint_kind"),
                pl.col("checkpoint_value").cast(pl.String),
                separator="=",
            )
        )
        .alias("checkpoint"),
    )
    .with_columns(
        pl.concat_str(
            "family",
            "config_name",
            "checkpoint",
            separator=" / ",
        ).alias("model_identity")
    )
    .select(
        "family",
        "config_name",
        "checkpoint",
        "model_identity",
        "ic_mean",
        "ic_std",
        "ic_t",
        "pct_positive",
        "prediction_hash",
    )
    .sort("family", "config_name", "checkpoint")
)
if analysis.select("ic_mean", "ic_std", "ic_t", "pct_positive").null_count().sum_horizontal().sum():
    raise RuntimeError("official model population has missing regression diagnostics")
analysis

# %% tags=["results"]
family_summary = (
    analysis.group_by("family")
    .agg(
        pl.len().alias("configuration_checkpoints"),
        pl.col("ic_mean").min().alias("ic_min"),
        pl.col("ic_mean").median().alias("ic_median"),
        pl.col("ic_mean").max().alias("ic_max"),
    )
    .sort("family")
)
family_summary

# %% tags=["results"]
fig = go.Figure()
for family in analysis.get_column("family").unique(maintain_order=True):
    rows = analysis.filter(pl.col("family") == family)
    fig.add_trace(
        go.Box(
            name=family,
            y=rows.get_column("ic_mean").to_list(),
            text=rows.get_column("model_identity").to_list(),
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            hovertemplate="%{text}<br>validation IC %{y:+.4f}<extra></extra>",
        )
    )
fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#666666")
fig.update_layout(
    title="Validation IC across declared configurations and checkpoints",
    xaxis_title="Model family",
    yaxis_title="Mean daily rank IC",
    showlegend=False,
)
fig.show()

# %% [markdown]
# ## Causal DML artifact
#
# The causal result is not mixed into the predictive population or its checkpoint summaries.

# %% tags=["results"]
causal = CausalResult.one(study, label="ret_to_expiry", execution_tier=EXECUTION_TIER)
if not causal.complete:
    raise RuntimeError("the causal DML artifact is incomplete")
causal_summary = pl.DataFrame(
    {
        "causal_hash": [causal.hash],
        "treatment": [causal.spec["computation"]["estimand"]["treatment"]],
        "outcome": [causal.spec["computation"]["estimand"]["outcome"]],
        "observations": [causal.metrics["n_obs"]],
        "effect": [causal.metrics["dml_effect"]],
        "hac_standard_error": [causal.metrics["dml_se_hac"]],
        "hac_p_value": [causal.metrics["p_value_hac"]],
        "placebo_p_value": [causal.metrics["refutation_p"]],
    }
)
causal_summary

# %% [markdown]
# Predictive diagnostics and the causal estimate are now available for reader inspection. Neither
# table changes the complete prediction population or selects a strategy.
