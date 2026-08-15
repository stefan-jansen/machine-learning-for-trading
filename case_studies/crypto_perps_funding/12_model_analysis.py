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
# # Model diagnostics for perpetual-funding signals
#
# This analysis reads only current, complete canonical prediction catalog rows. A row retains its
# training, prediction, fold, and checkpoint identity. IC, AUC, and loss diagnose models; they do
# not select the trading configuration. Selection occurs after validation backtests and uses Sharpe.
#
# **Learning objectives**
#
# - audit the complete model and checkpoint population through the prediction catalog;
# - compare diagnostic metrics without using them as the strategy selection rule; and
# - interpret causal evidence separately from predictive performance.
#
# **Book reference:** Chapters 11 to 19, model comparison and diagnostics.
#
# **Prerequisites:** complete canonical outputs from the model execution notebooks and the causal
# notebook.

# %%
import os

import plotly.express as px
import polars as pl
from IPython.display import Markdown

from case_studies.crypto_perps_funding.research_workflow import (
    OFFICIAL_POPULATION,
    open_study,
    plan_official_models,
)
from case_studies.research import CausalResult, OfficialPopulation
from utils.style import COLORS, ml4t_palette, show_plotly_with_alt

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE = os.environ.get("ML4T_OUTPUT_DIR", "")

# %% [markdown]
# ## Complete predictive catalog

# %% tags=["results"]
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
catalog = study.predictions.table(include_preview=EXECUTION_TIER == "preview").filter(
    (pl.col("identity_status") == "current")
    & (pl.col("execution_tier") == EXECUTION_TIER)
    & (pl.col("split") == "validation")
    & pl.col("complete")
)
if catalog.is_empty() or catalog["prediction_hash"].n_unique() != catalog.height:
    raise RuntimeError("the canonical model catalog is empty or has duplicate identities")

# %% [markdown]
# Canonical analysis also reconstructs the complete request population. Exact hash equality makes a
# missing, extra, or differently resolved checkpoint a blocking error before diagnostics begin.

# %% tags=["results"]
if EXECUTION_TIER == "canonical":
    population = OfficialPopulation.one(study, name=OFFICIAL_POPULATION)
    population.require_complete()
    declared_hashes = set(plan_official_models(study).expected_prediction_hashes)
    frozen_hashes = set(population.members)
    if frozen_hashes != declared_hashes:
        raise RuntimeError("the frozen model population differs from the declared requests")
    if set(catalog["prediction_hash"]) != frozen_hashes:
        raise RuntimeError("the canonical model catalog differs from the declared population")

# %% tags=["results"]
catalog.select(
    "family",
    "label",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
    "ic_mean",
    "auc_roc",
    "log_loss",
).sort("label", "family", "config_name", "checkpoint_value")

# %% [markdown]
# ## Diagnostic summaries
#
# These tables describe the registered validation predictions without collapsing checkpoint
# identity or intersecting away missing keys. Any incomplete row was rejected before this analysis.

# %% tags=["results"]
diagnostics = catalog.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
    "n_folds",
    "ic_mean",
    "ic_std",
    "auc_roc",
    "log_loss",
).sort("label", "family", "config_name", "checkpoint_value")
diagnostics

# %% [markdown]
# The distribution below retains every checkpoint. A family with more configured checkpoints is
# therefore shown with more points rather than receiving extra weight in a collapsed ranking.

# %% tags=["results"]
ic_points = catalog.filter(pl.col("ic_mean").is_not_null()).sort(
    "label", "family", "config_name", "checkpoint_value"
)
fig = px.strip(
    ic_points,
    x="ic_mean",
    y="label",
    color="family",
    hover_data=["config_name", "checkpoint_kind", "checkpoint_value"],
    title="Validation IC for every complete model checkpoint",
    labels={"ic_mean": "Mean validation rank IC", "label": "Prediction target"},
    color_discrete_sequence=ml4t_palette(5, categorical=True),
)
fig.add_vline(x=0, line_width=1, line_color=COLORS["neutral"])
fig.update_layout(legend_title_text="Model family")
show_plotly_with_alt(
    fig,
    "Horizontal strip plot with one point per complete model checkpoint, grouped by prediction "
    "target and colored by model family. A vertical zero line separates positive and negative "
    "validation rank correlations.",
)

# %% [markdown]
# ## Separate causal result
#
# The DML estimand is not placed in the predictive catalog. A reader-facing label selection must
# resolve to exactly one current canonical causal identity.

# %% tags=["results"]
causal = CausalResult.one(study, label="fwd_ret_8h", execution_tier=EXECUTION_TIER)
if not causal.complete:
    raise RuntimeError("causal result is incomplete")
causal_summary = pl.DataFrame(
    {
        "causal_hash": [causal.hash],
        "n_obs": [causal.metrics["n_obs"]],
        "effect": [causal.metrics["dml_effect"]],
        "hac_standard_error": [causal.metrics["dml_se_hac"]],
        "refutation_p": [causal.metrics["refutation_p"]],
    }
)
causal_summary

# %% tags=["results"]
# refutation_p is nullable by contract (utils/causal.py:1111), so it is reported as absent
# rather than formatted. A run whose refutation did not produce an empirical p-value is a
# weaker result, and saying so is more useful than omitting the sentence or failing to render.
_refutation = causal.metrics["refutation_p"]
_refutation_text = (
    f"The temporal refutation p-value is **{_refutation:.3f}**."
    if _refutation is not None
    else "No temporal refutation p-value was registered for this estimate."
)
Markdown(
    f"The registered DML estimate is **{causal.metrics['dml_effect']:+.4g}** with "
    f"a HAC standard error of **{causal.metrics['dml_se_hac']:.4g}**. {_refutation_text} "
    "This result describes the declared causal estimand and does not rank predictive models "
    "or trading strategies."
)

# %% [markdown]
# ## Key takeaways and limitations
#
# - Catalog completeness is checked against the declared canonical checkpoint population before any
#   diagnostic is interpreted.
# - IC, AUC, and loss describe predictions. Validation backtest Sharpe selects the trading
#   configuration later in the pipeline.
# - Model comparisons remain conditional on the finalized features, labels, universe, and validation
#   period. The causal estimate also depends on its declared observed-confounder assumptions.
