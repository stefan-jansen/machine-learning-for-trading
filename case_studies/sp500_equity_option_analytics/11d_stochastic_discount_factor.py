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
# # Stochastic Discount Factor for the S&P 500 Equity-and-Options Cross-Section
#
# The stochastic discount factor (SDF, *Chapter 14*; Chen, Pelger & Zhu 2021) is the adversarial
# no-arbitrage estimator of the latent-factor suite. It learns the pricing kernel directly from
# no-arbitrage moment conditions - a minimax game against a network that builds the worst-case
# mispriced portfolio - and so it collapses the three-stage adapter, leaving no factor-return history
# to hand to a separate forecaster. For comparability with the other families the case study selects
# the reported SDF checkpoint on the validation IC (not the validation-Sharpe criterion of the
# original CPZ protocol). Each estimator is scored by its average daily information coefficient (IC)
# with a HAC-corrected 95% interval; an interval that excludes zero is the bar for signal. This is a
# torch model that also ingests macro inputs; cross-model comparison lives in `13_model_analysis`.

# %%
"""S&P 500 equity+options SDF case-study run via the shared library path."""

import sqlite3
import warnings

import polars as pl

from case_studies.utils.analytics import _registry_path
from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
)

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
N_FACTORS = 5
N_EPOCHS = 50
USE_CACHE = True
FORCE_RETRAIN = False
MAX_FOLDS = 0
MAX_VARIANT_LABELS = -1
USE_MACRO = True
MODEL_NAME = "sdf"

# %%
context = load_case_study_context(
    CASE_STUDY_ID,
    primary_label=PRIMARY_LABEL,
    max_symbols=MAX_SYMBOLS,
    max_folds=MAX_FOLDS,
    max_variant_labels=MAX_VARIANT_LABELS,
    use_macro=USE_MACRO,
)
if MODEL_NAME not in configured_models(context):
    raise ValueError(f"{MODEL_NAME!r} is not configured for {CASE_STUDY_ID}")

# %% [markdown]
# ## Walk-forward cross-validation
#
# The run cell reports the pipeline's fixed reporting-epoch fold-mean IC (`model_results`). For a torch
# model whose validation IC drifts across checkpoints this differs from the best validation checkpoint
# in the significance table below - the number the cross-model comparison in `13_model_analysis` uses.

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="11d_stochastic_discount_factor",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
)
print(result["model_results"])
print(result["fold_metrics"][MODEL_NAME])


# %% [markdown]
# ## Validation significance
#
# The registry stores a HAC-corrected 95% interval for every checkpoint. We report the best validation
# checkpoint (argmax daily IC) - the number the cross-model comparison in `13_model_analysis` uses, the
# same convention applied to the *Chapter 13* deep-learning heads.


# %%
def significance_summary(case_study_id: str, family: str, config_name: str) -> pl.DataFrame:
    """Best validation checkpoint per label, with HAC-corrected IC interval, from the registry."""
    query = """
        SELECT t.label, ps.checkpoint_value AS epoch, COALESCE(pm.ic_mean_daily, pm.ic_mean) AS ic_mean, pm.ic_t_hac,
               pm.ic_p_hac, pm.ic_ci_lo, pm.ic_ci_hi, pm.ic_n_days
        FROM training_runs t
        JOIN prediction_sets ps ON ps.training_hash = t.training_hash
        JOIN prediction_metrics pm ON pm.prediction_hash = ps.prediction_hash
        WHERE t.family = ? AND t.config_name = ? AND ps.split = 'validation'
    """
    with sqlite3.connect(_registry_path(case_study_id)) as con:
        cursor = con.execute(query, [family, config_name])
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    frame = pl.DataFrame(rows, schema=columns, orient="row")
    return (
        frame.sort("ic_mean", descending=True)
        .group_by("label", maintain_order=False)
        .first()
        .sort("label")
        .with_columns(
            pl.col("ic_mean").round(4),
            pl.col("ic_t_hac").round(2).alias("hac_t"),
            pl.col("ic_p_hac").round(3).alias("hac_p"),
            pl.col("ic_ci_lo").round(4).alias("ci_lo"),
            pl.col("ic_ci_hi").round(4).alias("ci_hi"),
        )
        .select("label", "epoch", "ic_mean", "hac_t", "hac_p", "ci_lo", "ci_hi", "ic_n_days")
    )


significance = significance_summary(CASE_STUDY_ID, "latent_factors", MODEL_NAME)
print(significance)

# %% [markdown]
# ## Takeaway
#
# SDF is the strongest latent estimator for this case study: its best validation checkpoint is
# **+0.0124** (HAC *t* 0.78, 95% CI [-0.0189, +0.0438]), above the supervised
# autoencoder (+0.0098) and, like every latent point at the 5-day horizon, covering
# zero - no credible signal. The SDF-over-SAE ordering remains the same as frozen
# *Table 14.3*.
# Read the point estimate and its interval off the same statistic: the table above reports the
# **daily-pooled IC** (`ic_mean_daily`), which is what the HAC *t* and interval are computed from. The
# legacy fold-mean `ic_mean` column ranks the SAE first; that
# is a different statistic, not a competing result, and mixing the two is what makes the book look wrong
# when it is not. The two estimators are statistically indistinguishable here in any case - the ranking
# is a coin flip on numbers that all overlap zero, and the adversarial no-arbitrage objective neither
# adds nor loses signal relative to the other neural estimators. The full comparison against the
# *Chapters 11-13* supervised models is in `13_model_analysis`.
