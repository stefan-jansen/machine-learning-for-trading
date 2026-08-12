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
# # Conditional Autoencoder for the S&P 500 Equity-and-Options Cross-Section
#
# The conditional autoencoder (CAE, *Chapter 14*) is the nonlinear reconstruction estimator of the
# latent-factor suite: it generalizes IPCA's characteristic-conditioned loadings through a neural
# network and fits the factors by minimizing reconstruction error on the return panel. Unlike the
# supervised autoencoder it has no forward-return target - its bottleneck is a conditional factor
# structure disciplined by reconstruction, interpretable as risk. Each estimator is scored by its
# average daily information coefficient (IC) with a HAC-corrected 95% interval; an interval that
# excludes zero is the bar for signal. This is a torch model; cross-model comparison lives in
# `13_model_analysis`.

# %%
"""S&P 500 equity+options CAE case-study run via the shared library path."""

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
USE_MACRO = False
MODEL_NAME = "cae"

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
# model whose validation IC drifts across epochs this differs from the best validation checkpoint in
# the significance table below. Notebook 13 first selects one daily-IC representative for the entire
# latent-factor family; the per-epoch `fold_metrics` table makes CAE's epoch-to-epoch wander visible.

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="11c_conditional_autoencoder",
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
# The registry stores a HAC-corrected 95% interval for every checkpoint. We report the best
# validation checkpoint (argmax daily IC). Notebook 13 then carries only the latent-family leader into
# its cross-family comparison, the same one-representative-per-family convention applied elsewhere.


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
# CAE's validation IC is unstable across epochs and sits at or below zero for most of them (the
# pipeline's fixed reporting-epoch fold mean is **-0.0049**). The best daily-IC checkpoint is
# **+0.0014** at epoch 50 (HAC *t* 0.08, 95% CI [-0.0300, +0.0327]),
# indistinguishable from zero. Reconstruction-error factors fit
# contemporaneous covariance rather than a forward target, and on this 5-day cross-section that objective
# carries no signal - the prediction-protocol caveat of *Chapter 14*, §14.5. Any latent signal in this
# study is target-specific: IPCA and PCA clear zero on secondary labels, while SDF is nominally the
# strongest latent estimator on the 5-day label though its interval also covers zero.
# The full comparison against the *Chapters 11-13* supervised models is in `13_model_analysis`.
