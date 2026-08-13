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
# # IPCA for the S&P 500 Equity-and-Options Cross-Section
#
# Instrumented PCA (IPCA, *Chapter 14*) is the conditioned linear estimator of the latent-factor
# suite: instead of estimating loadings from returns alone, it lets each asset's factor betas vary
# with observable characteristics, so the panel does not need to stay balanced the way plain PCA
# does. It is the one linear estimator in the suite that carries a characteristic signal into the
# factor structure. Each estimator is scored by its average daily information coefficient (IC) with
# a HAC-corrected 95% interval; an interval that excludes zero is the bar for signal. Cross-model
# comparison lives in `13_model_analysis`.

# %%
"""S&P 500 equity+options IPCA case-study run via the shared library path."""

import sqlite3
import warnings

import polars as pl

from case_studies.utils.analytics import _registry_path
from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
    run_case_study_variants,
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
RUN_VARIANTS = True
USE_MACRO = False
MODEL_NAME = "ipca"

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
# ## Walk-forward cross-validation (primary label)

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="11b_ipca",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
)
print(result["model_results"])
print(result["fold_metrics"][MODEL_NAME])

# %% [markdown]
# ## Alternative labels
#
# The same estimator is scored against the 10-day return and the risk-adjusted 5-day return, the
# two secondary labels this case study carries.

# %%
variant_results = {}
if RUN_VARIANTS and context.variant_labels:
    variant_results = run_case_study_variants(
        context,
        model_name=MODEL_NAME,
        notebook="11b_ipca",
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        use_cache=USE_CACHE,
        force_retrain=FORCE_RETRAIN,
    )
    for label, variant_result in variant_results.items():
        print(label, variant_result["model_results"])


# %% [markdown]
# ## Validation significance
#
# The registry holds a HAC-corrected 95% interval for every checkpoint. For each label we report
# the best validation checkpoint (the one the model-selection rule keeps), so the interval below
# is the number carried forward to `13_model_analysis`.


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
# On the plain 5-day label IPCA's validation IC is **-0.0018** (HAC *t* -0.12,
# 95% CI [-0.0317, +0.0281]) and on the 10-day label **-0.0062** (*t* -0.59,
# CI [-0.0269, +0.0145]); both sit squarely on zero. The result that stands out is
# the **risk-adjusted 5-day label: +0.0419** (HAC *t* 2.93, *p* 0.004,
# CI [+0.0138, +0.0701]), an interval clear of zero. Conditioning factor betas on
# characteristics recovers target-specific evidence once the label is scaled by risk,
# even though the primary weekly target remains null. PCA also clears zero on two
# secondary targets. The full comparison against the *Chapters 11-13* supervised
# models is in `13_model_analysis`.
