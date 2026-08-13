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
# # Latent Factor Model Suite for S&P 500 Equity-and-Options
#
# *Chapter 14* fits five latent-factor estimators on this cross-section, one per notebook, from the
# unconditioned baseline through to the two deep models that bypass the three-stage adapter:
#
# - `11a_pca` - unconditioned variance baseline
# - `11b_ipca` - characteristic-conditioned linear betas
# - `11c_conditional_autoencoder` - nonlinear reconstruction factors
# - `11d_stochastic_discount_factor` - adversarial no-arbitrage pricing kernel
# - `11e_supervised_autoencoder` - end-to-end return prediction
#
# Each head notebook holds the per-model detail; this index ranks them on the primary 5-day label. The
# full comparison against the *Chapters 11-13* supervised models is in `13_model_analysis`.

# %%
"""Latent factor notebook index for the S&P 500 equity+options case study."""

import sqlite3
import warnings

import polars as pl

from case_studies.utils.analytics import _registry_path, load_best_ic_per_family

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
PRIMARY_LABEL = "fwd_ret_5d"

# %% [markdown]
# ## Latent-factor leaderboard (primary label)
#
# Each estimator's best validation checkpoint on the 5-day label (argmax daily IC), with its
# HAC-corrected 95% interval - the same selection the cross-model comparison in `13_model_analysis`
# uses. An interval that excludes zero is the bar for signal.


# %%
def latent_leaderboard(case_study_id: str, label: str) -> pl.DataFrame:
    """Best validation checkpoint per latent-factor config for one label, ranked by IC."""
    with sqlite3.connect(_registry_path(case_study_id)) as con:
        metric_columns = {
            row[1] for row in con.execute("PRAGMA table_info(prediction_metrics)").fetchall()
        }
        ic_expr = (
            "COALESCE(pm.ic_mean_daily, pm.ic_mean)"
            if "ic_mean_daily" in metric_columns
            else "pm.ic_mean"
        )
        query = f"""
            SELECT t.config_name, ps.checkpoint_value AS epoch, {ic_expr} AS ic_mean,
                   pm.ic_t_hac, pm.ic_p_hac, pm.ic_ci_lo, pm.ic_ci_hi
            FROM training_runs t
            JOIN prediction_sets ps ON ps.training_hash = t.training_hash
            JOIN prediction_metrics pm ON pm.prediction_hash = ps.prediction_hash
            WHERE t.family = 'latent_factors' AND t.label = ? AND ps.split = 'validation'
        """
        cursor = con.execute(query, [label])
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    frame = pl.DataFrame(rows, schema=columns, orient="row")
    return (
        frame.sort("ic_mean", descending=True)
        .group_by("config_name", maintain_order=False)
        .first()
        .sort("ic_mean", descending=True)
        .with_columns(
            pl.col("ic_mean").round(4),
            pl.col("ic_t_hac").round(2).alias("hac_t"),
            pl.col("ic_p_hac").round(3).alias("hac_p"),
            pl.col("ic_ci_lo").round(4).alias("ci_lo"),
            pl.col("ic_ci_hi").round(4).alias("ci_hi"),
        )
        .select("config_name", "epoch", "ic_mean", "hac_t", "hac_p", "ci_lo", "ci_hi")
    )


leaderboard = latent_leaderboard(CASE_STUDY_ID, PRIMARY_LABEL)
print(leaderboard)

# %% [markdown]
# ## Family best (registry SSOT)
#
# `load_best_ic_per_family` is the single source of truth the cross-model comparison reads: it selects
# the best validation checkpoint of the best config per family.

# %%
best = load_best_ic_per_family(
    families=["latent_factors"],
    case_studies=[CASE_STUDY_ID],
)

if best.is_empty():
    print("No latent-factor results are registered yet for this case study.")
else:
    print(best)

# %% [markdown]
# ## Takeaway
#
# On the primary 5-day label the stochastic discount factor leads (**+0.0124**, HAC *t* 0.78),
# above the supervised autoencoder (**+0.0098**, *t* 0.68); CAE (+0.0014), IPCA
# (-0.0018), and the unconditioned PCA baseline (-0.0126) trail. Every one of these intervals covers zero -
# the equity-and-options study is the one case in *Chapter 14* where the strongest latent estimator
# overlaps zero, so no latent estimator delivers a credible 5-day signal. Secondary targets differ:
# IPCA clears zero on the risk-adjusted 5-day label (+0.0419), while PCA clears zero on both the
# 10-day (+0.0815) and risk-adjusted 5-day (+0.0444) labels. The SDF-over-SAE primary-label ordering
# remains the frozen ordering, but only when the point estimate is read off the same
# statistic as its interval: the leaderboard above uses the **daily-pooled IC** (`ic_mean_daily`), which
# is what the HAC *t* and CI are computed from. The legacy fold-mean `ic_mean` column reverses the top
# two. The gap between them is a coin flip on numbers that all overlap zero, so nothing downstream turns
# on it. The cross-model comparison against the supervised families is in `13_model_analysis`.
