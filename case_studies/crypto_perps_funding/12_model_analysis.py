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
# refutation_p is nullable by contract (`case_studies/utils/causal.py`), so it is reported as absent
# rather than formatted. A run whose refutation did not produce an empirical p-value is a
# weaker result, and saying so is more useful than omitting the sentence or failing to render.
_refutation = causal.metrics["refutation_p"]
_n_placebo = causal.spec["computation"]["refutation"]["n_placebo"]
_floor = 1.0 / (_n_placebo + 1)
if _refutation is None:
    _refutation_text = "No temporal refutation p-value was registered for this estimate."
elif _refutation <= _floor:
    # Reaching the floor is only possible when every placebo fit succeeded and none was
    # as extreme as the observed effect: p = (1 + exceedances) / (1 + successful), which
    # is at or below 1/(n+1) only when the numerator is 1 and the denominator is the full
    # requested count. So the requested count is the right denominator on this branch even
    # though `empirical_permutation_p` drops placebos whose second stage returned NaN.
    #
    # What the floor does NOT establish is that the underlying permutation tail probability
    # is at most 1/(n+1). That is a property of all permutations; this is a sample of n of
    # them, and zero exceedances in a sample bounds the sample, not the population.
    _refutation_text = (
        f"The temporal refutation p-value is **{_refutation:.3f}**, the smallest value "
        f"{_n_placebo} permutations can produce: every placebo fit succeeded and none was "
        f"as extreme as the observed effect. This is the Monte Carlo estimate sitting at "
        f"its resolution floor of 1/{_n_placebo + 1}, not a bound on the permutation tail "
        f"probability itself - resolving that finer needs more permutations, not a smaller "
        f"reported number."
    )
else:
    _refutation_text = (
        f"The temporal refutation p-value is **{_refutation:.3f}**, a Monte Carlo estimate "
        f"over {_n_placebo} requested permutations and no finer than that sample supports."
    )

_effect = causal.metrics["dml_effect"]
_se = causal.metrics["dml_se_hac"]
_t = _effect / _se if _se else float("nan")
# The block and the bandwidth are sized by different rules and the reader is being asked to
# weigh a ratio built from the second. The bandwidth is not statsmodels': it requires an
# explicit `maxlags` for both `HAC` and `hac-groupsum` and supplies no default. The
# cube-root-of-decision-times rule is this repository's own fallback in `run_dml_analysis`,
# applied because no `hac_maxlags` is passed (`case_studies/utils/causal.py:467-472`), and
# raised only to `horizon - 1`. The realized bandwidth is not registered - `causal_runs`
# stores neither `hac_maxlags` nor `n_periods` - so the notebook states the rule rather than
# a number it cannot read back.
#
# The caveat is worded off `block_size_basis`, not off `block_size` alone. The block is
# `max(horizon_steps, treatment_window_steps or 1)`, so on a run whose treatment declares no
# window it collapses to the label horizon - and then the block is not treatment persistence
# and the bandwidth no longer covers fewer lags than it, since `hac_maxlags >= horizon - 1`.
# Saying so unconditionally would invert the moment someone re-runs this against a treatment
# the resolver cannot size.
_refutation_spec = causal.spec["computation"]["refutation"]
_block = _refutation_spec["block_size"]
_basis = _refutation_spec["block_size_basis"]
if _basis == "treatment_window":
    _bandwidth_text = (
        f"One caveat on the ratio: the placebo block spans **{_block}** bars of treatment "
        "persistence, but the standard error behind this ratio does not. Its HAC bandwidth "
        "is this repository's cube-root-of-decision-times fallback raised to cover the label "
        "horizon, and nothing ties it to the treatment's window, so it covers materially "
        "fewer lags than the block. Which way that moves the standard error is not something "
        "the mismatch settles: a HAC estimate is not monotonic in its bandwidth, because the "
        "autocovariances a longer bandwidth admits can carry either sign, and the treatment's "
        "construction window is an argument for the block rather than a derivation of the "
        "right bandwidth. Read the ratio as provisional until the estimate is recomputed "
        "across a range of defensible bandwidths. "
    )
else:
    _bandwidth_text = (
        f"The placebo block spans **{_block}** bars on the label horizon, the resolver having "
        "found no declared construction window for the treatment, so the block is not sized "
        "by treatment persistence and the standard error's bandwidth already covers it. "
    )
Markdown(
    f"The registered DML estimate is **{_effect:+.4g}** with "
    f"a HAC standard error of **{_se:.4g}**, a ratio of **{_t:.2f}**. {_refutation_text} "
    "Read the two together: the placebo test asks whether the estimation procedure "
    "manufactures this effect out of permuted treatment, and the standard error asks "
    "whether the effect is separable from zero at all. Surviving the first while failing "
    "the second is not evidence of a causal effect. "
    f"{_bandwidth_text}"
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
