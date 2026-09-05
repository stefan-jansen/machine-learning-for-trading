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
# # NASDAQ-100 Microstructure - Order Flow: Causal DML Estimation
#
# Does signed volume share *cause* future 15-minute returns, or is the observed price impact
# already accounted for by spread and volatility? This notebook applies double machine
# learning (DML) to `signed_vol_share` across NASDAQ-100 stocks on a one-minute observation
# grid.
#
# DML separates a treatment effect from confounding in two stages. The first stage predicts
# the outcome from the confounders, and separately predicts the treatment from the same
# confounders, using cross-fitting so no observation helps predict itself. The second stage
# regresses the outcome residual on the treatment residual, which measures the part of the
# treatment the confounders do not explain. That remainder is the quantity a causal claim
# needs.
#
# **The treatment** is signed volume share: the fraction of a bar's traded volume classified
# as buyer-initiated by the Lee-Ready rule, which assigns a trade to the buyer when it prints
# above the prevailing quote midpoint. **The confounders** are the relative bid-ask spread at
# the close of the bar, which stands for how costly the name is to trade; five-minute
# realized volatility, which stands for how noisy it currently is; and the trailing one-month
# cumulative return, which stands for slower drift the bar inherits.
#
# **Learning objectives**
#
# - Say what a causal estimand adds to a specification that a predictive one does not carry.
# - Read the analysis population, the temporal geometry and the refutation design out of the
#   resolved identity before anything is fitted.
# - Explain why an embargo and a permutation block on this panel are sized from the label
#   horizon rather than from the bar, and what a wrong reading of "period" costs.
# - Read a permutation refutation against a parametric standard error, and say which to
#   believe when they disagree.
# - Say why a causal row is registered separately from every prediction set and selects
#   nothing.
#
# **Book reference**: Chapter 15, Section 15.6 (cross-dataset causal evidence).
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) for the treatment
# and the confounders, [`04_model_based_features`](04_model_based_features.ipynb) for the rest
# of the panel, and [`05_evaluation`](05_evaluation.ipynb) for the holdout boundary this
# analysis stays behind.
#
# **What it writes**: one row in `causal_runs`, for the primary label, carrying the estimate,
# its Newey-West standard error, the naive comparison and the refutation p-value, under a
# hashed identity derived from the estimand, the analysis population and the refutation
# design. [`13_model_analysis`](13_model_analysis.ipynb) reads that row as causal evidence and
# keeps it out of the predictive comparison. **It selects nothing**: selection is validation
# backtest Sharpe in [`14_backtest`](14_backtest.ipynb), and a causal estimate is not a
# candidate.

# %% [markdown]
# ## Identifying assumptions
#
# DML estimates a causal effect under three assumptions, and none of them is tested here:
#
# 1. **Conditional ignorability**: every backdoor path between treatment and outcome is
#    blocked by the declared confounders. There is no unobserved third thing driving both.
# 2. **Overlap**: every name could have carried any level of the treatment, given its
#    confounders.
# 3. **SUTVA**: one stock's order flow does not change another stock's return.
#
# The refutation test in section 3 is indirect evidence about the estimator, not about these.
# It asks whether the estimate stands once the treatment's timing is destroyed, which a
# spurious estimate often does not - but standing there is not the same as the assumptions
# holding.

# %%
"""Resolve, estimate and register the NASDAQ-100 microstructure causal DML specification."""

import numpy as np
import plotly.graph_objects as go
import polars as pl

from case_studies.research import ExecutionTier, causal_supersedes, open_study, primary_label
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
LABEL = ""
RANDOM_SEED = 42
MAX_SYMBOLS = 0
CV_FOLDS = 0
MAX_SAMPLES = 0
N_PLACEBO = 0
# Declared rather than inferred from whether a reduction happens to be set. Inferring the tier
# left a reduced run writing into the case study's own artifacts, which is the production
# path. WORKSPACE is the other half: a preview has nowhere else to write.
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
# The causal identity this run retires, if any. A label resolves to exactly one current
# identity, so a refit under changed code produces a second and the registry refuses it until
# it is told which one it replaces. This case study has registered no causal run yet, so there
# is nothing to retire and the declaration is empty; the error raised on the first refit names
# the hash to put here. `causal_supersedes` resolves it against the registry in hand and
# withholds it where there is nothing to retire, so one declaration is right for a reader
# whose gitignored `run_log/` holds no causal rows at all.
SUPERSEDES_CAUSAL: str = ""

# %% [markdown]
# ## 1. Declaring the request
#
# Nothing about the estimand is written here. The treatment, the confounders and the method
# come from `config/setup.yaml`; the fold count, the placebo count and the seed come from the
# shared `causal_dml` configuration. What this cell declares is which label to estimate the
# effect on and under which execution tier.
#
# A nonzero fold, sample, symbol or placebo limit is a **preview**: a reduced run that writes
# to a throwaway workspace and is excluded from canonical evidence. A canonical run cannot
# carry one, and a preview must declare every one of them - a preview missing `max_samples`
# would otherwise resolve the full population, which is the opposite of what the tier is for.
#
# **A canonical run is uncapped, and on this panel that is the whole difference.** Until this
# notebook was converted it built its own specification and applied `max_samples: 50000` from
# `case_studies/config/dml/dml.yaml` by hand - about one trading day of a one-minute panel
# across roughly 115 names. `resolve_causal_request` ignores that key deliberately, so the
# canonical estimate below is taken over every admissible row instead. Expect the number to
# differ from any earlier hand-capped one by more than a rounding: on
# `sp500_equity_option_analytics` the same removal took the fit from 37,240 rows to 376,871
# and flipped the sign of the point estimate.

# %%
set_global_seeds(RANDOM_SEED)

tier = ExecutionTier(EXECUTION_TIER)
study = open_study(CASE_STUDY_ID, execution_tier=tier, workspace=WORKSPACE or None)
label = LABEL or primary_label(study)

REDUCTION_PARAMETERS = {
    "max_symbols": MAX_SYMBOLS or None,
    "n_folds": CV_FOLDS or None,
    "max_samples": MAX_SAMPLES or None,
    "n_placebo": N_PLACEBO or None,
}
reductions = {key: value for key, value in REDUCTION_PARAMETERS.items() if value is not None}
if tier is ExecutionTier.PREVIEW and not reductions:
    raise ValueError("preview execution must declare its reductions")
if tier is ExecutionTier.CANONICAL and reductions:
    raise ValueError(f"canonical execution cannot carry reductions: {sorted(reductions)}")

request = study.causal(
    method="dml",
    label=label,
    config_name="dml",
    execution_tier=tier,
    preview_reductions=reductions,
    overrides={},
    supersedes=causal_supersedes(
        study, SUPERSEDES_CAUSAL, label, labels=[label], execution_tier=tier.value
    ),
)
print(f"{CASE_STUDY_ID}: causal DML on {label}, tier {tier.value}")

# %% [markdown]
# ## 2. What resolving the request settles
#
# **Resolving** goes and finds everything the declaration left open: it loads the label and
# feature artifacts, checks the treatment and every confounder is present, computes the fold
# boundaries and the embargo from the panel's own observed cadence, and works out the exact
# set of symbol-timestamp pairs the estimate is taken over. It fits nothing, so all of it can
# be read before any cost is paid.
#
# Three things worth reading in it, and on this panel the first two are the notebook's whole
# subject.
#
# - **The cadence, measured rather than declared.** `observation_cadence` comes from the
#   spacing the data actually has, which here is one minute. `decision.bar_frequency` in
#   `setup.yaml` says fifteen minutes, and that is how often the strategy *acts*, not how far
#   apart the observations are. An embargo and a block length derived from the decision
#   frequency would come out as one period instead of fifteen, and one period is a number that
#   looks entirely normal in a printed table.
# - **The temporal controls, sized from the label horizon.** The label looks forward fifteen
#   minutes while the panel is observed once a minute, so a row's outcome overlaps the next
#   fourteen rows' outcomes. `embargo_periods` is the gap left between a training block and
#   the validation block after it: shorter than the horizon and the last training rows resolve
#   inside the validation window, so the fit has seen part of what it is scored on.
#   `holdout_endpoint_cutoff` is the same argument at the holdout boundary - the last admitted
#   decision time is one horizon before it, because a bar timestamped just inside would
#   resolve outside.
# - **The refutation design.** `block_size` is how many consecutive decision times move
#   together when the placebo shuffles the treatment, and `block_size_basis` names what set
#   it: the label buffer or the treatment's own construction window, whichever is longer.
#   Blocks shorter than the overlap leave the serial dependence in place, and the placebo
#   distribution then comes out narrower than the data supports - a p-value that reads like a
#   refutation without being one.
#
# All of it goes into the hashed identity, so an estimate made under a different population, a
# different embargo or a different placebo design registers beside this one rather than over
# it.

# %%
resolved = request.resolve()
computation = resolved.spec["computation"]
estimand = computation["estimand"]

print(f"Outcome:     {estimand['outcome']} over {estimand['outcome_horizon']}")
print(f"Treatment:   {estimand['treatment']}")
print(f"Confounders: {', '.join(estimand['confounders'])}")
print(f"Last admissible decision time: {estimand['holdout_endpoint_cutoff']}")
print(f"Identity:    {resolved.identity}")

# %% tags=["results"]
_settled = pl.DataFrame(
    {
        "setting": [
            "observation cadence",
            "label horizon",
            "embargo",
            "cross-fitting folds",
            "fold unit",
            "placebo method",
            "placebo block",
            "block basis",
            "placebo draws",
            "analysis rows",
            "decision times",
            "nuisance model",
        ],
        "value": [
            str(computation["refutation"]["observation_cadence"]),
            str(estimand["outcome_horizon"]),
            f"{computation['cv']['embargo_periods']} periods",
            str(computation["cv"]["n_folds"]),
            str(computation["cv"]["fold_unit"]),
            str(computation["refutation"]["method"]),
            f"{computation['refutation']['block_size']} periods",
            str(computation["refutation"]["block_size_basis"]),
            str(computation["refutation"]["n_placebo"]),
            f"{computation['analysis_population']['n_rows']:,}",
            f"{computation['analysis_population']['n_timestamps']:,}",
            str(computation["model"]["class"]),
        ],
    }
)
_settled

# %% [markdown]
# The embargo and the block are both fifteen periods on a one-minute cadence, which is the
# label horizon and not a coincidence: both answer the same overlap. `block_size_basis` says
# which of the two candidates set the block - `label_buffer` here, because the treatment's own
# construction window is one bar and the label's is fifteen, so the label's is what the
# placebo has to preserve.

# %% [markdown]
# ## 3. Estimating and registering
#
# `run()` fits the two nuisance models with cross-fitting and the embargo above, estimates the
# effect from the residuals, runs the placebo draws, and registers one row - **only after** the
# fit returns a finite effect and a finite standard error. A missing treatment, a missing
# confounder or an empty analysis population fails before anything is written.
#
# The registration goes through the resolver's own path rather than a convenience wrapper, and
# that is not a stylistic choice. A row written by `register_causal_run` carries no
# `identity_version`, and `current_causal_identities` - the set `CausalResult.one` resolves and
# the ambiguity check is computed from - skips exactly those rows. Such a row is written, sits
# in the table, and answers nothing; `13_model_analysis` would show no causal evidence at all
# while `causal_runs` held a row. This notebook was the last one in the corpus on that path.
#
# **A second run of this notebook fits nothing.** The identity is re-derived from the inputs,
# the registry already holds the matching row, and the cache answers. That is why everything
# reported below is read back from the registered row rather than from the return value of a
# fit: a reader re-running this notebook has to see the same numbers, and a cached run has no
# placebo draws to show.

# %% tags=["results"]
result = resolved.run()
if not result.complete:
    raise RuntimeError(f"the registered causal result for {label} is incomplete")
# The identity-bearing computation, not the whole spec. `provenance` records the commit of the
# run that registered the row, so comparing whole specs would assert that nothing had been
# committed since - which is not a property of the estimate.
if result.spec["computation"] != resolved.spec["computation"]:
    raise RuntimeError(f"the registered causal computation for {label} differs from the resolved")
if request.resolve().run().hash != result.hash:
    raise RuntimeError("re-resolving the same request changed its identity")

metrics = result.metrics
print(f"Registered causal identity: {result.hash}")

# %% [markdown]
# ## 4. What came out
#
# Every number below is read back from the registered row.
#
# `naive_effect` is the coefficient on the treatment with the confounders entered linearly and
# nothing orthogonalized. `dml_effect` is the same quantity after both the outcome and the
# treatment have had the confounders' flexible predictions removed. The distance between them
# is what adjustment moved, and `confounding_bias_pct` normalizes it - by the DML estimate, so
# when that estimate is near zero the percentage is unstable and can run past the whole of it
# with both raw effects tiny. Read the raw effects.
#
# `dml_se_hac` is a Newey-West standard error with the lag count set from the label horizon
# rather than inferred. Left to infer, it falls back to a rule based only on sample size,
# which under-lags an overlapping label and reports a t-statistic that is too large.
#
# The refutation p-value is a permutation p-value and reads `not run` when fewer than the
# minimum placebo draws returned a finite effect. That is a missing measurement, not a
# p-value of 1: the registry stores NULL for it and this table says so rather than printing a
# number for a test that did not happen.

# %% tags=["results"]
summary = pl.DataFrame(
    {
        "quantity": [
            "observations",
            "naive effect",
            "DML effect",
            "Newey-West SE",
            "t statistic",
            "p-value (Newey-West)",
            "confounding bias %",
            "refutation p (block permutation)",
            "successful placebo draws",
            "refutation class",
        ],
        "value": [
            f"{metrics['n_obs']:,}",
            f"{metrics['naive_effect']:+.3e}",
            f"{metrics['dml_effect']:+.3e}",
            f"{metrics['dml_se_hac']:.3e}",
            f"{metrics['dml_effect'] / metrics['dml_se_hac']:+.2f}",
            f"{metrics['p_value_hac']:.4f}",
            f"{metrics['confounding_bias_pct']:+.1f}",
            "not run" if metrics["refutation_p"] is None else f"{metrics['refutation_p']:.4f}",
            str(metrics["refutation_n_successful"]),
            str(metrics["refutation_class"]),
        ],
    }
)
summary

# %% [markdown]
# ### The observed effect against its placebos
#
# The permutation test asks how often the same estimator recovers an effect this large from
# data whose treatment has been shuffled in blocks - which keeps the panel's shape but
# destroys the treatment's alignment with the outcome. The two significance readings answer
# different questions and can disagree. The Newey-West p-value asks whether the effect is
# distinguishable from zero under a parametric model of the errors; the permutation p-value
# needs no such model, but can only reject in proportion to the draws run. When the
# permutation test does not corroborate the parametric one, the parametric standard error is
# the reading to distrust: it rests on an assumption about the errors that the shuffle does
# not need.

# %%
placebo = np.asarray(metrics.get("placebo_effects") or [], dtype=float)
if placebo.size:
    fig = go.Figure()
    fig.add_histogram(x=placebo, nbinsx=30, name="Placebo effects", marker_color=COLORS["blue"])
    fig.add_vline(
        x=float(metrics["dml_effect"]),
        line_color=COLORS["amber"],
        line_width=2,
        annotation_text="observed",
    )
    fig.update_layout(
        title="Observed effect against its block-permutation placebos",
        xaxis_title="Treatment effect",
        yaxis_title="Placebo replications",
    )
    show_plotly_with_alt(
        fig,
        "Histogram of block-permutation placebo treatment effects with the observed DML "
        "effect marked, showing where the estimate falls in the null distribution.",
    )
else:
    print("No placebo draws are stored on this row, so there is no null distribution to show.")

# %% [markdown]
# ## Key takeaways
#
# 1. **Size the embargo from the label horizon, not the bar.** On a one-minute grid a
#    15-minute label overlaps the next fourteen observations. An embargo counted in bars
#    rather than in horizons leaves most of that overlap inside the training block, and the
#    effect it produces is partly the model reading its own training window. The resolver
#    measures the cadence from the data for this reason, rather than trusting the declared
#    decision frequency.
#
# 2. **Seal the holdout one horizon early.** Filtering on the holdout date alone is not enough
#    when the label looks forward: the last admitted bar resolves after the boundary.
#    `holdout_endpoint_cutoff` is that subtraction, computed rather than typed.
#
# 3. **A reduction is declared, not applied by hand.** The cap this notebook used to apply
#    itself is now a preview reduction that a canonical run refuses, and the tier is stated
#    rather than inferred from whether a limit happens to be set. A capped estimate and a full
#    one then cannot be confused for each other, because they are different tiers writing to
#    different places under different identities.
#
# 4. **The two significance readings are not interchangeable.** A parametric standard error
#    assumes a model for the errors; the permutation test does not, but it can only reject in
#    proportion to the replications run. Report both, and treat disagreement as a reason to
#    distrust the parametric one rather than a result to average away.
#
# **Known limitations.** The three identifying assumptions above are untestable from this data
# and the refutation is indirect evidence about the estimator rather than about them. The
# effect is measured per unit of `signed_vol_share`; converting it into anything tradable
# needs that unit's dispersion, which [`17_costs`](17_costs.ipynb) supplies. And the estimate
# is a statement about the pre-holdout panel this case study admits, not about NASDAQ-100
# order flow in general.
