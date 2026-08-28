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
# # ETF momentum: a causal estimate, not a forecast
#
# Every model from [`06_linear`](06_linear.ipynb) to
# [`11e_supervised_autoencoder`](11e_supervised_autoencoder.ipynb) answers the same question: given
# this fund's features today, what is its return going to be. This notebook asks a different one.
# Skip-recent momentum - `skip_recent_6_1`, the six-month return that skips the most recent month -
# is one column of that feature matrix. **If two otherwise identical funds differed only in that
# column, how much would their next 21-day returns differ?**
#
# That is not what a predictive model estimates, and the gap matters. Momentum moves with
# volatility and with the risk-on/risk-off cycle, and so does the forward return. A regression of
# return on momentum absorbs all of that into the momentum coefficient. Double machine learning
# separates them: it fits one flexible model for the return given the confounders, another for the
# treatment given the same confounders, and estimates the effect from what neither model could
# explain. What is left is the part of the momentum-return relationship the confounders cannot
# account for.
#
# **The confounders are declared in `config/setup.yaml`, not chosen here**: realized volatility at
# 21 and 126 days, a regime indicator, and the yield-curve slope. Each one moves both the treatment
# and the outcome, which is what makes it a confounder rather than a control.
#
# **Learning objectives**
#
# - Say what a causal estimand adds to a specification that a predictive one does not carry.
# - Read the analysis population, the temporal geometry and the refutation design out of the
#   resolved identity before anything is fitted.
# - Read a Driscoll-Kraay standard error against a block-permutation p-value, and say which to
#   believe when they disagree.
# - Say why a causal row is registered separately from every prediction set and selects nothing.
#
# **Book reference**: Chapter 15, Section 15.6 (cross-dataset causal evidence).
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) for the treatment and
# the confounders, [`04_model_based_features`](04_model_based_features.ipynb) for the rest of the
# panel, and [`05_evaluation`](05_evaluation.ipynb) for the holdout boundary this analysis stays
# behind.
#
# **What it writes**: one row in `causal_runs`, for the primary label, carrying the estimate, its
# Driscoll-Kraay standard error, the naive comparison and the refutation p-value, under a hashed
# identity derived from the estimand, the analysis population and the refutation design.
# [`13_model_analysis`](13_model_analysis.ipynb) reads that row as causal evidence and keeps it out
# of the predictive comparison. **It selects nothing**: selection is validation backtest Sharpe in
# [`14_backtest`](14_backtest.ipynb), and a causal estimate is not a candidate.

# %% [markdown]
# ## Identifying assumptions
#
# DML estimates a causal effect under three assumptions, and none of them is tested here:
#
# 1. **Conditional ignorability**: every backdoor path between treatment and outcome is blocked by
#    the declared confounders. There is no unobserved third thing driving both.
# 2. **Overlap**: every fund could have carried any level of the treatment, given its confounders.
# 3. **SUTVA**: one fund's momentum does not change another fund's return.
#
# The refutation test in section 4 is indirect evidence about the estimator, not about these. It
# asks whether the estimate stands once the treatment's timing is destroyed, which a spurious
# estimate often does not - but standing there is not the same as the assumptions holding.
#
# **Research-design limitations.** This is a stability analysis on the fixed ETF research universe
# and the materialized feature panel. `regime` and `yield_curve_slope` are finalized FRED values
# rather than point-in-time vintages, and their feature timestamps are used as recorded. The
# analysis does not claim those macro observations were available at the same close. That narrows
# the reading to this retrospective panel, which is what it is for.

# %%
"""Resolve, estimate and register the ETF causal DML specification."""

import plotly.graph_objects as go
import polars as pl

from case_studies.research import ExecutionTier, open_study, primary_label, supersedes_for
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
LABEL = ""
RANDOM_SEED = 42
MAX_SYMBOLS = 0
CV_FOLDS = 0
MAX_SAMPLES = 0
N_PLACEBO = 0
# The tier is declared rather than inferred from whether a reduction happens to be set. Inferring
# it left a reduced run writing into the case study's own artifacts, which is the production path.
# WORKSPACE is the other half: a preview has nowhere else to write.
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
# The identity this run retires: the estimate fitted with a placebo block sized by the label
# buffer, before the block was sized by the treatment window. Empty on a clean clone, which has
# nothing to retire; the registry names the hash in the error it raises when there is.
SUPERSEDES_CAUSAL: str = "c4abd6b2f6ff"

# %% [markdown]
# ## 1. Declaring the request
#
# Nothing about the estimand is written here. The treatment, the confounders and the method come
# from `config/setup.yaml`; the fold count, the placebo count and the seed come from the shared
# `causal_dml` configuration. What this cell declares is which label to estimate the effect on and
# under which execution tier.
#
# A nonzero fold, sample, symbol or placebo limit is a **preview**: a reduced run that writes to a
# throwaway workspace and is excluded from canonical evidence. A canonical run cannot carry one,
# and a preview must declare every one of them - a preview missing `max_samples` would otherwise
# resolve the full population, which is the opposite of what the tier is for.

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
    supersedes=supersedes_for(SUPERSEDES_CAUSAL, label, labels=[label]),
)
print(f"{CASE_STUDY_ID}: causal DML on {label}, tier {tier.value}")

# %% [markdown]
# ## 2. What resolving the request settles
#
# **Resolving** goes and finds everything the declaration left open: it loads the label and feature
# artifacts, checks the treatment and every confounder is present, computes the fold boundaries and
# the embargo from the panel's own observed cadence, and works out the exact set of fund-date pairs
# the estimate is taken over. It fits nothing, so all of it can be read before any cost is paid.
#
# The three things worth reading in it:
#
# - **The estimand.** Outcome, treatment, confounders, the horizon the outcome runs over, and
#   `holdout_endpoint_cutoff` - the last decision date whose outcome resolves before the holdout
#   begins. Every row after it is dropped. The cutoff is computed **per fund** against that fund's
#   own observation count rather than by stepping back a calendar duration, because a 21-day buffer
#   is about fifteen sessions on a five-session week and subtracting it as calendar time leaves the
#   last few retained rows resolving inside the holdout.
# - **The temporal controls.** Cross-fitting groups whole decision timestamps rather than rows, so
#   no date is split across a fold boundary and the cross-section stays intact. The embargo is at
#   least the outcome horizon, which is what stops a training fold from containing the return a
#   validation row is still earning.
# - **The refutation design.** The placebo permutes **contiguous blocks within each fund**, not
#   rows, so some of the treatment's serial dependence is carried into the null rather than
#   destroyed by the shuffle. How much depends on how long the blocks are, which the subsection
#   after the table takes up - and it is the difference between a null that means what it reads as
#   and one that does not.
#
# All of it goes into the hashed identity, so an estimate made under a different population, a
# different embargo or a different placebo design registers beside this one rather than over it.

# %%
resolved = request.resolve()
computation = resolved.spec["computation"]
estimand = computation["estimand"]

print(f"Outcome:     {estimand['outcome']} over {estimand['outcome_horizon']}")
print(f"Treatment:   {estimand['treatment']}")
print(f"Confounders: {', '.join(estimand['confounders'])}")
print(f"Last admissible decision date: {estimand['holdout_endpoint_cutoff'][:10]}")
print(f"Identity:    {resolved.identity}")

# %%
_settled = pl.DataFrame(
    {
        "field": [
            "cross_fitting_folds",
            "embargo_periods",
            "fold_unit",
            "observation_cadence",
            "placebo_method",
            "placebo_block_size",
            "placebo_block_basis",
            "label_buffer_steps",
            "treatment_window_steps",
            "placebo_draws",
            "analysis_rows",
            "analysis_timestamps",
            "analysis_key_digest",
        ],
        "value": [
            str(computation["cv"]["n_folds"]),
            str(computation["cv"]["embargo_periods"]),
            computation["cv"]["fold_unit"],
            computation["refutation"]["observation_cadence"],
            computation["refutation"]["method"],
            str(computation["refutation"]["block_size"]),
            computation["refutation"]["block_size_basis"],
            str(computation["refutation"]["label_buffer_steps"]),
            str(computation["refutation"]["treatment_window_steps"]),
            str(computation["refutation"]["n_placebo"]),
            f"{computation['analysis_population']['n_rows']:,}",
            f"{computation['analysis_population']['n_timestamps']:,}",
            computation["analysis_population"]["key_digest"],
        ],
    }
)
# Every row and every value in full. Polars elides the middle of a frame longer than ten rows -
# which is where the block size and the scale that set it fall - and truncates a value wider than
# the default, which is where the placebo method's name falls.
with pl.Config(tbl_rows=_settled.height, fmt_str_lengths=60):
    print(_settled)

# %% [markdown]
# ### What the block size has to cover
#
# A block permutation preserves serial dependence for as long as its blocks span, and there are two
# sources of that dependence here. The labels overlap: consecutive rows of a 21-day forward return
# share twenty of their twenty-one days, which is what `label_buffer_steps` measures. The treatment
# overlaps too, and by much more: `skip_recent_6_1` is `close.shift(21) / close.shift(126) - 1`, so
# the oldest price it reads is 126 sessions back and two of its values a month apart are far from
# independent.
#
# The block spans the longer of the two, and `placebo_block_basis` names which one bound it. That
# field is not decoration: at the same block size the two bases mean different things about what
# the null preserved, and a reader who has only the number cannot tell them apart.
#
# **A block long enough to span the treatment window cannot move every row.** A fund whose history
# has a gap shorter than two blocks has nowhere to permute to, so those rows are held at their
# observed values and contribute no variation to the null. That makes the placebo distribution
# narrower than it would otherwise be in a direction that is *conservative* - the p-value is biased
# toward 1, not toward 0 - and the run warns with the frozen share whenever it is nonzero. It is the
# price of the longer block, and it is the right way round: a refutation that under-rejects is a
# weaker claim, not a false one.
#
# **An earlier registered estimate for this label used the shorter of the two.** It permuted the
# treatment in blocks of 21 observations against a construction window of 126, which is close to an
# independent shuffle of a column whose values are anything but. That narrows the placebo
# distribution, so its empirical p-value read as more refutation than the evidence supported. The
# row is retired rather than corrected - the fit was valid for the block it declared, and the block
# was the wrong one - and `SUPERSEDES_CAUSAL` above names it. Section 4 reads the current
# diagnostics knowing the previous ones were measured against a null that was too tight.

# %% [markdown]
# ## 3. Estimating and registering
#
# `run()` fits the two nuisance models with cross-fitting and the embargo above, estimates the
# effect from the residuals, runs the placebo draws, and registers one row - **only after** the fit
# returns a finite effect and a finite standard error. A missing treatment, a missing confounder or
# an empty analysis population fails before anything is written.
#
# The registration goes through the resolver's own path rather than a convenience wrapper, and that
# is not a stylistic choice. A row written by the wrapper carries no `identity_version`, and
# `current_causal_identities` - the set `CausalResult.one` resolves and the ambiguity check is
# computed from - skips exactly those rows. Such a row is written, sits in the table, and answers
# nothing. This notebook's single row was one of them until this conversion.
#
# **A second run of this notebook fits nothing.** The identity is re-derived from the inputs, the
# registry already holds the matching row, and the cache answers. That is why everything reported
# below is read back from the registered row rather than from the return value of a fit: a reader
# re-running this notebook has to see the same numbers, and a cached run has no placebo draws to
# show.
#
# `SUPERSEDES_CAUSAL` names the causal identity this run retires. A label resolves to exactly one
# current identity, so a refit under changed code produces a second and the registry refuses it
# until it is told which one it replaces. Leave it empty when nothing is being retired; the error
# raised on the attempt names the hash to give it.

# %% tags=["results"]
result = resolved.run()
if not result.complete:
    raise RuntimeError(f"the registered causal result for {label} is incomplete")
# The identity-bearing computation, not the whole spec. `provenance` records the commit of the run
# that registered the row, so comparing whole specs would assert that nothing had been committed
# since - which is not a property of the estimate.
if result.spec["computation"] != resolved.spec["computation"]:
    raise RuntimeError(f"the registered causal computation for {label} differs from the resolved")
if request.resolve().run().hash != result.hash:
    raise RuntimeError("re-resolving the same request changed its identity")

metrics = result.metrics
print(f"Registered causal identity: {result.hash}")

# %% [markdown]
# ## 4. What came out
#
# Four numbers, all read back from the registered row.
#
# `naive_effect` is the coefficient on the treatment with the confounders entered linearly and
# nothing orthogonalized. `dml_effect` is the same quantity after both the outcome and the
# treatment have had the confounders' flexible predictions removed. The distance between them is
# what adjustment moved, and `confounding_bias_pct` normalizes it - by the DML estimate, so when
# that estimate is near zero the percentage is unstable and can run past the whole of it with both
# raw effects tiny. Read the raw effects.
#
# `dml_se_hac` is a Driscoll-Kraay standard error: it allows the residuals to be correlated across
# funds on the same date and correlated over time within a fund. Both matter here. Every ETF in the
# panel loads on the same market, and the overlapping 21-day label makes consecutive residuals
# dependent by construction, so an ordinary standard error would be far too small.

# %% tags=["results"]
summary = pl.DataFrame(
    {
        "quantity": [
            "observations",
            "naive effect",
            "DML effect",
            "Driscoll-Kraay SE",
            "t statistic",
            "p-value (Driscoll-Kraay)",
            "confounding bias %",
            "refutation p (block permutation)",
            "successful placebo draws",
            "refutation class",
        ],
        "value": [
            f"{metrics['n_obs']:,}",
            f"{metrics['naive_effect']:+.4f}",
            f"{metrics['dml_effect']:+.4f}",
            f"{metrics['dml_se_hac']:.4f}",
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
# ### The estimate against zero, and against the unadjusted one
#
# The interval is the conventional two-sided ninety-five percent one, taken on the Driscoll-Kraay
# standard error. The naive coefficient is marked as a point because no comparable standard error
# is registered for it - it is there to show how far adjustment moved the estimate, not to be
# tested.

# %%
effect = float(metrics["dml_effect"])
se = float(metrics["dml_se_hac"])
naive = float(metrics["naive_effect"])
low, high = effect - 1.96 * se, effect + 1.96 * se

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[low, high],
        y=["DML", "DML"],
        mode="lines",
        line=dict(color=COLORS["blue"], width=4),
        name="95% Driscoll-Kraay interval",
    )
)
fig.add_trace(
    go.Scatter(
        x=[effect],
        y=["DML"],
        mode="markers",
        marker=dict(color=COLORS["blue"], size=12),
        name=f"DML effect ({effect:+.4f})",
    )
)
fig.add_trace(
    go.Scatter(
        x=[naive],
        y=["Naive OLS"],
        mode="markers",
        marker=dict(color=COLORS["neutral"], size=12, symbol="diamond"),
        name=f"Naive OLS ({naive:+.4f})",
    )
)
fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=COLORS["negative"])
fig.update_xaxes(title_text=f"Effect of {estimand['treatment']} on {estimand['outcome']}")
fig.update_layout(
    title=(
        "The adjusted effect's interval spans zero"
        if low < 0 < high
        else "The adjusted effect's interval excludes zero"
    ),
    height=340,
    width=800,
    margin=dict(t=90),
)
show_plotly_with_alt(
    fig,
    "Horizontal interval plot of the causal effect of skip-recent momentum on the 21-day forward "
    "return, with a dashed line at zero. Counted from the registered row: DML effect "
    f"{effect:+.4f} with a 95% Driscoll-Kraay interval from {low:+.4f} to {high:+.4f}, and the "
    f"unadjusted OLS coefficient at {naive:+.4f}.",
)

# %% [markdown]
# ## 5. What to notice
#
# **The two diagnostics answer the same question under different assumptions, and both now hold
# the same dependence.** The Driscoll-Kraay standard error allows residuals to be correlated across
# funds on a date and over time within a fund; the block permutation reproduces whatever its blocks
# are long enough to span. Section 2's table names which scale bound the block, and it is the
# treatment's construction window rather than the label horizon - the longer of the two. Where an
# earlier estimate for this label used the shorter, its placebo distribution was too narrow and its
# permutation p-value too small, so the two diagnostics were not comparable and the covariance
# estimator was the only one worth quoting. They are comparable here.
#
# **A small estimate relative to its standard error says the data do not locate the effect.** Read
# the two raw effects in the table rather than the bias percentage: adjustment moves the estimate,
# and how far it moved is informative, but neither the direction nor the size means anything while
# the interval covers zero.
#
# **The causal row is not a model candidate and cannot become one.** It carries no prediction set,
# it enters no population, and [`14_backtest`](14_backtest.ipynb) never sees it.
# [`13_model_analysis`](13_model_analysis.ipynb) reports it beside the predictive families and
# explicitly outside their ranking, because "which model orders the cross-section best" and "does
# this feature move the return" are different questions and a single table answering both would be
# answering neither.
#
# **Known limitations.** The three identifying assumptions above are untestable and the refutation
# does not address them. The confounder set is declared rather than discovered, so an unobserved
# driver of both momentum and returns would sit inside the estimate with nothing here to reveal it.
# The macro confounders are finalized rather than point-in-time. And the block permutation
# preserves dependence at the scale its blocks span and no finer, so it is evidence about serial
# structure rather than about whether the confounder set is complete.

# %% [markdown]
# **Next**: [`13_model_analysis`](13_model_analysis.ipynb) puts every family's published population
# in one place, reads this causal row beside them as separate evidence, and states the rule
# [`14_backtest`](14_backtest.ipynb) selects by.
