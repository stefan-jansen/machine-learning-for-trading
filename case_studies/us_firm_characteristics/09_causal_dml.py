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
# # Causal DML for US Firm Characteristics
#
# This notebook asks whether 12-to-2-month momentum predicts next-month returns after conditioning
# on four observed risk characteristics. It uses expanding-window double machine learning (DML),
# keeps every firm at a decision month on the same side of each fold boundary, and leaves the final
# 2016 holdout out of the analysis entirely. What the estimate and the two uncertainty checks come
# to is read off the row the run registers.
#
# **Learning objectives**
#
# - Residualize momentum and returns with nuisance models fit on earlier decision months.
# - Compare the adjusted effect with naive OLS on the exact same out-of-fold sample.
# - Use panel-robust inference and an entity-aware block-permutation refutation.
# - Distinguish a conditional causal estimate from proof that all confounding is observed.
#
# **Book reference:** Chapter 15, cross-dataset causal evidence.
#
# **Prerequisites:** `03_financial_features` and `04_evaluation`.

# %%
"""Panel-aware causal DML that leaves the final holdout out of the analysis."""

import warnings

import matplotlib.pyplot as plt
import polars as pl

from case_studies.research import causal_supersedes, open_study, primary_label
from utils.modeling import load_configs
from utils.style import COLORS, add_message_title

# `matplotlibrc` turns constrained layout on for every figure in this repo, and
# `add_message_title` reserves the title pad through `tight_layout`, so the two together
# warn on every figure. The warning is raised from the kernel's own scratch module, which
# puts a `/tmp/ipykernel_<pid>/` path into the committed cell output; the sanitizer cannot
# rewrite it and refuses to stamp the notebook. Filtered by message rather than by a bare
# `filterwarnings("ignore")`, so a warning about the analysis still reaches the reader.
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
PREVIEW_REDUCTIONS: dict = {}
LABEL = ""
CONFIG_NAME = ""
# The causal identity this run retires, as a bare hash. `_causal_source_identity` hashes
# `case_studies/utils/causal.py` whole, so any edit to that file gives the same fit a new
# identity, and a second current identity for one label makes `CausalResult.one`
# unresolvable. Registration refuses the write rather than leaving the ambiguity for a
# downstream notebook to hit hours later, so a refit has to say here which identity it
# replaces. Empty means the fit must leave exactly one current identity on its own.
# Papermill passes parameters through as strings, which is why this is a str and not a
# mapping.
SUPERSEDES_CAUSAL: str = ""

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# %% [markdown]
# ## 1. The request
#
# A causal request names three things: the method, the label whose return is the outcome, and the
# configuration that carries the DML budget. Everything else the analysis needs comes from the
# case study's own declarations and is resolved below rather than chosen here.
#
# The label is the one the strategy chapters trade, and the configuration is the first the case
# study declares for it. **Its `max_samples` does not apply here.** A canonical request resolves
# the full declared population, and a row cap reaches the resolver only through
# `preview_reductions`, which a canonical request may not carry; the resolved spec records
# `max_samples: 0` accordingly. The preset still declares one because six case-study DML stages
# have not migrated to this path and read it as their own default. What that changes for this
# notebook is the size of the analysis: the cap used to be spent on the most recent whole decision
# months that fit under it, so the estimate was computed on the last two years of the panel rather
# than on all of it.

# %%
label = LABEL or primary_label(study)
config_name = CONFIG_NAME or str(load_configs(CASE_STUDY_ID, label, "causal_dml")[0]["config_name"])
request = study.causal(
    method="dml",
    label=label,
    config_name=config_name,
    execution_tier=EXECUTION_TIER,
    preview_reductions=PREVIEW_REDUCTIONS,
    # The declared hash is only meaningful where this registry already holds a current
    # identity for the label. A reader's clone holds no causal rows at all, and naming a
    # predecessor that does not exist fails at registration, after the fit and every placebo
    # refit have been paid for. The resolution lives in shared code so no notebook branches
    # on the tier.
    supersedes=causal_supersedes(
        study,
        SUPERSEDES_CAUSAL,
        label,
        labels=[label],
        execution_tier=EXECUTION_TIER,
    ),
)
resolved = request.resolve()
print(f"{request.method} | label {label} | configuration {config_name}")
print(f"Causal identity: {resolved.identity}")

# %% [markdown]
# ## 2. What the resolver decided
#
# Resolving reads the panel, seals it against the holdout, and fixes every quantity the estimate
# depends on before a single model is fit. Two of those quantities are the sample: `n_rows` and
# `n_timestamps` describe the analysis frame after the seal, and the seal is per firm, so a firm
# that does not hold more observations than the buffer contributes nothing rather than dragging
# the boundary back for everyone.
#
# `holdout_endpoint_cutoff` is the loosest of the per-firm cutoffs, which is the honest scalar to
# record when each firm is sealed against its own: no firm retains a row at or after its own
# cutoff, and none of those cutoffs is later than this one.

# %%
spec = resolved.spec
estimand = spec["computation"]["estimand"]
population = spec["computation"]["analysis_population"]
cv = spec["computation"]["cv"]
design = pl.DataFrame(
    {
        "outcome": [estimand["outcome"]],
        "treatment": [estimand["treatment"]],
        "confounders": [", ".join(estimand["confounders"])],
        "n_rows": [population["n_rows"]],
        "n_timestamps": [population["n_timestamps"]],
        "folds": [cv["n_folds"]],
        "embargo_periods": [cv["embargo_periods"]],
        "holdout_endpoint_cutoff": [estimand["holdout_endpoint_cutoff"][:10]],
    }
)
design

# %% [markdown]
# ### The embargo, the permutation block and the outcome horizon answer different questions
#
# The **embargo** separates a fold's training months from its test months, so a label still running
# at the end of training cannot also be scored in test. The **permutation block** is the scale of
# serial dependence the placebo has to leave intact: shuffling in blocks shorter than that scale
# pulls dependent observations apart, and the placebo degrades towards an independent draw, which
# is the permutation that is too easy to pass and therefore establishes nothing. The **outcome
# horizon** is how far past its own timestamp a label resolves, and it sets the floor on the
# Newey-West bandwidth, because that is the span over which consecutive outcomes overlap.
#
# The three are separate numbers here and the table below prints all of them, counted in the
# panel's own observations. `labels.buffer` is `1M`, one observation on a monthly panel.
# `labels.horizons` declares `0D` for this label: the release pairs characteristics observed at
# the close of one month with the return earned over the next and dates the row by the month the
# return was earned, so the outcome is already realised on the timestamp its row carries and no
# two labels overlap at all. `label_horizon_steps` reads 1 rather than 0 because the step count
# floors at one observation - a span shorter than a bar still covers the bar it resolves inside -
# so read that column as the floor and the `0D` declaration as the statement about overlap. With
# no overlap the bandwidth falls to the sample-size rule.
#
# The block is neither of those. The treatment is a cumulative return from twelve months back to
# two months back, so it is autocorrelated over its whole construction window whatever the label
# does, and `block_size_basis` records which of the two quantities the block was taken from.
# `causal.treatment_window` in `config/setup.yaml` declares that window rather than inferring it,
# because guessing which element of a window list a column was built from puts a wrong number
# behind a right-looking one.
#
# **An earlier version of this notebook sized the block from the label buffer**, which is one
# month here, and permuted a twelve-month momentum measure in one-month blocks. That is close
# enough to an independent shuffle of momentum against time that the p-value it produced does not
# mean what it reads as, and the refutation it registered was weaker than it looked. That version
# also registered through a wrapper that wrote no `identity_version`, which
# `current_causal_identities` requires, so its rows resolve for nobody. Both are fixed by
# resolving the request here rather than assembling the call by hand: the block and the identity
# are decided in shared code, once, for every case study that runs DML.
#
# The observed cadence is recorded as well, which is what lets a hole in a firm's history end a
# block rather than be permuted across.

# %%
refutation_design = pl.DataFrame(
    {k: [v] for k, v in spec["computation"]["refutation"].items() if not isinstance(v, dict)}
)
refutation_design.select(
    "block_size",
    "block_size_basis",
    "label_buffer_steps",
    "label_horizon_steps",
    "treatment_window_steps",
    "n_placebo",
    "observation_cadence",
)

# %% [markdown]
# ## 3. Fit, refute and register
#
# Each fold trains on earlier months and scores later months after the embargo. Driscoll-Kraay
# inference aggregates the second-stage score by month, allowing general dependence across firms at
# the same decision time. The naive comparison uses exactly the rows with out-of-fold DML
# residuals. The placebo then permutes each firm's treatment history in contiguous blocks and
# refits, which is the expensive part of the call.
#
# Causal DML has its own registry table because it estimates a treatment effect rather than a
# cross-sectional prediction score. Everything that would change the estimate is in the identity
# printed above: the fold and embargo geometry, the placebo design and its seed, the outcome
# horizon, the configuration the case study declares, and the sealed sample. Two runs that differ
# in any of them are two identities rather than one row overwriting another. Re-running this
# notebook against a registry that already holds the identity serves the row from the registry
# instead of refitting.

# %%
result = resolved.run()
metrics = result.metrics
if not result.complete:
    # A run whose placebo refits mostly failed registers a null p-value, and every number
    # below would then be formatted from a null. The stop matters more than the
    # formatting: the row is already in the registry, so the next run reads it from the
    # cache and nothing recomputes the refutation that went missing.
    raise ValueError(
        f"causal identity {result.hash} registered an incomplete result - "
        f"n_obs={metrics.get('n_obs')}, effect={metrics.get('dml_effect')}, "
        f"refutation_p={metrics.get('refutation_p')}. Delete the row and re-run rather "
        "than reading it."
    )
# The draw count is a separate question from completeness: it arrived with a migration,
# so a row written before that column existed carries None there whatever its run did.
# Completeness cannot require it without condemning those rows, which leaves the two
# numbers derived from it - the p-value's floor and the verdict - to be reported as
# unavailable here rather than computed from a null.
draws = metrics.get("refutation_n_successful")
placebo_p_floor = 1.0 / (draws + 1) if draws is not None else None
# A null p-value reaches this point as a COMPLETE row, so the stop above does not catch it: a
# configuration that asks for no placebos is complete without one by contract
# (research/causal.py, `requested == 0`). Formatting it would raise from inside the figure's
# subtitle, after the row was registered and after the axes were drawn - the same trap the draw
# count on the line above already avoids, which is why it is avoided the same way.
refutation_p = metrics["refutation_p"]
refutation_clause = (
    "no permutation test was run"
    if refutation_p is None
    else f"permutation p {refutation_p:.3f} over "
    + (f"{draws} successful refits" if draws is not None else "an unrecorded draw count")
)
print(f"Registered causal identity: {result.hash} ({result.execution_tier})")

# %% [markdown]
# ## 4. Adjusted and naive estimates use the same sample
#
# A difference between the two estimates measures sensitivity to the observed confounder set. It
# does not establish that conditional ignorability, overlap, or no-interference assumptions hold.

# %%
effect_table = pl.DataFrame(
    {
        "estimate": ["Naive OLS", "DML adjusted"],
        "monthly_effect": [metrics["naive_effect"], metrics["dml_effect"]],
        "basis_points": [metrics["naive_effect"] * 10_000, metrics["dml_effect"] * 10_000],
        "observations": [metrics["n_obs"], metrics["n_obs"]],
    }
)
effect_table

# %% [markdown]
# ## 5. Read the estimate against its own uncertainty
#
# The band below is the DML coefficient plus and minus 1.96 Driscoll-Kraay standard errors, which
# is a normal-approximation reading of that standard error and **not** the interval implied by the
# registered `p_hac`: that p-value comes from a Student-t on the second-stage period count, which
# the registry does not carry, so an exact inversion is not available to a reader of the row. The
# two agree closely at this many periods and are not the same statement, and the band is drawn
# from the quantity that is recorded. The naive estimate is a point on the same axis because it is
# computed on the same rows, so the distance between them is the adjustment rather than a
# difference in sample. Both are monthly effects in basis points.
#
# Every number in this figure is read back from the registered row, so it redraws identically
# whether this run fitted the model or the registry served it. The placebo draws themselves are
# not persisted - the row records the empirical p-value and how many refits succeeded - so the
# refutation is reported as those two numbers rather than as a histogram that a re-run could not
# reproduce.

# %%
effect_bps = metrics["dml_effect"] * 10_000
half_width = 1.96 * metrics["dml_se_hac"] * 10_000
lo, hi = effect_bps - half_width, effect_bps + half_width
separable = lo > 0 or hi < 0

fig, ax = plt.subplots(figsize=(9, 3))
ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
ax.hlines(0, lo, hi, color=COLORS["blue"], linewidth=3)
ax.scatter([effect_bps], [0], color=COLORS["blue"], zorder=3, label="DML adjusted")
ax.scatter(
    [metrics["naive_effect"] * 10_000],
    [0],
    color=COLORS["copper"],
    marker="D",
    zorder=3,
    label="Naive OLS",
)
ax.set(xlabel="Monthly treatment effect (basis points)", yticks=[])
ax.legend(frameon=False, loc="upper left")
add_message_title(
    ax,
    (
        "Adjusting moves the estimate, and its own standard error separates it from zero"
        if separable
        else "Adjusting moves the estimate, but its own standard error does not separate it from zero"
    ),
    subtitle=(
        f"effect +/-1.96 Driscoll-Kraay standard errors, [{lo:+.1f}, {hi:+.1f}] bps; "
        + refutation_clause
    ),
)
fig.tight_layout(rect=(0, 0, 1, 0.82))
fig.show()

# %% [markdown]
# ## What the run produced
#
# One cell, so every number this notebook publishes is in one place and moves together on a re-run.
# `placebo_p_floor` is one over the number of successful placebo draws plus one, the smallest
# p-value a permutation test of this size can report. `refutation_class` is derived from the
# p-value and that draw count together, so a run whose draws could not have rejected at all is
# not reported as having failed to reject.

# %% tags=["results"]
pl.DataFrame(
    {
        "causal_hash": [result.hash],
        "observations": [metrics["n_obs"]],
        "dml_bps_per_month": [metrics["dml_effect"] * 10_000],
        "naive_bps_per_month": [metrics["naive_effect"] * 10_000],
        "se_hac_bps": [metrics["dml_se_hac"] * 10_000],
        "p_hac": [metrics["p_value_hac"]],
        "confounding_bias_pct": [metrics["confounding_bias_pct"]],
        "placebo_p": [metrics["refutation_p"]],
        "placebo_p_floor": [placebo_p_floor],
        "placebo_block_months": [spec["computation"]["refutation"]["block_size"]],
        "refutation_class": [metrics["refutation_class"]],
    }
)

# %% [markdown]
# ## Takeaways
#
# - This is a development-stage robustness diagnostic for the momentum signal. Prediction and
#   strategy selection are downstream tasks and read different evidence.
# - Read the coefficient against its Driscoll-Kraay standard error rather than against zero: that
#   standard error allows for dependence along time and across firms at the same decision month,
#   which an IID standard error on a panel this wide does not.
# - Naive OLS and the adjusted estimate are computed on the same out-of-fold rows, so the gap
#   between them measures sensitivity to the observed confounder set rather than two estimators
#   reading different numbers of observations.
# - The permutation p-value cannot fall below its floor however extreme the estimate is, so a value
#   at the floor is a bound and not a measurement. A block permutation disturbs the timing of the
#   treatment; it does not disturb confounding.
# - The block spans the treatment's own construction window, so what the placebo holds fixed is the
#   dependence a twelve-to-two-month momentum measure carries, and the p-value is readable as a
#   statement about timing. That changes what the number means rather than guaranteeing the number
#   moves: a permutation test can report the same value for a sound reason and an unsound one, and
#   it is the block, not the p-value, that says which of the two is on offer.
# - Read the refutation against the coefficient's own uncertainty rather than on its own. Where
#   the estimate is not separable from zero by its Driscoll-Kraay standard error, a placebo test
#   of that estimate could not have said much whichever way it came out.
# - Conditioning on Beta, IdioVol, LME and Variance does not establish that conditional
#   ignorability, overlap or no-interference hold. Every step above ends before the 2016 holdout.
