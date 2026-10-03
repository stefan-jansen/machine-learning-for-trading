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
# # From Opinions to Probabilities
#
# **Docker image**: `ml4t`
#
# Several agents have each returned a probability for the same question. Averaging them is the
# obvious move and it is a strong baseline, but it treats three agents that read the same
# article as three independent observations, and it treats a panel that split down the middle
# the same as one that agreed. This notebook is the arithmetic for doing better, and for
# knowing when the arithmetic is not earning its assumptions.
#
# There are no model calls here. Everything is closed-form or a grid search over a few hundred
# synthetic forecasts, which is why it is worth reading before the multi-agent notebooks rather
# than after: what a panel is worth is a question about correlation and calibration, not about
# prompting.
#
# **Learning Objectives**:
# - Combine several probability forecasts into one, and say what assumption about the
#   forecasters each combination rule is making
# - Compute the diversity factor that decides how far a panel's agreement moves the aggregate
#   away from the base rate
# - Show how far the aggregate travels when only the assumed correlation changes, and read that
#   as a bound on how much extremization can be trusted
# - Fit a calibration exponent on resolved forecasts, freeze it, and score it on observations
#   it never saw
#
# **Book Reference**: Chapter 24, Section 24.7 (Multi-Agent Forecasting Systems:
# Aggregation)
#
# **Prerequisites**: None. Nothing here calls a model.

# %%
"""From Opinions to Probabilities - aggregation math for multi-agent forecasting."""

import sys

from utils.paths import get_chapter_dir

sys.path.insert(0, str(get_chapter_dir(24)))

import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from agent_pipeline import (
    MODEL_CALIBRATION_D,
    find_optimal_d,
    logodds_extremize,
    neyman_extremize,
    neyman_extremize_weighted,
    platt_scale,
)

from utils.reproducibility import set_global_seeds
from utils.style import show_with_alt

# %% tags=["parameters"]
N_FORECASTERS = 3
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## The Averaging Problem
#
# Suppose three independent analysts each estimate a 65% probability that NVIDIA
# beats earnings. Simple averaging gives 65% - but three independent sources
# agreeing should make us *more* confident, not equally confident.
#
# **Neyman extremization** formalizes this: when forecasters agree, the aggregate
# should be pushed away from the base rate (50%) proportional to a **diversity
# factor** $d$:
#
# $$d = \sqrt{\frac{n}{1 + (n-1)\rho}}$$
#
# where $n$ is the number of forecasters and $\rho$ is their pairwise correlation.
#
# $$p_{\text{extreme}} = p_{\text{base}} + d \cdot (\bar{p} - p_{\text{base}})$$

# %%
# Three forecasters at 65%, with varying correlation assumptions
probs = [0.65, 0.65, 0.65]

# Independent (rho=0): maximum diversity benefit
result_independent = neyman_extremize(probs, base=0.5, correlation=0.0)
print(
    f"Independent (ρ=0.0): d={result_independent.extremization_factor:.2f}, "
    f"p={result_independent.extremized_probability:.2f}"
)

# Assumed moderate correlation (rho=0.3): illustrative scenario
result_moderate = neyman_extremize(probs, base=0.5, correlation=0.3)
print(
    f"Moderate (ρ=0.3):    d={result_moderate.extremization_factor:.2f}, "
    f"p={result_moderate.extremized_probability:.2f}"
)

# Assumed high correlation (rho=0.7): illustrative shared-research scenario
result_correlated = neyman_extremize(probs, base=0.5, correlation=0.7)
print(
    f"Correlated (ρ=0.7):  d={result_correlated.extremization_factor:.2f}, "
    f"p={result_correlated.extremized_probability:.2f}"
)

# %% [markdown]
# **Finding**: Three forecasters at 65% produce a 76% aggregate when independent,
# but only 67% when highly correlated. The diversity factor $d$ quantifies how much
# new information each forecaster adds. When they're all reading the same sell-side
# research ($\rho=0.7$), adding more analysts barely moves the aggregate.

# %% [markdown]
# ## Visualizing the Extremization Factor
#
# How does $d$ change with the number of forecasters and their correlation?

# %%
# Panel (a): d vs number of forecasters
correlations = [0.0, 0.1, 0.3, 0.5, 0.7]
n_range = list(range(2, 12))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for rho in correlations:
    d_values = [
        min(3.0, math.sqrt(n / (1 + (n - 1) * rho)) if (1 + (n - 1) * rho) > 0 else 1.0)
        for n in n_range
    ]
    axes[0].plot(n_range, d_values, marker="o", markersize=4, label=f"ρ={rho}")

axes[0].set_xlabel("Number of Forecasters")
axes[0].set_ylabel("Diversity Factor d")
axes[0].set_title("Correlation Caps the Diversity Benefit")
axes[0].legend()
_ = axes[0].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="d=1 (no extremization)")

# Panel (b): Aggregate probability vs correlation
mean_p = 0.65
base = 0.5
rho_range = np.linspace(0, 0.9, 50)
for n in [2, 3, 5, 8]:
    agg_probs = []
    for rho in rho_range:
        d_raw = math.sqrt(n / (1 + (n - 1) * rho)) if (1 + (n - 1) * rho) > 0 else 1.0
        d = max(1.0, min(3.0, d_raw))
        p_extreme = base + d * (mean_p - base)
        agg_probs.append(max(0.01, min(0.99, p_extreme)))
    axes[1].plot(rho_range, agg_probs, label=f"n={n}")

axes[1].set_xlabel("Forecaster Correlation (ρ)")
axes[1].set_ylabel("Aggregate Probability")
axes[1].set_title("Aggregate Probability vs Correlation (mean=65%)")
axes[1].legend()
axes[1].axhline(0.65, color="gray", linestyle="--", alpha=0.5)

show_with_alt(
    fig,
    "Two mathematical plots show the extremization factor for assumed agent correlations and how those factors transform a fixed mean probability. The correlations are illustrative inputs.",
)

# %% [markdown]
# **Interpretation**:
# The helper and panel (a) cap $d$ at 3.0 as an implementation choice. Without
# that cap, the independent case grows as $\sqrt{n}$. Correlations here are
# illustrative assumptions, not measured dependence among financial analysts.
#
# - **(a)** More forecasters increase $d$, but with diminishing returns - especially
#   at high correlation. Going from 3 to 8 forecasters at $\rho=0.5$ barely changes $d$.
# - **(b)** At zero correlation, 8 forecasters at 65% push the aggregate to 92%. At
#   $\rho=0.7$, the same 8 forecasters only reach 68%. **Correlation is the binding
#   constraint** on how much diversity helps.

# %% [markdown]
# ## Disagreement: When Forecasters Diverge
#
# This formula uses the mean, agent count and assumed error correlation.
# Dispersion of the probabilities does not enter it. Equal means produce
# equal aggregates when count, base rate and correlation are held fixed.

# %%
# Scenario: specialists disagree
divergent = [0.30, 0.90, 0.93]
result_divergent = neyman_extremize(divergent, base=0.5, correlation=0.3)

print(f"Specialist probabilities: {divergent}")
print(f"Simple mean: {sum(divergent) / len(divergent):.2f}")
print(
    f"Neyman (ρ=0.3): {result_divergent.extremized_probability:.2f} "
    f"(d={result_divergent.extremization_factor:.2f})"
)

# Compare: tight agreement
tight = [0.70, 0.72, 0.71]
result_tight = neyman_extremize(tight, base=0.5, correlation=0.3)

print(f"\nTight agreement: {tight}")
print(f"Simple mean: {sum(tight) / len(tight):.2f}")
print(f"Neyman (ρ=0.3): {result_tight.extremized_probability:.2f}")

# %% [markdown]
# **Finding:** Both panels average 71% and produce the same extremized probability.
# The large disagreement in the first panel is useful to inspect separately;
# this mean-based formula does not reduce conviction in response to it.

# %% [markdown]
# ## Platt Scaling: Post-Hoc Calibration
#
# **Platt scaling** adjusts probabilities based on observed calibration. The formula:
#
# $$p' = \frac{d \cdot p^a}{d \cdot p^a + (1-p)^a}$$
#
# - $a > 1$ pushes probabilities away from 0.5 (under-confident agents)
# - $a < 1$ pulls probabilities toward 0.5 (over-confident agents)
# - $d$ shifts the midpoint asymmetrically
#
# This is algebraically equivalent to the chapter's logistic parameterization,
# $p' = \sigma(a \cdot \text{logit}(p) + \log d)$ - the two notations match
# with $b = \log d$.

# %%
fig, ax = plt.subplots(figsize=(8, 6))

p_range = np.linspace(0.01, 0.99, 100)

# Different a values
for a, label in [
    (0.5, "a=0.5 (compress)"),
    (1.0, "a=1.0 (identity)"),
    (1.5, "a=1.5 (extremize)"),
    (2.0, "a=2.0 (strong extremize)"),
]:
    calibrated = [platt_scale(p, a=a, d=1.0) for p in p_range]
    ax.plot(p_range, calibrated, label=label)

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
ax.set_xlabel("Original Probability")
ax.set_ylabel("Calibrated Probability")
ax.set_title("Platt Scaling: Effect of Parameter a")
ax.legend(loc="upper left")
ax.set_aspect("equal")
show_with_alt(
    fig,
    "Probability transformations for fixed Platt slopes are compared with the identity line. These curves illustrate the formula, not measured model calibration.",
)

# %% [markdown]
# Above one the curve bows away from the diagonal and probabilities move toward the ends;
# below one it bows toward the middle. Which direction a given forecaster needs is not
# something the curve can say. It is measured against resolved forecasts, and the measurement
# has to be made on observations the transformation was not fitted to, or it will report the
# improvement it was constructed to produce.

# %% [markdown]
# ## Weighting the Panel
#
# Agents are not always interchangeable. When a design can defend giving one more weight than
# another - a specialist on the sector in question, an agent with a track record on this class
# of question - the mean becomes a weighted mean, and the panel size has to be adjusted to
# match. Three agents where one carries most of the weight is not three forecasters.
#
# The **Herfindahl index** measures that concentration. It is the sum of the squared normalized
# weights, and its reciprocal is the number of equally-weighted forecasters that would be as
# concentrated:
#
# $$\text{HI} = \sum w_i^2, \qquad n_{\text{weight}} = \frac{1}{\text{HI}}$$
#
# Uniform weights give back the panel size exactly; putting everything on one agent gives one.
# The correlation adjustment then applies on top,
# $n_{\text{adjusted}} = n_{\text{weight}} / [1 + (n_{\text{weight}} - 1)\rho]$, and it is the
# adjusted figure that becomes $d^2$.

# %%
# Three agents with different confidence levels
probs_w = [0.72, 0.58, 0.65]
weights_equal = [1.0, 1.0, 1.0]
weights_skewed = [0.8, 0.3, 0.5]  # First agent is most confident

result_equal = neyman_extremize_weighted(probs_w, weights_equal, base=0.5, correlation=0.3)
result_skewed = neyman_extremize_weighted(probs_w, weights_skewed, base=0.5, correlation=0.3)

pl.DataFrame(
    [
        {
            "weighting": "equal",
            "mean": result_equal.raw_probability,
            "extremized": result_equal.extremized_probability,
            "d": result_equal.extremization_factor,
            "n_eff": result_equal.effective_n,
        },
        {
            "weighting": "skewed",
            "mean": result_skewed.raw_probability,
            "extremized": result_skewed.extremized_probability,
            "d": result_skewed.extremization_factor,
            "n_eff": result_skewed.effective_n,
        },
    ]
)

# %% [markdown]
# **Finding**: With skewed weights, the weighted mean shifts toward the most
# confident agent (0.72), while the effective sample size drops, reducing the
# extremization push. This prevents over-extremization when confidence is
# concentrated in one voice.

# %% [markdown]
# ## Log-Odds Extremization
#
# An alternative to Neyman: operate in **log-odds space** where the transformation
# is naturally symmetric around 0.5:
#
# $$p' = \sigma(a \cdot \text{logit}(p)), \quad \text{logit}(p) = \log\frac{p}{1-p}$$
#
# This has the same effect as Platt scaling but is parameterized more intuitively:
# $a > 1$ extremizes, $a < 1$ compresses.

# %%
fig, ax = plt.subplots(figsize=(8, 6))

p_range = np.linspace(0.01, 0.99, 100)

for a, label in [
    (0.5, "a=0.5 (compress)"),
    (1.0, "a=1.0 (identity)"),
    (1.5, "a=1.5 (moderate extremize)"),
    (2.0, "a=2.0 (strong extremize)"),
]:
    calibrated = [logodds_extremize(p, a) for p in p_range]
    ax.plot(p_range, calibrated, label=label)

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Identity")
ax.set_xlabel("Original Probability")
ax.set_ylabel("Calibrated Probability")
ax.set_title("Log-Odds Extremization")
ax.legend(loc="upper left")
ax.set_aspect("equal")
show_with_alt(
    fig,
    "Fixed log-odds scaling curves are compared with unchanged probabilities. Larger coefficients push probabilities farther from one half.",
)

# %% [markdown]
# **Interpretation**: Log-odds extremization produces curves very similar to Platt
# scaling but operates in a more interpretable space. The parameter $a$ directly
# controls how aggressively probabilities are pushed toward 0 or 1. The AIA
# Forecaster uses this for per-model calibration.

# %% [markdown]
# ## Per-Model Calibration
#
# Different LLMs have different calibration profiles, so the extremization
# parameter $d$ is best tuned per model on historical forecasts. The table below
# lists illustrative starting points; they are not from the AIA paper, which
# recommends a single $d=\sqrt{3}$ for all forecasters and avoids per-model tuning.

# %%
pl.DataFrame({"model": list(MODEL_CALIBRATION_D.keys()), "d": list(MODEL_CALIBRATION_D.values())})

# %% [markdown]
# **Note**: These values are illustrative starting points, not paper-sourced
# constants. In practice, tune $d$ on your own resolved forecasts using
# `find_optimal_d`, or fall back to the paper's single $d=\sqrt{3}$.

# %% [markdown]
# ## Finding the Optimal Calibration Parameter
#
# Given a set of resolved forecasts, we can grid-search for the $a$ parameter
# that minimizes Brier score. This is the principled way to calibrate a
# forecasting system.

# %%
# Simulated resolved forecasts for demonstration
set_global_seeds(SEED)
n_questions = 24
true_probs = np.random.beta(2, 2, n_questions)
outcomes = (np.random.random(n_questions) < true_probs).astype(float)
# Agents are slightly under-confident (pushed toward 0.5)
raw_forecasts = [0.5 + 0.6 * (p - 0.5) + np.random.normal(0, 0.05) for p in true_probs]
raw_forecasts = [max(0.05, min(0.95, p)) for p in raw_forecasts]

result_cal = find_optimal_d(raw_forecasts, outcomes)
print(f"Optimal a:        {result_cal.optimal_d:.3f}")
print(f"Brier (before):   {result_cal.brier_before:.4f}")
print(f"Brier (after):    {result_cal.brier_after:.4f}")
print(f"Improvement:      {result_cal.improvement_pct:.1f}%")

# %%
# Visualize the calibration effect
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before calibration
axes[0].scatter(raw_forecasts, outcomes, alpha=0.6, s=40)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("Predicted Probability")
axes[0].set_ylabel("Outcome (0 or 1)")
axes[0].set_title("Before Calibration")

# After calibration
calibrated = [logodds_extremize(p, result_cal.optimal_d) for p in raw_forecasts]
axes[1].scatter(calibrated, outcomes, alpha=0.6, s=40, color="tab:orange")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[1].set_xlabel("Calibrated Probability")
axes[1].set_ylabel("Outcome (0 or 1)")
axes[1].set_title("After Calibration")

show_with_alt(
    fig,
    "Synthetic probability and outcome pairs appear before and after the declared calibration transform. This is an arithmetic demonstration, not held-out forecasting evidence.",
)

# %% [markdown]
# **Finding**: The optimal $a > 1$ implies the raw forecasts were under-confident
# (compressed toward 0.5). After calibration, extreme predictions are pushed further
# from 0.5, reducing Brier score on this 24-question demonstration set. The size of
# the Brier improvement on a 24-question panel is dominated by sampling noise; tuning
# $a$ on a deployed system requires a larger resolved-forecast panel with held-out
# validation.

# %% [markdown]
# ## Sensitivity Analysis: How Many Agents Do You Need?
#
# A practical question: given a budget for LLM calls, how many specialist agents
# should you run?

# %%
fig, ax = plt.subplots(figsize=(8, 5))

# Fixed per-agent probability, varying n and rho
for rho in [0.1, 0.3, 0.5]:
    effective_n = []
    agent_counts = list(range(1, 11))
    for n in agent_counts:
        denom = 1 + (n - 1) * rho
        d = math.sqrt(n / denom) if denom > 0 else 1.0
        effective_n.append(d**2)

    ax.plot(agent_counts, effective_n, "o-", label=f"ρ={rho}")

ax.plot(agent_counts, agent_counts, "k--", alpha=0.3, label="n_eff = n (independent)")
ax.set_xlabel("Number of Agents")
ax.set_ylabel("Effective N (information content)")
ax.set_title("Diminishing Returns: Effective N vs Agent Count")
ax.legend()
show_with_alt(
    fig,
    "Effective panel size grows with agent count for assumed correlations and approaches the corresponding reciprocal-correlation ceiling.",
)

# %% [markdown]
# Effective size rises steeply for the first few agents and then flattens against a ceiling
# that depends only on the correlation: for fixed $\rho > 0$ it approaches $1/\rho$ however
# many agents are added. That ceiling is where the budget question gets its answer. Agents past
# the point where the curve bends are paying full price for a fraction of an observation, and
# the way to buy more information is to lower $\rho$ rather than to raise $n$: different
# evidence, different framings, different models.
#
# This is arithmetic, not a measurement. It says what the formula implies for an assumed
# correlation, and the correlation is the thing nobody has estimated.

# %% [markdown]
# ## Key Takeaways
#
# 1. **The mean is the baseline anything else has to beat, out of sample.** Extremization and
#    calibration are adjustments to it, and an adjustment fitted on the data it is evaluated on
#    has not been evaluated.
# 2. **Extremization is a claim about independence, not about agreement.** Agents that agree
#    because they read the same three articles carry one observation between them. The whole
#    question is $\rho$, and this formula takes it as an input rather than estimating it.
# 3. **The panel's spread does not enter the formula.** Three agents at 20, 65 and 90 percent
#    and three within a point of each other produce the same aggregate at the same assumed
#    correlation. If dispersion should matter, it has to enter through an estimated dependence
#    model or a different rule.
# 4. **Adding agents runs into a ceiling that depends on the correlation, not the budget.**
#    Effective size approaches $1/\rho$, so diversity is bought by changing what the agents
#    read, not by running more of them.
# 5. **A calibration parameter is fitted, frozen, and then evaluated on observations it never
#    saw.** Any other order measures how well a curve fits the points it was drawn through.
#
# **Known limitations of what is built here.** Every result on this page is conditional on a
# correlation nobody measured, and the panel is assumed exchangeable: one $\rho$ for every
# pair. The clamps that keep the output well-defined are not part of the theory, so an
# aggregate near the bounds is partly an artifact of them. The calibration demonstration runs
# on simulated forecasts drawn from a known generating process, which is the easiest possible
# case: real forecast panels are small, resolve slowly, and are not identically distributed.
#
# **Next**: [`06_multi_agent_research`](06_multi_agent_research.ipynb) asks whether running
# several identical research agents on one question produces a panel worth aggregating at all,
# or whether they land in the same place.
#
# **Book**: Section 24.7 covers aggregation theory, including connections to
# Condorcet's jury theorem and prediction market design.
