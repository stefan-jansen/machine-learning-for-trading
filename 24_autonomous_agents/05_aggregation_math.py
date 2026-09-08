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
"""From Opinions to Probabilities: aggregation math for multi-agent forecasting."""

import math

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from agent_pipeline import (
    brier_score,
    fit_extremization_exponent,
    logodds_extremize,
    neyman_extremize,
    neyman_extremize_weighted,
    platt_scale,
)

from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, format_pct_axis, ml4t_palette

# %% [markdown]
# ## Settings
#
# `SEED` fixes the simulated forecast panel used in the calibration section, so the fitted
# exponent and the held-out scores are the same on every machine. Nothing else in the notebook
# is random.

# %% tags=["parameters"]
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## The Averaging Problem
#
# Three analysts each put the probability of the same event at the same level, comfortably
# above even odds. The mean of three identical numbers is that number, so simple averaging
# returns exactly what any one of them said and the panel has bought nothing.
#
# That is the right answer if the three are the same analyst three times over: three people who
# read the same wire story and reached the same conclusion are one observation. It is the wrong
# answer if they worked independently, because three independent routes to the same place is
# stronger evidence than one, and the aggregate should sit further from the base rate than any
# individual estimate.
#
# **Neyman extremization** makes that adjustment explicit. It pushes the mean away from the
# base rate by a **diversity factor** $d$:
#
# $$d = \sqrt{\frac{n}{1 + (n-1)\rho}}, \qquad
# p_{\text{extreme}} = p_{\text{base}} + d \cdot (\bar{p} - p_{\text{base}})$$
#
# where $n$ is the number of forecasters and $\rho$ is the correlation assumed between any two
# of them. With $\rho$ at zero, $d$ is $\sqrt{n}$ and the panel counts fully; as $\rho$ rises
# toward one, $d$ falls toward one and the panel counts as a single forecaster. The
# implementation clamps $d$ to lie between one and three, and clamps the result just inside the
# unit interval, so an extreme assumption cannot invert the adjustment or return a certainty.

# %%
probs = [0.65, 0.65, 0.65]
assumptions = [
    ("independent", 0.0),
    ("moderately correlated", 0.3),
    ("heavily correlated", 0.7),
]
correlation_results = [
    (label, rho, neyman_extremize(probs, base=0.5, correlation=rho)) for label, rho in assumptions
]

pl.DataFrame(
    {
        "forecasters assumed": [label for label, _, _ in correlation_results],
        "rho": [rho for _, rho, _ in correlation_results],
        "diversity factor d": [r.extremization_factor for _, _, r in correlation_results],
        "aggregate": [r.extremized_probability for _, _, r in correlation_results],
    }
)

# %% [markdown]
# The three correlation assumptions are the same three forecasts read three ways. Treated as
# independent, they carry three observations and the aggregate moves furthest from the base
# rate; treated as heavily correlated, they carry barely more than one and the aggregate stays
# close to what any single analyst said. Nothing about the forecasts changed between the three
# lines. The whole difference is an assumption, and it is the assumption a real deployment has
# to earn rather than choose.

# %% [markdown]
# ## How the Diversity Factor Behaves
#
# Two questions decide whether extremization is worth having. How much does $d$ grow as agents
# are added, and how much does the aggregate move when the correlation assumption changes?

# %%
correlations = [0.0, 0.1, 0.3, 0.5, 0.7]
n_range = list(range(2, 12))
curve_colors = ml4t_palette(len(correlations), categorical=True)
line_styles = ["-", "--", ":", "-.", "-"]
diversity_series = [
    (
        rho,
        [min(3.0, math.sqrt(n / (1 + (n - 1) * rho))) for n in n_range],
        color,
        line_style,
    )
    for rho, color, line_style in zip(correlations, curve_colors, line_styles, strict=True)
]

# %% [markdown]
# The second panel holds the mean forecast and the base rate fixed and sweeps the correlation,
# so the only thing moving is the assumption.

# %%
mean_p = 0.65
base = 0.5
rho_range = np.linspace(0, 0.9, 50)
aggregate_series = []
for n, color, line_style in zip(
    [2, 3, 5, 8],
    ml4t_palette(4, categorical=True),
    line_styles[:4],
    strict=True,
):
    agg_probs = []
    for rho in rho_range:
        d_raw = math.sqrt(n / (1 + (n - 1) * rho)) if (1 + (n - 1) * rho) > 0 else 1.0
        d = max(1.0, min(3.0, d_raw))
        p_extreme = base + d * (mean_p - base)
        agg_probs.append(max(0.01, min(0.99, p_extreme)))
    aggregate_series.append((n, agg_probs, color, line_style))

# %% [markdown]
# Both panels are drawn together because they answer two halves of one question: what the
# diversity factor does, and what it does to an aggregate.

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])
for rho, values, color, line_style in diversity_series:
    axes[0].plot(
        n_range,
        values,
        marker="o",
        markersize=3,
        color=color,
        linestyle=line_style,
        label=f"ρ={rho}",
    )
axes[0].set(xlabel="Number of forecasters", ylabel="Diversity factor d")
add_message_title(axes[0], "Correlation, not panel size, sets the ceiling on d")
axes[0].legend()
axes[0].axhline(1.0, color=COLORS["neutral"], linestyle="--", alpha=0.6)

for n, values, color, line_style in aggregate_series:
    axes[1].plot(rho_range, values, color=color, linestyle=line_style, label=f"n={n}")
axes[1].set(xlabel="Assumed forecaster correlation (ρ)", ylabel="Aggregate probability")
add_message_title(
    axes[1],
    "The correlation you assume moves the aggregate more than the panel size",
    subtitle="Mean forecast held fixed; dashed line marks the unextremized mean",
)
axes[1].legend()
axes[1].axhline(mean_p, color=COLORS["neutral"], linestyle="--", alpha=0.6)
format_pct_axis(axes[1])
fig.tight_layout()
plt.show()

# %% [markdown]
# The left panel is the diminishing return: each additional forecaster adds less than the last,
# and the curves flatten sooner the higher the correlation. The right panel is the part worth
# sitting with. Hold the mean forecast fixed and sweep only the assumed correlation, and the
# aggregate travels a long way. Nothing about the evidence changed along any of those lines.
# An analyst choosing $\rho$ by feel is choosing the answer, which is why the number has to
# come from somewhere defensible or the extremization step should be left out.

# %% [markdown]
# ## The Panel's Spread Does Not Enter the Formula
#
# The diversity factor reads three things: the panel size, the assumed correlation, and the
# mean. It does not read how far apart the forecasts are. Two panels can therefore be handed the
# same aggregate while telling a supervisor completely different stories - one where the agents
# converged, and one where they split and happened to average out.

# %%
panels = [("split", [0.80, 0.65, 0.50]), ("agreed", [0.64, 0.65, 0.66])]
panel_results = [
    (label, panel, neyman_extremize(panel, base=0.5, correlation=0.3)) for label, panel in panels
]

pl.DataFrame(
    {
        "panel": [label for label, _, _ in panel_results],
        "forecasts": [str(panel) for _, panel, _ in panel_results],
        "mean": [r.raw_probability for _, _, r in panel_results],
        "diversity factor d": [r.extremization_factor for _, _, r in panel_results],
        "aggregate": [r.extremized_probability for _, _, r in panel_results],
    }
)
# %% [markdown]
# Identical aggregates from panels a reader would treat very differently. The rule reads three
# inputs - the mean, the panel size, and the assumed correlation - and the spread is not one of
# them, so this is a limit on what the formula can be asked rather than a claim that dispersion
# carries nothing. If the spread should change the answer, it has to enter somewhere else:
# through a dependence model estimated from the panel, or through a rule that reads the
# distribution instead of its mean. Spread is not wasted meanwhile:
# [`07_adversarial_debate`](07_adversarial_debate.ipynb) uses a split panel as the trigger for
# making the agents argue, which is a use for dispersion that does not require putting it in
# the aggregation.

# %% [markdown]
# ## Platt Scaling: Post-Hoc Calibration
#
# Extremization asks how much a panel's agreement is worth. **Calibration** asks a different
# question: whether this forecaster's stated probabilities mean what they say. A forecaster
# who says seventy percent on a hundred questions and is right ninety times is systematically
# under-confident, and the fix is a transformation fitted on resolved forecasts rather than an
# argument about correlation.
#
# **Platt scaling** is that transformation:
#
# $$p' = \frac{d \cdot p^a}{d \cdot p^a + (1-p)^a}$$
#
# The exponent $a$ controls how far probabilities are pushed toward the ends: above one they
# spread out, below one they pull toward the middle, and at one nothing happens. The factor
# $d$ tilts the whole curve, moving the probability that maps to itself away from even odds,
# which is what corrects a forecaster biased toward one outcome.
#
# The chapter's other notebooks write the same function as
# $p' = \sigma(a \cdot \text{logit}(p) + \log d)$, which is the same map with $b = \log d$.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])

p_range = np.linspace(0.01, 0.99, 100)

# Different a values
for (a, label), color, line_style in zip(
    [
        (0.5, "a=0.5 (compress)"),
        (1.0, "a=1.0 (identity)"),
        (1.5, "a=1.5 (extremize)"),
        (2.0, "a=2.0 (strong extremize)"),
    ],
    ml4t_palette(4, categorical=True),
    line_styles[:4],
    strict=True,
):
    calibrated = [platt_scale(p, a=a, d=1.0) for p in p_range]
    ax.plot(p_range, calibrated, color=color, linestyle=line_style, label=label)

ax.plot(
    [0, 1],
    [0, 1],
    color=COLORS["neutral"],
    linestyle="--",
    alpha=0.6,
    label="Identity mapping",
)
ax.set_xlabel("Original Probability")
ax.set_ylabel("Transformed Probability")
add_message_title(ax, "Parameter a controls compression or extremization")
ax.legend(loc="upper left")
ax.set_aspect("equal")
format_pct_axis(ax, axis="both")
fig.tight_layout()
plt.show()

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
probs_w = [0.72, 0.58, 0.65]
weights_equal = [1.0, 1.0, 1.0]
weights_skewed = [0.8, 0.3, 0.5]

result_equal = neyman_extremize_weighted(probs_w, weights_equal, base=0.5, correlation=0.3)
result_skewed = neyman_extremize_weighted(probs_w, weights_skewed, base=0.5, correlation=0.3)
normalized_equal = np.array(weights_equal) / sum(weights_equal)
normalized_skewed = np.array(weights_skewed) / sum(weights_skewed)
weight_n_equal = 1 / float(np.square(normalized_equal).sum())
weight_n_skewed = 1 / float(np.square(normalized_skewed).sum())

pl.DataFrame(
    [
        {
            "weighting": "equal",
            "mean": result_equal.raw_probability,
            "extremized": result_equal.extremized_probability,
            "d": result_equal.extremization_factor,
            "weight_effective_n": weight_n_equal,
            "correlation_adjusted_n": result_equal.effective_n,
        },
        {
            "weighting": "skewed",
            "mean": result_skewed.raw_probability,
            "extremized": result_skewed.extremized_probability,
            "d": result_skewed.extremization_factor,
            "weight_effective_n": weight_n_skewed,
            "correlation_adjusted_n": result_skewed.effective_n,
        },
    ]
)

# %% [markdown]
# Two things move when the weights are skewed, and they move in opposite directions. The mean
# shifts toward the agent carrying the most weight, which raises the aggregate. The effective
# panel size falls, because concentrating weight on one forecaster is closer to consulting one
# forecaster, which lowers $d$ and pushes the aggregate back toward the base rate. The table
# above shows both, and the correlation adjustment shrinks the effective size again on top of
# %% [markdown]
# ## Log-Odds Extremization
#
# The same transformation, written where it is easiest to reason about. **Log-odds**, or the
# **logit**, is $\log\frac{p}{1-p}$: the log of the ratio of the two outcomes' probabilities.
# It maps the unit interval onto the whole real line, sends even odds to zero, and is symmetric
# about it, so "twice as confident" becomes multiplication rather than a curve:
#
# $$p' = \sigma(a \cdot \text{logit}(p)), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}$$
#
# Multiplying the log-odds by $a$ says the aggregate carries $a$ times the evidence the raw
# probability did. Above one extremizes, below one compresses, and one leaves the probability
# alone.
# This has the same effect as Platt scaling but is parameterized more intuitively:
# $a > 1$ extremizes, $a < 1$ compresses.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])

p_range = np.linspace(0.01, 0.99, 100)

for (a, label), color, line_style in zip(
    [
        (0.5, "a=0.5 (compress)"),
        (1.0, "a=1.0 (identity)"),
        (1.5, "a=1.5 (moderate extremize)"),
        (2.0, "a=2.0 (strong extremize)"),
    ],
    ml4t_palette(4, categorical=True),
    line_styles[:4],
    strict=True,
):
    calibrated = [logodds_extremize(p, a) for p in p_range]
    ax.plot(p_range, calibrated, color=color, linestyle=line_style, label=label)

ax.plot([0, 1], [0, 1], color=COLORS["neutral"], linestyle="--", alpha=0.6, label="Identity")
ax.set_xlabel("Original Probability")
ax.set_ylabel("Transformed Probability")
add_message_title(ax, "Log-odds scaling preserves symmetry around 50%")
ax.legend(loc="upper left")
ax.set_aspect("equal")
format_pct_axis(ax, axis="both")
fig.tight_layout()
plt.show()

# %% [markdown]
# The curves are the same family as Platt's, and the parameterisation is the one worth keeping:
# $a$ multiplies the log-odds, so it says directly how much more evidence the aggregate is
# being credited with than the raw probability carried. Whether $a$ should exceed one is not
# something the shape of the curve can answer; it comes from resolved forecasts, which is the
# subject of the next section.

# %% [markdown]
# ## Fitting the Exponent on Resolved Forecasts
#
# Everything above takes $a$ as given. Choosing it needs forecasts whose outcomes are known:
# grid-search the exponent that minimizes Brier score over a training panel, freeze it, and
# score it on observations it never saw. `fit_extremization_exponent` does the search; the
# separation is the caller's job and is the part that gets skipped.
#
# The panel here is simulated, with the generating process visible: a true probability drawn
# from a symmetric Beta, an outcome drawn against it, and a forecast that shrinks the true
# probability toward even odds and adds noise. That shrinkage is the miscalibration the
# exponent is supposed to undo, so the demonstration has a known right answer, which is what
# makes it a check on the procedure rather than a result about agents.

# %%
set_global_seeds(SEED)
n_questions = 240
train_size = 160
true_probs = np.random.beta(2, 2, n_questions)
outcomes = (np.random.random(n_questions) < true_probs).astype(float)
raw_forecasts = [0.5 + 0.6 * (p - 0.5) + np.random.normal(0, 0.05) for p in true_probs]
raw_forecasts = [max(0.05, min(0.95, p)) for p in raw_forecasts]

train_forecasts = raw_forecasts[:train_size]
train_outcomes = outcomes[:train_size]
test_forecasts = raw_forecasts[train_size:]
test_outcomes = outcomes[train_size:]

calibration_fit = fit_extremization_exponent(train_forecasts, train_outcomes)
test_calibrated = [
    logodds_extremize(probability, calibration_fit.optimal_exponent)
    for probability in test_forecasts
]
test_brier_before = brier_score(test_forecasts, test_outcomes)
test_brier_after = brier_score(test_calibrated, test_outcomes)
test_improvement = (test_brier_before - test_brier_after) / test_brier_before

print(f"Train observations: {len(train_forecasts)}")
print(f"Test observations:  {len(test_forecasts)}")
print(f"Fitted exponent a:  {calibration_fit.optimal_exponent:.3f}")
print(f"Test Brier before:  {test_brier_before:.4f}")
print(f"Test Brier after:   {test_brier_after:.4f}")
print(f"Test improvement:   {test_improvement:.1%}")

# %% [markdown]
# The comparison below is drawn from the held-out panel alone: the exponent was chosen on the
# training observations and applied to these unchanged.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
test_briers = [test_brier_before, test_brier_after]
bars = ax.bar(
    ["Raw", "Calibrated"],
    test_briers,
    color=[COLORS["blue"], COLORS["copper"]],
    width=0.58,
)
ax.bar_label(bars, labels=[f"{value:.3f}" for value in test_briers], padding=3)
ax.set_xlabel("Held-out forecasts")
ax.set_ylabel("Brier score (lower is better)")
ax.set_ylim(0, max(test_briers) * 1.2)
add_message_title(
    ax,
    "A calibration fitted on one panel carries to another",
    subtitle="Exponent fit on the training observations only, then frozen",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# The improvement is small, and on this data it should be. The forecasts were simulated by
# shrinking each probability linearly toward even odds and adding noise. A log-odds exponent
# does not invert a linear shrinkage: it can only be the single exponent that minimizes Brier
# score across the range, closer at some probabilities than at others, and the added noise is
# not correctable at all. So the fit is an approximate correction to an approximate
# description of the miscalibration, which is the ordinary case.
#
# What the comparison establishes is the procedure: fit on one panel, freeze, score on another.
# Read the same two numbers off a fit evaluated on its own training observations and they say
# nothing, because the exponent was chosen to make them look that way.

# %% [markdown]
# ## How Many Agents Are Worth Running
#
# Each agent costs model calls, and the previous sections have already said what the answer
# depends on. Plotting effective panel size against agent count makes the shape of the
# trade-off explicit: the vertical axis is $d^2$, the number of independent forecasters the
# panel is worth, against the number actually being paid for.

# %%
agent_counts = list(range(1, 11))
fig, ax = plt.subplots(figsize=FIGSIZE["single"])

for rho, color, line_style in zip(
    [0.1, 0.3, 0.5],
    ml4t_palette(3, categorical=True),
    line_styles[:3],
    strict=True,
):
    effective_n = []
    for n in agent_counts:
        denom = 1 + (n - 1) * rho
        d = math.sqrt(n / denom) if denom > 0 else 1.0
        effective_n.append(d**2)

    ax.plot(
        agent_counts,
        effective_n,
        marker="o",
        color=color,
        linestyle=line_style,
        label=f"ρ={rho}",
    )

ax.plot(
    agent_counts,
    agent_counts,
    color=COLORS["neutral"],
    linestyle="--",
    alpha=0.6,
    label="Independent agents",
)
ax.set_xlabel("Agents run")
ax.set_ylabel("Effective panel size")
add_message_title(
    ax,
    "Correlated agents stop adding information long before the budget runs out",
    subtitle=r"Effective size is $d^2$; for fixed ρ > 0 it approaches 1/ρ",
)
ax.legend()
fig.tight_layout()
plt.show()

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
