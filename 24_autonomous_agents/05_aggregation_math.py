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

# %%
"""From Opinions to Probabilities: aggregation math for multi-agent forecasting."""

import math
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from agent_pipeline import (
    brier_score,
    find_optimal_d,
    logodds_extremize,
    neyman_extremize,
    neyman_extremize_weighted,
    platt_scale,
)

from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, format_pct_axis, ml4t_palette

# %% tags=["parameters"]
N_FORECASTERS = 3
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## The Averaging Problem

# %%
probs = [0.65, 0.65, 0.65]

result_independent = neyman_extremize(probs, base=0.5, correlation=0.0)
print(
    f"Independent (ρ=0.0): d={result_independent.extremization_factor:.2f}, "
    f"p={result_independent.extremized_probability:.2f}"
)

result_moderate = neyman_extremize(probs, base=0.5, correlation=0.3)
print(
    f"Moderate (ρ=0.3):    d={result_moderate.extremization_factor:.2f}, "
    f"p={result_moderate.extremized_probability:.2f}"
)

result_correlated = neyman_extremize(probs, base=0.5, correlation=0.7)
print(
    f"Correlated (ρ=0.7):  d={result_correlated.extremization_factor:.2f}, "
    f"p={result_correlated.extremized_probability:.2f}"
)

# %% [markdown]
# ## Visualizing the Extremization Factor

# %%
correlations = [0.0, 0.1, 0.3, 0.5, 0.7]
n_range = list(range(2, 12))
curve_colors = ml4t_palette(len(correlations), categorical=True)
line_styles = ["-", "--", ":", "-.", "-"]

# HATA DÜZELTME: Sınırlandırma mantığı max(1.0, min(3.0, ...)) olarak tutarlı hale getirildi.
diversity_series = [
    (
        rho,
        [max(1.0, min(3.0, math.sqrt(n / (1 + (n - 1) * rho)))) for n in n_range],
        color,
        line_style,
    )
    for rho, color, line_style in zip(correlations, curve_colors, line_styles, strict=True)
]

# %% [markdown]
# ### Aggregate sensitivity

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
        denom = 1 + (n - 1) * rho
        d_raw = math.sqrt(n / denom) if denom > 0 else 1.0
        d = max(1.0, min(3.0, d_raw))
        p_extreme = base + d * (mean_p - base)
        agg_probs.append(max(0.01, min(0.99, p_extreme)))
    aggregate_series.append((n, agg_probs, color, line_style))

# %% [markdown]
# ### Combined sensitivity view

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
axes[0].set(xlabel="Number of Forecasters", ylabel="Diversity Factor d")
add_message_title(axes[0], "Correlation limits diversity")
axes[0].legend()
axes[0].axhline(1.0, color=COLORS["neutral"], linestyle="--", alpha=0.6)

for n, values, color, line_style in aggregate_series:
    axes[1].plot(rho_range, values, color=color, linestyle=line_style, label=f"n={n}")
axes[1].set(xlabel="Forecaster Correlation (ρ)", ylabel="Aggregate Probability")
add_message_title(axes[1], "Aggregate sensitivity", subtitle="From a 65% mean forecast")
axes[1].legend()
axes[1].axhline(0.65, color=COLORS["neutral"], linestyle="--", alpha=0.6)
format_pct_axis(axes[1])

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Observed Dispersion Is Not in the Formula

# %%
divergent = [0.80, 0.65, 0.50]
result_divergent = neyman_extremize(divergent, base=0.5, correlation=0.3)

print(f"Specialist probabilities: {divergent}")
print(f"Simple mean: {np.mean(divergent):.2f}")
print(
    f"Neyman (ρ=0.3): {result_divergent.extremized_probability:.2f} "
    f"(d={result_divergent.extremization_factor:.2f})"
)

tight = [0.64, 0.65, 0.66]
result_tight = neyman_extremize(tight, base=0.5, correlation=0.3)

print(f"\nTight agreement: {tight}")
print(f"Simple mean: {np.mean(tight):.2f}")
print(f"Neyman (ρ=0.3): {result_tight.extremized_probability:.2f}")

# %% [markdown]
# ## Platt Scaling: Post-Hoc Calibration

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
p_range = np.linspace(0.01, 0.99, 100)

# HATA DÜZELTME: Vektörleştirilmiş fonksiyon kullanımı
v_platt = np.vectorize(lambda p, a: platt_scale(p, a=a, d=1.0))

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
    calibrated = v_platt(p_range, a)
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
# ## Weighted Neyman Extremization

# %%
probs_w = [0.72, 0.58, 0.65]
weights_equal = [1.0, 1.0, 1.0]
weights_skewed = [0.8, 0.3, 0.5]

result_equal = neyman_extremize_weighted(probs_w, weights_equal, base=0.5, correlation=0.3)
result_skewed = neyman_extremize_weighted(probs_w, weights_skewed, base=0.5, correlation=0.3)

normalized_equal = np.array(weights_equal) / np.sum(weights_equal)
normalized_skewed = np.array(weights_skewed) / np.sum(weights_skewed)
weight_n_equal = 1 / float(np.sum(np.square(normalized_equal)))
weight_n_skewed = 1 / float(np.sum(np.square(normalized_skewed)))

# HATA DÜZELTME: Çıktı bir değişkene atanarak görünür kılındı.
df_weighted_summary = pl.DataFrame(
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
print(df_weighted_summary)

# %% [markdown]
# ## Log-Odds Extremization

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
v_logodds = np.vectorize(logodds_extremize)

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
    calibrated = v_logodds(p_range, a)
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
# ## Finding the Optimal Calibration Parameter

# %%
set_global_seeds(SEED)
n_questions = 240
train_size = 160
true_probs = np.random.beta(2, 2, n_questions)
outcomes = (np.random.random(n_questions) < true_probs).astype(float)
raw_forecasts = np.clip(0.5 + 0.6 * (true_probs - 0.5) + np.random.normal(0, 0.05, n_questions), 0.05, 0.95)

train_forecasts = raw_forecasts[:train_size]
train_outcomes = outcomes[:train_size]
test_forecasts = raw_forecasts[train_size:]
test_outcomes = outcomes[train_size:]

calibration_fit = find_optimal_d(train_forecasts, train_outcomes)
test_calibrated = v_logodds(test_forecasts, calibration_fit.optimal_d)

test_brier_before = brier_score(test_forecasts, test_outcomes)
test_brier_after = brier_score(test_calibrated, test_outcomes)
test_improvement = (test_brier_before - test_brier_after) / test_brier_before

print(f"Train observations: {len(train_forecasts)}")
print(f"Test observations:  {len(test_forecasts)}")
print(f"Train-fitted a:     {calibration_fit.optimal_d:.3f}")
print(f"Test Brier before:  {test_brier_before:.4f}")
print(f"Test Brier after:   {test_brier_after:.4f}")
print(f"Test improvement:   {test_improvement:.1%}")

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
ax.set_xlabel("Held-Out Forecast")
ax.set_ylabel("Brier Score (Lower Is Better)")
ax.set_ylim(0, max(test_briers) * 1.2)
add_message_title(
    ax,
    f"Held-out Brier changes by {test_improvement:.1%}",
    subtitle=f"Parameter a={calibration_fit.optimal_d:.2f} fit on training observations only",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Sensitivity Analysis: How Many Agents Do You Need?

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])

for rho, color, line_style in zip(
    [0.1, 0.3, 0.5],
    ml4t_palette(3, categorical=True),
    line_styles[:3],
    strict=True,
):
    effective_n = []
    agent_counts = list(range(1, 11))
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
    label="n_eff = n (independent)",
)
ax.set_xlabel("Number of Agents")
ax.set_ylabel("Effective N (information content)")
add_message_title(
    ax,
    "Positive correlation bounds effective panel size",
    subtitle=r"For fixed ρ > 0, effective N approaches 1/ρ",
)
ax.legend()
fig.tight_layout()
plt.show()
