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
# # Yield Curve Decomposition: Level, Slope, and Curvature
#
# **Docker image**: `ml4t`
#
# **Chapter 14: Latent Factors**
#
# This notebook demonstrates one of PCA's most celebrated applications:
# decomposing the Treasury yield curve into its three primary factors.
#
# **Learning Objectives:**
# - Apply PCA to Treasury yield changes and interpret the resulting factors
# - Understand why PCA works exceptionally well for yield curves (low-dimensional macro drivers)
# - Visualize factor loadings, time series, and reconstruction quality
# - Connect the decomposition to practical factor-based hedging
#
# ## Where this fits in the framework
#
# Yield-curve PCA is a textbook **Stage 1** decomposition (Figure 14.9):
# the cross-section of yields is compressed to three orthogonal factors.
# Stage 1 is essentially where the yield curve story ends in this notebook:
# the factors are interpretable risk dimensions used for hedging and risk
# decomposition, not return forecasts. The two-step framework's Stage 2 +
# Stage 3 mechanics become relevant when factors are used predictively;
# see [`04_ipca`](04_ipca.ipynb) for that pipeline on equity panels.
# This notebook uses the complete sample only for descriptive decomposition and
# local sensitivity analysis. It contains no target, model selection, backtest,
# or performance claim, so a train/validation/test split is not applicable.
#
# **Key Concepts (Litterman & Scheinkman, 1991):** the first three components of a
# yield-curve panel take a recognisable form, and the loading figure below is where each
# is read off rather than taken on trust.
#
# - PC1 (Level): a shift of the whole curve in one direction, so its loadings share a sign
# - PC2 (Slope): steepening or flattening, so its loadings change sign between the short
#   and long ends
# - PC3 (Curvature): a butterfly twist, so the middle of the curve loads opposite to both
#   ends
#
# The share of variance each carries is printed with the loadings, alongside a bootstrap
# interval, because it is a property of this sample rather than of the yield curve.
#
# **Prerequisites**: Complete [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb); requires FRED macro data.
#
# **Data Source**: FRED macro parquet (canonical data, no API calls). Uses 8
# Treasury constant-maturity series (1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y), a
# dense enough grid for the classical Level / Slope / Curvature pattern to
# emerge clearly.
#
# **Book Reference**: Chapter 14, Section 14.4 (Decoding the yield curve)

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""Yield Curve Decomposition: Level, Slope, and Curvature via PCA."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import load_macro
from utils.style import (
    COLORS,
    FIGSIZE,
    add_message_title,
    label_line_ends,
    ml4t_palette,
    show_with_alt,
    zero_line,
)

# %%

# Maturities in years, used for x-axis positions on all yield-curve plots
MATURITIES_YEARS = [1, 2, 3, 5, 7, 10, 20, 30]
MATURITY_LABELS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# %% [markdown]
# ## 2. Data Loading: Treasury Yields from FRED Parquet
#
# We load eight Treasury constant-maturity yields (DGS1 through DGS30) from the
# canonical FRED macro parquet. This is the densest grid available from FRED's
# constant-maturity series and spans the yield curve from the front end (1Y)
# to the long end (30Y) in eight points.

# %% tags=["parameters"]
# Production defaults (Papermill overrides for CI testing)
START_DATE = "2000-01-01"
END_DATE = "2024-12-01"

# %%
# Treasury constant-maturity series from FRED
YIELD_SERIES = {
    "dgs1": "1Y",
    "dgs2": "2Y",
    "dgs3": "3Y",
    "dgs5": "5Y",
    "dgs7": "7Y",
    "dgs10": "10Y",
    "dgs20": "20Y",
    "dgs30": "30Y",
}

yield_cols = list(YIELD_SERIES.keys())

# %%
fred_data = load_macro(start_date=START_DATE, end_date=END_DATE)

yields = (
    fred_data.select(["timestamp"] + yield_cols)
    .sort("timestamp")
    .rename({c: YIELD_SERIES[c] for c in yield_cols})
    .drop_nulls()
)

yield_names = list(YIELD_SERIES.values())
timestamps = yields["timestamp"].to_numpy()

# %%
print(
    f"Loaded {len(yields):,} calendar observations from {timestamps[0]} "
    f"through {timestamps[-1]} across {len(yield_names)} maturities."
)

# %% [markdown]
# ## 3. Yield Curve Visualization
#
# Sample yield curves at different dates show how the term structure has
# evolved over the sample period, from the post-dot-com era to today's
# higher-rate regime.

# %%
observed_curve_levels = (
    yields.with_columns(
        pl.any_horizontal(pl.col(yield_names).diff() != 0).alias("observed_curve_update")
    )
    .filter(pl.col("observed_curve_update"))
    .drop("observed_curve_update")
)
n_observed_levels = len(observed_curve_levels)
sample_indices = [
    0,
    n_observed_levels // 4,
    n_observed_levels // 2,
    3 * n_observed_levels // 4,
    n_observed_levels - 1,
]

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"], constrained_layout=True)

curve_colors = ml4t_palette(4, categorical=True) + [COLORS["neutral"]]
for idx, color in zip(sample_indices, curve_colors, strict=False):
    row = observed_curve_levels.row(idx, named=True)
    curve = [row[c] for c in yield_names]
    label = str(row["timestamp"])
    ax.plot(
        MATURITIES_YEARS,
        curve,
        marker="o",
        color=color,
        label=label,
        linewidth=2,
        markersize=7,
    )

ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Yield (%)")
ax.set_xticks([1, 5, 10, 20, 30])
ax.set_xticklabels(["1Y", "5Y", "10Y", "20Y", "30Y"])
label_line_ends(ax, expand_right=0.16)
add_message_title(
    ax,
    "Treasury yield by maturity, on five selected dates",
    subtitle="Five observed Treasury curves across eight maturities, 2000-2024",
)
show_with_alt(
    fig,
    "A line chart of Treasury yield in percent against maturity in years, one line per selected "
    "date, each labelled at its right end with that date. The lines differ in level and in shape, "
    "some rising with maturity and some falling.",
)

# %% [markdown]
# ## 4. Yield Changes for PCA
#
# PCA is typically applied to yield *changes* rather than levels, as changes
# are more stationary and capture the dynamics we want to model.

# %%
calendar_changes = yields.with_columns(pl.col(c).diff() for c in yield_names).drop_nulls()

# The macro panel is calendar-daily and forward-fills FRED observations. Weekend
# and holiday rows therefore contain eight exact zeros and are not new yield-curve
# observations. Keep genuine observations only; an unchanged single maturity is valid.
stale_calendar_row = pl.all_horizontal(pl.col(yield_names) == 0)
yield_changes = calendar_changes.filter(~stale_calendar_row)
n_stale_rows = len(calendar_changes) - len(yield_changes)

change_timestamps = yield_changes["timestamp"].to_numpy()

# %%
print(
    f"Retained {len(yield_changes):,} observed curve changes after removing "
    f"{n_stale_rows:,} forward-filled zero-change calendar rows."
)

# %% [markdown]
# We standardize yield changes before PCA (correlation-matrix PCA). This
# equalizes the influence of each maturity regardless of its volatility level.
# Some practitioners apply PCA to the covariance matrix directly when
# maturity-level variance differences are economically meaningful, for example,
# to let the volatile short end dominate the first component. With standardization,
# each maturity contributes equally to the factor structure.

# %%
scaler = StandardScaler()
changes_np = yield_changes.select(yield_names).to_numpy()
changes_scaled = scaler.fit_transform(changes_np)

# %% [markdown]
# ## 5. PCA on Yield Changes
#
# PCA decomposes yield changes into orthogonal factors:
#
# $$z(\Delta y_i) = \beta_{i,1} f_1 + \beta_{i,2} f_2 + \beta_{i,3} f_3 + \epsilon_i$$
#
# where $z(\Delta y_i)$ is a standardized yield change, $f_k$ is a principal
# component score, and $\beta_{i,k}$ is its correlation-PCA loading. Section 10
# converts the fitted moves back to basis points before applying key-rate DV01s.
# With eight maturities, we expect the first three components to reproduce the
# classical Litterman–Scheinkman (1991) Level / Slope / Curvature pattern.

# %%
# svd_solver='full' is default for small matrices; explicit for clarity
pca = PCA(svd_solver="full")
pca.fit(changes_scaled)

var_explained = pca.explained_variance_ratio_
cumvar_explained = np.cumsum(var_explained)

n_meaningful = 3

# %% [markdown]
# An eigenvector's sign is arbitrary - $v$ and $-v$ describe the same direction - so the
# signs below are fixed to make each component's name mean what it says: all-positive
# loadings for Level, so the component is a parallel shift up; long-end positive and
# short-end negative for Slope, so a positive score is a steepening; and a positive
# middle for Curvature, so a positive score lifts the belly against the wings. Without
# this, half the runs would name a flattening a steepening.

# %%
loadings = pca.components_[:n_meaningful].copy()  # numpy array (n_meaningful x n_rates)
loading_names = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
scores_np = pca.transform(changes_scaled)

signs = np.ones(n_meaningful)
if loadings[0].mean() < 0:
    signs[0] = -1.0
if loadings[1, -1] - loadings[1, 0] < 0:
    signs[1] = -1.0
belly = np.array([1, 2, 3, 4, 5])
wings = np.array([0, 6, 7])
if loadings[2, belly].mean() - loadings[2, wings].mean() < 0:
    signs[2] = -1.0
loadings = loadings * signs[:, None]
scores_np[:, :n_meaningful] = scores_np[:, :n_meaningful] * signs

# Store key statistics
pc1_variance = var_explained[0] * 100
pc2_variance = var_explained[1] * 100
pc3_variance = var_explained[2] * 100
top3_cumvar = cumvar_explained[2] * 100

# %%
print(
    f"Variance shares: Level {pc1_variance:.2f}%, Slope {pc2_variance:.2f}%, "
    f"Curvature {pc3_variance:.2f}%; cumulative {top3_cumvar:.2f}%."
)

# %% [markdown]
# **Interpretation**: PC1 explains the bulk of the variance, PC2 captures a
# steepening/flattening dimension orthogonal to it, and PC3 captures a
# middle-vs-ends curvature twist. Together the three components account for
# essentially all of the daily-change variance, confirming the classical
# Litterman–Scheinkman finding that the yield curve is effectively
# three-dimensional. The loadings below establish the economic interpretation
# of each factor.

# %% [markdown]
# ### Moving-block stability check
#
# A single full-sample PCA does not show whether the economic labels are stable.
# We resample contiguous 21-observation blocks, refit the scaler and PCA inside
# each bootstrap sample, and align all components to the full-sample basis by
# permutation and sign. The intervals quantify sampling variation in the loading
# curves while respecting short-run dependence in yield changes.

# %%
N_BOOTSTRAP = 500
BLOCK_LENGTH = 21
RANDOM_SEED = 42

# %% [markdown]
# The circular index sampler joins randomly selected contiguous blocks until it
# reaches the original sample length. Circular wrapping avoids giving the sample
# endpoints special treatment.


# %%
def moving_block_indices(
    n_observations: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a circular moving-block bootstrap index of the requested length."""
    n_blocks = int(np.ceil(n_observations / block_length))
    starts = rng.integers(0, n_observations, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets) % n_observations).ravel()[:n_observations]


# %% [markdown]
# Bootstrap components may change sign or order without changing their economic
# content. A one-to-one Hungarian assignment aligns the complete orthonormal basis
# to the reference before intervals are computed.


# %%
def align_to_reference(
    candidate: np.ndarray,
    candidate_variance: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match component order and signs to a reference orthonormal basis."""
    similarity = candidate @ reference.T
    candidate_idx, reference_idx = linear_sum_assignment(-np.abs(similarity))
    aligned = np.empty_like(candidate)
    aligned_variance = np.empty_like(candidate_variance)
    aligned_cosine = np.empty_like(candidate_variance)
    for source, target in zip(candidate_idx, reference_idx, strict=True):
        direction = 1.0 if similarity[source, target] >= 0 else -1.0
        aligned[target] = direction * candidate[source]
        aligned_variance[target] = candidate_variance[source]
        aligned_cosine[target] = abs(similarity[source, target])
    return aligned, aligned_variance, aligned_cosine


# %%
rng = np.random.default_rng(RANDOM_SEED)
reference_components = pca.components_.copy()
reference_components[:n_meaningful] = loadings
bootstrap_loadings = np.empty((N_BOOTSTRAP, n_meaningful, len(yield_names)))
bootstrap_variance = np.empty((N_BOOTSTRAP, n_meaningful))
bootstrap_cosine = np.empty((N_BOOTSTRAP, n_meaningful))

for bootstrap_id in range(N_BOOTSTRAP):
    sample_idx = moving_block_indices(len(changes_np), BLOCK_LENGTH, rng)
    sample_scaled = StandardScaler().fit_transform(changes_np[sample_idx])
    sample_pca = PCA(svd_solver="full").fit(sample_scaled)
    aligned, aligned_variance, aligned_cosine = align_to_reference(
        sample_pca.components_, sample_pca.explained_variance_ratio_, reference_components
    )
    bootstrap_loadings[bootstrap_id] = aligned[:n_meaningful]
    bootstrap_variance[bootstrap_id] = aligned_variance[:n_meaningful]
    bootstrap_cosine[bootstrap_id] = aligned_cosine[:n_meaningful]

# %%
loading_ci_low, loading_ci_high = np.percentile(bootstrap_loadings, [2.5, 97.5], axis=0)
variance_ci_low, variance_ci_high = np.percentile(bootstrap_variance * 100, [2.5, 97.5], axis=0)
median_cosine = np.median(bootstrap_cosine, axis=0)

print(
    "Median aligned loading cosine: "
    + ", ".join(
        f"{name.split()[0]} {cosine:.3f}"
        for name, cosine in zip(loading_names, median_cosine, strict=True)
    )
)

# %% [markdown]
# ## 6. Factor Loadings (Figure 14.4)
#
# The loadings reveal how each factor distributes across maturities. With a
# dense maturity grid spanning 1Y to 30Y, the classical Level / Slope / Curvature
# pattern emerges in the first three components.

# %%
loading_messages = [
    "PC1 loading across maturities: the level factor",
    "PC2 loading across maturities: the slope factor",
    "PC3 loading across maturities: the curvature factor",
]
component_colors = ml4t_palette(3, categorical=True)
loading_limit = 1.08 * np.max(np.abs([loading_ci_low, loading_ci_high]))

# %%
fig, axes = plt.subplots(
    3,
    1,
    figsize=FIGSIZE["grid_3x2"],
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for i, (ax, message, color) in enumerate(
    zip(axes, loading_messages, component_colors, strict=False)
):
    loading = loadings[i]

    zero_line(ax)
    ax.fill_between(
        MATURITIES_YEARS,
        loading_ci_low[i],
        loading_ci_high[i],
        color=color,
        alpha=0.2,
        label="95% block-bootstrap interval",
    )
    ax.plot(MATURITIES_YEARS, loading, color=color, marker="o", linewidth=2.5, markersize=6)

    ax.set_ylabel("Loading")
    ax.set_ylim(-loading_limit, loading_limit)
    ax.set_xticks([1, 5, 10, 20, 30])
    ax.set_xticklabels(["1Y", "5Y", "10Y", "20Y", "30Y"])
    add_message_title(
        ax,
        message,
        subtitle="Point estimate with its block-bootstrap interval",
    )
axes[-1].set_xlabel("Maturity (years)")
show_with_alt(
    fig,
    "Three stacked panels sharing a maturity axis in years and a common loading scale. "
    "Each plots one component's loading at each maturity as a marked line, with a "
    "shaded block-bootstrap interval around it and a dashed line at zero.",
)
for i in range(3):
    print(
        f"PC{i + 1}: {var_explained[i] * 100:.1f}% of variance "
        f"[95% CI {variance_ci_low[i]:.1f}-{variance_ci_high[i]:.1f}%]"
    )

# %% [markdown]
# **PC1 (Level)**: Approximately equal positive loadings across all maturities,
# a near-parallel shift of the entire curve. This is the dominant mode of yield
# variation, driven by changes in inflation expectations and the overall level
# of interest rates.
#
# **PC2 (Slope)**: Monotonically increasing loadings from short to long
# maturities (negative at the short end, positive at the long end). Positive
# PC2 corresponds to *steepening* (long rates rise more than short rates).
# Slope variation reflects business-cycle dynamics: the curve typically flattens
# during tightening cycles and steepens during easing.
#
# **PC3 (Curvature)**: Opposite signs on the middle of the curve versus the
# ends, a "butterfly" twist where intermediate maturities move differently
# from the short and long ends. This factor reflects nuances in term-premium
# pricing and convexity effects.

# %% [markdown]
# ## 7. Scree Plot

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], constrained_layout=True)

n_components = len(var_explained)

# Individual variance
ax1 = axes[0]
ax1.bar(range(1, n_components + 1), var_explained * 100, color=COLORS["blue"])
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Variance Explained (%)")
ax1.set_xticks(range(1, n_components + 1))
add_message_title(
    ax1,
    "Variance explained per principal component",
    subtitle="Individual correlation-PCA shares",
)

# Cumulative variance
ax2 = axes[1]
ax2.plot(
    range(1, n_components + 1),
    cumvar_explained * 100,
    color=COLORS["blue"],
    marker="o",
    linewidth=2,
    markersize=8,
)
ax2.axhline(y=95, color=COLORS["amber"], linestyle="--", alpha=0.7, label="95% threshold")
ax2.axhline(y=99, color=COLORS["copper"], linestyle="--", alpha=0.7, label="99% threshold")
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Variance Explained (%)")
ax2.set_xticks(range(1, n_components + 1))
ax2.set_ylim(0, 105)
ax2.legend()
add_message_title(
    ax2,
    "Cumulative variance explained by component count",
    subtitle="With dashed references at the 95% and 99% levels",
)

show_with_alt(
    fig,
    "Two stacked panels. The upper is a bar chart of the share of variance each principal "
    "component explains, in percent, against component number. The lower plots the cumulative "
    "share against the number of components retained, with dashed horizontal references at the 95 "
    "and 99 percent levels.",
)

# %% [markdown]
# The first component dominates, three components capture essentially all of
# the variance, and components beyond PC3 add negligible explanatory power.
# The yield curve's effective dimension is three, exactly the Level / Slope /
# Curvature structure documented in Litterman & Scheinkman (1991) for the US
# Treasury market.

# %% [markdown]
# ## 8. Factor-Shock Time Series
#
# PCA was fitted to yield *changes*, so its scores are daily shocks rather than
# yield levels. Dividing each score by its fitted standard deviation places the
# three components on a comparable scale. A 63-observation rolling mean reveals
# sustained directions in recent curve changes without relabeling those changes
# as the level or inversion state of the curve.

# %%
factor_names = ["Level", "Slope", "Curvature"]
scores_for_plot = scores_np[:, :n_meaningful]
factor_zscores = scores_for_plot / np.sqrt(pca.explained_variance_[:n_meaningful])
ROLLING_WINDOW = 63
rolling_factor_shocks = np.column_stack(
    [
        np.convolve(
            factor_zscores[:, component],
            np.ones(ROLLING_WINDOW) / ROLLING_WINDOW,
            mode="valid",
        )
        for component in range(n_meaningful)
    ]
)
rolling_timestamps = change_timestamps[ROLLING_WINDOW - 1 :]

# %%
fig, axes = plt.subplots(
    n_meaningful,
    1,
    figsize=FIGSIZE["grid_3x2"],
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
shock_messages = [
    "Level factor score over time",
    "Slope factor score over time",
    "Curvature factor score over time",
]
shock_limit = 1.05 * np.max(np.abs(rolling_factor_shocks))

for i, (ax, name, message, color) in enumerate(
    zip(axes, factor_names, shock_messages, component_colors, strict=False)
):
    ax.plot(rolling_timestamps, rolling_factor_shocks[:, i], color=color, linewidth=1.1)
    zero_line(ax)
    ax.set_ylabel(name, fontsize=12)
    ax.set_ylim(-shock_limit, shock_limit)
    ax.tick_params(axis="both", labelsize=11)
    add_message_title(ax, message, subtitle="63-observation mean standardized shock")

axes[-1].set_xlabel("Date", fontsize=12)
show_with_alt(
    fig,
    "Three stacked panels sharing a date axis, one per component, each plotting that component's "
    "standardized factor score over time against a dashed line at zero.",
)

# %% [markdown]
# The rolling Level shock turns strongly negative during rapid easing and positive
# during tightening episodes. The Slope and Curvature series isolate recent
# steepening/flattening and butterfly directions. These are descriptive changes,
# not recession signals or estimates of the current curve level; interpreting the
# economic cause of any episode requires information outside this PCA.

# %% [markdown]
# ## 9. Reconstruction Quality
#
# The reconstructed yield changes use the first $K$ components:
#
# $$\Delta \hat{y} = F_K \Lambda_K^T$$
#
# where $F_K$ is the $(T \times K)$ score matrix and $\Lambda_K$ the $(K \times N)$ loading matrix.
# With three factors explaining essentially all of the variance, reconstructed
# changes should closely track actuals across every maturity.

# %%
# Reconstruct yield changes from the first 3 PCs
n_reconstruct = 3
# Use the SIGN-CORRECTED scores and loadings consistently
scores_3pc = scores_np[:, :n_reconstruct]
reconstructed_scaled = scores_3pc @ loadings[:n_reconstruct]
reconstructed_np = scaler.inverse_transform(reconstructed_scaled)

# Reconstruction RMSE per maturity (in basis points)
actual_std_bps = changes_np.std(axis=0) * 100
rmse_bps = np.sqrt(((changes_np - reconstructed_np) ** 2).mean(axis=0)) * 100
rmse_ratio_pct = rmse_bps / actual_std_bps * 100

# %%
# Overlay recent actual vs reconstructed changes and summarize all maturities
target = "10Y"
target_idx = yield_names.index(target)
recent_observations = 504
recent_slice = slice(-recent_observations, None)
recent_tick_indices = np.linspace(
    len(change_timestamps) - recent_observations, len(change_timestamps) - 1, 3, dtype=int
)
recent_tick_dates = change_timestamps[recent_tick_indices]
recent_tick_labels = [np.datetime_as_string(date, unit="M") for date in recent_tick_dates]

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"], constrained_layout=True)
axes[0].plot(
    change_timestamps[recent_slice],
    changes_np[recent_slice, target_idx] * 100,
    color=COLORS["blue"],
    linewidth=0.9,
    label="Actual",
)
axes[0].plot(
    change_timestamps[recent_slice],
    reconstructed_np[recent_slice, target_idx] * 100,
    color=COLORS["amber"],
    linewidth=0.9,
    label=f"Reconstructed ({n_reconstruct} PCs)",
)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Observed change (bps)")
axes[0].set_xticks(recent_tick_dates, recent_tick_labels)
axes[0].legend()
add_message_title(
    axes[0],
    "Observed and three-component reconstructed 10Y changes",
    subtitle="The most recent stretch of the sample",
)

rmse_positions = np.arange(len(yield_names))
axes[1].bar(rmse_positions, rmse_ratio_pct, color=COLORS["blue"])
axes[1].set_xlabel("Maturity")
axes[1].set_ylabel("RMSE / observed standard deviation (%)")
axes[1].set_ylim(0, 22)
axes[1].set_xticks(rmse_positions, yield_names)
axes[1].tick_params(axis="x", labelsize=8)
for position, ratio in zip(rmse_positions, rmse_ratio_pct, strict=True):
    axes[1].text(position, ratio + 0.5, f"{ratio:.1f}", ha="center", fontsize=7)
add_message_title(
    axes[1],
    "Reconstruction error by maturity",
    subtitle="RMSE as a percentage of that maturity's observed-change volatility",
)

show_with_alt(
    fig,
    "Two panels. The left plots the observed daily change in the 10-year yield in basis points "
    "against date, over the most recent stretch of the sample, with the three-component "
    "reconstruction drawn over it. The right is a bar chart of reconstruction RMSE as a "
    "percentage of observed-change volatility, one bar per maturity, each labelled with its "
    "value.",
)

# %% [markdown]
# **Reading it**: the right panel's bars are the question - each is a maturity's
# reconstruction error as a share of how much that maturity actually moves, so a low bar
# means the three factors carry most of that maturity's variation and a high one means
# they do not. Comparing bars across maturities is what the normalisation makes possible;
# comparing raw RMSE would just rank maturities by volatility.
#
# The left panel is the same fact at daily resolution, and it is worth looking at
# alongside the bars: a maturity can have a respectable aggregate error and still miss
# individual days badly. Whatever is left over is variation the three-factor basis does
# not represent, and PCA on its own says nothing about what causes it.

# %% [markdown]
# ## 10. Practical Application: Generalized Duration
#
# The low-dimensional structure enables efficient hedging. Rather than managing
# exposure to dozens of individual bonds, a portfolio manager neutralizes three
# factor exposures: level, slope, and curvature sensitivity:
#
# 1. **Level Duration**: Sensitivity to parallel shifts (traditional DV01)
# 2. **Slope Duration**: Sensitivity to curve steepening/flattening
# 3. **Curvature Duration**: Sensitivity to butterfly twists
#
# This "generalized duration" framework enables more precise hedging using
# liquid instruments (Treasury futures, interest rate swaps) that target
# specific yield curve movements. See Chapter 14, Section 14.4 for the full
# hedging discussion.

# %% [markdown]
# Each loading is in standardized-yield space, which is the wrong unit for a hedge. The
# conversion below puts it back into basis points, which is what a key-rate DV01 profile
# is quoted in, so the two can be multiplied to get an exposure per unit of factor score.

# %%
factor_moves_bps = loadings * scaler.scale_[None, :] * 100

# Illustrative portfolio key-rate DV01 profile, in $1,000 per basis point.
portfolio_krd = np.array([0.10, 0.25, 0.40, 0.80, 1.20, 1.60, 1.20, 0.80])
unhedged_factor_exposure = factor_moves_bps @ portfolio_krd

# Use 2Y, 10Y, and 30Y key-rate instruments as three independent hedge directions.
hedge_indices = np.array([1, 5, 7])
hedge_positions = np.linalg.solve(factor_moves_bps[:, hedge_indices], -unhedged_factor_exposure)
hedged_krd = portfolio_krd.copy()
hedged_krd[hedge_indices] += hedge_positions
hedged_factor_exposure = factor_moves_bps @ hedged_krd

assert np.max(np.abs(hedged_factor_exposure)) < 1e-10

# %%
positions = np.arange(len(yield_names))
factor_positions = np.arange(n_meaningful)
width = 0.36
panel_specs = [
    (
        positions,
        portfolio_krd,
        hedged_krd,
        yield_names,
        "After hedge",
        "Key-rate maturity",
        "Key-rate DV01 ($1,000 per bp)",
        "Key-rate DV01 by maturity, before and after the hedge",
    ),
    (
        factor_positions,
        unhedged_factor_exposure,
        hedged_factor_exposure,
        factor_names,
        "After hedge",
        "Factor",
        "Factor exposure ($1,000 per score unit)",
        "Factor exposure by component, before and after the hedge",
    ),
]

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], constrained_layout=True)
for ax, spec in zip(axes, panel_specs, strict=True):
    x, before, after, tick_labels, after_label, xlabel, ylabel, title = spec
    ax.bar(x - width / 2, before, width, label="Before hedge", color=COLORS["blue"])
    ax.bar(x + width / 2, after, width, label=after_label, color=COLORS["amber"])
    zero_line(ax)
    if len(x) > 3:
        shown = np.array([0, 1, 3, 5, 7])
        ax.set_xticks(x[shown], np.asarray(tick_labels)[shown])
    else:
        ax.set_xticks(x, tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    add_message_title(ax, title)
show_with_alt(
    fig,
    "Two stacked panels comparing the position before and after the hedge, with paired bars in "
    "each. The upper gives key-rate DV01 in dollars per basis point at each maturity, against a "
    "line at zero; the lower gives exposure to each of the three factors in dollars per unit of "
    "factor score.",
)

# %% [markdown]
# **Finding**: In this local linear example, positions in three independent
# key-rate instruments reduce all three fitted factor exposures to numerical
# zero. This is an algebraic exposure match, not a backtest. A production hedge
# must map the target weights to actual futures or swaps and account for
# convexity, carry, basis risk, liquidity, and transaction costs.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Eight maturities, three directions that matter.** The cumulative variance figure
#    is where that claim is checked, and the printed shares say how the total splits
#    across the three. The loading shapes are the Level, Slope and Curvature structure
#    Litterman and Scheinkman documented in 1991, and the moving-block bootstrap says how
#    much of each shape is pinned down by this sample rather than by the resampling.
#
# 2. **Low-dimensional macro drivers**: The yield curve's compressibility
#    reflects that its underlying drivers (inflation expectations, business
#    cycle, term premium) are themselves low-dimensional, in sharp contrast
#    to equities, where thousands of idiosyncratic factors operate alongside
#    a smaller systematic core.
#
# 3. **Practical hedging**: The illustrative key-rate example shows how three
#    independent hedge directions can neutralize fitted Level, Slope, and
#    Curvature exposure. It is a local sensitivity calculation, not evidence of
#    realized hedge performance.
#
# **Next**: See [`04_ipca`](04_ipca.ipynb) for time-varying factor models that extend PCA
# by conditioning on observable characteristics.
