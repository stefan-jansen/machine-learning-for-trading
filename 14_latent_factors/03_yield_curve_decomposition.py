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
# **Key Concepts (Litterman & Scheinkman, 1991):**
# - PC1 (Level): Parallel shift of the entire curve (~82% of variance in this sample)
# - PC2 (Slope): Steepening/flattening, with opposite moves at short and long maturities (~12%)
# - PC3 (Curvature): Butterfly twist, with the middle moving opposite to the ends (~3%)
#
# **Prerequisites**: Complete [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb); requires FRED macro data.
#
# **Data Source**: FRED macro parquet (canonical data, no API calls). Uses 8
# Treasury constant-maturity series (1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y), a
# dense enough grid for the classical Level / Slope / Curvature pattern to
# emerge clearly, with PC1/PC2/PC3 explaining ~82/12/3 % of the variance in
# this 2000–2024 sample.
#
# **Book Reference**: Chapter 14, Section 14.4 (The Yield Curve Decoded)

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""Yield Curve Decomposition: Level, Slope, and Curvature via PCA."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from data import load_macro
from utils.style import (
    COLORS,
    FIGSIZE,
    add_message_title,
    label_line_ends,
    ml4t_palette,
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
    "Curve shapes span flat, steep, and inverted regimes",
    subtitle="Five observed Treasury curves across eight maturities, 2000-2024",
)
fig.show()

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
loadings = pca.components_[:n_meaningful].copy()  # numpy array (n_meaningful x n_rates)
loading_names = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
scores_np = pca.transform(changes_scaled)

# PCA component signs are arbitrary. Flip so each PC's economic interpretation
# matches its name:
#   Level: all-positive loadings (parallel shift up)
#   Slope: long-end positive, short-end negative (positive PC2 = steepening)
#   Curvature: middle of curve positive (butterfly upward)
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
    "Level is a parallel shift",
    "Slope pivots the curve",
    "Curvature bends the belly",
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
        subtitle=(
            f"PC{i + 1}: {var_explained[i] * 100:.1f}% "
            f"[95% CI {variance_ci_low[i]:.1f}-{variance_ci_high[i]:.1f}%]"
        ),
    )
axes[-1].set_xlabel("Maturity (years)")
fig.show()

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
    "Level dominates variance",
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
    "Three PCs reach 97.8%",
    subtitle="Cumulative share with 95% and 99% references",
)

fig.show()

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
    "Level shocks alternate between easing and tightening",
    "Slope shocks isolate steepening and flattening",
    "Curvature shocks move the belly against the wings",
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
fig.show()

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
    "Three PCs track 10Y changes",
    subtitle=f"Latest {recent_observations} observed changes",
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
    "Error stays below 20%",
    subtitle="RMSE / observed-change volatility",
)

fig.show()

# %% [markdown]
# **Finding**: Three PCs reconstruct every maturity's daily changes with RMSE
# between roughly 7% and 20% of the maturity's standard deviation (the 1Y is
# the tightest at ~7%, most maturities fall in the 13-20% range). The recent
# 10Y overlay shows where the approximation misses individual daily moves. The
# remaining residual is maturity-specific variation that the three-factor basis
# does not represent; PCA alone cannot identify its economic cause.

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

# %%
# Each PCA loading is expressed in standardized-yield space. Convert it back to
# basis-point moves so a key-rate DV01 profile can be mapped into factor exposure.
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
        "Hedge reshapes key-rate DV01",
    ),
    (
        factor_positions,
        unhedged_factor_exposure,
        hedged_factor_exposure,
        factor_names,
        "After hedge",
        "Factor",
        "Factor exposure ($1,000 per score unit)",
        "Factor exposure closes to zero",
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
fig.show()

# %% [markdown]
# **Finding**: In this local linear example, positions in three independent
# key-rate instruments reduce all three fitted factor exposures to numerical
# zero. This is an algebraic exposure match, not a backtest. A production hedge
# must map the target weights to actual futures or swaps and account for
# convexity, carry, basis risk, liquidity, and transaction costs.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Three-dimensional yield curve**: PC1 (Level), PC2 (Slope), and PC3
#    (Curvature) together explain 97.8% of the variance in observed daily
#    yield-change cross-sections, an 82/12/3% split in this correlation-PCA sample. A
#    moving-block bootstrap preserves the loading shapes while quantifying their
#    sampling variation. This is the same
#    Level/Slope/Curvature structure documented in Litterman & Scheinkman
#    (1991). The yield curve's effective dimension is three, not eight.
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
