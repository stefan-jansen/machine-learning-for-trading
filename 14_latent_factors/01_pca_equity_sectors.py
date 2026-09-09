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
# # PCA on Sector ETFs with Bootstrap Loading Stability
#
# **Docker image**: `ml4t`
#
# **Chapter 14: Latent Factors**
# **Section Reference**: See Section 14.2 for PCA theory and Section 14.3 for eigenportfolios
#
# ## Purpose
# This notebook applies PCA to sector ETFs to extract latent risk factors and quantifies
# loading stability using bootstrap resampling. We demonstrate how PCA captures market
# and rotation factors, and how to assess whether factor loadings are statistically reliable.
#
# ## Where this fits in the framework
#
# PCA is the simplest realisation of **Stage 1** in the two-step latent-factor
# framework (Figure 14.9): it compresses an $(T \times N)$ returns panel to
# a $(T \times K)$ factor history via SVD. This notebook focuses on the
# Stage 1 outputs - variance decomposition, loadings, bootstrap stability -
# and the structural interpretation of the principal components. The full
# Stage 1 + 2 + 3 pipeline (with a Stage 2 forecaster turning factor history
# into asset signals) is exercised in [`04_ipca`](04_ipca.ipynb) and
# [`05_rp_pca`](05_rp_pca.ipynb).
#
# ## Learning Objectives
# - LO1: Apply PCA to sector ETF returns and interpret variance decomposition
# - LO2: Quantify loading stability with bootstrap confidence intervals
# - LO3: Interpret sector factor exposures (market vs rotation)
# - LO4: Analyze temporal stability of factor structure using rolling PCA
#
# ## Cross-References
# - **Upstream**: Chapter 3 (ETF data loading)
# - **Downstream**: Chapter 16 (factor investing), Chapter 18 (factor-based backtesting)
# - **Related**: [`02_eigenportfolios`](02_eigenportfolios.ipynb) (stock-level PCA), Section 9.4 (HMM regimes)
#
# ## Data Source
# ETF Universe parquet (canonical data, no API calls)
#
# **Prerequisites**: Requires ETF Universe data (see Chapter 3).

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""PCA on Sector ETFs - Variance decomposition and bootstrap loading stability."""

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from matplotlib.ticker import PercentFormatter
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import (
    COLORS,
    FIGSIZE,
    add_message_title,
    show_plotly_with_alt,
    show_with_alt,
    zero_line,
)

# %% tags=["parameters"]
# Production defaults (Papermill overrides for CI testing)
START_DATE = "2010-01-01"
END_DATE = "2024-12-01"
N_BOOTSTRAP = 100
BLOCK_LENGTH = 20
SEED = 42

# %%
set_global_seeds(SEED)
rng = np.random.default_rng(SEED)

SECTOR_ETFS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Staples",
    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLY": "Discretionary",
}

print(
    f"PCA Sector: {len(SECTOR_ETFS)} ETFs, {START_DATE} to {END_DATE}, {N_BOOTSTRAP} bootstrap samples"
)

# %% [markdown]
# ## 2. Load Sector ETF Data
#
# Load sector ETFs from the canonical ETF Universe parquet file.

# %%
etf_data = load_etfs()

# Filter to sector ETFs and date range
tickers = list(SECTOR_ETFS.keys())
start_dt = date.fromisoformat(START_DATE)
end_dt = date.fromisoformat(END_DATE)
sector_data = (
    etf_data.filter(pl.col("symbol").is_in(tickers))
    .filter(pl.col("timestamp") >= start_dt)
    .filter(pl.col("timestamp") <= end_dt)
    .select(["timestamp", "symbol", "close"])
    .sort(["timestamp", "symbol"])
)

# Pivot to wide format
prices = (
    sector_data.pivot(on="symbol", index="timestamp", values="close")
    .sort("timestamp")
    .to_pandas()
    .set_index("timestamp")
)

# Ensure column order matches SECTOR_ETFS
prices = prices[[t for t in tickers if t in prices.columns]]

# Clean data
prices = prices.dropna(how="all").ffill()

# Calculate returns
returns = prices.pct_change().dropna()
print(
    f"Loaded: {prices.shape[0]} days, {prices.shape[1]} sectors, {returns.shape[0]} return observations"
)

# %% [markdown]
# Daily return distributions reveal whether a few volatile sectors could dominate covariance-PCA.
# The interquartile ranges and whiskers make the scale differences visible without a printed
# `describe()` table.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"], constrained_layout=True)
ax.boxplot(
    (returns * 100).to_numpy(),
    tick_labels=returns.columns,
    showfliers=False,
    boxprops={"color": COLORS["blue"]},
    medianprops={"color": COLORS["amber"], "linewidth": 1.5},
    whiskerprops={"color": COLORS["neutral"]},
    capprops={"color": COLORS["neutral"]},
)
zero_line(ax)
ax.set_xlabel("Sector ETF")
ax.set_ylabel("Daily return (%)")
add_message_title(
    ax,
    "Daily return distribution by sector ETF",
    subtitle="Outliers hidden so the central ranges are comparable",
)
show_with_alt(
    fig,
    "A vertical box plot with one box per sector ETF along the horizontal axis and "
    "daily return in percent on the vertical, with outliers hidden and a dashed line "
    "at zero. The height of each box is that sector's interquartile range and the "
    "whiskers give its wider central spread, so the boxes can be compared against each "
    "other for scale.",
)

# %% [markdown]
# ## 3. PCA on Sector Returns
#
# We use **correlation-PCA** (standardized returns) rather than covariance-PCA. Standardizing
# to unit variance prevents high-volatility sectors (e.g., Energy) from dominating the first
# component. For cross-sectional equity analysis this is the standard choice - see the Scale
# Sensitivity discussion in Section 14.2.
#
# PCA decomposes the covariance matrix as:
#
# $$\Sigma = V \Lambda V^T$$
#
# where $V$ contains eigenvectors (loadings) and $\Lambda$ is the diagonal matrix of
# eigenvalues.

# %%
# Standardize returns (correlation-PCA)
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# Fit PCA - default solver is appropriate for small N (9 sectors);
# use svd_solver='randomized' for N > 500
n_components = min(len(SECTOR_ETFS), 5)
pca = PCA(n_components=n_components, svd_solver="full")
factors = pca.fit_transform(returns_scaled)

# Create factor DataFrame
factor_cols = [f"PC{i + 1}" for i in range(n_components)]
factors_df = pd.DataFrame(factors, index=returns.index, columns=factor_cols)

# %% [markdown]
# ### Variance Decomposition
#
# The fraction of total variance explained by component $k$ is:
#
# $$\text{VE}_k = \frac{\lambda_k}{\sum_{i=1}^{N} \lambda_i}$$

# %%
cum_var = np.cumsum(pca.explained_variance_ratio_)
for i, (share, cumulative) in enumerate(
    zip(pca.explained_variance_ratio_, cum_var, strict=True), 1
):
    print(f"PC{i}: {share:6.1%} of variance, {cumulative:6.1%} cumulative")

# %% [markdown]
# ### Scree Plot
#
# The scree plot (Figure 14.2 in the text) shows eigenvalues in descending order. The
# "elbow" where explained variance levels off suggests how many components to retain.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"], constrained_layout=True)

# Bar chart of individual variance shares
ax.bar(
    range(1, n_components + 1),
    pca.explained_variance_ratio_,
    alpha=0.7,
    color=COLORS["blue"],
    label="Individual",
)

# Cumulative line on same axis
ax.plot(
    range(1, n_components + 1),
    cum_var,
    "o-",
    color=COLORS["amber"],
    label="Cumulative",
)

ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xticks(range(1, n_components + 1))
ax.legend()
add_message_title(
    ax,
    "Variance explained per principal component",
    subtitle="Correlation-PCA on the sector ETF panel",
)
show_with_alt(
    fig,
    "A bar chart with component number on the horizontal axis and share of variance "
    "explained, in percent, on the vertical. Each bar is one component's own share, "
    "ordered from largest to smallest, and a line over the bars gives the running "
    "cumulative share.",
)

# %% [markdown]
# **Reading it**: the height of the first bar against the rest is the question. One
# component carrying most of the variance is what a market factor looks like in a
# sector panel - the sectors move together, and how much they move together is the
# first bar.
#
# Where the cumulative line flattens, each further component is adding little to the
# variance already accounted for, and the "elbow" is a retention heuristic built on
# that. It is a heuristic and not a finding: a component explaining little variance can
# still carry systematic structure, and variance explained says nothing about whether a
# direction is priced or predictive. Nothing in this notebook tests either. The printed
# shares above the chart are the numbers; the shape is what the figure is for.

# %% [markdown]
# ## 4. Loadings Interpretation
#
# Each eigenvector defines a principal component as a linear combination of sector returns.
# The loadings can be interpreted as portfolio weights.

# %%
loadings = pd.DataFrame(
    pca.components_.T,
    index=returns.columns,
    columns=factor_cols,
)
loadings["Sector"] = loadings.index.map(SECTOR_ETFS)

# %% [markdown]
# **Interpretation**: PC1 loadings are uniformly positive - the classic "market factor" where
# all sectors move together. PC2 separates defensive sectors (positive loadings: Utilities,
# Staples) from cyclical sectors (negative loadings: Energy, Financials, Discretionary),
# capturing the sector rotation dimension.

# %% [markdown]
# ## 5. Bootstrap Loading Stability
#
# Are the loadings statistically reliable, or could they be estimation noise? We use a moving-block
# bootstrap to construct 95% confidence intervals while preserving short-run return dependence.
# The default `N_BOOTSTRAP=100` is a teaching budget; increase it for final inference.


# %% [markdown]
# ### Component Matching
#
# PCA signs are arbitrary, and nearby eigenvalues can swap component order across resamples. We use
# maximum absolute loading similarity to match each bootstrap component to the full-sample basis,
# then align its sign.


# %%
def align_components(loadings_new: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match component order and signs to a reference loading basis."""
    if loadings_new.shape != reference.shape:
        raise ValueError("Candidate and reference loading bases must have the same shape")
    similarity = np.abs(reference.T @ loadings_new)
    reference_idx, new_idx = linear_sum_assignment(-similarity)
    aligned = np.zeros_like(loadings_new)
    for ref_col, candidate_col in zip(reference_idx, new_idx, strict=True):
        candidate = loadings_new[:, candidate_col]
        sign = np.sign(reference[:, ref_col] @ candidate) or 1.0
        aligned[:, ref_col] = sign * candidate
    return aligned


# %% [markdown]
# ### Moving-Block Resamples
#
# Twenty-day blocks retain local dependence while resampling the historical sequence. Each draw
# concatenates random contiguous blocks until it reaches the original sample length.


# %%
def moving_block_indices(n_obs: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one moving-block bootstrap index of length `n_obs`."""
    n_blocks = int(np.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs - block_length + 1, size=n_blocks)
    return np.concatenate([np.arange(start, start + block_length) for start in starts])[:n_obs]


# %% [markdown]
# ### Bootstrap Confidence Intervals
#
# Each resample receives its own scaler and PCA fit. The component-matching step prevents a PC2/PC3
# swap from being misread as loading uncertainty.


# %%
def bootstrap_pca(
    returns_df: pd.DataFrame,
    reference_components: np.ndarray,
    n_bootstrap: int = 100,
    n_components: int = 2,
    block_length: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate loading uncertainty with a moving-block bootstrap."""
    if rng is None:
        rng = np.random.default_rng()
    n_obs = len(returns_df)
    n_features = returns_df.shape[1]
    bootstrap_loadings = np.zeros((n_bootstrap, n_features, n_components))

    for b in range(n_bootstrap):
        idx = moving_block_indices(n_obs, block_length, rng)
        sample = returns_df.iloc[idx]

        sample_scaled = StandardScaler().fit_transform(sample)
        pca_boot = PCA(n_components=n_components, svd_solver="full")
        pca_boot.fit(sample_scaled)

        loadings_boot = pca_boot.components_.T
        reference_loadings = reference_components[:n_components].T
        bootstrap_loadings[b] = align_components(loadings_boot, reference_loadings)

    return bootstrap_loadings


# %%
bootstrap_results = bootstrap_pca(
    returns,
    reference_components=pca.components_,
    n_bootstrap=N_BOOTSTRAP,
    n_components=2,
    block_length=BLOCK_LENGTH,
    rng=rng,
)

# Confidence intervals
loading_mean = bootstrap_results.mean(axis=0)
loading_lower = np.percentile(bootstrap_results, 2.5, axis=0)
loading_upper = np.percentile(bootstrap_results, 97.5, axis=0)

# %% [markdown]
# ## 6. Visualization: Loading Confidence Intervals

# %%
fig = make_subplots(rows=1, cols=2, subplot_titles=("PC1: market", "PC2: sector rotation"))
for component_idx, color in enumerate((COLORS["blue"], COLORS["copper"])):
    order = loading_mean[:, component_idx].argsort()
    sector_names = [SECTOR_ETFS.get(returns.columns[i], returns.columns[i]) for i in order]
    fig.add_trace(
        go.Scatter(
            x=loading_mean[order, component_idx],
            y=sector_names,
            mode="markers",
            marker={"size": 10, "color": color},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": loading_upper[order, component_idx] - loading_mean[order, component_idx],
                "arrayminus": loading_mean[order, component_idx]
                - loading_lower[order, component_idx],
                "color": COLORS["neutral"],
                "thickness": 1.5,
            },
            showlegend=False,
        ),
        row=1,
        col=component_idx + 1,
    )
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=COLORS["neutral"],
        row=1,
        col=component_idx + 1,
    )

# %% [markdown]
# Both panels get the same loading range, so an interval's width means the same thing
# on each and the two components can be compared by eye.

# %%
fig.update_layout(
    title="Bootstrap loading intervals for the first two components",
    height=500,
)
loading_limit = 1.1 * max(abs(loading_lower.min()), abs(loading_upper.max()))
fig.update_xaxes(title_text="Loading", range=[-loading_limit, loading_limit])
fig.update_yaxes(title_text="Sector")
show_plotly_with_alt(
    fig,
    "Two panels, one per component, sharing a loading axis. Each row is a sector, "
    "drawn as its bootstrap point estimate with a horizontal confidence interval, "
    "sorted by loading, against a dashed vertical line at zero. The width of an "
    "interval is how much the resamples disagreed about that sector's loading.",
)

# %% [markdown]
# **Reading it**: two things are visible at once. Whether every sector falls on the same
# side of zero on a component says whether that component is a common direction or a
# contrast between groups, and which sectors sit at the two ends of a contrast says what
# the contrast is between. Separately, the width of an interval says how much the
# resamples disagreed - a loading whose interval spans zero is one the data does not
# pin down, and building a portfolio on it is building on an estimate the bootstrap
# cannot distinguish from no exposure at all.

# %% [markdown]
# ## 7. Temporal Stability: Rolling PCA
#
# Factor structure is not constant. Correlations often rise during stress, increasing PC1's share,
# and relax in calm markets. Each point below fits a fresh correlation-PCA on the trailing 252
# observations ending strictly before the plotted date.

# %%
window = 252

rolling_var_explained = []

for i in range(window, len(returns)):
    window_returns = returns.iloc[i - window : i]

    scaled = StandardScaler().fit_transform(window_returns)
    pca_roll = PCA(n_components=2, svd_solver="full")
    pca_roll.fit(scaled)
    rolling_var_explained.append(pca_roll.explained_variance_ratio_)

rolling_var_explained = np.array(rolling_var_explained)

roll_dates = returns.index[window:]

# %% [markdown]
# Plot the two leading variance shares on the same scale so their relative importance remains
# visually honest throughout the sample.

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=roll_dates, y=rolling_var_explained[:, 0], name="PC1", line=dict(color=COLORS["blue"])
    )
)
fig.add_trace(
    go.Scatter(
        x=roll_dates, y=rolling_var_explained[:, 1], name="PC2", line=dict(color=COLORS["copper"])
    )
)
fig.update_layout(
    title="Variance explained by the first two components, rolling window",
    xaxis_title="Date",
    yaxis_title="Variance Explained",
    yaxis_tickformat=".0%",
    height=400,
)
show_plotly_with_alt(
    fig,
    "Two lines against date, one per component, giving the share of variance each "
    "explains within a trailing window. The vertical axis is in percent. The first "
    "component's line moves over a wide range across the sample while the second's "
    "stays much lower.",
)
print(
    f"PC1 rolling variance explained: {rolling_var_explained[:, 0].min():.1%} to "
    f"{rolling_var_explained[:, 0].max():.1%}; PC2: "
    f"{rolling_var_explained[:, 1].min():.1%} to {rolling_var_explained[:, 1].max():.1%}"
)

# %% [markdown]
# **Reading it**: the first component's share is a measure of how much the sectors are
# moving together, so the line is a picture of co-movement over time rather than of
# performance. Read where it rises and where it falls against what was happening, and
# note that stress does not have one signature: an episode where everything sells off
# together and an episode where sectors respond in different directions both count as
# stress and move this line in opposite directions.
#
# Every point is computed from the window ending at that date, so the line describes
# what had already happened. Nothing here forecasts the next window's value.

# %% [markdown]
# ## 8. Factor Score Analysis
#
# PC scores from correlation-PCA are mean-zero standardized projections - they cannot be
# compounded as portfolio returns because PCA centers the input. What they *can* do is
# expose the cross-sectional structure each component captures. We rescale all PC scores by
# a single constant so PC1's daily standard deviation matches the equal-weight sector
# portfolio; this preserves the eigenvalue hierarchy ($\sigma_{\text{PC2}}/\sigma_{\text{PC1}} =
# \sqrt{\lambda_2/\lambda_1}$) while putting PC1 on the same daily scale as the broad market.

# %%
ew_daily_std = returns.mean(axis=1).std()
score_scale = ew_daily_std / factors_df["PC1"].std()
factor_returns = factors_df * score_scale

# %% [markdown]
# A component's daily volatility is $\sqrt{\lambda_k}$, so the ratio
# $\sqrt{\lambda_k / \lambda_1}$ says how large each later component is beside the first.
# The values are printed below rather than stated here, because they follow the
# eigenvalues and change with the sample.

# %%
volatility_ratio = np.sqrt(pca.explained_variance_ / pca.explained_variance_[0])
for i, ratio in enumerate(volatility_ratio, 1):
    print(f"PC{i} daily volatility as a share of PC1's: {ratio:.0%}")

# %% [markdown]
# Prepare two diagnostic series: PC1 vs the equal-weight market (both standardized to unit
# variance so the scatter slope is the correlation), and the rolling 63-day PC1-PC2
# correlation (in-sample orthogonality means the long-run mean is zero by construction).

# %%
ew_returns = returns.mean(axis=1)
pc1_z = (factor_returns["PC1"] - factor_returns["PC1"].mean()) / factor_returns["PC1"].std()
ew_z = (ew_returns - ew_returns.mean()) / ew_returns.std()
roll_corr = factor_returns["PC1"].rolling(63).corr(factor_returns["PC2"])
pc1_market_corr = factor_returns["PC1"].corr(ew_returns)
scatter_min = min(ew_z.min(), pc1_z.min())
scatter_max = max(ew_z.max(), pc1_z.max())

# %% [markdown]
# Build the two-panel subplot grid: scatter on the left, time series on the right.

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "PC1 vs Equal-Weight Market (standardized)",
        "Rolling 63d PC1-PC2 Correlation",
    ),
    column_widths=[0.45, 0.55],
    horizontal_spacing=0.15,
)

# %% [markdown]
# Add the scatter, 45° reference line, and rolling-correlation traces to the grid.

# %%
fig.add_trace(
    go.Scatter(
        x=ew_z,
        y=pc1_z,
        mode="markers",
        marker=dict(size=3, color=COLORS["blue"], opacity=0.4),
        name="Daily",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[scatter_min, scatter_max],
        y=[scatter_min, scatter_max],
        mode="lines",
        line=dict(color=COLORS["neutral"], dash="dash"),
        name="45°",
        showlegend=False,
    ),
    row=1,
    col=1,
)
_ = fig.add_trace(
    go.Scatter(
        x=roll_corr.index,
        y=roll_corr,
        line=dict(color=COLORS["slate"]),
        name="PC1-PC2",
        showlegend=False,
    ),
    row=1,
    col=2,
)

# %% [markdown]
# Apply axis labels and honest correlation limits before rendering.

# %%
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], row=1, col=2)
fig.update_xaxes(
    title_text="Equal-weight return (z)", range=[scatter_min, scatter_max], row=1, col=1
)
fig.update_yaxes(title_text="PC1 score (z)", range=[scatter_min, scatter_max], row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_yaxes(title_text="63-day correlation", range=[-1, 1], row=1, col=2)
fig.update_layout(
    height=420,
    title_text="PC1 against the equal-weight market, and PC1-PC2 correlation over time",
)
show_plotly_with_alt(
    fig,
    "Two panels. The left scatters the first component's daily score against the "
    "equal-weight sector return, both standardized, on equal axes with a dashed "
    "diagonal; a cloud lying on that diagonal means the two are the same series up to "
    "scale. The right plots the rolling correlation between the first two components "
    "against date, on an axis from minus one to one with a line at zero.",
)
print(
    f"PC1 against equal-weight market: correlation {pc1_market_corr:.3f}. "
    f"Rolling PC1-PC2 correlation: mean {roll_corr.mean():+.3f}, "
    f"range {roll_corr.min():+.2f} to {roll_corr.max():+.2f}"
)

# %% [markdown]
# **Reading it**: the left panel asks what the first component *is*. If its score is
# nearly the equal-weight return, then PCA has recovered the market and given it a name,
# rather than found something the equal-weight portfolio does not already capture. The
# printed correlation is the number; the diagonal is what makes it legible.
#
# The right panel is about a property that is easy to over-trust. PCA makes the
# components orthogonal *over the sample it was fitted on*, and that is a statement
# about the whole period, not about any window inside it. The rolling correlation shows
# how far short windows depart from zero. Anything that relies on the components being
# uncorrelated - a risk decomposition, a hedge built from one against the other - needs
# them re-fitted on the window it will be used in, because a single in-sample
# decomposition does not deliver orthogonality inside that window.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Standardize before decomposing, or the loudest sector wins.** Correlation-PCA
#    puts every sector on unit variance first, so the components describe co-movement
#    rather than which sector happens to be most volatile. The box plot at the top is
#    why: the sectors differ enough in spread for that choice to change the answer.
# 2. **A loading is an estimate, and the bootstrap says how good.** The interval, not
#    the point, is what a portfolio decision should read. An interval spanning zero is
#    an exposure the data does not establish.
# 3. **The first component's share is a measure of co-movement, and it moves.** A single
#    full-sample number hides that; the rolling window is what shows it, and it rises
#    both when everything sells off together and falls when sectors respond in different
#    directions. Neither is "the stress signature".
# 4. **Orthogonality is a property of the fitting sample, not of every window in it.**
#    The rolling correlation between the first two components shows how far a short
#    window departs from zero, which is what anything built on their independence has to
#    contend with.
# 5. **This is a description, not a forecast.** PCA on the full sample says what the
#    covariance structure was. Turning that into a return prediction needs a second
#    stage, which is the subject of `04_ipca` and `05_rp_pca`.
#
# ### PCA in the Two-Step Framework
#
# Everything above is **Stage 1**. To turn it into a return forecast, a
# Stage 2 factor-premium forecaster is required (see Figure 14.10 for the
# full catalog). PCA with a sample-mean Stage 2 reduces to a per-asset
# historical-mean predictor - a useful sanity-check baseline but not a
# forecaster in any meaningful sense. Non-trivial cross-sectional ranking
# emerges only when Stage 2 conditions on the factor path (AR(1), EWMA, or
# richer ML forecasters). The IPCA notebook demonstrates the full pipeline
# end-to-end.
#
# **Next Steps**:
# - For eigenportfolio construction on a broader stock universe, see [`02_eigenportfolios`](02_eigenportfolios.ipynb)
# - For Stage 1 + 2 + 3 end-to-end with characteristic-driven loadings, see [`04_ipca`](04_ipca.ipynb)
# - For production PCA with walk-forward CV on ETFs, see [`11_latent_factors`](../case_studies/etfs/11_latent_factors.ipynb)
