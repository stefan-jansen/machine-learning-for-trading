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
# # Factor-based regime detection
#
# **Chapter 1 · §1.4 Market Regimes: Change Is the Constant**
#
# **Docker image**: `ml4t`
#
# Markets do not behave the same way in every year. The same value strategy that earns a
# steady premium through a long expansion can lose money for a decade, and the same
# momentum strategy can give back years of gains in a single quarter. A **regime** is a
# stretch of time over which the joint behaviour of returns is stable enough to be treated
# as one environment: the averages, the volatilities and the correlations hold roughly
# still inside it and shift when it ends.
#
# Nobody publishes a regime label. This notebook estimates one from the data with a
# **Gaussian mixture model** (GMM), which treats each month's vector of factor returns as
# a draw from one of a small number of multivariate normal distributions and infers both
# the distributions and which one each month most likely came from. The inputs are monthly
# returns to nine factor strategies from AQR's Century of Factor Premia dataset, which
# starts in 1927.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Fit a Gaussian mixture model to a panel of monthly factor returns and read off the
#   cluster it assigns to each month.
# - Choose how many clusters to keep by comparing three criteria that measure different
#   things, and say what each one rewards.
# - Read a regime timeline against dated market events and judge whether the clusters
#   correspond to periods a reader would recognise.
# - Measure what each factor earned inside each cluster, and say what that implies about
#   which factors diversify each other when conditions deteriorate.
# - Explain why a label fitted on the whole sample cannot be used as a trading signal.
#
# ## Book reference
#
# Chapter 1, Section 1.4, "Market Regimes: Change Is the Constant". `macro_regimes` is the
# companion notebook and estimates regimes from macroeconomic indicators instead.
#
# ## Prerequisites
#
# - Monthly return series and standardization (subtracting a mean, dividing by a standard
#   deviation) so that series with different scales contribute comparably to a distance.
# - The AQR Century of Factor Premia parquet under `data/aqr_factors/`. Download it with
#   `uv run python data/factors/aqr_download.py` if it is missing.
#
# ## What this notebook establishes, and what it does not
#
# The model is fitted on the whole history and the labels are assigned back over that same
# history. What comes out is a description of how the factor space partitions, and a way
# of seeing where dated events fall inside that partition. Every month's label is informed
# by every other month, including later ones, so the labels describe the sample rather than
# forecast it. Building regime labels a strategy could have acted on requires walk-forward
# fitting on information available at each date; Chapter 6 onward introduces that machinery
# and the case-study chapters apply it.

# %% [markdown]
# ## Setup
#
# The AQR provider logs its download and cache decisions through `structlog`, which writes
# to the notebook rather than through the standard library's logging module. Raising its
# threshold to `WARNING` keeps the data-loading cells readable while leaving anything that
# reports a problem visible.

# %%
"""Factor-based regime detection - Gaussian mixture regimes over AQR factor returns."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import structlog
from IPython.display import Markdown, display
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from ml4t.data.providers import AQRFactorProvider
from ml4t.diagnostic.metrics import sharpe_ratio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import utils.style as style
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

COLORS = style.COLORS

# %% tags=["parameters"]
# Production defaults (Papermill overrides these when the test suite runs the notebook)
SEED = 42

# %% [markdown]
# ### Settings, and what each one decides
#
# `SEED` fixes the initialization of every estimator below. A Gaussian mixture is fitted by
# expectation-maximization from a random start, so two runs from different starts can land
# on different local optima and swap which cluster is numbered 0; fixing the seed is what
# makes the figures in this notebook reproduce.
#
# `N_REGIMES_GRID` is the set of cluster counts to fit and compare. Two is the smallest
# number that partitions anything. Six is where the search stops: the point of the sweep is
# to show how the criteria move as clusters are added, and by six they have already
# separated.
#
# `MONTHS_PER_YEAR` annualizes monthly statistics. Multiply a mean monthly return by it and
# a monthly standard deviation by its square root.
#
# `ROLLING_VOL_MONTHS` is the width of the moving window used to draw a volatility series.
# Twelve months is short enough to move inside a regime and long enough that a single month
# does not dominate the estimate.

# %%
N_REGIMES_GRID = [2, 3, 4, 5, 6]
MONTHS_PER_YEAR = 12
ROLLING_VOL_MONTHS = 12

OUTPUT_DIR = get_output_dir(1, "factor_regimes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_global_seeds(SEED)

# %% [markdown]
# ## Load the factor panel
#
# The AQR Century of Factor Premia dataset holds monthly returns to long-short factor
# strategies built inside several asset classes, plus the returns to holding each asset
# class outright. The provider returns one row per month and one column per strategy.

# %%
aqr_raw = AQRFactorProvider().fetch("century_premia")

date_range = aqr_raw.select(
    pl.col("timestamp").min().alias("first"), pl.col("timestamp").max().alias("last")
)
print(f"AQR Century of Factor Premia: {aqr_raw.height} months, {aqr_raw.width - 1} series")
print(f"Covering {date_range['first'][0]:%Y-%m} to {date_range['last'][0]:%Y-%m}")

# %% [markdown]
# ### The nine series used here
#
# Four of them are cross-asset long-short strategies, applied across equity indices, bonds,
# currencies and commodities at once:
#
# - **Value** buys what is cheap relative to a fundamental anchor and sells what is
#   expensive.
# - **Momentum** buys what has risen over the past year and sells what has fallen.
# - **Carry** buys what pays a high yield and sells what pays a low one.
# - **Defensive** buys low-risk assets and sells high-risk ones.
#
# Two apply the value and momentum definitions inside the US stock market alone. The last
# three are the returns to simply holding each asset class - equity indices, fixed income,
# commodities - rather than a long-short strategy, and they are what places a month in a
# risk environment rather than a style environment.

# %%
FACTOR_COLUMNS = [
    "All asset classes Value",
    "All asset classes Momentum",
    "All asset classes Carry",
    "All asset classes Defensive",
    "US Stock Selection Value",
    "US Stock Selection Momentum",
    "Equity indices Market",
    "Fixed income Market",
    "Commodities Market",
]
EQUITY_COLUMN = "Equity indices Market"

# Short names for the figures and tables below; the full AQR column names do not fit on an
# axis.
SHORT_NAMES = {
    "All asset classes Value": "Value",
    "All asset classes Momentum": "Momentum",
    "All asset classes Carry": "Carry",
    "All asset classes Defensive": "Defensive",
    "US Stock Selection Value": "US Value",
    "US Stock Selection Momentum": "US Momentum",
    "Equity indices Market": "Equity",
    "Fixed income Market": "Bonds",
    "Commodities Market": "Commodities",
}

# %% [markdown]
# ### What is actually in the panel
#
# Clustering treats every month as a point in nine dimensions, so a month is usable only
# when all nine series are quoted. The table below reports, for each series, the month it
# starts, how many observations it has, and its mean and standard deviation over the raw
# file. Read it before anything computed from the panel: it says which series shortens the
# usable history and how far apart the series are in scale, which is why they are
# standardized before the distance is taken.

# %%
coverage = pd.DataFrame(
    [
        {
            "Series": SHORT_NAMES[column],
            "First month": aqr_raw.filter(pl.col(column).is_not_null())["timestamp"].min(),
            "Months quoted": aqr_raw.select(pl.col(column).is_not_null().sum()).item(),
            "Mean (% / month)": aqr_raw.select(pl.col(column).mean()).item() * 100,
            "Std (% / month)": aqr_raw.select(pl.col(column).std()).item() * 100,
        }
        for column in FACTOR_COLUMNS
    ]
).set_index("Series")
coverage["First month"] = pd.to_datetime(coverage["First month"]).dt.strftime("%Y-%m")
coverage.style.format({"Mean (% / month)": "{:.2f}", "Std (% / month)": "{:.2f}"})

# %% [markdown]
# ## Prepare the panel for clustering
#
# Months in which any of the nine series is unquoted are dropped rather than filled. A
# forward fill would repeat the previous month's return, and repeating a return is not a
# missing observation recovered - it is an invented one, and a mixture model would treat it
# as evidence for a cluster.
#
# The nine series are then standardized to zero mean and unit variance. The coverage table
# above shows why: the widest series in the panel has several times the monthly standard
# deviation of the narrowest, and a Euclidean distance is dominated by whichever coordinate
# moves furthest. Standardizing puts every series on the same footing, so the partition
# reflects the direction the panel moved in rather than which column happened to be measured
# on the largest scale.

# %%
factors_pl = aqr_raw.select(["timestamp", *FACTOR_COLUMNS]).sort("timestamp").drop_nulls()

factors_df = factors_pl.select(FACTOR_COLUMNS).to_pandas()
factors_df.index = pd.DatetimeIndex(factors_pl["timestamp"].to_pandas())

factors_scaled = StandardScaler().fit_transform(factors_df)

start_year = factors_df.index.min().year
end_year = factors_df.index.max().year
print(f"Complete months: {len(factors_df)} ({start_year} to {end_year})")

# %% [markdown]
# ## Fit a grid of mixture models
#
# There is no test that returns the true number of regimes, so the count is a modelling
# choice made by comparing candidates. The helper below fits one mixture per candidate
# count and records three quantities for each.
#
# **BIC**, the Bayesian information criterion, is the fitted log-likelihood penalised by the
# number of free parameters times the log of the sample size. **AIC**, the Akaike
# information criterion, applies the same idea with a constant penalty of two per parameter.
# Both are better when lower, and BIC's penalty grows with the sample, so on a long panel it
# is far more reluctant than AIC to buy an extra cluster. A full-covariance mixture spends
# parameters quickly - each additional cluster costs a mean vector and a covariance matrix
# over nine dimensions - which is what the two criteria are pricing differently.
#
# The **silhouette score** measures something else entirely. For each month it compares the
# average distance to the other months in its own cluster against the average distance to
# the months of the nearest other cluster, and reports the normalized difference; the score
# is the average over months. It runs from -1 to 1, it is better when higher, and a negative
# value means the typical month sits closer to a different cluster than to its own. It is a
# geometric statement about separation and knows nothing about likelihood.
#
# K-means is fitted alongside for the same counts. It partitions on distance to a centroid
# with no covariance and no probabilities, so putting its silhouette beside the mixture's
# says how much of the separation is coming from the shape the mixture is allowed to fit.
#
# The last two columns count things none of the three criteria look at: how often consecutive
# months carry different labels, and how many months the emptiest cluster holds. A criterion
# can only say how well a partition fits. Whether the partition holds still long enough to be
# called a regime, and whether every cluster in it has enough months to say anything about,
# are separate questions, and these two counts are the cheapest way to ask them.


# %%
@dataclass(frozen=True)
class GmmFitResult:
    """One fitted mixture and the three quantities used to compare it against the others."""

    model: GaussianMixture
    labels: np.ndarray
    probabilities: np.ndarray
    bic: float
    aic: float
    silhouette: float


# %%
def fit_gmm_grid(
    x: np.ndarray,
    n_components_list: Iterable[int],
    random_state: int = SEED,
) -> dict[int, GmmFitResult]:
    """Fit one full-covariance mixture per requested cluster count."""
    results: dict[int, GmmFitResult] = {}
    for n in n_components_list:
        model = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=random_state,
            n_init=10,
            reg_covar=1e-6,
        )
        model.fit(x)
        labels = model.predict(x)
        results[n] = GmmFitResult(
            model=model,
            labels=labels,
            probabilities=model.predict_proba(x),
            bic=float(model.bic(x)),
            aic=float(model.aic(x)),
            silhouette=float(silhouette_score(x, labels)),
        )
    return results


# %%
gmm_grid = fit_gmm_grid(factors_scaled, N_REGIMES_GRID)

kmeans_labels = {
    n: KMeans(n_clusters=n, random_state=SEED, n_init=10).fit_predict(factors_scaled)
    for n in N_REGIMES_GRID
}


def switch_count(labels: np.ndarray) -> int:
    """How many times consecutive months carry different labels."""
    return int(np.sum(np.diff(labels) != 0))


selection = pd.DataFrame(
    {
        "Clusters": N_REGIMES_GRID,
        "BIC": [gmm_grid[n].bic for n in N_REGIMES_GRID],
        "AIC": [gmm_grid[n].aic for n in N_REGIMES_GRID],
        "Silhouette (mixture)": [gmm_grid[n].silhouette for n in N_REGIMES_GRID],
        "Silhouette (k-means)": [
            silhouette_score(factors_scaled, kmeans_labels[n]) for n in N_REGIMES_GRID
        ],
        "Switches": [switch_count(gmm_grid[n].labels) for n in N_REGIMES_GRID],
        "Smallest cluster (months)": [
            int(np.bincount(gmm_grid[n].labels).min()) for n in N_REGIMES_GRID
        ],
    }
).set_index("Clusters")
selection.style.format(
    {
        "BIC": "{:,.0f}",
        "AIC": "{:,.0f}",
        "Silhouette (mixture)": "{:.3f}",
        "Silhouette (k-means)": "{:.3f}",
    }
)

# %% [markdown]
# ### Reading the three criteria against each other
#
# The bars below draw three of the table's columns. Both information criteria are drawn on a
# zoomed vertical axis, because the differences between candidate models are small relative
# to their level and would be invisible on a scale that started at zero; the silhouette
# panel is drawn on its natural scale, which includes zero.


# %%
def plot_selection_criterion(ax: Axes, values: list[float], label: str, best: str) -> None:
    """Draw one criterion as bars, highlighting the count it favours."""
    bars = ax.bar(N_REGIMES_GRID, values, color=COLORS["recede"])
    chosen = int(np.argmin(values)) if best == "min" else int(np.argmax(values))
    bars[chosen].set_color(COLORS["amber"])
    ax.set_xlabel("Clusters")
    ax.set_ylabel(label)
    ax.set_xticks(N_REGIMES_GRID)


# %%
fig, axes = plt.subplots(1, 3, figsize=style.FIGSIZE["triple_h_tall"])

plot_selection_criterion(axes[0], selection["BIC"].tolist(), "BIC (lower is better)", "min")
plot_selection_criterion(axes[1], selection["AIC"].tolist(), "AIC (lower is better)", "min")
plot_selection_criterion(
    axes[2], selection["Silhouette (mixture)"].tolist(), "Silhouette (higher is better)", "max"
)
for ax, values in ((axes[0], selection["BIC"]), (axes[1], selection["AIC"])):
    span = values.max() - values.min()
    ax.set_ylim(values.min() - 0.15 * span, values.max() + 0.15 * span)

style.add_message_title(
    axes[0],
    "BIC, AIC and silhouette do not pick the same number of regimes",
    subtitle="Highlighted bar is the count each criterion favours; note the zoomed axes on BIC and AIC",
)
style.show_with_alt(
    fig,
    "Three bar charts over cluster counts two to six. BIC rises with the count, AIC falls "
    "with it, and the silhouette score falls sharply after the smallest count and hovers "
    "near zero at the larger ones.",
)

# %%
bic_choice = int(selection["BIC"].idxmin())
aic_choice = int(selection["AIC"].idxmin())
silhouette_choice = int(selection["Silhouette (mixture)"].idxmax())
display(
    Markdown(
        f"BIC is lowest at **{bic_choice} clusters** and the silhouette score is highest at "
        f"**{silhouette_choice}**, while AIC keeps falling out to **{aic_choice}**, the "
        "largest count fitted."
    )
)

# %% [markdown]
# When the criteria disagree, the choice is made on what the model is for. AIC targets
# predictive likelihood and keeps buying clusters as long as they raise it, and the last two
# columns say what it is buying at the larger counts: a partition that changes label more
# often, containing at least one cluster with too few months to describe. A sliver of the
# sample with its own covariance improves the fit and is not a regime anyone can name.
#
# BIC's sample-scaled penalty and the silhouette's separation test both ask for clusters
# distinct enough to survive being described, which is what a regime has to be if it is going
# to appear in a risk report. The rest of this notebook works with the two-cluster model on
# that basis, and the timelines in the next section show what the larger counts do with the
# clusters they add.

# %%
N_REGIMES_SELECTED = 2

# %% [markdown]
# ## What each cluster count partitions
#
# Each band below is one fitted model. A month is shaded in the row of the cluster it was
# assigned to, so a solid horizontal stretch is a period the model held in one regime and a
# speckled stretch is one it switched through repeatedly. Comparing the bands shows what the
# extra clusters buy: whether an added cluster carves out a recognisable era, or whether it
# scatters across the whole century.


# %%
def decade_ticks(dates: pd.DatetimeIndex) -> tuple[list[int], list[str]]:
    """Positions and labels for the first month of each decade in a monthly index."""
    years = dates.year.tolist()
    ticks = [
        j for j, year in enumerate(years) if year % 10 == 0 and (j == 0 or years[j - 1] != year)
    ]
    return ticks, [str(years[j]) for j in ticks]


# %%
def draw_swimlanes(ax: Axes, rows: np.ndarray, n_rows: int, cmap: ListedColormap) -> None:
    """Shade each month in the row given by *rows*, one row per regime."""
    matrix = np.zeros((n_rows, len(rows)))
    matrix[rows, np.arange(len(rows))] = 1
    ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=(0, len(rows), -0.5, n_rows - 0.5),
        interpolation="nearest",
        origin="lower",
    )
    for boundary in range(1, n_rows):
        ax.axhline(y=boundary - 0.5, color=COLORS["bg_light"], linewidth=1.5)


# %%
lane_cmap = ListedColormap([COLORS["bg_light"], COLORS["blue"]])
ticks, tick_labels = decade_ticks(factors_df.index)

fig, axes = plt.subplots(
    len(N_REGIMES_GRID),
    1,
    figsize=(style.PAGE_WIDTH, 1.0 * len(N_REGIMES_GRID) + 1.0),
    sharex=True,
)
for ax, n in zip(axes, N_REGIMES_GRID):
    draw_swimlanes(ax, gmm_grid[n].labels, n, lane_cmap)
    ax.set_yticks([])
    ax.set_ylabel(f"K = {n}", rotation=0, ha="right", va="center")

axes[-1].set_xticks(ticks)
axes[-1].set_xticklabels(tick_labels)
axes[-1].set_xlabel("Year")
style.add_message_title(
    axes[0],
    "Adding clusters makes the timeline switch more often, not cleaner",
    subtitle="One panel per fitted model; a month is shaded in the row of the cluster it was assigned",
)
style.show_with_alt(
    fig,
    "Five stacked panels, one per cluster count. The two-cluster panel shows one row "
    "occupied most of the time and the other taking short scattered stretches. Each panel "
    "below it is more finely speckled than the one above, with no row holding a long "
    "uninterrupted stretch by the largest count.",
)

# %% [markdown]
# No band settles down as clusters are added. An added cluster does not carve out an era the
# earlier models had merged; it takes months from across the whole century, so each panel is
# more finely speckled than the one above it. A mixture has no notion of time and nothing in
# it prefers a month to keep its neighbour's label, so there is no mechanism by which more
# clusters could produce longer stretches. That absence is the reason the last section of
# the section on how long the regimes last points at a hidden Markov model, which supplies
# exactly the mechanism this one lacks.

# %% [markdown]
# ## The two-cluster model: risk-on and risk-off
#
# The mixture returns cluster numbers, not names, and the numbering carries no meaning - a
# different seed can swap it. To describe the two clusters they have to be identified by
# something outside the model, and the natural choice is the series that says what the
# market as a whole was doing: the average return to holding equity indices inside each
# cluster. **Risk-on** and **risk-off** are the market's ordinary terms for the environments
# that result, meaning periods when investors were adding exposure to risky assets and
# periods when they were shedding it.
#
# Naming the clusters this way uses the whole sample, exactly as fitting them did. It is a
# description of what the partition turned out to be, not a rule that could have been
# applied in 1929.

# %%
labels_2 = gmm_grid[N_REGIMES_SELECTED].labels
equity_returns = factors_df[EQUITY_COLUMN]

mean_equity_by_cluster = equity_returns.groupby(labels_2).mean()
risk_on_cluster = int(mean_equity_by_cluster.idxmax())
risk_off_cluster = int(mean_equity_by_cluster.idxmin())

regime_name = pd.Series(
    np.where(labels_2 == risk_on_cluster, "Risk-on", "Risk-off"),
    index=factors_df.index,
    name="regime",
)

# %% [markdown]
# ### The timeline against dated events
#
# The upper panel is the same swim-lane band as above, restricted to the two-cluster model,
# with seven dated episodes marked. The lower panel is the cumulative product of the equity
# index returns on a logarithmic scale, drawn in the colour of the regime each month was
# assigned to. A logarithmic scale is what makes the 1930s comparable to the 2010s: equal
# vertical distances are equal percentage moves, so a fall of a given percentage takes up the
# same space on the axis wherever in the century it happens.

# %%
HISTORICAL_EVENTS = {
    1929: ("Great Crash", "top"),
    1937: ("Recession", "bottom"),
    1973: ("Oil crisis", "top"),
    1987: ("Black Monday", "bottom"),
    2000: ("Dot-com peak", "top"),
    2008: ("Global financial crisis", "bottom"),
    2020: ("COVID-19", "top"),
}


# %%
def mark_events(ax: Axes, years: list[int]) -> None:
    """Draw a rule and a label at the first month of each dated episode."""
    for event_year, (event_label, side) in HISTORICAL_EVENTS.items():
        position = next((j for j, year in enumerate(years) if year == event_year), None)
        if position is None:
            continue
        ax.axvline(x=position, color=COLORS["neutral"], alpha=0.6, linewidth=1.0)
        ax.annotate(
            event_label,
            xy=(position, 1.7 if side == "top" else -0.7),
            ha="center",
            va="bottom" if side == "top" else "top",
            fontsize=7,
            color=COLORS["neutral"],
        )


# %%
def plot_by_regime(ax: Axes, values: np.ndarray, in_risk_on: np.ndarray) -> None:
    """Draw one line whose every segment takes the colour of the month it starts in.

    Colouring by masking each regime into its own line would drop the segment that spans a
    transition, and would draw nothing at all for a one-month episode, which is most of
    them here.
    """
    y = np.asarray(values, dtype=float)
    points = np.column_stack([np.arange(len(y)), y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    drawable = np.isfinite(segments[:, :, 1]).all(axis=1)
    colours = np.where(in_risk_on[:-1], COLORS["recede"], COLORS["blue"])
    ax.add_collection(LineCollection(segments[drawable], colors=colours[drawable], linewidths=1.2))
    finite = y[np.isfinite(y)]
    ax.set_xlim(0, len(y) - 1)
    ax.set_ylim(finite.min() * 0.9, finite.max() * 1.1)


# %%
fig, (ax_band, ax_equity) = plt.subplots(
    2, 1, figsize=style.FIGSIZE["dual_v"], height_ratios=[1, 2], sharex=True
)

rows = np.where(labels_2 == risk_on_cluster, 1, 0)
draw_swimlanes(ax_band, rows, 2, lane_cmap)
ax_band.set_yticks([0, 1])
ax_band.set_yticklabels(["Risk-off", "Risk-on"])
ax_band.set_ylim(-0.5, 1.5)
mark_events(ax_band, factors_df.index.year.tolist())

cumulative = (1 + equity_returns).cumprod()
in_risk_on = labels_2 == risk_on_cluster
plot_by_regime(ax_equity, cumulative.to_numpy(), in_risk_on)
ax_equity.set_yscale("log")
ax_equity.set_ylabel("Growth of 1 unit invested (log scale)")
ax_equity.set_xlabel("Year")
ax_equity.set_xticks(ticks)
ax_equity.set_xticklabels(tick_labels)
ax_equity.grid(True, alpha=0.3)

style.add_message_title(
    ax_band,
    "Risk-off months come in short bursts, not in long bear markets",
    subtitle="Upper: assigned regime. Lower: cumulative equity index return, coloured by regime",
)
style.show_with_alt(
    fig,
    "A two-row regime band above a rising cumulative return curve on a log scale. The "
    "risk-off row is occupied in many short bursts spread across the whole century rather "
    "than in a few long blocks, and the curve is drawn dark through those same stretches.",
)

# %% [markdown]
# ## Volatility inside each regime
#
# A regime label is only worth carrying if it corresponds to something a risk manager would
# act on, and the first thing to check is whether the two environments differ in how much
# the market moves. The series below is the standard deviation of the equity index return
# over a trailing twelve-month window, annualized by multiplying by the square root of
# twelve, drawn in the colour of the regime assigned to each month. The dashed rules are the
# average of that rolling series within each regime.

# %%
rolling_vol = equity_returns.rolling(ROLLING_VOL_MONTHS).std() * np.sqrt(MONTHS_PER_YEAR)

rolling_vol_by_regime = (
    rolling_vol.groupby(regime_name)
    .agg(["mean", "median", "max"])
    .rename(columns={"mean": "Mean (%)", "median": "Median (%)", "max": "Max (%)"})
    .mul(100)
    .loc[["Risk-on", "Risk-off"]]
)
rolling_vol_by_regime.index.name = "Regime"
rolling_vol_by_regime.style.format("{:.1f}")

# %%
fig, (ax_band, ax_vol) = plt.subplots(
    2, 1, figsize=style.FIGSIZE["dual_v"], height_ratios=[1, 3], sharex=True
)

draw_swimlanes(ax_band, rows, 2, lane_cmap)
ax_band.set_yticks([0, 1])
ax_band.set_yticklabels(["Risk-off", "Risk-on"])
ax_band.set_ylim(-0.5, 1.5)

plot_by_regime(ax_vol, rolling_vol.to_numpy() * 100, in_risk_on)
for regime, colour in (("Risk-on", COLORS["copper"]), ("Risk-off", COLORS["amber"])):
    ax_vol.axhline(
        rolling_vol_by_regime.loc[regime, "Mean (%)"],
        color=colour,
        linestyle="--",
        linewidth=1.2,
        label=f"{regime} average",
    )
ax_vol.set_ylabel(f"Trailing {ROLLING_VOL_MONTHS}-month volatility, annualized (%)")
ax_vol.set_xlabel("Year")
ax_vol.set_xticks(ticks)
ax_vol.set_xticklabels(tick_labels)
ax_vol.legend(loc="upper right")
ax_vol.grid(True, alpha=0.3)

style.add_message_title(
    ax_band,
    "The risk-off cluster sits on the higher-volatility stretches of the century",
    subtitle=f"Trailing {ROLLING_VOL_MONTHS}-month standard deviation of the equity index, "
    f"annualized by the square root of {MONTHS_PER_YEAR}",
)
style.show_with_alt(
    fig,
    "A two-row regime band above a rolling volatility series in percent. The dark risk-off "
    "segments concentrate on the peaks of the volatility series, and the risk-off average "
    "rule sits above the risk-on one.",
)

# %% [markdown]
# ### Figure 1.5 inputs
#
# The print version of Figure 1.5 is drawn by
# `book/01_process_is_edge/figures/scripts/generate_figure_1_5_factor_regimes_volatility.py`
# in the book repository. It reads the arrays written below, so the book build renders the
# figure from this notebook's fit rather than re-estimating the mixture.

# %%
ARTIFACT_DIR = OUTPUT_DIR / "figure_1_5"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    ARTIFACT_DIR / "inputs.npz",
    dates=factors_df.index.astype("datetime64[ns]").astype("int64"),
    labels_2=np.asarray(labels_2, dtype=np.int64),
    good_regime=np.int64(risk_on_cluster),
    rolling_vol=rolling_vol.to_numpy(dtype=float),
    risk_on_mean_vol=float(rolling_vol_by_regime.loc["Risk-on", "Mean (%)"]) / 100,
    risk_off_mean_vol=float(rolling_vol_by_regime.loc["Risk-off", "Mean (%)"]) / 100,
    start_year=np.int64(start_year),
    end_year=np.int64(end_year),
)

# %% [markdown]
# ## What each factor earned in each regime
#
# The volatility check says the two environments differ in how much the market moved. The
# more useful question for portfolio construction is whether the strategies behave
# differently in them, because a strategy that earns its premium in both is a diversifier
# and one that only earns it in the calm environment is leverage on the market in disguise.
#
# Each bar below is the mean monthly return of one series over the months assigned to one
# regime, multiplied by twelve. This is an average within a set of months, not the return of
# a strategy that switched between them - no rule available in real time would have produced
# this partition.

# %%
annualized_by_regime = (
    factors_df.groupby(regime_name).mean().mul(MONTHS_PER_YEAR * 100).rename(columns=SHORT_NAMES)
).loc[["Risk-on", "Risk-off"]]

# %%
fig, ax = plt.subplots(figsize=style.FIGSIZE["single_tall"])

x = np.arange(annualized_by_regime.shape[1])
width = 0.38
ax.bar(
    x - width / 2,
    annualized_by_regime.loc["Risk-on"],
    width,
    label="Risk-on",
    color=COLORS["recede"],
)
ax.bar(
    x + width / 2,
    annualized_by_regime.loc["Risk-off"],
    width,
    label="Risk-off",
    color=COLORS["blue"],
)
ax.set_xticks(x)
ax.set_xticklabels(annualized_by_regime.columns, rotation=30, ha="right")
ax.set_ylabel("Mean return within the regime, annualized (%)")
style.zero_line(ax)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

style.add_message_title(
    ax,
    "Value and bonds hold up in the risk-off months; carry and defensive do not",
    subtitle="Mean monthly return within each regime times twelve, by series",
)
style.show_with_alt(
    fig,
    "Paired bars per factor. Value and bonds have taller risk-off bars than risk-on bars; "
    "momentum, equity, carry and defensive have taller risk-on bars, with carry and "
    "defensive falling below zero in the risk-off months.",
)

# %%
annualized_by_regime.T.style.format("{:+.1f}")

# %% [markdown]
# The two strategies a calm market reads as safe income - carry, which collects a yield
# spread, and defensive, which is long low-risk assets - are the ones that stop paying when
# conditions deteriorate. Value and bonds go the other way and earn more.
#
# A strategy's average return over a full sample therefore says nothing about whether it
# will be there in the environment a portfolio needs it in. Finding out means conditioning
# on the environment and looking, which is what makes a regime label worth estimating even
# when nothing can act on it.

# %% [markdown]
# ## Regime statistics
#
# The table below characterises each regime through the equity index alone. Every statistic
# is computed on one stream: the monthly returns of the months the model assigned to that
# regime, concatenated in date order with the months in the other regime removed.
#
# For the mean, the standard deviation and the Sharpe ratio that is an ordinary conditional
# statistic - the average of a set of months does not depend on their order. **Maximum
# drawdown** is different, because it is a path statistic: it compounds the stream, tracks
# its running high-water mark, and reports the largest percentage fall from that mark to a
# later point. Compounding a stream whose gaps have been closed up is not the same as
# compounding the index, so the number below describes an investor who held equities during
# one regime and held nothing during the other. The section after the table shows how far
# apart the two readings are.


# %%
def drawdown_path(returns: pd.Series) -> pd.Series:
    """Fractional decline from the running high-water mark of the compounded stream."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


# %%
regime_statistics = pd.DataFrame(
    [
        {
            "Regime": regime,
            "Months": int(len(returns)),
            "Share of sample": len(returns) / len(equity_returns),
            "Return, annualized": returns.mean() * MONTHS_PER_YEAR,
            "Volatility, annualized": returns.std() * np.sqrt(MONTHS_PER_YEAR),
            "Sharpe ratio": sharpe_ratio(returns, periods_per_year=MONTHS_PER_YEAR),
            "Max drawdown": float(drawdown_path(returns).min()),
        }
        for regime, returns in (
            (name, equity_returns[regime_name == name]) for name in ("Risk-on", "Risk-off")
        )
    ]
).set_index("Regime")
regime_statistics.style.format(
    {
        "Share of sample": "{:.1%}",
        "Return, annualized": "{:+.1%}",
        "Volatility, annualized": "{:.1%}",
        "Sharpe ratio": "{:.2f}",
        "Max drawdown": "{:.1%}",
    }
)

# %%
pooled_vol_ratio = (
    regime_statistics.loc["Risk-off", "Volatility, annualized"]
    / regime_statistics.loc["Risk-on", "Volatility, annualized"]
)
rolling_vol_ratio = (
    rolling_vol_by_regime.loc["Risk-off", "Mean (%)"]
    / rolling_vol_by_regime.loc["Risk-on", "Mean (%)"]
)
display(
    Markdown(
        f"Risk-off volatility is **{pooled_vol_ratio:.2f}x** risk-on volatility measured on "
        f"the pooled monthly returns, against **{rolling_vol_ratio:.2f}x** measured as the "
        "average of the trailing twelve-month series."
    )
)

# %% [markdown]
# The two ratios answer different questions and a reader should expect them to differ. The
# rolling series smooths each month against the eleven around it, so a single extreme month
# raises twelve windows a little; the pooled standard deviation lets it enter once, at full
# size. The pooled figure is the one that describes what a portfolio holding through the
# regime experiences, and the rolling figure is the one a monitoring dashboard would show.
# Reporting either alone would be defensible; reporting one and calling it "the" volatility
# ratio would not.

# %% [markdown]
# ### What the restricted drawdown actually measures
#
# The drawdown reported for the risk-off regime is large, and it is worth reading where it
# came from before quoting it. The cell below finds the high-water mark of the restricted
# stream and the month it fell furthest below it, and puts the drawdown of the index itself
# over the same span beside it.

# %%
risk_off_returns = equity_returns[regime_name == "Risk-off"]
risk_off_drawdown = drawdown_path(risk_off_returns)
trough_month = risk_off_drawdown.idxmin()
peak_month = (1 + risk_off_returns).cumprod().loc[:trough_month].idxmax()

index_over_span = equity_returns.loc[peak_month:trough_month]
index_drawdown = float(drawdown_path(index_over_span).min())
display(
    Markdown(
        f"The restricted stream peaks in **{peak_month:%B %Y}** and reaches its low in "
        f"**{trough_month:%B %Y}**, a fall of "
        f"**{risk_off_drawdown.min():.1%}** spread over "
        f"**{trough_month.year - peak_month.year} years**. Over the same span the equity "
        f"index itself fell at most **{index_drawdown:.1%}** from its own high-water mark."
    )
)

# %% [markdown]
# Nobody lived through the first of those two numbers. The risk-off months are selected by
# the mixture's assignment, not by the sign of the equity return, so the stream holds gains
# as well as losses - its average monthly return is positive. What it does not hold is the
# months in between, and closing those gaps puts a loss from one decade immediately beside a
# loss from the next, with none of the recovery that separated them. Compounded, decades of
# separate episodes read as one uninterrupted decline.
#
# The statistic is not wrong; it is the correct drawdown of the object it was computed on.
# But the object is a construction of this notebook, and a reader who takes it for a crash
# has been misled by a table that never said which object it meant.
#
# That is the general form: a conditional statistic inherits the assumptions of the
# conditioning. Averages survive it, because a mean does not care in what order its terms
# arrive. Anything that reads a path - a drawdown, a run length, a time to recovery, a
# stop-loss - is a different statistic once the path has been cut up, and reporting it
# without saying so is how a number that means one thing gets read as another.

# %% [markdown]
# ## How long the regimes last
#
# A regime label is useful to a strategy only if it persists long enough to act on. An
# **episode** below is a maximal run of consecutive months carrying the same label, and the
# table reports how many episodes each regime had and how long they ran.

# %%
episode_id = (labels_2 != np.roll(labels_2, 1)).cumsum()
episode_lengths = (
    pd.DataFrame({"regime": regime_name.to_numpy(), "episode": episode_id})
    .groupby(["regime", "episode"])
    .size()
    .rename("months")
    .reset_index()
)

duration = (
    episode_lengths.groupby("regime")["months"]
    .agg(["mean", "max", "count"])
    .rename(
        columns={"mean": "Mean length (months)", "max": "Longest (months)", "count": "Episodes"}
    )
    .loc[["Risk-on", "Risk-off"]]
)
duration.index.name = "Regime"
duration.style.format({"Mean length (months)": "{:.1f}"})

# %%
switches = int(np.sum(np.diff(labels_2) != 0))
months_between_switches = len(labels_2) / switches
display(
    Markdown(
        f"The two-cluster model changes regime **{switches}** times over "
        f"{end_year - start_year} years, about once every "
        f"**{months_between_switches:.1f} months**."
    )
)

# %% [markdown]
# That is the finding that decides what this model is for. A label that changes several
# times a year cannot drive a reallocation: the turnover would be paid on every switch,
# including the ones the model reverses a month later, and the transaction costs would
# arrive whether or not the regime call was right. The same label is useful as a lens on
# history and as one input to a risk report, which is the use Section 1.4 argues for.
#
# A model built to be acted on would be built differently. It would carry a transition
# structure that makes staying in a regime more likely than leaving it - a hidden Markov
# model is the standard choice - and it would be fitted forward in time so that each
# month's label used only the months before it. `09_model_based_features/11_hmm_regimes`
# builds the first, and `09_model_based_features/13_regime_as_feature` turns the result into
# a feature a model downstream can read.

# %% [markdown]
# ## Key takeaways
#
# - **The number of clusters is a decision, not an estimate.** BIC, AIC and the silhouette
#   score reward different things, so they will disagree; pick the one whose objective
#   matches what the labels are for, and say which you picked and why.
# - **Cluster numbers carry no meaning until something outside the model names them.** Here
#   the average equity return within each cluster does the naming, and a different seed can
#   renumber the clusters without changing the partition.
# - **Condition on the environment before trusting a factor's average.** A strategy's
#   full-sample premium can be entirely earned in one environment, which is invisible in the
#   pooled number and decisive for whether it diversifies anything.
# - **State how a conditional statistic was built.** Restricting a return stream to the
#   months in a regime and concatenating them produces a drawdown no investor experienced
#   unless they held only during that regime; the same number is either informative or
#   misleading depending on whether that construction is stated.
# - **Fitting on the whole sample buys description, not prediction.** Every label here is
#   informed by the months after it, so the labels cannot be used as features. Walk-forward
#   fitting is what makes them usable, and Chapter 6 onward introduces it.
#
# **Known limitations.** The mixture has no notion of time: it can assign January 1932 and
# January 2019 to the same cluster and has no mechanism preferring a month to keep the
# previous month's label, which is why the episodes are short. The AQR series are monthly, so
# nothing here resolves anything faster than a month. And the panel is a century long, over
# which the composition of the asset classes and the microstructure of the markets changed
# considerably; the model treats 1927 and 2024 as draws from the same set of distributions.
#
# **Next**: `macro_regimes` asks the same question of macroeconomic indicators rather than
# factor returns, and checks the resulting clusters against realized equity volatility.
