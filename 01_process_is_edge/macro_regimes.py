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
# # Macro-based regime detection
#
# **Chapter 1 · §1.4 Keeping up with changing market regimes**
#
# **Docker image**: `ml4t`
#
# `factor_regimes` estimated regimes from what the factors themselves earned. This notebook
# asks whether the same partition can be recovered from outside the market, using
# macroeconomic series published by the Federal Reserve Bank of St. Louis through its FRED
# database: unemployment, the policy rate, the shape of the yield curve and inflation.
#
# The appeal of macro inputs is that they are not returns. A regime estimated from returns
# is at risk of restating the volatility it was fitted on; one estimated from the labour
# market and the rate environment is a statement about conditions, which can then be
# checked against what the equity market did. That check is the second half of this
# notebook, and it is what separates a regime model that has found something from one that
# has partitioned the calendar.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Turn irregularly published economic series into an aligned monthly panel without
#   letting a later release inform an earlier month.
# - Explain why a series that trends has to enter a clustering as a rate of change rather
#   than as a level, and recognise the failure that follows from ignoring it.
# - Attach economic names to unnamed clusters through a stated rule, and say what the rule
#   assumes.
# - Check estimated regimes against realized equity volatility and drawdown - evidence the
#   clustering never saw - and say which half of that check passed.
# - Judge a clustering by whether it recovers recurring conditions rather than by its
#   separation score, and show why the two can point in opposite directions.
# - Measure how much of a partition is left after a different random start or a slightly
#   shorter sample, and report that beside the result rather than instead of it.
#
# ## Book reference
#
# Chapter 1, Section 1.4, "Keeping up with changing market regimes". Figure 1.6 in the
# chapter is the regime-and-volatility panel drawn below.
#
# ## Prerequisites
#
# - `factor_regimes`, which introduces Gaussian mixture models, the silhouette score, and
#   why a model fitted on the whole sample describes rather than predicts.
# - The FRED panel, which `load_macro` reads. It needs a `FRED_API_KEY` and one run of
#   `uv run python data/macro/download.py`; `data/macro/README.md` covers both.
# - Daily S&P 500 index closes, which `load_sp500_index` reads. This one ships with the
#   repository and needs no download.
#
# ## A caveat that applies to every number below
#
# FRED serves the latest revision of each series. Unemployment, industrial production and
# the price indices are all restated after their first publication, sometimes substantially,
# so the value this notebook reads for a month in 2008 is not the value anyone could see in
# 2008. That is acceptable here because the labels are descriptive and nothing acts on them.
# A strategy would have to read `load_macro_initial_release`, which serves the value as it
# stood at each date, and would also have to wait out the publication lag - the
# unemployment rate for a month is released in the first days of the next one.

# %% [markdown]
# ## Setup

# %%
"""Macro-based regime detection - clustering FRED indicators and validating against the S&P 500."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import Markdown, display
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, scale

import utils.style as style
from data import load_macro, load_macro_metadata, load_sp500_index
from utils.paths import get_output_dir
from utils.reproducibility import set_global_seeds

COLORS = style.COLORS

# %% tags=["parameters"]
# Production defaults (Papermill overrides these when the test suite runs the notebook)
SEED = 42

# %% [markdown]
# ### Settings, and what each one decides
#
# `SEED` fixes every estimator's random start, which is what makes the cluster numbering
# and the figures reproduce.
#
# `N_REGIMES` is the number of clusters fitted throughout. Four is taken from the Two Sigma
# study this notebook follows, which reports four regimes over an eighteen-factor panel. It
# is a choice, not a result: `factor_regimes` shows how the model-selection criteria are
# compared when the count is being decided rather than inherited.
#
# `PANEL_START` is the first month the panel is allowed to contain. The FRED file begins in
# 2000, and the series used here are all quoted from that point, so the bound is set two
# years later for the same reason a rolling statistic drops its warm-up: the year-over-year
# inflation rate needs twelve prior months before it exists.
#
# `MAX_MISSING_SHARE` is how much of a series may be unquoted before it is dropped from the
# extended panel. `MIN_COPHENETIC` is the level above which a dendrogram is usually taken to
# summarise its distance matrix faithfully; it is a convention, quoted here so the number
# the notebook computes has something to be read against.

# %%
N_REGIMES = 4
PANEL_START = pl.datetime(2002, 1, 1)
MAX_MISSING_SHARE = 0.5
MIN_COPHENETIC = 0.7
MONTHS_PER_YEAR = 12
ROLLING_VOL_MONTHS = 12
DATE_COL = "timestamp"

OUTPUT_DIR = get_output_dir(1, "macro_regimes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_global_seeds(SEED)

# %% [markdown]
# ## The FRED panel
#
# The loader returns one wide frame with a daily date index and one column per series.
# Series that are not published daily carry their most recent value forward, so the file has
# no gaps to interpolate and the frequency a series was actually observed at has to be read
# from its metadata rather than from the column.

# %%
macro_raw = load_macro()
if macro_raw[DATE_COL].dtype == pl.Date:
    macro_raw = macro_raw.with_columns(pl.col(DATE_COL).cast(pl.Datetime))

print(f"FRED panel: {macro_raw.height} rows, {macro_raw.width - 1} series")
print(f"Covering {macro_raw[DATE_COL].min():%Y-%m-%d} to {macro_raw[DATE_COL].max():%Y-%m-%d}")

# %% [markdown]
# ### What the panel holds
#
# The metadata companion says what each column is, how often the source publishes it, and
# whether it is an observed series or one this repository derives from others. Both facts
# matter downstream: a quarterly series entering a monthly clustering repeats each value
# three times, and a derived column can duplicate a series that is already in the panel
# under its own name.

# %%
metadata = load_macro_metadata().to_pandas().set_index("series")
inventory = metadata.loc[
    [c for c in macro_raw.columns if c != DATE_COL], ["description", "native_frequency", "formula"]
].rename(
    columns={
        "description": "Series",
        "native_frequency": "Published",
        "formula": "Derived as",
    }
)
inventory["Derived as"] = inventory["Derived as"].fillna("-")
inventory.index.name = "Column"
inventory

# %% [markdown]
# ## Four indicators that describe conditions
#
# The core model uses four series, each standing for one thing a reader can name:
#
# - **UNRATE**, the unemployment rate, for the state of the labour market.
# - **DFF**, the effective federal funds rate, for the stance of monetary policy.
# - **T10Y2Y**, the ten-year Treasury yield minus the two-year, for the shape of the yield
#   curve. A negative value means short-dated debt yields more than long-dated debt, an
#   inversion that has preceded most post-war US recessions.
# - **CPIAUCSL**, the consumer price index, for inflation - entered as a rate of change,
#   for the reason given below.
#
# Four is few enough that every cluster can be described in a sentence, which is the point
# of a regime label. The extended panel later in the notebook is the counter-experiment.

# %%
CORE_SERIES = ["unrate", "dff", "t10y2y", "cpiaucsl"]

# %% [markdown]
# ### From daily rows to a monthly panel
#
# `group_by_dynamic` cuts the daily frame into calendar months and `last()` takes the final
# observation inside each. Labelling each window by its right edge is what keeps the panel
# honest: the window covering January is stamped 1 February, so the row can only contain
# observations from January and nothing dated later reaches it.
#
# That stamp is then stepped back a day, onto the last day of the month the row actually
# describes. Without it every date in the notebook is a month late - the row holding
# April's unemployment rate reads as May - and a figure axis, a printed date and the arrays
# handed to the book build would all inherit the error.


# %%
def to_monthly(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Last observation of each calendar month, stamped on the month it describes."""
    return (
        frame.select([DATE_COL, *columns])
        .sort(DATE_COL)
        .group_by_dynamic(DATE_COL, every="1mo", label="right")
        .agg([pl.col(c).last() for c in columns])
        .with_columns(pl.col(DATE_COL).dt.offset_by("-1d"))
    )


# %%
macro_monthly = (
    to_monthly(macro_raw, CORE_SERIES)
    .drop_nulls(subset=CORE_SERIES)
    .filter(pl.col(DATE_COL) >= PANEL_START)
)

print(f"Monthly rows: {macro_monthly.height}")
print(f"Covering {macro_monthly[DATE_COL].min():%Y-%m} to {macro_monthly[DATE_COL].max():%Y-%m}")

# %% [markdown]
# ### Why the price index enters as a rate, not a level
#
# The consumer price index rises almost every month of the sample. Clustering on the level
# would put every early month in one cluster and every late month in another, because that
# is the largest source of variance in the column, and the resulting partition would be a
# statement about the calendar rather than about inflation. The year-over-year percentage
# change removes the trend and leaves the quantity a reader means by inflation: how much
# prices rose over the past twelve months. Taking it costs the first twelve months of the
# panel, which is why the panel starts a year after `PANEL_START`.
#
# The other three series are already rates and need no such treatment. All four are then
# standardized. They share a unit - percentage points - but not a range: unemployment moves
# over several points across a cycle while the yield spread moves over a fraction of one, so
# without standardizing, the distances would be mostly unemployment.

# %%
macro_df = macro_monthly.select(CORE_SERIES).to_pandas()
macro_df.index = pd.DatetimeIndex(macro_monthly[DATE_COL].to_pandas())
macro_df["cpi_yoy"] = macro_df["cpiaucsl"].pct_change(MONTHS_PER_YEAR) * 100
macro_df = macro_df.drop(columns=["cpiaucsl"]).dropna()

macro_scaled = StandardScaler().fit_transform(macro_df)

PANEL_FIRST_YEAR = macro_df.index.min().year
PANEL_LAST_YEAR = macro_df.index.max().year
print(f"Clustering panel: {len(macro_df)} months, {PANEL_FIRST_YEAR} to {PANEL_LAST_YEAR}")

# %% [markdown]
# ### The four series before anything is fitted
#
# The table reports each series in its own units over the clustering panel. Read each
# quartile spread against its own minimum and maximum: the policy rate covers the whole
# distance from the zero bound to the peak of a tightening cycle, while the unemployment
# rate sits tight around its median and then makes one excursion far outside it. Those
# shapes are what the mixture is about to partition, and the second one is why a cluster
# ends up holding very few months.

# %%
UNITS = {
    "unrate": "Unemployment rate (%)",
    "dff": "Fed funds rate (%)",
    "t10y2y": "10Y minus 2Y Treasury (percentage points)",
    "cpi_yoy": "CPI, change over 12 months (%)",
}
macro_df.rename(columns=UNITS).describe().T.style.format("{:.2f}")

# %% [markdown]
# ## Fit the mixture
#
# The same estimator as `factor_regimes`: a full-covariance Gaussian mixture, restarted ten
# times from different initializations, keeping the fit with the highest likelihood.

# %%
gmm_core = GaussianMixture(
    n_components=N_REGIMES,
    covariance_type="full",
    random_state=SEED,
    n_init=10,
    reg_covar=1e-6,
)
gmm_core.fit(macro_scaled)
core_labels = gmm_core.predict(macro_scaled)
core_probabilities = gmm_core.predict_proba(macro_scaled)
core_silhouette = float(silhouette_score(macro_scaled, core_labels))

print(f"Core model silhouette: {core_silhouette:.3f}")

# %% [markdown]
# ### What the clusters contain
#
# The mixture returns numbers. The table below is what those numbers mean: the average of
# each indicator over the months assigned to each cluster, in the original units.

# %%
cluster_means = macro_df.groupby(core_labels).mean()
cluster_means.index.name = "Cluster"
cluster_means.rename(columns=UNITS).style.format("{:.2f}")

# %% [markdown]
# ### Naming the clusters
#
# A cluster number is not a regime name, and the names have to come from somewhere the model
# cannot supply. The rule below reads each cluster's average indicators and applies the first
# description that fits, in a fixed order. Both the ordering and the thresholds are
# judgements about the post-2002 US economy, chosen so each name means what a reader expects:
# unemployment above ten percent is a crisis whatever the rate environment; high unemployment
# with the policy rate at the floor is the aftermath rather than the crisis itself; a high
# policy rate with a flat curve is a tightening cycle.
#
# Not every branch has to fire. Which names come out depends on which clusters the mixture
# found, and the table below is where to see what this fit produced rather than what the rule
# can produce.
#
# The rule is written in one place so that it can be argued with. A different reader would set
# different thresholds and get different names over the same partition, and that is the honest
# situation: the clustering is estimated, the naming is asserted.


# %%
def name_clusters(means: pd.DataFrame) -> dict[int, str]:
    """Attach an economic description to each cluster from its average indicators."""
    names: dict[int, str] = {}
    for cluster in means.index:
        row = means.loc[cluster]
        if row["unrate"] > 10:
            names[cluster] = "Crisis"
        elif row["unrate"] > 6 and row["dff"] < 0.5:
            names[cluster] = "Recovery"
        elif row["dff"] > 3 and row["t10y2y"] < 0.5:
            names[cluster] = "Tightening"
        elif row["cpi_yoy"] > 4:
            names[cluster] = "Inflation"
        elif row["unrate"] < 5 and row["dff"] < 2:
            names[cluster] = "Expansion"
        else:
            names[cluster] = "Transition"
    return names


# %%
cluster_names = name_clusters(cluster_means)
if len(set(cluster_names.values())) != len(cluster_names):
    raise ValueError(f"Two clusters received the same name: {cluster_names}")

regime_name = pd.Series(
    [cluster_names[label] for label in core_labels], index=macro_df.index, name="regime"
)
ordered_names = [cluster_names[cluster] for cluster in cluster_means.index]
pd.DataFrame(
    {"Name": ordered_names, "Months": regime_name.value_counts()[ordered_names].to_numpy()},
    index=cluster_means.index,
)

# %% [markdown]
# ## Do the regimes correspond to anything the market did?
#
# The clustering has seen no market data at all. If the four groups are picking up real
# conditions rather than an arbitrary carve-up of a four-dimensional cloud, then equity
# returns should behave differently inside them, and the difference should be in the
# direction economic reasoning predicts.
#
# The S&P 500 daily series is resampled to month-end closes and joined to the macro panel on
# the date. Both are now stamped on the last day of the month they describe, so the join is
# exact and each macro row meets the close of the month whose conditions it reports.

# %%
sp500 = load_sp500_index().to_pandas()
sp500[DATE_COL] = pd.to_datetime(sp500[DATE_COL])
sp500 = sp500.set_index(DATE_COL)

sp500_monthly = sp500["close"].resample("ME").last().to_frame()
sp500_monthly["returns"] = sp500_monthly["close"].pct_change()

aligned = sp500_monthly.reindex(macro_df.index)
if aligned["close"].isna().any():
    raise ValueError("A macro month found no S&P 500 close on the same month-end")

print(f"Aligned {len(aligned)} months of S&P 500 closes to the macro panel")

# %% [markdown]
# ### Two statistics per regime, and what each one is
#
# **Annualized volatility** is the standard deviation of the monthly returns of the months
# assigned to a regime, multiplied by the square root of twelve. It describes the months in
# that regime and nothing else.
#
# **Maximum drawdown** is taken differently, and the difference matters. The running peak is
# computed over the whole index, in calendar order, and each month's decline from that peak
# is measured; the number reported for a regime is the deepest such decline observed in one
# of its months. It is therefore the worst point of the index reached while conditions were
# in that regime, not the drawdown of a portfolio held only during it. Where the deepest
# reading lands is therefore a fact about which regime was in force when the index bottomed,
# and that is the fact worth reading off the table.

# %%
aligned["regime"] = regime_name.to_numpy()
aligned["drawdown"] = aligned["close"] / aligned["close"].cummax() - 1

regime_stats = aligned.groupby("regime").agg(
    months=("returns", "count"),
    monthly_std=("returns", "std"),
    worst_drawdown=("drawdown", "min"),
)
regime_stats["annual_vol"] = regime_stats["monthly_std"] * np.sqrt(MONTHS_PER_YEAR) * 100
regime_stats["max_dd_pct"] = -regime_stats["worst_drawdown"] * 100
regime_stats = regime_stats.sort_values("annual_vol")
regime_order = regime_stats.index.tolist()

regime_stats[["months", "annual_vol", "max_dd_pct"]].rename(
    columns={
        "months": "Months",
        "annual_vol": "Annualized volatility (%)",
        "max_dd_pct": "Deepest index drawdown reached (%)",
    }
).style.format(
    {"Annualized volatility (%)": "{:.1f}", "Deepest index drawdown reached (%)": "{:.1f}"}
)

# %% [markdown]
# ### The regime panel
#
# One row per regime, ordered by the volatility of the months it holds. The left panel shows
# when each regime was in force, with three dated episodes marked; the two right panels are
# the statistics above, so the ordering of the bars can be read against the pattern of the
# bands. This is the panel that appears as Figure 1.6.

# %%
EVENTS = {2008: "Global financial crisis", 2020: "COVID-19", 2022: "Inflation"}
REGIME_SHADES = [COLORS["recede"], COLORS["copper"], COLORS["amber"], COLORS["blue"]]


# %%
def year_ticks(dates: pd.DatetimeIndex, every: int) -> tuple[list[int], list[str]]:
    """Positions and labels for the first month of every *every*-th year."""
    years = dates.year.tolist()
    ticks = [j for j, y in enumerate(years) if y % every == 0 and (j == 0 or years[j - 1] != y)]
    return ticks, [str(years[j]) for j in ticks]


# %%
def draw_regime_row(ax, occupied: np.ndarray, colour: str, label: str) -> None:
    """Shade the months a regime was in force across one full-width row."""
    for j in np.flatnonzero(occupied):
        ax.axvspan(j, j + 1, color=colour)
    ax.set_xlim(0, len(occupied))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center")


# %%
def draw_stat_bar(ax, value: float, colour: str, limit: float, heading: str | None) -> None:
    """Draw one horizontal bar with its value written at the end."""
    ax.barh([0], [value], color=colour, height=0.6)
    ax.set_xlim(0, limit)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.text(value + limit * 0.02, 0, f"{value:.0f}%", ha="left", va="center", fontsize=8)
    if heading:
        ax.set_title(heading, fontsize=8)


# %%
def mark_events(ax, dates: pd.DatetimeIndex, label_them: bool) -> None:
    """Draw a rule at the first month of each dated episode, labelling it on the top row."""
    for event_year, event_label in EVENTS.items():
        position = next((j for j, y in enumerate(dates.year) if y == event_year), None)
        if position is None:
            continue
        ax.axvline(x=position, color=COLORS["neutral"], alpha=0.35, linewidth=0.8)
        if label_them:
            ax.annotate(
                event_label,
                xy=(position, 1.3),
                xycoords=("data", "axes fraction"),
                ha="center",
                fontsize=6,
                color=COLORS["neutral"],
            )


# %%
def draw_panel_row(fig, grid, i: int, regime: str, first: bool, last: bool) -> None:
    """Draw one regime's timeline row and its two statistic bars."""
    shade = REGIME_SHADES[i]
    ax_band = fig.add_subplot(grid[i, 0])
    draw_regime_row(ax_band, regime_name.to_numpy() == regime, shade, regime)
    mark_events(ax_band, macro_df.index, label_them=first)
    if last:
        ax_band.set_xticks(ticks)
        ax_band.set_xticklabels(tick_labels, fontsize=7)
        ax_band.set_xlabel("Year")
    else:
        ax_band.set_xticks([])

    stats = regime_stats.loc[regime]
    heading = ("Annualized\nvolatility", "Deepest index\ndrawdown") if first else (None, None)
    draw_stat_bar(fig.add_subplot(grid[i, 1]), stats["annual_vol"], shade, 25, heading[0])
    draw_stat_bar(fig.add_subplot(grid[i, 2]), stats["max_dd_pct"], shade, 70, heading[1])


# %%
ticks, tick_labels = year_ticks(macro_df.index, every=4)
n_rows = len(regime_order)

fig = plt.figure(figsize=(style.PAGE_WIDTH, 0.85 * n_rows + 1.4))
grid = GridSpec(n_rows, 3, figure=fig, width_ratios=[3.5, 1, 1], wspace=0.15, hspace=0.15)

for i, regime in enumerate(regime_order):
    draw_panel_row(fig, grid, i, regime, first=(i == 0), last=(i == n_rows - 1))

fig.suptitle(
    "Ordering the macro regimes by volatility does not order them by drawdown",
    fontsize=10,
    ha="left",
    x=0.02,
)
fig.text(
    0.5,
    0.0,
    "Regimes from unemployment, the fed funds rate, the 10Y-2Y spread and 12-month CPI change",
    ha="center",
    fontsize=7,
    color=COLORS["neutral"],
)
style.show_with_alt(
    fig,
    "Four stacked timeline rows, one per regime, beside two columns of horizontal bars for "
    "annualized volatility and the deepest index drawdown. The volatility bars grow "
    "steadily down the rows while the drawdown bars do not follow the same order.",
)

# %% [markdown]
# ### Reading the panel honestly
#
# The volatility ordering is what the panel was sorted on, so it is monotone by
# construction and says nothing on its own. The drawdown column is the informative one,
# because nothing forced it to agree, and it does not.

# %%
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]


def rank_by_volatility(regime: str) -> str:
    """Where a regime sits in the volatility ordering the panel is drawn in."""
    return ORDINALS[regime_order.index(regime)]


calmest_regime = str(regime_stats["annual_vol"].idxmin())
loudest_regime = str(regime_stats["annual_vol"].idxmax())
vol_spread = float(regime_stats["annual_vol"].max() - regime_stats["annual_vol"].min())
deepest_month = aligned["drawdown"].idxmin()
deepest_regime = str(aligned.loc[deepest_month, "regime"])
shallowest_regime = str(regime_stats["max_dd_pct"].idxmin())
display(
    Markdown(
        f"Annualized volatility runs from **{calmest_regime}** to **{loudest_regime}**, a "
        f"span of **{vol_spread:.0f} percentage points**. The deepest decline of the whole "
        f"panel, **{regime_stats['max_dd_pct'].max():.0f}%**, falls in "
        f"**{deepest_month:%B %Y}**, which the naming rule calls **{deepest_regime}** - the "
        f"**{rank_by_volatility(deepest_regime)}** of the four by volatility. The shallowest "
        f"decline belongs to **{shallowest_regime}**, the "
        f"**{rank_by_volatility(shallowest_regime)}**."
    )
)

# %% [markdown]
# The volatility ordering is economically legible: a tightening cycle is a calm market with an
# expensive policy rate, and the two regimes with an unusual price level or an unusual
# unemployment rate are the volatile ones. That is the check passing. Nothing about the
# clustering forced it, because the clustering never saw a return.
#
# The drawdown column is where the check bites. It does not follow the volatility order, and
# the regime holding the deepest decline is the one the rule names for the aftermath of a
# crisis - high unemployment with the policy rate on the floor. That combination is what a
# central bank produces in response to a crash, so the label arrives with the crash rather
# than before it. The regime a risk report would most want warning of is the one this model
# can only confirm.
#
# That is the shape of what macro regimes are for, and it follows from what the inputs are.
# Unemployment is measured over a month and published in the next; the policy rate moves in
# response to conditions the market has already priced. A label built from them describes the
# environment a portfolio is in. Section 1.4 argues for exactly that use - connecting an
# identified environment to a predefined risk action - and against reading it as a forecast.

# %% [markdown]
# ### How much of this partition would survive a small change?
#
# Everything above describes one fit. Before any of it is written down, it is worth asking how
# much of it is a property of the data and how much is a property of this particular run.
#
# Two perturbations, neither of which changes what the data says. The first refits from a
# different random start, which a mixture is entitled to answer differently because
# expectation-maximization climbs to a local optimum. The second drops the first few months of
# the panel, which is the kind of choice made once and never revisited - a start date is a
# judgement about coverage, not a measurement.
#
# Agreement between two partitions is measured by the **adjusted Rand index**: the share of
# month pairs that two partitions agree about - either together in both, or separated in both -
# rescaled so that a random relabelling scores zero and an identical partition scores one. It
# does not care how the clusters are numbered, which is what makes it usable here.


# %%
def refit(data: np.ndarray, seed: int) -> np.ndarray:
    """Fit the core mixture to *data* from one random start and return its labels."""
    return (
        GaussianMixture(
            n_components=N_REGIMES,
            covariance_type="full",
            random_state=seed,
            n_init=10,
            reg_covar=1e-6,
        )
        .fit(data)
        .predict(data)
    )


# %%
perturbations = []
for seed in (0, 7, 123, 2024):
    labels = refit(macro_scaled, seed)
    perturbations.append(
        {
            "Change": f"random start {seed}",
            "Agreement with the fit above": adjusted_rand_score(core_labels, labels),
            "Silhouette": silhouette_score(macro_scaled, labels),
            "Smallest cluster (months)": int(np.bincount(labels).min()),
        }
    )
for dropped in (1, 3, 6, 12):
    labels = refit(macro_scaled[dropped:], SEED)
    perturbations.append(
        {
            "Change": f"drop first {dropped} months",
            "Agreement with the fit above": adjusted_rand_score(core_labels[dropped:], labels),
            "Silhouette": silhouette_score(macro_scaled[dropped:], labels),
            "Smallest cluster (months)": int(np.bincount(labels).min()),
        }
    )
stability = pd.DataFrame(perturbations).set_index("Change")
stability.style.format({"Agreement with the fit above": "{:.2f}", "Silhouette": "{:.3f}"})

# %%
worst_agreement = float(stability["Agreement with the fit above"].min())
smallest_ever = int(stability["Smallest cluster (months)"].min())
display(
    Markdown(
        f"The least agreement any perturbation reaches is **{worst_agreement:.2f}**, and the "
        f"smallest cluster any of them produces holds **{smallest_ever} months** against "
        f"**{int(np.bincount(core_labels).min())}** in the fit above."
    )
)

# %% [markdown]
# None of these partitions is wrong. They are answers to the same question from starts the
# data does not distinguish between, and they disagree about a meaningful share of the months.
# Some of them split off a cluster small enough that the naming rule reaches a different
# branch, so the same panel comes back with a different set of regime names.
#
# The response is not to pick the fit whose names read best. It is to say what was fitted, on
# what window, from which start, and to publish the perturbation table beside the result, so
# that a reader can see how much of the story is load-bearing. Anything downstream that would
# break under an agreement of this size is not supported by this model, whatever the figure
# looks like.
#
# This is the chapter's argument arriving in one table. The model is ordinary and takes four
# lines to fit; what separates a usable result from an anecdote is the check that comes after
# it, and the discipline to report the check when it is unflattering.

# %% [markdown]
# ### Figure 1.6 inputs
#
# The print version of Figure 1.6 is drawn by
# `book/01_process_is_edge/figures/scripts/generate_figure_1_6_macro_regimes_volatility.py`
# in the book repository, which reads the arrays written below rather than re-fitting the
# clustering.

# %%
ARTIFACT_DIR = OUTPUT_DIR / "figure_1_6"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    ARTIFACT_DIR / "inputs.npz",
    dates=macro_df.index.astype("datetime64[ns]").astype("int64"),
    macro_labels=np.asarray(core_labels, dtype=np.int64),
    regime_order=np.asarray(regime_order, dtype=object),
    raw_regime_for_order=np.asarray(
        [next(c for c, name in cluster_names.items() if name == regime) for regime in regime_order],
        dtype=np.int64,
    ),
    annual_vol=regime_stats.loc[regime_order, "annual_vol"].to_numpy(dtype=float),
    max_dd_pct=regime_stats.loc[regime_order, "max_dd_pct"].to_numpy(dtype=float),
    event_years=np.asarray(list(EVENTS), dtype=np.int64),
    event_labels=np.asarray(list(EVENTS.values()), dtype=object),
    start_year=np.int64(PANEL_FIRST_YEAR),
    end_year=np.int64(PANEL_LAST_YEAR),
)

# %% [markdown]
# ## How the four indicators move together
#
# A clustering on four correlated inputs is not a clustering in four independent
# directions, so the correlations are worth seeing before the extended panel raises the
# question at a larger scale. Only one triangle is drawn: the matrix is symmetric and the
# other half carries no additional information. The scale is fixed to the full range a
# correlation can take, so the colours mean the same thing here as in any other correlation
# figure in the book.

# %%
correlations = macro_df.rename(columns=UNITS).corr()

fig, ax = plt.subplots(figsize=style.FIGSIZE["single_tall"])
sns.heatmap(
    correlations,
    mask=style.upper_triangle_mask(correlations),
    annot=True,
    fmt=".2f",
    cmap=style.ml4t_diverging(),
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7, "label": "Correlation"},
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("")
style.add_message_title(
    ax,
    "No pair of the four indicators is close to independent",
    subtitle="Pearson correlation over the clustering panel",
)
style.show_with_alt(
    fig,
    "A lower-triangular correlation heatmap of four macro indicators on a red-to-blue scale "
    "fixed between minus one and one. The unemployment rate and the yield spread are "
    "strongly positive, the fed funds rate is strongly negative against the yield spread, "
    "and inflation is moderately negative against unemployment.",
)

# %% [markdown]
# Unemployment and the yield spread move together, and the fed funds rate moves against the
# spread. Both are the same mechanism seen from two sides: the Federal Reserve cuts the
# short rate when the labour market weakens, which pulls the front of the curve down and
# steepens the spread, and it raises the short rate when the economy runs hot, which flattens
# or inverts it. Inflation is the least entangled of the four and is still not independent of
# them.
#
# That is the case for clustering jointly rather than thresholding one series at a time. Any
# single indicator tells the same story as its neighbours plus its own noise; the joint model
# reads all four positions at once, which is how it can separate the tightening cycle from
# the recovery even though both involve an unusual short rate.

# %% [markdown]
# ## The counter-experiment: every series in the panel
#
# The core model uses four series chosen for what they mean. The obvious question is whether
# adding the rest of the file would do better, and the obvious way to answer it is to fit the
# same mixture to everything and compare the separation scores.
#
# That comparison is the trap this section exists to spring. The extended panel is assembled
# with no judgement at all beyond a coverage filter, which is exactly how such a panel is
# usually built.

# %%
value_columns = [c for c in macro_raw.columns if c != DATE_COL]
monthly_full = to_monthly(macro_raw, value_columns).filter(pl.col(DATE_COL) >= PANEL_START)

missing_share = {
    column: monthly_full.select(pl.col(column).is_null().sum()).item() / monthly_full.height
    for column in value_columns
}
kept_columns = [c for c, share in missing_share.items() if share <= MAX_MISSING_SHARE]

extended_pl = monthly_full.select([DATE_COL, *kept_columns]).drop_nulls()
extended_df = extended_pl.select(kept_columns).to_pandas()
extended_df.index = pd.DatetimeIndex(extended_pl[DATE_COL].to_pandas())

# Hold both models to the same months. Left alone the two panels start a month apart - the
# core one waits twelve months for the year-over-year change, this one waits for the Fed
# balance sheet to begin - and every comparison below would then be partly about the window.
extended_df = extended_df.loc[extended_df.index.intersection(macro_df.index)].apply(scale)
if not extended_df.index.equals(macro_df.index):
    raise ValueError("The extended panel does not cover the same months as the core panel")

display(
    Markdown(
        f"The coverage filter keeps **{len(kept_columns)} of {len(value_columns)}** series, "
        f"and holding them to the core panel's months leaves **{len(extended_df)} months** "
        "across both models."
    )
)

# %% [markdown]
# ### What the filter let through
#
# Nothing in that assembly asked what the columns are, and the metadata table at the top of
# the notebook says what got in. Nine of the columns are Treasury yields at maturities from
# one to thirty years, which move almost as one series. `YIELD_CURVE_SLOPE` is defined as the
# ten-year yield minus the two-year, which is the definition of `t10y2y`, already in the
# panel - the same spread enters twice under two names. Nominal and real GDP are both
# present, published quarterly, so each repeats its value for three months at a time. And
# three price indices enter as levels, which is the treatment the core panel rejected for one
# of them.

# %%
duplicate_check = (extended_df["t10y2y"] - extended_df["YIELD_CURVE_SLOPE"]).abs().max()
gdp_distinct = int(extended_df["gdp"].round(6).nunique())
display(
    Markdown(
        f"The largest gap between `t10y2y` and `YIELD_CURVE_SLOPE` anywhere in the panel is "
        f"**{duplicate_check:.1e}** after standardization, and nominal GDP takes "
        f"**{gdp_distinct} distinct values** across **{len(extended_df)} monthly rows**."
    )
)

# %% [markdown]
# ### The series, drawn
#
# Every kept column standardized and plotted on shared axes. The point of the small multiples
# is the shape of each panel, not its level: how many of them are the same line.

# %%
n_columns = 5
n_rows_grid = int(np.ceil(len(extended_df.columns) / n_columns))
fig, axes = plt.subplots(
    n_rows_grid,
    n_columns,
    figsize=(style.PAGE_WIDTH, 0.85 * n_rows_grid + 0.6),
    sharex=True,
    sharey=True,
)
for ax, column in zip(axes.flat, extended_df.columns):
    ax.plot(extended_df.index, extended_df[column], color=COLORS["blue"], linewidth=0.7)
    ax.set_title(column, fontsize=6)
    ax.tick_params(labelsize=5)
for ax in axes.flat[len(extended_df.columns) :]:
    ax.set_visible(False)

fig.suptitle(
    "Slow trends dominate the panel; only the VIX and claims spike",
    fontsize=10,
    ha="left",
    x=0.02,
)
style.show_with_alt(
    fig,
    "A grid of small standardized time-series panels sharing their axes. Several rise "
    "steadily from one end of the window to the other, the Treasury yields move together "
    "in broad waves, and only the VIX and initial jobless claims show sharp isolated "
    "spikes against an otherwise flat line.",
)

# %% [markdown]
# ### How much independent variation is left
#
# Principal component analysis puts a number on the redundancy the figure shows. Each
# component is the direction of greatest remaining variance, and the cumulative share says
# how much of the panel's total variation the first few directions account for.

# %%
pca = PCA(n_components=min(10, extended_df.shape[1]))
reduced = pca.fit_transform(extended_df)

explained = pd.DataFrame(
    {
        "Component": [f"PC{i + 1}" for i in range(pca.n_components_)],
        "Share of variance": pca.explained_variance_ratio_,
        "Cumulative": np.cumsum(pca.explained_variance_ratio_),
    }
).set_index("Component")
explained.style.format("{:.1%}")

# %%
two_component_share = float(explained["Cumulative"].iloc[1])
display(
    Markdown(
        f"Two directions account for **{two_component_share:.0%}** of the variance across "
        f"**{extended_df.shape[1]} columns**."
    )
)

# %% [markdown]
# ### Which columns each direction is made of
#
# The share of variance says how much a direction carries; the loadings say what it is. Each
# column below lists the four series with the largest absolute weight in that component,
# heaviest first.

# %%
loadings = pd.DataFrame(pca.components_.T, index=extended_df.columns, columns=explained.index)
pd.DataFrame(
    {
        component: loadings[component].abs().sort_values(ascending=False).index[:4].tolist()
        for component in explained.index[:4]
    },
    index=[f"Heaviest {i + 1}" for i in range(4)],
)

# %% [markdown]
# The leading direction is the trend the small multiples showed, and the second is the level
# of the long end of the Treasury curve. Neither is a regime. The series that move when
# conditions break - the VIX, initial jobless claims - do not appear until the third and
# fourth components, which between them carry a small share of the variance. A mixture fitted
# on this panel is therefore fitted mostly on where in the sample a month sits, which is what
# the episode count is about to show.

# %% [markdown]
# ## Fit the same models to the extended panel
#
# A mixture and a k-means partition, both at the same cluster count as the core model, plus a
# mixture on the ten leading principal components to see whether reducing the redundancy
# changes the answer.

# %%
gmm_extended = GaussianMixture(
    n_components=N_REGIMES, covariance_type="full", random_state=SEED, n_init=10, reg_covar=1e-6
).fit(extended_df)
extended_labels = gmm_extended.predict(extended_df)
extended_probabilities = gmm_extended.predict_proba(extended_df)

kmeans_labels = KMeans(n_clusters=N_REGIMES, random_state=SEED, n_init=10).fit_predict(extended_df)

gmm_pca = GaussianMixture(
    n_components=N_REGIMES, covariance_type="full", random_state=SEED, n_init=10, reg_covar=1e-6
).fit(reduced)
pca_labels = gmm_pca.predict(reduced)

comparison = pd.DataFrame(
    {
        "Model": [
            f"Core, {len(CORE_SERIES)} chosen series",
            f"Extended, {extended_df.shape[1]} series",
            "Extended, 10 principal components",
            f"Extended, {extended_df.shape[1]} series, k-means",
        ],
        "Silhouette": [
            core_silhouette,
            silhouette_score(extended_df, extended_labels),
            silhouette_score(reduced, pca_labels),
            silhouette_score(extended_df, kmeans_labels),
        ],
    }
).set_index("Model")
comparison.style.format("{:.3f}")

# %% [markdown]
# ### The score says the extended panel is better. Look at what it bought.
#
# An **episode** is a maximal run of consecutive months carrying the same label. Counting
# them separates a model that recovers recurring conditions from one that has cut the sample
# into consecutive blocks: a condition that recurs produces many episodes and revisits
# earlier labels, while a chronological cut produces exactly as many episodes as it has
# clusters and never returns to one.


# %%
def episode_count(labels: np.ndarray) -> int:
    """Number of maximal runs of consecutive months sharing a label."""
    return int(np.sum(np.diff(labels) != 0)) + 1


def revisited_clusters(labels: np.ndarray) -> int:
    """How many clusters the model returns to after having left them."""
    starts = [labels[0], *labels[1:][np.diff(labels) != 0]]
    return sum(1 for cluster in set(starts) if starts.count(cluster) > 1)


episodes = pd.DataFrame(
    {
        "Model": [
            f"Core, {len(CORE_SERIES)} chosen series",
            f"Extended, {extended_df.shape[1]} series",
        ],
        "Clusters": [N_REGIMES, N_REGIMES],
        "Episodes": [episode_count(core_labels), episode_count(extended_labels)],
        "Clusters revisited": [
            revisited_clusters(core_labels),
            revisited_clusters(extended_labels),
        ],
    }
).set_index("Model")
episodes

# %%
extended_episodes = episode_count(extended_labels)
core_episodes = episode_count(core_labels)
display(
    Markdown(
        f"The extended model splits {len(extended_df)} months into **{extended_episodes} "
        f"episodes** from **{N_REGIMES} clusters**, returning to "
        f"**{revisited_clusters(extended_labels)}** of them, against "
        f"**{core_episodes} episodes** and "
        f"**{revisited_clusters(core_labels)}** revisited for the core model."
    )
)

# %% [markdown]
# A model with as many episodes as clusters and none of them revisited has assigned every
# month to the cluster of its neighbours and never returned to a condition it had left. It has
# partitioned the calendar into consecutive eras. The core model does return, which is what
# lets it call the 2008 aftermath and the 2020 aftermath the same thing. That is what the trending levels put into the panel - three
# price indices, two GDP series, money stock, payrolls - and it is precisely the failure the
# core panel avoided by taking a rate of change instead of a level. The silhouette score
# rewards it, because consecutive eras of a trending panel are far apart in the space the
# score measures distance in.
#
# The lesson generalises past this dataset. A separation score answers "are these groups far
# apart", and a regime model has to answer "would this label have told me something the next
# time conditions like these arrived". Nothing forces those two questions to have the same
# answer, and a panel assembled without judgement is where they come apart.

# %% [markdown]
# ### The two partitions, drawn
#
# Assignment probabilities from both mixtures over the same months. Darker means the model
# was more confident the month belonged to that cluster.


# %%
def plot_assignment_heatmap(probabilities: np.ndarray, dates: pd.DatetimeIndex, claim: str) -> None:
    """Draw per-month cluster assignment probabilities as a heatmap."""
    fig, ax = plt.subplots(figsize=style.FIGSIZE["single_wide"])
    image = ax.imshow(probabilities.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    positions, labels = year_ticks(dates, every=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(range(probabilities.shape[1]))
    ax.set_yticklabels([f"Cluster {i}" for i in range(probabilities.shape[1])])
    ax.set_xlabel("Year")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Assignment probability")
    style.add_message_title(ax, claim)
    style.show_with_alt(
        fig,
        "A heatmap with one row per cluster and one column per month, shaded by assignment "
        "probability.",
    )


# %%
plot_assignment_heatmap(
    core_probabilities, macro_df.index, "The core model returns to conditions it has seen before"
)

# %%
plot_assignment_heatmap(
    extended_probabilities, extended_df.index, "The extended model never returns to a cluster"
)

# %% [markdown]
# ## Clustering the indicators rather than the months
#
# Everything so far has grouped months. The same machinery can group the columns instead,
# which answers a different question: which indicators carry the same information? Ward
# linkage builds a tree by repeatedly merging the two groups whose merger adds least to the
# within-group variance.
#
# The **cophenetic correlation** measures how well such a tree preserves the distances it was
# built from. For each pair of items it reads the height at which the tree first joins them,
# and correlates those heights against the original pairwise distances. A tree that scores
# near one is a faithful summary of the distance matrix; one that scores low has imposed a
# hierarchy the data does not have.
#
# Two trees are built below and they answer different questions, so their cophenetic
# correlations are not interchangeable. The first is over the columns and is what the block
# structure in the figure refers to. The second is over the months.

# %%
column_distances = pdist(extended_df.T)
column_linkage = linkage(column_distances, method="ward")
column_cophenetic, _ = cophenet(column_linkage, column_distances)

month_distances = pdist(extended_df)
month_linkage = linkage(month_distances, method="ward")
month_cophenetic, _ = cophenet(month_linkage, month_distances)

display(
    Markdown(
        f"Cophenetic correlation is **{column_cophenetic:.2f}** for the tree over the "
        f"indicators and **{month_cophenetic:.2f}** for the tree over the months, against a "
        f"conventional bar of {MIN_COPHENETIC}."
    )
)

# %%
fig, ax = plt.subplots(figsize=style.FIGSIZE["single_tall"])
dendrogram(
    column_linkage,
    labels=extended_df.columns.tolist(),
    orientation="right",
    ax=ax,
    color_threshold=0,
    link_color_func=lambda _: COLORS["neutral"],
)
ax.set_xlabel("Ward linkage distance")
ax.tick_params(axis="y", labelsize=6)
style.add_message_title(
    ax,
    "The two names for the same spread merge first, at zero distance",
    subtitle="Ward linkage over the standardized indicator columns",
)
style.show_with_alt(
    fig,
    "A horizontal dendrogram of the panel's columns. One pair joins at the left edge with "
    "no visible branch length; the Treasury maturities and the trending level series form "
    "two further groups, and the VIX and initial claims stand apart until the last merge.",
)

# %%
first_merge_distance = float(column_linkage[0, 2])
merged_first = [extended_df.columns[int(i)] for i in column_linkage[0, :2]]
display(
    Markdown(
        f"The first merge the tree makes is `{merged_first[0]}` with `{merged_first[1]}`, at "
        f"a distance of **{first_merge_distance:.1e}**."
    )
)

# %% [markdown]
# A merge at zero distance is two copies of one column, and the tree finds it before it
# considers anything else. Above that the panel resolves into groups a reader can name: the
# series that only trend, the short and medium Treasury yields, the curve-shape series
# alongside unemployment, and - joining last, at the greatest distance - the VIX and initial
# jobless claims. That last pair is the only part of the panel that moves on the timescale a
# crisis moves on, which is why the principal components leave it until third and fourth
# place while spending the first two on trend and on the level of long rates.
#
# What the tree does not do is choose the panel. It says which columns duplicate each other;
# which of a duplicated pair to keep is decided on grounds the data cannot supply - what the
# column means, and whether it is published in time to be read when a decision has to be
# made.

# %% [markdown]
# ## Key takeaways
#
# - **Level or rate is the first decision, not a preprocessing detail.** A trending column
#   entering a clustering makes elapsed time the dominant direction of variance, and the
#   partition that comes back is a partition of the calendar wearing economic names.
# - **A separation score is not a validation.** Silhouette answers whether the groups are far
#   apart in the space they were fitted in. Whether the groups mean anything is a separate
#   question, and counting episodes - does the model ever return to a cluster? - is a cheap
#   way to ask it.
# - **More series is not more information.** Nine Treasury maturities, a spread stored twice,
#   and two GDP measures are one panel with a few directions in it, which principal components
#   and the linkage tree both report directly.
# - **Attaching names to clusters is an assertion.** The mixture supplies a partition; every
#   economic name here comes from a threshold rule written by hand, and a different rule would
#   rename the same partition.
# - **Validate a regime model against something it never saw.** These regimes were fitted
#   without any market data, so equity volatility and drawdown are an independent check. Half
#   of it passed - the volatility ordering is economically legible - and half of it did not:
#   the deepest drawdown sits in the regime named for a crisis already under way.
# - **Perturb the fit before you write about it.** A different random start, or a sample three
#   months shorter, moves this partition enough to change which names the rule assigns. Report
#   that alongside the result; a story the perturbation table takes apart was never supported
#   by the data.
#
# **Known limitations.** FRED serves revised values, so no number here was available on the
# date it is attached to. The mixture has no transition structure and no notion of time, so
# nothing prefers a month to keep its neighbour's label; the long episodes here come from the
# indicators moving slowly, not from the model. The panel starts in 2002 and holds one severe
# recession, one pandemic and one inflation episode, which is too few repetitions of any
# condition to say how often it recurs - and it is the direct cause of the instability the
# perturbation table reports. The naming rule's thresholds were set for the post-2002 US
# economy and would not transfer to another country or another era.
#
# **Next**: Chapter 1 closes on what a research process needs to survive changes like these.
# Walk-forward estimation, which is what turns a descriptive regime label into one a strategy
# could act on, is introduced from Chapter 6.
