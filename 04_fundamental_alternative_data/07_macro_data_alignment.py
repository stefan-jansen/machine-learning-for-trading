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
# # Macro Data Alignment: Multi-Frequency Integration
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.3 (Fundamentals Across the Asset-Class Spectrum)
#
# ## Purpose
#
# The previous notebook established that the shipped macro panel sits on a calendar-day grid
# with every series carried forward from the date it is stamped with. That grid is convenient
# and, used directly, wrong for a backtest, because a macro observation is stamped with the
# period it *measures* and not with the day it was *published*. The January unemployment rate is
# stamped 1 January and reaches the public in early February. Read straight off the grid, a
# model dated 15 January is trading on a number nobody had for another three weeks.
#
# This notebook closes that gap. It attaches a publication date to each observation, rebuilds
# the daily panel so that every date carries only what had been released by then, measures how
# far the two versions differ, and then builds features on the corrected panel. It finishes with
# the second source of the same error: the value that was published first is often not the value
# that is in the database today.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Recover the period an observation measures from the date it is stamped with, given the
#   series' release frequency.
# - Compute the date an observation became public, from the end of its period and its release
#   lag, and state where that lag comes from.
# - Rebuild a daily panel with a backward as-of join so that each date carries only what was
#   published by then, and measure how many days of look-ahead the naive version carried.
# - Build stationary features - changes, z-scores and regimes - with windows counted on the
#   grid they are named for.
# - Compare a series as first published against the same series as it stands after revision,
#   and say what that difference does to a backtest.
# - Join a macro panel onto a price series without letting the macro grid create trading days
#   that do not exist.
#
# ## Prerequisites
#
# - [`06_fred_macro_eda`](06_fred_macro_eda.ipynb) for what the shipped panel contains.
# - `python data/macro/download.py` for the panel itself, and
#   `python data/macro/download_alfred.py` for the initial-release panel used in Part 6.
#
# ## Cross-References
#
# - **Upstream**: [`06_fred_macro_eda`](06_fred_macro_eda.ipynb), `data/macro/download.py`
# - **Downstream**: `08_financial_features/04_fundamentals_macro_calendar.py` (macro regime features)
# - **Related**: [`09_onchain_fundamentals`](09_onchain_fundamentals.ipynb) (crypto fundamentals)

# %%
"""Macro Data Alignment - align multi-frequency macro data for daily trading models with PIT correctness."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_etfs, load_macro, load_macro_metadata
from data.macro.loader import load_macro_initial_release

# Importing utils.style registers and activates the ML4T Plotly template
# (palette, fonts, backgrounds) as the repo-wide default.
from utils.style import COLORS

# %% [markdown]
# Two settings decide what the notebook shows rather than how it computes. The charts open in
# 2020 so the COVID shock and the tightening cycle are both in frame, and the price join at the
# end uses the most liquid US equity ETF as the thing a macro feature would be attached to.

# %% tags=["parameters"]
VIZ_START = "2020-01-01"  # left edge of every windowed chart
PRICE_SYMBOL = "SPY"  # the price series the macro features are joined onto in Part 7

# %% [markdown]
# ## 1. Two dates, and why only one of them is usable
#
# Every macro observation carries a **reference period**, the stretch of time it measures, and a
# **release date**, the day the statistical agency published it. FRED stamps an observation with
# the first day of its reference period, which is convenient for lining series up and is not a
# date on which anyone knew the value.
#
# The gap between them is the **release lag**, and it has two parts: the period has to finish,
# and then the agency needs time to collect and publish. Both parts have to be paid.

# %% [markdown]
# The lags below are counted in calendar days from the **end** of the reference period, which is
# how each agency states its own schedule. The length of the period is added separately, in the
# next section, from the series' release frequency. Writing the lag this way rather than as a
# distance from the stamp is what keeps a monthly series from appearing four weeks early: the
# Bureau of Labor Statistics publishes the employment report for a month on the first Friday of
# the month after, which is six days past the period end and about thirty-six days past the
# stamp FRED puts on it.

# %%
RELEASE_LAGS = {
    "icsa": 5,  # weekly claims, Thursday for the week ending the previous Saturday
    "walcl": 1,  # Fed H.4.1, Thursday for the Wednesday balance sheet
    "unrate": 6,  # employment situation, first Friday after the month
    "payems": 6,
    "civpart": 6,
    "cpiaucsl": 13,  # CPI, around the middle of the following month
    "cpilfesl": 13,
    "indpro": 16,  # industrial production, G.17
    "m2sl": 24,  # money stock, H.6
    "pcepi": 28,  # personal income and outlays
    "gdp": 26,  # GDP advance estimate, around four weeks after the quarter
    "gdpc1": 26,
}
# Daily market series carry no lag: the value for a date is published that evening.
DAILY_SERIES = [
    "dff",
    "dgs1",
    "dgs2",
    "dgs3",
    "dgs5",
    "dgs7",
    "dgs10",
    "dgs20",
    "dgs30",
    "t10y2y",
    "vixcls",
]

meta = load_macro_metadata()
lag_table = (
    meta.select("series", "description", "native_frequency")
    .filter(pl.col("series").is_in(list(RELEASE_LAGS)))
    .with_columns(
        lag_after_period_end=pl.col("series").replace_strict(RELEASE_LAGS, return_dtype=pl.Int32)
    )
    .sort("native_frequency", "lag_after_period_end")
)
lag_table

# %% [markdown]
# ## 2. The panel as shipped
#
# The daily panel is what Part 3 corrects, so it is worth seeing its shape and its stamping
# convention once more before it is changed. A monthly series changes value on the first day of
# each month, which is the stamping convention stated above rather than a property of the data.

# %%
macro = load_macro().sort("timestamp")
print(f"Rows: {len(macro):,}   Series: {len(macro.columns) - 1}")
print(f"Span: {macro['timestamp'].min()} to {macro['timestamp'].max()}")

first_changes = (
    macro.select("timestamp", "cpiaucsl")
    .with_columns(changed=pl.col("cpiaucsl") != pl.col("cpiaucsl").shift(1))
    .filter(pl.col("changed") & (pl.col("timestamp").dt.year() == 2024))
    .select("timestamp", "cpiaucsl")
    .head(4)
)
print("The dates CPI takes a new value in 2024:")
first_changes

# %% [markdown]
# ## 3. Re-dating an observation to the day it was published
#
# The correction has three steps, and none of them needs anything the panel does not already
# carry. Take the reference period each observation belongs to; add the length of that period to
# reach its end; add the release lag to reach the day it was published. Then a date in the daily
# panel may carry an observation only if that publication date has already passed.
#
# The last step is a **backward as-of join**: for each row of the daily grid, take the most
# recent observation whose publication date is on or before it. `join_asof` does exactly this
# and does it in one pass, which matters here only for clarity - the alternative is a nested
# loop that is easy to get subtly wrong at period boundaries.


# %%
def published_observations(panel: pl.DataFrame, series: str, frequency: str, lag: int):
    """One row per release of `series`: its reference period, and the date it was published."""
    every = {"weekly": "1w", "monthly": "1mo", "quarterly": "1q"}[frequency]
    offset = {"weekly": "1w", "monthly": "1mo", "quarterly": "1q"}[frequency]
    observations = (
        panel.select("timestamp", pl.col(series).alias("value"))
        .drop_nulls()
        .group_by_dynamic("timestamp", every=every)
        .agg(pl.col("value").first())
        .rename({"timestamp": "period_start"})
    )
    return observations.with_columns(
        # A weekly series is already stamped at the end of its week, so its period end is the
        # stamp itself; a monthly or quarterly one is stamped at the start and runs to the day
        # before the next period begins.
        period_end=pl.col("period_start")
        if frequency == "weekly"
        else pl.col("period_start").dt.offset_by(offset).dt.offset_by("-1d"),
    ).with_columns(published_on=pl.col("period_end").dt.offset_by(f"{lag}d"))


# %% [markdown]
# Applied to CPI, the three columns say the whole thing: the value stamped at the start of a
# month describes that month, is complete at its end, and reaches the public around the middle
# of the month after.

# %%
cpi_releases = published_observations(macro, "cpiaucsl", "monthly", RELEASE_LAGS["cpiaucsl"])
cpi_releases.filter(pl.col("period_start").dt.year() == 2024).head(4)

# %% [markdown]
# The as-of join then rebuilds the daily column. Every series that is published on a lag is
# rebuilt this way; the daily market series are carried through unchanged, because for them the
# reference period and the publication date are the same day.


# %%
def point_in_time_panel(panel: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Rebuild `panel` so each date carries only what had been published by that date."""
    frequency_of = dict(zip(metadata["series"], metadata["native_frequency"], strict=True))
    result = panel.select("timestamp", *DAILY_SERIES)

    for series, lag in RELEASE_LAGS.items():
        releases = published_observations(panel, series, frequency_of[series], lag).select(
            "published_on", pl.col("value").alias(series)
        )
        result = result.join_asof(
            releases.sort("published_on"),
            left_on="timestamp",
            right_on="published_on",
            strategy="backward",
        ).drop("published_on")
    return result


pit = point_in_time_panel(macro, meta)
print(f"Point-in-time panel: {pit.shape[0]:,} rows x {pit.shape[1]} columns")
pit.select("timestamp", "cpiaucsl", "unrate", "gdp").tail(5)

# %% [markdown]
# ### How much look-ahead the shipped panel carried
#
# The two panels can be compared directly: on any date, how old is the newest observation each
# of them offers? The naive panel offers the observation stamped at the start of the current
# period; the point-in-time panel offers the newest one that had actually been published.

# %%
comparison = (
    macro.select("timestamp", pl.col("cpiaucsl").alias("as_shipped"))
    .join(pit.select("timestamp", pl.col("cpiaucsl").alias("as_published")), on="timestamp")
    .drop_nulls()
    .with_columns(same=pl.col("as_shipped") == pl.col("as_published"))
)
print(f"Dates where the two panels agree on the CPI level: {comparison['same'].mean():.1%}")

lead = (
    comparison.filter(pl.col("timestamp") >= pl.lit(VIZ_START).str.to_date())
    .select("timestamp", "as_shipped", "as_published")
    .unpivot(index="timestamp", variable_name="panel", value_name="cpi")
)
fig = px.line(
    lead.to_pandas(),
    x="timestamp",
    y="cpi",
    color="panel",
    color_discrete_map={"as_shipped": COLORS["copper"], "as_published": COLORS["blue"]},
    title="The shipped panel steps up about six weeks before the release it reports",
    labels={"timestamp": "Date", "cpi": "CPI index level", "panel": ""},
)
fig.update_layout(height=420, legend=dict(orientation="h", y=1.02, yanchor="bottom"))
fig.show()

# %% [markdown]
# The two lines are the same staircase offset horizontally, and the offset is the whole point.
# Everything a model reads off the earlier line, it reads before the number existed. The offset
# is largest for GDP, whose reference period is a quarter, and smallest for weekly claims.

# %%
staleness = []
for series, lag in RELEASE_LAGS.items():
    frequency = meta.filter(pl.col("series") == series)["native_frequency"][0]
    releases = published_observations(macro, series, frequency, lag)
    staleness.append(
        {
            "series": series,
            "frequency": frequency,
            "median_days_period_start_to_publication": int(
                (releases["published_on"] - releases["period_start"]).dt.total_days().median()
            ),
        }
    )
pl.DataFrame(staleness).sort("median_days_period_start_to_publication", descending=True)

# %% [markdown]
# ## 4. Features, with windows counted on the grid they name
#
# A raw macro level trends and a model fitted on it will fit the trend. The usual corrections
# are a change over a window, a z-score against a rolling window, and a threshold classification
# into regimes.
#
# The one thing to be careful about is what a window means here. This panel has a row per
# **calendar** day, so a window of 252 rows is 252 calendar days, which is eight months and not
# the trading year the number is normally shorthand for. Every window below is therefore named
# and counted in calendar days: 365 for a year, 90 for a quarter, 30 for a month.

# %%
CALENDAR_YEAR, CALENDAR_QUARTER, CALENDAR_MONTH = 365, 90, 30

features = pit.with_columns(
    # Yield curve: level, momentum over a quarter, and position within its own year.
    yield_curve_change_90d=pl.col("t10y2y") - pl.col("t10y2y").shift(CALENDAR_QUARTER),
    yield_curve_zscore_365d=(pl.col("t10y2y") - pl.col("t10y2y").rolling_mean(CALENDAR_YEAR))
    / pl.col("t10y2y").rolling_std(CALENDAR_YEAR),
    yield_curve_regime=pl.when(pl.col("t10y2y") < 0)
    .then(pl.lit("inverted"))
    .when(pl.col("t10y2y") < 0.5)
    .then(pl.lit("flat"))
    .when(pl.col("t10y2y") < 1.5)
    .then(pl.lit("normal"))
    .otherwise(pl.lit("steep"))
    .alias("yield_curve_regime"),
    # Volatility: a shorter memory, because the VIX mean-reverts in weeks rather than years.
    vix_change_30d=pl.col("vixcls") - pl.col("vixcls").shift(CALENDAR_MONTH),
    vix_zscore_90d=(pl.col("vixcls") - pl.col("vixcls").rolling_mean(CALENDAR_QUARTER))
    / pl.col("vixcls").rolling_std(CALENDAR_QUARTER),
    volatility_regime=pl.when(pl.col("vixcls") < 15)
    .then(pl.lit("calm"))
    .when(pl.col("vixcls") < 25)
    .then(pl.lit("normal"))
    .when(pl.col("vixcls") < 35)
    .then(pl.lit("unsettled"))
    .otherwise(pl.lit("frightened"))
    .alias("volatility_regime"),
    # Labour market: the change over a year, which is what recession dating reads.
    unemployment_change_365d=pl.col("unrate") - pl.col("unrate").shift(CALENDAR_YEAR),
    labor_market_regime=pl.when(pl.col("unrate") < 4.0)
    .then(pl.lit("tight"))
    .when(pl.col("unrate") < 6.0)
    .then(pl.lit("normal"))
    .otherwise(pl.lit("slack"))
    .alias("labor_market_regime"),
    # Inflation: a year-over-year rate off an index level, on a calendar-day grid.
    inflation_yoy=pl.col("cpiaucsl") / pl.col("cpiaucsl").shift(CALENDAR_YEAR) - 1,
)
features.select(
    "timestamp",
    "yield_curve_regime",
    "volatility_regime",
    "labor_market_regime",
    "inflation_yoy",
).tail(5)

# %% [markdown]
# ### The four-week claims average, computed on weekly observations
#
# The Department of Labor reports a four-week moving average of initial claims alongside the
# weekly number, because the weekly series is noisy. The average is over four weekly
# observations, so it is computed on the weekly observations and then joined back onto the daily
# grid. A rolling mean taken over the forward-filled daily panel is a different statistic: it
# averages carried-forward days, which weights each week by how many days of the window it
# happened to occupy.

# %%
claims_releases = published_observations(
    macro, "icsa", "weekly", RELEASE_LAGS["icsa"]
).with_columns(claims_4wk_average=pl.col("value").rolling_mean(4))
features = features.join_asof(
    claims_releases.select("published_on", "claims_4wk_average").sort("published_on"),
    left_on="timestamp",
    right_on="published_on",
    strategy="backward",
).drop("published_on")
features.select("timestamp", "icsa", "claims_4wk_average").tail(5)

# %% [markdown]
# ## 5. What the regimes look like
#
# Three of the four regime classifications are worth seeing against the series that produced
# them, because a threshold that never fires and a threshold that fires constantly are both
# useless and both invisible from the definition alone.

# %%
viz = features.filter(pl.col("timestamp") >= pl.lit(VIZ_START).str.to_date()).to_pandas()

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "Yield curve spread, 10-year minus 2-year",
        "Unemployment rate, as published",
        "VIX",
    ),
    vertical_spacing=0.08,
)
fig.add_trace(
    go.Scatter(x=viz["timestamp"], y=viz["t10y2y"], mode="lines", line={"color": COLORS["blue"]}),
    row=1,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["negative"], row=1, col=1)
fig.add_trace(
    go.Scatter(x=viz["timestamp"], y=viz["unrate"], mode="lines", line={"color": COLORS["slate"]}),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=viz["timestamp"], y=viz["vixcls"], mode="lines", line={"color": COLORS["copper"]}),
    row=3,
    col=1,
)
fig.add_hline(y=20, line_dash="dash", line_color=COLORS["neutral"], row=3, col=1)
fig.update_yaxes(title_text="Percentage points", row=1, col=1)
fig.update_yaxes(title_text="% of labour force", row=2, col=1)
fig.update_yaxes(title_text="Index", row=3, col=1)
fig.update_layout(
    height=700,
    title_text="The three series the regime classifications are cut from",
    showlegend=False,
)
fig.show()

# %%
regime_columns = [c for c in features.columns if c.endswith("_regime")]
shares = pl.concat(
    [
        features.filter(pl.col("timestamp") >= pl.lit(VIZ_START).str.to_date())
        .drop_nulls(column)
        .group_by(column)
        .len()
        .rename({column: "regime"})
        .with_columns(classification=pl.lit(column), share=pl.col("len") / pl.col("len").sum())
        for column in regime_columns
    ]
)
fig = px.bar(
    shares.to_pandas(),
    x="share",
    y="classification",
    color="regime",
    orientation="h",
    title="Every regime label is reached, and none covers the whole window",
    labels={"share": "Share of days since the window opened", "classification": ""},
)
fig.update_layout(height=320, xaxis_tickformat=".0%")
fig.show()

# %% [markdown]
# ## 6. The second look-ahead: revisions
#
# Re-dating an observation to its publication date fixes *when* a value becomes visible. It does
# nothing about *which* value. A statistical agency publishes an estimate and then revises it,
# sometimes for years, and the number sitting in the database today is the revised one. A
# backtest that reads today's database reads a number that did not exist on the date it trades.
#
# The archive that answers this is ALFRED, which keeps every vintage FRED has ever published.
# `data/macro/download_alfred.py` materializes one slice of it: for each observation date, the
# value as first released. Comparing that against the current panel measures the revisions
# directly, on real data, rather than describing them.

# %%
initial = load_macro_initial_release()
revisable = [c for c in initial.columns if c != "timestamp" and c in macro.columns]

vintages = initial.select("timestamp", *revisable).join(
    macro.select("timestamp", *revisable), on="timestamp", how="inner", suffix="_current"
)
revision_summary = pl.DataFrame(
    [
        {
            "series": column,
            "observations_compared": len(vintages),
            "observations_revised": int(
                ((vintages[column] - vintages[f"{column}_current"]).abs() > 1e-9).sum()
            ),
            "largest_revision": float(
                (vintages[column] - vintages[f"{column}_current"]).abs().max()
            ),
        }
        for column in revisable
    ]
).sort("largest_revision", descending=True)
revision_summary

# %% [markdown]
# These are daily market rates, the series least likely to be revised at all, and they are
# revised: a handful of observations in each, by amounts that are small against the level and
# large against a day's move. Series built from surveys move far more. Gross domestic product is
# published as an advance estimate about four weeks after the quarter, revised twice within
# three months, and revised again in each of the following years' annual updates.
#
# The consequence for a strategy is not that the numbers are slightly different. It is that a
# rule with a threshold in it can fire on the revised series and not on the released one, and a
# backtest reading the revised series will report a trade that could not have been taken. The
# chart below shows where the two vintages of the ten-year yield actually part company.

# %%
gap = (
    vintages.select(
        "timestamp",
        (pl.col("dgs10") - pl.col("dgs10_current")).alias("revision"),
    )
    .filter(pl.col("revision").abs() > 1e-9)
    .sort("timestamp")
)
print(f"Observation dates on which the 10-year yield was revised: {len(gap)}")
fig = px.bar(
    gap.to_pandas(),
    x="timestamp",
    y="revision",
    title="The 10-year yield is revised rarely, and not by a rounding error",
    labels={
        "timestamp": "Observation date",
        "revision": "First published minus current (percentage points)",
    },
    color_discrete_sequence=[COLORS["copper"]],
)
fig.update_layout(height=380)
fig.show()

# %% [markdown]
# ## 7. Joining macro onto prices
#
# The join direction is the last place this can go wrong. Prices exist on trading days and the
# macro panel exists on every calendar day, so joining the price series onto the macro panel
# would invent a row for every weekend and holiday, each with a null price that a later
# forward-fill would quietly fill. Prices go on the left; the macro columns are attached to the
# days on which trading actually happened.

# %%
prices = (
    load_etfs(symbols=[PRICE_SYMBOL], start_date=VIZ_START)
    .sort("timestamp")
    .select("timestamp", close="close")
    .with_columns(daily_return=pl.col("close").pct_change())
)
combined = prices.join(features, on="timestamp", how="left")

print(f"Trading days: {len(prices):,}")
print(f"Rows after the join: {len(combined):,}")
print(f"Rows with no macro attached: {combined['yield_curve_regime'].is_null().sum()}")
combined.select(
    "timestamp", "close", "daily_return", "yield_curve_regime", "volatility_regime"
).tail(5)

# %% [markdown]
# ## Key Takeaways
#
# 1. A macro observation is stamped with the period it measures, not the day it was published.
#    The distance between the two is the length of the period plus the agency's release lag, and
#    both parts have to be subtracted before a backtest may read the value.
# 2. A backward as-of join on the publication date is the operation that enforces it: for each
#    date, the most recent observation that had already been released. It is one call, and the
#    hand-rolled alternatives get period boundaries wrong.
# 3. A window is counted in rows, so its meaning depends on the grid. On a calendar-day panel,
#    252 rows is eight months rather than a trading year. Name the window for the grid it runs
#    on, or move to the grid the name assumes.
# 4. A statistic whose inputs were carried forward is not the statistic its name claims. A
#    four-week average of a weekly series is computed on the weekly observations; a rolling mean
#    over the forward-filled daily panel weights each week by how many days it happened to
#    cover.
# 5. Publication date and vintage are two separate corrections, and doing the first does not do
#    the second. Even daily market rates are revised after the fact, and any rule with a
#    threshold in it can fire on the revised series and not on the released one.
# 6. Join prices onto macro rather than macro onto prices. The macro panel has a row for every
#    calendar day and will otherwise manufacture trading days that never existed.
