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
# # FX Pairs: Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. It trades twenty currency pairs,
# sorts them once a day, buys the ones at the top of the ordering and sells the ones at the bottom.
# That file says which pairs it trades, what moment of the day it decides at, what it assumes a
# trade costs, and how the history is divided between designing the strategy and testing it. This
# notebook checks each of those assumptions against the data and reports what it finds.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Divide a price feed that never closes into trading days, and pick the one bar in each day that a
#   decision taken at the daily cut-off could actually have traded at
# - Count how many instruments are quoting on each date a strategy changes its positions, and
#   compare that count against the number of positions it has to fill
# - Separate the number of instruments in a universe from the number of independent bets they
#   carry, when several of them are exposed to the same thing
# - Read off one chart what fraction of price moves are larger than the cost of trading them
# - Measure how much of one day's return carries into the next, computing the correlation inside
#   each instrument rather than across the whole universe stacked into one series
# - Check that a walk-forward split of the history fits the sample available and leaves the test
#   period unread
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. This notebook reads four-hour spot bars from OANDA and
# `config/setup.yaml`, and writes nothing.
#
# ## Prerequisites
#
# None beyond what the sections below define. A reader who has not traded currencies or split a
# sample for walk-forward evaluation will find both explained where they are first used.

# %%
"""FX Pairs Case Study - Feasibility Analysis."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.splitters.calendar import TradingCalendar

from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import load_fx_pairs
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
START_DATE = "2011-01-01"
END_DATE = "2025-12-31"

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads those
# values rather than repeating them, so the two can never disagree. Four groups of settings matter
# here, and each one decides something the sections below test.
#
# **How the history is divided.** The sample runs from 2011 to the end of 2025. The last two years
# are the *holdout*: a stretch of history that is not looked at while the strategy is being
# designed, so that when it is finally evaluated there, the result is not a rehearsal of choices
# already tuned on the same data. Everything computed in this notebook uses the earlier part,
# called the development period, and `holdout_start` is where the line falls.
#
# **What the strategy trades.** `setup.yaml` names twenty currency pairs. It takes a position in
# every one of them - up to ten bought and up to ten sold - so all twenty have to be quoting on any
# date it rebalances. That floor of twenty comes from the grid of position counts the strategy will
# later search over, not from a separate assumption.
#
# **What a trade is assumed to cost.** A range of *spreads*: the gap between the price at which a
# pair can be bought and the price at which it can be sold, one range for pairs quoted against the
# US dollar and a wider one for the rest. Section B.3 says why the two differ and turns the range
# into a cost that can be compared to a return.
#
# **What is being predicted.** The return over the next trading day, with variants looking five and
# twenty-one days ahead. The one-day horizon is the primary one, and it sets both how often
# positions change and how wide a gap has to separate training data from validation data.
#
# Two calendars are declared and they answer different questions. `decision.session_calendar` is
# the venue whose daily rollover decides which trading day a four-hour bar belongs to, which is
# what this notebook needs in order to find each day's last bar. `evaluation.calendar` is the one
# the walk-forward splitter counts training and validation windows on.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
DECLARED_PAIRS = sorted(SETUP["universe"]["symbols"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = SETUP["labels"]["buffer"]
HORIZONS = sorted(int(n.split("_")[-1][:-1]) for n in [PRIMARY_LABEL, *SETUP["labels"]["variants"]])
PRIMARY_HORIZON = int(PRIMARY_LABEL.split("_")[-1][:-1])
BREADTH_FLOOR = 2 * max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
SPREAD_BPS = SETUP["costs"]["spread_bps"]
SESSION_CALENDAR = SETUP["decision"]["session_calendar"]
PERIODS_PER_YEAR = SETUP["evaluation"]["periods_per_year"]

print(f"Sample: {START_DATE} to {END_DATE}")
print(f"  Development period, used everywhere below:  {START_DATE} to {HOLDOUT_START}")
print(f"  Holdout, not read by this notebook:         {HOLDOUT_START} to {HOLDOUT_END}")
print(f"Universe: {len(DECLARED_PAIRS)} currency pairs declared")
print(
    f"  Up to {BREADTH_FLOOR // 2} bought and {BREADTH_FLOOR // 2} sold at once, so all "
    f"{BREADTH_FLOOR} must be quoting on a rebalancing date to fill the book"
)
print(
    f"Assumed cost: a spread of {SPREAD_BPS['major_pairs'][0]} to {SPREAD_BPS['major_pairs'][-1]} "
    f"basis points on pairs quoted against the US dollar, and "
    f"{SPREAD_BPS['cross_pairs'][0]} to {SPREAD_BPS['cross_pairs'][-1]} on the rest, charged once "
    f"on the way in and once on the way out"
)
print(
    f"Forecast horizons: {', '.join(f'{h} day' if h == 1 else f'{h} days' for h in HORIZONS)} "
    f"ahead; the {PRIMARY_HORIZON}-day horizon is the primary one and sets how often positions "
    f"change"
)
print(
    f"Calendars: {SESSION_CALENDAR} decides which day a bar belongs to, "
    f"{SETUP['evaluation']['calendar']} counts the walk-forward windows"
)

# %% [markdown]
# ## A. Orientation
#
# ### What a currency pair is
#
# A currency pair is a price of one currency in units of another. `EUR_USD` is how many US dollars
# one euro buys; the first currency is the **base**, the second is the **quote**, and the number is
# the quote currency per unit of base. Buying the pair means buying euros and paying for them in
# dollars, so every position is long one currency and short another at the same time. There is no
# cash side to a currency trade the way there is for a stock, which has one consequence that
# matters below: selling a pair short is the same operation as buying it, run the other way, and
# costs the same. Nothing has to be borrowed.
#
# A pair with the US dollar on one side is a **dollar pair**, and there are seven of them here. A
# pair without one - `EUR_JPY`, `AUD_NZD` - is a **cross**. Crosses are quoted by combining two
# dollar pairs, so a bank quoting `EUR_JPY` is standing behind two prices rather than one, and
# charges accordingly. Section B.3 is where that shows up as a number.
#
# ### Why sorting currency pairs is a strategy
#
# The strategy takes no view on any single currency. Once a day it sorts the twenty pairs by
# something read off their recent prices - `setup.yaml::mapping.entry_logic` names **momentum**,
# how much a pair has moved over a trailing window, and **carry**, the interest-rate difference
# between the two currencies, which a holder of the position earns or pays for keeping it
# overnight - buys the pairs at the top of the ordering, and sells the ones at the bottom. What it
# is betting on is that the ordering carries: that a pair near the top today is more likely than
# not to be above average tomorrow. Whether that is true is a question for Chapter 7 onwards. What
# this notebook asks is whether the data could support the attempt.
#
# A strategy of that shape needs many instruments moving somewhat independently of each other.
# Sorting twenty pairs that all rise and fall together is one bet dressed as twenty, and
# Section B.2 measures which of the two this universe is.
#
# ### The market has no closing bell
#
# Spot currencies trade continuously from Sunday evening to Friday evening, New York time. There is
# no exchange and no closing auction, so "the daily close" is a convention rather than an event:
# somebody has to choose an instant, and `setup.yaml::decision.snapshot` chooses 5PM in New York,
# which is the rollover the interbank market treats as the start of a new value date. The calendar
# named in `decision.session_calendar` implements that rollover, so a four-hour bar printed after
# 5PM counts toward the next trading day rather than the one whose date it carries.
#
# ### The three questions this notebook asks
#
# 1. **Does the universe quote when the strategy trades?** Positions change once a day, and both
#    sides of the book have to be filled on each of those days.
# 2. **Is a typical price move worth more than it costs to capture?** Every round trip crosses the
#    spread twice, and the spread is wider on some pairs than others.
# 3. **Is there enough history to evaluate this honestly?** Enough to split into training and
#    validation periods several times over, with the holdout left untouched.

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 Load the data and look at the universe
#
# The loader returns four-hour bars. Two steps turn them into the daily series the strategy
# decides on.
#
# First, each bar is assigned to a trading day by the calendar, which applies the 5PM New York
# rollover described above. Second, within each day the **decision bar** is the latest four-hour
# timestamp that printed anywhere in the universe. Taking the maximum across the universe rather
# than one pair at a time is what makes the panel one instant wide: a pair with no bar at that
# timestamp is simply absent from that day, rather than carried forward at a stale price of its
# own. It is also what Section B.2 counts.
#
# That leaves one thing to check. A decision taken at the daily cut-off can only act on a price
# that had already printed, so the bar picked should be the last one the day scheduled - and if the
# whole universe were missing its final bar, the rule above would quietly move the snapshot hours
# earlier and nothing would look wrong. The calendar declares when each trading day closes, so the
# check is that the chosen bar sits within one four-hour bar of that close. Two further properties
# are checked before anything is computed: that the set of pairs in the data is exactly the set
# `setup.yaml` declares, in both directions, since a pair present in the data and missing from the
# configuration would be dropped by the cost join in B.3 without changing any row count; and that
# no close is at or below zero, since every ratio below divides by one.

# %%
bars = load_fx_pairs(start_date=START_DATE, end_date=END_DATE)
calendar = TradingCalendar(SESSION_CALENDAR)
sessions = calendar.get_sessions(pd.DatetimeIndex(bars["timestamp"].to_pandas()))
bars = bars.with_columns(pl.Series("session", sessions.values).cast(pl.Date)).drop_nulls("session")

research = bars.filter(pl.col("session") < pl.lit(HOLDOUT_START).str.to_date())
decision_bar = research.group_by("session").agg(pl.col("timestamp").max().alias("timestamp"))
daily = research.join(decision_bar, on=["session", "timestamp"]).sort(["symbol", "session"])
returns = daily.with_columns(pl.col("close").pct_change().over("symbol").alias("ret"))

# %% [markdown]
# The calendar's schedule gives the declared close of every trading day, and the three checks below
# run against it and against the panel that was just built. Both sides of the gap are converted to
# seconds since the epoch before being subtracted, so that the comparison holds whether the bar
# timestamps arrive with a time zone attached or without one.

# %%
schedule = calendar.calendar.schedule(pd.Timestamp(START_DATE), pd.Timestamp(HOLDOUT_START))
declared_close = pl.DataFrame(
    {
        "session": pd.Series(schedule.index.date),
        "close_at": schedule["market_close"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
    }
).with_columns(pl.col("session").cast(pl.Date))
to_close = (pl.col("close_at").dt.epoch("s") - pl.col("timestamp").dt.epoch("s")) // 60
lag_minutes = decision_bar.join(declared_close, "session", how="left").select(to_close).to_series()

loaded = set(daily["symbol"].unique().to_list())
assert loaded == set(DECLARED_PAIRS), (
    f"data and setup.yaml disagree: {loaded ^ set(DECLARED_PAIRS)}"
)
assert daily["close"].min() > 0, "a non-positive close is not a denominator"
assert lag_minutes.max() <= 240, "a decision bar sits more than one bar before the declared close"
print(
    f"{daily['symbol'].n_unique()} pairs, {len(daily):,} daily closes, "
    f"{daily['session'].min()} to {daily['session'].max()}\n"
    f"widest gap from a decision bar to the {SESSION_CALENDAR} close: {lag_minutes.max()} minutes, "
    f"the width of one bar"
)

# %% [markdown]
# Twenty tickers are a list, not a description. The way the market groups them is the way the cost
# model already groups them, by whether the US dollar is one of the two currencies, so that is how
# the table below is sorted. Alongside each pair it shows how large a typical day's move is, in
# **basis points** - one basis point is one hundredth of one percent, the conventional unit for
# quantities this small - and the annualized volatility, which is the standard deviation of the
# daily return scaled to a year so that pairs can be compared on a familiar scale.
#
# Two things are worth noticing before anything is computed from this panel. Every pair has a price
# on every trading day of the sample, so nothing enters or leaves the universe and no eligibility
# rule is needed. And the pairs are not interchangeable: the widest daily move in the table is
# more than twice the narrowest, which is the first hint that a single cost line drawn across all
# twenty would answer the question for no pair in particular.

# %%
universe = (
    returns.group_by("symbol")
    .agg(
        pl.len().alias("days"),
        (pl.col("ret").abs().median() * 1e4).round(1).alias("median_move_bps"),
        (pl.col("ret").std() * np.sqrt(PERIODS_PER_YEAR) * 100).round(1).alias("annual_vol_pct"),
    )
    .with_columns(
        pl.col("symbol").str.split("_").list.first().alias("base"),
        pl.col("symbol").str.split("_").list.last().alias("quote"),
        pl.when(pl.col("symbol").str.contains("USD"))
        .then(pl.lit("dollar pair"))
        .otherwise(pl.lit("cross"))
        .alias("quoted_as"),
    )
    .select("quoted_as", "symbol", "base", "quote", "days", "median_move_bps", "annual_vol_pct")
    .sort(["quoted_as", "symbol"], descending=[True, False])
)
with pl.Config(tbl_rows=universe.height, tbl_cols=universe.width):
    display(universe)

# %% [markdown]
# ### B.2 How many pairs are quoting when the strategy trades
#
# A book that buys ten pairs and sells ten others needs all twenty quoting on the date it
# rebalances. The line below counts them at each decision bar. Against it, each dot marks a
# four-hour bar at which at least one pair did not print - the cadence an alternative design would
# trade on, drawn only where it falls short, since most four-hour bars carry the full universe.
# The dots are why the daily snapshot exists: a timestamp some pairs did not print at cannot
# support a decision across the whole universe, and the daily rule picks the instant they all did.

# %%
snap = daily.group_by("session").agg(pl.col("symbol").n_unique().alias("n")).sort("session")
grid = research.group_by("timestamp").agg(pl.col("symbol").n_unique().alias("n")).sort("timestamp")

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
gaps = grid.filter(pl.col("n") < BREADTH_FLOOR)
floor = dict(color=COLORS["copper"], lw=5, alpha=0.35, zorder=1)
ax.axhline(BREADTH_FLOOR, label="positions to fill on both sides", **floor)
ax.plot(snap["session"], snap["n"], color=COLORS["blue"], lw=1.4, label="decision bar", zorder=3)
ax.plot(
    gaps["timestamp"],
    gaps["n"],
    ".",
    ms=3,
    color=COLORS["neutral"],
    label="four-hour bar missing pairs",
)
ax.set_ylim(0, len(DECLARED_PAIRS) + 2)
ax.set_ylabel("Pairs quoting")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
add_message_title(
    ax,
    "Every pair quotes at the decision bar; the four-hour grid drops some",
    subtitle="Pairs with a price at each daily snapshot, and at each four-hour bar that falls short",
)
show_with_alt(
    fig,
    "Count of pairs quoting, plotted against time from 2011 to 2024. A solid band sits at "
    "the full universe for the whole history, because every daily decision bar carries "
    "every pair. Below it, scattered points mark four-hour bars that carry fewer - dense "
    "through 2011 to 2013 and again around 2019 and 2020, and rare in between.",
)

# %% [markdown]
# Twenty pairs are not twenty bets. Only eight currencies appear across the twenty, so each one
# turns up in several pairs at once, as the table below counts. When the euro moves against
# everything, every pair with a euro leg moves with it, and a portfolio holding all of them holds
# one position several times over.

# %%
legs = (
    pl.concat([universe.select(currency="base"), universe.select(currency="quote")])
    .group_by("currency")
    .agg(pl.len().alias("appears_in_pairs"))
    .sort(["appears_in_pairs", "currency"], descending=[True, False])
)
display(legs)

# %% [markdown]
# The figure below turns that overlap into a single number. The daily returns of the twenty pairs
# are correlated with each other, and the eigenvalues of that correlation matrix say how the
# variance is distributed across independent directions: a few large eigenvalues mean most of the
# movement is a handful of shared themes, while twenty equal ones would mean twenty unrelated
# series. The **participation ratio**, $(\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$, summarizes that
# distribution in one number. It equals the number of pairs when they are independent and one when
# they all move identically, so it reads as the number of independent bets the cross-section
# actually carries.

# %%
wide = returns.select("symbol", "session", "ret").pivot(index="session", on="symbol", values="ret")
spectrum = np.corrcoef(wide.drop("session").drop_nulls().to_numpy(), rowvar=False)
eigenvalues = np.sort(np.linalg.eigvalsh(spectrum))[::-1]
share = eigenvalues / eigenvalues.sum()
participation = float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
pc = np.arange(1, len(share) + 1)
ax.bar(pc, share * 100, color=COLORS["blue"], width=0.7, label="per component")
ax.plot(pc, np.cumsum(share) * 100, "o-", color=COLORS["amber"], lw=1.4, ms=3, label="cumulative")
ax.axvline(participation, color=COLORS["copper"], ls="--", lw=1.5, label="independent bets")
ax.set_xlabel("Principal component of the daily return correlation matrix")
ax.set_ylabel("Variance explained (%)")
ax.set_xticks(pc)
ax.legend(frameon=False, fontsize=8, loc="center right")
add_message_title(
    ax,
    "Shared currencies leave far fewer bets than there are pairs",
    subtitle="Variance explained by each component, with the participation ratio marked",
)
show_with_alt(
    fig,
    "Bars of the variance each principal component of the daily return correlation matrix "
    "explains, falling from about thirty percent for the first to near zero by the "
    "eighth, with a cumulative curve rising over them and flattening at one hundred "
    "percent. A dashed vertical rule marks the participation ratio at just over five, "
    "well short of the twenty pairs the matrix is built from.",
)

# %% [markdown]
# ### B.3 What a round trip costs, and what a move is worth
#
# Buying a pair and later selling it crosses the spread twice: once paying the higher of the two
# prices on the way in, once receiving the lower on the way out. `setup.yaml::costs` states that
# spread as a range in basis points rather than a single number, one range for the dollar pairs and
# a wider one for the crosses, for the reason Section A gave - a cross has to be assembled from two
# dollar quotes, and the bank assembling it charges for both. The top of each range is what gets
# charged here, which is the conservative end of the assumption.
#
# This data carries no bid and no ask, only what traded, so the spread cannot be measured from it.
# The declared assumption is what gets charged, and `15_costs` re-runs the strategy under harsher
# ones to see how much the answer depends on it. Because the assumption has only the two levels the
# configuration declares - one for the dollar pairs, one for the crosses - the cost across this
# universe is two numbers rather than a curve, and they are printed rather than drawn.

# %%
cost = universe.select(
    "symbol",
    pl.when(pl.col("quoted_as") == "dollar pair")
    .then(2 * SPREAD_BPS["major_pairs"][-1])
    .otherwise(2 * SPREAD_BPS["cross_pairs"][-1])
    .alias("cost_bps"),
).sort(["cost_bps", "symbol"])
COST_BPS = float(cost["cost_bps"].median())
by_class = cost.group_by("cost_bps").len().sort("cost_bps").rows()
print(
    "Assumed round trip: "
    + " | ".join(f"{n} pairs at {bps} bps" for bps, n in by_class)
    + f" | universe median {COST_BPS:.0f} bps"
)

# %% [markdown]
# Because the cost differs across the universe, a single cost line drawn across raw returns would
# answer the question for no pair in particular: a move that clears the charge on `EUR_USD` need
# not clear it on `GBP_JPY`. Each move is therefore divided by what its own pair charges, which
# puts break-even at 1 for every pair whatever its own cost happens to be, and lets every candidate
# horizon sit on one axis.
#
# The chart below is an **exceedance curve**, and it reads from the right: for each multiple on the
# horizontal axis, the curve gives the fraction of moves at least that large. Where a curve crosses
# the line at 1 is the fraction of moves bigger than the cost of trading them. It is drawn over
# every pair and every day of the development period, which is the whole panel here, because
# Section B.1 established that all twenty pairs quote throughout and no eligibility rule removes
# any of them.
#
# One thing this chart is not. It is the distribution of how far prices move, ignoring direction.
# It is not the return a strategy would earn: nothing here is signed, nothing waits a bar to enter,
# and nothing is restricted to the moment positions change. Whether the strategy can pick which
# moves to be on the right side of is the question Chapter 7 onwards asks. This is only whether the
# moves are large enough to be worth trying.

# %%
moves = (
    daily.with_columns(
        pl.col("close").pct_change(h).abs().over("symbol").alias(f"h{h}") for h in HORIZONS
    )
    .join(cost, "symbol")
    .with_columns(pl.col(f"h{h}") * 1e4 / pl.col("cost_bps") for h in HORIZONS)
)
spacing = pl.col("timestamp").diff().over("symbol") == pl.duration(hours=4)
intraday = (
    research.sort(["symbol", "timestamp"])
    .with_columns(pl.col("close").pct_change().abs().over("symbol").alias("bar"))
    .filter(spacing)
    .join(cost, "symbol")
    .with_columns(pl.col("bar") * 1e4 / pl.col("cost_bps"))
)
curves = [("four-hour bar", intraday["bar"])]
curves += [(f"{h}-day move", moves[f"h{h}"]) for h in HORIZONS]

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
styles = [
    (COLORS["neutral"], "--"),
    (COLORS["blue"], "-"),
    (COLORS["amber"], "-"),
    (COLORS["copper"], "-"),
]
for (label, series), (color, style) in zip(curves, styles, strict=True):
    magnitude, fraction = exceedance_curve(series.drop_nulls().to_numpy())
    ax.plot(magnitude, fraction, color=color, ls=style, lw=1.6, label=label)
ax.axvline(1, color=COLORS["slate"], ls=":", lw=1.8, label="break-even on the round trip")
ax.set(xscale="log", xlim=(0.01, 200))
ax.set_xlabel("Absolute move as a multiple of the pair's own round trip (log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Each longer horizon puts more of the move clear of the round trip",
    subtitle="Absolute returns scaled by each pair's own assumed cost, development period",
)
show_with_alt(
    fig,
    "Four curves on a logarithmic horizontal axis, each giving the fraction of absolute "
    "moves at least as large as a given multiple of the pair's own round-trip cost. All "
    "four fall away to the right, and they are ordered by horizon throughout: the "
    "four-hour bar is lowest and the twenty-one-day move highest. At the break-even "
    "multiple of one, marked by a dotted rule, the four-hour curve has already dropped "
    "below half while the longer horizons are still near the top of the axis.",
)

# %% [markdown]
# ### B.4 How much of one day's return carries into the next
#
# A position opened at one daily snapshot and closed at the next earns that pair's return over the
# interval. Before building anything that forecasts that return, it is worth asking how much of it
# the pair's own recent history already accounts for. If pairs that rose yesterday tend to rise
# again, the simplest imaginable ordering - buy yesterday's risers - is already a strategy, and the
# daily schedule has to be fast enough to act on that tendency before it fades.
#
# The measurement is an **autocorrelation**: the correlation between a pair's return on one day and
# its return some number of days later. Plotted against that number of days, it shows how much of
# the series its own past accounts for, and how quickly that fades.
#
# It is computed inside each pair and then averaged across pairs. Stacking twenty pairs into one
# long series and correlating that returns a number too, and the number is wrong: at every point
# where one pair's history ends and the next begins, it correlates the euro against the dollar with
# the Australian dollar against the New Zealand dollar. The shaded region shows how much the result
# varies from pair to pair, and the band around zero shows how large a correlation could plausibly
# be if a pair's returns carried no information about their own past at all.
#
# What this bounds is the raw return series, not a feature built over many days. Chapter 7 is where
# a candidate signal is measured against the returns that follow it, and Chapter 8 is where the
# features it reads are constructed.

# %%
acf = panel_acf(returns, entity_col="symbol", value_col="ret", max_lags=max(HORIZONS))
acf = acf.filter(pl.col("lag") > 0)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
band, spread = dict(alpha=0.3, zorder=1), dict(alpha=0.15, zorder=2)
ax.axhspan(
    -acf["band"][0],
    acf["band"][0],
    color=COLORS["copper"],
    label="range expected from no information",
    **band,
)
ax.fill_between(
    acf["lag"],
    acf["acf_p10"],
    acf["acf_p90"],
    color=COLORS["blue"],
    label="10th to 90th percentile across pairs",
    **spread,
)
ax.bar(
    acf["lag"], acf["acf"], color=COLORS["blue"], width=0.6, zorder=3, label="average across pairs"
)
ax.set(xlim=(0.4, max(HORIZONS) + 0.6), ylim=(-0.06, 0.06))
ax.set_xticks(range(1, max(HORIZONS) + 1, 2))
ax.set_xlabel("Days between the two returns")
ax.set_ylabel("Autocorrelation of the daily return")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
add_message_title(
    ax,
    "A pair's own past return accounts for almost none of its next one",
    subtitle="Averaged within each pair",
)
show_with_alt(
    fig,
    "Bars of the average within-pair autocorrelation of daily returns at lags one to "
    "twenty-one. Every bar is small and they alternate in sign with no run of one "
    "direction, and all of them sit inside the shaded band the chart marks as the range "
    "expected from no information. A wider shaded band behind them gives the tenth to "
    "ninetieth percentile across pairs, and it too straddles zero at every lag.",
)

# %% [markdown]
# ### B.5 Move size against cost
#
# Two numbers summarise what B.3 drew. The first is the median absolute move over one rebalancing
# interval divided by the median round trip, which says how much larger a typical move is than a
# typical cost. The second is the share of moves larger than what their own pair charges, which is
# where the exceedance curve crosses its break-even line. The same share at the four-hour bar is
# reported alongside, because Section C.1 uses the comparison between the two.
#
# Neither says the strategy earns anything. Both count a move down exactly as they count a move up,
# and nothing here decides which side of it a position would have been on. What they rule out is
# the case where the design fails immediately, because a typical move is smaller than the cost of
# capturing it.

# %%
primary = f"h{PRIMARY_HORIZON}"
cleared = moves.drop_nulls(primary)
median_move_bps = float((cleared[primary] * cleared["cost_bps"]).median())
clears_cost = float((cleared[primary] > 1).mean())
clears_intraday = float((intraday["bar"].drop_nulls() > 1).mean())
print(
    f"Round trip {cost['cost_bps'].min()} to {cost['cost_bps'].max()} bps across pairs, median "
    f"{COST_BPS:.0f} bps\n"
    f"Median absolute {PRIMARY_HORIZON}-day move {median_move_bps:.1f} bps, "
    f"{median_move_bps / COST_BPS:.1f}x the median round trip\n"
    f"Share of moves larger than their own pair's round trip {clears_cost:.3f} at one day, "
    f"{clears_intraday:.3f} at the four-hour bar"
)

# %% [markdown] tags=["results"]
# The assumed round trip is 6 bps on the seven pairs quoted against the dollar and 16 bps on the
# thirteen crosses, a median of 16 bps. The median absolute one-day move is 31.3 bps, 2.0 times
# that median, and 0.776 of one-day moves are larger than the cost their own pair would charge. At
# the four-hour bar that share falls to 0.461.

# %% [markdown]
# ## C. Design decisions
#
# The sections above are evidence. This section is where that evidence meets the choices recorded
# in `setup.yaml`, and says what each one rests on.
#
# ### C.1 How often to rebalance
#
# `setup.yaml::decision.cadence` sorts the pairs at the daily snapshot and trades on the next bar's
# open. Section B.3 supports that interval over the faster alternative: a four-hour move clears its
# round trip about as often as not, where a one-day move clears it far more often, so the shorter
# interval spends the same cost on a smaller move. Section B.2 supports it from the other side,
# because the four-hour grid drops pairs that the daily snapshot keeps, and an ordering computed on
# a bar that part of the universe did not print at ranks only the part that did.
#
# ### C.2 What would send this design back
#
# A feasibility study is only useful if some result would have stopped it. Three would, and each is
# measured where its evidence exists rather than here.
#
# The one this notebook could have produced is a cost failure: if a typical move were smaller than
# the round trip needed to capture it, the ordering would pay more to trade than the move it is
# trying to catch. Section B.5 is that measurement, and Chapter 18 repeats it against the trades a
# backtest actually places rather than against raw moves.
#
# The other two are outcomes of the strategy rather than properties of the data. Chapter 7 asks
# whether the ordering has any relationship at all to the returns that follow it, at any horizon.
# And an ordering whose result comes entirely from being long or short the US dollar across the
# board is a directional bet on one currency wearing twenty tickers rather than a cross-sectional
# strategy; `12_model_analysis` measures how much of what a model reads is that single exposure.
#
# ### C.3 What the strategy does with the ordering
#
# `setup.yaml::mapping.class` holds both sides: it buys the pairs at the top of the ordering and
# sells the ones at the bottom. Section A gave the reason this is cheap here where it is expensive
# in equities - a pair is already a relative price, so the sold side needs no borrow and costs what
# the bought side costs. Holding both sides also puts a position on every pair the ordering covers,
# which is what spends the whole of the breadth Section B.2 measures.
#
# Each position gets an equal share of the money. A weighting optimised for risk would fold an
# estimate of how the pairs move together into the result, and the ordering's own contribution
# could no longer be separated from that estimate's. Chapter 17 compares the alternatives with the
# ordering held fixed.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 How much an evaluation has to spend
#
# A panel of four-hour bars looks large, but a strategy that changes its positions once a day does
# not get to treat every row as an independent opportunity. What it spends is decision dates. The
# calendar says how many trading days the development period holds, and comparing that against the
# days actually observed, and against the number of pair-days a complete panel would carry, says
# whether anything is missing.

# %%
expected = calendar.trading_days_between(pd.Timestamp(START_DATE), pd.Timestamp(HOLDOUT_START))
print(
    f"Calendar trading days {expected:,} | observed {daily['session'].n_unique():,} "
    f"| pair-days {len(daily):,} of {expected * len(DECLARED_PAIRS):,}"
)

# %% [markdown]
# ### D.2 The folds
#
# A model is fitted on one stretch of history and evaluated on the stretch that follows it, then
# the pair moves forward and the process repeats. Each fit-then-evaluate pair is a **fold**, and
# evaluating this way is called **walk-forward**, because the split always runs in the direction
# time does.
#
# One detail decides whether the evaluation is honest. The return being predicted lands one day
# ahead, so a training row dated on the last day of its block is labelled with a price from after
# the block ends. Validating on the day immediately after training would score the model on data it
# had partly seen already. The fix is to leave a gap between the two, at least as wide as the
# horizon, and that gap is called **purging**. Its width comes from `labels.buffer` in `setup.yaml`.
# The two longer variants declare wider gaps of their own in `labels.variant_buffers` - five days
# for `fwd_ret_5d`, twenty-one for `fwd_ret_21d` - and every stage that generates folds resolves the
# gap for the label it is about to train on, so the timeline below is the primary label's design
# rather than one design shared by all three.
#
# The splitter is given the whole sample, holdout included, and applies the holdout boundary itself
# from `evaluation.holdout_start`, which is what every later stage does too. It is handed trading
# days and no prices, so nothing the holdout contains reaches a number computed above.
#
# `generate_cv_splits` numbers folds from zero backwards from the most recent, so fold 0 is the last
# one before the holdout and the highest number is the earliest. The figure draws them earliest-first
# and labels each with that number, which is why the labels count down; every later stage prints
# the same ones. The three assertions below establish what the figure cannot: the gap is one trading
# day against training blocks measured in years, far too narrow to see, so only counting it off the
# timeline can confirm it matches the label horizon. The other two check that the number of folds is
# the number `setup.yaml` declares, and that no validation window reaches into the holdout. The
# figure then draws the boundaries the splitter returned rather than recomputing them, so the
# picture and the folds cannot disagree.

# %%
timeline = bars.select(pl.col("session").alias("timestamp")).unique().sort("timestamp")
splits = generate_cv_splits(
    timeline,
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)

grid = timeline["timestamp"].to_numpy()
purge_gaps = {
    int(((grid > np.datetime64(s["train_end"])) & (grid < np.datetime64(s["val_start"]))).sum())
    for s in splits
}
last_val = max(split["val_end"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < np.datetime64(HOLDOUT_START), "a fold reaches into the holdout"
assert purge_gaps == {PRIMARY_HORIZON}, "a purge gap is not the primary label horizon"
print(
    f"{len(splits)} folds | purge gap {min(purge_gaps)} trading day at every boundary, from "
    f"labels.buffer {LABEL_BUFFER} against the {PRIMARY_HORIZON}-day primary horizon | last "
    f"validation ends {pd.Timestamp(last_val).date()}, the holdout opens {HOLDOUT_START}"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
ax.set_xlabel("Trading day")
add_message_title(
    ax,
    "Folds roll forward and stop short of the holdout",
    subtitle="Boundaries as generate_cv_splits returned them; the one-day purge is too narrow to see",
)
show_with_alt(
    fig,
    "One horizontal bar per walk-forward fold, each a long training stretch running "
    "straight into a shorter validation stretch. The bars step up and to the right from "
    "fold seven at the bottom of the history to fold zero at the top, so the lowest fold "
    "number is the most recent window. A shaded region on the right marks the holdout, "
    "and no bar reaches into it.",
)

# %% [markdown]
# ## E. What this notebook hands on
#
# Nothing. The universe is fixed and declared in `setup.yaml::universe.symbols`, which is where
# this notebook reads it and where a later stage that needs the list reads it too. Section B.1
# established that every declared pair quotes on every trading day of the sample, so there is no
# eligibility table for a later notebook to filter on.

# %% [markdown]
# ## F. What the evidence says about each setting
#
# One row per setting: the evidence behind it, and the condition under which a reader working on
# their own data would choose differently.
#
# | Setting | Evidence | Choose differently when |
# |---|---|---|
# | `universe.symbols` | B.2, pairs quoting on each decision date and the independent bets they carry | a pair stops quoting at the daily snapshot, or the independent bets fall further |
# | `decision.cadence` | B.2 pairs quoting by snapshot, B.3 move sizes against cost | a shorter interval starts clearing its round trip on a grid that carries the whole universe |
# | `costs.spread_bps` | B.3, the declared range charged at its wide end for each of the two groups | spreads estimated from quotes sit outside the declared range |
# | `evaluation.n_splits` | D.1 trading days available, D.2 fold boundaries | the folds no longer fit the development period |

# %%
print(
    f"universe.n_assets {SETUP['universe']['n_assets']}, pairs quoting per decision date "
    f"{snap['n'].min()} to {snap['n'].max()}, short of {BREADTH_FLOOR} on "
    f"{snap.filter(pl.col('n') < BREADTH_FLOOR).height} of {len(snap)} dates, independent bets "
    f"{participation:.1f}\n"
    f"decision.cadence {SETUP['decision']['cadence']} | labels.primary {PRIMARY_LABEL} | "
    f"costs.spread_bps per crossing, dollar pairs {SPREAD_BPS['major_pairs']}, crosses "
    f"{SPREAD_BPS['cross_pairs']}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {pd.Timestamp(last_val).date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# All twenty pairs quote on every one of the 3,355 decision dates, so the book can always be
# filled, but the correlation spectrum puts the independent bets at 5.3 - about a quarter of the
# nominal count. Eight folds are generated, the last validation window ending 2023-12-28.

# %% [markdown]
# ## Key takeaways
#
# 1. **Count the universe at the moment the strategy decides, and take the price from that same
#    moment.** Falling back to each instrument's own last bar fills a missing price with a stale
#    earlier one, and the panel is then no longer one instant wide.
# 2. **Count the independent bets, not the instruments.** Where members of a universe are exposed
#    to the same thing, the participation ratio of the correlation spectrum is the breadth a
#    portfolio actually gets, and it can be a small fraction of the count.
# 3. **Scale each move by its own instrument's cost before comparing horizons**, so one axis
#    answers what fraction of moves is larger than the cost of trading them, for every instrument
#    at once.
# 4. **Compute a panel autocorrelation inside each instrument, then average.** Stacking instruments
#    into one series measures the joins between them.
# 5. **Derive the decision timestamp from a calendar, not only from the data.** A rule that takes
#    the last bar observed will silently move the snapshot when the data is incomplete, and
#    checking it against the declared close is what turns that into an error rather than a number.
#
# ### Known limitations
#
# - The spread is an assumption, not a measurement: this data records what traded, not what was
#   quoted, so a pair whose realized spread sits outside the declared range does not show up here.
# - The swap points a position pays or earns for being held overnight are priced at no stage of
#   this case study, so every cost figure it reports is the cost of crossing the spread and not the
#   cost of holding the position. A pair whose interest-rate difference runs against the position
#   costs more than anything here charges.
# - The twenty pairs were chosen knowing which of them are liquid today, and no rule inside this
#   notebook could remove a bias in the list itself.
#
# **Next**: labels at the declared horizons, built on this development period.
