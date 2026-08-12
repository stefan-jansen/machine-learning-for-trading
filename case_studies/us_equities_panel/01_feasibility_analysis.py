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
# # US Equities Panel: Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. It trades the broad US stock
# market, sorts the stocks it is allowed to hold once a day, buys the tenth that rank highest and
# sells the tenth that rank lowest. That file says how often positions change, what a trade is
# assumed to cost, and how the history is divided between designing the strategy and testing it.
# This notebook checks each of those assumptions against the data and reports what it finds.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Decide which stocks a strategy was allowed to hold on a given date using only information that
#   existed before that date, and count how many that leaves on each date it trades
# - Tell an adjusted price from the price that printed, and use each for what it is right for:
#   returns from the adjusted series, and the price and volume screen from what actually traded
# - Compare two ways of charging for a trade - a fraction of the price, and a fixed number of cents
#   per share - and decide which one a price distribution this wide can support
# - Read off one chart what fraction of price moves are larger than the cost of trading them
# - Measure how much of one day's return carries into the next, computing the correlation inside
#   each stock rather than across thousands of stocks stacked into one series
# - Check that a walk-forward split of the history fits the sample available and leaves the test
#   period unread
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. This notebook reads the daily US equity panel and
# `config/setup.yaml`, and writes nothing.
#
# ## Prerequisites
#
# None beyond what the sections below define. A reader who has not built a point-in-time universe
# or split a sample for walk-forward evaluation will find both explained where they are first used.

# %%
"""US Equities Panel Case Study - Feasibility Analysis."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.splitters.calendar import TradingCalendar

from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import load_us_equities
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
START_DATE = "1990-01-01"
END_DATE = "2018-03-31"  # the archive's last session
MIN_PRICE = 5.0
MIN_ADV_USD = 1_000_000
ADV_WINDOW = 21

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads those
# values rather than repeating them, so the two can never disagree. Four groups of settings matter
# here, and each one decides something the sections below test.
#
# **How the history is divided.** The archive runs from 1990 to the end of the first quarter of
# 2018. The last two years and a quarter are the *holdout*: a stretch of history that is not looked
# at while the strategy is being designed, so that when it is finally evaluated there, the result is
# not a rehearsal of choices already tuned on the same data. Everything computed in this notebook
# uses the earlier part, called the development period, and `holdout_start` is where the line falls.
#
# **What the strategy trades.** Not every stock in the archive, and the rule deciding which is not
# in `setup.yaml`: it is three thresholds declared in the parameters cell above - a minimum price, a
# minimum daily turnover, and the number of sessions the turnover is averaged over. Section B.2
# explains what each one is for and applies them. What `setup.yaml` does fix is how many positions
# have to be filled: the sort takes as many as fifty positions on each side, so at least a hundred
# stocks have to pass the screen on any date the strategy trades.
#
# **What a trade is assumed to cost.** A round trip is charged as a fraction of the money traded,
# from a per-leg range that `setup.yaml` states as a band rather than a point; the midpoint of that
# band, doubled for the two legs, is what gets charged. A second way of charging - a fixed number of
# cents per share - is carried alongside as a comparison, and Section B.3 is where the two are put
# against each other.
#
# **What is being predicted.** The return over the next trading day, with variants looking five and
# twenty-one days ahead. The one-day horizon is the primary one, and it sets both how often
# positions change and how wide a gap has to separate training data from validation data.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = SETUP["labels"]["buffer"]
VARIANT_BUFFERS = SETUP["labels"]["variant_buffers"]
LABELS = [PRIMARY_LABEL, *SETUP["labels"]["variants"]]
HORIZONS = sorted(int(name.rsplit("_", 1)[-1].removesuffix("d")) for name in LABELS)
PRIMARY_HORIZON = int(PRIMARY_LABEL.rsplit("_", 1)[-1].removesuffix("d"))
BREADTH_FLOOR = 2 * max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
PER_LEG_RANGE = SETUP["costs"]["per_leg_cost_bps_range"]
COST_BPS = 2 * sum(PER_LEG_RANGE) / len(PER_LEG_RANGE)
HALF_SPREADS = [c for c in SETUP["backtest"]["sweep"]["cost_grid_half_spread_usd"] if c > 0]
HALF_SPREAD_USD = HALF_SPREADS[len(HALF_SPREADS) // 2]
PER_SHARE = SETUP["costs"]["per_share"]
CALENDAR = SETUP["evaluation"]["calendar"]

print(f"Archive: {START_DATE} to {END_DATE}")
print(f"  Development period, used everywhere below:  {START_DATE} to {HOLDOUT_START}")
print(f"  Holdout, not read by this notebook:         {HOLDOUT_START} to {HOLDOUT_END}")
print(f"Universe: {SETUP['universe']['n_assets']:,} stocks in the archive, screened down each day")
print(
    f"  A stock qualifies on a date when it printed above ${MIN_PRICE:.0f} and traded more than "
    f"${MIN_ADV_USD / 1e6:.0f}M a day on average over the previous {ADV_WINDOW} sessions"
)
print(
    f"  Up to {BREADTH_FLOOR // 2} bought and {BREADTH_FLOOR // 2} sold at once, so at least "
    f"{BREADTH_FLOOR} must qualify on a date for the book to be filled"
)
print(
    f"Assumed cost: {PER_LEG_RANGE[0]} to {PER_LEG_RANGE[1]} basis points of the money traded per "
    f"leg, so {COST_BPS:.0f} bps for a round trip at the midpoint of that band"
)
print(
    f"  Compared against a per-share charge of ${HALF_SPREAD_USD} of spread plus ${PER_SHARE} of "
    f"commission per share, per leg"
)
print(
    f"Forecast horizons: {', '.join(f'{h} day' if h == 1 else f'{h} days' for h in HORIZONS)} "
    f"ahead; the {PRIMARY_HORIZON}-day horizon is the primary one and sets how often positions "
    f"change. Trading calendar: {CALENDAR}"
)

# %% [markdown]
# ## A. Orientation
#
# ### What the archive holds
#
# One row per stock and trading day, for every US common stock that was trading into the first
# quarter of 2018 - including the ones that stopped trading along the way, because they were
# acquired, went private or failed. That inclusion is what makes a universe formed on a past date
# the universe that existed on it. An archive holding only the survivors would let a strategy be
# tested on a list nobody could have written at the time.
#
# Each row carries two versions of the price. The **printed price** is what the exchange quoted that
# day. The **adjusted price** is the same series rescaled backwards so that a stock split or a
# dividend payment does not appear as a price move: a two-for-one split halves the price overnight
# without making anybody poorer, and an unadjusted return would record that as a fall of half.
# Differences of the adjusted series are therefore returns, and differences of the printed series
# are not.
#
# The two are used for different things below, and swapping them is a mistake that runs silently in
# both directions. Returns come from the adjusted series. The screen in Section B.2 comes from the
# printed one, because whether a stock was worth more than five dollars on a date in 1997 is a fact
# about that date, and the adjusted price for that date is a number computed from everything that
# happened afterwards.
#
# ### Why sorting stocks against each other is a strategy
#
# The strategy takes no view on the market as a whole. Once a day it sorts the stocks it is allowed
# to hold by some measure of their recent behaviour, splits the sorted list into ten equal groups -
# **deciles** - buys the top group and sells the bottom one. Selling a stock the portfolio does not
# own means borrowing the shares in order to sell them, and paying a fee to whoever lent them, which
# is a cost the long side does not carry; `setup.yaml::costs` declares an assumption for it.
#
# What the strategy is betting on is that the ordering carries: that a stock near the top today is
# more likely than not to be above average tomorrow. Whether that is true is a question for
# Chapter 7 onwards. What this notebook asks is whether the data could support the attempt.
#
# A strategy of that shape needs breadth. Sorting fifty stocks into deciles leaves five per group,
# and the result is then a story about five companies rather than about the ordering, so it matters
# more that many stocks qualify at once than that any one of them is a good name to hold.
#
# ### The three questions this notebook asks
#
# 1. **Is the tradable universe wide enough on the dates the strategy acts?** Positions change every
#    day, so both ends of the sort have to be fillable on each of them.
# 2. **Is a typical price move worth more than it costs to capture?** Every round trip pays two
#    legs, and the two ways of charging for those legs disagree sharply about cheap stocks.
# 3. **Is there enough history to evaluate this honestly?** Enough to split into training and
#    validation periods many times over, with the holdout left untouched.

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 Load the data and look at the universe
#
# The loader returns one row per stock and session over the whole archive, and everything computed
# below is taken from the development period alone. One property is checked before anything else:
# that the panel holds no more stocks than `setup.yaml` declares.

# %%
panel = load_us_equities(start_date=START_DATE, end_date=END_DATE)
research = panel.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date()).sort(
    ["symbol", "timestamp"]
)

n_declared = SETUP["universe"]["n_assets"]
assert panel["symbol"].n_unique() <= n_declared, "the panel holds more stocks than setup.yaml"
print(
    f"{panel['symbol'].n_unique():,} stocks against {n_declared:,} declared | development period "
    f"{research['symbol'].n_unique():,} stocks, {len(research):,} daily bars, "
    f"{research['timestamp'].n_unique():,} dates, to {research['timestamp'].max()}"
)

# %% [markdown]
# Three thousand tickers are a list, not a description. The property that matters most for
# everything below is how far apart these stocks are in price, because both the screen in
# Section B.2 and the cost comparison in Section B.3 turn on it. The table groups the universe by
# each stock's median printed price over the development period, and shows how much money changed
# hands in a typical day at each level.
#
# *Turnover* is the number of shares that traded multiplied by the price, so it measures money
# rather than shares, and it is the quantity the liquidity part of the screen reads. Read the table
# for the size of the gap: the cheapest group trades a small fraction of what the dearest group
# trades, so a charge quoted in cents per share falls on it very differently. That is the
# observation Section B.3 turns into a chart.

# %%
price_band = (
    pl.when(pl.col("median_close_usd") < MIN_PRICE)
    .then(pl.lit(f"1. under ${MIN_PRICE:.0f}"))
    .when(pl.col("median_close_usd") < 20)
    .then(pl.lit(f"2. ${MIN_PRICE:.0f} to $20"))
    .when(pl.col("median_close_usd") < 100)
    .then(pl.lit("3. $20 to $100"))
    .otherwise(pl.lit("4. $100 and above"))
)
by_stock = research.group_by("symbol").agg(
    pl.col("close").median().alias("median_close_usd"),
    (pl.col("close") * pl.col("volume")).median().alias("turnover_usd"),
    pl.len().alias("bars"),
)
bands = (
    by_stock.with_columns(price_band.alias("price_band"))
    .group_by("price_band")
    .agg(
        pl.len().alias("stocks"),
        pl.col("median_close_usd").median().round(2).alias("median_close_usd"),
        (pl.col("turnover_usd").median() / 1e6).round(2).alias("median_turnover_musd"),
        pl.col("bars").sum().alias("daily_bars"),
    )
    .with_columns((100 * pl.col("daily_bars") / pl.col("daily_bars").sum()).round(1).alias("pct"))
    .sort("price_band")
)
with pl.Config(tbl_rows=bands.height, tbl_cols=bands.width):
    display(bands)

# %% [markdown]
# ### B.2 How many stocks the strategy is allowed to hold when it trades
#
# Not every stock in the archive could have been bought on every date it appears. Two thresholds
# decide, and both are read from information that existed before the date they apply to - a rule
# built that way is called **point-in-time**, and it is the difference between a backtest and a
# rehearsal.
#
# The first threshold is a price floor. A stock trading below a few dollars moves in increments that
# are a large fraction of its own price, and the strategy holds a hundred positions rather than
# betting on one, so admitting them buys noise. The floor is read off the price that printed, for
# the reason Section A gave.
#
# The second is a turnover floor. A position the portfolio wants to open has to be small relative to
# what trades that day, or the act of buying moves the price against the buyer. Turnover is averaged
# over a window of recent sessions rather than read off a single day, because one unusual day is not
# evidence a stock can absorb a position. That average is only an average over a window when the
# window's rows are consecutive sessions, so a stock returning from a trading halt does not qualify
# on volume from before the halt: the sessions are numbered, and a stock qualifies only where the
# window it spans is unbroken. This also means the screen cannot decide anything until the archive
# is that many sessions old.
#
# The count that matters is the one taken on each date the strategy acts, not a total over the
# sample. An average taken over every date hides whether the book could have been filled on the
# dates that decide the result.

# %% [markdown]
# One preliminary. The archive carries stray prints on dates the exchange did not hold a session,
# and a date that is not a session is not a date the strategy can act on. `get_sessions` maps each
# timestamp to the session that settles it, so a date mapping to itself is one the exchange held.
# Numbering those dates in order gives a session counter, which is what makes "twenty-one sessions
# back" mean sessions rather than rows.

# %%
dates = research.select("timestamp").unique().sort("timestamp")
mapped = pl.Series(
    TradingCalendar(CALENDAR)
    .get_sessions(pd.DatetimeIndex(dates["timestamp"].to_list(), tz="UTC"))
    .to_numpy()
).cast(pl.Date)
calendar = dates.filter(mapped == pl.col("timestamp")).with_row_index("session")

# %% [markdown]
# The screen itself. `covered` marks the rows whose trailing window is unbroken, `eligible` marks
# the rows that also clear both thresholds, and `breadth` counts the eligible stocks on each date
# the screen can decide.

# %%
dollar_volume = (pl.col("close") * pl.col("volume")).rolling_mean(ADV_WINDOW)
covered = pl.col("session") - pl.col("session").shift(ADV_WINDOW - 1) == ADV_WINDOW - 1
qualifies = pl.col("covered") & (pl.col("close") > MIN_PRICE) & (pl.col("adv") > MIN_ADV_USD)
screened = (
    research.join(calendar, on="timestamp")
    .sort(["symbol", "timestamp"])
    .with_columns(
        dollar_volume.over("symbol").alias("adv"), covered.over("symbol").alias("covered")
    )
    .with_columns(qualifies.alias("eligible"))
)
breadth = (
    screened.filter("covered")
    .group_by("timestamp")
    .agg(pl.col("eligible").sum().alias("n_eligible"))
    .sort("timestamp")
)
tradable = screened.filter("eligible")

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(
    breadth["timestamp"],
    breadth["n_eligible"],
    color=COLORS["blue"],
    linewidth=0.8,
    label="stocks passing the screen",
)
ax.axhline(
    BREADTH_FLOOR,
    color=COLORS["copper"],
    ls="--",
    lw=1.5,
    label="positions to fill across both sides",
)
ax.set_ylabel("Stocks passing the screen")
ax.set_ylim(0, None)
ax.legend(frameon=False, fontsize=8, loc="upper left")
add_message_title(
    ax,
    "Far more stocks qualify than the strategy has positions to fill",
    subtitle="Stocks clearing the price and turnover thresholds, counted on each date",
)
show_with_alt(
    fig,
    "A single line traces how many stocks pass the screen on each date across the sample. "
    "It rises steadily over the period, dips sharply in the 2008 crisis, recovers to its "
    "highest level, and falls back over the last two years. A dashed horizontal line near "
    "the bottom of the panel marks the number of positions the strategy fills, and the "
    "line stays far above it throughout.",
)

# %% [markdown]
# ### B.3 What a round trip costs, and what a move is worth
#
# `setup.yaml::costs.model` charges a round trip as a fraction of the money traded, and carries a
# per-share charge alongside it as a comparison: a half-spread plus a commission, both quoted in
# cents per share. The two are different kinds of assumption, not different levels of the same one.
# A fraction of the money traded costs the same on a five-dollar stock and a five-hundred-dollar
# one. A fixed number of cents per share is a hundred times heavier on the first than on the second,
# and Section B.1 showed that this universe spans exactly that range.
#
# The chart converts the per-share charge into the same unit as the proportional one - **basis
# points**, one hundredth of one percent - by dividing it by each stock's own median price, and
# draws the result against the declared proportional round trip. The vertical axis is logarithmic
# because the curve spans several orders of magnitude, which is itself the finding: one number
# quoted in cents cannot describe this universe.

# %%
per_share_leg = HALF_SPREAD_USD + PER_SHARE
cost = (
    tradable.group_by("symbol")
    .agg(pl.col("close").median().alias("price"))
    .drop_nulls("price")
    .with_columns((2 * per_share_leg / pl.col("price") * 1e4).alias("per_share_bps"))
    .sort("per_share_bps")
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(range(len(cost)), cost["per_share_bps"], color=COLORS["blue"], lw=1.6, label="per share")
ax.axhline(
    COST_BPS, color=COLORS["copper"], ls="--", lw=1.5, label="declared proportional round trip"
)
ax.set_yscale("log")
ax.set_xlabel("Stocks, ordered from the highest median printed price to the lowest")
ax.set_ylabel("Round-trip cost (bps, log scale)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
add_message_title(
    ax,
    "The per-share charge costs low-priced stocks orders of magnitude more",
    subtitle="The configured per-share spread and commission over each stock's median price",
)
show_with_alt(
    fig,
    "Round-trip cost in basis points on a logarithmic axis, against stocks ordered from "
    "the highest median printed price to the lowest. The curve climbs almost vertically "
    "out of the bottom left, flattens across most of the middle of the range, then turns "
    "up again and rises steeply at the right-hand end. A dashed horizontal line marks the "
    "declared proportional round trip; the curve crosses it about two thirds of the way "
    "along and finishes an order of magnitude above it.",
)

# %% [markdown]
# That is why the production cost model is the proportional one. One proportional number is wrong
# for every stock by a bounded amount; one per-share number is wrong for the cheap end by orders of
# magnitude, and the cheap end is where a broad screen admits the most names.
#
# With one cost line to compare against, moves at each candidate horizon go on one axis. The chart
# below is an **exceedance curve**, and it reads from the right: for each move size on the
# horizontal axis, the curve gives the fraction of moves at least that large. Where a curve crosses
# the cost line is the fraction of moves bigger than the cost of trading them.
#
# It is drawn over the stock-dates the screen admits, not over every row of the panel. A move in a
# stock the strategy was not allowed to hold that day was never an opportunity, and counting it
# would overstate how often a move covers its own cost.
#
# One thing this chart is not. It is the distribution of how far prices move, ignoring direction. It
# is not the return a strategy would earn: nothing here is signed, nothing waits a day to enter, and
# nothing decides which side of a move a position would have been on. Whether the strategy can pick
# that side is the question Chapter 7 onwards asks. This is only whether the moves are large enough
# to be worth trying.

# %% [markdown]
# The forward return has to be counted in sessions, not in rows. A stock's rows are the sessions it
# traded, so a stock that was halted for a week has rows five sessions apart, and reading five rows
# ahead would return a move over a longer stretch of calendar than the label describes. The session
# counter built above is what says which pairs of rows really are the declared horizon apart.

# %%
ahead = {h: pl.col("session").shift(-h) - pl.col("session") == h for h in HORIZONS}
returns = screened.with_columns(
    pl.when(ahead[h].over("symbol"))
    .then((pl.col("adj_close").shift(-h) / pl.col("adj_close") - 1).over("symbol"))
    .alias(f"h{h}")
    for h in HORIZONS
)
moves = returns.filter("eligible")

styles = ((COLORS["blue"], "-"), (COLORS["amber"], "-"), (COLORS["neutral"], "-."))
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for h, (color, ls) in zip(HORIZONS, styles, strict=True):
    magnitude, fraction = exceedance_curve(moves[f"h{h}"].abs().drop_nulls().to_numpy() * 1e4)
    ax.plot(magnitude, fraction, color=color, ls=ls, lw=1.6, label=f"{h}-day move")
ax.axvline(COST_BPS, color=COLORS["slate"], ls=":", lw=1.8, label="round-trip cost")
ax.set_xscale("log")
ax.set_xlim(1, 2e4)
ax.set_xlabel("Absolute move (bps, log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Most moves at every horizon are larger than the cost of trading them",
    subtitle="Absolute returns from the adjusted price, against the configured round trip",
)
show_with_alt(
    fig,
    "Three curves show the fraction of absolute moves at least as large as the value on a "
    "logarithmic basis-point axis, one for each of the one-day, five-day and twenty-one-day "
    "horizons. Each falls from one to zero, and the longer the horizon the further right "
    "the curve sits. A dotted vertical line marks the round-trip cost, and all three curves "
    "are still close to the top of the panel where it stands.",
)

# %% [markdown]
# ### B.4 How much of one day's return carries into the next
#
# Changing positions every day is only worth the trading it causes if something about a stock's
# recent behaviour says something about the day ahead. The cheapest version of that question is
# whether a stock's own return predicts its next one.
#
# The measurement is an **autocorrelation**: the correlation between a stock's return on one day and
# its return some number of days later. Plotted against that number of days, it shows how much of
# the series its own past accounts for, and how quickly that fades.
#
# It is computed inside each stock and then averaged across stocks. Stacking thousands of stocks
# into one long series and correlating that returns a number too, and the number is wrong: at every
# point where one stock's history ends and the next begins, it correlates two unrelated companies. A
# stock contributes a curve only when it has a year of sessions behind it, since a correlation from
# a handful of observations is mostly noise. The shaded region shows how much the result varies from
# stock to stock, and the band around zero shows how large a correlation could plausibly be if a
# stock's returns carried no information about their own past at all.

# %%
acf = panel_acf(
    returns,
    entity_col="symbol",
    value_col=f"h{PRIMARY_HORIZON}",
    max_lags=max(HORIZONS),
    min_obs=252,
).filter(pl.col("lag") > 0)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.axhspan(
    -acf["band"][0],
    acf["band"][0],
    color=COLORS["copper"],
    alpha=0.3,
    zorder=0,
    label="range expected from no information",
)
ax.fill_between(
    acf["lag"],
    acf["acf_p10"],
    acf["acf_p90"],
    color=COLORS["blue"],
    alpha=0.15,
    label="10th to 90th percentile across stocks",
)
ax.bar(acf["lag"], acf["acf"], color=COLORS["blue"], width=0.6, zorder=2, label="average")
ax.set_xlabel("Days between the two returns")
ax.set_ylabel("Autocorrelation of the daily return")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
add_message_title(
    ax,
    "Only the one-day lag clears the band, and it points down",
    subtitle="Averaged within each stock, over the range expected from no information",
)
show_with_alt(
    fig,
    "Bars give the average autocorrelation of the daily return at each lag from one to "
    "twenty-one days. A shaded horizontal band spans the range expected from no "
    "information, and a second, wider shaded region traces the tenth to ninetieth "
    "percentile across stocks. Only the bar at lag one reaches below the no-information "
    "band, and it points downwards; every other bar is short and sits inside it.",
)

# %%
beyond_one = acf.filter(pl.col("lag") > 1)["acf"].abs().max()
print(
    f"Lag one average {acf['acf'][0]:+.4f} against a band of {acf['band'][0]:.4f}, over a "
    f"per-stock 10th-90th spread of {acf['acf_p90'][0] - acf['acf_p10'][0]:.4f}\n"
    f"Largest absolute average beyond lag one {beyond_one:.4f}, measured on "
    f"{acf['n_entities'][0]:,} stocks"
)

# %% [markdown]
# One day back is the only lag whose average clears the band, and it points down: a weak tendency
# for a day's move to be partly given back the next day. It is small next to the spread across
# stocks behind it, so it is not something a book can be built on by itself. Every longer lag sits
# inside the band. Whatever the sort ends up reading therefore has to come from the cross-section -
# from how a stock compares to the others on the same day - rather than from its own last return.
#
# ### B.5 Move size against cost
#
# Two numbers summarise what B.3 drew. The first is the median absolute move over one rebalancing
# interval divided by the declared round trip, which says how much larger a typical move is than a
# typical cost. The second is the share of moves larger than that round trip, which is where the
# exceedance curve crosses its cost line.
#
# Neither says the strategy earns anything. Both count a move down exactly as they count a move up,
# and both legs of the round trip are paid whichever way the move went. What they rule out is the
# case where the design fails immediately, because a typical move is smaller than the cost of
# capturing it.

# %%
move_bps = moves[f"h{PRIMARY_HORIZON}"].abs().drop_nulls() * 1e4
per_share = cost["per_share_bps"]
print(
    f"Declared proportional round trip {COST_BPS:.0f} bps\n"
    f"Median absolute {PRIMARY_HORIZON}-day move {move_bps.median():.0f} bps, "
    f"{move_bps.median() / COST_BPS:.1f}x that round trip, share larger than it "
    f"{(move_bps > COST_BPS).mean():.3f}\n"
    f"Per-share round trip over the same universe {per_share.min():.2f} to {per_share.max():.0f} "
    f"bps, median {per_share.median():.1f} bps"
)

# %% [markdown] tags=["results"]
# The median absolute one-day move is 116 bps against a declared round trip of 25 bps, a ratio of
# 4.6 times, and 0.867 of one-day moves are larger than it. Charged per share instead, the same
# round trip runs from 0.01 bps on the highest-priced stock in the eligible universe to 112 bps on
# the lowest, with a median of 23.8 bps.

# %% [markdown]
# ## C. Design decisions
#
# The sections above are evidence. This section is where that evidence meets the choices recorded in
# `setup.yaml`, and says what each one rests on.
#
# ### C.1 How often to rebalance
#
# `setup.yaml::decision.cadence` sorts the stocks at the close and trades at the next open.
# Section B.3 supports trading that often: moves over one day are several times larger than the
# round trip, so cost is not what would force a slower schedule. Section B.4 is why the sort is also
# labelled at five and twenty-one days. A stock's own last return accounts for almost none of its
# next one, so there is no fast-fading tendency that a daily schedule exists to catch, and a longer
# holding period may well suit whatever the sort ends up reading better than a daily one.
#
# ### C.2 What would send this design back
#
# `setup.yaml::kill_conditions` declares four thresholds, and each is measured where its evidence
# exists rather than here. Two are about whether the sort works at all: whether it has any
# cross-sectional relationship to the returns that follow it, and whether the strategy's return per
# unit of risk stays above its floor once the borrow fee on the short leg is charged. Two are about
# whether it works where it can be traded: whether the return per unit of risk is large enough
# relative to what trading costs, and whether the result is concentrated in the least liquid fifth
# of the universe, which would mean the strategy is being paid for taking positions it could not
# have taken at size.
#
# The one this notebook could have produced is a cost failure, if a typical move had been smaller
# than the round trip needed to capture it. Section B.5 is that measurement, and Chapter 18 repeats
# it against the trades a backtest actually places rather than against raw moves.
#
# ### C.3 What the strategy does with the ordering
#
# `setup.yaml::mapping.class` sorts the eligible stocks into deciles and holds the top group against
# the bottom one. Holding both ends uses the whole ordering: the bottom of a sort carries as much
# information as the top, and a long-only version would discard half of what the sort says.
#
# Each stock held gets an equal share of the money inside its group. A weighting optimised for risk
# would fold an estimate of how the stocks move together into the result, and the ordering's own
# contribution could no longer be separated from that estimate's. Chapter 17 compares the
# alternatives with the ordering held fixed.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 How much an evaluation has to spend
#
# A panel of this size looks like a large sample, but a strategy that changes its positions once a
# day does not get to treat every row as an independent opportunity. What it spends is decision
# dates. A wider cross-section on one date buys precision in what that date says; it does not buy
# another date.

# %%
eligible_per_date = breadth["n_eligible"]
print(
    f"Sessions {len(calendar):,} of {len(dates):,} dates in the archive, of which the screen can "
    f"decide on {len(breadth):,}\n"
    f"Eligible stocks per date: {eligible_per_date.mean():.0f} on average, "
    f"{eligible_per_date.min():,} at the fewest, {eligible_per_date.max():,} at the widest, "
    f"below the {BREADTH_FLOOR} positions to fill on {(eligible_per_date < BREADTH_FLOOR).sum()} "
    f"of them"
)

# %% [markdown]
# ### D.2 The folds
#
# A model is fitted on one stretch of history and evaluated on the stretch that follows it, then the
# pair moves forward and the process repeats. Each fit-then-evaluate pair is a **fold**, and
# evaluating this way is called **walk-forward**, because the split always runs in the direction
# time does.
#
# One detail decides whether the evaluation is honest. The return being predicted lands one day
# ahead, so a training row dated on the last day of its block is labelled with a price from after
# the block ends. Validating on the day immediately after training would score the model on data it
# had partly seen already. The fix is to leave a gap between the two, at least as wide as the
# horizon, and that gap is called **purging**. Its width comes from `labels.buffer` in `setup.yaml`.
# The two longer variants declare wider gaps of their own in `labels.variant_buffers`, and every
# stage that generates folds resolves the gap for the label it is about to train on, so the timeline
# below is the primary label's design rather than one design shared by all three.
#
# The splitter is given the whole archive, holdout included, and applies the holdout boundary itself
# from `evaluation.holdout_start`, which is what every later stage does too. It is handed dates and
# no prices, so nothing the holdout contains reaches a number computed above.
#
# `generate_cv_splits` numbers folds from zero backwards from the most recent, so fold 0 is the last
# one before the holdout and the highest number is the earliest. The figure draws them earliest-first
# and labels each with that number, which is why the labels count down; every later stage prints
# the same ones. The two assertions below check what the figure cannot show at this scale: that the
# number of folds is the number `setup.yaml` declares, and that no validation window reaches into
# the holdout. The figure then draws the boundaries the splitter returned rather than recomputing
# them, so the picture and the folds cannot disagree.

# %%
splits = generate_cv_splits(
    panel.select("timestamp"),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)
last_val = max(split["val_end"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert str(last_val.date()) < HOLDOUT_START, "a fold reaches into the holdout"
print(
    f"{len(splits)} folds | training {SETUP['evaluation']['train_size']} and validation "
    f"{SETUP['evaluation']['val_size']} each, purged by labels.buffer {LABEL_BUFFER}; the variants "
    f"declare {', '.join(f'{k} at {v}' for k, v in VARIANT_BUFFERS.items())}\n"
    f"Last validation ends {last_val.date()}, the holdout opens {HOLDOUT_START}"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
ax.set_xlabel("Date")
add_message_title(
    ax,
    "Folds roll forward and stop short of the holdout",
    subtitle=f"Boundaries as generate_cv_splits returned them; the {LABEL_BUFFER} purge is one "
    "session and too narrow to see",
)
show_with_alt(
    fig,
    "One horizontal row per walk-forward fold, the highest-numbered at the top and fold "
    "zero at the bottom. Each row is a long dark training bar followed immediately by a "
    "short lighter validation bar. Reading down the rows, both bars shift later in time, "
    "so the folds roll forward across the sample. A shaded holdout block stands at the "
    "right-hand edge, and no fold's validation bar reaches it.",
)

# %% [markdown]
# ## E. What this notebook hands on
#
# Nothing. The screen in Section B.2 is a rule over the panel rather than a table, so each later
# notebook applies it to the rows it is working with rather than reading a list from here.

# %% [markdown]
# ## F. What the evidence says about each setting
#
# One row per setting: the evidence behind it, and the condition under which a reader working on
# their own data would choose differently.
#
# | Setting | Evidence | Choose differently when |
# |---|---|---|
# | `universe.n_assets` | B.1 the archive against the declared count, B.2 stocks passing the screen on each date | fewer stocks qualify than both ends of the sort need to fill |
# | `costs.model` | B.3, the per-share charge converted to the same unit and drawn across the universe | quotes per stock and era replace the assumption, or prices stop spanning orders of magnitude |
# | `decision.cadence`, `labels.primary` | B.3 move sizes against cost, B.4 how much of a day's return carries, B.5 the ratio | moves stop covering the round trip, or a longer horizon earns more than the trading it saves |
# | `evaluation.n_splits` | D.1 decision dates available, D.2 fold boundaries | the folds no longer fit ahead of the holdout |

# %%
print(
    f"universe.n_assets {SETUP['universe']['n_assets']:,}, eligible per decision date "
    f"{eligible_per_date.min():,} to {eligible_per_date.max():,} against {BREADTH_FLOOR} positions "
    f"to fill\n"
    f"costs.model {SETUP['costs']['model']}, round trip {COST_BPS:.0f} bps | decision.cadence "
    f"{SETUP['decision']['cadence']} | labels.primary {PRIMARY_LABEL}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {last_val.date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# The number of stocks passing the screen runs from 290 at its narrowest to 2,693 at its widest,
# against the 100 positions the sort has to fill across both sides, and no date the screen can
# decide on falls below that. Sixteen folds are generated from the declared design, the last
# validation window ending 2015-12-30, and the holdout is untouched.

# %% [markdown]
# ## Key takeaways
#
# 1. **Decide what a strategy was allowed to hold from information that existed before the date the
#    decision applies to.** A screen applied to the whole sample at once admits exactly the stocks
#    that turned out to stay liquid, and a backtest run on that universe is measuring a choice
#    nobody could have made at the time.
# 2. **Take returns from the adjusted series and the screen from the printed one.** An unadjusted
#    return records a two-for-one split as a fall of half, and an adjusted price for a date in the
#    past is a number computed from everything that happened after it.
# 3. **Count in sessions, not in rows.** A stock's rows are the sessions it traded, so a fixed
#    number of rows back is a fixed number of sessions back only where the stock traded every one of
#    them - which a halt, a suspension or a late listing breaks.
# 4. **Check a per-share cost assumption against the price distribution before adopting it.** A
#    fixed number of cents is a different fraction of every stock, and across a universe spanning
#    two orders of magnitude in price it is wrong at the cheap end by the same factor.
# 5. **Compute a panel autocorrelation inside each entity, then average.** Stacking entities into
#    one series measures the joins between them.
#
# ### Known limitations
#
# - The archive ends in the first quarter of 2018, so the holdout is the most recent history
#   available and there is nothing after it to check against.
# - Cost is one proportional assumption for every stock and every date. Spreads were far wider
#   before decimalization in 2001 than after it, which `setup.yaml::costs.era_dependent` records
#   without applying.
# - The borrow fee on the short leg is a flat annual assumption, and in practice it is neither flat
#   across stocks nor stable through time - it rises exactly on the names most people want to sell.
# - The price and turnover thresholds are fixed dollar amounts that are not adjusted for inflation,
#   so the screen is stricter at the start of the sample than at the end.
#
# **Next**: labels at the declared horizons, built on this development period.
