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
# # S&P 500 Options: Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. Once a week it sells an option
# combination on several hundred S&P 500 constituents, offsets the position's exposure to the share
# price at each close, and holds until the options expire about a month later. That file says which
# stocks it trades, on which day of the week it acts, what it assumes a trade costs, and how the
# history is divided between designing the strategy and testing it. This notebook checks each of
# those assumptions against the data and reports what it finds.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Pair a call and a put quoted at the same strike and expiration into a single position, and read
#   its price out of an option chain that carries one end-of-session quote per contract
# - Express what a trade costs in the unit the position earns in - a share of the option premium -
#   and see why the same cost measured against the share price orders the universe differently
# - Count how many stocks carry an option at the maturity a strategy targets on each date it
#   trades, and compare that count against the number of positions it has to fill
# - Read off one chart what fraction of price moves are larger than the cost of trading them
# - Measure how long the gap between implied and realized volatility persists, computing the
#   correlation inside each stock rather than across hundreds of stocks stacked into one series
# - Check that a walk-forward split fits the history available and leaves the test period unread,
#   when a position's outcome is not known until the options it holds expire
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. This notebook reads the daily straddle panel, the option chains it
# was built from, daily share prices, the daily implied-volatility summary and `config/setup.yaml`,
# and writes nothing.
#
# ## Prerequisites
#
# None beyond what the sections below define. A reader who has not traded an option will find the
# instrument, the premium, implied volatility and hedging explained where they are first used.

# %%
"""S&P 500 Options Case Study - Feasibility Analysis."""

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from IPython.display import display
from ml4t.diagnostic.splitters.calendar import TradingCalendar

from case_studies.sp500_options._straddle_moves import straddle_premium_moves
from case_studies.sp500_options._underlying_returns import reconcile_underlying_log_returns
from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import (
    load_sp500_daily_bars,
    load_sp500_options_straddles,
    load_sp500_options_straddles_raw,
    load_sp500_options_surface,
)
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_options"
START_DATE = "2017-01-01"
END_DATE = "2021-12-31"

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads those
# values rather than repeating them, so the two can never disagree. Four groups of settings matter
# here, and each one decides something the sections below test.
#
# **How the history is divided.** The sample runs from the start of 2017 to the end of 2021. The
# final year is the *holdout*: a stretch of history that is not looked at while the strategy is
# being designed, so that when it is finally evaluated there, the result is not a rehearsal of
# choices already tuned on the same data. Everything computed in this notebook uses the earlier
# part, called the development period, and `holdout_start` is where the line falls.
#
# **What the strategy trades.** Not a fixed list of tickers. Any S&P 500 constituent quoting the
# option combination described in Section A, at the maturity `features.target_dte` names, is
# available on the date it quotes it. From those, the strategy ranks only the cheapest fifth to
# trade - `backtest.sweep.htm_cost_cascade.liquid_quantile` - and holds the leaders of that
# ranking, up to `top_k` of them. Filling that book therefore needs five times `top_k` stocks
# quoting on a trading date, and Section B.2 counts them.
#
# **What a trade is assumed to cost.** The gap between the price at which the position can be sold
# and the price at which it can be bought back, stated in `costs.components.option_spread` as a
# share of the premium rather than in the units an equity cost model uses. Because the position is
# held until the options expire, it is never bought back, so it crosses that gap once rather than
# twice, and `htm_cost_cascade.cost_fractions` sweeps how much of the one crossing a trader
# actually pays. Section B.3 measures the gap and Section B.5 compares it against the moves.
#
# **What is being predicted.** The return the position earns between the day it is opened and the
# day the options expire, which `labels.primary` names. Positions are opened weekly, so a new one
# starts before the previous has finished, and the settings below give both the weekly step and the
# holding period the design assumes. `labels.buffer` is the separation the walk-forward split has
# to leave for an outcome that is not known until expiration, and Section D reads it.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
PRIMARY_LABEL = SETUP["labels"]["primary"]
LABEL_BUFFER = SETUP["labels"]["buffer"]
BUFFER_SESSIONS = int(LABEL_BUFFER.rstrip("D"))
CASCADE = SETUP["backtest"]["sweep"]["htm_cost_cascade"]
TOP_K = max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
BREADTH_FLOOR = int(np.ceil(TOP_K / CASCADE["liquid_quantile"]))
# The cheapest rung assumes a trader crosses this share of the quoted half-spread, and holding to
# expiration crosses one leg rather than two, so it is that share of half the round trip.
ENTRY_COST_SHARE = min(CASCADE["cost_fractions"]) / 2
TARGET_DTE = SETUP["features"]["target_dte"]
VRP_WINDOW = SETUP["features"]["windows"]["vrp_reference"]
HORIZONS = sorted(
    {SETUP["labels"]["rebalance_step"][PRIMARY_LABEL], SETUP["decision"]["holding_period_days"]}
)
SESSIONS = TradingCalendar(SETUP["evaluation"]["calendar"]).trading_days_between(
    START_DATE, HOLDOUT_START
)

print(f"Sample: {START_DATE} to {END_DATE}")
print(f"  Development period, used everywhere below:  {START_DATE} to {HOLDOUT_START}")
print(f"  Holdout, not read by this notebook:         {HOLDOUT_START} to {HOLDOUT_END}")
print(f"Universe: every S&P 500 constituent quoting a {TARGET_DTE}-day option combination that day")
print(
    f"  Ranked inside the cheapest {CASCADE['liquid_quantile']:.0%} of a date, {TOP_K} positions "
    f"held, so at least {BREADTH_FLOOR} must be quoting for that book to be filled"
)
print(
    f"Assumed cost: {SETUP['costs']['components']['option_spread']['estimate_pct_of_premium'][0]} "
    f"to {SETUP['costs']['components']['option_spread']['estimate_pct_of_premium'][-1]} percent of "
    f"the premium per crossing, of which the cascade charges "
    f"{', '.join(f'{f:.0%}' for f in CASCADE['cost_fractions'])} in turn"
)
print(
    f"Prediction: {PRIMARY_LABEL}, the return from entry to expiration; positions opened every "
    f"{SETUP['labels']['rebalance_step'][PRIMARY_LABEL]} sessions and assumed held for "
    f"{SETUP['decision']['holding_period_days']}"
)
print(
    f"Walk-forward: {SETUP['evaluation']['n_splits']} folds on the {SETUP['evaluation']['calendar']}"
    f" calendar, with {BUFFER_SESSIONS} sessions of separation for an outcome known at expiration"
)

# %% [markdown]
# ## A. Orientation
#
# ### What a straddle is
#
# An option is a contract on a stock. A **call** gives its holder the right to buy the stock at a
# fixed price, the **strike**, up to a fixed date, the **expiration**; a **put** gives the right to
# sell it on the same terms. The holder pays for that right, and the amount paid is the
# **premium**. A **straddle** is a call and a put on the same stock, at the same strike and the same
# expiration, treated as one position: its price is the sum of the two premiums. **At the money**
# means the strike sits nearest the current share price, which is where a straddle carries the least
# opinion about direction.
#
# Whoever buys an at-the-money straddle profits if the stock ends far from the strike, up or down,
# and loses the premium if it ends near it. Whoever sells the straddle takes the opposite side: the
# premium is collected at the outset and paid back to the extent the stock moves. Selling is what
# this strategy does, on several hundred stocks at once.
#
# ### What the seller is paid for
#
# What a straddle is worth depends on how far the market expects the stock to travel before
# expiration, so the traded price implies a movement, expressed the way statisticians express one -
# as an annualized standard deviation of returns. That number is the **implied volatility**. What the stock then actually does, measured the same way from its daily
# returns, is the **realized volatility**. Across equity markets the first has on average sat above
# the second, and the gap between them is the **variance risk premium**. A straddle seller who is
# right about that gap keeps the difference; ranking stocks by how wide their own gap looks is what
# turns it into a cross-sectional strategy.
#
# Two things stand between the gap and the money. The first is direction. As the share price moves
# away from the strike, a straddle stops being an even bet and starts behaving like a position in
# the stock; **delta** measures that, as the change in the position's value per one-dollar change in
# the share price. Holding an offsetting amount of the stock cancels it, which is called **delta
# hedging**, and `setup.yaml::hedging_protocol` does it at each close. What the hedge leaves behind
# is the volatility gap rather than the share's direction. The second is cost, and it is Section B.3.
#
# ### Why the premium is the denominator
#
# A stock position's return is measured against the share price, and the cost of trading it is a few
# cents against tens or hundreds of dollars. A short straddle earns on the premium instead, which is
# a few percent of the share price, and the gap between the bid and the ask is quoted on that same
# premium. The same few cents is therefore a far heavier charge here than it is on the share. Every
# cost in this notebook is expressed as a fraction of the premium for that reason, and Section B.1
# shows both units side by side so the difference between them is visible before anything depends on
# it.
#
# ### The three questions this notebook asks
#
# 1. **Does the universe exist when the strategy trades?** Positions are opened once a week, and a
#    stock only qualifies on the weeks a listed expiration falls in the maturity window, so enough
#    stocks have to qualify on each of those dates to fill the book.
# 2. **Is a typical move in the premium worth more than it costs to trade?** Selling the straddle
#    crosses the gap between the bid and the ask, and holding to expiration decides whether that
#    happens once or twice.
# 3. **Is there enough history to evaluate this honestly?** Enough to split into training and
#    validation periods more than once, with the holdout left untouched, and with room for outcomes
#    that are not known until the options expire.

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 Load the data and look at the universe
#
# The loader returns the straddle panel: one row per stock and session on which a listed expiration
# falls inside the target maturity window, carrying the call and the put nearest the money at a
# common strike and expiration, their combined price, and the gap between the combined bid and the
# combined ask.
#
# Three properties are checked before anything is computed. The combined price is above zero,
# because every ratio below divides by it. The panel holds at most one straddle per stock and
# session, since a second one would double that stock's weight in every average taken across the
# panel. And the combined price and the combined spread are each the sum of the two legs, which is
# the arithmetic every cost figure in Section B.3 rests on.

# %%
straddles = load_sp500_options_straddles(start_date=START_DATE, end_date=END_DATE)
research = straddles.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date())
legs = research.select(
    (pl.col("instr_mid") - pl.col("call_mid") - pl.col("put_mid")).abs().max().alias("mid"),
    (
        pl.col("instr_spread")
        - (pl.col("call_ask") - pl.col("call_bid"))
        - (pl.col("put_ask") - pl.col("put_bid"))
    )
    .abs()
    .max()
    .alias("spread"),
)

assert research["instr_mid"].min() > 0, "a straddle mid is not a usable denominator"
assert not research.select(pl.struct("symbol", "timestamp").is_duplicated().any()).item(), (
    "a stock carries more than one straddle on a session"
)
assert legs["mid"][0] < 1e-9 and legs["spread"][0] < 1e-9, (
    "a straddle price or spread is not the sum of its two legs"
)

timeline = research.select("timestamp").unique().sort("timestamp").with_row_index("s")
print(
    f"{research['symbol'].n_unique()} stocks, {len(research):,} straddle-days over "
    f"{len(timeline):,} sessions, {timeline['timestamp'][0]} to {timeline['timestamp'][-1]}\n"
    f"maturity at entry {research['instr_dte'].min()} to {research['instr_dte'].max()} calendar "
    f"days, against the {TARGET_DTE} days features.target_dte asks for"
)

# %% [markdown]
# Six hundred tickers are a list, not a description. The way the options market groups stocks is by
# how much they are expected to move, because that is what the premium is a price for, so the table
# below sorts the universe into four equal groups by each stock's own median implied volatility and
# reports what a straddle on it costs.
#
# Two columns say the same thing in different units, and the contrast between them is the reason
# this case study exists. The round trip as a share of the premium is what the position gives up out
# of what it earns on; the round trip in **basis points** of the share price - one basis point is
# one hundredth of one percent - is what an equity cost model would charge. Read down the first, the
# four groups are almost indistinguishable. Read down the second, the most volatile group looks
# nearly twice as expensive as the calmest. The premium is what grows with volatility, and the
# spread grows with it in step, so a cost model quoted against the share price ranks the universe by
# how volatile its members are rather than by how expensive they are to trade.

# %%
by_symbol = research.group_by("symbol").agg(
    pl.col("iv_atm").median().alias("implied_vol"),
    (pl.col("instr_pct_of_S") * 100).median().alias("premium_pct_of_share"),
    (pl.col("instr_rel_spread") * 100).median().alias("round_trip_pct_of_premium"),
    (pl.col("instr_spread") / pl.col("underlying_price") * 1e4).median().alias("round_trip_bps"),
    pl.len().alias("straddle_days"),
)
groups = (
    by_symbol.with_columns(
        ((pl.col("implied_vol").rank("ordinal") - 1) * 4 // pl.len()).alias("group")
    )
    .group_by("group")
    .agg(
        pl.len().alias("stocks"),
        (pl.col("implied_vol") * 100).min().round(0).alias("implied_vol_from_pct"),
        (pl.col("implied_vol") * 100).max().round(0).alias("implied_vol_to_pct"),
        pl.col("premium_pct_of_share").median().round(1),
        pl.col("round_trip_pct_of_premium").median().round(1),
        pl.col("round_trip_bps").median().round(0),
        pl.col("straddle_days").sum(),
    )
    .sort("group")
    .drop("group")
)
with pl.Config(tbl_rows=groups.height, tbl_cols=groups.width):
    display(groups)

# %% [markdown]
# ### B.2 How many stocks carry a straddle when the strategy trades
#
# `setup.yaml::decision.entry_cadence` opens positions at the last session of each week, so that is
# where the universe has to exist. A stock qualifies on that date only if one of its listed
# expirations happens to fall in the maturity window, and which stocks those are changes from week
# to week, and the reason is how options are listed rather than anything the strategy does.
#
# Every optionable US stock has **monthly** expirations, one in the third week of each month - the
# third Friday, or the session before it when that Friday is a market holiday. Liquid names
# additionally have **weekly** expirations, on most of the other Fridays. So in a week whose
# maturity window reaches a monthly expiration, the whole universe qualifies; in the other weeks
# only the names carrying weeklies do. The figure separates the count on that basis, and draws the
# number of stocks the book needs against it. That floor is five times the positions held, because
# the ranking is run inside the cheapest fifth of the date rather than over everything quoting.
#
# The assertion states what the split assumes: expirations landing in the third week arrive one per
# month, so an expiration in that window is the monthly one and every other is a weekly.

# %%
decisions = research.group_by(pl.col("timestamp").dt.truncate("1w")).agg(
    pl.col("timestamp").max().alias("decision_date")
)
monthly = ((pl.col("expiration").dt.day() - 1) // 7 + 1) == 3
entries = research.join(decisions, left_on="timestamp", right_on="decision_date").with_columns(
    monthly.alias("monthly_expiry")
)
third_week = research.filter(monthly).select("expiration").unique()
assert third_week.height == third_week.select(pl.col("expiration").dt.truncate("1mo")).n_unique(), (
    "more than one expiration falls in the third week of a month"
)
breadth = (
    entries.group_by("timestamp")
    .agg(
        pl.col("symbol").n_unique().alias("n_symbols"),
        pl.col("symbol").filter(pl.col("monthly_expiry")).n_unique().alias("n_monthly"),
        pl.col("symbol").filter(~pl.col("monthly_expiry")).n_unique().alias("n_weekly"),
    )
    .sort("timestamp")
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.stackplot(
    breadth["timestamp"],
    breadth["n_weekly"],
    breadth["n_monthly"],
    colors=[COLORS["blue"], COLORS["amber"]],
    labels=["expiration listed weekly", "expiration listed monthly"],
)
ax.axhline(
    BREADTH_FLOOR,
    color=COLORS["copper"],
    ls="--",
    lw=1.5,
    label="straddles the cheapest fifth needs to fill the book",
)
ax.set_ylim(0, None)
ax.set_ylabel("Stocks quoting a straddle")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
add_message_title(
    ax,
    "The universe swells whenever a monthly expiration falls in the window",
    subtitle="Stocks with a target-maturity straddle at each weekly decision date, by listing",
)
show_with_alt(
    fig,
    "Stacked area chart of the count of stocks quoting a straddle at each weekly decision "
    "date from 2017 to 2021, split into weekly-listed and monthly-listed expirations. The "
    "weekly-listed base sits between roughly 150 and 250 names throughout, and the monthly "
    "band spikes to between 400 and 470 in the weeks a monthly expiration falls inside the "
    "target-maturity window, giving the series a regular sawtooth. A dashed line at 100 marks "
    "the straddles the cheapest fifth needs to fill the book; the total falls below it on 4 of "
    "the 209 decision dates, all of them consecutive weeks in the March-April 2020 trough, "
    "which reads as a single notch in the series.",
)

# %% [markdown]
# ### B.3 What a round trip costs, and what a move is worth
#
# Selling a straddle means selling both legs, and each is sold at the bid while the quoted mid sits
# above it. Buying the position back later pays the ask on both legs. A trade that opens and closes
# therefore gives up the full gap between the combined bid and the combined ask, and dividing that
# gap by the combined mid states it as the share of the premium the position surrenders before the
# stock has done anything. That is what `costs.components.option_spread` assumes, and what the curve
# below measures, one point per stock, sorted.
#
# It is measured at the decision dates Section B.2 counted rather than over every session, because
# what a position gives up is the spread quoted around the time it is opened. Those dates are the
# Fridays the ranking is formed on, and the fill itself lands one session later, at the following
# close, for the reason Section C.1 gives. Both this cost and the move in B.4 are read at that same
# Friday quote, which is what lets B.5 compare them; what neither of them is, is the price of any
# particular fill. The line marking the cheapest
# fifth is the threshold the strategy trades inside; it is drawn here over the whole development
# period, while the strategy applies it separately on each date, which is what the second figure
# looks at.
#
# `costs.components.option_spread` states what one crossing is assumed to cost, as a range in
# percent of the premium, and half the round trip is one crossing. The assertion below holds that
# range against the cheapest fifth, which is the part of the universe the strategy trades. Where the
# rest of the universe sits against it is a number, and Section F reports it.

# %%
cost = (
    entries.group_by("symbol")
    .agg(
        pl.col("instr_rel_spread").median().alias("round_trip"),
        (pl.col("instr_spread") / pl.col("underlying_price") * 1e4).median().alias("spread_bps"),
    )
    .sort("round_trip")
)
COST_SHARE = float(cost["round_trip"].median())
LIQUID_CUT = float(cost["round_trip"].quantile(CASCADE["liquid_quantile"]))
ASSUMED_PCT = SETUP["costs"]["components"]["option_spread"]["estimate_pct_of_premium"]
assert min(ASSUMED_PCT) <= LIQUID_CUT * 50 <= max(ASSUMED_PCT), (
    "the cheapest fifth crosses at a cost outside costs.components.option_spread"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(range(len(cost)), cost["round_trip"] * 100, color=COLORS["blue"], lw=1.6)
ax.axhline(COST_SHARE * 100, color=COLORS["copper"], ls="--", lw=1.5, label="universe median")
ax.axhline(
    LIQUID_CUT * 100, color=COLORS["amber"], ls=":", lw=1.8, label="cheapest fifth of stocks"
)
ax.set_xlabel("Stocks, ordered by their own round-trip cost")
ax.set_ylabel("Round trip (% of straddle premium)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
add_message_title(
    ax,
    "Where a stock sits in this ordering decides what its move has to clear",
    subtitle="Median gap between the combined bid and ask over the combined mid, per stock",
)
show_with_alt(
    fig,
    "Line chart of each of about 600 stocks' median round-trip cost, as a percentage of the "
    "straddle premium, with the stocks sorted along the horizontal axis by that cost. The "
    "curve rises from under 4 percent at the cheapest names, passes 10 percent around the "
    "200th stock, and steepens after the 500th to reach nearly 27 percent at the most "
    "expensive. A dashed line marks the universe median near 12 percent and a dotted line the "
    "cheapest-fifth threshold near 9 percent, so the ordering is shallow across the middle of "
    "the universe and steep only at its expensive tail.",
)

# %% [markdown]
# A cost measured once over five years hides whether it is the same cost every week. The figure
# below recomputes it at each decision date: the median across the stocks quoting that day, and the
# threshold the cheapest fifth of that day sits under, which is the rule the strategy actually
# applies.
#
# Two patterns are worth separating. The week-to-week sawtooth is the listing cycle of Section B.2
# arriving in a different quantity: on the weeks a monthly expiration is in reach, the median is
# taken over roughly twice as many stocks, and the ones that only appear then are not the same
# stocks. The slower movement is the spread itself, and its largest excursion falls in the weeks
# Section B.2's count falls to its lowest - so the position gets more expensive to open exactly
# where there is least to choose from.

# %%
by_date = (
    entries.group_by("timestamp")
    .agg(
        pl.col("instr_rel_spread").median().alias("median_round_trip"),
        pl.col("instr_rel_spread").quantile(CASCADE["liquid_quantile"]).alias("liquid_cut"),
    )
    .sort("timestamp")
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(
    by_date["timestamp"],
    by_date["median_round_trip"] * 100,
    color=COLORS["blue"],
    lw=1.2,
    label="median stock that day",
)
ax.plot(
    by_date["timestamp"],
    by_date["liquid_cut"] * 100,
    color=COLORS["amber"],
    lw=1.2,
    label="cheapest fifth that day",
)
ax.set_ylim(0, None)
ax.set_ylabel("Round trip (% of straddle premium)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(frameon=False, fontsize=8, loc="upper left")
add_message_title(
    ax,
    "Spreads widen and the universe thins in the same weeks",
    subtitle="Cost of crossing, recomputed across the stocks quoting at each decision date",
)
show_with_alt(
    fig,
    "Two lines tracking the round trip as a percentage of straddle premium at each weekly "
    "decision date from 2017 to 2021: the median stock that day, and the cheapest fifth that "
    "day. The cheapest-fifth line runs three to five points below the median line throughout "
    "and the two move together. The median runs mostly between 9 and 16 percent and the "
    "cheapest fifth between 4 and 11. Both jump sharply in March 2020, the median peaking "
    "above 22 percent, and both settle after it at a level higher than they held before.",
)

# %% [markdown]
# Against that cost sits the move in the premium itself. Each straddle the panel selected on a
# decision date is followed forward through its own strike and expiration rather than through the
# panel, because the panel re-picks whichever contract is nearest the money each session and
# differencing it would measure the switch of contract instead of a change in price. The offset is
# counted in sessions the chains quoted on, so a day without a two-sided quote yields no value
# rather than a mistimed one. `straddle_premium_moves` carries all three constructions.
#
# Each move is then divided by the round trip that same entry would have crossed, which puts
# break-even at one for every stock whatever its own spread happens to be. The two horizons are the
# weekly step at which positions are opened and the holding period the design assumes; both sit
# inside the life of the contract, which expires within about four weeks of entry.
#
# The chart is an **exceedance curve**, and it reads from the right: for each multiple on the
# horizontal axis, the curve gives the fraction of moves at least that large. Two cost lines are
# drawn rather than one. The first is the full round trip, which is what a position that is bought
# back pays. The second is what this strategy pays instead: holding to expiration means the position
# is never bought back, so it crosses one leg rather than two, and the cheapest rung of
# `htm_cost_cascade.cost_fractions` assumes a trader gives up only part of even that one crossing.
# The distance between the two lines is the whole of the design's cost argument.
#
# One thing this chart is not. It is the distribution of how far the premium moves, ignoring
# direction. It is not the return the strategy would earn: nothing here is signed, and a seller
# keeps the premium only when the move is small. Whether the ranking can pick which straddles move
# least is the question Chapter 7 onwards asks.

# %%
KEYS = ["symbol", "strike", "expiration", "timestamp"]
moves = straddle_premium_moves(
    load_sp500_options_straddles_raw(start_date=START_DATE, end_date=HOLDOUT_START, lazy=True),
    entries,
    horizons=HORIZONS,
).join(entries.select(*KEYS, "instr_rel_spread"), on=KEYS)
moves = moves.with_columns(
    (pl.col(f"h{h}") / pl.col("instr_rel_spread")).alias(f"h{h}") for h in HORIZONS
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for h, color in zip(HORIZONS, (COLORS["blue"], COLORS["amber"]), strict=True):
    multiple, fraction = exceedance_curve(moves[f"h{h}"].drop_nulls().to_numpy())
    ax.plot(multiple, fraction, color=color, lw=1.6, label=f"{h}-session move")
ax.axvline(1, color=COLORS["copper"], ls="--", lw=1.5, label="both legs crossed")
ax.axvline(ENTRY_COST_SHARE, color=COLORS["copper"], ls=":", lw=1.8, label="one leg, cheapest rung")
ax.set_xscale("log")
ax.set_xlim(0.03, 100)
ax.set_xlabel("Absolute premium move as a multiple of the entry's own round trip (log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Which cost line the position pays decides whether its move clears it",
    subtitle="Exceedance of absolute premium moves at entry, over the spread that entry crosses",
)
show_with_alt(
    fig,
    "Two exceedance curves on a logarithmic horizontal axis, giving the fraction of absolute "
    "premium moves at least as large as a given multiple of the entry's own round trip, for "
    "the 5-session and the 10-session move. Both start near 1.0 at the left and fall to "
    "almost zero beyond ten times the round trip, with the 10-session curve above the "
    "5-session curve everywhere. A dashed vertical line at one times the round trip marks "
    "both legs crossed, where the 5-session curve reads about 0.5 and the 10-session about "
    "0.7; a dotted vertical line near one tenth marks the one-leg cheapest rung, where the "
    "5-session curve still reads 0.948 and the 10-session sits just above it.",
)

# %% [markdown]
# ### B.4 How long the volatility premium lasts
#
# Re-ranking every week is worth the trading it causes only if what the data says on one decision
# date still says something on the next. The quantity being ranked is the gap Section A described:
# each stock's at-the-money implied volatility less the volatility it has just realized, the latter
# from the standard deviation of its daily log returns over the window
# `features.windows.vrp_reference` declares, annualized. Returns are computed inside each stable
# security identity, so a ticker changing hands between companies does not enter as a return, and
# implied volatility is read from the daily surface summary, which records the contract nearest the
# money in the maturity bucket on every session.
#
# The measurement is an **autocorrelation**: the correlation between a stock's gap on one session
# and its gap some number of sessions later. Plotted against that number, it shows how much of the
# series its own past accounts for and how quickly that fades.
#
# It is computed inside each stock and then averaged. Stacking hundreds of stocks into one long
# series and correlating that returns a number too, and the number is wrong: wherever one stock's
# history ends and the next begins, it correlates two unrelated companies. A lag is also a row
# offset rather than a date difference, so only stocks quoted on every session between their first
# and last observation can contribute one, and the filter below keeps those with at least half a
# development period of them. The shaded region shows how much the result varies from stock to
# stock, and the band around zero shows how large a correlation could plausibly be if the gap
# carried no information about its own past at all.

# %%
bars = load_sp500_daily_bars(start_date=START_DATE, end_date=HOLDOUT_START)
rolling = pl.col("clean_log_return").rolling_std(VRP_WINDOW, min_samples=VRP_WINDOW)
realized = (
    reconcile_underlying_log_returns(bars)
    .select(
        "timestamp",
        "symbol",
        (
            rolling.over(["symbol", "sec_id"]) * np.sqrt(SETUP["evaluation"]["periods_per_year"])
        ).alias("realized_vol"),
    )
    .drop_nulls()
)
surface = load_sp500_options_surface(start_date=START_DATE, end_date=HOLDOUT_START)
span = pl.col("s").max().over("symbol") - pl.col("s").min().over("symbol") + 1
premium = (
    surface.select("timestamp", "symbol", "iv_30_atm")
    .drop_nulls()
    .join(realized, on=["timestamp", "symbol"], how="inner")
    .join(timeline, on="timestamp")
    .with_columns((pl.col("iv_30_atm") - pl.col("realized_vol")).alias("vol_gap"))
    .filter((pl.len().over("symbol") == span) & (pl.len().over("symbol") >= SESSIONS // 2))
    .sort(["symbol", "timestamp"])
)
acf = panel_acf(
    premium,
    entity_col="symbol",
    value_col="vol_gap",
    max_lags=max(HORIZONS) * 2,
    min_obs=SESSIONS // 2,
).filter(pl.col("lag") > 0)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.axhspan(
    -acf["band"][0],
    acf["band"][0],
    color=COLORS["copper"],
    alpha=0.3,
    zorder=1,
    label="range expected from no information",
)
ax.fill_between(
    acf["lag"],
    acf["acf_p10"],
    acf["acf_p90"],
    color=COLORS["blue"],
    alpha=0.15,
    zorder=2,
    label="10th to 90th percentile across stocks",
)
ax.bar(
    acf["lag"], acf["acf"], color=COLORS["blue"], width=0.6, zorder=3, label="average across stocks"
)
ax.set_xticks(range(1, acf["lag"].max() + 1, 2))
ax.set_xlabel("Sessions between the two observations")
ax.set_ylabel("Correlation with its own past")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
add_message_title(
    ax,
    "A week on, the volatility gap still resembles what it was",
    subtitle="Mean within-stock autocorrelation of implied less realized volatility",
)
show_with_alt(
    fig,
    "Bar chart of the mean within-stock autocorrelation of implied less realized volatility "
    "against the number of sessions between the two observations, from 1 to 20, with a shaded "
    "band showing the 10th to 90th percentile across stocks and a horizontal band showing the "
    "range expected from no information. The bars decay almost linearly, from about 0.92 at "
    "one session to 0.74 at five, 0.47 at ten and 0.01 at twenty. They stay clear of the "
    "no-information band, which spans roughly plus or minus 0.065, out to session 18, so the "
    "gap a week later still resembles the gap observed at entry.",
)

# %% [markdown]
# ### B.5 Move size against cost
#
# Three numbers summarise what B.3 and B.4 drew. The first is the median move over one weekly step,
# divided by the round trip quoted at that same decision session, which says how much larger a
# typical move is than a typical cost. The second pair is the share of moves larger than each of
# the two cost lines, which is where the exceedance curve crosses them. The third is how much of
# the volatility gap is still there a week later.
#
# None of them says the strategy earns anything. A move counts the same whether the premium rises or
# falls, and a seller of the straddle is hurt by one and helped by the other. What they rule out is
# the case where the design fails immediately, because a typical move is smaller than the cost of
# taking the position.

# %%
short = moves[f"h{HORIZONS[0]}"].drop_nulls()
print(
    f"Round trip {cost['round_trip'].min():.3f} to {cost['round_trip'].max():.3f} of premium, "
    f"median {COST_SHARE:.4f}; the same charge on the share price is "
    f"{cost['spread_bps'].median():.1f} bps\n"
    f"Median {HORIZONS[0]}-session move {short.median():.2f}x the round trip; larger than both legs "
    f"on {(short > 1).mean():.3f} of entries, larger than one leg at the cheapest rung on "
    f"{(short > ENTRY_COST_SHARE).mean():.3f}\n"
    f"Volatility gap averages {premium['vol_gap'].mean():.4f} across "
    f"{premium['symbol'].n_unique()} stocks, is positive on {(premium['vol_gap'] > 0).mean():.3f} "
    f"of sessions, and correlates {acf.filter(pl.col('lag') == HORIZONS[0])['acf'][0]:.3f} with "
    f"itself a week later"
)

# %% [markdown] tags=["results"]
# The median stock gives up 0.1184 of the straddle premium to cross both legs, which is the same
# charge as 70.1 bps of the share price. The median entry's five-session move matches the round trip
# it would cross, at 1.00x, and 0.497 of entries move more than that; measured against the one leg
# the cheapest rung of the cascade charges, 0.948 do. The volatility gap averages -0.0008 volatility
# points across 172 stocks, is positive on 0.596 of sessions, and correlates 0.735 with itself a
# week later.

# %% [markdown]
# ## C. Design decisions
#
# The sections above are evidence. This section is where that evidence meets the choices recorded in
# `setup.yaml`, and says what each one rests on.
#
# ### C.1 How often to trade, and at what price
#
# `setup.yaml::decision.entry_cadence` ranks the stocks at the Friday close, and
# `decision.execution_delay` prices the resulting trade at the close of the next session. That
# convention is set by the data rather than chosen: this option chain records one end-of-session
# quote per contract per day and carries no opening, high or low price, so a close is the only price
# a fill can be struck at, and the first one a decision taken at Friday's close can reach is
# Monday's. `hedge_cadence` then re-hedges the position's direction at each close until it expires.
#
# Section B.4 supports acting weekly: inside a stock the volatility gap still resembles itself a
# week later, so the ranking is not being rebuilt out of noise between one decision and the next.
# Whether the *ordering across* stocks is as stable is a different question, and Chapter 7 is where
# it is measured. Section B.2 constrains the cadence from the other side, because the set of stocks
# available to rank turns over with the expiration calendar rather than with anything the strategy
# controls.
#
# ### C.2 What would send this design back
#
# A feasibility study is only useful if some result would have stopped it. Three would, and each is
# measured where its evidence exists rather than here.
#
# The one this notebook could have produced is a cost failure: if a typical move in the premium were
# smaller than what the position gives up to take it, no ranking would repair that. Section B.5 is
# that measurement, and Chapter 18 repeats it against the trades a backtest actually places, sweeping
# the cascade rather than assuming one rung of it.
#
# The other two are outcomes of the strategy rather than properties of the data. Chapter 7 asks
# whether the ranking has any relationship at all to the returns that follow it. Chapter 19 asks the
# question specific to selling options: a short straddle loses more the further the stock travels,
# without limit, so a run of large moves can cost more than every premium collected around it, and
# the risk overlay is where that is measured against the premium the book takes in.
#
# ### C.3 What the strategy does with the ranking
#
# `setup.yaml::mapping.class` sells the straddles at the top of the ranking and takes no position in
# the rest. Section B.3 is why the ranking is run inside the cheapest fifth of each date rather than
# over everything quoting: the spread on the widest names is a large enough share of the premium
# that a position in them is trading the spread rather than the volatility gap. `sizing` then scales
# each position by its sensitivity to a change in implied volatility, so that a high-priced stock
# does not dominate a book whose purpose is exposure to volatility.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 How much an evaluation has to spend
#
# A panel of straddle-days looks large, but a strategy that opens positions once a week does not get
# to treat every row as an independent opportunity. What it spends is decision dates.
#
# One quantity here has no counterpart in a case study trading shares. A straddle sold on the last
# decision date of the development period is not resolved on that date: it resolves when the options
# expire, `labels.buffer` sessions later. So the last date this notebook may draw a conclusion from
# is not the last date before the holdout but the last one whose outcome lands before it, and the
# figure in D.2 has to stop there rather than at the boundary.

# %%
outcome_seal = timeline["timestamp"][-(BUFFER_SESSIONS + 1)]
print(
    f"Trading sessions {SESSIONS} | decision dates {len(decisions):,} | stocks per decision date "
    f"{breadth['n_symbols'].min()} to {breadth['n_symbols'].max()}, median "
    f"{breadth['n_symbols'].median():.0f}\n"
    f"Below the {BREADTH_FLOOR} the book needs on "
    f"{breadth.filter(pl.col('n_symbols') < BREADTH_FLOOR).height} of {len(breadth)} decision "
    f"dates | last date whose outcome resolves before the holdout {outcome_seal}"
)

# %% [markdown]
# ### D.2 The folds
#
# A model is fitted on one stretch of history and evaluated on the stretch that follows it, then the
# pair moves forward and the process repeats. Each fit-then-evaluate pair is a **fold**, and
# evaluating this way is called **walk-forward**, because the split always runs in the direction
# time does.
#
# One detail decides whether the evaluation is honest. A position opened near the end of a training
# block is labelled with a price from after the block ends, so validating on the session immediately
# after training would score the model on data it had partly seen. The fix is to leave a gap between
# the two, at least as wide as the outcome takes to arrive, and that gap is called **purging**. Its
# width comes from `labels.buffer`, which `generate_cv_splits` counts in trading sessions on the
# calendar `evaluation.calendar` names.
#
# The splitter is handed the sessions this panel quotes on inside the development period, which is
# the same timeline every later stage builds its folds from, because a decision date exists only
# where a listed expiration falls in the maturity window and a label exists only where the outcome
# lands before the holdout. The splitter applies the same buffer at the far end, which is why the
# last validation session is the one D.1 identified rather than the last session of the period.
#
# `generate_cv_splits` numbers folds from zero backwards from the most recent, so fold 0 is the
# last one before the holdout. The figure draws them earliest-first and labels each with that
# number, which is why the labels count down; every later stage prints the same ones.
#
# The four checks below establish what the figure cannot: the gap is
# narrow against training blocks measured in years, so only counting it off the session timeline can
# confirm it is as wide as the buffer, and only comparing the last validation date against D.1 can
# confirm the outcome of the last position evaluated does not land in the holdout. The other two
# check that the number of folds is the number `setup.yaml` declares and that no validation window
# reaches the holdout. The figure then draws the boundaries the splitter returned rather than
# recomputing them, so the picture and the folds cannot disagree.

# %%
splits = generate_cv_splits(
    research.select("timestamp").unique(),
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
assert purge_gaps == {BUFFER_SESSIONS}, "a purge gap is not the declared label buffer"
assert last_val == np.datetime64(outcome_seal), (
    "the last validation outcome resolves after the seal"
)
print(
    f"{len(splits)} folds | purge gap {min(purge_gaps)} sessions at every boundary, from "
    f"labels.buffer {LABEL_BUFFER} | last validation ends {last_val.date()}, the last date whose "
    f"outcome resolves before the holdout opens on {HOLDOUT_START}"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
ax.set_xlabel("Session")
add_message_title(
    ax,
    "Folds roll forward and stop where the last outcome still resolves",
    subtitle="Boundaries as generate_cv_splits returned them; the purge gap separates each pair",
)
show_with_alt(
    fig,
    "Horizontal timeline with one row per cross-validation fold, fold 1 on the upper row and "
    "fold 0 on the lower, each drawn as a training span, a narrow purge gap and a validation "
    "span, with the holdout drawn as a separate pale band running from 2021 to 2022. Fold 1 is "
    "the earlier pass, training from early 2017 to late 2018 and validating through 2019 into "
    "the first days of 2020; fold 0 trains from early 2018 to late 2019 and validates through "
    "2020 to November. Each training span is separated from its own validation span by the "
    "purge gap, and neither validation span reaches the holdout.",
)

# %% [markdown]
# ## E. What this notebook hands on
#
# Nothing. Membership is decided by the option listings themselves - a stock is available on a date
# when one of its expirations falls in the maturity window - so there is no eligibility table for a
# later stage to filter on. The cheapest-fifth rule is applied per date from the spreads quoted that
# day, which is what the second figure in Section B.3 draws, and the cost stage recomputes it there.

# %% [markdown]
# ## F. What the evidence says about each setting
#
# One row per setting: the evidence behind it, and the condition under which a reader working on
# their own data would choose differently.
#
# | Setting | Evidence | Choose differently when |
# |---|---|---|
# | `universe.underlying` | B.2, stocks quoting a straddle at each decision date | the count falls under what the cheapest fifth needs to fill the book on dates outside a market dislocation |
# | `decision.entry_cadence` | B.2 the listing cycle, B.4 how long the volatility gap lasts | the gap decays inside one week, or coverage stops turning over with the expiration calendar |
# | `costs.components.option_spread` | B.3, half the round trip at the cheapest fifth and at the median stock | the crossing the traded universe pays leaves the declared range in either direction |
# | `backtest.sweep.htm_cost_cascade` | B.1 cost in both units, B.3 cost per stock and through time | the spread narrows enough that buying the position back stops deciding whether a move clears its cost |
# | `evaluation.n_splits` | D.1 decision dates and the outcome seal, D.2 fold boundaries | the folds no longer fit the development period ahead of the date whose outcome still resolves |

# %%
thin = breadth.filter(pl.col("n_symbols") < BREADTH_FLOOR).sort("timestamp")
print(
    f"universe.underlying {SETUP['universe']['underlying']}, {research['symbol'].n_unique()} stocks"
    f" quoted, {len(cost)} of them on a decision date, cheapest fifth of those at or under "
    f"{LIQUID_CUT:.4f} of premium\n"
    f"decision.entry_cadence {SETUP['decision']['entry_cadence']} | "
    f"decision.execution_delay {SETUP['decision']['execution_delay']} | "
    f"labels.primary {PRIMARY_LABEL} | labels.buffer {LABEL_BUFFER}\n"
    f"costs.components.option_spread {ASSUMED_PCT[0]} to {ASSUMED_PCT[-1]} percent of premium per "
    f"crossing; measured {LIQUID_CUT * 50:.2f} at the cheapest fifth, {COST_SHARE * 50:.2f} at the "
    f"median stock, {cost['round_trip'].max() * 50:.2f} at the widest\n"
    f"htm_cost_cascade top_k {CASCADE['top_k']} of the cheapest {CASCADE['liquid_quantile']:.0%}, "
    f"cost_fractions {CASCADE['cost_fractions']}, below the floor on {thin.height} of "
    f"{len(breadth)} dates ({thin['timestamp'].min()} to {thin['timestamp'].max()})\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {last_val.date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# The development period quotes 605 stocks, 599 of which appear on at least one decision date, and
# the cheapest fifth of those crosses at or under 0.0887 of the premium. Between 77 and 469 stocks
# carry a straddle on a decision date, and the count falls under the 100 a top-20 book drawn from
# the cheapest fifth needs on 4 of 209 dates, all of them between 2020-03-06 and 2020-04-09. One
# crossing costs 4.44 percent of the premium at that cheapest fifth, inside the 2 to 5 percent the
# configuration assumes, and 5.92 percent at the median stock, above it. Two folds are generated,
# the last validation ending 2020-11-10, which is also the last date whose outcome resolves before
# the holdout.

# %% [markdown]
# ## Key takeaways
#
# 1. **Express a cost in the unit the position earns in.** A short option position earns on the
#    premium, and its spread is quoted on the premium, so a cost model built for shares ranks the
#    universe by how volatile its members are rather than by how expensive they are to trade.
# 2. **Charge the crossings the strategy actually makes.** A position held to expiration is never
#    bought back, so it pays one leg where a position that is closed pays two, and the two cost
#    lines sit an order of magnitude apart on the same distribution of moves.
# 3. **Count the universe on the dates the strategy acts.** Where membership follows a listing
#    calendar rather than a liquidity rule, the count cycles for reasons the strategy does not
#    control, and an average over all sessions hides it.
# 4. **Follow a derivative through its own contract.** A panel that re-picks the nearest contract
#    each session is not one series, and differencing it measures the switch rather than the move;
#    index the horizon in sessions the contract was quoted on, not in rows.
# 5. **Compute a panel autocorrelation inside each entity, then average.** Stacking entities into
#    one series measures the joins between them.
# 6. **Where the outcome arrives after the decision, the last usable date is earlier than the
#    boundary.** With an outcome known only at expiration, the walk-forward split has to stop a
#    label horizon short of the holdout, and that date is what the latest fold validates to.
#
# ### Known limitations
#
# - Cost here is the quoted option spread alone. The commission, the share trades the daily hedge
#   requires and the margin the position ties up all need a notional to be expressed against, and
#   they enter at the cost stage.
# - The moves in Section B.3 are unhedged marks. Re-hedging the direction at each close removes part
#   of what a seller would collect, and the labels are where the hedged outcome is built.
# - Section B.4's implied volatility is the contract nearest the money in a maturity bucket rather
#   than a fixed tenor, so a change of expiration moves the series alongside the premium it is meant
#   to measure.
# - The universe thins and the spread widens in the same weeks, which are the weeks a short
#   volatility position is most exposed. Neither figure separates a stock that stopped quoting from
#   one whose quotes stopped meeting the panel's quality conditions.
#
# **Next**: labels at the declared horizon, built on this development period.
