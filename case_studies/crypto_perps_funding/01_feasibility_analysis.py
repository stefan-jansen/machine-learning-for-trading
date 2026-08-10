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
# # Crypto Perpetuals Funding: Feasibility Analysis
#
# Before building a trading strategy it is worth asking whether the data can support one at all.
# This notebook does that and nothing else: it fits no model and makes no forecast.
#
# The strategy being checked is described in `config/setup.yaml`. It trades perpetual futures on
# Binance, sorts them every eight hours by how expensive it currently is to hold each one long, and
# buys one end of that ordering while selling the other. That file says which contracts it trades,
# when it is allowed to change positions, what a trade is assumed to cost, and how the history is
# divided between designing the strategy and testing it. This notebook checks each of those
# assumptions against the data and reports what it finds.
#
# ## Learning objectives
#
# By the end of this notebook you will be able to:
#
# - Count the members of a panel whose entities start at different dates, at the moments a strategy
#   is allowed to trade, and compare that count against the number of positions it wants to hold
# - Read off one chart what fraction of price moves are larger than the fee charged to capture them
# - Measure how long a per-contract quantity keeps describing the same contract, computing the
#   correlation inside each contract rather than across contracts stacked into one series
# - Turn that persistence into the number of independent observations a sample holds, which is
#   smaller than the number of rows it contains
# - Check that a walk-forward split of the history fits the sample available and leaves the test
#   period unread
#
# ## Book reference
#
# Chapter 6, Sections 6.2-6.6. This notebook reads Binance perpetual bars and `config/setup.yaml`,
# and writes nothing.
#
# ## Prerequisites
#
# None beyond what the sections below define. A reader who has not traded a perpetual future or
# split a sample for walk-forward evaluation will find both explained where they are first used.

# %%
"""Crypto Perpetuals Funding Case Study - Feasibility Analysis."""

import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml
from IPython.display import display

from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import load_crypto_perps
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt, zero_line

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "crypto_perps_funding"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
MAX_SYMBOLS = 0

# %% [markdown]
# ## Configuration
#
# Everything the strategy assumes is declared in `config/setup.yaml`, and this notebook reads those
# values rather than repeating them, so the two can never disagree. Four groups of settings matter
# here, and each one decides something the sections below test.
#
# **How the history is divided.** The sample runs from 2020 to the end of 2025. The last two years
# are the *holdout*: a stretch of history that is not looked at while the strategy is being
# designed, so that when it is finally evaluated there, the result is not a rehearsal of choices
# already tuned on the same data. Everything computed in this notebook uses the earlier part,
# called the development period. `holdout_start` is where the line falls.
#
# **What the strategy trades.** `setup.yaml` names 19 contracts. They do not all exist for the whole
# sample: a perpetual starts when the venue lists it and has no history before that, so the panel is
# *unbalanced* and its width is something to measure rather than assume. The strategy holds both
# ends of its ranking, up to 10 contracts per side, so at least 20 have to be quoting whenever it
# rebalances. That floor comes from the grid of book sizes the strategy will later search over, not
# from a separate assumption.
#
# **When it is allowed to act.** `decision.cadence` puts every decision on the venue's eight-hour
# funding schedule, three times a day. This is an information schedule rather than a parameter to
# tune: a new observation of what the strategy ranks on exists only when a funding period settles.
#
# **What a trade is assumed to cost.** A flat exchange fee, quoted in basis points and charged on
# each side of a round trip, in two tiers. Unlike a bid-ask spread it is published by the venue
# rather than measured from the data, and Section B.3 compares it against the moves it is charged
# on.

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

HOLDOUT_START = str(SETUP["evaluation"]["holdout_start"])
HOLDOUT_END = str(SETUP["evaluation"]["holdout_end"])
HOLDOUT_TS = pl.lit(HOLDOUT_START).str.to_datetime().dt.replace_time_zone("UTC")
PRIMARY_LABEL, LABEL_BUFFER = SETUP["labels"]["primary"], SETUP["labels"]["buffer"]
DECLARED = set(SETUP["universe"]["symbols"])
BREADTH_FLOOR = 2 * max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
BAR_HOURS = int(SETUP["decision"]["cadence"].split("_")[0])
HORIZONS = sorted(
    int(re.search(r"(\d+)h$", name).group(1))
    for name in (PRIMARY_LABEL, *SETUP["labels"]["variants"])
    if name.startswith("fwd_ret")
)
MAKER_RT, TAKER_RT = (2 * SETUP["costs"]["fee_schedule"][k] for k in ("maker_bps", "taker_bps"))
MAJORS = set(SETUP["features"]["majors"])
PREMIUM = "premium_index_close"
AVAILABLE = pl.col("timestamp") + pl.duration(hours=BAR_HOURS)  # a bar is known when it closes
ACF_LAGS = 21 * 24 // BAR_HOURS  # settlements in three weeks, the span Section B.4 draws

print(f"Sample: {START_DATE} to {END_DATE}")
print(f"  Development period, used everywhere below:  {START_DATE} to {HOLDOUT_START}")
print(f"  Holdout, not read by this notebook:         {HOLDOUT_START} to {HOLDOUT_END}")
print(f"Universe: {len(DECLARED)} perpetual contracts declared")
print(
    f"  Up to {BREADTH_FLOOR // 2} held per side, so at least {BREADTH_FLOOR} must be quoting at a "
    f"settlement to fill both legs"
)
print(
    f"Decision times: every {BAR_HOURS} hours on the funding schedule, "
    f"{24 // BAR_HOURS} a day, execution at the settlement itself"
)
print(
    f"Assumed cost: {MAKER_RT} bps round trip at the maker tier, {TAKER_RT} bps at the taker tier"
)
print(
    f"Forecast horizons: {' and '.join(f'{h} hours' for h in HORIZONS)} ahead; {HORIZONS[0]} hours "
    f"is the primary horizon and sets the gap that separates training from validation"
)

# %% [markdown]
# ## A. Orientation
#
# ### What a perpetual future is
#
# An ordinary futures contract has an expiry date, and its price is pulled towards the price of what
# it tracks as that date approaches, because on the day itself the two have to agree. A *perpetual*
# future never expires, so nothing pulls it back on its own. The venue supplies the force instead.
# Every eight hours it compares the contract's price against an *index price* built from spot markets
# on several exchanges, and whichever side of the contract is on the expensive end of that comparison
# pays the other. That payment is called *funding*, the moment it settles is the *funding timestamp*,
# and the running measure the payment is computed from is the *premium index*.
#
# The premium index takes both signs. It is positive when the contract trades above the index it
# tracks, which is what happens when leveraged buyers crowd in, and negative when it trades below.
# Funding is a transfer between the two sides of the contract rather than a fee the venue keeps, so
# it is not a trading cost: it is what the crowded side pays the other side for the privilege.
#
# ### Why ranking contracts is a strategy at all
#
# This strategy does not collect that transfer. It reads the premium as a statement about crowding -
# a contract whose longs are paying heavily to stay long is one that a lot of borrowed money is
# leaning on - and bets on what the price does next rather than on the payment itself. Every eight
# hours it sorts the contracts it can trade by their premium, buys one end of the ordering and sells
# the other, and holds until the following settlement. Whether the ordering carries any information
# is a question for Chapter 7 onwards; what this notebook asks is whether the data could support the
# attempt.
#
# A strategy of that shape needs breadth more than it needs depth. Choosing ten contracts out of six
# is not a choice, so it matters more that many contracts are quoting at once than that any one of
# them is quoting well.
#
# ### The three questions this notebook asks
#
# 1. **Does the universe exist when the strategy trades?** Positions change at every settlement, so
#    enough contracts have to be quoting at each of those moments to fill both sides of the book.
# 2. **Is a typical price move worth more than the fee charged to capture it?** Every round trip pays
#    the exchange fee twice, and that fee is the same fraction of the price for every contract.
# 3. **Is there enough independent history to evaluate this honestly?** Enough to split into training
#    and validation periods more than once, with the holdout left untouched - and rows are not the
#    same thing as independent observations when consecutive readings resemble each other.

# %% [markdown]
# ## B. Universe and cost feasibility
#
# ### B.1 Load and verify the declared universe
#
# The loader aggregates raw hourly bars onto the funding grid, so one row is one contract at one
# settlement, labelled at the bar's opening timestamp. That timestamp moves onto the availability
# clock once, here, and the seal, the folds and the horizons all read it. The panel is unbalanced.

# %%
bars = load_crypto_perps(
    frequency=f"{BAR_HOURS}h", start_date=START_DATE, end_date=END_DATE, max_symbols=MAX_SYMBOLS
)
bars = bars.with_columns(AVAILABLE).sort(["symbol", "timestamp"])
research = bars.filter(pl.col("timestamp") < HOLDOUT_TS)

loaded = set(research["symbol"].unique().to_list())
assert not loaded - DECLARED, f"in the data, undeclared in setup.yaml: {sorted(loaded - DECLARED)}"
print(
    f"{len(loaded)} of {len(DECLARED)} declared contracts, {len(research):,} funding bars, "
    f"{research['timestamp'].min().date()} to {research['timestamp'].max().date()}"
)

# %% [markdown]
# Nineteen tickers are a list, not a description. The table below groups them the way the cost model
# already does. `costs.fee_schedule` charges two tiers, and `features.majors` names the five
# contracts that clear at the cheaper one; everything else pays the taker fee. That assignment comes
# from which contracts carry the most volume rather than from anything measured here, and the
# turnover column is the check on it.
#
# Three columns describe each contract. *First settlement* is the earliest funding bar in the
# development window, which for most of these is the date the venue listed the contract rather than
# the start of the sample. *Settled share* is the fraction of the eight-hour slots between that date
# and the end of the development window for which a bar exists, so a value below one means the
# contract has holes in its history - Section B.3 has to work around exactly those. *Median premium*
# is the middle value of the quantity the strategy ranks on, in basis points of the index the
# contract tracks, which shows both its typical sign and how far the contracts differ from one
# another.

# %%
LAST_SETTLEMENT = research["timestamp"].max()
slots_since_listing = (
    (pl.lit(LAST_SETTLEMENT) - pl.col("listed")).dt.total_hours() // BAR_HOURS
) + 1
roster = (
    research.with_columns(turnover=pl.col("close") * pl.col("volume"))
    .group_by("symbol")
    .agg(
        pl.col("timestamp").min().alias("listed"),
        pl.len().alias("settlements"),
        pl.col(PREMIUM).median().mul(1e4).alias("median_premium"),
        pl.col("turnover").median().alias("median_turnover"),
    )
    .with_columns(
        pl.when(pl.col("symbol").is_in(MAJORS))
        .then(pl.lit("maker"))
        .otherwise(pl.lit("taker"))
        .alias("fee_tier"),
        (pl.col("settlements") / slots_since_listing).alias("settled_share"),
    )
    .select(
        "fee_tier",
        "symbol",
        pl.col("listed").dt.date().alias("first_settlement"),
        "settlements",
        pl.col("settled_share").round(3),
        pl.col("median_premium").round(1).alias("median_premium_bps"),
        (pl.col("median_turnover") / 1e6).round(1).alias("median_turnover_musd"),
    )
    .sort(["fee_tier", "first_settlement", "symbol"])
)
with pl.Config(tbl_rows=roster.height, tbl_cols=roster.width, tbl_width_chars=200):
    display(roster)

# %% [markdown]
# ### B.2 Breadth at every funding timestamp
#
# A single count over the whole sample would hide the question a strategy of this shape has to
# answer, which is how many contracts are quoting *at the moment it has to choose between them*.
# The reference line is the number the largest entry in `backtest.sweep.top_k_grid` requires: ten
# positions on each side, so twenty contracts quoting at once. Where breadth is below that line, the
# strategy cannot fill the book it declares, and it would have to hold fewer names or drop that grid
# entry.

# %%
breadth = research.group_by("timestamp").agg(n=pl.col("symbol").n_unique()).sort("timestamp")

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(breadth["timestamp"], breadth["n"], color=COLORS["blue"], lw=1.0)
ax.axhline(
    BREADTH_FLOOR, color=COLORS["copper"], ls="--", lw=1.5, label="contracts the largest book needs"
)
ax.set_ylim(0, BREADTH_FLOOR + 2)
ax.set_yticks(range(0, BREADTH_FLOOR + 3, 5))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Contracts quoting at the funding timestamp")
ax.legend(frameon=False, fontsize=8, loc="lower right")
add_message_title(
    ax,
    "The universe never fills the largest book the strategy may hold",
    subtitle="Perpetuals with a settled funding bar at each timestamp, against the 20 a top-10 book needs",
)
show_with_alt(
    fig,
    "Step line of the number of perpetuals quoting at each funding timestamp, rising from 2 at "
    "the start of 2020 to 7 by the spring, then to 16 by the end of that year. It holds at 16 "
    "through 2021 and 2022, apart from two narrow drops to 12 in early 2022, and steps up again "
    "to 19 during 2023. A dashed line at 20, the contracts a top-10 book needs on both sides, "
    "runs above the series for the whole window and is never reached.",
)

# %% [markdown]
# ### B.3 What a move is worth against the fee
#
# `setup.yaml::costs.fee_schedule` charges a flat fee per trade in two tiers rather than a
# per-contract spread, so cost here is a level the venue publishes, not something this data
# measures. The question is what fraction of price moves are larger than that fee, at each horizon
# the labels will be built at.
#
# The chart below is an *exceedance curve*: at each move size on the horizontal axis it shows the
# fraction of moves that were at least that large. Reading up from the round-trip fee gives the
# share of moves big enough to pay for themselves. The moves are unsigned, so this measures how much
# the price travels, not how much of that travel a strategy could capture - the second is a question
# about forecasting, and Chapter 7 is where it starts.
#
# Two details of the construction matter. This case study declares no eligibility rule - the
# contract list in `setup.yaml` is fixed - so the population is every settlement in the panel rather
# than a filtered subset of it. And a move counts only when the bar it ends on sits exactly one
# horizon ahead. The table in B.1 shows contracts with holes in their history, so the endpoint is
# matched by timestamp rather than by row position, which keeps a three-day gap out of the
# eight-hour distribution.

# %%
moves = bars
for h in HORIZONS:
    endpoint = pl.col("timestamp") + pl.duration(hours=h)
    ahead = pl.col("close").shift(-h // BAR_HOURS).over("symbol")
    on_grid = pl.col("timestamp").shift(-h // BAR_HOURS).over("symbol") == endpoint
    known = endpoint < HOLDOUT_TS
    moves = moves.with_columns(
        pl.when(on_grid & known).then((ahead / pl.col("close") - 1).abs() * 1e4).alias(f"h{h}")
    )

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
for h, color in zip(HORIZONS, (COLORS["blue"], COLORS["amber"]), strict=True):
    magnitude, fraction = exceedance_curve(moves[f"h{h}"].drop_nulls().to_numpy())
    ax.plot(magnitude, fraction, color=color, lw=1.6, label=f"{h}-hour move")
ax.axvline(MAKER_RT, color=COLORS["neutral"], ls=":", lw=1.5, label="maker round trip")
ax.axvline(TAKER_RT, color=COLORS["copper"], ls="--", lw=1.5, label="taker round trip")
ax.set_xscale("log")
ax.set_xlim(left=1)
ax.set_xlabel("Absolute move (bps, log scale)")
ax.set_ylabel("Fraction of moves at least this large")
ax.legend(frameon=False, fontsize=8, loc="lower left")
add_message_title(
    ax,
    "Moves at both horizons clear the round trip that captures them",
    subtitle="Exceedance of absolute perpetual returns against the declared maker and taker round trips",
)
show_with_alt(
    fig,
    "Two exceedance curves on a logarithmic move axis running from 1 bp to beyond 10,000 bps. "
    "Both sit at essentially 1.0 out to about 10 bps, fall away through the hundreds, and reach "
    "zero by roughly 2,000 bps. The 24-hour curve lies above the 8-hour one at every size, so a "
    "longer horizon carries more mass at every threshold. A dotted line at the 4 bp maker round "
    "trip and a dashed line at the 8 bp taker round trip both fall inside the flat left-hand "
    "stretch, where almost every move is larger than the fee charged to capture it.",
)

# %% [markdown]
# ### B.4 How long the premium describes the contract
#
# Ranking contracts every eight hours is only worth the trading it causes if what the premium says
# at one settlement still describes the same contract at the next. The statistic that answers this
# is *autocorrelation*: the correlation between a series and the same series shifted back by a fixed
# number of steps, called the lag. At lag 1 it asks whether a contract that is expensive to hold now
# was also expensive to hold at the previous settlement; at lag 21 it asks whether it was expensive
# a week ago.
#
# Two choices about how it is computed change the answer. It is computed *inside each contract* and
# then averaged, because stacking every contract into one series and correlating that would mostly
# measure the point where one contract's history ends and the next begins. And it is computed over
# each contract's longest unbroken run of settlements, because the helper counts lags by row: across
# a hole in the history, row 1 and row 2 are not eight hours apart. The shaded band shows how much
# contracts differ from one another, and the horizontal strip is the range within which a
# correlation is indistinguishable from zero at this sample size.

# %%
gap = pl.col("timestamp").diff().over("symbol").ne(pl.duration(hours=BAR_HOURS)).fill_null(True)
unbroken = (
    research.with_columns(gap.cum_sum().over("symbol").alias("run"))
    .with_columns(pl.len().over("symbol", "run").alias("run_len"))
    .filter(pl.col("run_len") == pl.col("run_len").max().over("symbol"))
)
# a series correlated with itself is 1 by construction, and that bar would flatten every other one
acf = panel_acf(unbroken, entity_col="symbol", value_col=PREMIUM, max_lags=ACF_LAGS)
drawn = acf.filter(pl.col("lag") > 0)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.axhspan(
    -acf["band"][0],
    acf["band"][0],
    color=COLORS["copper"],
    alpha=0.35,
    zorder=0,
    label="range expected from no information",
)
ax.fill_between(
    drawn["lag"],
    drawn["acf_p10"],
    drawn["acf_p90"],
    color=COLORS["blue"],
    alpha=0.15,
    label="10th to 90th percentile across contracts",
)
ax.bar(drawn["lag"], drawn["acf"], color=COLORS["blue"], width=0.7)
ax.set_xlabel("Settlements between the two readings")
ax.set_ylabel("Correlation with the contract's own past")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
add_message_title(
    ax,
    "Premium persistence outlasts every horizon the labels declare",
    subtitle="Mean within-contract autocorrelation, shaded 10th-90th percentile across contracts",
)
show_with_alt(
    fig,
    "Bars of the mean within-contract autocorrelation of the premium index against lag, from one "
    "settlement to sixty-three. The first bar is about 0.48, the next few fall to about 0.40, and "
    "the decay is slow from there to roughly 0.22 at the longest lag drawn. A shaded band for the "
    "10th to 90th percentile across contracts runs from about 0.68 down to about 0.42 at its "
    "upper edge and is wide throughout. A narrow strip around zero marks the range expected from "
    "no information, and every bar sits far above it, so persistence outlasts both the 8-hour and "
    "the 24-hour label horizon.",
)

# %% [markdown]
# A cross-sectional book also needs contracts to disagree at one timestamp, so quantiles are taken
# there and thinned to a daily median only after: pooling first folds the level into the band.

# %%
BANDS = (("lo", 0.1), ("mid", 0.5), ("hi", 0.9))
spread = (
    research.group_by("timestamp")
    .agg(pl.col(PREMIUM).quantile(q).mul(1e4).alias(name) for name, q in BANDS)
    .group_by(pl.col("timestamp").dt.truncate("1d").alias("day"))
    .agg(pl.col(name).median() for name, _ in BANDS)
    .sort("day")
    .drop_nulls()
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.fill_between(
    spread["day"],
    spread["lo"],
    spread["hi"],
    color=COLORS["blue"],
    alpha=0.25,
    label="10th to 90th percentile of contracts",
)
ax.plot(spread["day"], spread["mid"], color=COLORS["blue"], lw=0.8, label="median contract")
zero_line(ax)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Premium index (bps of the tracked index)")
add_message_title(
    ax,
    "The premium moves as a common level more than contracts disperse",
    subtitle="Cross-sectional quantiles at each funding timestamp, shown at their daily median",
)
show_with_alt(
    fig,
    "The 10th to 90th percentile of the premium index across contracts at each funding timestamp, "
    "shaded, with the median contract drawn through it, both thinned to a daily median. Through "
    "2020 and 2021 the median swings between roughly minus 20 and plus 40 basis points and the "
    "band moves with it rather than around it. From 2022 the whole picture narrows to within a "
    "few basis points of zero, apart from isolated spikes to about minus 35, and turns positive "
    "again through late 2023. The band is narrow relative to how far the median travels, which is "
    "the level a long-short book cancels.",
)

# %% [markdown]
# ### B.5 Move scale against cost
#
# The ratio divides the median absolute move at the primary horizon by the taker round trip, the
# tier a contract outside the majors pays. It says nothing about whether the move is forecastable.

# %%
primary = moves[f"h{HORIZONS[0]}"].drop_nulls()
print(
    f"Round trip {MAKER_RT} bps at maker and {TAKER_RT} bps at taker | median {HORIZONS[0]}-hour "
    f"move {primary.median():.0f} bps, ratio {primary.median() / TAKER_RT:.0f}x, share above the "
    f"taker round trip {(primary > TAKER_RT).mean():.3f}"
)

# %% [markdown] tags=["results"]
# The maker round trip costs 4 bps and the taker round trip 8 bps. The median absolute 8-hour move
# is 138 bps, 17 times the taker round trip, and 0.963 of moves are larger than it. The numerator is
# an unsigned magnitude, so it bounds the room a forecast has and says nothing about whether one
# exists.

# %% [markdown]
# ## C. Design decisions
#
# ### C.1 Cadence
#
# `setup.yaml::decision.cadence` rebalances on the funding grid and executes at the funding
# timestamp. That is an information schedule rather than a hyperparameter to sweep: a new premium
# observation exists only when a period settles, so a decision between settlements reads the same
# premium twice and pays twice for it. B.4 supports holding through at least one period.
#
# ### C.2 Kill conditions
#
# Four falsifiable checkpoints send the strategy back to the drawing board, each tested where its
# evidence exists: a gross return the fee erases, in Chapter 16; a premium that stops predicting by
# the next funding timestamp, in Chapter 7 through the information coefficient; a venue change to
# the funding formula, cap or interval, leaving the training distribution describing a product that
# no longer exists; and an equal-weight cross-section with a higher Sharpe, in Chapter 17.
#
# ### C.3 Mapping class
#
# `setup.yaml::mapping.class` ranks contracts on the premium and holds both legs, which a perpetual
# allows from either side and which cancels the level B.4 shows is most of what the premium does.
# Sizing is equal weight or risk parity: Chapter 16 fixes the first, Chapter 17 sweeps the rest.

# %% [markdown]
# ## D. Walk-forward structure
#
# ### D.1 Effective sample size
#
# What evaluation spends is independent observations, not rows. Summing the initial positive
# sequence of B.4's mean curve gives the integrated autocorrelation time, the funding periods one
# independent premium observation is worth - for one contract's own premium series, not for the
# decisions a portfolio takes across contracts. The sequence never turns negative here, so the
# count below is a ceiling.

# %%
curve = acf["acf"].to_numpy()
pairs = curve[1::2][: len(curve[2::2])] + curve[2::2]
turns = np.flatnonzero(pairs <= 0)
tau = 1 + 2 * pairs[: turns[0] if turns.size else len(pairs)].sum()
raw_periods = SETUP["evaluation"]["periods_per_year"] * 24 // BAR_HOURS
print(
    f"Funding timestamps {len(breadth):,} | contracts per timestamp {breadth['n'].mean():.1f} | "
    f"integrated autocorrelation {tau:.0f} funding periods, so at most {raw_periods / tau:.0f} "
    f"independent premium observations per contract a year against {raw_periods:,} settlements"
)

# %% [markdown]
# ### D.2 Fold demonstration
#
# A walk-forward split cuts the development period into consecutive blocks: a *training* window the
# model is fitted on, then a *validation* window it is scored on, with the pair sliding forward to
# make the next fold. Between the two sits a *purge gap*, a stretch that is dropped from both. It is
# needed because a label is a statement about the future: a target computed at the last moment of
# training resolves some hours later, and without the gap that resolution falls inside validation
# and the score is partly a score on data the model was fitted on.
#
# `generate_cv_splits` places those boundaries from the widths in `setup.yaml::evaluation` and the
# gap from the label buffer, and the figure draws the boundaries it returned rather than
# recomputing them, so the picture and the folds cannot disagree.
#
# The splitter is given the whole sample, holdout included, and applies the holdout boundary itself
# from `evaluation.holdout_start`, which is what every later stage does too. Trimming the data first
# would shift the first training settlement of most folds, and the figure would then show a training
# window the pipeline never trains on. The splitter numbers folds from zero backwards from the most
# recent, so fold 0 is the one that ends against the holdout. The figure draws them earliest-first
# and labels each with that number, which is why the labels count down; every later stage prints the
# same ones.
#
# The gap drawn is the buffer for the primary label, `labels.buffer`. The longest declared variant,
# `fwd_ret_24h`, resolves three settlements out and carries its own wider buffer in
# `labels.variant_buffers`; the figure shows the primary one, and a fold built for that variant has
# a proportionally wider gap.

# %%
splits = generate_cv_splits(
    bars.select("timestamp"),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)
last_val = max(split["val_end"] for split in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
holdout_opens = pd.Timestamp(HOLDOUT_START, tz="UTC")
assert last_val + pd.Timedelta(LABEL_BUFFER) < holdout_opens, (
    "a fold's last label reaches the holdout"
)
BUFFER_SLOTS = int(pd.Timedelta(LABEL_BUFFER) / pd.Timedelta(hours=BAR_HOURS))
purged = {
    int((s["val_start"] - s["train_end"]) / pd.Timedelta(hours=BAR_HOURS)) - 1 for s in splits
}
assert purged == {BUFFER_SLOTS}, (
    f"{sorted(purged)} settlements purged, not the {BUFFER_SLOTS} the {LABEL_BUFFER} buffer needs"
)
widths = {
    ((s["train_end"] - s["train_start"]).days, (s["val_end"] - s["val_start"]).days) for s in splits
}
assert len(widths) == 1, f"folds differ in width: {sorted(widths)}"
train_days, val_days = widths.pop()
print(
    f"{len(splits)} folds | training {train_days} days and validating {val_days} days each | "
    f"{purged.pop()} settlement purged between them, matching the {LABEL_BUFFER} label buffer"
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
fold_timeline(ax, splits, holdout=(HOLDOUT_START, HOLDOUT_END))
add_message_title(
    ax,
    "Folds roll forward and stop short of the holdout",
    subtitle="Boundaries as generate_cv_splits returned them; the one-settlement purge is too narrow to see",
)
show_with_alt(
    fig,
    "Two horizontal fold bars against a date axis. Fold 1 trains from the start of 2020 to the "
    "end of 2021 and validates through 2022; fold 0 trains from the start of 2021 to the end of "
    "2022 and validates through 2023, so the pair slides forward by one year and the later fold "
    "carries the higher position on the axis while holding the lower number. A shaded holdout "
    "block covers 2024 and 2025 and neither validation window reaches it. The one-settlement "
    "purge gap between each training and validation window is present in the legend and too "
    "narrow to resolve at this scale.",
)

# %% [markdown]
# ## E. Derived artifacts
#
# This notebook writes nothing. `setup.yaml::universe.symbols` fixes the contract list and the
# loader carries no row before a contract was listed, so there is no eligibility table for a later
# notebook to filter on - the panel itself already contains only what could have been traded.

# %% [markdown]
# ## F. Findings vs `setup.yaml`
#
# Each declared setting is paired below with the evidence in this notebook that motivates it, and
# with the condition under which a reader working on their own data would revise it.
#
# | Knob | Evidence | Revise it when |
# |---|---|---|
# | `universe.symbols`, `backtest.sweep.top_k_grid` | B.2 breadth against the contracts a top-10 book needs on both sides | breadth falls below what the book needs, as it does here at every settlement for the top-10 entry |
# | `decision.cadence` | B.3 exceedance, B.4 persistence | moves stop clearing the round trip, or the premium decays inside one funding period |
# | `costs.fee_schedule` | B.3 the two declared round trips | the venue changes a tier, or a contract moves between them |
# | `evaluation.n_splits` | D.1 independent observations, D.2 boundaries | the folds no longer fit the development window |

# %%
print(
    f"universe.n_assets {SETUP['universe']['n_assets']}, breadth {breadth['n'].min()} to "
    f"{breadth['n'].max()}, under the floor of {BREADTH_FLOOR} on "
    f"{breadth.filter(pl.col('n') < BREADTH_FLOOR).height:,} of {len(breadth):,} timestamps\n"
    f"decision.cadence {SETUP['decision']['cadence']} | labels.primary {PRIMARY_LABEL}\n"
    f"evaluation.n_splits {SETUP['evaluation']['n_splits']}, generated {len(splits)}, last "
    f"validation ends {last_val.date()}, holdout untouched"
)

# %% [markdown] tags=["results"]
# The declared universe holds 19 contracts. Breadth at a funding timestamp runs from 2 to 19 and
# stays under the 20 that the largest declared book needs on all 4,382 of them, so the
# top-10 grid entry cannot fill both legs anywhere in the development window. Two folds are
# generated, the last validation ending 2023-12-31. An integrated autocorrelation of 37 funding
# periods leaves at most 30 independent premium observations per contract a year, of 1,095.

# %% [markdown]
# ## Key takeaways
#
# 1. **Count the panel at the moments the strategy is allowed to trade.** Contracts entering at
#    their listing dates give breadth a history, and the book the strategy wants has to fit inside
#    that history at every one of those moments, not on average.
# 2. **Compute a panel autocorrelation inside each entity, over unbroken stretches.** Stacking
#    contracts measures where two of them meet; counting lags by row measures across the gaps.
# 3. **Turn persistence into an observation count before trusting a sample size**, and read it as a
#    ceiling where the initial positive sequence runs past the lags you drew.
# 4. **Separate the common level from the cross-sectional spread.** A ranking reads what is left
#    once the level both legs cancel comes out, and here the level is the larger part.
#
# ### Known limitations
#
# - The contract list is fixed and was drawn knowing which perpetuals stayed listed, so it carries
#   selection and delisting bias, and the earliest folds see a much narrower cross-section.
# - Cost is the published fee alone; slippage and the entry spread need a notional and enter at the
#   cost stage. Funding itself is not part of the cost: the strategy holds the price move rather
#   than collecting the transfer, so the labels and the registered backtest measure the move net of
#   fees only, and the premium enters as a feature. Chapter 13 replays the selected configuration a
#   second time with the official settlements added, outside the registry, to show separately what
#   the transfer would have been worth.
#
# **Next**: labels at the declared horizons, built on this development window.
