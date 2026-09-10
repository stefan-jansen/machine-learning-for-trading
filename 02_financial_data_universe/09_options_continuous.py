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
# # Constructing Continuous Options Series
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Demonstrate why a "constant-maturity" option series chains together different
# contracts, produces phantom price jumps at every roll, and contaminates label
# construction. Implement two clean alternatives: same-contract holding returns
# (correct labels) and a roll-zeroed continuous reconstruction (backtestable
# mark-to-market).
#
# ## Learning Objectives
#
# - Detect contract rolls in a 30-day ATM straddle time series and quantify the
#   resulting bias against non-roll-day returns.
# - Compute same-contract holding returns by looking up the entry contract's
#   exit price in the raw option chain.
# - Build a continuous price series suitable for backtesting, and measure what the choice of
#   roll-day return does to it.
# - Tell apart the construction a label needs from the one a mark-to-market series needs.
#
# ## Book Reference
#
# §2.2, "The asset-class market data landscape" - the derivatives part of it. The futures
# analogue is `06_futures_continuous`; the case-study scale-up is
# `case_studies/sp500_options/02_labels`.
#
# ## Prerequisites
#
# - `07_sp500_options_eda` for option-chain structure.
# - `08_options_greeks_computation` for theta and time decay.
# - The AlgoSeek S&P 500 options EDA parquet at `$ML4T_DATA_PATH/sp500_options/`.

# %%
"""Constructing Continuous Options Series — constant-maturity roll adjustment."""

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_sp500_options_eda
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
DEMO_SYMBOL = "AAPL"
DEMO_YEAR = 2019

# Constant-maturity selection. These decide which contract the series holds on each day, so
# they belong with the other declared inputs rather than beside the code that applies them.
DTE_WINDOW = (25, 35)
TARGET_DELTA = 0.50
DELTA_TOL = 0.15
MIN_BID = 0.01
MAX_REL_SPREAD = 0.30

# Holding period for the same-contract return, in trading days.
HOLDING_PERIOD = 10

# %% [markdown]
# ## 1. The Constant-Maturity Construction
#
# A 30-day ATM straddle is a common volatility instrument: you buy (or sell) both
# an ATM call and an ATM put with ~30 days to expiration. To build a daily time
# series, we select the "best" straddle each day — the one closest to 30 DTE and
# 50-delta.
#
# The problem: this selection is **independent** each day. The contract identity
# (strike, expiration) changes whenever a new weekly/monthly option enters the
# 25–35 DTE window or the underlying moves enough to shift which strike is ATM.

# %% [markdown]
# ### Loading the raw chains for one symbol

# %%
raw = load_sp500_options_eda(
    symbols=[DEMO_SYMBOL],
    start_date=f"{DEMO_YEAR}-01-01",
    end_date=f"{DEMO_YEAR}-12-31",
).rename({"timestamp": "date"})

print(f"Raw {DEMO_SYMBOL} options ({DEMO_YEAR}): {len(raw):,} rows")
print(f"  Dates: {raw['date'].n_unique()}")
print(
    f"  Unique (strike, expiration): {raw.select(pl.struct('strike', 'expiration')).n_unique():,}"
)

# %% [markdown]
# ### Selecting a constant-maturity straddle
#
# For each trading day, select the ATM straddle closest to 30 DTE.
# This replicates the logic in `materialize_options.py`.

# %% [markdown]
# ### Straddle Selection Logic
#
# For each trading day, select the ATM straddle closest to 30 DTE.


# %%
def select_constant_maturity_straddle(raw_options: pl.DataFrame) -> pl.DataFrame:
    """Select each day's ATM straddle: one call and one put on the SAME strike and expiration.

    Ranking the two legs independently and joining them on the date alone does not produce a
    straddle. It produces whichever call and whichever put each sat closest to the target, and
    when those differ the result is a strangle or a diagonal - priced, tracked and rolled as
    though it were a single instrument. The legs are therefore paired on strike and expiration
    first, and the ranking is applied to the pair.
    """
    opts = raw_options.with_columns(
        pl.col("delta").abs().alias("abs_delta"),
        ((pl.col("ask") - pl.col("bid")) / pl.col("mid_price").clip(lower_bound=0.01)).alias(
            "rel_spread"
        ),
    )
    filtered = opts.filter(
        pl.col("days_to_maturity").is_between(DTE_WINDOW[0], DTE_WINDOW[1])
        & (pl.col("bid") >= MIN_BID)
        & (pl.col("rel_spread") <= MAX_REL_SPREAD)
        & (pl.col("iv_convergence") == "Converged")
        & pl.col("abs_delta").is_between(TARGET_DELTA - DELTA_TOL, TARGET_DELTA + DELTA_TOL)
    )

    key = ["date", "strike", "expiration"]
    calls = filtered.filter(pl.col("call_put") == "C").select(
        *key,
        "days_to_maturity",
        "underlying_price",
        pl.col("mid_price").alias("call_mid"),
        pl.col("bid").alias("call_bid"),
        pl.col("ask").alias("call_ask"),
        pl.col("delta").alias("call_delta"),
        pl.col("theta").alias("call_theta"),
        pl.col("abs_delta").alias("call_abs_delta"),
    )
    puts = filtered.filter(pl.col("call_put") == "P").select(
        *key,
        pl.col("mid_price").alias("put_mid"),
        pl.col("bid").alias("put_bid"),
        pl.col("ask").alias("put_ask"),
        pl.col("delta").alias("put_delta"),
        pl.col("theta").alias("put_theta"),
    )

    # An inner join on the full key is what makes this a straddle: a strike and expiration
    # survive only if BOTH legs are quoted and pass the filters there.
    pairs = calls.join(puts, on=key, how="inner").with_columns(
        (pl.col("call_mid") + pl.col("put_mid")).alias("instr_mid"),
        (pl.col("call_theta") + pl.col("put_theta")).alias("instr_theta"),
        (pl.col("call_delta") + pl.col("put_delta")).alias("instr_delta"),
        (pl.col("call_abs_delta") - TARGET_DELTA).abs().alias("_delta_gap"),
        (pl.col("days_to_maturity") - sum(DTE_WINDOW) / 2).abs().alias("_dte_gap"),
    )

    return (
        pairs.sort(["date", "_delta_gap", "_dte_gap", "strike"])
        .group_by("date")
        .first()
        .drop(["_delta_gap", "_dte_gap"])
        .sort("date")
    )


# %%
cm_straddles = select_constant_maturity_straddle(raw)
print(f"Constant-maturity straddle series: {len(cm_straddles)} days")
print(f"  Date range: {cm_straddles['date'].min()} to {cm_straddles['date'].max()}")
cm_straddles.select(["date", "strike", "expiration", "days_to_maturity", "instr_mid"]).head(10)

# %% [markdown]
# ## 2. The Roll Problem
#
# How often does the underlying contract change?

# %%
cm_straddles = cm_straddles.with_columns(
    pl.col("expiration").shift(1).alias("prev_exp"),
    pl.col("strike").shift(1).alias("prev_strike"),
    pl.col("instr_mid").shift(1).alias("prev_mid"),
)

cm_straddles = cm_straddles.with_columns(
    (pl.col("expiration") != pl.col("prev_exp")).fill_null(False).alias("expiry_moved"),
    (pl.col("strike") != pl.col("prev_strike")).fill_null(False).alias("strike_moved"),
    ((pl.col("instr_mid") - pl.col("prev_mid")) / pl.col("prev_mid")).alias("daily_return"),
).with_columns(
    (pl.col("expiry_moved") | pl.col("strike_moved")).alias("is_roll"),
    pl.when(pl.col("expiry_moved"))
    .then(pl.lit("expiry changed"))
    .when(pl.col("strike_moved"))
    .then(pl.lit("strike moved, same expiry"))
    .otherwise(pl.lit("same contract"))
    .alias("change_kind"),
)

n_rolls = cm_straddles.filter(pl.col("is_roll")).height
n_total = len(cm_straddles) - 1
print(f"Contract changes: {n_rolls} out of {n_total} days ({n_rolls / n_total:.0%})")

# %% [markdown]
# ### Roll days against non-roll days
#
# If the constant-maturity series were a genuine instrument, days when the contract changed and
# days when it did not would have similar return characteristics. They do not.
#
# **But "the contract changed" is two different events.** The selection can move to a different
# strike at the same expiration, because spot drifted and a neighbouring strike is now closer
# to the money. Or it can move to a different expiration, because a new listing entered the
# day-count window. Only the second resets time value, so only the second should produce the
# phantom step - and treating them as one category would average a large effect with a small
# one and report something in between that describes neither.

# %%
by_change = (
    cm_straddles.drop_nulls("daily_return")
    .group_by("change_kind")
    .agg(
        pl.len().alias("days"),
        pl.col("daily_return").mean().alias("mean_return"),
        pl.col("daily_return").median().alias("median_return"),
        (pl.col("daily_return") > 0).mean().alias("pct_positive"),
    )
    .sort("mean_return", descending=True)
)
print("Daily return by what changed in the selection:")
print(by_change)

roll_rets = cm_straddles.filter(pl.col("is_roll"))["daily_return"].drop_nulls()
nonroll_rets = cm_straddles.filter(~pl.col("is_roll"))["daily_return"].drop_nulls()

roll_summary = pl.DataFrame(
    {
        "metric": ["count", "mean_return", "median_return", "pct_positive"],
        "roll_days": [
            float(roll_rets.len()),
            roll_rets.mean(),
            roll_rets.median(),
            (roll_rets > 0).mean(),
        ],
        "non_roll_days": [
            float(nonroll_rets.len()),
            nonroll_rets.mean(),
            nonroll_rets.median(),
            (nonroll_rets > 0).mean(),
        ],
    }
)
roll_summary = roll_summary.with_columns(
    (pl.col("roll_days") - pl.col("non_roll_days")).alias("difference"),
)
roll_summary

# %% [markdown]
# The split is the result, and only one of the three groups is positive. Expiration changes
# carry a large positive mean. Days when only the strike moved and days when nothing changed
# are close to each other and both negative, which is ordinary time decay showing through.
#
# The expiration result is what the mechanism predicts. Moving to a later expiration buys back
# the time value decay had removed, and the series records the difference as a return no
# position earned.
#
# The strike result invites a conclusion it does not support. Moving to a neighbouring strike
# at the same expiration exchanges one contract for a similar one with no time value to reset,
# and the group mean duly lands beside the group mean for days when nothing changed. That is a
# statement about two averages. It is not a statement about the individual days, which is
# where a return series is used, and Section 5 measures those directly once the held contract's
# own price is available. The answer there is not the one this table suggests.
#
# Lumping all the changes into a single "roll day" average blends one large positive effect
# with two indistinguishable negative ones and reports a figure describing none of the three.
#
# What this table does not establish is that strike changes are harmless. Two group means can
# agree while the individual days behind them are wrong in both directions, and the quantity
# that matters is per-day: the gap between the return the series records and the return the
# contract actually held would have earned. That gap needs the held contract's own price,
# which Section 5 recovers; the comparison is made there, on the days themselves.
#
# The futures analogue is roll yield, and the difference is one of degree that becomes a
# difference in kind. A quarterly futures roll contributes a small adjustment four times a
# year. This series reselects on most days, so the phantom component is not an occasional
# correction to an otherwise sound return stream - it is a large part of the stream.
#
# **A note on sign, because everything below depends on it.** Every return in this notebook is
# the return of the *instrument*: today's price over yesterday's, minus one. Positive means the
# straddle got more expensive.
#
# That is the long convention, and it is used here for every series without exception, because
# a price series and its returns should mean the same thing in every section. The case study
# this feeds sells straddles rather than buying them. A seller's P&L over a *single* period,
# before costs and for a fixed quantity, is the negation of the long return printed below.
#
# That does not extend to a cumulative return, and the difference is not small. Negating each
# period's return and compounding is a different series from negating the compounded result:
# a long series of $+10\%$ then $-10\%$ compounds to $-1\%$, and so does its per-period
# negation. A cumulative short return is only defined once the position's sizing is - whether
# the quantity is fixed for the whole period or reset against equity each day - so this
# notebook prints the long series and states the convention where a short figure is needed.

# %% [markdown]
# ### The sawtooth

# %%
dates = cm_straddles["date"].to_list()
prices = cm_straddles["instr_mid"].to_list()
dtes = cm_straddles["days_to_maturity"].to_list()
rolls = cm_straddles["is_roll"].to_list()

roll_dates = [d for d, r in zip(dates, rolls, strict=False) if r]
roll_prices = [p for p, r in zip(prices, rolls, strict=False) if r]
nonroll_dates = [d for d, r in zip(dates, rolls, strict=False) if not r]
nonroll_prices = [p for p, r in zip(prices, rolls, strict=False) if not r]

# %% [markdown]
# Both panels are built in one cell so the inline backend does not flush the figure
# mid-construction and publish a half-drawn copy above the finished one.

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        f"{DEMO_SYMBOL} constant-maturity straddle mid",
        "Days to expiration",
    ],
    vertical_spacing=0.08,
)

fig.add_trace(
    go.Scatter(
        x=nonroll_dates,
        y=nonroll_prices,
        mode="markers",
        name="Same contract",
        marker=dict(size=4, color=COLORS["blue"]),
        opacity=0.7,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=roll_dates,
        y=roll_prices,
        mode="markers",
        name="Contract switch",
        marker=dict(size=6, color=COLORS["negative"], symbol="diamond"),
        opacity=0.9,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode="lines",
        name="Price",
        line=dict(width=1, color=COLORS["neutral"]),
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=dtes,
        mode="lines+markers",
        name="DTE",
        marker=dict(size=3, color=COLORS["positive"]),
        line=dict(width=1),
    ),
    row=2,
    col=1,
)

fig.update_yaxes(title_text="Straddle Mid ($)", row=1, col=1)
fig.update_yaxes(title_text="DTE", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_layout(height=600, legend=dict(x=0.01, y=0.99))
show_plotly_with_alt(
    fig,
    "Two stacked panels sharing a date axis across the year. The upper panel plots the daily mid price of the selected straddle as a line with the points marked, distinguishing days when the contract stayed the same from days when it switched; the trace is a dense jagged band with no smooth trend, and almost every point is marked as a contract switch rather than a continuation. The lower panel plots days to expiration, which zigzags rapidly and continuously between the two edges of a narrow band, with no sustained run in either direction.",
)

# %% [markdown]
# %% [markdown]
# Run lengths below come from the contract identity itself. Encoding runs of an "unchanged"
# flag instead drops the first observation of every contract - identities A, B, B, B give two
# unchanged flags although B is held for three days - and understates every holding period by
# exactly one.

# %%
_dte = cm_straddles["days_to_maturity"]
_delta_dte = (_dte - _dte.shift(1)).drop_nulls()
_held_runs = (
    cm_straddles.select("strike", "expiration")
    .with_columns(
        (
            (pl.col("strike") != pl.col("strike").shift(1))
            | (pl.col("expiration") != pl.col("expiration").shift(1))
        )
        .fill_null(True)
        .cum_sum()
        .alias("spell")
    )
    .group_by("spell")
    .agg(pl.len().alias("days"))
)
print(f"Days to expiration stays within {_dte.min()} and {_dte.max()} all year")
print(f"  it falls on {(_delta_dte < 0).sum()} days and rises on {(_delta_dte > 0).sum()}")
print(f"  longest stretch holding one contract: {_held_runs['days'].max()} days")
print(f"  mean stretch: {_held_runs['days'].mean():.2f} days")

# %% [markdown]
# The teeth of the sawtooth are much finer than the description usually attached to one. Days
# to expiration never sweeps the width of the selection window and resets: the longest run on
# one contract is a handful of days and the average is under two, so what the panel shows is a
# rapid zigzag inside a narrow band rather than a slow decay with occasional jumps.
#
# The direction is still mostly downward, which is decay doing its work, and the upward steps
# are the expiration changes identified above. But the scale matters for what follows. A series
# that resets every few days is not a lightly contaminated version of a real instrument with a
# few bad days in it; the contamination is the majority of its observations, and any adjustment
# applied to those days touches most of the sample.

# %% [markdown]
# ## 3. Same-Contract Holding Returns
#
# The correct return for a straddle trade is: enter a specific contract at mid on
# day $t$, exit the **same contract** at mid on day $t + h$. This requires
# looking up that specific contract in the raw option chain $h$ days later.
#
# $$r_{same} = \frac{P_{exit}^{mid}(K, T) - P_{entry}^{mid}(K, T)}{P_{entry}^{mid}(K, T)}$$
#
# where $(K, T)$ identifies the specific strike and expiration.

# %% [markdown]
# ### An exit-price lookup from the raw chain


# %%
def build_exit_lookup(raw_options: pl.DataFrame) -> pl.DataFrame:
    """Build a lookup table: (date, strike, expiration, call_put) → mid_price.

    Used to find the exit price of a specific contract h days after entry.
    """
    return raw_options.filter(
        pl.col("bid") >= MIN_BID,
        pl.col("iv_convergence") == "Converged",
    ).select(
        [
            "date",
            "strike",
            "expiration",
            "call_put",
            "mid_price",
            "bid",
            "ask",
            "delta",
            "theta",
            "days_to_maturity",
        ]
    )


# %%
# Build lookup from the raw chain
lookup = build_exit_lookup(raw)
print(f"Exit price lookup: {len(lookup):,} rows")
print(
    f"  Unique contracts: {lookup.select(pl.struct('strike', 'expiration', 'call_put')).n_unique():,}"
)

# %% [markdown]
# ### Computing same-contract returns
#
# For each day's constant-maturity straddle, find the same contract's price
# $h$ trading days later in the raw chain.

# %%
# Get the trading calendar (ordered dates)
trading_dates = cm_straddles["date"].unique().sort().to_list()
date_to_idx = {d: i for i, d in enumerate(trading_dates)}


def get_exit_date(entry_date, h: int, cal: list):
    """Get the trading date h business days after entry_date."""
    idx = date_to_idx.get(entry_date)
    if idx is None or idx + h >= len(cal):
        return None
    return cal[idx + h]


# Build entry-exit pairs
entries = cm_straddles.select(
    [
        "date",
        "strike",
        "expiration",
        "instr_mid",
        "call_mid",
        "put_mid",
        "instr_delta",
        "is_roll",
    ]
).rename({"date": "entry_date", "instr_mid": "entry_mid"})

# Add exit dates
exit_dates = [
    get_exit_date(d, HOLDING_PERIOD, trading_dates) for d in entries["entry_date"].to_list()
]
entries = entries.with_columns(pl.Series("exit_date", exit_dates).cast(pl.Date))

# Drop entries where exit date is beyond our data
entries = entries.filter(pl.col("exit_date").is_not_null())
print(f"Entry-exit pairs to look up: {len(entries):,}")

# %%
# Look up call exit price
call_exit = lookup.filter(pl.col("call_put") == "C").select(
    [
        pl.col("date").alias("exit_date"),
        "strike",
        "expiration",
        pl.col("mid_price").alias("call_exit_mid"),
    ]
)

# Look up put exit price
put_exit = lookup.filter(pl.col("call_put") == "P").select(
    [
        pl.col("date").alias("exit_date"),
        "strike",
        "expiration",
        pl.col("mid_price").alias("put_exit_mid"),
    ]
)

# Join: entry contract → exit prices for same (strike, expiration)
same_contract = entries.join(call_exit, on=["exit_date", "strike", "expiration"], how="left").join(
    put_exit, on=["exit_date", "strike", "expiration"], how="left"
)

# Compute same-contract straddle exit mid
same_contract = same_contract.with_columns(
    (pl.col("call_exit_mid") + pl.col("put_exit_mid")).alias("exit_mid"),
)

same_contract = same_contract.with_columns(
    ((pl.col("exit_mid") - pl.col("entry_mid")) / pl.col("entry_mid")).alias("same_contract_ret"),
)

# How many lookups succeeded?
found = same_contract.filter(pl.col("exit_mid").is_not_null()).height
print(f"Exit prices found: {found} / {len(same_contract)} ({found / len(same_contract):.1%})")

# %% [markdown]
# ### Naive against same-contract

# %% [markdown]
# The naive return is the one a `shift` on the constant-maturity series produces. For the
# comparison to isolate the construction, both sides must cover the same window: the
# same-contract return runs from the selection day to `HOLDING_PERIOD` days later, so the naive
# one has to as well. Entering a day later, as a label-shifted version would, means the two
# series differ partly because they cover different days and partly because of the roll, and
# the comparison could not separate them.

# %%
naive_rets = cm_straddles.with_columns(
    pl.col("instr_mid").shift(-HOLDING_PERIOD).alias("naive_exit"),
).with_columns(
    ((pl.col("naive_exit") - pl.col("instr_mid")) / pl.col("instr_mid")).alias("naive_ret"),
)

# Align for comparison
comparison = (
    same_contract.filter(pl.col("same_contract_ret").is_not_null())
    .select(["entry_date", "same_contract_ret"])
    .join(
        naive_rets.select([pl.col("date").alias("entry_date"), "naive_ret"]).drop_nulls(),
        on="entry_date",
        how="inner",
    )
)

print(f"Paired observations: {len(comparison):,}")

naive_vs_same = pl.DataFrame(
    {
        "metric": ["mean_return", "std", "median_return", "pct_positive"],
        "naive_chained": [
            comparison["naive_ret"].mean(),
            comparison["naive_ret"].std(),
            comparison["naive_ret"].median(),
            (comparison["naive_ret"] > 0).mean(),
        ],
        "same_contract": [
            comparison["same_contract_ret"].mean(),
            comparison["same_contract_ret"].std(),
            comparison["same_contract_ret"].median(),
            (comparison["same_contract_ret"] > 0).mean(),
        ],
    }
)
corr = comparison.select(pl.corr("naive_ret", "same_contract_ret").alias("correlation")).item()
print(f"Correlation(naive, same-contract) = {corr:.3f}")
naive_vs_same

# %% [markdown]
# The correlation is the number to read, and it is low. Squaring it gives the share of variance
# the naive series shares with the return an actual trade would have earned, and that share is
# small: most of what the naive series moves on is not the trade.
#
# The two are computed over the same days, on the same instrument, for the same holding period.
# The only difference between them is that one follows the contract it entered and the other
# follows whatever the selection rule picked up along the way. A label built from the second is
# mostly a record of the selection rule.

# %%
# Scatter plot: naive vs same-contract
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=comparison["naive_ret"].to_list(),
        y=comparison["same_contract_ret"].to_list(),
        mode="markers",
        marker=dict(size=3, color=COLORS["blue"], opacity=0.4),
        name="Returns",
    )
)
fig.add_trace(
    go.Scatter(
        x=[-0.5, 0.5],
        y=[-0.5, 0.5],
        mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"]),
        name="45° line",
    )
)
fig.update_layout(
    title=f"{DEMO_SYMBOL}: same-contract against naive chained returns",
    xaxis_title="Naive (chained constant-maturity)",
    yaxis_title="Same-contract holding return",
    width=600,
    height=500,
)
show_plotly_with_alt(
    fig,
    "A scatter of the same-contract holding return against the naive chained return over the same days, with a dashed forty-five degree line. The cloud is broad and only loosely oriented along that line.",
)

# %% [markdown]
# ## 4. Building a Continuous Series for Backtesting
#
# For backtesting we need a price series whose daily returns are actual P&L, with no phantom
# jumps at rolls. Futures roll quarterly and suit Panama adjustment; straddles reselect on most
# days, so the adjustment has to be return-based:
#
# 1. On a non-roll day, the daily return is the held contract's own move. Uncontroversial.
# 2. On a roll day, the naive-looking answer is to set the return to zero, on the grounds that a
#    roll is a transaction rather than a return.
# 3. Reconstruct prices from whichever returns you chose: $P_{adj,t} = P_0 \prod (1 + r_i)$.
#
# **Step 2 is wrong, and the section measures how wrong.** On a roll day you did hold a
# position: yesterday's contract, from yesterday's close to today's. It moved with the market,
# and that move is real P&L. Zeroing it does not remove the phantom jump alone - it removes the
# day's genuine return along with it.
#
# The correct return for a roll day is the *held* contract's return: yesterday's strike and
# expiration, priced yesterday and today. Section 3 already built the raw-chain lookup that
# makes this available, so it costs nothing but the lookup.
#
# Both are computed below and the difference is reported. Roll costs, the bid-ask actually paid
# to switch, are a separate matter and are modelled as transaction costs in Chapter 18.


# %%
def held_contract_returns(cm_series: pl.DataFrame, raw_options: pl.DataFrame) -> pl.DataFrame:
    """Return each day's move of the contract actually held coming into that day.

    On a non-roll day that is the same contract the series already shows. On a roll day it is
    yesterday's strike and expiration, repriced from the raw chain at today's close.
    """
    previous = cm_series.select(
        "date",
        pl.col("date").shift(1).alias("prev_date"),
        pl.col("strike").shift(1).alias("prev_strike"),
        pl.col("expiration").shift(1).alias("prev_expiration"),
        pl.col("instr_mid").shift(1).alias("prev_mid"),
    ).drop_nulls("prev_date")

    calls = raw_options.filter(pl.col("call_put") == "C").select(
        "date", "strike", "expiration", pl.col("mid_price").alias("call_now")
    )
    puts = raw_options.filter(pl.col("call_put") == "P").select(
        "date", "strike", "expiration", pl.col("mid_price").alias("put_now")
    )

    return (
        previous.join(
            calls,
            left_on=["date", "prev_strike", "prev_expiration"],
            right_on=["date", "strike", "expiration"],
            how="left",
        )
        .join(
            puts,
            left_on=["date", "prev_strike", "prev_expiration"],
            right_on=["date", "strike", "expiration"],
            how="left",
        )
        .with_columns(
            ((pl.col("call_now") + pl.col("put_now")) / pl.col("prev_mid") - 1).alias("held_ret")
        )
        .select("date", "held_ret")
    )


def build_continuous_straddle_series(cm_series: pl.DataFrame, held: pl.DataFrame) -> pl.DataFrame:
    """Rebuild a straddle price series two ways, so the roll-day choice can be compared.

    ``zeroed_daily_ret`` sets roll days to zero. ``held_daily_ret`` uses the return of the
    contract that was actually held into the roll. They agree on every non-roll day.
    """
    df = cm_series.join(held, on="date", how="left").with_columns(
        (pl.col("instr_mid") / pl.col("instr_mid").shift(1) - 1).alias("raw_daily_ret"),
    )

    df = df.with_columns(
        pl.when(pl.col("is_roll"))
        .then(0.0)
        .otherwise(pl.col("raw_daily_ret"))
        .alias("zeroed_daily_ret"),
        pl.when(pl.col("is_roll"))
        .then(pl.col("held_ret"))
        .otherwise(pl.col("raw_daily_ret"))
        .alias("held_daily_ret"),
    )

    start_price = cm_series["instr_mid"][0]
    for source, target in [
        ("zeroed_daily_ret", "price_zeroed"),
        ("held_daily_ret", "price_held"),
    ]:
        prices = [start_price]
        for r in df[source].fill_null(0.0).to_list()[1:]:
            prices.append(prices[-1] * (1.0 + r))
        df = df.with_columns(pl.Series(target, prices))

    return df


# %%
held = held_contract_returns(cm_straddles, raw)
adjusted = build_continuous_straddle_series(cm_straddles, held)

adj_roll = adjusted.filter(pl.col("is_roll"))
_found = adj_roll["held_ret"].drop_nulls().len()
print(f"Roll days: {adj_roll.height} of {adjusted.height}")
print(f"  held contract found in the raw chain on {_found} of them")
print(
    f"  phantom return the raw series shows on those days: {adj_roll['raw_daily_ret'].mean():+.4f}"
)
print(f"  return the held contract actually earned:          {adj_roll['held_ret'].mean():+.4f}")

# %% [markdown]
# The held contract is recoverable on every roll day, so there is no data reason to discard the
# day. And the return being discarded is not noise around zero: on average the position held
# into a roll made money, so zeroing those days does not remove a bias, it introduces one.
#
# This is also where the question Section 2 left open can be answered. Splitting the roll days
# by what changed showed strike-only days averaging close to the days where nothing changed,
# which says the two groups have similar means and nothing more. The construction error is a
# per-day quantity - what the series recorded minus what the contract held would have earned -
# and it can be large on individual days while averaging to little.

# %%
_error_by_kind = (
    adjusted.filter(pl.col("is_roll"))
    .drop_nulls("held_ret")
    .with_columns((pl.col("raw_daily_ret") - pl.col("held_ret")).alias("construction_error"))
    .group_by("change_kind")
    .agg(
        pl.len().alias("days"),
        pl.col("construction_error").mean().alias("mean_error"),
        pl.col("construction_error").abs().mean().alias("mean_abs_error"),
        pl.col("construction_error").abs().max().alias("worst_abs_error"),
    )
    .sort("mean_abs_error", descending=True)
)
print("Per-day construction error, raw series minus the contract actually held:")
print(_error_by_kind)

# %% [markdown]
# Strike-only days are not clean. Their mean error is several percentage points, and the error
# is defined as the raw series minus the held contract, so the sign says which way each group
# is wrong on average: expiration days come out positive, meaning the raw series records more
# than the position earned, and strike-only days come out negative, meaning it records less.
# The two biases point in opposite directions and neither is small.
#
# The mean absolute error exceeds the absolute mean in both groups, which says the individual
# days are not all wrong in their group's direction - the average is a net of errors both
# ways, and the worst single day in each group is tens of percentage points. A bias that
# happens to be modest in the average can still be large on the day a label is taken.
#
# So the two group means in Section 2 agreed for a reason that has nothing to do with the
# series being right on those days. A day on which the strike moved records a return that
# belongs partly to a contract nobody held, and the resemblance between that group's average
# and the do-nothing group's average is a coincidence of aggregation. Comparing averages of a
# quantity cannot establish that the quantity is correct case by case; only the case-by-case
# comparison can, and here it fails.

# %%
_zeroed_total = (1 + adjusted["zeroed_daily_ret"].fill_null(0.0)).product() - 1
_held_total = (1 + adjusted["held_daily_ret"].fill_null(0.0)).product() - 1
_short_held_total = (1 - adjusted["held_daily_ret"].fill_null(0.0)).product() - 1
print(f"Cumulative return of holding the straddle long over {DEMO_YEAR}:")
print(f"  zeroing roll-day returns:         {_zeroed_total:+.1%}")
print(f"  using the held contract's return: {_held_total:+.1%}")
print(f"  the two series differ by:         {_held_total - _zeroed_total:+.1%}")
print(f"Same contracts, same days, held over {DEMO_YEAR}, sold rather than bought:")
print(f"  short series, notional reset against equity each day: {_short_held_total:+.1%}")
print(f"  what negating the long cumulative return would say:  {-_held_total:+.1%}")

# %% [markdown]
# The short figure is constructed here, by negating each day's held-contract return and
# compounding the result over the same year. That is the P&L of a position whose notional is
# reset against equity every day, and it has to be named because the convention decides the
# answer.
#
# Under *this* convention the short return is not the negation of the long one, and the two
# printed numbers show how far from it: both lose money over the year, so a sign flip gets
# the direction wrong and not merely the size. Both series reset exposure against their own
# equity each day, and two paths that rebalance separately do not stay mirror images.
#
# A different convention gives a different answer, and one of them gives exactly the
# negation: holding the opposite quantity of the same contracts every day, never
# rebalancing, produces the opposite dollar P&L before costs, which on the same initial
# capital is the long return negated. The lesson is not that the negation is wrong in
# general. It is that a cumulative short return is undefined until the sizing is stated, and
# negating a compounded long return silently assumes one particular sizing that the series
# above does not use.
#
# The two roll-convention series part company by tens of percentage points of cumulative
# return over a single year, from the same contracts, the same days and the same prices. Only the roll-day
# convention separates them. On a series whose stated purpose is to be backtested, that is the
# difference between two materially different answers about the same strategy.
#
# Zeroing is the more conservative-looking choice and it is not the safer one. It deletes a
# real return on most of the sample, because the contract changes on the majority of days, and
# the deleted returns do not average to zero - so it is a one-sided subtraction repeated on the
# majority of observations, not noise that cancels. Here it makes the instrument look worse
# than it was.
#
# The general form of the error is worth naming, because it is not specific to straddles. A
# roll is two facts at once - the instrument changed, and the market moved - and an adjustment
# that suppresses the day suppresses both. Futures get away with the crude version because
# quarterly rolls make it four days a year. Here it is most of them.

# %%
# Visualize raw vs adjusted price series
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Constant-maturity straddle mid",
        "Reconstructed continuous price, two roll conventions",
    ],
    vertical_spacing=0.08,
)

fig.add_trace(
    go.Scatter(
        x=adjusted["date"].to_list(),
        y=adjusted["instr_mid"].to_list(),
        mode="lines",
        name="Raw",
        line=dict(color=COLORS["negative"], width=1.5),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=adjusted["date"].to_list(),
        y=adjusted["price_zeroed"].to_list(),
        mode="lines",
        name="Roll days zeroed",
        line=dict(color=COLORS["amber"], width=1.5, dash="dot"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=adjusted["date"].to_list(),
        y=adjusted["price_held"].to_list(),
        mode="lines",
        name="Held contract's return",
        line=dict(color=COLORS["blue"], width=1.5),
    ),
    row=2,
    col=1,
)

fig.update_yaxes(title_text="Straddle Mid ($)", row=1, col=1)
fig.update_yaxes(title_text="Reconstructed price ($)", row=2, col=1)
fig.update_layout(
    height=560,
    title=f"{DEMO_SYMBOL} straddle: selected price and two reconstructions",
    legend=dict(orientation="h", yanchor="bottom", y=-0.16, x=0),
)
show_plotly_with_alt(
    fig,
    "Two stacked panels sharing a date axis. The upper panel shows the selected straddle's mid price with its repeated sawtooth. The lower panel shows two reconstructed price series that start together and separate steadily through the year, the roll-zeroed one falling further than the one built from the held contract's returns, with the gap widening throughout.",
)

# %% [markdown]
# Both reconstructions remove the sawtooth, which is what the adjustment was for, and they
# separate from each other steadily over the year because they disagree on most days. The
# zeroed series is the lower of the two and its gap widens monotonically: every roll day
# contributes another deleted return.
#
# The held-contract series is the one to backtest. Roll costs, meaning the bid-ask actually
# crossed to switch contracts, are a separate deduction and belong with transaction costs in
# Chapter 18 rather than inside the price series.
#
# For **labels**, neither reconstruction is the right tool. A label is the return of a trade
# someone could have placed and held, which is the same-contract calculation in Section 3.

# %% [markdown]
# ## 5. Summary: Three Approaches Compared
#
# | Approach | Mechanism | Suitable For |
# |----------|-----------|--------------|
# | Naive (chained) | `shift(-h)` on the constant-maturity series | Nothing. It contaminates labels |
# | Same-contract | Look up the entry contract h days later | Labels, being per-trade P&L |
# | Continuous, held-contract | Roll days take the held contract's return | Backtesting mark-to-market |
# | Continuous, roll-zeroed | Roll days take zero | Nothing. It deletes real P&L |
#
# Three of the four need the raw option chain; only roll-zeroing avoids it, and that is the
# whole of its appeal. On a series that reselects quarterly the shortcut is defensible. On one
# that reselects most days it costs the majority of the sample's return, which the section
# above measures rather than argues.

# %% [markdown]
# The final table puts the forward return from each construction side by side over the same
# days.

# %%
adjusted = adjusted.with_columns(
    pl.col("instr_mid").shift(-HOLDING_PERIOD).alias("raw_exit"),
).with_columns(
    ((pl.col("raw_exit") - pl.col("instr_mid")) / pl.col("instr_mid")).alias("naive_fwd_ret"),
)

# Continuous-adjusted
adjusted = adjusted.with_columns(
    pl.col("price_held").shift(-HOLDING_PERIOD).alias("adj_exit"),
).with_columns(
    ((pl.col("adj_exit") - pl.col("price_held")) / pl.col("price_held")).alias("cont_fwd_ret"),
)

# Merge with same-contract returns
comparison_all = (
    adjusted.select([pl.col("date").alias("entry_date"), "naive_fwd_ret", "cont_fwd_ret"])
    .join(
        same_contract.select(["entry_date", "same_contract_ret"]),
        on="entry_date",
        how="left",
    )
    .drop_nulls()
)

print(f"Paired observations: {len(comparison_all):,}")

three_way = pl.DataFrame(
    {
        "metric": ["mean", "std", "corr_with_same_contract"],
        "naive": [
            comparison_all["naive_fwd_ret"].mean(),
            comparison_all["naive_fwd_ret"].std(),
            comparison_all.select(pl.corr("naive_fwd_ret", "same_contract_ret")).item(),
        ],
        "continuous": [
            comparison_all["cont_fwd_ret"].mean(),
            comparison_all["cont_fwd_ret"].std(),
            comparison_all.select(pl.corr("cont_fwd_ret", "same_contract_ret")).item(),
        ],
        "same_contract": [
            comparison_all["same_contract_ret"].mean(),
            comparison_all["same_contract_ret"].std(),
            1.0,
        ],
    }
)
three_way

# %% [markdown]
# ## Key Takeaways
#
# 1. **A constant-maturity series is not an instrument.** The rule picks whichever straddle
#    sits closest to the target each day, independently, so the contract identity changes on
#    most days of the year. Each change resets time value and prints a price step that no position
#    experienced.
#
# 2. **The roll contamination is systematic and it is large.** Roll days and non-roll days have
#    means of opposite sign, and both are printed above rather than quoted here. The futures
#    analogue is roll yield, but a quarterly roll contributes four such days a year and this
#    series contributes most of them, which turns a correction into the bulk of the signal.
#    Splitting the roll days by what changed is worth doing, and it is worth not over-reading:
#    the strike-only group's mean sits beside the do-nothing group's, yet its per-day
#    construction error against the held contract averages several percentage points, in the
#    opposite direction from the error on expiration days. Agreement between two averages says
#    nothing about whether the individual values are right.
#
# 3. **A short position's cumulative return is not the long's negation.** Per period and for a
#    fixed quantity it is, which is why the notebook prints one convention and states the other.
#    Compounding breaks the symmetry: over the demo year the long series and the daily-reset
#    short series are both negative, so reading the seller's result off a sign flip gets the
#    direction wrong, not merely the magnitude.
#
# 4. **The construction error is a per-day quantity.** What the series records minus what the
#    contract actually held earned is the thing that reaches a label or a backtest, and it is
#    measured day by day in Section 5 rather than inferred from group averages.
#
# 5. **Same-contract returns are the correct label.** Pricing the entry contract in the raw
#    chain h days later gives the P&L of a trade someone could have placed. The correlation
#    between that and the naive chained return is low enough that a label built from the naive
#    series is mostly a record of the selection rule.
#
# 6. **Zeroing roll-day returns deletes real P&L.** On a roll day you held yesterday's
#    contract, and it moved; that move is a return, not a transaction. The held contract is
#    recoverable from the raw chain on every roll day here, and over a single year the two
#    reconstructions differ by tens of percentage points of cumulative return from identical
#    prices. The discarded returns do not average to zero, so the error is one-sided and it
#    compounds over the majority of the sample. The continuous series should take the held
#    contract's return.
#
# 7. **Roll costs are transaction costs, not returns.** The bid-ask actually crossed to switch
#    contracts is execution cost, accounted in Chapter 18 rather than inside the price series.
#    That is what a roll-day adjustment is for; it is not a licence to delete the day.
#
# ## Next
#
# The S&P 500 options case study
# ([`02_labels`](../case_studies/sp500_options/02_labels.ipynb)) builds its labels with the
# same-contract construction from Section 3, priced from the raw chain, across the full
# universe and a multi-year sample. It is a label pipeline rather than a backtest, so it has no
# continuous mark-to-market series and does not need one.
