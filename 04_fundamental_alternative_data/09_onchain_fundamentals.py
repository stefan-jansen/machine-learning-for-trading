# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # On-Chain Fundamentals: DeFi TVL as Alternative Data
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.4 (Understanding Alternative Data)
#
# ## Purpose
#
# A public blockchain records every transaction, so a quantity that would be a trade secret in
# any other market is simply readable: how much capital is deposited in each lending pool,
# exchange and vault. Summed across the protocols on a chain, that is **total value locked**,
# and it is the closest thing decentralized finance has to a fundamental.
#
# It is also a good specimen for the question this section of the chapter is about, which is not
# "what does this dataset measure" but "is it worth integrating". This notebook takes the
# obvious hypothesis - that capital flowing into DeFi precedes a rising ether price - and tries
# to measure it. Most of the work turns out to be establishing how little the available data can
# say, which is the usual outcome of an honest alternative-data evaluation and the reason the
# evaluation happens before the integration.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Define total value locked and say what it does and does not measure.
# - Load a chain-level TVL history and a matched price series, and identify which of the two
#   bounds the window you can study.
# - Show a composition breakdown against the true total rather than against the subset you
#   selected.
# - Turn a level into momentum and regime features, and state the window each is measured over.
# - Test whether a signal predicts a forward return, and correct the test for the overlap that
#   forward returns create.
# - Count the independent observations behind a result, and decide from that count whether the
#   result can be acted on.
#
# ## Prerequisites
#
# Both feeds are free and are cached locally by one downloader, so the notebook does no
# network access:
#
# ```bash
# python data/crypto/onchain/download.py --dataset defillama
# python data/crypto/onchain/download.py --dataset coingecko
# ```
#
# ## Cross-References
#
# - **Related**: [`07_macro_data_alignment`](07_macro_data_alignment.ipynb) (the same publication-timing discipline on macro series)
# - **Downstream**: [`11_defi_tvl_evaluation`](11_defi_tvl_evaluation.ipynb) (the full due-diligence framework applied to this dataset)
#
# ## Key Concepts
#
# - **Total value locked (TVL)**: the dollar value of the crypto assets deposited in a chain's
#   decentralized finance protocols, valued at current prices.
# - **Chain TVL**: that figure for one blockchain. **Protocol TVL** is the same for one
#   application.
# - **Forward return**: the return realized over a stated window *after* the date a signal is
#   observed, which is the quantity a signal has to predict to be worth anything.
# - **Overlapping windows**: consecutive forward returns computed over a window longer than the
#   sampling interval share most of their days, so consecutive observations are not independent
#   draws and a test that assumes they are overstates its own significance.

# %%
"""On-Chain Fundamentals: DeFi TVL as Alternative Data - source and analyze DeFi TVL for crypto trading signals."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm
from plotly.subplots import make_subplots

from data import load_coingecko_ohlcv, load_defillama_chain_tvl
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# The forward horizon is the setting that decides what is being tested. Thirty days asks whether
# TVL predicts a month of ether returns, which is the horizon the conventional story is told at;
# it is also what creates the overlap Part 6 has to correct for, since the series is daily.

# %% tags=["parameters"]
CHAINS = ["Ethereum", "Solana", "BSC", "Arbitrum"]  # the four largest by TVL
FORWARD_DAYS = 30  # the return horizon the signal is tested against
MOMENTUM_DAYS = 30  # the window TVL growth is measured over
ZSCORE_DAYS = 90  # the window a TVL level is judged unusual against
REGIME_Z = 1.0  # standard deviations from the mean that separate the three regimes
RECENT_DAYS = 30  # the trailing window the composition breakdown averages over

# %% [markdown]
# ## 1. What total value locked measures
#
# When someone deposits ether into a lending protocol, the deposit sits in a smart contract
# whose balance anyone can read. TVL is the sum of those balances across a chain's protocols,
# converted to dollars at current prices.
#
# Two properties follow from that definition and both matter for how the number can be read.
# It is a **stock**, not a flow: it does not distinguish new capital arriving from existing
# deposits appreciating, so a chain whose TVL doubled while its native token doubled has
# attracted nothing. And it is **denominated in dollars while being held in crypto**, so it
# falls when prices fall whether or not anyone withdrew.
#
# The conventional readings below are the ones a data vendor's pitch deck offers. They are
# hypotheses, and Part 6 tests one of them.
#
# | Pattern | Conventional reading |
# |---------|----------------------|
# | TVL growing faster than the ether price | Capital arriving rather than deposits appreciating |
# | TVL falling while the price holds | Capital leaving; risk appetite falling |
# | A TVL spike | A new protocol or a yield opportunity pulling deposits in |
# | A TVL collapse | An exploit, a cascade of liquidations, or a general panic |

# %% [markdown]
# ## 2. The TVL history
#
# DeFi Llama publishes the series free and it goes back to the start of the sector. Seeing its
# full length first matters, because the joined panel later in this notebook is a small fraction
# of it and the reason is worth knowing before the results are read.

# %%
total_tvl = (
    load_defillama_chain_tvl("total").sort("timestamp").with_columns(tvl_bn=pl.col("tvl_usd") / 1e9)
)
print(f"Observations: {len(total_tvl):,}")
print(f"History: {total_tvl['timestamp'].min()} to {total_tvl['timestamp'].max()}")
print(f"Highest level reached: ${total_tvl['tvl_bn'].max():.0f}bn")
total_tvl.tail(3)

# %%
fig = px.line(
    total_tvl.to_pandas(),
    x="timestamp",
    y="tvl_bn",
    title="DeFi's capital base grew, collapsed, and rebuilt over eight years",
    labels={"timestamp": "Date", "tvl_bn": "Total value locked (USD billions)"},
    color_discrete_sequence=[COLORS["blue"]],
)
fig.update_layout(height=380)
show_plotly_with_alt(
    fig,
    "Line chart of total value locked across all DeFi from 2017 to 2026, near zero until 2020, "
    "rising steeply to a peak at the end of 2021, falling by about four fifths into late 2023, "
    "and recovering to roughly half the peak since.",
)

# %% [markdown]
# ### Which chains hold it
#
# Chain-level series let the total be decomposed. The four loaded here are the largest, and the
# breakdown below measures them against the **total** rather than against each other, so that
# the share held by everything else is visible rather than assumed away.

# %%
chain_tvl = {}
for chain in CHAINS:
    series = load_defillama_chain_tvl(chain).sort("timestamp")
    chain_tvl[chain] = series
    print(f"{chain}: {len(series):,} observations from {series['timestamp'].min()}")

# %%
recent_total = float(total_tvl.tail(RECENT_DAYS)["tvl_bn"].mean())
composition = pl.DataFrame(
    [
        {
            "chain": chain,
            "tvl_bn": float(series.tail(RECENT_DAYS)["tvl_usd"].mean()) / 1e9,
        }
        for chain, series in chain_tvl.items()
    ]
).sort("tvl_bn", descending=True)
composition = pl.concat(
    [
        composition,
        pl.DataFrame(
            {"chain": ["All other chains"], "tvl_bn": [recent_total - composition["tvl_bn"].sum()]}
        ),
    ]
).with_columns(share=pl.col("tvl_bn") / recent_total)
composition

# %%
fig = px.bar(
    composition.to_pandas(),
    x="share",
    y="chain",
    orientation="h",
    title="One chain holds more than all the others together",
    labels={"share": f"Share of total value locked, {RECENT_DAYS}-day average", "chain": ""},
    color_discrete_sequence=[COLORS["blue"]],
)
fig.update_layout(height=320, xaxis_tickformat=".0%", yaxis=dict(categoryorder="total ascending"))
show_plotly_with_alt(
    fig,
    "Horizontal bar chart of each chain's share of total value locked over the trailing thirty "
    "days. The largest bar is longer than the other four combined, and the residual bar for all "
    "remaining chains is the second longest.",
)

# %% [markdown]
# ## 3. The price series, and what bounds the study
#
# Testing whether TVL predicts returns needs a price. CoinGecko's free tier serves the trailing
# 365 days and no more, so the joined panel is one year long however far back the TVL series
# reaches. That is not a detail: it is the constraint that decides what this notebook can
# conclude, and Part 6 comes back to it.
#
# The free tier also appends a live intraday snapshot on top of the current day's midnight bar,
# so the last calendar day can arrive twice. Collapsing to the most recent row per day is done
# before anything joins to it.

# %%
eth = load_coingecko_ohlcv("ethereum").unique(subset="timestamp", keep="last", maintain_order=True)
print(f"Price observations: {len(eth):,}")
print(f"Window: {eth['timestamp'].min()} to {eth['timestamp'].max()}")
print(
    f"TVL history that window discards: {(eth['timestamp'].min() - total_tvl['timestamp'].min()).days:,} days"
)

# %%
panel = (
    total_tvl.join(
        eth.rename({"price_usd": "eth_price", "volume_usd": "eth_volume"}),
        on="timestamp",
        how="inner",
    )
    .sort("timestamp")
    .select("timestamp", "tvl_bn", "eth_price", "eth_volume")
)
print(
    f"Joined panel: {len(panel):,} rows, {panel['timestamp'].min()} to {panel['timestamp'].max()}"
)
panel.tail(3)

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Total value locked", "Ether price"),
)
fig.add_trace(
    go.Scatter(
        x=panel["timestamp"],
        y=panel["tvl_bn"],
        line=dict(color=COLORS["blue"]),
        fill="tozeroy",
        fillcolor="rgba(10, 22, 40, 0.15)",  # translucent COLORS["blue"]
        name="TVL",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=panel["timestamp"], y=panel["eth_price"], line=dict(color=COLORS["amber"]), name="ETH"
    ),
    row=2,
    col=1,
)
fig.update_yaxes(title_text="USD billions", row=1, col=1)
fig.update_yaxes(title_text="USD", row=2, col=1)
fig.update_layout(height=560, showlegend=False, title="TVL and the ether price move together")
show_plotly_with_alt(
    fig,
    "Two stacked panels over the joined one-year window: total value locked and the ether price. "
    "The two lines rise and fall at the same times, which is what a dollar-denominated stock of "
    "crypto assets does.",
)

# %% [markdown]
# The two lines moving together is the first thing to be careful about. TVL is a dollar value of
# crypto holdings, so it mechanically follows the price of those holdings. Any test of whether
# TVL predicts the price has to work with a quantity that is not simply the price again, which
# is why the features below are growth rates and z-scores rather than levels.

# %% [markdown]
# ## 4. Features
#
# Three quantities, each with the window it is measured over stated in its name. Growth over the
# momentum window; the level as a z-score against a longer window, which is what makes "high" or
# "low" mean something; and the regime label that z-score falls into.

# %%
features = panel.with_columns(
    tvl_growth=pl.col("tvl_bn").pct_change(MOMENTUM_DAYS),
    eth_return=pl.col("eth_price").pct_change(MOMENTUM_DAYS),
    tvl_zscore=(pl.col("tvl_bn") - pl.col("tvl_bn").rolling_mean(ZSCORE_DAYS))
    / pl.col("tvl_bn").rolling_std(ZSCORE_DAYS),
).with_columns(
    # A row without a full z-score window is unclassified rather than neutral: pooling the
    # warm-up into the middle band would put a hundred days of "no measurement" into a bucket
    # the analysis then reads as a measurement.
    tvl_regime=pl.when(pl.col("tvl_zscore").is_null())
    .then(pl.lit(None, dtype=pl.String))
    .when(pl.col("tvl_zscore") > REGIME_Z)
    .then(pl.lit("expansion"))
    .when(pl.col("tvl_zscore") < -REGIME_Z)
    .then(pl.lit("contraction"))
    .otherwise(pl.lit("neutral"))
)
features.select("timestamp", "tvl_bn", "tvl_growth", "tvl_zscore", "tvl_regime").tail(5)

# %% [markdown]
# ## 5. The forward return
#
# The quantity a signal has to predict is the return *after* it is observed. Shifting the
# trailing return back by its own window turns it into the return over the days that follow each
# date, which is the only construction that puts the signal before the outcome.

# %%
tested = features.with_columns(forward_return=pl.col("eth_return").shift(-FORWARD_DAYS)).drop_nulls(
    ["tvl_growth", "forward_return"]
)

print(f"Rows with both a signal and a forward return: {len(tested):,}")
print(f"Signal dates: {tested['timestamp'].min()} to {tested['timestamp'].max()}")

# %% [markdown]
# ## 6. Testing the hypothesis, and counting the evidence
#
# The regime table is the obvious first cut: average the forward return within each regime and
# compare. It is also where an alternative-data evaluation most often goes wrong, because the
# table looks like evidence and is not yet.

# %%
by_regime = (
    tested.drop_nulls("tvl_regime")
    .group_by("tvl_regime")
    .agg(
        pl.len().alias("days"),
        pl.col("forward_return").mean().alias("mean_forward_return"),
        pl.col("forward_return").std().alias("std_forward_return"),
    )
    .with_columns(
        # Consecutive forward returns share all but one of their days, so the number of
        # independent windows a bucket holds is its day count divided by the horizon.
        independent_windows=(pl.col("days") / FORWARD_DAYS).round(1)
    )
    .sort("mean_forward_return", descending=True)
)
by_regime

# %%
fig = px.bar(
    by_regime.to_pandas(),
    x="tvl_regime",
    y="mean_forward_return",
    error_y=by_regime.with_columns(
        se=pl.col("std_forward_return") / (pl.col("independent_windows") ** 0.5)
    )["se"].to_list(),
    title="The error bars are wider than the differences between the regimes",
    labels={
        "tvl_regime": f"TVL regime, {ZSCORE_DAYS}-day z-score",
        "mean_forward_return": f"Mean {FORWARD_DAYS}-day forward return",
    },
    color_discrete_sequence=[COLORS["blue"]],
)
fig.update_layout(height=400, yaxis_tickformat=".0%")
show_plotly_with_alt(
    fig,
    "Bar chart of the mean forward ether return in each of the three TVL regimes, with error "
    "bars scaled by the number of independent windows in each bucket. Every error bar spans "
    "zero and overlaps the other bars.",
)

# %% [markdown]
# The error bars are what the table alone does not show. Each bucket holds a few hundred daily
# rows and a handful of independent thirty-day windows, so the standard error of each mean is
# larger than the distance between the means. Nothing in the ordering of those three bars can be
# distinguished from the ordering three coin flips would produce.
#
# ### The same question as a regression
#
# The linear version is a regression of the forward return on TVL growth. Its uncorrected
# t-statistic assumes each daily observation is an independent draw, which the overlap makes
# false; the Newey-West correction widens the standard error by the amount of serial dependence
# actually present, and the lag is set one short of the horizon because that is how far the
# overlap reaches.

# %%
signal = tested["tvl_growth"].to_numpy()
outcome = tested["forward_return"].to_numpy()
design = sm.add_constant(signal)

naive = sm.OLS(outcome, design).fit()
corrected = sm.OLS(outcome, design).fit(cov_type="HAC", cov_kwds={"maxlags": FORWARD_DAYS - 1})

print(
    f"Correlation between TVL growth and the forward return: {tested.select(pl.corr('tvl_growth', 'forward_return')).item():+.3f}"
)
print(f"Slope: {naive.params[1]:+.3f}")
print(f"t-statistic assuming independent days: {naive.tvalues[1]:+.2f}")
print(f"t-statistic with the overlap corrected: {corrected.tvalues[1]:+.2f}")
print(f"Independent thirty-day windows in the sample: {len(tested) / FORWARD_DAYS:.0f}")

# %% [markdown]
# Both statistics are small, and the correction moves the smaller one toward zero. The reason is
# in the last line: a year of daily observations of a thirty-day forward return is about ten
# independent windows, and ten observations cannot establish a relationship of this size whatever
# the daily row count suggests.
#
# **What binds is the price feed, not the TVL series.** DeFi Llama publishes eight years of TVL
# for free; the free price tier serves one. Extending the study needs a longer price history,
# which the exchange feeds in Chapter 2 provide, and that is the change that would make this
# question answerable rather than any refinement of the signal.

# %% [markdown]
# ## Key Takeaways
#
# 1. Total value locked is a dollar-denominated stock of crypto assets, so it moves with the
#    prices of those assets by construction. A test of whether it predicts price has to be built
#    on growth or on a standardized level, never on the level itself.
# 2. Show a composition against the true total. Four chains plotted against each other will
#    always fill the chart, whatever share of the market they actually hold.
# 3. A forward return sampled daily over a thirty-day horizon gives thirty overlapping
#    observations of each window. The row count is not the sample size, and the number that
#    matters is the row count divided by the horizon.
# 4. Correct the test for that overlap before reading it. The Newey-West standard error is the
#    standard correction and it needs a lag at least as long as the overlap; here it moves an
#    already-weak statistic closer to zero.
# 5. A regime table with three buckets and a few independent windows in each will always produce
#    an ordering. Put the standard error on the chart and the ordering usually stops being
#    interesting.
# 6. The binding constraint on an alternative-data study is often not the alternative data. Here
#    the free TVL history is eight years and the free price history is one, so the price feed
#    decides what can be concluded.
#
# **Next**: [`11_defi_tvl_evaluation`](11_defi_tvl_evaluation.ipynb) applies the chapter's full
# due-diligence framework - signal, quality, legal risk and cost - to this same dataset.
