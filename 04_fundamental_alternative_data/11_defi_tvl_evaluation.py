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
# # Alternative Data Evaluation: DeFi TVL Case Study
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.4 (Understanding Alternative Data)
#
# ## Purpose
#
# A data vendor's pitch is always the same shape: here is a series nobody else has, and here is a
# backtest in which it works. Deciding whether to buy it, or in this case whether to spend two
# weeks integrating a free one, is a different exercise, and the previous notebook did one part of
# it. This one runs the whole thing.
#
# Four questions have to be answered before a dataset enters a research pipeline, and they are not
# interchangeable. Does it predict anything? Is the data itself sound? Is using it legal, and is
# any of it material non-public information? And does the value justify the cost of carrying it?
# The first and last are measured; the second is measured and then judged; the third is a
# **hard gate**, one that blocks integration whatever the other three say.
#
# The dataset under evaluation is the DeFi Llama total value locked series, which is free, which
# means the commercial question is about engineering time rather than a licence fee, and which
# makes the exercise cleaner: nothing here is being justified by its price.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - Lay out an alternative-data evaluation as four questions, and say which of them can block on
#   their own.
# - Measure a signal's relationship to forward returns across several definitions and horizons,
#   with the overlap those horizons create priced into every statistic.
# - Recognize what selecting the strongest of many measured relationships does to its
#   significance, and report the count.
# - Measure the stability of a relationship over time rather than reporting one number for the
#   whole sample.
# - Audit a series for gaps, nulls and implausible moves, and separate a data defect from an
#   early-history artifact.
# - Explain why the absence of historical vintages is a hard gate for a backtest, whatever the
#   rest of the audit says.
# - Compute the gross return a signal must earn to cover the cost of carrying it, at a stated
#   fund size and allocation.
#
# ## Prerequisites
#
# Both feeds are free and cached locally by one downloader:
#
# ```bash
# python data/crypto/onchain/download.py
# ```
#
# ## Cross-References
#
# - **Data source**: [`09_onchain_fundamentals`](09_onchain_fundamentals.ipynb) introduces the TVL series
# - **Related**: [`07_macro_data_alignment`](07_macro_data_alignment.ipynb) measures revisions on a series that does ship vintages
#
# ## The four questions
#
# | Question | What answers it | Can it block on its own? |
# |----------|-----------------|--------------------------|
# | **Signal** | Does it relate to forward returns, and does that relation hold up over time? | No; a weak signal is a reason not to prioritize |
# | **Data** | Coverage, gaps, methodology, and whether history can be reconstructed as it stood | Yes, on the last of those |
# | **Legal** | How the data was obtained, whether it is material non-public information, what the licence permits | Yes |
# | **Commercial** | The cost of carrying it against the capital it would inform | No; it sets the bar the signal has to clear |

# %%
"""Alternative Data Evaluation: DeFi TVL Case Study - measure four evaluation dimensions on a real alt-data feed."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm

from data import load_coingecko_ohlcv, load_defillama_chain_tvl
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# The settings below fall into three groups. The signal definitions and horizons decide how many
# relationships get measured, which is the number the multiple-comparison discussion turns on.
# The audit thresholds decide what counts as an implausible move and where the early history
# stops. The cost assumptions decide the break-even calculation and are the ones most worth
# replacing with a reader's own.

# %% tags=["parameters"]
HORIZONS = [7, 14, 30, 60]  # forward-return horizons in days
ZSCORE_DAYS = 90  # window the TVL level is standardized against
ROLLING_WINDOW = 180  # window the stability of the relationship is measured over
MODERN_ERA_START = "2020-01-01"  # where the audit stops treating the history as a launch curve
EXTREME_DAILY_MOVE = 0.20  # a one-day change larger than this is flagged for inspection
INTEGRATION_HOURS = 40  # one-off engineering to bring the feed into a pipeline
MAINTENANCE_HOURS = 20  # per year, to keep it running
HOURLY_RATE = 150  # fully loaded cost of an engineer-hour, USD
DATA_FEES = 0  # DeFi Llama charges nothing for this series
ALLOCATION_SHARE = 0.10  # fraction of a fund's capital this signal would inform
TARGET_RETURN_ON_COST = 3.0  # the multiple of cost a research budget expects back

# %% [markdown]
# ## 1. The data under evaluation
#
# Two series: the TVL history being evaluated, and the ether price it would be used to trade.

# %%
tvl = (
    load_defillama_chain_tvl("total").sort("timestamp").with_columns(tvl_bn=pl.col("tvl_usd") / 1e9)
)
# CoinGecko's free tier appends a live snapshot on top of the current day's daily bar, so the
# last date can arrive twice; the later row is the one to keep.
eth = (
    load_coingecko_ohlcv("ethereum")
    .select("timestamp", pl.col("price_usd").alias("eth_price"))
    .group_by("timestamp")
    .agg(pl.col("eth_price").last())
    .sort("timestamp")
)
panel = tvl.join(eth, on="timestamp", how="inner").sort("timestamp")

print(f"TVL history: {len(tvl):,} days, {tvl['timestamp'].min()} to {tvl['timestamp'].max()}")
print(f"Price history: {len(eth):,} days, {eth['timestamp'].min()} to {eth['timestamp'].max()}")
print(f"Joined: {len(panel):,} days, {panel['timestamp'].min()} to {panel['timestamp'].max()}")

# %% [markdown]
# ## 2. Signal: does it relate to forward returns?
#
# Three ways of turning the level into a signal, four horizons to test each against. Twelve
# relationships, which is a number to keep in mind rather than a detail: the largest of twelve
# measured correlations is larger than the largest of one, whether or not anything is there.
#
# The forward returns are computed from the price at the two ends of each window. Consecutive
# windows overlap by all but one day, so every statistic below carries a Newey-West standard
# error at a lag one short of its own horizon.

# %%
SIGNALS = {
    "growth_7d": pl.col("tvl_bn").pct_change(7),
    "growth_30d": pl.col("tvl_bn").pct_change(30),
    f"zscore_{ZSCORE_DAYS}d": (pl.col("tvl_bn") - pl.col("tvl_bn").rolling_mean(ZSCORE_DAYS))
    / pl.col("tvl_bn").rolling_std(ZSCORE_DAYS),
}
FORWARD = {f"fwd_{h}d": pl.col("eth_price").shift(-h) / pl.col("eth_price") - 1 for h in HORIZONS}
measured = panel.with_columns(**SIGNALS).with_columns(**FORWARD)
measured.select("timestamp", *SIGNALS, *FORWARD).tail(3)


# %%
def relationship(frame: pl.DataFrame, signal: str, forward: str, horizon: int) -> dict:
    """Correlation of `signal` with `forward`, and a t-statistic that prices in the overlap."""
    pair = frame.select(signal, forward).drop_nulls()
    x, y = pair[signal].to_numpy(), pair[forward].to_numpy()
    fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": horizon - 1})
    return {
        "signal": signal,
        "horizon_days": horizon,
        "observations": len(pair),
        "independent_windows": round(len(pair) / horizon, 1),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "t_statistic": float(fit.tvalues[1]),
    }


relationships = pl.DataFrame(
    [
        relationship(measured, signal, f"fwd_{horizon}d", horizon)
        for signal in SIGNALS
        for horizon in HORIZONS
    ]
)
relationships.sort(pl.col("correlation").abs(), descending=True)

# %% [markdown]
# Reading the correlations and the t-statistics side by side is the point of the table. The
# largest correlation in it is the one a pitch deck would lead with; its t-statistic is the reason
# not to.

# %%
grid = relationships.pivot(on="horizon_days", index="signal", values="t_statistic")
signals_order = list(SIGNALS)
fig = go.Figure(
    go.Heatmap(
        z=[grid.filter(pl.col("signal") == s).drop("signal").row(0) for s in signals_order],
        x=[f"{h} days" for h in HORIZONS],
        y=signals_order,
        colorscale=[[0.0, COLORS["negative"]], [0.5, COLORS["silver"]], [1.0, COLORS["positive"]]],
        zmid=0,
        zmin=-2,
        zmax=2,
        text=[
            [f"{v:.2f}" for v in grid.filter(pl.col("signal") == s).drop("signal").row(0)]
            for s in signals_order
        ],
        texttemplate="%{text}",
        textfont={"size": 14},
        colorbar=dict(title="t"),
    )
)
fig.update_layout(
    title="No signal and horizon pair reaches a t-statistic of two",
    xaxis_title="Forward return horizon",
    yaxis_title="Signal",
    height=340,
)
show_plotly_with_alt(
    fig,
    "Heatmap of overlap-corrected t-statistics for three TVL signals against four forward-return "
    "horizons, on a diverging scale bounded at plus and minus two. Every cell sits well inside "
    "the bounds and the colours are pale throughout.",
)

# %% [markdown]
# The colour scale is fixed at plus and minus two, the conventional threshold, so a cell that
# reached it would saturate. None does. Every relationship in the grid is smaller than its own
# uncertainty, and that is before accounting for having measured twelve of them: the largest
# absolute t-statistic among twelve draws from a null would routinely exceed what any one of these
# reaches.
#
# ### Stability
#
# A correlation over a whole sample is one number and hides how it got there. Rolling a window
# across the sample shows whether the relationship persisted or whether one episode produced it.

# %%
stability_signal, stability_horizon = "growth_30d", 30
rolling = measured.select("timestamp", stability_signal, f"fwd_{stability_horizon}d").drop_nulls()
rolling_correlations = [
    rolling.slice(i - ROLLING_WINDOW, ROLLING_WINDOW)
    .select(pl.corr(stability_signal, f"fwd_{stability_horizon}d"))
    .item()
    for i in range(ROLLING_WINDOW, len(rolling))
]
rolling_ic = pl.DataFrame(
    {
        "timestamp": rolling["timestamp"][ROLLING_WINDOW:],
        "rolling_correlation": rolling_correlations,
    }
).drop_nulls()

print(f"Windows: {len(rolling_ic):,} of {ROLLING_WINDOW} days each")
print(f"Mean: {rolling_ic['rolling_correlation'].mean():+.3f}")
print(
    f"Range: {rolling_ic['rolling_correlation'].min():+.3f} to {rolling_ic['rolling_correlation'].max():+.3f}"
)
print(
    f"Share of windows with a positive correlation: {(rolling_ic['rolling_correlation'] > 0).mean():.0%}"
)

# %%
fig = px.area(
    rolling_ic.to_pandas(),
    x="timestamp",
    y="rolling_correlation",
    title="The relationship changes sign inside the one year of data available",
    labels={
        "timestamp": "Window ending",
        "rolling_correlation": f"Correlation over the trailing {ROLLING_WINDOW} days",
    },
    color_discrete_sequence=[COLORS["slate"]],
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.update_layout(height=380)
show_plotly_with_alt(
    fig,
    "Filled line chart of the rolling correlation between thirty-day TVL growth and the "
    "thirty-day forward ether return. The line crosses zero and spends time on both sides of it.",
)

# %% [markdown]
# The rolling windows overlap each other as heavily as the forward returns do, so the picture is
# a description rather than a test. What it describes is enough: a relationship that changes sign
# within a single year of data has not been shown to exist, and the sample is too short to tell
# an unstable relationship from no relationship at all.
#
# **The signal question closes as unproven rather than as failed**, and the reason is the sample.
# One year of daily observations of a monthly horizon is about ten independent windows. What
# would change the answer is a longer price history, not a cleverer signal.

# %% [markdown]
# ## 3. Data: is the series itself sound?
#
# Four things to establish, and they are cheap: how long the history runs, whether any days are
# missing, whether any values are missing, and whether any single-day moves are too large to be
# real.

# %%
gaps = tvl.with_columns(
    gap_days=(pl.col("timestamp") - pl.col("timestamp").shift(1)).dt.total_days()
)
moves = tvl.with_columns(daily_change=pl.col("tvl_bn").pct_change())
modern = moves.filter(pl.col("timestamp") >= pl.lit(MODERN_ERA_START).str.to_date())

audit = pl.DataFrame(
    {
        "check": [
            "history in years",
            "days absent from the history",
            "missing values",
            f"daily moves above {EXTREME_DAILY_MOVE:.0%}, whole history",
            f"daily moves above {EXTREME_DAILY_MOVE:.0%}, since {MODERN_ERA_START}",
        ],
        "result": [
            f"{(tvl['timestamp'].max() - tvl['timestamp'].min()).days / 365.25:.1f}",
            str(
                int(
                    gaps.select(
                        (pl.col("gap_days") - 1).filter(pl.col("gap_days") > 1).sum()
                    ).item()
                    or 0
                )
            ),
            str(tvl["tvl_usd"].null_count()),
            str(int((moves["daily_change"].abs() > EXTREME_DAILY_MOVE).sum())),
            str(int((modern["daily_change"].abs() > EXTREME_DAILY_MOVE).sum())),
        ],
    }
)
audit

# %% [markdown]
# The two move counts are the same check over two windows, and the difference between them is
# the finding. Almost every implausible daily move is in the years when total value locked was
# measured in millions and a single protocol launching moved it by half.
#
# Filter those years out by **date**. A level threshold looks equivalent and is not: the series
# passes back through any low level on the way down from a peak, so a threshold on the value
# removes days from the middle of later drawdowns. The series then has holes where it had none,
# and a gap audit reports them as a defect in the data rather than in the filter.

# %%
fig = px.scatter(
    moves.drop_nulls("daily_change").to_pandas(),
    x="timestamp",
    y="daily_change",
    title="Implausible daily moves belong to the launch years, not to the series",
    labels={"timestamp": "Date", "daily_change": "One-day change in total value locked"},
    color_discrete_sequence=[COLORS["blue"]],
    opacity=0.5,
)
for level in (EXTREME_DAILY_MOVE, -EXTREME_DAILY_MOVE):
    fig.add_hline(y=level, line_dash="dot", line_color=COLORS["negative"])
fig.add_vline(x=MODERN_ERA_START, line_dash="dash", line_color=COLORS["neutral"])
fig.update_layout(height=400, yaxis_tickformat=".0%")
show_plotly_with_alt(
    fig,
    "Scatter of every one-day change in total value locked against date, with dotted rules at "
    "plus and minus twenty percent and a dashed vertical rule at the start of 2020. Points "
    "outside the rules cluster almost entirely to the left of the vertical.",
)

# %% [markdown]
# ### The hard gate: no vintages
#
# Every check above passes, and one thing that was not checked decides the gate anyway.
#
# DeFi Llama serves the series **as it currently understands it**. When a protocol is added to
# its coverage, that protocol's history is added too; when a price source for a deposited token
# is corrected, every past day using that token is recomputed. The series is therefore restated,
# and the API offers no way to ask what it said on a given past date.
#
# A backtest reading it is reading today's understanding of 2021, including protocols nobody was
# tracking in 2021, and there is no way to measure how large that difference is because the
# earlier version is not retained anywhere. The previous notebook could measure revisions on
# Treasury yields because an archive of vintages exists; here the same question has no answer.
#
# That is what makes it a hard gate rather than a caveat. A signal computed on a restated history
# cannot be shown to have been computable at the time, and no amount of signal strength repairs
# it. The two ways past it are to build a vintage archive going forward by snapshotting the feed
# daily, which starts producing usable history a year later, or to restrict any claim to the
# period since snapshotting began.

# %% [markdown]
# ## 4. Legal: how the data was obtained, and what it is
#
# This gate is qualitative and it is a gate: a single failure blocks regardless of everything
# else. The four inputs are recorded rather than scored.

# %%
legal_review = pl.DataFrame(
    {
        "question": [
            "How was the data obtained?",
            "Could it be material non-public information?",
            "What does the licence permit?",
            "What jurisdictional issues attach?",
        ],
        "finding": [
            "Aggregated from public blockchain state, readable by anyone running a node.",
            "No. On-chain balances are public the moment they are written; the aggregation adds "
            "convenience, not access.",
            "Free access with attribution requested and published rate limits; no redistribution "
            "of the raw feed as a product.",
            "Crypto regulation differs by jurisdiction, and an institution's compliance review "
            "attaches to the smart-contract exposure a strategy would take, not to the data.",
        ],
    }
)
legal_review

# %% [markdown]
# The distinction in the second row is the one that matters and it generalizes past this dataset.
# What makes information material non-public is not that it is hard to get; it is that the person
# supplying it was not entitled to. A satellite image of a car park and a scraped public web page
# are both hard to obtain and neither is non-public. A dataset assembled from a company's own
# systems by someone bound by a duty to that company is non-public however cheaply it arrives.

# %% [markdown]
# ## 5. Commercial: what carrying it costs
#
# Free data is not costless. Someone builds the loader, someone keeps it running when the schema
# changes, and the capital the signal informs could have been informed by something else. The
# question is what gross return the signal has to earn to return a multiple of that cost.


# %%
def break_even_alpha(aum: float) -> dict:
    """Two bars: the return that covers the cost, and the one that returns a multiple of it."""
    annual_cost = DATA_FEES + (INTEGRATION_HOURS + MAINTENANCE_HOURS) * HOURLY_RATE
    capital = aum * ALLOCATION_SHARE
    cost_recovery = annual_cost / capital * 10_000
    return {
        "aum_usd": int(aum),
        "annual_cost_usd": int(annual_cost),
        "capital_informed_usd": int(capital),
        "cost_recovery_bps": round(cost_recovery, 1),
        "target_bps": round(cost_recovery * TARGET_RETURN_ON_COST, 1),
    }


costs = pl.DataFrame([break_even_alpha(aum) for aum in (10e6, 50e6, 500e6, 5e9)])
costs

# %% [markdown]
# The two columns are different bars and both are worth having. **Cost recovery** is the return
# at which the signal pays for itself and nothing more. **Target** is that multiplied by the
# return a research budget expects on what it funds, which is the bar a project has to clear to be
# worth starting rather than merely worth continuing.
#
# The cost is the same whatever the fund's size, so both bars fall with capital. That is the
# general shape of an alternative-data decision and it is why the same dataset is worth buying at
# one firm and not at another: the question is never whether a signal is real, but whether it is
# real and large enough for the capital it would inform.
#
# Here the first year carries the integration hours as well, so a second year is cheaper. What
# the table does not include is the opportunity cost of the research time, which is usually the
# larger number and is not a figure a spreadsheet supplies.

# %% [markdown]
# ## 6. The four answers together
#
# The point of separating the questions is that they combine by rule rather than by arithmetic. A
# weighted score would let a strong signal outvote a failed hard gate, which is exactly the
# mistake the structure exists to prevent.

# %%
verdict = pl.DataFrame(
    {
        "question": ["Signal", "Data", "Legal", "Commercial"],
        "what was measured": [
            f"{len(relationships)} signal-horizon pairs; largest |t| "
            f"{relationships['t_statistic'].abs().max():.2f}; rolling correlation changes sign",
            f"{(tvl['timestamp'].max() - tvl['timestamp'].min()).days / 365.25:.1f} years, no "
            f"missing days, no missing values, {int((modern['daily_change'].abs() > EXTREME_DAILY_MOVE).sum())} "
            f"implausible move since {MODERN_ERA_START}",
            "Public on-chain state; no material non-public information; attribution requested",
            f"${(INTEGRATION_HOURS + MAINTENANCE_HOURS) * HOURLY_RATE:,} a year in engineering, "
            f"no data fees; cost recovery from "
            f"{costs['cost_recovery_bps'].min():.1f} to {costs['cost_recovery_bps'].max():.1f} bps",
        ],
        "outcome": [
            "Unproven on this sample",
            "Blocked: no vintages",
            "Clear",
            "Affordable above mid size",
        ],
        "hard gate": [False, True, True, False],
    }
)
verdict

# %% [markdown]
# The decision follows from the second row alone. The data is clean, the licence is permissive
# and the cost is small, and none of that matters while the history is restated and no archive of
# what it previously said exists. The action that changes the answer is not more analysis: it is
# to start snapshotting the feed daily, and to revisit the signal question in a year when there
# is a point-in-time history and enough of it to test against.

# %% [markdown]
# ## Key Takeaways
#
# 1. Separate the questions and keep the hard gates separate from the scored ones. A composite
#    score lets a strong signal outvote a legal failure or an unreconstructible history, which is
#    the arithmetic that produces the decisions this framework exists to prevent.
# 2. Count the relationships you measured. Twelve signal-horizon pairs produce a largest
#    correlation whether or not anything is there, and reporting the largest without the count is
#    how a screening exercise turns into a finding.
# 3. Price the overlap into every statistic. Forward returns sampled daily share all but one day
#    with their neighbours, and the correction is the same Newey-West covariance in a screening
#    table as in a formal test.
# 4. A relationship that changes sign inside the available sample has not been shown to exist.
#    Report the rolling picture beside the whole-sample number, and read a short sample as
#    unproven rather than as refuted.
# 5. Filter a history by date, not by level. A series passes back through low levels on the way
#    down from a peak, so a level threshold removes days from the middle of later drawdowns and
#    leaves holes that a gap audit then reports as a defect in the data.
# 6. A restated series with no vintage archive cannot support a backtest, however clean it is. The
#    fix is to start snapshotting, which is cheap and only pays off later, and the decision in the
#    meantime is to wait rather than to proceed carefully.
# 7. Separate the return that recovers the cost from the return a budget requires on what it
#    funds. Both fall with the capital a signal would inform, which is why the same dataset is a
#    reasonable purchase at one firm and not at another, and why the question is never whether a
#    signal is real on its own.
