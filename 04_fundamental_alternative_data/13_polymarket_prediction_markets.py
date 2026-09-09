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
# # Polymarket Prediction Markets: Crypto-Settled Event Contracts
#
# **Chapter 4: Fundamental and Alternative Data**
# **Docker image**: `ml4t`
# **Section Reference**: Section 4.4 (Understanding Alternative Data)
#
# ## Purpose
#
# The previous notebook read the regulated prediction market. This one reads the other kind.
# Polymarket runs on the Polygon blockchain, settles in a dollar stablecoin rather than dollars,
# imposes no position limit, and lists whatever its users ask for. It is reported to carry far
# more volume than the regulated venue, which Part 5 explains this feed cannot confirm, and it is
# not available to US persons.
#
# The pair is worth studying together because the differences between them are the differences
# that matter when sourcing any alternative dataset from a venue rather than a vendor: who is
# allowed to trade there decides who sets the price, and what the venue lists decides what
# questions its data can answer.
#
# ## Learning Objectives
#
# After completing this notebook, you will be able to:
#
# - State the structural differences between a regulated and an unregulated prediction market,
#   and say which of them affect the data rather than the trading.
# - Establish how many markets, bars and dates a snapshot actually contains before computing
#   anything from it.
# - Recognize a threshold ladder on either venue, and check the monotonicity it has to satisfy.
# - Explain why the volume fields of two venues cannot be compared, and what can be compared
#   instead.
# - Build the same probability features on both venues, and read them against the sample size.
#
# ## Prerequisites
#
# ```bash
# python data/prediction_markets/download.py
# ```
#
# ## Cross-References
#
# - **Upstream**: `data/prediction_markets/download.py`
# - **Related**: [`12_kalshi_prediction_markets`](12_kalshi_prediction_markets.ipynb) (the CFTC-regulated venue)

# %%
"""Polymarket Prediction Markets - compare crypto-based event contracts with Kalshi for ML feature engineering."""

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data.prediction_markets.loader import load_kalshi, load_polymarket
from utils.paths import get_output_dir
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
CONFIDENT_PROBABILITY = 0.2  # a contract within this of zero or one is treated as settled
LIVE = False  # set True to query the live Polymarket market list; off for reproducible runs

# %%
OUTPUT_DIR = get_output_dir(4, "polymarket")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. The two venues
#
# Both list binary contracts whose price is a probability. Everything else about them differs,
# and the differences fall into two groups.
#
# | | Polymarket | Kalshi |
# |---|---|---|
# | Regulator | None | Commodity Futures Trading Commission |
# | Settlement asset | USDC, a dollar stablecoin | US dollars |
# | Position limit | None | Twenty-five thousand dollars per contract |
# | Available to US persons | No | Yes |
# | Listing | User-proposed, broad | Exchange-defined, narrower |
#
# The first four rows are about **who can trade and how much**, and they decide whose views the
# price aggregates. A venue closed to US persons prices a US interest-rate decision using the
# opinions of everyone except the people closest to it, which is a selection effect, not a
# defect, and one to keep in mind before treating either price as the market's view.
#
# The last row is about **what gets listed**, and it decides which questions the data can answer
# at all. It is also the row most often overstated, as Part 3 shows.

# %% [markdown]
# ## 2. What the snapshot contains
#
# The downloader ships a small, reproducible sample rather than a live query, so the first thing
# to establish is how small. Every statement later in this notebook is bounded by these three
# numbers.

# %%
poly = load_polymarket().sort("symbol", "timestamp")
print(f"Rows: {len(poly)}")
print(f"Markets: {poly['symbol'].n_unique()}")
print(
    f"Dates: {poly['timestamp'].n_unique()}, {poly['timestamp'].min()} to {poly['timestamp'].max()}"
)
print(f"Categories: {', '.join(sorted(poly['category'].unique()))}")

# %%
latest = (
    poly.sort("timestamp")
    .group_by("symbol")
    .agg(
        pl.col("category").last(),
        pl.col("close").last().alias("probability"),
        pl.col("volume").sum().alias("price_points"),
        pl.len().alias("bars"),
    )
    .sort("probability", descending=True)
)
latest

# %%
print(f"Markets appearing on both dates: {(latest['bars'] == 2).sum()}")
print(f"Markets appearing on one date only: {(latest['bars'] == 1).sum()}")

# %% [markdown]
# A snapshot in which most markets appear once and none appears more than twice is a smoke test,
# not a history. It is enough to read a cross-section of prices and to compare the two venues'
# structure; it is not enough to compute a distribution, a volatility, or anything else that needs
# a series. The sections below stay inside that limit and say where it binds.

# %%
fig = px.scatter(
    latest.with_columns(
        label=pl.col("symbol").str.replace(":YES", "").str.slice(0, 44)
    ).to_pandas(),
    x="probability",
    y="label",
    color="category",
    title="Almost every listed market is priced at one end of the range",
    labels={"probability": "Implied probability", "label": "", "category": "Category"},
)
fig.update_layout(height=460, xaxis_tickformat=".0%", xaxis_range=[-0.05, 1.05], margin=dict(l=300))
show_plotly_with_alt(
    fig,
    "Dot plot of every market in the snapshot against its implied probability, coloured by "
    "category. The points cluster hard against both ends of the axis with almost nothing in the "
    "middle.",
)

# %% [markdown]
# The gap in the middle of that axis is the normal state of a prediction market and it is worth
# understanding before building anything on one. A contract only sits near even odds while the
# question is genuinely open; most listed questions are either already decided by events or were
# never close. A feature built on "the market's probability" is therefore a feature that is
# informative on a small and shifting subset of the listed universe, and the rest of the time it
# is reporting a settled fact.

# %% [markdown]
# ## 3. Both venues price ladders
#
# The obvious way to describe the difference between the two venues is that Kalshi lists a ladder
# of rate thresholds and Polymarket lists individual event bets. The snapshot refutes it: several
# of these markets ask whether Bitcoin is above each of a series of levels on the same date,
# which is the same construction as Kalshi's rate ladder applied to a different underlying.

# %% [markdown]
# Two things have to match before a set of contracts is a ladder rather than a collection.
# They must resolve on the **same terms** - a Bitcoin level on 2 May and the same level on
# 31 May are different questions - and they must be priced on the **same date**, since a
# comparison across dates measures the market moving rather than the shape of the distribution.
# The ticker carries the resolution term, so both conditions can be enforced rather than assumed.

# %%
LADDER_TICKER = r"(?i)^BITCOIN-ABOVE-(?<threshold_k>\d+)K-ON-(?<resolves>[A-Z]+-\d+)"

candidates = (
    poly.with_columns(parsed=pl.col("symbol").str.extract_groups(LADDER_TICKER))
    .unnest("parsed")
    .drop_nulls("threshold_k")
    .with_columns(pl.col("threshold_k").cast(pl.Int64))
)
if candidates.is_empty():
    ladder = candidates.select("symbol", "threshold_k", "resolves", "timestamp", "close")
    breaks = ladder
    print("No threshold ladder in this snapshot.")
else:
    # The resolution term and date with the most contracts is the one worth drawing.
    # Ties on contract count are broken by the most recent date and then by the resolution
    # term, so the same group is chosen on every run over the same input.
    best = (
        candidates.group_by("resolves", "timestamp")
        .len()
        .sort(["len", "timestamp", "resolves"], descending=[True, True, False])
        .row(0, named=True)
    )
    ladder = (
        candidates.filter(
            (pl.col("resolves") == best["resolves"]) & (pl.col("timestamp") == best["timestamp"])
        )
        .sort("threshold_k")
        .select("symbol", "threshold_k", "resolves", "timestamp", "close")
    )
    breaks = ladder.filter(pl.col("close").diff() > 0)
    print(f"Contracts resolving on {best['resolves']}, priced {best['timestamp']}: {len(ladder)}")
    print(f"Places where a higher threshold's bid exceeds a lower threshold's: {len(breaks)}")
ladder

# %%
fig = px.line(
    ladder.to_pandas(),
    x="threshold_k",
    y="close",
    markers=True,
    title="An unregulated venue prices the same ladder shape as a regulated one",
    labels={
        "threshold_k": "Bitcoin price threshold (thousands of USD)",
        "close": "Probability the price is above the threshold",
    },
    color_discrete_sequence=[COLORS["amber"]],
)
fig.update_layout(height=380, yaxis_tickformat=".0%", yaxis_range=[-0.02, 1.05])
show_plotly_with_alt(
    fig,
    "Line chart with markers of the probability Bitcoin exceeds each listed price threshold on a "
    "single date, falling from near certainty at the lowest threshold to almost nothing at the "
    "highest, with the steep fall between the second and third.",
)

# %% [markdown]
# The curve has the same shape as the Fed ladder in the previous notebook and the same constraint
# holds: a higher threshold cannot be priced above a lower one. What differs between the venues is
# not the mechanism but the subject. Kalshi's ladders are on the outcomes its regulator has
# approved, which in practice means economic releases and policy decisions; Polymarket's are on
# whatever its users proposed, which in this snapshot means crypto prices, a space launch and a
# central bank appointment.

# %% [markdown]
# ## 4. The same event on both venues
#
# Both list contracts on Federal Reserve decisions, which is the one place a direct comparison is
# available. The comparison is still not like for like, and the reason is instructive.

# %%
kalshi = load_kalshi()
kalshi_latest = (
    kalshi.sort("timestamp")
    .group_by("symbol")
    .agg(
        pl.col("close").last().alias("probability"),
        pl.col("volume").sum().alias("volume_contracts"),
    )
    .with_columns(threshold=pl.col("symbol").str.split("-T").list.last().cast(pl.Float64))
    .sort("threshold")
)
poly_fed = latest.filter(
    pl.col("symbol").str.contains(r"(?i)fed|powell|interest-rate|shelton")
).sort("probability", descending=True)

print(f"Kalshi rate contracts: {len(kalshi_latest)}")
print(f"Polymarket policy contracts: {len(poly_fed)}")

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Kalshi: one meeting's rate thresholds", "Polymarket: separate policy events"),
    horizontal_spacing=0.35,
)
fig.add_trace(
    go.Bar(
        x=kalshi_latest["probability"],
        y=[f"above {t}%" for t in kalshi_latest["threshold"]],
        orientation="h",
        marker_color=COLORS["blue"],
        text=[f"{p:.0%}" for p in kalshi_latest["probability"]],
        textposition="outside",
        cliponaxis=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=poly_fed["probability"],
        y=[s.replace(":YES", "").replace("-", " ").title()[:38] for s in poly_fed["symbol"]],
        orientation="h",
        marker_color=COLORS["amber"],
        text=[f"{p:.1%}" for p in poly_fed["probability"]],
        textposition="outside",
        cliponaxis=False,
    ),
    row=1,
    col=2,
)
fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title_text="Implied probability", row=1, col=1)
# A download configured for other categories can leave the policy subset empty, in which case
# there is no maximum to scale by and the right thing is a default range rather than a crash.
poly_ceiling = 1.0 if poly_fed.is_empty() else float(poly_fed["probability"].max()) * 1.5
fig.update_xaxes(
    range=[0, poly_ceiling],
    tickformat=".1%",
    title_text="Implied probability, own scale",
    row=1,
    col=2,
)
fig.update_layout(height=440, showlegend=False, margin=dict(l=170, r=90))
show_plotly_with_alt(
    fig,
    "Two horizontal bar panels on separate probability scales. The left panel's bars span nearly "
    "the whole range; the right panel's all sit within a few percent of zero, which is why the "
    "scales are not shared.",
)

# %% [markdown]
# The two panels carry separate scales deliberately. Kalshi's ladder spans the whole probability
# range because a ladder always does: some threshold is nearly certain and some is nearly
# impossible. Polymarket's policy contracts are all tail events in this snapshot, so a shared axis
# would compress them to invisibility and would also suggest a like-for-like comparison that does
# not exist. These are different questions about the same institution, not two prices for one
# contract.
#
# Where the venues do list the same question, the difference between their prices is the
# interesting quantity, and it is a measure of who is allowed to trade rather than of who is
# right. That comparison needs both venues to list a genuinely identical contract, which this
# snapshot does not contain.

# %% [markdown]
# ## 5. One of these columns is not a volume
#
# Both feeds carry a column called `volume`, and reading the providers rather than the column
# names is what settles what they hold. Kalshi's is contracts traded, which is a volume. The
# Polymarket loader builds its bars from a price history that carries no size at all, and fills
# the column with **the number of price observations in the day** - the provider's own comment
# calls it a proxy. It is a sampling frequency, and it rises when the price is quoted often, not
# when size changes hands.

# %%
pl.DataFrame(
    {
        "venue": ["Polymarket", "Kalshi"],
        "column_total": [float(poly["volume"].sum()), float(kalshi["volume"].sum())],
        "what_it_counts": ["price observations in the bar", "contracts traded"],
        "is_a_volume": [False, True],
        "bars": [len(poly), len(kalshi)],
    }
)

# %% [markdown]
# No comparison of trading activity across these two feeds is available, and the honest response
# is to say so rather than to divide one total by the other. Polymarket does publish a traded
# volume per market, on the market metadata the live path in Part 7 fetches; it is simply not in
# the bars. A column name is a claim like any other, and this is the cheapest kind of defect to
# inherit: everything downstream still runs, and every number it produces is about something
# else.

# %% [markdown]
# ## 6. Features
#
# The features are the same ones the previous notebook built, which is the point: a probability
# path is a probability path whatever venue quoted it, and a pipeline that ingests both wants one
# feature definition rather than two.
#
# **Conviction** is how far the price sits from even odds. **Extreme** flags the contracts priced
# close to zero or one. Neither says the question is settled: a contract at one percent has the
# whole range above it, and the events that move a market most are the ones it had written off.
# What the flag does is separate the contracts whose price carries information about a live
# question from those where a change would be news in itself, and a strategy handles the two
# differently rather than ignoring either.

# %%
features = poly.select(
    "timestamp",
    "symbol",
    "category",
    "open",
    "high",
    "low",
    "close",
    "volume",
    conviction=(pl.col("close") - 0.5).abs(),
    extreme=(
        (pl.col("close") > 1 - CONFIDENT_PROBABILITY) | (pl.col("close") < CONFIDENT_PROBABILITY)
    ).cast(pl.Int8),
    intraday_range=pl.col("high") - pl.col("low"),
)
print(f"Rows: {len(features)}")
print(f"Rows flagged extreme: {int(features['extreme'].sum())}")
print(f"Rows with any intraday range: {int((features['intraday_range'] > 0).sum())}")
features.head(8)

# %% [markdown]
# Every row is flagged extreme, which is the dot plot from Part 2 restated as a column. On a
# universe where that is true of everything, the flag separates nothing and the conviction feature
# is nearly constant. Neither is a defect in the definition; both are the snapshot telling you
# that a feature needs markets spread across the probability range, and that selecting them is the
# first step rather than an afterthought.

# %% [markdown]
# ## 7. Querying the live venue
#
# Everything above reads the shipped snapshot, which is what makes this notebook reproducible.
# Setting `LIVE` true instead queries the exchange for its current market list, which is the right
# path for exploration and the wrong one for a notebook that has to run the same way twice.

# %%
if LIVE:
    from ml4t.data.providers.polymarket import PolymarketProvider

    provider = PolymarketProvider()
    live_markets = provider.list_markets(active=True, closed=False, limit=200)
    provider.close()
    live = pl.DataFrame(
        {
            "slug": [m.get("slug", "") for m in live_markets],
            "question": [m.get("question", "")[:80] for m in live_markets],
            "volume": [float(m.get("volume", 0)) for m in live_markets],
            "liquidity": [float(m.get("liquidity", 0)) for m in live_markets],
        }
    ).sort("volume", descending=True)
    print(f"Active markets returned: {len(live)}")
else:
    live = None
    print("LIVE is false; reading the shipped snapshot only.")

live.head(10) if live is not None else None

# %% [markdown]
# ## 8. Saving the feature panel

# %%
output_file = OUTPUT_DIR / "polymarket_features.parquet"
features.write_parquet(output_file)
print(f"Wrote {len(features)} rows to {output_file}")

# %% [markdown]
# ## Key Takeaways
#
# 1. The structural differences between the two venues are about who may trade and what gets
#    listed. The first is a selection effect on the price - a venue closed to US persons prices a
#    US policy decision without the people closest to it - and the second decides which questions
#    the data can answer.
# 2. The listing difference is easy to overstate. Both venues price threshold ladders, and both
#    ladders have to be monotone in the threshold. The mechanism is shared; the subjects differ.
# 3. Read the provider before trusting a column name. Kalshi's `volume` is contracts traded;
#    Polymarket's is a count of price observations, because the bars are built from a price
#    history with no size in it. Nothing errors when the two are compared, and the answer is
#    about neither.
# 4. Most listed contracts are priced as near-settled almost all of the time. Selecting the
#    markets with an open question is the first step in using this data, not a refinement of it.
# 5. Establish the snapshot's size before computing on it. Two bars per market over two days
#    supports a cross-section and a structural comparison, and supports no statement about how
#    anything moved.
#
# **Previous**: [`12_kalshi_prediction_markets`](12_kalshi_prediction_markets.ipynb) reads the
# regulated venue and its rate ladder in full.
