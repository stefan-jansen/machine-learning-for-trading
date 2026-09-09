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

# %% [markdown] tags=[]
# # US Equities — Exploratory Data Analysis
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Profile the **US equities panel**: its schema, its price conventions, and — the part
# that decides whether it can carry a backtest — the shape of its coverage through time.
#
# The panel is sourced from the legacy Quandl WIKI end-of-day file: community-maintained,
# free, and discontinued in March 2018. That provenance is not trivia. It determines what
# the panel can and cannot tell us, and this notebook's job is to find out which.
#
# ## Learning objectives
#
# - Load the panel via `data.load_us_equities` and inspect its canonical schema.
# - Distinguish raw from split/dividend-adjusted price columns.
# - Chart how many symbols the panel covers in each year, and when symbols enter and leave it.
# - Read the coverage chart as evidence about the *data collection process*, not about the market.
# - Check OHLC invariants and null rates across the full panel.
#
# ## How this notebook relates to `15_survivorship_bias_detection`
#
# This notebook establishes **what is in the file**. It ends by surfacing one anomaly in the
# coverage record that it deliberately does not resolve.
#
# `15_survivorship_bias_detection` picks that anomaly up and asks the follow-on question:
# *given what the panel records about symbols that leave it, how wrong is a backtest that
# ignores them?* Read this one first; it is the shorter one, and it sets up the problem.
#
# ## Book reference
#
# §2.2, "The asset-class market data landscape" - the equities part of it.
#
# ## Prerequisites
#
# `data` package on `PYTHONPATH`; the panel's parquet present at
# `ML4T_DATA_PATH/equities/market/us_equities/`. Run
# `python data/equities/market/us_equities/download.py` if missing.

# %% tags=[]
"""US Equities — Exploratory data analysis of the US equities panel."""

import plotly.graph_objects as go
import polars as pl
from IPython.display import Markdown, display

from data import load_us_equities
from utils.data_quality import check_ohlc_invariants
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all symbols; a positive value subsets for fast execution

# %% [markdown] tags=[]
# ## 1. Load and inspect

# %% tags=[]
wiki = load_us_equities()

if MAX_SYMBOLS:
    keep = sorted(wiki["symbol"].unique().to_list())[:MAX_SYMBOLS]
    wiki = wiki.filter(pl.col("symbol").is_in(keep))

print("=== US equities panel ===")
print(f"Shape: {wiki.shape}")
print(f"Columns: {wiki.columns}")

# %% tags=[]
print("Schema:")
for col, dtype in wiki.schema.items():
    print(f"  {col}: {dtype}")

# %% [markdown] tags=[]
# ### Raw versus adjusted prices
#
# This panel carries **both** price conventions. Which one you reach for is not a
# style preference — it changes the numbers.
#
# | Column type | Columns | Use for |
# |---|---|---|
# | Raw | `open`, `high`, `low`, `close`, `volume` | Analysis at actually traded price levels |
# | Adjusted | `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_volume` | Return calculations and backtesting |
#
# The adjusted columns are meant to absorb splits and dividends so that a return computed
# across a corporate action is a return an investor could have earned. Use `adj_*` for
# returns; use the raw columns when you need the price a trade would have printed at.
#
# The panel also ships the two corporate-action inputs themselves, `ex-dividend` and
# `split_ratio`, which is what lets `02_corporate_actions` verify the adjustment on a
# worked example rather than take it on trust. That check passes on AAPL. It does not
# pass everywhere — see `15_survivorship_bias_detection` §3.

# %% [markdown] tags=[]
# ## 2. Coverage: the cross-section

# %% tags=[]
print("=== Coverage ===")
print(f"Unique symbols: {wiki['symbol'].n_unique():,}")
print(f"Date range: {wiki['timestamp'].min()} to {wiki['timestamp'].max()}")
print(f"Total rows: {len(wiki):,}")

# %% tags=[]
# One row per symbol: when it enters the panel, when it leaves, how long it trades.
dataset_end = wiki.select(pl.col("timestamp").max()).item()

lifespans = (
    wiki.group_by("symbol")
    .agg(
        [
            pl.len().alias("days"),
            pl.col("timestamp").min().alias("first_date"),
            pl.col("timestamp").max().alias("last_date"),
            pl.col("adj_close").mean().alias("avg_price"),
        ]
    )
    .with_columns(
        [
            pl.col("first_date").dt.year().alias("entry_year"),
            pl.col("last_date").dt.year().alias("exit_year"),
            (pl.col("last_date") < dataset_end).alias("leaves_early"),
        ]
    )
)

# %% [markdown] tags=[]
# Per-symbol coverage distribution — number of trading days and mean adjusted price.

# %% tags=[]
lifespans.select(["days", "avg_price"]).describe()

# %% [markdown] tags=[]
# ## 3. Coverage through time
#
# The cross-section above says how many symbols the panel holds *in total*. It says nothing
# about **when**. The same total is consistent with every symbol being quoted in every year,
# and with a universe that grows and shrinks as firms list and delist. Those are different
# datasets and only one of them can carry a backtest.
#
# Two views answer this:
#
# - **Universe size** — how many distinct symbols are quoted in each calendar year.
# - **Entries and exits** — how many symbols record their *first* observation in each year,
#   and how many record their *last*.
#
# For a panel tracking a real equity market, both flows should run continuously: firms IPO
# every year, and firms are acquired or fail every year.

# %% tags=[]
active_by_year = (
    wiki.select(pl.col("timestamp").dt.year().alias("year"), "symbol")
    .group_by("year")
    .agg(pl.col("symbol").n_unique().alias("active"))
    .sort("year")
)

peak = active_by_year.sort("active", descending=True).head(1)
peak_year, peak_active = peak["year"][0], peak["active"][0]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=active_by_year["year"].to_list(),
        y=active_by_year["active"].to_list(),
        mode="lines",
        name="Symbols quoted",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig.add_annotation(
    x=peak_year,
    y=peak_active,
    text=f"Peak: {peak_active:,} symbols ({peak_year})",
    showarrow=True,
    arrowhead=2,
    ax=-70,
    ay=-30,
    font=dict(color=COLORS["copper"]),
)
fig.update_layout(
    title="The quoted universe grows for decades, peaks, and then shrinks",
    xaxis_title="Year",
    yaxis_title="Distinct symbols",
    height=420,
)
show_plotly_with_alt(
    fig,
    "A line of the number of distinct symbols quoted each year. It rises steadily from the "
    "early 1960s, reaches its highest point a few years before the panel ends, and falls "
    "away sharply after that.",
)

# %% [markdown] tags=[]
# The universe grows for fifty years, peaks, and then declines. A market does not do that.
# A *data collection process* does. Hold that thought; the next chart names the mechanism.

# %% tags=[]
entries = (
    lifespans.group_by("entry_year").agg(pl.len().alias("entries")).rename({"entry_year": "year"})
)
# A symbol whose last observation is the panel's last date has not left — it is still quoted.
exits = (
    lifespans.filter(pl.col("leaves_early"))
    .group_by("exit_year")
    .agg(pl.len().alias("exits"))
    .rename({"exit_year": "year"})
)

years = pl.DataFrame(
    {"year": list(range(lifespans["entry_year"].min(), lifespans["exit_year"].max() + 1))}
)
flows = (
    years.join(entries, on="year", how="left")
    .join(exits, on="year", how="left")
    .with_columns([pl.col("entries").fill_null(0), pl.col("exits").fill_null(0)])
    .sort("year")
)

first_exit_year = flows.filter(pl.col("exits") > 0)["year"].min()

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=flows["year"].to_list(),
        y=flows["entries"].to_list(),
        name="Entries (first observation)",
        marker_color=COLORS["slate"],
    )
)
fig.add_trace(
    go.Bar(
        x=flows["year"].to_list(),
        y=[-n for n in flows["exits"].to_list()],
        name="Exits (last observation)",
        marker_color=COLORS["copper"],
    )
)
fig.add_vline(x=first_exit_year - 0.5, line_dash="dash", line_color=COLORS["amber"])
fig.add_annotation(
    x=first_exit_year - 0.5,
    y=max(flows["entries"].to_list()),
    text=f"First exit on record: {first_exit_year}",
    showarrow=False,
    xanchor="right",
    font=dict(color=COLORS["amber"]),
)
fig.add_hline(y=0, line_color=COLORS["neutral"], line_width=1)
fig.update_layout(
    title="Entries run through the whole panel; exits appear only at the end",
    xaxis_title="Year",
    yaxis_title="Symbols",
    barmode="relative",
    height=420,
)
show_plotly_with_alt(
    fig,
    "Bars above the axis count symbols recording their first observation in each year and "
    "bars below it count those recording their last. The upper bars are present in every "
    "year of the panel; the lower bars are empty until a dashed rule near the right edge, "
    "after which they appear in every remaining year.",
)

# %% [markdown] tags=[]
# ## 4. What the exit record actually shows
#
# Entries run continuously from 1962. Exits do not: **the panel records no symbol leaving
# before 2014.**
#
# Firms plainly did exit before 2014 — they were acquired, they went bankrupt, they went
# private. Enron delisted in 2002. What the chart shows is therefore not a fact about the
# equity market. It is a fact about the file: **exit information was not being captured until
# roughly 2014, and started being captured around then.**
#
# This is the tax on free, community-maintained data. The panel was assembled from the
# symbols that existed when the project ran, each backfilled to its own IPO. Symbols that had
# already died were never added, so their disappearance was never recorded. From 2014 the
# panel is live, so symbols leaving it *are* recorded. Both flows then decay as contributors
# drift away, until the feed stops in March 2018.
#
# What follows for a backtest:
#
# - Before 2014 the panel holds **only firms that survived to 2014**. A 1990s backtest run on
#   it will not see a single failure, because none are there to see.
# - From 2014 the panel does record exits, and can support survivorship-aware work.
# - The absence of exits before 2014 is **absence of evidence**, not evidence that nothing
#   left. We do not know what left. The panel cannot tell us.

# %% tags=[]
exit_summary = (
    lifespans.filter(pl.col("leaves_early"))
    .group_by("exit_year")
    .agg(pl.len().alias("symbols"))
    .sort("exit_year")
    .rename({"exit_year": "year"})
)
exit_summary

# %% tags=[]
n_total = lifespans.height
n_left = lifespans.filter(pl.col("leaves_early")).height
n_still = n_total - n_left
first_year, last_year = lifespans["entry_year"].min(), flows["year"].max()

longest_run = lifespans["days"].max()
display(
    Markdown(
        f"**Coverage summary.** The panel holds **{n_total:,} symbols** over "
        f"**{first_year}-{last_year}**, and its longest single-symbol history runs "
        f"**{longest_run:,} trading days**. **{n_still:,}** ({n_still / n_total:.1%}) are "
        f"still quoted on the final date ({dataset_end}); **{n_left:,}** "
        f"({n_left / n_total:.1%}) leave earlier. Every one of those exits falls in "
        f"**{first_exit_year} or later**, so the panel records no exit across its first "
        f"**{first_exit_year - first_year} years**."
    )
)

# %% [markdown] tags=[]
# ## 5. Data quality

# %% tags=[]
null_counts = wiki.null_count()
total_nulls = null_counts.sum_horizontal()[0]
print("=== Nulls ===")
print(f"Total null values: {total_nulls:,}")
for col in null_counts.columns:
    val = null_counts[col][0]
    if val > 0:
        print(f"  {col}: {val:,} ({val / len(wiki) * 100:.4f}%)")
print(f"\nNull rate: {total_nulls / (len(wiki) * len(wiki.columns)) * 100:.4f}% of values")

# %% tags=[]
invariants = check_ohlc_invariants(
    wiki,
    open_col="adj_open",
    high_col="adj_high",
    low_col="adj_low",
    close_col="adj_close",
    volume_col="adj_volume",
)

print("OHLC invariants (adjusted prices):")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

# %% [markdown] tags=[]
# Both checks pass, and neither one would catch the coverage gap in §4. Nulls and OHLC
# invariants are *within-row* checks: they ask whether each observation is internally
# consistent. Survivorship is a property of **which rows exist at all**, and no amount of
# row-level validation will surface a symbol that was never written to the file.

# %% [markdown] tags=[]
# ## 6. Example: a single symbol

# %% tags=[]
aapl = wiki.filter(pl.col("symbol") == "AAPL").sort("timestamp")

print("=== AAPL ===")
print(f"Trading days: {len(aapl):,}")
print(f"Date range: {aapl['timestamp'].min()} to {aapl['timestamp'].max()}")

# %% [markdown] tags=[]
# Five most recent trading days for AAPL, adjusted prices.

# %% tags=[]
aapl.select(["timestamp", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]).tail(5)

# %% [markdown] tags=[]
# ## Key takeaways
#
# - **Profile the coverage before profiling the rows.** Nulls and OHLC invariants ask whether
#   each observation is internally consistent. Survivorship is a property of which rows exist
#   at all, and no within-row check can raise a symbol that was never written to the file.
# - **Plot both flows, not the total.** A count of symbols per year and a count of first and
#   last observations per year answer different questions, and it is the second that showed
#   the exit record switching on partway through the sample. The total is consistent with
#   either dataset.
# - **A discontinuity in a data-collection record is not a market event.** Firms were acquired
#   and did fail throughout this history; the panel simply did not record it until collection
#   went live. Read a break in coverage as a fact about the file first, and look for the
#   collection mechanism that produced it.
# - **Absence of exits is absence of evidence.** For the years before the record starts, this
#   panel cannot say which symbols left, so it cannot say how wrong a backtest run on it would
#   be. That is a stronger statement than "the bias is small", and a weaker one than any number.
# - **Say which price convention a number came from.** Raw prices are what a trade would have
#   printed at; the adjusted columns are what a return should be computed on. The panel carries
#   both, and the two disagree across every split and dividend.
#
# **Known limitations.** The panel is the discontinued Quandl WIKI file, so it stops in 2018
# and nothing here extends past it. Symbols are identified by ticker, which is reused after a
# delisting, so a long history under one ticker is not proof of one continuous company. And the
# adjustment columns are taken on trust in this notebook: `02_corporate_actions` checks them on
# a worked example, and `15_survivorship_bias_detection` finds where the check fails.
#
# **Next**: `02_corporate_actions` validates the adjustment factors behind the `adj_*` columns
# on a worked example. `15_survivorship_bias_detection` takes the coverage finding and asks how
# much a backtest that ignores the leavers gets wrong. **Book reference**: §2.2.
