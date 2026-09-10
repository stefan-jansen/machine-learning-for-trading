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
# # ETFs — Exploratory Data Analysis
#
# **Docker image**: `ml4t`
#
# **Purpose**: Profile the 100-ETF universe sourced from Yahoo Finance and confirm the
# group coverage, history, and data-quality characteristics that drive the ETF rotation
# case study.
#
# **Learning objectives**:
#
# - Load the ETF panel via `data.load_etfs` and inspect its canonical schema.
# - Quantify per-symbol coverage and see why this universe *grows and plateaus*
#   rather than rising and falling (contrast `01_us_equities_eda`).
# - Attach the nine-group classification from the universe dictionary and check the
#   panel and dictionary agree symbol-for-symbol.
# - Check OHLC invariants and null rates, and compare liquidity across the nine groups.
#
# **Book reference**: §2.2, "The asset-class market data landscape" - the ETP part of it.
#
# **Prerequisites**: `data` package on `PYTHONPATH`; ETF parquet present at
# `ML4T_DATA_PATH/etfs/market/`. Run `python data/etfs/market/download.py` if missing.

# %%
"""ETFs — Exploratory data analysis of the multi-asset ETF universe."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from ml4t.data.etfs import ETFDataManager

from data import load_etfs
from utils.data_quality import check_ohlc_invariants
from utils.paths import REPO_ROOT
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all

# %% [markdown]
# ## 1. Load and inspect
#
# The ETF universe is stored as a single Parquet file of daily OHLCV data spanning the major
# asset classes. The load below reports how many symbols and how many rows that comes to.

# %%
etfs = load_etfs()

print("=== ETF dataset ===")
print(f"Shape: {etfs.shape}")
print(f"Columns: {etfs.columns}")

# %%
# Schema overview
print("\nSchema:")
for col, dtype in etfs.schema.items():
    print(f"  {col}: {dtype}")

# %% [markdown]
# ### Adjusted prices
#
# Yahoo Finance returns split- and dividend-adjusted OHLC. The `close` column is the
# adjusted close, so returns can be computed directly without a separate `adj_close`
# column. This is the opposite convention to the US equities panel in
# `01_us_equities_eda`, which ships raw *and* adjusted columns — worth keeping straight
# when you move between the two.

# %% [markdown]
# ## 2. Coverage
#
# How many ETFs, over what window, and how much of it does each one cover?

# %%
symbols = etfs["symbol"].unique().sort().to_list()
date_range = etfs.select(
    pl.col("timestamp").min().alias("start"),
    pl.col("timestamp").max().alias("end"),
    pl.col("timestamp").n_unique().alias("unique_dates"),
)
full_start = date_range["start"][0]
full_end = date_range["end"][0]

print("=== Coverage ===")
print(f"Number of ETFs: {len(symbols)}")
print(f"Date range:     {full_start} to {full_end}")
print(f"Trading days:   {date_range['unique_dates'][0]:,}")

# %%
# Per-symbol first/last observation and row count
symbol_stats = etfs.group_by("symbol").agg(
    pl.col("timestamp").min().alias("start"),
    pl.col("timestamp").max().alias("end"),
    pl.len().alias("rows"),
)
partial = symbol_stats.filter((pl.col("start") != full_start) | (pl.col("end") != full_end))

print(f"Symbols with full coverage:    {symbol_stats.height - partial.height}")
print(f"Symbols with partial coverage: {partial.height}")

# %% [markdown]
# Most ETFs predate the 2006 start of the panel; a sizeable minority were launched
# later and so start after `2006-01-03`. None *end* early — every symbol is still
# quoted on the final date.

# %%
partial.select(["symbol", "start", "end", "rows"]).sort("start")

# %% [markdown]
# ### The universe grows and then plateaus
#
# Counting how many ETFs have data available at each year-end tells a different story from
# the equities panel in `01_us_equities_eda`. There the count rose for decades and then fell,
# which is the signature of a collection that started recording exits partway through. Here it
# rises and then holds flat, because the universe was chosen once and every member is still
# quoted; the climb is new products reaching their launch date, not the universe changing.

# %%
years = list(range(full_start.year, full_end.year + 1))
available = [
    symbol_stats.filter((pl.col("start").dt.year() <= y) & (pl.col("end").dt.year() >= y)).height
    for y in years
]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=years,
        y=available,
        mode="lines+markers",
        name="ETFs available",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig.update_layout(
    title="ETFs with data available at each year-end",
    xaxis_title="Year",
    yaxis_title="ETFs available",
    yaxis_range=[0, 105],
    height=420,
)
show_plotly_with_alt(
    fig,
    "A line counting how many ETFs have data available at each year-end. It climbs steadily "
    "through the first decade as new products launch, reaches the full universe, and then "
    "runs flat to the end of the panel with no decline.",
)

# %% [markdown]
# The count climbs while new products list and then runs flat: nothing leaves. That is the
# opposite of the equities panel in `01_us_equities_eda`, which falls away at the end, and the
# difference is not a fact about the two markets. ETFs close down too. What differs is how each
# panel was assembled, and a universe that never loses a member is the signature of a
# survivorship-filtered one.

# %% [markdown]
# ## 3. The nine groups
#
# The universe config that drove the download — `data/etfs/market/config.yaml`, read
# through `ETFDataManager` — classifies every symbol into one of **nine groups**. Sourcing
# the classification from the same config that generated the panel (rather than a separate
# metadata file) keeps the two in lockstep by construction; we still cross-check that the
# configured universe and the price panel agree symbol-for-symbol.

# %%
config_path = REPO_ROOT / "data" / "etfs" / "market" / "config.yaml"
etf_mgr = ETFDataManager.from_config(str(config_path))

groups = pl.DataFrame(
    [
        {"symbol": symbol, "group": group, "description": info.get("description", "")}
        for group, info in etf_mgr.config.tickers.items()
        for symbol in info["symbols"]
    ]
)

panel_symbols = set(symbols)
config_symbols = set(groups["symbol"].to_list())
print("=== Panel vs config universe ===")
print(f"In panel but not config: {sorted(panel_symbols - config_symbols) or 'none'}")
print(f"In config but not panel: {sorted(config_symbols - panel_symbols) or 'none'}")

group_sizes = groups.group_by("group").agg(pl.len().alias("etfs")).sort("etfs", descending=True)
print(f"\nGroups: {group_sizes.height} | ETFs classified: {group_sizes['etfs'].sum()}")
group_sizes

# %%
fig = go.Figure(
    go.Bar(
        x=group_sizes["etfs"].to_list(),
        y=group_sizes["group"].to_list(),
        orientation="h",
        marker_color=COLORS["slate"],
        text=group_sizes["etfs"].to_list(),
        textposition="outside",
    )
)
fig.update_layout(
    title="ETFs per group",
    xaxis_title="ETFs",
    yaxis=dict(autorange="reversed"),
    height=420,
)
show_plotly_with_alt(
    fig,
    "A horizontal bar chart of how many ETFs each group holds, sorted from largest to "
    "smallest, with the count written at the end of each bar. The largest group holds "
    "several times what the smallest does.",
)

# %% [markdown]
# The groups are uneven - the largest holds several times what the smallest does - and the
# counts sum to the universe total, so no symbol is dropped or counted twice. That matters for
# anything that aggregates by group later: an equal-weighted average across groups is not an
# equal-weighted average across ETFs.

# %% [markdown]
# ## 4. Data quality

# %%
null_counts = etfs.null_count()
total_nulls = null_counts.sum_horizontal()[0]
zero_volume = etfs.filter(pl.col("volume") == 0)

print("=== Data quality ===")
print(f"Total null values: {total_nulls}")
print(f"Zero-volume rows:  {zero_volume.height} ({100 * zero_volume.height / etfs.height:.3f}%)")

# %% [markdown]
# Two checks of the same invariant follow, and they are meant to disagree.
#
# `check_ohlc_invariants` compares against a tolerance scaled to the price. The filter
# below it compares strictly, the way the check itself used to. Run them side by side and
# the third number, the size of the breach, says which of the two is describing the panel
# and which is describing its own arithmetic.

# %%
# high should be the max and low the min of {open, high, low, close}
invariants = check_ohlc_invariants(etfs)
print("OHLC invariants:")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

violations = etfs.filter(
    (pl.col("high") < pl.col("low"))
    | (pl.col("high") < pl.col("open"))
    | (pl.col("high") < pl.col("close"))
    | (pl.col("low") > pl.col("open"))
    | (pl.col("low") > pl.col("close"))
)
print(
    f"\nTotal OHLC violations: {violations.height} ({100 * violations.height / etfs.height:.3f}%)"
)

# How far outside the bound, as a fraction of the price on that row. A count says something
# failed; the size says what kind of thing failed.
breach = violations.select(
    (
        pl.max_horizontal(
            pl.col("close") - pl.col("high"),
            pl.col("open") - pl.col("high"),
            pl.col("low") - pl.col("close"),
            pl.col("low") - pl.col("open"),
            pl.col("low") - pl.col("high"),
        )
        / pl.col("close")
    ).alias("relative")
)["relative"]
if breach.len():
    print(f"Largest breach: {breach.max():.2e} of the close (median {breach.median():.2e})")
    print(f"Float64 epsilon: {float(np.finfo(np.float64).eps):.2e}")

# %% [markdown]
# Three numbers, and they answer different questions. The per-invariant table says *which*
# ordering fails, and reports every check clean. The strict union count says *how many* rows
# fail at least one, and reports several hundred. The breach size says *by how much*, and it
# is the one that settles which of the two to believe.
#
# The largest breach in the whole panel is about the size of float64 epsilon, printed beside
# it: the smallest relative difference a double-precision number can represent. On those rows
# the close and the high are the same price, and the adjustment arithmetic left them one bit
# apart. An adjusted panel multiplies each of the four price fields by the same cumulative
# ratio in four separate operations, and those four products do not round identically.
# Nothing is wrong with the data, and the tolerant check is the one telling the truth.
#
# The strict comparison is not wrong about what it measured; it is wrong about what that
# means. `high >= close` on two independently rounded products fails on ties, so it counts
# ties as violations. A check written that way reports on its own arithmetic, and it will do
# so on any adjusted price panel, in every dataset in this chapter and yours.
#
# This is worth dwelling on because the plausible explanation is the wrong one. Per-field
# vendor rounding would also produce failed ordering checks, and it would produce them at a
# size a price could notice - a fraction of a cent, not a fraction of a trillionth. The count
# alone cannot tell the two apart, and a reader who stops at the count will believe whichever
# story they were told. Measuring the breach is what separates them, and it is three lines.

# %% [markdown]
# ## 5. Liquidity across the groups
#
# Average daily volume varies by more than an order of magnitude across groups. That
# spread drives transaction-cost assumptions in later chapters: a rotation that trades
# the currency bucket cannot assume the fills a broad-equity bucket gets.

# %%
by_group_vol = (
    etfs.join(groups.select(["symbol", "group"]), on="symbol", how="left")
    .group_by("group")
    .agg(pl.col("volume").mean().alias("avg_volume"))
    .sort("avg_volume", descending=True)
)

print("=== Average daily volume by group ===")
for row in by_group_vol.iter_rows(named=True):
    print(f"  {row['group']:<24} {row['avg_volume']:>14,.0f} shares/day")

# %%
fig = go.Figure(
    go.Bar(
        x=by_group_vol["avg_volume"].to_list(),
        y=by_group_vol["group"].to_list(),
        orientation="h",
        marker_color=COLORS["copper"],
    )
)
fig.update_layout(
    title="Average daily share volume by group",
    xaxis_title="Shares/day (log scale)",
    xaxis_type="log",
    yaxis=dict(autorange="reversed"),
    height=420,
)
show_plotly_with_alt(
    fig,
    "A horizontal bar chart of average daily share volume by group on a logarithmic axis. "
    "The broad US equity group sits at the far right and the currency group at the far "
    "left, with the rest spread between them.",
)

# %% [markdown]
# The axis is logarithmic because the spread demands it: average daily volume runs over more
# than an order of magnitude from the broad US equity group down to the currency group. Any
# cost or capacity assumption applied uniformly across this universe is wrong at one end of it.

# %% [markdown]
# ## 6. Price levels
#
# ETF price levels span a wide range — from single digits to several thousand dollars —
# which is why later work compares *returns*, not price levels, across the universe.

# %%
latest = etfs.filter(pl.col("timestamp") == full_end)
price_dist = latest.select(
    pl.col("close").min().alias("min_price"),
    pl.col("close").max().alias("max_price"),
    pl.col("close").median().alias("median_price"),
    pl.col("close").mean().alias("mean_price"),
)
print("=== Price distribution (latest date) ===")
price_dist

# %% [markdown]
# ## 7. Loading a subset or using the ml4t-data library
#
# `load_etfs(symbols=[...])` filters the panel to a subset; `ETFDataManager` (loaded in
# §3) is the config-driven entry point used by the production download/refresh workflow in
# `data/etfs/market/`.

# %%
spy = load_etfs(symbols=["SPY"])
print(f"SPY via loader: {spy.shape}")

# %%
configured = sum(len(group["symbols"]) for group in etf_mgr.config.tickers.values())
print(f"ETFDataManager loaded from {config_path.relative_to(REPO_ROOT)}")
print(f"  Provider:           {etf_mgr.config.provider}")
print(f"  Date range:         {etf_mgr.config.start} to {etf_mgr.config.end}")
print(f"  Configured symbols: {configured} across {len(etf_mgr.config.tickers)} groups")

# %% [markdown]
# ## Key takeaways
#
# - **Ask which convention a `close` column follows before computing a return.** Here it is
#   already split- and dividend-adjusted, so no further adjustment is needed. The equities
#   panel in `01_us_equities_eda` ships raw and adjusted side by side under different names,
#   so the column called `close` means a different thing in each file.
# - **A universe that fills and holds was chosen; one that rises and falls was collected.**
#   The same year-by-year count that exposed a collection artifact in the equities panel
#   confirms the opposite here, and the shape of the curve is what distinguishes them.
# - **Check the panel against the dictionary that built it, both ways.** Symbols in the
#   config with no prices, and prices with no group, are different failures with different
#   causes, and a single count of matches hides both.
# - **A failed check needs a magnitude before it needs an explanation.** The count of ordering
#   violations here looks like a data-quality finding and is an artifact of comparing floats
#   with `>=`. The same count, at a size a price could notice, would be a real defect. Measure
#   the breach before writing the sentence that explains it.
# - **A spread in liquidity is a constraint on the strategy, not a footnote about the data.**
#   The groups here differ by more than an order of magnitude in daily volume, so a rotation
#   that can trade one bucket at negligible cost cannot assume the same fills in another.
#
# **Known limitations.** The universe is fixed and
# every member is still quoted, so this panel says nothing about ETF closures - a real ETF universe does lose members, and a backtest that
# selects from this pool inherits the survivorship its construction removed. Volume is in
# shares rather than notional, so it is not comparable across price levels without converting.
# And the group classification comes from the download config, which is a choice made by this
# repository rather than a standard taxonomy.
#
# **Next**: `13_data_quality_framework` runs systematic checks across these datasets, and
# `15_survivorship_bias_detection` works the equity panel this one is the contrast to.
# Chapter 6 constructs a trading universe from this candidate pool.
