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
# # Crypto Perps: Exploratory Data Analysis
#
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Profile the Binance Futures hourly OHLCV dataset for 19 perpetual contracts
# alongside the 8-hourly Premium Index that captures the perpetual–spot
# basis. The notebook anchors data shape, units, coverage, and OHLC integrity
# before the strategy work begins in Chapter 6.
#
# ## Learning Objectives
#
# - Load hourly OHLCV and 8-hourly premium index data via the canonical loaders.
# - Document premium-index units (decimal, multiply by 100 for percent).
# - Quantify cross-frequency join coverage between OHLCV and premium data.
# - Run OHLC invariant checks and inspect for time-stamp gaps.
#
# ## Book Reference
#
# §2.2, "The asset-class market data landscape" - the digital-asset part of it.
#
# ## Prerequisites
#
# - Familiarity with daily OHLCV equity data (`01_us_equities_eda`).
# - The Binance Futures parquet at `$ML4T_DATA_PATH/crypto/` (OHLCV + premium).
# - Methodology continues in `11_crypto_premium_analysis`; the case-study
#   pipeline lives under `case_studies/crypto_perps_funding/`.

# %%
"""Crypto perps EDA: hourly OHLCV and premium index exploration."""

import plotly.graph_objects as go
import polars as pl

from data import load_crypto_perps, load_crypto_premium
from utils.data_quality import check_ohlc_invariants, per_asset_stats
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# ### Declared parameters
#
# `MAX_SYMBOLS` of zero loads every symbol. CI overrides it to reduce the notebook, so it has
# to be passed to the loaders rather than sit beside them unused.

# %% tags=["parameters"]
MAX_SYMBOLS = 0

# Half-width, in percent, of the premium histogram's x-axis. The distribution's standard
# deviation is around a tenth of a percent, so this is several standard deviations either way
# and still a small fraction of the range the tail reaches.
PREMIUM_AXIS_PCT = 0.5

# %% [markdown]
# ## 1. Load and Inspect OHLCV
#
# Hourly OHLCV data from Binance Futures for 19 cryptocurrencies.
# Trading is 24/7 (8,760 hours/year vs 252 days for equities).

# %%
ohlcv = load_crypto_perps(frequency="1h", max_symbols=MAX_SYMBOLS)

print("=== OHLCV Dataset ===")
print(f"Shape: {ohlcv.shape}")
print(f"Columns: {ohlcv.columns}")

# %%
# Schema overview
print("\nSchema:")
for col, dtype in ohlcv.schema.items():
    print(f"  {col}: {dtype}")

# %% [markdown]
# ## 2. Coverage Summary

# %%
# Symbols and date range
symbols = ohlcv["symbol"].unique().sort().to_list()
print("=== Coverage ===")
print(f"Number of symbols: {len(symbols)}")
print(f"\nSymbols: {', '.join(symbols)}")

# %%
# Overall date range
date_range = ohlcv.select(
    [
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        pl.col("timestamp").n_unique().alias("unique_hours"),
    ]
)

print(f"\nDate range: {date_range['start'][0]} to {date_range['end'][0]}")
print(f"Unique hours: {date_range['unique_hours'][0]:,}")

# %%
# Per-symbol statistics
symbol_stats = per_asset_stats(
    ohlcv,
    time_col="timestamp",
    asset_col="symbol",
    price_col="close",
    volume_col="volume",
)

print("\nSymbol Statistics (top 5 by volume):")
symbol_stats.sort("avg_volume", descending=True).head(5)

# %% [markdown]
# ### Contracts list at different dates
#
# The universe is not fixed: BTC and a handful of majors are present from 2020,
# and newer contracts (SUI lists in 2023) switch on later. Counting distinct
# symbols per month makes the staggered onboarding explicit, and it is coverage that any
# cross-sectional signal has to account for.

# %%
active_by_month = (
    ohlcv.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))
    .group_by("month")
    .agg(pl.col("symbol").n_unique().alias("contracts"))
    .sort("month")
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=active_by_month["month"].to_list(),
        y=active_by_month["contracts"].to_list(),
        mode="lines",
        line=dict(color=COLORS["blue"], width=2, shape="hv"),
        name="Contracts with data",
    )
)
fig.update_layout(
    title="Perpetual contracts with data, by month",
    xaxis_title="Month",
    yaxis_title="Contracts",
    yaxis_range=[0, 20],
    height=420,
)
show_plotly_with_alt(
    fig,
    "A step line counting how many perpetual contracts have data in each month, from 2020 to the end of the sample. It rises in discrete jumps as new contracts list and does not fall.",
)

# %% [markdown]
# ## 3. Premium Index Data
#
# The premium index measures how far the perpetual is trading from the underlying index. It is
# often described as the simple relative difference between the two prices, and that
# description is close enough to build intuition on but wrong in a way that matters later:
# Binance computes it from the *impact bid and ask* against the price index, with a term that
# is zero whenever the index sits between them. The exact definition and its consequence are in
# the data-quality section below.
#
# - Positive premium: the perpetual is bid above the index, and longs pay shorts
# - Negative premium: the perpetual is offered below it, and shorts pay longs
#
# ### Units
#
# Premium values are stored as **decimals**, not percentages: a value of one thousandth means
# a tenth of one percent. Multiply by a hundred before displaying. Nothing in the column name
# records the convention, so the printed range below is what settles it - the magnitudes are
# only plausible under one reading.

# %%
premium = load_crypto_premium(frequency="8h", max_symbols=MAX_SYMBOLS)
print("=== Premium Dataset ===")
print(f"Shape: {premium.shape}")
print(f"Columns: {premium.columns}")

# %%

premium_range = premium.select(
    [
        pl.col("premium_index_close").min().alias("min"),
        pl.col("premium_index_close").max().alias("max"),
        pl.col("premium_index_close").mean().alias("mean"),
        pl.col("premium_index_close").std().alias("std"),
    ]
)

print("\nPremium Range (decimals):")
print(f"  Mean: {premium_range['mean'][0]:.6f} ({premium_range['mean'][0] * 100:.4f}%)")
print(f"  Std:  {premium_range['std'][0]:.6f} ({premium_range['std'][0] * 100:.4f}%)")
print(f"  Min:  {premium_range['min'][0]:.6f} ({premium_range['min'][0] * 100:.4f}%)")
print(f"  Max:  {premium_range['max'][0]:.6f} ({premium_range['max'][0] * 100:.4f}%)")

# %% [markdown]
# ### The basis is small, until it is not
#
# The premium sits within a fraction of a percent almost all the time, and the printed range
# above shows how far the extremes reach. The distribution is not symmetric around that central
# mass, and the asymmetry is what the funding strategy in the case study is built on, so it is
# worth drawing rather than tabulating.
#
# The axis is clipped so the central mass is legible at all. Unclipped, a single dislocation
# sets the range and the observations the strategy actually trades collapse into one bar. The
# clip half-width is declared as a parameter rather than typed into the figure, and the
# annotation carries the number the clipping hides, so the choice conceals nothing.

# %%
premium_pct = (premium["premium_index_close"] * 100).to_list()

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=premium_pct,
        xbins=dict(start=-PREMIUM_AXIS_PCT, end=PREMIUM_AXIS_PCT, size=PREMIUM_AXIS_PCT / 40),
        marker_color=COLORS["slate"],
        name="Premium (%)",
    )
)
fig.add_vline(x=0, line_color=COLORS["amber"], line_width=1)
fig.add_annotation(
    x=-PREMIUM_AXIS_PCT,
    y=1,
    xref="x",
    yref="paper",
    text=f"tail reaches {min(premium_pct):.1f}%",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color=COLORS["copper"]),
)
fig.update_layout(
    title="Premium index distribution, 8-hourly, axis clipped",
    xaxis_title="Premium (%)",
    yaxis_title="8-hour observations",
    xaxis_range=[-PREMIUM_AXIS_PCT, PREMIUM_AXIS_PCT],
    height=420,
)
show_plotly_with_alt(
    fig,
    f"A histogram of the eight-hourly premium index in percent, with a vertical line at zero "
    f"and the horizontal axis clipped to plus or minus {PREMIUM_AXIS_PCT} percent. The mass is "
    f"a single narrow spike centred on zero, taller than its neighbours, tapering within about "
    f"a fifth of a percent in both directions. An annotation at the top left names how far the "
    f"negative tail runs beyond the clipped range.",
)

# %% [markdown]
# ## 4. Data Quality
#
# These are raw exchange bars, not an adjusted panel, so the OHLC relations are exact rather
# than approximate: the high is a maximum of prices that were printed and the low a minimum of
# the same. Any breach at all is a defect in the capture, so the check reports a count rather
# than a percentage against a tolerance.

# %%
# OHLC invariants
invariants = check_ohlc_invariants(ohlcv)
print(f"OHLC invariants over {len(ohlcv):,} raw hourly bars:")
for row in invariants.iter_rows(named=True):
    breaches = round((100 - row["valid_pct"]) / 100 * len(ohlcv))
    status = "[OK]" if breaches == 0 else "[FAIL]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.4f}%  ({breaches} bars outside)")

# %%
ohlcv_nulls = ohlcv.null_count().sum_horizontal()[0]
premium_nulls = premium.null_count().sum_horizontal()[0]
print(f"Null values: OHLCV={ohlcv_nulls}, Premium={premium_nulls}")

# %% [markdown]
# ### A point mass, and how to tell what makes one
#
# Neither file has a null in it. That is worth exactly as much as the encoding behind it,
# because absence recorded as a null is visible to a null check and absence recorded as a
# valid-looking number is not. So it is worth looking for the second kind.
#
# The premium index has an obvious candidate. It is a continuous quantity stored to eight
# decimal places with a standard deviation around a thousandth, and yet a large share of its
# closes are exactly zero - a value that a continuous distribution at that resolution should
# essentially never produce.

# %%
_close = pl.col("premium_index_close")
_zero_rows = premium.filter(_close == 0)
_nonzero = premium.filter(_close != 0)["premium_index_close"].abs()

print(
    f"Premium closes of exactly zero: {_zero_rows.height:,} of {len(premium):,} "
    f"({100 * _zero_rows.height / len(premium):.2f}%)"
)
print(f"Smallest non-zero magnitude in the file: {_nonzero.min():.10f}")
print(f"Standard deviation of the close:         {premium['premium_index_close'].std():.6f}")
print(
    f"  so the storage quantum is 1/{premium['premium_index_close'].std() / _nonzero.min():,.0f} "
    f"of a standard deviation"
)

# %% [markdown]
# One observation in seven sits on a value the rest of the distribution never visits. The
# tempting reading is that these are missing values written as zero - a placeholder the schema
# accepts, which is why the null count sees nothing.
#
# **That reading is wrong, and the way it goes wrong is the point of this section.** Before
# treating a point mass as a data defect, read the definition of the quantity. Binance computes
# the premium index as
#
# $$\frac{\max(0,\ \text{Impact Bid} - \text{Price Index}) - \max(0,\ \text{Price Index} - \text{Impact Ask})}{\text{Price Index}}$$
#
# Both numerator terms are zero whenever the price index lies between the impact bid and the
# impact ask. The formula has a **dead zone** exactly the width of the impact spread, and
# inside it the output is not approximately zero but exactly zero. A point mass there is
# what the definition predicts, and it means "no premium is owed", which is a measurement
# rather than the absence of one.
#
# ### Distinguishing a dead zone from missing data
#
# Both produce a point mass, so the count cannot separate them. What separates them is what the
# rate correlates with. Non-publication is an operational failure and has no reason to sort
# itself by market structure. A dead zone whose width is the impact spread has every reason to:
# the spread is wide where liquidity is thin, so the zone is easier to sit inside.

# %%
zero_rate_by_symbol = (
    premium.group_by("symbol")
    .agg((_close == 0).mean().alias("zero_rate"))
    .join(
        ohlcv.with_columns((pl.col("close") * pl.col("volume")).alias("dollar_volume"))
        .group_by("symbol")
        .agg(pl.col("dollar_volume").median().alias("median_hourly_dollar_volume")),
        on="symbol",
        how="inner",
    )
    .sort("zero_rate")
)
print("Share of premium closes at exactly zero, against a liquidity proxy:")
zero_rate_by_symbol

# %% [markdown]
# The ordering is close to monotone across the whole universe, from the most liquid contract to
# the thinnest, and it spans a factor of more than fifty. That is the dead-zone prediction and
# it is not something an outage would produce.
#
# The remaining checks all point the same way once the formula is known. The rate declines
# steadily year on year, which reads as spreads tightening rather than as a fault being
# repaired. Fewer than one percent of the affected rows have all four premium fields at zero,
# so it is not a blank record. And the zeros are present in the exchange's own one-minute
# source, so nothing in the download or the resampling creates them.

# %%
_all_four_zero = premium.filter(
    (pl.col("premium_index_open") == 0)
    & (pl.col("premium_index_high") == 0)
    & (pl.col("premium_index_low") == 0)
    & (_close == 0)
).height

zero_by_year = (
    premium.with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by("year")
    .agg(pl.len().alias("observations"), (_close == 0).mean().alias("share_exactly_zero"))
    .sort("year")
)
print(f"Rows where all four premium fields are zero: {_all_four_zero:,} of {_zero_rows.height:,}")
print(f"Symbols affected: {_zero_rows['symbol'].n_unique()} of {premium['symbol'].n_unique()}")
zero_by_year

# %% [markdown]
# ### What to carry forward
#
# The zeros are measurements, so nothing should be filtered on them. Treating them as unknown
# would discard one observation in seven, and a third of the thinnest contracts' history, on
# the strength of a hypothesis the definition refutes.
#
# What does follow is that `premium_index_close` carries a large tie group at exactly zero,
# reaching a third of the observations for the least liquid contracts. Any feature built by
# ranking or standardising this column - a trailing percentile, a z-score - is operating on a
# distribution with a third of its mass on a single point for those symbols, and rank-based
# transforms handle ties in ways that are worth choosing deliberately rather than inheriting.
#
# And the general lesson, which outlives this dataset: **a point mass in a derived quantity is
# a property of its formula before it is a defect in its data.** Reading the definition costs
# minutes. Every diagnostic computed above is consistent with both explanations, so no amount
# of measurement on this file alone would have settled it - the frequency, the non-zero
# extremes, the yearly decline and the null count are all equally compatible with a dead zone
# and with a placeholder. The one measurement that discriminates is the one the formula tells
# you to make.

# ### Gaps in the hourly grid
#
# The check runs over every symbol rather than a reference one. Checking BTC alone would sample
# the single contract least likely to have gaps - it is the most liquid and the longest-listed -
# and report its cleanliness as the dataset's. The thin and recently listed contracts are where
# a gap is plausible, so they are the reason to run the check at all.

# %%
gaps = (
    ohlcv.sort(["symbol", "timestamp"])
    .with_columns(
        pl.col("timestamp").diff().dt.total_hours().over("symbol").alias("hours_since_previous")
    )
    .filter(pl.col("hours_since_previous") > 1)
)

gaps_by_symbol = (
    gaps.group_by("symbol")
    .agg(
        pl.len().alias("gaps"),
        pl.col("hours_since_previous").max().alias("longest_gap_hours"),
        (pl.col("hours_since_previous") - 1).sum().alias("missing_hours"),
    )
    .sort("missing_hours", descending=True)
)

print(f"Symbols with at least one gap: {gaps_by_symbol.height} of {ohlcv['symbol'].n_unique()}")
if gaps_by_symbol.height:
    print(f"Longest single gap: {gaps['hours_since_previous'].max():.0f} hours")
gaps_by_symbol.head(10)

# %% [markdown]
# ## 5. Joining OHLCV and Premium
#
# The two frames are published on different clocks: OHLCV every hour, the premium index every
# eight. Joining them on an exact timestamp match therefore lands a premium on one hourly bar
# in eight and leaves the rest null.
#
# That is worth doing once, because the number it produces looks like a coverage statistic and
# is not one. It measures the ratio of the two publication frequencies. Nothing is missing from
# the premium file, and no amount of better data would raise it.

# %%
exact = ohlcv.join(premium, on=["timestamp", "symbol"], how="left")
_matched = exact.filter(pl.col("premium_index_close").is_not_null()).height

print(f"OHLCV rows:   {len(ohlcv):,}")
print(f"Premium rows: {len(premium):,}")
print(f"Exact-timestamp match: {_matched:,} of {len(exact):,} ({100 * _matched / len(exact):.1f}%)")
print(f"Ratio of the two publication frequencies: 1 in {len(ohlcv) / len(premium):.1f}")

# %% [markdown]
# The match rate and the frequency ratio are the same number, which is the tell. Reporting the
# first as a data-quality finding would describe the calendar rather than the data.
#
# **What downstream work needs is the premium in force at each hour**, which is the most recent
# one that was actually available at that hour. Those are different things, and the difference
# is a look-ahead bug waiting to happen.
#
# Binance stamps each bar with the time it **opened**. A premium bar stamped midnight covers
# midnight to eight, so its close is not known until eight - and joining it to hourly rows from
# midnight onward hands the notebook eight hours of the future. The case study states the same
# convention and corrects for it the same way
# (`case_studies/crypto_perps_funding/02_labels.py`, section B).
#
# So the premium timestamps are advanced by one bar length first. After that shift a row's
# timestamp is the moment its close became available, and an as-of join on it is point-in-time
# correct rather than merely looking it.
#
# Both frames are sorted by symbol and then timestamp, which is what an as-of join within
# groups requires. `check_sortedness=False` asserts that rather than asking polars to verify
# it, which it cannot do once `by` groups are involved; left at its default it warns to that
# effect on every run and the warning lands in the rendered notebook.

# %%
PREMIUM_BAR_HOURS = 8

premium_available = premium.with_columns(
    (pl.col("timestamp") + pl.duration(hours=PREMIUM_BAR_HOURS)).alias("timestamp"),
    pl.col("timestamp").alias("premium_bar_opened"),
).sort(["symbol", "timestamp"])

combined = ohlcv.sort(["symbol", "timestamp"]).join_asof(
    premium_available,
    on="timestamp",
    by="symbol",
    strategy="backward",
    check_sortedness=False,
    suffix="_premium",
)

missing_premium = combined.filter(pl.col("premium_index_close").is_null()).height
print(
    f"As-of join leaves {missing_premium:,} of {len(combined):,} hourly bars without a premium "
    f"({100 * missing_premium / len(combined):.2f}%)"
)

# %% [markdown]
# That count alone would be a poor coverage check, and it is worth saying why rather than
# quoting it as a result. A backward as-of join with no tolerance matches every hour after a
# symbol's first publication to *something*. If the exchange skipped a settlement, the join
# does not report a gap; it silently carries the previous value across it. Zero unmatched rows
# is therefore consistent with complete data and with a file full of holes.
#
# What distinguishes them is staleness. A premium published on an eight-hour grid should never
# be more than eight hours old at the moment it is read.

# %%
staleness = combined.drop_nulls("premium_index_close").with_columns(
    (pl.col("timestamp") - pl.col("premium_bar_opened")).dt.total_hours().alias("premium_age_hours")
)
_stale = staleness.filter(pl.col("premium_age_hours") > 2 * PREMIUM_BAR_HOURS)

print(
    f"Premium age when read: median {staleness['premium_age_hours'].median():.0f}h, "
    f"max {staleness['premium_age_hours'].max():.0f}h"
)
print(
    f"Hourly bars reading a premium more than two settlement periods old: {_stale.height:,} "
    f"({100 * _stale.height / staleness.height:.2f}%)"
)
if _stale.height:
    print(
        _stale.group_by("symbol")
        .agg(pl.len().alias("bars"), pl.col("premium_age_hours").max().alias("worst_age_hours"))
        .sort("bars", descending=True)
        .head(10)
    )

# %% [markdown]
# The staleness check finds what the match count could not. Almost every hourly bar matched
# something, and a fraction of a percent of them matched a premium that was days or weeks out
# of date - one symbol reading a value more than two months old, and a shorter outage that
# most of the universe shares on the same dates.
#
# Both are interior gaps in the premium file: settlements the index skipped while the contract
# went on trading. The case study's `02_labels.py` names the same property from the other side,
# noting that the index is absent at some settlements where the contract traded and that the
# two are therefore not the same set of rows.
#
# The lesson generalises past this file. An as-of join never fails, so it never reports a gap;
# it reports the last thing it found. Any coverage claim built on its null count is a claim
# about whether the series ever started, not about whether it kept going. What detects a gap is
# comparing the age of the match against the schedule the source publishes on.

# %% [markdown]
# What remains unmatched is a different kind of absence from the one the exact join reported,
# and the next cell checks which kind. If these are hours before a symbol's first premium
# publication, the gap is a start-of-history edge and closes on its own. If they are scattered
# through the middle of a symbol's life, the premium file has holes.

# %%
_first_premium = premium.group_by("symbol").agg(pl.col("timestamp").min().alias("premium_starts"))
_unmatched = (
    combined.filter(pl.col("premium_index_close").is_null())
    .join(_first_premium, on="symbol", how="left")
    .with_columns((pl.col("timestamp") < pl.col("premium_starts")).alias("before_first_premium"))
)

if _unmatched.height:
    _before = _unmatched.filter(pl.col("before_first_premium")).height
    print(f"Unmatched bars before the symbol's first premium: {_before:,} of {_unmatched.height:,}")
    print("Remaining unmatched bars by symbol:")
    print(
        _unmatched.filter(~pl.col("before_first_premium"))
        .group_by("symbol")
        .agg(pl.len().alias("bars"), pl.col("timestamp").min().alias("earliest"))
        .sort("bars", descending=True)
        .head(10)
    )
else:
    print("Every hourly bar carries a premium.")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Crypto trades continuously**, so an hourly grid has roughly 8,760 rows per symbol-year
#    against 252 daily rows for equities. Every count above follows from that, and none of them
#    is quoted here, because they move whenever the snapshot is extended.
#
# 2. **The universe is not fixed.** Contracts list at different dates and the count only ever
#    rises, so any cross-sectional signal is computed over a membership that changes underneath
#    it. The monthly chart makes the staggering explicit rather than leaving it to be
#    discovered downstream.
#
# 3. **The premium index is stored as a decimal, not a percentage.** Getting that backwards
#    scales every basis figure by a hundred, and nothing in the column name records which
#    convention the file uses. The printed range is what settles it.
#
# 4. **A cross-frequency exact join reports the calendar, not the data.** OHLCV is hourly and
#    the premium index is eight-hourly, so an exact-timestamp join matches one bar in eight -
#    and that fraction is the ratio of the publication frequencies, not a coverage problem.
#
# 5. **An as-of join needs two corrections before it means anything.** Binance stamps a bar
#    with the time it *opened*, so joining on the raw timestamp hands each hour a premium that
#    was not known for another eight; the premium clock is advanced by one bar length first.
#    And an as-of join never fails - it returns the last value it found - so its null count
#    says only whether the series ever started. Measuring the age of each match against the
#    publication grid is what finds the interior gaps, and it finds one symbol reading a
#    premium more than two months stale.
#
# 6. **The gap check covers every symbol**, because the thin and recently listed contracts are
#    the reason it exists. BTC is the most liquid and longest-listed of the nineteen, so a
#    check confined to it samples the contract with least to find and reports that as the
#    dataset's condition.
#
# 7. **A point mass is a property of a formula before it is a defect in data.** One premium
#    close in seven is exactly zero, which looks like a placeholder and is not. Binance's
#    definition has a dead zone the width of the impact spread, and inside it the index is
#    exactly zero by construction. The measurement that settles it is the one the formula
#    predicts: the zero rate tracks liquidity across the universe, from well under one percent
#    on the most traded contract to a third on the thinnest. Frequency, non-zero extremes and
#    a yearly decline are all equally consistent with a placeholder, so none of them could have
#    settled it. Read the definition first.
#
# 8. **These are raw exchange bars, so the OHLC relations are exact.** The check reports the
#    number of bars outside each bound rather than a percentage against a tolerance, because on
#    unadjusted data there is no rounding for a tolerance to absorb.
#
# ## Next Steps
#
# - `11_crypto_premium_analysis`: Premium dynamics, basis seasonality, and
#   alignment to the 8-hourly funding cadence.
# - Chapter 8: Feature engineering for premium signals
#   (`case_studies/crypto_perps_funding/03_financial_features.py`).
# - Chapter 16: Backtests for the funding-arbitrage case study.
