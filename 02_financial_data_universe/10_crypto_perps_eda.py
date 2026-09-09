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
# The premium index measures the spread between perpetual futures and spot prices:
#
# **Premium = (Perpetual Price - Spot Price) / Spot Price**
#
# - Positive premium: Futures above spot (bullish sentiment)
# - Negative premium: Futures below spot (bearish sentiment)
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
    "A histogram of the eight-hourly premium index in percent, with a vertical line at zero and the horizontal axis clipped to plus or minus two percent. The mass is a narrow peak close to zero, slightly to its positive side. An annotation at the left edge names how far the tail runs beyond the clipped range.",
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
# ### A null count is not a completeness check
#
# Neither file has a null in it, and that is worth exactly as much as the encoding behind it.
# Absence recorded as a null is visible to the check above; absence recorded as a valid-looking
# number is not.
#
# The premium index gives a way to test for the second kind. It is a continuous quantity stored
# to eight decimal places, and its standard deviation is around a thousandth. A quantum that
# small against a spread that large means landing on exactly zero should be vanishingly rare -
# so if exact zeros are common, they are not measurements.

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
# Fourteen percent of the observations sit on a value that a smooth distribution at this
# resolution would essentially never produce. They are not rounding, and they are not the
# perpetual and spot happening to agree to the eighth decimal place.
#
# The bars they sit in settle it. If the premium were genuinely zero for the whole eight-hour
# period, the open, high and low of that bar should be at or near zero too.

# %%
_shape = _zero_rows.select(
    (pl.col("premium_index_open") == 0).mean().alias("open_also_zero"),
    (pl.col("premium_index_high") == 0).mean().alias("high_also_zero"),
    (pl.col("premium_index_low") == 0).mean().alias("low_also_zero"),
)
print("On the rows whose close is exactly zero, share where the other prices are also zero:")
print(_shape)
print(f"Symbols affected: {_zero_rows['symbol'].n_unique()} of {premium['symbol'].n_unique()}")
print(f"Spanning {_zero_rows['timestamp'].min()} to {_zero_rows['timestamp'].max()}")

# %% [markdown]
# The high and the low are almost never zero on those rows, so the index moved during the
# period and then "closed" at a value it never plausibly reached. That is a placeholder written
# into a price column, not a price.
#
# **This is why the null count passed.** The absence is encoded as a number the schema accepts,
# so every completeness check that looks for nulls reports the file as complete. It affects
# every symbol and runs the length of the sample, so it cannot be dismissed as an early-history
# artifact either.
#
# What follows for the case study in `case_studies/crypto_perps_funding/` is that a premium of
# exactly zero has to be treated as unknown rather than as a basis of zero. The two are opposite
# instructions to a strategy that trades the basis: one says stand aside, the other says the
# spread has closed.

# %% [markdown]
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
        pl.col("hours_since_previous").sum().alias("total_missing_hours"),
    )
    .sort("total_missing_hours", descending=True)
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
# one published at or before that hour. That is an as-of join, and it is point-in-time correct
# by construction: it never reaches forward to a value that had not been published yet.
#
# Both frames are sorted by symbol and then timestamp, which is what an as-of join within
# groups requires. `check_sortedness=False` asserts that rather than asking polars to verify
# it, which it cannot do once `by` groups are involved; left at its default it warns to that
# effect on every run and the warning lands in the rendered notebook.

# %%
combined = ohlcv.sort(["symbol", "timestamp"]).join_asof(
    premium.sort(["symbol", "timestamp"]),
    on="timestamp",
    by="symbol",
    strategy="backward",
    check_sortedness=False,
)

missing_premium = combined.filter(pl.col("premium_index_close").is_null()).height
print(
    f"As-of join leaves {missing_premium:,} of {len(combined):,} hourly bars without a premium "
    f"({100 * missing_premium / len(combined):.2f}%)"
)

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
#    The number to compute instead is the as-of join, which carries the premium in force at
#    each hour and is point-in-time correct by construction.
#
# 5. **The gap check covers every symbol**, because the thin and recently listed contracts are
#    the reason it exists. BTC is the most liquid and longest-listed of the nineteen, so a
#    check confined to it samples the contract with least to find and reports that as the
#    dataset's condition.
#
# 6. **A file with no nulls is not a complete file.** Fourteen percent of premium closes are
#    exactly zero, on a quantity stored to eight decimals with a spread three orders of
#    magnitude wider - a value a smooth distribution would essentially never produce. The same
#    bars have non-zero highs and lows, so the index moved and then "closed" where it never
#    traded. Absence is encoded as a number the schema accepts, which is why the null check
#    reports the file as complete. Downstream, a zero premium has to mean unknown rather than
#    a basis of zero.
#
# 7. **These are raw exchange bars, so the OHLC relations are exact.** The check reports the
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
