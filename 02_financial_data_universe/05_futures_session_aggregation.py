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
# # Futures Session Aggregation: Hourly to Daily
#
# **Docker image**: `ml4t`
#
# **Purpose**: Convert hourly continuous futures bars (Databento, UTC) to
# session-aware daily bars that respect the 4:00 PM Central Time CME session
# boundary, applying ratio back-adjustment to eliminate roll-induced price
# gaps.
#
# **Learning objectives**:
#
# - Understand why CME session dates differ from UTC calendar dates and how
#   Sunday-evening bars are counted into Monday's session.
# - Apply ratio (multiplicative) back-adjustment to a continuous series so
#   percentage returns are preserved across rolls.
# - Aggregate hourly bars to session-correct daily OHLCV across all 30 products
#   and three tenors (front month, first deferred, second deferred).
#
# **Book reference**: §2.2, "The asset-class market data landscape" - the futures part of
# it. `06_futures_continuous` compares the adjustment methods.
#
# **Prerequisites**: `data` package on `PYTHONPATH`; hourly continuous parquet
# present at `ML4T_DATA_PATH/futures/market/continuous/`. See
# [`06_futures_continuous`](06_futures_continuous.ipynb) for the teaching
# explanation of ratio vs Panama adjustment.

# %%
"""Session-aware aggregation of hourly futures to daily bars."""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_cme_futures
from utils import ML4T_DATA_PATH
from utils.paths import REPO_ROOT, get_chapter_dir
from utils.style import COLORS, ml4t_palette, show_plotly_with_alt


def _rel(path):
    """Repo-relative display path (keeps absolute machine paths out of outputs)."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


# %% [markdown]
# ### Where the aggregated bars are written
#
# `WRITE_TO_DATA=1` materialises the canonical daily parquet under `ML4T_DATA_PATH`, which is
# what the later chapters and case studies read. The default writes to a chapter-local output
# directory instead, so running this notebook to see how it works cannot overwrite the data
# everything downstream depends on.

# %% tags=["parameters"]
WRITE_TO_DATA = os.environ.get("WRITE_TO_DATA", "0") == "1"

# %%
OUTPUT_DIR = (
    ML4T_DATA_PATH / "futures" / "market" / "continuous" / "daily"
    if WRITE_TO_DATA
    else get_chapter_dir(2) / "output" / "futures_daily"
)

# %% [markdown]
# ## 1. CME Session Boundaries
#
# ### Session Definition
#
# CME Globex sessions follow this schedule:
# - **Session Start**: Sunday 5:00 PM CT (for Monday session)
# - **Session End**: 4:00 PM CT (defines the session date)
# - **Daily Maintenance**: 4:00-5:00 PM CT (1-hour break)
#
# ### Why This Matters
#
# If we aggregate by calendar day (midnight UTC), we split a single trading
# session across two days, creating incorrect daily bars:
#
# | Approach | Sunday 11 PM UTC | Monday 3 PM UTC |
# |----------|------------------|-----------------|
# | **Calendar Day (Wrong)** | Sunday | Monday |
# | **CME Session (Correct)** | Monday | Monday |
#
# Both bars fall inside Monday's session, which ends Monday 4 PM CT.

# %%
# Timezone constants
CT = ZoneInfo("America/Chicago")
UTC = ZoneInfo("UTC")

# CME session ends at 4 PM CT
SESSION_END_HOUR_CT = 16  # 4:00 PM


def assign_cme_session_date(ts: datetime) -> datetime:
    """
    Assign CME session date to a UTC timestamp.

    The session date is the date when the session ENDS (4 PM CT).
    A bar at Sunday 11 PM UTC belongs to Monday's session.

    CME closes Friday at 4 PM CT and reopens Sunday 5 PM CT.
    Bars after Friday 4 PM CT still belong to Friday's session —
    they must NOT roll to Saturday.

    Args:
        ts: UTC timestamp

    Returns:
        Session date (as date, no time component)
    """
    # Convert to Central Time
    ts_ct = ts.astimezone(CT)

    # If we're past 4 PM CT, this belongs to tomorrow's session
    if ts_ct.hour >= SESSION_END_HOUR_CT:
        candidate = ts_ct.date() + timedelta(days=1)
        # Friday after 4 PM CT → keep as Friday (no Saturday session)
        # isoweekday: Mon=1, Fri=5, Sat=6
        if candidate.isoweekday() == 6:  # Saturday
            candidate = ts_ct.date()  # Keep as Friday
        session_date = candidate
    else:
        session_date = ts_ct.date()

    return session_date


# %%
# Quick test
test_times = [
    datetime(2024, 1, 7, 23, 0, tzinfo=UTC),  # Sunday 11 PM UTC = Sunday 5 PM CT -> Monday
    datetime(2024, 1, 8, 15, 0, tzinfo=UTC),  # Monday 3 PM UTC = Monday 9 AM CT -> Monday
    datetime(2024, 1, 8, 22, 0, tzinfo=UTC),  # Monday 10 PM UTC = Monday 4 PM CT -> Tuesday
    datetime(
        2024, 1, 12, 22, 0, tzinfo=UTC
    ),  # Friday 10 PM UTC = Friday 4 PM CT -> Friday (NOT Saturday)
]

print("Session Assignment Examples:")
for ts in test_times:
    ts_ct = ts.astimezone(CT)
    session = assign_cme_session_date(ts)
    print(f"  {ts} ({ts_ct.strftime('%a %I:%M %p CT')}) -> Session: {session}")

# %% [markdown]
# ## 2. Load Hourly Continuous Data
#
# We load all products and tenors from the DataBento hourly data.

# %%
hourly = load_cme_futures(continuous=True, frequency="hourly")
products = sorted(hourly["product"].unique().to_list())

print(f"Loaded {len(hourly):,} hourly bars")
print(f"Products: {hourly['product'].n_unique()}")
print(f"Tenors: {sorted(hourly['tenor'].unique().to_list())}")
print(f"Date range: {hourly['timestamp'].min()} to {hourly['timestamp'].max()}")
print(f"Available products: {', '.join(products)}")

# %%
hourly.filter(pl.col("product") == "ES").select(
    "timestamp", "product", "tenor", "open", "high", "low", "close", "volume"
).head(10)

# %% [markdown]
# ## 3. Assign Session Dates
#
# We add a `session_date` column using Polars expressions for efficiency.


# %%
def add_session_date(df: pl.DataFrame) -> pl.DataFrame:
    """Add session_date column based on CME session boundaries.

    Friday after 4 PM CT stays as Friday — CME has no Saturday session.
    """
    return (
        df.with_columns(pl.col("timestamp").dt.convert_time_zone("America/Chicago").alias("ts_ct"))
        .with_columns(
            pl.col("ts_ct").dt.date().alias("_ct_date"),
            (pl.col("ts_ct").dt.hour() >= SESSION_END_HOUR_CT).alias("_after_close"),
            # isoweekday: Mon=1 ... Fri=5, Sat=6, Sun=7
            (pl.col("ts_ct").dt.weekday() == 5).alias("_is_friday"),
        )
        .with_columns(
            # After 4 PM CT → next day, UNLESS it's Friday (no Saturday session)
            pl.when(pl.col("_after_close") & ~pl.col("_is_friday"))
            .then(pl.col("_ct_date") + pl.duration(days=1))
            .otherwise(pl.col("_ct_date"))
            .alias("session_date")
        )
        .drop("ts_ct", "_ct_date", "_after_close", "_is_friday")
    )


# %%
hourly_with_sessions = add_session_date(hourly)

print("Session dates assigned (ES sample):")
hourly_with_sessions.filter(pl.col("product") == "ES").select(
    "timestamp", "session_date", "product", "tenor", "close", "volume"
).head(15)

# %% [markdown]
# Walk a single calendar day for the ES front month: bars with `timestamp <
# 2024-01-08 22:00 UTC` carry session_date 2024-01-08; bars at or after 22:00
# UTC (= 16:00 CT, the close) carry session_date 2024-01-09.

# %%
(
    hourly_with_sessions.filter(
        (pl.col("product") == "ES")
        & (pl.col("tenor") == 0)
        & (pl.col("timestamp").dt.date() == pl.lit("2024-01-08").str.to_date())
    )
    .sort("timestamp")
    .select("timestamp", "session_date", "close", "volume")
)

# %% [markdown]
# The same point is clearer on the tape. Below, four days of ES front-month
# hourly closes are colored by the session each bar belongs to, in Central Time.
# The Friday session ends at 16:00 CT; there is no Saturday session; trading
# reopens Sunday 17:00 CT — and those Sunday-evening bars carry **Monday's**
# color, not Sunday's. Calendar day and session date part ways every weekend.

# %%
_boundary_window = (
    hourly_with_sessions.filter(
        (pl.col("product") == "ES")
        & (pl.col("tenor") == 0)
        & (pl.col("timestamp").dt.date() >= pl.lit("2024-01-05").str.to_date())
        & (pl.col("timestamp").dt.date() <= pl.lit("2024-01-09").str.to_date())
    )
    .with_columns(pl.col("timestamp").dt.convert_time_zone("America/Chicago").alias("ts_ct"))
    .sort("timestamp")
)

_sessions = _boundary_window["session_date"].unique().sort().to_list()
_session_colors = dict(zip(_sessions, ml4t_palette(len(_sessions), categorical=True)))

fig = go.Figure()
for sess in _sessions:
    seg = _boundary_window.filter(pl.col("session_date") == sess)
    fig.add_trace(
        go.Scatter(
            x=seg["ts_ct"].to_list(),
            y=seg["close"].to_list(),
            mode="lines+markers",
            line=dict(color=_session_colors[sess], width=1.5),
            marker=dict(size=5),
            name=str(sess),
        )
    )
fig.update_layout(
    title="Hourly bars coloured by CME session date",
    xaxis_title="Timestamp (Central Time)",
    yaxis_title="ES front-month close",
    height=420,
    legend_title="Session date",
)
show_plotly_with_alt(
    fig,
    "A close-price line over several days in Central Time, coloured by the session each bar "
    "is assigned to. The colour changes at 4 PM rather than at midnight, so a Sunday evening "
    "and the Monday after it carry one colour between them.",
)

# %% [markdown]
# The colours make the session boundary visible: bars that trade on Sunday evening carry
# Monday's colour, because their CME session runs on until Monday afternoon. A calendar date
# would have split them off into a day of their own holding a few evening hours and nothing
# else.

# %% [markdown]
# ## 3b. Ratio Back-Adjustment
#
# Databento's continuous contracts are **unadjusted**, so the price steps at every roll and a
# return computed across that step is the gap between two contracts rather than a market move.
# The roll table below reports how large the steps are. We apply
# **ratio (multiplicative)** back-adjustment using `instrument_id` to detect roll points:
#
# 1. Detect where `instrument_id` changes between adjacent hourly bars
# 2. Compute ratio = new contract open / old contract close at each roll
# 3. Accumulate ratios backward (most recent prices stay unadjusted)
# 4. Multiply all OHLC prices by cumulative ratio
#
# Ratio adjustment preserves **percentage returns** (critical for IC, momentum features,
# and backtesting) unlike Panama (additive) which distorts returns for old data and can
# push prices negative for commodities with large cumulative adjustments.
#
# See [`06_futures_continuous`](06_futures_continuous.ipynb) for a teaching explanation of adjustment methods.

# %%
# Sort and detect roll transitions per (product, tenor)
hourly_sorted = hourly_with_sessions.sort(["product", "tenor", "timestamp"])

# Detect instrument_id changes within each (product, tenor) group
hourly_sorted = hourly_sorted.with_columns(
    pl.col("instrument_id").shift(1).over("product", "tenor").alias("_prev_instrument_id"),
    pl.col("close").shift(1).over("product", "tenor").alias("_prev_close"),
)

# Roll points: where instrument_id changes (excluding first row of each group)
rolls = hourly_sorted.filter(
    pl.col("_prev_instrument_id").is_not_null()
    & (pl.col("instrument_id") != pl.col("_prev_instrument_id"))
)

# Ratio = new contract's open / old contract's close (adjacent hourly bars)
roll_ratios = rolls.select(
    "product",
    "tenor",
    "timestamp",
    (pl.col("open") / pl.col("_prev_close")).alias("ratio"),
)

print(f"Roll transitions detected: {len(roll_ratios)}")
print(f"Products with rolls: {roll_ratios['product'].n_unique()}")

# Front month only: deferred tenors roll far more often, because thin volume flips
# leadership back and forth, so mixing tenors overstates the front-month count.
es_rolls = roll_ratios.filter((pl.col("product") == "ES") & (pl.col("tenor") == 0)).sort(
    "timestamp"
)
print(f"ES front-month roll ratios ({len(es_rolls)} rolls):")
es_rolls.select("timestamp", "ratio").head(10)

# %% [markdown]
# ### Ratio Back-Adjustment Function
#
# Walk backward through each (product, tenor) group, accumulating roll ratios to
# build a cumulative multiplier for all OHLC prices.


# %%
def ratio_adjust(group: pl.DataFrame) -> pl.DataFrame:
    """Apply ratio back-adjustment to a single (product, tenor) group."""
    group = group.sort("timestamp")

    # Get roll ratios for this group
    group_rolls = roll_ratios.filter(
        (pl.col("product") == group["product"][0]) & (pl.col("tenor") == group["tenor"][0])
    ).select("timestamp", "ratio")

    if len(group_rolls) == 0:
        return group.with_columns(pl.lit(1.0).alias("_cumulative_ratio"))

    # Join roll ratios
    group = group.join(group_rolls, on="timestamp", how="left").with_columns(
        pl.col("ratio").fill_null(1.0)
    )

    # Cumulative ratio: product of all FUTURE ratios (reverse cumprod)
    # Bars BEFORE a roll get multiplied; bars ON and AFTER the roll do not
    n = len(group)
    ratios = group["ratio"].to_numpy()
    adj = np.ones(n)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        adj[i] = cumulative
        if ratios[i] != 1.0:
            cumulative *= ratios[i]

    return group.with_columns(pl.Series("_cumulative_ratio", adj)).drop("ratio")


# %%
# Apply per group
adjusted_groups = []
products_tenors = hourly_sorted.select("product", "tenor").unique().sort("product", "tenor")
n_groups = len(products_tenors)

for i, row in enumerate(products_tenors.iter_rows(named=True)):
    group = hourly_sorted.filter(
        (pl.col("product") == row["product"]) & (pl.col("tenor") == row["tenor"])
    )
    adjusted = ratio_adjust(group)
    adjusted_groups.append(adjusted)
    if (i + 1) % 30 == 0 or i == n_groups - 1:
        print(f"  Adjusted {i + 1}/{n_groups} groups")

hourly_adjusted = pl.concat(adjusted_groups)

# %% [markdown]
# ### Two price series, and no unlabelled one
#
# Ratio adjustment preserves returns within a tenor and moves price levels, so the two things
# a reader might want from this frame cannot come out of one column.
#
# - `adj_*` is roll-continuous, and is what returns, momentum, volatility and labels are built
#   from.
# - `raw_*` is the price as it was quoted, and is what carry, term structure, roll yield,
#   notional and costs need, because each of those compares contracts at one moment.
#
# Differencing adjusted front-month and deferred levels reads accumulated roll history rather
# than the shape of the curve, and produces a plausible-looking number while doing it. So the
# frame ships both series, carries `cum_ratio` so they reconcile as
# `adj_close == raw_close * cum_ratio`, and ships no bare `open`/`high`/`low`/`close` at all:
# a consumer has to say which one it means.

# %%
hourly_adjusted = hourly_adjusted.with_columns(
    pl.col("open").alias("raw_open"),
    pl.col("high").alias("raw_high"),
    pl.col("low").alias("raw_low"),
    pl.col("close").alias("raw_close"),
)

# Adjusted OHLC (multiply by cumulative ratio, not add).
hourly_adjusted = hourly_adjusted.with_columns(
    (pl.col("open") * pl.col("_cumulative_ratio")).alias("adj_open"),
    (pl.col("high") * pl.col("_cumulative_ratio")).alias("adj_high"),
    (pl.col("low") * pl.col("_cumulative_ratio")).alias("adj_low"),
    (pl.col("close") * pl.col("_cumulative_ratio")).alias("adj_close"),
).drop("open", "high", "low", "close")

print(f"\nRatio adjustment applied to {len(hourly_adjusted):,} hourly bars")

# Show adjustment magnitude for ES front month
es_adj = hourly_adjusted.filter((pl.col("product") == "ES") & (pl.col("tenor") == 0)).sort(
    "timestamp"
)
print(
    f"ES front month cumulative ratio range: "
    f"{es_adj['_cumulative_ratio'].min():.4f} to {es_adj['_cumulative_ratio'].max():.4f}"
)

# %% [markdown]
# The adjustment is easiest to see side by side. The top panel plots the raw
# (unadjusted) ES front-month close against the ratio-adjusted series; the two
# coincide at the right edge (recent prices are the anchor) and separate going
# back in time as each roll's ratio compounds. The bottom panel is that
# cumulative multiplier — every downward step is a roll where the new contract
# opened below the old one's close. Raw prices carry those roll gaps as spurious
# returns; the adjusted series does not.

# %%
es_adj_daily = (
    es_adj.group_by("session_date")
    .agg(
        pl.col("raw_close").last(),
        pl.col("adj_close").last(),
        pl.col("_cumulative_ratio").last().alias("cum_ratio"),
    )
    .sort("session_date")
)
_x = es_adj_daily["session_date"].to_list()

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.06,
)
fig.add_trace(
    go.Scatter(
        x=_x,
        y=es_adj_daily["raw_close"].to_list(),
        mode="lines",
        line=dict(color=COLORS["copper"], width=1),
        name="Raw (unadjusted)",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=_x,
        y=es_adj_daily["adj_close"].to_list(),
        mode="lines",
        line=dict(color=COLORS["blue"], width=1),
        name="Ratio-adjusted",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=_x,
        y=es_adj_daily["cum_ratio"].to_list(),
        mode="lines",
        line=dict(color=COLORS["slate"], width=1),
        name="Cumulative ratio",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_hline(y=1.0, line=dict(color=COLORS["neutral"], width=1, dash="dot"), row=2, col=1)
fig.update_layout(
    title="ES front month, raw and ratio-adjusted close",
    height=560,
    legend_title="Price series",
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Cumulative ratio", row=2, col=1)
fig.update_xaxes(title_text="Session date", row=2, col=1)
show_plotly_with_alt(
    fig,
    "Above, the unadjusted and ratio-adjusted front-month price series over the full "
    "history. The unadjusted line steps at each roll while the adjusted one runs through the "
    "same dates without a break. Below, the cumulative adjustment ratio: a staircase that "
    "changes only on roll dates and stays close to one.",
)

# %% [markdown]
# The adjusted frame replaces the hourly one for the aggregation below, keeping `cum_ratio`
# so the raw and adjusted series stay reconcilable.

# %%
hourly_with_sessions = hourly_adjusted.drop("_prev_instrument_id", "_prev_close").rename(
    {"_cumulative_ratio": "cum_ratio"}
)

# %% [markdown]
# ## 4. Aggregate to Daily OHLCV
#
# Aggregate hourly bars to daily using session boundaries. Both the adjusted
# (`adj_*`) and raw (`raw_*`) series are aggregated the same way:
# - **Open**: First bar's open
# - **High**: Maximum high
# - **Low**: Minimum low
# - **Close**: Last bar's close
# - **Volume**: Sum of all volumes

# %%
# Aggregate to daily by session_date, product, tenor
daily = (
    hourly_with_sessions.sort(["product", "tenor", "timestamp"])
    .group_by(["session_date", "product", "tenor"])
    .agg(
        [
            pl.col("adj_open").first(),
            pl.col("adj_high").max(),
            pl.col("adj_low").min(),
            pl.col("adj_close").last(),
            pl.col("raw_open").first(),
            pl.col("raw_high").max(),
            pl.col("raw_low").min(),
            pl.col("raw_close").last(),
            pl.col("cum_ratio").last(),
            pl.col("volume").sum(),
            pl.len().alias("bar_count"),
            pl.col("timestamp").min().alias("session_start"),
            pl.col("timestamp").max().alias("session_end"),
        ]
    )
    .sort(["product", "tenor", "session_date"])
)

print(f"Daily bars: {len(daily):,}")
print(f"Products: {daily['product'].n_unique()}")
print(f"Session date range: {daily['session_date'].min()} to {daily['session_date'].max()}")

# %%
es_daily = daily.filter((pl.col("product") == "ES") & (pl.col("tenor") == 0))
print("ES front month daily bars (first 20 sessions):")
es_daily.select(
    "session_date", "adj_open", "adj_high", "adj_low", "adj_close", "volume", "bar_count"
).head(20)

# %% [markdown]
# ## 5. Validate Aggregation
#
# Two things are worth measuring on the daily frame, and they establish different things.
#
# The **bar count per session** is genuinely informative: it is the number of hourly bars that
# went into each daily bar, and its distribution says which sessions were short and why.
#
# The **OHLC invariants** are not. Every daily price here is selected from an hourly price
# rather than computed from several: the open is the session's first open, the close its last
# close, the high the maximum of the hourly highs and the low the minimum of the hourly lows.
# So if each hourly bar satisfies low ≤ open, close ≤ high, the daily bar inherits it - the
# minimum over the session is at or below the first bar's own low, which is at or below its
# open. The check below cannot detect a bad aggregation. What it can detect is an hourly bar
# that arrived broken, or a later edit that replaces a selection with an arithmetic. Because
# the relation is exact rather than approximate, any breach at all is a finding.

# %%
bar_counts = daily.group_by("bar_count").len().sort("bar_count")
typical_sessions = daily.filter(pl.col("bar_count").is_between(20, 24))
print(f"Typical sessions (20-24 bars): {len(typical_sessions):,} / {len(daily):,}")

# Highlight the modal (most common) bucket; a full 23-hour session dominates.
_modal_bars = bar_counts.sort("len", descending=True)["bar_count"][0]
fig = go.Figure(
    go.Bar(
        x=bar_counts["bar_count"].to_list(),
        y=bar_counts["len"].to_list(),
        marker_color=[
            COLORS["amber"] if bc == _modal_bars else COLORS["slate"]
            for bc in bar_counts["bar_count"].to_list()
        ],
    )
)
fig.add_annotation(
    x=_modal_bars,
    y=bar_counts.filter(pl.col("bar_count") == _modal_bars)["len"][0],
    text=f"{_modal_bars}-hour session",
    showarrow=True,
    arrowhead=2,
    yshift=6,
)
fig.update_layout(
    title="Hourly bars per session",
    xaxis_title="Hourly bars in the session",
    yaxis_title="Number of daily bars",
    height=420,
    showlegend=False,
    xaxis=dict(dtick=2),
)
show_plotly_with_alt(
    fig,
    "A histogram of how many hourly bars each daily session contains. One column towers over "
    "the rest at the full-length session, with a thin tail of shorter sessions to its left.",
)

# %% [markdown]
# One bucket dominates: the full-length session is the overwhelming mode, and everything to its
# left has a reason - holidays, half days, and deferred tenors thin enough to stop printing for
# part of the day. The tail is small in count and worth keeping visible, because a session with
# a handful of bars produces a daily bar whose high and low mean much less than the others.

# %%
# OHLC invariant check
ohlc_check = daily.with_columns(
    [
        (pl.col("adj_low") <= pl.col("adj_open")).alias("low_le_open"),
        (pl.col("adj_low") <= pl.col("adj_close")).alias("low_le_close"),
        (pl.col("adj_high") >= pl.col("adj_open")).alias("high_ge_open"),
        (pl.col("adj_high") >= pl.col("adj_close")).alias("high_ge_close"),
    ]
)

print(f"OHLC invariants over {len(ohlc_check):,} daily bars:")
for col in ["low_le_open", "low_le_close", "high_ge_open", "high_ge_close"]:
    breaches = int((~ohlc_check[col]).sum())
    status = "[OK]" if breaches == 0 else "[FAIL]"
    print(f"  {status} {col}: {breaches} breaches")

# %% [markdown]
# ## 6. Coverage Summary
#
# Summary of daily data coverage by product.

# %%
# Coverage by product
coverage = (
    daily.group_by("product")
    .agg(
        [
            pl.col("session_date").min().alias("start_date"),
            pl.col("session_date").max().alias("end_date"),
            pl.len().alias("total_bars"),
            pl.col("tenor").n_unique().alias("tenors"),
        ]
    )
    .sort("product")
)

print("Daily data coverage by product:")
coverage

# %%
tenor_coverage = (
    daily.group_by("tenor")
    .agg(
        pl.col("product").n_unique().alias("products"),
        pl.len().alias("total_bars"),
    )
    .sort("tenor")
)
print("Coverage by tenor:")
tenor_coverage

# %% [markdown]
# ## 7. Save Daily Data
#
# Save the session-aggregated daily data for downstream use.

# %%
# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save combined daily file
output_path = OUTPUT_DIR / "continuous_daily.parquet"
daily.write_parquet(output_path)
print(f"Saved: {_rel(output_path)}")
print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")

# %%
# Also save per-product files for convenience
per_product_dir = OUTPUT_DIR / "by_product"
per_product_dir.mkdir(exist_ok=True)

for product in products:
    product_df = daily.filter(pl.col("product") == product)
    product_path = per_product_dir / f"{product}.parquet"
    product_df.write_parquet(product_path)

print(f"\nSaved per-product files to: {_rel(per_product_dir)}/")
print(f"Products: {len(products)}")

# %% [markdown]
# ## 8. Using the Daily Data
#
# The daily data is now available via `load_cme_futures()` (daily is the default frequency).
# This loader is defined in `data/__init__.py` and can be used by downstream chapters.

# %%
es_nq_2024 = (
    pl.read_parquet(OUTPUT_DIR / "continuous_daily.parquet")
    .filter(
        pl.col("product").is_in(["ES", "NQ"])
        & (pl.col("tenor") == 0)
        & (pl.col("session_date") >= pl.lit("2024-01-01").str.to_date())
        & (pl.col("session_date") <= pl.lit("2024-12-31").str.to_date())
    )
    .sort("session_date", "product")
)
print(f"ES + NQ front month, 2024: {len(es_nq_2024)} daily bars")
es_nq_2024.head(10)

# %% [markdown]
# ## Key Takeaways
#
# 1. **CME sessions end at 4 PM CT**, not midnight UTC. The session date is
#    the date the session ends, so Sunday-evening trading is counted into Monday.
# 2. **Aggregate on the session, not the calendar.** Grouping by UTC date splits one trading
#    session across two rows and produces a daily bar whose open, high, low and close come
#    from two different sessions. Nothing downstream can recover from that, and nothing about
#    the resulting frame looks wrong.
# 3. **Ratio back-adjustment is applied per product and tenor, before aggregation.** The order
#    matters: adjusting after aggregating would compute the daily high and low from prices on
#    two sides of a roll. The cumulative ratio stays near one over this history, which is what
#    a series of quarterly rolls in a liquid contract looks like.
# 4. **The session length has a mode, and everything else is a reason.** Most sessions run the
#    full trading day; the shorter ones are holidays, partial days, and deferred tenors thin
#    enough to stop printing. The distribution is worth drawing rather than summarising,
#    because the tail is the part that needs explaining.
# 5. **A check that cannot fail is not a check.** The daily OHLC invariants follow from the
#    aggregation being four selections rather than four calculations, so they hold for any
#    valid hourly input. Running them is still worth the line, because they catch a broken
#    input bar or a future edit that computes where it used to select - but the section says
#    which of those it would be finding, rather than implying the aggregation is on trial.
#
# ## Next Steps
#
# - [`06_futures_continuous`](06_futures_continuous.ipynb): Roll detection and
#   alternative adjustment methods (Panama / calendar).
# - **Chapter 8**: Feature engineering on daily futures data.
# - **Chapter 16**: Backtesting with session-correct returns.
