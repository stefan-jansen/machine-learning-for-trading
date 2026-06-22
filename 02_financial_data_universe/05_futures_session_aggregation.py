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
#   bars on Sunday evening belong to Monday's session.
# - Apply ratio (multiplicative) back-adjustment to a continuous series so
#   percentage returns are preserved across rolls.
# - Aggregate hourly bars to session-correct daily OHLCV across all 30 products
#   and three tenors (front month, first deferred, second deferred).
#
# **Book reference**: §2.2 ("The Asset-Class Market Data Landscape" — Futures);
# adjustment methodology compared in `06_futures_continuous`.
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
import polars as pl

from data import load_cme_futures
from utils import ML4T_DATA_PATH
from utils.paths import get_chapter_dir

# Output path for session-aggregated daily data. Default writes to a
# chapter-local directory; set WRITE_TO_DATA=1 to materialize the canonical
# daily parquet under ML4T_DATA_PATH for downstream notebooks.
WRITE_TO_DATA = os.environ.get("WRITE_TO_DATA", "0") == "1"
OUTPUT_DIR = (
    ML4T_DATA_PATH / "futures" / "market" / "continuous" / "daily"
    if WRITE_TO_DATA
    else get_chapter_dir(2) / "output" / "futures_daily"
)


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

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
# Both bars belong to Monday's session (which ends Monday 4 PM CT).

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
# Vectorized session date assignment using Polars
# Convert to Central Time, then check if hour >= 16 (4 PM)


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
# ## 3b. Ratio Back-Adjustment
#
# Databento's continuous contracts are **unadjusted** — price gaps at roll transitions
# produce spurious returns (e.g., ES Mar 2020: -11.08% artificial gap). We apply
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

es_rolls = roll_ratios.filter(pl.col("product") == "ES").sort("timestamp")
print(f"ES roll ratios ({len(es_rolls)} rolls):")
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

# Apply ratio adjustment to OHLC (multiply, not add)
hourly_adjusted = hourly_adjusted.with_columns(
    (pl.col("open") * pl.col("_cumulative_ratio")).alias("open"),
    (pl.col("high") * pl.col("_cumulative_ratio")).alias("high"),
    (pl.col("low") * pl.col("_cumulative_ratio")).alias("low"),
    (pl.col("close") * pl.col("_cumulative_ratio")).alias("close"),
)

print(f"\nRatio adjustment applied to {len(hourly_adjusted):,} hourly bars")

# Show adjustment magnitude for ES front month
es_adj = hourly_adjusted.filter((pl.col("product") == "ES") & (pl.col("tenor") == 0)).sort(
    "timestamp"
)
print(
    f"ES front month cumulative ratio range: "
    f"{es_adj['_cumulative_ratio'].min():.4f} to {es_adj['_cumulative_ratio'].max():.4f}"
)

# Replace hourly_with_sessions with adjusted data for downstream aggregation
hourly_with_sessions = hourly_adjusted.drop(
    "_prev_instrument_id", "_prev_close", "_cumulative_ratio"
)

# %% [markdown]
# ## 4. Aggregate to Daily OHLCV
#
# Aggregate hourly bars to daily using session boundaries:
# - **Open**: First bar's open (ratio-adjusted)
# - **High**: Maximum high (ratio-adjusted)
# - **Low**: Minimum low (ratio-adjusted)
# - **Close**: Last bar's close (ratio-adjusted)
# - **Volume**: Sum of all volumes

# %%
# Aggregate to daily by session_date, product, tenor
daily = (
    hourly_with_sessions.sort(["product", "tenor", "timestamp"])
    .group_by(["session_date", "product", "tenor"])
    .agg(
        [
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
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
es_daily.select("session_date", "open", "high", "low", "close", "volume", "bar_count").head(20)

# %% [markdown]
# ## 5. Validate Aggregation
#
# Check that daily aggregation is correct:
# - Bar counts should be ~23 per session (23-hour trading day)
# - OHLC relationships should hold (Low ≤ Open/Close ≤ High)

# %%
bar_counts = daily.group_by("bar_count").len().sort("bar_count")
typical_sessions = daily.filter(pl.col("bar_count").is_between(20, 24))
print(f"Typical sessions (20-24 bars): {len(typical_sessions):,} / {len(daily):,}")
print("Bar counts per session:")
bar_counts

# %%
# OHLC invariant check
ohlc_check = daily.with_columns(
    [
        (pl.col("low") <= pl.col("open")).alias("low_le_open"),
        (pl.col("low") <= pl.col("close")).alias("low_le_close"),
        (pl.col("high") >= pl.col("open")).alias("high_ge_open"),
        (pl.col("high") >= pl.col("close")).alias("high_ge_close"),
    ]
)

print("OHLC Invariant Check:")
for col in ["low_le_open", "low_le_close", "high_ge_open", "high_ge_close"]:
    pct = ohlc_check[col].mean() * 100
    status = "[OK]" if pct > 99.9 else "[FAIL]"
    print(f"  {status} {col}: {pct:.2f}%")

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
print(f"Saved: {output_path}")
print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")

# %%
# Also save per-product files for convenience
per_product_dir = OUTPUT_DIR / "by_product"
per_product_dir.mkdir(exist_ok=True)

for product in products:
    product_df = daily.filter(pl.col("product") == product)
    product_path = per_product_dir / f"{product}.parquet"
    product_df.write_parquet(product_path)

print(f"\nSaved per-product files to: {per_product_dir}/")
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
#    the date the session ends — Sunday-evening trading belongs to Monday's
#    session.
# 2. **Volume here is 5,463,741 hourly bars across 30 products and 3 tenors**,
#    aggregating to 312,859 daily bars over 2011-01-03 through 2025-12-31.
# 3. **Ratio back-adjustment** is applied per (product, tenor) before
#    aggregation — for ES front month the cumulative ratio ranges 0.87–1.16
#    over 427 rolls, preserving percentage returns across roll boundaries.
# 4. **Full sessions have 23 hourly bars** (23-hour trading day): the 23-bar
#    bucket is by far the largest in the bar-count distribution. Shorter
#    sessions arise from holidays, deferred tenors with thin trading, and
#    partial days.
# 5. **OHLC invariants hold at 100%** on the aggregated daily bars across all
#    four checks.
#
# ## Next Steps
#
# - [`06_futures_continuous`](06_futures_continuous.ipynb): Roll detection and
#   alternative adjustment methods (Panama / calendar).
# - **Chapter 8**: Feature engineering on daily futures data.
# - **Chapter 16**: Backtesting with session-correct returns.
