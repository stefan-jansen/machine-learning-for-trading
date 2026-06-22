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
# # Constructing Continuous Futures Contracts
#
# **Docker image**: `ml4t`
#
# **Purpose**: Walk through the construction of a continuous futures price
# series from individual expiring contracts: detect rolls, compare
# adjustment methods (raw, Panama / additive back-adjustment, ratio /
# multiplicative back-adjustment), and validate against the vendor-built
# continuous series.
#
# **Learning objectives**:
#
# - Detect roll dates using volume-based front-month identification (with a
#   no-rollback constraint to avoid spurious switches).
# - Apply Panama (additive) back-adjustment to preserve dollar P&L across
#   rolls.
# - Apply ratio (multiplicative) back-adjustment to preserve percentage
#   returns across rolls.
# - Cross-check constructed continuous prices against Databento's pre-built
#   continuous series and quantify the disagreement.
#
# **Book reference**: §2.2 ("The Asset-Class Market Data Landscape" —
# Futures); the methodology comparison underpins the engineering decision
# in §2.2 to store raw contract histories alongside one or more continuous
# variants.
#
# **Prerequisites**: `data` package on `PYTHONPATH`; individual ES contract
# parquet at `ML4T_DATA_PATH/futures/market/individual/ES/data.parquet` and
# the contract-definitions parquet at
# `ML4T_DATA_PATH/futures/market/contract_definitions.parquet`.

# %%
"""Continuous Futures Construction."""

import re
from datetime import UTC, date, datetime

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_cme_futures
from utils import ML4T_DATA_PATH

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ## 1. Understanding the Data
#
# ### 1.1 Load Individual Contracts

# %%
es_individual = load_cme_futures(products=["ES"], frequency="hourly", continuous=False)

print(f"Individual contracts: {es_individual.shape}")
print(f"Unique contracts (by instrument_id): {es_individual['instrument_id'].n_unique()}")
print(f"Date range: {es_individual['timestamp'].min()} to {es_individual['timestamp'].max()}")
print("Sample:")
es_individual.head()

# %%
contract_stats = (
    es_individual.group_by("instrument_id")
    .agg(
        pl.col("timestamp").min().alias("first_trade"),
        pl.col("timestamp").max().alias("last_trade"),
        pl.col("volume").sum().alias("total_volume"),
        pl.len().alias("trading_days"),
    )
    .sort("first_trade")
)

print(f"Contracts: {len(contract_stats)} (sorted by first trade)")
contract_stats.head(10)

# %% [markdown]
# ### 1.2 Understanding Contract Naming
#
# ES contract symbols follow the pattern: ES + Month Code + Year
#
# Month codes:
# - H = March, M = June, U = September, Z = December (standard quarterly)
# - F = January, G = February, J = April, K = May, N = July, Q = August, V = October, X = November

# %%
_MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
_SYMBOL_RE = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d+)$")


def parse_contract_symbol(symbol: str) -> dict:
    """Parse a futures contract symbol like ESH24 or RTYM25 into product / month / year."""
    match = _SYMBOL_RE.match(symbol)
    if not match:
        raise ValueError(f"Cannot parse symbol: {symbol}")
    product, month_code, year_str = match.groups()
    year = int(year_str)
    year = year + 2000 if year < 50 else year + 1900
    return {
        "product": product,
        "month_code": month_code,
        "month": _MONTH_CODES[month_code],
        "year": year,
    }


# %%
defn_path = ML4T_DATA_PATH / "futures" / "market" / "contract_definitions.parquet"
contract_defs = pl.read_parquet(defn_path).filter(pl.col("product") == "ES")
contract_df = (
    pl.DataFrame(
        [
            {**parse_contract_symbol(r["symbol"]), "symbol": r["symbol"]}
            for r in contract_defs.iter_rows(named=True)
        ]
    )
    .join(contract_defs.select("symbol", "expiration"), on="symbol")
    .sort("year", "month")
)
print(f"ES contract definitions: {contract_df.height} contracts")
contract_df.select("symbol", "month_code", "month", "year", "expiration").head(10)

# %% [markdown]
# ### 1.3 Contract Expiration from Symbols
#
# Without a separate definitions file, we can derive expiration information
# from contract symbols. For ES contracts, the pattern is ESH24 (March 2024),
# ESM24 (June 2024), etc.

# %% [markdown]
# `parse_contract_symbol` and the contract-definitions parquet give us actual
# expiration dates. For products where we only see the symbol (no definitions
# file), the expiration can be approximated as the 15th of the contract month
# — close enough for roll detection but not for delivery scheduling.

# %%
es_definition = contract_df.select(
    "symbol",
    "year",
    "month",
    pl.struct("year", "month")
    .map_elements(lambda x: date(x["year"], x["month"], 15), return_dtype=pl.Date)
    .alias("expiration"),
)
print(f"ES definition rows: {es_definition.height}")
es_definition.head(10)

# %% [markdown]
# ## 2. Roll Detection
#
# The "roll" is when we switch from the near-month contract to the next contract.
# There are several approaches:
#
# 1. **Volume-based**: Roll when the next contract has higher daily volume
# 2. **Open Interest-based**: Roll when next contract has higher open interest
# 3. **Fixed Schedule**: Roll N days before expiration (e.g., first Thursday of expiry month)
#
# We'll implement volume-based rolling.


# %%
def identify_front_month(
    individual_df: pl.DataFrame, min_outright_price: float = 500.0
) -> pl.DataFrame:
    """Volume-based front-month detection with no-rollback constraint."""
    # Filter to outright contracts only (exclude calendar spreads at ~$50-100)
    outrights = individual_df.filter(pl.col("close") >= min_outright_price)

    # Aggregate to daily volume per contract
    daily_volume = (
        outrights.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by(["date", "instrument_id"])
        .agg(pl.col("volume").sum().alias("daily_volume"))
    )

    daily_leader = (
        daily_volume.group_by("date")
        .agg(pl.col("instrument_id").sort_by("daily_volume").last().alias("leader"))
        .sort("date")
    )

    # No-rollback constraint — switch to new leaders, never go back
    leader_ids = daily_leader["leader"].to_list()
    dates = daily_leader["date"].to_list()
    used_contracts = {leader_ids[0]}
    current_front = leader_ids[0]
    front = [current_front]

    for i in range(1, len(leader_ids)):
        if leader_ids[i] != current_front and leader_ids[i] not in used_contracts:
            current_front = leader_ids[i]
            used_contracts.add(current_front)
        front.append(current_front)

    daily_front = pl.DataFrame({"date": dates, "front_symbol": front})

    # Expand back to hourly bars
    hourly = individual_df.select("timestamp").unique().sort("timestamp")
    hourly = hourly.with_columns(pl.col("timestamp").dt.date().alias("date"))
    front_month = hourly.join(daily_front, on="date", how="left").drop("date")

    front_month = front_month.with_columns(
        pl.col("front_symbol").shift(1).alias("prev_front"),
    ).with_columns(
        (pl.col("front_symbol") != pl.col("prev_front")).alias("is_roll"),
    )
    return front_month


# %%
front_months = identify_front_month(es_individual)
print("Front month identification (2024 sample):")
front_months.filter(pl.col("timestamp") >= datetime(2024, 1, 1, tzinfo=UTC)).head(20)

# %%
roll_dates = front_months.filter(pl.col("is_roll"))
print(f"Total roll events: {len(roll_dates)}")
print("Most recent 10 roll dates:")
roll_dates.tail(10).select("timestamp", "prev_front", "front_symbol")

# %% [markdown]
# ### 2.2 Calendar-Based Roll (Alternative)
#
# An alternative to volume-based rolling is **calendar-based**: roll a fixed number
# of days before contract expiration. This is simpler and more predictable, but may
# not track liquidity as well as volume-based methods.
#
# Common calendar roll schedules:
# - 5 business days before expiry (conservative)
# - First notice day (for physical delivery commodities)
# - 2 weeks before expiry (popular for equity index futures)


# %%
def identify_front_month_calendar(
    individual_df: pl.DataFrame,
    definition_df: pl.DataFrame,
    roll_days_before: int = 5,
) -> pl.DataFrame:
    """Identify front month using calendar-based roll (fixed days before expiry)."""
    # Get expiration dates from definitions
    # NOTE: Requires individual data to have a "symbol" column with contract names
    expirations = definition_df.select(["symbol", "expiration"]).with_columns(
        pl.col("expiration").cast(pl.Date).alias("expiry_date")
    )

    # Join with individual data (requires symbol column)
    with_expiry = individual_df.join(expirations, on="symbol", how="left").with_columns(
        pl.col("timestamp").cast(pl.Date).alias("trade_date")
    )

    # Calculate days to expiry
    with_expiry = with_expiry.with_columns(
        (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days().alias("days_to_expiry")
    )

    # Filter to contracts with more than roll_days_before to expiry
    # Then select the nearest such contract for each day
    front_month = (
        with_expiry.filter(pl.col("days_to_expiry") > roll_days_before)
        .sort(["timestamp", "days_to_expiry"])
        .group_by("timestamp")
        .first()
        .select(["timestamp", pl.col("symbol").alias("front_symbol")])
        .sort("timestamp")
    )

    # Add roll indicators
    front_month = front_month.with_columns(
        pl.col("front_symbol").shift(1).alias("prev_front"),
    ).with_columns(
        (pl.col("front_symbol") != pl.col("prev_front")).alias("is_roll"),
    )

    return front_month


# %% [markdown]
# Calendar-based roll detection requires the individual data to carry a symbol
# column that joins to the contract-definitions table. The Databento individual
# parquet uses numeric `instrument_id` rather than ESH24-style symbols, so we
# present the calendar logic above as a teaching reference and use volume-based
# detection for the rest of the notebook.

# %%
volume_rolls = front_months.filter(pl.col("is_roll"))
print(f"Volume-based rolls (ES, 2016-2025): {len(volume_rolls)}")

# %% [markdown]
# **Volume vs Calendar Trade-offs**:
# - **Volume-based**: Follows liquidity naturally, but roll timing varies
# - **Calendar-based**: Predictable timing, easier to automate, but may roll into less liquid contract
#
# For this notebook, we use **volume-based** roll detection as our primary method since it
# better reflects actual market liquidity transitions.

# %% [markdown]
# ## 3. Adjustment Methods
#
# When we roll from contract A to contract B, there's usually a price gap.
# If we don't adjust, our time series will have artificial jumps.
#
# ### 3.1 No Adjustment (Raw)
#
# Simply use prices as-is. Returns calculated on roll dates are invalid.


# %%
def create_continuous_raw(individual_df: pl.DataFrame, front_months: pl.DataFrame) -> pl.DataFrame:
    """Create continuous series with no adjustment (raw prices)."""
    # Join individual prices with front month info
    continuous = (
        individual_df.join(
            front_months.select(["timestamp", "front_symbol"]), on="timestamp", how="inner"
        )
        .filter(pl.col("instrument_id") == pl.col("front_symbol"))
        .select(["timestamp", "open", "high", "low", "close", "volume", "instrument_id"])
        .sort("timestamp")
    )

    return continuous


# %%
es_continuous_raw = create_continuous_raw(es_individual, front_months)
print(f"Raw continuous series: {len(es_continuous_raw)} hourly bars")
es_continuous_raw.head(10)

# %% [markdown]
# ### 3.2 Panama (Back-Adjustment)
#
# Add the price gap to all historical prices. This preserves dollar P&L
# but distorts percentage returns for old data.
#
# Gap = Close_new_contract - Close_old_contract
# Adjusted_price = Price + cumulative_gap
#
# Note: We add (not subtract) because we're bringing old prices UP to the
# current contract's level, eliminating the discontinuity at roll dates.


# %%
def _compute_roll_gaps(individual_df: pl.DataFrame, front_months: pl.DataFrame) -> pl.DataFrame:
    """Compute price gaps at each roll date (new - old contract close)."""
    roll_info = front_months.filter(pl.col("is_roll"))
    prices_lookup = individual_df.select(["timestamp", "instrument_id", "close"])

    old_prices = (
        roll_info.select(["timestamp", pl.col("prev_front").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "old_close"})
    )

    new_prices = (
        roll_info.select(["timestamp", pl.col("front_symbol").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "new_close"})
    )

    return (
        old_prices.select(["timestamp", "old_close"])
        .join(new_prices.select(["timestamp", "new_close"]), on="timestamp", how="inner")
        .with_columns((pl.col("new_close") - pl.col("old_close")).alias("gap"))
        .select(["timestamp", "gap"])
        .drop_nulls()
    )


# %% [markdown]
# ### Panama Adjustment
#
# Apply the computed gaps cumulatively backwards through the raw series.


# %%
def create_continuous_panama(
    individual_df: pl.DataFrame, front_months: pl.DataFrame
) -> pl.DataFrame:
    """Create continuous series with Panama (back) adjustment.

    Uses vectorized Polars joins instead of row-by-row iteration for O(n) complexity.
    """
    raw = create_continuous_raw(individual_df, front_months)
    roll_info = front_months.filter(pl.col("is_roll"))

    if len(roll_info) == 0:
        return raw.with_columns(pl.lit(0.0).alias("cumulative_adjustment"))

    gaps_df = _compute_roll_gaps(individual_df, front_months)

    if len(gaps_df) == 0:
        return raw.with_columns(pl.lit(0.0).alias("cumulative_adjustment"))

    # Adjustment applies to dates STRICTLY BEFORE each roll date
    raw_with_gaps = raw.join(gaps_df, on="timestamp", how="left").with_columns(
        pl.col("gap").fill_null(0.0)
    )

    # Cumulative sum in reverse, shift by 1 to exclude roll date from adjustment
    raw_with_gaps = raw_with_gaps.with_columns(
        pl.col("gap")
        .reverse()
        .cum_sum()
        .shift(1)
        .fill_null(0.0)
        .reverse()
        .alias("cumulative_adjustment")
    )

    adjusted = raw_with_gaps.with_columns(
        [
            (pl.col("open") + pl.col("cumulative_adjustment")).alias("adj_open"),
            (pl.col("high") + pl.col("cumulative_adjustment")).alias("adj_high"),
            (pl.col("low") + pl.col("cumulative_adjustment")).alias("adj_low"),
            (pl.col("close") + pl.col("cumulative_adjustment")).alias("adj_close"),
        ]
    )

    return adjusted


# %%
es_continuous_panama = create_continuous_panama(es_individual, front_months)
panama_first = es_continuous_panama["cumulative_adjustment"][0]
print(
    f"Panama-adjusted: cumulative_adjustment at the start of the series = {panama_first:+.2f} "
    f"(adjusts every historical price up by this amount so the most recent contract is unchanged)"
)
es_continuous_panama.select(
    "timestamp", "close", "adj_close", "cumulative_adjustment", "instrument_id"
).head(10)

# %% [markdown]
# ### 3.3 Ratio Adjustment
#
# Multiply historical prices by the ratio of new/old contract prices.
# This preserves percentage returns but distorts dollar amounts.
#
# Ratio = Close_new_contract / Close_old_contract
# Adjusted_price = Price * cumulative_ratio


# %%
def _compute_roll_ratios(individual_df: pl.DataFrame, front_months: pl.DataFrame) -> pl.DataFrame:
    """Compute price ratios (new/old) at each roll date."""
    roll_info = front_months.filter(pl.col("is_roll"))
    prices_lookup = individual_df.select(["timestamp", "instrument_id", "close"])

    old_prices = (
        roll_info.select(["timestamp", pl.col("prev_front").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "old_close"})
    )

    new_prices = (
        roll_info.select(["timestamp", pl.col("front_symbol").alias("instrument_id")])
        .join(prices_lookup, on=["timestamp", "instrument_id"], how="left")
        .rename({"close": "new_close"})
    )

    return (
        old_prices.select(["timestamp", "old_close"])
        .join(new_prices.select(["timestamp", "new_close"]), on="timestamp", how="inner")
        .filter(pl.col("old_close") != 0)
        .with_columns((pl.col("new_close") / pl.col("old_close")).alias("ratio"))
        .select(["timestamp", "ratio"])
        .drop_nulls()
    )


# %% [markdown]
# ### Ratio Adjustment
#
# Apply the computed ratios cumulatively backwards through the raw series.


# %%
def create_continuous_ratio(
    individual_df: pl.DataFrame, front_months: pl.DataFrame
) -> pl.DataFrame:
    """Create continuous series with ratio adjustment.

    Uses vectorized Polars joins instead of row-by-row iteration for O(n) complexity.
    """
    raw = create_continuous_raw(individual_df, front_months)
    roll_info = front_months.filter(pl.col("is_roll"))

    if len(roll_info) == 0:
        return raw.with_columns(pl.lit(1.0).alias("cumulative_ratio"))

    ratios_df = _compute_roll_ratios(individual_df, front_months)

    if len(ratios_df) == 0:
        return raw.with_columns(pl.lit(1.0).alias("cumulative_ratio"))

    # Adjustment applies to dates STRICTLY BEFORE each roll date
    raw_with_ratios = raw.join(ratios_df, on="timestamp", how="left").with_columns(
        pl.col("ratio").fill_null(1.0)
    )

    # Cumulative product in reverse, shift by 1 to exclude roll date
    raw_with_ratios = raw_with_ratios.with_columns(
        pl.col("ratio")
        .reverse()
        .cum_prod()
        .shift(1)
        .fill_null(1.0)
        .reverse()
        .alias("cumulative_ratio")
    )

    adjusted = raw_with_ratios.with_columns(
        [
            (pl.col("open") * pl.col("cumulative_ratio")).alias("adj_open"),
            (pl.col("high") * pl.col("cumulative_ratio")).alias("adj_high"),
            (pl.col("low") * pl.col("cumulative_ratio")).alias("adj_low"),
            (pl.col("close") * pl.col("cumulative_ratio")).alias("adj_close"),
        ]
    )

    return adjusted


# %%
es_continuous_ratio = create_continuous_ratio(es_individual, front_months)
ratio_first = es_continuous_ratio["cumulative_ratio"][0]
print(
    f"Ratio-adjusted: cumulative_ratio at the start of the series = {ratio_first:.4f} "
    f"(historical prices are scaled up by this factor)"
)
es_continuous_ratio.select(
    "timestamp", "close", "adj_close", "cumulative_ratio", "instrument_id"
).head(10)

# %% [markdown]
# ## 4. Validation
#
# Let's compare our construction to DataBento's pre-built continuous series.
#
# We compare our construction against DataBento's production continuous series.
# Both use volume-based roll detection, but the implementations differ in detail
# (daily aggregation window, exact crossover threshold, etc.), so some divergence
# on roll timing is expected — typically by a day or two around the roll date.

# %%
es_databento = load_cme_futures(products=["ES"], tenors=[0], frequency="hourly", continuous=True)
print(f"DataBento continuous: {es_databento.shape}")
es_databento.head()

# %%
comparison = (
    es_continuous_raw.select("timestamp", pl.col("close").alias("our_close"))
    .join(
        es_databento.select("timestamp", pl.col("close").alias("databento_close")),
        on="timestamp",
        how="inner",
    )
    .with_columns(
        (pl.col("our_close") - pl.col("databento_close")).alias("diff"),
    )
)

mean_abs = comparison["diff"].abs().mean()
median_diff = comparison["diff"].median()
max_abs = comparison["diff"].abs().max()
large_diffs = comparison.filter(pl.col("diff").abs() > 1)

print(f"Hours compared: {len(comparison)}")
print(f"Mean absolute difference: ${mean_abs:.2f}")
print(f"Median signed difference:  ${median_diff:+.2f}")
print(f"Max absolute difference:   ${max_abs:.2f}")
print(
    f"Hourly bars with >$1 difference: {len(large_diffs)} ({100 * len(large_diffs) / len(comparison):.1f}%)"
)
comparison.describe()

# %%
print("Sample of bars with large differences:")
large_diffs.head(10)

# %% [markdown]
# Most differences come from roll-timing disagreements — when our volume-based
# detector rolls a day earlier or later than Databento's, the two series report
# the price of a different contract for those hours, and the
# contango/backwardation spread between expiries produces a gap. The median
# signed difference is essentially zero, but mean absolute difference is on the
# order of tens of dollars, with occasional larger gaps around roll dates where
# the two algorithms disagree by more than a day.

# %% [markdown]
# ### 4.1 Visualize the Difference

# %%
# Plot both series
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price Comparison", "Difference")
)

comp_pd = comparison.to_pandas()

fig.add_trace(
    go.Scatter(x=comp_pd["timestamp"], y=comp_pd["our_close"], name="Our Construction"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=comp_pd["timestamp"], y=comp_pd["databento_close"], name="DataBento"), row=1, col=1
)
fig.add_trace(
    go.Scatter(x=comp_pd["timestamp"], y=comp_pd["diff"], name="Difference"), row=2, col=1
)

fig.update_layout(height=600, title="ES Continuous: Our Construction vs DataBento")
fig.show()

# %% [markdown]
# ## 5. Construct + Validate Helper
#
# The construction logic is generic across products. The Databento subscription
# bundled with the book ships individual contract data for ES only; the other
# 29 products are delivered exclusively as pre-built continuous series. We
# therefore wrap the construction-plus-validation in a single function and
# apply it to ES — the only product where individual data is currently
# available on disk.


# %%
def construct_and_validate(product: str) -> dict:
    """Construct a raw continuous series and validate it against the vendor continuous."""
    individual = load_cme_futures(products=[product], frequency="hourly", continuous=False)
    fronts = identify_front_month(individual)
    continuous_raw = create_continuous_raw(individual, fronts)
    databento = load_cme_futures(
        products=[product], tenors=[0], frequency="hourly", continuous=True
    )
    cmp = continuous_raw.select("timestamp", pl.col("close").alias("our_close")).join(
        databento.select("timestamp", pl.col("close").alias("db_close")),
        on="timestamp",
        how="inner",
    )
    diff = (cmp["our_close"] - cmp["db_close"]).abs()
    return {
        "product": product,
        "rows": len(continuous_raw),
        "contracts_used": continuous_raw["instrument_id"].n_unique(),
        "validation_rows": len(cmp),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }


# %%
validation_summary = pl.DataFrame([construct_and_validate("ES")])
print("Construction-vs-vendor validation (ES):")
validation_summary

# %% [markdown]
# ## 6. Production Pipeline
#
# The teaching examples above demonstrate roll detection and adjustment methods on a single product.
# For production use, the pipeline is:
#
# 1. **Download**: Databento provides pre-rolled continuous contracts (hourly OHLCV) for
#    front, second, and third month tenors → `data/futures/market/continuous/hourly/`
# 2. **Session aggregation**: [`05_futures_session_aggregation`](05_futures_session_aggregation.ipynb) assigns CME session dates
#    and aggregates hourly bars into daily OHLCV → `data/futures/market/continuous/daily/continuous_daily.parquet`
# 3. **Loading**: `load_cme_futures()` reads the daily parquet for downstream analysis

# %% [markdown]
# ---
#
# ## Key Takeaways
#
# 1. **Roll detection (volume-based, ES, 2016-2025)** finds **40 rolls** —
#    matching the four quarterly rolls per year × 10 years. The
#    no-rollback constraint is necessary because raw daily-volume leadership
#    can flicker between contracts during the roll window.
# 2. **Calendar spreads contaminate raw individual data**: CME ships outright
#    contracts alongside calendar spreads that trade at the inter-month
#    price difference (~$50–100) rather than the index level (~$5,000). The
#    `min_outright_price` filter in `identify_front_month` is what prevents
#    a high-volume spread from being selected as "front month".
# 3. **Panama (additive) adjustment** for the ES series adds about
#    **$625 to the earliest 2016 prices** (so the start-of-history close is
#    ~30% above the original quote). This preserves dollar P&L across rolls
#    but distorts percentage returns the further back you go.
# 4. **Ratio (multiplicative) adjustment** for the same window has a
#    **cumulative ratio of ~1.11 at the start of the series** (about
#    +11% scaling). Returns stay correct in percentage terms across the
#    whole window — the right choice for IC, momentum features, and any
#    statistical analysis.
# 5. **Validation against Databento's continuous** (2,581 daily-aligned bars):
#    **mean absolute difference ~$27**, **median signed difference ~$2.50**
#    (essentially zero relative to ~$3,800 average price). 2,444 of those
#    bars differ by more than $1 (most by a few dollars; **maximum absolute
#    gap ~$583**). Differences come almost entirely from roll-timing
#    disagreements — when our detector rolls a day earlier or later than the
#    vendor's algorithm, the two series report the price of a different
#    contract for those hours. The signed median near zero means the
#    disagreements wash out: neither algorithm is systematically high or
#    low.
#
# ### Adjustment Method Selection
# | Use Case | Recommended Method | Reason |
# |----------|-------------------|--------|
# | Backtesting P&L | Panama (additive) | Preserves dollar gains/losses across rolls |
# | Statistical analysis | Ratio | Preserves percentage returns accurately |
# | Live trading | Raw + position management | Handle rolls in execution layer |
#
# ### Next Steps
#
# - **Chapter 8**: Carry and momentum features built on continuous series.
# - **Chapter 16**: Backtesting with adjusted P&L.
