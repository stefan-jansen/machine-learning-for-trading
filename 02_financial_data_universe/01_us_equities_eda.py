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
# # US Equities — Exploratory Data Analysis
#
# **Docker image**: `ml4t`
#
# **Purpose**: Profile the Wiki Prices dataset of US equity OHLCV history and confirm
# the inactive-symbol coverage that makes it usable for survivorship-bias-free
# backtests.
#
# **Learning objectives**:
#
# - Load the equity panel via `data.load_us_equities` and inspect its canonical schema.
# - Distinguish raw and split/dividend-adjusted price columns.
# - Quantify the share of symbols that stop trading before the dataset end date.
# - Check OHLC invariants and null rates across the full panel.
#
# **Book reference**: §2.2 ("The Asset-Class Market Data Landscape" — Equities).
#
# **Prerequisites**: `data` package on `PYTHONPATH`; Wiki Prices parquet present at
# `ML4T_DATA_PATH/equities/market/us_equities/`. Run
# `python data/equities/market/us_equities/download.py` if missing.

# %%
"""US Equities — Exploratory data analysis of the Wiki Prices dataset."""

import polars as pl

from data import load_us_equities
from utils.data_quality import check_ohlc_invariants

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all

# %% [markdown]
# ## 1. Load and Inspect

# %%
wiki = load_us_equities()

print("=== Wiki Prices Dataset ===")
print(f"Shape: {wiki.shape}")
print(f"Columns: {wiki.columns}")

# %%
# Schema overview
print("\nSchema:")
for col, dtype in wiki.schema.items():
    print(f"  {col}: {dtype}")

# %% [markdown]
# ### Adjusted vs Raw Prices
#
# **Important**: This dataset contains both raw and adjusted prices.
#
# | Column Type | Examples | Use Case |
# |-------------|----------|----------|
# | Raw | `open`, `high`, `low`, `close`, `volume` | Historical analysis at actual prices |
# | Adjusted | `adj_open`, `adj_high`, `adj_low`, `adj_close`, `adj_volume` | **Backtesting** (handles splits/dividends) |
#
# Always use `adj_*` columns for return calculations and strategy backtesting.

# %% [markdown]
# ## 2. Coverage Summary

# %%
# Overall coverage
print("=== Coverage ===")
print(f"Unique tickers: {wiki['symbol'].n_unique():,}")
print(f"Date range: {wiki['timestamp'].min()} to {wiki['timestamp'].max()}")
print(f"Total rows: {len(wiki):,}")

# %%
# Per-stock statistics
stock_stats = wiki.group_by("symbol").agg(
    [
        pl.len().alias("days"),
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        pl.col("adj_close").mean().alias("avg_price"),
    ]
)

# %% [markdown]
# Per-symbol coverage distribution — number of trading days and mean adjusted price.

# %%
stock_stats.select(["days", "avg_price"]).describe()

# %% [markdown]
# ## 3. Survivorship Analysis
#
# A key feature of this dataset is that it includes stocks that ceased trading
# before the dataset end date. This is critical for avoiding survivorship bias.
#
# **Note**: Stocks marked as "inactive before end" include both:
# - Actually delisted companies (bankruptcy, acquisition, etc.)
# - Stocks with incomplete coverage in this dataset
#
# The important point: these stocks are included, preventing survivorship bias.

# %%
dataset_end = wiki.select(pl.col("timestamp").max()).item()

# Identify stocks that stopped trading before dataset end
stock_stats = stock_stats.with_columns((pl.col("end") < dataset_end).alias("inactive_before_end"))

n_active = stock_stats.filter(~pl.col("inactive_before_end")).height
n_inactive = stock_stats.filter(pl.col("inactive_before_end")).height
total = n_active + n_inactive
inactive_pct = n_inactive / total * 100

print("=== Survivorship Analysis ===")
print(f"Dataset end: {dataset_end}")
print(f"Active at dataset end: {n_active:,} ({n_active / total * 100:.1f}%)")
print(f"Inactive before end: {n_inactive:,} ({inactive_pct:.1f}%)")
print(f"\nThis {inactive_pct:.0f}% inactive rate helps mitigate survivorship bias in backtests.")

# %% [markdown]
# ## 4. Data Quality

# %%
# Check for nulls across columns
null_counts = wiki.null_count()
total_nulls = null_counts.sum_horizontal()[0]
print("=== Data Quality ===")
print(f"Total null values: {total_nulls:,}")

# Show per-column breakdown (only columns with nulls)
for col in null_counts.columns:
    val = null_counts[col][0]
    if val > 0:
        print(f"  {col}: {val:,} ({val / len(wiki) * 100:.4f}%)")

print(f"\nNull rate: {total_nulls / (len(wiki) * len(wiki.columns)) * 100:.4f}%")

# %%
# OHLC invariants on adjusted prices
invariants = check_ohlc_invariants(
    wiki,
    open_col="adj_open",
    high_col="adj_high",
    low_col="adj_low",
    close_col="adj_close",
    volume_col="adj_volume",
)

print("\nOHLC Invariants (adjusted prices):")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

# %% [markdown]
# ## 5. Example: Single Stock

# %%
# AAPL as example
aapl = wiki.filter(pl.col("symbol") == "AAPL").sort("timestamp")

print("=== AAPL Example ===")
print(f"Trading days: {len(aapl):,}")
print(f"Date range: {aapl['timestamp'].min()} to {aapl['timestamp'].max()}")

# %% [markdown]
# Five most recent trading days for AAPL on adjusted prices.

# %%
aapl.select(["timestamp", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]).tail(5)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Mitigates survivorship bias**: 24% of symbols stop trading before the
#    dataset end — including these inactive tickers is what makes the panel
#    usable for unbiased backtests.
# 2. **Always use adjusted prices for returns**: the `adj_*` columns absorb
#    splits and dividends; the raw `open/high/low/close/volume` columns remain
#    available for analyses that need actual traded levels.
# 3. **Long history, broad cross-section**: 3,199 symbols with a max single-symbol
#    span of ~56 years (14,155 trading days), covering multiple market regimes.
# 4. **Clean panel**: null rate is 0.0006% of values and the six adjusted-price
#    OHLC invariants hold on 100% of rows.
#
# **Next**: `02_corporate_actions` validates the adjustment factors that
# produce the `adj_*` columns. **Book reference**: §2.2 (Equities).
