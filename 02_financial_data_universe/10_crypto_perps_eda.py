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
# # Crypto Perps — Exploratory Data Analysis
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
# Chapter 2 §2.2 (asset-class market data landscape — digital assets).
#
# ## Prerequisites
#
# - Familiarity with daily OHLCV equity data (`01_us_equities_eda`).
# - The Binance Futures parquet at `$ML4T_DATA_PATH/crypto/` (OHLCV + premium).
# - Methodology continues in `11_crypto_premium_analysis`; the case-study
#   pipeline lives under `case_studies/crypto_perps_funding/`.

# %%
"""Crypto Perps EDA — hourly OHLCV and premium index exploration."""

import polars as pl

from data import load_crypto_perps, load_crypto_premium
from utils.data_quality import check_ohlc_invariants, per_asset_stats

# %% tags=["parameters"]
MAX_SYMBOLS = 0  # 0 = all symbols

# %% [markdown]
# ## 1. Load and Inspect OHLCV
#
# Hourly OHLCV data from Binance Futures for 19 cryptocurrencies.
# Trading is 24/7 (8,760 hours/year vs 252 days for equities).

# %%
ohlcv = load_crypto_perps(frequency="1h")

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
# Premium values are stored as **decimals** (0.001 = 0.1%). When displaying,
# multiply by 100 for percentage representation.

# %%
premium = load_crypto_premium(frequency="8h")
print("=== Premium Dataset ===")
print(f"Shape: {premium.shape}")
print(f"Columns: {premium.columns}")

# %%
# Premium range — values are decimals, not percentages
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
# ## 4. Data Quality

# %%
# OHLC invariants
invariants = check_ohlc_invariants(ohlcv)
print("=== OHLC Invariants ===")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

# %%
# Check for nulls
ohlcv_nulls = ohlcv.null_count().sum_horizontal()[0]
premium_nulls = premium.null_count().sum_horizontal()[0]
print(f"\nNull values: OHLCV={ohlcv_nulls}, Premium={premium_nulls}")

# %%
# Check for gaps > 1 hour (use BTC as reference)
btc = ohlcv.filter(pl.col("symbol") == "BTCUSDT").sort("timestamp")
btc_gaps = btc.with_columns(pl.col("timestamp").diff().dt.total_hours().alias("hours_diff")).filter(
    pl.col("hours_diff") > 1
)

print(f"\nGaps > 1 hour in BTC: {len(btc_gaps)}")
if len(btc_gaps) > 0:
    print("(Small gaps expected during exchange maintenance)")

# %% [markdown]
# ## 5. Joining OHLCV and Premium
#
# Use left join to preserve all OHLCV rows and identify missing premium coverage.

# %%
# Left join to identify missing premium data
combined = ohlcv.join(premium, on=["timestamp", "symbol"], how="left")

# Coverage analysis
total_rows = len(combined)
missing_premium = combined.filter(pl.col("premium_index_close").is_null()).height
coverage_pct = (total_rows - missing_premium) / total_rows * 100

print("=== Join Coverage ===")
print(f"OHLCV rows: {len(ohlcv):,}")
print(f"Premium rows: {len(premium):,}")
print(f"Combined rows: {total_rows:,}")
print(f"Missing premium: {missing_premium:,} ({100 - coverage_pct:.2f}%)")
print(f"Coverage: {coverage_pct:.2f}%")

# %%
# Where does the missing premium concentrate?
missing_by_symbol = (
    combined.filter(pl.col("premium_index_close").is_null())
    .group_by("symbol")
    .len()
    .sort("len", descending=True)
)
print("Missing premium by symbol (top 5):")
missing_by_symbol.head(5)

# %% [markdown]
# ## Key Takeaways
#
# 1. **24/7 trading**: Crypto runs continuously — 52,608 unique hours across
#    six full years (2020-01-01 to 2025-12-31), ~8,760 hours/year for the
#    longest-history symbols.
# 2. **Universe**: 19 perpetual contracts span 866,484 OHLCV bars; symbol
#    coverage is non-uniform because contracts list at different dates.
# 3. **Premium units**: Stored as decimals (0.001 = 0.1%); always multiply by
#    100 for percentage display in figures or text.
# 4. **Frequency mismatch**: OHLCV is hourly, premium is 8-hourly, so a left
#    join lands ~12.4% premium coverage on the hourly grid by design.
# 5. **Clean data**: OHLC invariants hold for 100% of records on this
#    snapshot; BTC has zero gaps > 1 hour.
#
# ## Next Steps
#
# - `11_crypto_premium_analysis`: Premium dynamics, basis seasonality, and
#   alignment to the 8-hourly funding cadence.
# - Chapter 8: Feature engineering for premium signals
#   (`case_studies/crypto_perps_funding/03_financial_features.py`).
# - Chapter 16: Backtests for the funding-arbitrage case study.
