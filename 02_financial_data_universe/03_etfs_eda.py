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
# **Purpose**: Profile the 100-ETF candidate universe sourced from Yahoo Finance
# and confirm the category coverage, history, and data-quality characteristics
# that drive the ETF rotation case study.
#
# **Learning objectives**:
#
# - Load the ETF panel via `data.load_etfs` and inspect its canonical schema.
# - Quantify per-symbol coverage and identify ETFs with shorter history.
# - Check OHLC invariants and null rates across the full panel.
# - Compare liquidity and price ranges across asset-class categories.
#
# **Book reference**: §2.2 ("The Asset-Class Market Data Landscape" — ETPs).
#
# **Prerequisites**: `data` package on `PYTHONPATH`; ETF parquet present at
# `ML4T_DATA_PATH/etfs/market/`. Run `python data/etfs/market/download.py` if
# missing.

# %%
"""ETFs — Exploratory data analysis of the multi-asset ETF universe."""

import polars as pl
from ml4t.data.etfs import ETFDataManager

from data import load_etfs
from utils.data_quality import check_ohlc_invariants, per_asset_stats
from utils.paths import REPO_ROOT

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all

# %% [markdown]
# ## 1. Load and Inspect
#
# The ETF universe is stored as a single Parquet file containing daily OHLCV data
# for 50 ETFs spanning multiple asset classes.

# %%
etfs = load_etfs()

print("=== ETF Dataset ===")
print(f"Shape: {etfs.shape}")
print(f"Columns: {etfs.columns}")

# %%
# Schema overview
print("\nSchema:")
for col, dtype in etfs.schema.items():
    print(f"  {col}: {dtype}")

# %% [markdown]
# ### Adjusted Prices
#
# Yahoo Finance returns split- and dividend-adjusted OHLC. The `close` column is
# the adjusted close, so returns can be computed directly without a separate
# `adj_close` column.

# %% [markdown]
# ## 2. Coverage Summary

# %%
# Unique symbols
symbols = etfs["symbol"].unique().sort().to_list()
print("=== Coverage ===")
print(f"Number of ETFs: {len(symbols)}")
print(f"\nSymbols: {', '.join(symbols)}")

# %%
# Overall date range
date_range = etfs.select(
    [
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
        pl.col("timestamp").n_unique().alias("unique_dates"),
    ]
)

print(f"\nDate range: {date_range['start'][0]} to {date_range['end'][0]}")
print(f"Trading days: {date_range['unique_dates'][0]}")

# %%
symbol_stats = per_asset_stats(
    etfs,
    time_col="timestamp",
    asset_col="symbol",
    price_col="close",
    volume_col="volume",
)

full_start = date_range["start"][0]
full_end = date_range["end"][0]

partial = symbol_stats.filter(
    (pl.col("start").cast(pl.Date) != pl.lit(full_start).cast(pl.Date))
    | (pl.col("end").cast(pl.Date) != pl.lit(full_end).cast(pl.Date))
)

print(f"Symbols with full coverage: {len(symbol_stats) - len(partial)}")
print(f"Symbols with partial coverage: {len(partial)}")

# %% [markdown]
# Most ETFs predate the 2006 start of the dataset, but a sizeable minority were
# launched later — visible below as ETFs whose first observation is after
# 2006-01-03.

# %%
partial.select(["symbol", "start", "end", "rows"]).sort("start")

# %% [markdown]
# ## 3. ETF Categories
#
# Six categories cover 50 ETFs that span the major asset-class buckets used by
# the rotation case study. The full universe contains 100 ETFs; the remaining
# 50 are kept as candidates for the universe-construction work in Chapter 6.

# %%
ETF_CATEGORIES = {
    "US Equity Broad": ["SPY", "QQQ", "IWM", "DIA", "VTI", "MDY", "IJR"],
    "US Sectors": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "VNQ", "IYR"],
    "International": [
        "EFA",
        "EEM",
        "VEA",
        "VWO",
        "EWJ",
        "EWG",
        "FXI",
        "EWZ",
        "EWY",
        "EWC",
        "EWA",
        "ACWI",
    ],
    "Fixed Income": ["AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB", "MUB"],
    "Commodities": ["GLD", "SLV", "USO", "UNG", "DBC", "GSG"],
    "Specialty": ["IBB", "SMH", "KRE", "OIH"],
}

print("=== ETF Categories ===")
for category, tickers in ETF_CATEGORIES.items():
    available = [t for t in tickers if t in symbols]
    print(f"  {category}: {len(available)}/{len(tickers)} ETFs")

# %% [markdown]
# ## 4. Data Quality

# %%
# Check for nulls
null_counts = etfs.null_count()
total_nulls = null_counts.sum_horizontal()[0]
print("=== Data Quality ===")
print(f"Total null values: {total_nulls}")

# %%
# Check for zero volume days
zero_volume = etfs.filter(pl.col("volume") == 0)
print(f"Zero volume rows: {len(zero_volume)} ({100 * len(zero_volume) / len(etfs):.2f}%)")

# %%
# OHLC invariants
invariants = check_ohlc_invariants(etfs)
print("\nOHLC Invariants:")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

# Count violations
violations = etfs.filter(
    (pl.col("high") < pl.col("low"))
    | (pl.col("high") < pl.col("open"))
    | (pl.col("high") < pl.col("close"))
)
print(f"\nTotal OHLC violations: {len(violations)}")

# %% [markdown]
# Violations occur where `high < close` or `low > close` after price adjustment.
# These arise from floating-point/rounding precision in the stored adjusted OHLC
# values (the same cumulative ratio is applied across all four fields, but small
# per-field rounding can break the strict invariants the raw quotes satisfied).
# At a tenth of a percent of rows they are immaterial for return and feature
# calculations but worth being aware of when computing intraday range statistics.

# %% [markdown]
# ## 5. Volume and Liquidity Distribution
#
# Understanding volume distributions helps identify liquidity constraints for trading.

# %%
# Average daily volume by ETF
volume_stats = (
    etfs.group_by("symbol")
    .agg(
        [
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("volume").median().alias("median_volume"),
            pl.col("volume").std().alias("volume_std"),
        ]
    )
    .sort("avg_volume", descending=True)
)

print("=== Volume Distribution (Top 10 Most Liquid) ===")
volume_stats.head(10)

# %%
# Volume by category
print("\n=== Average Daily Volume by Category ===")
for category, tickers in ETF_CATEGORIES.items():
    category_vol = (
        etfs.filter(pl.col("symbol").is_in(tickers)).select(pl.col("volume").mean()).item()
    )
    if category_vol is not None:
        print(f"  {category}: {category_vol:,.0f} shares/day")

# %% [markdown]
# ## 6. Price Distribution
#
# Price levels across the ETF universe span a wide range.

# %%
# Current price levels (most recent date)
latest = etfs.filter(pl.col("timestamp") == etfs["timestamp"].max())
price_dist = latest.select(
    [
        pl.col("close").min().alias("min_price"),
        pl.col("close").max().alias("max_price"),
        pl.col("close").median().alias("median_price"),
        pl.col("close").mean().alias("mean_price"),
    ]
)

print("=== Price Distribution (Latest Date) ===")
price_dist

# %%
# Price range by category
print("=== Price Range by Category ===")
for category, tickers in ETF_CATEGORIES.items():
    category_prices = latest.filter(pl.col("symbol").is_in(tickers))
    min_p = category_prices["close"].min()
    max_p = category_prices["close"].max()
    print(f"  {category}: ${min_p:.2f} - ${max_p:.2f}")

# %% [markdown]
# ## 7. Loading by Symbol or via the ml4t-data Library
#
# `load_etfs(symbols=[...])` filters the panel to a subset; `ETFDataManager` is
# the config-driven entry point used by the production download/refresh workflow
# in `data/etfs/market/`.

# %%
spy = load_etfs(symbols=["SPY"])
print(f"SPY via loader: {spy.shape}")

# %%
config_path = REPO_ROOT / "data" / "etfs" / "market" / "config.yaml"
etf_mgr = ETFDataManager.from_config(str(config_path))
configured = sum(len(group["symbols"]) for group in etf_mgr.config.tickers.values())
print(f"ETFDataManager loaded from {config_path}")
print(f"  Provider:           {etf_mgr.config.provider}")
print(f"  Date range:         {etf_mgr.config.start} to {etf_mgr.config.end}")
print(f"  Configured symbols: {configured} across {len(etf_mgr.config.tickers)} categories")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Pre-adjusted prices**: Yahoo Finance returns split- and dividend-adjusted
#    OHLC. The `close` column is the adjusted close — return calculations need
#    no further adjustment.
# 2. **Coverage**: 100 ETFs across daily history from 2006-01-03 to 2025-12-31
#    (5,031 trading days), with 59 ETFs spanning the full window and 41 starting
#    later as new products were launched.
# 3. **Categorization**: Six categories (US Equity Broad, US Sectors,
#    International, Fixed Income, Commodities, Specialty) cover 50 of the 100
#    ETFs and form the candidate set for the rotation case study; the other 50
#    are reserved for the universe-construction work in Chapter 6.
# 4. **Mostly clean data**: Zero nulls and 473 OHLC violations
#    (`high < close` or `low > close`) — about 0.1% of rows, immaterial for
#    return and feature calculations.
# 5. **Liquidity variation**: SPY averages 126M shares/day; the Commodities
#    bucket averages 5.5M — a 23× difference that matters for transaction-cost
#    modeling in later chapters.
#
# ### Next Steps
#
# - **§2.6 / `13_data_quality_framework`**: Systematic data-quality checks across
#   the seven datasets.
# - **§2.7 / `15_survivorship_bias_detection`**: Survivorship and selection bias
#   in equity panels — a contrast to the ETF universe shown here.
# - **Chapter 6**: Universe construction filters this 100-ETF candidate pool down
#   to the trading universe used by the ETF rotation case study.
