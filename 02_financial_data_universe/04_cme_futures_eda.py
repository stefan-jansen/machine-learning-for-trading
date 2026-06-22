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
# # CME Futures — Exploratory Data Analysis
#
# **Docker image**: `ml4t`
#
# **Purpose**: Profile the 30-product CME futures dataset (Databento, hourly,
# 2011–2025) and surface the contract / continuous structure that downstream
# notebooks rely on.
#
# **Learning objectives**:
#
# - Understand the futures data hierarchy: product → contract → continuous series.
# - Load individual contracts and continuous (rolled) series via
#   `load_cme_futures` and inspect the canonical `timestamp` / `product`
#   schema.
# - Summarize per-product coverage and group products by asset class.
# - Verify OHLC invariants on a representative continuous series.
#
# **Book reference**: §2.2 ("The Asset-Class Market Data Landscape" — Futures).
#
# **Prerequisites**: `data` package on `PYTHONPATH`; CME parquet present at
# `ML4T_DATA_PATH/futures/`. Run `python data/futures/market/download.py` if
# missing (Databento API key required).

# %%
"""CME Futures — Exploratory data analysis of the futures universe."""

import polars as pl

from data import list_cme_products, load_cme_futures
from utils.data_quality import check_ohlc_invariants

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
MAX_SYMBOLS = 0  # 0 = all

# %% [markdown]
# ## 1. Configuration and Data Discovery
#
# The futures data uses a Hive-partitioned structure for efficient queries:
# - `futures/continuous/product={PRODUCT}/`: Volume-rolled continuous contracts (hourly)
# - `futures/individual/{PRODUCT}/data.parquet`: Individual contract price data
#
# We use `load_cme_futures()` for proper data loading with partition pruning.

# %%
# Discover available products via the CME loader
products = list_cme_products()

print("=== Futures Universe ===")
print(f"Available products: {len(products)}")
print(f"\nProducts: {', '.join(products)}")

# %% [markdown]
# Map each product to a coarse asset-class bucket. The mapping covers every
# product in the dataset; downstream chapters use the same bucket labels for
# universe-construction and risk reporting.

# %%
ASSET_CLASS_MAP = {
    "ES": "Equity Index",
    "NQ": "Equity Index",
    "YM": "Equity Index",
    "RTY": "Equity Index",
    "ZN": "Rates",
    "ZB": "Rates",
    "ZF": "Rates",
    "ZT": "Rates",
    "CL": "Energy",
    "NG": "Energy",
    "HO": "Energy",
    "RB": "Energy",
    "GC": "Metals",
    "SI": "Metals",
    "HG": "Metals",
    "PL": "Metals",
    "6E": "FX",
    "6J": "FX",
    "6B": "FX",
    "6A": "FX",
    "6C": "FX",
    "6S": "FX",
    "ZC": "Grains",
    "ZS": "Grains",
    "ZW": "Grains",
    "ZM": "Grains",
    "ZL": "Grains",
    "LE": "Livestock",
    "HE": "Livestock",
    "GF": "Livestock",
}

# %%
# Count products by asset class
class_counts = (
    pl.DataFrame({"product": products})
    .with_columns(asset_class=pl.col("product").replace(ASSET_CLASS_MAP))
    .group_by("asset_class")
    .len()
    .sort(["len", "asset_class"], descending=[True, False])
)

print("Products by Asset Class:")
class_counts

# %% [markdown]
# ## 2. Data Structure Example: E-mini S&P 500 (ES)
#
# ### Futures Key Hierarchy
#
# | Level | Example | Description |
# |-------|---------|-------------|
# | **Product** | ES | The underlying (E-mini S&P 500) |
# | **Contract** | ESH4 | Specific expiration (March 2024) |
# | **Continuous** | c0, c1 | Front month, first deferred |

# %%
es_individual = load_cme_futures(products=["ES"], continuous=False, frequency="hourly")

print("=== ES Individual Contracts ===")
print(f"Shape: {es_individual.shape}")
print(f"Columns: {es_individual.columns}")
print(f"Date range: {es_individual['timestamp'].min()} to {es_individual['timestamp'].max()}")
print(f"Unique contracts: {es_individual['instrument_id'].n_unique()}")

# %%
es_continuous = load_cme_futures(products=["ES"], tenors=[0], continuous=True, frequency="hourly")

print("=== ES Continuous Series (front month) ===")
print(f"Shape: {es_continuous.shape}")
print(f"Date range: {es_continuous['timestamp'].min()} to {es_continuous['timestamp'].max()}")

# %% [markdown]
# Each individual contract trades for a finite window before expiry. Aggregating
# by `instrument_id` shows the rollover pattern — quarterly contracts overlap
# during the roll period.

# %%
contract_stats = (
    es_individual.group_by("instrument_id")
    .agg(
        pl.col("timestamp").min().alias("first_trade"),
        pl.col("timestamp").max().alias("last_trade"),
        pl.col("volume").sum().alias("total_volume"),
        pl.len().alias("observations"),
    )
    .sort("first_trade")
)
print(f"Total ES contracts: {len(contract_stats)}")
print("Most recent 5 contracts:")
contract_stats.tail(5)

# %% [markdown]
# ## 3. Coverage Summary
#
# Check data availability across all products.


# %% [markdown]
# Summarize per-product coverage by loading the front-month continuous series
# (`tenor=0`) for every product and recording its row count and date range.


# %%
def get_product_coverage(product_list: list[str]) -> pl.DataFrame:
    """Summarize continuous series coverage for each product (front month)."""
    summaries = []
    for product in product_list:
        df = load_cme_futures(products=[product], tenors=[0], continuous=True, frequency="hourly")
        summaries.append(
            {
                "product": product,
                "asset_class": ASSET_CLASS_MAP[product],
                "rows": len(df),
                "start_date": str(df["timestamp"].min())[:10],
                "end_date": str(df["timestamp"].max())[:10],
            }
        )
    return pl.DataFrame(summaries)


# %%
coverage = get_product_coverage(products)
print(f"Products with data: {len(coverage)} / {len(products)}")
print("Coverage by asset class:")
coverage.group_by("asset_class").len().sort(["len", "asset_class"], descending=[True, False])

# %% [markdown]
# A handful of representative products from each asset-class bucket:

# %%
key_products = ["ES", "NQ", "CL", "GC", "ZN", "6E", "ZC"]
coverage.filter(pl.col("product").is_in(key_products))

# %% [markdown]
# ## 4. Data Quality

# %%
invariants = check_ohlc_invariants(es_continuous)
print("=== OHLC Invariants (ES Continuous) ===")
for row in invariants.iter_rows(named=True):
    status = "[OK]" if row["valid_pct"] >= 99.99 else "[WARN]"
    print(f"  {status} {row['check']}: {row['valid_pct']:.2f}%")

# %% [markdown]
# ## Key Takeaways
#
# 1. **30 products across 7 asset-class buckets**: FX (6), Grains (5),
#    Energy / Equity Index / Metals / Rates (4 each), Livestock (3).
# 2. **Hierarchy**: each product has 100+ individual contracts (194 for ES) and
#    one or more continuous series; downstream notebooks operate on the
#    continuous front month unless they specifically need contract-level data.
# 3. **Hourly granularity**, full coverage 2011-01-02 through 2025-12-30 for
#    products with the longest history. ES individual contract data starts
#    later (2016) because earlier contracts have already rolled off.
# 4. **Canonical schema**: `timestamp` for time and `product` for entity (CME's
#    contract identity is non-trivial — see also `instrument_id` for individual
#    contracts).
# 5. **OHLC invariants hold at 100% for ES continuous** — all six checks pass
#    on every observation.
#
# ### Next Steps
#
# - **`05_futures_session_aggregation`**: Aligning hourly bars to CME trading
#   sessions.
# - **`06_futures_continuous`**: Roll detection and the three adjustment
#   methods (ratio, difference, calendar).
# - **Chapter 8**: Feature engineering on term structure and roll yield.
