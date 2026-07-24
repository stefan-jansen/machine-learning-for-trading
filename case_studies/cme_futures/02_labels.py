# ---
# jupyter:
#   jupytext:
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
# # CME Futures: Label Engineering
#
# This notebook implements label engineering for the CME Futures case study.
# The primary prediction target is the 1-week (5-day) forward return, matching
# the Friday-close / Monday-open decision cadence from Ch6.
#
# **Learning Objectives**:
# - Construct forward return labels from the roll-continuous (ratio-adjusted) series
# - Build ATR-scaled triple-barrier labels for path-dependent classification
# - Generate walk-forward CV splits with proper purge and embargo
# - Evaluate label quality across products and sectors
#
# **Book Reference**: Chapter 7, Section 7.2 (Label Engineering)
#
# **Prerequisites**: [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) verifies the dataset against the canonical `config/setup.yaml`.

# %%
"""CME Futures: Label Engineering."""

import warnings

import polars as pl
from ml4t.engineer.features.volatility import atr
from ml4t.engineer.labeling import atr_triple_barrier_labels

from data import load_cme_futures
from utils.modeling import get_cv_config
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %%

CASE_DIR = get_case_study_dir("cme_futures")
LABELS_DIR = CASE_DIR / "labels"

STRATEGY_ID = "cme_futures"
HORIZON = 5  # 5 trading days = 1 week

# Product-to-sector mapping (from setup.yaml)
PRODUCT_GROUPS = {
    "equity_index": ["ES", "NQ", "YM", "RTY"],
    "treasuries": ["ZN", "ZB", "ZF", "ZT"],
    "energy": ["CL", "NG", "HO", "RB"],
    "metals": ["GC", "SI", "HG", "PL"],
    "currencies": ["6E", "6J", "6B", "6A", "6C", "6S"],
    "agriculture": ["ZC", "ZS", "ZW", "ZM", "ZL"],
    "livestock": ["LE", "HE", "GF"],
}
PRODUCT_TO_SECTOR = {}
for sector, products in PRODUCT_GROUPS.items():
    for p in products:
        PRODUCT_TO_SECTOR[p] = sector

# %% [markdown]
# ## 1. Load Futures Term Structure Data
#
# The loader returns daily session data with a `position` column (renamed from
# the loader's `tenor`):
# - position 0 = front month (c0)
# - position 1 = second month (c1)
# - position 2 = third month (c2)
#
# **Two price series, two jobs.** Every row carries both a raw traded level
# (`raw_close`) and a roll-continuous, **ratio-adjusted** level
# (`adj_close == raw_close × cum_ratio`). Forward-return labels ride the
# *adjusted* series: ratio back-adjustment rescales the pre-roll history so a
# contract roll does **not** register as a real price move — the return across a
# roll date reflects only the genuine price change, not the front–deferred
# basis gap. The adjustment is applied for all 30 products in this dataset
# (`cum_ratio` departs from 1 on essentially every session). Contemporaneous
# term-structure quantities — carry, roll yield, notional, costs — instead read
# `raw_close`, the level actually traded that day; differencing the *adjusted*
# tenors would read accumulated roll history rather than the live curve. See
# the Ch2 notebook
# [`06_futures_continuous`](../../02_financial_data_universe/06_futures_continuous.ipynb)
# for how the ratio-adjusted continuous series is constructed.

# %%
df = load_cme_futures().rename({"session_date": "timestamp", "tenor": "position"})

print(f"Loaded {len(df):,} rows")
print(f"Products: {sorted(df['product'].unique().to_list())}")
print(f"Positions: {sorted(df['position'].unique().to_list())}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown]
# ## 2. Create Labels
#
# ### Label Types
#
# | Label | Type | Description |
# |-------|------|-------------|
# | `fwd_ret_5d` | Regression | 5-day forward return (total, primary) |
# | `fwd_ret_21d` | Regression | 21-day forward return (monthly variant) |
# | `fwd_tb_5d` | Triple-barrier | ATR-scaled path-dependent labels |


# %%
def create_futures_labels(
    data: pl.DataFrame,
    horizon_primary: int = 5,
    horizon_monthly: int = 21,
    pt_atr: float = 2.0,
    sl_atr: float = 2.0,
    atr_period: int = 14,
) -> pl.DataFrame:
    """Create diverse label types for futures modeling.

    Args:
        data: Raw futures data with product, date, position, OHLC.
        horizon_primary: Primary forward return horizon (5d = 1 week).
        horizon_monthly: Monthly forward return horizon (21d).
        pt_atr: Profit-take threshold in ATR multiples.
        sl_atr: Stop-loss threshold in ATR multiples.
        atr_period: Period for ATR calculation.

    Returns:
        DataFrame with all label columns added.
    """
    data = data.sort(["product", "position", "timestamp"])

    # --- 1. Forward returns (total) ---
    # Returns/labels use the roll-continuous adjusted series (within-tenor returns
    # are identical to raw, but the adjusted series is the correct target).
    # A non-positive base price has no return: CL tenor 2 settled at -13.10 on
    # 2020-04-20. Clipping it to 1e-8 would manufacture a ~1.3e9 return rather
    # than admit the quantity is undefined, so the denominator nulls instead.
    base = pl.when(pl.col("adj_close") > 0).then(pl.col("adj_close"))
    data = data.with_columns(
        (
            pl.col("adj_close").shift(-horizon_primary).over(["product", "position"]) / base - 1
        ).alias(f"fwd_ret_{horizon_primary}d"),
        (
            pl.col("adj_close").shift(-horizon_monthly).over(["product", "position"]) / base - 1
        ).alias(f"fwd_ret_{horizon_monthly}d"),
    )

    # --- 2. Triple-barrier labels ---
    # ATR barriers must ride the adjusted series so roll gaps don't register as
    # true-range spikes.
    data = data.with_columns(
        atr("adj_high", "adj_low", "adj_close", period=atr_period)
        .over(["product", "position"])
        .alias("_atr")
    )

    tb_parts = []
    for (product, position), group in data.group_by(["product", "position"]):
        # Cast Date→Datetime for numpy compatibility in TB labeling
        group = group.sort("timestamp").with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        # atr_triple_barrier_labels computes ATR internally from bare high/low/close.
        # Expose the adjusted series under those names so the barriers ride the
        # roll-continuous path.
        group = group.with_columns(
            pl.col("adj_high").alias("high"),
            pl.col("adj_low").alias("low"),
            pl.col("adj_close").alias("close"),
        )
        tb = atr_triple_barrier_labels(
            data=group,
            atr_tp_multiple=pt_atr,
            atr_sl_multiple=sl_atr,
            atr_period=atr_period,
            max_holding_bars=horizon_primary,
            price_col="close",
            timestamp_col="timestamp",
            group_col="product",
        )
        # Cast back to Date for consistency
        tb = tb.with_columns(
            pl.col("timestamp").cast(pl.Date),
            pl.lit(position).alias("position"),
        )
        tb_parts.append(tb.select(["timestamp", "product", "position", "label"]))

    if not tb_parts:
        msg = "Triple-barrier labeling produced no groups — refusing to write empty labels."
        raise RuntimeError(msg)

    all_tb = pl.concat(tb_parts)
    data = data.join(
        all_tb.rename({"label": f"fwd_tb_{horizon_primary}d"}),
        on=["timestamp", "product", "position"],
        how="left",
    )

    data = data.drop("_atr")
    return data


# %%
print("Creating labels...")
df_labeled = create_futures_labels(df, horizon_primary=HORIZON)
print(f"Labeled data: {len(df_labeled):,} rows")

# %% [markdown]
# ## 3. Label Distribution Summary

# %%
print("=" * 60)
print("LABEL DISTRIBUTION SUMMARY")
print("=" * 60)

# Diagnose the frame that is actually saved and modeled: front month only.
# The deferred tenors carry contracts that are never traded by this strategy,
# so their label distribution says nothing about the training target.
df_front = df_labeled.filter(pl.col("position") == 0)

# Primary regression label
ret_valid = df_front.select("fwd_ret_5d").drop_nulls()
print(f"\nRegression: fwd_ret_5d ({len(ret_valid):,} valid, front month)")
print(f"  Mean: {ret_valid['fwd_ret_5d'].mean():.4f}")
print(f"  Std:  {ret_valid['fwd_ret_5d'].std():.4f}")
print(f"  Skew: {ret_valid['fwd_ret_5d'].skew():.4f}")

# Triple-barrier
tb_dist = df_front.group_by("fwd_tb_5d").agg(pl.len().alias("count")).sort("fwd_tb_5d")
print("\nTriple-Barrier (fwd_tb_5d):")
tb_map = {1: "Profit-Take", -1: "Stop-Loss", 0: "Time-Out"}
for row in tb_dist.iter_rows(named=True):
    if row["fwd_tb_5d"] is not None:
        label = tb_map.get(row["fwd_tb_5d"], "Unknown")
        print(f"  {label}: {row['count']:,}")

# Monthly horizon
ret_21d = df_front.select("fwd_ret_21d").drop_nulls()
print(f"\nMonthly: fwd_ret_21d ({len(ret_21d):,} valid, front month)")
print(f"  Mean: {ret_21d['fwd_ret_21d'].mean():.4f}")
print(f"  Std:  {ret_21d['fwd_ret_21d'].std():.4f}")

# %% [markdown]
# ## 4. Generate CV Configuration
#
# Walk-forward splits per setup.yaml:
# - 5 folds, 8Y train / 1Y test
# - Purge: 5 days (label horizon)
# - Embargo: 5 days (carry rank autocorrelation > 0.9 at weekly lag)
# - Holdout: 2024-2025 (sealed)

# %%
cv_config = get_cv_config("cme_futures")
print("CV Configuration:")
print(f"  Splits: {cv_config.n_splits}")
print(f"  Train size: {cv_config.train_size}")
print(f"  Test size: {cv_config.test_size}")
print(f"  Embargo: {cv_config.embargo_td}")
print(f"  Label horizon: {cv_config.label_horizon}")

# %% [markdown]
# ## 5. Save Artifacts

# %%
label_cols = ["timestamp", "product", "position"]

LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Labels: front month only (position 0). Prediction targets are the traded
# front-month contract; deferred tenors are dropped before writing.
front = df_front
print(f"Front-month labels: {len(front):,} rows (filtered from {len(df_labeled):,})")

# Primary: 5-day forward return
front.select(label_cols + ["fwd_ret_5d"]).drop_nulls().write_parquet(
    LABELS_DIR / "fwd_ret_5d.parquet"
)
print("Saved labels/fwd_ret_5d.parquet")

# Monthly forward return
front.select(label_cols + ["fwd_ret_21d"]).drop_nulls().write_parquet(
    LABELS_DIR / "fwd_ret_21d.parquet"
)
print("Saved labels/fwd_ret_21d.parquet")

# Triple-barrier
front.select(label_cols + ["fwd_tb_5d"]).drop_nulls().write_parquet(
    LABELS_DIR / "fwd_tb_5d.parquet"
)
print("Saved labels/fwd_tb_5d.parquet")

# CV config
cv_config.to_json(CASE_DIR / "config" / "cv_config.json")
print(f"Saved cv_config.json (n_splits={cv_config.n_splits})")

# %% [markdown]
# ## 6. Evaluation Summary
#
# Per-sector dispersion of the primary label. Energy and metals carry the
# widest 5-day return distributions; treasuries and currencies the narrowest
# (close to a 7-to-1 spread in standard deviation). This heterogeneity matters
# downstream: it argues for rank/quantile targets and per-sector normalization
# rather than a single pooled scale.

# %%
# Per-sector statistics of the primary label (front month, non-null).
front_labels = df_labeled.filter(pl.col("position") == 0).drop_nulls(subset=["fwd_ret_5d"])
front_labels = front_labels.with_columns(
    pl.col("product").replace_strict(PRODUCT_TO_SECTOR, default="unknown").alias("sector")
)

sector_stats = (
    front_labels.group_by("sector")
    .agg(
        pl.len().alias("n_obs"),
        pl.col("fwd_ret_5d").mean().round(4).alias("mean_ret"),
        pl.col("fwd_ret_5d").std().round(4).alias("std_ret"),
    )
    .sort("std_ret", descending=True)
)
print("Per-sector fwd_ret_5d (front month), sorted by dispersion:")
print(sector_stats)

# %% [markdown]
# ## Key Takeaways
#
# 1. **5-day primary label** matches the weekly decision cadence from Ch6,
#    balancing cost efficiency with signal responsiveness
# 2. **21-day variant** tests whether the carry signal decays slowly enough
#    for monthly rebalancing (higher edge/cost ratio at lower turnover)
# 3. **Embargo of 5 days** accounts for high autocorrelation in carry ranks
#    (>0.9 at weekly lag) that could leak across train/test boundaries
#
# **Artifacts**: `cv_config.json`,
# `labels/{fwd_ret_5d,fwd_ret_21d,fwd_tb_5d}.parquet`
#
# **Next**: `03_financial_features.py` for carry, momentum, seasonal, and composite features.
