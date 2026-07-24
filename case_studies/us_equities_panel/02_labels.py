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
# # US Equities Panel: Label Engineering
#
# This notebook implements label engineering for the broad US equities panel
# case study. Labels encode forward returns at daily, weekly, and monthly
# horizons for cross-sectional prediction of ~3,200 stocks.
#
# ## Learning Objectives
#
# - Compute multi-horizon forward return labels (1-day primary, 5-day and 21-day variants)
# - Enforce point-in-time universe membership (include delisted until delist date)
# - Apply decision-time liquidity filters (ADV > $1M, price > $5)
# - Generate walk-forward CV configuration (16 splits, 10Y train / 1Y test)
# - Evaluate label quality: distribution, class balance, horizon analysis
#
# ## Book Reference
#
# Chapter 7, Section 7.2 (Label Engineering)
#
# ## Prerequisites
#
# - [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) reviewed (universe and cost feasibility)
# - US equities data available via `load_us_equities()`
#
# ## Output Contract
#
# Artifacts saved to `case_studies/us_equities_panel/labels/`:
# - `fwd_ret_1d.parquet` -- Primary regression labels (1-day forward return)
# - `fwd_ret_5d.parquet` -- Variant: 1-week forward return
# - `fwd_ret_21d.parquet` -- Variant: 1-month forward return
# - `cv_config.json` -- Walk-forward CV configuration (saved to case study root)

# %%
"""US Equities Panel: Label Engineering."""

import subprocess
import warnings
from datetime import UTC, date, datetime

import numpy as np
import polars as pl

from data import load_us_equities
from utils.modeling import get_cv_config
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

CASE_DIR = get_case_study_dir("us_equities_panel")
LABELS_DIR = CASE_DIR / "labels"

# Configuration
START_DATE = "1990-01-01"
END_DATE = "2018-03-31"
HOLDOUT_START = "2016-01-01"

# Label horizons
PRIMARY_HORIZON = 1  # 1-day forward return
VARIANT_HORIZONS = [5, 21]  # 1-week, 1-month

# Liquidity filters (applied at decision time)
MIN_ADV_USD = 1_000_000  # $1M average daily volume
MIN_PRICE = 5.0  # Exclude penny stocks
ADV_WINDOW = 21  # 21-day rolling ADV


# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI

# %% [markdown]
# ## 1. Load and Prepare Data
#
# The Wiki Prices dataset contains daily OHLCV for ~3,200 US equities from
# 1962-2018. We use adjusted prices to account for splits and dividends.
#
# **Key design decisions**:
# - Include delisted stocks until their delist date (survivorship-safe)
# - Apply liquidity and price filters at decision time, not retroactively
# - Use `adj_close` for returns; raw prices for volume calculations

# %%
raw_df = load_us_equities(start_date=START_DATE, end_date=END_DATE)

# Normalize column names and types
if raw_df.schema["timestamp"] == pl.Datetime:
    raw_df = raw_df.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

raw_df = raw_df.sort(["symbol", "timestamp"])

print(f"Raw data: {len(raw_df):,} rows, {raw_df['symbol'].n_unique()} symbols")
print(f"Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")

# %% [markdown]
# ### Point-in-Time Eligibility Filter
#
# The strategic review flagged that the previous implementation used a global
# `n_obs >= 252` filter -- a look-ahead violation that excludes stocks based
# on their full-sample history. Instead, we apply rolling filters at each
# decision date:
#
# 1. **Price > $5**: Excludes penny stocks at decision time
# 2. **ADV > $1M**: Ensures sufficient liquidity for execution
#
# Stocks that are delisted remain in the universe until their last trading date.
# They are not excluded retroactively.


# %%
def apply_pit_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Apply point-in-time eligibility filters.

    Filters applied at each decision date (no look-ahead):
    - Price > $5 (current adj_close)
    - 21-day average dollar volume > $1M

    Stocks are included until they delist (survivorship-safe).
    """
    df = df.sort(["symbol", "timestamp"])

    # Compute rolling ADV (21-day average dollar volume)
    df = df.with_columns(
        (pl.col("adj_close") * pl.col("adj_volume"))
        .rolling_mean(ADV_WINDOW)
        .over("symbol")
        .alias("adv_21d")
    )

    # Apply filters at decision time
    eligible = df.filter((pl.col("adj_close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD))

    n_removed = len(df) - len(eligible)
    pct_removed = 100 * n_removed / len(df)
    print(f"Eligibility filter: {n_removed:,} rows removed ({pct_removed:.1f}%)")
    print(f"  Remaining: {len(eligible):,} rows, {eligible['symbol'].n_unique()} symbols")

    return eligible


# %%
df = apply_pit_filters(raw_df)

# %% [markdown]
# ### Universe Coverage Over Time
#
# Verify that the eligible universe is stable and sufficient for
# cross-sectional analysis at each date.

# %%
daily_counts = (
    df.group_by("timestamp").agg(pl.col("symbol").n_unique().alias("n_stocks")).sort("timestamp")
)

print("\nUniverse coverage:")
print(f"  Min stocks/day: {daily_counts['n_stocks'].min()}")
print(f"  Max stocks/day: {daily_counts['n_stocks'].max()}")
print(f"  Median stocks/day: {daily_counts['n_stocks'].median()}")

# %% [markdown]
# ## 2. Compute Forward Return Labels
#
# Three label horizons matched to the signal families this case study explores:
#
# | Label | Horizon | Signal Family | Purge Required |
# |-------|---------|---------------|----------------|
# | `fwd_ret_1d` (primary) | 1 day | Short-term reversal, microstructure | 1 day |
# | `fwd_ret_5d` (variant) | 5 days | Weekly momentum/reversal | 5 days |
# | `fwd_ret_21d` (variant) | 21 days | Monthly momentum | 21 days |
#
# The 1-day horizon is primary because it matches the daily decision cadence,
# has clean purge/embargo (1-day purge, 0 embargo), and maximizes the number
# of non-overlapping observations.

# %%
# Compute daily returns for label construction
df = df.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("returns")
)

# Compute forward return labels at all horizons
all_horizons = [PRIMARY_HORIZON] + VARIANT_HORIZONS
for h in all_horizons:
    df = df.with_columns(
        (pl.col("adj_close").shift(-h).over("symbol") / pl.col("adj_close") - 1).alias(f"ret_{h}d")
    )

print("Forward return labels computed:")
for h in all_horizons:
    col = f"ret_{h}d"
    valid = df[col].drop_nulls()
    print(f"  {col}: {len(valid):,} valid obs, mean={valid.mean():.6f}, std={valid.std():.4f}")

# %% [markdown]
# ### Effective Sample Size for Overlapping Labels
#
# The primary 1-day label is non-overlapping, but variant labels (5d, 21d) create
# overlapping windows. For $h$-day labels:
#
# $$N_{\text{eff}} \approx \frac{N}{h}$$
#
# The purge gap must also increase to $h$ days. For significance testing of
# cross-sectional IC, use HAC standard errors (Newey-West with bandwidth $\geq h$).

# %% [markdown]
# ## 3. Label Distribution Summary
#
# Report distributions by horizon to verify labels are well-behaved.

# %%
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION SUMMARY")
print("=" * 60)

label_stats = {}
for h in all_horizons:
    col = f"ret_{h}d"
    valid = df.filter(pl.col(col).is_not_null())
    vals = valid[col]

    stats = {
        "horizon_days": h,
        "n_obs": len(valid),
        "n_symbols": valid["symbol"].n_unique(),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skew": float(vals.skew()),
        "p1": float(vals.quantile(0.01)),
        "p99": float(vals.quantile(0.99)),
        "pct_positive": float((vals > 0).mean()),
    }
    label_stats[col] = stats

    print(f"\n{col} ({h}-day forward return):")
    print(f"  Observations: {stats['n_obs']:,} across {stats['n_symbols']} symbols")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std:  {stats['std']:.4f}")
    print(f"  Skew: {stats['skew']:.3f}")
    print(f"  [P1, P99]: [{stats['p1']:.4f}, {stats['p99']:.4f}]")
    print(f"  % Positive: {stats['pct_positive']:.1%}")

# %% [markdown]
# ## 4. CV Configuration
#
# Generate 16 walk-forward splits from `setup.yaml`:
# - **Training**: 10-year rolling window
# - **Test**: 1-year (annual stepping)
# - **First test year**: 2000 (or later if start_date is after 1990)
# - **Holdout**: 2016-2018 (sealed)
#
# The 1-day primary label requires only 1-day purge and 0-day embargo,
# maximizing the usable data at each fold boundary.

# %%
cv_config = get_cv_config("us_equities_panel")

print("CV Configuration:")
print(f"  n_splits: {cv_config.n_splits}")
print(f"  train_size: {cv_config.train_size}")
print(f"  test_size: {cv_config.test_size}")
print(f"  embargo: {cv_config.embargo_td}")
print(f"  label_horizon: {cv_config.label_horizon}")

# %% [markdown]
# ## 5. Label Quality Metrics
#
# Compute basic quality metrics:
# - **Annual label statistics**: Detect non-stationarity across decades
# - **Baseline IC**: Cross-sectional Spearman IC of naive signals vs labels
# - **Label autocorrelation**: Verify labels are not trivially correlated
# - **CV config verification**: Cross-check against setup.yaml

# %%
# Annual label statistics (detect non-stationarity)
annual_stats = (
    df.filter(pl.col("ret_1d").is_not_null())
    .with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.col("ret_1d").mean().alias("mean_ret"),
        pl.col("ret_1d").std().alias("std_ret"),
        pl.col("ret_1d").count().alias("n_obs"),
        pl.col("symbol").n_unique().alias("n_symbols"),
    )
    .sort("year")
)

print("Annual Label Statistics (ret_1d):")
print(annual_stats)

# %% [markdown]
# ### Baseline IC: Naive Signal vs Forward Returns
#
# Cross-sectional Spearman IC of past returns (21-day and 252-day) against
# the 1-day forward return label. This sets the bar that features in
# `03_financial_features.py` must beat. Expected: IC ~ 0.02 for momentum signals.
# Measured on train+validation rows only (timestamp < holdout_start) so this
# expectation-setting diagnostic never reads the sealed holdout.

# %%
from scipy.stats import spearmanr

labeled = df.filter(pl.col("ret_1d").is_not_null()).sort(["symbol", "timestamp"])
labeled = labeled.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(21).over("symbol") - 1).alias("past_ret_21d"),
    (pl.col("adj_close") / pl.col("adj_close").shift(252).over("symbol") - 1).alias(
        "past_ret_252d"
    ),
)
labeled = labeled.drop_nulls(subset=["past_ret_21d", "past_ret_252d", "ret_1d"])

# Sample every 5th date for IC computation, restricted to pre-holdout rows
ic_dates = (
    labeled.filter(pl.col("timestamp") < date.fromisoformat(HOLDOUT_START))["timestamp"]
    .unique()
    .sort()
    .gather_every(5)
    .to_list()
)
labeled_sample = labeled.filter(pl.col("timestamp").is_in(ic_dates))

baseline_ic = {}
for signal_col in ["past_ret_21d", "past_ret_252d"]:
    ic_per_date = []
    for date_val in ic_dates:
        day_data = labeled_sample.filter(pl.col("timestamp") == date_val)
        if len(day_data) < 30:
            continue
        corr, _ = spearmanr(day_data[signal_col].to_numpy(), day_data["ret_1d"].to_numpy())
        if not np.isnan(corr):
            ic_per_date.append(corr)
    if ic_per_date:
        ic_arr = np.array(ic_per_date)
        baseline_ic[signal_col] = {
            "mean_ic": float(ic_arr.mean()),
            "std_ic": float(ic_arr.std()),
            "t_stat": float(ic_arr.mean() / (ic_arr.std() / np.sqrt(len(ic_arr)))),
            "n_dates": len(ic_arr),
        }

print("\nBaseline IC (naive signals vs ret_1d):")
for sig, stats in baseline_ic.items():
    print(f"  {sig}: IC={stats['mean_ic']:.4f} (t={stats['t_stat']:.2f}, n={stats['n_dates']})")

del labeled, labeled_sample

# %% [markdown]
# ### Label Autocorrelation
#
# Cross-sectional IC of $\text{ret\_1d}(t)$ vs $\text{ret\_1d}(t-k)$ for
# $k=1 \ldots 5$. High autocorrelation indicates overlapping information
# and affects purge requirements.

# %%
auto_df = df.filter(pl.col("ret_1d").is_not_null()).sort(["symbol", "timestamp"])
for lag in range(1, 6):
    auto_df = auto_df.with_columns(
        pl.col("ret_1d").shift(lag).over("symbol").alias(f"ret_1d_lag{lag}")
    )

auto_sample_dates = auto_df["timestamp"].unique().sort().gather_every(5).to_list()
auto_sample = auto_df.filter(pl.col("timestamp").is_in(auto_sample_dates))

label_autocorr = {}
for lag in range(1, 6):
    lag_col = f"ret_1d_lag{lag}"
    ic_per_date = []
    for date_val in auto_sample_dates:
        day_data = auto_sample.filter(pl.col("timestamp") == date_val).drop_nulls(
            subset=["ret_1d", lag_col]
        )
        if len(day_data) < 30:
            continue
        corr, _ = spearmanr(day_data[lag_col].to_numpy(), day_data["ret_1d"].to_numpy())
        if not np.isnan(corr):
            ic_per_date.append(corr)
    if ic_per_date:
        ic_arr = np.array(ic_per_date)
        label_autocorr[lag] = float(ic_arr.mean())

print("\nLabel autocorrelation (cross-sectional IC of ret_1d(t) vs ret_1d(t-k)):")
for lag, ac in label_autocorr.items():
    print(f"  lag {lag}: IC = {ac:.4f}")

del auto_df, auto_sample

# %% [markdown]
# **Interpretation**: Lag-1 autocorrelation for 1-day returns is typically
# negative (short-term reversal), confirming that the 1-day label is
# non-overlapping and that a 1-day purge is sufficient.

# %% [markdown]
# ### CV Config Verification
#
# Cross-check the generated CV config against `setup.yaml` to ensure
# they are consistent.

# %%
import yaml

with open(CASE_DIR / "config" / "setup.yaml") as f:
    setup = yaml.safe_load(f)

_eval = setup.get("evaluation", {})
checks = {
    "n_splits": (cv_config.n_splits, _eval.get("n_splits")),
}

print("\nCV Config vs setup.yaml:")
all_ok = True
for param, (cv_val, setup_val) in checks.items():
    match = str(cv_val) == str(setup_val)
    status = "OK" if match else "MISMATCH"
    if not match:
        all_ok = False
    print(f"  {param}: cv_config={cv_val}, setup={setup_val} [{status}]")

if all_ok:
    print("  All checks passed")
else:
    print("  WARNING: CV config and setup.yaml are inconsistent")

# %% [markdown]
# ## 6. Save Artifacts
#
# Save all label files, prices, and CV config for downstream consumption.

# %%
# Prepare label DataFrames (fwd_ prefix distinguishes labels from backward-looking features)
labels_dict = {}
for h in all_horizons:
    src_col = f"ret_{h}d"
    label_id = f"fwd_ret_{h}d"
    label_df = df.select(["symbol", "timestamp", src_col]).rename({src_col: label_id}).drop_nulls()
    labels_dict[label_id] = label_df

# Evaluation metadata (used in results JSON below)
evaluation = {
    "case_study_id": "us_equities_panel",
    "primary_label": "fwd_ret_1d",
    "horizons": {f"fwd_ret_{h}d": label_stats.get(f"ret_{h}d", {}) for h in all_horizons},
    "universe": {
        "n_symbols_total": df["symbol"].n_unique(),
        "median_stocks_per_day": int(daily_counts["n_stocks"].median()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
    },
    "filters": {
        "min_adv_usd": MIN_ADV_USD,
        "min_price": MIN_PRICE,
        "adv_window": ADV_WINDOW,
    },
}

# %%
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Save labels (flat layout: labels/*.parquet)
for name, label_df in labels_dict.items():
    path = LABELS_DIR / f"{name}.parquet"
    label_df.write_parquet(path)
    print(f"Saved {name}.parquet ({len(label_df):,} rows)")

# Save CV config (case study root)
cv_config.to_json(CASE_DIR / "config" / "cv_config.json")
print(f"Saved cv_config.json (n_splits={cv_config.n_splits})")
# %% [markdown]
# ### Symbol Count Reconciliation
#
# The universe shrinks at each pipeline stage due to point-in-time filters and
# horizon-dependent edge trimming:
#
# - **Raw universe**: 3,199 symbols (from [`01_feasibility_analysis`](01_feasibility_analysis.ipynb))
# - **After PIT filters (1d)**: ~3,172 symbols (ADV > $1M, price > $5 removes thinly traded stocks)
# - **After PIT filters (5d)**: ~3,170 symbols (5-day horizon trims a few more at edges)
# - **After PIT filters (21d)**: ~3,157 symbols (21-day horizon trims the most)
#
# The Fundamental Law calculations should use the 1-day count (~3,172) since
# that is the primary label's cross-sectional breadth.

# %%
print("Symbol count reconciliation:")
print(f"  Raw universe:       {df['symbol'].n_unique():,}")
for name, ldf in labels_dict.items():
    print(f"  {name}: {ldf['symbol'].n_unique():,} symbols, {len(ldf):,} obs")

# %% [markdown]
# ## Results Collection
#
# Collect label metadata for pipeline tracking.


# %%
def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


primary_stats = label_stats.get("ret_1d", {})

results = {
    "case_study_id": "us_equities_panel",
    "chapter": 7,
    "stage": "labels",
    "timestamp": datetime.now(UTC).isoformat(),
    "git_commit": _git_commit_hash(),
    "notebook": "case_studies/us_equities_panel/02_labels.py",
    "summary": {
        "primary_label": "fwd_ret_1d",
        "horizons": [f"fwd_ret_{h}d" for h in all_horizons],
        "n_observations": primary_stats.get("n_obs", 0),
        "n_symbols": primary_stats.get("n_symbols", 0),
        "label_mean": primary_stats.get("mean", 0),
        "label_std": primary_stats.get("std", 0),
    },
    "techniques": {
        "label_types": ["regression_forward_return"],
        "horizons_days": all_horizons,
        "pit_filters": ["adv_gt_1m", "price_gt_5"],
        "survivorship_handling": "include_delisted_until_delist_date",
    },
    "diagnostics": {
        "n_splits": cv_config.n_splits,
        "pct_positive_1d": primary_stats.get("pct_positive", 0),
        "baseline_ic": {
            sig: {"mean_ic": round(s["mean_ic"], 4), "t_stat": round(s["t_stat"], 2)}
            for sig, s in baseline_ic.items()
        },
        "label_autocorrelation": {f"lag_{k}": round(v, 4) for k, v in label_autocorr.items()},
    },
    "key_findings": [
        f"1-day primary label: {primary_stats.get('n_obs', 0):,} obs across {primary_stats.get('n_symbols', 0)} symbols",
        f"PIT filters remove ~{100 * (len(raw_df) - len(df)) / len(raw_df):.0f}% of raw observations",
        f"Label std ranges from {label_stats.get('ret_1d', {}).get('std', 0):.4f} (1d) to {label_stats.get('ret_21d', {}).get('std', 0):.4f} (21d)",
        "16-fold walk-forward CV with 10Y train / 1Y test",
    ],
}


# %% [markdown]
# ## Key Takeaways
#
# 1. **1-day forward return** is the primary label, matching the daily decision
#    cadence and enabling clean purge (1-day, no overlap).
#
# 2. **Point-in-time filters** (ADV > $1M, price > $5) are applied at each
#    decision date, not retroactively. This avoids look-ahead bias in universe
#    construction.
#
# 3. **Survivorship-safe**: Delisted stocks remain in the universe until their
#    last trading date. No future knowledge is used to exclude stocks.
#
# 4. **Multi-horizon variants** (5-day, 21-day) enable testing which signal
#    families work best at different frequencies -- reversal at daily,
#    momentum at monthly.
#
# **Next**: `03_financial_features.py` in Ch8 adds momentum, volatility, and composite
# cross-sectional features.
