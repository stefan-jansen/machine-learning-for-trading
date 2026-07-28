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
# # FX Pairs: Label Engineering
#
# **Chapter 7: Defining the Learning Task**
#
# This notebook implements label engineering for the FX Pairs case study.
# Labels encode the economic hypothesis: can cross-sectional reversal
# predict forward returns across a 20-pair FX universe?
#
# **Learning Objectives**:
# - Aggregate 4-hour bars to daily using NY 5PM rollover convention
# - Compute forward return labels at multiple horizons (1d, 5d, 21d)
# - Build walk-forward CV configuration respecting FX calendar
# - Evaluate baseline IC of raw 126-day reversal signal against labels
#
# **Book Reference**: Chapter 7, Section 7.2 (Label Engineering)
#
# **Prerequisites**: [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) completed; FX data via `load_fx_pairs()`
#
# **Output Contract**:
# - `labels/fwd_ret_1d.parquet` -- Primary: 1-day forward returns
# - `labels/fwd_ret_5d.parquet` -- Variant: 1-week forward returns
# - `labels/fwd_ret_21d.parquet` -- Variant: 1-month forward returns
#
# The walk-forward CV configuration is defined in `config/setup.yaml`; this notebook
# validates that config and materializes the splits via `generate_cv_splits`.
# `config/cv_config.json` is a snapshot frozen at release time -- downstream
# notebooks derive folds from `setup.yaml` via `modeling_fold_boundaries`, not
# from that file, since `generate_cv_splits` no longer reproduces it exactly.

# %%
"""FX Pairs: Label Engineering."""

import subprocess
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from scipy.stats import spearmanr

from data import load_fx_pairs
from utils.cv_splits import generate_cv_splits, load_evaluation_config
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
START_DATE = "2011-01-01"
MAX_SYMBOLS = 0

# %%
CASE_DIR = get_case_study_dir("fx_pairs")
LABELS_DIR = CASE_DIR / "labels"

START_DATE = "2011-01-01"
END_DATE = "2025-12-31"
HOLDOUT_START = "2024-01-01"

# %% [markdown]
# ## 1. Load and Aggregate to Daily
#
# FX data arrives as 4-hour bars. The setup (Ch6) specifies daily NY 5PM close
# as the decision cadence. We aggregate 4H bars to daily using the CME_FX
# trading calendar, which implements the standard NY 5PM session rollover.


# %%
def aggregate_4h_to_daily(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 4-hour FX bars to daily OHLCV using CME_FX trading calendar.

    Uses ml4t-diagnostic's TradingCalendar to assign each 4H bar to its
    correct FX trading session (Mon-Fri, NY 5PM rollover).
    """
    cal = TradingCalendar("CME_FX")
    sessions = cal.get_sessions(pd.DatetimeIndex(df["timestamp"].to_pandas()))

    # Keep the original 4H timestamp as `bar_ts` so first()/last() can sort
    # within the session group — polars group_by does not contractually
    # preserve input row order.
    daily = (
        df.rename({"timestamp": "bar_ts"})
        .with_columns(pl.Series("timestamp", sessions.values).cast(pl.Date))
        .drop_nulls("timestamp")
        .group_by(["symbol", "timestamp"])
        .agg(
            pl.col("open").sort_by("bar_ts").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").sort_by("bar_ts").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
        .sort(["symbol", "timestamp"])
    )

    assert daily.filter(pl.col("timestamp").dt.weekday() > 5).height == 0, (
        "Weekend dates found in prices"
    )
    return daily


# %%
# Load 4H data and aggregate to daily
fx_4h = load_fx_pairs(
    frequency="4h",
    start_date=START_DATE,
    end_date=END_DATE,
).select(["symbol", "timestamp", "open", "high", "low", "close", "volume"])

prices = aggregate_4h_to_daily(fx_4h)

n_assets = prices["symbol"].n_unique()
n_dates = prices["timestamp"].n_unique()
print(f"Daily FX data: {n_assets} pairs, {n_dates} dates, {len(prices):,} rows")
print(f"Period: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
print(f"Pairs: {sorted(prices['symbol'].unique().to_list())}")

# %% [markdown]
# ## 2. Forward Return Labels
#
# We compute forward returns at three horizons:
# - **1-day** (primary): Matches daily decision cadence; tight 1-day purge
# - **5-day** (variant): Tests slower signal decay; academic FX literature uses weekly
# - **21-day** (variant): Matches concept note's reversal hypothesis horizon
#
# $r_{t \to t+h} = \frac{P_{t+h}}{P_t} - 1$
#
# where $P_t$ is the NY 5PM close.


# %%
def create_forward_returns(df: pl.DataFrame, horizon: int, label_name: str) -> pl.DataFrame:
    """Create forward return labels.

    Args:
        df: Daily DataFrame with asset, date, close
        horizon: Forward horizon in trading days
        label_name: Output column name
    """
    return df.with_columns(
        (pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1).alias(label_name)
    )


# %%
labels_df = prices.sort(["symbol", "timestamp"])

# Primary: 1-day forward return
labels_df = create_forward_returns(labels_df, horizon=1, label_name="fwd_ret_1d")
print("Created: fwd_ret_1d (primary, 1-day)")

# Variant: 5-day forward return
labels_df = create_forward_returns(labels_df, horizon=5, label_name="fwd_ret_5d")
print("Created: fwd_ret_5d (variant, 1-week)")

# Variant: 21-day forward return
labels_df = create_forward_returns(labels_df, horizon=21, label_name="fwd_ret_21d")
print("Created: fwd_ret_21d (variant, 1-month)")

# %% [markdown]
# ## 3. Label Distribution Summary

# %%
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION SUMMARY")
print("=" * 60)

label_stats = {}

for label_col, horizon_name in [
    ("fwd_ret_1d", "1-day"),
    ("fwd_ret_5d", "5-day"),
    ("fwd_ret_21d", "21-day"),
]:
    valid = labels_df.select(label_col).drop_nulls()
    n_valid = len(valid)
    mean_val = valid[label_col].mean()
    std_val = valid[label_col].std()
    pct_pos = (valid[label_col] > 0).mean()

    label_stats[label_col] = {
        "n_valid": n_valid,
        "mean": float(mean_val),
        "std": float(std_val),
        "pct_positive": float(pct_pos),
    }

    print(f"\n{horizon_name} ({label_col}):")
    print(f"  Valid samples: {n_valid:,}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std:  {std_val:.6f}")
    print(f"  % Positive: {pct_pos:.1%}")

# %% [markdown]
# **Class balance**: ~50% positive for all horizons is expected for FX forward
# returns (no systematic drift in exchange rates). This is not a class-imbalance
# problem -- the challenge is predicting the sign of small deviations from zero.

# %% [markdown]
# ### Label Autocorrelation
#
# Overlapping labels (5d, 21d) induce mechanical autocorrelation.
# We verify this and note it for downstream purge/embargo design.

# %%
from statsmodels.tsa.stattools import acf as acf_func

# Compute pooled label ACF for each horizon
print("\nLabel autocorrelation (pooled across pairs):")
for label_col, horizon_name, overlap in [
    ("fwd_ret_1d", "1-day", 0),
    ("fwd_ret_5d", "5-day", 4),
    ("fwd_ret_21d", "21-day", 20),
]:
    pooled = labels_df[label_col].drop_nulls().to_numpy()
    acf_vals = acf_func(pooled, nlags=5, alpha=None)
    lags_str = ", ".join(f"lag{i}={acf_vals[i]:.3f}" for i in range(1, 6))
    print(f"  {horizon_name}: {lags_str}  (expected overlap={overlap}d)")

# %%
# Effective sample size: N_eff = N / (1 + 2 * sum(ACF[1:h]))
n_eff_stats = {}
print("\nEffective sample size (overlapping label adjustment):")
for label_col_name, horizon_name, h in [
    ("fwd_ret_1d", "1-day", 1),
    ("fwd_ret_5d", "5-day", 5),
    ("fwd_ret_21d", "21-day", 21),
]:
    pooled = labels_df[label_col_name].drop_nulls().to_numpy()
    N = len(pooled)
    acf_vals = acf_func(pooled, nlags=h, alpha=None)
    denom = 1 + 2 * sum(acf_vals[1 : h + 1])
    n_eff = int(N / max(denom, 1.0))
    ratio = n_eff / N
    n_eff_stats[label_col_name] = {"N": N, "N_eff": n_eff, "ratio": round(ratio, 3)}
    print(f"  {horizon_name}: N={N:,}, N_eff={n_eff:,} ({ratio:.1%} effective)")

# %% [markdown]
# **Interpretation**: The 1-day label shows near-zero autocorrelation (no overlap),
# so $N_\text{eff} \approx N$. The 5-day and 21-day labels show significant
# autocorrelation from overlapping return windows, reducing effective sample sizes
# to ~21% and ~5% of raw counts respectively. The CV purge (1-day) is sufficient
# for the primary 1-day label; downstream models using 5d/21d labels should
# account for this overlap when interpreting statistical significance.

# %% [markdown]
# ## 4. Baseline Reversal IC
#
# Before any feature engineering we measure the cross-sectional rank IC of the
# raw 126-day reversal signal (past 126-day return) against each forward-return
# label. This sets the floor that engineered features must beat and tests the
# reversal hypothesis: a *negative* IC means past losers outperform.
#
# **Holdout seal.** This IC is a development-time signal-validation readout, so it
# is measured only on signal dates whose entire label window closes before
# `HOLDOUT_START` (2024-01-01). We seal on the **label endpoint**
# (`timestamp.shift(-h) < HOLDOUT_START`), not the signal date -- otherwise the
# last few pre-holdout signal dates of the 21-day label would read holdout prices.
# The 2024-2025 holdout stays sealed against every quantity used to motivate
# feature work.

# %%
# Compute 126-day lookback return as the reversal signal
reversal_df = labels_df.with_columns(
    (pl.col("close") / pl.col("close").shift(126).over("symbol") - 1).alias("ret_126d")
)

holdout_start = datetime.strptime(HOLDOUT_START, "%Y-%m-%d").date()

# Compute IC per date (cross-sectional rank correlation), sealed to the
# development window on the label endpoint so no forward label reads the holdout.
ic_results = {}

for label_col, horizon_name, horizon in [
    ("fwd_ret_1d", "1d", 1),
    ("fwd_ret_5d", "5d", 5),
    ("fwd_ret_21d", "21d", 21),
]:
    # Seal: keep only signal dates whose label window closes before the holdout.
    sealed = reversal_df.with_columns(
        pl.col("timestamp").shift(-horizon).over("symbol").alias("_label_end")
    ).filter(pl.col("_label_end") < holdout_start)

    # Group by date, compute Spearman rank correlation
    date_ics = []
    valid = sealed.drop_nulls(subset=["ret_126d", label_col])

    for date_val in valid["timestamp"].unique().sort().to_list():
        day_data = valid.filter(pl.col("timestamp") == date_val)
        if len(day_data) >= 10:  # Require at least 10 pairs for meaningful IC
            signal = day_data["ret_126d"].to_numpy()
            label = day_data[label_col].to_numpy()
            ic, _ = spearmanr(signal, label)
            if not np.isnan(ic):
                date_ics.append(ic)

    if date_ics:
        mean_ic = np.mean(date_ics)
        ic_std = np.std(date_ics)
        ic_tstat = mean_ic / (ic_std / np.sqrt(len(date_ics)))
        ic_results[horizon_name] = {
            "mean_ic": float(mean_ic),
            "ic_std": float(ic_std),
            "ic_tstat": float(ic_tstat),
            "n_dates": len(date_ics),
        }

print("\n" + "=" * 60)
print("BASELINE IC: 126-Day Reversal Signal")
print("=" * 60)
print(f"\n{'Horizon':<10} {'Mean IC':>10} {'IC Std':>10} {'t-stat':>10} {'N dates':>10}")
print("-" * 52)
for horizon_name, stats in ic_results.items():
    print(
        f"{horizon_name:<10} {stats['mean_ic']:>10.4f} {stats['ic_std']:>10.4f} "
        f"{stats['ic_tstat']:>10.2f} {stats['n_dates']:>10,}"
    )

# %% [markdown]
# **Interpretation**: The reversal IC is essentially zero at the 1-day horizon and
# turns increasingly negative as the horizon lengthens -- the 21-day label carries
# the largest-magnitude, most significant IC, so the reversal mechanism operates at
# monthly rather than daily frequency, consistent with the macro-driven hypothesis.
# All three horizons still sit **below the 0.05 absolute-IC floor** set by the
# feasibility kill condition KC1, so the raw signal alone is weak: feature
# engineering in `03_financial_features.py` must lift it. Note the naive per-date
# t-statistic assumes independent daily ICs; for the overlapping 5-day and 21-day
# labels it overstates significance (see the effective-sample-size analysis above),
# so treat those t-values as upper bounds.

# %% [markdown]
# ## 5. Save Labels and Materialize CV Splits

# %%
label_key_cols = ["timestamp", "symbol"]

LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Save individual label files
labels_df.select(label_key_cols + ["fwd_ret_1d"]).drop_nulls().write_parquet(
    LABELS_DIR / "fwd_ret_1d.parquet"
)
print("Saved labels/fwd_ret_1d.parquet")

labels_df.select(label_key_cols + ["fwd_ret_5d"]).drop_nulls().write_parquet(
    LABELS_DIR / "fwd_ret_5d.parquet"
)
print("Saved labels/fwd_ret_5d.parquet")

labels_df.select(label_key_cols + ["fwd_ret_21d"]).drop_nulls().write_parquet(
    LABELS_DIR / "fwd_ret_21d.parquet"
)
print("Saved labels/fwd_ret_21d.parquet")

# CV config from setup.yaml
eval_config = load_evaluation_config("fx_pairs")
assert eval_config["n_splits"] == 8
assert eval_config["train_size"] in ("P5Y", "5Y")
assert eval_config["val_size"] in ("P1Y", "1Y")

cv_splits = generate_cv_splits(prices, case_study_id="fx_pairs", label_buffer="1D")
print(f"Generated {len(cv_splits)} walk-forward splits")
# %% [markdown]
# ## 6. Results Collection


# %%
def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


results = {
    "case_study_id": "fx_pairs",
    "chapter": 7,
    "stage": "labels",
    "timestamp": datetime.now(UTC).isoformat(),
    "git_commit": _git_hash(),
    "notebook": "case_studies/fx_pairs/02_labels.py",
    "summary": {
        "n_pairs": int(n_assets),
        "n_dates": int(n_dates),
        "date_range": [str(prices["timestamp"].min()), str(prices["timestamp"].max())],
        "frequency": "daily_ny_5pm",
        "primary_label": "fwd_ret_1d",
        "variant_labels": ["fwd_ret_5d", "fwd_ret_21d"],
    },
    "techniques": {
        "aggregation": "4h_to_daily_ny_5pm_rollover",
        "label_type": "forward_returns",
        "horizons_days": [1, 5, 21],
        "cv_method": "walk_forward_rolling_5Y_train_1Y_test",
    },
    "diagnostics": {
        "label_stats": label_stats,
        "baseline_ic": ic_results,
        "n_eff": n_eff_stats,
    },
    "key_findings": [
        f"Aggregated 4H bars to daily using NY 5PM rollover convention ({n_dates} dates)",
        f"126-day reversal IC against 1d label: {ic_results.get('1d', {}).get('mean_ic', 0):.4f}",
        f"126-day reversal IC against 21d label: {ic_results.get('21d', {}).get('mean_ic', 0):.4f}",
        "CV config: 8 folds, 5Y rolling train, 1Y test (2016-2023), holdout 2024-2025",
    ],
}


# %% [markdown]
# ## Key Takeaways
#
# 1. **Daily aggregation**: 4H bars aggregated to daily using NY 5PM rollover --
#    this resolves the frequency mismatch between Ch6 setup and Ch8 features
# 2. **Three label horizons** capture different signal decay speeds: 1-day
#    (high frequency, noisy), 5-day (weekly, standard academic), 21-day
#    (monthly, matches reversal hypothesis)
# 3. **Baseline IC** confirms whether reversal effect (negative IC) is present
#    before any feature engineering
# 4. **CV configuration**: 8-fold walk-forward with 5Y rolling train
#    captures regime diversity (post-GFC, COVID, Fed hiking)
#
# **Next**: `03_financial_features.py` uses these daily prices and labels for feature
# engineering.
