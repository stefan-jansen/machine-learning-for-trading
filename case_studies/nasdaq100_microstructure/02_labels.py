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
# # NASDAQ-100 Microstructure: Label Engineering
#
# **Chapter 7: Defining the Learning Task**
#
# This notebook implements label engineering for the **NASDAQ-100 Microstructure**
# case study. We construct forward midprice returns at multiple horizons (1-bar,
# 5-bar, 60-bar) using execution-consistent conventions.
#
# **Learning Objectives**:
# - Compute midprice-based labels that avoid bid-ask bounce contamination
# - Apply session-bounded forward returns (no overnight gap leakage)
# - Perform cost-sanity checks to assess horizon feasibility
# - Generate walk-forward CV splits with appropriate purge/embargo
#
# **Label Types**:
# - `fwd_ret_15m`: Primary regression label (1-bar / 15-min forward mid return)
# - `fwd_ret_5m`: Variant 1 (5-min forward mid return, from 5-bar shift)
# - `fwd_ret_60m`: Variant 2 (60-min forward mid return, from 60-bar shift)
# - `fwd_dir_15m`: Classification variant (ternary: up/flat/down, 5 bps threshold)
#
# **Execution Convention**:
# - Decision time $t$ = bar close
# - Enter at $t+1$ midprice (1-bar execution delay)
# - Label: $y_H = \text{mid}_{t+H} / \text{mid}_{t+1} - 1$
#
# **Output Contract**:
# - `labels/fwd_ret_15m.parquet` — primary label
# - `labels/fwd_ret_5m.parquet` — fast variant
# - `labels/fwd_ret_60m.parquet` — slow variant
# - `labels/fwd_dir_15m.parquet` — classification variant
# - `cv_config.json` — walk-forward CV configuration
#
# **Cross-References**:
# - **Upstream**: Ch3 (AlgoSeek TAQ data), Ch6 ([`01_feasibility_analysis`](01_feasibility_analysis.ipynb))
# - **Downstream**: Ch8 (`03_financial_features.py`), Ch9 (`04_temporal.py`)

# %%
"""NASDAQ-100 Microstructure: Label Engineering (Ch7)."""

import warnings
from datetime import UTC, date, datetime

import numpy as np
import polars as pl

from data import load_nasdaq100_bars
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
START_DATE = "2020-01-01"
END_DATE = "2021-12-31"
MAX_SYMBOLS = 0

# %%
# Configuration
CASE_DIR = get_case_study_dir("nasdaq100_microstructure")
LABELS_DIR = CASE_DIR / "labels"

print(f"Date range: {START_DATE} to {END_DATE}")

# %% [markdown]
# ## 1. Load Minute Bar Data
#
# Load AlgoSeek minute bars with microstructure fields. We need NBBO quotes
# (CloseBidPrice, CloseAskPrice) to compute midprice-based labels that avoid
# bid-ask bounce per Hasbrouck (2007).
#
# **Memory requirement**: the full universe (114 symbols, 2020–2021) is ~40M
# minute bars across 60+ microstructure columns and materializes to roughly
# **28 GB** in memory — more than most laptops have. The raw load is about
# **0.25 GB per symbol**, so:
#
# | Symbols (`MAX_SYMBOLS`) | Approx. loaded size |
# |---|---|
# | 10 | ~2.5 GB |
# | 25 | ~6 GB |
# | 50 | ~13 GB |
# | 114 (all, `0`) | ~28 GB |
#
# Peak usage runs higher than these figures because label computation holds
# additional intermediates, so budget headroom. To run on a memory-constrained
# machine, set `MAX_SYMBOLS` to a smaller value in the parameters cell (a
# seed-deterministic subset), and/or narrow `START_DATE`/`END_DATE`. The
# microstructure patterns this case study teaches are visible on any subset;
# only the published full-universe results require the complete load.

# %%
df = load_nasdaq100_bars(
    start_date=START_DATE,
    end_date=END_DATE,
    include_microstructure=True,
    max_symbols=MAX_SYMBOLS,
)

print(f"Loaded {len(df):,} minute bars")
print(f"Symbols: {df['symbol'].n_unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown]
# ## 2. Filter to Regular Trading Hours
#
# Restrict to 09:30--16:00 ET. Pre-market and after-hours bars have
# wider spreads, lower liquidity, and different microstructure dynamics.

# %%
df = df.filter(
    (pl.col("timestamp").dt.hour() >= 10)  # >= 10:00 captures 09:30 due to bar labeling
    | ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() >= 30))
)
df = df.filter(pl.col("timestamp").dt.hour() < 16)

# Sort for time-series operations
df = df.sort(["symbol", "timestamp"])

# Add session_date for session-bounded operations
df = df.with_columns(pl.col("timestamp").dt.date().alias("session_date"))

print(f"Regular hours bars: {len(df):,}")
print(f"Sessions: {df['session_date'].n_unique()}")

# %% [markdown]
# ## 3. Compute Midprice
#
# Midprice = (CloseBidPrice + CloseAskPrice) / 2 reduces bid-ask bounce that
# contaminates trade-price-based returns. This is standard practice for
# intraday label construction (Hasbrouck, 2007).

# %%
has_bid_ask = "close_bid_price" in df.columns and "close_ask_price" in df.columns

if has_bid_ask:
    df = df.with_columns(
        mid_close=((pl.col("close_bid_price") + pl.col("close_ask_price")) / 2),
        half_spread=(
            (pl.col("close_ask_price") - pl.col("close_bid_price"))
            / (pl.col("close_bid_price") + pl.col("close_ask_price") + 1e-8)
        ),
    )
    print("Using midprice from NBBO quotes")
else:
    df = df.with_columns(
        mid_close=pl.col("last_trade_price"),
        half_spread=pl.lit(0.0005),  # Default 5 bps if no quotes
    )
    print("WARNING: No NBBO available, using last trade price")

# Filter bars with valid midprice (positive, non-null)
df = df.filter(pl.col("mid_close").is_not_null() & (pl.col("mid_close") > 0))

print(f"Bars with valid midprice: {len(df):,}")

# %% [markdown]
# ## 4. Create Labels
#
# Forward midprice returns at three horizons. The shift(-1) in the denominator
# enforces the 1-bar execution delay: you observe at bar $t$ close, enter at
# $t+1$ midprice, and the label measures the return from entry to the target bar.
#
# **Implementation note**: Labels are computed at *minute* bar resolution using
# `shift(-15)` rather than resampling to 15-min bars and using `shift(-1)`.
# Both produce the same 15-minute forward midprice return. The minute-bar
# approach avoids information loss from resampling and aligns the feature
# matrix (also at minute resolution) with labels for exact row-level joins.
# The setup notebook's "1-bar at 15-min frequency" specification is equivalent.
#
# **Session-bounded**: `.over(["symbol", "session_date"])` ensures shifts do not
# cross overnight boundaries. This prevents overnight gap contamination in
# intraday labels.
#
# **Horizons** (in minute bars):
# - 5 bars = 5 minutes (fast, most cost-dominated)
# - 15 bars = 15 minutes (primary, balances signal vs cost)
# - 60 bars = 60 minutes (slow, tests residual microstructure content)


# %%
def create_intraday_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Create forward midprice return labels at multiple horizons.

    Each label computes: mid[t+H] / mid[t+1] - 1
    where t+1 is the entry bar (execution delay) and t+H is the exit bar.

    Session-bounded: shifts do not cross overnight boundaries.
    """
    group_cols = ["symbol", "session_date"]

    df = df.with_columns(
        # Entry midprice (1-bar execution delay)
        entry_mid=pl.col("mid_close").shift(-1).over(group_cols),
        # Exit midprices at each horizon
        exit_5m=pl.col("mid_close").shift(-5).over(group_cols),
        exit_15m=pl.col("mid_close").shift(-15).over(group_cols),
        exit_60m=pl.col("mid_close").shift(-60).over(group_cols),
    )

    # Compute returns from entry to exit
    df = df.with_columns(
        fwd_ret_5m=(pl.col("exit_5m") / pl.col("entry_mid") - 1),
        fwd_ret_15m=(pl.col("exit_15m") / pl.col("entry_mid") - 1),
        fwd_ret_60m=(pl.col("exit_60m") / pl.col("entry_mid") - 1),
    )

    # Ternary classification: up / flat / down with cost-motivated threshold
    # 5 bps threshold: moves below this are within the friction floor
    flat_threshold = 0.0005  # 5 bps

    df = df.with_columns(
        fwd_dir_15m=pl.when(pl.col("fwd_ret_15m").is_null())
        .then(None)
        .when(pl.col("fwd_ret_15m") > flat_threshold)
        .then(1)
        .when(pl.col("fwd_ret_15m") < -flat_threshold)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8),
    )

    # Drop intermediate columns
    df = df.drop(["entry_mid", "exit_5m", "exit_15m", "exit_60m"])

    return df


# %%
total_before_labels = len(df)
df = create_intraday_labels(df)
print("Labels created successfully")

# Session boundary data loss reporting
print("\nSession boundary data loss:")
for label_col, shift_bars in [("fwd_ret_5m", 5), ("fwd_ret_15m", 15), ("fwd_ret_60m", 60)]:
    valid = df[label_col].drop_nulls().len()
    lost = total_before_labels - valid
    pct_lost = 100 * lost / total_before_labels
    print(
        f"  {label_col}: {lost:,} rows lost ({pct_lost:.1f}%) — last {shift_bars} bars/session + 1-bar entry"
    )

# %% [markdown]
# ## 5. Label Quality Assessment
#
# Evaluate label distributions, class balance, and cost-feasibility.

# %%
# Return statistics by horizon
label_stats = {}
for label_col, horizon_name in [
    ("fwd_ret_5m", "5m"),
    ("fwd_ret_15m", "15m"),
    ("fwd_ret_60m", "60m"),
]:
    vals = df[label_col].drop_nulls()
    label_stats[horizon_name] = {
        "count": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(vals.median()),
        "p5": float(vals.quantile(0.05)),
        "p95": float(vals.quantile(0.95)),
    }

print("=== Forward Return Statistics ===")
for horizon, stats in label_stats.items():
    print(
        f"  fwd_ret_{horizon}: n={stats['count']:,}, "
        f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"[p5={stats['p5']:.5f}, p95={stats['p95']:.5f}]"
    )

# %%
# Classification label distribution
dir_dist = df.group_by("fwd_dir_15m").len().sort("fwd_dir_15m")
print("\nDirection label distribution (15m, threshold=5bps):")
print(dir_dist)

# Compute class balance
total = dir_dist["len"].sum()
for row in dir_dist.iter_rows(named=True):
    pct = 100 * row["len"] / total
    label_name = {-1: "Down", 0: "Flat", 1: "Up"}.get(row["fwd_dir_15m"], "?")
    print(f"  {label_name}: {row['len']:,} ({pct:.1f}%)")

# %% [markdown]
# ### Cost-Sanity Check
#
# Compare the typical return magnitude to the half-spread cost. If the median
# absolute return is close to or less than the half-spread, the horizon is
# cost-dominated and unlikely to be tradeable.

# %%
# Cost sanity: compare |return| distribution to half-spread
cost_check = {}
for label_col, horizon_name in [
    ("fwd_ret_5m", "5m"),
    ("fwd_ret_15m", "15m"),
    ("fwd_ret_60m", "60m"),
]:
    joined = df.select([label_col, "half_spread"]).drop_nulls()
    abs_ret = joined[label_col].abs()
    hs = joined["half_spread"]

    median_abs_ret_bps = float(abs_ret.median()) * 10000
    median_half_spread_bps = float(hs.median()) * 10000
    pct_exceeds_cost = float((abs_ret > hs).mean()) * 100

    cost_check[horizon_name] = {
        "median_abs_ret_bps": round(median_abs_ret_bps, 2),
        "median_half_spread_bps": round(median_half_spread_bps, 2),
        "pct_exceeds_cost": round(pct_exceeds_cost, 1),
    }

print("\n=== Cost-Sanity Check ===")
pl.DataFrame(
    [
        {
            "horizon": h,
            "median_abs_ret_bps": c["median_abs_ret_bps"],
            "median_half_spread_bps": c["median_half_spread_bps"],
            "pct_exceeds_cost": c["pct_exceeds_cost"],
        }
        for h, c in cost_check.items()
    ]
)

# %% [markdown]
# **Interpretation**: If `% > cost` is below 50%, most trades would lose money
# to the spread alone. This confirms the cost-dominant regime identified in
# [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) and supports the pedagogical framing: microstructure signals
# exist but are not tradeable at institutional horizons.

# %% [markdown]
# ### Baseline IC: Lagged Return vs Forward Return
#
# The simplest predictor of the next 15-minute return is the current 15-minute
# return (testing for mean-reversion or momentum). This baseline IC provides
# the reference that Ch8 features must beat. Measured on train+validation rows
# only (timestamp < holdout_start) so this expectation-setting diagnostic never
# reads the sealed holdout.

# %%
import yaml

_holdout_start = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())["evaluation"][
    "holdout_start"
]

# Compute lagged 15-minute return as baseline signal, pre-holdout rows only
df_baseline = (
    df.with_columns(
        lag_15m=(
            pl.col("mid_close") / pl.col("mid_close").shift(15).over(["symbol", "session_date"]) - 1
        )
    )
    .filter(pl.col("timestamp").dt.date() < date.fromisoformat(_holdout_start))
    .drop_nulls(subset=["lag_15m", "fwd_ret_15m"])
)

# Sample every 15th timestamp for approximately independent observations
all_ts = df_baseline["timestamp"].unique().sort()
sample_ts = all_ts.gather_every(15)
baseline_sample = df_baseline.filter(pl.col("timestamp").is_in(sample_ts))

min_cs_size = min(10, baseline_sample["symbol"].n_unique())
baseline_ic = (
    baseline_sample.group_by("timestamp")
    .agg(
        pl.corr("lag_15m", "fwd_ret_15m", method="spearman").alias("ic"),
        pl.len().alias("n"),
    )
    .filter(pl.col("n") >= min_cs_size)
)

if baseline_ic.height > 0 and baseline_ic["ic"].null_count() < baseline_ic.height:
    baseline_ic_mean = float(baseline_ic["ic"].mean())
    baseline_ic_std = float(baseline_ic["ic"].std())
    baseline_ic_t = (
        baseline_ic_mean / (baseline_ic_std / np.sqrt(len(baseline_ic)))
        if baseline_ic_std > 0
        else 0.0
    )
else:
    baseline_ic_mean, baseline_ic_std, baseline_ic_t = 0.0, 0.0, 0.0
    print("  Warning: insufficient cross-sectional data for baseline IC")

print("Baseline IC: Lagged 15-min Return → fwd_ret_15m")
print(f"  Mean IC: {baseline_ic_mean:.5f}")
print(f"  IC t-stat: {baseline_ic_t:.2f}")
print(f"  IC observations: {len(baseline_ic):,}")
if abs(baseline_ic_t) < 2.0:
    print("  → NOT significant. Ch8 features must create signal from scratch.")
else:
    sign = "mean-reverting" if baseline_ic_mean < 0 else "momentum"
    print(f"  → Significant ({sign}). Ch8 features should improve on this.")

# %% [markdown]
# **Reconciliation with Ch6 baseline**: The feasibility notebook ([`01_feasibility_analysis`](01_feasibility_analysis.ipynb)) reports a
# much stronger baseline IC (~-0.17) because it uses close-to-close (trade price)
# returns, which suffer from bid-ask bounce — an artifact that creates artificial
# negative autocorrelation (Hasbrouck, 2007). The near-zero IC here, computed on
# midprice returns, is the uncontaminated measure and is authoritative for downstream
# feature evaluation.

# %% [markdown]
# ### Label Autocorrelation
#
# Autocorrelation of `fwd_ret_15m` at different lags reveals how quickly predictive
# content decays and informs purge/embargo settings.

# %%
# Label autocorrelation at 1-min, 15-min, and 60-min lags
autocorr_lags = {"1_bar": 1, "15_bar": 15, "60_bar": 60}
autocorr_results = {}

for lag_name, lag_bars in autocorr_lags.items():
    lag_col = f"fwd_ret_15m_lag_{lag_bars}"
    valid = df.with_columns(
        pl.col("fwd_ret_15m").shift(lag_bars).over(["symbol", "session_date"]).alias(lag_col)
    ).drop_nulls(subset=["fwd_ret_15m", lag_col])
    rho = float(valid.select(pl.corr("fwd_ret_15m", lag_col)).item())
    autocorr_results[lag_name] = round(rho, 5)
    print(f"  fwd_ret_15m autocorrelation at lag {lag_bars}: {rho:.5f}")

# %% [markdown]
# ## 6. Save Labels and CV Configuration

# %%
# Drop incomplete rows (NaN labels from session boundaries)
df_clean = df.drop_nulls(subset=["fwd_ret_15m"])
print(f"Rows after dropping incomplete labels: {len(df_clean):,}")

# %%
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Save individual label files
key_cols = ["timestamp", "symbol"]

for label_col in ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "fwd_dir_15m"]:
    subset = df_clean.select(key_cols + [label_col]).drop_nulls(subset=[label_col])
    subset.write_parquet(LABELS_DIR / f"{label_col}.parquet")
    print(f"Saved labels/{label_col}.parquet ({len(subset):,} rows)")

# %%
# CV splits (generated at runtime from setup.yaml, not saved as file)
splits = generate_cv_splits(
    df_clean,
    case_study_id="nasdaq100_microstructure",
    label_buffer="15min",
    date_col="timestamp",
)
print(f"CV config: {len(splits)} folds from setup.yaml evaluation section")

# %%
# Results JSON for chapter agent
results = {
    "case_study_id": "nasdaq100_microstructure",
    "chapter": 7,
    "stage": "labels",
    "timestamp": datetime.now(UTC).isoformat(),
    "git_commit": "unknown",
    "notebook": "case_studies/nasdaq100_microstructure/code/02_labels.py",
    "summary": {
        "n_rows": len(df_clean),
        "n_symbols": df_clean["symbol"].n_unique(),
        "n_sessions": df_clean["session_date"].n_unique(),
        "labels": ["fwd_ret_5m", "fwd_ret_15m", "fwd_ret_60m", "fwd_dir_15m"],
        "primary_label": "fwd_ret_15m",
    },
    "techniques": [
        "midprice-based forward returns",
        "session-bounded computation",
        "1-bar execution delay",
        "cost-sanity check",
    ],
    "diagnostics": {
        "label_stats": label_stats,
        "cost_check": cost_check,
        "baseline_ic": {
            "signal": "lagged_15m_return",
            "mean_ic": round(baseline_ic_mean, 5),
            "ic_t_stat": round(baseline_ic_t, 2),
            "n_observations": len(baseline_ic),
        },
        "label_autocorrelation": autocorr_results,
    },
    "key_findings": [
        f"Primary label fwd_ret_15m: mean={label_stats['15m']['mean']:.6f}, std={label_stats['15m']['std']:.6f}",
        f"Cost dominance confirmed: {cost_check['15m']['pct_exceeds_cost']:.0f}% of 15m returns exceed half-spread",
        f"Baseline IC (lagged 15m return): {baseline_ic_mean:.5f} (t={baseline_ic_t:.2f})",
        "Direction label (5bps threshold): ternary up/flat/down with 5 bps cost-motivated threshold",
    ],
}
# %% [markdown]
# ## Key Takeaways
#
# 1. **Midprice labels** avoid bid-ask bounce contamination that would add noise
#    to trade-price returns — essential for intraday microstructure research
# 2. **Session-bounded** forward returns prevent overnight gap artifacts
# 3. **1-bar execution delay** reflects realistic entry timing: you observe at
#    bar close, enter at next bar's midprice
# 4. **Cost-sanity check** quantifies what fraction of returns exceed the
#    half-spread — the primary feasibility gate for intraday strategies
# 5. **Three horizons** (5m, 15m, 60m) demonstrate signal decay: microstructure
#    content is highest at short horizons but so are costs
# 6. **Baseline IC** from lagged returns establishes the reference bar that
#    Ch8 engineered features must exceed
# 7. **Label autocorrelation** reveals how quickly predictive content decays,
#    informing purge/embargo settings for walk-forward CV
#
# **Next**: `03_financial_features.py` engineers microstructure features from the raw
# minute bar data (order flow, liquidity, volatility, Kyle's lambda).
