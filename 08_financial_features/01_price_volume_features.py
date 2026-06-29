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
# # Price and Volume Feature Families
#
# **Chapter 8: Feature Engineering**
# **Section Reference**: 8.2 - Price-Derived Features
# **Docker image**: `ml4t`
#
# ## Purpose
#
# This notebook demonstrates the **core feature families** derived from a single
# asset's price and volume history. These are the workhorse features of most
# quantitative strategies — available for every tradeable instrument.
#
# ## Learning Objectives
#
# 1. Understand the core feature families for price/volume data
# 2. Implement features with explicit time-safety (sorting, window alignment)
# 3. Choose appropriate normalization for cross-sectional vs time-series use
# 4. Compare volatility estimators (close-to-close, Parkinson, Garman-Klass, Yang-Zhang)
# 5. Build volatility state features (vol ratio, percentile, decile)
# 6. Avoid common leakage patterns in feature construction
#
# ## Feature Families Covered
#
# | Family | Representative Features | Key Use Case |
# |--------|------------------------|--------------|
# | **Returns & Horizons** | Simple, log, skip-1, cumulative | Base signals |
# | **Trend & Reversal** | MA distance, regression slope, dist-to-MA | Momentum/reversion |
# | **Volatility** | CC, Parkinson, GK, YZ, vol-of-vol | Risk scaling |
# | **Volatility State** | Vol ratio, percentile, decile | Regime conditioning |
# | **Volume & Liquidity** | Dollar volume, relative volume, VWAP | Capacity signals |
# | **Risk** | VaR, CVaR, tail ratio | Position sizing |
# | **Cross-Sectional** | Ranks, z-scores | Universe normalization |
#
# ## Data Policy
#
# All examples use **real ETF data** (no synthetic data).

# %%
"""Price and Volume Feature Families — core feature families derived from a single asset's price and volume history."""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from utils.paths import get_chapter_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
START_DATE = "2015-01-01"

# %% [markdown]
# ## 0. Feature Discovery with ml4t-engineer
#
# Before building features manually, let's see what the `ml4t-engineer` library
# offers. The registry provides 120+ pre-built, validated features that the
# case study notebooks use throughout Chapters 8-12.
#
# For a full tour of the library ecosystem (data loaders, feature computation,
# evaluation tools), see `10_ml4t_library_ecosystem` in Chapter 7.

# %%
from ml4t.engineer import compute_features
from ml4t.engineer.core.registry import get_registry

registry = get_registry()
all_features = registry.list_all()

# Features by category
categories = {}
for name in all_features:
    metadata = registry.get(name)
    categories.setdefault(metadata.category, []).append(name)

print(f"Total features available: {len(all_features)}\n")
print("Features by category:")
for cat, feats in sorted(categories.items()):
    examples = ", ".join(feats[:4])
    suffix = ", ..." if len(feats) > 4 else ""
    print(f"  {cat:20s}: {len(feats):3d}  ({examples}{suffix})")

# %% [markdown]
# ### Feature metadata
#
# Each registry entry carries self-documenting metadata: formula, parameters,
# input type, and description. This makes features discoverable without reading
# source code.

# %%
rsi_meta = registry.get("rsi")
print(f"Feature:     {rsi_meta.name}")
print(f"Category:    {rsi_meta.category}")
print(f"Formula:     {rsi_meta.formula}")
print(f"Parameters:  {rsi_meta.parameters}")
print(f"Input type:  {rsi_meta.input_type}")

# %% [markdown]
# ### Quick computation
#
# `compute_features()` accepts a list of feature names (default parameters)
# or dicts (custom parameters). We'll use it throughout the case study
# notebooks; the sections below show the manual implementations for teaching.

# %%
from data import load_etfs

spy_quick = load_etfs().filter(pl.col("symbol") == "SPY").sort("timestamp").tail(500)

result = compute_features(spy_quick, ["rsi", "atr", "sma"])
new_cols = [c for c in result.columns if c not in spy_quick.columns]
print(f"Computed {len(new_cols)} feature columns: {new_cols}")
result.select(["timestamp", "close"] + new_cols).tail(5)

# %% [markdown]
# The notebooks that follow build these features manually to explain the
# economics and implementation details, then use the registry for production
# pipelines.

# %% [markdown]
# ## 1. Data Loading and Sorting
#
# **Critical**: All rolling/window operations require chronological ordering.
# We establish sorting once at data load, not implicitly per operation.

# %%
# Load and sort by symbol, then timestamp (CRITICAL for .over() operations)
etfs = load_etfs().sort(["symbol", "timestamp"])

# Filter date range
etfs = etfs.filter(pl.col("timestamp") >= datetime.strptime(START_DATE, "%Y-%m-%d"))

# For single-asset demos: SPY
spy = etfs.filter(pl.col("symbol") == "SPY").sort("timestamp")

# For cross-sectional demos: subset of liquid ETFs
cs_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLE", "XLK"]
cs_etfs = etfs.filter(pl.col("symbol").is_in(cs_symbols)).sort(["symbol", "timestamp"])

print(f"SPY: {len(spy):,} rows")
print(f"Cross-sectional universe: {len(cs_etfs):,} rows, {cs_etfs['symbol'].n_unique()} symbols")
print(f"Date range: {spy['timestamp'].min()} to {spy['timestamp'].max()}")

# %% [markdown]
# ## 2. Returns and Horizons
#
# Returns are the foundation of all momentum features. Key variants:
#
# | Variant | Formula | Use Case |
# |---------|---------|----------|
# | Simple return | $(P_t - P_{t-h}) / P_{t-h}$ | Standard momentum |
# | Log return | $\ln(P_t / P_{t-h})$ | Additive across time |
# | Skip-1 momentum | $r_{t-1:t-h}$ | Avoid microstructure noise |
# | Cumulative | $\sum_{i=0}^{h} r_{t-i}$ | Multi-period signals |

# %% [markdown]
# ### 2.1 Manual Implementation (Teaching)
#
# Understanding the mechanics of return computation.


# %%
def compute_returns_manual(df: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """
    Compute returns for multiple horizons using pure Polars.

    Note: This is for teaching. Use ml4t-engineer in production.
    """
    return_exprs = []

    for h in horizons:
        # Simple returns
        return_exprs.append(pl.col("close").pct_change(h).alias(f"ret_{h}d"))
        # Log returns
        return_exprs.append(
            (pl.col("close").log() - pl.col("close").log().shift(h)).alias(f"logret_{h}d")
        )

    return df.with_columns(return_exprs)


# Single with_columns for efficiency
returns_df = compute_returns_manual(spy, horizons=[1, 5, 21])

print("Return features computed:")
returns_df.select(["timestamp", "close", "ret_1d", "ret_5d", "ret_21d"]).tail(10)

# %% [markdown]
# ### 2.2 Skip-1 Momentum
#
# Skip the most recent day to avoid microstructure reversals (bid-ask bounce).
#
# $$\text{Skip-1 Momentum}_{21d} = \frac{P_{t-1}}{P_{t-21}} - 1$$

# %%
# Skip-1 momentum: shift(1) before computing the horizon return
skip1_df = spy.with_columns(
    [
        # Standard 21-day momentum
        pl.col("close").pct_change(21).alias("mom_21d"),
        # Skip-1: use yesterday's close as numerator
        ((pl.col("close").shift(1) / pl.col("close").shift(21)) - 1).alias("mom_21d_skip1"),
    ]
)

# Show correlation - should be high but not identical
correlation = skip1_df.drop_nulls().select(
    [pl.corr("mom_21d", "mom_21d_skip1").alias("correlation")]
)
print(f"Correlation between standard and skip-1 momentum: {correlation[0, 0]:.4f}")

# %% [markdown]
# **Interpretation**: A correlation of ~0.97 confirms the two series carry
# very similar information at the 21-day horizon. The skip-1 variant removes
# the last day's microstructure noise (bid-ask bounce), so it is preferred for
# daily-rebalanced strategies where the most recent close is noisiest.

# %% [markdown]
# ### 2.3 Session-Based Returns
#
# Decomposing returns into overnight (gap) and intraday components:
#
# - **Overnight return**: $\frac{Open_t}{Close_{t-1}} - 1$
# - **Intraday return**: $\frac{Close_t}{Open_t} - 1$

# %%
session_df = spy.with_columns(
    [
        # Overnight (gap)
        ((pl.col("open") / pl.col("close").shift(1)) - 1).alias("overnight_ret"),
        # Intraday
        ((pl.col("close") / pl.col("open")) - 1).alias("intraday_ret"),
        # Total for reference
        pl.col("close").pct_change().alias("total_ret"),
    ]
)

# Verify decomposition: (1 + overnight) * (1 + intraday) ≈ (1 + total)
session_df = session_df.with_columns(
    ((1 + pl.col("overnight_ret")) * (1 + pl.col("intraday_ret")) - 1).alias("reconstructed")
)

print("Session return decomposition:")
session_df.select(
    ["timestamp", "overnight_ret", "intraday_ret", "total_ret", "reconstructed"]
).tail(10)

# %% [markdown]
# ## 3. Trend and Reversal Features
#
# These features capture where price is relative to historical patterns.
#
# | Feature | Signal Type | Interpretation |
# |---------|-------------|----------------|
# | MA Distance | Trend | >0 bullish, <0 bearish |
# | Regression Slope | Trend strength | Steeper = stronger trend |
# | Short-term Reversal | Mean reversion | Oversold/overbought |

# %% [markdown]
# ### 3.1 MA Distance (Volatility-Scaled)
#
# Raw MA distance varies with price level. Scaling by volatility creates
# a standardized signal comparable across assets and time.
#
# $$\text{MA Distance} = \frac{P_t - SMA_{21}}{ATR_{21}}$$

# %%
from ml4t.engineer.features.volatility import atr

# MA distance scaled by ATR
ma_df = spy.with_columns(
    [
        pl.col("close").rolling_mean(21).alias("sma_21"),
        atr("high", "low", "close", period=21).alias("atr_21"),
    ]
).with_columns(
    [
        # Raw distance (not comparable across assets)
        (pl.col("close") - pl.col("sma_21")).alias("ma_dist_raw"),
        # Vol-scaled (standardized)
        ((pl.col("close") - pl.col("sma_21")) / pl.col("atr_21")).alias("ma_dist_scaled"),
    ]
)

# %%
# Visualize MA distance
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["SPY Close with SMA(21)", "Raw MA Distance", "Vol-Scaled MA Distance"],
    vertical_spacing=0.08,
)

n = 252  # Last year
fig.add_trace(
    go.Scatter(x=ma_df["timestamp"].to_list()[-n:], y=ma_df["close"].to_list()[-n:], name="Close"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ma_df["timestamp"].to_list()[-n:],
        y=ma_df["sma_21"].to_list()[-n:],
        name="SMA(21)",
        line=dict(dash="dash"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ma_df["timestamp"].to_list()[-n:],
        y=ma_df["ma_dist_raw"].to_list()[-n:],
        name="Raw",
        fill="tozeroy",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ma_df["timestamp"].to_list()[-n:],
        y=ma_df["ma_dist_scaled"].to_list()[-n:],
        name="Vol-Scaled",
        fill="tozeroy",
    ),
    row=3,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

fig.update_layout(height=600, title="MA Distance: Raw vs Vol-Scaled")
fig.show()


# %% [markdown]
# ### 3.2 Rolling Regression Slope
#
# OLS slope over a rolling window measures trend strength more robustly than
# endpoint-to-endpoint returns: it uses all intermediate prices and is less
# sensitive to start/end outliers.
#
# For evenly spaced data with $x = [0, 1, \ldots, n-1]$:
#
# $$\hat\beta = \frac{\text{Cov}(x, P)}{\text{Var}(x)}$$
#
# We normalize by mean price to get a percentage slope comparable across assets.


# %%
def rolling_regression_slope(
    df: pl.DataFrame,
    price_col: str = "close",
    period: int = 21,
) -> pl.DataFrame:
    """Rolling OLS slope, normalized by mean price."""
    return df.with_columns(
        pl.col(price_col)
        .rolling_map(
            lambda s: np.polyfit(np.arange(len(s)), s.to_numpy(), 1)[0] / s.mean() * len(s),
            window_size=period,
            min_periods=period,
        )
        .alias(f"slope_{period}d")
    )


slope_df = rolling_regression_slope(spy, period=21)
print("Rolling regression slope (normalized %-change over 21d):")
slope_df.select(["timestamp", "close", "slope_21d"]).tail(10)

# %% [markdown]
# ### 3.3 Short-Term Reversal
#
# Recent underperformers tend to outperform in the short term (1-5 days).
# This is the complement to momentum.
#
# $$\text{Reversal} = -r_{1d}$$

# %%
reversal_df = spy.with_columns(
    [
        pl.col("close").pct_change(1).alias("ret_1d"),
        (-pl.col("close").pct_change(1)).alias("reversal_1d"),
        # Z-scored reversal for comparability
        (
            (-pl.col("close").pct_change(1) - (-pl.col("close").pct_change(1)).rolling_mean(63))
            / (-pl.col("close").pct_change(1)).rolling_std(63)
        ).alias("reversal_zscore"),
    ]
)

print("Short-term reversal features:")
reversal_df.select(["timestamp", "close", "ret_1d", "reversal_1d"]).tail(10)

# %% [markdown]
# ### 3.4 Directional Persistence: Distance to MA in ATR Units
#
# This extends the vol-scaled MA distance from §3.1 with different parameters
# and interpretation. Where §3.1 uses SMA(21)/ATR(21) to measure *current*
# deviation, here MA(50)/ATR(14) captures *directional persistence* — how
# far price has trended away from a slower anchor.
#
# $$\text{dist\_to\_ma\_atr} = \frac{P_t - MA_{50}}{ATR_{14}}$$
#
# ATR is used here for normalization (price-denominated), not as a volatility
# estimator — see §4 for the distinction.

# %%
dist_ma_df = spy.with_columns(
    [
        pl.col("close").rolling_mean(50).alias("ma_50"),
        atr("high", "low", "close", period=14).alias("atr_14"),
    ]
).with_columns(
    ((pl.col("close") - pl.col("ma_50")) / pl.col("atr_14").clip(1e-10, None)).alias(
        "dist_to_ma_atr"
    ),
)

print("Distance to MA in ATR units:")
dist_ma_df.select(["timestamp", "close", "ma_50", "atr_14", "dist_to_ma_atr"]).tail(10)

# %% [markdown]
# Values beyond $\pm 3$ indicate extreme directional extension. This feature
# works well as both a signal (contrarian at extremes) and a state variable
# (conditioning faster signals on trend context).

# %% [markdown]
# ## 4. Volatility Features
#
# Volatility is essential for:
# - **Risk scaling**: Adjust signals by volatility
# - **Position sizing**: Kelly criterion, risk parity
# - **Regime detection**: High vs low vol environments
#
# ### Estimator Efficiency Comparison
#
# | Estimator | Efficiency | Best For |
# |-----------|------------|----------|
# | Close-to-Close | 1x | Baseline |
# | Parkinson (H-L) | ~5x | High-low only |
# | Garman-Klass | ~7× | Full OHLC |
# | Yang-Zhang | ~8–14× | Best overall |

# %% [markdown]
# ### 4.1 Realized Volatility (Close-to-Close)
#
# $$\sigma_{CC} = \sqrt{252} \times \text{std}(r_t)$$

# %%
from ml4t.engineer.features.volatility import realized_volatility, yang_zhang_volatility

vol_df = spy.with_columns(
    pl.col("close").pct_change().alias("ret"),
).with_columns(
    [
        # Close-to-close (annualized) — note: realized_volatility expects returns, not prices
        realized_volatility("ret", period=21, annualize=True).alias("vol_cc_21"),
        # Yang-Zhang (most efficient)
        yang_zhang_volatility("open", "high", "low", "close", period=21, annualize=True).alias(
            "vol_yz_21"
        ),
    ]
)

print("Volatility comparison:")
vol_df.select(["timestamp", "close", "vol_cc_21", "vol_yz_21"]).tail(10)

# %% [markdown]
# ### 4.2 Range-Based Estimators
#
# Range-based estimators use OHLC data for much higher efficiency than
# close-to-close. Key formulas:
#
# **Parkinson (1980)** — uses high-low range only:
#
# $$\hat{\sigma}^2_P = \frac{1}{4 \ln 2} (\ln H_t - \ln L_t)^2$$
#
# **Garman-Klass (1980)** — adds open-close information:
#
# $$\hat{\sigma}^2_{GK} = 0.5 (\ln H_t - \ln L_t)^2 - (2\ln 2 - 1)(\ln C_t - \ln O_t)^2$$
#
# | Estimator | Efficiency vs CC | Data Required |
# |-----------|-----------------|---------------|
# | Close-to-Close | 1x | Close |
# | Parkinson | ~5x | High, Low |
# | Garman-Klass | ~7× | Open, High, Low, Close |
# | Yang-Zhang | ~8–14× | Open, High, Low, Close |
#
# **Note**: ATR (Average True Range) measures price *range* in dollar terms,
# not *volatility* in return terms. ATR is useful for stop placement and
# normalization but is not a volatility estimator.


# %%
# Manual implementations for teaching
def parkinson_vol(period: int = 21) -> pl.Expr:
    """Parkinson range-based volatility (annualized)."""
    log_hl_sq = ((pl.col("high").log() - pl.col("low").log()) ** 2) / (4 * np.log(2))
    return (log_hl_sq.rolling_mean(period) * 252).sqrt()


# %%
def garman_klass_vol(period: int = 21) -> pl.Expr:
    """Garman-Klass OHLC volatility (annualized)."""
    log_hl_sq = 0.5 * (pl.col("high").log() - pl.col("low").log()) ** 2
    log_co_sq = (2 * np.log(2) - 1) * (pl.col("close").log() - pl.col("open").log()) ** 2
    return ((log_hl_sq - log_co_sq).rolling_mean(period) * 252).sqrt()


# %%
# Compare all four estimators
vol_compare_df = spy.with_columns(
    pl.col("close").pct_change().alias("ret"),
).with_columns(
    [
        # Close-to-close — realized_volatility expects returns, not prices
        realized_volatility("ret", period=21, annualize=True).alias("vol_cc"),
        # Parkinson (manual — works on OHLC directly)
        parkinson_vol(period=21).alias("vol_parkinson"),
        # Garman-Klass (manual — works on OHLC directly)
        garman_klass_vol(period=21).alias("vol_gk"),
        # Yang-Zhang (library — works on OHLC directly, most efficient)
        yang_zhang_volatility("open", "high", "low", "close", period=21, annualize=True).alias(
            "vol_yz"
        ),
    ]
)

# %%
# 4-panel comparison
n = 504  # Last ~2 years for visualization
fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=["Close-to-Close", "Parkinson (H-L)", "Garman-Klass (OHLC)", "Yang-Zhang"],
    vertical_spacing=0.10,
    horizontal_spacing=0.06,
)

for idx, (col, name) in enumerate(
    [("vol_cc", "CC"), ("vol_parkinson", "Park"), ("vol_gk", "GK"), ("vol_yz", "YZ")]
):
    row, c = idx // 2 + 1, idx % 2 + 1
    fig.add_trace(
        go.Scatter(
            x=vol_compare_df["timestamp"].to_list()[-n:],
            y=vol_compare_df[col].to_list()[-n:],
            name=name,
        ),
        row=row,
        col=c,
    )

fig.update_layout(height=500, title="Volatility Estimator Comparison — SPY (21-day)")
fig.show()

# %% [markdown]
# **Interpretation**: Range-based estimators (Parkinson, GK, YZ) are smoother
# and more responsive than close-to-close. During high-volatility events
# (e.g., March 2020), the range-based estimators capture intraday dynamics
# that CC misses. Yang-Zhang is preferred for most applications.

# %% [markdown]
# ### Book Figure: Four OHLC Estimators Overlaid
#
# Static matplotlib version for print publication — all four estimators on a
# single panel with line-style variation for grayscale readability.

# %%
import matplotlib.pyplot as plt

# Prepare data — use full history for a window that includes a volatility spike
vol_plot = vol_compare_df.drop_nulls(subset=["vol_cc", "vol_parkinson", "vol_gk", "vol_yz"])

fig_mpl, ax = plt.subplots(figsize=(12, 5))

# SPY close price on secondary axis (context for vol spikes)
ax2 = ax.twinx()
dates = vol_plot["timestamp"].to_list()
ax2.fill_between(dates, vol_plot["close"].to_list(), alpha=0.08, color="black")
ax2.set_ylabel("SPY Close ($)", color="0.5")
ax2.tick_params(axis="y", labelcolor="0.5")

# Vol estimators on primary axis (convert to percentage for readability)
estimators = [
    ("vol_cc", "Close-to-Close (eff. 1.0\u00d7)", "-", "black"),
    ("vol_parkinson", "Parkinson (eff. 5.2\u00d7)", "--", "#555555"),
    ("vol_gk", "Garman-Klass (eff. 7.4\u00d7)", ":", "#333333"),
    ("vol_yz", "Yang-Zhang (eff. 8.4\u00d7)", "-.", "#666666"),
]

for col, label, ls, color in estimators:
    vals = [v * 100 for v in vol_plot[col].to_list()]
    ax.plot(dates, vals, ls=ls, color=color, label=label, linewidth=1.2)

ax.set_ylabel("Annualized Volatility (%)")
ax.set_title("Rolling Volatility Estimates: Four OHLC Estimators on SPY")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)

plt.show()

# Persist source data so book/08_financial_features/figures/scripts/generate_figure_8_3_*.py
# can re-render at print resolution without re-executing this notebook (Hard Rule 15).
_FIG_8_3_ARTIFACT = (
    get_chapter_dir(8)
    / "output"
    / "book_figure_artifacts"
    / "figure_8_3_ohlc_volatility_estimators.parquet"
)
_FIG_8_3_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
vol_plot.select(
    ["timestamp", "close", "vol_cc", "vol_parkinson", "vol_gk", "vol_yz"]
).write_parquet(_FIG_8_3_ARTIFACT)

# %% [markdown]
# ### 4.3 Volatility-of-Volatility (Vol-of-Vol)
#
# Second moment of volatility — useful for detecting unstable regimes.

# %%
from ml4t.engineer.features.volatility import volatility_of_volatility

vov_df = spy.with_columns(
    [
        yang_zhang_volatility("open", "high", "low", "close", period=21, annualize=True).alias(
            "vol"
        ),
        volatility_of_volatility("close", vol_period=21, vov_period=21).alias("vol_of_vol"),
    ]
)

print("Vol-of-vol (last 10 rows):")
vov_df.select(["timestamp", "vol", "vol_of_vol"]).tail(10)

# %% [markdown]
# ### 4.4 Volatility State Features
#
# Volatility state features transform continuous vol into conditioning variables:
#
# - **Vol ratio** (short/long): Detects expansion/contraction
# - **Vol percentile**: Where current vol sits in its 252-day history
# - **Vol decile**: Binned version for evaluation slicing

# %%
SHORT_VOL_WINDOW = 10
LONG_VOL_WINDOW = 63
PERCENTILE_LOOKBACK = 252

vol_state_df = (
    spy.with_columns(
        pl.col("close").log().diff().alias("log_return"),
    )
    .with_columns(
        [
            (pl.col("log_return").rolling_std(SHORT_VOL_WINDOW) * np.sqrt(252)).alias("vol_short"),
            (pl.col("log_return").rolling_std(LONG_VOL_WINDOW) * np.sqrt(252)).alias("vol_long"),
        ]
    )
    .with_columns(
        # Vol ratio: short/long — >1 means expansion, <1 contraction
        (pl.col("vol_short") / pl.col("vol_long").clip(1e-10, None)).alias("vol_ratio"),
    )
)

# Vol percentile: rolling rank over 252 days
vol_state_df = vol_state_df.with_columns(
    pl.col("vol_short")
    .rolling_map(
        lambda s: 100 * (s.rank().last() - 1) / max(len(s) - 1, 1),
        window_size=PERCENTILE_LOOKBACK,
        min_periods=PERCENTILE_LOOKBACK // 2,
    )
    .alias("vol_percentile")
)

# Vol decile: clipped 0-9
vol_state_df = vol_state_df.with_columns(
    (pl.col("vol_percentile").fill_nan(None).fill_null(-10.0) / 10)
    .floor()
    .clip(0, 9)
    .cast(pl.Int32)
    .alias("vol_decile")
)

print("Volatility state features:")
vol_state_df.select(["timestamp", "vol_ratio", "vol_percentile", "vol_decile"]).tail(10)

# %% [markdown]
# ### 4.5 Price-Derived Regime Indicators
#
# `ml4t-engineer` provides rolling statistical tests that detect market regime
# changes without relying on parametric models (HMM, GARCH are in Chapter 9):
#
# | Indicator | Test/Signal | Interpretation |
# |-----------|-------------|----------------|
# | **Variance Ratio** | Lo-MacKinlay RW test | >1 trending, <1 mean-reverting |
# | **Fractal Efficiency** | Mandelbrot efficiency | 1=trending, 0=noise |
# | **Trend Intensity** | ADX-like directional | High=strong trend |

# %%
from ml4t.engineer.features.regime import fractal_efficiency, trend_intensity_index, variance_ratio

vr_exprs = variance_ratio("close", periods=[5], window=21)
regime_ind_df = spy.with_columns(
    [
        vr_exprs["vr_5"].alias("variance_ratio_21d"),
        fractal_efficiency("close", period=21).alias("fractal_eff_21d"),
        trend_intensity_index("close", period=21).alias("trend_intensity_21d"),
    ]
)

print("Regime indicators:")
regime_ind_df.select(
    ["timestamp", "variance_ratio_21d", "fractal_eff_21d", "trend_intensity_21d"]
).tail(10)

# %% [markdown]
# **Variance Ratio**: Values above 1 indicate positive autocorrelation (trending);
# below 1 indicates mean-reversion. This is the Lo-MacKinlay (1988) test as a
# rolling feature.
#
# **Fractal Efficiency**: Measures path efficiency — 1 means price moves in a
# straight line (trend), 0 means random wandering.
#
# **Note**: These are *rolling-window* regime indicators derived purely from price.
# Parametric regime models (HMM, Markov-switching GARCH) appear in Chapter 9.

# %% [markdown]
# ## 5. Volume and Liquidity Features
#
# Volume features proxy for:
# - **Attention**: High volume = information event
# - **Liquidity**: Capacity to trade
# - **Conviction**: Volume confirms price moves

# %% [markdown]
# ### 5.1 Dollar Volume
#
# Raw volume is not comparable across assets. Dollar volume normalizes.
#
# $$\text{Dollar Volume} = \text{Volume} \times \text{Close}$$

# %%
volume_df = spy.with_columns(
    [
        (pl.col("volume") * pl.col("close")).alias("dollar_volume"),
        # Log transform for better distribution
        (pl.col("volume") * pl.col("close")).log().alias("log_dollar_volume"),
    ]
)

print("Dollar volume:")
volume_df.select(["timestamp", "close", "volume", "dollar_volume"]).tail(10)

# %% [markdown]
# ### 5.2 Relative Volume
#
# Volume relative to its recent average. >1 means above-average activity.
#
# $$\text{Relative Volume} = \frac{V_t}{\text{SMA}_{21}(V)}$$

# %%
rel_vol_df = spy.with_columns(
    [
        (pl.col("volume") / pl.col("volume").rolling_mean(21)).alias("rel_volume"),
        # Volume shock: z-score of log volume
        (
            (pl.col("volume").log() - pl.col("volume").log().rolling_mean(21))
            / pl.col("volume").log().rolling_std(21)
        ).alias("volume_zscore"),
    ]
)

# %%
# Visualize relative volume features
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["SPY Close", "Relative Volume", "Volume Z-Score"],
    vertical_spacing=0.08,
)

fig.add_trace(
    go.Scatter(
        x=rel_vol_df["timestamp"].to_list()[-n:], y=rel_vol_df["close"].to_list()[-n:], name="Close"
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=rel_vol_df["timestamp"].to_list()[-n:],
        y=rel_vol_df["rel_volume"].to_list()[-n:],
        name="Rel Vol",
        fill="tozeroy",
    ),
    row=2,
    col=1,
)
fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
fig.add_trace(
    go.Scatter(
        x=rel_vol_df["timestamp"].to_list()[-n:],
        y=rel_vol_df["volume_zscore"].to_list()[-n:],
        name="Vol Z-Score",
    ),
    row=3,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
fig.add_hline(y=2, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=-2, line_dash="dash", line_color="red", row=3, col=1)

fig.update_layout(height=600, title="Volume Features")
fig.show()

# %% [markdown]
# ### 5.3 VWAP Distance
#
# How far price has moved from the volume-weighted average price.
# Useful for intraday strategies and execution.
#
# $$\text{VWAP Distance} = \frac{P_t - VWAP_t}{VWAP_t}$$

# %%
# Daily VWAP using typical price as proxy
vwap_df = (
    spy.with_columns(
        [
            # Typical price as VWAP proxy for daily data
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price"),
        ]
    )
    .with_columns(
        [
            # Rolling VWAP (volume-weighted rolling mean)
            (
                (pl.col("typical_price") * pl.col("volume")).rolling_sum(5)
                / pl.col("volume").rolling_sum(5)
            ).alias("vwap_5d"),
        ]
    )
    .with_columns(
        [
            ((pl.col("close") / pl.col("vwap_5d")) - 1).alias("vwap_distance"),
        ]
    )
)

print("VWAP distance:")
vwap_df.select(["timestamp", "close", "vwap_5d", "vwap_distance"]).tail(10)

# %% [markdown]
# **Interpretation**: Positive VWAP distance means the close is above the
# volume-weighted average — buying pressure exceeded selling pressure over the
# lookback. Negative indicates the opposite. For intraday strategies this is a
# key mean-reversion anchor; for daily data it proxies volume-weighted trend.

# %% [markdown]
# ## 6. Cross-Sectional Normalization
#
# For multi-asset strategies, raw features are not comparable. Normalization
# creates standardized signals across the universe.
#
# **Warning**: Cross-sectional operations can introduce leakage if not careful
# about timing. Always use point-in-time data.

# %% [markdown]
# ### 6.1 Cross-Sectional Ranks
#
# Rank-based features are robust to outliers.
#
# $$\text{Rank}_{t,i} = \frac{\text{rank}(f_{t,i})}{\text{N}_t}$$

# %%
# Add momentum to cross-sectional data
cs_mom = cs_etfs.with_columns(
    [
        pl.col("close").pct_change(21).over("symbol").alias("mom_21d"),
    ]
)

# Cross-sectional rank within each day
cs_ranked = cs_mom.with_columns(
    [
        # Rank: 0 = lowest momentum, 1 = highest
        (
            pl.col("mom_21d").rank().over("timestamp") / pl.col("symbol").count().over("timestamp")
        ).alias("mom_rank"),
    ]
)

# Show one day
sample_date = cs_ranked["timestamp"].max()
print(f"Cross-sectional ranks for {sample_date}:")
(
    cs_ranked.filter(pl.col("timestamp") == sample_date)
    .select(["symbol", "mom_21d", "mom_rank"])
    .sort("mom_rank", descending=True)
)

# %% [markdown]
# ### 6.2 Vol-Scaled Cross-Sectional Momentum
#
# The text's spec-table formula: cumulative return divided by realized volatility,
# then cross-sectional percentile rank. Vol-scaling penalizes momentum driven by
# high volatility — a stock that rose 10% with 40% vol is less compelling than one
# that rose 10% with 15% vol.
#
# $$\text{Vol-Scaled Mom} = \frac{r_{21d}}{\sigma_{21d}}$$

# %%
# Vol-scaled momentum: return / realized vol
cs_vol_scaled = cs_etfs.with_columns(
    [
        pl.col("close").pct_change(21).over("symbol").alias("mom_21d"),
        (
            pl.col("close").pct_change().over("symbol").rolling_std(21).over("symbol")
            * np.sqrt(252)
        ).alias("vol_21d"),
    ]
).with_columns(
    (pl.col("mom_21d") / pl.col("vol_21d").clip(1e-10, None)).alias("vol_scaled_mom"),
)

# Cross-sectional percentile rank
cs_vol_ranked = cs_vol_scaled.with_columns(
    (
        pl.col("vol_scaled_mom").rank().over("timestamp")
        / pl.col("symbol").count().over("timestamp")
    ).alias("vol_scaled_rank"),
)

# Compare raw vs vol-scaled rankings
print(f"Vol-scaled cross-sectional momentum ({sample_date}):")
(
    cs_vol_ranked.filter(pl.col("timestamp") == sample_date)
    .select(["symbol", "mom_21d", "vol_21d", "vol_scaled_mom", "vol_scaled_rank"])
    .sort("vol_scaled_rank", descending=True)
)

# %% [markdown]
# **Interpretation**: Vol-scaling reshuffles the ranking — high-momentum assets
# with elevated volatility drop, while steady trending assets rise. This is the
# same intuition as the Sharpe ratio applied cross-sectionally.

# %% [markdown]
# ### 6.3 Cross-Sectional Z-Scores
#
# Z-score normalization assumes (roughly) normal distribution.
#
# $$z_{t,i} = \frac{f_{t,i} - \mu_t}{\sigma_t}$$
#
# **Robustness**: Use median/MAD instead of mean/std for robustness to outliers.

# %%
# Z-score within each day
cs_zscored = cs_mom.with_columns(
    [
        # Standard z-score
        (
            (pl.col("mom_21d") - pl.col("mom_21d").mean().over("timestamp"))
            / pl.col("mom_21d").std().over("timestamp")
        ).alias("mom_zscore"),
        # Robust z-score (using median and MAD)
        # MAD × 1.4826 ≈ σ for normal data (1.4826 = 1/Φ⁻¹(3/4))
        (
            (pl.col("mom_21d") - pl.col("mom_21d").median().over("timestamp"))
            / (
                (pl.col("mom_21d") - pl.col("mom_21d").median().over("timestamp"))
                .abs()
                .median()
                .over("timestamp")
                * 1.4826
            )
        ).alias("mom_zscore_robust"),
    ]
)

# Show one day
print(f"Cross-sectional z-scores for {sample_date}:")
(
    cs_zscored.filter(pl.col("timestamp") == sample_date)
    .select(["symbol", "mom_21d", "mom_zscore", "mom_zscore_robust"])
    .sort("mom_zscore", descending=True)
)

# %% [markdown]
# ### 6.4 Leakage Warning: Point-in-Time Discipline
#
# **Common Leakage Patterns**:
#
# 1. **Using future data**: Z-score computed over future values
# 2. **Survivorship bias**: Only including current constituents
# 3. **Look-ahead in ranks**: Ranking before data was available
#
# **Safe Pattern**: Always use `.over("timestamp")` for cross-sectional
# operations — this guarantees that each date's statistics use only
# contemporaneous data:
#
# ```python
# # WRONG: full-sample z-score includes future data
# pl.col("mom").zscore()
#
# # CORRECT: point-in-time cross-sectional z-score
# (pl.col("mom") - pl.col("mom").mean().over("timestamp"))
# / pl.col("mom").std().over("timestamp")
# ```

# %% [markdown]
# ## 7. Risk Features
#
# Risk features capture the *shape* of the return distribution beyond simple volatility.
# Tail risk measures like VaR, CVaR, and tail ratio are essential for:
# - **Position sizing**: Scale down exposure when tail risk is elevated
# - **Feature conditioning**: Momentum works differently in fat-tail vs thin-tail regimes
# - **Risk-adjusted signals**: Sharpe ratios penalize return/vol; CVaR penalizes tail events

# %%
from ml4t.engineer.features.risk import (
    conditional_value_at_risk,
    downside_deviation,
    tail_ratio,
    value_at_risk,
)

risk_df = spy.with_columns(
    [
        pl.col("close").pct_change().alias("ret"),
    ]
).with_columns(
    [
        value_at_risk("ret", confidence_level=0.95, window=63).alias("var_5pct_63d"),
        conditional_value_at_risk("ret", confidence_level=0.95, window=63).alias("cvar_5pct_63d"),
        downside_deviation("ret", window=63).alias("downside_dev_63d"),
        tail_ratio("ret", confidence_level=0.95, window=63).alias("tail_ratio_63d"),
    ]
)

print("Risk features (last 10 rows):")
risk_df.select(
    ["timestamp", "var_5pct_63d", "cvar_5pct_63d", "downside_dev_63d", "tail_ratio_63d"]
).tail(10)

# %% [markdown]
# | Risk Feature | Interpretation | Trading Use |
# |-------------|----------------|-------------|
# | **VaR (5% tail)** | Max expected loss in worst 5% of days | Position sizing threshold |
# | **CVaR (5%)** | Expected loss beyond VaR | Tail risk penalty |
# | **Downside Deviation** | Volatility of negative returns only | Sortino ratio denominator |
# | **Tail Ratio** | Right tail / left tail size | Asymmetry of return distribution |

# %% [markdown]
# ## 8. ML-Specific Transforms
#
# ML-specific transforms prepare features for tree and linear models:
# - **Fractional differencing**: Makes features stationary while preserving memory
# - **Volatility-adjusted returns**: Standardizes by recent volatility

# %%
from ml4t.engineer.features.fdiff import ffdiff

# Fractional differencing preserves memory while achieving stationarity
# d=0.5 is a common starting point; use find_optimal_d() for data-driven choice
ffd_df = spy.with_columns(
    [
        ffdiff("close", d=0.5).alias("close_ffd_05"),
        ffdiff("close", d=1.0).alias("close_ffd_10"),  # Equivalent to first difference
    ]
)

print("Fractional differencing (d=0.5 vs d=1.0):")
ffd_df.select(["timestamp", "close", "close_ffd_05", "close_ffd_10"]).tail(10)

# %% [markdown]
# | Transform | d Value | Stationarity | Memory | Use Case |
# |-----------|---------|-------------|--------|----------|
# | Original | 0.0 | Non-stationary | Full | Not for ML |
# | Fractional | 0.3-0.5 | Near-stationary | Preserved | Tree models, regressions |
# | First diff | 1.0 | Stationary | Lost | Benchmark comparison |
#
# Fractional differencing with $d \approx 0.5$ is a good default for financial
# time series. It achieves stationarity (required by most ML models) while
# retaining long-range dependence that purely differenced series lose.

# %% [markdown]
# **Caveat**: With finite truncation, `d=1.0` *approximates* but does not
# exactly equal first differencing. The truncated weight series drops small
# high-lag coefficients that a true first difference implicitly includes. For
# practical purposes the difference is negligible, but be aware when comparing
# FFD output to `pct_change()`.

# %% [markdown]
# **ml4t-engineer Production API**: For production feature computation using
# config-driven batch processing and library RSI/MACD/Bollinger implementations,
# see `10_ml4t_library_ecosystem` (Chapter 7).

# %% [markdown]
# ## Summary
#
# ### Feature Family Decision Guide
#
# | Family | When to Use | Key Considerations |
# |--------|-------------|-------------------|
# | **Returns** | Base signals, momentum | Skip-1 for short horizons |
# | **Trend** | Trend-following | Vol-scale for comparability |
# | **Volatility** | Risk scaling, sizing | Yang-Zhang most efficient |
# | **Vol State** | Regime conditioning | Percentile > decile for granularity |
# | **Regime Indicators** | Trend detection | Rolling-window, model-free |
# | **Volume** | Liquidity, conviction | Use relative, not raw |
# | **Risk** | Position sizing, tail risk | VaR, CVaR, downside dev |
# | **Cross-sectional** | Multi-asset | Point-in-time discipline |
# | **ML transforms** | Stationarity | Fractional differencing |
#
# ### Implementation Rules
#
# 1. **Sort once at load**: `df.sort(["symbol", "timestamp"])`
# 2. **Single with_columns**: All transforms in one call for parallelism
# 3. **Vol-scale for comparability**: Raw features vary with price level
# 4. **Library for production**: ml4t-engineer for validated implementations
# 5. **Point-in-time for cross-sectional**: Use `.over("timestamp")`
#
# ### Next Notebooks
#
# - `02_microstructure_features` — Trade-based features (§8.2)
# - `03_structural_cross_instrument_features` — Carry, cross-asset, options (§8.3)
# - `04_fundamentals_macro_calendar` — Fundamentals, macro, calendar (§8.4)
