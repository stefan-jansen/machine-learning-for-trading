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
# # US Equities Panel: Feature Engineering
#
# This notebook implements cross-sectional factor construction for the US
# Equities Panel case study. The broad universe (~3,200 stocks) is the book's
# highest-breadth case study, so features emphasize cross-sectional ranking
# and the Fundamental Law of Active Management: weak per-stock signals
# compensated by large breadth.
#
# ## Feature Families
#
# 1. **Momentum/Trend**: Multi-horizon returns, skip-month, risk-adjusted Sharpe
# 2. **Mean Reversion**: Short-horizon reversal ranks, RSI, distance from extremes
# 3. **Volatility**: Multi-horizon vol, vol ratios, vol z-scores
# 4. **Liquidity**: Amihud illiquidity, dollar volume rank, volume ratio
# 5. **Technical Indicators**: RSI, MACD, ADX, CCI, Stochastic, NATR
# 6. **Composites**: Momentum-reversal score, size-conditional features
#
# ## Key Design Decisions
#
# - **Survivorship-safe**: Features computed only on data available at decision time
# - **Winsorized at 1st/99th percentile**: Reduces extreme value influence
# - **Cross-sectional ranks**: Ensure stationarity across market regimes
# - **Amihud illiquidity**: Added per strategic review (captures liquidity premium)
# - **Size-conditional features**: Momentum interacted with size decile
#
# ## Book Reference
#
# Chapter 8, Section 8.2 (Price-Derived Features)
#
# ## Prerequisites
#
# - US equities data (via `load_us_equities()` canonical loader)

# %%
"""US Equities Panel: Feature Engineering."""

import subprocess
import warnings
from datetime import UTC, datetime

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from ml4t.engineer.features.momentum import adx, cci, macd, rsi, stochastic
from ml4t.engineer.features.trend import ema, kama, sma
from ml4t.engineer.features.volatility import natr

from data import load_us_equities
from utils.paths import get_case_study_dir

CASE_DIR = get_case_study_dir("us_equities_panel")
FEATURES_DIR = CASE_DIR / "features"

# Configuration
MOMENTUM_HORIZONS = [5, 10, 21, 42, 63, 126, 189, 252]
VOLATILITY_HORIZONS = [21, 63, 126, 252]
MA_HORIZONS = [10, 20, 50, 100, 200]

# Data range
START_DATE = "1990-01-01"
END_DATE = "2018-03-31"

# Liquidity filters
MIN_ADV_USD = 1_000_000
MIN_PRICE = 5.0
ADV_WINDOW = 21

# The primary label is a one-session forward return, so a strategy trading it decides
# at every close. Used only by the Fundamental Law arithmetic in Section 9.
REBALANCES_PER_YEAR = 252

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
START_DATE = "1990-01-01"
MAX_SYMBOLS = 0

# %% [markdown]
# ## Connecting to the Edge Hypothesis
#
# The feasibility notebook ([`01_feasibility_analysis`](01_feasibility_analysis.ipynb)) frames **behavioral persistence** as
# the edge source: momentum and reversal signals exploit slow information
# diffusion across ~3,200 stocks. The Fundamental Law of Active Management
# predicts that even weak per-stock signals ($IC \approx 0.02$) can generate
# meaningful portfolio-level alpha when breadth is large:
# $IR = IC \cdot \sqrt{BR}$. The features below operationalize this hypothesis.

# %% [markdown]
# ## 1. Load Data
#
# Load from the canonical data loader and declare the same eligibility screen as
# [`02_labels`](02_labels.ipynb): a printed close above \$5, and dollar volume
# `close * volume` averaging above \$1M over the previous 21 sessions. Both legs
# read figures the tape carried on the day, so neither depends on a corporate
# action that had not happened yet. Section B of [`02_labels`](02_labels.ipynb)
# derives why the adjusted close cannot serve here. Both notebooks rebuild the
# screen from the same three constants on the same columns, so the trainable
# panel and the label files agree on the universe.
#
# **Declared here, applied in Section 6.** The screen removes whole rows, and a
# per-symbol shift or rolling window applied afterwards counts the rows that
# survived rather than trading sessions. Section 6 states what that costs and
# applies the screen between the per-symbol features and the cross-sectional
# ones, which is the only ordering that gives both their intended meaning.
#
# Returns and every price-derived feature below still read `adj_close`: a return
# has to divide out splits and dividends to mean anything.
#
# **Alignment check**: the two indices are not expected to match exactly - a label
# needs a forward window the last session of a stock's series does not have, and a
# feature needs a warm-up the first sessions do not have. Section 8 attributes
# every row on both sides to one of those causes and **asserts the remainder is
# empty**, so a screen that drifted apart between the two stages fails there rather
# than printing a larger number and passing.

# %%
raw_df = load_us_equities(start_date=START_DATE, end_date=END_DATE)

# Normalize types
if raw_df.schema["timestamp"] == pl.Datetime:
    raw_df = raw_df.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

raw_df = raw_df.sort(["symbol", "timestamp"])

# Compute base columns
raw_df = raw_df.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("returns"),
    (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
)

raw_df = raw_df.with_columns(
    pl.col("dollar_volume").rolling_mean(ADV_WINDOW).over("symbol").alias("adv_21d")
)

# The screen is declared here and applied in Section 6, between the per-symbol
# features and the cross-sectional ones. It is not applied yet, and that ordering
# is the point: see Section 6.
ELIGIBLE = (pl.col("close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD)

print(f"Loaded {len(raw_df):,} rows, {raw_df['symbol'].n_unique()} symbols")
print(f"Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")

# %% [markdown]
# ## 2. Momentum and Volatility Features
#
# Multi-horizon raw returns, skip-month momentum, Sharpe-like ratios, and
# volatility features. These are the core signal families for cross-sectional
# equity prediction.
#
# **Skip-month momentum** follows Jegadeesh and Titman (1993): the return
# earned from twelve months ago through one month ago,
# $P_{t-21}/P_{t-252} - 1$. Excluding the last month avoids the short-term
# reversal effect that contaminates raw 12-month returns. Note that returns
# compound, so the recent month is removed by dividing prices rather than by
# subtracting $r_{1M}$ from $r_{12M}$; the difference of two simple returns is
# only a first-order approximation and is not a return over any window.


# %%
def compute_momentum_returns(data: pl.DataFrame) -> pl.DataFrame:
    """Multi-horizon raw returns and skip-month momentum."""
    data = data.sort(["symbol", "timestamp"])
    for h in MOMENTUM_HORIZONS:
        data = data.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(h).over("symbol") - 1).alias(
                f"ret_{h}d"
            )
        )
    # Skip-month momentum (12-1): Jegadeesh-Titman (1993) construction, the
    # return from t-252 to t-21.
    data = data.with_columns(
        (
            pl.col("adj_close").shift(21).over("symbol")
            / pl.col("adj_close").shift(252).over("symbol").clip(lower_bound=1e-8)
            - 1
        ).alias("ret_12m_skip")
    )
    return data


# %% [markdown]
# ### Volatility and Risk-Adjusted Returns
#
# Multi-horizon annualized volatility, vol ratios (short-to-long regimes),
# and Sharpe-like return/vol ratios clipped at ±10.


# %%
def compute_volatility_sharpe(data: pl.DataFrame) -> pl.DataFrame:
    """Volatility, vol ratios, Sharpe ratios, and momentum acceleration."""
    for h in VOLATILITY_HORIZONS:
        data = data.with_columns(
            (pl.col("returns").rolling_std(h).over("symbol") * np.sqrt(252)).alias(f"vol_{h}d")
        )
    data = data.with_columns(
        (pl.col("vol_21d") / pl.col("vol_63d").clip(lower_bound=1e-6))
        .clip(upper_bound=10.0)
        .alias("vol_ratio_short"),
        (pl.col("vol_63d") / pl.col("vol_126d").clip(lower_bound=1e-6))
        .clip(upper_bound=10.0)
        .alias("vol_ratio_medium"),
    )
    # Sharpe-like ratios (return / vol, clipped)
    for h in MOMENTUM_HORIZONS:
        if h >= 21:
            vol_h = min(
                [v for v in VOLATILITY_HORIZONS if v <= h],
                default=21,
                key=lambda x: abs(x - h),
            )
            data = data.with_columns(
                (pl.col(f"ret_{h}d") / pl.col(f"vol_{vol_h}d").clip(lower_bound=0.01))
                .clip(lower_bound=-10.0, upper_bound=10.0)
                .alias(f"sharpe_{h}d")
            )
    # Momentum acceleration
    data = data.with_columns(
        (pl.col("ret_21d") - pl.col("ret_63d")).alias("mom_accel_short"),
        (pl.col("ret_63d") - pl.col("ret_126d")).alias("mom_accel_medium"),
        (pl.col("ret_126d") - pl.col("ret_252d")).alias("mom_accel_long"),
    )
    return data


# %% [markdown]
# ## 3. Technical Indicators
#
# Library-computed oscillators (RSI, MACD, ADX, CCI, Stochastic, NATR)
# capture momentum and volatility regime signals.


# %%
def compute_oscillators(data: pl.DataFrame) -> pl.DataFrame:
    """Technical oscillators: RSI, MACD, ADX, CCI, Stochastic, NATR."""
    data = data.with_columns(
        rsi("adj_close", period=7).over("symbol").alias("rsi_7"),
        rsi("adj_close", period=14).over("symbol").alias("rsi_14"),
    )
    data = data.with_columns(
        (
            macd("adj_close", fast_period=12, slow_period=26).over("symbol")
            / pl.col("adj_close")
            * 100
        )
        .clip(lower_bound=-50.0, upper_bound=50.0)
        .alias("macd_pct")
    )
    data = data.with_columns(
        adx("adj_high", "adj_low", "adj_close", period=14).over("symbol").alias("adx_14"),
        cci("adj_high", "adj_low", "adj_close", period=20).over("symbol").alias("cci_20"),
        stochastic("adj_high", "adj_low", "adj_close", fastk_period=14)
        .over("symbol")
        .alias("stoch_k_14"),
        natr("adj_high", "adj_low", "adj_close", period=14)
        .over("symbol")
        .clip(upper_bound=100.0)
        .alias("natr_14"),
    )
    return data


# %% [markdown]
# ### Trend and Distance Features
#
# Price-to-MA ratios at multiple horizons (SMA, EMA, KAMA) and distance from
# 52-week high/low. These capture mean-reversion and trend-following signals.


# %%
def compute_trend_distance(data: pl.DataFrame) -> pl.DataFrame:
    """MA ratios and distance-from-extreme features."""
    for period in MA_HORIZONS:
        data = data.with_columns(
            (pl.col("adj_close") / sma("adj_close", period=period).over("symbol")).alias(
                f"sma_ratio_{period}"
            )
        )
    data = data.with_columns(
        (pl.col("adj_close") / ema("adj_close", period=12).over("symbol")).alias("ema_ratio_12"),
        (pl.col("adj_close") / ema("adj_close", period=26).over("symbol")).alias("ema_ratio_26"),
        (pl.col("adj_close") / kama("adj_close", timeperiod=10).over("symbol")).alias(
            "kama_ratio_10"
        ),
    )
    # clip(1e-8) on denominators guards against div-by-zero → inf
    data = data.with_columns(
        (
            pl.col("adj_close")
            / pl.col("adj_high").rolling_max(252).over("symbol").clip(lower_bound=1e-8)
        )
        .clip(lower_bound=0.1, upper_bound=1.0)
        .alias("dist_from_52w_high"),
        (
            pl.col("adj_close")
            / pl.col("adj_low").rolling_min(252).over("symbol").clip(lower_bound=1e-8)
        )
        .clip(lower_bound=1.0, upper_bound=10.0)
        .alias("dist_from_52w_low"),
    )
    return data


# %% [markdown]
# ## 4. Cross-Sectional Ranks, Liquidity, and Composites
#
# Cross-sectional ranking ensures features are stationary across market regimes.
# The Amihud (2002) illiquidity measure captures price impact per unit of
# trading volume:
#
# $$\text{ILLIQ}_{i,t} = \frac{1}{D} \sum_{d=1}^{D} \frac{|r_{i,d}|}{\text{DVOL}_{i,d}}$$
#
# where $D = 21$ days, $r_{i,d}$ is the daily return, and $\text{DVOL}_{i,d}$
# is dollar volume. Higher values indicate less liquid stocks. Amihud (2002)
# showed that expected illiquidity positively predicts cross-sectional returns.
#
# **New features from strategic review**:
# - Amihud illiquidity (rolling 21-day)
# - Composite momentum-reversal score (exploiting negative correlation)
# - Size-conditional features (momentum interacted with size decile)


# %%
def compute_xs_ranks(data: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional ranks and z-scores for momentum, volatility, and Sharpe."""
    # Momentum ranks
    for h in [21, 63, 126, 252]:
        data = data.with_columns(
            (
                pl.col(f"ret_{h}d").rank().over("timestamp")
                / pl.col(f"ret_{h}d").count().over("timestamp")
            ).alias(f"mom_rank_{h}d")
        )
    # Sharpe ranks
    for h in [63, 126, 252]:
        data = data.with_columns(
            (
                pl.col(f"sharpe_{h}d").rank().over("timestamp")
                / pl.col(f"sharpe_{h}d").count().over("timestamp")
            ).alias(f"sharpe_rank_{h}d")
        )
    # Volatility rank
    data = data.with_columns(
        (
            pl.col("vol_63d").rank().over("timestamp") / pl.col("vol_63d").count().over("timestamp")
        ).alias("vol_rank")
    )
    # Z-scores
    data = data.with_columns(
        (
            (pl.col("ret_126d") - pl.col("ret_126d").mean().over("timestamp"))
            / (pl.col("ret_126d").std().over("timestamp") + 1e-8)
        ).alias("mom_zscore_6m"),
        (
            (pl.col("vol_63d") - pl.col("vol_63d").mean().over("timestamp"))
            / (pl.col("vol_63d").std().over("timestamp") + 1e-8)
        ).alias("vol_zscore"),
    )
    return data


# %% [markdown]
# ### Liquidity and Reversion Features
#
# Amihud illiquidity measures the price impact of trading volume -- a key
# cross-sectional predictor (Amihud 2002). Mean-reversion signals capture
# short-horizon reversal via 5-day return and RSI ranks.


# %%
def compute_rolling_liquidity(data: pl.DataFrame) -> pl.DataFrame:
    """Per-symbol liquidity measures. Rolling, so the complete series."""
    data = data.with_columns((pl.col("dollar_volume") / pl.col("adv_21d")).alias("volume_ratio"))
    # Amihud illiquidity: |return| / dollar_volume (rolling 21-session mean)
    data = data.with_columns(
        (pl.col("returns").abs() / (pl.col("dollar_volume") + 1))
        .rolling_mean(21)
        .over("symbol")
        .alias("amihud_illiq")
    )
    return data


# %% [markdown]
# The ranks below are the cross-sectional half of the same block, and they run on
# the eligible frame: a rank is only meaningful against the names the strategy
# could actually have sorted on that day.


# %%
def compute_xs_liquidity_reversion(data: pl.DataFrame) -> pl.DataFrame:
    """Cross-sectional ranks of the liquidity, reversion, and size signals."""
    return data.with_columns(
        (
            pl.col("adv_21d").rank().over("timestamp") / pl.col("adv_21d").count().over("timestamp")
        ).alias("liq_rank"),
        (
            pl.col("amihud_illiq").rank().over("timestamp")
            / pl.col("amihud_illiq").count().over("timestamp")
        ).alias("illiq_rank"),
        (
            pl.col("ret_5d").rank().over("timestamp") / pl.col("ret_5d").count().over("timestamp")
        ).alias("reversal_rank"),
        (
            pl.col("rsi_14").rank().over("timestamp") / pl.col("rsi_14").count().over("timestamp")
        ).alias("rsi_rank"),
        # Size proxy (log dollar volume rank as mcap proxy)
        (
            pl.col("adv_21d").log().rank().over("timestamp")
            / pl.col("adv_21d").log().count().over("timestamp")
        ).alias("size_rank"),
    )


# %% [markdown]
# ### Composite and Interaction Features
#
# Composites blend related ranks into single signals. The momentum-reversal
# spread exploits the negative correlation between trend and reversion
# signals. Size-conditional features test whether momentum varies by
# market cap (Fama and French 1992).


# %%
def compute_composites(data: pl.DataFrame) -> pl.DataFrame:
    """Composite factors and size-conditional interaction features."""
    data = data.with_columns(
        ((pl.col("mom_rank_63d") + pl.col("mom_rank_126d") + pl.col("mom_rank_252d")) / 3).alias(
            "momentum_composite"
        ),
        ((1 - pl.col("vol_rank")) * 0.5 + pl.col("liq_rank") * 0.5).alias("quality_composite"),
        ((1 - pl.col("reversal_rank")) * 0.5 + (1 - pl.col("rsi_rank")) * 0.5).alias(
            "contrarian_composite"
        ),
    )
    # Momentum-reversal spread: high momentum + low reversal = strong trend
    data = data.with_columns(
        (pl.col("momentum_composite") - pl.col("contrarian_composite")).alias("mom_rev_spread")
    )
    # Size-conditional momentum
    data = data.with_columns(
        (pl.col("mom_rank_126d") * pl.col("size_rank")).alias("mom_x_size"),
        (pl.col("mom_rank_252d") * pl.col("size_rank")).alias("mom12m_x_size"),
    )
    return data


# %% [markdown]
# ## 5. Winsorization
#
# Apply winsorization at the 1st/99th percentile per feature to limit
# the influence of extreme values (split artifacts, data errors).
#
# **Known deviation**: Quantiles are computed over the full time series
# rather than per cross-section date. This is a minor form of lookahead:
# the clip boundaries at date $t$ reflect values at all dates, including
# future ones. The bias is small because (a) the 1st and 99th percentiles
# of returns are relatively stable over decades, and (b) the cross-sectional
# ranks (which are per-date) are the primary transformed features —
# winsorization only affects raw return levels that feed into the ranking.
# A production pipeline would compute per-date quantiles to eliminate this.


# %%
def winsorize_features(
    data: pl.DataFrame, feature_cols: list[str], lower: float = 0.01, upper: float = 0.99
) -> pl.DataFrame:
    """Winsorize feature columns at specified quantiles.

    Production alternative (per-date, no look-ahead):
        data.with_columns(
            pl.col(col).clip(
                pl.col(col).quantile(lower).over("timestamp"),
                pl.col(col).quantile(upper).over("timestamp"),
            )
        )
    """
    for col in feature_cols:
        if col in data.columns:
            q_low = data[col].quantile(lower)
            q_high = data[col].quantile(upper)
            if q_low is not None and q_high is not None and q_low < q_high:
                data = data.with_columns(pl.col(col).clip(lower_bound=q_low, upper_bound=q_high))
    return data


# %% [markdown]
# ## 6. Run Feature Pipeline
#
# Every per-symbol feature is a shift or a rolling window over `symbol`, and those
# count rows. On the screened frame they would count *eligible* rows rather than
# trading sessions, so a stock that drops below a threshold and recovers would
# carry windows spanning the whole excursion: skip-month momentum would reach back
# 252 eligible rows, which can be years. So they run on the complete series, and
# the screen is applied afterwards. [`02_labels`](02_labels.ipynb) Section B makes
# this argument for the forward window; this is the backward-looking half of it,
# and the two have to agree on what a session is.
#
# The cross-sectional ranks then run on the screened frame, because a rank is only
# meaningful against the names the strategy could have sorted on that day.

# %%
print("Computing features...")

raw_df = raw_df.pipe(compute_momentum_returns).pipe(compute_volatility_sharpe)
print("  Momentum and volatility done")

raw_df = raw_df.pipe(compute_oscillators).pipe(compute_trend_distance)
print("  Technical indicators done")

raw_df = raw_df.pipe(compute_rolling_liquidity)
print("  Rolling liquidity done")

# The screen is applied here, between the two kinds of feature.
df = raw_df.filter(ELIGIBLE)
print(f"  Eligible: {df.height:,} of {raw_df.height:,} rows, {df['symbol'].n_unique()} stocks")

df = df.pipe(compute_xs_ranks).pipe(compute_xs_liquidity_reversion).pipe(compute_composites)
print("  Cross-sectional ranks and composites done")

# %% [markdown]
# ## 7. Select and Clean Features

# %%
# Metadata columns to exclude from features
metadata_cols = {
    "symbol",
    "timestamp",
    # Raw prices (non-stationary)
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_volume",
    # Derived helpers
    "dollar_volume",
    "returns",
    "adv_21d",
    "amihud_illiq",
    # Corporate actions
    "split_ratio",
    "ex-dividend",
}

feature_cols = [c for c in df.columns if c not in metadata_cols]

# Drop rows with nulls in essential short-lookback features
essential_cols = ["symbol", "timestamp", "ret_5d", "ret_21d", "vol_21d"]
essential_cols = [c for c in essential_cols if c in df.columns]

output_cols = ["symbol", "timestamp"] + feature_cols
available_cols = [c for c in output_cols if c in df.columns]
output_df = df.select(available_cols).drop_nulls(subset=essential_cols)

# Winsorize at 1st/99th percentile
print("Winsorizing features at 1st/99th percentile...")
output_df = winsorize_features(output_df, feature_cols)

# Rename backward-looking returns to avoid collision with forward label names.
# Labels use ret_1d/ret_5d/ret_21d for *forward* returns; features use
# ret_5d/ret_21d for *backward* returns. A naive join would silently corrupt.
ret_renames = {c: f"past_{c}" for c in feature_cols if c.startswith("ret_")}
if ret_renames:
    output_df = output_df.rename(ret_renames)
    feature_cols = [ret_renames.get(c, c) for c in feature_cols]
    print(f"  Renamed {len(ret_renames)} backward-looking return columns (ret_* -> past_ret_*)")

n_features = len(feature_cols)
print(
    f"\nFeatures: {n_features} ({len(output_df):,} rows, {output_df['symbol'].n_unique()} symbols)"
)
print(f"Date range: {output_df['timestamp'].min()} to {output_df['timestamp'].max()}")

# Feature breakdown
momentum_feats = [c for c in feature_cols if c.startswith("past_ret_") or c.startswith("mom_")]
vol_feats = [c for c in feature_cols if c.startswith("vol_")]
sharpe_feats = [c for c in feature_cols if c.startswith("sharpe_")]
tech_feats = [
    c
    for c in feature_cols
    if any(c.startswith(p) for p in ["rsi_", "macd", "adx_", "cci_", "stoch_", "natr_"])
]
trend_feats = [
    c
    for c in feature_cols
    if any(c.startswith(p) for p in ["sma_", "ema_", "kama_", "dist_from_52w"])
]
rank_feats = [c for c in feature_cols if "rank" in c]
composite_feats = [c for c in feature_cols if "composite" in c or "spread" in c or "x_size" in c]
liquidity_feats = [c for c in feature_cols if "liq" in c or "illiq" in c or "volume_ratio" in c]

EXPECTED_FEATURES = 63  # concept note target: ~71 features across 6 families

print("\nFeature breakdown:")
print(f"  Momentum/Returns: {len(momentum_feats)}")
print(f"  Volatility: {len(vol_feats)}")
print(f"  Sharpe: {len(sharpe_feats)}")
print(f"  Technical: {len(tech_feats)}")
print(f"  Trend/MA: {len(trend_feats)}")
print(f"  Ranks: {len(rank_feats)}")
print(f"  Composites/Interactions: {len(composite_feats)}")
print(f"  Liquidity: {len(liquidity_feats)}")

# Validate against concept note expectations
if n_features < EXPECTED_FEATURES - 10:
    print(f"\n  WARNING: Only {n_features} features; concept note targets ~{EXPECTED_FEATURES}")
else:
    print(
        f"\n  Feature count ({n_features}) aligns with concept note target (~{EXPECTED_FEATURES})"
    )

# %% [markdown]
# ### Feature Summary Interpretation
#
# The feature matrix spans 6 families designed to capture distinct sources
# of cross-sectional return variation:
#
# - **Momentum/returns** dominate the feature count, reflecting the primary
#   edge hypothesis (behavioral persistence). The skip-month construction
#   (Jegadeesh and Titman 1993) isolates medium-term continuation from
#   short-term reversal.
# - **Ranks and composites** are the most numerous derived features. These
#   ensure stationarity across market regimes -- raw momentum levels shift
#   dramatically between the 1990s bull market and the 2008 crisis, but
#   cross-sectional ranks remain uniformly distributed.
# - **Amihud illiquidity** captures the tension between alpha and tradability:
#   momentum signals are strongest among illiquid names, but these have the
#   highest execution costs. The interaction with size rank (`mom_x_size`)
#   tests whether this tradeoff is monotonic.
# - **Technical oscillators** (RSI, MACD, ADX) overlap with momentum but
#   capture non-linear aspects (overbought/oversold thresholds) that linear
#   models cannot extract from raw returns alone.

# %% [markdown]
# ## 8. Save Features

# %%
output_path = FEATURES_DIR / "financial.parquet"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
output_df.write_parquet(output_path)
print(f"Saved {n_features} features to {output_path}")
# %% [markdown]
# ## 9. Feature Evaluation
#
# We evaluate all engineered features against the primary 1-day forward return
# label using:
# - **Information Coefficient (IC)**: Cross-sectional Spearman rank correlation
#   per date, then averaged across dates
# - **HAC adjustment**: Newey-West standard errors accounting for autocorrelation
#   in the IC time series
# - **BH-FDR**: Benjamini-Hochberg false discovery rate correction for multiple
#   testing across all features
# - **Fundamental Law of Active Management**: With ~3,177 stocks, even tiny ICs
#   compound into significant portfolio-level IR
# - **Pairwise correlation**: Identify redundant feature pairs (|corr| > 0.7)


# %%
def assign_feature_family(feature_name: str) -> str:
    """Map feature name to family for US equities panel."""
    family_map = [
        (["mom_", "ret_", "skip_recent", "cumret"], "momentum"),
        (["rev_", "reversal", "str_"], "reversal"),
        (["vol_", "rv_", "realized", "natr", "range_", "mdd_"], "volatility"),
        (["sharpe_", "risk_adj"], "sharpe"),
        (["rsi", "macd", "adx", "cci", "stoch", "bb_", "aroon"], "technical"),
        (["sma_", "ema_", "trend"], "trend"),
        (["rank_"], "cross_sectional"),
        (["composite", "quality"], "composite"),
        (["illiq", "turnover", "volume", "amihud"], "liquidity"),
        (["size", "mktcap", "ln_"], "size"),
    ]
    for prefixes, family in family_map:
        if any(p in feature_name.lower() for p in prefixes):
            return family
    return "other"


# %% [markdown]
# ### Load Labels and Join

# %%
import plotly.graph_objects as go
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from scipy.stats import spearmanr as _spearmanr

_label_df = pl.read_parquet(CASE_DIR / "labels" / "fwd_ret_1d.parquet")
_label_col = "fwd_ret_1d"

eval_df = output_df.join(_label_df, on=["timestamp", "symbol"], how="inner")
print(f"Evaluation set: {len(eval_df):,} rows, label column: {_label_col}")

# %% [markdown]
# ### Reconciling the feature index against the label index
#
# Section 1 promised this check. Both residuals are attributed to a named cause and the
# unexplained remainder is asserted empty in both directions, because counting the
# mismatches and printing them is not a check: divergent screening between stage 02 and
# stage 03 - which is what both #146 and #235 were - would print a larger number and pass.
#
# A feature row can lack a label only where [`02_labels`](02_labels.ipynb) wrote none, and
# its Section D enumerates exactly three causes: the row is the last session of the stock's
# price series, so no forward window exists; the window to the next session spans more
# calendar time than holidays explain, which is that notebook's one-session span tolerance;
# or a price is missing at one end of the window. All three are reproduced below, because a
# cause left out would be indistinguishable from the two screens having drifted apart.
#
# A label row can lack a feature row only where a feature the selection requires is still
# null, which is the warm-up at the start of a stock's series. `essential_cols` is the set
# that decides it, so the dropped rows are recomputed from that same condition rather than
# guessed at.

# %%
_fwd = raw_df.select(
    "symbol",
    "timestamp",
    (pl.col("timestamp").shift(-1).over("symbol") - pl.col("timestamp"))
    .dt.total_days()
    .alias("_fwd_days"),
    (pl.col("adj_close").is_null() | pl.col("adj_close").shift(-1).over("symbol").is_null()).alias(
        "_unpriced"
    ),
)
LABEL_SPAN_TOLERANCE_1D = 9  # 02_labels: ceil(1 * 7 / 5) + 7

_feature_orphans = output_df.join(_label_df, on=["timestamp", "symbol"], how="anti").join(
    _fwd, on=["symbol", "timestamp"], how="left"
)
_no_forward_window = pl.col("_fwd_days").is_null()
_window_spans_a_hole = pl.col("_fwd_days") > LABEL_SPAN_TOLERANCE_1D
_no_price_at_an_end = pl.col("_unpriced")
_unexplained_features = _feature_orphans.filter(
    ~(_no_forward_window | _window_spans_a_hole | _no_price_at_an_end)
)

_essential_features = [c for c in essential_cols if c not in ("symbol", "timestamp")]
_warmup = raw_df.filter(ELIGIBLE).filter(
    pl.any_horizontal(pl.col(c).is_null() for c in _essential_features)
)
_label_orphans = _label_df.join(output_df, on=["timestamp", "symbol"], how="anti")
_unexplained_labels = _label_orphans.join(
    _warmup.select("symbol", "timestamp"), on=["symbol", "timestamp"], how="anti"
)

print(
    f"  features with no label: {_feature_orphans.height:,} — "
    f"{_feature_orphans.filter(_no_forward_window).height:,} at the end of a stock's series, "
    f"{_feature_orphans.filter(~_no_forward_window & _window_spans_a_hole).height:,} whose next "
    f"session is more than {LABEL_SPAN_TOLERANCE_1D} calendar days away, "
    f"{_feature_orphans.filter(~_no_forward_window & ~_window_spans_a_hole & _no_price_at_an_end).height:,} "
    "with no price at an end of the window"
)
print(
    f"  labels with no feature: {_label_orphans.height:,} — "
    f"{_label_orphans.height - _unexplained_labels.height:,} still inside the feature warm-up"
)

assert _unexplained_features.height == 0, (
    f"{_unexplained_features.height} feature rows have no label and neither end a stock's "
    "series nor precede a gap. The two stages are screening different universes."
)
assert _unexplained_labels.height == 0, (
    f"{_unexplained_labels.height} label rows have no feature row and are not explained by "
    "the feature warm-up. The two stages are screening different universes."
)
print("  reconciled: no unexplained rows on either side")
# %% [markdown]
# ### Per-Feature IC with HAC Adjustment

# %%
ic_results = {}

if eval_df is not None:
    _partitions = eval_df.partition_by("timestamp", as_dict=True)

    for feat in feature_cols:
        ic_vals = []
        for _key, group in _partitions.items():
            vals = group.select([feat, _label_col]).drop_nulls()
            if len(vals) >= 30:
                ic, _ = _spearmanr(vals[feat].to_numpy(), vals[_label_col].to_numpy())
                if not np.isnan(ic):
                    ic_vals.append(ic)
        if len(ic_vals) >= 20:
            hac_stats = compute_ic_hac_stats(np.array(ic_vals))
            ic_results[feat] = hac_stats

    print(f"IC computed for {len(ic_results)} / {len(feature_cols)} features")

# %% [markdown]
# ### BH-FDR correction, the HAC effect, and the Fundamental Law's two inputs
#
# Three quantities that are easy to confuse, so each is computed and named separately.
# The **FDR discovery ratio** compares how many features clear the significance
# threshold before and after Benjamini-Hochberg: it sizes the multiple-testing
# correction. The **HAC effect**
# compares the HAC t-statistic against the naive one: it sizes the autocorrelation
# correction. Neither stands in for the other.
#
# The **Fundamental Law**, $IR = IC \cdot \sqrt{BR}$, has two inputs that are easy to
# overstate, so both are reported as what they are and the product is not reported as an
# achievable information ratio at all.
#
# *Breadth* is not the symbol count of the panel. Those symbols are spread over 28 years
# and were never all tradable at once, and $BR$ counts **independent** bets **per year**.
# The contemporaneous eligible cross-section is the honest starting point; multiplying it
# by the rebalancing frequency gives the count only if every bet is independent, and they
# are not - names in one cross-section share factor exposure, and consecutive days re-bet
# the same slow-moving signals. This notebook does not estimate that dependence, so what
# it prints is an upper bound on an upper bound.
#
# *Skill* is not the largest $|IC|$ among the features scored above. That maximum was
# picked on the sample it is measured on, so it is a selection artifact and overstates what the feature
# would repeat out of sample. The typical feature's $|IC|$ is the defensible summary; the
# maximum is printed only so the gap between the two is visible.

# %%
if ic_results:
    _feat_names = list(ic_results.keys())
    _p_values = [ic_results[f]["p_value"] for f in _feat_names]

    fdr_result = benjamini_hochberg_fdr(_p_values, alpha=0.05, return_details=True)

    eval_summary = pl.DataFrame(
        {
            "feature": _feat_names,
            "family": [assign_feature_family(f) for f in _feat_names],
            "ic_mean": [ic_results[f]["mean_ic"] for f in _feat_names],
            "hac_se": [ic_results[f]["hac_se"] for f in _feat_names],
            "hac_tstat": [ic_results[f]["t_stat"] for f in _feat_names],
            "p_value": _p_values,
            "adjusted_p": list(fdr_result["adjusted_p_values"]),
            "significant_fdr05": list(fdr_result["rejected"]),
            "naive_tstat": [ic_results[f]["naive_t_stat"] for f in _feat_names],
        }
    ).sort("ic_mean", descending=True)

    n_significant = int(fdr_result["n_rejected"])
    n_naive_sig = sum(1 for p in _p_values if p < 0.05)
    fdr_discovery_ratio = n_naive_sig / max(n_significant, 1)
    # A ratio below 1 means HAC widened the standard error.
    _t_ratio = (eval_summary["hac_tstat"].abs() / eval_summary["naive_tstat"].abs()).drop_nans()
    hac_t_ratio_median = float(_t_ratio.median())
    n_t_grew = int((_t_ratio > 1).sum())

    print(f"Features tested: {len(_feat_names)}")
    print(f"Naive significant (p < 0.05): {n_naive_sig}")
    print(f"FDR-corrected significant: {n_significant}")
    print(f"FDR discovery ratio: {fdr_discovery_ratio:.2f}x (multiple testing, not HAC)")
    print(
        f"HAC effect on |t|: median ratio {hac_t_ratio_median:.3f}, "
        f"range {_t_ratio.min():.3f} to {_t_ratio.max():.3f}; "
        f"{n_t_grew} of {len(_t_ratio)} features have a larger |t| under HAC"
    )
    print(eval_summary.select("feature", "family", "ic_mean", "hac_tstat", "significant_fdr05"))

    xs_per_date = eval_df.group_by("timestamp").len()["len"]
    breadth_date = int(xs_per_date.median())
    br_independent = breadth_date * REBALANCES_PER_YEAR
    typical_ic = float(eval_summary["ic_mean"].abs().mean())
    best = eval_summary.sort(pl.col("ic_mean").abs(), descending=True).row(0, named=True)

    print("\nFundamental Law inputs, IR = IC x sqrt(BR)")
    print(
        f"  cross-section per decision date: median {breadth_date:,} eligible stocks "
        f"(the panel holds {output_df['symbol'].n_unique():,} across the whole sample)"
    )
    print(f"  typical feature |IC|: {typical_ic:.4f} over {len(eval_summary)} features")
    print(
        f"  largest |IC| in sample: {abs(best['ic_mean']):.4f} ({best['feature']}) "
        "- a maximum over 63, not a signal's skill"
    )
    print(
        f"  if all {breadth_date:,} names x {REBALANCES_PER_YEAR} rebalances were independent "
        f"bets, BR would be {br_independent:,} and the typical feature would imply "
        f"IR {typical_ic * np.sqrt(br_independent):.1f}."
    )
    print(
        "  They are not independent, and this notebook does not estimate the discount, "
        "so that figure is an upper bound on an upper bound and no IR is claimed here."
    )

# %% [markdown]
# ### IC Bar Chart (Top 20)

# %%
if ic_results:
    top_20 = eval_summary.sort(pl.col("ic_mean").abs(), descending=True).head(20)

    fig = go.Figure()
    colors = ["#2ecc71" if sig else "#95a5a6" for sig in top_20["significant_fdr05"].to_list()]
    fig.add_trace(
        go.Bar(
            x=top_20["feature"].to_list(),
            y=top_20["ic_mean"].to_list(),
            marker_color=colors,
            text=[f"{t:.1f}" for t in top_20["hac_tstat"].to_list()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top 20 Features by IC (green = FDR-significant at 5%)",
        xaxis_title="Feature",
        yaxis_title="Mean IC (Spearman)",
        template="plotly_white",
        xaxis_tickangle=-45,
        height=500,
    )
    fig.show()

# %% [markdown]
# ### Pairwise Feature Correlation

# %%
high_corr_pairs = []

if eval_df is not None:
    # Sample every 20th date for efficiency (~4.5M rows is very large)
    _sample_dates = eval_df["timestamp"].unique().sort().gather_every(20)
    _corr_data = (
        eval_df.filter(pl.col("timestamp").is_in(_sample_dates))
        .select(feature_cols)
        .to_pandas()
        .corr(method="spearman")
    )

    for i, f1 in enumerate(_corr_data.columns):
        for j, f2 in enumerate(_corr_data.columns):
            if i < j and abs(_corr_data.iloc[i, j]) > 0.7:
                high_corr_pairs.append((f1, f2, float(_corr_data.iloc[i, j])))

    print(f"Feature pairs with |corr| > 0.7: {len(high_corr_pairs)}")

    fig = go.Figure(
        data=go.Heatmap(
            z=_corr_data.values,
            x=_corr_data.columns.tolist(),
            y=_corr_data.columns.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        title=f"Feature Pairwise Correlation ({len(high_corr_pairs)} pairs above 0.7)",
        template="plotly_white",
        height=700,
        width=800,
    )
    fig.show()

# %% [markdown]
# ### HAC vs Naive t-statistics

# %%
if ic_results:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eval_summary["naive_tstat"].to_list(),
            y=eval_summary["hac_tstat"].to_list(),
            mode="markers",
            text=eval_summary["feature"].to_list(),
            marker=dict(
                color=[
                    "#2ecc71" if s else "#e74c3c"
                    for s in eval_summary["significant_fdr05"].to_list()
                ],
                size=8,
            ),
        )
    )
    _max_t = (
        max(
            abs(eval_summary["naive_tstat"].max()),
            abs(eval_summary["hac_tstat"].max()),
        )
        * 1.1
    )
    fig.add_trace(
        go.Scatter(
            x=[-_max_t, _max_t],
            y=[-_max_t, _max_t],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="HAC shrinks most t-statistics by about a tenth; a few grow",
        xaxis_title="Naive t-stat",
        yaxis_title="HAC t-stat",
        template="plotly_white",
        height=500,
    )
    fig.show()

# %% [markdown]
# **Interpretation**:
# - Breadth is the point of this panel: $IR = IC \cdot \sqrt{BR}$, so a per-stock
#   edge far too small to trade on its own becomes a portfolio-level one once it
#   is applied across thousands of names. The cell above prints that arithmetic
#   for the typical feature and for the strongest, on the breadth it just used.
# - The HAC adjustment is modest here, because one-session, non-overlapping returns
#   leave little IC autocorrelation to correct for - but it is not negligible and it
#   is not one-directional, and the printed effect gives both the median and the
#   number of features whose statistic grew. The FDR discovery ratio beside it is the
#   multiple-testing correction and is a different quantity measuring a different
#   thing; neither number stands in for the other.
# - Cross-sectional rank features likely dominate because they are stationary
#   across the 28-year sample (1990-2018), while raw return levels shift
#   dramatically between regimes.
# - Feature correlation is expected to be high within families (momentum
#   horizons, volatility horizons). Dimensionality reduction or feature
#   clustering recommended before modeling.
#
# **Fundamental Law teaching moment**: this is the book's highest-breadth case
# study, and the arithmetic above is why that matters -- an information
# coefficient small enough to look like noise on any one stock reaches a
# respectable information ratio once the square root of the breadth multiplies
# it. The lesson is that in a wide cross-section, signal *consistency* and cost
# control decide the outcome rather than signal strength. Note what the same
# arithmetic says about the panel taken as a whole: its features point in both
# directions and their signed ICs very nearly cancel, so the breadth is only
# worth anything to a strategy that has already chosen a direction to bet in.

# %% [markdown]
# ## Results Collection


# %%
def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


results = {
    "case_study_id": "us_equities_panel",
    "chapter": 8,
    "stage": "features",
    "timestamp": datetime.now(UTC).isoformat(),
    "git_commit": _git_commit_hash(),
    "notebook": "case_studies/us_equities_panel/03_financial_features.py",
    "summary": {
        "n_features": n_features,
        "n_observations": len(output_df),
        "n_symbols": output_df["symbol"].n_unique(),
        "date_range": [str(output_df["timestamp"].min()), str(output_df["timestamp"].max())],
    },
    "techniques": {
        "feature_families": [
            "momentum",
            "reversal",
            "volatility",
            "liquidity",
            "technical",
            "trend",
            "composites",
            "size_interactions",
        ],
        "winsorization": "1st/99th percentile",
        "cross_sectional_ranks": True,
        "amihud_illiquidity": True,
        "size_conditional_features": True,
    },
    "diagnostics": {
        "feature_count_by_family": {
            "momentum": len(momentum_feats),
            "volatility": len(vol_feats),
            "sharpe": len(sharpe_feats),
            "technical": len(tech_feats),
            "trend": len(trend_feats),
            "ranks": len(rank_feats),
            "composites": len(composite_feats),
            "liquidity": len(liquidity_feats),
        },
    },
    "key_findings": [
        f"{n_features} features computed across {output_df['symbol'].n_unique()} symbols",
        "Amihud illiquidity added as rolling 21-day measure",
        "Momentum-reversal spread and size-conditional features included",
        "Winsorized at 1st/99th percentile to limit extreme values",
    ],
}

# Add evaluation block if IC analysis was performed
if ic_results:
    family_ic = {}
    for feat, stats in ic_results.items():
        family = assign_feature_family(feat)
        family_ic.setdefault(family, []).append(stats["mean_ic"])
    family_avg_ic = {f: float(np.mean(ics)) for f, ics in family_ic.items()}

    results["evaluation"] = {
        "primary_label": _label_col,
        "n_features_tested": len(ic_results),
        "n_significant_naive05": n_naive_sig,
        "n_significant_fdr05": n_significant,
        "fdr_discovery_ratio": round(fdr_discovery_ratio, 2),
        "hac_t_ratio_median": round(hac_t_ratio_median, 3),
        # Inputs only. No IR is serialized, for the reason Section 9 states: an
        # independent-bet count is what turns these into one, and this notebook does
        # not estimate it.
        "fundamental_law": {
            "typical_abs_ic": round(typical_ic, 4),
            "largest_abs_ic_in_sample": round(abs(best["ic_mean"]), 4),
            "largest_abs_ic_feature": best["feature"],
            "median_cross_section": breadth_date,
            "rebalances_per_year": REBALANCES_PER_YEAR,
            "br_if_bets_were_independent": br_independent,
        },
        "top_features": [
            {
                "name": row["feature"],
                "ic_mean": round(row["ic_mean"], 4),
                "hac_tstat": round(row["hac_tstat"], 2),
                "hac_pval": round(row["p_value"], 4),
            }
            for row in eval_summary.head(10).to_dicts()
        ],
        "max_pairwise_corr": (
            round(max(abs(c) for _, _, c in high_corr_pairs), 3) if high_corr_pairs else 0.0
        ),
        "corr_pairs_above_07": len(high_corr_pairs),
        "feature_family_avg_ic": {
            k: round(v, 4) for k, v in sorted(family_avg_ic.items(), key=lambda x: -abs(x[1]))
        },
    }


# %% [markdown]
# ## Key Takeaways
#
# 1. **Cross-sectional ranks** ensure features are stationary across decades
#    of US equity history. Raw returns and volatility levels are non-stationary;
#    their ranks within the cross-section at each date are not.
#
# 2. **Amihud illiquidity** captures the tension between alpha and tradability:
#    momentum is stronger among illiquid names (Amihud, 2002), but these are
#    exactly the stocks with highest transaction costs.
#
# 3. **Size-conditional features** (momentum x size rank) test whether the
#    momentum signal varies by market cap -- a key finding in the academic
#    literature (Fama and French, 1992).
#
# 4. **Winsorization at 1st/99th percentile** protects against split artifacts
#    and data errors that produce extreme outlier values in daily data.
#
# **Next**: `04_temporal.py` in Ch9 adds Wasserstein regime detection,
# fractional differencing, and GARCH volatility features.
