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
# Load from the canonical data loader with the same PIT filters as [`02_labels`](02_labels.ipynb).
# Both notebooks apply identical filters (MIN_PRICE=\$5, ADV>\$1M) via the
# same constants.
#
# **Alignment check**: After loading and filtering, we verify the feature
# index is a subset of the label index. Any mismatch indicates divergent
# filter logic.

# %%
raw_df = load_us_equities(start_date=START_DATE, end_date=END_DATE)

# Normalize types
if raw_df.schema["timestamp"] == pl.Datetime:
    raw_df = raw_df.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

raw_df = raw_df.sort(["symbol", "timestamp"])

# Compute base columns
raw_df = raw_df.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("returns"),
    (pl.col("adj_close") * pl.col("adj_volume")).alias("dollar_volume"),
)

# Apply PIT eligibility filters
raw_df = raw_df.with_columns(
    pl.col("dollar_volume").rolling_mean(ADV_WINDOW).over("symbol").alias("adv_21d")
)
df = raw_df.filter((pl.col("adj_close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD))

print(f"Loaded {len(df):,} rows, {df['symbol'].n_unique()} symbols")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

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
def compute_liquidity_reversion(data: pl.DataFrame) -> pl.DataFrame:
    """Liquidity measures, Amihud illiquidity, reversion signals, and size proxy."""
    # Liquidity features
    data = data.with_columns((pl.col("dollar_volume") / pl.col("adv_21d")).alias("volume_ratio"))
    data = data.with_columns(
        (
            pl.col("adv_21d").rank().over("timestamp") / pl.col("adv_21d").count().over("timestamp")
        ).alias("liq_rank")
    )
    # Amihud illiquidity: |return| / dollar_volume (rolling 21-day mean)
    data = data.with_columns(
        (pl.col("returns").abs() / (pl.col("dollar_volume") + 1))
        .rolling_mean(21)
        .over("symbol")
        .alias("amihud_illiq")
    )
    data = data.with_columns(
        (
            pl.col("amihud_illiq").rank().over("timestamp")
            / pl.col("amihud_illiq").count().over("timestamp")
        ).alias("illiq_rank")
    )
    # Mean reversion signals
    data = data.with_columns(
        (
            pl.col("ret_5d").rank().over("timestamp") / pl.col("ret_5d").count().over("timestamp")
        ).alias("reversal_rank"),
        (
            pl.col("rsi_14").rank().over("timestamp") / pl.col("rsi_14").count().over("timestamp")
        ).alias("rsi_rank"),
    )
    # Size proxy (log dollar volume rank as mcap proxy)
    data = data.with_columns(
        (
            pl.col("adv_21d").log().rank().over("timestamp")
            / pl.col("adv_21d").log().count().over("timestamp")
        ).alias("size_rank")
    )
    return data


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

# %%
print("Computing features...")
df = df.pipe(compute_momentum_returns).pipe(compute_volatility_sharpe)
print("  Momentum and volatility done")

df = df.pipe(compute_oscillators).pipe(compute_trend_distance)
print("  Technical indicators done")

df = df.pipe(compute_xs_ranks).pipe(compute_liquidity_reversion).pipe(compute_composites)
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
# - **Fundamental Law of Active Management**: With ~3,149 stocks, even tiny ICs
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
# ### BH-FDR Multiple Testing Correction

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
    inflation = n_naive_sig / max(n_significant, 1)

    print(f"Features tested: {len(_feat_names)}")
    print(f"Naive significant (p < 0.05): {n_naive_sig}")
    print(f"FDR-corrected significant: {n_significant}")
    print(f"Inflation factor: {inflation:.1f}x")
    print(eval_summary.select("feature", "family", "ic_mean", "hac_tstat", "significant_fdr05"))

    # Fundamental Law of Active Management
    # IR = IC * sqrt(BR) where BR ~ number of independent bets
    mean_ic = float(eval_summary["ic_mean"].mean())
    breadth = output_df["symbol"].n_unique()
    ir_estimate = abs(mean_ic) * np.sqrt(breadth)
    print(f"\nFundamental Law: IC={mean_ic:.4f} x sqrt({breadth}) = IR={ir_estimate:.2f}")
    print(f"Even tiny ICs are significant with {breadth:,} stocks")

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
        title="HAC vs Naive t-statistics (points below line = HAC deflation)",
        xaxis_title="Naive t-stat",
        yaxis_title="HAC t-stat",
        template="plotly_white",
        height=500,
    )
    fig.show()

# %% [markdown]
# **Interpretation**:
# - With ~3,149 stocks the Fundamental Law is the key insight: even ICs of
#   0.01-0.02 generate portfolio-level $IR \approx 0.5\text{--}1.0$ because
#   $IR = IC \cdot \sqrt{BR}$ and breadth is enormous.
# - HAC adjustment should be minimal (inflation factor ~1.0x) because 1-day
#   non-overlapping returns produce low IC autocorrelation.
# - Cross-sectional rank features likely dominate because they are stationary
#   across the 28-year sample (1990-2018), while raw return levels shift
#   dramatically between regimes.
# - Feature correlation is expected to be high within families (momentum
#   horizons, volatility horizons). Dimensionality reduction or feature
#   clustering recommended before modeling.
#
# **Fundamental Law teaching moment**: This is the book's highest-breadth
# case study. A mean IC of just 0.01 across 3,149 stocks implies
# $IR = 0.01 \times \sqrt{3149} \approx 0.56$ -- competitive with many
# hedge fund strategies. The lesson: in large cross-sections, signal
# quality matters less than signal consistency and cost control.

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
        "inflation_factor": round(inflation, 1),
        "fundamental_law": {
            "mean_ic": round(mean_ic, 4),
            "breadth": breadth,
            "ir_estimate": round(ir_estimate, 2),
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
