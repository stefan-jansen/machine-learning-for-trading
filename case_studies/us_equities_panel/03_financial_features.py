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
# The panel is the book's widest cross-section, so what a feature has to do here is rank
# thousands of stocks against each other rather than time any one of them. This notebook
# builds eight families of price-derived features on that panel, applies the eligibility
# screen where it cannot change the window of a row it keeps, clips each feature against
# the cross-section it was measured in, and scores every one of them against the primary
# label on the development window alone.
#
# ## Learning objectives
#
# - Order per-symbol windows, the eligibility screen and cross-sectional ranks so each is
#   computed on the frame that gives it its meaning
# - Winsorize against the cross-section a feature is ranked in, rather than against a
#   sample the strategy has not lived through yet
# - Seal a feature evaluation on the label's endpoint, so the holdout scores nothing here
# - Separate the multiple-testing correction from the autocorrelation correction, and read
#   the Fundamental Law's two inputs as the bounds they are
#
# ## Book reference, prerequisites and artifacts
#
# Chapter 8, Section 8.2. Reads the adjusted daily panel through `load_us_equities()`,
# `config/setup.yaml` for the holdout boundary, and `labels/fwd_ret_1d.parquet` written by
# [`02_labels`](02_labels.ipynb). Writes `features/financial.parquet`.
# [`04_model_based_features`](04_model_based_features.ipynb) writes a second matrix beside this
# one and does not read it; `utils/modeling.py::load_modeling_dataset` joins the two when a
# model stage loads the dataset.

# %%
"""US Equities Panel: Feature Engineering."""

import warnings
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import yaml
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr
from ml4t.diagnostic.metrics import compute_ic_hac_stats
from ml4t.diagnostic.splitters.calendar import TradingCalendar
from ml4t.engineer.features.momentum import adx, cci, macd, rsi, stochastic
from ml4t.engineer.features.trend import ema, kama, sma
from ml4t.engineer.features.volatility import natr

from case_studies.utils.artifact_digest import read_digest, value_digest, write_artifact
from data import load_us_equities
from utils.artifact_specs import resolve_label_horizon
from utils.paths import display_path, get_case_study_dir
from utils.style import (
    COLORS,
    FIGSIZE,
    add_message_title,
    ml4t_diverging,
    show_plotly_with_alt,
    show_with_alt,
)

CASE_DIR = get_case_study_dir("us_equities_panel")
FEATURES_DIR = CASE_DIR / "features"

# Feature horizons, in trading sessions. These define the features rather than the
# strategy, so they are declared here; everything that defines the strategy is bound
# from setup.yaml below.
MOMENTUM_HORIZONS = [5, 10, 21, 42, 63, 126, 189, 252]
VOLATILITY_HORIZONS = [21, 63, 126, 252]
MA_HORIZONS = [10, 20, 50, 100, 200]
# 12-1 momentum, the case study's declared treatment. `02_labels` Section G carries the same
# two numbers and scores this construction as the baseline.
MOMENTUM_LOOKBACK, MOMENTUM_SKIP = 252, 21

# The tradability screen, declared in Section 1 and applied in Section 6.
# `02_labels` carries the same three constants and rebuilds the screen from them.
MIN_PRICE, MIN_ADV_USD, ADV_WINDOW = 5.0, 1_000_000, 21

# Winsorization and redundancy thresholds, named once so prose and code cannot drift.
WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99
REDUNDANT_CORR = 0.7
FDR_ALPHA = 0.05

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
START_DATE = "1990-01-01"

# %% [markdown]
# ## Configuration
#
# The holdout boundary and the primary label come from `config/setup.yaml`, which is also
# what [`02_labels`](02_labels.ipynb) read to write the label file this notebook scores
# against. The decision cadence fixes the rebalance count the Fundamental Law's breadth
# input is multiplied by: the primary label is a one-session forward return, so a strategy
# trading it decides at every close.

# %%
SETUP = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

PRIMARY_LABEL = SETUP["labels"]["primary"]
# The horizon the primary label resolves over, in trading sessions, read from the same place
# `02_labels` reads it. Section 9 stops on it and corrects the IC standard error for it, so a
# hardcoded 1 here would silently misstate both if the primary label were ever changed.
PRIMARY_HORIZON = int(resolve_label_horizon(CASE_STUDY_ID, PRIMARY_LABEL, SETUP).rstrip("Dd"))
HOLDOUT_START = date.fromisoformat(SETUP["evaluation"]["holdout_start"])
END_DATE = str(SETUP["evaluation"]["holdout_end"])
REBALANCES_PER_YEAR = SETUP["evaluation"]["periods_per_year"]
CALENDAR = SETUP["evaluation"]["calendar"]

print(f"Primary label {PRIMARY_LABEL}, holdout opens {HOLDOUT_START}, panel ends {END_DATE}")
print(
    f"Screen: printed close over ${MIN_PRICE:.0f}, {ADV_WINDOW}-session ADV over ${MIN_ADV_USD:,}"
)

# %% [markdown]
# ## Connecting to the edge hypothesis
#
# [`01_feasibility_analysis`](01_feasibility_analysis.ipynb) found no time-series signal in
# the panel worth trading - only the one-session lag clears its band, and it is a weak
# reversal - so whatever edge exists has to come from the cross-section. That is the
# hypothesis the features below operationalize: slow information diffusion across thousands
# of stocks, read as a ranking rather than as a forecast for any one name.
#
# The Fundamental Law of Active Management, $IR = IC \cdot \sqrt{BR}$, is why a per-stock
# correlation far too small to trade on its own is worth building here: breadth multiplies
# it. Section 9 computes both of its inputs and reports what each of them actually bounds.

# %% [markdown]
# ## 1. Load Data
#
# Load from the canonical data loader and declare the same eligibility screen as
# [`02_labels`](02_labels.ipynb): a printed close above \$5, and dollar volume
# `close * volume` averaging above \$1M over 21 sessions. Both legs
# read figures the tape carried on the day, so neither depends on a corporate
# action that had not happened yet. Section B of [`02_labels`](02_labels.ipynb)
# derives why the adjusted close cannot serve here. Both notebooks rebuild the
# screen from the same three constants on the same columns, so the trainable
# panel and the label files agree on the universe.
#
# **Sessions are numbered first, as they are in [`02_labels`](02_labels.ipynb) and
# [`01_feasibility_analysis`](01_feasibility_analysis.ipynb).** The archive carries stray
# prints on dates the exchange held no market, and `get_sessions` identifies them: a date
# that maps to itself is a session, a stray print maps to a neighbour. Dropping them and
# numbering what is left gives a counter whose difference between two rows is a count of
# sessions. Two things in this notebook need it. The turnover leg of the screen is only
# meaningful over an unbroken window, so a stock returning from a halt cannot qualify on the
# volume it traded before the halt. And Section 9 reconciles this stage's index against the
# label index, which was built on that same counter - a feature row sitting on a date no
# label file can carry would otherwise show up there as two stages disagreeing.
#
# **Declared here, applied in Section 6.** The screen removes whole rows, and a
# per-symbol shift or rolling window applied afterwards counts the rows that
# survived rather than trading sessions. Section 6 states what that costs and
# applies the screen between the per-symbol features and the cross-sectional
# ones, which is the only ordering that gives both their intended meaning. Its third leg,
# `adv_covered`, is the coverage condition [`01_feasibility_analysis`](01_feasibility_analysis.ipynb)
# and [`02_labels`](02_labels.ipynb) also apply: the 21 rows the average runs over have to be
# the 21 consecutive sessions ending on the row, or the average describes a stretch of calendar
# the stock was not trading through.
#
# The digest printed below is the panel this stage read, taken over the same five columns
# [`02_labels`](02_labels.ipynb) digests. The two stages screen the same universe only if they
# read the same download, and printing it here is what makes that checkable: it has to equal the
# `market_data` digest in the label sidecars, so the assertions in Section 9 are reconciling two
# files rather than one file against a stale copy of another.
#
# Returns and every price-derived feature below still read `adj_close`: a return
# has to divide out splits and dividends to mean anything.
#
# **Alignment check**: the two indices are not expected to match exactly - a label
# needs a forward window the last session of a stock's series does not have, and a
# feature needs a warm-up the first sessions do not have. Section 9 attributes
# every row on both sides to one of those causes and **asserts the remainder is
# empty**, so a screen that drifted apart between the two stages fails there rather
# than printing a larger number and passing. That check is only as good as the two
# stages sharing one definition of a session, which is why the counter below is
# built the same way in both.

# %%
raw_df = load_us_equities(start_date=START_DATE, end_date=END_DATE)

# Normalize types
if raw_df.schema["timestamp"] == pl.Datetime:
    raw_df = raw_df.with_columns(pl.col("timestamp").dt.date().alias("timestamp"))

raw_df = raw_df.sort(["symbol", "timestamp"])

MARKET_DATA_DIGEST = value_digest(raw_df, ["symbol", "timestamp", "close", "volume", "adj_close"])
LABEL_INPUT_DIGEST = read_digest(CASE_DIR / "labels" / f"{PRIMARY_LABEL}.parquet")["inputs"][
    "market_data"
]
print(f"market_data digest: {MARKET_DATA_DIGEST}")
assert MARKET_DATA_DIGEST == LABEL_INPUT_DIGEST, (
    f"the labels were written against market_data {LABEL_INPUT_DIGEST} and this stage read "
    f"{MARKET_DATA_DIGEST}. Re-run 02_labels before scoring features against its output."
)

# The session counter, built exactly as `02_labels` builds it: the dates that map to
# themselves under the exchange calendar, numbered in order. The join drops the stray prints,
# so no row here sits on a date a label file cannot carry.
dates = raw_df.select("timestamp").unique().sort("timestamp")
settling_session = pl.Series(
    TradingCalendar(CALENDAR)
    .get_sessions(pd.DatetimeIndex(dates["timestamp"].to_list(), tz="UTC"))
    .to_numpy()
).cast(pl.Date)
sessions = (
    dates.filter(settling_session == pl.col("timestamp"))
    .with_row_index("session")
    .with_columns(pl.col("session").cast(pl.Int64))
)
_archive_rows = raw_df.height
raw_df = raw_df.join(sessions, on="timestamp", how="inner").sort(["symbol", "timestamp"])
print(
    f"{sessions.height:,} of {dates.height:,} dates in the archive are {CALENDAR} sessions; "
    f"the other {dates.height - sessions.height} carry stray prints and take "
    f"{_archive_rows - raw_df.height:,} rows with them"
)

# Compute base columns
raw_df = raw_df.with_columns(
    (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("returns"),
    (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
)

raw_df = raw_df.with_columns(
    pl.col("dollar_volume").rolling_mean(ADV_WINDOW).over("symbol").alias("adv_21d"),
    (pl.col("session") - pl.col("session").shift(ADV_WINDOW - 1) == ADV_WINDOW - 1)
    .over("symbol")
    .alias("adv_covered"),
)

# Declared here, applied in Section 6.
ELIGIBLE = pl.col("adv_covered") & (pl.col("close") > MIN_PRICE) & (pl.col("adv_21d") > MIN_ADV_USD)

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
#
# Its two lookbacks are read off the **session counter**, not off the stock's own rows, and
# it is the one feature here built that way. `setup.yaml` declares this construction as the
# case study's causal treatment, and [`02_labels`](02_labels.ipynb) Section G measures what it
# earns as the baseline every feature below has to beat. A row-counted version would reach
# past twelve months in any stock that missed a session, so the column the models consume
# would not be the quantity the baseline scored - the two have to be the same signal for the
# comparison to mean anything.
#
# The other windows on this page - the raw return horizons, the rolling volatilities, and the
# library oscillators - still count the stock's own rows. What that costs is bounded and
# one-directional: a stock that missed sessions inside the window has its feature measured
# over a slightly longer stretch of calendar than the name says, which widens the window
# rather than shifting it forward, and none of them is a quantity another stage recomputes
# independently. Rewriting the library indicators to a dense session grid is a change to the
# feature definitions rather than a correction to them, so it is not made here.


# %%
def close_sessions_back(data: pl.DataFrame, lag: int, name: str) -> pl.DataFrame:
    """Each stock's close `lag` sessions earlier, re-keyed to the session reading it."""
    return data.select(
        "symbol", (pl.col("session") + lag).alias("session"), pl.col("adj_close").alias(name)
    )


def compute_momentum_returns(data: pl.DataFrame) -> pl.DataFrame:
    """Multi-horizon raw returns and skip-month momentum."""
    data = data.sort(["symbol", "timestamp"])
    for h in MOMENTUM_HORIZONS:
        data = data.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(h).over("symbol") - 1).alias(
                f"ret_{h}d"
            )
        )
    # Skip-month momentum (12-1): Jegadeesh-Titman (1993) construction, the return from
    # t-252 to t-21, both counted on the market's session list as `02_labels` counts them.
    data = (
        data.join(
            close_sessions_back(data, MOMENTUM_SKIP, "_skip_close"),
            on=["symbol", "session"],
            how="left",
        )
        .join(
            close_sessions_back(data, MOMENTUM_LOOKBACK, "_start_close"),
            on=["symbol", "session"],
            how="left",
        )
        .with_columns(
            (pl.col("_skip_close") / pl.col("_start_close").clip(lower_bound=1e-8) - 1).alias(
                "ret_12m_skip"
            )
        )
        .drop("_skip_close", "_start_close")
        .sort(["symbol", "timestamp"])
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
# A raw feature level carries whatever the whole panel was doing that day, so its
# distribution moves with the regime; its rank within that day's cross-section is
# bounded in the unit interval whatever the regime is, which is what makes one
# model coefficient mean the same thing in 1997 and in 2009.
#
# The Amihud (2002) illiquidity measure captures price impact per unit of
# trading volume:
#
# $$\text{ILLIQ}_{i,t} = \frac{1}{D} \sum_{d=1}^{D} \frac{|r_{i,d}|}{\text{DVOL}_{i,d}}$$
#
# where $D = 21$ days, $r_{i,d}$ is the daily return, and $\text{DVOL}_{i,d}$
# is dollar volume. Higher values indicate less liquid stocks. Amihud (2002)
# showed that expected illiquidity positively predicts cross-sectional returns.


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
    )


# %% [markdown]
# ### Composite and Interaction Features
#
# Composites blend related ranks into single signals. The momentum-reversal
# spread exploits the negative correlation between trend and reversion
# signals. The interactions multiply a momentum rank by the dollar-volume
# rank, so a model can let momentum act differently on the names that trade
# and on the ones that barely do.
#
# The panel carries prices and share volume and no shares outstanding, so it
# has no market capitalization and this is a liquidity interaction rather than
# the size-conditional momentum of Fama and French (1992). Taking the logarithm
# of dollar volume before ranking does not recover one: a rank is invariant
# under any increasing transform, so `rank(log(ADV))` and `rank(ADV)` are the
# same column to the last bit.


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
    # Liquidity-conditional momentum
    data = data.with_columns(
        (pl.col("mom_rank_126d") * pl.col("liq_rank")).alias("mom_x_liq"),
        (pl.col("mom_rank_252d") * pl.col("liq_rank")).alias("mom12m_x_liq"),
    )
    return data


# %% [markdown]
# ## 5. Winsorization
#
# A split the vendor did not adjust, or a price of a fraction of a cent, enters a return
# feature as a move of thousands of percent. Clipping each feature at its first and
# ninety-ninth percentile keeps those rows in the panel without letting them set the scale
# every other value is measured against.
#
# **The percentiles are taken within each cross-section, not over the whole sample.** A
# bound estimated over every date is estimated partly on dates the strategy has not reached
# yet, including the holdout, and it is then applied to rows that precede them - the
# clip a stock receives in 1994 would depend on what the panel did in 2017. It is also the
# wrong bound: the width of a daily cross-section is not a constant of the panel.
# [`02_labels`](02_labels.ipynb) measured that directly, and cross-sectional dispersion more
# than doubles between its quietest and loudest year, so one pair of bounds clips almost
# nothing in a crisis and cuts into the body of the distribution in a calm decade. The
# figure below shows the second effect on the primary momentum feature.
#
# It shows it on the development window alone. The flat pair it draws is the counterfactual,
# and computing the true sample-wide one is the very read the holdout boundary forbids, so the
# counterfactual is estimated on development rows only - which understates the case rather
# than overstating it, since the sample-wide pair would additionally be wrong by whatever
# the two holdout years did. The clip itself still runs on every row the file carries: a
# per-date quantile reads nothing but the date it is applied to, so it crosses no boundary.
#
# Taking the percentiles per date fixes the leak and the mis-scaling with the same
# expression, and it is the same frame the cross-sectional ranks are already computed in.


# %%
def winsorize_features(
    data: pl.DataFrame,
    feature_cols: list[str],
    lower: float = WINSOR_LOWER,
    upper: float = WINSOR_UPPER,
) -> pl.DataFrame:
    """Clip each feature at per-date quantiles, so no bound crosses a decision date."""
    present = [c for c in feature_cols if c in data.columns]
    return data.with_columns(
        pl.col(col)
        .clip(
            pl.col(col).quantile(lower).over("timestamp"),
            pl.col(col).quantile(upper).over("timestamp"),
        )
        .alias(col)
        for col in present
    )


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
print(
    f"  {raw_df.filter(~pl.col('adv_covered').fill_null(False)).height:,} of those rows carry no "
    f"unbroken {ADV_WINDOW}-session volume window and cannot be screened on turnover at all"
)

df = df.pipe(compute_xs_ranks).pipe(compute_xs_liquidity_reversion).pipe(compute_composites)
print("  Cross-sectional ranks and composites done")

# %% [markdown]
# ## 7. Select and Clean Features
#
# Two things happen here that a row count would hide.
#
# **A missing value is a null, and several of the library oscillators return a float NaN
# instead** - an efficiency ratio that divides by zero on a flat window, a true range that does
# the same. Polars treats the two as different things: `drop_nulls` keeps a NaN, and every
# summary it reaches returns NaN rather than skipping the row. The consequence is not a dropped
# row but a dropped *date*: `spearmanr` propagates NaN, so a single stock carrying one poisons
# that feature's correlation for the whole cross-section it sits in, and the feature is then
# scored on whatever dates happen to be left. Converting NaN to null once, here, is what makes
# the rest of the notebook's null handling apply to them.
#
# **The winsorization bounds are measured before the clip is applied**, so the figure below can
# show what a single flat pair would have done to each cross-section it was applied to. Both are
# read from development rows only - see the section note below on why the counterfactual is not
# the true sample-wide pair.

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
    "adv_covered",
    "session",
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

# `df` carried the price columns and the helpers alongside the features and is superseded here.
# On the full panel it is about eight gigabytes, and Section 9 partitions the evaluation frame by
# date while every frame still referenced stays resident - which is where this notebook peaks.
del df

_nan_counts = {
    c: int(output_df[c].is_nan().sum())
    for c in feature_cols
    if output_df.schema[c] in (pl.Float32, pl.Float64)
}
_nan_columns = {c: n for c, n in _nan_counts.items() if n}
output_df = output_df.with_columns(
    pl.col(c).fill_nan(None)
    for c in feature_cols
    if output_df.schema[c] in (pl.Float32, pl.Float64)
)
print(
    f"NaN converted to null in {len(_nan_columns)} of {len(_nan_counts)} float features, "
    f"{sum(_nan_columns.values()):,} values: {sorted(_nan_columns)}"
)

WINSOR_EXAMPLE = "ret_21d"
_winsor_dev = output_df.filter(pl.col("timestamp") < HOLDOUT_START)
per_date_bounds = (
    _winsor_dev.group_by("timestamp")
    .agg(
        pl.col(WINSOR_EXAMPLE).quantile(WINSOR_LOWER).alias("lower"),
        pl.col(WINSOR_EXAMPLE).quantile(WINSOR_UPPER).alias("upper"),
    )
    .sort("timestamp")
)
flat_lower = _winsor_dev[WINSOR_EXAMPLE].quantile(WINSOR_LOWER)
flat_upper = _winsor_dev[WINSOR_EXAMPLE].quantile(WINSOR_UPPER)

print("Winsorizing each feature against its own cross-section...")
output_df = winsorize_features(output_df, feature_cols)

# %% [markdown]
# The two lines are the percentiles the clip uses on each date; the flat pair is what one
# estimate over all of them would have applied to every one of them. Where the flat bound
# sits outside the daily pair it clips nothing, and where it sits inside it cuts into the
# body of that day's cross-section. The gap is a regime effect, not noise: it tracks the
# crises, and the print below gives the narrowest and widest daily pair against the flat one.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
ax.plot(per_date_bounds["timestamp"], per_date_bounds["upper"], color=COLORS["blue"], lw=0.7)
ax.plot(per_date_bounds["timestamp"], per_date_bounds["lower"], color=COLORS["blue"], lw=0.7)
ax.fill_between(
    per_date_bounds["timestamp"],
    per_date_bounds["lower"],
    per_date_bounds["upper"],
    color=COLORS["blue"],
    alpha=0.15,
)
for bound in (flat_lower, flat_upper):
    ax.axhline(bound, color=COLORS["copper"], ls="--", lw=1.4)
ax.axhline(0, color=COLORS["neutral"], lw=0.6)
ax.set_xlabel("Date")
ax.set_ylabel(f"Clip bounds on {WINSOR_EXAMPLE}")
add_message_title(
    ax,
    "One flat clip bound is too wide in calm years and too tight in crises",
    subtitle="Per-date first and ninety-ninth percentile against a single flat pair (dashed), "
    "development window",
)
show_with_alt(
    fig,
    "Two ragged lines run across the development window, one above zero and one below, "
    "tracing the per-date upper and lower percentile of the example return. Both widen "
    "sharply around the 2000 and 2008 crises and narrow in the calm stretches between. A "
    "shaded band between them marks the region a single flat pair of clip bounds would "
    "keep, and two dashed horizontal lines draw those flat bounds; the ragged lines spend "
    "long periods well inside them and spike far outside them in the crises.",
)

print(
    f"{WINSOR_EXAMPLE}: flat bounds {flat_lower:.4f} to {flat_upper:.4f} | per-date "
    f"width from {(per_date_bounds['upper'] - per_date_bounds['lower']).min():.4f} to "
    f"{(per_date_bounds['upper'] - per_date_bounds['lower']).max():.4f}, median "
    f"{(per_date_bounds['upper'] - per_date_bounds['lower']).median():.4f} against the "
    f"flat {flat_upper - flat_lower:.4f}"
)

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
composite_feats = [c for c in feature_cols if "composite" in c or "spread" in c or "x_liq" in c]
liquidity_feats = [c for c in feature_cols if "liq" in c or "illiq" in c or "volume_ratio" in c]

print("\nFeature breakdown:")
for _family, _members in (
    ("Momentum/returns", momentum_feats),
    ("Volatility", vol_feats),
    ("Sharpe", sharpe_feats),
    ("Technical", tech_feats),
    ("Trend/MA", trend_feats),
    ("Ranks", rank_feats),
    ("Composites/interactions", composite_feats),
    ("Liquidity", liquidity_feats),
):
    print(f"  {_family}: {len(_members)}")

# Every feature the matrix carries belongs to a family the notebook can name; a column that
# matched none of the prefixes above would be a feature nobody could interpret downstream.
_unfamilied = sorted(
    set(feature_cols)
    - set(momentum_feats + vol_feats + sharpe_feats + tech_feats)
    - set(trend_feats + rank_feats + composite_feats + liquidity_feats)
)
assert not _unfamilied, f"features in no family: {_unfamilied}"

# %% [markdown]
# ### What the matrix is made of
#
# The eight families are meant to carry distinct sources of cross-sectional variation, and
# the counts above say which of them the matrix is weighted towards. They are groups rather
# than a partition, so they do not sum to the column count: a momentum rank is counted under
# both Momentum/returns and Ranks, because it is one of each.
#
# - **Momentum and returns** dominate the count, which follows from the edge hypothesis:
#   the skip-month construction of Jegadeesh and Titman (1993) separates medium-term
#   continuation from the short-term reversal inside the last month.
# - **Ranks and composites** are the next largest group. A raw momentum level in 1997 and
#   the same level in 2009 mean different things; its rank within that day's cross-section
#   does not, which is what makes the ranks usable across a 28-year sample.
# - **Amihud illiquidity** carries the tension between alpha and tradability: the names
#   whose prices move most per dollar traded are the ones a position moves against itself.
#   Multiplying a momentum rank by the dollar-volume rank lets a model price momentum
#   differently in the part of the panel where it could be traded.
# - **Technical oscillators** overlap with momentum by construction. Section 9 measures
#   that overlap rather than assuming it away.

# %% [markdown]
# ## 8. Save Features
#
# Beside the parquet, `write_artifact` leaves a small JSON file with the same name and a
# `.digest.json` suffix, the same way [`02_labels`](02_labels.ipynb) writes its label files. Its
# job is to make the matrix self-describing, so that a later reader can tell which build of the
# features a result came from.
#
# It holds a hash computed over the values in the file; the number of rows; the columns that
# identify a row, here the symbol and the timestamp; the notebook that wrote them; and a hash of
# each input the values were built from. Prices are the only input - the labels are read in
# Section 9, after this write, and score the features rather than shaping them.

# %%
output_path = FEATURES_DIR / "financial.parquet"
# No NaN reaches the artifact. A column that carried one would be scored on a different set
# of dates from every other column, and nothing downstream would say so.
_still_nan = [
    c
    for c in feature_cols
    if output_df.schema[c] in (pl.Float32, pl.Float64) and output_df[c].is_nan().any()
]
assert not _still_nan, f"features reaching the artifact with NaN: {_still_nan}"

record = write_artifact(
    output_df,
    output_path,
    keys=["symbol", "timestamp"],
    written_by="03_financial_features",
    inputs={"market_data": MARKET_DATA_DIGEST},
)
print(f"Saved {n_features} features to {display_path(output_path)}")
print(f"financial.parquet: {record['n_rows']:,} rows, digest {record['digest']}")
# %% [markdown]
# ## 9. Feature Evaluation
#
# Every feature is scored against the primary label with four quantities, each of which
# answers a different question and none of which stands in for another:
#
# - **Information coefficient**: the cross-sectional Spearman correlation on each date,
#   averaged over dates - the quantity a ranking model is scored on.
# - **HAC standard errors**: Newey-West, because the IC series carries autocorrelation of
#   its own even where the label does not overlap.
# - **Benjamini-Hochberg**: the panel is scored on dozens of features at once, so some
#   clear a nominal threshold by construction.
# - **Pairwise correlation**: which features are close enough to be one feature.
#
# **The evaluation stops on the label's endpoint**, not on the observation date. A row
# observed the session before the holdout opens resolves inside it, so a filter on the
# observation date reads holdout prices while appearing not to - this is the boundary
# [`02_labels`](02_labels.ipynb) Section E derives. The feature file written above keeps
# every eligible row, holdout included, because the boundary governs what this notebook reads
# rather than what it writes: the model stages need holdout features to score the holdout
# once, and nothing here may look at them.


# %%
def assign_feature_family(feature_name: str) -> str:
    """Map feature name to family for US equities panel.

    The blends and the interactions are matched first, so a composite of momentum
    ranks is filed as a composite rather than as momentum. Everything after that is
    the construction the feature comes from, which is what makes a diagonal block of
    the correlation matrix mean anything: a rank and the level it ranks belong to the
    same family. The classification is total, and the cell below asserts it is.
    """
    family_map = [
        (["composite", "quality_", "spread", "_x_liq"], "composite"),
        (["mom_", "ret_", "skip_recent", "cumret"], "momentum"),
        (["rev_", "reversal", "str_"], "reversal"),
        (["vol_", "rv_", "realized", "natr", "range_", "mdd_"], "volatility"),
        (["sharpe_", "risk_adj"], "sharpe"),
        (["rsi", "macd", "adx", "cci", "stoch", "bb_", "aroon"], "technical"),
        (["sma_", "ema_", "kama_", "dist_from_52w", "trend"], "trend"),
        (["liq", "turnover", "volume", "amihud"], "liquidity"),
    ]
    for prefixes, family in family_map:
        if any(p in feature_name.lower() for p in prefixes):
            return family
    return "other"


# A feature filed under "other" would sit in a block of the correlation heatmap that shares
# no construction, and would carry a bar in the family chart that means nothing.
_unfamilied_eval = sorted(f for f in feature_cols if assign_feature_family(f) == "other")
assert not _unfamilied_eval, f"features in no evaluation family: {_unfamilied_eval}"


# %% [markdown]
# ### Load labels, join, and stop on the label endpoint
#
# The label's endpoint is the session numbered `PRIMARY_HORIZON` higher, looked up in the
# stock's own series - the identical construction [`02_labels`](02_labels.ipynb) writes the
# labels with, so this notebook stops on the date each label actually resolves on rather than
# on an approximation of it. Reading the endpoint off the next *row* would return a later date
# wherever the stock missed a session, and reading it off the screened frame would return the
# next *eligible* session, which depends on what happens after the decision. It is derived on
# the complete price frame for that second reason.

# %%
_label_col = PRIMARY_LABEL
_label_df = pl.read_parquet(CASE_DIR / "labels" / f"{_label_col}.parquet")


def rows_sessions_ahead(data: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Each stock's row `horizon` sessions later, re-keyed to the session the window opens on."""
    return data.select(
        "symbol",
        (pl.col("session") - horizon).alias("session"),
        pl.col("session").alias("_end_session"),
        pl.col("timestamp").alias("_label_end"),
        pl.col("adj_close").alias("_end_close"),
    )


_panel_ends = (
    raw_df.select("symbol", "timestamp", "session", "adj_close")
    .with_columns((pl.col("session").max().over("symbol") - pl.col("session")).alias("_from_end"))
    .join(rows_sessions_ahead(raw_df, PRIMARY_HORIZON), on=["symbol", "session"], how="left")
)

_joined = output_df.join(_label_df, on=["timestamp", "symbol"], how="inner")
eval_df = (
    _joined.join(
        _panel_ends.select("symbol", "timestamp", "_label_end"),
        on=["symbol", "timestamp"],
        how="left",
    )
    .filter(pl.col("_label_end") < HOLDOUT_START)
    .drop("_label_end")
)
assert eval_df["timestamp"].max() < HOLDOUT_START, "a scored row resolves inside the holdout"

_n_joined = _joined.height
del _joined  # superseded by eval_df; see the note at the end of Section 7

print(f"Feature rows joined to a label: {_n_joined:,}, label column {_label_col}")
print(
    f"Evaluation set after the endpoint restriction: {eval_df.height:,} rows through "
    f"{eval_df['timestamp'].max()}, holdout opens {HOLDOUT_START}"
)

# %% [markdown]
# ### Reconciling the feature index against the label index
#
# Section 1 promised this check. Both residuals are attributed to a named cause and the
# unexplained remainder is asserted empty in both directions, because counting the
# mismatches and printing them is not a check: two stages that screened on different prices
# would print a larger number and pass, which is how the two defects this check was written
# for survived a review each.
#
# A feature row can lack a label only where [`02_labels`](02_labels.ipynb) wrote none, and
# its Section D enumerates exactly three causes: the window reaches past the last session the
# stock has; the stock has no observation on the session that closes the window; or a price is
# missing at one of its two ends. All three are reproduced below on the same session counter
# that notebook used, because a cause left out would be indistinguishable from the two screens
# having drifted apart, and a cause stated in calendar days rather than in sessions would be an
# approximation of the rule instead of the rule.
#
# A label row can lack a feature row only where a feature the selection requires is still
# null, which is the warm-up at the start of a stock's series. `essential_cols` is the set
# that decides it, so the dropped rows are recomputed from that same condition rather than
# guessed at.

# %%
_feature_orphans = output_df.join(_label_df, on=["timestamp", "symbol"], how="anti").join(
    _panel_ends, on=["symbol", "timestamp"], how="left"
)
_past_last_session = pl.col("_from_end") < PRIMARY_HORIZON
_missed_the_closing_session = ~_past_last_session & pl.col("_end_session").is_null()
_no_price_at_an_end = (
    ~_past_last_session
    & ~_missed_the_closing_session
    & (pl.col("adj_close").is_null() | pl.col("_end_close").is_null())
)
_unexplained_features = _feature_orphans.filter(
    ~(_past_last_session | _missed_the_closing_session | _no_price_at_an_end)
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
    f"{_feature_orphans.filter(_past_last_session).height:,} within {PRIMARY_HORIZON} session(s) "
    f"of the end of a stock's series, "
    f"{_feature_orphans.filter(_missed_the_closing_session).height:,} whose stock did not trade "
    f"on the session that closes the window, "
    f"{_feature_orphans.filter(_no_price_at_an_end).height:,} with no price at an end of it"
)
print(
    f"  labels with no feature: {_label_orphans.height:,} — "
    f"{_label_orphans.height - _unexplained_labels.height:,} still inside the feature warm-up"
)

assert _unexplained_features.height == 0, (
    f"{_unexplained_features.height} feature rows have no label, and none of the three causes "
    "02_labels enumerates explains them. The two stages are screening different universes."
)
assert _unexplained_labels.height == 0, (
    f"{_unexplained_labels.height} label rows have no feature row and are not explained by "
    "the feature warm-up. The two stages are screening different universes."
)
print("  reconciled: no unexplained rows on either side")

# The reconciliation was the last reader of the complete price panel and of the frames built
# from it. The IC loop below partitions the evaluation frame by date, so what is still
# referenced here is what the notebook has to hold at its peak.
del raw_df, _panel_ends, _feature_orphans, _label_orphans, _warmup

# %% [markdown]
# ### Per-feature IC with a HAC standard error
#
# Two properties of this loop decide whether the standard error means anything.
#
# **The dates are visited in order.** A Newey-West correction reads the autocovariances of
# the series it is handed, so a series assembled in whatever order the partitions came back
# in is a permutation of time and its lag structure is an artifact of that permutation.
# `partition_by` gives no ordering guarantee, so the keys are sorted before the loop runs.
#
# **The minimum cross-section is half the median**, as in
# [`02_labels`](02_labels.ipynb) Section G, rather than a fixed count. A rank correlation
# over a handful of names is mostly noise, and a bare threshold means something different on
# a panel of a hundred names than on one of three thousand.

# %%
ic_results = {}

# The order comes from a sort on the time axis, not from the partition scan.
_partitions = eval_df.partition_by("timestamp", as_dict=True)
_dates_in_order = [
    (d,) for d in eval_df.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
]
assert set(_dates_in_order) == set(_partitions), "the scored dates and the partitions disagree"
_min_cross_section = int(eval_df.group_by("timestamp").len()["len"].median() // 2)
print(
    f"Scoring {len(_dates_in_order):,} dates, minimum cross-section {_min_cross_section:,} stocks"
)

ic_series = {}  # kept per feature, so the stability figure below reads the same numbers

for feat in feature_cols:
    ic_vals, ic_dates = [], []
    for _key in _dates_in_order:
        vals = _partitions[_key].select([feat, _label_col]).drop_nulls()
        if len(vals) >= _min_cross_section:
            ic, _ = spearmanr(vals[feat].to_numpy(), vals[_label_col].to_numpy())
            if not np.isnan(ic):
                ic_vals.append(ic)
                ic_dates.append(_key[0])
    if len(ic_vals) >= 20:
        ic_results[feat] = compute_ic_hac_stats(np.array(ic_vals), label_horizon=PRIMARY_HORIZON)
        ic_series[feat] = pl.DataFrame({"timestamp": ic_dates, "ic": ic_vals})

print(f"IC computed for {len(ic_results)} of {len(feature_cols)} features")
assert ic_results, "no feature carried enough scored dates to compute an IC"

# %% [markdown]
# ### BH-FDR correction, the HAC effect, and the Fundamental Law's two inputs
#
# Three quantities that are easy to confuse, so each is computed and named separately.
# The **FDR discovery ratio** compares how many features clear the significance
# threshold before and after Benjamini-Hochberg, both counted from the same
# HAC-corrected p-values: it sizes the multiple-testing correction alone. The **HAC effect**
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
#
# Both counts behind the discovery ratio are taken from the same HAC p-values, so the ratio
# prices the multiple testing and nothing else. "Nominal" there means uncorrected for
# multiplicity, not uncorrected for autocorrelation - that second correction is already inside
# every p-value on both sides of the ratio, and a ratio of HAC to naive $|t|$ below one means
# HAC widened the standard error.
#
# One assertion guards the ranking. Features are compared against each other, so they have to
# have been scored over near enough the same span: a feature scored on a fraction of the dates
# is not a weaker signal, it is a different sample, and its place in the ranking means nothing.
# A feature's IC series begins once its own rolling window has filled, so the spread between
# the widest and the narrowest support is bounded by the longest window in the set, and that is
# the bound applied - in the sessions the windows are declared in rather than as a share of the
# sample. On this panel the observed spread is far under it, because names enter over decades
# and the early dates fall below the minimum cross-section for every feature at once; a denser
# panel whose names all start on the same day loses the first window from its longest features
# alone, and nothing is wrong in either case. What the bound still rejects is the defect it was
# written for: a feature scored on a fraction of the dates before the NaN conversion in
# Section 7, which is short by thousands of sessions, not by one window.

# %%
if ic_results:
    _feat_names = list(ic_results.keys())
    _p_values = [ic_results[f]["p_value"] for f in _feat_names]

    fdr_result = benjamini_hochberg_fdr(_p_values, alpha=FDR_ALPHA, return_details=True)

    eval_summary = pl.DataFrame(
        {
            "feature": _feat_names,
            "family": [assign_feature_family(f) for f in _feat_names],
            "n_dates": [ic_series[f].height for f in _feat_names],
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
    n_nominal_sig = sum(1 for p in _p_values if p < FDR_ALPHA)
    fdr_discovery_ratio = n_nominal_sig / max(n_significant, 1)
    _t_ratio = (eval_summary["hac_tstat"].abs() / eval_summary["naive_tstat"].abs()).drop_nans()
    hac_t_ratio_median = float(_t_ratio.median())
    n_t_grew = int((_t_ratio > 1).sum())

    _max_warmup = max(*MOMENTUM_HORIZONS, *VOLATILITY_HORIZONS, *MA_HORIZONS)
    _date_floor, _date_ceiling = eval_summary["n_dates"].min(), eval_summary["n_dates"].max()
    _short = (
        eval_summary.filter(pl.col("n_dates") < _date_ceiling - _max_warmup)
        .sort("n_dates")
        .select("feature", "n_dates")
    )
    assert _date_floor >= _date_ceiling - _max_warmup, (
        f"features were scored on {_date_floor:,} to {_date_ceiling:,} dates, a spread the "
        f"{_max_warmup}-session longest window cannot explain, so their ICs are not measured "
        f"on comparable samples. Short of it: {_short.rows()}"
    )

    print(f"Features tested: {len(_feat_names)}")
    print(
        f"Support per feature: {_date_floor:,} to {_date_ceiling:,} dates of the "
        f"{len(_dates_in_order):,} scored"
    )
    print(f"Nominally significant (p < {FDR_ALPHA}, no multiplicity correction): {n_nominal_sig}")
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
        f"- a maximum over {len(eval_summary)}, not a signal's skill"
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
# ### The twenty strongest features, and which of them clear the correction
#
# The bars are signed, because the sign is the claim: a feature the panel ranks in one
# direction and a feature it ranks in the other are different signals, and the sorted
# magnitude hides that. The label on each bar is its HAC t-statistic, and colour marks
# whether Benjamini-Hochberg still rejects the null - which at the top of the ranking it
# does for every one of them, so what separates these twenty is their direction and not
# their significance. The count that does vary is printed above.

# %%
top_20 = eval_summary.sort(pl.col("ic_mean").abs(), descending=True).head(20)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=top_20["feature"].to_list(),
        y=top_20["ic_mean"].to_list(),
        marker_color=[
            COLORS["blue"] if sig else COLORS["silver_muted"]
            for sig in top_20["significant_fdr05"].to_list()
        ],
        marker_line=dict(color=COLORS["neutral"], width=0.6),
        text=[f"{t:.1f}" for t in top_20["hac_tstat"].to_list()],
        textposition="outside",
    )
)
fig.update_layout(
    title="The strongest features point in both directions, and none is large",
    xaxis_title="Feature, ordered by the size of its information coefficient",
    yaxis_title="Mean IC (Spearman)",
    xaxis_tickangle=-45,
    height=620,
    margin=dict(b=190),
)
show_plotly_with_alt(
    fig,
    "A bar chart of the mean rank correlation of each of the twenty strongest features "
    "against the next session's return, ordered by the size of that correlation and "
    "labelled with each feature's name. Most bars hang below the zero line and only three "
    "rise above it. Every bar is short against the axis, and the largest in each direction "
    "are of similar length, so the strongest features disagree on sign without any of them "
    "being large.",
)

# %% [markdown]
# ### Pairwise feature correlation
#
# Sixty-odd features built from one price series are not sixty-odd signals. The heatmap is
# ordered by family, so the blocks along the diagonal are the horizons of one construction
# and the off-diagonal blocks are where two families measure the same thing. Every twentieth
# date is sampled: a Spearman matrix over every row of the development panel costs hours and
# moves nothing here, because what the figure has to establish is block structure.

# %%
_sample_dates = eval_df["timestamp"].unique().sort().gather_every(20)
_corr_order = [
    f
    for f, _ in sorted(
        ((c, assign_feature_family(c)) for c in feature_cols), key=lambda p: (p[1], p[0])
    )
]
_corr_data = (
    eval_df.filter(pl.col("timestamp").is_in(_sample_dates))
    .select(_corr_order)
    .to_pandas()
    .corr(method="spearman")
)

high_corr_pairs = [
    (f1, f2, float(_corr_data.iloc[i, j]))
    for i, f1 in enumerate(_corr_data.columns)
    for j, f2 in enumerate(_corr_data.columns)
    if i < j and abs(_corr_data.iloc[i, j]) > REDUNDANT_CORR
]
print(
    f"Sampled {len(_sample_dates):,} of {eval_df['timestamp'].n_unique():,} dates | feature pairs "
    f"correlated above {REDUNDANT_CORR}: {len(high_corr_pairs):,} of "
    f"{len(feature_cols) * (len(feature_cols) - 1) // 2:,}"
)

fig = go.Figure(
    data=go.Heatmap(
        z=_corr_data.values,
        x=_corr_data.columns.tolist(),
        y=_corr_data.columns.tolist(),
        colorscale=ml4t_diverging(),
        zmid=0,
        zmin=-1,
        zmax=1,
    )
)
fig.update_layout(
    title="The features cluster into blocks, so they carry fewer signals than columns",
    height=900,
    width=1000,
    xaxis=dict(tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8)),
    margin=dict(l=170, b=170),
)
show_plotly_with_alt(
    fig,
    "A square correlation heat map over the feature set, with the features on both axes "
    "ordered so that related ones sit together, and a diverging green-to-red colour scale "
    "running from one to minus one. Solid green squares stand along the diagonal: a large "
    "block covering the moving-average, oscillator, Sharpe and past-return columns, a "
    "smaller one over the volatility columns, and a two-column one over the liquidity "
    "ranks. Away from those blocks most of the map is pale, and a few narrow red stripes "
    "mark the columns that move opposite to a block.",
)

# %% [markdown]
# ### The HAC correction against the naive one

# %%
fig = go.Figure()
_max_t = max(abs(eval_summary["naive_tstat"]).max(), abs(eval_summary["hac_tstat"]).max()) * 1.1
fig.add_trace(
    go.Scatter(
        x=[-_max_t, _max_t],
        y=[-_max_t, _max_t],
        mode="lines",
        line=dict(dash="dash", color=COLORS["neutral"], width=1),
        name="no correction",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=eval_summary["naive_tstat"].to_list(),
        y=eval_summary["hac_tstat"].to_list(),
        mode="markers",
        text=eval_summary["feature"].to_list(),
        name="feature",
        marker=dict(
            color=[
                COLORS["blue"] if s else COLORS["copper"]
                for s in eval_summary["significant_fdr05"].to_list()
            ],
            size=8,
        ),
    )
)
fig.update_layout(
    title="The autocorrelation correction pulls almost every feature toward zero",
    xaxis_title="t-statistic under the naive standard error",
    yaxis_title="t-statistic under the HAC standard error",
    height=500,
)
show_plotly_with_alt(
    fig,
    "A scatter of each feature's t-statistic computed with the naive standard error "
    "against the same statistic computed with the autocorrelation-consistent one, with a "
    "dashed diagonal marking where the two would agree and reference lines at zero on both "
    "axes, with points shaded by whether the feature survives the false-discovery "
    "correction. The points lie close to the diagonal and span both signs, thinning out "
    "away from the origin. Most sit just inside it, where allowing for serial correlation "
    "has shrunk the statistic; a small number sit just outside, where it has grown "
    "instead. The line above the chart reports how many go each way.",
)

# %% [markdown]
# **Interpretation**:
# - The scatter puts almost every feature between the diagonal and the horizontal axis -
#   above the line where the naive statistic is negative, below it where it is positive,
#   which in both cases is closer to zero. That is the
#   autocorrelation correction doing its work: a daily IC series is persistent enough
#   that the naive standard error is too narrow. It is modest here, because
#   one-session, non-overlapping returns leave little of that persistence to price in,
#   and the print above gives both the median ratio and the count of features whose
#   statistic grew instead. The FDR discovery ratio beside it is the multiple-testing
#   correction, a different quantity measuring a different thing; neither number stands
#   in for the other.
# - The bars in the top-twenty chart run in both directions and none of them is large,
#   which is the shape a cross-sectional edge is supposed to have. It is also why the
#   colouring matters more than the height: clearing Benjamini-Hochberg is the claim,
#   and the ranking by magnitude is not.
# - The heatmap's diagonal blocks are wide, and the count printed beside it says how many
#   pairs exceed the redundancy threshold. The matrix carries fewer signals than columns,
#   and the selection that acts on that happens downstream rather than here.
#
# **What the Fundamental Law does and does not license here**: this is the book's
# highest-breadth case study, and $IR = IC \cdot \sqrt{BR}$ is why an information
# coefficient this small is worth
# building for at all. But the arithmetic above stops at an upper bound on an upper
# bound, and it stops there deliberately: breadth counts *independent* bets, the names
# in one cross-section share factor exposure, consecutive days re-bet the same
# slow-moving signals, and nothing in this notebook estimates that dependence. So the
# figure printed above is what the law permits, not what a strategy would earn. The
# figures below say something the law's arithmetic cannot: the families rank the panel in
# opposing directions, so a model that has not chosen between them inherits the
# disagreement rather than the strength of its strongest member, and the strongest member
# itself varies by more than an order of magnitude between years. Breadth multiplies a
# direction; it neither supplies one nor holds it steady.

# %% [markdown]
# ### What the families are worth, signed
#
# The interpretation above rests on a claim the top-twenty chart cannot settle, because it
# sorts on magnitude: that the panel's features do not agree on a direction. Averaging the
# signed IC within each family is the direct test. A family whose mean sits near zero is
# not a family without signal - it is one whose members disagree, and a model that has not
# chosen a direction inherits that disagreement rather than the strength of its strongest
# member.

# %%
family_ic = {}
for feat, stats in ic_results.items():
    family_ic.setdefault(assign_feature_family(feat), []).append(stats["mean_ic"])
family_summary = pl.DataFrame(
    {
        "family": list(family_ic),
        "mean_ic": [float(np.mean(v)) for v in family_ic.values()],
        "n_features": [len(v) for v in family_ic.values()],
    }
).sort("mean_ic")

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.barh(
    family_summary["family"],
    family_summary["mean_ic"],
    color=[
        COLORS["blue"] if v > 0 else COLORS["copper"] for v in family_summary["mean_ic"].to_list()
    ],
)
ax.axvline(0, color=COLORS["neutral"], lw=0.8)
ax.set_xlabel(f"Mean signed IC against {_label_col}")
add_message_title(
    ax,
    "The families split on direction, and the negative side is the larger",
    subtitle="Mean of the signed information coefficients of each family's members",
)
show_with_alt(
    fig,
    "A horizontal bar for each feature family, giving the mean signed information "
    "coefficient of its members against the one-day forward return, with a vertical line "
    "at zero. Three families - composite, liquidity and sharpe - extend to the right of "
    "zero and are shaded dark; the remaining five, led by reversal and volatility, extend "
    "to the left and are shaded copper. The longest bars on the negative side reach "
    "further from zero than the longest on the positive side.",
)

print(family_summary)
print(
    f"  signed mean across all {len(ic_results)} features "
    f"{float(np.mean([s['mean_ic'] for s in ic_results.values()])):+.5f}, against a mean "
    f"absolute IC of {typical_ic:.5f}"
)

# %% [markdown]
# ### How stable the strongest feature is
#
# One number for a 26-year sample says nothing about whether a feature would have been
# worth trading in any particular part of it. The annual mean of the same daily IC series
# the statistics above are computed from is the cheapest test of that, and it separates two
# things a single average cannot: whether the direction of the effect held, and whether its
# size did. They fail independently, and only the first is what "the sign is stable" means.
#
# The bars begin later than the panel does: a date is scored only where its eligible
# cross-section reaches the minimum set above, and the panel's first years do not reach it.

# %%
_strongest = best["feature"]
annual_ic = (
    ic_series[_strongest]
    .with_columns(pl.col("timestamp").dt.year().alias("year"))
    .group_by("year")
    .agg(pl.col("ic").mean())
    .sort("year")
)
_full_sample = float(ic_series[_strongest]["ic"].mean())

fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
ax.bar(
    annual_ic["year"],
    annual_ic["ic"],
    color=[COLORS["blue"] if v > 0 else COLORS["copper"] for v in annual_ic["ic"].to_list()],
    width=0.7,
)
ax.axhline(_full_sample, color=COLORS["amber"], ls="--", lw=1.4, label="full-sample mean")
ax.axhline(0, color=COLORS["neutral"], lw=0.8)
ax.set_xticks(annual_ic["year"].to_list()[::4])
ax.set_xlabel("Year")
ax.set_ylabel(f"Mean IC of {_strongest}")
ax.legend(frameon=False, fontsize=8, loc="lower right")
add_message_title(
    ax,
    "The strongest feature holds its sign between years but not its size",
    subtitle="Annual mean of the daily information coefficient behind its full-sample average",
)
show_with_alt(
    fig,
    "One bar per year giving the mean daily information coefficient of the strongest "
    "feature. Every bar hangs below the zero line, so the sign never changes, but their "
    "lengths vary by more than an order of magnitude: the earliest years reach far down "
    "the axis and the later ones are short. A dashed horizontal line marks the "
    "full-sample mean, and the bars sit on both sides of it.",
)

_negative_years = int((annual_ic["ic"] < 0).sum())
_by_size = annual_ic["ic"].abs()
print(
    f"{_strongest}: full-sample mean IC {_full_sample:+.5f} | annual means from "
    f"{annual_ic['ic'].min():+.4f} to {annual_ic['ic'].max():+.4f}, negative in "
    f"{_negative_years} of {annual_ic.height} years, and the largest year is "
    f"{_by_size.max() / _by_size.min():.0f}x the smallest"
)

# %% [markdown]
# ## Key takeaways
#
# 1. **Order the screen against the windows.** Per-symbol shifts and rolling windows count
#    rows, so they run on the complete price series; cross-sectional ranks are only
#    meaningful against the names that were sortable that day, so they run on the screened
#    one. The screen goes between them, and it is the same screen
#    [`02_labels`](02_labels.ipynb) applies for the same reason.
# 2. **Winsorize against the cross-section, not against the sample.** A bound estimated over
#    every date is estimated partly on dates that have not happened yet, and it is the wrong
#    width on almost all of them, because the panel's dispersion more than doubles between
#    regimes.
# 3. **A feature evaluation stops on the label's endpoint.** The last development
#    session's label resolves after the holdout opens, so an observation-date filter reads
#    holdout prices while appearing not to.
# 4. **Sort the IC series before correcting it.** A Newey-West standard error reads the
#    autocovariances of the series it is handed, and `partition_by` returns groups in no
#    particular order, so the correction is otherwise computed over a permutation of time.
# 5. **The multiple-testing correction and the autocorrelation correction are different
#    quantities.** One says how many features clear a threshold by construction; the other
#    says how much of a single feature's statistic is left once its own persistence is
#    priced in.
#
# ### Known limitations
#
# - Every feature is price-derived. No fundamentals, no ownership, no text.
# - The families are highly correlated by construction, and this notebook measures that
#   without acting on it; the selection happens downstream.
# - The Fundamental Law arithmetic here is an upper bound on an upper bound, because
#   nothing in it estimates how dependent the bets are.
#
# **Next**: [`04_model_based_features`](04_model_based_features.ipynb) fits features that have
# to be learned per fold. It writes them to a matrix of their own rather than into this one,
# and the two are joined when a model stage loads the dataset.
