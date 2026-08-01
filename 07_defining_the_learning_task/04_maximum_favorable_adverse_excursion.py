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
# # Maximum Favorable/Adverse Excursion (MFE/MAE) Analysis
#
# **Docker image**: `ml4t`
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.2 - Label Engineering
#
# ## Purpose
#
# This notebook provides **empirical justification** for triple-barrier parameter
# choices. Rather than picking arbitrary barrier widths, we analyze actual price
# excursions to determine appropriate thresholds.
#
# ## Key Questions Answered
#
# 1. How far do prices typically move in our favor before reversing?
# 2. How far do prices move against us before recovering?
# 3. How do these distributions differ by asset class and volatility regime?
# 4. What barrier widths capture meaningful price moves without excessive stops?
#
# ## MFE/MAE Definitions
#
# Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) were
# introduced by John Sweeney in *Campaign Trading* (1996) to analyze trade
# management. For a **long** entry at time $t$ with holding period $H$:
#
# $$\text{MFE}(t) = \max_{u \in [t+1, t+H]} \left( \frac{\text{high}(u)}{\text{entry}(t)} - 1 \right)$$
#
# $$\text{MAE}(t) = \max_{u \in [t+1, t+H]} \left( 1 - \frac{\text{low}(u)}{\text{entry}(t)} \right)$$
#
# **Note**: Both MFE and MAE are non-negative by definition. For **short** positions,
# the definitions reverse: MFE uses lows (favorable moves down) and MAE uses highs
# (adverse moves up).
#
# The window opens at $t+1$. The entry fills at bar $t$'s close, so bar $t$'s own
# high and low have already happened: counting them measures movement the position
# was never exposed to, and stretches an $H$-bar holding period over $H+1$ bars.
# Both excursions are inflated, and unevenly, since where the close sits inside bar
# $t$'s range decides which side gains more.
#
# ## Data Coverage
#
# - **ETF Universe** (daily): SPY as representative equity exposure
# - **Crypto Premium** (hourly): BTC for high-volatility comparison
# - **CME Futures** (daily): ES for institutional context
#
# ## Prerequisites
#
# - `03_label_methods` - defines the triple-barrier label whose parameters
#   this notebook empirically justifies.
# - Familiarity with OHLCV bar data and ATR (true range with Wilder smoothing).
# - Polars `max_horizontal` / `min_horizontal` and forward-shifted columns.

# %%
"""Maximum Favorable and Adverse Excursion - ATR-normalized trade path analysis for barrier calibration."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.excursion import analyze_excursions
from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.features.volatility import atr as library_atr
from ml4t.engineer.labeling import triple_barrier_labels
from plotly.subplots import make_subplots

from data import load_cme_futures, load_crypto_perps, load_etfs
from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS  # activates the ml4t Plotly template + house palette

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
ETF_START_DATE = "2015-01-01"
CRYPTO_START_DATE = "2021-01-01"
FUTURES_START_DATE = "2015-01-01"
SAVE_OUTPUT = True

# %%
# Output directory for JSON export
CHAPTER_DIR = get_chapter_dir(7)
set_global_seeds(SEED)
OUTPUT_DIR = CHAPTER_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# %% [markdown]
# ## 1. Vectorized MFE/MAE Computation
#
# We use Polars' `max_horizontal` and `min_horizontal` functions to compute
# forward-looking extremes without Python loops. This approach builds $H$ shifted
# columns - shifts $-1$ through $-H$, so the entry bar is excluded - resulting in
# $O(n \times H)$ work and memory. While vectorized (no Python loops), memory scales
# with horizon. For very large horizons the equivalent reverse-rolling form is
# `high.shift(-1).reverse().rolling_max(H).reverse()`; note the `shift(-1)`, without
# which the window reopens at bar $t$ and reintroduces the pre-entry extremes.


# %%
def compute_mfe_mae(
    prices: pl.DataFrame,
    timestamp_col: str,
    close_col: str,
    horizon_bars: int,
    high_col: str = "high",
    low_col: str = "low",
    side: Literal[1, -1] = 1,
    unit: Literal["pct", "decimal"] = "pct",
) -> pl.DataFrame:
    """
    Compute maximum favorable excursion (MFE) and maximum adverse excursion (MAE).

    Definitions are for a long position when side=1, entering at close(t):
    - MFE(t) = max_{u in [t+1, t+h]} (high(u)/entry(t) - 1)
    - MAE(t) = max_{u in [t+1, t+h]} (1 - low(u)/entry(t))

    The window opens at ``t+1``, not ``t``. Bar ``t``'s own high and low are
    already in the past when the entry fills at bar ``t``'s close, so including
    them measures movement the position never had the chance to experience and
    inflates both excursions. It also makes the window ``h+1`` bars long while the
    caller asked for ``h``.

    For short positions (side=-1), favorable and adverse are swapped.

    Results are clipped at 0 to reflect the standard non-negative excursion notion.

    Parameters
    ----------
    prices : pl.DataFrame
        OHLCV data with timestamp, high, low, close columns
    timestamp_col : str
        Name of timestamp column
    close_col : str
        Name of close price column (used as entry price)
    horizon_bars : int
        Number of bars in the forward window
    high_col : str
        Name of high price column (default: "high")
    low_col : str
        Name of low price column (default: "low")
    side : {1, -1}
        Trade direction: 1 for long, -1 for short
    unit : {"pct", "decimal"}
        Output format: "pct" for percentage, "decimal" for raw

    Returns
    -------
    pl.DataFrame
        Columns: timestamp, mfe_pct (or mfe), mae_pct (or mae), final_return_pct (or final_return)
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if side not in (1, -1):
        raise ValueError("side must be 1 (long) or -1 (short)")

    df = prices.sort(timestamp_col)

    # Use close if high/low not available
    effective_high_col = high_col if high_col in df.columns else close_col
    effective_low_col = low_col if low_col in df.columns else close_col

    # Compute forward-looking max high and min low using horizontal operations.
    # This is vectorized and efficient for reasonable horizon sizes.
    #
    # k runs from 1, not 0: the entry price is bar t's close, so bar t's own high
    # and low are pre-entry. See the docstring.
    forward_high = pl.max_horizontal(
        [pl.col(effective_high_col).shift(-k) for k in range(1, horizon_bars + 1)]
    )
    forward_low = pl.min_horizontal(
        [pl.col(effective_low_col).shift(-k) for k in range(1, horizon_bars + 1)]
    )

    entry = pl.col(close_col)
    exit_ = pl.col(close_col).shift(-horizon_bars)

    # Long position formulas
    long_mfe = (forward_high / entry - 1.0).clip(lower_bound=0.0)
    long_mae = (1.0 - forward_low / entry).clip(lower_bound=0.0)
    long_final = exit_ / entry - 1.0

    # Short position formulas (swap favorable/adverse)
    short_mfe = (1.0 - forward_low / entry).clip(lower_bound=0.0)
    short_mae = (forward_high / entry - 1.0).clip(lower_bound=0.0)
    short_final = 1.0 - exit_ / entry

    # Select based on side
    mfe = pl.when(pl.lit(side) == 1).then(long_mfe).otherwise(short_mfe)
    mae = pl.when(pl.lit(side) == 1).then(long_mae).otherwise(short_mae)
    final_ret = pl.when(pl.lit(side) == 1).then(long_final).otherwise(short_final)

    scale = 100.0 if unit == "pct" else 1.0

    # Output column names based on unit
    mfe_name = "mfe_pct" if unit == "pct" else "mfe"
    mae_name = "mae_pct" if unit == "pct" else "mae"
    final_name = "final_return_pct" if unit == "pct" else "final_return"

    out = df.select(
        [
            pl.col(timestamp_col).alias("timestamp"),
            (mfe * scale).alias(mfe_name),
            (mae * scale).alias(mae_name),
            (final_ret * scale).alias(final_name),
        ]
    ).drop_nulls()

    return out


# %% [markdown]
# ## 2. ATR Computation
#
# **True Range** (accounts for gaps) with Wilder's smoothing:
#
# $$TR_t = \max\left( H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}| \right)$$
#
# $$ATR_t = \frac{n-1}{n} ATR_{t-1} + \frac{1}{n} TR_t$$
#
# The Wilder smoothing is equivalent to an EMA with $\alpha = 1/n$.


# %%
def compute_atr(
    prices: pl.DataFrame,
    timestamp_col: str,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    period: int = 14,
    unit: Literal["price", "pct"] = "pct",
) -> pl.DataFrame:
    """
    Compute ATR using true range with Wilder-style smoothing.

    Parameters
    ----------
    prices : pl.DataFrame
        OHLCV data
    timestamp_col : str
        Name of timestamp column
    high_col, low_col, close_col : str
        Column names for OHLC data
    period : int
        Smoothing period (default: 14, Wilder's original)
    unit : {"price", "pct"}
        Output format: "price" for absolute ATR, "pct" for ATR/close * 100

    Returns
    -------
    pl.DataFrame
        Columns: timestamp, atr, and optionally atr_pct
    """
    if period <= 0:
        raise ValueError("period must be positive")

    df = prices.sort(timestamp_col)

    # True Range: max of (H-L, |H-prev_close|, |L-prev_close|)
    prev_close = pl.col(close_col).shift(1)
    tr = pl.max_horizontal(
        [
            (pl.col(high_col) - pl.col(low_col)).abs(),
            (pl.col(high_col) - prev_close).abs(),
            (pl.col(low_col) - prev_close).abs(),
        ]
    ).alias("true_range")

    # Wilder smoothing: EMA with alpha = 1/period
    atr = tr.ewm_mean(alpha=1.0 / period, adjust=False).alias("atr")

    out = df.with_columns([tr, atr]).select(
        [pl.col(timestamp_col).alias("timestamp"), pl.col("atr"), pl.col(close_col)]
    )

    if unit == "pct":
        out = out.with_columns([(pl.col("atr") / pl.col(close_col) * 100.0).alias("atr_pct")])

    return out.drop_nulls().select(["timestamp", "atr"] + (["atr_pct"] if unit == "pct" else []))


# %%
def compute_percentiles(series: pl.Series, percentiles: list[float]) -> dict[float, float]:
    """Compute percentiles for a series."""
    return {p: float(series.quantile(p / 100)) for p in percentiles}


# %% [markdown]
# ### 2.1 Library ATR: ml4t-engineer
#
# The manual `compute_atr()` above teaches Wilder's smoothing. `ml4t-engineer`
# provides the same algorithm with edge-case handling and panel data support -
# a modern Python alternative to TA-Lib's `ATR()` function.

# %% [markdown]
# ## 3. ETF Analysis (Daily)
#
# We start with SPY as the representative low-volatility daily asset.

# %%
try:
    etfs = load_etfs()

    # Focus on SPY
    spy = etfs.filter(pl.col("symbol") == "SPY").sort("timestamp")

    # Filter date range
    spy = spy.filter(
        pl.col("timestamp")
        >= pl.lit(datetime.strptime(ETF_START_DATE, "%Y-%m-%d")).cast(spy["timestamp"].dtype)
    )

    print(f"SPY daily data: {len(spy):,} bars")
    print(f"Date range: {spy['timestamp'].min()} to {spy['timestamp'].max()}")

except Exception as e:
    print(f"ETF data not found - skipping: {e}")
    spy = None

# %% [markdown]
# ### MFE/MAE computation (21-day horizon)

# %%
spy_mfe_mae = None
spy_atr = None

if spy is not None:
    etf_horizon = 21  # trading days
    spy_mfe_mae = compute_mfe_mae(spy, "timestamp", "close", etf_horizon, unit="pct", side=1)

    print(f"SPY MFE/MAE Statistics (21d horizon, n={len(spy_mfe_mae):,}):")
    print(
        f"  MFE mean: {spy_mfe_mae['mfe_pct'].mean():.2f}%  median: {spy_mfe_mae['mfe_pct'].median():.2f}%"
    )
    print(
        f"  MAE mean: {spy_mfe_mae['mae_pct'].mean():.2f}%  median: {spy_mfe_mae['mae_pct'].median():.2f}%"
    )

    # Percentiles for barrier selection
    mfe_pctls = compute_percentiles(spy_mfe_mae["mfe_pct"], [25, 50, 75, 90, 95])
    mae_pctls = compute_percentiles(spy_mfe_mae["mae_pct"], [25, 50, 75, 90, 95])

    print(f"\nMFE Percentiles: {mfe_pctls}")
    print(f"MAE Percentiles: {mae_pctls}")

# %% [markdown]
# ### ATR comparison: manual vs library

# %%
if spy is not None:
    spy_atr = compute_atr(spy, "timestamp", period=14, unit="pct")
    avg_atr = float(spy_atr["atr_pct"].mean())
    print(f"SPY 14-day ATR: {avg_atr:.2f}% (average)")
    # Deliberately not a barrier recommendation. The conventional "TP=2xATR,
    # SL=1xATR" is printed here only so the reader can compare it against what
    # section 9 derives from the measured excursions, where the stop this panel
    # supports is over three times ATR rather than one.
    print(
        f"Conventional rule of thumb, for comparison only: "
        f"TP=2xATR ({avg_atr * 2:.2f}%), SL=1xATR ({avg_atr:.2f}%)"
    )

    # Library ATR comparison
    lib_atr_values = library_atr(
        spy["high"].to_numpy(),
        spy["low"].to_numpy(),
        spy["close"].to_numpy(),
        period=14,
    )
    lib_atr_avg = float(np.nanmean(lib_atr_values / spy["close"].to_numpy() * 100))
    print(f"  Library ATR (ml4t-engineer): {lib_atr_avg:.2f}%")
    print(f"  Difference: {abs(avg_atr - lib_atr_avg):.4f}pp")

# %% [markdown]
# ### ETF MFE/MAE Distribution

# %%
if spy_mfe_mae is not None:
    # Shared x and y ranges across the two panels. The title makes a comparison
    # between them, and panels on independent axes cannot support one: plotly's
    # default fits each histogram to its own extent, so the narrower distribution
    # is drawn as wide as the broader one. The shared limit is the 99.5th
    # percentile of whichever series runs further, which keeps the bulk legible
    # without letting a handful of crisis observations set the scale.
    SPY_HIST_MAX = max(
        float(spy_mfe_mae["mfe_pct"].quantile(0.995)),
        float(spy_mfe_mae["mae_pct"].quantile(0.995)),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Favorable (MFE)", "Adverse (MAE)"],
        shared_yaxes=True,
    )

    # MFE histogram
    fig.add_trace(
        go.Histogram(
            x=spy_mfe_mae["mfe_pct"].to_numpy(),
            xbins=dict(start=0, end=SPY_HIST_MAX, size=SPY_HIST_MAX / 50),
            name="MFE",
            marker_color=COLORS["positive"],
        ),
        row=1,
        col=1,
    )

    # MAE histogram
    fig.add_trace(
        go.Histogram(
            x=spy_mfe_mae["mae_pct"].to_numpy(),
            xbins=dict(start=0, end=SPY_HIST_MAX, size=SPY_HIST_MAX / 50),
            name="MAE",
            marker_color=COLORS["negative"],
        ),
        row=1,
        col=2,
    )

    # Median markers, so the claim in the title is readable off the chart.
    for col, median in ((1, mfe_pctls[50]), (2, mae_pctls[50])):
        fig.add_vline(
            x=median,
            line_dash="dash",
            line_color=COLORS["neutral"],
            row=1,
            col=col,
            annotation_text="median",
            annotation_font_size=10,
        )

    fig.update_layout(
        title="SPY favorable moves run wider than adverse ones at the median",
        showlegend=False,
        height=400,
    )
    fig.update_xaxes(title_text="Excursion (%)", range=[0, SPY_HIST_MAX], row=1, col=1)
    fig.update_xaxes(title_text="Excursion (%)", range=[0, SPY_HIST_MAX], row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.show()

# %% [markdown]
# ## 4. Crypto Analysis (Hourly)
#
# Crypto exhibits higher volatility, requiring wider barriers.

# %%
try:
    crypto = load_crypto_perps()

    # Focus on BTC for analysis
    btc = crypto.filter(pl.col("symbol") == "BTCUSDT").sort("timestamp")

    # Filter to recent data (cast literal to match column dtype)
    btc = btc.filter(
        pl.col("timestamp")
        >= pl.lit(datetime.strptime(CRYPTO_START_DATE, "%Y-%m-%d").replace(tzinfo=UTC)).cast(
            btc["timestamp"].dtype
        )
    )

    print(f"BTC hourly data: {len(btc):,} bars")
    print(f"Date range: {btc['timestamp'].min()} to {btc['timestamp'].max()}")

    # Compute MFE/MAE for 8-hour horizon (matches funding rate cycle)
    crypto_horizon = 8  # hours
    btc_mfe_mae = compute_mfe_mae(btc, "timestamp", "close", crypto_horizon, unit="pct", side=1)

    # Summary statistics
    print(f"\nBTC MFE/MAE Statistics (8h horizon, n={len(btc_mfe_mae):,}):")
    print(f"  MFE mean: {btc_mfe_mae['mfe_pct'].mean():.2f}%")
    print(f"  MFE median: {btc_mfe_mae['mfe_pct'].median():.2f}%")
    print(f"  MAE mean: {btc_mfe_mae['mae_pct'].mean():.2f}%")
    print(f"  MAE median: {btc_mfe_mae['mae_pct'].median():.2f}%")

    # Percentiles for barrier selection
    btc_mfe_pctls = compute_percentiles(btc_mfe_mae["mfe_pct"], [25, 50, 75, 90, 95])
    btc_mae_pctls = compute_percentiles(btc_mfe_mae["mae_pct"], [25, 50, 75, 90, 95])

    print(f"\nMFE Percentiles: {btc_mfe_pctls}")
    print(f"MAE Percentiles: {btc_mae_pctls}")

except Exception as e:
    print(f"Crypto data not found - skipping: {e}")
    btc_mfe_mae = None

# %% [markdown]
# ### Crypto MFE/MAE Distribution

# %%
if btc_mfe_mae is not None:
    # Shared limits, for the same reason as the SPY panels above: the title
    # compares the two distributions, so they have to be drawn on one scale.
    BTC_HIST_MAX = max(
        float(btc_mfe_mae["mfe_pct"].quantile(0.995)),
        float(btc_mfe_mae["mae_pct"].quantile(0.995)),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Favorable (MFE)", "Adverse (MAE)"],
        shared_yaxes=True,
    )

    # MFE histogram
    fig.add_trace(
        go.Histogram(
            x=btc_mfe_mae["mfe_pct"].to_numpy(),
            xbins=dict(start=0, end=BTC_HIST_MAX, size=BTC_HIST_MAX / 50),
            name="MFE",
            marker_color=COLORS["positive"],
        ),
        row=1,
        col=1,
    )

    # MAE histogram
    fig.add_trace(
        go.Histogram(
            x=btc_mfe_mae["mae_pct"].to_numpy(),
            xbins=dict(start=0, end=BTC_HIST_MAX, size=BTC_HIST_MAX / 50),
            name="MAE",
            marker_color=COLORS["negative"],
        ),
        row=1,
        col=2,
    )

# %%
if btc_mfe_mae is not None:
    # Reference line at each side's own 75th percentile.
    #
    # This line used to sit at a flat 2.0% "from typical crypto settings" - a round
    # number asserted as typical, in the notebook whose argument is that barrier
    # widths should be measured rather than assumed. The measured p75 is what the
    # rest of the notebook calibrates against, and drawing each panel's own value is
    # what makes the near-symmetry in the title checkable.
    for col, pctl in ((1, btc_mfe_pctls[75]), (2, btc_mae_pctls[75])):
        fig.add_vline(
            x=pctl,
            line_dash="dash",
            line_color=COLORS["copper"],
            row=1,
            col=col,
            annotation_text="75th pctl",
            annotation_font_size=10,
        )

    fig.update_layout(
        title="BTC favorable and adverse excursions are near mirror images",
        showlegend=False,
        height=400,
    )
    fig.update_xaxes(title_text="Excursion (%)", range=[0, BTC_HIST_MAX], row=1, col=1)
    fig.update_xaxes(title_text="Excursion (%)", range=[0, BTC_HIST_MAX], row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.show()

# %% [markdown]
# ## 5. Futures Analysis (Daily)
#
# Futures have institutional flow and roll considerations.

# %%
try:
    es = load_cme_futures(products=["ES"])

    # Normalize timestamp column (daily data uses session_date)
    if "session_date" in es.columns:
        es = es.rename({"session_date": "timestamp"})
    elif "ts_event" in es.columns:
        es = es.rename({"ts_event": "timestamp"})
    elif "date" in es.columns:
        es = es.rename({"date": "timestamp"})
    # Keep the front contract only, BEFORE sorting.
    #
    # load_cme_futures returns one row per (session, tenor): 11,458 rows over 3,866
    # sessions, tenors 0, 1 and 2. compute_mfe_mae walks a forward window with
    # shift(-k) over whatever row order it is handed, so on the un-filtered frame the
    # 21-bar excursion window steps across three different contracts rather than
    # forward in time on one. That inflates every excursion statistic, and these
    # statistics are exported to mfe_mae_summary.json as the source for the chapter's
    # barrier-width references.
    if "tenor" in es.columns:
        n_before = len(es)
        es = es.filter(pl.col("tenor") == 0)
        print(f"Front contract only: {len(es):,} of {n_before:,} rows (tenors 1 and 2 dropped)")

    es = es.sort("timestamp")

    # Filter date range (cast literal to match column dtype)
    es = es.filter(
        pl.col("timestamp")
        >= pl.lit(datetime.strptime(FUTURES_START_DATE, "%Y-%m-%d").replace(tzinfo=UTC)).cast(
            es["timestamp"].dtype
        )
    )

    print(f"ES futures data: {len(es):,} bars")
    print(f"Date range: {es['timestamp'].min()} to {es['timestamp'].max()}")

    # Compute MFE/MAE for 21-day horizon. Excursions ride the roll-adjusted
    # series (adj_*) so roll gaps don't register as favorable/adverse moves.
    futures_horizon = 21
    es_mfe_mae = compute_mfe_mae(
        es,
        "timestamp",
        "adj_close",
        futures_horizon,
        high_col="adj_high",
        low_col="adj_low",
        unit="pct",
        side=1,
    )

    # Summary statistics
    print(f"\nES MFE/MAE Statistics (21d horizon, n={len(es_mfe_mae):,}):")
    print(f"  MFE mean: {es_mfe_mae['mfe_pct'].mean():.2f}%")
    print(f"  MFE median: {es_mfe_mae['mfe_pct'].median():.2f}%")
    print(f"  MAE mean: {es_mfe_mae['mae_pct'].mean():.2f}%")
    print(f"  MAE median: {es_mfe_mae['mae_pct'].median():.2f}%")

    es_mfe_pctls = compute_percentiles(es_mfe_mae["mfe_pct"], [25, 50, 75, 90, 95])
    es_mae_pctls = compute_percentiles(es_mfe_mae["mae_pct"], [25, 50, 75, 90, 95])
    print(f"\nMFE Percentiles: {es_mfe_pctls}")
    print(f"MAE Percentiles: {es_mae_pctls}")

    es_atr = compute_atr(
        es, "timestamp", high_col="adj_high", low_col="adj_low", close_col="adj_close", unit="pct"
    )
    es_avg_atr = float(es_atr["atr_pct"].mean())
    print(f"\nES 14-day ATR: {es_avg_atr:.2f}% (average)")

except Exception as e:
    print(f"Futures data not found - skipping: {e}")
    es_mfe_mae = None
    es_atr = None
    es_avg_atr = None

# %% [markdown]
# ## 6. MFE/MAE Scatter Plot
#
# A scatter plot reveals the joint distribution and helps identify
# candidate barrier rectangles.

# %%
if spy_mfe_mae is not None:
    # Sample for performance (scatter with 10k+ points is slow)
    sample_size = min(2000, len(spy_mfe_mae))
    sample = spy_mfe_mae.sample(sample_size, seed=SEED)

    # Fixed axis range for reproducibility
    x_max = float(spy_mfe_mae["mae_pct"].quantile(0.99))
    y_max = float(spy_mfe_mae["mfe_pct"].quantile(0.99))

    fig = go.Figure()

    # Scatter plot - larger markers, lower opacity for print clarity
    fig.add_trace(
        go.Scatter(
            x=sample["mae_pct"].to_numpy(),
            y=sample["mfe_pct"].to_numpy(),
            mode="markers",
            marker=dict(size=5, opacity=0.3, color=COLORS["neutral"]),
            name="Observations",
        )
    )

# %%
if spy_mfe_mae is not None:
    # House palette with distinct line styles for percentile levels; the
    # yshift staggers the three MAE (vertical) labels so they don't collide
    # along the top axis.
    pctl_styles = [
        (50, COLORS["blue"], "solid", "p50", 0),
        (75, COLORS["copper"], "dash", "p75", -16),
        (90, COLORS["neutral"], "dot", "p90", -32),
    ]

    for pctl, color, dash, label, yshift in pctl_styles:
        mae_val = float(spy_mfe_mae["mae_pct"].quantile(pctl / 100))
        mfe_val = float(spy_mfe_mae["mfe_pct"].quantile(pctl / 100))

        # Horizontal line for MFE threshold (take profit)
        fig.add_hline(
            y=mfe_val,
            line_dash=dash,
            line_color=color,
            line_width=1.5,
            annotation_text=f"MFE {label}: {mfe_val:.1f}%",
            annotation_font_size=10,
        )

        # Vertical line for MAE threshold (stop loss)
        fig.add_vline(
            x=mae_val,
            line_dash=dash,
            line_color=color,
            line_width=1.5,
            annotation_text=f"MAE {label}: {mae_val:.1f}%",
            annotation_font_size=10,
            annotation_yshift=yshift,
        )

    fig.update_layout(
        title="SPY 21-day excursions concentrate below the 75th-percentile barriers",
        xaxis_title="Maximum Adverse Excursion (%)",
        yaxis_title="Maximum Favorable Excursion (%)",
        xaxis_range=[0, x_max],
        yaxis_range=[0, y_max],
        height=500,
        width=600,
        font=dict(size=12),
    )
    fig.show()

# %% [markdown]
# ## 7. Regime-Conditional Analysis
#
# Barrier effectiveness varies by market regime. High volatility periods
# require wider barriers to avoid premature stops.

# %%
if spy_mfe_mae is not None and spy_atr is not None:
    # Join MFE/MAE with ATR for regime conditioning
    spy_regime = spy_mfe_mae.join(
        spy_atr.select(["timestamp", "atr_pct"]),
        on="timestamp",
        how="left",
    ).drop_nulls()

    # Define regimes based on ATR percentiles
    low_vol_thresh = spy_regime["atr_pct"].quantile(0.33)
    high_vol_thresh = spy_regime["atr_pct"].quantile(0.67)

    spy_regime = spy_regime.with_columns(
        [
            pl.when(pl.col("atr_pct") <= low_vol_thresh)
            .then(pl.lit("Low Vol"))
            .when(pl.col("atr_pct") >= high_vol_thresh)
            .then(pl.lit("High Vol"))
            .otherwise(pl.lit("Normal"))
            .alias("regime")
        ]
    )

    print("\n=== Regime-Conditional Analysis (SPY) ===")
    for regime in ["Low Vol", "Normal", "High Vol"]:
        regime_data = spy_regime.filter(pl.col("regime") == regime)
        if len(regime_data) > 50:
            print(f"\n{regime} (n={len(regime_data):,}):")
            print(f"  MFE median: {regime_data['mfe_pct'].median():.2f}%")
            print(f"  MAE median: {regime_data['mae_pct'].median():.2f}%")
            print(f"  ATR avg: {regime_data['atr_pct'].mean():.2f}%")

# %% [markdown]
# ## 8. Barrier Validation via Hit-Type Analysis
#
# MFE/MAE alone doesn't tell us which barrier hits first.
# We validate by running triple-barrier and inspecting hit distributions.

# %%
if spy is not None:
    # Barriers taken from the measured excursion percentiles, not from round numbers.
    #
    # This is the notebook's stated purpose - "rather than picking arbitrary barrier
    # widths, we analyze actual price excursions" - and mfe_pctls / mae_pctls were
    # already computed above and then not used here. Take-profit comes from the MFE
    # distribution and stop-loss from the MAE distribution, at the same percentile, so
    # each label genuinely names the quantity it is derived from.
    configs = [
        (f"{label} (p{p})", mfe_pctls[p] / 100, mae_pctls[p] / 100)
        for label, p in (("Tight", 50), ("Medium", 75), ("Wide", 90))
    ]

    print("Barrier widths derived from SPY's own excursion distribution:")
    for name, tp, sl in configs:
        print(f"  {name}: TP={tp:.2%} (MFE), SL={sl:.2%} (MAE)")

    print("\n=== Barrier Hit Validation (SPY 21d) ===")

    # triple_barrier_labels requires Datetime timestamps (numpy conversion)
    spy_dt = spy.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    for name, tp, sl in configs:
        config = LabelingConfig.triple_barrier(
            upper_barrier=tp,
            lower_barrier=sl,
            max_holding_period=21,
            side=1,
        )

        labels = triple_barrier_labels(spy_dt, config=config, price_col="close")

        # Hit type distribution
        hit_dist = labels.group_by("barrier_hit").len().sort("barrier_hit")

        # Resolution time statistics
        if "resolution_time" in labels.columns:
            avg_res_time = labels["resolution_time"].mean()
        else:
            avg_res_time = None

        print(f"\n{name} (TP={tp:.1%}, SL={sl:.1%}):")
        for row in hit_dist.iter_rows(named=True):
            pct = row["len"] / len(labels) * 100
            print(f"  {row['barrier_hit']}: {row['len']:,} ({pct:.1f}%)")

        if avg_res_time:
            print(f"  Avg resolution: {avg_res_time:.1f} bars")

# %% [markdown]
# ## 9. Recommendations Summary
#
# Each instrument's barriers come from its own excursion distribution, at the same
# percentile, so the take-profit is read off the MFE and the stop off the MAE. The
# table below is built from the measured quantiles rather than typed, which is the
# whole point of the exercise: a barrier width is a claim about how far price
# travels, and this notebook measured how far it travels.
#
# For the two daily instruments the table also reports the ATR multiple a
# volatility-scaled stop would use. That multiple is the 75th percentile of the
# per-entry ratio `MAE(t) / ATR(t)`, **not** the p75 excursion divided by the mean
# ATR. The two are different numbers and only the first answers the question an
# ATR-scaled stop asks. A stop placed at `k x ATR(t)` is re-sized every day, so
# what has to hold at the 75th percentile is the *ratio*; dividing one aggregate by
# another gives the ratio of typical values, which ignores that ATR and the
# excursion it scales move together. It is a *consequence* of the measured
# distribution rather than an input to it, which is why the multiple differs by
# instrument instead of being a round number chosen in advance.


# %%
def atr_scaled_stop_multiple(
    mfe_mae: pl.DataFrame, atr: pl.DataFrame, quantile: float = 0.75
) -> float | None:
    """p75 of MAE(t) / ATR(t), the multiple a volatility-scaled stop needs."""
    joined = mfe_mae.join(atr.select(["timestamp", "atr_pct"]), on="timestamp", how="inner")
    ratio = (
        joined.filter(pl.col("atr_pct") > 0)
        .select((pl.col("mae_pct") / pl.col("atr_pct")).alias("r"))
        .drop_nulls()["r"]
    )
    return float(ratio.quantile(quantile)) if len(ratio) else None


barrier_rows = []

if spy_mfe_mae is not None:
    barrier_rows.append(
        ("ETF (SPY)", "21d", mfe_pctls, mae_pctls, atr_scaled_stop_multiple(spy_mfe_mae, spy_atr))
    )
if btc_mfe_mae is not None:
    btc_mfe_pctls = compute_percentiles(btc_mfe_mae["mfe_pct"], [50, 75])
    btc_mae_pctls = compute_percentiles(btc_mfe_mae["mae_pct"], [50, 75])
    barrier_rows.append(("Crypto (BTC)", "8h", btc_mfe_pctls, btc_mae_pctls, None))
if es_mfe_mae is not None:
    barrier_rows.append(
        (
            "Futures (ES)",
            "21d",
            es_mfe_pctls,
            es_mae_pctls,
            atr_scaled_stop_multiple(es_mfe_mae, es_atr),
        )
    )

print("Barrier widths derived from each instrument's own excursion distribution\n")
header = f"{'Instrument':<14}{'Horizon':<9}{'TP p50':>8}{'SL p50':>8}{'TP p75':>8}{'SL p75':>8}"
print(header + f"{'stop, p75 of MAE/ATR':>23}")
print("-" * (len(header) + 23))
for name, horizon, mfe_p, mae_p, stop_mult in barrier_rows:
    line = (
        f"{name:<14}{horizon:<9}"
        f"{mfe_p[50]:>7.2f}%{mae_p[50]:>7.2f}%{mfe_p[75]:>7.2f}%{mae_p[75]:>7.2f}%"
    )
    line += f"{stop_mult:>22.2f}x" if stop_mult is not None else f"{'n/a':>23}"
    print(line)

if spy_mfe_mae is not None and es_mfe_mae is not None:
    print("\nES vs SPY adverse excursions, same 21-day horizon:")
    for label, q in (("median", 50), ("p75", 75)):
        print(f"  MAE {label:<7} SPY {mae_pctls[q]:.2f}%   ES {es_mae_pctls[q]:.2f}%")

# %% [markdown]
# **What the table says.** BTC's 8-hour excursions are an order of magnitude smaller
# than the daily instruments' 21-day excursions, which is horizon rather than asset:
# a fixed percentage barrier is workable there because the funding cycle fixes the
# holding period. For SPY and ES the stop that the p75 adverse excursion implies is
# *wider* than one ATR, so an implementation that reaches for a round `1xATR` stop
# will be stopped out by ordinary movement.
#
# Note the ES-versus-SPY comparison printed above. ES adverse excursions are wider
# than SPY's at both quantiles, so the intuition that an index future deserves a
# *tighter* stop than the matching ETF is contradicted by the measurement. This is
# the same trap as the take-profit/stop-loss ordering in §8: a plausible-sounding
# asymmetry, asserted rather than measured, pointing the wrong way.

# %% [markdown]
# **Key findings:**
#
# 1. A barrier width is a property of one instrument at one horizon, not of an asset
#    class - the printed table is the calibration, recomputed whenever the data is.
# 2. Regime conditioning matters more than the choice of instrument: §7 shows the
#    high-volatility tercile carrying roughly twice the excursion of the low one, a
#    ratio larger than any gap between SPY and ES.
# 3. ATR-scaled barriers adapt to that regime shift; fixed percentage barriers are
#    defensible only where the holding period is short and externally fixed.

# %% [markdown]
# ## 10. Export Statistics for Documentation
#
# Save summary statistics for chapter reference.

# %%
# Compile summary statistics
summary = {
    "generated_at": datetime.now(UTC).isoformat(),
    "horizons": {
        "etf_spy": 21,
        "crypto_btc": 8,
        "futures_es": 21,
    },
    "datasets": {},
}

if spy_mfe_mae is not None:
    summary["datasets"]["etf_spy_21d"] = {
        "n_observations": len(spy_mfe_mae),
        "mfe_median": float(spy_mfe_mae["mfe_pct"].median()),
        "mfe_75th": float(spy_mfe_mae["mfe_pct"].quantile(0.75)),
        "mfe_90th": float(spy_mfe_mae["mfe_pct"].quantile(0.90)),
        "mae_median": float(spy_mfe_mae["mae_pct"].median()),
        "mae_75th": float(spy_mfe_mae["mae_pct"].quantile(0.75)),
        "mae_90th": float(spy_mfe_mae["mae_pct"].quantile(0.90)),
    }
    if spy_atr is not None:
        summary["datasets"]["etf_spy_21d"]["atr_pct_avg"] = float(spy_atr["atr_pct"].mean())

# %%
# BTC and ES summary statistics
if btc_mfe_mae is not None:
    summary["datasets"]["crypto_btc_8h"] = {
        "n_observations": len(btc_mfe_mae),
        "mfe_median": float(btc_mfe_mae["mfe_pct"].median()),
        "mfe_75th": float(btc_mfe_mae["mfe_pct"].quantile(0.75)),
        "mfe_90th": float(btc_mfe_mae["mfe_pct"].quantile(0.90)),
        "mae_median": float(btc_mfe_mae["mae_pct"].median()),
        "mae_75th": float(btc_mfe_mae["mae_pct"].quantile(0.75)),
        "mae_90th": float(btc_mfe_mae["mae_pct"].quantile(0.90)),
    }

if es_mfe_mae is not None:
    summary["datasets"]["futures_es_21d"] = {
        "n_observations": len(es_mfe_mae),
        "mfe_median": float(es_mfe_mae["mfe_pct"].median()),
        "mfe_75th": float(es_mfe_mae["mfe_pct"].quantile(0.75)),
        "mfe_90th": float(es_mfe_mae["mfe_pct"].quantile(0.90)),
        "mae_median": float(es_mfe_mae["mae_pct"].median()),
        "mae_75th": float(es_mfe_mae["mae_pct"].quantile(0.75)),
        "mae_90th": float(es_mfe_mae["mae_pct"].quantile(0.90)),
    }

# %%
# Print summary
print("\n=== Summary Statistics for Chapter Reference ===")
for dataset, stats in summary.get("datasets", {}).items():
    print(f"\n{dataset}:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

# Save to JSON for chapter reference
if SAVE_OUTPUT:
    output_path = OUTPUT_DIR / "mfe_mae_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {output_path}")

# %% [markdown]
# ## 11. Multi-Horizon Excursion Analysis
#
# The manual implementation above analyzes one horizon at a time.
# `ml4t-diagnostic` provides `analyze_excursions()` for multi-horizon analysis
# in a single call - useful for comparing barrier widths across holding periods.

# %%
if spy is not None:
    spy_close = spy["close"]
    horizons = [5, 10, 21, 42]
    result = analyze_excursions(
        spy_close,
        horizons=horizons,
        percentiles=[25, 50, 75, 90],
    )

    print(f"Multi-Horizon Excursion Analysis (SPY) - observations: {result.n_samples:,}")

    multi_horizon_rows = [
        {
            "horizon_days": h,
            "mfe_p50_pct": round(result.get_percentile(h, 50, "mfe") * 100, 2),
            "mfe_p75_pct": round(result.get_percentile(h, 75, "mfe") * 100, 2),
            "mae_p50_pct": round(result.get_percentile(h, 50, "mae") * 100, 2),
            "mae_p75_pct": round(result.get_percentile(h, 75, "mae") * 100, 2),
        }
        for h in horizons
    ]
    multi_horizon_df = pl.DataFrame(multi_horizon_rows)
    display(multi_horizon_df)

# %% [markdown]
# Two conventions appear in the table above. The manual MFE/MAE block (Section 3)
# clips both excursions to be non-negative - adverse moves report as positive
# percentages. The `analyze_excursions` library reports adverse excursions as
# *signed* deviations (negative when the price drops below entry), which is why
# `mae_p50_pct` shows negative values. Both conventions are valid; barrier
# calibration only needs the magnitude. The two are close but not identical: the
# library measures excursions on close prices, while the manual block uses
# intraday high/low, so the manual magnitudes run wider (quantified in the
# comparison below).

# %%
# Compare manual single-horizon vs library multi-horizon
if spy_mfe_mae is not None:
    print("--- Manual vs Library (21d horizon) ---")
    manual_mfe_50 = spy_mfe_mae["mfe_pct"].quantile(0.5)
    library_mfe_50 = result.get_percentile(21, 50, "mfe") * 100
    manual_mae_50 = spy_mfe_mae["mae_pct"].quantile(0.5)
    library_mae_50_abs = abs(result.get_percentile(21, 50, "mae") * 100)
    print(f"Manual  MFE p50 (uses high):  {manual_mfe_50:.2f}%")
    print(f"Library MFE p50 (uses close): {library_mfe_50:.2f}%")
    print(f"Manual  |MAE| p50 (uses low):   {manual_mae_50:.2f}%")
    print(f"Library |MAE| p50 (uses close): {library_mae_50_abs:.2f}%")
    print(
        "\nManual uses high/low prices for intraday extremes; the library uses "
        "close prices only. High/low captures wider excursions and is the "
        "more appropriate reference for barrier-width selection."
    )

# %% [markdown]
# ## Summary
#
# This notebook provides empirical justification for triple-barrier parameters:
#
# 1. **Vectorized MFE/MAE**: Efficient computation using Polars horizontal operations
# 2. **ATR**: True range with Wilder smoothing
# 3. **Non-negative excursions**: MFE/MAE clipped at 0 by definition
# 4. **Regime conditioning**: High-vol periods require wider barriers
# 5. **Validation via hit-types**: Barrier widths validated by hit distribution
#
# ### Key Results
#
# - **Barriers are measured, not chosen**: §8's widths come from the MFE and MAE
#   quantiles and §9's table reports them per instrument, so a re-run recalibrates
#   them rather than confirming a number typed here.
# - **The asymmetry is not the one intuition offers**: at the 90th percentile SPY's
#   stop belongs wider than its take-profit, and ES's adverse excursions are wider
#   than SPY's - both the reverse of the conventional framing.
# - **Volatility scaling**: required for daily strategies; less critical where the
#   holding period is short and externally fixed, as with the crypto funding cycle.
#
# ### Production Usage
#
# Use the exported `mfe_mae_summary.json` for chapter references.
# For custom calibration, run this notebook with your specific asset and horizon.
