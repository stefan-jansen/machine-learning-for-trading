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
# # Spread Estimation from Market Data
#
# **Docker image**: `ml4t`
#
# The **bid-ask spread** is the gap between the highest price a buyer is currently willing to pay
# and the lowest a seller will accept. An order that has to execute now crosses it and gives up
# about half of it, which for most strategies is the largest single component of trading cost. The
# spread is directly observable only if you hold quote data, and most historical datasets carry
# nothing but daily open, high, low, close and volume.
#
# Two classical estimators recover a spread from those daily bars alone. This notebook implements
# both, checks what they produce against real quotes on a market where quotes are available, and
# then runs them across six markets where they are not. Because the same estimator is applied
# everywhere, the comparison is honest about what it measures: the estimator's output, which is not
# the same thing as an observed cost.
#
# **Learning Objectives:**
# - Estimate a spread from daily high and low prices with the Corwin-Schultz estimator, and say
#   which assumption about where highs and lows occur it rests on
# - Estimate a spread from the serial covariance of returns with Roll's estimator, and say when
#   that covariance stops carrying the information the estimator needs
# - Check an estimator against observed quotes by separating whether it *ranks* instruments
#   correctly from whether it gets their *level* right, and report both
# - Aggregate panels recorded at different frequencies onto one daily grid before comparing them
# - Recognize when two cost figures are quoted in units that cannot be compared, and say what
#   further information each one would need
#
# **Book Reference:** Chapter 18, Section 18.3
#
# **Prerequisites:** Access to six OHLCV datasets and licensed NASDAQ-100 minute bars with
# microstructure columns.

# %%
"""Spread estimation from market data with unit-aware validation."""

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from _cost_analysis import corwin_schultz_spread, roll_spread
from IPython.display import Markdown, display

from data import (
    load_cme_futures,
    load_crypto_perps,
    load_etfs,
    load_fx_pairs,
    load_macro,
    load_nasdaq100_bars,
    load_sp500_daily_bars,
)
from utils.paths import get_case_study_source_dir
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

# %% tags=["parameters"]
MAX_SYMBOLS = 50
NQ100_SYMBOLS = 12
NQ100_START_DATE = "2021-10-01"
NQ100_END_DATE = "2021-12-31"
CS_WINDOW = 20
ROLL_WINDOW = 20

# %% [markdown]
# What each setting decides:
#
# - `MAX_SYMBOLS` caps how many instruments each of the six cross-asset panels contributes. The
#   panels are trimmed to their most actively traded names because a spread estimator on a barely
#   traded instrument mostly measures the gaps between its trades.
# - `NQ100_SYMBOLS` caps the NASDAQ-100 validation sample. Twelve names is a small sample, and the
#   validation section reports the correlation on that basis rather than implying more precision
#   than twelve points can carry.
# - `NQ100_START_DATE` and `NQ100_END_DATE` bound the validation quarter. The minute-bar quote data
#   is licensed and large, and one quarter is enough to compare estimator output against observed
#   quotes; a longer window would change the sample, not the lesson.
# - `CS_WINDOW` and `ROLL_WINDOW` are both 20 sessions, roughly one trading month. Both estimators
#   average a noisy per-period quantity, so the window trades responsiveness against how much
#   of that noise is averaged away. Shorter windows track a widening spread sooner and are less
#   stable.

# %% [markdown]
# ## 1. Two OHLCV Estimators
#
# The Corwin-Schultz estimator starts from an observation about where the day's extremes occur: the
# high is usually a trade at the ask and the low is usually a trade at the bid, so the high-low
# range contains the spread once as well as whatever the price genuinely moved. Volatility scales
# with the length of the interval and the spread does not, so comparing the range over one day with
# the range over two days separates them. For consecutive periods,
#
# $$
# \beta = \mathbb{E}\left[\sum_{j=1}^{2}
# \ln\left(\frac{H_j}{L_j}\right)^2\right], \qquad
# \gamma = \mathbb{E}\left[
# \ln\left(\frac{H_{[1,2]}}{L_{[1,2]}}\right)^2\right].
# $$
#
# The spread estimate follows from
#
# $$
# \alpha = \frac{\sqrt{2\beta}-\sqrt{\beta}}{3-2\sqrt{2}}
# -\sqrt{\frac{\gamma}{3-2\sqrt{2}}}, \qquad
# \widehat{S}_{CS}=\frac{2(e^\alpha-1)}{1+e^\alpha}.
# $$
#
# A day whose price movement swamps the spread can drive $\alpha$ below zero, which would imply a
# negative spread; those estimates are clamped to zero. The derivation and the empirical validation
# are in Corwin, S. A., and Schultz, P., "A Simple Way to Estimate Bid-Ask Spreads from Daily High
# and Low Prices", *Journal of Finance* 67(2), 719-760.

# %% [markdown]
# Roll's estimator starts from a different observation. If a stock's true value were unchanged and
# buy and sell orders arrived at random, the traded price would alternate between the bid and the
# ask - **bid-ask bounce** - and consecutive returns would be negatively correlated purely from
# that alternation. The size of that negative covariance is what the spread has to be for the
# bounce to explain it. The shared implementation uses percentage returns, so its output is a
# relative spread:
#
# $$
# r_t=\frac{P_t}{P_{t-1}}-1, \qquad
# \widehat{S}_{Roll}=2\sqrt{-\operatorname{Cov}(r_t,r_{t-1})}.
# $$
#
# Real returns also carry genuine price discovery, which is positively correlated over short
# horizons and works against the bounce. Where the covariance comes out positive the square root is
# undefined and the estimate is set to zero, which is the first sign that the assumption behind the
# estimator has stopped holding on that sample. Roll, R., "A Simple Implicit Measure of the
# Effective Bid-Ask Spread in an Efficient Market", *Journal of Finance* 39(4), 1127-1139.

# %% [markdown]
# ## 2. A Regular-Session Quote Benchmark
#
# An estimator is only worth using if you know what it does where the answer is checkable. The
# NASDAQ-100 minute bars carry the bid and the ask at every minute close, so on this market the
# spread is observed rather than inferred, and both estimators can be scored against it.
#
# Two things have to be right before the comparison means anything. The raw microstructure feed
# deliberately bypasses the loader's regular-hours filter, so the 09:30-16:00 ET session has to be
# restored before any symbol is selected or any day is aggregated - overnight and pre-market
# minutes trade at far wider spreads and would dominate the average. And the trade panel is kept
# separate from the quote panel, so that discarding a minute for a bad quote cannot also discard a
# legitimate trade that set the day's high or low.


# %%
def load_nq_microstructure() -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Load raw, regular-session, and valid-quote NASDAQ-100 minute panels."""
    raw = load_nasdaq100_bars(
        start_date=NQ100_START_DATE,
        end_date=NQ100_END_DATE,
        include_microstructure=True,
        lazy=True,
    ).select(
        "timestamp",
        "symbol",
        "volume",
        "close_bid_price",
        "close_ask_price",
        "high_trade_price",
        "low_trade_price",
        "first_trade_price",
        "last_trade_price",
    )
    hour = pl.col("timestamp").dt.hour()
    minute = pl.col("timestamp").dt.minute()
    regular = raw.filter(((hour > 9) | ((hour == 9) & (minute >= 30))) & (hour < 16))
    valid_quotes = (
        regular.filter(
            pl.col("close_bid_price").is_not_null()
            & pl.col("close_ask_price").is_not_null()
            & (pl.col("close_bid_price") > 0)
            & (pl.col("close_ask_price") > pl.col("close_bid_price"))
            & (pl.col("volume").fill_null(0) > 0)
        )
        .with_columns(
            midpoint=(pl.col("close_bid_price") + pl.col("close_ask_price")) / 2,
            quoted_spread=pl.col("close_ask_price") - pl.col("close_bid_price"),
        )
        .with_columns(quoted_spread_rel=pl.col("quoted_spread") / pl.col("midpoint"))
    )
    return raw, regular, valid_quotes


# %% [markdown]
# ### Select a Descriptive Liquid Sample
#
# The names are ranked by traded volume over the whole quarter, so the selection uses information
# from the end of the window to decide what to include at the start of it. That is fine here, where
# the sample only has to describe a set of liquid large caps, and would not be fine in a backtest,
# where it would hand the strategy a universe nobody could have picked in advance.

# %%
raw_nq_lf, regular_nq_lf, quote_nq_lf = load_nq_microstructure()

if NQ100_SYMBOLS > 0:
    top_symbols = (
        regular_nq_lf.group_by("symbol")
        .agg(pl.col("volume").sum())
        .sort("volume", descending=True)
        .head(NQ100_SYMBOLS)
        .collect()["symbol"]
        .to_list()
    )
    raw_nq_lf = raw_nq_lf.filter(pl.col("symbol").is_in(top_symbols))
    regular_nq_lf = regular_nq_lf.filter(pl.col("symbol").is_in(top_symbols))
    quote_nq_lf = quote_nq_lf.filter(pl.col("symbol").is_in(top_symbols))

# %% [markdown]
# ### Account for Every Minute the Filters Remove
#
# Each filter above discards rows, and a discarded row that nobody counts is how a benchmark
# quietly becomes a different benchmark. The table below adds up to the raw row count, so every
# exclusion is visible. The quote filter removes minutes with no quote at all, with a non-positive
# bid, with no trade, and two conditions that indicate a stale or crossed book: a **locked** market,
# where the bid equals the ask, and a **crossed** one, where the bid is above the ask. Neither can
# be executed against, so neither describes a real cost of trading.

# %%
raw_rows = raw_nq_lf.select(pl.len()).collect().item()
regular_rows = regular_nq_lf.select(pl.len()).collect().item()
valid_quote_rows = quote_nq_lf.select(pl.len()).collect().item()
regular_keys = regular_nq_lf.select(pl.struct("symbol", "timestamp").n_unique()).collect().item()

quote_integrity = pl.DataFrame(
    {
        "population": [
            "Selected raw minutes",
            "Outside regular session",
            "Regular-session minutes",
            "Invalid/locked/zero-volume quotes",
            "Valid quote minutes",
            "Duplicate keys",
        ],
        "rows": [
            raw_rows,
            raw_rows - regular_rows,
            regular_rows,
            regular_rows - valid_quote_rows,
            valid_quote_rows,
            regular_rows - regular_keys,
        ],
    }
)
quote_integrity

# %% [markdown]
# ### Aggregate Trades and Quotes Independently
#
# The daily bar the estimators consume is built from every regular-session trade minute: the day's
# high is the highest trade, the low the lowest, the open and close the first and last trades. The
# benchmark it is scored against is built only from the valid quote minutes, and each minute's
# relative spread is weighted by the volume that traded in it, so a wide spread quoted while nobody
# was trading does not count as much as a wide spread traded through. Joining the two on symbol and
# date gives the days where both a bar and a benchmark exist.

# %%
nq_trade_daily = (
    regular_nq_lf.filter(
        pl.col("high_trade_price").is_not_null()
        & pl.col("low_trade_price").is_not_null()
        & (pl.col("low_trade_price") > 0)
    )
    .with_columns(date=pl.col("timestamp").dt.date())
    .sort(["symbol", "timestamp"])
    .group_by(["date", "symbol"])
    .agg(
        high=pl.col("high_trade_price").max(),
        low=pl.col("low_trade_price").min(),
        open=pl.col("first_trade_price").drop_nulls().first(),
        close=pl.col("last_trade_price").drop_nulls().last(),
        volume=pl.col("volume").fill_null(0).sum(),
    )
    .filter(
        pl.col("open").is_not_null()
        & pl.col("close").is_not_null()
        & (pl.col("high") > pl.col("low"))
    )
    .collect()
)

# %% tags=["results"]
nq_quote_daily = (
    quote_nq_lf.with_columns(date=pl.col("timestamp").dt.date())
    .group_by(["date", "symbol"])
    .agg(
        quoted_close_spread=(
            (pl.col("quoted_spread_rel") * pl.col("volume")).sum() / pl.col("volume").sum()
        ),
        quote_minutes=pl.len(),
    )
    .collect()
)

nq_daily = nq_trade_daily.join(nq_quote_daily, on=["date", "symbol"], how="inner").sort(
    ["symbol", "date"]
)
if nq_daily.is_empty():
    raise ValueError("No regular-session NASDAQ-100 symbol-days remain after quote validation.")

median_quote_bps = nq_daily["quoted_close_spread"].median() * 10_000
display(
    Markdown(
        f"The benchmark contains **{len(nq_daily):,} symbol-days** across "
        f"**{nq_daily['symbol'].n_unique()} symbols**. Its median volume-weighted minute-close "
        f"quoted spread is **{median_quote_bps:.1f} bps**."
    )
)

# %% [markdown]
# ## 3. Estimating One Symbol at a Time
#
# Both estimators look backwards: Corwin-Schultz reads yesterday's range alongside today's, and
# Roll correlates today's return with yesterday's. In a panel stacked symbol after symbol, the row
# before one company's first day is another company's last, so an unguarded shift would pair one
# firm's return with another's and a rolling window would average across the seam. Wrapping the
# whole expression in `over("symbol")` restarts both at each symbol boundary.


# %%
def estimate_spreads(df: pl.DataFrame) -> pl.DataFrame:
    """Add symbol-isolated Corwin-Schultz and Roll relative-spread estimates."""
    return df.sort(["symbol", "timestamp"]).with_columns(
        cs_spread=corwin_schultz_spread(pl.col("high"), pl.col("low"), window=CS_WINDOW).over(
            "symbol"
        ),
        roll_spread_est=roll_spread(pl.col("close"), window=ROLL_WINDOW).over("symbol"),
    )


# %%
nq_estimator_input = nq_daily.select(
    pl.col("date").cast(pl.Datetime).alias("timestamp"),
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quoted_close_spread",
)
nq_estimated = estimate_spreads(nq_estimator_input).filter(
    pl.col("cs_spread").is_not_null()
    & pl.col("roll_spread_est").is_not_null()
    & pl.col("quoted_close_spread").is_not_null()
)
if nq_estimated.is_empty():
    raise ValueError("The configured windows leave no matched NASDAQ-100 observations.")

nq_validation = (
    nq_estimated.group_by("symbol")
    .agg(
        quoted_bps=pl.col("quoted_close_spread").median() * 10_000,
        cs_bps=pl.col("cs_spread").median() * 10_000,
        roll_bps=pl.col("roll_spread_est").median() * 10_000,
    )
    .sort("quoted_bps")
)
if len(nq_validation) < 2:
    raise ValueError("At least two symbols are required for validation metrics.")

# %% [markdown]
# ### Separate Getting the Order Right from Getting the Level Right
#
# An estimator can be useful in two different ways, and they fail independently. It can **rank**
# instruments correctly - the wider-spread name comes out wider - which is enough to choose between
# two candidates. Or it can get the **level** right, which is what a backtest needs before it can
# subtract the number from a return. Four measures separate the two, all of them taken over the
# per-symbol medians so that one symbol counts once:
#
# - **Pearson $r$** between the estimate and the quoted spread answers the ranking question. It is
#   invariant to scale, so it says nothing about the level.
# - **Mean absolute error** is the typical distance from the quoted spread, in basis points.
# - **Bias** is the average signed error: positive means the estimator reads wider than the market
#   actually quoted.
# - The **identity-line $R^2$**, $1-\sum(y-\hat y)^2/\sum(y-\bar y)^2$, scores the estimate against
#   the 45-degree line rather than against a fitted line. It goes negative as soon as the estimate
#   is further from the truth than the benchmark's own mean would be, which is what makes a level
#   failure impossible to miss. It is not the square of the correlation above.


# %%
def compute_validation_metrics(validation: pl.DataFrame) -> pl.DataFrame:
    """Compute association and identity-line accuracy for both estimators, over symbol medians."""
    observed = validation["quoted_bps"].to_numpy()
    rows = []
    for column, estimator in (("cs_bps", "Corwin-Schultz"), ("roll_bps", "Roll")):
        estimated = validation[column].to_numpy()
        mask = np.isfinite(observed) & np.isfinite(estimated)
        y, y_hat = observed[mask], estimated[mask]
        sst = np.sum((y - y.mean()) ** 2)
        rows.append(
            {
                "estimator": estimator,
                "pearson_r": np.corrcoef(y, y_hat)[0, 1],
                "identity_r2": 1 - np.sum((y - y_hat) ** 2) / sst,
                "mae_bps": np.mean(np.abs(y - y_hat)),
                "bias_bps": np.mean(y_hat - y),
            }
        )
    return pl.DataFrame(rows)


# %% tags=["results"]
validation_metrics = compute_validation_metrics(nq_validation)
metric_rows = {row["estimator"]: row for row in validation_metrics.to_dicts()}
cs_metrics = metric_rows["Corwin-Schultz"]
roll_metrics = metric_rows["Roll"]

display(
    Markdown(
        f"The {len(nq_estimated):,} matched symbol-days reduce to one median per symbol, so all "
        f"four measures below are computed over **{len(nq_validation)} points**. Corwin-Schultz "
        f"has Pearson $r={cs_metrics['pearson_r']:.2f}$, MAE **{cs_metrics['mae_bps']:.1f} bps**, "
        f"and bias **{cs_metrics['bias_bps']:+.1f} bps**. Roll has "
        f"$r={roll_metrics['pearson_r']:.2f}$, MAE **{roll_metrics['mae_bps']:.1f} bps**, and bias "
        f"**{roll_metrics['bias_bps']:+.1f} bps**. Their identity-line $R^2$ values are "
        f"**{cs_metrics['identity_r2']:.1f}** and **{roll_metrics['identity_r2']:.1f}**. A "
        "correlation on that few points is a weak reading on its own; the level measures beside it "
        "are not, because they need no sampling argument to be interpreted."
    )
)

# %% [markdown]
# ### Estimated Versus Quoted Spreads
#
# One point per symbol, the estimate on the vertical axis against the quoted spread on the
# horizontal. A perfectly calibrated estimator would put every point on the dashed 45-degree line;
# points above it read wider than the market quoted and points below read narrower. Both panels
# share axis limits, so how far each estimator sits from the line is comparable by eye.

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True, sharey=True)
plot_specs = (
    (axes[0], "cs_bps", "Corwin-Schultz", COLORS["blue"]),
    (axes[1], "roll_bps", "Roll", COLORS["amber"]),
)
limit = 1.08 * max(
    nq_validation["quoted_bps"].max(),
    nq_validation["cs_bps"].max(),
    nq_validation["roll_bps"].max(),
)

for ax, column, estimator, color in plot_specs:
    metrics = metric_rows[estimator]
    ax.scatter(
        nq_validation["quoted_bps"],
        nq_validation[column],
        color=color,
        alpha=0.75,
        s=28,
    )
    ax.plot([0, limit], [0, limit], color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Quoted close spread (bps)")
    ax.set_ylabel("Estimated spread (bps)")
    direction = "above" if metrics["bias_bps"] > 0 else "below"
    add_message_title(
        ax,
        f"{estimator} symbol medians sit {direction} the quoted spread",
        subtitle="One point per symbol; the dashed line is exact agreement",
    )

fig.tight_layout()
show_with_alt(
    fig,
    "Two scatter panels of estimated against quoted spread, one point per symbol, sharing axes. "
    "Corwin-Schultz points cluster near the low-spread corner slightly above the 45-degree line; "
    "Roll points sit far above it, an order of magnitude away from exact agreement.",
)

# %% [markdown]
# The quoted spread used here is what the market was showing at each minute close. It is not the
# **effective spread**, which measures where a trade actually printed relative to the midpoint, nor
# the **realized spread**, which measures what the liquidity provider kept after the price moved.
# An order large enough to move the market pays more than the quoted spread; an order that rests
# and gets filled passively can pay less.

# %% [markdown]
# ## 4. A Frequency-Aligned Cross-Asset Map
#
# Both estimators count in rows, not in time: a 20-row window is a month on daily bars and a week
# on 8-hour bars. Comparing markets therefore means putting every panel on the same daily grid
# first. Crypto's funding-aligned 8-hour bars are aggregated to UTC calendar days before any window
# is applied, and every other panel is already daily.


# %%
def keep_top_symbols(df: pl.DataFrame, symbol_col: str) -> pl.DataFrame:
    """Restrict a volume-bearing panel to a full-sample liquid subset."""
    if MAX_SYMBOLS <= 0:
        return df
    top = (
        df.group_by(symbol_col)
        .agg(pl.col("volume").mean())
        .sort("volume", descending=True)
        .head(MAX_SYMBOLS)[symbol_col]
    )
    return df.filter(pl.col(symbol_col).is_in(top))


# %% [markdown]
# Crypto perpetuals trade continuously, so there is no session close to define a day. UTC calendar
# days are used because the funding bars are already aligned to them and because any reader can
# reproduce the boundary. The day's open is the first bar's open, its high the highest high, its
# low the lowest low, its close the last bar's close, and its volume the sum of the three.


# %%
def aggregate_crypto_daily(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate funding-aligned crypto bars to UTC calendar-day OHLCV."""
    return (
        df.with_columns(day=pl.col("timestamp").dt.date())
        .sort(["symbol", "timestamp"])
        .group_by(["symbol", "day"])
        .agg(
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            volume=pl.col("volume").sum(),
        )
        .rename({"day": "timestamp"})
        .with_columns(pl.col("timestamp").cast(pl.Datetime))
        .sort(["symbol", "timestamp"])
    )


# %% [markdown]
# ### Assemble the Six Panels
#
# A futures contract expires, so a continuous futures price series is stitched together from
# successive contracts, and the prices are **back-adjusted** by a multiplicative factor so the
# stitch does not appear as a jump. Both estimators are safe on the adjusted series: Corwin-Schultz
# reads the high-low ratio, from which a common factor cancels, and Roll reads percentage returns,
# which the adjustment is designed to leave intact. Volume is not adjusted and is used only to
# choose which products to keep.

# %%
datasets: dict[str, pl.DataFrame] = {}
datasets["ETFs"] = keep_top_symbols(load_etfs(), "symbol")

crypto_8h = keep_top_symbols(load_crypto_perps(frequency="8h"), "symbol")
datasets["Crypto Perps"] = aggregate_crypto_daily(crypto_8h)

datasets["CME Futures"] = (
    keep_top_symbols(load_cme_futures(tenors=[0]), "product")
    .rename({"session_date": "timestamp", "product": "symbol"})
    .with_columns(
        pl.col("adj_open").alias("open"),
        pl.col("adj_high").alias("high"),
        pl.col("adj_low").alias("low"),
        pl.col("adj_close").alias("close"),
    )
    .select("timestamp", "symbol", "open", "high", "low", "close", "volume")
)

# %%
fx = load_fx_pairs(frequency="daily")
if MAX_SYMBOLS > 0:
    fx_symbols = fx["symbol"].unique().sort().head(MAX_SYMBOLS)
    fx = fx.filter(pl.col("symbol").is_in(fx_symbols))
datasets["FX Pairs"] = fx

sp500 = load_sp500_daily_bars()
if "timestamp" not in sp500.columns:
    raise ValueError("S&P 500 loader must provide canonical timestamp keys.")
datasets["S&P 500 Equities"] = keep_top_symbols(sp500, "symbol").select(
    "timestamp", "symbol", "open", "high", "low", "close", "volume"
)

datasets["NASDAQ-100"] = nq_estimator_input.select(
    "timestamp", "symbol", "open", "high", "low", "close", "volume"
)

# %% [markdown]
# ### What Each Panel Actually Contains
#
# Read this table before the chart that follows it. The six panels cover different histories,
# because the licensed and public datasets start and end in different places, and they are trimmed
# to their liquid names in three different ways. Any level difference between two rows of the chart
# is therefore partly a difference between two samples, and the table is what lets a reader see how
# much of it could be.

# %%
inventory_rows = []
for asset_class, panel in datasets.items():
    inventory_rows.append(
        {
            "asset_class": asset_class,
            "frequency": "daily",
            "start": str(panel["timestamp"].min())[:10],
            "end": str(panel["timestamp"].max())[:10],
            "rows": len(panel),
            "symbols": panel["symbol"].n_unique(),
            "selection": "full-sample liquid subset"
            if asset_class not in {"FX Pairs", "NASDAQ-100"}
            else ("alphabetical subset" if asset_class == "FX Pairs" else "validation subset"),
        }
    )

panel_inventory = pl.DataFrame(inventory_rows).sort("asset_class")
panel_inventory

# %% [markdown]
# ### Estimate Symbol-Level Spreads
#
# The same two expressions now run on each daily panel, one symbol at a time. Only the NASDAQ-100
# panel has quotes to check the answer against; on the other five the number is whatever the
# estimator produces, and Section 3 has already shown how far that can sit from an observed spread.
#
# Both estimators can return exactly zero, and for different reasons: Corwin-Schultz when a
# window's volatility swamps the spread and drives $\alpha$ negative, Roll when the return
# covariance comes out positive. A zero is not an estimate of a zero spread; it is the estimator
# saying it could not separate the spread from the price movement. How often that happens is
# recorded alongside the level, because a median taken over a column that is mostly zeros is a
# statement about the estimator, not about the market.

# %% tags=["results"]
spread_results = []
clamp_rows = []
for asset_class, panel in datasets.items():
    estimated = estimate_spreads(panel).filter(
        pl.col("cs_spread").is_not_null() | pl.col("roll_spread_est").is_not_null()
    )
    clamp_rows.append(
        {
            "asset_class": asset_class,
            "sessions": len(estimated),
            "cs_zero_share": estimated.select((pl.col("cs_spread") == 0).mean()).item(),
            "roll_zero_share": estimated.select((pl.col("roll_spread_est") == 0).mean()).item(),
        }
    )
    spread_results.append(
        estimated.group_by("symbol")
        .agg(
            cs_bps=pl.col("cs_spread").median() * 10_000,
            roll_bps=pl.col("roll_spread_est").median() * 10_000,
        )
        .with_columns(asset_class=pl.lit(asset_class))
    )

all_spreads = pl.concat(spread_results)
clamp_summary = pl.DataFrame(clamp_rows)
spread_summary = (
    all_spreads.group_by("asset_class")
    .agg(
        symbols=pl.col("symbol").n_unique(),
        cs_median=pl.col("cs_bps").median(),
        cs_p25=pl.col("cs_bps").quantile(0.25),
        cs_p75=pl.col("cs_bps").quantile(0.75),
        roll_median=pl.col("roll_bps").median(),
    )
    .join(clamp_summary, on="asset_class")
    .sort("cs_median")
)

most_clamped = clamp_summary.sort("cs_zero_share", descending=True).row(0, named=True)
largest_cs = spread_summary.sort("cs_median", descending=True).row(0, named=True)
display(
    Markdown(
        f"The aligned map contains **{len(all_spreads)} symbol-level estimates**. The largest "
        f"median Corwin-Schultz output is **{largest_cs['asset_class']} "
        f"({largest_cs['cs_median']:.1f} bps)**. At the other end, Corwin-Schultz returns exactly "
        f"zero on **{most_clamped['cs_zero_share']:.0%}** of "
        f"**{most_clamped['asset_class']}** sessions, so that market's median is zero and says "
        "nothing about its spread."
    )
)

# %% [markdown]
# ### Estimator Choice Changes the Level, and Sometimes There Is No Level
#
# The upper panel shows both estimators on the same sample, joined by a line so the gap between
# them is what the eye lands on. The bar through the Corwin-Schultz marker is the interquartile
# range of that market's symbol-level estimates: it says how much the instruments within a market
# differ from one another, and is not a confidence interval.
#
# The lower panel is what makes the upper one readable. It shows how often each estimator returned
# zero on that market. Where that share is high, the marker above it is not a spread estimate that
# happens to be small - it is mostly the clamp.

# %%
summary_pd = spread_summary.to_pandas()
y = np.arange(len(summary_pd))
fig, (ax, ax_zero) = plt.subplots(
    2,
    1,
    figsize=FIGSIZE["dual_v"],
    gridspec_kw={"height_ratios": [2, 1]},
)

for idx, row in summary_pd.iterrows():
    ax.hlines(
        y[idx],
        row["cs_median"],
        row["roll_median"],
        color=COLORS["silver_muted"],
        linewidth=2,
        zorder=1,
    )
    ax.errorbar(
        row["cs_median"],
        y[idx],
        xerr=[
            [max(row["cs_median"] - row["cs_p25"], 0)],
            [max(row["cs_p75"] - row["cs_median"], 0)],
        ],
        fmt="o",
        color=COLORS["blue"],
        capsize=3,
        label="Corwin-Schultz median and IQR" if idx == 0 else None,
        zorder=3,
    )
    ax.scatter(
        row["roll_median"],
        y[idx],
        marker="s",
        color=COLORS["amber"],
        label="Roll median" if idx == 0 else None,
        zorder=3,
    )

ax.set_yticks(y, summary_pd["asset_class"])
ax.set_xlim(left=0)
ax.set_xlabel("Estimated relative spread (bps)")
ax.set_ylabel("")
add_message_title(
    ax,
    "Estimator choice changes cross-asset spread levels",
    subtitle="20-session windows on daily liquid samples; lines connect paired medians",
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

bar_height = 0.38
ax_zero.barh(
    y - bar_height / 2,
    summary_pd["cs_zero_share"],
    height=bar_height,
    color=COLORS["blue"],
    label="Corwin-Schultz",
)
ax_zero.barh(
    y + bar_height / 2,
    summary_pd["roll_zero_share"],
    height=bar_height,
    color=COLORS["amber"],
    label="Roll",
)
ax_zero.set_yticks(y, summary_pd["asset_class"])
ax_zero.set_xlim(0, 1)
ax_zero.set_xlabel("Share of sessions the estimator returned zero")
add_message_title(
    ax_zero,
    "On the equity panels the estimate is zero most of the time",
    subtitle="A zero means the window's volatility hid the spread, not that the spread was zero",
)
fig.tight_layout()
show_with_alt(
    fig,
    "Upper panel: one row per market with a circle for the Corwin-Schultz median, a square for "
    "the Roll median, and a line joining them; Roll sits to the right on every row. Lower panel: "
    "paired horizontal bars of how often each estimator returned zero, near the full width for "
    "Corwin-Schultz on the two equity panels and far lower elsewhere.",
)

# %% [markdown]
# ## 5. Which Configured Costs Are Even Spreads
#
# The spread this notebook estimates is one number: a fraction of price, applied to a round trip.
# The nine case studies each record a cost assumption, and almost none of them record that number.
# Some record a fee, which is charged in addition to the spread. Some record an all-in cost, which
# already contains the spread along with commission and impact. Some record dollars per share or
# ticks per contract, which are not fractions of anything until a price is supplied.
#
# The classifier below decides each row from the keys actually present in that case study's
# `costs:` block, so a configuration change moves the table rather than leaving it stale.


# %%
def classify_cost_units(costs: dict) -> tuple[str, bool, str]:
    """Return the unit a costs block states, whether it is a spread in bps, and what is missing."""
    if "spread_bps" in costs:
        return (
            "spread in bps, as a range per pair class",
            False,
            "one rate per pair, and whether the range is a half or a full spread",
        )
    if "fee_schedule" in costs:
        return (
            "exchange fee in bps per side",
            False,
            "a spread; the configured figure is a fee charged on top of one",
        )
    if "round_trip_cost_bps" in costs:
        return (
            "all-in round trip in bps",
            False,
            "the spread's share of an all-in figure that also holds commission and impact",
        )
    if "per_leg_cost_bps_range" in costs:
        return (
            "all-in cost per leg in bps, as a range",
            False,
            "the spread's share of the range, and a point inside it",
        )
    if costs.get("model") == "per_share_plus_spread":
        return (
            "USD per share commission plus a USD half spread",
            False,
            "the share price, to turn dollars into a fraction of it",
        )
    if "commission_per_contract" in costs:
        return (
            "USD per contract plus a spread counted in ticks",
            False,
            "each product's tick size and price, to turn ticks into a fraction",
        )
    if isinstance(costs.get("components"), dict):
        return (
            "percent of option premium, bps of hedge notional, and per-contract fees",
            False,
            "a single cost base; the components are quoted against three different ones",
        )
    return ("unrecognized", False, "a rule for this schema")


# %%
case_studies = [
    "etfs",
    "crypto_perps_funding",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "us_firm_characteristics",
    "fx_pairs",
    "cme_futures",
    "sp500_options",
    "us_equities_panel",
]

cost_rows = []
for case_study in case_studies:
    setup_path = get_case_study_source_dir(case_study) / "config" / "setup.yaml"
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing required setup: {case_study}")
    costs = yaml.safe_load(setup_path.read_text()).get("costs", {})
    if not costs:
        raise ValueError(f"Missing costs configuration: {case_study}")
    native_unit, is_spread_bps, missing = classify_cost_units(costs)
    cost_rows.append(
        {
            "case_study": case_study,
            "native_representation": native_unit,
            "is_full_spread_bps": "Yes" if is_spread_bps else "No",
            "still_needed": missing,
        }
    )

cost_inventory = pl.DataFrame(cost_rows)
unrecognized = cost_inventory.filter(pl.col("native_representation") == "unrecognized")
if not unrecognized.is_empty():
    raise ValueError(f"Unclassified cost schemas: {unrecognized['case_study'].to_list()}")
cost_inventory

# %% tags=["results"]
comparable_count = cost_inventory.filter(pl.col("is_full_spread_bps") == "Yes").height
display(
    Markdown(
        f"All **{len(cost_inventory)}** configurations were read and classified, and "
        f"**{comparable_count}** of them state a full spread in basis points. Each row's "
        "`still_needed` column names what would have to be supplied before that case study's cost "
        "assumption and this notebook's estimate could be placed on the same axis."
    )
)

# %% [markdown]
# ## 6. Does the Estimate Move with Market Stress
#
# A cost assumption fixed at one number treats a calm day and a crisis day alike, which is the
# mistake Section 18.3 of the chapter is about. The VIX - the market's expectation of S&P 500
# volatility over the next month, read off option prices and quoted in annualized percentage points
# - is the standard measure of how stressed the market is, so it is the natural variable to
# condition on.
#
# The four states below are the quartiles of the VIX over its whole history, which means the
# boundaries are set using every observation including the future ones. That is fine for a
# description of what already happened and would be look-ahead in a strategy that had to decide,
# on a given day, which state it was in.
#
# One thing to keep in mind while reading this section: Corwin-Schultz reads the high-low range,
# and the high-low range widens when the price moves more. Some of what follows is therefore the
# estimator responding to volatility rather than evidence that spreads themselves widened. Section
# 3's benchmark is what would settle that, and it exists only for one quarter of one market.

# %%
vix = (
    load_macro(series=["vixcls"])
    .select(
        pl.col("timestamp").cast(pl.Date),
        pl.col("vixcls").alias("vix"),
    )
    .filter(pl.col("vix").is_not_null())
    .sort("timestamp")
)
vix_q = tuple(vix["vix"].quantile(q) for q in (0.25, 0.50, 0.75))

etf_with_spreads = estimate_spreads(datasets["ETFs"]).filter(pl.col("cs_spread").is_not_null())
etf_vix = (
    etf_with_spreads.join(
        vix,
        left_on=pl.col("timestamp").cast(pl.Date),
        right_on="timestamp",
        how="inner",
    )
    .with_columns(cs_bps=pl.col("cs_spread") * 10_000)
    .with_columns(
        vix_regime=pl.when(pl.col("vix") < vix_q[0])
        .then(pl.lit("Q1 (Low)"))
        .when(pl.col("vix") < vix_q[1])
        .then(pl.lit("Q2"))
        .when(pl.col("vix") < vix_q[2])
        .then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4 (High)"))
    )
)

# %% tags=["results"]
regime_order = {"Q1 (Low)": 1, "Q2": 2, "Q3": 3, "Q4 (High)": 4}
regime_summary = (
    etf_vix.group_by("vix_regime")
    .agg(
        mean_bps=pl.col("cs_bps").mean(),
        positive_share=(pl.col("cs_bps") > 0).mean(),
        mean_when_positive_bps=pl.col("cs_bps").filter(pl.col("cs_bps") > 0).mean(),
        observations=pl.len(),
    )
    .with_columns(order=pl.col("vix_regime").replace_strict(regime_order))
    .sort("order")
)

low_regime = regime_summary.filter(pl.col("vix_regime") == "Q1 (Low)")
high_regime = regime_summary.filter(pl.col("vix_regime") == "Q4 (High)")
low_mean, high_mean = low_regime["mean_bps"].item(), high_regime["mean_bps"].item()
low_share, high_share = (
    low_regime["positive_share"].item(),
    high_regime["positive_share"].item(),
)
low_level, high_level = (
    low_regime["mean_when_positive_bps"].item(),
    high_regime["mean_when_positive_bps"].item(),
)
display(
    Markdown(
        f"The full-sample VIX boundaries are **{vix_q[0]:.1f}**, **{vix_q[1]:.1f}**, and "
        f"**{vix_q[2]:.1f}**. Mean ETF Corwin-Schultz output moves from **{low_mean:.2f} bps** "
        f"in the calmest quartile to **{high_mean:.2f} bps** in the most stressed. That rise is "
        "not the estimator producing an answer more often: it returns a positive estimate on "
        f"**{low_share:.1%}** of calm-quartile sessions and **{high_share:.1%}** of stressed "
        "ones, which is essentially the same rate. What changes is the size of the estimate on "
        f"the sessions that produce one, from **{low_level:.1f} bps** to **{high_level:.1f} bps**."
    )
)

# %% [markdown]
# ### The Level Rises; How Often an Estimate Appears Does Not
#
# The mean over every session mixes two different things: how often the estimator produced any
# estimate at all, and how large it was when it did. Separating them says which one the rise in
# the mean is made of, and only one of the two answers is about spreads. A rising rate would say
# volatility had stopped hiding the spread inside the high-low range on more days. A rising level
# says that where the estimator does read a spread, it reads a wider one.

# %%
regime_pd = regime_summary.to_pandas()
x = np.arange(len(regime_pd))
fig, (ax_rate, ax_level) = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

ax_rate.bar(x, regime_pd["positive_share"], color=COLORS["blue"])
ax_rate.set_ylabel("Share of sessions with an estimate")
ax_rate.set_ylim(bottom=0)
add_message_title(
    ax_rate,
    "The share of sessions producing any estimate barely moves with VIX",
    subtitle="A session with no estimate is one where volatility swamped the high-low range",
)

ax_level.bar(x, regime_pd["mean_when_positive_bps"], color=COLORS["amber"])
ax_level.set_xticks(x, regime_pd["vix_regime"])
ax_level.set_ylim(bottom=0)
ax_level.set_xlabel("Full-sample VIX quartile")
ax_level.set_ylabel("Mean estimate when positive (bps)")
add_message_title(
    ax_level,
    "On the sessions that do produce one, the estimate more than doubles",
    subtitle="ETF daily panel; quartile boundaries set on the whole VIX history",
)
fig.tight_layout()
show_with_alt(
    fig,
    "Two stacked bar panels across the four VIX quartiles. The upper panel, the share of sessions "
    "on which Corwin-Schultz returns a positive estimate, is close to flat across all four. The "
    "lower panel, the mean estimate on those sessions, rises steadily and is more than twice as "
    "large in the most stressed quartile as in the calmest.",
)

# %% [markdown]
# ### Keep VIX and Spread Estimates on Separate Axes
#
# The two series are drawn one above the other on a shared time axis rather than on two vertical
# scales in one frame. Two scales can be stretched until any pair of series appears to move
# together, and stacked panels leave the reader to judge the timing themselves.

# %%
etf_ts = (
    etf_vix.group_by(pl.col("timestamp").cast(pl.Date).alias("timestamp"))
    .agg(mean_spread_bps=pl.col("cs_bps").mean())
    .sort("timestamp")
    .join(vix, on="timestamp", how="inner")
)
etf_ts_pd = etf_ts.to_pandas()
daily_association = etf_ts.select(pl.corr("mean_spread_bps", "vix")).item()

fig, (ax_spread, ax_vix) = plt.subplots(
    2,
    1,
    figsize=FIGSIZE["dual_v"],
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)
ax_spread.plot(
    etf_ts_pd["timestamp"],
    etf_ts_pd["mean_spread_bps"],
    color=COLORS["blue"],
    linewidth=1,
)
ax_spread.set_ylabel("Mean CS estimate (bps)")
add_message_title(
    ax_spread,
    "The estimate spikes in the same weeks the VIX does",
    subtitle="Cross-sectional mean across the ETF panel, one point per session",
)

ax_vix.plot(etf_ts_pd["timestamp"], etf_ts_pd["vix"], color=COLORS["amber"], linewidth=1)
ax_vix.set_ylabel("VIX (index points)")
ax_vix.set_xlabel("Date")
ax_vix.xaxis.set_major_locator(mdates.YearLocator(3))
ax_vix.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
show_with_alt(
    fig,
    "Two stacked time series over the ETF sample: the daily cross-sectional mean Corwin-Schultz "
    "estimate above, the VIX below, on a shared date axis. Both are quiet for long stretches and "
    "spike together in the same episodes.",
)

# %% tags=["results"]
cs_lead = spread_summary.sort("cs_median", descending=True).row(0, named=True)
display(
    Markdown(
        f"Corwin-Schultz reads **{cs_metrics['bias_bps']:+.1f} bps** and Roll "
        f"**{roll_metrics['bias_bps']:+.1f} bps** against the quoted benchmark. Across the six "
        f"aligned panels, **{cs_lead['asset_class']}** carries the largest median Corwin-Schultz "
        f"output at **{cs_lead['cs_median']:.1f} bps**, while on "
        f"**{most_clamped['asset_class']}** the same estimator returns zero on "
        f"**{most_clamped['cs_zero_share']:.0%}** of sessions. Over the ETF panel the estimate "
        f"appears about as often in stressed states as in calm ones (**{high_share:.1%}** against "
        f"**{low_share:.1%}** of sessions) and is more than twice as large when it does, "
        f"**{high_level:.1f} bps** against **{low_level:.1f} bps**. Of the nine case-study "
        f"configurations, **{comparable_count}** state a full spread in basis points."
    )
)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Check an estimator where the answer is observable before trusting it where it is not.** The
#    same two lines of code run on every market in this notebook, and only one of those markets
#    could say whether the output resembled a real spread. Build that check first, and let what it
#    finds set how much weight the rest of the numbers carry.
#
# 2. **Score ranking and level separately, because they fail separately.** An estimator that ranks
#    instruments usefully can be off by an order of magnitude in level, which is fine for choosing
#    between two candidates and fatal for a backtest that subtracts the number from a return. A
#    correlation will not reveal that; an error against the 45-degree line will.
#
# 3. **Read what an estimator assumes, then ask whether your data satisfies it.** Roll needs the
#    bid-ask bounce to be the dominant source of return autocorrelation. At daily frequency on
#    liquid large caps it is not, genuine price discovery swamps it, and the estimator degrades in
#    exactly the way its derivation says it should.
#
# 4. **Put panels on a common grid before comparing them.** Both estimators count rows, not time,
#    so a 20-row window means something different on 8-hour bars than on daily ones. Aggregate
#    first, then estimate.
#
# 5. **Do not compare two cost figures until you have checked they are the same kind of quantity.**
#    A fee, an all-in cost, a dollar-per-share commission and a spread are four different things,
#    and the arithmetic that puts them on one axis will run without complaint on any of them.
#
# 6. **Check how often an estimator returns nothing before you average its output.** Both
#    estimators clamp to zero when their assumption fails, and a median over a column that is
#    mostly zeros reads as a narrow spread when it is really a silent estimator. On the two equity
#    panels here that share is high enough to make the median meaningless.
#
# 7. **Condition the cost assumption on market state, and decompose the conditioning before
#    believing it.** An average taken over every session confounds how often a measurement exists
#    with how large it is. Splitting the two here shows the rise with volatility is entirely in
#    the level, which is the half that is about spreads.
#
# ### Known limitations
#
# - The validation rests on twelve symbols in one quarter of one market. It supports a statement
#   about level calibration on liquid US large caps and nothing broader.
# - The benchmark is the quoted spread at each minute close. It is neither the effective spread a
#   trade actually paid nor the realized spread the liquidity provider kept.
# - The six panels cover different histories and are trimmed to their liquid names in three
#   different ways, so a level difference between two of them is partly a sample difference.
# - The VIX quartile boundaries use the whole history, so the states are a description of what
#   happened rather than something a strategy could have known at the time.
# - Corwin-Schultz reads the high-low range, which widens with volatility on its own. The
#   volatility conditioning in Section 6 cannot separate that from a genuine widening of spreads.
# - On the ETF and S&P 500 panels the Corwin-Schultz estimate is zero on most sessions, so those
#   two rows of the cross-asset chart carry a clamp rather than a level. The chart records the
#   share; it does not repair it.
#
# **Next:** `03_market_impact_calibration` estimates the impact component, which is the part of
# cost the spread does not cover and the part that grows with order size.
#
# **Book:** Chapter 18, Section 18.3.
