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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Panel Features
#
# **Chapter 9 | Section 9.6**
#
# **Docker image**: `ml4t`
#
# Every feature so far has been computed from one series. This notebook computes features that
# need two or more, and they come in two shapes that have almost nothing to do with each other
# beyond both requiring a panel.
#
# A **pairwise** feature reads two price series and asks whether a combination of them is
# stationary when neither is. That is the cointegration question, and its output is a spread, a
# hedge ratio, and a speed of reversion.
#
# A **cross-sectional** feature reads one quantity across many assets on the same date and
# replaces its level with its position among them. An annualized volatility of a quarter means
# one thing for a Treasury fund and another for an energy fund, and a rank says which of the
# two a reader is looking at without needing to know.
#
# **Learning objectives**
#
# - Test a pair for cointegration two ways and read what it means when the two disagree.
# - Estimate a hedge ratio that changes over time with a Kalman filter, and see where it
#   differs from the single number a full-sample regression gives.
# - Estimate a mean-reversion half-life on a first block and use it to size the window a
#   trading signal is computed over, so the window length is not chosen with future data.
# - Convert a temporal feature into a cross-sectional rank and a benchmark-relative value, and
#   say which of the two a downstream model wants.
# - Aggregate a per-asset regime measure across a universe without ranking each asset against
#   its own future.
#
# **Book reference**
#
# Chapter 9, Section 9.6 (Cross-sectional and panel features).
#
# **Prerequisites**
#
# `04_kalman_filter` for the recursion the hedge ratio uses. `11_hmm_regimes` for the regime
# measures the last section aggregates.

# %% [markdown]
# ## Setup

# %%
"""Panel features - cointegrated pairs, cross-sectional ranks and universe aggregates."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ml4t.engineer.features.cross_asset import (
    beta_to_market,
    co_integration_score,
    correlation_regime_indicator,
    rolling_correlation,
)
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from data import load_etfs
from utils.style import COLORS, FIGSIZE, show_with_alt

with warnings.catch_warnings():
    # filterpy carries an invalid escape sequence in a docstring, which Python reports the first
    # time it byte-compiles the module and never again. Nothing about the filter is affected.
    warnings.simplefilter("ignore", SyntaxWarning)
    from filterpy.kalman import KalmanFilter

# %% tags=["parameters"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
MAX_SYMBOLS = 0  # zero loads every symbol in the universe
SIGNIFICANCE = 0.05
TRAIN_FRACTION = 0.5
ENTRY_THRESHOLD = 2.0
MINIMUM_LOOKBACK = 20
MINIMUM_SESSIONS = 252
CORRELATION_WINDOW = 60
COINTEGRATION_WINDOW = 120
FEATURE_WINDOW = 60
REGIME_WINDOW = 252
SESSIONS_PER_YEAR = 252

# %% [markdown]
# ## The universe
#
# The ETF universe, because a pair needs two series whose relationship has a reason behind it
# and exchange-traded funds come with the reason written on them. The pair used through the
# first half is an energy sector fund against a crude oil fund, and the fallback pair for a
# reduced fixture is two broad equity funds.

# %%
universe = load_etfs().filter(
    pl.col("timestamp").is_between(pl.lit(START_DATE).str.to_date(), pl.lit(END_DATE).str.to_date())
)
if MAX_SYMBOLS > 0:
    kept = universe["symbol"].unique().sort().to_list()[:MAX_SYMBOLS]
    universe = universe.filter(pl.col("symbol").is_in(kept))

available = set(universe["symbol"].unique().to_list())
DEPENDENT, INDEPENDENT = ("XLE", "USO") if {"XLE", "USO"} <= available else ("SPY", "QQQ")


def pair_frame(dependent: str, independent: str) -> pd.DataFrame:
    """Closing prices of two symbols on the sessions both traded, indexed by date."""
    columns = []
    for symbol, name in ((dependent, "dependent"), (independent, "independent")):
        columns.append(
            universe.filter(pl.col("symbol") == symbol)
            .select(["timestamp", "close"])
            .rename({"close": name})
            .sort("timestamp")
        )
    joined = columns[0].join(columns[1], on="timestamp", how="inner").drop_nulls()
    frame = joined.to_pandas().set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


pair = pair_frame(DEPENDENT, INDEPENDENT)
train_end = int(len(pair) * TRAIN_FRACTION)

print(f"Universe: {len(available)} symbols, {universe.height:,} rows")
print(f"Pair: {DEPENDENT} against {INDEPENDENT}, {len(pair):,} shared sessions")
print(f"Sessions used to estimate before any signal is taken: {train_end:,}")
display(pair.tail(3))

# %% [markdown]
# ## Two tests for one question
#
# Two price series are **cointegrated** when some fixed combination of them is stationary even
# though neither of them is. That is a stronger statement than correlation and a different one:
# two series can move together every day and drift apart without limit over years, and it is
# the drift that decides whether a spread between them comes back.
#
# The two standard tests reach the question differently. **Engle-Granger** regresses one series
# on the other and tests the residual for a unit root, so it takes one of the two as the
# dependent variable and gives a different answer if the roles are swapped. **Johansen** treats
# the pair as a system and tests how many stationary combinations exist, which is symmetric in
# the two series and extends past two of them.
#
# Both are run on the whole sample here, and that is the correct thing for what they are being
# used for. Whether a pair is worth trading is a question asked before any trading, out of the
# history available at that point; the answer is not a feature and does not enter a design
# matrix. What must not use the whole sample is anything the signal below reads.

# %%
statistic, p_value, _ = coint(pair["dependent"], pair["independent"], trend="c")[:3]

johansen = coint_johansen(pair[["dependent", "independent"]].to_numpy(), det_order=0, k_ar_diff=1)
trace_statistics = johansen.lr1
critical_values = johansen.cvt[:, 1]
johansen_rejects = bool(trace_statistics[0] > critical_values[0])
johansen_vector = johansen.evec[:, 0]

display(
    pd.DataFrame(
        [
            {
                "test": "Engle-Granger",
                "statistic": statistic,
                "threshold": np.nan,
                "p value": p_value,
                "rejects no cointegration": bool(p_value < SIGNIFICANCE),
            },
            {
                "test": "Johansen, no stationary combination",
                "statistic": trace_statistics[0],
                "threshold": critical_values[0],
                "p value": np.nan,
                "rejects no cointegration": johansen_rejects,
            },
            {
                "test": "Johansen, at most one",
                "statistic": trace_statistics[1],
                "threshold": critical_values[1],
                "p value": np.nan,
                "rejects no cointegration": bool(trace_statistics[1] > critical_values[1]),
            },
        ]
    ).set_index("test")
)

print(
    f"Hedge ratio implied by the leading Johansen vector: {-johansen_vector[1] / johansen_vector[0]:.4f}"
)

# %% [markdown]
# This pair is the lesson rather than the demonstration. An energy sector fund and a crude oil
# fund are related by something a reader can state in a sentence, and neither test calls them
# cointegrated over this sample. The reason is visible in the funds: one holds equity in companies
# whose earnings depend on the oil price, the other holds futures and pays to roll them, and a
# roll cost accumulates as a drift that no fixed combination removes.
#
# The implied hedge ratio printed above is a good place to see what a rejected test means. It is
# far from the regression's, and on a pair where no stationary combination is found the leading
# eigenvector is not estimating one: it is the least badly behaved direction in a system that has
# no well behaved one, and its ratio is not a number to hold a position on.
#
# The mechanics below are worth working through anyway, because every step still computes and
# every step's output says something about a pair that fails.

# %% [markdown]
# ## Two hedge ratios
#
# The spread needs a ratio: how many units of one series to hold against one unit of the other.
# A regression over the whole sample gives one number and a filter gives a series, and the
# difference between them is not a matter of preference.
#
# The regression is fitted on everything, which makes its ratio unavailable at every date
# except the last. It is reported because it is the number the cointegration test used, and its
# spread is what the stationarity test below is run on.
#
# The **Kalman filter** treats the intercept and the ratio as a two-dimensional state following
# a random walk and updates it one session at a time. Its estimate at session $t$ has seen
# sessions up to $t$ and no more, which is what a hedge ratio has to be if a position is taken
# on it. The two parameters that decide its behavior are the measurement noise and the process
# noise, and their ratio is the whole tuning: a larger process noise lets the ratio move faster
# and tracks a genuine structural change sooner, at the cost of chasing noise.

# %%
MEASUREMENT_NOISE = 1e-3
PROCESS_NOISE = 1e-5

regression = LinearRegression().fit(pair[["independent"]], pair["dependent"])
static_ratio = float(regression.coef_[0])
pair["spread_static"] = pair["dependent"] - static_ratio * pair["independent"]

adf_statistic, adf_p_value = adfuller(pair["spread_static"], autolag="AIC")[:2]


def kalman_hedge_ratio(
    dependent: np.ndarray, independent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Filter an intercept and a hedge ratio forward, one session at a time."""
    filter_ = KalmanFilter(dim_x=2, dim_z=1)
    filter_.F = np.eye(2)
    filter_.x = np.array([[0.0], [1.0]])
    filter_.P = np.eye(2)
    filter_.R = np.array([[MEASUREMENT_NOISE]])
    filter_.Q = np.eye(2) * PROCESS_NOISE

    intercepts = np.empty(len(dependent))
    ratios = np.empty(len(dependent))
    for step in range(len(dependent)):
        filter_.H = np.array([[1.0, independent[step]]])
        filter_.predict()
        filter_.update(np.array([[dependent[step]]]))
        intercepts[step] = filter_.x[0, 0]
        ratios[step] = filter_.x[1, 0]
    return ratios, intercepts


pair["hedge_ratio"], _ = kalman_hedge_ratio(
    pair["dependent"].to_numpy(), pair["independent"].to_numpy()
)
pair["spread"] = pair["dependent"] - pair["hedge_ratio"] * pair["independent"]

print(f"Hedge ratio from a regression on the whole sample: {static_ratio:.4f}")
print(
    f"Its spread, augmented Dickey-Fuller: statistic {adf_statistic:.4f}, p value {adf_p_value:.4f}"
)
print(
    f"Filtered hedge ratio: from {pair['hedge_ratio'].min():.4f} to "
    f"{pair['hedge_ratio'].max():.4f}, ending at {pair['hedge_ratio'].iloc[-1]:.4f}"
)

# %% [markdown]
# ## How fast the spread comes back
#
# Fit an autoregression of order one to the spread and the coefficient says what fraction of a
# deviation is still there one session later. Written as a change,
#
# $$\Delta s_t = \alpha + \phi\, s_{t-1} + \varepsilon_t$$
#
# a negative $\phi$ means the spread is pulled back toward its level, and the **half-life** is
# the number of sessions over which half of a deviation is expected to have gone:
# $-\ln 2 / \ln(1 + \phi)$. A non-negative $\phi$ says the spread is not being pulled anywhere
# and there is no half-life to report.
#
# The half-life is used below to choose the window the trading signal's mean and standard
# deviation are computed over, which is why it is estimated on the first `TRAIN_FRACTION` of the
# sample only. A window length chosen from the whole sample is a small leak and an easy one to
# miss, because a length does not look like a parameter.


# %%
def estimate_half_life(spread: pd.Series) -> float:
    """Sessions over which half a deviation decays, from an autoregression of order one."""
    level = spread.dropna()
    change = level.diff().dropna()
    lagged = level.shift(1).dropna().loc[change.index]

    coefficient = float(
        LinearRegression().fit(lagged.to_numpy().reshape(-1, 1), change.to_numpy()).coef_[0]
    )
    if coefficient >= 0.0:
        return float("inf")
    return float(-np.log(2.0) / np.log1p(coefficient))


half_life_train = estimate_half_life(pair["spread"].iloc[:train_end])
half_life_full = estimate_half_life(pair["spread"])
LOOKBACK = max(int(2 * half_life_train), MINIMUM_LOOKBACK)

print(f"Half-life on the first block: {half_life_train:.1f} sessions")
print(f"Half-life on the whole sample, for comparison only: {half_life_full:.1f} sessions")
print(f"Signal window, twice the first block's half-life: {LOOKBACK} sessions")

# %% [markdown]
# The two half-lives differ, and the gap between them is how far the window length would have
# moved. Neither is more correct as an estimate. The first is the only one available at the
# moment the window has to be chosen.

# %% [markdown]
# ## From spread to position
#
# The signal is the spread measured against its own recent history: subtract a rolling mean,
# divide by a rolling standard deviation, and the result is in units of its own variability. A
# spread `ENTRY_THRESHOLD` standard deviations below its recent mean is an entry on the long
# side, meaning long the dependent series and short `hedge_ratio` units of the other, and the
# position is held until the spread crosses back through the mean.
#
# Both the mean and the standard deviation are rolling and both use the window chosen above, so
# every value is available on its own session. The position is stepped forward one session at a
# time rather than computed with a vector operation, because it depends on its own previous
# value: an entry that has not yet reverted stays open.

# %%
rolling_mean = pair["spread"].rolling(LOOKBACK).mean()
rolling_deviation = pair["spread"].rolling(LOOKBACK).std()
pair["spread_mean"] = rolling_mean
pair["upper_band"] = rolling_mean + ENTRY_THRESHOLD * rolling_deviation
pair["lower_band"] = rolling_mean - ENTRY_THRESHOLD * rolling_deviation
pair["z_score"] = (pair["spread"] - rolling_mean) / rolling_deviation

scores = pair["z_score"].to_numpy()
position = np.zeros(len(pair), dtype=np.int64)
held = 0
for step in range(1, len(pair)):
    if step < train_end:
        continue
    if scores[step] < -ENTRY_THRESHOLD:
        held = 1
    elif scores[step] > ENTRY_THRESHOLD:
        held = -1
    elif held * scores[step] > 0:
        held = 0
    position[step] = held

pair["position"] = position

held_positions = pair["position"].to_numpy()
previous_positions = np.concatenate([[0], held_positions[:-1]])
entries = int(((held_positions != previous_positions) & (held_positions != 0)).sum())

print(f"Sessions with a position open: {int((held_positions != 0).sum()):,}")
print(f"Entries, counting a reversal as a new one: {entries:,}")

# %%
fig, axes = plt.subplots(4, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(pair.index, pair["dependent"], linewidth=0.8, color=COLORS["blue"], label=DEPENDENT)
ax.plot(pair.index, pair["independent"], linewidth=0.8, color=COLORS["copper"], label=INDEPENDENT)
ax.set_ylabel("US dollars")
ax.set_title("Two funds with a reason to move together", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(pair.index, pair["hedge_ratio"], linewidth=0.9, color=COLORS["blue"], label="filtered")
ax.axhline(static_ratio, color=COLORS["amber"], linestyle="--", linewidth=0.9, label="whole sample")
ax.set_ylabel("Units held short")
ax.set_title("The filtered hedge ratio moves; the regression cannot", fontsize=9)
ax.legend(fontsize=7)

ax = axes[2]
ax.plot(pair.index, pair["spread"], linewidth=0.8, color=COLORS["blue"])
ax.plot(pair.index, pair["spread_mean"], linewidth=0.8, color=COLORS["amber"], linestyle="--")
ax.fill_between(
    pair.index, pair["lower_band"], pair["upper_band"], alpha=0.2, color=COLORS["silver_muted"]
)
ax.set_ylabel("US dollars")
ax.set_title("The spread against a band of its own recent variability", fontsize=9)

ax = axes[3]
ax.plot(pair.index, pair["z_score"], linewidth=0.8, color=COLORS["blue"])
for level in (-ENTRY_THRESHOLD, ENTRY_THRESHOLD):
    ax.axhline(level, color=COLORS["negative"], linestyle="--", linewidth=0.7)
ax.axhline(0, color=COLORS["recede"], linewidth=0.7)
ax.axvline(pair.index[train_end], color=COLORS["neutral"], linestyle=":", linewidth=0.9)
ax.set_ylabel("Standard deviations")
ax.set_xlabel("Session")
ax.set_title("Entries at the dashed lines, exits at the crossing", fontsize=9)

fig.suptitle(f"{DEPENDENT} against {INDEPENDENT}: spread, band and signal")
show_with_alt(
    fig,
    "Four stacked panels for one pair of funds. The top plots both price series. The second "
    "plots the filtered hedge ratio as a moving line against a flat dashed line for the "
    "whole-sample regression, and the two are far apart for long stretches. The third plots the "
    "spread with a shaded band around a dashed rolling mean. The bottom plots the spread's "
    "standardized deviation with dashed entry lines above and below zero and a dotted vertical "
    "line where the estimation block ends.",
)

# %% [markdown]
# ## What the position would have returned
#
# What was held over session $t$ is one share of the dependent fund against $\beta_{t-1}$ shares
# of the other, because $\beta_{t-1}$ is the ratio the filter had produced by the previous close.
# Its profit in dollars is the two price changes weighted by those holdings:
#
# $$\Delta P_t - \beta_{t-1}\, \Delta Q_t$$
#
# and to express it as a return it is divided by what holding both legs committed, which is the
# value of one share of the first plus $|\beta_{t-1}|$ shares of the second.
#
# That is deliberately not the first difference of the `spread` column. The spread at $t$ is
# defined with $\beta_t$, so differencing it mixes the session's price moves with the change in
# the ratio itself, and a holder cannot rebalance at a ratio they only learn at that session's
# close. The two agree only where the filtered ratio is flat.
#
# It is also not the difference of the two funds' returns. That quantity is the profit of equal
# dollar amounts in the two legs, which is a hedge in money rather than in shares, and it uses no
# estimated ratio at all. Which of the two a strategy wants is a choice; reporting one while
# having estimated the other is not.
#
# Three limits, all structural rather than fixable here. No transaction costs and no slippage are
# charged, and a spread strategy trades often enough to be sensitive to both. The short leg is
# assumed available at no borrowing cost. And the pair was chosen by looking at the same history,
# which the screening section takes up.

# %%
evaluated = pair.index >= pair.index[train_end]

profit = pair["position"].shift(1) * (
    pair["dependent"].diff() - pair["hedge_ratio"].shift(1) * pair["independent"].diff()
)
committed = pair["dependent"].shift(1) + pair["hedge_ratio"].shift(1).abs() * pair[
    "independent"
].shift(1)
pair["strategy_return"] = (profit / committed).fillna(0.0)

realized = pair.loc[evaluated, "strategy_return"]
curve = (1.0 + realized).cumprod()

display(
    pd.DataFrame(
        [
            {
                "sessions evaluated": int(len(realized)),
                "sessions with a position": int((pair.loc[evaluated, "position"] != 0).sum()),
                "total return": float(curve.iloc[-1] - 1.0),
                "annualized mean return": float(realized.mean() * SESSIONS_PER_YEAR),
                "annualized volatility": float(realized.std() * np.sqrt(SESSIONS_PER_YEAR)),
                "deepest drawdown": float((curve / curve.cummax() - 1.0).min()),
            }
        ]
    ).T.rename(columns={0: DEPENDENT + " against " + INDEPENDENT})
)

# %% [markdown]
# The entry count is the number to read, and it is the section's conclusion. A half-life of
# months sets a window of about a year, a band that wide is crossed rarely, and the result is a
# position that opens a couple of times across the evaluated years and is then held for more than
# half of their sessions. That is not
# a mean-reversion strategy; it is a slow directional bet on a spread, which is what the
# cointegration tests said would happen when they declined to find a stationary combination.
#
# The return is therefore not evidence about the method. It is one draw from as many trades as the
# entry count reports. The
# drawdown is the one figure here worth carrying: it is what a position of that duration exposed
# a holder to while waiting for a reversion the spread had no mechanism to deliver.

# %% [markdown]
# ## Screening, and what a screen costs
#
# Six candidate pairs, each with a stated reason to be related, run through the same two tests.
# The table is the point of the section: a reason is not evidence, and the two tests do not
# always agree.
#
# The screen also carries a cost that the table cannot show. Testing six pairs at the same
# level and reporting whichever passes makes the reported level wrong, because the chance that
# at least one of six independent tests rejects a true null is far above the level of any one of
# them. What follows from that is not a correction to apply here but a rule for reading: a pair
# that passes a screen has cleared a lower bar than a pair tested on its own, and its spread
# needs to hold up on data the screen did not see.

# %%
CANDIDATE_PAIRS = [
    ("GLD", "SLV", "gold against silver"),
    ("XLE", "USO", "energy equity against crude oil"),
    ("QQQ", "SMH", "the Nasdaq 100 against semiconductors"),
    ("SPY", "VTI", "the S&P 500 against the total market"),
    ("TLT", "IEF", "long against intermediate Treasuries"),
    ("EEM", "VWO", "two emerging market funds"),
]

screen_rows = []
for dependent, independent, description in CANDIDATE_PAIRS:
    if not {dependent, independent} <= available:
        continue
    candidate = pair_frame(dependent, independent)
    if len(candidate) < MINIMUM_SESSIONS:
        continue

    candidate_p_value = coint(candidate["dependent"], candidate["independent"], trend="c")[1]
    candidate_johansen = coint_johansen(
        candidate[["dependent", "independent"]].to_numpy(), det_order=0, k_ar_diff=1
    )
    ratio = float(
        LinearRegression().fit(candidate[["independent"]], candidate["dependent"]).coef_[0]
    )
    spread = candidate["dependent"] - ratio * candidate["independent"]

    screen_rows.append(
        {
            "pair": f"{dependent} / {independent}",
            "why it might hold": description,
            "sessions": len(candidate),
            "Engle-Granger p value": candidate_p_value,
            "Johansen rejects": bool(candidate_johansen.lr1[0] > candidate_johansen.cvt[0, 1]),
            "hedge ratio": ratio,
            "half-life in sessions": estimate_half_life(spread),
        }
    )

screen = pd.DataFrame(screen_rows).set_index("pair")
display(screen)

print(
    f"Pairs both tests reject no cointegration for: {int(((screen['Engle-Granger p value'] < SIGNIFICANCE) & screen['Johansen rejects']).sum())}"
)
print(
    f"Pairs exactly one test rejects for: {int(((screen['Engle-Granger p value'] < SIGNIFICANCE) ^ screen['Johansen rejects']).sum())}"
)

# %% [markdown]
# Read the two counts printed under the table first. Over this sample no pair clears the
# Engle-Granger test at the stated level, and one pair is called cointegrated by Johansen alone.
# Six pairs with a stated reason, one weak signal: that is the result, and it is more useful than
# a table of successes would have been, because the reasons were not bad ones.
#
# The half-life column decides something different from whether the relationship exists. A
# half-life of a few sessions leaves room to enter and exit inside the reversion. Every half-life
# here is measured in months, and the shortest of them still holds a position through a quarter in
# which the relationship may have changed for reasons no test on past data can see. A pair needs
# both: a spread that reverts, and reversion fast enough to trade.

# %% [markdown]
# ## The same features as library expressions
#
# The pairwise work above is written out with statsmodels and numpy, one pair at a time.
# `ml4t-engineer` supplies the rolling versions as Polars expressions, which is the form a
# pipeline over many pairs wants: `rolling_correlation` and `beta_to_market` read return
# columns, `co_integration_score` reads price columns, and `correlation_regime_indicator` reads
# a correlation column and emits flags for its level.
#
# What these are not is a replacement for the tests above. A rolling score computed over a
# window is a description of that window, and cointegration is a statement about a relationship
# holding over a long sample. The two answer different questions.

# %%
LIBRARY_PAIR = ("GLD", "SLV") if {"GLD", "SLV"} <= available else (DEPENDENT, INDEPENDENT)

library = (
    pair_frame(*LIBRARY_PAIR)
    .pipe(pl.from_pandas, include_index=True)
    .rename({"dependent": "first_close", "independent": "second_close"})
    .with_columns(
        first_return=pl.col("first_close").pct_change(),
        second_return=pl.col("second_close").pct_change(),
    )
    .drop_nulls()
    .with_columns(
        correlation=rolling_correlation("first_return", "second_return", window=CORRELATION_WINDOW),
        beta=beta_to_market("second_return", "first_return", window=CORRELATION_WINDOW),
        cointegration_score=co_integration_score(
            "first_close", "second_close", window=COINTEGRATION_WINDOW
        ),
    )
)
library = library.with_columns(**correlation_regime_indicator("correlation"))

print(f"Library pair: {LIBRARY_PAIR[0]} against {LIBRARY_PAIR[1]}, {library.height:,} sessions")
display(
    library.select(["correlation", "beta", "cointegration_score"])
    .describe()
    .filter(pl.col("statistic").is_in(["mean", "std", "min", "max"]))
)
display(library.select(pl.col("^corr_regime.*$")).mean())

# %% [markdown]
# ## A level replaced by a position
#
# The second half of the notebook changes what a feature is computed over. Nothing here is a
# rolling window: the operation reads one date across many assets and returns each asset's
# standing among them on that date.
#
# Three versions of the same idea, and they are not interchangeable. A **rank** is an integer
# position and is bounded by the number of assets, so it says nothing about how far apart the
# assets are. A **percentile** divides the rank by the count, which makes a universe of ten
# comparable with one of five hundred. A **z-score** subtracts the cross-sectional mean and
# divides by the cross-sectional standard deviation, which keeps the distances and therefore
# keeps the outliers, and on a date when one asset moved five times as far as the rest it is the
# only one of the three that says so.
#
# None of the three uses a future date. That is worth stating because the causality question
# looks the same as everywhere else in the chapter and has a different answer: a cross-sectional
# operation reads across assets at one date, so it cannot look forward in time whatever else it
# does.

# %%
RANK_SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "XLE", "XLF", "XLV"]
selected = [symbol for symbol in RANK_SYMBOLS if symbol in available]

panel = (
    universe.filter(pl.col("symbol").is_in(selected))
    .select(["timestamp", "symbol", "close"])
    .sort(["symbol", "timestamp"])
    .with_columns(returns=pl.col("close").pct_change().over("symbol"))
    .with_columns(
        momentum=pl.col("returns").rolling_mean(FEATURE_WINDOW).over("symbol"),
        volatility=pl.col("returns").rolling_std(FEATURE_WINDOW).over("symbol")
        * np.sqrt(SESSIONS_PER_YEAR),
    )
    .drop_nulls()
    .with_columns(
        momentum_rank=pl.col("momentum").rank().over("timestamp"),
        volatility_rank=pl.col("volatility").rank().over("timestamp"),
        momentum_percentile=(pl.col("momentum").rank().over("timestamp") - 1)
        / (pl.col("momentum").count().over("timestamp") - 1),
        momentum_z_score=(pl.col("momentum") - pl.col("momentum").mean().over("timestamp"))
        / pl.col("momentum").std().over("timestamp"),
    )
)

print(f"Panel: {panel['symbol'].n_unique()} symbols, {panel.height:,} rows")
snapshot_date = panel["timestamp"].unique().sort().to_list()[-REGIME_WINDOW]
display(
    panel.filter(pl.col("timestamp") == snapshot_date)
    .select(["symbol", "momentum", "momentum_rank", "momentum_percentile", "momentum_z_score"])
    .sort("momentum_rank")
)

# %% [markdown]
# Read the snapshot's last two columns against each other. The percentiles are evenly spaced by
# construction, one step per asset, whatever the momentum values were. The z-scores are not, and
# where they bunch together the percentile has manufactured a distinction between assets that
# were nearly identical. That is the trade: a rank is robust because it discards the distances,
# and it is misleading for the same reason.

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)
HIGHLIGHT = [symbol for symbol in ("SPY", "GLD", "XLE") if symbol in selected]
palette = [COLORS["blue"], COLORS["amber"], COLORS["copper"]]

for symbol, color in zip(HIGHLIGHT, palette, strict=False):
    series = panel.filter(pl.col("symbol") == symbol).to_pandas()
    axes[0].plot(
        series["timestamp"],
        series["momentum_percentile"],
        linewidth=0.8,
        color=color,
        label=symbol,
    )
    axes[1].plot(
        series["timestamp"], series["volatility_rank"], linewidth=0.8, color=color, label=symbol
    )

axes[0].axhline(0.5, color=COLORS["recede"], linestyle="--", linewidth=0.7)
axes[0].set_ylabel("Percentile")
axes[0].set_title("Momentum measured against the rest of the universe", fontsize=9)
axes[0].legend(fontsize=7)

axes[1].set_ylabel("Rank")
axes[1].set_xlabel("Session")
axes[1].set_title("Volatility rank, where one is the calmest of the universe", fontsize=9)

fig.suptitle("A cross-sectional feature is bounded whatever the market does")
show_with_alt(
    fig,
    "Two stacked panels tracking three funds. The top plots each fund's momentum percentile "
    "against a dashed line at one half; the lines travel the full range and cross often. The "
    "bottom plots each fund's volatility rank, which steps between integer levels and holds "
    "for months at a time.",
)

# %% [markdown]
# The rank in the lower panel holds for long stretches and then steps, which is a different
# statistical object from the series it was computed from: a rolling standard deviation moves
# every session, and its rank moves only when the ordering changes. A downstream model reading
# the rank sees a step function, and one reading the raw volatility sees a continuous series.
# Which is wanted depends on whether the question is how volatile an asset is or which assets
# are the volatile ones.

# %% [markdown]
# ## A level replaced by a difference from a benchmark
#
# The other way to place a feature in context is to subtract a benchmark's version of it, which
# keeps the units. Momentum minus the market's momentum is a momentum, in the same units, with
# the part every asset shared removed. Volatility divided by the market's volatility is a
# ratio, and above one says the asset moved more than the market did.
#
# How the benchmark enters the feature decides which of the two operations to use. A momentum is
# a sum of returns and the market component is additive in it, so subtraction removes it. A
# volatility is a scale and the market's stress multiplies it, so a ratio removes it and a
# difference does not.

# %%
BENCHMARK = "SPY" if "SPY" in selected else selected[0]
COMPARED = "XLE" if "XLE" in selected else selected[-1]

benchmark = (
    panel.filter(pl.col("symbol") == BENCHMARK)
    .select(["timestamp", "momentum", "volatility"])
    .rename({"momentum": "market_momentum", "volatility": "market_volatility"})
)

relative = panel.join(benchmark, on="timestamp", how="inner").with_columns(
    momentum_less_market=pl.col("momentum") - pl.col("market_momentum"),
    volatility_over_market=pl.col("volatility") / pl.col("market_volatility"),
)

compared = relative.filter(pl.col("symbol") == COMPARED).to_pandas()

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

axes[0].plot(
    compared["timestamp"],
    compared["momentum"],
    linewidth=0.8,
    color=COLORS["blue"],
    label="its own",
)
axes[0].plot(
    compared["timestamp"],
    compared["momentum_less_market"],
    linewidth=0.8,
    color=COLORS["copper"],
    label="less the market's",
)
axes[0].axhline(0, color=COLORS["recede"], linestyle="--", linewidth=0.7)
axes[0].set_ylabel("Mean daily return")
axes[0].set_title("Subtracting the market leaves what was specific to the fund", fontsize=9)
axes[0].legend(fontsize=7)

axes[1].plot(
    compared["timestamp"],
    compared["volatility_over_market"],
    linewidth=0.8,
    color=COLORS["blue"],
)
axes[1].axhline(1.0, color=COLORS["recede"], linestyle="--", linewidth=0.7)
axes[1].set_ylabel("Ratio")
axes[1].set_xlabel("Session")
axes[1].set_title("Volatility as a multiple of the market's, not a difference", fontsize=9)

fig.suptitle(f"{COMPARED} against {BENCHMARK}, in units that survive the comparison")
show_with_alt(
    fig,
    f"Two stacked panels for {COMPARED}. The top plots its own momentum and its momentum less "
    f"{BENCHMARK}'s, both against a dashed zero line; the two diverge most where the market "
    "itself moved. The bottom plots the ratio of its volatility to the market's against a "
    "dashed line at one, and the ratio stays above one for most of the sample.",
)

# %% [markdown]
# The lower panel is the one that changes a reading. A ratio near one during a market-wide
# selloff says this fund was no more volatile than everything else, which the absolute
# volatility could not have said: that number was high because every number was high.

# %% [markdown]
# ## One measure aggregated across the universe
#
# The last construction turns a per-asset regime measure into universe-level features. Three
# aggregates, each answering a different question about the same date:
#
# - The **mean** stress across assets, which is the level.
# - The **breadth**, the fraction of assets above a threshold, which distinguishes a few assets
#   in trouble from all of them.
# - The **dispersion**, the standard deviation across assets, which is low when everything moves
#   together and high when the universe has split.
#
# The per-asset measure here is each asset's volatility ranked within a trailing window of
# `REGIME_WINDOW` sessions, a stand-in for the filtered regime probability that
# `11_hmm_regimes` produces. `rolling_rank` is what makes it a stand-in rather than a leak:
# ranking each asset's volatility over its whole history would place today's value against
# values from years that had not happened, and the resulting series would look like a regime
# indicator while being unavailable on every date but the last.

# %%
STRESS_THRESHOLD = 0.7

stress = (
    panel.select(["timestamp", "symbol", "volatility"])
    .sort(["symbol", "timestamp"])
    .with_columns(
        stress=pl.col("volatility").rolling_rank(window_size=REGIME_WINDOW).over("symbol")
        / REGIME_WINDOW
    )
    .drop_nulls()
)

aggregates = (
    stress.group_by("timestamp")
    .agg(
        mean_stress=pl.col("stress").mean(),
        breadth=(pl.col("stress") > STRESS_THRESHOLD).mean(),
        dispersion=pl.col("stress").std(),
        assets=pl.len(),
    )
    .sort("timestamp")
)

benchmark_level = (
    universe.filter(pl.col("symbol") == BENCHMARK)
    .select(["timestamp", "close"])
    .sort("timestamp")
    .with_columns(cumulative=(pl.col("close") / pl.col("close").first() - 1.0) * 100)
)

aggregated = aggregates.join(
    benchmark_level.select(["timestamp", "cumulative"]), on="timestamp", how="inner"
).to_pandas()

print(
    f"Aggregate rows: {len(aggregated):,}, assets per date: {aggregates['assets'].min()} to {aggregates['assets'].max()}"
)
display(aggregates.select(["mean_stress", "breadth", "dispersion"]).describe())

# %%
fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

axes[0].plot(
    aggregated["timestamp"], aggregated["cumulative"], linewidth=0.8, color=COLORS["neutral"]
)
axes[0].set_ylabel("Percent")
axes[0].set_title(f"{BENCHMARK} cumulative return, for context", fontsize=9)

axes[1].fill_between(
    aggregated["timestamp"], 0, aggregated["breadth"], alpha=0.4, color=COLORS["copper"]
)
axes[1].set_ylabel("Fraction")
axes[1].set_title("Breadth: how much of the universe is in its own high-stress range", fontsize=9)

axes[2].plot(aggregated["timestamp"], aggregated["dispersion"], linewidth=0.8, color=COLORS["blue"])
axes[2].set_ylabel("Standard deviation")
axes[2].set_xlabel("Session")
axes[2].set_title("Dispersion: whether the universe agrees about the stress", fontsize=9)

fig.suptitle("Three aggregates of one per-asset measure, each saying something different")
show_with_alt(
    fig,
    f"Three stacked panels. The top plots {BENCHMARK}'s cumulative return. The middle fills the "
    "fraction of the universe in its own high-stress range, which rises to near one during the "
    "sharp declines above it. The bottom plots the cross-sectional dispersion of the stress "
    "measure, which falls when the breadth is at its highest.",
)

# %% [markdown]
# Breadth and dispersion move against each other at the extremes, and that is the reason to
# carry both. When everything is stressed the breadth is near one and there is nothing left for
# the assets to disagree about, so the dispersion falls to near zero. A high dispersion is a
# universe where some assets are under stress and others are not, which is a different
# environment and not a milder version of the same one.
#
# Both are computed from a trailing rank, so both are available on the session they describe.
# What neither is, is a regime probability: a rank says where today sits among the last
# `REGIME_WINDOW` sessions of the same asset, and a fitted model says how likely a state is
# given everything observed. The aggregation mechanics are the same either way, which is what
# this section is for.

# %% [markdown]
# ## Takeaways
#
# 1. **Cointegration is a claim about drift, not about co-movement.** Two funds can track each
#    other daily and still fail both tests, and one of the two funds carrying a roll cost is
#    enough to do it. A stated economic reason is where a screen starts and not evidence: six
#    pairs with good reasons produced no Engle-Granger rejection at the level used here.
# 2. **Engle-Granger and Johansen can disagree, and the disagreement is information.** One takes
#    a dependent variable and one treats the pair as a system. A pair only one of them rejects
#    for is a pair whose relationship depends on how the question was asked.
# 3. **A hedge ratio used to hold a position has to be filtered.** A regression over the whole
#    sample gives one number that was unavailable on every date but the last. The Kalman
#    recursion gives a series, and its process noise is the choice between tracking a real
#    change and chasing noise.
# 4. **A window length is a parameter and leaks like one.** The half-life that sizes the signal
#    window is estimated on the first block only, because a length chosen from the whole sample
#    is a leak that does not look like one.
# 5. **Price the position that was held, not the column that was plotted.** The profit is the two
#    price changes weighted by the previous session's holdings. Differencing a spread whose ratio
#    moves adds the ratio's own change to it, and differencing the two funds' returns prices equal
#    dollar amounts in each leg, which is a hedge in money and not the share ratio estimated.
# 6. **Rank, percentile and z-score discard different things.** Ranks and percentiles throw away
#    the distances between assets, which is what makes them robust and what makes them invent
#    distinctions between assets that were nearly identical.
# 7. **A cross-sectional operation cannot look forward, and a rank over an asset's own history
#    can.** The first reads one date across assets. The second is the same word applied along
#    time, where it needs a trailing window like every other temporal feature.
#
# **Previous**: `13_regime_as_feature` puts a regime column into a downstream model.
# **Next**: `case_study_temporal_summary` collects what this chapter's notebooks produced.
