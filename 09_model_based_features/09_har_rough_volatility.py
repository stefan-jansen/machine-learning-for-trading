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
# # HAR and Rough Volatility
#
# **Chapter 9 | Section 9.3**
#
# **Docker image**: `ml4t`
#
# The GARCH model in the previous notebook has one memory: a single decay rate that
# governs how long a shock lasts. This notebook covers two things that memory cannot
# express. **HAR** replaces the single decay with three horizons at once, on the argument
# that the people trading at a daily, weekly and monthly rhythm are different people.
# **Roughness** asks a different question entirely: not how long volatility remembers, but
# how jagged its path is, and the answer for real markets is far jaggeder than the models
# that were standard for thirty years assume.
#
# Both need a measurement of volatility to work on, and getting that measurement right is
# the first third of the notebook.
#
# **Learning objectives**
#
# - Compute volatility from within-session data, and use it to check what the estimators
#   available from daily bars alone are actually measuring.
# - Fit a model of volatility across three horizons, and read its coefficients as a
#   statement about which horizon is driving the current level.
# - Correct the standard errors of that fit for the overlap its own regressors contain.
# - Estimate how jagged a series is over time, apply it to returns and to volatility, and
#   read the two answers as the different things they are.
#
# **Book reference**
#
# Chapter 9, Section 9.3 (Volatility Features).
#
# **Prerequisites**
#
# `08_garch_volatility` for conditional volatility and for what persistence means.
# Chapter 8 introduced the range-based estimators this notebook tests.

# %% [markdown]
# ## Setup

# %%
"""HAR and rough volatility - multi-horizon volatility and the Hurst exponent."""

import warnings

import exchange_calendars as xcals
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from arch import arch_model
from IPython.display import display
from ml4t.engineer.features.regime import hurst_exponent
from ml4t.engineer.features.volatility import (
    garman_klass_volatility,
    parkinson_volatility,
    realized_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
)
from pandas.errors import PerformanceWarning
from scipy import stats

from data import load_etfs, load_nasdaq100_bars
from utils.style import COLORS, FIGSIZE, show_with_alt

# Building the exchange session index applies a date offset element by element, which pandas
# reports as slow twenty times over. It is a speed notice about one calendar build.
warnings.filterwarnings(
    "ignore", category=PerformanceWarning, module="exchange_calendars.exchange_calendar"
)

# %% tags=["parameters"]
START_DATE = "2006-01-01"
END_DATE = "2024-12-31"
INTRADAY_SYMBOL = "AAPL"
DAILY_SYMBOL = "SPY"
MINIMUM_BARS_PER_SESSION = 360  # a full session is about 390 one-minute bars
EXCHANGE = "XNYS"  # the calendar that says which dates are sessions at all
SESSIONS_PER_YEAR = 252
ROLLING_WINDOW = 20  # sessions in every rolling average below: about one month

# %% [markdown]
# # Part 1: what "volatility" is being measured
#
# Everything in this notebook models a series called realized volatility, and there are
# several different things that name refers to. The definition with the strongest claim to
# it is computed from within-session data: sum the squared returns of every minute in a
# session, and the total estimates that session's variance with an error that falls as the
# bars get finer.
#
# $$RV_t = \sum_{i=1}^{M} r_{t,i}^2$$
#
# It measures the **trading session** and nothing else, and that is the first thing to be
# clear about, because a daily estimator's target is the whole day. This section
# establishes the relation between the two on a symbol where both are available, and turns
# up two data traps on the way.

# %%
minute_bars = (
    load_nasdaq100_bars(symbols=[INTRADAY_SYMBOL], include_microstructure=True, lazy=True)
    .filter((pl.col("time") >= "09:30") & (pl.col("time") < "16:00"))
    .select(["date", "time", "first_trade_price", "last_trade_price"])
    .collect()
    .sort(["date", "time"])
)

print(
    f"{INTRADAY_SYMBOL} minute bars: {minute_bars.height:,} rows over "
    f"{minute_bars['date'].n_unique()} sessions, "
    f"{minute_bars['date'].min()} to {minute_bars['date'].max()}"
)

# %% [markdown]
# Two choices in the cell above are worth stating rather than leaving in the code. The
# returns are taken from the **last trade price** of each minute rather than from the
# minute's volume-weighted average. A volume-weighted average is an average over the
# minute, so a series of them is smoother than the price actually was, and squared returns
# built from it understate the variance. On this sample the difference is not small.
#
# And the bars cover 09:30 to 16:00 only, so the sum below is the variance of the trading
# session. What happens between one session's close and the next one's open is a separate
# quantity, measured separately here.
#
# Three details in the cell below are load-bearing. The session's open and close come from
# the bars as loaded, not from a frame reduced to the minutes that have a previous close;
# an open read from such a frame would be the second minute's price, and the overnight
# return would then swallow the opening minute. The opening minute keeps its own
# open-to-close return in the variance sum for the same reason: without it the sum covers
# one minute less than the session return it is compared against.
#
# And an overnight return is only an overnight return if the two rows are consecutive
# sessions of the exchange. Bar counts cannot establish that, and neither can the number of
# calendar days between the dates: a Monday-to-Wednesday pair is three days apart and still
# contains all of Tuesday's trading if Tuesday is missing from the data. What settles it is
# the exchange calendar, which says which dates are sessions independently of what the data
# holds. Each pair is required to be one session apart on that calendar, which keeps
# holiday weekends and drops a missing weekday. On a reduced test fixture, which keeps
# separated blocks of sessions, an unchecked gap can be weeks long.

# %%
boundaries = (
    minute_bars.group_by("date")
    .agg(
        session_open=pl.col("first_trade_price").first(),
        session_close=pl.col("last_trade_price").last(),
        bars=pl.len(),
    )
    .sort("date")
)

variances = (
    minute_bars.with_columns(
        # The first bar of a session has no previous close, and dropping it leaves the
        # opening minute out of the sum while the aggregate open-to-close return contains
        # it. Its own open-to-close return is what belongs there.
        minute_return=pl.when(pl.col("last_trade_price").log().diff().over("date").is_null())
        .then((pl.col("last_trade_price") / pl.col("first_trade_price")).log())
        .otherwise(pl.col("last_trade_price").log().diff().over("date"))
    )
    .group_by("date")
    .agg(minute_variance=pl.col("minute_return").pow(2).sum())
)

exchange_sessions = xcals.get_calendar(EXCHANGE).sessions_in_range(
    minute_bars["date"].min(), minute_bars["date"].max()
)
session_numbers = pl.DataFrame(
    {
        "date": pl.Series("date", exchange_sessions).cast(pl.Date),
        "session_number": np.arange(len(exchange_sessions), dtype=np.int64),
    }
)

intraday = (
    boundaries.join(variances, on="date", how="left")
    .join(session_numbers, on="date", how="inner")
    .sort("date")
    .with_columns(
        previous_close=pl.col("session_close").shift(1),
        previous_bars=pl.col("bars").shift(1),
        sessions_apart=pl.col("session_number") - pl.col("session_number").shift(1),
        days_apart=(pl.col("date") - pl.col("date").shift(1)).dt.total_days(),
    )
    .with_columns(
        session_return=(pl.col("session_close") / pl.col("session_open")).log(),
        overnight_return=(pl.col("session_open") / pl.col("previous_close")).log(),
        close_to_close_return=(pl.col("session_close") / pl.col("previous_close")).log(),
    )
    .filter(
        (pl.col("bars") >= MINIMUM_BARS_PER_SESSION)
        & (pl.col("previous_bars") >= MINIMUM_BARS_PER_SESSION)
        & (pl.col("sessions_apart") == 1)
    )
    .drop_nulls()
)

print(f"Sessions kept: {intraday.height:,} of {boundaries.height}")
print(f"Exchange sessions in the sample period: {len(exchange_sessions):,}")
print(f"Longest calendar gap among the kept pairs: {intraday['days_apart'].max()} days")

# %% [markdown]
# ## The first trap: these prices are not adjusted
#
# A close-to-close return computed from raw trade prices crosses every corporate action as
# though it were a price move. The check below finds any session whose return is too large
# to be one, and on this symbol and period it finds exactly one.

# %%
IMPLAUSIBLE_RETURN = 0.25  # a one-session log return larger than this is a corporate action

implausible = intraday.filter(pl.col("close_to_close_return").abs() > IMPLAUSIBLE_RETURN)
print(f"Sessions with a return beyond {IMPLAUSIBLE_RETURN:.0%}: {implausible.height}")
for row in implausible.iter_rows(named=True):
    print(f"  {row['date']}: {row['close_to_close_return']:+.3f} in logs")

clean = intraday.filter(pl.col("close_to_close_return").abs() <= IMPLAUSIBLE_RETURN)

# %% [markdown]
# One session, and it is the four-for-one share split. The intraday variance of that
# session is unaffected, because a split happens between sessions and every minute return
# inside the session is a real price move; what is corrupted is anything computed across
# the boundary. That asymmetry is worth carrying: the same raw data can be sound for one
# statistic and unusable for another, and which is which depends on whether the statistic
# crosses a session boundary.

# %% [markdown]
# ## The second trap: a session is not a day
#
# A close-to-close return is the overnight move plus the session that follows it, exactly,
# because both are logarithms. So the identity below holds term by term with no
# approximation:
#
# $$\mathbb{E}[r_{cc}^2] = \mathbb{E}[r_{on}^2] + \mathbb{E}[r_{oc}^2]
#   + 2\,\mathbb{E}[r_{on} r_{oc}]$$
#
# where $r_{oc}$ is the **aggregate** open-to-close return of the session. That is not the
# same quantity as the realized variance measured from minute bars: the realized variance
# is the sum of squared minute returns and the aggregate is the square of their sum, so
# they differ by every cross-product between minutes. Both are reported below, separately,
# because they answer different questions and conflating them is how a decomposition comes
# out looking exact when it is not.

# %%
annualize = np.sqrt(SESSIONS_PER_YEAR)

overnight_variance = float((clean["overnight_return"] ** 2).mean())
session_aggregate_variance = float((clean["session_return"] ** 2).mean())
cross_product = float((clean["overnight_return"] * clean["session_return"]).mean())
close_to_close_variance = float((clean["close_to_close_return"] ** 2).mean())
realized_variance = float(clean["minute_variance"].mean())

display(
    pd.DataFrame(
        [
            {"term": "overnight, squared", "value": overnight_variance},
            {"term": "session open to close, squared", "value": session_aggregate_variance},
            {"term": "twice their cross-product", "value": 2 * cross_product},
            {
                "term": "the three added",
                "value": overnight_variance + session_aggregate_variance + 2 * cross_product,
            },
            {"term": "close to close, squared", "value": close_to_close_variance},
        ]
    )
)
print(
    f"Share of the daily square that happens overnight: {overnight_variance / close_to_close_variance:.1%}"
)
print(f"Session variance minute by minute: {np.sqrt(realized_variance) * annualize:.4f} annualized")
print(
    "Session variance from the aggregate open-to-close return: "
    f"{np.sqrt(session_aggregate_variance) * annualize:.4f} annualized"
)
print(f"Ratio of the two: {realized_variance / session_aggregate_variance:.3f}")

# %% [markdown]
# The identity closes to the last digit, which it must; what it buys is the size of each
# term. The overnight square is a large fraction of the daily square for a symbol whose
# market is open six and a half hours out of twenty-four, and the cross-product is small
# beside the other two, so the two components are close to uncorrelated.
#
# The last ratio is the separate question. Summing squared minute returns and squaring the
# aggregate open-to-close return are two estimates of the same session's variance and they
# do not agree: the minute-by-minute sum picks up movement that reverses before the close
# and the aggregate does not. Neither is wrong, and which one a model wants depends on
# whether reversals inside a session are risk it is exposed to. A position held through the
# session sees the aggregate; one traded inside it sees the sum.
#
# Either way, a comparison between an intraday measurement and a daily estimator has to
# account for the overnight term, and a strategy holding overnight is exposed to a risk no
# intraday measurement sees.

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

sessions = clean["date"].to_list()
ax = axes[0]
ax.plot(
    sessions,
    clean["minute_variance"].sqrt() * annualize,
    linewidth=0.7,
    color=COLORS["blue"],
    label="Trading session",
)
ax.plot(
    sessions,
    clean["overnight_return"].abs() * annualize,
    linewidth=0.5,
    alpha=0.7,
    color=COLORS["amber"],
    label="Overnight",
)
ax.set_ylabel("Annualized volatility")
ax.set_title("Two components of the same day")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(
    sessions,
    (clean["minute_variance"].sqrt() * annualize).rolling_mean(ROLLING_WINDOW),
    linewidth=1,
    color=COLORS["blue"],
    label="Trading session",
)
ax.plot(
    sessions,
    (clean["overnight_return"].abs() * annualize).rolling_mean(ROLLING_WINDOW),
    linewidth=1,
    color=COLORS["amber"],
    label="Overnight",
)
ax.set_ylabel("Annualized volatility")
ax.set_xlabel("Session")
ax.set_title(f"The same two, averaged over {ROLLING_WINDOW} sessions")
ax.legend(fontsize=7)

fig.suptitle("The market is shut for most of the day, and it moves then too")
show_with_alt(
    fig,
    "Two stacked panels sharing a time axis over two years. The top draws the session "
    "volatility measured from minute bars against the absolute overnight move, both noisy "
    "day to day, with the session series generally the larger and both spiking in March "
    "2020. The bottom draws the same two averaged over twenty sessions, where the session "
    "series stays above the overnight one throughout and the two move together.",
)

# %% [markdown]
# ## Which daily estimator to use
#
# Ranking estimators by accuracy needs a target measured the same way, and the section
# above is the reason there is not one here: the intraday measurement covers the session
# and every daily estimator covers something else.
#
# What is measured below instead is each estimator's **dispersion around its own trailing
# average**, session by session on the ETF panel's adjusted daily bars. Be clear about what
# that is and is not. It is not the estimator's sampling error: the underlying volatility
# genuinely moves within twenty sessions, so real movement is in the number, and the
# trailing average in the denominator carries the estimator's own noise as well. Nor are
# the four measuring the same interval, since the close-to-close estimator spans a session
# boundary and the other three do not.
#
# What it does show is how jagged each column would be if a model read it, which is a
# property of the column rather than of the estimator, and is the thing that decides
# whether a feature needs smoothing before use.


# %%
def range_volatility(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-session Garman-Klass volatility, annualized, plus the close-to-close return."""
    return frame.with_columns(
        log_return=pl.col("close").log().diff(),
        volatility=(
            (
                0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
                - (2 * np.log(2) - 1) * (pl.col("close") / pl.col("open")).log().pow(2)
            ).clip(lower_bound=0)
            * SESSIONS_PER_YEAR
        ).sqrt(),
    ).drop_nulls()


spy = (
    load_etfs(symbols=[DAILY_SYMBOL])
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .sort("timestamp")
    .select(["timestamp", "open", "high", "low", "close"])
)

per_session = spy.with_columns(
    close_to_close=(pl.col("close").log().diff().pow(2) * SESSIONS_PER_YEAR).sqrt(),
    parkinson=(
        (pl.col("high") / pl.col("low")).log().pow(2) / (4 * np.log(2)) * SESSIONS_PER_YEAR
    ).sqrt(),
    garman_klass=(
        (
            0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
            - (2 * np.log(2) - 1) * (pl.col("close") / pl.col("open")).log().pow(2)
        ).clip(lower_bound=0)
        * SESSIONS_PER_YEAR
    ).sqrt(),
    rogers_satchell=(
        (
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
        ).clip(lower_bound=0)
        * SESSIONS_PER_YEAR
    ).sqrt(),
).drop_nulls()

ESTIMATORS = ["close_to_close", "parkinson", "garman_klass", "rogers_satchell"]

efficiency = pd.DataFrame(
    [
        {
            "estimator": name,
            "mean, annualized": per_session[name].mean(),
            "dispersion around its trailing average": float(
                (per_session[name] / per_session[name].rolling_mean(ROLLING_WINDOW))
                .drop_nulls()
                .std()
            ),
            "sessions estimated at zero": int((per_session[name] < 1e-6).sum()),
        }
        for name in ESTIMATORS
    ]
).sort_values("dispersion around its trailing average")
display(efficiency)

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
smooth = per_session.select(
    [pl.col("timestamp")]
    + [pl.col(name).rolling_mean(ROLLING_WINDOW).alias(name) for name in ESTIMATORS]
).drop_nulls()
for name, color in zip(
    ESTIMATORS, [COLORS["recede"], COLORS["amber"], COLORS["blue"], COLORS["copper"]]
):
    ax.plot(smooth["timestamp"], smooth[name], linewidth=0.8, alpha=0.9, color=color, label=name)
ax.set_ylabel("Annualized volatility")
ax.set_xlabel("Session")
ax.set_title(f"Four estimators of the same thing, averaged over {ROLLING_WINDOW} sessions")
ax.legend(fontsize=7)
show_with_alt(
    fig,
    "Four volatility estimators for the same symbol, each averaged over twenty sessions and "
    "drawn over the whole sample. All four rise and fall together at the same dates and are "
    "separated by a roughly constant vertical offset, with the close-to-close series the "
    "lowest and the range-based ones above it.",
)

# %% [markdown]
# The close-to-close column is the one that separates from the rest, on both of the
# quantities reported. It is the most dispersed around its own average, and it is the only
# one that puts a non-trivial number of sessions at zero: a session that closed where it
# opened has no close-to-close movement to read, whatever happened in between. Reading the
# high and the low is what removes that failure, and Chapter 8 is where the efficiency of
# each formula is derived rather than measured.
#
# The mean column ranks them on level, and there the ordering says nothing about which is
# right. Each formula assumes something about the process, they cover different parts of
# the day, and this notebook has no target to check any of them against. Read the level as
# a scale each estimator carries, and never mix two of them in one feature without
# rescaling.

# %% [markdown]
# ## The library's versions, and the one that sees the gap
#
# `ml4t.engineer.features.volatility` supplies all five as Polars expressions. They take a
# window rather than working session by session, and they average variances before taking
# the square root where the cells above take the square root first, so the two do not agree
# exactly even where they implement the same formula. The difference is the usual one
# between a mean of roots and a root of means, and it is small next to the difference
# between estimators.
#
# The fifth estimator is the reason to run this: **Yang-Zhang** adds an overnight component
# to an intraday one, so it is the only one of the five whose target is the whole day that
# Part 1 decomposed. Its level should sit above the others for exactly that reason.

# %%
library = (
    per_session.with_columns(log_return=pl.col("close").log().diff())
    .with_columns(
        library_close=realized_volatility("log_return", period=ROLLING_WINDOW),
        library_parkinson=parkinson_volatility("high", "low", period=ROLLING_WINDOW),
        library_garman_klass=garman_klass_volatility(
            "open", "high", "low", "close", period=ROLLING_WINDOW
        ),
        library_rogers_satchell=rogers_satchell_volatility(
            "open", "high", "low", "close", period=ROLLING_WINDOW
        ),
        library_yang_zhang=yang_zhang_volatility(
            "open", "high", "low", "close", period=ROLLING_WINDOW
        ),
    )
    .drop_nulls()
)

display(
    pd.DataFrame(
        [
            {
                "expression": name.removeprefix("library_"),
                "mean, annualized": library[name].mean(),
                "sees the overnight gap": name.endswith(("close", "yang_zhang")),
            }
            for name in library.columns
            if name.startswith("library_")
        ]
    )
)

# %% [markdown]
# Yang-Zhang and the close-to-close estimator are the two that cross a session boundary,
# and they are the two whose target is the day rather than the session. The three purely
# intraday estimators are measuring something smaller and should not be compared against
# them on level, which is the same point Part 1 made with the decomposition and is the
# reason a feature set should not mix them.

# %% [markdown]
# # Part 2: three horizons instead of one
#
# The **heterogeneous autoregressive** model (Corsi, 2009) predicts tomorrow's volatility
# from three averages of today's: yesterday's, the last week's, and the last month's.
#
# $$RV_{t+1} = c + \beta_d\,RV^{(d)}_t + \beta_w\,RV^{(w)}_t + \beta_m\,RV^{(m)}_t + \varepsilon_{t+1}$$
#
# The argument for that shape is economic rather than statistical. Traders operating at
# different frequencies watch volatility at different horizons and act on what they watch,
# so the volatility a market produces is a superposition of their responses rather than a
# single decay. The model is a linear regression, which makes it far easier to fit than
# GARCH and far easier to read: each coefficient is the weight one horizon carries.
#
# The regressors are built from a range-based estimator for the reason Part 1 established,
# and each is lagged so that nothing on the right-hand side of the regression is known
# later than the session it predicts from.

# %%
HORIZONS = {"daily": 1, "weekly": 5, "monthly": 22}

spy = (
    load_etfs(symbols=[DAILY_SYMBOL])
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .sort("timestamp")
    .select(["timestamp", "open", "high", "low", "close"])
)


def range_volatility(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-session Garman-Klass volatility, annualized, plus the close-to-close return."""
    return frame.with_columns(
        log_return=pl.col("close").log().diff(),
        volatility=(
            (
                0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
                - (2 * np.log(2) - 1) * (pl.col("close") / pl.col("open")).log().pow(2)
            ).clip(lower_bound=0)
            * SESSIONS_PER_YEAR
        ).sqrt(),
    ).drop_nulls()


def har_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """The three lagged averages and the next session's volatility, as one table."""
    return (
        frame.with_columns(
            [
                pl.col("volatility").shift(1).rolling_mean(length).alias(name)
                for name, length in HORIZONS.items()
            ]
        )
        .rename({"volatility": "target"})
        .drop_nulls()
    )


spy_volatility = range_volatility(spy)
har = har_frame(spy_volatility)
print(f"{DAILY_SYMBOL}: {spy_volatility.height:,} sessions, {har.height:,} usable for the model")
display(har.select(["timestamp", *HORIZONS, "target"]).tail(3))

# %% [markdown]
# ## Fitting it, and correcting the standard errors
#
# The regression is ordinary least squares. Its textbook standard errors assume the
# residuals are homoscedastic and serially uncorrelated, and neither assumption is safe on
# a volatility series: volatility is heteroscedastic almost by definition, and a
# misspecified dynamic leaves autocorrelation behind in the residuals.
#
# Note what the overlapping regressors do and do not imply. The weekly and monthly
# regressors are rolling averages, so consecutive rows share most of their inputs; that
# makes the *regressors* correlated across rows, which ordinary least squares handles.
# It does not by itself make the *residuals* correlated: a correctly specified model of
# this shape can have independent innovations. So overlap is a reason to check rather than
# a proof of a problem.
#
# Four standard errors for the same coefficients are computed below, and the reason there
# are four rather than two is that isolating one relaxation requires holding the other
# estimator choices fixed. The textbook ones assume homoscedastic, serially uncorrelated
# residuals. `HC3` relaxes the first assumption with a leverage adjustment. The last two
# are the same **Newey-West** estimator run twice, once with a zero-lag window and once
# over a window set to the longest horizon in the model, which is where any dependence the
# regressors' overlap could induce would reach. Those two differ in the lag window and in
# nothing else, so their ratio is what the lagged covariance terms contribute; `HC3`
# against the textbook column is a separate reading of heteroscedasticity alone.

# %%
NEWEY_WEST_LAGS = max(HORIZONS.values())


def fit_har(frame: pl.DataFrame):
    """OLS with Newey-West standard errors over the longest horizon's overlap."""
    design = sm.add_constant(frame.select(list(HORIZONS)).to_pandas())
    return sm.OLS(frame["target"].to_numpy(), design).fit(
        cov_type="HAC", cov_kwds={"maxlags": NEWEY_WEST_LAGS}
    )


design = sm.add_constant(har.select(list(HORIZONS)).to_pandas())
target = har["target"].to_numpy()

har_fit = fit_har(har)
plain_fit = sm.OLS(target, design).fit()
leverage_fit = sm.OLS(target, design).fit(cov_type="HC3")
zero_lag_fit = sm.OLS(target, design).fit(cov_type="HAC", cov_kwds={"maxlags": 0})

display(
    pd.DataFrame(
        {
            "coefficient": har_fit.params,
            "standard error, textbook": plain_fit.bse,
            "standard error, HC3": leverage_fit.bse,
            "standard error, Newey-West at zero lags": zero_lag_fit.bse,
            f"standard error, Newey-West at {NEWEY_WEST_LAGS} lags": har_fit.bse,
            "ratio, HC3 to textbook": leverage_fit.bse / plain_fit.bse,
            "ratio, the two Newey-West columns": har_fit.bse / zero_lag_fit.bse,
        }
    )
)
print(f"R-squared: {har_fit.rsquared:.4f}")

# %% [markdown]
# Nothing about the coefficients changed. What changed is how much confidence the fit
# reports in them, and the two ratio columns say where that change comes from.
#
# The first ratio is heteroscedasticity, read on its own. The second is the lag window, read
# with everything else held fixed. A second ratio near one says the lagged covariance terms
# add little to the zero-lag version; it is a statement about the estimate, not about the
# residuals, because what Newey-West sums over the window is the covariance of the
# regressor-residual products rather than of the residuals themselves. Serially correlated
# residuals whose products with these regressors happen to be close to uncorrelated would
# produce the same ratio.
#
# The coefficients themselves are the reason to fit HAR rather than GARCH. Each one is the
# weight the model puts on one horizon, so their relative sizes say which horizon is
# carrying the current level, and that is a statement a person can act on. GARCH answers
# the same forecasting question with one decay rate and no such decomposition.
#
# Two things to watch when reading them. They are not constrained to be positive, and a
# negative coefficient on one horizon usually means it is collinear with another rather
# than that volatility at that horizon predicts less volatility next session. And they sum
# to close to one on a persistent series, so a rise in one is generally a fall in another;
# read the three together rather than one at a time.

# %% [markdown]
# ## Against GARCH, out of sample
#
# Both models are fitted on the same training block and asked for one-step-ahead forecasts
# over the same test block. They are not forecasting exactly the same quantity, which is
# the first thing the comparison has to state: HAR targets the range-based volatility it
# was built from, and GARCH targets the variance of close-to-close returns. The levels
# therefore differ for a reason that has nothing to do with forecast quality, and the error
# measures below carry that difference.

# %%
TRAIN_FRACTION = 0.7

split = int(har.height * TRAIN_FRACTION)
train, test = har.head(split), har.tail(har.height - split)

har_out_of_sample = fit_har(train).predict(sm.add_constant(test.select(list(HORIZONS)).to_pandas()))

returns_percent = pd.Series(
    spy_volatility["log_return"].to_numpy() * 100,
    index=pd.DatetimeIndex(spy_volatility["timestamp"].to_list()),
)
split_timestamp = pd.Timestamp(test["timestamp"][0])
garch_fit = arch_model(returns_percent, mean="Constant", vol="GARCH", p=1, q=1).fit(
    disp="off", last_obs=split_timestamp
)
origin = returns_percent.index[returns_percent.index.get_loc(split_timestamp) - 1]
garch_out_of_sample = (
    np.sqrt(garch_fit.forecast(horizon=1, start=origin, reindex=False).variance["h.1"].to_numpy())
    * np.sqrt(SESSIONS_PER_YEAR)
    / 100
)[: test.height]

actual = test["target"].to_numpy()
display(
    pd.DataFrame(
        [
            {
                "model": name,
                "root mean squared error": float(np.sqrt(np.mean((forecast - actual) ** 2))),
                "mean absolute error": float(np.mean(np.abs(forecast - actual))),
                "mean forecast": float(np.mean(forecast)),
                "correlation with the target": float(np.corrcoef(forecast, actual)[0, 1]),
            }
            for name, forecast in [
                ("HAR", har_out_of_sample.to_numpy()),
                ("GARCH", garch_out_of_sample),
            ]
        ]
        + [{"model": "the target itself", "mean forecast": float(np.mean(actual))}]
    )
)

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

test_sessions = test["timestamp"].to_list()
ax = axes[0]
ax.plot(test_sessions, actual, linewidth=0.5, alpha=0.6, color=COLORS["neutral"], label="Target")
ax.plot(test_sessions, har_out_of_sample, linewidth=0.9, color=COLORS["blue"], label="HAR")
ax.plot(test_sessions, garch_out_of_sample, linewidth=0.9, color=COLORS["amber"], label="GARCH")
ax.set_ylabel("Annualized volatility")
ax.set_title("One-step-ahead forecasts, parameters fitted before the block")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(test_sessions, har_out_of_sample - actual, linewidth=0.5, color=COLORS["blue"], label="HAR")
ax.plot(
    test_sessions, garch_out_of_sample - actual, linewidth=0.5, color=COLORS["amber"], label="GARCH"
)
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Forecast minus target")
ax.set_xlabel("Session")
ax.set_title("Errors, on the same scale")
ax.legend(fontsize=7)

fig.suptitle("Two models of the same thing, forecasting two slightly different things")
show_with_alt(
    fig,
    "Two stacked panels over the test block. The top draws the target volatility against "
    "the HAR and GARCH forecasts: both track its large movements, the HAR line sitting "
    "closer to the target and the GARCH line at a visibly higher level throughout. The "
    "bottom draws each forecast minus the target, with the GARCH errors offset above zero "
    "and both widening at the same dates.",
)

# %% [markdown]
# Read the mean forecast column before the error columns. The two models sit at different
# levels because they are estimating different quantities, and a squared error between a
# forecast and a target measured a different way charges the model for that difference as
# though it were inaccuracy. The correlation column is the part of the comparison that
# the level difference does not touch, because a correlation is invariant to it.
#
# So the comparison establishes less than it looks like it does. What it does establish is
# that a linear regression on three lagged averages tracks realized volatility about as
# well as a fitted GARCH does, at a fraction of the machinery, which is the reason HAR is
# the standard baseline in the realized-volatility literature.

# %% [markdown]
# ## The coefficients as features
#
# Refitting on a moving window turns the three weights into three slowly varying columns,
# and their movement is a statement about which horizon is driving volatility now. The
# window is long because the coefficients are estimated rather than observed, and the step
# is a month because refitting daily would produce three columns whose variation is mostly
# estimation noise.

# %%
REFIT_WINDOW = 504  # sessions each fit reads: about two years
REFIT_STEP = 22  # sessions between fits: about one month

rolling_rows = []
for start in range(0, har.height - REFIT_WINDOW, REFIT_STEP):
    window = har.slice(start, REFIT_WINDOW)
    fit = fit_har(window)
    rolling_rows.append(
        {
            "timestamp": window["timestamp"][-1],
            **{name: fit.params[name] for name in HORIZONS},
            "r_squared": fit.rsquared,
        }
    )

rolling_har = pd.DataFrame(rolling_rows).set_index("timestamp")
print(f"Refits: {len(rolling_har)}, each on {REFIT_WINDOW} sessions")

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

ax = axes[0]
for name, color in zip(HORIZONS, [COLORS["blue"], COLORS["amber"], COLORS["copper"]]):
    ax.plot(rolling_har.index, rolling_har[name], linewidth=1, color=color, label=name)
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Coefficient")
ax.set_title("Which horizon carries the weight, and when")
ax.legend(fontsize=7)

ax = axes[1]
ax.fill_between(rolling_har.index, 0, rolling_har["r_squared"], alpha=0.3, color=COLORS["blue"])
ax.plot(rolling_har.index, rolling_har["r_squared"], linewidth=1, color=COLORS["blue"])
ax.set_ylabel("R-squared")
ax.set_xlabel("Session the window ends")
ax.set_title("How much of the variation the fit explains")

fig.suptitle(f"Refitted every {REFIT_STEP} sessions, each fit reading {REFIT_WINDOW}")
show_with_alt(
    fig,
    "Two stacked panels over the refit dates. The top draws the three HAR coefficients, "
    "which move substantially and cross each other repeatedly; the weekly one is the "
    "largest over much of the sample, the daily one overtakes it in places, and the monthly "
    "one dips below zero in a few windows. The bottom draws the fit's R-squared, which "
    "rises and falls between about a sixth and three quarters across the sample.",
)

# %% [markdown]
# ## The shape of the term structure
#
# The ratio of the short horizon to the long one is a single number saying whether
# volatility right now is above or below where it has been. Above one, the recent past has
# been more volatile than the month; below one, less. It is bounded in practice, means the
# same thing in every year, and is the form a conditioning feature usually wants for the
# reason `08_garch_volatility` gave about ranks. The figure draws it smoothed over a
# month, because the unsmoothed daily ratio is dominated by whichever single session
# sits in its numerator and reaches several times the average.

# %%
term_structure = (
    spy_volatility.with_columns(
        short=pl.col("volatility").rolling_mean(HORIZONS["daily"]),
        long=pl.col("volatility").rolling_mean(HORIZONS["monthly"]),
    )
    .with_columns(ratio=pl.col("short") / pl.col("long"))
    .drop_nulls()
)

frame = term_structure.select(["timestamp", "ratio"]).to_pandas().set_index("timestamp")
smoothed = frame["ratio"].rolling(HORIZONS["monthly"]).mean()

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(frame.index, frame["ratio"], linewidth=0.4, alpha=0.4, color=COLORS["recede"])
ax.plot(frame.index, smoothed, linewidth=1, color=COLORS["blue"])
ax.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=0.6)
ax.fill_between(
    frame.index, 1.0, smoothed, where=smoothed > 1.0, alpha=0.25, color=COLORS["copper"]
)
ax.fill_between(frame.index, 1.0, smoothed, where=smoothed <= 1.0, alpha=0.25, color=COLORS["blue"])
ax.set_ylabel("Short horizon over long")
ax.set_xlabel("Session")
ax.set_title("Volatility above its own month, and below it")
show_with_alt(
    fig,
    "The ratio of a one-session volatility to a twenty-two-session average, drawn faintly "
    "day by day with a smoothed line over it and a dashed reference at one. The smoothed "
    "line oscillates around the reference, shaded on one side when above and the other "
    "when below, with the largest excursions above during the sharpest sell-offs.",
)

# %% [markdown]
# # Part 3: how jagged is the path
#
# The **Hurst exponent** $H$ describes how the size of a series' movement scales with the
# length of the interval measured. For a random walk the distance covered grows with the
# square root of the interval, which is $H = 1/2$. Above a half the series is persistent:
# a move is more likely to be followed by another in the same direction, so the path
# wanders further than a random walk would. Below a half it is anti-persistent, or
# **rough**: moves tend to be reversed, so the path is jagged and covers less ground.
#
# Two things are worth separating before estimating it. Applied to **returns**, $H$ says
# whether a price is trending or reverting, and for a liquid market it comes out near a
# half because anything else would be a tradable pattern. Applied to **log-volatility**,
# it says how jagged the volatility path is, and Gatheral, Jaisson and Rosenbaum (2018)
# found it comes out near a tenth across every asset they measured, far below the half
# that the standard volatility models had assumed for decades.

# %% [markdown]
# ## Two ways to estimate it
#
# **Rescaled range** is the original method. Cut the series into windows of size $s$;
# inside each, take the cumulative deviations from that window's mean, measure the range
# they cover, and divide by the window's standard deviation. Average over windows, repeat
# for several sizes, and the slope of the average against the size on a log scale is $H$.
#
# **Detrended fluctuation analysis** does the same and is unaffected by a trend. It
# integrates the series first, removes a straight line from each window rather than a
# mean, and measures what is left. The slope of that against the window size is the same
# exponent, and it does not move when the series has a drift the rescaled range would
# absorb into its estimate.
#
# Both fit a straight line on a log scale, so both report an $R^2$ alongside the exponent,
# and that $R^2$ is the check on whether the scaling relationship the method assumes holds
# at all. An exponent read off a set of points that are not on a line is not an estimate
# of anything.

# %%
MINIMUM_WINDOW = 10
WINDOW_GROWTH = 1.5


def window_sizes(n: int, minimum: int, maximum: int | None = None) -> list[int]:
    """Geometrically spaced window sizes, so the log axis is evenly covered."""
    maximum = maximum if maximum is not None else n // 4
    sizes, size = [], minimum
    while size <= maximum:
        sizes.append(size)
        size = int(size * WINDOW_GROWTH)
    return sorted(set(sizes))


def rescaled_range(series: np.ndarray, minimum: int = 20, maximum: int | None = None):
    """Hurst exponent by rescaled range, with the points the line was fitted to."""
    log_sizes, log_statistic = [], []
    for size in window_sizes(len(series), minimum, maximum):
        values = []
        for index in range(len(series) // size):
            window = series[index * size : (index + 1) * size]
            spread = window.std(ddof=1)
            if spread > 0:
                deviations = np.cumsum(window - window.mean())
                values.append((deviations.max() - deviations.min()) / spread)
        if values:
            log_sizes.append(np.log(size))
            log_statistic.append(np.log(np.mean(values)))
    fit = stats.linregress(log_sizes, log_statistic)
    return np.array(log_sizes), np.array(log_statistic), fit.slope, fit.rvalue**2


def detrended_fluctuation(
    series: np.ndarray, minimum: int = MINIMUM_WINDOW, maximum: int | None = None, order: int = 1
):
    """Hurst exponent by detrended fluctuation analysis, with its fitted points."""
    profile = np.cumsum(series - series.mean())
    log_sizes, log_fluctuation = [], []
    for size in window_sizes(len(series), minimum, maximum):
        steps = np.arange(size)
        fluctuations = []
        for index in range(len(series) // size):
            segment = profile[index * size : (index + 1) * size]
            trend = np.polyval(np.polyfit(steps, segment, order), steps)
            fluctuations.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if fluctuations:
            log_sizes.append(np.log(size))
            log_fluctuation.append(np.log(np.mean(fluctuations)))
    fit = stats.linregress(log_sizes, log_fluctuation)
    return np.array(log_sizes), np.array(log_fluctuation), fit.slope, fit.rvalue**2


# %% [markdown]
# ## The two series, and what each one says
#
# Returns go in directly. Volatility goes in as the **increments of its logarithm**,
# which is the quantity the rough-volatility result is about: the model it argues against
# says those increments behave like a random walk's, and the finding is that they do not.

# %%
returns_series = spy_volatility["log_return"].to_numpy()
log_volatility = np.log(spy_volatility["volatility"].to_numpy().clip(min=1e-10))
log_volatility_increments = np.diff(log_volatility)

estimates = []
for name, series in [
    ("returns", returns_series),
    ("log-volatility increments", log_volatility_increments),
]:
    rs_sizes, rs_values, rs_exponent, rs_fit = rescaled_range(series)
    dfa_sizes, dfa_values, dfa_exponent, dfa_fit = detrended_fluctuation(series)
    estimates.append(
        {
            "series": name,
            "H, rescaled range": rs_exponent,
            "R-squared, rescaled range": rs_fit,
            "H, detrended fluctuation": dfa_exponent,
            "R-squared, detrended fluctuation": dfa_fit,
            "_points": (rs_sizes, rs_values, dfa_sizes, dfa_values),
        }
    )

display(pd.DataFrame(estimates).drop(columns="_points"))

# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["grid_3x2"])

for row, estimate in enumerate(estimates):
    rs_sizes, rs_values, dfa_sizes, dfa_values = estimate["_points"]
    for column, (sizes, values, exponent, label) in enumerate(
        [
            (rs_sizes, rs_values, estimate["H, rescaled range"], "Rescaled range"),
            (dfa_sizes, dfa_values, estimate["H, detrended fluctuation"], "Detrended fluctuation"),
        ]
    ):
        ax = axes[row, column]
        ax.scatter(sizes, values, s=14, color=COLORS["blue"], zorder=3)
        ax.plot(
            sizes,
            exponent * sizes + (values[0] - exponent * sizes[0]),
            linestyle="--",
            linewidth=1,
            color=COLORS["negative"],
        )
        ax.set_title(f"{label}, {estimate['series']}")
        ax.set_xlabel("Log window size")
        ax.set_ylabel("Log statistic")

fig.suptitle("The exponent is a slope, so the points have to be on a line")
show_with_alt(
    fig,
    "A two by two grid of log-log scatter plots with a dashed fitted line through each. "
    "The top row is for returns and the bottom for log-volatility increments; the left "
    "column is the rescaled range and the right the detrended fluctuation. The returns "
    "points lie close to their lines with a clear positive slope; the log-volatility "
    "points lie on much flatter lines.",
)

# %%
for estimate in estimates:
    print(
        f"{estimate['series']}: "
        f"rescaled range {estimate['H, rescaled range']:.3f}, "
        f"detrended fluctuation {estimate['H, detrended fluctuation']:.3f}, "
        f"against 0.5 for a random walk"
    )

# %% [markdown]
# Read the two rows against a half and against each other. Whichever method is used, the
# two series land in different places, and the direction of the difference is the finding:
# returns scale close to a random walk's, and the increments of log-volatility scale far
# below it, meaning volatility reverses direction much more often than a random walk does.
#
# That is what "rough" means, and its consequence is practical. A model built on the
# assumption that log-volatility is a random walk, which describes most of the standard
# volatility models, is assuming a path smoother than the one the data traces. It will
# be slow to follow volatility down after a spike, because it expects the level to persist
# when in fact the next increment is more likely than not to reverse the last.
#
# The two methods do not agree exactly, and neither is a measurement of a physical
# constant. Each estimates a slope from a handful of points, the points are averages over
# overlapping structure, and the estimate moves with the window range chosen. Read the
# exponent together with the $R^2$ beside it, and treat a difference of a few hundredths
# between methods as noise rather than as a finding.

# %% [markdown]
# ## Roughness over time
#
# Estimated on a moving window, the exponent becomes a column. For returns it is a slow
# statement about whether the market is trending or reverting; for log-volatility it says
# how bursty the current volatility regime is.

# %%
HURST_WINDOW = 252
HURST_STEP = 10  # sessions between estimates; each estimate is a fit over many sub-windows

rolling_rows = []
for end in range(HURST_WINDOW, len(returns_series), HURST_STEP):
    _, _, returns_exponent, _ = detrended_fluctuation(
        returns_series[end - HURST_WINDOW : end], maximum=HURST_WINDOW // 4
    )
    # The increments series is one shorter than the returns series, so its window ends one
    # index earlier and covers the same sessions.
    _, _, volatility_exponent, _ = detrended_fluctuation(
        log_volatility_increments[max(0, end - HURST_WINDOW - 1) : end - 1],
        maximum=HURST_WINDOW // 4,
    )
    rolling_rows.append(
        {
            "timestamp": spy_volatility["timestamp"][end],
            "returns": returns_exponent,
            "log-volatility": volatility_exponent,
        }
    )

rolling_hurst = pd.DataFrame(rolling_rows).set_index("timestamp")
print(f"Rolling estimates: {len(rolling_hurst)}, each fitted on {HURST_WINDOW} sessions")
print(
    "Median exponent: "
    f"returns {rolling_hurst['returns'].median():.3f}, "
    f"log-volatility {rolling_hurst['log-volatility'].median():.3f}"
)
print(
    "Share of windows above one half: "
    f"returns {(rolling_hurst['returns'] > 0.5).mean():.1%}, "
    f"log-volatility {(rolling_hurst['log-volatility'] > 0.5).mean():.1%}"
)

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

for ax, column, color, title in [
    (axes[0], "returns", COLORS["blue"], "Returns: trending above the line, reverting below"),
    (axes[1], "log-volatility", COLORS["copper"], "Log-volatility: rough throughout"),
]:
    ax.plot(rolling_hurst.index, rolling_hurst[column], linewidth=0.9, color=color)
    ax.axhline(0.5, color=COLORS["negative"], linestyle="--", linewidth=0.7)
    ax.annotate(
        "A random walk",
        xy=(0.995, 0.5),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color=COLORS["negative"],
    )
    ax.set_ylabel("Exponent")
    ax.set_title(title)
axes[1].set_xlabel("Session the window ends")

fig.suptitle(f"The exponent on a {HURST_WINDOW}-session moving window")
show_with_alt(
    fig,
    "Two stacked panels, each drawing a rolling Hurst exponent against a dashed reference "
    "at one half. The top, for returns, oscillates around the reference and crosses it "
    "repeatedly. The bottom, for log-volatility increments, stays well below the reference "
    "throughout the sample without approaching it.",
)

# %% [markdown]
# The top panel is a feature and the bottom panel is a fact about markets. Returns cross
# the line in both directions, so a column holding that exponent separates periods; the
# log-volatility exponent never comes near it, so a column holding that one separates
# almost nothing and its value is a property of the asset class rather than of the moment.
#
# That asymmetry is the practical reading of the rough-volatility literature. The finding
# is important because it says the standard model is wrong, and it is a poor feature for
# exactly the same reason: it is wrong everywhere, in the same direction, all the time.

# %% [markdown]
# ## The library expression, and what it measures
#
# `hurst_exponent` computes a rolling estimate as a Polars expression, and it takes a
# **price** column. That is a third series, different from both of the ones above: the
# rescaled range of a price level rather than of its returns or of its volatility. The
# three answer different questions and there is no reason for them to agree.

# %%
library_hurst = spy.with_columns(hurst=hurst_exponent("close", period=HURST_WINDOW))
library_values = library_hurst["hurst"].drop_nulls()

display(
    pd.DataFrame(
        [
            {
                "series": "close price, library expression",
                "median": library_values.median(),
                "share above one half": float((library_values > 0.5).mean()),
            },
            {
                "series": "returns, detrended fluctuation",
                "median": rolling_hurst["returns"].median(),
                "share above one half": float((rolling_hurst["returns"] > 0.5).mean()),
            },
        ]
    )
)

# %% [markdown]
# The two differ, and the difference is the input rather than the implementation. Use the
# expression where a rolling exponent of the price is what is wanted and the cost of the
# manual version across a panel is prohibitive; use the manual version where the series
# being characterised is a return or a volatility, which is most of the time in this
# chapter.

# %% [markdown]
# # Part 4: across the panel
#
# Every number above comes from one symbol. Fitting the same three quantities across the
# ETF panel says which of them are properties of this symbol and which are properties of
# the asset class.

# %%
MINIMUM_PANEL_SESSIONS = 600

panel_source = (
    load_etfs()
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .sort(["symbol", "timestamp"])
)


def panel_row(symbol: str) -> dict | None:
    """HAR coefficients and both exponents for one symbol, or None where the history is short."""
    frame = panel_source.filter(pl.col("symbol") == symbol).select(
        ["timestamp", "open", "high", "low", "close"]
    )
    if frame.height < MINIMUM_PANEL_SESSIONS:
        return None

    volatility = range_volatility(frame)
    model_frame = har_frame(volatility)
    if model_frame.height < MINIMUM_PANEL_SESSIONS // 2:
        return None

    fit = fit_har(model_frame)
    increments = np.diff(np.log(volatility["volatility"].to_numpy().clip(min=1e-10)))
    _, _, returns_exponent, _ = detrended_fluctuation(volatility["log_return"].to_numpy())
    _, _, volatility_exponent, _ = detrended_fluctuation(increments)

    return {
        "symbol": symbol,
        **{name: fit.params[name] for name in HORIZONS},
        "r_squared": fit.rsquared,
        "H returns": returns_exponent,
        "H log-volatility": volatility_exponent,
    }


panel = pd.DataFrame(
    [row for symbol in panel_source["symbol"].unique().sort() if (row := panel_row(symbol))]
)
print(f"Symbols fitted: {len(panel)} of {panel_source['symbol'].n_unique()}")
display(panel.describe().loc[["mean", "std", "min", "max"]].round(4))

# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["grid_3x2"])

ax = axes[0, 0]
for name, color in zip(HORIZONS, [COLORS["blue"], COLORS["amber"], COLORS["copper"]]):
    ax.hist(panel[name], bins=25, alpha=0.6, color=color, label=name)
ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.6)
ax.set_xlabel("Coefficient")
ax.set_ylabel("Symbols")
ax.set_title("The three HAR weights across the panel")
ax.legend(fontsize=7)

ax = axes[0, 1]
ax.hist(panel["r_squared"], bins=25, color=COLORS["blue"])
ax.set_xlabel("R-squared")
ax.set_ylabel("Symbols")
ax.set_title("How much of volatility the model explains")

for ax, column, reference, title in [
    (axes[1, 0], "H returns", 0.5, "Exponent on returns"),
    (axes[1, 1], "H log-volatility", 0.5, "Exponent on log-volatility"),
]:
    ax.hist(panel[column], bins=25, color=COLORS["copper"])
    ax.axvline(reference, color=COLORS["negative"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Exponent")
    ax.set_ylabel("Symbols")
    ax.set_title(title)

fig.suptitle("What varies across symbols, and what does not")
show_with_alt(
    fig,
    "A two by two grid of histograms across the ETF panel. Top left overlays the three HAR "
    "coefficients, which occupy different and partly overlapping ranges. Top right is the "
    "R-squared, concentrated over a moderate range. Bottom left is the exponent on returns, "
    "clustered near the dashed reference at one half. Bottom right is the exponent on "
    "log-volatility, clustered far below that reference with no overlap between the two.",
)

# %%
print(
    "Exponent on returns: "
    f"median {panel['H returns'].median():.3f}, "
    f"{(panel['H returns'] > 0.5).sum()} of {len(panel)} symbols above one half"
)
print(
    "Exponent on log-volatility: "
    f"median {panel['H log-volatility'].median():.3f}, "
    f"{(panel['H log-volatility'] > 0.5).sum()} of {len(panel)} symbols above one half"
)
print(
    "HAR coefficient with the largest median: "
    + max(HORIZONS, key=lambda name: panel[name].median())
)

# %% [markdown]
# The two bottom panels are the answer to the question the panel was fitted to settle. The
# exponent on returns clusters close to a half with symbols on both sides of it, which is
# what a liquid market should look like. The exponent on log-volatility sits far below a
# half for every symbol, with no overlap at all between the two distributions, which
# reproduces the rough-volatility finding across a hundred assets rather than one.
#
# Compare the returns figure here against the rolling one two sections up before reading
# either as a level. This one estimates the exponent once over each symbol's whole history;
# that one estimates it over 252 sessions at a time, and their medians differ. An exponent
# is a slope through averaged points and the estimate moves with the range of window sizes
# the fit was given, so the two numbers are answers to slightly different questions rather
# than a disagreement about the market. What both agree on is the direction, which is that
# returns sit near a half and volatility does not.
#
# The HAR coefficients vary more across symbols than either exponent does, and that
# variation is what makes them worth carrying per symbol rather than fitting once.

# %% [markdown]
# ## The features this notebook produces
#
# | Column | What it is | Causal |
# |---|---|---|
# | `daily`, `weekly`, `monthly` | lagged averages of range-based volatility over three horizons | yes |
# | HAR coefficients | the weight on each horizon, refitted on a moving window | yes, refit on past windows |
# | term structure ratio | the short horizon over the long one | yes |
# | exponent on returns | how trending or reverting the price has been, on a moving window | yes |
# | exponent on log-volatility | how jagged the volatility path has been | yes |
#
# The realized volatility measured from minute bars is not on the list, because it exists
# for two symbols and two years and the point of Part 1 was to establish which daily
# estimator stands in for it.

# %% [markdown]
# ## Key takeaways
#
# 1. **Check the estimator against a measurement where you can.** A squared close-to-close
#    return misses a session that moved and came back; the range-based estimators see it,
#    and comparing them against volatility measured from minute bars is how you find out by
#    how much and in which direction each is biased.
# 2. **HAR is a linear regression, and that is its advantage.** Three lagged averages track
#    realized volatility about as closely as a fitted GARCH does, and unlike GARCH each
#    coefficient is readable as the weight one horizon carries.
# 3. **Overlapping regressors are a reason to widen the standard errors, not a proof the
#    residuals are correlated.** A correctly specified model can have independent
#    innovations despite rolling-average regressors. To read what one relaxed assumption
#    costs, change only that assumption: the same Newey-West estimator at zero lags and at
#    the model's longest horizon differs in the window alone, which no comparison across two
#    different covariance estimators can claim.
# 4. **Two models forecasting differently-measured targets cannot be ranked by their
#    errors.** Report the mean of each forecast next to the mean of the target, and read
#    the correlation, which the level difference does not touch.
# 5. **Roughness is a finding about markets and a poor feature.** The exponent on
#    log-volatility sits far below a half for every symbol in the panel, which is why the
#    standard models are wrong and also why a column holding it separates nothing. The
#    exponent on returns crosses a half in both directions and is the one worth carrying.
#
# **Known limitations.** Part 1 compares estimators on one symbol over two years, both
# chosen by what intraday data exists rather than by what would be representative. The HAR
# and GARCH comparison uses one split at one date. Every Hurst estimate is a slope through
# a handful of averaged points and moves with the window range chosen, so differences of a
# few hundredths between methods carry no information. And the panel's ETFs overlap
# heavily in what they hold, so its hundred rows are far fewer than a hundred independent
# observations.
#
# **Previous**: `08_garch_volatility` for a single-horizon volatility model.
# **Next**: `10_uncertainty_features`, which treats the model's own uncertainty as the
# feature rather than its estimate.
