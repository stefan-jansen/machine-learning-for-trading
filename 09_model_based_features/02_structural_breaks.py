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
# # Structural Break Detection
#
# **Chapter 9 | Section 9.1**
#
# **Docker image**: `ml4t`
#
# A **structural break** is a date after which the process generating a series is not the
# one that generated it before: the average level moves, the size of a typical move
# changes, or the way one observation depends on the last changes. Every model fitted on
# the earlier period is describing something that has stopped happening, which is why
# locating breaks is worth doing before fitting anything and worth monitoring afterwards.
#
# This notebook covers three ways of getting at them, in increasing generality: a test
# that allows exactly one break, a segmentation that finds several, and a classifier that
# learns what a break looks like from a set of statistics comparing the period before a
# candidate date with the period after it.
#
# **Learning objectives**
#
# - Test for a break in a price series with a test that estimates the break date rather
#   than requiring you to name it in advance.
# - Split a long series into segments whose average level is stable inside each segment,
#   and read the dates the split produces.
# - Turn a detected break into a column a model can read on a given date without using
#   anything that happened after it.
# - Monitor for a break as new observations arrive, using a statistic that accumulates
#   evidence and one that looks only at a recent window, and say what each is good for.
# - Build the statistics that compare two adjacent windows - their averages, their
#   spreads, their whole distributions, and their dependence - and combine them into a
#   single detector.
#
# **Book reference**
#
# Chapter 9, Section 9.1 (Diagnostics and stationarity features).
#
# **Prerequisites**
#
# `01_visual_diagnostics` for stationarity testing and the ADF test. Familiarity with
# hypothesis testing and with reading a p-value.

# %% [markdown]
# ## Setup

# %%
"""Structural Break Detection - classical tests, causal features, and classification."""

import logging
import warnings
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import ruptures as rpt
from arch.unitroot import ZivotAndrews
from IPython.display import display
from lightgbm import LGBMClassifier
from ml4t.engineer.features.statistics import (
    coefficient_of_variation,
    rolling_drift,
    rolling_kl_divergence,
    rolling_wasserstein,
)
from ml4t.engineer.logging import setup_logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict, cross_val_score

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, add_message_title, show_with_alt

# Per-call timing notices from the feature library are about this machine, not the data.
# Each feature logger reads this level when created, so the call precedes the first use.
setup_logging(level=logging.ERROR)
# `ruptures` warns once per fit that its default jump argument will change; the fits
# below all pass their own arguments, so the change does not reach them.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ruptures")

# %% tags=["parameters"]
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## The series
#
# SPY, the exchange-traded fund tracking the S&P 500, is the working example: two decades
# of daily closes containing two episodes nobody disputes were breaks, the 2008 credit
# crisis and the March 2020 shutdown, and a great deal of ordinary market that was not.
# A detector worth having separates those.
#
# `START_DATE` and `END_DATE` bound the sessions every method here may read. The ETF
# panel's SPY history opens in 2006, so the requested start binds nothing and the sample
# is what the panel holds.
#
# Returns are **log returns**, the difference of the logged close, scaled to percent.
# Logs make the return over several days the sum of the daily returns, which is the
# property the cumulative statistics later in the notebook rely on.

# %%
etfs = load_etfs(symbols=["SPY"])

START = datetime.strptime(START_DATE, "%Y-%m-%d")
END = datetime.strptime(END_DATE, "%Y-%m-%d")

spy = (
    etfs.filter((pl.col("timestamp") >= START.date()) & (pl.col("timestamp") <= END.date()))
    .sort("timestamp")
    .select(["timestamp", "close"])
)

spy_pd = spy.to_pandas().set_index("timestamp")
spy_pd.index = pd.DatetimeIndex(spy_pd.index)
spy_pd["returns"] = np.log(spy_pd["close"]).diff() * 100
returns = spy_pd["returns"].dropna()

print(
    f"SPY: {len(spy_pd):,} sessions "
    f"({spy_pd.index.min().date()} to {spy_pd.index.max().date()}), "
    f"{len(returns):,} daily returns"
)

# %% [markdown]
# # Part 1: finding breaks in a price series

# %% [markdown]
# ## One break, with its date estimated: Zivot-Andrews
#
# The ADF test asks whether a series has a unit root under the assumption that one
# process generated the whole sample. If the sample contains a break, that assumption is
# wrong in a specific direction: a stationary series whose mean jumps once looks, to a
# test that cannot see the jump, exactly like a series that wanders. The ADF test then
# fails to reject a unit root that is not there.
#
# The **Zivot-Andrews** test removes the assumption. It fits the ADF regression once for
# every candidate break date, allowing the intercept and the trend to change at that
# date, and reports the date whose fit is most hostile to the unit root. Its null
# hypothesis is a unit root with no break; rejecting it says the series is stationary
# around a level and trend that shift once.
#
# Three settings decide what the test can find. `trend="ct"` allows both the level and
# the slope to shift at the break, which is the general case for a price series.
# `trim` refuses candidate dates in the first and last fraction of the sample it is
# given, here fifteen percent at each end, because a break estimated from a handful of
# observations is not estimated at all. `lags=12` is the number of lagged differences in the regression: enough to
# absorb short-run dynamics at daily frequency without spending the sample on them.


# %%
def run_zivot_andrews(series: pd.Series, name: str) -> dict:
    """Run the Zivot-Andrews test and report the break date it selected."""
    series = series.dropna()
    za = ZivotAndrews(series, lags=12, trend="ct", trim=0.15)

    return {
        "series": name,
        "statistic": za.stat,
        "p_value": za.pvalue,
        "critical_1pct": za.critical_values["1%"],
        "critical_5pct": za.critical_values["5%"],
        "rejects_unit_root": bool(za.pvalue < 0.05),
    }


za_results = pd.DataFrame(
    [
        run_zivot_andrews(spy_pd["close"], "SPY close"),
        run_zivot_andrews(returns, "SPY daily return"),
    ]
)
display(za_results)

# %% [markdown]
# The test carries the same caution as any single-break test. Failing to reject leaves
# two possibilities open that it cannot distinguish: the series has a unit root, or it
# has more than one break and no single shift describes it. The next section allows
# several.

# %% [markdown]
# ## Several breaks: segmenting the series
#
# **Segmentation** takes a different route. Rather than testing a hypothesis it partitions
# the series into consecutive segments, choosing the split points that minimise the total
# within-segment cost. With a squared-error cost the fitted value inside each segment is
# its mean, so the algorithm is looking for the places where the average level moves.
#
# Two algorithms in `ruptures` implement this, and the difference between them is what
# you have to supply:
#
# - **PELT** is given a penalty per additional break and returns however many breaks pay
#   for themselves. The penalty here is the log of the sample size times the variance of
#   the series, which is the shape of the Bayesian information criterion: the variance
#   puts the penalty in the units of the cost being minimised, and the log makes the
#   price of a break grow slowly with the sample.
# - **Binary segmentation** is given the number of breaks and splits recursively, taking
#   the split that reduces the cost most, then splitting each part the same way.
#
# `min_size=60` refuses a segment shorter than about a quarter, which stops both
# algorithms from spending breaks on individual selloffs.

# %%
MIN_SEGMENT = 60
REQUESTED_BREAKS = 4


def detect_multiple_breaks(series: np.ndarray, n_breaks: int, model: str = "l2") -> dict:
    """Segment *series* two ways: PELT under a penalty, and a fixed number of splits."""
    penalty = np.log(len(series)) * series.var()
    pelt = rpt.Pelt(model=model, min_size=MIN_SEGMENT).fit(series).predict(pen=penalty)
    binseg = rpt.Binseg(model=model, min_size=MIN_SEGMENT).fit(series).predict(n_bkps=n_breaks)

    # Both predictors return the end of the series as the final boundary; it is not a break.
    return {"pelt": pelt[:-1], "binseg": binseg[:-1]}


prices = spy_pd["close"].to_numpy()
dates = spy_pd.index
breaks = detect_multiple_breaks(prices, n_breaks=REQUESTED_BREAKS)

print(f"PELT, under the penalty: {len(breaks['pelt'])} breaks")
print(f"Binary segmentation, as requested: {len(breaks['binseg'])} breaks")

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

for ax, key, title in [
    (axes[0], "pelt", "PELT chooses how many breaks the penalty will pay for"),
    (axes[1], "binseg", f"Binary segmentation splits {REQUESTED_BREAKS} times as asked"),
]:
    ax.plot(dates, prices, linewidth=0.8, color=COLORS["blue"])
    for position in breaks[key]:
        ax.axvline(dates[position], color=COLORS["negative"], linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("US dollars")

axes[1].set_xlabel("Session")
fig.suptitle("Where the average level of the SPY close moves")
show_with_alt(
    fig,
    "Two stacked panels, each drawing the SPY closing price over the sample with vertical "
    "dashed lines at the detected breaks. The upper panel, from PELT under a penalty, "
    "carries eight lines, none of them before 2012 and three of them after 2020. The lower "
    "panel, from binary segmentation, carries four lines spread across the second half of "
    "the sample. Neither panel marks the 2008 trough or the 2020 drop.",
)

# %% [markdown]
# The two panels disagree, and the disagreement is the point rather than a defect. Binary
# segmentation was told how many breaks to find and found exactly that many wherever they
# fit best; PELT was told what a break costs and returned as many as cleared the price.
# Neither number is the true number of breaks, because the segmentation has no notion of
# one: it reports the lowest-cost partition under a cost function and a constraint you
# chose.
#
# What neither panel marks is more instructive than what they do. Neither crash gets a
# line, and the reason is the cost being minimised: squared error around a segment mean,
# on a price that rose several hundred percent over the sample. The two quantities printed
# below are the scale of the two things competing for a split, and the drawdown is much
# the smaller, so the splits land where the average price moved rather than where the
# market broke.
#
# That is a property of the question asked, not a shortcoming of the algorithm. A
# mean-shift cost on a price level finds shifts in the price level. Volatility breaks live
# in a different series and need a different cost, which is what the rest of this notebook
# is about.

# %%
annual_mean = spy_pd["close"].resample("YE").mean()
deepest_fall = (spy_pd["close"].cummax() - spy_pd["close"]).max()
print(f"Deepest peak-to-trough fall in the sample: {deepest_fall:,.0f} dollars")
print(
    "Widest gap between two annual average closes: "
    f"{annual_mean.max() - annual_mean.min():,.0f} dollars"
)

# %% [markdown]
# ## Turning a break into a feature the reader could have had
#
# Everything above places its breaks using the whole sample. For describing history that
# is what you want. For building a feature it is a look-ahead: on the session PELT marks
# in 2012, the algorithm had not seen the twelve years that made that session look like a
# break, and a model trained on a column built this way is reading a date it could not
# have known.
#
# The fix is to refit. At each refit date the segmentation runs on sessions available up
# to that date and nothing later, and the columns it produces hold until the next refit.
# Three settings decide the shape of that. `REFIT_EVERY` is how often the fit is repeated,
# quarterly here, which is the cadence at which a feature of this cost would realistically
# be rebuilt. `LOOKBACK` is how much history each fit reads: five years rather than
# everything, because the question the feature answers is where the current regime began,
# a break twenty years back does not change that answer, and the cost of a fit grows with
# the history it is given. `LEVEL_WINDOW` bounds how much data either side of a break is
# averaged to measure how far the level moved.
#
# A lookback that does not reach back forever has a consequence to state rather than hide:
# `sessions_since_break` is censored at the lookback. A window containing no break reports
# no break, which means "none in five years", not "none ever".

# %%
LOOKBACK = 1260  # sessions each refit reads: about five years
REFIT_EVERY = 63  # sessions between refits: about one quarter
LEVEL_WINDOW = 252  # sessions either side of a break averaged for its level shift


def causal_break_features(series: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Refit the segmentation forward through the sample, reading only the past."""
    rows = []
    for stop in range(LOOKBACK, len(series), REFIT_EVERY):
        history = series[stop - LOOKBACK : stop]
        penalty = np.log(len(history)) * history.var()
        found = rpt.Pelt(model="l2", min_size=MIN_SEGMENT).fit(history).predict(pen=penalty)
        # Positions are offsets into the lookback; the last one is its end, not a break.
        found = [stop - LOOKBACK + position for position in found if position < LOOKBACK]

        last = max(found) if found else None
        row = {
            "timestamp": index[stop],
            "breaks_in_lookback": len(found),
            "break_detected": int(bool(found)),
            "sessions_since_break": stop - last if last is not None else np.nan,
            "level_shift_usd": np.nan,
        }
        if last is not None:
            before = series[max(0, last - LEVEL_WINDOW) : last]
            after = series[last : min(stop, last + LEVEL_WINDOW)]
            if len(before) and len(after):
                row["level_shift_usd"] = after.mean() - before.mean()
        rows.append(row)

    refits = pd.DataFrame(rows).set_index("timestamp")
    # A refit answers for every session until the next one, so the quarterly rows are
    # carried forward onto the session index. Sessions before the first refit stay empty:
    # nothing had been fitted yet, and filling them backwards would invent a value.
    return refits.reindex(index, method="ffill")


break_features = causal_break_features(prices, dates)
display(break_features.dropna(how="all").head(10))

# %% [markdown]
# Read the first rows against the retrospective figure above and the difference is
# visible: a break the full-sample fit places does not appear in this table until a refit
# whose lookback both reached it and had enough data after it to pay the penalty.
# `sessions_since_break` counts from a break that was detectable at the time, so a model
# reading the value stamped on a date is reading something a person could have computed
# that morning. `level_shift_usd` is in the units of the series it was measured on, which
# for an unadjusted price level means the same shift is a different number in 2008 and in
# 2024; a model reading it across the whole sample wants it scaled.
#
# `breaks_in_lookback` is comparable across refits and not comparable with the count from
# the full-sample fit above. The penalty is the log of the sample size times the variance
# of the series being segmented, so it is denominated in the variance of whatever window
# it was computed on. Every refit here reads the same number of sessions, which is what
# makes the column readable over time; the full-sample fit priced a break against nineteen
# years of variance and therefore bought far fewer of them.
#
# Two properties of this construction are worth carrying to any other refitted feature.
# The columns are step functions between refits, so a model reading them daily sees the
# same value for a quarter at a time, and their variation across the sample is far
# smaller than the daily variation of the underlying series. And a break's estimated
# position can move at the next refit as more data arrives, so the column is a running
# estimate rather than a record of events.

# %% [markdown]
# ## Monitoring: CUSUM and MOSUM
#
# Segmentation answers "where were the breaks", offline, over a sample you already have.
# Monitoring answers "has one just happened", online, as observations arrive. Both
# statistics below compare what is arriving against a reference estimated once, from a
# **burn-in** period at the start of the sample that is treated as representative and
# never re-estimated. That is what makes them causal: at any later date the statistic uses
# the burn-in and the observations up to that date, and nothing else.
#
# The **cumulative sum** (CUSUM) adds up the deviations from the reference mean. Under no
# change it wanders around zero; after a shift it drifts steadily in the direction of the
# shift and stays there, because every subsequent observation adds to the total.
#
# The **moving sum** (MOSUM) adds up the same deviations over a fixed-width window ending
# at the current date, standardized by the reference scale:
#
# $$M_t = \frac{1}{\hat{\sigma}\sqrt{h}} \sum_{i=t-h+1}^{t} (x_i - \bar{x}_{\text{burn-in}})$$
#
# Because the sum has a fixed number of terms, its value depends only on the last $h$
# observations: each arrival pushes one observation out of the window. That is the whole
# difference from CUSUM, and it is worth being precise about what it does and does not
# buy. A **temporary** excursion leaves the statistic once the window has slid past it,
# which is what makes MOSUM readable for the next event and useful for dating one. A
# **permanent** shift of size $\Delta$ does not go away, but it also stops accumulating:
# the expected value of the statistic moves to $\Delta\sqrt{h}/\hat{\sigma}$ and stays
# there while the statistic itself keeps fluctuating around it. A mean that keeps moving
# keeps moving the statistic, so a fixed window bounds what a *constant* shift can do to
# it, not what any shift can.

# %%
BURN_IN = 252  # sessions used to estimate the reference mean and scale
MOSUM_BANDWIDTHS = [20, 50]


def cusum_and_mosum(series: np.ndarray, burn_in: int, bandwidths: list[int]) -> pd.DataFrame:
    """Both monitoring statistics against a reference fixed on the first *burn_in* points."""
    reference_mean = series[:burn_in].mean()
    reference_std = series[:burn_in].std()
    deviations = series - reference_mean

    out = {"cusum": np.cumsum(deviations)}
    for bandwidth in bandwidths:
        moving = pd.Series(deviations).rolling(bandwidth).sum().to_numpy()
        out[f"mosum_{bandwidth}"] = moving / (reference_std * np.sqrt(bandwidth))

    frame = pd.DataFrame(out)
    frame.iloc[:burn_in] = np.nan  # the burn-in is the reference, not a monitored period
    return frame


monitoring = cusum_and_mosum(returns.to_numpy(), BURN_IN, MOSUM_BANDWIDTHS)
monitoring.index = returns.index

# %%
fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(returns.index, returns.to_numpy(), linewidth=0.3, alpha=0.5, color=COLORS["blue"])
ax.set_title("SPY daily log return")
ax.set_ylabel("Percent")

ax = axes[1]
ax.plot(monitoring.index, monitoring["cusum"], linewidth=0.8, color=COLORS["blue"])
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_title("CUSUM keeps every deviation it has seen")
ax.set_ylabel("Cumulative percent")

ax = axes[2]
for bandwidth, color in zip(MOSUM_BANDWIDTHS, [COLORS["amber"], COLORS["blue"]]):
    ax.plot(
        monitoring.index,
        monitoring[f"mosum_{bandwidth}"],
        linewidth=0.8,
        color=color,
        alpha=0.8,
        label=f"{bandwidth}-session window",
    )
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_title("MOSUM forgets everything outside its window")
ax.set_ylabel("Standardized sum")
ax.set_xlabel("Session")
ax.legend(fontsize=7, loc="lower left")

fig.suptitle(f"Two monitors against a {BURN_IN}-session reference")
show_with_alt(
    fig,
    "Three stacked panels sharing a time axis. Top: SPY daily log returns, with visible "
    "bursts of large moves in 2008 and 2020. Middle: the CUSUM statistic, which falls "
    "steeply through 2008 and then stays far below zero for the remaining fifteen years "
    "without recovering. Bottom: two MOSUM series at twenty and fifty session windows, "
    "both oscillating around zero with sharp excursions at the crisis dates and a return "
    "to zero within a few months afterwards.",
)

# %% [markdown]
# The two panels show the trade the choice is between. CUSUM keeps every deviation, so a
# shift moves the statistic permanently and a second shift is then read against a baseline
# the first one moved; the excursions in 2008 and 2020 never come back. MOSUM holds only
# what is inside its window, so a shift it has fully absorbed stops adding to it and the
# statistic stays readable for the next event, at the cost of being blind to a drift too
# gradual to register across $h$ observations. It
# comes back to zero here because the average return after each crisis is close to the
# burn-in average again, not because the statistic forgets a permanent shift.
#
# The bandwidth is the same trade at a smaller scale: a short window reacts sooner and
# crosses a threshold more often on noise alone, a long one is steadier and later. Neither
# is a default. Set it from how long a shift has to persist before you would want to act
# on it.

# %% [markdown]
# # Part 2: break detection as a classification problem
#
# The ADIA Lab structural break challenge poses the problem in a third form. Given a
# series and a candidate boundary date, decide whether the process changed at that
# boundary. The input is one series and one date; the output is a probability. That
# framing makes the problem supervised, so the detector is learned from labelled examples
# rather than derived from a null hypothesis, and it accepts evidence of any kind that
# separates the two sides of the boundary.
#
# The work is then in the features. Each one summarises a comparison between the window
# before the boundary and the window after it, and the families below cover the four ways
# two samples can differ: where they sit, how spread out they are, what shape they have,
# and how each observation depends on the one before it.

# %% [markdown]
# ## Labelled examples
#
# Real breaks are scarce and their dates are arguable, so the detector is trained on
# series whose construction is known. Each example is 500 steps with a candidate boundary
# at the midpoint. Half have nothing at the boundary; the other half have one of four
# changes, each of which a different family of features is built to see.
#
# The figure below draws three of the four. The fourth, a change in how strongly each
# value depends on the one before it, is left out because neither panel would show it: it
# alters no level, no spread and no distribution, only the order the same values arrive
# in. That is exactly why one of the five families measures dependence and none of the
# other four would catch it.
#
# Holding the spread fixed while the dependence changes takes care in the generator, in
# three places. A first-order autoregressive series with coefficient $\phi$ and
# unit-variance innovations has marginal variance $1/(1-\phi^2)$, so raising $\phi$ would
# raise the spread too: scaling the innovations by $\sqrt{1-\phi^2}$ fixes the marginal
# variance at one on both sides. Keeping $\phi$ below one keeps both halves stationary.
# And the recursion runs once through the whole series, starting from a draw with the
# stationary variance rather than from zero, because a series restarted at zero has
# variance $1 - \phi^{2i}$ at step $i$: starting each half separately would put a variance
# transient at exactly the boundary being tested, which is the thing the example is
# supposed not to contain.

# %%
N_EXAMPLES = 200
SERIES_LENGTH = 500
BREAK_TYPES = ["mean_shift", "var_shift", "trend_shift", "autocorr_shift"]
AR_PHI_BEFORE = 0.2  # the coefficient before the boundary; after it rises with magnitude


def generate_break_series(
    n: int, break_type: str, magnitude: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """A series of length *n* with the named change at its midpoint, or none."""
    rng = np.random.RandomState(seed)
    mid = n // 2

    if break_type == "none":
        return rng.randn(n)
    if break_type == "mean_shift":
        return np.concatenate([rng.randn(mid), rng.randn(n - mid) + magnitude])
    if break_type == "var_shift":
        return np.concatenate([rng.randn(mid), rng.randn(n - mid) * (1 + magnitude)])
    if break_type == "trend_shift":
        steps = np.arange(n, dtype=float)
        before = 0.001 * steps[:mid]
        after = before[-1] + magnitude * 0.01 * (steps[mid:] - steps[mid])
        return np.concatenate([before, after]) + rng.randn(n) * 0.5
    if break_type == "autocorr_shift":
        phi_after = AR_PHI_BEFORE + 0.35 * magnitude
        values = np.zeros(n)
        values[0] = rng.randn()
        for i in range(1, n):
            phi = AR_PHI_BEFORE if i < mid else phi_after
            values[i] = phi * values[i - 1] + np.sqrt(1 - phi**2) * rng.randn()
        return values

    msg = f"Unknown break_type: {break_type}"
    raise ValueError(msg)


magnitude_rng = np.random.RandomState(SEED)
series_list = []
labels = []

for i in range(N_EXAMPLES):
    if i < N_EXAMPLES // 2:
        series_list.append(generate_break_series(SERIES_LENGTH, "none", seed=i))
        labels.append(0)
    else:
        break_type = BREAK_TYPES[(i - N_EXAMPLES // 2) % len(BREAK_TYPES)]
        magnitude = magnitude_rng.uniform(0.3, 2.0)
        series_list.append(
            generate_break_series(SERIES_LENGTH, break_type, magnitude=magnitude, seed=i)
        )
        labels.append(1)

labels = np.array(labels)
print(f"{N_EXAMPLES} examples: {(labels == 0).sum()} without a break, {(labels == 1).sum()} with")

# %%
fig, axes = plt.subplots(2, 4, figsize=FIGSIZE["dashboard_2x3"], sharex="row")

for column, break_type in enumerate(["none", *BREAK_TYPES[:3]]):
    example = generate_break_series(SERIES_LENGTH, break_type, magnitude=1.0, seed=99)
    mid = SERIES_LENGTH // 2

    ax = axes[0, column]
    ax.plot(example, linewidth=0.5, color=COLORS["blue"])
    ax.axvline(mid, color=COLORS["negative"], linestyle="--", alpha=0.7)
    ax.set_title(break_type.replace("_", " "))
    if column == 0:
        ax.set_ylabel("Value")

    ax = axes[1, column]
    ax.hist(example[:mid], bins=30, alpha=0.55, density=True, color=COLORS["blue"], label="Before")
    ax.hist(example[mid:], bins=30, alpha=0.55, density=True, color=COLORS["amber"], label="After")
    if column == 0:
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

axes[0, 0].set_xlabel("Step")
axes[1, 0].set_xlabel("Value")
fig.suptitle("Three of the four constructed changes, and a boundary with none")
show_with_alt(
    fig,
    "A two by four grid. The top row draws one example series per column with a dashed "
    "line at the midpoint boundary: no change, a level that steps up, a spread that "
    "widens, and a slope that steepens. The bottom row overlays the histogram of the "
    "values before the boundary on the histogram after it, so the level shift appears as "
    "two offset humps and the variance shift as one hump inside a wider one.",
)

# %% [markdown]
# ## Family 1: where the two windows sit
#
# **Welch's t-test** compares two means without assuming the two samples have the same
# variance, which matters here because a break often moves both. It is run on the raw
# values, and again on absolute values at three window widths: taking absolute values
# converts a change in spread into a change in level, so the same test picks up a
# variance shift that the raw comparison would miss.
#
# The three widths are then combined by **Fisher's method**, which turns independent
# p-values into a single one by summing their logs. It is the standard way to ask whether
# a set of weak signals is jointly stronger than chance. The windows here overlap, so
# their p-values are not independent and the combined value is not a calibrated
# probability; it is used as a feature, where being monotone in the evidence is what is
# required of it.

# %%
FISHER_WINDOWS = [50, 100, 250]


def location_shift_features(series: np.ndarray, boundary: int) -> dict:
    """Compare where the two windows sit: Welch's t-test, raw and on absolute values."""
    features = {}
    before = series[:boundary]
    after = series[boundary:]

    t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)
    features["welch_t_pval"] = p_value
    features["welch_t_stat"] = abs(t_stat)

    for window in FISHER_WINDOWS:
        width = min(window, len(before), len(after))
        if width < 10:
            features[f"local_t_abs_{window}"] = 0.5
            continue
        _, p_value = stats.ttest_ind(
            np.abs(before[-width:]), np.abs(after[:width]), equal_var=False
        )
        features[f"local_t_abs_{window}"] = p_value

    # Fisher's method over the same three windows, floored so log(0) cannot arise.
    p_values = [max(features[f"local_t_abs_{w}"], 1e-300) for w in FISHER_WINDOWS]
    fisher_stat = -2 * sum(np.log(p) for p in p_values)
    features["fisher_location"] = stats.chi2.sf(fisher_stat, df=2 * len(p_values))

    pooled_std = np.sqrt((before.var() + after.var()) / 2)
    features["mean_diff_std"] = (
        abs(before.mean() - after.mean()) / pooled_std if pooled_std > 0 else 0.0
    )
    return features


# %% [markdown]
# ## Family 2: how spread out they are
#
# Three tests of the same question, differing in what they assume. The **F-test** on the
# ratio of variances is exact when both samples are normal and unreliable when they are
# not, which for financial returns they never are. **Levene's** test compares mean
# absolute deviations from the group centre, which stays accurate on non-normal samples.
# **Fligner-Killeen** ranks the deviations before comparing them, so the size of the
# largest observation cannot drive the answer at all. On financial data the last two are
# the ones to read; all three are kept because the classifier can learn where they
# disagree.


# %%
def scale_shift_features(series: np.ndarray, boundary: int) -> dict:
    """Compare how spread out the two windows are: variance ratio, Levene, Fligner."""
    features = {}
    before = series[:boundary]
    after = series[boundary:]

    var_before = max(before.var(ddof=1), 1e-10)
    var_after = max(after.var(ddof=1), 1e-10)
    features["var_ratio"] = var_after / var_before
    features["log_var_ratio"] = abs(np.log(features["var_ratio"]))

    # The F statistic puts the larger variance on top, so its degrees of freedom follow
    # whichever sample that is; using a fixed order returns the wrong tail probability
    # every time the second window is the quieter one.
    if var_after >= var_before:
        f_stat, df_numerator, df_denominator = (
            var_after / var_before,
            len(after) - 1,
            len(before) - 1,
        )
    else:
        f_stat, df_numerator, df_denominator = (
            var_before / var_after,
            len(before) - 1,
            len(after) - 1,
        )
    f_pval = 2 * stats.f.sf(f_stat, df_numerator, df_denominator)
    features["f_test_pval"] = f_pval
    features["neg_log10_f_pval"] = -np.log10(max(f_pval, 1e-300))

    features["fligner_pval"] = stats.fligner(before, after).pvalue
    features["levene_pval"] = stats.levene(before, after).pvalue
    return features


# %% [markdown]
# ## Family 3: what shape they are
#
# A break can leave the mean and the variance where they were and still change the
# distribution, so this family compares the two samples whole. The
# **Kolmogorov-Smirnov** statistic is the largest gap between their cumulative
# distributions. The **Jensen-Shannon divergence** and the **Hellinger distance** both
# measure how far apart two histograms are, the first through the information lost by
# treating one as the other and the second geometrically; both are bounded, which keeps
# them comparable across series. The **Wasserstein distance** is the cost of moving one
# distribution onto the other, so unlike the other three it grows with how far the
# probability has to travel and carries the units of the data.


# %%
def distribution_shift_features(series: np.ndarray, boundary: int, n_bins: int = 50) -> dict:
    """Compare the two windows as distributions: KS, Jensen-Shannon, Hellinger, Wasserstein."""
    features = {}
    before = series[:boundary]
    after = series[boundary:]

    ks = stats.ks_2samp(before, after)
    features["ks_stat"] = ks.statistic
    features["ks_pval"] = ks.pvalue

    # The three divergences below are defined on distributions, so both windows are binned
    # on one shared grid and each histogram is floored away from zero before normalising.
    edges = np.linspace(series.min() - 0.1, series.max() + 0.1, n_bins + 1)
    before_hist = np.histogram(before, bins=edges)[0] + 1e-10
    after_hist = np.histogram(after, bins=edges)[0] + 1e-10
    before_hist = before_hist / before_hist.sum()
    after_hist = after_hist / after_hist.sum()

    features["jsd"] = jensenshannon(before_hist, after_hist) ** 2
    features["hellinger"] = np.sqrt(1 - np.sum(np.sqrt(before_hist * after_hist)))
    features["wasserstein"] = stats.wasserstein_distance(before, after)
    return features


# %% [markdown]
# ## Family 4: how each observation depends on the last
#
# Two windows can agree on every distributional summary above and still differ in
# ordering, because a distribution says nothing about sequence. Lag-one autocorrelation
# captures the ordering of the values themselves; the same statistic on squared values
# captures whether large moves cluster, which is the dependence that changes when a
# volatility regime does.


# %%
def dependence_shift_features(series: np.ndarray, boundary: int) -> dict:
    """Compare lag-one dependence across the boundary, in the values and in their squares."""

    def lag_one(window: np.ndarray) -> float:
        if len(window) < 3 or window.std() == 0:
            return 0.0
        return float(np.corrcoef(window[:-1], window[1:])[0, 1])

    before = series[:boundary]
    after = series[boundary:]

    features = {
        "autocorr_pre": lag_one(before),
        "autocorr_post": lag_one(after),
        "sq_autocorr_diff": abs(lag_one(after**2) - lag_one(before**2)),
    }
    features["autocorr_diff"] = abs(features["autocorr_post"] - features["autocorr_pre"])
    return features


# %% [markdown]
# ## Family 5: does the data agree on where the boundary is
#
# The four families above take the boundary as given and ask what changed there. This one
# asks the boundary itself: run a change point detector that is free to pick any date,
# and measure how far its answer lands from the candidate. A break at the candidate date
# should attract both detectors to it; a series with no break puts them anywhere.


# %%
def alignment_features(series: np.ndarray, boundary: int) -> dict:
    """Measure how close a free change point estimate lands to the candidate boundary."""
    features = {}
    n = len(series)

    cusum = np.cumsum(series - series.mean())
    cusum_position = int(np.argmax(np.abs(cusum)))
    features["cusum_dist_to_boundary"] = abs(cusum_position - boundary) / n
    features["cusum_max"] = abs(cusum[cusum_position]) / (series.std() * np.sqrt(n))

    # Segmentation is an optimization and can return nothing on a short, near-constant
    # series; a distance of one is the largest the normalisation can produce.
    try:
        found = rpt.Pelt(model="l2", min_size=20).fit(series).predict(pen=np.log(n) * series.var())
        found = [position for position in found if position < n]
    except (ValueError, RuntimeError):
        found = []
    features["ruptures_dist_to_boundary"] = (
        min(abs(position - boundary) for position in found) / n if found else 1.0
    )
    return features


# %% [markdown]
# ## The feature matrix
#
# Each family is a function of the series and the boundary, so the matrix is the five of
# them applied to every example. The map from column to family is built from the same
# functions rather than typed out, which is what keeps the contribution chart below
# honest when a family gains or loses a column.

# %%
FEATURE_FAMILIES = {
    "location": location_shift_features,
    "scale": scale_shift_features,
    "distribution": distribution_shift_features,
    "dependence": dependence_shift_features,
    "alignment": alignment_features,
}


def compute_all_break_features(series: np.ndarray, boundary: int) -> dict:
    """Every family, for one series and one candidate boundary."""
    features = {}
    for family in FEATURE_FAMILIES.values():
        features.update(family(series, boundary))
    return features


X = (
    pd.DataFrame([compute_all_break_features(series, len(series) // 2) for series in series_list])
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)
y = labels

column_family = {
    column: name
    for name, family in FEATURE_FAMILIES.items()
    for column in family(series_list[0], SERIES_LENGTH // 2)
}

print(f"Feature matrix: {X.shape[0]} examples, {X.shape[1]} columns")
display(
    pd.Series(column_family)
    .groupby(lambda column: column_family[column])
    .apply(lambda group: ", ".join(group.index))
    .rename("columns")
    .to_frame()
)
display(X.head())

# %% [markdown]
# ## Classification
#
# A gradient-boosted tree ensemble combines the columns. It is the right shape of model
# for this: the families disagree with each other in ways that depend on the kind of
# break, which is an interaction, and trees represent interactions without being told
# where to look. Five-fold cross-validation gives the score, and the same folds supply
# out-of-fold probabilities for the ROC curve, so no example is scored by a model that
# saw it.

# %%
CV_FOLDS = 5

classifier = LGBMClassifier(n_estimators=100, max_depth=3, random_state=SEED, verbose=-1)
cv_auc = cross_val_score(classifier, X, y, cv=CV_FOLDS, scoring="roc_auc")
out_of_fold = cross_val_predict(classifier, X, y, cv=CV_FOLDS, method="predict_proba")[:, 1]

classifier.fit(X, y)
importances = pd.Series(classifier.feature_importances_, index=X.columns).sort_values()

print(
    f"{CV_FOLDS}-fold cross-validated AUC: {cv_auc.mean():.4f} (standard deviation {cv_auc.std():.4f})"
)

# %%
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"])

ax = axes[0]
top = importances.tail(15)
ax.barh(top.index, top.to_numpy(), color=COLORS["blue"])
ax.set_title("Which columns the trees split on")
ax.set_xlabel("Split count")
ax.tick_params(axis="y", labelsize=6)

ax = axes[1]
false_positive, true_positive, _ = roc_curve(y, out_of_fold)
ax.plot(false_positive, true_positive, linewidth=2, color=COLORS["blue"])
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.5, color=COLORS["neutral"])
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Out-of-fold separation")

fig.suptitle("Constructed breaks are easy; the score belongs to the construction")
show_with_alt(
    fig,
    "Two panels. Left: a horizontal bar chart of the fifteen most-split feature columns, "
    "led by the absolute-value t-test at the widest window and by the autocorrelation of "
    "the window after the boundary. Right: the out-of-fold ROC curve, which rises to the "
    "top left corner and runs flat along the top, far above the diagonal chance line.",
)

# %% [markdown]
# The score describes the examples it was measured on, and those were built to carry
# the changes the features were built to see. It says the wiring works. It says nothing
# about how the same detector behaves on a market series, which the last section tests.

# %% [markdown]
# ## Which family carries the signal
#
# Summing the split counts within each family says where the trees found their splits.
# Read it as a statement about these examples: the generator produces four kinds of break
# in equal proportion, so a family that sees only one of them can carry at most a quarter
# of the work however good it is at that quarter.

# %%
family_totals = importances.groupby(importances.index.map(column_family)).sum().sort_values()

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.barh(family_totals.index, family_totals.to_numpy(), color=COLORS["blue"])
add_message_title(
    ax,
    "Location leads the splits and every family gets used",
    subtitle="Total LightGBM split count per family, over the constructed examples",
)
ax.set_xlabel("Split count")
show_with_alt(
    fig,
    "A horizontal bar chart of five feature families ordered by total split count. "
    "Location is much the longest bar, followed by dependence and distribution, with "
    "scale and alignment short but not zero.",
)

# %% [markdown]
# ## The same detector on market data
#
# The test that matters is whether a detector trained on constructed breaks transfers.
# Each period below takes an equal number of sessions either side of a named date and
# scores that date as the candidate boundary. Three of the dates are episodes nobody
# disputes; three are quiet mid-year dates chosen for having no such episode near them.

# %%
PERIOD_SESSIONS = 250

PERIODS = [
    ("2008-09-15", "2008 credit crisis"),
    ("2020-03-11", "2020 shutdown"),
    ("2011-08-05", "2011 US downgrade"),
    ("2015-06-15", "mid-2015"),
    ("2017-06-15", "mid-2017"),
    ("2019-06-15", "mid-2019"),
]


def features_around(series: pd.Series, center_date: str, label: str) -> dict[str, Any] | None:
    """Break features for the candidate boundary at *center_date*, in session counts."""
    position = int(series.index.searchsorted(pd.Timestamp(center_date)))
    start = max(0, position - PERIOD_SESSIONS)
    stop = min(len(series), position + PERIOD_SESSIONS)
    window = series.iloc[start:stop].to_numpy()
    if len(window) < 100:
        return None

    features = compute_all_break_features(window, position - start)
    features["period"] = label
    features["sessions"] = len(window)
    return features


period_df = pd.DataFrame(
    [
        features
        for date, label in PERIODS
        if (features := features_around(returns, date, label)) is not None
    ]
)
period_df["break_probability"] = classifier.predict_proba(
    period_df[X.columns].replace([np.inf, -np.inf], np.nan).fillna(0)
)[:, 1]

display(
    period_df[
        ["period", "sessions", "ks_stat", "jsd", "wasserstein", "var_ratio", "break_probability"]
    ]
)

# %%
ordered = period_df.sort_values("break_probability")

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.barh(ordered["period"], ordered["break_probability"], color=COLORS["blue"], alpha=0.85)
ax.set_xlabel("Break probability")
ax.set_xlim(0, 1)
add_message_title(
    ax,
    "The detector calls almost every market window a break",
    subtitle=f"{PERIOD_SESSIONS} SPY sessions either side of each date, scored by the "
    "classifier fitted above",
)
show_with_alt(
    fig,
    "A horizontal bar chart of six SPY windows ordered by the probability the detector "
    "assigns them. All but one bar reach the right-hand edge at close to certainty, "
    "including two of the three quiet mid-year dates, and a single bar sits at the far "
    "left near zero.",
)

# %%
quiet = period_df[period_df["period"].str.startswith("mid-")]
flagged = int((quiet["break_probability"] > 0.5).sum())
print(f"Quiet windows scored above one half: {flagged} of {len(quiet)}")
print(
    "Widest distributional gap among the quiet windows: "
    f"{quiet['wasserstein'].max():.3f}, against "
    f"{period_df.loc[~period_df.index.isin(quiet.index), 'wasserstein'].max():.3f} "
    "among the crisis windows"
)

# %% [markdown]
# The detector is not miscoded, and the features are not useless: the distributional
# statistics in the table rank the periods sensibly, with the crisis windows separating
# further than the quiet ones on the same measures. What has failed is calibration, and
# the reason is in the training set. The examples were 500 constant-variance steps with a
# single change of a stated size at the midpoint. A 500-session window of a real index
# contains a dozen changes of comparable size before anything a person would call a
# crisis, so almost every real window looks, to this detector, like the positive class.
#
# The lesson generalises past this example. A supervised detector inherits the definition
# of its positive class from its labels, and synthetic labels define the class by
# construction rather than by consequence. Two ways out: label real history against dated
# events, which buys a definition someone can argue with, or drop the probability and feed
# the feature columns to the downstream model directly, which is what the rest of this
# chapter does with model-based features generally.

# %% [markdown]
# ## The same statistics as Polars expressions
#
# The functions above take NumPy arrays and one boundary, which is the right shape for
# scoring a candidate date. A production feature has the opposite shape: one long frame,
# one column per statistic, recomputed on a rolling window. `ml4t.engineer` supplies these
# as Polars expressions over the same frame the rest of the chapter uses.
#
# `coefficient_of_variation` is the standard deviation over the absolute mean, so it is
# only meaningful on a series whose mean stays away from zero. Daily returns average
# almost exactly zero, which makes the ratio explode; it is applied here to the absolute
# return, whose mean is safely positive, and then reads as the relative variability of
# move size.

# %%
ENGINEER_WINDOWS = {"short": 50, "long": 100}

engineered = (
    spy.with_columns(returns=pl.col("close").pct_change())
    .drop_nulls()
    .with_columns(
        cv_abs_return=coefficient_of_variation(
            pl.col("returns").abs(), window=ENGINEER_WINDOWS["short"]
        ),
        kl_divergence=rolling_kl_divergence("returns", window=ENGINEER_WINDOWS["long"]),
        wasserstein=rolling_wasserstein("returns", window=ENGINEER_WINDOWS["long"]),
        drift=rolling_drift("returns", window=ENGINEER_WINDOWS["long"]),
    )
)

display(
    engineered.select(["cv_abs_return", "kl_divergence", "wasserstein", "drift"])
    .drop_nulls()
    .describe()
    .to_pandas()
)

# %% [markdown]
# Each column compares a recent window against the one before it, so together they are a
# rolling version of the distribution family above and can be read on any session without
# refitting anything. That makes them usable next to the monitoring statistics as
# continuous inputs, which is the form the caveat two sections up recommends.

# %% [markdown]
# ## Key takeaways
#
# 1. **A single-break test and a segmentation answer different questions.**
#    Zivot-Andrews tests a hypothesis and estimates one date; a segmentation minimises a
#    cost and returns as many dates as the penalty or the requested count allows. Neither
#    reports the true number of breaks, because neither has a notion of one.
# 2. **A break detected on the full sample is not a feature.** Refit the detector forward
#    through the sample so that every value is computable from what preceded it, and
#    expect the causal column to be coarser and to revise itself as history accumulates.
# 3. **CUSUM and MOSUM trade memory against readability.** The cumulative statistic keeps
#    every deviation and therefore never resets; the moving one forgets outside its window
#    and stays readable for the next break. Both need a reference period that is fixed
#    rather than re-estimated, or they are not causal.
# 4. **Five families cover the ways two windows can differ**: where they sit, how spread
#    out they are, what shape they have, how each value depends on the last, and whether a
#    free detector agrees the boundary is where the change is. A detector that reads only
#    one family can only see the breaks that family is sensitive to.
# 5. **A learned detector inherits its definition of a break from its labels.** Trained on
#    constructed examples it separates constructed examples; on market windows the same
#    model returns a probability whose threshold means nothing, while the underlying
#    statistics still rank the windows sensibly.
#
# **Known limitations.** The classification section trains and evaluates on the same
# constructed distribution, so its cross-validated score bounds nothing about market data.
# The causal feature refits quarterly, which is a cost decision rather than a statistical
# one, and a break occurring just after a refit is invisible for up to a quarter. And the
# monitoring statistics are drawn without decision thresholds: the level at which a
# crossing should trigger action depends on how much a false alarm costs, which is a
# question this notebook does not have the information to answer.
#
# **Previous**: `01_visual_diagnostics` for stationarity testing.
# **Next**: `03_fractional_differencing` for reaching stationarity without discarding the
# level.
