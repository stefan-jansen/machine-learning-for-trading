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
# # Event Studies
#
# **Chapter 8: Feature Engineering**
# **Section Reference**: 8.6 - Combining Features and Controlling Search
# **Docker image**: `ml4t`
#
# ## Purpose
#
# Event studies measure abnormal returns around specific events (signal triggers,
# macro announcements, earnings) to assess their predictive power. This is a key
# validation technique for trading signals.
#
# ## Learning Objectives
#
# 1. Understand event study methodology (MacKinlay 1997)
# 2. Implement correct abnormal return computation
# 3. Calculate CAAR with proper confidence bands
# 4. Use event studies for signal validation
# 5. Recognize common pitfalls (clustering, overlapping windows)
#
# ## Key Concepts
#
# **Event Study Workflow**:
# 1. Define events (signal triggers, announcements)
# 2. Estimate "normal" returns in estimation window
# 3. Calculate abnormal returns in event window
# 4. Aggregate across events (CAAR)
# 5. Test statistical significance
#
# ## References
#
# - MacKinlay, A.C. (1997). "Event Studies in Economics and Finance"
# - Boehmer et al. (1991). Event-induced variance adjustments
#
# ## Data Policy
#
# All examples use **real ETF data**.

# %%
"""Event Studies: abnormal returns around signal triggers and macro announcements."""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import stats

from utils.reproducibility import set_global_seeds
from utils.style import (  # importing utils.style activates the ml4t Plotly template
    COLORS,
    show_plotly_with_alt,
)

# %% tags=["parameters"]
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
# The two-sided normal critical value the confidence bands, the bar shading and the
# significance verdicts all read. Declared once so a reader changing it changes all three.
Z_CRIT = 1.96
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Data Loading
#
# We use ETF data to demonstrate event studies. Events will be generated from
# momentum breakouts (trading signal) as a validation example.

# %%
from data import load_etfs

etfs = load_etfs()


# Select liquid ETFs for event study
SYMBOLS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# Filter
etf_filtered = (
    etfs.filter(pl.col("symbol").is_in(SYMBOLS))
    .filter(
        (pl.col("timestamp") >= datetime.strptime(START_DATE, "%Y-%m-%d"))
        & (pl.col("timestamp") < datetime.strptime(END_DATE, "%Y-%m-%d"))
    )
    .sort(["symbol", "timestamp"])
)

print(f"ETF data: {len(etf_filtered):,} rows")
print(f"Symbols: {etf_filtered['symbol'].n_unique()}")
print(f"Date range: {etf_filtered['timestamp'].min()} to {etf_filtered['timestamp'].max()}")

# %%
# Compute daily returns
returns_df = (
    etf_filtered.select(["timestamp", "symbol", "close"])
    .with_columns(pl.col("close").pct_change().over("symbol").alias("return"))
    .drop_nulls()
)

print(f"Returns: {len(returns_df):,} observations")

# Benchmark: SPY as market proxy
benchmark_returns = (
    returns_df.filter(pl.col("symbol") == "SPY")
    .select(["timestamp", "return"])
    .rename({"return": "benchmark_return"})
)

print(f"Benchmark: {len(benchmark_returns):,} days")

# %% [markdown]
# ## Generate Events
#
# For demonstration, we generate events from **momentum breakouts** (new 20-day highs).
# In practice, events could be:
# - Trading signal triggers
# - Earnings announcements
# - FOMC meetings
# - Index rebalances


# %%
def generate_momentum_breakout_events(
    prices: pl.DataFrame,
    lookback: int = 20,
    min_gap_days: int = 21,
) -> pl.DataFrame:
    """
    Generate events when price makes a new N-day high.

    Parameters
    ----------
    prices : DataFrame with timestamp, symbol, close
    lookback : Days to look back for high
    min_gap_days : Minimum gap between events (avoid clustering)

    Returns
    -------
    DataFrame with timestamp, symbol, event_type
    """
    events = []

    for symbol in prices["symbol"].unique().to_list():
        if symbol == "SPY":  # Skip benchmark
            continue

        symbol_data = prices.filter(pl.col("symbol") == symbol).sort("timestamp")
        close_prices = symbol_data["close"].to_numpy()
        timestamps = symbol_data["timestamp"].to_list()

        last_event_idx = -min_gap_days - 1

        for i in range(lookback, len(close_prices) - 30):  # Leave room for event window
            # Check if new high
            if close_prices[i] >= max(close_prices[i - lookback : i]):
                # Check minimum gap
                if i - last_event_idx >= min_gap_days:
                    events.append(
                        {
                            "timestamp": timestamps[i],
                            "symbol": symbol,
                            "event_type": "momentum_breakout",
                        }
                    )
                    last_event_idx = i

    return pl.DataFrame(events)


# %%
# Generate events
events_df = generate_momentum_breakout_events(
    etf_filtered.select(["timestamp", "symbol", "close"]),
    lookback=20,
    min_gap_days=30,  # At least 30 days between events per symbol
)

print(f"Generated {len(events_df)} events")
print("\nEvents by symbol:")
events_df.group_by("symbol").len().sort("symbol")

# %% [markdown]
# ## Event Study: Manual Implementation
#
# We align every symbol and the benchmark on a single shared date index (a
# wide-format returns table) so that event windows are located by integer
# offset from the event row - never by a label lookup that could match
# duplicate dates.
#
# The manual implementation shows the mechanics:
# 1. For each event, extract estimation and event windows
# 2. Estimate market model (CAPM) in estimation window
# 3. Calculate abnormal returns in event window
# 4. Aggregate across events


# %% [markdown]
# ### 3a. Market Model Estimation
#
# For each event, estimate the CAPM parameters ($\alpha$, $\beta$) from
# the pre-event estimation window. This establishes the "normal return"
# baseline.


# %%
def _estimate_market_model(
    returns_wide: pl.DataFrame,
    event_idx: int,
    symbol: str,
    estimation_window: tuple[int, int],
    min_estimation_obs: int,
) -> tuple[float, float] | None:
    """Estimate market model alpha and beta from estimation window."""
    est_start = event_idx + estimation_window[0]
    est_end = event_idx + estimation_window[1]

    if est_start < 0:
        return None

    est_slice = returns_wide.slice(est_start, est_end - est_start + 1)
    asset_est = est_slice[symbol].to_numpy()
    bench_est = est_slice["benchmark_return"].to_numpy()

    valid = np.isfinite(asset_est) & np.isfinite(bench_est)
    if np.sum(valid) < min_estimation_obs:
        return None

    try:
        slope, intercept, _, _, _ = stats.linregress(bench_est[valid], asset_est[valid])
        return intercept, slope  # alpha, beta
    except Exception:
        return None


# %% [markdown]
# ### 3b. Abnormal Return Computation
#
# Given estimated $\alpha$ and $\beta$, compute abnormal returns in the event
# window: $AR_t = R_{actual} - (\alpha + \beta \cdot R_{market})$.


# %%
def _compute_abnormal_returns(
    returns_wide: pl.DataFrame,
    event_idx: int,
    event_date,
    symbol: str,
    alpha: float,
    beta: float,
    event_window: tuple[int, int],
) -> tuple[list[dict], dict | None]:
    """Compute abnormal returns in the event window."""
    evt_start = event_idx + event_window[0]
    evt_end = event_idx + event_window[1]

    if evt_end >= len(returns_wide):
        return [], None

    evt_slice = returns_wide.slice(evt_start, evt_end - evt_start + 1)
    asset_evt = evt_slice[symbol].to_numpy()
    bench_evt = evt_slice["benchmark_return"].to_numpy()
    evt_dates = evt_slice["timestamp"].to_list()

    ars = []
    car = 0.0
    for i, (date, r_actual, r_market) in enumerate(
        zip(evt_dates, asset_evt, bench_evt, strict=False)
    ):
        if not (np.isfinite(r_actual) and np.isfinite(r_market)):
            continue
        r_expected = alpha + beta * r_market
        ar = r_actual - r_expected
        car += ar
        day_relative = event_window[0] + i
        ars.append(
            {
                "event_date": event_date,
                "symbol": symbol,
                "day": day_relative,
                "ar": ar,
                "car_to_day": car,
            }
        )

    event_car = {
        "event_date": event_date,
        "symbol": symbol,
        "car": car,
        "alpha": alpha,
        "beta": beta,
    }
    return ars, event_car


# %% [markdown]
# ### 3c. Aggregation: AAR and CAAR
#
# Average across events to get the Average Abnormal Return (AAR) per relative
# day, then cumulate to get the CAAR with correct standard errors:
#
# $$\text{SE}(\text{CAAR}_t) = \sqrt{\sum_{s=1}^{t} \frac{\sigma_s^2}{n_s}}$$


# %%
def _aggregate_to_caar(
    all_ars: list[dict],
    event_cars: list[dict],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Aggregate abnormal returns to AAR and CAAR."""
    ar_df = pl.DataFrame(all_ars) if all_ars else pl.DataFrame()
    car_df = pl.DataFrame(event_cars) if event_cars else pl.DataFrame()

    if len(ar_df) > 0:
        daily_aar = (
            ar_df.group_by("day")
            .agg(
                [
                    pl.col("ar").mean().alias("aar"),
                    pl.col("ar").std().alias("std"),
                    pl.len().alias("n"),
                ]
            )
            .sort("day")
            .with_columns((pl.col("std") / pl.col("n").sqrt()).alias("se"))
        )
        # CAAR and its standard error
        daily_aar = daily_aar.with_columns(pl.col("aar").cum_sum().alias("caar"))
        daily_aar = daily_aar.with_columns(
            (pl.col("std").pow(2) / pl.col("n")).cum_sum().sqrt().alias("caar_se")
        )
        daily_aar = daily_aar.with_columns(
            [
                (pl.col("aar") / pl.col("se")).alias("t_stat"),
                (pl.col("caar") / pl.col("caar_se")).alias("caar_t_stat"),
            ]
        )
    else:
        daily_aar = pl.DataFrame()

    return ar_df, car_df, daily_aar


# %% [markdown]
# ### 3d. Event Study Wrapper
#
# The wrapper orchestrates the three stages: estimate market model, compute
# abnormal returns, and aggregate to CAAR.


# %%
def compute_event_study(
    returns_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    events_df: pl.DataFrame,
    estimation_window: tuple[int, int] = (-60, -6),
    event_window: tuple[int, int] = (-5, 10),
    min_estimation_obs: int = 30,
) -> dict:
    """
    Compute event study using the three-stage pipeline above.

    Parameters
    ----------
    returns_df : Long-format returns (timestamp, symbol, return)
    benchmark_df : Benchmark returns (timestamp, benchmark_return)
    events_df : Events (timestamp, symbol)
    estimation_window : (start, end) days relative to event
    event_window : (start, end) days relative to event
    min_estimation_obs : Minimum observations in estimation window

    Returns
    -------
    Dict with:
        - abnormal_returns: DataFrame of AR by event-day
        - event_cars: DataFrame of CAR by event
        - daily_aar: DataFrame of AAR by relative day
    """
    # Create wide-format returns for efficient lookup
    returns_wide = returns_df.pivot(on="symbol", index="timestamp", values="return").sort(
        "timestamp"
    )
    returns_wide = returns_wide.join(benchmark_df, on="timestamp", how="inner")

    dates = returns_wide["timestamp"].to_list()
    date_to_idx = {d: i for i, d in enumerate(dates)}
    symbols = [c for c in returns_wide.columns if c not in ["timestamp", "benchmark_return"]]

    all_ars = []
    event_cars = []

    for row in events_df.iter_rows(named=True):
        event_date = row["timestamp"]
        symbol = row["symbol"]
        if symbol not in symbols or event_date not in date_to_idx:
            continue
        event_idx = date_to_idx[event_date]

        # Check event window upper bound
        if event_idx + event_window[1] >= len(dates):
            continue

        # Stage 1: Estimate market model
        model = _estimate_market_model(
            returns_wide, event_idx, symbol, estimation_window, min_estimation_obs
        )
        if model is None:
            continue
        alpha, beta = model

        # Stage 2: Compute abnormal returns
        ars, event_car = _compute_abnormal_returns(
            returns_wide, event_idx, event_date, symbol, alpha, beta, event_window
        )
        all_ars.extend(ars)
        if event_car:
            event_cars.append(event_car)

    # Stage 3: Aggregate
    ar_df, car_df, daily_aar = _aggregate_to_caar(all_ars, event_cars)

    return {
        "abnormal_returns": ar_df,
        "event_cars": car_df,
        "daily_aar": daily_aar,
        "n_events": len(car_df),
    }


# %%
# Run event study
result = compute_event_study(
    returns_df.select(["timestamp", "symbol", "return"]),
    benchmark_returns,
    events_df,
    estimation_window=(-60, -6),
    event_window=(-5, 10),
)

print(f"Processed {result['n_events']} events")

if len(result["event_cars"]) > 0:
    print("\nCAR Summary:")
    cars = result["event_cars"]["car"].to_numpy()
    print(f"  Mean CAR: {np.mean(cars) * 100:.2f}%")
    print(f"  Median CAR: {np.median(cars) * 100:.2f}%")
    print(f"  Std CAR: {np.std(cars) * 100:.2f}%")

# %% [markdown]
# ## Visualize CAAR with Confidence Bands
#
# The variance of the CAAR is the **cumulative sum** of the daily AAR variances,
# not a rolling calculation - abnormal returns accumulate day by day, so their
# variances add:
#
# $$\text{Var}(\text{CAAR}_t) = \sum_{s=1}^{t} \text{Var}(\text{AAR}_s)$$
#
# $$\text{SE}(\text{CAAR}_t) = \sqrt{\sum_{s=1}^{t} \frac{\sigma_s^2}{n_s}}$$

# %%
if len(result["daily_aar"]) > 0:
    daily_aar = result["daily_aar"]

    fig = go.Figure()

    days = daily_aar["day"].to_list()
    caar = daily_aar["caar"].to_numpy() * 100  # Convert to percent
    caar_se = daily_aar["caar_se"].to_numpy() * 100

    # 95% confidence band (Var(CAAR_t) = cumulative sum of daily AAR variances)
    upper = caar + Z_CRIT * caar_se
    lower = caar - Z_CRIT * caar_se

    fig.add_trace(
        go.Scatter(
            x=days + days[::-1],
            y=np.concatenate([upper, lower[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(10, 22, 40, 0.15)",  # COLORS["blue"] at 15% opacity
            line=dict(width=0),
            name="95% CI",
        )
    )

    # CAAR line drawn on top of the band
    fig.add_trace(
        go.Scatter(
            x=days,
            y=caar,
            mode="lines+markers",
            name="CAAR",
            line=dict(color=COLORS["blue"], width=2),
        )
    )

    # Event day marker and zero reference
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=COLORS["amber"],
        annotation_text="Event day",
        annotation_position="top left",
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])

    fig.update_layout(
        title="Cumulative average abnormal return around the event, with a 95% band",
        xaxis_title="Trading days relative to event",
        yaxis_title="Cumulative average abnormal return (%)",
        height=500,
    )

    show_plotly_with_alt(
        fig,
        (
            "A line chart of cumulative average abnormal return, in percent, against trading "
            "days relative to the event, running from five days before to ten days after. A "
            "grey band marks the ninety-five percent confidence interval and an amber "
            "vertical line labelled Event day marks day zero. The line starts at zero five "
            "days before, climbs steadily through the pre-event days, and rises most "
            "steeply between the day before and the event day itself, where it reaches its "
            "highest level. After the event it is flat to slightly declining for the rest "
            "of the window, ending a little below its peak. The confidence band is narrow "
            "before the event and widens steadily after it, with its lower edge "
            "coming closer to zero at the right without reaching it."
        ),
    )

# %% [markdown]
# Two features of this curve are worth separating before reading it as a result.
#
# The pre-event rise is not evidence of anything predictive. The event is *defined* by a
# momentum breakout, so the days leading up to it are days on which the price rose by
# construction; a cumulative abnormal return that climbs from day minus five to day zero
# is that definition showing up in the chart. The quantity to read is what happens after
# day zero, because only that part was unknown at the moment the signal fired.
#
# What happens after day zero is a flat to gently declining line inside a band that widens
# with every additional day. Widening is mechanical, since each day adds variance to a
# cumulative sum, and it is the reason a CAAR window has to be fixed in advance: extending
# it until the band excludes zero is a search, and the band was chosen to make that easy.
#
# ### Daily abnormal returns
#
# Decomposing the CAAR into its per-day contributions shows *where* the abnormal
# return is earned. Bars whose t-statistic clears `Z_CRIT` are drawn in the primary
# colour, the rest are muted, and the event day is highlighted whatever its significance.

# %%
if len(result["daily_aar"]) > 0:
    daily_aar = result["daily_aar"]

    days = daily_aar["day"].to_list()
    aar_pct = (daily_aar["aar"] * 100).to_list()
    t_stats = daily_aar["t_stat"].to_list()

    # Color by significance; highlight the event day
    bar_colors = [
        COLORS["amber"]
        if d == 0
        else (COLORS["blue"] if abs(t) > Z_CRIT else COLORS["silver_muted"])
        for d, t in zip(days, t_stats, strict=True)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=days, y=aar_pct, marker_color=bar_colors, name="AAR"))
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["amber"])

    fig.update_layout(
        title="Average abnormal return by day, shaded by significance",
        xaxis_title="Trading days relative to event",
        yaxis_title="Average abnormal return (%)",
        height=400,
    )

    show_plotly_with_alt(
        fig,
        (
            "A bar chart of average abnormal return in percent by trading day relative to the "
            "event, from five days before to ten days after, against a dashed zero line. "
            "Bars are shaded by significance: those whose t-statistic clears the critical "
            "value are dark, the rest are pale grey, and the event-day bar is amber. The "
            "amber bar at day zero is by far the tallest, several times the height of any "
            "other. Two pre-event bars, three days and one day before, are dark and "
            "positive, so they clear the threshold too. The remaining bars are small and "
            "pale, scattered on both sides of zero, with the post-event days slightly more "
            "often negative than positive."
        ),
    )

# %% [markdown]
# ## 4b. Library Alternative: EventStudyAnalysis
#
# The manual implementation above teaches the MacKinlay (1997) mechanics.
# The `ml4t-diagnostic` library adds robust variance adjustment (BMP test,
# Boehmer et al. 1991) and non-parametric testing (Corrado rank test).

# %%
from ml4t.diagnostic.config import EventConfig
from ml4t.diagnostic.config.event_config import WindowSettings
from ml4t.diagnostic.evaluation import EventStudyAnalysis

# Prepare data in library format
# Returns: date, asset, return
lib_returns = returns_df.select(
    pl.col("timestamp").alias("date"),
    pl.col("symbol").alias("asset"),
    pl.col("return"),
)

# Benchmark: date, return
lib_benchmark = benchmark_returns.rename({"timestamp": "date", "benchmark_return": "return"})

# Events: date, asset
lib_events = events_df.select(
    pl.col("timestamp").alias("date"),
    pl.col("symbol").alias("asset"),
)

# Configure event study
config = EventConfig(
    window=WindowSettings(
        estimation_start=-60,
        estimation_end=-6,
        event_start=-5,
        event_end=10,
    ),
    model="market_model",
    min_estimation_obs=30,
)

# Run library event study
lib_analysis = EventStudyAnalysis(
    returns=lib_returns,
    events=lib_events,
    benchmark=lib_benchmark,
    config=config,
)
# The library warns when it drops events it cannot fit. That is a fact about this
# analysis rather than a problem with it, so catch the warnings and report them as part of
# the output; a reader needs to know how many events the numbers below actually rest on.
with warnings.catch_warnings(record=True) as _caught:
    warnings.simplefilter("always")
    lib_result = lib_analysis.run()

_skips = [str(w.message) for w in _caught if "Skipped" in str(w.message)]
print(f"events supplied: {len(lib_events)}")
for _msg in _skips:
    print(f"  {_msg}")
if not _skips:
    print("  none skipped")

# %%
# Compare results
print("=== Library EventStudyAnalysis Results ===\n")
print(lib_result.summary())

# The library defaults to the Boehmer et al. (1991) BMP test, which is robust to
# event-induced variance - the manual t-test above assumes constant variance.
print("\n=== Robust significance test (library) ===")
print(f"Test: {lib_result.test_name}")
print(f"Test statistic: {lib_result.test_statistic:.2f}")
print(f"P-value: {lib_result.p_value:.4f}")
print(f"Significant at 5%: {'Yes' if lib_result.p_value < 0.05 else 'No'}")

# %% [markdown]
# The manual implementation teaches the market model ($R_i = \alpha + \beta R_m$)
# and CAAR computation. The library adds:
#
# | Feature | Manual | Library |
# |---------|--------|---------|
# | Market model | Yes | Yes |
# | Mean-adjusted model | No | Yes |
# | BMP test (robust variance) | No | Yes |
# | Corrado rank test | No | Yes |
# | Event clustering handling | No | Yes |

# %% [markdown]
# ## CAR Distribution
#
# Examining the distribution of individual event CARs reveals whether the
# aggregate effect is driven by many small effects or few large ones.

# %%
if len(result["event_cars"]) > 0:
    cars = result["event_cars"]["car"].to_numpy() * 100

    # The paragraph under this figure turns on a mean-median comparison and on whether the
    # distribution is skewed, so compute both rather than inviting the reader to eyeball
    # them off a histogram.
    print(f"cumulative abnormal return over the event window, across {len(cars)} events:")
    print(f"  mean   {cars.mean():+.2f}%")
    print(f"  median {np.median(cars):+.2f}%")
    print(f"  skewness {float(stats.skew(cars)):+.2f}")
    print(f"  share positive {float((cars > 0).mean()):.1%}")

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=cars,
            nbinsx=25,
            marker_color=COLORS["blue"],
            name="CAR distribution",
        )
    )

    # Zero reference and mean (mean highlighted in amber)
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color=COLORS["neutral"],
        annotation_text="Zero",
        annotation_position="top left",
    )
    fig.add_vline(
        x=float(np.mean(cars)),
        line_dash="dash",
        line_color=COLORS["amber"],
        annotation_text=f"Mean: {np.mean(cars):.2f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Cumulative abnormal return per event, with zero and mean marked",
        xaxis_title="Cumulative abnormal return over event window (%)",
        yaxis_title="Number of events",
        height=400,
    )

    show_plotly_with_alt(
        fig,
        (
            "A histogram of cumulative abnormal return per event, in percent, with the count "
            "of events on the vertical axis. The bulk of the distribution sits between "
            "about minus five and plus ten percent, with a tall bar just above zero and a "
            "roughly symmetric fall-off on both sides; a few isolated events sit far out "
            "in each tail. Two vertical dashed lines cross the distribution near its "
            "centre, one labelled Zero and one labelled with the mean, the mean line "
            "sitting a little to the right of zero. The gap between them is small relative "
            "to the width of the distribution."
        ),
    )

    # Statistical test: Mean CAR = 0
    t_stat, p_value = stats.ttest_1samp(cars, 0)

    print("\nStatistical Test (H0: Mean CAR = 0):")
    print(f"  Mean CAR: {np.mean(cars):.2f}%")
    print(f"  Median CAR: {np.median(cars):.2f}%")
    print(f"  T-statistic: {t_stat:.2f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")

# %% [markdown]
# **Interpretation**: read the four numbers printed above the figure together. The
# question they answer is whether the aggregate effect is a property of the typical event
# or the work of a few large ones, and the mean alone cannot tell you.
#
# A mean and a median close to each other, with a skewness near zero and a share of
# positive events meaningfully above half, describe an effect spread across the sample:
# the aggregate CAAR then says something about what to expect from the next breakout. A
# mean well above a median near zero would say the opposite, that a handful of events
# carry the average, and an aggregate built that way is not something to size a position
# against however significant its t-statistic looks.
#
# The histogram shows the same thing in shape rather than in numbers, and the two dashed
# lines are there so the gap between zero and the mean can be read against the width of
# the distribution, which is the comparison that matters.

# %% [markdown]
# ## Event Study Heatmap
#
# Visualize abnormal returns across events and days to identify patterns.

# %%
if len(result["abnormal_returns"]) > 0:
    ar_df = result["abnormal_returns"]

    # Create unique event identifier (date + symbol can have multiple events)
    ar_df = ar_df.with_columns(
        (pl.col("event_date").dt.strftime("%Y-%m-%d") + "_" + pl.col("symbol")).alias("event_id")
    )

    # Pivot to wide format (event x day)
    ar_pivot = ar_df.pivot(on="day", index="event_id", values="ar").sort("event_id")

    # Convert to numpy for heatmap
    day_cols = sorted([c for c in ar_pivot.columns if c != "event_id"], key=lambda x: int(x))
    ar_matrix = ar_pivot.select(day_cols).to_numpy() * 100

    # Limit to 30 events for readability
    if len(ar_matrix) > 30:
        ar_matrix = ar_matrix[:30]
        event_ids = ar_pivot["event_id"].to_list()[:30]
    else:
        event_ids = ar_pivot["event_id"].to_list()

    # ML4T diverging scale centered at zero (red = negative, green = positive AR)
    diverging_scale = [
        [0.0, COLORS["negative"]],
        [0.5, COLORS["silver_muted"]],
        [1.0, COLORS["positive"]],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=ar_matrix,
            x=day_cols,
            y=event_ids,
            colorscale=diverging_scale,
            zmid=0,
            colorbar=dict(title="AR (%)"),
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["amber"], line_width=2)

    fig.update_layout(
        title="Abnormal return by event and day, one row per event",
        xaxis_title="Trading days relative to event",
        yaxis_title="Event (date + symbol)",
        height=600,
    )

    show_plotly_with_alt(
        fig,
        (
            "A heatmap with one row per event, labelled by date and ticker down the left "
            "edge, and trading days relative to the event across the bottom from five "
            "before to ten after. Cells are shaded by abnormal return on a diverging "
            "scale, green for positive and red for negative, with a colour bar at the "
            "right. A dashed amber vertical line marks the event day. Most of the grid is "
            "pale, including the event-day column, where only one or two rows show a "
            "strong green; the largest individual cells, in both directions, are scattered "
            "through the pre- and post-event days rather than concentrated at day zero."
        ),
    )

# %% [markdown]
# ## Caveats and Best Practices
#
# ### Event Clustering
#
# When multiple events occur on the same day (e.g., sector-wide announcements),
# the cross-sectional correlation inflates the t-statistics. Solutions:
# - Use portfolio-level returns
# - Adjust standard errors for clustering
# - Aggregate to one "event" per day
#
# ### Overlapping Windows
#
# If events are close together, estimation and event windows may overlap,
# contaminating the "normal return" estimate. Solutions:
# - Enforce minimum gap between events (we used 30 days)
# - Use shorter estimation windows
# - Use calendar-time portfolio approach
#
# ### Confounding Events
#
# Other events in the window (earnings, macro news) can confound results.
# Solutions:
# - Screen for confounding events
# - Use matched controls
# - Analyze subsamples

# %% [markdown]
# ## Using Event Studies for Signal Validation
#
# Event studies validate trading signals by testing whether signal-generated
# "events" produce abnormal returns.


# %%
def validate_signal_with_event_study(
    returns_df: pl.DataFrame,
    benchmark_df: pl.DataFrame,
    signal_df: pl.DataFrame,
    signal_column: str = "signal",
    threshold: float = 2.0,
    event_window: tuple[int, int] = (-5, 10),
) -> dict:
    """
    Validate a trading signal using event study methodology.

    Parameters
    ----------
    returns_df : Long-format returns
    benchmark_df : Benchmark returns
    signal_df : Signal values (timestamp, symbol, signal)
    signal_column : Column name for signal
    threshold : Z-score threshold for event trigger
    event_window : Days around event to analyze

    Returns
    -------
    Dict with long and short event study results
    """
    # Z-score signals cross-sectionally
    signal_zscored = signal_df.with_columns(
        (
            (pl.col(signal_column) - pl.col(signal_column).mean().over("timestamp"))
            / pl.col(signal_column).std().over("timestamp")
        ).alias("zscore")
    )

    # Generate events from extreme signals
    long_events = (
        signal_zscored.filter(pl.col("zscore") > threshold)
        .select(["timestamp", "symbol"])
        .with_columns(pl.lit("long_signal").alias("event_type"))
    )

    short_events = (
        signal_zscored.filter(pl.col("zscore") < -threshold)
        .select(["timestamp", "symbol"])
        .with_columns(pl.lit("short_signal").alias("event_type"))
    )

    print(f"Long signal events: {len(long_events)}")
    print(f"Short signal events: {len(short_events)}")

    results = {}

    if len(long_events) > 10:
        results["long"] = compute_event_study(
            returns_df, benchmark_df, long_events, event_window=event_window
        )

    if len(short_events) > 10:
        results["short"] = compute_event_study(
            returns_df, benchmark_df, short_events, event_window=event_window
        )

    return results


# %% [markdown]
# A worked example: validating a momentum signal by treating each of its triggers as an
# event and asking whether abnormal returns follow.

# %%
prices_wide = (
    etf_filtered.select(["timestamp", "symbol", "close"])
    .pivot(on="symbol", index="timestamp", values="close")
    .sort("timestamp")
)

symbols = [c for c in prices_wide.columns if c != "timestamp"]

# 21-day momentum
momentum = prices_wide.select(
    pl.col("timestamp"), *[(pl.col(s) / pl.col(s).shift(21) - 1).alias(s) for s in symbols]
)

# Melt to long format
momentum_long = (
    momentum.unpivot(index="timestamp", variable_name="symbol", value_name="momentum")
    .drop_nulls()
    .filter(pl.col("momentum").is_finite())
)

print(f"Momentum signal: {len(momentum_long):,} observations")

# %%
# Validate momentum signal
validation = validate_signal_with_event_study(
    returns_df.select(["timestamp", "symbol", "return"]),
    benchmark_returns,
    momentum_long,
    signal_column="momentum",
    threshold=1.5,
)

if "long" in validation and len(validation["long"]["daily_aar"]) > 0:
    print("\nLong Signal Validation:")
    aar_long = validation["long"]["daily_aar"]
    final_caar = aar_long["caar"].to_numpy()[-1] * 100
    final_t = aar_long["caar_t_stat"].to_numpy()[-1]
    print(f"  Final CAAR: {final_caar:.2f}%")
    print(f"  CAAR t-stat: {final_t:.2f}")
    print(f"  Significant: {'Yes' if abs(final_t) > Z_CRIT else 'No'}")

if "short" in validation and len(validation["short"]["daily_aar"]) > 0:
    print("\nShort Signal Validation:")
    aar_short = validation["short"]["daily_aar"]
    final_caar = aar_short["caar"].to_numpy()[-1] * 100
    final_t = aar_short["caar_t_stat"].to_numpy()[-1]
    print(f"  Final CAAR: {final_caar:.2f}%")
    print(f"  CAAR t-stat: {final_t:.2f}")
    print(f"  Significant: {'Yes' if abs(final_t) > Z_CRIT else 'No'}")

# %% [markdown]
# ## Summary
#
# ### Methodology
#
# - **Estimation window**: 60 days before event (excluding 5-day gap)
# - **Event window**: 5 days before to 10 days after
# - **Model**: Market model ($R_i = \alpha + \beta \cdot R_{market}$)
#
# ### Key Formulas
#
# | Metric | Formula |
# |--------|---------|
# | Abnormal Return | $AR = R_{actual} - (\alpha + \beta \cdot R_{market})$ |
# | CAR | Cumulative sum of AR over event window |
# | CAAR | Average CAR across events |
# | CAAR SE | $\sqrt{\sum_{s=1}^{t} \sigma_s^2 / n_s}$ |
#
# ### Interpretation Guide
#
# | Pattern | Meaning | Trading Implication |
# |---------|---------|---------------------|
# | Pre-event drift | Information leakage | Limited post-event alpha |
# | Event-day jump | Clean announcement | Event timing matters |
# | Post-event drift | Underreaction | Post-event momentum |
# | Reversal | Overreaction | Mean-reversion |
#
# ### Caveats
#
# - Event clustering inflates t-stats
# - Overlapping windows contaminate estimates
# - Confounding events require screening

# %% [markdown]
# ## Key Takeaways
#
# 1. **Alignment matters**: locate event windows by integer offset on a single
#    shared date index, so duplicate or missing dates cannot misalign the windows.
#
# 2. **CAAR variance is cumulative**: $\text{SE}(\text{CAAR}_t) = \sqrt{\sum_s
#    \sigma_s^2 / n_s}$ - the daily variances add, they are not propagated by a
#    rolling formula.
#
# 3. **Event studies validate signals**: Signal-triggered "events" should produce
#    significant abnormal returns if the signal has predictive power.
#
# 4. **Watch for clustering**: Events on the same day violate independence
#    assumptions underlying the t-tests.
#
# 5. **Minimum gap prevents overlap**: Enforce at least 20-30 day gaps between
#    events per symbol to keep estimation windows clean.
#
# ### Next Notebook
#
# - `case_study_feature_summary`: cross-case-study feature inventory
