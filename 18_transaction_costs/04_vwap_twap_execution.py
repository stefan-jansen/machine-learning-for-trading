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
# # VWAP and TWAP Execution Algorithms
#
# **Docker image**: `ml4t`
#
# An order too large to send at once is broken into **child orders** and worked across the session.
# The rule deciding how much to send in each interval is an execution **schedule**, and the two
# schedules everything else is measured against are:
#
# - **TWAP**, the time-weighted average price, which sends the same number of shares every
#   interval. It needs no forecast of anything.
# - **VWAP**, the volume-weighted average price, which sends shares in proportion to how much the
#   market is expected to trade in each interval. It needs a forecast of the day's volume shape.
#
# Each is also a **benchmark**: a desk's execution is scored by how close its average fill price
# came to the day's actual volume-weighted average price. This notebook builds both schedules,
# estimates the volume forecast VWAP needs from real minute bars, runs both against real NASDAQ-100
# sessions the forecast was not estimated on, and compares how tightly each tracks that benchmark.
#
# **Learning Objectives**
# - Turn an order size and a session into a TWAP schedule, and a volume forecast into a VWAP one
# - Estimate an intraday volume profile from minute bars on one window and apply it on another, so
#   the forecast is tested rather than fitted
# - Simulate what a schedule would have paid on a real session, with impact charged against the
#   volume actually available in each interval
# - Compare two schedules by the distribution of their benchmark slippage across many sessions
#   rather than by one session's outcome, and separate a schedule's bias from its dispersion
#
# **Book Reference:** Chapter 18, Section 18.5
#
# **Prerequisites:** Read [`03_market_impact_calibration`](03_market_impact_calibration.ipynb) for impact assumptions
# and [`05_almgren_chriss_optimal_execution`](05_almgren_chriss_optimal_execution.ipynb) for the transition from control
# algorithms to explicit cost-risk optimization.

# %% [markdown]
# ## Why TWAP and VWAP?
#
# Both exist to solve the problem the previous notebook measured: impact grows with the share of an
# interval's volume an order takes, so the same order costs less spread over a session than sent at
# once. Neither tries to predict where the price is going. A schedule is fixed before trading
# starts, which is what makes it auditable - a desk can be held to it - and what makes it a
# benchmark other execution can be scored against.
#
# The difference between them is what they assume:
#
# | Schedule | Assumes | Fails when |
# |---|---|---|
# | TWAP | Every interval is as good as any other | Liquidity varies through the day, so equal slices take a larger share of the quiet intervals |
# | VWAP | Today's volume shape resembles the recent average | The day is unusual - news, an index event, an unexpected halt |

# %% [markdown]
# ## Imports & Settings

# %%
"""VWAP and TWAP Execution - Algorithm implementation and evaluation on real NASDAQ-100 sessions."""

from datetime import date, datetime, time, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from data import load_nasdaq100_bars
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# %% tags=["parameters"]
EXEC_SYMBOL = "AAPL"
TAQ_START_DATE = "2021-10-01"
TAQ_END_DATE = "2021-12-31"
PROFILE_END_DATE = "2021-11-15"
INTERVAL_MINUTES = 15
ORDER_SHARES = 100_000
IMPACT_BPS = 5.0
SEED = 42

# %%
set_global_seeds(SEED)

SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE = 9, 30
SESSION_CLOSE_HOUR = 16

# %% [markdown]
# What each setting decides:
#
# - `EXEC_SYMBOL` is the stock both schedules trade. One liquid name keeps the volume profile and
#   the session-by-session comparison readable; the code takes a list of symbols.
# - `TAQ_START_DATE` and `TAQ_END_DATE` bound the minute bars read, and `PROFILE_END_DATE` splits
#   them. Sessions on or before it estimate the volume profile; sessions after it are the ones both
#   schedules are run on. The split is a fixed calendar date rather than a fraction of the sample,
#   so extending the data cannot move the estimation window under a result.
# - `INTERVAL_MINUTES` is how often a child order is sent. Fifteen minutes divides the 09:30-16:00
#   session into 26 intervals. A finer grid tracks the volume shape more closely and sends smaller,
#   more numerous orders; a coarser one is easier to supervise and matches the profile less well.
# - `ORDER_SHARES` is the size of the parent order. It matters only through its ratio to the
#   volume available, which is what the impact model charges against.
# - `IMPACT_BPS` scales the square-root impact charged in the simulation: it is the cost in basis
#   points of a slice equal to an interval's entire volume. It is a stated figure, not one
#   calibrated from executions, and it moves both schedules' costs together.

# %% [markdown]
# ## Part 1: TWAP Algorithm
#
# **TWAP** divides the order equally across time intervals:
#
# $$\text{Trade Size}_t = \frac{\text{Total Order}}{\text{Number of Intervals}}$$
#
# The only inputs are the order size and the number of intervals, so nothing about the schedule can
# be wrong in the way a forecast can be wrong. What it gives up is that an interval carrying two
# percent of the day's volume receives the same number of shares as one carrying eight, and by the
# square-root model the first of those slices costs twice as much per share.


# %%
def build_time_grid(
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
) -> list[datetime]:
    """Create the execution clock used by both TWAP and VWAP.

    Each returned timestamp is a *decision time* - the instant a slice is
    scheduled - not the boundary of a closed time interval. Slice volume is
    attributed to the interval beginning at that decision time.
    """
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if end_time <= start_time:
        raise ValueError("end_time must be later than start_time")
    current = start_time
    times = []
    while current < end_time:
        times.append(current)
        current += timedelta(minutes=interval_minutes)
    return times


# %% [markdown]
# ### Build the TWAP Share Schedule


# %%
def build_twap_schedule(
    total_shares: int,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
) -> pl.DataFrame:
    """Allocate shares evenly across the available execution intervals."""
    if total_shares <= 0:
        raise ValueError("total_shares must be positive")
    times = build_time_grid(start_time, end_time, interval_minutes)
    n_intervals = len(times)
    shares_per_interval = total_shares // n_intervals
    remainder = total_shares % n_intervals

    schedule = []
    cumulative = 0
    for i, ts in enumerate(times):
        shares = shares_per_interval + (1 if i < remainder else 0)
        cumulative += shares
        schedule.append(
            {
                "timestamp": ts,
                "shares": shares,
                "cumulative": cumulative,
                "pct_complete": cumulative / total_shares,
            }
        )

    return pl.DataFrame(schedule)


# %% [markdown]
# ### Target Lookup Helper


# %%
def latest_cumulative_target(schedule: pl.DataFrame, current_time: datetime) -> int:
    """Return the most recent cumulative target in the schedule."""
    mask = schedule["timestamp"].to_numpy() <= current_time
    if not mask.any():
        return 0
    latest_index = int(np.flatnonzero(mask)[-1])
    return int(schedule["cumulative"].item(latest_index))


# %% [markdown]
# ### TWAP Summary Helper


# %%
def summarize_twap(algo) -> dict:
    """Create a compact TWAP schedule summary."""
    return {
        "algorithm": "TWAP",
        "total_shares": algo.total_shares,
        "n_intervals": len(algo.schedule),
        "interval_minutes": algo.interval_minutes,
        "shares_per_interval": algo.total_shares // len(algo.schedule),
    }


# %% [markdown]
# ### TWAP Class Definition


# %%
class TWAPAlgorithm:
    """Time-Weighted Average Price execution algorithm."""

    def __init__(
        self, total_shares: int, start_time: datetime, end_time: datetime, interval_minutes: int = 5
    ):
        self.total_shares = total_shares
        self.start_time = start_time
        self.end_time = end_time
        self.interval_minutes = interval_minutes
        self.schedule = build_twap_schedule(
            total_shares=self.total_shares,
            start_time=self.start_time,
            end_time=self.end_time,
            interval_minutes=self.interval_minutes,
        )

    def get_target_at_time(self, current_time: datetime) -> int:
        return latest_cumulative_target(self.schedule, current_time)

    def summary(self) -> dict:
        return summarize_twap(self)


# %% [markdown]
# ### The Session Clock
#
# Every schedule and every chart below runs on the same 09:30-16:00 grid. The helper builds it for
# a named session, so an axis showing a day's execution carries that day's date rather than a
# placeholder.


# %%
def session_grid(day: date) -> tuple[datetime, datetime]:
    """Return the regular-session open and close for one date."""
    return (
        datetime.combine(day, time(SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE)),
        datetime.combine(day, time(SESSION_CLOSE_HOUR, 0)),
    )


# %%
start, end = session_grid(date.fromisoformat(TAQ_START_DATE))

twap = TWAPAlgorithm(
    total_shares=100_000,
    start_time=start,
    end_time=end,
    interval_minutes=15,
)

print("TWAP Schedule Summary:")
for k, v in twap.summary().items():
    print(f"  {k}: {v}")

if twap.schedule["shares"].sum() != twap.total_shares:
    raise ValueError("The TWAP schedule must allocate exactly the parent order")

# %% [markdown]
# **Reading the summary**: The whole schedule is described by four numbers, none of which came
# from the market. That is the point of TWAP: there is nothing in it that can be estimated wrong.

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Shares sent per interval", "Order completed"],
    vertical_spacing=0.12,
)
times = twap.schedule["timestamp"].to_list()

_ = fig.add_bar(
    x=times,
    y=twap.schedule["shares"].to_list(),
    marker_color=COLORS["blue"],
    row=1,
    col=1,
)
_ = fig.add_scatter(
    x=times,
    y=twap.schedule["pct_complete"].to_list(),
    mode="lines",
    line=dict(color=COLORS["amber"], width=2),
    row=2,
    col=1,
)
fig.update_yaxes(title_text=f"Shares per {twap.interval_minutes}-minute interval", row=1, col=1)
fig.update_yaxes(title_text="Order completed", tickformat=".0%", range=[0, 1], row=2, col=1)
fig.update_xaxes(title_text="Execution time (ET)", tickformat="%H:%M", row=2, col=1)
fig.update_layout(
    title="TWAP sends the same number of shares every interval",
    height=520,
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "Upper panel: bars of equal height across every interval of the session. Lower panel: the "
    "resulting completion curve, a straight line from zero to fully filled.",
)

# %% [markdown]
# **Finding**: Equal bars in the upper panel become a straight line in the lower one. The schedule
# is fully determined before the session starts and takes no input from the market, which is what
# makes it robust to a bad volume forecast and what guarantees it trades through the quiet part of
# the day at the same rate as the busy part.

# %% [markdown]
# ## Part 2: VWAP Algorithm
#
# **VWAP** matches the market's volume profile:
#
# $$\text{Trade Size}_t = \text{Total Order} \times \frac{\text{Expected Volume}_t}{\text{Total Expected Volume}}$$
#
# Sending shares in proportion to expected volume holds the participation rate roughly constant
# through the day, which is what the square-root model says minimizes total impact for a fixed
# order worked over a fixed horizon. It also tracks the VWAP benchmark by construction, since the
# benchmark is itself a volume-weighted average.
#
# The cost is that the schedule is only as good as the forecast. And because the shape is
# predictable, a large VWAP order is something other participants can anticipate and trade ahead
# of - the schedule's predictability is a liability as well as a control.

# %% [markdown]
# ### Intraday Volume Patterns from Real Minute Bars
#
# US equity markets exhibit a consistent **U-shaped** volume pattern - high at the
# open, thin at midday, high into the close. Rather than assume that shape, we
# measure it directly from AlgoSeek NASDAQ-100 minute bars and use the measured
# profile as the VWAP volume forecast. We restrict to the regular session
# (09:30-16:00) and aggregate minute bars onto the execution grid.


# %%
def load_intraday_panel(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval_minutes: int,
) -> pl.DataFrame:
    """Aggregate minute bars by symbol, session, and execution-grid bucket."""
    session_start = SESSION_OPEN_HOUR * 60 + SESSION_OPEN_MINUTE
    session_end = SESSION_CLOSE_HOUR * 60
    panel = (
        load_nasdaq100_bars(
            start_date=start_date,
            end_date=end_date,
            include_microstructure=True,
            lazy=True,
        )
        .filter(pl.col("symbol").is_in(symbols))
        .select("timestamp", "symbol", "volume", "last_trade_price")
        .filter(pl.col("last_trade_price").is_not_null() & (pl.col("volume") > 0))
        .with_columns(
            session_date=pl.col("timestamp").dt.date(),
            minute_of_day=pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int32),
        )
        .filter(
            (pl.col("minute_of_day") >= session_start) & (pl.col("minute_of_day") < session_end)
        )
        .with_columns(
            bucket=((pl.col("minute_of_day") - session_start) // interval_minutes).cast(pl.Int32)
        )
        .group_by("symbol", "session_date", "bucket")
        .agg(
            volume=pl.col("volume").sum(),
            price=(pl.col("last_trade_price") * pl.col("volume")).sum() / pl.col("volume").sum(),
        )
        .sort("symbol", "session_date", "bucket")
        .collect()
    )
    return panel


# %%
N_BUCKETS = (
    SESSION_CLOSE_HOUR * 60 - (SESSION_OPEN_HOUR * 60 + SESSION_OPEN_MINUTE)
) // INTERVAL_MINUTES
panel = load_intraday_panel([EXEC_SYMBOL], TAQ_START_DATE, TAQ_END_DATE, INTERVAL_MINUTES)
print(
    f"Loaded {panel.height:,} interval rows for {panel['symbol'].n_unique()} symbols "
    f"across {panel['session_date'].n_unique()} sessions ({N_BUCKETS} intervals per session)"
)


# %% [markdown]
# ### Extract Per-Day Price and Volume Paths
#
# Keep only sessions with full interval coverage so each day yields aligned
# price and volume arrays of length `N_BUCKETS`.


# %%
def day_paths(panel: pl.DataFrame, symbol: str, n_buckets: int) -> dict:
    """Return {session_date: (price_path, volume_path)} for complete sessions."""
    sym = panel.filter(pl.col("symbol") == symbol).sort("session_date", "bucket")
    out = {}
    for d in sym["session_date"].unique().to_list():
        g = sym.filter(pl.col("session_date") == d)
        if g.height == n_buckets:
            out[d] = (g["price"].to_numpy(), g["volume"].to_numpy())
    return out


# %% [markdown]
# ### Split the Sessions Before Estimating Anything
#
# The sessions are divided at the calendar date declared in the settings. Everything on or before
# it estimates the volume profile; everything after it is where both schedules run. Splitting on a
# date rather than on a fraction of the sample means that adding more data later extends the test
# window instead of silently moving the estimation window under a result already reported.

# %%
profile_symbol = EXEC_SYMBOL
paths_by_day = day_paths(panel, profile_symbol, N_BUCKETS)
all_days = sorted(paths_by_day)
profile_end_date = datetime.fromisoformat(PROFILE_END_DATE).date()
estimation_days = [day for day in all_days if day <= profile_end_date]
execution_days = [day for day in all_days if day > profile_end_date]
if not estimation_days or not execution_days:
    raise ValueError("The fixed profile boundary must leave estimation and execution sessions")

volume_matrix = np.vstack([paths_by_day[d][1] / paths_by_day[d][1].sum() for d in estimation_days])
base_curve = volume_matrix.mean(axis=0)
base_curve = base_curve / base_curve.sum()
print(
    f"{profile_symbol}: {len(estimation_days)} estimation sessions, "
    f"{len(execution_days)} execution sessions"
)

# %%
# Plot the measured average profile against the actual per-day realizations.
fig = go.Figure()
profile_start, profile_end = session_grid(estimation_days[0])
bucket_times = build_time_grid(profile_start, profile_end, INTERVAL_MINUTES)
for d in estimation_days:
    daily = paths_by_day[d][1] / paths_by_day[d][1].sum()
    _ = fig.add_scatter(
        x=bucket_times,
        y=daily * 100,
        mode="lines",
        line=dict(width=1, color=COLORS["silver"]),
        opacity=0.4,
        showlegend=False,
    )
_ = fig.add_scatter(
    x=bucket_times,
    y=base_curve * 100,
    mode="lines",
    name="Mean Volume %",
    line=dict(color=COLORS["blue"], width=3),
)
fig.update_layout(
    title=f"{profile_symbol} volume concentrates around the open and close",
    xaxis_title="Execution time (ET)",
    yaxis_title=f"Share of daily volume per {INTERVAL_MINUTES}-minute interval (%)",
    height=450,
)
fig.update_xaxes(tickformat="%H:%M")
show_plotly_with_alt(
    fig,
    "One faint line per estimation session showing that day's share of volume by interval, with "
    "the average across them drawn heavy on top. The average traces a U: highest at the open, "
    "falling to a midday floor, rising into the close. The individual days scatter widely around "
    "it, most so at the open.",
)

# %%
open_slice = base_curve[: max(1, N_BUCKETS // 6)]
midday_slice = base_curve[N_BUCKETS // 3 : 2 * N_BUCKETS // 3]
close_slice = base_curve[-max(1, N_BUCKETS // 6) :]
even_share = 1 / N_BUCKETS
for label, part in (
    ("Opening hour", open_slice),
    ("Midday", midday_slice),
    ("Closing hour", close_slice),
):
    print(
        f"{label:<13} {len(part):>2} intervals, {part.sum():>5.1%} of the day's volume, "
        f"{part.mean() / even_share:>4.1f}x an evenly traded interval"
    )

# %% [markdown]
# **Finding**: The measured profile concentrates volume near the open and close
# and thins out midday - the empirical U-shape, not an assumed one. The spread of
# the faint per-day lines around the mean is the real day-to-day forecast
# uncertainty a VWAP desk faces; it is wider than any smooth parametric curve
# would suggest, which is exactly why VWAP tracking is imperfect in practice.


# %% [markdown]
# ### Volume-Curve Alignment Helper
#
# Resamples any input volume curve onto the execution grid so that VWAP
# schedules can consume forecasts at arbitrary granularity.


# %%
def normalize_volume_curve(volume_curve: np.ndarray, n_intervals: int) -> np.ndarray:
    """Match any input curve to the target execution grid."""
    curve = np.asarray(volume_curve, dtype=float)
    if curve.ndim != 1 or curve.size == 0:
        raise ValueError("volume_curve must be a non-empty one-dimensional array")
    if not np.isfinite(curve).all() or (curve < 0).any() or curve.sum() <= 0:
        raise ValueError("volume_curve must contain finite nonnegative values with positive sum")
    curve = curve / curve.sum()
    if len(curve) == n_intervals:
        return curve

    aligned_curve = np.interp(
        np.linspace(0, 1, n_intervals),
        np.linspace(0, 1, len(curve)),
        curve,
    )
    return aligned_curve / aligned_curve.sum()


# %% [markdown]
# ### Convert a Volume Curve into an Execution Schedule


# %%
def build_vwap_schedule(
    total_shares: int,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
    volume_curve: np.ndarray,
) -> pl.DataFrame:
    """Allocate shares in proportion to the expected intraday volume profile."""
    if total_shares <= 0:
        raise ValueError("total_shares must be positive")
    times = build_time_grid(start_time, end_time, interval_minutes)
    n_intervals = len(times)
    weights = normalize_volume_curve(volume_curve, n_intervals)
    shares_float = total_shares * weights
    shares_int = np.floor(shares_float).astype(int)

    remainder = total_shares - shares_int.sum()
    if remainder > 0:
        fractions = shares_float - shares_int
        top_idx = np.argsort(fractions)[-int(remainder) :]
        shares_int[top_idx] += 1

    cumulative = np.cumsum(shares_int)
    return pl.DataFrame(
        {
            "timestamp": times,
            "shares": shares_int,
            "volume_weight": weights,
            "cumulative": cumulative,
            "pct_complete": cumulative / total_shares,
        }
    )


# %% [markdown]
# ### VWAP Summary Helper


# %%
def summarize_vwap(algo) -> dict:
    """Create a compact VWAP schedule summary."""
    return {
        "algorithm": "VWAP",
        "total_shares": algo.total_shares,
        "n_intervals": len(algo.schedule),
        "interval_minutes": algo.interval_minutes,
        "max_shares_interval": algo.schedule["shares"].max(),
        "min_shares_interval": algo.schedule["shares"].min(),
    }


# %% [markdown]
# ### VWAP Class Definition


# %%
class VWAPAlgorithm:
    """Volume-Weighted Average Price execution algorithm."""

    def __init__(
        self,
        total_shares: int,
        start_time: datetime,
        end_time: datetime,
        volume_curve: np.ndarray | None = None,
        interval_minutes: int = 5,
    ):
        self.total_shares = total_shares
        self.start_time = start_time
        self.end_time = end_time
        self.interval_minutes = interval_minutes
        if volume_curve is None:
            raise ValueError("VWAP requires a measured volume_curve (the empirical profile).")
        self.schedule = build_vwap_schedule(
            total_shares=self.total_shares,
            start_time=self.start_time,
            end_time=self.end_time,
            interval_minutes=self.interval_minutes,
            volume_curve=volume_curve,
        )
        self.volume_curve = self.schedule["volume_weight"].to_numpy()

    def get_target_at_time(self, current_time: datetime) -> int:
        return latest_cumulative_target(self.schedule, current_time)

    def summary(self) -> dict:
        return summarize_vwap(self)


# %%
# Example VWAP schedule, driven by the measured intraday volume profile
vwap = VWAPAlgorithm(
    total_shares=ORDER_SHARES,
    start_time=start,
    end_time=end,
    volume_curve=base_curve,
    interval_minutes=15,
)

print("VWAP Schedule Summary:")
for k, v in vwap.summary().items():
    print(f"  {k}: {v}")

if vwap.schedule["shares"].sum() != vwap.total_shares:
    raise ValueError("The VWAP schedule must allocate exactly the parent order")

# %% [markdown]
# **Reading the summary**: The gap between the largest and smallest interval is the U-curve
# turned into share counts. Every one of those counts came out of the forecast, so the schedule
# inherits whatever the forecast got wrong.

# %%
# Compare TWAP vs VWAP schedules
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Shares per Interval", "Cumulative Execution"],
    vertical_spacing=0.1,
)

# TWAP provides the neutral baseline.
twap_times = twap.schedule["timestamp"].to_list()
_ = fig.add_bar(
    x=twap_times,
    y=twap.schedule["shares"].to_list(),
    name="TWAP",
    marker_color=COLORS["neutral"],
    row=1,
    col=1,
)
_ = fig.add_scatter(
    x=twap_times,
    y=twap.schedule["pct_complete"].to_list(),
    mode="lines",
    name="TWAP Cumulative",
    line=dict(color=COLORS["neutral"], width=2, dash="dash"),
    row=2,
    col=1,
)

# %% [markdown]
# #### Add VWAP Schedule Overlay

# %%
# VWAP is the focal schedule.
vwap_times = vwap.schedule["timestamp"].to_list()
_ = fig.add_bar(
    x=vwap_times,
    y=vwap.schedule["shares"].to_list(),
    name="VWAP",
    marker_color=COLORS["blue"],
    row=1,
    col=1,
)
_ = fig.add_scatter(
    x=vwap_times,
    y=vwap.schedule["pct_complete"].to_list(),
    mode="lines",
    name="VWAP Cumulative",
    line=dict(color=COLORS["blue"], width=2),
    row=2,
    col=1,
)

fig.update_yaxes(tickformat=".0%", row=2, col=1)
fig.update_yaxes(title_text=f"Shares per {INTERVAL_MINUTES}-minute interval", row=1, col=1)
fig.update_yaxes(title_text="Order completed", row=2, col=1)
fig.update_xaxes(title_text="Execution time (ET)", tickformat="%H:%M", row=2, col=1)
fig.update_layout(
    title=f"VWAP shifts {profile_symbol} shares toward the liquid open and close",
    height=600,
    barmode="group",
)
show_plotly_with_alt(
    fig,
    "Upper panel: paired bars per interval, TWAP flat and VWAP taller at the open and close and "
    "shorter through midday. Lower panel: their completion curves, TWAP a straight line and VWAP "
    "steeper at both ends and flatter in the middle.",
)

# %% [markdown]
# **Finding**: VWAP reduces share count in the lunch-hour trough and shifts risk
# toward the opening and closing intervals. Relative to TWAP, it accepts forecast risk
# in exchange for lower expected footprint when the market is thin.

# %% [markdown]
# ## Part 3: Executing Against a Real Trading Day
#
# We now run the TWAP and VWAP schedules against an actual NASDAQ-100 session -
# real interval prices and real interval volumes - and measure realized execution
# price against that day's market VWAP.


# %%
def shares_from_profile(profile: np.ndarray, total_shares: int) -> np.ndarray:
    """Allocate integer shares across intervals in proportion to a profile."""
    if total_shares <= 0:
        raise ValueError("total_shares must be positive")
    weights = normalize_volume_curve(profile, len(profile))
    raw = total_shares * weights
    shares = np.floor(raw).astype(int)
    remainder = total_shares - int(shares.sum())
    if remainder > 0:
        top_idx = np.argsort(raw - shares)[-remainder:]
        shares[top_idx] += 1
    return shares


# %% [markdown]
# ### Execute a Schedule Against Real Interval Prices and Volumes
#
# Impact uses the interval's **actual market volume** as the participation
# denominator under the square-root law: a slice arriving in a high-volume
# interval pays less impact than the same slice in the midday lull. This is the
# participation-rate edge VWAP is designed to exploit.


# %%
def execute_day(
    shares: np.ndarray,
    price_path: np.ndarray,
    volume_path: np.ndarray,
    impact_bps: float,
) -> tuple[float, np.ndarray]:
    """Return (realized VWAP, per-interval execution price) for a share schedule."""
    if not (len(shares) == len(price_path) == len(volume_path)):
        raise ValueError("shares, price_path, and volume_path must have equal length")
    if shares.sum() <= 0 or (shares < 0).any():
        raise ValueError("shares must be nonnegative with a positive total")
    if not np.isfinite(price_path).all() or not np.isfinite(volume_path).all():
        raise ValueError("price and volume paths must be finite")
    if (price_path <= 0).any() or (volume_path < 0).any():
        raise ValueError("prices must be positive and volumes nonnegative")
    interval_volume = np.maximum(volume_path.astype(float), 1.0)
    participation = np.maximum(shares, 0) / interval_volume
    impact_fraction = impact_bps / 10_000 * np.sqrt(participation)
    exec_price = price_path * (1 + impact_fraction)
    realized_vwap = float((exec_price * shares).sum() / shares.sum())
    return realized_vwap, exec_price


# %% [markdown]
# ### Market VWAP Benchmark
#
# The benchmark weights each interval's observed price by its share of realized
# market volume. It is used only after the predetermined schedules execute.


# %%
def market_vwap_of(price_path: np.ndarray, volume_path: np.ndarray) -> float:
    """Volume-weighted average price actually traded over the session."""
    if len(price_path) != len(volume_path) or len(price_path) == 0:
        raise ValueError("price_path and volume_path must have equal positive length")
    if not np.isfinite(price_path).all() or not np.isfinite(volume_path).all():
        raise ValueError("price and volume paths must be finite")
    if (price_path <= 0).any() or (volume_path < 0).any() or volume_path.sum() <= 0:
        raise ValueError("prices must be positive and volumes nonnegative with positive total")
    return float((price_path * volume_path).sum() / volume_path.sum())


# %%
# Build the two schedules on the execution grid and run them against one real day.
twap_shares = shares_from_profile(np.ones(N_BUCKETS), ORDER_SHARES)
vwap_shares = shares_from_profile(base_curve, ORDER_SHARES)

demo_day = execution_days[0]
demo_price, demo_volume = paths_by_day[demo_day]
demo_start, demo_end = session_grid(demo_day)
demo_times = build_time_grid(demo_start, demo_end, INTERVAL_MINUTES)

market_vwap = market_vwap_of(demo_price, demo_volume)
twap_realized, twap_exec_price = execute_day(twap_shares, demo_price, demo_volume, IMPACT_BPS)
vwap_realized, vwap_exec_price = execute_day(vwap_shares, demo_price, demo_volume, IMPACT_BPS)

twap_vs_benchmark = (twap_realized / market_vwap - 1) * 10000
vwap_vs_benchmark = (vwap_realized / market_vwap - 1) * 10000

print(f"Session: {demo_day} ({profile_symbol})")
print("=" * 50)
print(f"Market VWAP:    ${market_vwap:.4f}")
print(f"TWAP Realized:  ${twap_realized:.4f} ({twap_vs_benchmark:+.1f} bps vs VWAP)")
print(f"VWAP Realized:  ${vwap_realized:.4f} ({vwap_vs_benchmark:+.1f} bps vs VWAP)")

# %% [markdown]
# **Reading the comparison**: Both schedules traded the same shares on the same day, so the gap
# between them is entirely down to when each one traded. The sign says which side of the day's
# average each finished on, and the size says by how much - but on a single session, either could
# land closer by chance, which is why Part 4 runs all of them.

# %%
# Visualize execution against the real session
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Price Path & Executions", "Cumulative Fill"],
    vertical_spacing=0.1,
)

_ = fig.add_scatter(
    x=demo_times,
    y=demo_price,
    mode="lines",
    name="Market Price",
    line=dict(color=COLORS["neutral"], width=1),
    row=1,
    col=1,
)

# %% [markdown]
# #### Realized Benchmark Comparison

# %%
for name, value, color, dash in (
    ("Market VWAP", market_vwap, COLORS["amber"], "dash"),
    (f"TWAP realized ({twap_vs_benchmark:+.1f} bps)", twap_realized, COLORS["neutral"], "dot"),
    (f"VWAP realized ({vwap_vs_benchmark:+.1f} bps)", vwap_realized, COLORS["blue"], "dash"),
):
    fig.add_scatter(
        x=[demo_times[0], demo_times[-1]],
        y=[value, value],
        mode="lines",
        name=name,
        line=dict(color=color, width=2, dash=dash),
        row=1,
        col=1,
    )

# %% [markdown]
# #### Cumulative Fill Comparison


# %%
_ = fig.add_scatter(
    x=demo_times,
    y=np.cumsum(twap_shares) / ORDER_SHARES,
    mode="lines",
    name="TWAP Fill",
    line=dict(color=COLORS["neutral"], width=2, dash="dash"),
    row=2,
    col=1,
)
_ = fig.add_scatter(
    x=demo_times,
    y=np.cumsum(vwap_shares) / ORDER_SHARES,
    mode="lines",
    name="VWAP Fill",
    line=dict(color=COLORS["blue"], width=2),
    row=2,
    col=1,
)
fig.update_xaxes(title_text="Execution time (ET)", tickformat="%H:%M", row=2, col=1)
fig.update_yaxes(title_text="Execution price (USD)", tickprefix="$", row=1, col=1)
fig.update_yaxes(title_text="Order completed", tickformat=".0%", row=2, col=1)
fig.update_layout(
    title=f"Both schedules against one held-out session, {demo_day}",
    height=600,
)
show_plotly_with_alt(
    fig,
    "Upper panel: the session's price path with three horizontal lines marking the market VWAP "
    "and each schedule's realized average price. Lower panel: cumulative fill against time, with "
    "the VWAP curve steeper at the open and close and flatter through midday than the straight "
    "TWAP line.",
)

# %% [markdown]
# **Reading the chart**: The lower panel shows where each schedule chose to be exposed. VWAP
# completes a larger share of the order early and late, so it holds less of the position through
# the middle of the day and more of its fills happen when the benchmark itself is being set.

# %% [markdown]
# ## Part 4: Cross-Session Distribution
#
# A single session can favor either schedule by luck. We run both schedules across
# every held-out execution session and compare the *distribution* of slippage
# versus each day's market VWAP. The volume profile feeding VWAP was estimated on
# the earlier window, so this is an out-of-sample test of the forecast.


# %%
def run_across_days(
    days: list,
    paths: dict,
    twap_shares: np.ndarray,
    vwap_shares: np.ndarray,
    impact_bps: float,
) -> pl.DataFrame:
    """Execute both schedules on each real session and collect slippage vs VWAP."""
    rows = []
    for d in days:
        price_path, volume_path = paths[d]
        mkt = market_vwap_of(price_path, volume_path)
        twap_realized, _ = execute_day(twap_shares, price_path, volume_path, impact_bps)
        vwap_realized, _ = execute_day(vwap_shares, price_path, volume_path, impact_bps)
        rows.append(
            {
                "session_date": d,
                "twap_vs_market_bps": (twap_realized / mkt - 1) * 10000,
                "vwap_vs_market_bps": (vwap_realized / mkt - 1) * 10000,
            }
        )
    return pl.DataFrame(rows)


# %%
session_results = run_across_days(
    execution_days,
    paths_by_day,
    twap_shares,
    vwap_shares,
    IMPACT_BPS,
)

# %% [markdown]
# The four measures answer different questions and can disagree. The signed mean says whether a
# schedule systematically pays above or below the benchmark. The standard deviation, the mean
# absolute error and the worst single session say how far it strays, regardless of direction. A
# schedule can have a mean near zero because it misses by a lot in both directions equally.

# %% [markdown]
# ### Benchmark Tracking Metrics
#
# A schedule can have a favorable signed bias while missing the benchmark widely.
# We therefore report bias separately from distance-based tracking errors.


# %%
def summarize_tracking(results: pl.DataFrame) -> pl.DataFrame:
    """Summarize signed bias and distance from market VWAP for each schedule."""
    rows = []
    for algorithm in ("TWAP", "VWAP"):
        values = results[f"{algorithm.lower()}_vs_market_bps"].to_numpy()
        rows.append(
            {
                "algorithm": algorithm,
                "mean_bias_bps": values.mean(),
                "tracking_std_bps": values.std(ddof=1),
                "mae_bps": np.abs(values).mean(),
                "rmse_bps": np.sqrt(np.mean(values**2)),
                "worst_abs_bps": np.abs(values).max(),
            }
        )
    return pl.DataFrame(rows)


# %%
tracking_summary = summarize_tracking(session_results)
print(f"Held-out tracking summary: {session_results.height} {profile_symbol} sessions")
tracking_summary

# %%
# Visualize results
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=["TWAP", "VWAP"],
)

all_slippage = np.concatenate(
    [
        session_results["twap_vs_market_bps"].to_numpy(),
        session_results["vwap_vs_market_bps"].to_numpy(),
    ]
)
hist_start = float(np.floor(all_slippage.min() / 5) * 5)
hist_end = float(np.ceil(all_slippage.max() / 5) * 5)
bin_size = max(5.0, (hist_end - hist_start) / 12)

_ = fig.add_histogram(
    x=session_results["twap_vs_market_bps"].to_list(),
    xbins=dict(start=hist_start, end=hist_end, size=bin_size),
    name="TWAP",
    marker_color=COLORS["neutral"],
    row=1,
    col=1,
)

_ = fig.add_histogram(
    x=session_results["vwap_vs_market_bps"].to_list(),
    xbins=dict(start=hist_start, end=hist_end, size=bin_size),
    name="VWAP",
    marker_color=COLORS["blue"],
    row=1,
    col=2,
)

# %% [markdown]
# #### Shared Benchmark and Bias Markers

# %%
# Zero is exact benchmark tracking; colored dashed lines show signed bias.
twap_mean = session_results["twap_vs_market_bps"].mean()
vwap_mean = session_results["vwap_vs_market_bps"].mean()
for column in (1, 2):
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["neutral"], row=1, col=column)

fig.add_vline(x=twap_mean, line_dash="dash", line_color=COLORS["slate"], row=1, col=1)
fig.add_vline(x=vwap_mean, line_dash="dash", line_color=COLORS["blue"], row=1, col=2)

fig.update_xaxes(title_text="Slippage vs market VWAP (bps)", range=[hist_start, hist_end])
fig.update_yaxes(title_text="Held-out sessions", row=1, col=1)
fig.update_layout(
    title="Held-out benchmark slippage, one session per observation",
    height=400,
    showlegend=False,
)
show_plotly_with_alt(
    fig,
    "Two histograms of slippage against market VWAP on a shared axis, TWAP on the left and VWAP "
    "on the right. Both centre near zero; the VWAP distribution is visibly narrower and its tails "
    "reach less far in both directions.",
)

# %% [markdown]
# **Reading the histograms**: The two panels share an axis, so their widths are directly
# comparable. The dotted line is exact benchmark tracking and the dashed one is each schedule's
# own average. What separates the two distributions is their width, not their centre - and a
# narrower distribution means a more predictable execution price, not a cheaper one.

# %% [markdown]
# ## Part 5: Choosing Between Them, and What Comes After
#
# The choice turns on one question: is there a volume forecast worth having? A liquid stock with a
# stable daily pattern gives VWAP something to work with. A thinly traded name, a day with a
# scheduled event, or a market whose shape shifts with the news does not, and TWAP's indifference
# to all of it becomes an advantage rather than a limitation.
#
# Three variants relax an assumption each:
#
# - **Adaptive VWAP** re-estimates the remaining day's volume shape as the session unfolds, instead
#   of committing to a forecast made before the open. `08_ml_dynamic_execution` builds one.
# - **Percent of volume** abandons a fixed schedule and instead trades a constant fraction of
#   whatever volume actually prints. The participation rate is then exactly controlled and the
#   completion time is not, so an order can fail to finish on a quiet day.
# - **Implementation shortfall** changes the benchmark rather than the schedule. It scores execution
#   against the price when the decision was made, not against the day's average, which makes
#   trading slowly a risk rather than a virtue - a price that moves away while an order is being
#   worked is a cost under that benchmark and invisible under this one.
#   `05_almgren_chriss_optimal_execution` optimizes against exactly that trade-off.

# %% [markdown]
# ## Key Takeaways
#
# 1. **Spread an order in proportion to the liquidity available, not in proportion to time.**
#    Impact scales with $\sqrt{\text{shares}/\text{interval volume}}$, so a slice sent into a
#    quiet interval costs more per share than the same slice sent into a busy one. Matching the
#    volume shape holds that ratio roughly constant across the session.
#
# 2. **Estimate the volume forecast on one window and test it on another.** A profile fitted on
#    the same days it is evaluated on will always look good. Splitting at a fixed calendar date
#    is what turns the comparison in this notebook into evidence about a forecast rather than a
#    description of one.
#
# 3. **Judge a schedule on the distribution of its outcomes, not on one session.** Any single day
#    can favour either schedule by chance. Report the spread across sessions and read bias
#    separately from dispersion: a schedule can sit close to the benchmark on average while
#    missing it widely in both directions.
#
# 4. **A tighter benchmark distribution is not a better price.** Both schedules are scored against
#    the day's own volume-weighted average, so tracking it closely means being average, reliably.
#    Under a benchmark set at the decision price instead, trading slowly is itself a risk.
#
# 5. **Predictability is what makes a schedule auditable and what makes it exploitable.** The same
#    property that lets a desk be held to a VWAP schedule lets other participants anticipate a
#    large one.
#
# ### Known limitations
#
# - One symbol, one quarter, one order size. A less liquid name would have a noisier volume
#   profile and a larger participation rate for the same order.
# - The impact coefficient is stated rather than calibrated, and it applies to both schedules
#   identically, so it moves the level of the comparison and not its direction.
# - Impact is charged against the volume that actually traded in each interval, which includes the
#   order's own hypothetical shares. A real order of this size would change the volume it is
#   measured against.
# - The simulation fills every slice at the interval's volume-weighted price. Real fills arrive at
#   individual prints, and a slice can go unfilled.
# - The volume profile is a single average over the estimation window, with no adjustment for
#   weekday, index events, or expiry.
#
# **Next**: `05_almgren_chriss_optimal_execution` derives the schedule that optimizes cost against
# timing risk rather than tracking a benchmark; `08_ml_dynamic_execution` re-estimates the forecast
# as the session unfolds.
