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
# # Point-in-Time (PIT) Data Validation
#
# **Docker image**: `ml4t`
#
# ## Purpose
# Demonstrate the operational checks needed to keep a backtest free of
# lookahead bias on real ETF and macroeconomic data. Three threads run
# through the notebook: feature-time vs decision-time alignment, a
# correlation-based heuristic for catching obvious feature leakage, and the
# bitemporal (vintage) view of macro data via the FRED provider.
#
# ## Learning Objectives
# - Distinguish *event time* (when something happened) from *knowledge time*
#   (when we learned about it), and the bitemporal axis for macro releases.
# - Show why a centered moving average leaks future information, and why a
#   level-versus-return correlation heuristic misses it.
# - Build the corrected scale-invariant heuristic (feature-return against
#   future-return) that does fire for `close.pct_change(-1)` features.
# - Determine, by measurement, what the macro panel's timestamps mean, and
#   what a forward-fill over them costs when they are period stamps.
# - Query FRED with `vintage_date` to see how the GDP advance estimate
#   differs from the revised value.
#
# ## Book reference
# Chapter 2, §2.3 (data quality framework — point-in-time correctness and
# bitemporal data). The figure below is the §2.3 illustration.
#
# ## Prerequisites
# - ETF parquet files materialized under `ML4T_DATA_PATH`.
# - Macro parquet (FRED snapshot) materialized under `ML4T_DATA_PATH` or
#   loadable via `data.load_macro`.
# - `FRED_API_KEY` environment variable for the live vintage query, which is
#   the last section so that everything before it runs without a key
#   (free key at https://fred.stlouisfed.org/docs/api/api_key.html).

# %%
"""Point-in-Time Data Validation."""

import os
from datetime import datetime

import plotly.graph_objects as go
import polars as pl
from ml4t.data.providers import FREDProvider

from data import load_etfs, load_macro
from utils.style import COLORS, show_plotly_with_alt

# %% [markdown]
# ### Declared parameters
#
# `LEAK_THRESHOLD` is the correlation above which the validator in the last-but-one section
# calls a feature leaky. It is declared here rather than living on the class so the value the
# reader sees and the value the check uses are the same one, and Section 4 prints the measured
# correlations it has to separate.
#
# `MONTHLY_RELEASE_LAG_DAYS` and `QUARTERLY_RELEASE_LAG_DAYS` are how long after a period ends
# its first estimate is published. They are round numbers standing in for a real release
# calendar: US payrolls and unemployment land on the first Friday after the reference month,
# CPI in the middle of the following month, and the GDP advance estimate about four weeks
# after the quarter closes. A production system reads the actual calendar; these two make the
# size of the correction visible.

# %% tags=["parameters"]
DEMO_SYMBOL = "SPY"
MA_WINDOW = 5  # window for the leakage demonstrations
MA_WINDOW_LONG = 20  # window for the figure
PLOT_TAIL_DAYS = 120
LEAK_THRESHOLD = 0.30  # |corr| above which the validator flags a feature
MAX_GAP_DAYS = 5
EXECUTION_LAG_ROWS = 1  # trading days between a close-of-day signal and its fill

MONTHLY_RELEASE_LAG_DAYS = 7
QUARTERLY_RELEASE_LAG_DAYS = 30
MACRO_DEMO_SERIES = "unrate"

# %% [markdown]
# ## 1. Load Real Market Data
#
# Daily bars for one ETF anchor every example below.

# %%
spy = load_etfs().filter(pl.col("symbol") == DEMO_SYMBOL).sort("timestamp")
print(f"{DEMO_SYMBOL}: {len(spy):,} rows; {spy['timestamp'].min()} to {spy['timestamp'].max()}")
spy.head()

# %% [markdown]
# ## 2. Lookahead Bias: A Visual Demonstration
#
# A trailing moving average of width $w$ at time $T$ averages $[T - w + 1,\ T]$, which is what
# a live system can compute. A *centered* one is the same window slid forward so that $T$ sits
# near its middle, which puts some of its inputs after $T$.
#
# Exactly how many depends on the library's centering convention and on whether the width is
# odd or even, so the next cell asks rather than assumes: running the centered average over a
# ramp makes each output reveal the window that produced it.


# %%
def centered_window_offsets(width: int) -> tuple[int, int]:
    """Return the first and last offsets, relative to T, of a centered window of `width`.

    Averaging a ramp gives back the midpoint of whatever window was used, so one output value
    identifies the window exactly.
    """
    ramp = pl.DataFrame({"x": [float(i) for i in range(4 * width)]}).with_columns(
        pl.col("x").rolling_mean(window_size=width, center=True).alias("centered")
    )
    row = next(i for i, v in enumerate(ramp["centered"]) if v is not None)
    # The mean of a ramp is its window's midpoint, so this recovers the window's first input.
    first_input = ramp["centered"][row] - (width - 1) / 2
    return int(first_input - row), int(first_input + width - 1 - row)


for _w in (MA_WINDOW, MA_WINDOW_LONG):
    _lo, _hi = centered_window_offsets(_w)
    print(
        f"Centered window of width {_w:>2}: [T{_lo:+d}, T{_hi:+d}], "
        f"{_hi} of {_w} inputs are in the future"
    )

# %%
spy_ma = (
    spy.with_columns(
        pl.col("close").rolling_mean(window_size=MA_WINDOW_LONG).alias("ma_trailing"),
        pl.col("close").rolling_mean(window_size=MA_WINDOW_LONG, center=True).alias("ma_centered"),
    )
    .drop_nulls()
    .tail(PLOT_TAIL_DAYS)
)

_tracking = spy_ma.select(
    (pl.col("close") - pl.col("ma_trailing")).abs().mean().alias("trailing"),
    (pl.col("close") - pl.col("ma_centered")).abs().mean().alias("centered"),
)
print(
    f"Mean absolute distance from the close over the plotted window: "
    f"trailing {_tracking['trailing'][0]:.2f}, centered {_tracking['centered'][0]:.2f}"
)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=spy_ma["timestamp"].to_list(),
        y=spy_ma["close"].to_list(),
        name="SPY close",
        line=dict(color=COLORS["blue"], width=1),
    )
)
fig.add_trace(
    go.Scatter(
        x=spy_ma["timestamp"].to_list(),
        y=spy_ma["ma_trailing"].to_list(),
        name=f"{MA_WINDOW_LONG}d trailing MA",
        line=dict(color=COLORS["slate"], width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=spy_ma["timestamp"].to_list(),
        y=spy_ma["ma_centered"].to_list(),
        name=f"{MA_WINDOW_LONG}d centered MA",
        line=dict(color=COLORS["copper"], width=2, dash="dash"),
    )
)
fig.update_layout(
    title=f"{DEMO_SYMBOL} close with trailing and centered moving averages",
    xaxis_title="Date",
    yaxis_title=f"{DEMO_SYMBOL} price",
    height=420,
)
show_plotly_with_alt(
    fig,
    "A price line rising across about six months, with two moving averages drawn over it. "
    "The solid trailing average runs below the price and turns after each move in it. The "
    "dashed centered average runs above the trailing one for the whole window, shifted "
    "left relative to it, and tracks the middle of the price's oscillation rather than "
    "lagging behind it.",
)

# %% [markdown]
# The centered average sits closer to the close, by the measured distance printed above, and
# it turns where the price turns rather than after it. Both follow from the same thing: half
# its inputs are prices that had not printed yet. No live system can produce that line, so a
# backtest that uses it reports skill the strategy could not have had.
#
# The share of a centered window that lies in the future is just under half and stays there as
# the width grows, as the two measured windows above show. A longer centered average does not
# leak proportionally less; it leaks further ahead.
#
# There is a second tell, and it is the one that shows up first in practice. A centered
# average has no value for the most recent rows, because they are still waiting for inputs,
# so `drop_nulls` above silently ends the plotted series short of the data. A feature that
# cannot be computed for today is a feature no live system can trade on, and that missing
# tail is visible long before any correlation test is run.

# %% [markdown]
# ## 3. Why the Naive Correlation Heuristic Fails
#
# A leakage test that suggests itself is to correlate a feature's *level* with the next
# period's *return*. It does not work, and the reason is not subtle once stated: a price
# level is dominated by where the series has drifted to over years, and next-day returns are
# close to mean-zero noise around that drift. The correlation is between a slow trend and a
# fast wiggle, and it comes out near zero whether or not the feature contains tomorrow's
# price. The next cell scores three features, one of which is tomorrow's close itself.


# %%
def naive_leakage_corr(df: pl.DataFrame, feature: str, price: str = "close") -> float:
    """Correlation between a level feature and the next-period price return.

    Reproduces the textbook-warning heuristic that *seems* like it should
    catch lookahead but rarely does because it compares incompatible scales.
    """
    enriched = df.with_columns(
        (pl.col(price).shift(-1) / pl.col(price) - 1).alias("next_ret")
    ).drop_nulls([feature, "next_ret"])
    return float(enriched.select(pl.corr(feature, "next_ret")).item())


# %%
naive_features = spy.with_columns(
    pl.col("close").rolling_mean(window_size=MA_WINDOW).alias("ma_trailing"),
    pl.col("close").rolling_mean(window_size=MA_WINDOW, center=True).alias("ma_centered"),
    pl.col("close").shift(-1).alias("tomorrow_close"),
).drop_nulls()

LEAKAGE_FEATURES = ["ma_trailing", "ma_centered", "tomorrow_close"]

naive_results = pl.DataFrame(
    [
        {"feature": f, "naive_corr_with_next_return": naive_leakage_corr(naive_features, f)}
        for f in LEAKAGE_FEATURES
    ]
)
naive_results

# %% [markdown]
# All three score near zero, including the one that *is* tomorrow's price. The test does not
# separate a clean feature from a leaking one, so a feature matrix passing it has learned
# nothing about itself.
#
# What the test needs is for both sides to be the same kind of quantity. Turning the feature
# into its own period-over-period return does that, and it is what the next section builds.

# %% [markdown]
# ## 4. A Correct Scale-Invariant Leakage Test
#
# Convert any feature to its own period-over-period return, then correlate that with the next
# period's price return. A feature built from $\mathrm{close}_{T+k}$ now exposes itself: for
# $k \ge 1$ the feature's return at $T$ *is* the price return $k$ periods ahead, so the
# correlation goes to one.


# %%
def leakage_corr_scale_invariant(df: pl.DataFrame, feature: str, price: str = "close") -> float:
    """Correlation between a feature's own return and the next price return.

    Most useful when `feature` is a price-like (level) series — for return-
    shaped features the pct_change is unstable near zero, so the caller
    should pass them through directly instead.
    """
    enriched = (
        df.with_columns(
            pl.col(feature).pct_change().alias("feat_ret"),
            (pl.col(price).shift(-1) / pl.col(price) - 1).alias("next_ret"),
        )
        .drop_nulls(["feat_ret", "next_ret"])
        .filter(pl.col("feat_ret").is_finite() & pl.col("next_ret").is_finite())
    )
    if enriched.is_empty():
        return float("nan")
    return float(enriched.select(pl.corr("feat_ret", "next_ret")).item())


# %%
fixed_results = pl.DataFrame(
    [
        {
            "feature": f,
            "scale_invariant_corr": leakage_corr_scale_invariant(naive_features, f),
        }
        for f in LEAKAGE_FEATURES
    ]
)
fixed_results

# %% [markdown]
# The same three features now sort: the trailing average stays low, the centered average rises
# because part of its window sits in the future, and `tomorrow_close` saturates near one
# because it is the next day's price.
#
# What separates them is a gap in the middle of the scale, and the validator later in this
# notebook puts its threshold there. The gap is what makes a threshold possible at all: a
# detector whose clean and leaking cases sat close together would need a cutoff chosen to
# produce the answer already known, which is not a detector.

# %% [markdown]
# ## 5. Signal-to-Trade Lag: Trading Days, Not Calendar Days
#
# A signal computed from the close of $T$ can be acted on no earlier than the open of $T+1$.
# The shift has to be by *rows* rather than by calendar days, because the next tradable
# moment after a Friday close is a Monday open and no amount of date arithmetic knows which
# Mondays are holidays. Shifting the timestamp column by rows inherits the trading calendar
# from the data itself.


# %%
def validate_signal_trade_lag(signals: pl.DataFrame, execution_lag: int = 1) -> pl.DataFrame:
    """Tag each signal with the earliest tradable date `execution_lag` rows ahead."""
    return signals.sort("timestamp").with_columns(
        pl.col("timestamp").alias("signal_date"),
        pl.col("timestamp").shift(-execution_lag).alias("earliest_execution"),
    )


# %%
spy_with_signal = spy.with_columns(
    (pl.col("close") / pl.col("close").shift(MA_WINDOW) - 1).alias("momentum_signal")
)
_lagged = validate_signal_trade_lag(spy_with_signal, execution_lag=EXECUTION_LAG_ROWS)
_calendar_gap = _lagged.select(
    (pl.col("earliest_execution") - pl.col("signal_date")).dt.total_days().alias("days")
).drop_nulls()
print(
    f"Calendar days between a signal and its earliest execution: "
    f"{_calendar_gap['days'].min()} to {_calendar_gap['days'].max()}, "
    f"for a lag of {EXECUTION_LAG_ROWS} trading day(s)"
)
print(_calendar_gap.group_by("days").len().sort("days"))
_lagged.select(
    ["timestamp", "close", "momentum_signal", "signal_date", "earliest_execution"]
).drop_nulls().head(8)

# %% [markdown]
# ## 6. What the Macro Panel's Timestamps Mean
#
# Macro series arrive at different cadences and the panel stores them side by side on one daily
# grid, so a column has to be carried forward between its own releases. The usual advice is to
# forward-fill and never back-fill, and that advice is correct and not sufficient: it is safe
# only if the timestamp is the date the number was *published*. If the timestamp is the period
# the number *describes*, forward-filling from it hands a backtest a figure weeks before anyone
# outside the statistical agency had it.
#
# Which convention this panel uses is a question about the file, so the next cells ask it.

# %%
macro = load_macro()
macro = macro.rename({col: col.lower() for col in macro.columns})

date_col = "timestamp" if "timestamp" in macro.columns else "date"
_series_cols = [c for c in macro.columns if c != date_col]

freshness = (
    pl.DataFrame(
        {
            "series": _series_cols,
            "nulls": [macro[c].null_count() for c in _series_cols],
            "value_changes": [
                macro.select((pl.col(c) != pl.col(c).shift(1)).sum()).item() for c in _series_cols
            ],
        }
    )
    .with_columns(
        (pl.col("value_changes") / (macro.height / 365.25)).round(1).alias("changes_per_year")
    )
    .sort("changes_per_year")
)
print(f"Macro panel: {macro.height:,} rows, {macro[date_col].min()} to {macro[date_col].max()}")
print(f"Distinct dates: {macro[date_col].n_unique():,} (a row per calendar day, weekends included)")
freshness

# %% [markdown]
# Two things are already visible. Every column is complete: there are no nulls to forward-fill,
# because the panel arrives pre-filled. And the change counts sort the columns into cadences
# without anyone declaring them, from a few changes a year up to one per business day.
#
# The convention question is answered by *when* the changes happen.

# %%
_change_day = (
    pl.concat(
        [
            macro.select(
                pl.lit(c).alias("series"),
                pl.col(date_col).dt.day().alias("day_of_month"),
            ).filter(
                macro.select((pl.col(c) != pl.col(c).shift(1)).alias("x"))["x"]
                & macro.select(pl.col(c).shift(1).is_not_null().alias("y"))["y"]
            )
            for c in _series_cols
        ]
    )
    .group_by("series", "day_of_month")
    .len()
)
_first_of_month = (
    _change_day.group_by("series")
    .agg(
        pl.col("len").sum().alias("changes"),
        pl.col("len").filter(pl.col("day_of_month") == 1).sum().alias("on_the_first"),
    )
    .with_columns((pl.col("on_the_first") / pl.col("changes")).alias("share_on_the_first"))
    .sort("share_on_the_first", descending=True)
)
print("Share of each series' value changes that land on the first of a month:")
_first_of_month

# %% [markdown]
# The low-cadence series change their value on the first of the month, every time. That is not
# a release calendar: no statistical agency publishes on the first of every month, and none
# publishes a quarter's output on the first day of that quarter. It is the period stamp. FRED
# labels a monthly observation with the first day of the month it describes and a quarterly one
# with the first day of the quarter, and the panel has carried each value forward from there.
#
# So the unemployment rate for a month is readable in this panel from that month's first day,
# before the month it measures has finished, let alone been surveyed and published. The
# forward-fill in this panel is not PIT-safe, and adding a forward-fill of its own would change
# nothing, because there is nothing left to fill.

# %%
_period_days = {"monthly": 31, "quarterly": 92}


def stamp_to_availability(frame: pl.DataFrame, column: str, cadence: str) -> pl.DataFrame:
    """Move a period-stamped series to the date its first estimate could have been read.

    The stamp is the first day of the period, so the period ends roughly a period length later
    and the first estimate follows the declared release lag after that.
    """
    lag_days = _period_days[cadence] + (
        MONTHLY_RELEASE_LAG_DAYS if cadence == "monthly" else QUARTERLY_RELEASE_LAG_DAYS
    )
    return frame.select(
        (pl.col(date_col) + pl.duration(days=lag_days)).alias("available_from"),
        pl.col(column).alias(f"{column}_pit"),
    )


_demo = macro.select(date_col, MACRO_DEMO_SERIES)
_pit = stamp_to_availability(_demo, MACRO_DEMO_SERIES, "monthly")
_joined = (
    _demo.join(_pit, left_on=date_col, right_on="available_from", how="left")
    .with_columns(pl.col(f"{MACRO_DEMO_SERIES}_pit").forward_fill())
    .drop_nulls()
)
_differs = _joined.filter(pl.col(MACRO_DEMO_SERIES) != pl.col(f"{MACRO_DEMO_SERIES}_pit"))

print(
    f"Rows where the panel's {MACRO_DEMO_SERIES} and the release-lagged version disagree: "
    f"{_differs.height:,} of {_joined.height:,} "
    f"({100 * _differs.height / _joined.height:.0f}%)"
)
print(
    "Largest disagreement: "
    f"{(_differs[MACRO_DEMO_SERIES] - _differs[f'{MACRO_DEMO_SERIES}_pit']).abs().max():.1f} "
    "percentage points"
)
_joined.filter(pl.col(date_col).dt.year() == 2020).head(12)

# %% [markdown]
# The two columns disagree on three days in four, and the widest disagreement is larger than
# the whole range the series occupies in an ordinary decade: the unemployment rate moved by
# more than ten points inside two months in 2020, and the panel's column carries the move
# weeks before it was published.
#
# The lag applied here is a round number standing in for a release calendar, so the corrected
# column is not itself production-grade. What it establishes is the direction and the magnitude:
# the uncorrected column is early, on most days, by an amount that is large exactly when the
# data is interesting. A macro feature built straight off this panel is a feature that knows the
# recession before the recession was announced.
#
# The general rule, and the reason this notebook is here: **a timestamp is a claim about when
# something was knowable, and the file rarely says which claim it is making.** Determine it by
# measurement, once, and record the answer where the loader is.

# ## 7. PIT Validator Walkthrough
#
# The scale-invariant leakage test, a date-monotonicity check and a gap audit, wrapped so a
# whole feature matrix can be run through them in one call. The leakage threshold is the
# declared `LEAK_THRESHOLD`, and Section 4's table is what justifies putting it where it is:
# the clean features and the leaking ones are separated by most of the scale, so the cutoff
# sits in empty space rather than between two adjacent measurements.


# %%
class PITValidator:
    """Point-in-time validation for daily price-and-feature panels."""

    def __init__(self, df: pl.DataFrame, date_col: str = "timestamp"):
        self.df = df
        self.date_col = date_col
        self.violations: list[dict] = []

    def check_future_leakage(self, feature_col: str, price_col: str = "close") -> dict:
        corr = leakage_corr_scale_invariant(self.df, feature_col, price_col)
        severity = "HIGH" if abs(corr) > LEAK_THRESHOLD else "LOW"
        result = {
            "feature": feature_col,
            "scale_invariant_corr": round(corr, 4),
            "severity": severity,
            "violation": severity == "HIGH",
        }
        if result["violation"]:
            self.violations.append(result)
        return result

    def check_date_gaps(self, max_gap_days: int = MAX_GAP_DAYS) -> dict:
        gaps = (
            self.df.sort(self.date_col)
            .with_columns(pl.col(self.date_col).diff().dt.total_days().alias("gap_days"))
            .filter(pl.col("gap_days") > max_gap_days)
        )
        return {
            "rows": len(self.df),
            "gaps_above_threshold": len(gaps),
            "max_gap_days": int(gaps["gap_days"].max()) if len(gaps) > 0 else 0,
        }

    def check_monotonic_dates(self) -> dict:
        diffs = self.df[self.date_col].diff().drop_nulls().dt.total_days()
        is_sorted = bool((diffs >= 0).all())
        return {"is_monotonic": is_sorted, "violation": not is_sorted}


# %%
spy_features = spy.with_columns(
    pl.col("close").rolling_mean(window_size=MA_WINDOW).alias("ma_short_trailing"),
    pl.col("close").rolling_mean(window_size=MA_WINDOW_LONG).alias("ma_long_trailing"),
    pl.col("close").rolling_mean(window_size=MA_WINDOW, center=True).alias("ma_short_centered"),
    pl.col("close").shift(-1).alias("tomorrow_close"),
).drop_nulls()

validator = PITValidator(spy_features)
validation_table = pl.DataFrame(
    [
        validator.check_future_leakage(f)
        for f in [
            "ma_short_trailing",
            "ma_long_trailing",
            "ma_short_centered",
            "tomorrow_close",
        ]
    ]
)
validation_table

# %%
gap_check = validator.check_date_gaps()
mono_check = validator.check_monotonic_dates()
print(
    f"Date gaps over {MAX_GAP_DAYS} calendar days: "
    f"{gap_check['gaps_above_threshold']} (max {gap_check['max_gap_days']} days); "
    f"monotonic={mono_check['is_monotonic']}"
)
print(f"Features flagged as leaking: {len(validator.violations)} of {validation_table.height}")

# %% [markdown]
# The two constructed-from-the-future features are flagged and the two trailing averages are
# not, which is the ordering the section set out to produce. It is worth being clear about
# what that does and does not establish. The validator was run on features whose status was
# known in advance, so this is a test of the validator, not of the features. Run against a
# feature matrix nobody has audited, it will catch leakage that shows up as correlation with
# the next return, and will not catch leakage that does not: a feature using a future value
# of something *other* than the price it is scored against passes this check untouched.

# ## 8. Bitemporal GDP via FRED `vintage_date`
#
# Macro data is revised. The FRED API's `realtime_start` / `realtime_end`
# parameters return the values **as known at** a chosen historical date.
# `FREDProvider.fetch_ohlcv` exposes that as `vintage_date`.
#
# The cell below requires `FRED_API_KEY` (free signup); if the key is
# missing the fetch fails loudly rather than substituting a synthetic
# illustration.

# %%
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError(
        "FRED_API_KEY not set. Get a free key at "
        "https://fred.stlouisfed.org/docs/api/api_key.html and export it."
    )

provider = FREDProvider()

gdp_early = provider.fetch_ohlcv(
    "GDP",
    "2023-01-01",
    "2023-09-30",
    frequency="quarterly",
    vintage_date="2023-11-01",
)
gdp_late = provider.fetch_ohlcv(
    "GDP",
    "2023-01-01",
    "2023-09-30",
    frequency="quarterly",
    vintage_date="2024-06-01",
)
provider.close()

revisions = (
    gdp_early.select(["timestamp", pl.col("close").alias("vintage_2023_11_01")])
    .join(
        gdp_late.select(["timestamp", pl.col("close").alias("vintage_2024_06_01")]),
        on="timestamp",
    )
    .with_columns(
        (
            (pl.col("vintage_2024_06_01") - pl.col("vintage_2023_11_01"))
            / pl.col("vintage_2023_11_01")
            * 100
        )
        .round(3)
        .alias("revision_pct")
    )
)
revisions

# %% [markdown]
# The earlier quarters agree between the two vintages because they had already been revised
# to their current values by the first vintage date. The most recent quarter does not: what a
# query in late 2023 returned for it was the advance estimate, and by mid-2024 that number
# had moved.
#
# The revision is small in percentage terms, which is the ordinary case and the reason this
# is easy to skip. The problem is not the size of one revision. It is that a backtest reading
# the current value is reading a number that did not exist at the decision date, for every
# revised series it touches, and the direction of a revision is not noise: estimates are
# revised toward what actually happened.

# %% [markdown]
# %% [markdown]
# ## Key Takeaways
#
# 1. **Just under half of a centered window lies in the future, at any width.** The window
#    offsets are measured above rather than assumed, because the convention differs between
#    libraries and between odd and even widths. Since the share does not shrink with width, a
#    longer centered average leaks further ahead rather than less. The figure shows the
#    consequence and the printed distances measure it.
#
# 2. **Correlating a level against a return does not detect leakage.** Tomorrow's close, used
#    directly as a feature, scores near zero on that test, alongside a clean trailing average.
#    A feature matrix that passes it has learned nothing about itself.
#
# 3. **Putting both sides in return space makes the same test work**, and the reason the
#    threshold is placeable is that the clean and leaking cases end up separated by most of the
#    scale. A cutoff between two adjacent measurements would be a cutoff chosen to produce the
#    answer already known.
#
# 4. **A trading-day lag has to be applied by rows, not by dates.** One trading day is one to
#    five calendar days depending on where in the week and the holiday calendar it falls, and
#    the counts are printed above. Shifting the timestamp column by rows inherits the calendar
#    from the data.
#
# 5. **The macro panel's timestamps are period stamps, not release dates, and this is
#    measurable.** Every low-cadence series changes value on the first of a month, which no
#    release calendar does. The panel arrives already carried forward from those stamps, so it
#    is complete, a further forward-fill is a no-op, and reading a column gives a figure weeks
#    before it was published. Compared against a release-lagged version, the panel's own
#    column disagrees on three days in four, and at the widest by more than ten points of
#    unemployment - the spring of 2020, when the difference between knowing and not knowing
#    was the whole trade.
#
# 6. **Macro values are revised, so the current value is not the value that was available.**
#    Vintage queries return what was knowable at a date. Revisions move estimates toward what
#    actually happened, which is precisely the direction that flatters a backtest.
#
# 7. **Detection is a backstop, not a substitute for construction.** Every check here was run
#    against features whose status was known in advance, which tests the check. On an unaudited
#    matrix these checks catch the leakage that correlates with the scored return and miss the
#    leakage that does not.
#
# **Next**: `15_survivorship_bias_detection` adds the universe-membership dimension to the
# temporal correctness shown here.
