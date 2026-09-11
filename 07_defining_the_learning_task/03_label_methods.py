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
# **Docker image**: `ml4t`

# %% [markdown]
# # Label Engineering Methods
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.2 - Label Engineering
#
# ## Purpose
#
# This notebook demonstrates **all major labeling methods** for ML-based trading
# strategies, with practical examples on real ETF data. It serves as the canonical
# reference for choosing and configuring labels across all modeling chapters.
#
# ## Learning Objectives
#
# 1. Understand fixed-horizon vs path-dependent labels
# 2. Compare time-series vs **cross-sectional** percentile approaches
# 3. Implement triple-barrier with fixed and ATR-based thresholds
# 4. Visualize barrier mechanics with price path examples
# 5. Understand anchor alignment (close-to-close vs next-open)
#
# ## Prerequisites
#
# - `01_data_quality_diagnostics` - establishes the ETF coverage assumptions used here.
# - Familiarity with leakage-aware splitting (Chapter 6, *Data leakage and estimation bias*)
#   and forward-return semantics.
# - Polars DataFrame manipulation; basic statistics (t-statistics, percentiles).
#
# ## Data Contract
#
# - **Input**: Real ETF OHLCV from data loaders (SPY for single-asset, full universe for cross-sectional)
# - **Output**: Example labels for teaching. Each case study builds its production labels in its
#   own `02_labels` notebook, reading the horizon from that case study's `config/setup.yaml`
#   through `utils.artifact_specs.resolve_label_horizon`.

# %%
"""Label Methods - fixed-horizon, cross-sectional, and event-driven labeling for supervised learning."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.engineer.config.labeling import LabelingConfig
from ml4t.engineer.labeling import (
    atr_triple_barrier_labels,
    calculate_label_uniqueness,
    compute_bet_size,
    fixed_time_horizon_labels,
    meta_labels,
    rolling_percentile_binary_labels,
    sequential_bootstrap,
    trend_scanning_labels,
    triple_barrier_labels,
)
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import (  # importing also registers the ML4T Plotly template
    COLORS,
    show_plotly_with_alt,
)

# %% tags=["parameters"]
SEED = 42
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# %%
set_global_seeds(SEED)


# %% [markdown]
# ## Helper Functions
#
# Robust label column discovery to avoid brittle hardcoded column names.


# %%
def first_col_matching_any(df: pl.DataFrame, needles: Sequence[str]) -> str:
    """
    Return the first column whose name contains any of the substrings in needles.
    Raises ValueError if no match is found.
    """
    lowered = [(c, c.lower()) for c in df.columns]
    for c, c_low in lowered:
        for n in needles:
            if n.lower() in c_low:
                return c
    raise ValueError(f"No column found matching: {needles}")


# %% [markdown]
# ## Load Sample Data
#
# We use the ETF universe for demonstrations. SPY serves as the single-asset
# example; the full universe enables cross-sectional analysis.

# %%
# Load ETF universe
etf = load_etfs()

# Filter date range
date_filter = (pl.col("timestamp") >= datetime.strptime(START_DATE, "%Y-%m-%d")) & (
    pl.col("timestamp") <= datetime.strptime(END_DATE, "%Y-%m-%d")
)

etf_filtered = (
    etf.filter(date_filter)
    .sort(["symbol", "timestamp"])
    .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
)

# SPY for single-asset demos
spy = etf_filtered.filter(pl.col("symbol") == "SPY").sort("timestamp")

print(f"ETF universe: {etf_filtered['symbol'].n_unique()} symbols")
print(f"SPY data: {len(spy):,} days from {spy['timestamp'].min()} to {spy['timestamp'].max()}")
spy.head()

# %% [markdown]
# ## Fixed Time Horizon Labels
#
# The simplest approach: compute forward returns over a fixed window.
# This is the workhorse of factor-based ML strategies.
#
# ### Methods
#
# | Method | Description | Use Case |
# |--------|-------------|----------|
# | `"returns"` | Raw percentage return | Regression targets |
# | `"binary"` | +1 if return > 0, else -1 | Classification |
# | `"log_returns"` | Log return | Regression with better properties |

# %%
# 21 trading days is a common "one month" convention in daily data
HORIZON = 21

labels_returns = fixed_time_horizon_labels(
    spy,
    horizon=HORIZON,
    method="returns",
    price_col="close",
)

# Discover the produced label column robustly
fh_label_col = first_col_matching_any(labels_returns, ["label_return", "label"])
print(f"Fixed Horizon Labels (horizon={HORIZON}):")
print(f"  Column added: {fh_label_col}")
labels_returns.select(["timestamp", "close", fh_label_col]).head(10)

# %% [markdown]
# ### Return vs binary comparison
#
# Both methods use the same forward window but produce different target types:
# continuous returns for regression, binary direction for classification.

# %%
labels_binary = fixed_time_horizon_labels(
    spy,
    horizon=HORIZON,
    method="binary",
    price_col="close",
)

binary_label_col = first_col_matching_any(labels_binary, ["direction", "label"])

print("Return distribution:")
display(labels_returns[fh_label_col].describe())
print("Binary label distribution:")
display(labels_binary.group_by(binary_label_col).len().sort(binary_label_col))

# %% [markdown]
# ## Anchor Alignment
#
# **Critical concept**: The anchor point determines when returns are measured.
# Different anchors produce different labels even with the same horizon.
#
# - **Close-to-close**: Decision at close, measure return from close to H-day close
# - **Next-open-to-open**: Decision at close, execute at next open, measure from there
#
# This is one of the most common sources of subtle lookahead bias.

# %%
# Compute both anchor alignments
spy_anchors = spy.with_columns(
    [
        # Close-to-close: standard approach
        (pl.col("close").shift(-HORIZON) / pl.col("close") - 1).alias("ret_close_to_close"),
        # Next-open-to-open: decision at close(t), execute at open(t+1), exit at open(t+horizon+1)
        # Holding for `horizon` trading days means exit is horizon+1 bars from decision
        (pl.col("open").shift(-(HORIZON + 1)) / pl.col("open").shift(-1) - 1).alias(
            "ret_next_open_to_open"
        ),
    ]
).drop_nulls()

# Compute the difference
spy_anchors = spy_anchors.with_columns(
    [(pl.col("ret_close_to_close") - pl.col("ret_next_open_to_open")).alias("anchor_diff")]
)

print(f"Mean close-to-close return:    {spy_anchors['ret_close_to_close'].mean():.4f}")
print(f"Mean next-open-to-open return: {spy_anchors['ret_next_open_to_open'].mean():.4f}")
print(f"Mean difference:               {spy_anchors['anchor_diff'].mean():.4f}")
print(f"Std difference:                {spy_anchors['anchor_diff'].std():.4f}")

# %%
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "Anchor Difference Over Time",
        "Distribution of Anchor Differences",
    ],
    row_heights=[0.6, 0.4],
    vertical_spacing=0.22,
)

# Raw daily differences (light) with 63-day rolling mean overlay
fig.add_trace(
    go.Scatter(
        x=spy_anchors["timestamp"].to_list(),
        y=spy_anchors["anchor_diff"].to_numpy(),
        mode="lines",
        name="Daily",
        line=dict(width=0.3, color="rgba(100,100,100,0.3)"),
    ),
    row=1,
    col=1,
)

# Rolling mean to show structural pattern
rolling_mean = spy_anchors["anchor_diff"].rolling_mean(63)
fig.add_trace(
    go.Scatter(
        x=spy_anchors["timestamp"].to_list(),
        y=rolling_mean.to_numpy(),
        mode="lines",
        name="63-day MA",
        line=dict(width=1.5, color=COLORS["amber"]),
    ),
    row=1,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.8, row=1, col=1)

# Histogram of differences
fig.add_trace(
    go.Histogram(
        x=spy_anchors["anchor_diff"].to_numpy(),
        nbinsx=50,
        name="Distribution",
        showlegend=False,
        marker_color=COLORS["blue"],
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Return difference", row=1, col=1)
fig.update_xaxes(title_text="Return difference", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_layout(
    height=550,
    title_text=f"Close-to-close minus next-open-to-open return ({HORIZON}-day)",
    font=dict(size=12),
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two stacked panels of the difference between the close-to-close and "
        "next-open-to-open 21-day SPY return, 2015 to 2024. The top panel plots the "
        "daily difference as a thin grey line that oscillates within roughly plus or "
        "minus two percent, widening to plus or minus ten percent around the March 2020 "
        "crash, with an amber 63-day moving average that stays close to the zero line, "
        "its largest excursion in either direction too shallow to reach one percent. The "
        "bottom panel is a histogram of the same differences: a tall "
        "narrow peak centred on zero, roughly symmetric, with most observations inside "
        "plus or minus two percent and thin tails reaching eight percent."
    ),
)

# %%
# Example timestamps showing anchor shift
spy_anchors.select(
    ["timestamp", "close", "open", "ret_close_to_close", "ret_next_open_to_open", "anchor_diff"]
).head(10)

# %% [markdown]
# The two anchors agree on average and disagree on every individual trade. The mean
# difference printed above is indistinguishable from zero, while its standard deviation
# is of the same order as a typical daily move - so the choice of anchor is invisible in
# a summary statistic and decisive for any single label. A model trained on close-to-close
# labels is being scored on a price its strategy could not have transacted at.
#
# For an end-of-day signal executed at the next open, anchor the label at the next open.

# %% [markdown]
# ## Time-Series Percentile Labels
#
# Labels are relative to recent history for a **single instrument**,
# making them adaptive to volatility regimes.

# %%
# Binary percentile: Is return in top 25%?
labels_ts_pct = rolling_percentile_binary_labels(
    spy,
    horizon=HORIZON,
    percentile=75,  # Top 25%
    direction="long",
    lookback_window=252,  # 1 year rolling window
    price_col="close",
)

# Find the label column robustly
ts_pct_label_col = first_col_matching_any(labels_ts_pct, ["label"])
print("Time-Series Percentile Labels (p75):")
print(f"  Column added: {ts_pct_label_col}")

# Show distribution
print("Label Distribution:")
display(labels_ts_pct.group_by(ts_pct_label_col).len().sort(ts_pct_label_col))

# %%
threshold_col = [c for c in labels_ts_pct.columns if "threshold" in c.lower()]
if threshold_col:
    fig = px.line(
        labels_ts_pct.to_pandas(),
        x="timestamp",
        y=threshold_col[0],
        title=f"Rolling 75th percentile of trailing {HORIZON}-day SPY returns",
    )
    fig.update_layout(height=350, xaxis_title="Date", yaxis_title="Return threshold")
    show_plotly_with_alt(
        fig,
        alt=(
            "A single navy line of the rolling 75th-percentile threshold for SPY 21-day "
            "returns from 2015 to 2024, computed over a 252-day lookback. The line starts "
            "near four percent, falls to below two percent through 2016, drifts up through "
            "2018 and 2019, and peaks near five and a half percent in early 2021. It falls "
            "back to below three percent in mid-2022 before rising again to around five "
            "percent in 2023 and easing to four percent by the end of 2024."
        ),
    )

# %% [markdown]
# The cell below correlates the threshold the labeler actually produced - not a
# re-derivation of it - against two quantities measured strictly backwards from each date,
# so nothing in the comparison sees the future the threshold is trying to anticipate.

# %% tags=["results"]
if threshold_col:
    _diag = (
        labels_ts_pct.select("timestamp", "close", threshold_col[0])
        .with_columns(
            daily=(pl.col("close") / pl.col("close").shift(1) - 1),
            completed=(pl.col("close") / pl.col("close").shift(HORIZON) - 1),
        )
        .with_columns(
            trailing_vol=pl.col("daily").rolling_std(252) * np.sqrt(252),
            trailing_drift=pl.col("completed").rolling_mean(252),
        )
        .drop_nulls()
    )
    print(f"Rolling threshold ({threshold_col[0]}), correlation against:")
    print(
        "  trailing 1-year realized volatility:          "
        f"{_diag.select(pl.corr(threshold_col[0], 'trailing_vol')).item():.2f}"
    )
    print(
        f"  trailing 1-year mean completed {HORIZON}-day return: "
        f"{_diag.select(pl.corr(threshold_col[0], 'trailing_drift')).item():.2f}"
    )

# %% [markdown]
# The threshold is not a fixed return; it is whatever the top quartile of the last year
# looked like. It moves mostly with trailing volatility, which is the property the method
# is usually sold on, but the correlation above is well short of one because it also moves
# with trailing drift: a year of steady gains raises the bar for "top quartile" without
# any change in dispersion. Both effects raise the threshold, and a reader cannot tell
# them apart from the line alone.

# %% [markdown]
# ## Cross-Sectional Percentile Labels
#
# **The most natural use of percentile labels**: rank assets within the universe
# at each decision time, then label top/bottom quantiles.
#
# This is the standard approach for equity and ETF rotation strategies.

# %%
# Compute forward returns and cross-sectional rank for the entire ETF universe
etf_with_fwd = etf_filtered.with_columns(
    [(pl.col("close").shift(-HORIZON) / pl.col("close") - 1).over("symbol").alias("fwd_return")]
).drop_nulls(subset=["fwd_return"])

etf_cs = etf_with_fwd.with_columns(
    [
        pl.col("fwd_return").rank(method="average").over("timestamp").alias("rank"),
        pl.col("fwd_return").count().over("timestamp").alias("n_symbols"),
    ]
).with_columns(
    [
        # Percentile rank: 0-100 scale (guard against single-symbol dates)
        pl.when(pl.col("n_symbols") > 1)
        .then((pl.col("rank") - 1) / (pl.col("n_symbols") - 1) * 100)
        .otherwise(None)
        .alias("pct_rank")
    ]
)

print(f"Cross-sectional ranking: {len(etf_cs):,} asset-date observations")

# %% [markdown]
# Cross-sectional percentile labels rank assets at each decision time $t$.
# This is inherently point-in-time: the ranking at $t$ uses only returns
# realized at $t$, so no future information leaks into label construction.

# %%
# Assign labels: top quintile = +1, bottom quintile = -1, else 0
QUINTILE_THRESHOLD = 20  # Top/bottom quintile
etf_cs = etf_cs.with_columns(
    [
        pl.when(pl.col("pct_rank") >= (100 - QUINTILE_THRESHOLD))
        .then(pl.lit(1))
        .when(pl.col("pct_rank") <= QUINTILE_THRESHOLD)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("cs_label")
    ]
)

print(f"Cross-Sectional Labels ({HORIZON}d horizon, {QUINTILE_THRESHOLD}th percentile cutoffs):")
print(f"  Total observations: {len(etf_cs):,}")
display(etf_cs.group_by("cs_label").len().sort("cs_label"))

# %% [markdown]
# Counts per class move with the size of the universe on each date; proportions do not,
# because the cut is a rank. Converting to proportions before plotting is what makes the
# construction visible rather than the coverage.

# %%
label_by_date = (
    etf_cs.group_by(["timestamp", "cs_label"])
    .len()
    .pivot(on="cs_label", index="timestamp", values="len")
    .sort("timestamp")
)

# Convert counts to proportions
count_cols = [c for c in label_by_date.columns if c != "timestamp"]
if count_cols:
    label_by_date = label_by_date.with_columns([pl.col(c).fill_null(0) for c in count_cols])
    label_by_date = label_by_date.with_columns(
        pl.sum_horizontal([pl.col(c) for c in count_cols]).alias("_total")
    )
    label_by_date = label_by_date.with_columns(
        [
            pl.when(pl.col("_total") > 0)
            .then(pl.col(c) / pl.col("_total"))
            .otherwise(None)
            .alias(c)
            for c in count_cols
        ]
    )

# %%
# Show class proportions over time
fig = go.Figure()
for label in [-1, 0, 1]:
    col_name = str(label)
    if col_name in label_by_date.columns:
        fig.add_trace(
            go.Scatter(
                x=label_by_date["timestamp"].to_list(),
                y=label_by_date[col_name].to_numpy(),
                mode="lines",
                name=f"Label {label}",
            )
        )

fig.update_layout(
    height=400,
    title=f"Cross-sectional label proportions by date (top/bottom {QUINTILE_THRESHOLD}%)",
    xaxis_title="Date",
    yaxis_title="Proportion",
)
show_plotly_with_alt(
    fig,
    alt=(
        "Three flat lines across 2015 to 2024 showing the share of ETFs in each "
        "cross-sectional label class on each date. The neutral class sits just under "
        "0.6 and the two extreme classes sit just above 0.2, where they coincide exactly "
        "and plot as one line. The lines are horizontal apart from a few one-pixel steps "
        "around 2016, 2018 and 2019 where the number of ETFs with a forward return "
        "changes and the quintile cut lands on a different count."
    ),
)

# %%
# Show cross-sectional threshold values over time
cs_thresholds = (
    etf_with_fwd.group_by("timestamp")
    .agg(
        [
            pl.col("fwd_return").quantile(QUINTILE_THRESHOLD / 100).alias("bottom_threshold"),
            pl.col("fwd_return").quantile(1 - QUINTILE_THRESHOLD / 100).alias("top_threshold"),
        ]
    )
    .sort("timestamp")
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=cs_thresholds["timestamp"].to_list(),
        y=cs_thresholds["top_threshold"].to_numpy(),
        mode="lines",
        name="Top 20% Threshold",
        line=dict(color=COLORS["positive"]),
    )
)
fig.add_trace(
    go.Scatter(
        x=cs_thresholds["timestamp"].to_list(),
        y=cs_thresholds["bottom_threshold"].to_numpy(),
        mode="lines",
        name="Bottom 20% Threshold",
        line=dict(color=COLORS["negative"]),
    )
)
fig.update_layout(
    height=400,
    title=f"Cross-sectional cut points for the top and bottom {QUINTILE_THRESHOLD}%",
    xaxis_title="Date",
    yaxis_title=f"{HORIZON}-day return threshold",
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two lines from 2015 to 2024 tracing the cross-sectional cut points that define "
        "the top and bottom ETF quintiles on each date. The green top-quintile line sits "
        "mostly between zero and plus ten percent and the red bottom-quintile line mostly "
        "between zero and minus ten percent, so the band between them is usually about "
        "ten percentage points wide. The band narrows to a few percentage points in the "
        "calm stretches of 2017 and 2024, and blows out in March 2020 to a single spike "
        "reaching plus twenty-six percent above and minus forty percent below."
    ),
)

# %% [markdown]
# The two figures make the same point from opposite sides. Class proportions are fixed by
# construction, so the first figure is flat by design and carries no information about the
# market; the small steps in it are changes in the number of ETFs, not changes in returns.
# Everything that varies has been pushed into the second figure: the return a fund needs
# to reach the top quintile is a few percent in a quiet month and tens of percent in a
# dislocation. A model trained on these labels sees a constant class balance and a target
# whose economic meaning changes underneath it.

# %% [markdown]
# ## Triple-Barrier Labels
#
# Path-dependent labeling that captures realistic trade outcomes:
# - **Upper barrier**: Take profit hit → +1
# - **Lower barrier**: Stop loss hit → -1
# - **Time barrier**: Neither hit → label based on final return
#
# This method is from De Prado's *Advances in Financial Machine Learning*.
#
# **What the barriers are tested against.** `triple_barrier_labels` takes `high_col`,
# `low_col` and `open_col` alongside `price_col`. Supplied, it detects a touch on the bar's
# range and executes a gap-through at the open. Omitted - as here, to keep the mechanics
# visible on a single series - the bar reduces to its close: a barrier is touched only if
# a *close* crosses it, and the trade is then booked at the barrier price rather than at the
# close that crossed it. Both simplifications flatter the label set, and the section below
# measures by how much.

# %%
# Fixed percentage barriers: 2% take profit, 1% stop loss
config = LabelingConfig.triple_barrier(
    upper_barrier=0.02,  # 2% take profit
    lower_barrier=0.01,  # 1% stop loss
    max_holding_period=20,  # 20 days max
    side=1,  # Long positions only
)

labels_tb = triple_barrier_labels(
    spy,
    config=config,
    price_col="close",
    timestamp_col="timestamp",
    calculate_uniqueness=True,  # Compute sample weights
)

print("Triple-Barrier Labels (Fixed %):")
print("Label Distribution:")
display(labels_tb.group_by("label").len().sort("label"))

print("Barrier Hit Distribution:")
display(labels_tb.group_by("barrier_hit").len().sort("barrier_hit"))

# %% [markdown]
# ### Triple-Barrier Path Visualization
#
# Each figure below is one trade. The navy segment is the window in which the position was
# open, the copper cross is the bar the label resolves on and the price it books, and the
# dotted grey continuation is the rest of the twenty-day window - drawn because a reader
# needs to see what the label deliberately ignores. The lower-barrier example is the one to
# study: the label records a one percent loss on a path that went on to fall twenty-eight
# percent, which is the whole point of a stop and the whole risk of trusting the label as a
# description of the market rather than of the trade.

# %%
# Find examples of each barrier hit type
tb_with_price = labels_tb.join(
    spy.select(["timestamp", "close"]), on="timestamp", how="left"
).with_row_index("row_idx")


def plot_triple_barrier_example(
    df: pl.DataFrame, entry_idx: int, config: LabelingConfig, title: str
) -> go.Figure:
    """Plot a single triple-barrier trade example with barriers and the exit overlaid."""
    # Get entry point
    entry_row = df.row(entry_idx, named=True)
    entry_price = entry_row["close"]
    entry_time = entry_row["timestamp"]
    exit_bars = entry_row["label_bars"]

    # Calculate barrier levels
    upper_level = entry_price * (1 + config.upper_barrier)
    lower_level = entry_price * (1 - config.lower_barrier)

    # Get the forward price path
    forward_rows = df.filter(pl.col("timestamp") >= entry_time).head(config.max_holding_period + 1)
    times = forward_rows["timestamp"].to_list()
    prices = forward_rows["close"].to_numpy()

    fig = go.Figure()

    # Bars after the exit are not part of the trade; draw them as context, not as price.
    if exit_bars is not None and exit_bars < len(times) - 1:
        fig.add_trace(
            go.Scatter(
                x=times[int(exit_bars) :],
                y=prices[int(exit_bars) :],
                mode="lines",
                name="After exit",
                line=dict(color=COLORS["neutral"], width=1, dash="dot"),
                opacity=0.5,
            )
        )
    held = int(exit_bars) + 1 if exit_bars is not None else len(times)

    # Price path while the position is open
    fig.add_trace(
        go.Scatter(
            x=times[:held],
            y=prices[:held],
            mode="lines+markers",
            name="Price (position open)",
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=4),
        )
    )

    # Exit point
    if exit_bars is not None and int(exit_bars) < len(times):
        fig.add_trace(
            go.Scatter(
                x=[times[int(exit_bars)]],
                y=[entry_row["label_price"]],
                mode="markers",
                name="Exit (booked)",
                marker=dict(color=COLORS["copper"], size=12, symbol="x"),
            )
        )

    # Entry point
    fig.add_trace(
        go.Scatter(
            x=[entry_time],
            y=[entry_price],
            mode="markers",
            name="Entry",
            marker=dict(color=COLORS["amber"], size=12, symbol="star"),
        )
    )

    # Upper barrier (horizontal line) - label on the left to clear the time-barrier text
    fig.add_hline(
        y=upper_level,
        line_dash="dash",
        line_color=COLORS["positive"],
        annotation_text=f"TP: {upper_level:.2f} (+{config.upper_barrier:.1%})",
        annotation_position="top left",
    )

    # Lower barrier (horizontal line)
    fig.add_hline(
        y=lower_level,
        line_dash="dash",
        line_color=COLORS["negative"],
        annotation_text=f"SL: {lower_level:.2f} (-{config.lower_barrier:.1%})",
        annotation_position="bottom left",
    )

    # Time barrier (vertical line at end)
    # Note: Use add_shape instead of add_vline with annotation to avoid Plotly datetime bug
    time_barrier = forward_rows["timestamp"].to_list()[-1] if len(forward_rows) > 0 else entry_time
    fig.add_shape(
        type="line",
        x0=time_barrier,
        x1=time_barrier,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=COLORS["neutral"], dash="dot"),
    )
    fig.add_annotation(
        x=time_barrier,
        y=1,
        yref="paper",
        text="Time Barrier",
        showarrow=False,
        yshift=10,
    )

    fig.update_layout(
        height=350,
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
    )
    return fig


# Find examples of each barrier type
if "barrier_hit" in labels_tb.columns:
    # Get indices for different barrier hits
    examples = []

    for barrier_type in ["upper", "lower", "time"]:
        matches = tb_with_price.filter(
            (pl.col("barrier_hit") == barrier_type) & (pl.col("close").is_not_null())
        )
        if len(matches) > 10:
            # Pick an example from the middle of the dataset
            idx = len(matches) // 2
            row_idx = matches["row_idx"][idx]
            examples.append((barrier_type, row_idx))

    alts = {
        "upper": (
            "One SPY trade entered at the close of 27 April 2020 near 264, with a dashed "
            "green take-profit line at 269.15 and a dashed red stop line at 261.23. The "
            "navy price line rises across two bars to the take-profit, where a copper cross "
            "marks the booked exit at 269.15. The remaining eighteen bars of the holding "
            "window are drawn as a faint dotted grey line that falls below the stop level "
            "in early May, recovers, and ends near 275 - none of it part of the trade, "
            "though a position still open would have been stopped out on the way."
        ),
        "lower": (
            "One SPY trade entered at the close of 25 February 2020 near 286, with a dashed "
            "green take-profit line at 291.44 and a dashed red stop line at 282.86. The navy "
            "price line falls across two bars through the stop, where a copper cross marks "
            "the booked exit at the barrier price. The rest of the holding window is a faint "
            "dotted grey line collapsing to about 205 by late March, a twenty-eight percent "
            "decline that the label does not see because the position closed on the second bar."
        ),
        "time": (
            "One SPY trade entered at the close of 1 June 2017 near 211, with a dashed green "
            "take-profit line at 215.25 well above the path and a dashed red stop line at "
            "208.92 well below it. The navy price line wanders between 210 and 213.2 for the "
            "whole twenty-bar window without touching either barrier, and a copper cross "
            "marks the exit at the final bar, where the vertical dotted time barrier sits. "
            "The closest approach is to the stop, roughly a point away near the end."
        ),
    }

    # Plot examples
    for barrier_type, idx in examples[:3]:  # Limit to 3 examples
        if idx < len(tb_with_price):
            fig = plot_triple_barrier_example(
                tb_with_price,
                idx,
                config,
                f"SPY triple-barrier trade closed by the {barrier_type} barrier",
            )
            show_plotly_with_alt(fig, alt=alts[barrier_type])

# %% [markdown]
# ### ATR-Based Barriers
#
# Volatility-adjusted barriers adapt to market conditions:
# - Low volatility → Tighter barriers (capture smaller moves)
# - High volatility → Wider barriers (avoid whipsaws)

# %%
# Compute ATR and convert to percentage-of-price barriers
atr_period = 14
atr_tp_multiple = 1.0  # 1x ATR take profit
atr_sl_multiple = 0.5  # 0.5x ATR stop loss (tighter asymmetric)

spy_atr = (
    spy.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs(),
        ).alias("true_range")
    )
    .with_columns(pl.col("true_range").rolling_mean(atr_period).alias("atr_dollar"))
    .with_columns(
        # Express barriers as fraction of entry price so they match the return-based engine
        (atr_tp_multiple * pl.col("atr_dollar") / pl.col("close")).alias("upper_barrier_pct"),
        (atr_sl_multiple * pl.col("atr_dollar") / pl.col("close")).alias("lower_barrier_pct"),
    )
    .drop_nulls()
)

# %%
# Use triple_barrier_labels with dynamic per-row percentage barriers
atr_config = LabelingConfig.triple_barrier(
    upper_barrier="upper_barrier_pct",
    lower_barrier="lower_barrier_pct",
    max_holding_period=20,
    side=1,
)
labels_atr = triple_barrier_labels(
    spy_atr,
    config=atr_config,
    price_col="close",
    timestamp_col="timestamp",
)

print("ATR Triple-Barrier Labels:")
print(f"  ATR period: {atr_period}, TP: {atr_tp_multiple}x ATR, SL: {atr_sl_multiple}x ATR")
print("Label Distribution:")
display(labels_atr.group_by("label").len().sort("label"))

if "barrier_hit" in labels_atr.columns:
    print("Barrier Hit Distribution:")
    display(labels_atr.group_by("barrier_hit").len().sort("barrier_hit"))

print("ATR as % of Close:")
display(spy_atr["upper_barrier_pct"].describe())

# %% [markdown]
# ### Sample Weights from Uniqueness
#
# Overlapping labels create mechanical dependence: high-concurrency periods
# dominate training loss. Weighting by uniqueness prevents these periods
# from overwhelming the model. De Prado introduces **uniqueness-based
# sample weights** where more unique samples (less overlap) get higher weights.

# %%
if "sample_weight" in labels_tb.columns:
    print("Sample Weight Statistics:")
    display(labels_tb["sample_weight"].describe())

    fig = px.histogram(
        labels_tb.filter(pl.col("sample_weight").is_not_null()).to_pandas(),
        x="sample_weight",
        nbins=50,
        title="Uniqueness-based sample weights, SPY triple-barrier labels",
    )
    fig.update_layout(height=350, xaxis_title="Sample weight", yaxis_title="Count")
    show_plotly_with_alt(
        fig,
        alt=(
            "A right-skewed histogram of uniqueness-based sample weights for the SPY "
            "triple-barrier labels. The weights are normalised to average one. The bulk "
            "sits between zero and one and a half with a mode near 0.4, and a long thin "
            "tail runs out past four. A minority of near-isolated labels therefore carry "
            "several times the weight of a label drawn from a crowded stretch."
        ),
    )

# %% [markdown]
# ### Rich Triple-Barrier Output
#
# Unlike simple forward-return labels, triple-barrier output includes the **full
# trade outcome**. This is critical for MFE/MAE analysis (NB04) and position
# sizing (Ch20).

# %%
# Display all output columns from triple_barrier_labels
output_cols = [
    c for c in labels_tb.columns if c.startswith("label") or c in ("barrier_hit", "sample_weight")
]
print("Triple-Barrier Output Columns:")
for col in output_cols:
    dtype = labels_tb[col].dtype
    print(f"  {col:<20} {str(dtype):<12} - {labels_tb[col].drop_nulls().head(1).to_list()}")

# %%
# Summary table: mean return and median holding period by barrier type
barrier_summary = (
    labels_tb.filter(pl.col("barrier_hit").is_not_null())
    .group_by("barrier_hit")
    .agg(
        count=pl.len(),
        mean_return=pl.col("label_return").mean(),
        median_bars=pl.col("label_bars").median(),
    )
    .sort("barrier_hit")
)
print("Trade Outcomes by Barrier Type:")
display(barrier_summary)

# %%
fig = px.histogram(
    labels_tb.filter(pl.col("label_return").is_not_null()).to_pandas(),
    x="label_return",
    color="barrier_hit",
    nbins=50,
    barmode="overlay",
    opacity=0.7,
    title="Booked label return by barrier type, SPY",
)
fig.update_layout(height=350, xaxis_title="Booked label return", yaxis_title="Count")
show_plotly_with_alt(
    fig,
    alt=(
        "A histogram of the booked label return for every SPY triple-barrier trade, "
        "coloured by which barrier closed it. Almost all the mass is two narrow spikes: "
        "about 1,250 lower-barrier trades at exactly minus one percent and about 1,100 "
        "upper-barrier trades at exactly plus two percent. Between them the axis is empty "
        "apart from a sliver of roughly 160 time-barrier trades scattered near plus one "
        "percent. Nothing lies beyond either barrier."
    ),
)

# %% [markdown]
# The two spikes sit exactly on the barriers because a close-only test books the barrier
# price, not the close that crossed it. That is an assumption, and it is measurable: the
# cell below compares what each trade was booked at against the close that actually
# triggered it.

# %% tags=["results"]
_triggered = (
    labels_tb.filter(pl.col("barrier_hit").is_in(["upper", "lower"]))
    .join(
        spy.select(pl.col("timestamp").alias("label_time"), pl.col("close").alias("exit_close")),
        on="label_time",
        how="left",
    )
    .with_columns(realized=(pl.col("exit_close") / pl.col("close") - 1))
)
print("Booked at the barrier vs realized at the close that crossed it:")
for hit in ("upper", "lower"):
    part = _triggered.filter(pl.col("barrier_hit") == hit)
    print(
        f"  {hit:<6} n={len(part):>5}  booked {part['label_return'].mean():+.4f}"
        f"   realized {part['realized'].mean():+.4f}"
        f"   gap {(part['realized'] - part['label_return']).mean():+.4f}"
    )
print(
    f"  all barrier exits: booked {_triggered['label_return'].mean():+.4f}"
    f"   realized {_triggered['realized'].mean():+.4f}"
)

# %% [markdown]
# The gap runs the wrong way on both sides, and further on the lower side than the upper
# one. An upper-barrier exit is booked at the take-profit when the close that triggered it
# was already above it, so the label gives away part of the gain; a lower-barrier exit is
# booked at the stop when the close that triggered it was well below, so the label hides
# part of the loss. Averaged over every barrier exit the label set is better than what
# those closes would have paid, and the bias is asymmetric rather than a wash.
#
# Read that as barrier-price against trigger-close accounting on a fixed set of exits, and
# nothing more. It is **not** a measurement of what OHLC detection is worth: supplying
# `high_col`, `low_col` and `open_col` changes which bar exits and which barrier is hit, not
# only the price each exit is booked at, and an intrabar touch that does not gap still
# executes at the barrier there too. Sizing that setting needs a second labeling run to
# compare against, not a repricing of this one.

# %% [markdown]
# ### Sequential Bootstrap
#
# Overlapping labels create sample dependence. The **sequential bootstrap**
# (De Prado, AFML Ch4) generates bootstrap indices that respect label
# uniqueness - favoring samples with less concurrent overlap.

# %%
# Extract label lifetimes as index ranges for the uniqueness calculation
tb_valid = labels_tb.filter(pl.col("label_bars").is_not_null()).with_row_index("idx")

starts = tb_valid["idx"].to_numpy().astype(np.int64)
ends = (starts + tb_valid["label_bars"].to_numpy().astype(np.int64)).clip(max=len(tb_valid) - 1)

# Compute uniqueness from indices
uniqueness = calculate_label_uniqueness(starts, ends, n_bars=len(tb_valid))

print(f"Label Uniqueness: mean={uniqueness.mean():.3f}, std={uniqueness.std():.3f}")
print(f"  Range: [{uniqueness.min():.3f}, {uniqueness.max():.3f}]")

# %%
# Sequential bootstrap vs naive random sampling
n_draws = min(len(starts), 500)
seq_indices = sequential_bootstrap(starts, ends, n_draws=n_draws, random_state=SEED)
naive_indices = np.random.default_rng(SEED).choice(len(starts), size=n_draws, replace=True)

# Compare uniqueness of selected samples
seq_uniqueness = uniqueness[seq_indices]
naive_uniqueness = uniqueness[naive_indices]

print("Bootstrap Comparison:")
print(f"  Sequential mean uniqueness: {seq_uniqueness.mean():.3f}")
print(f"  Naive mean uniqueness:      {naive_uniqueness.mean():.3f}")
print(f"  Improvement:                {(seq_uniqueness.mean() / naive_uniqueness.mean() - 1):.1%}")

# %% [markdown]
# Overlaid on one axis with one binning. Side-by-side panels rescale independently, which
# would let two nearly identical distributions look like a finding.

# %%
bins = dict(start=0.0, end=float(uniqueness.max()), size=float(uniqueness.max()) / 30)
fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=naive_uniqueness,
        xbins=bins,
        name="Naive",
        marker_color=COLORS["neutral"],
        opacity=0.6,
    )
)
fig.add_trace(
    go.Histogram(
        x=seq_uniqueness,
        xbins=bins,
        name="Sequential",
        marker_color=COLORS["amber"],
        opacity=0.6,
    )
)
for values, color in ((naive_uniqueness, COLORS["neutral"]), (seq_uniqueness, COLORS["amber"])):
    fig.add_vline(x=float(values.mean()), line_dash="dash", line_color=color, line_width=1.5)
fig.update_layout(
    height=320,
    barmode="overlay",
    title=f"Uniqueness of {n_draws} draws, naive vs sequential bootstrap",
    xaxis_title="Average uniqueness of the drawn label",
    yaxis_title="Count",
)
show_plotly_with_alt(
    fig,
    alt=(
        "Two overlaid histograms on one axis of the average uniqueness of the labels "
        "drawn by naive random sampling in slate and by sequential bootstrap in amber, "
        "with a dashed vertical line at each mean. Both are right-skewed over the same "
        "range from about 0.04 to 0.52 with a mode near 0.09, and they overlap almost "
        "everywhere. The naive histogram stands above the sequential one through the "
        "crowded low end below 0.15, and the sequential one is the taller of the two from "
        "about 0.25 outward. The two mean lines sit close together between 0.15 and 0.20, "
        "a separation far smaller than the spread of either distribution."
    ),
)

# %% [markdown]
# The two histograms are nearly the same shape, and the printed comparison above says how
# far apart their means are. The shift is real and it is small: the sequential draw tilts
# toward less-overlapped labels, it does not select them. What it cannot do is manufacture
# independent observations that the sampling scheme never generated - the overlap is a
# property of labelling every bar over a horizon of many bars, and the next section
# measures what that leaves.

# %% [markdown]
# ### Effective Sample Size
#
# The section text defines $N_{\text{eff}} = \sum_{t,a} w_{t,a}$ and notes that
# for fixed-horizon labels sampled at every bar, $N_{\text{eff}} \approx N / H$.
#
# That is an approximation. Below we *measure* $N_{\text{eff}}$ by computing each
# label's average uniqueness $w_{t,a}$ directly - per symbol, since concurrency
# accumulates along the time axis, not across the cross-section - and compare it
# to the $N/H$ shortcut. We report both the single-asset (SPY) and the full-panel
# figures, because they answer different questions: how many independent
# observations does *one* series carry, and how many does the *panel* carry.

# %% [markdown]
# Uniqueness depends only on the index geometry of the labels, not on prices: label $i$ is
# alive over bars $[i, i+H]$, and $w_i$ averages $1/c(u)$ over that span. Concurrency is a
# per-symbol quantity, because two different ETFs' labels do not overlap each other in the
# sense the weight measures, so the panel figure is a sum over symbols rather than one
# calculation on the stacked frame.

# %%
N_nominal = len(etf_with_fwd)
n_symbols = etf_with_fwd["symbol"].n_unique()

# For fixed-horizon labels sampled at every bar, uniqueness ≈ 1/H so N_eff ≈ N/H.
# This is what the section text quotes; we keep it to compare against the measurement.
N_eff_approx = N_nominal / HORIZON


def measure_n_eff(n_labels: int, h: int) -> tuple[float, float]:
    """Return (mean average-uniqueness, N_eff) for n_labels contiguous H-bar labels."""
    starts = np.arange(n_labels)
    w = calculate_label_uniqueness(starts, starts + h, n_bars=n_labels + h)
    return float(w.mean()), float(w.sum())


# Single asset: SPY. This is the number quoted for a one-series study.
spy_labels = etf_with_fwd.filter(pl.col("symbol") == "SPY").height
spy_mean_u, spy_n_eff = measure_n_eff(spy_labels, HORIZON)

# Full panel: sum N_eff across symbols (each symbol has its own length).
panel_n_eff = 0.0
for (_sym,), grp in etf_with_fwd.group_by("symbol"):
    if grp.height > HORIZON:
        panel_n_eff += measure_n_eff(grp.height, HORIZON)[1]

panel_mean_u = panel_n_eff / N_nominal
se_inflation = np.sqrt(N_nominal / panel_n_eff)

print(f"Fixed-horizon ETF labels (H={HORIZON}):")
print(f"  Symbols:              {n_symbols}")
print(f"  Nominal N (panel):    {N_nominal:,}")
print()
print(f"  SPY: labels           {spy_labels:,}")
print(f"       avg uniqueness   {spy_mean_u:.4f}   (≈ 1/(H+1) = {1 / (HORIZON + 1):.4f})")
print(f"       N_eff (measured) {spy_n_eff:,.0f}")
print()
print(f"  Panel: avg uniqueness {panel_mean_u:.4f}")
print(f"         N_eff measured {panel_n_eff:,.0f}")
print(
    f"         N_eff ≈ N/H    {N_eff_approx:,.0f}   (the shortcut, off by "
    f"{100 * (N_eff_approx / panel_n_eff - 1):.1f}%)"
)
print(
    f"  SE inflation:         √(N/N_eff) = {se_inflation:.2f}× "
    f"(confidence intervals based on N are this much too narrow)"
)
print()
print("  Note: a label spans H+1 bars inclusive of both endpoints, so maximal")
print(f"  overlap gives w = 1/(H+1) = {1 / (HORIZON + 1):.4f}, not 1/H = {1 / HORIZON:.4f}.")
print("  That is why the measured N_eff sits slightly below the N/H shortcut.")

# %% [markdown]
# ## Trend Scanning Labels
#
# De Prado's adaptive approach that identifies trends using t-statistics.
# The method scans forward with varying windows and selects the one
# with the highest statistical significance.

# %%
labels_trend = trend_scanning_labels(
    spy,
    min_window=5,  # Minimum 5 days
    max_window=20,  # Maximum 20 days
    step=1,  # Check every window size
    price_col="close",
)

print("Trend Scanning Labels:")
print("Label Distribution:")
display(labels_trend.group_by("label").len().sort("label"))

if "t_value" in labels_trend.columns:
    print("T-Value Statistics:")
    display(labels_trend["t_value"].describe())

# %% [markdown]
# ### What the Trend-Scanning t-statistic Measures
#
# For each bar the method regresses the **price level** on a time index over every window
# from five to twenty bars and keeps the window with the largest absolute t-statistic. Two
# separate things inflate that number, and only one of them is a multiple-comparisons
# problem:
#
# 1. **Selection.** The reported t is a maximum over sixteen candidates, not a single draw.
# 2. **Dependence.** Prices are close to a random walk, so the residuals of a regression on
#    their levels are strongly autocorrelated. The ordinary least-squares standard error
#    assumes they are not, and is therefore far too small.
#
# The usual answer addresses the first: raise the critical value to $\alpha/k$. The cells
# below apply it, and then test whether it was the binding problem by running the identical
# scan on a price path with no trend structure left in it at all.

# %%
if "optimal_window" in labels_trend.columns:
    horizon_col = "optimal_window"
elif "best_window" in labels_trend.columns:
    horizon_col = "best_window"
else:
    horizon_col = None

N_CANDIDATES = 20 - 5 + 1  # max_window - min_window + 1
ALPHA = 0.05
bonferroni_crit = sp_stats.norm.ppf(1 - ALPHA / (2 * N_CANDIDATES))

# %% [markdown]
# The null the t-statistic is compared against says: no trend. Rather than assume its
# shape, build it. Subtract the sample mean from SPY's daily log returns and shuffle what
# is left: the result has SPY's own return dispersion and shape, no drift, and no temporal
# ordering, so no window of it contains a trend to find. Demeaning matters - a plain
# shuffle would keep the decade's compounded gain and leave a real slope in every window.
# Whatever the scan reports on this path is what the method reports when there is nothing
# there.

# %%
_log_returns = np.diff(np.log(spy["close"].to_numpy()))
_driftless = np.random.default_rng(SEED).permutation(_log_returns - _log_returns.mean())
_shuffled = np.log(spy["close"][0]) + np.concatenate([[0.0], np.cumsum(_driftless)])
spy_permuted = spy.with_columns(pl.Series("close", np.exp(_shuffled)))

labels_permuted = trend_scanning_labels(
    spy_permuted,
    min_window=5,
    max_window=20,
    step=1,
    price_col="close",
)

raw_t = labels_trend["t_value"].drop_nulls()
null_t = labels_permuted["t_value"].drop_nulls()

# %%
if horizon_col is not None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Selected window", "|t|: SPY vs driftless"],
    )

    selected_horizons = labels_trend[horizon_col].drop_nulls().cast(pl.Int32, strict=False)
    fig.add_trace(
        go.Histogram(
            x=selected_horizons.to_numpy(),
            xbins=dict(start=4.5, end=20.5, size=1),
            name="Selected window",
            marker_color=COLORS["blue"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    T_AXIS_MAX = 45  # a handful of near-perfect fits run far past this; disclosed on the axis
    t_bins = dict(start=0, end=T_AXIS_MAX, size=1)
    n_clipped = int((raw_t.abs() > T_AXIS_MAX).sum() + (null_t.abs() > T_AXIS_MAX).sum())
    fig.add_trace(
        go.Histogram(
            x=raw_t.abs().to_numpy(),
            xbins=t_bins,
            name="SPY",
            marker_color=COLORS["blue"],
            opacity=0.6,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=null_t.abs().to_numpy(),
            xbins=t_bins,
            name="Driftless path",
            marker_color=COLORS["amber"],
            opacity=0.6,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=1.96,
        line_dash="dash",
        line_color=COLORS["negative"],
        annotation_text="t=1.96",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Window (bars)", row=1, col=1)
    fig.update_xaxes(
        title_text=f"|t| of the selected window ({n_clipped} beyond {T_AXIS_MAX} not shown)",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(
        height=360,
        barmode="overlay",
        font=dict(size=12),
        title_text="Trend scanning: selected window, and |t| against a driftless path",
    )
    show_plotly_with_alt(
        fig,
        alt=(
            "Two panels. The left panel is a histogram of the window length the scan "
            "selects for each SPY bar, from five to twenty. The bars are near-level across "
            "the windows from six to nineteen, a little taller at five, and dwarfed by a "
            "single dominant spike at twenty, the longest window offered. "
            "The right panel overlays the absolute t-statistic of the selected window for "
            "SPY in navy and for a driftless random walk built from SPY's own demeaned "
            "returns in amber. The two distributions sit almost on top of each other: both "
            "climb steeply from the left edge of the axis, peak between five and seven, "
            "and trail off past twenty. The dashed red critical-value line stands at that "
            "left edge with almost no mass to its left. A single observation from the "
            "driftless path falls beyond the plotted range and is noted on the axis."
        ),
    )

# %% tags=["results"]
raw_significant = int((raw_t.abs() > 1.96).sum())
corrected_significant = int((raw_t.abs() > bonferroni_crit).sum())
null_significant = int((null_t.abs() > 1.96).sum())
at_max_window = int((selected_horizons == 20).sum())

print(f"Candidate windows scanned per bar:   {N_CANDIDATES}")
print(f"Bonferroni critical value:           {bonferroni_crit:.2f} (vs 1.96 uncorrected)")
print(
    f"  longest window selected:           {at_max_window:,} / {len(selected_horizons):,}"
    f" ({at_max_window / len(selected_horizons):.1%})"
)
print()
print(
    f"SPY, significant at 5% raw:          {raw_significant:,} / {len(raw_t):,}"
    f" ({raw_significant / len(raw_t):.1%})"
)
print(
    f"SPY, significant after Bonferroni:   {corrected_significant:,} / {len(raw_t):,}"
    f" ({corrected_significant / len(raw_t):.1%})"
)
print(
    f"Driftless path, significant at 5%:   {null_significant:,} / {len(null_t):,}"
    f" ({null_significant / len(null_t):.1%})   <- a 5% test should reject 5% here"
)
print()
print(f"Median |t|, SPY:                     {float(raw_t.abs().median()):.2f}")
print(f"Median |t|, driftless path:          {float(null_t.abs().median()):.2f}")

# %% [markdown]
# The driftless path has no trend in it anywhere, by construction, and the scan calls
# almost every bar significant anyway - at a rate indistinguishable from SPY's, with a
# median absolute t-statistic of the same size and a matching distribution out into the
# tail. A test that rejects this often under a null it was built to accept is not measuring
# whether a trend exists; it is measuring that the residuals of a regression on price levels
# are autocorrelated, which they are whatever the prices do.
#
# Against that, the Bonferroni correction moves the rejection rate by a few percentage
# points. It is doing what it claims - the reported t really is a maximum over sixteen
# candidates - and the correction is not where the problem is. The selected-window panel
# shows the same thing from the other side: the scan lands on the longest window far more
# often than on any other, because a longer window buys more observations and a larger t
# under drift of any sign, so the sixteen candidates are neither independent nor exchangeable.
#
# Trend scanning still produces a usable label: the **sign** of the fitted slope is a
# statement about the path, and that is what `label` records. It is the accompanying
# t-statistic that should not be read as significance, before or after correction. To use
# one, calibrate it against a permutation null like the one above rather than against a
# Student-t table.

# %% [markdown]
# ## Meta-Labeling
#
# **Meta-labeling** separates the signal from the sizing decision:
#
# 1. A primary model generates directional signals (+1 long, -1 short)
# 2. Triple-barrier labels determine whether each signal was profitable
# 3. A secondary (meta) model learns *when to act* and *how much to bet*
#
# This decomposes the problem: the primary model handles *direction*,
# the meta-model handles *confidence*. The cells below illustrate the
# construction on SPY; the case studies in this book do not adopt
# meta-labeling (each case study trains a single model on one label
# horizon and sizes positions through an allocator), but the pattern
# transfers directly to any directional model already in place.

# %%
# Simple primary signal: buy when 20-day momentum is positive
spy_meta = spy.with_columns(
    signal=pl.when(pl.col("close") > pl.col("close").shift(20)).then(1).otherwise(-1),
    fwd_return=(pl.col("close").shift(-HORIZON) / pl.col("close") - 1),
).drop_nulls()

# Create meta-labels: was the signal profitable?
spy_meta = meta_labels(spy_meta, signal_col="signal", return_col="fwd_return")

print("Meta-Label Distribution:")
display(spy_meta.group_by("meta_label").len().sort("meta_label"))

# %% [markdown]
# The second half of meta-labeling is turning the meta-model's probability into a position
# size. `compute_bet_size` offers three mappings; the figure below is the mapping itself,
# plotted over the whole probability range, which is the part that transfers to any model.

# %%
BET_SCALE = 5.0
prob_grid = pl.DataFrame({"p": np.linspace(0.0, 1.0, 201)})
curves = prob_grid.with_columns(
    linear=compute_bet_size("p", method="linear"),
    sigmoid=compute_bet_size("p", method="sigmoid", scale=BET_SCALE),
    discrete=compute_bet_size("p", method="discrete", threshold=0.5),
)

fig = go.Figure()
for name, color, dash in (
    ("linear", COLORS["neutral"], "dot"),
    ("sigmoid", COLORS["blue"], "solid"),
    ("discrete", COLORS["amber"], "solid"),
):
    fig.add_trace(
        go.Scatter(
            x=curves["p"].to_numpy(),
            y=curves[name].to_numpy(),
            mode="lines",
            name=name,
            line=dict(color=color, width=2, dash=dash),
        )
    )
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], line_width=0.8)
fig.update_layout(
    height=360,
    title=f"Bet size against meta-model probability (sigmoid scale {BET_SCALE:g})",
    xaxis_title="Meta-model probability of success",
    yaxis_title="Bet size",
)
show_plotly_with_alt(
    fig,
    alt=(
        "Three curves mapping a meta-model probability on the horizontal axis, from zero "
        "to one, to a bet size on the vertical axis. The dotted slate linear mapping is a "
        "straight line from minus one to plus one crossing zero at a probability of 0.5. "
        "The navy sigmoid at scale five is an S through the same crossing point: steeper "
        "than the straight line through the middle and flatter at the ends, where it stops "
        "short of the full range the linear mapping reaches. The amber discrete mapping is "
        "a step that "
        "sits at zero below 0.5 and jumps to one above it."
    ),
)

# %%
spy_meta = spy_meta.with_columns(
    # Stand in for a trained meta-model: two confidence levels, not a calibrated score.
    pseudo_prob=pl.col("meta_label").cast(pl.Float64) * 0.3 + 0.5,
).with_columns(
    bet_size=compute_bet_size("pseudo_prob", method="sigmoid", scale=BET_SCALE),
)

print("Bet Size Statistics (sigmoid method):")
display(spy_meta["bet_size"].describe())

# %% [markdown]
# The statistics above have a minimum, a maximum and no spread worth the name, because the
# stand-in probability takes two values and therefore touches the curve in two places. That
# is a property of the placeholder, not of meta-labeling: a trained classifier emits a
# continuous score and the whole curve comes into play.
#
# It is also where meta-labeling earns its keep and where it can mislead. Sizing on a
# probability is only as good as the calibration of that probability, and a classifier
# optimised for accuracy or AUC is not calibrated by default. Ranking well and being right
# about *how often* are different properties, and only the second one sizes a position.

# %% [markdown]
# ## Label Diagnostics
#
# The function below provides a reusable diagnostic template. Run it on any
# label column to check distribution stability and class balance-the two
# properties that determine whether a label is learnable.


# %%
def label_diagnostics(
    df: pl.DataFrame,
    label_col: str,
    timestamp_col: str = "timestamp",
    title_prefix: str = "",
) -> None:
    """
    Generate diagnostic plots for any label column.

    Works with both continuous (returns) and discrete (classification) labels.
    """
    labels = df[label_col].drop_nulls()
    n_unique = labels.n_unique()
    is_discrete = n_unique <= 10  # Heuristic: discrete if few unique values

    if is_discrete:
        # Discrete label diagnostics
        print(f"\n{'=' * 60}")
        print(f"{title_prefix} Discrete Label Diagnostics")
        print(f"{'=' * 60}")
        print(f"Unique values: {labels.unique().sort().to_list()}")

        # Exclude nulls from value counts
        df_non_null = df.drop_nulls(subset=[label_col])
        print("Value Counts:")
        display(df_non_null.group_by(label_col).len().sort(label_col))

        counts = df_non_null.group_by(label_col).len().sort(label_col)
        fig = px.bar(
            counts.to_pandas(),
            x=label_col,
            y="len",
            title=f"{title_prefix} label counts by class",
        )
        fig.update_layout(
            height=300,
            xaxis_title=f"{label_col} value",
            yaxis_title="Count",
            xaxis=dict(type="category"),
        )
        show_plotly_with_alt(
            fig,
            alt=(
                f"A bar chart of how many observations carry each value of {label_col}. "
                f"The classes and their counts are "
                + ", ".join(
                    f"{v}: {n:,}"
                    for v, n in zip(
                        counts[label_col].to_list(), counts["len"].to_list(), strict=True
                    )
                )
                + "."
            ),
        )

        # Class balance over time
        if timestamp_col in df.columns:
            by_date = (
                df.group_by([timestamp_col, label_col])
                .len()
                .pivot(on=label_col, index=timestamp_col, values="len")
                .sort(timestamp_col)
            )
            # Compute class proportions (robust to missing classes on a date)
            count_cols = [c for c in by_date.columns if c != timestamp_col]
            if count_cols:
                by_date = by_date.with_columns([pl.col(c).fill_null(0) for c in count_cols])
                by_date = by_date.with_columns(
                    pl.sum_horizontal([pl.col(c) for c in count_cols]).alias("_total")
                )
                by_date = by_date.with_columns(
                    [
                        pl.when(pl.col("_total") > 0)
                        .then(pl.col(c) / pl.col("_total"))
                        .otherwise(None)
                        .alias(f"{c}_pct")
                        for c in count_cols
                    ]
                )

            # Plot proportions
            fig = go.Figure()
            for col in by_date.columns:
                if col.endswith("_pct"):
                    fig.add_trace(
                        go.Scatter(
                            x=by_date[timestamp_col].to_list(),
                            y=by_date[col].to_numpy(),
                            mode="lines",
                            name=col.replace("_pct", ""),
                        )
                    )
            fig.update_layout(
                height=300,
                title=f"{title_prefix} class proportions by date",
                xaxis_title="Date",
                yaxis_title="Proportion",
            )
            show_plotly_with_alt(
                fig,
                alt=(
                    f"One line per class showing the share of observations carrying each "
                    f"value of {label_col} on each date, over "
                    f"{by_date[timestamp_col].min()} to {by_date[timestamp_col].max()}. "
                    f"The classes plotted are "
                    + ", ".join(
                        str(c).replace("_pct", "")
                        for c in by_date.columns
                        if str(c).endswith("_pct")
                    )
                    + "."
                ),
            )
    else:
        # Continuous label diagnostics
        print(f"\n{'=' * 60}")
        print(f"{title_prefix} Continuous Label Diagnostics")
        print(f"{'=' * 60}")
        display(labels.describe())

        fig = px.histogram(
            x=labels.to_numpy(),
            nbins=50,
            title=f"{title_prefix} label distribution",
        )
        fig.update_layout(height=300, xaxis_title="Label value", yaxis_title="Count")
        show_plotly_with_alt(
            fig,
            alt=(
                f"A histogram of {len(labels):,} values of {label_col}. The distribution "
                f"runs from {labels.min():.3f} to {labels.max():.3f} with a median of "
                f"{labels.median():.3f} and a standard deviation of {labels.std():.3f}, "
                f"so the plotted range is roughly "
                f"{(labels.max() - labels.min()) / labels.std():.0f} standard deviations "
                f"wide and the mass is concentrated near the median."
            ),
        )

        # Time series if available
        if timestamp_col in df.columns:
            fig = px.line(
                df.select([timestamp_col, label_col]).drop_nulls().to_pandas(),
                x=timestamp_col,
                y=label_col,
                title=f"{title_prefix} label value by date",
            )
            fig.update_layout(height=300, xaxis_title="Date", yaxis_title="Label value")
            show_plotly_with_alt(
                fig,
                alt=(
                    f"A single line of {label_col} plotted against {timestamp_col}, "
                    f"oscillating around a median of {labels.median():.3f} within a range "
                    f"of {labels.min():.3f} to {labels.max():.3f}. Because the label is a "
                    f"forward return, neighbouring points share most of their window and "
                    f"the line is far smoother than a series of independent draws."
                ),
            )


# Example: run diagnostics on fixed horizon labels
label_diagnostics(labels_returns, fh_label_col, title_prefix=f"Fixed Horizon ({HORIZON}d)")

# %% [markdown]
# ## Label Method Comparison
#
# Continuous and discrete targets go on separate axes: they are different quantities and
# putting them on one would be a category error. Within each figure the panels share their
# axes, so a difference in spread is a difference in the labels rather than an artefact of
# letting each panel pick its own range.

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[f"Fixed horizon, {HORIZON}-day", "Triple-barrier, booked return"],
    horizontal_spacing=0.1,
    shared_xaxes=True,
    shared_yaxes=True,
)

fh_values = labels_returns[fh_label_col].drop_nulls().to_numpy()
tb_values = labels_tb["label_return"].drop_nulls().to_numpy()
span = float(max(np.abs(fh_values).max(), np.abs(tb_values).max()))
ret_bins = dict(start=-span, end=span, size=2 * span / 60)

fig.add_trace(
    go.Histogram(
        x=fh_values,
        xbins=ret_bins,
        name="Fixed Horizon",
        marker_color=COLORS["blue"],
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=tb_values,
        xbins=ret_bins,
        name="TB Label Return",
        marker_color=COLORS["amber"],
        showlegend=False,
    ),
    row=1,
    col=2,
)

for col in (1, 2):
    fig.update_xaxes(title_text="Return", range=[-span, span], row=1, col=col)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_layout(height=360, title_text="Continuous targets on a shared return axis")
show_plotly_with_alt(
    fig,
    alt=(
        "Two histograms of SPY label values on the same return axis running from about "
        "minus 0.33 to plus 0.33. The left panel, fixed-horizon 21-day returns, is a broad "
        "bell centred slightly above zero with visible tails on both sides. The right "
        "panel, the booked triple-barrier return, is two thin spikes crowded together near "
        "the middle at minus one and plus two percent, with the small remainder of "
        "time-barrier exits spread too thinly between them to see at this scale. "
        "The barrier target occupies a small fraction of the range the fixed-horizon "
        "target spans."
    ),
)

# %% [markdown]
# On a shared axis the barriers stop looking like a variation on the forward return and
# start looking like what they are: a target whose mass is pinned to two values a few
# percent apart, with only the trades that reach the time barrier keeping a return of
# their own.
# Everything the fixed-horizon label says about the size of a move has been discarded, in
# exchange for a statement about which of two thresholds arrived first. That is the right
# trade when the strategy really does exit at those thresholds and the wrong one when the
# model is meant to forecast magnitude.

# %%
discrete_sets = [
    ("Fixed horizon binary", labels_binary[binary_label_col], COLORS["blue"]),
    ("Triple barrier", labels_tb["label"], COLORS["amber"]),
    ("ATR barrier", labels_atr["label"], COLORS["copper"]),
]

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[name for name, _, _ in discrete_sets],
    horizontal_spacing=0.08,
    shared_yaxes=True,
)

for i, (name, series, color) in enumerate(discrete_sets, start=1):
    counts = series.drop_nulls().value_counts().sort(series.name).rename({series.name: "value"})
    fig.add_trace(
        go.Bar(
            x=[str(v) for v in counts["value"].to_list()],
            y=counts["count"].to_numpy(),
            name=name,
            marker_color=color,
            showlegend=False,
        ),
        row=1,
        col=i,
    )
    fig.update_xaxes(title_text="Class", type="category", row=1, col=i)

fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_layout(height=360, title_text="Discrete targets, shared count axis")
show_plotly_with_alt(
    fig,
    alt=(
        "Three bar charts of class counts on a shared vertical axis. Fixed-horizon binary "
        "labels are lopsided: about 780 bars at minus one against about 1,710 at plus one. "
        "The triple-barrier labels are close to balanced, roughly 1,250 at minus one and "
        "1,100 at plus one, with a short bar near 160 at zero for trades that reached the "
        "time barrier. The ATR-barrier labels are balanced in the same way, about 1,270 "
        "and 1,230, with the zero class almost invisible because volatility-scaled "
        "barriers are nearly always reached inside the holding window."
    ),
)

# %% [markdown]
# The three targets disagree about what a positive month is. The fixed-horizon binary label
# inherits the drift of the sample - over a decade in which SPY mostly rose, "up" is simply
# more common - so a classifier that predicts the majority class scores well above half
# without learning anything. Both barrier variants are near balanced instead, because the
# stop is set tighter than the take-profit and is therefore hit more often; that balance is
# a property of the barrier geometry, not evidence that the barrier label is more
# informative. The ATR variant almost never reaches the time barrier, since barriers scaled
# to recent volatility sit inside the range the price covers in twenty days.

# %% [markdown]
# ## Choosing a Method
#
# | Strategy Type | Recommended Method | Rationale |
# |--------------|-------------------|-----------|
# | Factor timing (monthly) | Fixed horizon | Simple, stationary targets |
# | Stat arb (intraday) | Fixed horizon binary | Speed matters |
# | Cross-sectional ranking | **Cross-sectional percentile** | Stable class balance |
# | Active trading | Triple barrier (ATR) | Matches trade mechanics |
# | Trend following | Trend scanning, sign only | Adaptive horizon; ignore its t-statistic |
#
# **Key Considerations**:
#
# 1. **Anchor alignment**: Match label computation to execution timing
# 2. **Cross-sectional vs time-series**: Most equity/ETF strategies need cross-sectional
# 3. **Path-dependence**: Use triple-barrier when stop losses are part of the strategy
# 4. **Bar resolution**: Give the barrier engine `high_col`, `low_col` and `open_col` in
#    production; a close-only test books every exit at the barrier price
# 5. **Volatility adaptation**: ATR-based barriers for changing market conditions

# %% [markdown]
# ## Summary
#
# ### Key Takeaways
#
# 1. **Fixed horizon** for simple regression and classification targets.
# 2. **Rolling percentile** thresholds adapt to the recent return distribution, which moves
#    with trailing volatility and with trailing drift together.
# 3. **Cross-sectional percentile** fixes the class balance by construction and pushes all
#    the variation into the cut point, where it is easy to stop looking at it.
# 4. **Triple barrier** describes the trade, not the market. A trade that touches a
#    barrier is booked at that barrier, so most of the target lands on one of two values;
#    only the minority that run to the time barrier carry a terminal return. Either way
#    the rest of the path is discarded.
# 5. **Trend scanning** gives a useful adaptive horizon and a sign. Its t-statistic is a
#    regression on price levels and rejects at almost any threshold under its own null.
# 6. **Anchor alignment** nets to zero on average and changes every individual label.
# 7. **Overlapping labels** cut the effective sample size by roughly the horizon; sample
#    weights and the sequential bootstrap tilt against the overlap without removing it.
#
# ### Production Usage
#
# Each case study computes its labels in its own `02_labels` notebook, which reads the
# horizon and the label set from that case study's `config/setup.yaml` rather than from
# constants typed into a cell. `utils.artifact_specs.resolve_label_horizon` is the accessor.
#
# ### References
#
# - Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
#   - Chapter 3: Labeling (Triple-Barrier, Meta-Labeling)
#   - Chapter 4: Sample Weights (Uniqueness)
# - See [`04_maximum_favorable_adverse_excursion`](04_maximum_favorable_adverse_excursion.ipynb)
#   for empirical barrier calibration
