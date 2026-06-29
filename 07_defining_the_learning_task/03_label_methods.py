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
# - `01_data_quality_diagnostics` — establishes the ETF coverage assumptions used here.
# - Familiarity with leakage-aware splitting (Chapter 6 §6.3) and forward-return semantics.
# - Polars DataFrame manipulation; basic statistics (t-statistics, percentiles).
#
# ## Data Contract
#
# - **Input**: Real ETF OHLCV from data loaders (SPY for single-asset, full universe for cross-sectional)
# - **Output**: Example labels for teaching (use `compute_labels()` for production)

# %%
"""Label Methods — fixed-horizon, cross-sectional, and event-driven labeling for supervised learning."""

from __future__ import annotations

import warnings
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

warnings.filterwarnings("ignore")

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
# ## 1. Load Sample Data
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
# ## 2. Fixed Time Horizon Labels
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
horizon = 21

labels_returns = fixed_time_horizon_labels(
    spy,
    horizon=horizon,
    method="returns",
    price_col="close",
)

# Discover the produced label column robustly
fh_label_col = first_col_matching_any(labels_returns, [f"{horizon}", "label_return", "label"])
print(f"Fixed Horizon Labels (horizon={horizon}):")
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
    horizon=horizon,
    method="binary",
    price_col="close",
)

binary_label_col = first_col_matching_any(labels_binary, ["direction", "label"])

print("Return distribution:")
display(labels_returns[fh_label_col].describe())
print("Binary label distribution:")
display(labels_binary.group_by(binary_label_col).len().sort(binary_label_col))

# %% [markdown]
# ## 3. Anchor Alignment Demo
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
        (pl.col("close").shift(-horizon) / pl.col("close") - 1).alias("ret_close_to_close"),
        # Next-open-to-open: decision at close(t), execute at open(t+1), exit at open(t+horizon+1)
        # Holding for `horizon` trading days means exit is horizon+1 bars from decision
        (pl.col("open").shift(-(horizon + 1)) / pl.col("open").shift(-1) - 1).alias(
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
# Visualize the difference over time
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "Anchor Difference Over Time",
        "Distribution of Anchor Differences",
    ],
    row_heights=[0.6, 0.4],
    vertical_spacing=0.15,
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
        line=dict(width=1.5, color="#2166ac"),
    ),
    row=1,
    col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8, row=1, col=1)

# Histogram of differences
fig.add_trace(
    go.Histogram(
        x=spy_anchors["anchor_diff"].to_numpy(),
        nbinsx=50,
        name="Distribution",
        showlegend=False,
        marker_color="#2166ac",
    ),
    row=2,
    col=1,
)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_yaxes(title_text="Return Difference", row=1, col=1)
fig.update_xaxes(title_text="Return Difference", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_layout(
    height=550,
    title_text=f"Anchor Alignment Impact — Close-to-Close minus Open-to-Open ({horizon}-day horizon)",
    font=dict(size=12),
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
)
fig.show()

# %%
# Example timestamps showing anchor shift
spy_anchors.select(
    ["timestamp", "close", "open", "ret_close_to_close", "ret_next_open_to_open", "anchor_diff"]
).head(10)

# %% [markdown]
# **Key Insight**: The anchor difference is noisy at the trade level (standard
# deviation ~100bps for SPY), even though it averages close to zero. This means
# individual label assignments can differ substantially between anchors, affecting
# model training. For end-of-day signals executed at next open, labels should use
# next-open anchoring to match the actual execution price.

# %% [markdown]
# ## 4. Time-Series Percentile Labels
#
# Labels are relative to recent history for a **single instrument**,
# making them adaptive to volatility regimes.

# %%
# Binary percentile: Is return in top 25%?
labels_ts_pct = rolling_percentile_binary_labels(
    spy,
    horizon=horizon,
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
# Visualize threshold adaptation
threshold_col = [c for c in labels_ts_pct.columns if "threshold" in c.lower()]
if threshold_col:
    fig = px.line(
        labels_ts_pct.to_pandas(),
        x="timestamp",
        y=threshold_col[0],
        title=f"Rolling 75th Percentile Threshold ({horizon}-day returns)",
    )
    fig.update_layout(height=350, yaxis_title="Return Threshold")
    fig.show()

# %% [markdown]
# ## 5. Cross-Sectional Percentile Labels
#
# **The most natural use of percentile labels**: rank assets within the universe
# at each decision time, then label top/bottom quantiles.
#
# This is the standard approach for equity and ETF rotation strategies.

# %%
# Compute forward returns and cross-sectional rank for the entire ETF universe
etf_with_fwd = etf_filtered.with_columns(
    [(pl.col("close").shift(-horizon) / pl.col("close") - 1).over("symbol").alias("fwd_return")]
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
quintile_threshold = 20  # Top/bottom 20%
etf_cs = etf_cs.with_columns(
    [
        pl.when(pl.col("pct_rank") >= (100 - quintile_threshold))
        .then(pl.lit(1))
        .when(pl.col("pct_rank") <= quintile_threshold)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("cs_label")
    ]
)

print(f"Cross-Sectional Labels ({horizon}d horizon, {quintile_threshold}th percentile cutoffs):")
print(f"  Total observations: {len(etf_cs):,}")
display(etf_cs.group_by("cs_label").len().sort("cs_label"))

# %%
# Verify stable class proportions over time
# Note: Counts vary if universe size changes; proportions are stable by construction.
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
    title="Cross-Sectional Label Proportions Over Time (Stable by Construction)",
    xaxis_title="Date",
    yaxis_title="Proportion",
)
fig.show()

# %%
# Show cross-sectional threshold values over time
cs_thresholds = (
    etf_with_fwd.group_by("timestamp")
    .agg(
        [
            pl.col("fwd_return").quantile(quintile_threshold / 100).alias("bottom_threshold"),
            pl.col("fwd_return").quantile(1 - quintile_threshold / 100).alias("top_threshold"),
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
        line=dict(color="green"),
    )
)
fig.add_trace(
    go.Scatter(
        x=cs_thresholds["timestamp"].to_list(),
        y=cs_thresholds["bottom_threshold"].to_numpy(),
        mode="lines",
        name="Bottom 20% Threshold",
        line=dict(color="red"),
    )
)
fig.update_layout(
    height=400,
    title="Cross-Sectional Threshold Values Over Time",
    xaxis_title="Date",
    yaxis_title=f"{horizon}-day Return Threshold",
)
fig.show()

# %% [markdown]
# **Key Insight**: Cross-sectional percentile labels maintain stable class distributions
# by construction, but the absolute return thresholds vary with market conditions.
# In high-volatility periods, larger absolute returns are needed to qualify as "top quintile".

# %% [markdown]
# ## 6. Triple-Barrier Labels
#
# Path-dependent labeling that captures realistic trade outcomes:
# - **Upper barrier**: Take profit hit → +1
# - **Lower barrier**: Stop loss hit → -1
# - **Time barrier**: Neither hit → label based on final return
#
# This method is from De Prado's *Advances in Financial Machine Learning*.

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
# ### 6.1 Triple-Barrier Path Visualization
#
# **Understanding triple-barrier requires seeing the price paths**.
# Below we plot several example trades showing how barriers are hit.

# %%
# Find examples of each barrier hit type
tb_with_price = labels_tb.join(
    spy.select(["timestamp", "close"]), on="timestamp", how="left"
).with_row_index("row_idx")


def plot_triple_barrier_example(
    df: pl.DataFrame, entry_idx: int, config: LabelingConfig, title: str
) -> go.Figure:
    """Plot a single triple-barrier trade example with barriers overlaid."""
    # Get entry point
    entry_row = df.row(entry_idx, named=True)
    entry_price = entry_row["close"]
    entry_time = entry_row["timestamp"]

    # Calculate barrier levels
    upper_level = entry_price * (1 + config.upper_barrier)
    lower_level = entry_price * (1 - config.lower_barrier)

    # Get the forward price path
    forward_rows = df.filter(pl.col("timestamp") >= entry_time).head(config.max_holding_period + 1)

    fig = go.Figure()

    # Price path
    fig.add_trace(
        go.Scatter(
            x=forward_rows["timestamp"].to_list(),
            y=forward_rows["close"].to_numpy(),
            mode="lines+markers",
            name="Price",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        )
    )

    # Entry point
    fig.add_trace(
        go.Scatter(
            x=[entry_time],
            y=[entry_price],
            mode="markers",
            name="Entry",
            marker=dict(color="black", size=12, symbol="star"),
        )
    )

    # Upper barrier (horizontal line)
    fig.add_hline(
        y=upper_level,
        line_dash="dash",
        line_color="green",
        annotation_text=f"TP: {upper_level:.2f} (+{config.upper_barrier:.1%})",
    )

    # Lower barrier (horizontal line)
    fig.add_hline(
        y=lower_level,
        line_dash="dash",
        line_color="red",
        annotation_text=f"SL: {lower_level:.2f} (-{config.lower_barrier:.1%})",
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
        line=dict(color="gray", dash="dot"),
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

    # Plot examples
    for barrier_type, idx in examples[:3]:  # Limit to 3 examples
        if idx < len(tb_with_price):
            fig = plot_triple_barrier_example(
                tb_with_price,
                idx,
                config,
                f"Triple-Barrier Example: {barrier_type.upper()} barrier hit",
            )
            fig.show()

# %% [markdown]
# ### 6.2 ATR-Based Barriers
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
# ### 6.3 Sample Weights from Uniqueness
#
# Overlapping labels create mechanical dependence: high-concurrency periods
# dominate training loss. Weighting by uniqueness prevents these periods
# from overwhelming the model. De Prado introduces **uniqueness-based
# sample weights** where more unique samples (less overlap) get higher weights.

# %%
if "sample_weight" in labels_tb.columns:
    print("Sample Weight Statistics:")
    display(labels_tb["sample_weight"].describe())

    # Visualize weight distribution
    fig = px.histogram(
        labels_tb.filter(pl.col("sample_weight").is_not_null()).to_pandas(),
        x="sample_weight",
        nbins=50,
        title="Triple-Barrier Sample Weight Distribution",
    )
    fig.update_layout(height=350)
    fig.show()

# %% [markdown]
# ### 6.4 Rich Triple-Barrier Output
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
    print(f"  {col:<20} {str(dtype):<12} — {labels_tb[col].drop_nulls().head(1).to_list()}")

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
# Return distribution colored by barrier hit type
fig = px.histogram(
    labels_tb.filter(pl.col("label_return").is_not_null()).to_pandas(),
    x="label_return",
    color="barrier_hit",
    nbins=50,
    barmode="overlay",
    opacity=0.7,
    title="Label Return Distribution by Barrier Hit Type",
)
fig.update_layout(height=350, xaxis_title="Label Return", yaxis_title="Count")
fig.show()

# %% [markdown]
# ### 6.5 Sequential Bootstrap
#
# Overlapping labels create sample dependence. The **sequential bootstrap**
# (De Prado, AFML Ch4) generates bootstrap indices that respect label
# uniqueness — favoring samples with less concurrent overlap.

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

# %%
# Visualize the difference
fig = make_subplots(rows=1, cols=2, subplot_titles=["Naive Bootstrap", "Sequential Bootstrap"])

fig.add_trace(
    go.Histogram(x=naive_uniqueness, nbinsx=30, name="Naive", marker_color="gray", opacity=0.7),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=seq_uniqueness, nbinsx=30, name="Sequential", marker_color="#2ca02c", opacity=0.7
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="Uniqueness", row=1, col=1)
fig.update_xaxes(title_text="Uniqueness", row=1, col=2)
fig.update_layout(height=300, title_text="Sequential Bootstrap Favors Higher-Uniqueness Samples")
fig.show()

# %% [markdown]
# Sequential bootstrap produces training sets where each sample contributes
# more independent information. This reduces effective sample size but
# improves model generalization on overlapping label data.

# %% [markdown]
# ### 6.6 Effective Sample Size
#
# The section text defines $N_{\text{eff}} = \sum_{t,a} w_{t,a}$ and gives
# a worked example: 100 ETFs $\times$ 20 years $\times$ 250 days $= 500{,}000$
# nominal labels; with $H=5$, $N_{\text{eff}} \approx 100{,}000$.
# Let us verify this empirically on our fixed-horizon ETF labels.

# %%
# Compute effective sample size for fixed-horizon labels on the ETF universe
N_nominal = len(etf_with_fwd)
n_symbols = etf_with_fwd["symbol"].n_unique()

# For fixed-horizon labels sampled at every bar, uniqueness ≈ 1/H
# so N_eff ≈ N / H (ignoring cross-sectional correlation)
N_eff_approx = N_nominal / horizon

# Exact uniqueness for fixed-horizon: each label is alive for H bars,
# and at each bar ~n_symbols labels are alive (one per asset).
# Concurrency c(u) ≈ n_symbols × H for cross-sectional panels,
# but uniqueness is computed per-asset: w ≈ 1/H for the time-series dimension.
print(f"Fixed-horizon ETF labels (H={horizon}):")
print(f"  Symbols:         {n_symbols}")
print(f"  Nominal N:       {N_nominal:,}")
print(f"  N_eff ≈ N/H:     {N_eff_approx:,.0f}")
print(
    f"  SE inflation:    √{horizon} ≈ {np.sqrt(horizon):.1f}× (confidence intervals based on N are this much too narrow)"
)

# %% [markdown]
# ## 7. Trend Scanning Labels
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
# ### 7.1 Selection Bias in Trend Scanning
#
# Trend scanning picks the horizon with the strongest t-statistic for each
# observation. This maximization introduces **selection bias**: the reported
# t-statistics are systematically inflated. The Bonferroni correction raises
# the critical value to account for the number of horizons tested, requiring
# each t-statistic to clear a higher bar for significance.

# %%
# Distribution of selected horizons
if "optimal_window" in labels_trend.columns:
    horizon_col = "optimal_window"
elif "best_window" in labels_trend.columns:
    horizon_col = "best_window"
else:
    horizon_col = None

# %%
if horizon_col is not None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Selected Horizon Distribution", "Raw t-statistics (with critical values)"],
    )

    # (a) Histogram of selected horizons
    selected_horizons = labels_trend[horizon_col].drop_nulls().cast(pl.Int32, strict=False)
    fig.add_trace(
        go.Histogram(
            x=selected_horizons.to_numpy(),
            nbinsx=16,
            name="Selected horizon",
            marker_color="#2166ac",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # (b) Raw t-statistics vs Bonferroni-adjusted critical value
    if "t_value" in labels_trend.columns:
        n_candidates = 20 - 5 + 1  # max_window - min_window + 1
        raw_t = labels_trend["t_value"].drop_nulls()

        # Bonferroni: raise the critical value by dividing alpha by n_candidates
        alpha = 0.05
        bonferroni_crit = sp_stats.norm.ppf(1 - alpha / (2 * n_candidates))

        fig.add_trace(
            go.Histogram(
                x=raw_t.to_numpy(),
                nbinsx=50,
                name="Raw t",
                marker_color="rgba(33,102,172,0.5)",
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

        # Mark both critical values
        fig.add_vline(
            x=1.96, line_dash="dash", line_color="blue", annotation_text="t=1.96", row=1, col=2
        )
        fig.add_vline(x=-1.96, line_dash="dash", line_color="blue", row=1, col=2)
        fig.add_vline(
            x=bonferroni_crit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Bonf={bonferroni_crit:.2f}",
            row=1,
            col=2,
        )
        fig.add_vline(x=-bonferroni_crit, line_dash="dash", line_color="red", row=1, col=2)

        # Significance counts
        raw_significant = (raw_t.abs() > 1.96).sum()
        corrected_significant = (raw_t.abs() > bonferroni_crit).sum()

    fig.update_xaxes(title_text="Horizon (bars)", row=1, col=1)
    fig.update_xaxes(title_text="t-statistic", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(height=350, font=dict(size=12))
    fig.show()

# %% [markdown]
# The Bonferroni correction is conservative but illustrates the magnitude of
# the selection effect. After correction, many trends that appeared "significant"
# under the uncorrected test lose significance — confirming the text's warning
# that uncorrected trend scanning t-statistics should not be taken at face value.

# %%
if horizon_col is not None and "t_value" in labels_trend.columns:
    sign_change_frac = 1 - corrected_significant / max(raw_significant, 1)
    print(f"Candidate horizons tested: {n_candidates}")
    print(f"Bonferroni critical value: {bonferroni_crit:.2f} (vs 1.96 uncorrected)")
    print(
        f"Significant at 5% (raw):        {raw_significant:,} / {len(raw_t):,} ({raw_significant / len(raw_t):.1%})"
    )
    print(
        f"Significant at 5% (Bonferroni): {corrected_significant:,} / {len(raw_t):,} ({corrected_significant / len(raw_t):.1%})"
    )
    print(f"Fraction losing significance:    {sign_change_frac:.1%}")

# %% [markdown]
# ## 7.5 Meta-Labeling Concept
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
    fwd_return=(pl.col("close").shift(-horizon) / pl.col("close") - 1),
).drop_nulls()

# Create meta-labels: was the signal profitable?
spy_meta = meta_labels(spy_meta, signal_col="signal", return_col="fwd_return")

print("Meta-Label Distribution:")
display(spy_meta.group_by("meta_label").len().sort("meta_label"))

# %%
# Bet sizing: convert meta-model probability to position size
# Here we use the meta_label directly as a proxy for probability
spy_meta = spy_meta.with_columns(
    # Simulate a meta-model probability (in practice, this comes from a trained classifier)
    pseudo_prob=pl.col("meta_label").cast(pl.Float64) * 0.3 + 0.5,
).with_columns(
    bet_size=compute_bet_size("pseudo_prob", method="sigmoid", scale=5.0),
)

print("Bet Size Statistics (sigmoid method):")
display(spy_meta["bet_size"].describe())

# %% [markdown]
# **Key Insight**: Meta-labeling turns a classification problem (direction)
# into a probability calibration problem (confidence). This enables
# Kelly-criterion-style position sizing from ML predictions.

# %% [markdown]
# ## 8. Label Diagnostics
#
# The function below provides a reusable diagnostic template. Run it on any
# label column to check distribution stability and class balance—the two
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

        # Bar chart of label distribution
        fig = px.bar(
            df_non_null.group_by(label_col).len().sort(label_col).to_pandas(),
            x=label_col,
            y="len",
            title=f"{title_prefix} Label Distribution",
        )
        fig.update_layout(height=300)
        fig.show()

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
                title=f"{title_prefix} Class Proportions Over Time",
                yaxis_title="Proportion",
            )
            fig.show()
    else:
        # Continuous label diagnostics
        print(f"\n{'=' * 60}")
        print(f"{title_prefix} Continuous Label Diagnostics")
        print(f"{'=' * 60}")
        display(labels.describe())

        # Histogram
        fig = px.histogram(
            x=labels.to_numpy(),
            nbins=50,
            title=f"{title_prefix} Label Distribution",
        )
        fig.update_layout(height=300)
        fig.show()

        # Time series if available
        if timestamp_col in df.columns:
            fig = px.line(
                df.select([timestamp_col, label_col]).drop_nulls().to_pandas(),
                x=timestamp_col,
                y=label_col,
                title=f"{title_prefix} Labels Over Time",
            )
            fig.update_layout(height=300)
            fig.show()


# Example: run diagnostics on fixed horizon labels
label_diagnostics(labels_returns, fh_label_col, title_prefix="Fixed Horizon (20d)")

# %% [markdown]
# ## 9. Label Method Comparison
#
# **Important**: We must compare continuous vs discrete labels separately.
# Mixing them on the same visual axis is conceptually misleading.

# %%
# Continuous targets comparison
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Fixed Horizon Returns", "Label Return (from Triple-Barrier)"],
    horizontal_spacing=0.1,
)

# Fixed horizon returns
fig.add_trace(
    go.Histogram(
        x=labels_returns[fh_label_col].drop_nulls().to_numpy(),
        nbinsx=50,
        name="Fixed Horizon",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Triple-barrier final returns (before discretization)
if "label_return" in labels_tb.columns:
    fig.add_trace(
        go.Histogram(
            x=labels_tb["label_return"].drop_nulls().to_numpy(),
            nbinsx=50,
            name="TB Label Return",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig.update_layout(
    height=350,
    title_text="Continuous Targets Comparison",
)
fig.show()

# %%
# Discrete targets comparison
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Fixed Horizon Binary", "Triple-Barrier Label", "ATR-Barrier Label"],
    horizontal_spacing=0.08,
)

# Fixed horizon binary
fig.add_trace(
    go.Histogram(
        x=labels_binary[binary_label_col].drop_nulls().to_numpy(),
        name="Binary",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Triple barrier discrete
fig.add_trace(
    go.Histogram(
        x=labels_tb.filter(pl.col("label").is_not_null())["label"].to_numpy(),
        name="Triple Barrier",
        showlegend=False,
    ),
    row=1,
    col=2,
)

# ATR barrier discrete
fig.add_trace(
    go.Histogram(
        x=labels_atr.filter(pl.col("label").is_not_null())["label"].to_numpy(),
        name="ATR Barrier",
        showlegend=False,
    ),
    row=1,
    col=3,
)

# %%
fig.update_layout(
    height=350,
    title_text="Discrete Targets Comparison",
)
fig.show()

# %% [markdown]
# ## 10. Method Comparison: Decision Guide
#
# | Strategy Type | Recommended Method | Rationale |
# |--------------|-------------------|-----------|
# | Factor timing (monthly) | Fixed horizon | Simple, stationary targets |
# | Stat arb (intraday) | Fixed horizon binary | Speed matters |
# | Cross-sectional ranking | **Cross-sectional percentile** | Stable class balance |
# | Active trading | Triple barrier (ATR) | Matches trade mechanics |
# | Trend following | Trend scanning | Data-driven trend ID |
#
# **Key Considerations**:
#
# 1. **Anchor alignment**: Match label computation to execution timing
# 2. **Cross-sectional vs time-series**: Most equity/ETF strategies need cross-sectional
# 3. **Path-dependence**: Use triple-barrier when stop losses are part of the strategy
# 4. **Volatility adaptation**: ATR-based barriers for changing market conditions

# %% [markdown]
# ## Summary
#
# ### Key Takeaways
#
# 1. **Fixed horizon** for simple regression/classification targets
# 2. **Rolling percentile** for time-series adaptive thresholds (single instrument)
# 3. **Cross-sectional percentile** for relative ranking within a universe
# 4. **Triple barrier (ATR)** for realistic trading simulations with stops
# 5. **Trend scanning** for data-driven trend identification
# 6. **Anchor alignment** is critical - close-to-close vs next-open matters
#
# ### Production Usage
#
# For production label computation, use the experiment configuration module
# which centralizes method, horizon, and threshold choices per case study.
# The case study label notebooks (NB09–NB17) demonstrate this pipeline.
#
# ### References
#
# - Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
#   - Chapter 3: Labeling (Triple-Barrier, Meta-Labeling)
#   - Chapter 4: Sample Weights (Uniqueness)
# - See [`04_minimum_favorable_adverse_excursion`](04_minimum_favorable_adverse_excursion.ipynb) for empirical barrier calibration
