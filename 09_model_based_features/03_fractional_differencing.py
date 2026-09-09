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
# # Fractional Differencing
#
# **Chapter 9 | Section 9.1**
#
# **Docker image**: `ml4t`
#
# Differencing a price series once makes it stationary and throws away the level. That is
# a real loss: whether a price is near the top or the bottom of its recent range is
# information, and a return series has none of it. **Fractional differencing** takes the
# difference a fractional number of times, which is enough to make the series stationary
# while leaving part of the level intact.
#
# **Learning objectives**
#
# - Explain what differencing a series a fractional number of times means, and read the
#   weights that do it.
# - Apply the transform across a range of the fractional order and read the trade it makes:
#   how much of the original level is left against how strongly the result tests
#   stationary.
# - Say how many observations at the start of a sample the transform costs, and why the
#   answer depends on a convention you choose rather than on the data.
# - Pick the fractional order without searching the sample you will be tested on.
#
# **Book reference**
#
# Chapter 9, Section 9.1 (Diagnostics and stationarity features).
#
# **Prerequisites**
#
# `01_visual_diagnostics` for the ADF test and what stationarity means.

# %% [markdown]
# ## Setup

# %%
"""Fractional Differencing - achieve stationarity while preserving memory."""

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.engineer.features.fdiff import (
    fdiff_diagnostics,
    ffdiff,
    find_optimal_d,
    get_ffd_weights,
)
from ml4t.engineer.logging import setup_logging
from plotly.subplots import make_subplots
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller

from data import load_etfs
from utils.style import COLORS, show_plotly_with_alt

# Per-call timing notices from the feature library are about this machine, not the data.
setup_logging(level=logging.ERROR)
# The ADF helper inside the library runs KPSS alongside it on some paths, and a statistic
# past the ends of the KPSS lookup table raises this on every call.
warnings.filterwarnings(
    "ignore",
    message="The test statistic is outside of the range of p-values",
    category=InterpolationWarning,
)

# %% tags=["parameters"]
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
FFD_THRESHOLD = 1e-4

# %% [markdown]
# ## What a fractional difference is
#
# Differencing once replaces $x_t$ with $x_t - x_{t-1}$. Write that as a weighted sum of
# the history, $1 \cdot x_t - 1 \cdot x_{t-1}$, and the weights are $[1, -1, 0, 0, \dots]$.
# Differencing twice gives $[1, -2, 1, 0, \dots]$. The **fractional** difference of order
# $d$ uses the weights the binomial expansion of $(1 - L)^d$ produces for a
# non-integer $d$, where $L$ is the operator that shifts a series back one period:
#
# $$w_0 = 1, \qquad w_k = -w_{k-1}\,\frac{d - k + 1}{k}$$
#
# For $d = 1$ this terminates after two terms and reproduces the ordinary difference. For
# $d$ between zero and one it never terminates. After $w_0 = 1$ every weight is negative
# and decays towards zero, so the transform is today's value minus a decaying weighted
# average of the entire past. At $d = 1$ that average reduces to yesterday alone,
# which is the ordinary difference; below one it spreads over hundreds of sessions, and
# that spread is the memory ordinary differencing discards.
#
# In practice the tail is cut where the weights become negligible. `FFD_THRESHOLD` is
# where: a weight smaller than this in magnitude is dropped, and the number of weights
# left is the width of the fixed window the transform applies. The threshold therefore
# decides both how faithful the transform is and how much history each output value
# needs, which is the subject of two sections below.

# %% [markdown]
# ## The series
#
# One ETF panel, nine years of daily closes, and the transform is applied to **log**
# prices throughout. Logs matter because the filter is linear: on log prices its output is
# a weighted combination of log returns plus a small multiple of the log level, where a
# multiple of the *price* would scale with the price itself. The residual level term does
# not vanish, and it is the subject of the weight-sum section below: multiplying every
# price by a constant shifts every output by the weight sum times the log of that
# constant. Reduced dependence on the price scale, not independence from it.

# %%
all_etfs = load_etfs()


def load_etf(symbol: str) -> pl.DataFrame:
    """One symbol from the panel, inside the requested window, in session order."""
    return (
        all_etfs.filter(pl.col("symbol") == symbol)
        .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
        .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
        .sort("timestamp")
    )


spy = load_etf("SPY")
log_prices = spy["close"].log()
print(f"SPY: {spy.height:,} sessions ({spy['timestamp'].min()} to {spy['timestamp'].max()})")

# %% [markdown]
# ## The weights, and what the truncation leaves behind
#
# Every weight sequence starts at $w_0 = 1$, so the differences between orders are all in
# the tail. Plotting the magnitude of the weights on a log scale, with the first one
# omitted because it is the same for every order, shows the decay rate directly.
#
# Read it in two parts. At lag one the weight is exactly $-d$, so a higher order applies
# the larger immediate correction and starts above the others. Every pair of lines then
# crosses, because a lower order decays more slowly and its weights end up larger at
# distant lags, but they cross at very different places: widely separated orders swap
# within the first ten or twenty lags, while two neighbouring low orders stay close and
# swap only after hundreds. The end of each line is where its window closes, and that
# ordering does not reverse anywhere.

# %%
D_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def sequential_shades(start: str, stop: str, n: int) -> list[str]:
    """*n* colors interpolated between two palette entries, for an ordered variable."""
    low = np.array([int(start[i : i + 2], 16) for i in (1, 3, 5)])
    high = np.array([int(stop[i : i + 2], 16) for i in (1, 3, 5)])
    return [
        "#{:02x}{:02x}{:02x}".format(*(low + (high - low) * t).astype(int))
        for t in np.linspace(0, 1, n)
    ]


d_shades = sequential_shades(COLORS["blue"], COLORS["copper"], len(D_GRID))

fig = go.Figure()
for d, color in zip(D_GRID, d_shades):
    weights = get_ffd_weights(d, threshold=FFD_THRESHOLD)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(weights))),
            y=np.abs(weights[1:]),
            mode="lines",
            name=f"d={d}",
            line=dict(color=color, width=2),
        )
    )

fig.update_layout(
    title="A lower order decays more slowly, so its window reaches further back",
    xaxis_title="Lag in sessions",
    yaxis_title="Weight magnitude, log scale",
    yaxis=dict(type="log", exponentformat="power"),
)
show_plotly_with_alt(
    fig,
    "Six lines on a log vertical axis, one per fractional order, each showing the "
    "magnitude of the weights against the lag they apply to. At the first lag the highest "
    "fractional order sits highest, because that weight is the order itself. The lines "
    "then cross, the widely separated orders within the first tens of lags and the "
    "neighbouring low orders only far to the right, after which the lower orders lie "
    "above. Each line ends where its weights fall under the truncation threshold, and the "
    "lower the order the further right that is.",
)

# %% [markdown]
# The two columns below are what the truncation costs. The window width is how many
# sessions of history each output value needs. The weight sum is what a constant input
# would come out as: for an untruncated fractional difference of any positive order the
# weights sum to zero, so a constant maps to zero and the level is removed entirely.
# Truncation leaves a small positive remainder, and the transformed series therefore
# carries that fraction of the level on top of the differenced fluctuation. At a low order
# the remainder is large, which is the same fact as "a low order preserves memory", seen
# from the other side.

# %%
weight_table = pd.DataFrame(
    [
        {
            "d": d,
            "window_sessions": len(get_ffd_weights(d, threshold=FFD_THRESHOLD)),
            "weight_sum": get_ffd_weights(d, threshold=FFD_THRESHOLD).sum(),
        }
        for d in D_GRID
    ]
)
display(weight_table)

# %% [markdown]
# ## The boundary convention, and the observations it costs
#
# `ffdiff` is **boundary-partial**: near the start of the sample, where the full window of
# history does not exist yet, it applies whatever weights it can and returns a value
# anyway. Nothing is null and the row count is preserved, which is convenient and means
# the earliest values were produced by a shorter filter than the later ones.
#
# The **full-window** convention (Lopez de Prado, 2018) treats those rows as unavailable.
# It is the one to use for a feature, because a column whose first few hundred values were
# computed a different way is a column with a silent regime change at its start.
#
# Imposing it is one line: the first `width - 1` rows, where `width` is the number of
# weights, do not have a full window and are set to null. The count of those rows is the
# **sample loss**, and it is reported everywhere below because it is the price of the
# order chosen.


# %%
def ffd_full_window(series: pl.Series, d: float, threshold: float = FFD_THRESHOLD) -> dict:
    """Fractional difference under the full-window convention, with its diagnostics."""
    width = len(get_ffd_weights(d, threshold=threshold))
    values = ffdiff(series, d=d, threshold=threshold).to_numpy().copy()

    valid = np.zeros(len(values), dtype=bool)
    valid[width - 1 :] = True
    valid &= ~np.isnan(values)
    values[~valid] = np.nan

    # Polars keeps NaN and null distinct and `drop_nulls` drops only the second, so the
    # warmup is converted to null for every downstream filter to agree on.
    return {
        "transformed": pl.Series(series.name, values).fill_nan(None),
        "valid": pl.Series("valid", valid),
        "sample_loss": int((~valid).sum()),
        "d": d,
        "window_sessions": width,
    }


# %% [markdown]
# ## Reading the trade across the grid
#
# Each row applies one order to the SPY log price and reports three things: how many
# observations the warmup cost, how strongly the result rejects a unit root, and how much
# of the original level survived, measured as the correlation between the transformed
# series and the log price it came from.

# %%
grid_rows = []
for d in D_GRID:
    result = ffd_full_window(log_prices, d=d)
    valid = result["valid"].to_numpy()
    transformed = result["transformed"].to_numpy()[valid]

    grid_rows.append(
        {
            "d": d,
            "window_sessions": result["window_sessions"],
            "sample_loss": result["sample_loss"],
            "sample_loss_pct": 100 * result["sample_loss"] / len(log_prices),
            "adf_pval": adfuller(transformed, autolag="AIC")[1],
            "corr_with_level": np.corrcoef(log_prices.to_numpy()[valid], transformed)[0, 1],
        }
    )

grid_df = pd.DataFrame(grid_rows)
grid_df["stationary"] = grid_df["adf_pval"] < 0.05
display(grid_df)

# %%
first_stationary = grid_df.loc[grid_df["stationary"], "d"].min()
at_first = grid_df.loc[grid_df["d"] == first_stationary].iloc[0]
print(
    f"Smallest order on the grid that rejects a unit root: d = {first_stationary}, "
    f"keeping correlation {at_first['corr_with_level']:.2f} with the log price "
    f"at a cost of {at_first['sample_loss']} warmup sessions"
)

# %% [markdown]
# Three things move together down that table, and they are the whole subject.
#
# The correlation with the level falls as the order rises: more differencing, less memory.
# The ADF p-value falls with it, because the part of the series that carries the memory is
# the part that wanders. And the sample loss falls too, which is the direction that
# surprises people: a *lower* order needs a *wider* window, because its weights take
# longer to fall under the threshold, so keeping memory is paid for twice, once in
# stationarity and once in observations.
#
# The grid brackets the crossing deliberately. Orders below it do not reject a unit root
# and orders above it do, and the row where that changes is the one to read against the
# correlation column: it is the most memory this series will give up while still testing
# stationary.

# %% [markdown]
# ## Choosing the order without searching
#
# The obvious next move is to search for the smallest order that passes on this sample.
# That is a selection made on the same data the model will be evaluated on, and it makes
# the transform a fitted object with all the look-ahead that implies.
#
# The alternative is a **fixed order per asset class**, set from what those series are
# known to be like and held constant. It is not optimal for any one symbol and it is not
# estimated from anything, which is what makes it safe to apply across a panel and across
# time. Persistent series get a higher order because they need more differencing; series
# that already mean-revert get less.

# %%
ASSET_CLASS_D = {
    "equities": 0.4,
    "fixed_income": 0.5,
    "crypto": 0.5,
    "commodities": 0.4,
    "fx": 0.35,
}

display(
    pd.DataFrame(
        [
            {"asset class": name, "d": d, "why": reason}
            for (name, d), reason in zip(
                ASSET_CLASS_D.items(),
                [
                    "moderate persistence in the level",
                    "rates trend for years at a time",
                    "strong trending, short history",
                    "similar persistence to equities",
                    "levels mean-revert, so less differencing is needed",
                ],
            )
        ]
    )
)

# %% [markdown]
# ## Across a panel
#
# Applying the fixed order to seven ETFs shows what a no-search rule costs. Every symbol
# in a class gets the same order, so every symbol in a class loses the same number of
# warmup sessions, and some symbols will not test stationary at that order. That is the
# trade being made, not a failure of the rule.

# %%
ETF_ASSETS = {
    "SPY": "equities",
    "QQQ": "equities",
    "IWM": "equities",
    "TLT": "fixed_income",
    "GLD": "commodities",
    "EFA": "equities",
    "EEM": "equities",
}

panel_rows = []
for symbol, asset_class in ETF_ASSETS.items():
    data = load_etf(symbol)
    if data.height < 100:
        continue
    d = ASSET_CLASS_D[asset_class]
    result = ffd_full_window(data["close"].log(), d=d)
    transformed = result["transformed"].drop_nulls().to_numpy()
    panel_rows.append(
        {
            "symbol": symbol,
            "asset_class": asset_class,
            "d": d,
            "sample_loss": result["sample_loss"],
            "sample_loss_pct": 100 * result["sample_loss"] / data.height,
            "adf_pval": adfuller(transformed, autolag="AIC")[1],
        }
    )

panel_df = pd.DataFrame(panel_rows)
panel_df["stationary"] = panel_df["adf_pval"] < 0.05
display(panel_df)

# %%
not_stationary = panel_df.loc[~panel_df["stationary"], "symbol"].tolist()
print(
    f"Symbols still testing non-stationary at their class order: "
    f"{', '.join(not_stationary) if not_stationary else 'none'} "
    f"({len(not_stationary)} of {len(panel_df)})"
)

# %% [markdown]
# The table above runs its test on each symbol's whole sample, so it describes the panel
# and must not select for it. A symbol that fails may be moved one step up the grid, but
# the diagnostic that triggers the move has to be computed on training observations alone.
# Run on the full sample it is the evaluation period choosing the parameter, and a single
# predetermined step is still a step the test data asked for.

# %% [markdown]
# ## The output a model reads
#
# The feature table pairs the transformed column with its validity mask. Downstream code
# filters on the mask, which keeps the warmup out of every model that reads the column
# without anyone having to carry the count separately.

# %%
D_EQUITIES = ASSET_CLASS_D["equities"]
equity_result = ffd_full_window(log_prices, d=D_EQUITIES)

spy_features = spy.select(["timestamp", "close"]).with_columns(
    log_close=pl.col("close").log(),
    return_1d=pl.col("close").pct_change(),
    ffd=equity_result["transformed"],
    ffd_valid=equity_result["valid"],
)

print(
    f"Rows: {spy_features.height}, valid: {equity_result['valid'].sum()}, "
    f"warmup dropped: {equity_result['sample_loss']}"
)
display(spy_features.filter(pl.col("ffd_valid")).tail(10))

# %% [markdown]
# Compare the `ffd` column against `log_close` in those rows and the weight sum from the
# earlier table is visible directly: the transformed value sits at roughly that fraction
# of the log price, with the differenced fluctuation on top of it. This is what
# "preserving memory" means concretely, and it is also the reason the transformed series
# is only approximately stationary: the surviving fraction of the level drifts with the
# level.

# %% [markdown]
# ## What the three transforms look like
#
# The same series undifferenced, differenced once, and differenced fractionally. The last
# 500 sessions, so the shapes are visible rather than compressed.

# %%
DISPLAY_SESSIONS = 500

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Log price, no differencing",
        "Simple return, differenced once",
        f"Fractional difference at d={D_EQUITIES}",
    ],
    vertical_spacing=0.08,
)

tail = slice(-DISPLAY_SESSIONS, None)
sessions = spy["timestamp"].to_list()[tail]
for row, values in enumerate(
    [
        log_prices.to_numpy()[tail],
        spy["close"].pct_change().to_numpy()[tail],
        equity_result["transformed"].to_numpy()[tail],
    ],
    start=1,
):
    fig.add_trace(
        go.Scatter(x=sessions, y=values, mode="lines", line=dict(color=COLORS["blue"])),
        row=row,
        col=1,
    )

fig.update_yaxes(title_text="Log price", row=1, col=1)
fig.update_yaxes(title_text="Return", row=2, col=1)
fig.update_yaxes(title_text="FFD value", row=3, col=1)
fig.update_xaxes(title_text="Session", row=3, col=1)
fig.update_layout(
    height=600,
    showlegend=False,
    title="The fractional difference keeps the slow movement returns discard",
)
show_plotly_with_alt(
    fig,
    "Three stacked panels over the last 500 sessions. The top panel, the log price, "
    "rises and falls across a wide range. The middle panel, the daily return, is a band "
    "of noise around zero with no visible trend. The bottom panel, the fractional "
    "difference, oscillates like the return panel but around a level that drifts with "
    "the shape of the top panel.",
)

# %% [markdown]
# ## If you must search, search forward
#
# Where a fixed order is not acceptable, the search has to be confined to data the model
# is not evaluated on. The function below takes a training cut-off, searches only inside
# it, and returns the smallest order on the grid that rejects a unit root there. That is
# the Lopez de Prado convention: take the least differencing the diagnostic will accept,
# because everything past it is memory given away for nothing.


# %%
SEARCH_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TRAIN_FRACTION = 0.8


def search_d_on_training(series: pl.Series, train_end: int) -> dict:
    """The smallest order on the grid that rejects a unit root inside the training cut."""
    train = series.head(train_end)
    for d in sorted(SEARCH_GRID):
        result = ffd_full_window(train, d=d)
        valid = result["valid"].to_numpy()
        transformed = result["transformed"].to_numpy()[valid]
        if len(transformed) < 50:
            continue
        adf_pval = adfuller(transformed, autolag="AIC")[1]
        if adf_pval < 0.05:
            return {
                "selected_d": d,
                "train_adf_pval": adf_pval,
                "train_corr": np.corrcoef(train.to_numpy()[valid], transformed)[0, 1],
            }
    # Nothing on the grid was stationary; fall back to the ordinary first difference.
    return {"selected_d": 1.0, "train_adf_pval": float("nan"), "train_corr": 0.0}


# %%
train_end = int(spy.height * TRAIN_FRACTION)
search = search_d_on_training(log_prices, train_end)

print(f"Training: {spy['timestamp'][0]} to {spy['timestamp'][train_end - 1]}")
print(f"Test:     {spy['timestamp'][train_end]} to {spy['timestamp'][-1]}")
print(f"Selected d on training data only: {search['selected_d']}")
print(f"  training ADF p-value: {search['train_adf_pval']:.4f}")
print(f"  training correlation with the level: {search['train_corr']:.4f}")

test_result = ffd_full_window(log_prices.tail(spy.height - train_end), d=search["selected_d"])
test_values = test_result["transformed"].drop_nulls().to_numpy()
if len(test_values) > 50:
    test_pval = adfuller(test_values, autolag="AIC")[1]
    print(f"Test ADF p-value at the selected d: {test_pval:.4f}")
print(f"Smallest stationary order over the full sample, for comparison: {first_stationary}")

# %% [markdown]
# The order selected on the training cut need not match the one the full sample would
# give, and on this sample it does not. Neither outcome tells you anything on its own:
# two searches over the same grid can land on the same point for perfectly good reasons.
# What makes a selection safe is which observations the procedure was allowed to read, not
# whether its answer happens to agree with a search that read more.

# %% [markdown]
# ## The library helpers, and the convention they use
#
# `find_optimal_d` searches a range for the smallest order that passes, and
# `fdiff_diagnostics` reports the ADF result, the correlation and the weight count at a
# given order. Both call the boundary-partial transform directly, with no validity mask,
# so their ADF test reads a series whose early values came from a partial filter.
#
# That is not a different opinion about the same question; it is a different question, and
# on this sample the difference is large enough to change what you would do. Compare the
# window width the helper's answer implies against the length of the sample it was
# measured on.

# %%
optimal = find_optimal_d(log_prices, d_range=(0.0, 1.0), step=0.05)
diagnostics = fdiff_diagnostics(log_prices, d=optimal["optimal_d"])

print(f"find_optimal_d selected d = {optimal['optimal_d']:.2f}")
print(f"  ADF p-value: {optimal['adf_pvalue']:.4f}")
print(f"  correlation with the level: {optimal['correlation']:.4f}")
print(f"  window it implies: {diagnostics['n_weights']} sessions")
print(f"  sessions in the sample: {spy.height}")
print(f"  weight sum: {diagnostics['weight_sum']:.4f}")

# %% [markdown]
# Read the last three lines together. The order the helper selected needs a window longer
# than the sample it was selected on, so under the full-window convention not one
# observation would have a complete window and the transform is undefined on this data.
# Every value it tested was a partial application of a filter that never fits. The helper
# is doing what it says; the convention it assumes is the one that has to be checked
# before its answer is used.

# %% [markdown]
# ## Three distributions
#
# The last comparison is what each transform does to the distribution of values, which is
# a different question from what it does to the path.

# %%
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Log price",
        "Simple return",
        f"Fractional difference, d={D_EQUITIES}",
    ],
)

for column, values in enumerate(
    [
        log_prices.drop_nulls().to_numpy(),
        spy["close"].pct_change().drop_nulls().to_numpy(),
        equity_result["transformed"].drop_nulls().to_numpy(),
    ],
    start=1,
):
    fig.add_trace(
        go.Histogram(x=values, nbinsx=50, marker=dict(color=COLORS["blue"])), row=1, col=column
    )

fig.add_vline(x=0, line=dict(color=COLORS["neutral"], dash="dash", width=1), row=1, col=2)
fig.update_xaxes(title_text="Log price", row=1, col=1)
fig.update_xaxes(title_text="Return", row=1, col=2)
fig.update_xaxes(title_text="FFD value", row=1, col=3)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_layout(
    height=350,
    showlegend=False,
    title="Differencing narrows the distribution; the fractional order stops part way",
)
show_plotly_with_alt(
    fig,
    "Three histograms side by side, each on its own horizontal scale. The log price "
    "spreads across more than a full unit with several separate humps. The simple return "
    "is one narrow spike centred on zero, marked by a dashed reference line. The "
    "fractional difference spans about the same width as the return but is centred near "
    "the level shown in the weight table rather than on zero, and its shape is broader "
    "and less peaked.",
)

# %% [markdown]
# The log price is spread out and multimodal, which is what a wandering level looks like
# as a histogram. The return is a narrow spike at zero: stationary and carrying nothing
# about where the price was. The fractional difference spans roughly the same width as
# the return, so it is comparable in scale, and it sits away from zero because of the
# surviving fraction of the level from the weight table.

# %% [markdown]
# ## Key takeaways
#
# 1. **The order controls one trade and pays for it twice.** A lower order keeps more of
#    the level and tests stationary less strongly, and it also needs a wider window, so it
#    costs more warmup observations.
# 2. **The boundary convention is a choice that changes the answer.** The
#    boundary-partial form keeps every row and computes the early ones with a shorter
#    filter; the full-window form drops them. Diagnostics computed under one convention do
#    not transfer to the other, which is why the library helper here selects an order the
#    full-window convention cannot use at all.
# 3. **Report the sample loss with the feature.** It is the number of rows at the start of
#    every series that a model must not read, and it changes with the order and the
#    truncation threshold.
# 4. **Fix the order rather than searching for it.** A fixed order per asset class is not
#    optimal anywhere and is not estimated from anything. Where a search is unavoidable,
#    confine it to a training cut and take the smallest order that passes.
# 5. **The transform is not exactly stationary.** Truncation leaves the weights summing to
#    a small positive number, so the output carries that fraction of the level and drifts
#    with it. It passes the test; it is not free of the level.
#
# **Known limitations.** The ADF test decides stationarity here, and it is one test with
# known low power against a slowly mean-reverting alternative, so an order it accepts is
# not established as sufficient. The asset-class orders are conventions rather than
# measurements. And the panel section applies one order per class over one nine-year
# window; a longer sample containing a different volatility regime can move which symbols
# pass.
#
# **Next**: `04_kalman_filter` for extracting a latent level from a noisy series, and
# `05_spectral_features` for describing a series by its frequencies.
