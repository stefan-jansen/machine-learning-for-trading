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
# # ARIMA as a Feature Extractor
#
# **Chapter 9 | Section 9.3**
#
# **Docker image**: `ml4t`
#
# ARIMA is normally introduced as a forecasting model, and judged by whether its forecasts
# are accurate. This notebook uses it differently: the forecast is not the answer, it is a
# **column**, one number per session that a later model reads alongside everything else.
# That changes what has to be checked. An accurate forecast is not required; what is
# required is that the column is computable on the day it is stamped, that it varies over
# time, and that what it carries is not already in the other columns.
#
# **Learning objectives**
#
# - Read an ACF and a PACF to propose an order for a model of this family, and check the
#   proposal against an information criterion over a grid.
# - Explain why a forecast made several steps ahead settles at a constant, and why that
#   makes it useless as a column however good the model is.
# - Build the one-step-ahead version that does vary, by filtering a series under parameters
#   fitted on an earlier block, and say what makes that construction causal.
# - Measure what the column is worth on one asset and then on a hundred, and read a
#   distribution of results rather than one number.
#
# **Book reference**
#
# Chapter 9, Section 9.3 (Volatility Features).
#
# **Prerequisites**
#
# `01_visual_diagnostics` for stationarity and for the ACF and the Ljung-Box test.

# %% [markdown]
# ## Setup

# %%
"""ARIMA as a feature extractor - one-step forecasts as a column, not as an answer."""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.autocorrelation import analyze_autocorrelation
from ml4t.diagnostic.metrics import pooled_ic
from plotly.subplots import make_subplots
from scipy.stats import ConstantInputWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

from case_studies.utils.temporal import arima_one_step_forecast
from data import load_etfs
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, show_plotly_with_alt

# statsmodels warns once per fit that a date index carries no declared frequency. Trading
# days have no frequency to declare, the models here index by position, and the notice
# repeats once per symbol.
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.base.tsa_model")
# A constant forecast has no rank correlation with anything. The order that produces one is
# in the table below on purpose, showing NaN, and the notice repeats for every such call.
warnings.filterwarnings(
    "ignore", category=ConstantInputWarning, module="ml4t.diagnostic.metrics.ic"
)

# %% tags=["parameters"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-01"
TEST_START = "2024-01-01"
MAX_SYMBOLS = 0  # 0 reads every symbol in the panel
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## The data
#
# The ETF panel, split at `TEST_START` into a block the models are fitted on and a block
# they are only ever read over. One symbol carries the walk-through; the whole panel
# carries the measurement at the end, because one symbol's result is one draw.

# %%
DEMONSTRATION_SYMBOL = "SPY"

etfs = (
    load_etfs()
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
)
symbols = etfs["symbol"].unique().sort().to_list()
if MAX_SYMBOLS > 0:
    symbols = symbols[:MAX_SYMBOLS]


def symbol_frame(symbol: str) -> pd.DataFrame:
    """One symbol's closes and returns, indexed by session, as statsmodels wants them."""
    frame = (
        etfs.filter(pl.col("symbol") == symbol)
        .sort("timestamp")
        .with_columns(returns=pl.col("close").pct_change())
        .drop_nulls()
        .select(["timestamp", "close", "returns"])
        .to_pandas()
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.set_index("timestamp")


spy = symbol_frame(DEMONSTRATION_SYMBOL)
split_at = pd.Timestamp(TEST_START)
train, test = spy[spy.index < split_at], spy[spy.index >= split_at]

print(f"Panel: {len(symbols)} symbols, {etfs.height:,} rows")
print(f"{DEMONSTRATION_SYMBOL}: {len(train):,} training sessions to {train.index.max().date()}")
print(f"{DEMONSTRATION_SYMBOL}: {len(test):,} test sessions from {test.index.min().date()}")

# %% [markdown]
# ## What the model is fitted to
#
# ARIMA has three orders. The **autoregressive** order $p$ is how many past values of the
# series enter; the **moving average** order $q$ is how many past shocks enter; the
# **integration** order $d$ is how many times the series is differenced before either of
# those applies. Differencing is what makes the model applicable to a series that wanders,
# and it is the reason the family is usually shown on prices.
#
# Here the model is fitted to returns, which are already the first difference of the log
# price, so $d = 0$ throughout. The ADF test below is the check that this is right; the
# reasoning behind it is in `01_visual_diagnostics`.

# %%
stationarity = pd.DataFrame(
    [
        {
            "series": name,
            "ADF statistic": adfuller(values.dropna(), autolag="AIC")[0],
            "ADF p-value": adfuller(values.dropna(), autolag="AIC")[1],
        }
        for name, values in [("close", spy["close"]), ("returns", spy["returns"])]
    ]
)
display(stationarity)

# %% [markdown]
# ## What the correlogram proposes
#
# The autocorrelation function suggests the moving-average order and the partial
# autocorrelation function suggests the autoregressive order: the lag at which each drops
# inside its confidence band is the order it proposes. On daily returns both are inside the
# band nearly everywhere, which proposes a very low order and is the first sign of how much
# there is here to model.

# %%
ACF_LAGS = 20


def correlogram(series: pd.Series, claim: str) -> go.Figure:
    """ACF and PACF side by side, with the band inside which a bar is indistinguishable from zero."""
    values = series.dropna()
    band = 1.96 / np.sqrt(len(values))

    figure = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    for column, estimates in enumerate(
        [acf(values, nlags=ACF_LAGS), pacf(values, nlags=ACF_LAGS)], start=1
    ):
        figure.add_trace(
            go.Bar(x=list(range(len(estimates))), y=estimates, marker_color=COLORS["blue"]),
            row=1,
            col=column,
        )
        for edge in (band, -band):
            figure.add_hline(
                y=edge, line_dash="dash", line_color=COLORS["neutral"], row=1, col=column
            )
        figure.update_xaxes(title_text="Lag, sessions", row=1, col=column)
    figure.update_yaxes(title_text="Correlation", row=1, col=1)
    figure.update_layout(title_text=claim, showlegend=False, height=340)
    return figure


show_plotly_with_alt(
    correlogram(
        spy["returns"],
        f"Daily {DEMONSTRATION_SYMBOL} returns carry almost no linear dependence"
        "<br><sup>Bars inside the dashed band are indistinguishable from zero at the five "
        "percent level</sup>",
    ),
    "Two bar charts side by side, the autocorrelation function and the partial "
    "autocorrelation function of daily returns at lags zero to twenty. Both start at one at "
    "lag zero and then sit close to zero, mostly inside the dashed confidence band, with a "
    "handful of bars reaching just outside it.",
)

# %%
suggestion = analyze_autocorrelation(spy["returns"].dropna().to_numpy())
print(f"Order suggested from the correlogram: {suggestion.suggested_arima_order}")

# %% [markdown]
# ## Why a multi-step forecast cannot be a feature
#
# The obvious way to produce a forecast for a test period is to fit once and ask for as
# many steps as the period is long. For a stationary model that produces something useless,
# and the reason is structural rather than a defect of the fit: with no new observations to
# condition on, the forecast at step $h$ is the model's expectation given the last observed
# value, and for a stationary process that expectation decays to the unconditional mean
# geometrically. Within a handful of steps every forecast is the same number.
#
# How fast that happens depends on the fitted persistence rather than on stationarity
# alone, so the demonstration below prints the first and last forecasts and the spread
# across the whole test period, against the spread of the returns the column is meant to
# track. A column whose variation is three orders of magnitude below its target's carries
# essentially nothing about it, and one that reaches an exactly constant vector has no rank
# correlation at all.

# %%
static_fit = ARIMA(train["returns"], order=(1, 0, 0)).fit()
static_forecast = static_fit.forecast(steps=len(test)).to_numpy()

print(f"First five forecasts: {np.round(static_forecast[:5] * 100, 4)} percent")
print(f"Last five forecasts:  {np.round(static_forecast[-5:] * 100, 4)} percent")
print(f"Spread across the whole test period: {static_forecast.std() * 100:.2e} percent")
print(f"Spread of the returns it is meant to track: {test['returns'].std() * 100:.4f} percent")

# %% [markdown]
# ## The forecast that does vary
#
# The version that works forecasts one step at a time: at each session, the model conditions
# on everything observed up to that session and predicts the next one. There are two ways to
# do that and both are causal. **Filtering** holds the parameters where the training block
# put them and runs the state recursion forward, so no test observation influences a
# parameter and the cost is one fit. **Refitting on an expanding window** re-estimates at
# each step from the observations that precede it, which costs one fit per session and lets
# the parameters follow the data. This notebook filters, because the cost of refitting a
# hundred symbols daily is real and the parameters of a model this small barely move; a
# series whose dynamics genuinely change would be a reason to pay it.
#
# `arima_one_step_forecast` is the shared implementation, and it exists because two
# neighbouring calls do something else. `apply(endog, refit=True)` re-estimates on the array
# it is handed, which would fit every prediction on the block it is emitted over;
# `forecast(h)` continues past the end of the data rather than filtering across it, so it
# returns as many values as you asked for rather than one per row. The helper passes
# `refit=False` explicitly, which is also the installed default, and then checks that the
# parameters did not in fact move, so a changed default upstream fails loudly rather than
# turning the column into an in-sample fit.

# %%
full_returns = spy["returns"].to_numpy()
one_step = arima_one_step_forecast(static_fit, full_returns)
one_step_test = one_step[len(train) :]

print(
    f"Spread of the one-step forecasts over the test period: {one_step_test.std() * 100:.4f} percent"
)
print(
    f"Spread of the multi-step forecasts, for comparison: {static_forecast.std() * 100:.2e} percent"
)

# %%
figure = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "The multi-step forecast settles; the one-step forecast tracks",
        "One is a spike, the other has a distribution",
    ),
    vertical_spacing=0.12,
)

sessions = test.index[: min(120, len(test))]
for name, values, color in [
    ("Realized return", test["returns"].to_numpy(), COLORS["neutral"]),
    ("One-step forecast", one_step_test, COLORS["blue"]),
    ("Multi-step forecast", static_forecast, COLORS["copper"]),
]:
    figure.add_trace(
        go.Scatter(x=sessions, y=values[: len(sessions)] * 100, name=name, line=dict(color=color)),
        row=1,
        col=1,
    )

for name, values, color in [
    ("Realized return", test["returns"].to_numpy(), COLORS["neutral"]),
    ("One-step forecast", one_step_test, COLORS["blue"]),
]:
    figure.add_trace(
        go.Histogram(
            x=values * 100, name=name, opacity=0.55, nbinsx=50, marker_color=color, showlegend=False
        ),
        row=2,
        col=1,
    )

figure.update_xaxes(title_text="Session", row=1, col=1)
figure.update_yaxes(title_text="Percent", row=1, col=1)
figure.update_xaxes(title_text="Daily return, percent", row=2, col=1)
figure.update_yaxes(title_text="Count", row=2, col=1)
figure.update_layout(
    height=560, barmode="overlay", title_text="Only one of the two forecasts is a column"
)
show_plotly_with_alt(
    figure,
    "Two stacked panels. The top draws the realized daily returns against two forecasts "
    "over the first months of the test period: the multi-step forecast is a flat line, "
    "while the one-step forecast moves with the returns at a much smaller amplitude. The "
    "bottom overlays the histogram of realized returns with the histogram of the one-step "
    "forecasts, which is far narrower and centred near zero.",
)

# %% [markdown]
# ## Choosing the order
#
# The grid below fits every combination of a small autoregressive and moving-average order
# and reports the **Akaike information criterion**, which scores a fit by its likelihood and
# charges it for each parameter. The lowest score is the usual choice.
#
# The grid also records whether the optimiser converged. That notice is a
# `ConvergenceWarning`, and it is worth catching rather than silencing: a criterion computed
# from a fit that did not converge is a number the optimiser stopped next to, not the
# maximum it was looking for, and comparing it against a converged one ranks them on
# different things.

# %%
AR_ORDERS, MA_ORDERS = range(4), range(3)


def fit_recording_convergence(series: pd.Series, order: tuple[int, int, int]):
    """Fit, and report whether the optimiser said it converged rather than printing it."""
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = ARIMA(series, order=order).fit()
    converged = not any(issubclass(entry.category, ConvergenceWarning) for entry in raised)
    return fitted, converged


grid_rows = []
for p in AR_ORDERS:
    for q in MA_ORDERS:
        try:
            fitted, converged = fit_recording_convergence(train["returns"], (p, 0, q))
        except (ValueError, np.linalg.LinAlgError):
            continue
        grid_rows.append(
            {"p": p, "q": q, "AIC": fitted.aic, "BIC": fitted.bic, "converged": converged}
        )

grid = pd.DataFrame(grid_rows).sort_values("AIC")
display(grid.head())
print(f"Grid fits that did not converge: {(~grid['converged']).sum()} of {len(grid)}")

# %% [markdown]
# ## What each order is worth as a column
#
# Every converged order is refitted on the training block and filtered one step at a time
# across the whole series, and the resulting column is scored over the test block against
# the return it was trying to predict. The **information coefficient** is the rank
# correlation between the two.
#
# A rank correlation needs a column that varies, so the multi-step forecast has no entry
# here: its rank correlation is undefined by construction, which is the point the previous
# section made.

# %%
scored_rows = []
for row in grid.itertuples():
    if not row.converged:
        continue
    order = (row.p, 0, row.q)
    try:
        fitted, _ = fit_recording_convergence(train["returns"], order)
        column = arima_one_step_forecast(fitted, full_returns)[len(train) :]
    except (ValueError, np.linalg.LinAlgError):
        continue
    scored_rows.append(
        {
            "order": f"ARIMA{order}",
            "AIC": row.AIC,
            "information coefficient": pooled_ic(column, test["returns"].to_numpy()),
            "RMSE": float(np.sqrt(np.mean((test["returns"].to_numpy() - column) ** 2))),
        }
    )

scored = pd.DataFrame(scored_rows).sort_values("information coefficient", ascending=False)
display(scored)

# %%
lowest_aic = scored.loc[scored["AIC"].idxmin()]
highest_ic = scored.iloc[0]
print(
    f"Lowest AIC:                  {lowest_aic['order']}, IC {lowest_aic['information coefficient']:+.4f}"
)
print(
    f"Highest information coefficient: {highest_ic['order']}, IC {highest_ic['information coefficient']:+.4f}"
)
print(f"RMSE across every order spans {scored['RMSE'].min():.6f} to {scored['RMSE'].max():.6f}")

# %% [markdown]
# The two criteria need not agree, and on this split they do not have to for a reason worth
# stating. The information criterion scores the fit on the training block, in likelihood;
# the information coefficient scores a column on the test block, in rank agreement. A model
# can win the first by fitting the training block's noise more closely and lose the second
# on the block it never saw. Selecting on the criterion and reporting the coefficient, as
# here, keeps that separation visible; selecting on the coefficient would be selecting on
# the block being reported.
#
# The RMSE column is the third thing to notice, and it is the sharpest of the three. Every
# forecast here is small next to the return it predicts, so the squared error is dominated
# by the return itself and barely notices which model produced the forecast. The order that
# scores lowest on RMSE is the one with no dynamics at all, whose forecast is a constant:
# predicting nothing is the least wrong thing to do when the thing being predicted is
# almost all noise. That order also has no information coefficient, because a rank
# correlation with a constant is undefined. Ranking these models on RMSE would be ranking
# them on the variance of the test block.

# %% [markdown]
# ## Across the panel
#
# One split on one symbol gives one information coefficient. Running the same construction
# on every symbol in the panel gives a distribution, and the distribution is what says
# whether the column carries anything.
#
# Each symbol is fitted on its own training block and filtered across its own series, so a
# symbol with a shorter history is handled the same way as one with a longer one.

# %%
PANEL_ORDER = (1, 0, 0)
MINIMUM_SESSIONS = 252


def one_step_column(symbol: str) -> dict:
    """Fit on the training block, filter across the whole series, score the test block."""
    frame = symbol_frame(symbol)
    if len(frame) < MINIMUM_SESSIONS:
        return {"symbol": symbol, "status": "too short", "information coefficient": np.nan}

    symbol_train = frame[frame.index < split_at]
    symbol_test = frame[frame.index >= split_at]
    if len(symbol_train) < 100 or len(symbol_test) < 10:
        return {"symbol": symbol, "status": "split too small", "information coefficient": np.nan}

    try:
        fitted, converged = fit_recording_convergence(symbol_train["returns"], PANEL_ORDER)
        column = arima_one_step_forecast(fitted, frame["returns"].to_numpy())
    except (ValueError, np.linalg.LinAlgError) as failure:
        return {
            "symbol": symbol,
            "status": type(failure).__name__,
            "information coefficient": np.nan,
        }

    return {
        "symbol": symbol,
        "status": "scored",
        "converged": converged,
        "test sessions": len(symbol_test),
        "information coefficient": pooled_ic(
            column[len(symbol_train) :], symbol_test["returns"].to_numpy()
        ),
    }


panel = pd.DataFrame([one_step_column(symbol) for symbol in symbols])
scored_panel = panel[panel["status"] == "scored"]

print(f"Symbols scored: {len(scored_panel)} of {len(panel)}")
print(f"Fits whose optimiser did not converge: {(~scored_panel['converged']).sum()}")
print(
    "Information coefficient across the panel: "
    f"median {scored_panel['information coefficient'].median():+.4f}, "
    f"quartiles {scored_panel['information coefficient'].quantile(0.25):+.4f} "
    f"to {scored_panel['information coefficient'].quantile(0.75):+.4f}"
)
print(f"Share above zero: {(scored_panel['information coefficient'] > 0).mean():.1%}")
print(
    "Median by convergence: "
    + ", ".join(
        f"{'converged' if converged else 'did not converge'} "
        f"{group['information coefficient'].median():+.4f} ({len(group)} symbols)"
        for converged, group in scored_panel.groupby("converged")
    )
)

# %%
figure = go.Figure(
    go.Histogram(x=scored_panel["information coefficient"], nbinsx=30, marker_color=COLORS["blue"])
)
figure.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
figure.update_layout(
    title_text=f"The panel's forecasts straddle zero at order {PANEL_ORDER}",
    xaxis_title="Information coefficient over the test block",
    yaxis_title="Symbols",
    height=340,
)
show_plotly_with_alt(
    figure,
    "A histogram of the information coefficient across every scored symbol in the panel, "
    "with a dashed line at zero. The distribution is broad and centred close to the line, "
    "with symbols on both sides and no clear separation from zero.",
)

# %% [markdown]
# The distribution is the answer, and it is a more useful one than a single symbol's number
# would have been. Its quartiles span zero while its median sits a little above it, which
# says the column is weakly and unreliably informative about direction rather than either
# useless or usable, and it puts the single-symbol result earlier inside the spread of what
# the panel produces rather than out on its own.
#
# Do not read the share above zero as a hundred votes. These are ETFs on overlapping
# universes, so their returns are highly correlated and so are their forecast errors; the
# number of independent observations behind that share is far smaller than the number of
# symbols, and no test here says how much smaller.
#
# The convergence split is the other thing to look at before believing the distribution.
# A sixth of these fits stopped without the optimiser reporting convergence, and their
# parameters are wherever it stopped rather than at a maximum. They are in the distribution
# above, and printing the two medians beside each other is the cheap check on whether that
# matters here. It does: the two medians are not the same, and the symbols whose fits did
# not converge sit closer to zero than the ones that did, so the panel's headline number is
# being pulled by fits that never reached a maximum. They stay in nonetheless, because
# removing them would select the panel on how easy each symbol was to fit, which is a
# property of the optimiser and not of the market. The honest report is both medians.
#
# Two properties of returns explain it, and both are the subject of what follows. The linear
# dependence a model of this family reads is nearly absent from daily returns, which the
# correlogram said at the start. And the variance of returns changes over time while this
# model assumes it does not, so the model is misspecified in the one dimension where daily
# returns are strongly predictable. That dimension is what `08_garch_volatility` models.

# %% [markdown]
# ## The features this notebook produces
#
# | Column | How it is built | Causal |
# |---|---|---|
# | one-step forecast | fitted on the training block, then filtered one step at a time across the series | yes |
# | fitted order | the grid's lowest-AIC order, chosen on the training block alone | yes |
# | residual | the realized return minus the one-step forecast, which is what a volatility model reads next | yes |
#
# The multi-step forecast is deliberately not on the list. It is constant after a few steps,
# so it carries no information at any session and no rank correlation is defined for it.

# %% [markdown]
# ## Key takeaways
#
# 1. **A forecast is a column, and a column has to vary.** A multi-step forecast from a
#    stationary model decays to the unconditional mean, which makes it constant, which makes
#    it useless whatever the model's accuracy.
# 2. **One-step-ahead is a horizon, not a parameter policy.** Filtering under fixed
#    parameters and refitting on an expanding window are both causal and differ in cost and
#    in whether the parameters follow the data. What is not causal is refitting on a window
#    that includes the value being predicted, which is what `apply(endog, refit=True)` on
#    the prediction block, or a whole-sample fit, does.
# 3. **An information criterion and an information coefficient measure different things on
#    different blocks**, and a model can lead on one and not the other. Select on the
#    criterion, which reads the training block, and report the coefficient.
# 4. **Do not silence a convergence warning.** A criterion from a fit that did not converge
#    is not comparable with one that did, and catching the warning turns it into a column
#    of the grid instead of noise the reader never sees.
# 5. **One symbol is one draw.** The panel's distribution of information coefficients
#    straddles zero, which is the finding; a single symbol's number sits inside that spread.
#
# **Known limitations.** One split at one date, so nothing here separates the model from the
# period it was tested in. The panel's symbols overlap heavily in what they hold, so its
# hundred results are far fewer than a hundred independent observations. And the whole
# notebook models the mean of returns, which is the part of a return series that is hardest
# to predict; the same family applied to a volatility proxy behaves quite differently, which
# is the next notebook's subject.
#
# **Next**: `08_garch_volatility` models the variance this notebook assumed was constant.
