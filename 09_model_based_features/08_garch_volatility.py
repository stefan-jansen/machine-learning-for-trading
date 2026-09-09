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
# # GARCH Volatility Features
#
# **Chapter 9 | Section 9.3**
#
# **Docker image**: `ml4t`
#
# The previous notebook fitted a model to the mean of returns and found almost nothing to
# predict. The variance is different: large moves arrive in bursts, so the size of today's
# move says a great deal about the size of tomorrow's. **GARCH** is the model of that, and
# what it produces is one number per session, an estimate of how volatile the next session
# is expected to be, which is exactly the shape of a feature.
#
# **Learning objectives**
#
# - Test whether a series has the property a volatility model assumes, before fitting one.
# - Read the two GARCH parameters as what they decide: how much a new move changes the
#   estimate, and how long the estimate remembers.
# - Extract the conditional volatility as a column and check whether the model captured
#   what it was fitted to capture.
# - Extend the model so that a fall raises the estimate more than a rise, and measure
#   whether it was worth the parameter.
# - Turn the estimate into a risk number, and check that number against what happened.
#
# **Book reference**
#
# Chapter 9, Section 9.3 (Volatility Features).
#
# **Prerequisites**
#
# `01_visual_diagnostics` for the ARCH-LM test and for what clustered volatility looks
# like. `07_arima_features` for the difference between a forecast and a column.

# %% [markdown]
# ## Setup

# %%
"""GARCH Volatility Features - conditional volatility, persistence, and asymmetry."""

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from arch import arch_model
from arch.univariate.base import DataScaleWarning
from IPython.display import display
from ml4t.diagnostic.evaluation.volatility import arch_lm_test
from ml4t.engineer.features.volatility import (
    ewma_volatility,
    garch_forecast,
    realized_volatility,
    volatility_of_volatility,
    volatility_percentile_rank,
)
from scipy.stats import norm, probplot
from statsmodels.graphics.tsaplots import plot_acf

from case_studies.utils.temporal import garch11_conditional_volatility
from data import load_etfs
from utils.style import COLORS, FIGSIZE, show_with_alt

# %% tags=["parameters"]
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
MAX_SYMBOLS = 0  # 0 reads every symbol in the panel
SEED = 42

# %% [markdown]
# ## Returns, and the scale they are measured in
#
# One detail decides several numbers below, so it is worth stating before anything is
# fitted. `arch` estimates by maximum likelihood over a variance, and a daily return
# expressed as a fraction has a variance around a hundredth of a percent, which is small
# enough that the optimiser works in a badly conditioned corner and says so. Multiplying
# returns by a hundred, so they are in percent, removes the problem and changes nothing
# about the model: the two parameters that matter are ratios and do not move, and the
# constant scales by the square of whatever factor was used.
#
# Every return in this notebook is therefore in **percent**, and every volatility that
# comes out of a fit is a percent-per-session standard deviation. Where an annualised
# figure is shown it is multiplied by the square root of the number of sessions in a year.

# %%
RETURN_SCALE = 100  # returns in percent, which is the scale `arch` is conditioned for
SESSIONS_PER_YEAR = 252
DEMONSTRATION_SYMBOL = "SPY"
MINIMUM_SESSIONS = 500

start, end = (datetime.strptime(value, "%Y-%m-%d").date() for value in (START_DATE, END_DATE))
etfs = (
    load_etfs()
    .filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    .sort("timestamp")
)

symbols = etfs["symbol"].unique().sort().to_list()
if MAX_SYMBOLS > 0:
    symbols = symbols[:MAX_SYMBOLS]


def symbol_returns(symbol: str) -> pd.Series | None:
    """Log returns in percent for one symbol, or None where the history is too short."""
    frame = (
        etfs.filter(pl.col("symbol") == symbol)
        .select(["timestamp", "close"])
        .with_columns(returns=pl.col("close").log().diff() * RETURN_SCALE)
        .drop_nulls()
    )
    if frame.height < MINIMUM_SESSIONS:
        return None
    series = frame.to_pandas().set_index("timestamp")["returns"]
    series.index = pd.DatetimeIndex(series.index)
    return series


returns = symbol_returns(DEMONSTRATION_SYMBOL)
assert returns is not None, f"{DEMONSTRATION_SYMBOL} has fewer than {MINIMUM_SESSIONS} sessions"

print(f"Panel: {len(symbols)} symbols over {START_DATE} to {END_DATE}")
print(
    f"{DEMONSTRATION_SYMBOL}: {len(returns):,} sessions, "
    f"{returns.index.min().date()} to {returns.index.max().date()}"
)
print(f"Daily return: mean {returns.mean():.4f} percent, standard deviation {returns.std():.4f}")

# %% [markdown]
# ## The property the model assumes
#
# GARCH is worth fitting only if the variance of the series changes over time in a way that
# depends on its own past. The **ARCH-LM test** asks exactly that: it regresses squared
# returns on their own lags and tests whether those lags jointly explain anything. Its null
# is that they do not, so rejecting it says the size of recent moves predicts the size of
# the next one.
#
# The three panels below are the same statement drawn three ways, and it is worth looking at
# them before the test: the returns themselves, their squares, and a rolling standard
# deviation.

# %%
ROLLING_WINDOW = 21  # sessions in the rolling standard deviation: about one month

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(returns.index, returns.to_numpy(), linewidth=0.4, color=COLORS["blue"])
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Percent")
ax.set_title("Daily log returns")

ax = axes[1]
ax.plot(returns.index, returns.to_numpy() ** 2, linewidth=0.4, color=COLORS["copper"])
ax.set_ylabel("Percent squared")
ax.set_title("Squared returns: the same series with the sign removed")

rolling = returns.rolling(ROLLING_WINDOW).std() * np.sqrt(SESSIONS_PER_YEAR)
ax = axes[2]
ax.fill_between(rolling.index, 0, rolling.to_numpy(), alpha=0.3, color=COLORS["blue"])
ax.plot(rolling.index, rolling.to_numpy(), linewidth=0.8, color=COLORS["blue"])
ax.axhline(rolling.median(), color=COLORS["negative"], linestyle="--", linewidth=0.7)
ax.annotate(
    "Median over the sample",
    xy=(0.005, rolling.median()),
    xycoords=("axes fraction", "data"),
    xytext=(0, 4),
    textcoords="offset points",
    fontsize=7,
    color=COLORS["negative"],
)
ax.set_ylabel("Percent, annualized")
ax.set_xlabel("Session")
ax.set_title(f"{ROLLING_WINDOW}-session rolling standard deviation")

fig.suptitle("Large moves arrive next to other large moves")
show_with_alt(
    fig,
    "Three stacked panels sharing a time axis. The top draws daily log returns, whose "
    "amplitude visibly widens and narrows in blocks rather than staying constant. The "
    "middle draws the squared returns, where those blocks become tall spikes clustered in "
    "2011, 2018, 2020 and 2022. The bottom draws a rolling annualized standard deviation "
    "against its median, spending long stretches on one side of it at a time.",
)

# %%
arch_test = arch_lm_test(returns.dropna().to_numpy())
print(f"ARCH-LM statistic: {arch_test.test_statistic:.2f}")
print(f"P-value:           {arch_test.p_value:.2e}")
print(f"ARCH effects:      {arch_test.has_arch_effects}")

# %% [markdown]
# ## The model
#
# GARCH(1,1) says the variance of the next session is a weighted sum of three things: a
# constant, the size of the last move, and the last variance estimate.
#
# $$\sigma^2_t = \omega + \alpha\,\varepsilon^2_{t-1} + \beta\,\sigma^2_{t-1}$$
#
# The two weights are what the model is. $\alpha$ is how much a new move moves the estimate,
# so a large $\alpha$ makes a jumpy series of estimates. $\beta$ is how much of yesterday's
# estimate carries over, so a large $\beta$ makes a smooth one that takes a long time to
# come back down. Their sum is the **persistence**, and it has to be below one for the
# variance to have a long-run average at all: at exactly one a shock never decays.
#
# Written out, the estimate is an exponentially weighted average of past squared returns
# with decay $\beta$, plus a floor set by $\omega$. That is worth holding onto, because it
# says what the model can and cannot do: it can track how volatile the recent past was, and
# it has no mechanism for anticipating anything the past does not already show.

# %%
garch = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")

omega, alpha, beta = (garch.params[name] for name in ("omega", "alpha[1]", "beta[1]"))
persistence = alpha + beta
half_life = np.log(0.5) / np.log(persistence)
long_run_volatility = np.sqrt(omega / (1 - persistence)) * np.sqrt(SESSIONS_PER_YEAR)

display(
    pd.DataFrame(
        [
            {
                "quantity": "omega, the constant",
                "value": omega,
                "what it sets": "the floor under the variance",
            },
            {
                "quantity": "alpha",
                "value": alpha,
                "what it sets": "how much a new move moves the estimate",
            },
            {
                "quantity": "beta",
                "value": beta,
                "what it sets": "how much of yesterday's estimate carries over",
            },
            {
                "quantity": "persistence, alpha + beta",
                "value": persistence,
                "what it sets": "how slowly a shock decays",
            },
            {
                "quantity": "half-life, sessions",
                "value": half_life,
                "what it sets": "when half a shock has decayed",
            },
            {
                "quantity": "long-run volatility, annualized percent",
                "value": long_run_volatility,
                "what it sets": "where the estimate returns to",
            },
        ]
    )
)

# %% [markdown]
# Read the table as one sentence. Persistence this close to one is what makes a volatility
# estimate useful and also what makes it slow: the half-life in the table is how many
# sessions it takes for half of a shock to work its way out, and until then the estimate is
# still carrying the last crisis. That is the trade the two parameters are making, and it is
# the same trade the window length made in every rolling statistic earlier in the chapter,
# with the difference that here it was estimated rather than chosen.

# %% [markdown]
# ## Asymmetry
#
# The model above treats a fall and a rise of the same size identically, because only
# $\varepsilon^2$ enters. In equities they are not identical: a fall raises subsequent
# volatility more than a rise of the same size does. **EGARCH** allows that by modelling the
# logarithm of the variance and adding a term that reads the sign of the last move, so the
# extra parameter is the size of the asymmetry.

# %%
egarch = arch_model(returns, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(
    disp="off"
)

display(
    pd.DataFrame(
        [
            {
                "model": "GARCH(1,1)",
                "parameters": len(garch.params),
                "AIC": garch.aic,
                "BIC": garch.bic,
            },
            {
                "model": "EGARCH(1,1,1)",
                "parameters": len(egarch.params),
                "AIC": egarch.aic,
                "BIC": egarch.bic,
            },
        ]
    )
)
print(f"Asymmetry parameter: {egarch.params['gamma[1]']:+.4f}")
print("A negative value means a fall raises the volatility estimate more than a rise")

# %% [markdown]
# ## Did the model capture what it was fitted to capture
#
# The check on a volatility model is its **standardized residuals**: each return divided by
# the volatility the model assigned to that session. If the model has the variance right,
# these have unit variance and no remaining clustering, so their squares should show no
# autocorrelation. Whatever is left in them is what the model missed.

# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["grid_3x2"])

annualize = np.sqrt(SESSIONS_PER_YEAR)
ax = axes[0, 0]
ax.plot(
    garch.conditional_volatility.index,
    garch.conditional_volatility.to_numpy() * annualize,
    linewidth=0.7,
    color=COLORS["blue"],
    label="GARCH",
)
ax.plot(
    egarch.conditional_volatility.index,
    egarch.conditional_volatility.to_numpy() * annualize,
    linewidth=0.7,
    color=COLORS["amber"],
    alpha=0.8,
    label="EGARCH",
)
ax.set_ylabel("Percent, annualized")
ax.set_title("The two conditional volatilities")
ax.legend(fontsize=7)

standardized = garch.std_resid.dropna()
ax = axes[0, 1]
ax.plot(standardized.index, standardized.to_numpy(), linewidth=0.3, color=COLORS["blue"])
for level in (-2, 0, 2):
    ax.axhline(level, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Standard deviations")
ax.set_title("Standardized residuals")

ax = axes[1, 0]
probplot(standardized, dist="norm", plot=ax)
ax.get_lines()[0].set(color=COLORS["blue"], markerfacecolor=COLORS["blue"], markersize=2)
ax.get_lines()[1].set_color(COLORS["negative"])
ax.set_title("Against a normal")
ax.set_xlabel("Normal quantile")
ax.set_ylabel("Sample quantile")

ax = axes[1, 1]
plot_acf(standardized**2, lags=40, zero=False, ax=ax)
ax.set_title("Squared residuals, ACF")
ax.set_xlabel("Lag, sessions")
ax.set_ylabel("Correlation")

fig.suptitle("The clustering is gone; the tails are not")
show_with_alt(
    fig,
    "Four panels. Top left overlays the GARCH and EGARCH annualized conditional volatility, "
    "which track each other closely and separate during the sharpest falls. Top right draws "
    "the standardized residuals against dashed lines at plus and minus two, with excursions "
    "well beyond them. Bottom left is a quantile plot of those residuals against a normal, "
    "bending away from the line at both ends. Bottom right is the autocorrelation of the "
    "squared residuals, whose bars sit close to zero at every lag.",
)

# %%
print(f"Standardized residuals: standard deviation {standardized.std():.4f}, target 1")
print(f"Excess kurtosis: {standardized.kurtosis():.3f}, target 0 for a normal")
beyond_three = float((standardized.abs() > 3).mean())
print(
    f"Beyond three standard deviations: {beyond_three:.2%}, against {2 * norm.sf(3):.2%} for a normal"
)

# %% [markdown]
# Two of the three checks pass and one does not, which is the usual result and the reason
# the next section matters. The squared residuals have lost their autocorrelation, so the
# clustering the model was fitted to remove is gone. But the residuals are far from normal:
# they exceed three standard deviations several times more often than a normal distribution
# allows, which is what the quantile panel is showing.
#
# That is a statement about the **distribution** assumed for the innovations, not about the
# variance model. GARCH got the variance right and the shape wrong. The response is a
# heavier-tailed innovation distribution, Student-t being the usual choice, and it matters
# most for exactly the calculation the next section does.

# %% [markdown]
# ## From a volatility estimate to a risk number
#
# **Value at risk** at a confidence level is the loss that should be exceeded only with the
# remaining probability. Under a normal distribution it is the forecast mean plus the
# forecast volatility times the corresponding quantile, which is one line given a volatility
# estimate.
#
# It is also the calculation most exposed to the fat tails the previous section found, so
# the honest version does not stop at producing the number. Below, the same rule is applied
# at every session in the sample using that session's own conditional volatility, and the
# exceedances are counted against what the confidence level promised.

# %%
CONFIDENCE_LEVELS = [0.95, 0.99]

conditional = garch.conditional_volatility
mean_return = garch.params["mu"]

# The historical count uses each session's own conditional volatility. The number a risk
# system would carry today is a forecast for the NEXT session, which is a different
# quantity: the last conditional value estimates the session that has already ended.
next_session = garch.forecast(horizon=1, reindex=False)
forecast_mean = float(next_session.mean.iloc[-1, 0])
forecast_volatility = float(np.sqrt(next_session.variance.iloc[-1, 0]))

backtest = pd.DataFrame(
    [
        {
            "confidence": level,
            "promised exceedance rate": 1 - level,
            "observed exceedance rate": float(
                (
                    returns.loc[conditional.index] < mean_return + conditional * norm.ppf(1 - level)
                ).mean()
            ),
            "next-session value at risk, percent": forecast_mean
            + forecast_volatility * norm.ppf(1 - level),
        }
        for level in CONFIDENCE_LEVELS
    ]
)
display(backtest)
print(f"Conditional volatility of the session just ended: {conditional.iloc[-1]:.4f} percent")
print(f"One-step forecast for the next session:           {forecast_volatility:.4f} percent")

# %% [markdown]
# The observed rates come out above what each level promised, and by more at the tighter
# level, which is exactly what a normal assumption does to a series with fat tails: the
# further into the tail the rule reaches, the more it understates. The volatility model is
# not the problem here, and swapping the innovation distribution rather than the variance
# equation is what closes the gap.
#
# The last column is the number a risk system would carry today. Read with the column beside
# it, it is a loss level that has historically been breached more often than its label
# claims.

# %% [markdown]
# ## The same estimate as a column, across the panel
#
# `arch` fits models. For building a feature over a whole panel there are two shapes worth
# knowing.
#
# `case_studies.utils.temporal.garch11_conditional_volatility` runs the recursion forward
# under parameters given to it, which is what a walk-forward feature needs: fit on the
# training block, then filter across the rest so no later return enters an earlier estimate.
# It is the volatility counterpart of the ARIMA filter used in `07_arima_features`.
#
# `ml4t.engineer.features.volatility` supplies Polars expressions that compute volatility
# features inside a `with_columns` call. Two things about them decide whether the columns
# mean anything, and neither is visible in the call.
#
# **Each expression takes either a price column or a return column, and computes the other
# itself.** `realized_volatility` and `garch_forecast` are given returns;
# `ewma_volatility`, `volatility_of_volatility` and `volatility_percentile_rank` are given a
# price and difference it internally. Handing a return series to one of the price-taking
# expressions produces a column: it differences the returns and carries on, and what comes
# out is a percentage change of a series that crosses zero. It looks like a volatility rank
# and is not one.
#
# **The parameters have defaults, and a default is not this symbol's fit.** `garch_forecast`
# takes an $\omega$ that defaults to a fixed small number, and $\omega$ carries the units of
# the returns squared, so a default would set the level of the feature from something with
# no relation to the data. The fitted values are passed below.

# %%
TRAIN_FRACTION = 0.7

split = int(len(returns) * TRAIN_FRACTION)
train_returns = returns.iloc[:split]
train_fit = arch_model(train_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
    disp="off"
)

filtered = garch11_conditional_volatility(
    returns.to_numpy(),
    mu=train_fit.params["mu"],
    omega=train_fit.params["omega"],
    alpha=train_fit.params["alpha[1]"],
    beta=train_fit.params["beta[1]"],
    backcast=float(np.mean(train_returns.to_numpy() ** 2)),
)

walk_forward = pd.Series(filtered, index=returns.index)
print(f"Fitted on {split:,} sessions, filtered across all {len(returns):,}")
print(
    "Correlation with the whole-sample fit over the held-out block: "
    f"{np.corrcoef(walk_forward.iloc[split:], conditional.iloc[split:])[0, 1]:.4f}"
)

# %% [markdown]
# The two agree closely over the held-out block, and that agreement is the argument for the
# filtered version rather than against it. The recursion is dominated by the data it reads
# rather than by the parameters, so refitting buys little; what the filtered version buys is
# that no estimate depends on a return that had not happened yet. The check is cheap and the
# guarantee is not available any other way.

# %%
EWMA_SPAN = 120
PERCENTILE_WINDOW = 60  # sessions in the volatility the rank is taken of
PERCENTILE_LOOKBACK = 252  # sessions the rank is taken over, the expression's default

features = (
    etfs.filter(pl.col("symbol") == DEMONSTRATION_SYMBOL)
    .select(["timestamp", "close"])
    .with_columns(returns=pl.col("close").log().diff() * RETURN_SCALE)
    .drop_nulls()
    .with_columns(
        realized=realized_volatility("returns", period=ROLLING_WINDOW),
        ewma=ewma_volatility("close", span=EWMA_SPAN),
        garch=garch_forecast("returns", horizon=1, omega=omega, alpha=alpha, beta=beta),
        vol_of_vol=volatility_of_volatility("close", vol_period=ROLLING_WINDOW),
        percentile=volatility_percentile_rank(
            "close", period=PERCENTILE_WINDOW, lookback=PERCENTILE_LOOKBACK
        ),
    )
)
display(features.select(["timestamp", "realized", "garch", "vol_of_vol", "percentile"]).tail(5))

# %%
frame = features.to_pandas().set_index("timestamp")

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(
    frame.index,
    frame["garch"] * annualize,
    linewidth=0.6,
    color=COLORS["recede"],
    label="GARCH",
)
ax.plot(
    frame.index,
    frame["realized"],
    linewidth=0.9,
    color=COLORS["blue"],
    label=f"Realized, {ROLLING_WINDOW} sessions",
)
ax.plot(
    frame.index,
    frame["ewma"] * annualize * RETURN_SCALE,
    linewidth=0.7,
    color=COLORS["amber"],
    label=f"EWMA, span {EWMA_SPAN}",
)
ax.set_ylabel("Percent, annualized")
ax.set_title("Three estimates of the same quantity")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(frame.index, frame["vol_of_vol"], linewidth=0.7, color=COLORS["copper"])
ax.set_ylabel("Ratio")
ax.set_title("How unstable the volatility estimate itself is")

ax = axes[2]
ax.fill_between(frame.index, 0, frame["percentile"], alpha=0.4, color=COLORS["blue"])
ax.set_ylim(0, 100)
ax.set_ylabel("Percentile")
ax.set_xlabel("Session")
ax.set_title(f"{PERCENTILE_WINDOW}-session volatility ranked over {PERCENTILE_LOOKBACK}")

fig.suptitle("One quantity, three estimators, and two ways of conditioning on it")
show_with_alt(
    fig,
    "Three stacked panels sharing a time axis. The top overlays a GARCH, a realized and an "
    "exponentially weighted volatility estimate: all three rise and fall together, the "
    "GARCH one updates daily and is the most jagged, the realized one is smoothed over its "
    "own window, and the exponentially weighted one is the smoothest and comes down from a "
    "spike most slowly. The middle shows the volatility of volatility, spiking at the same "
    "dates. The bottom is a filled percentile rank cycling between zero and one hundred and "
    "reaching both ends repeatedly.",
)

# %% [markdown]
# Two things to take from the top panel. The three estimators agree about where the
# volatile periods are and disagree about how quickly to leave them: the GARCH estimate
# updates every session, the realized one is an average over its own window, and the
# exponentially weighted one with a long span comes down from a spike over months. Which is
# right depends on how quickly the decision reading it needs to react.
#
# The percentile rank in the bottom panel is the one to notice as a feature. The other
# series are in percent per year and move over an order of magnitude across the sample, so a
# model reading them has to learn what counts as high; the rank is bounded, has the same
# meaning in 2012 and in 2020, and answers the question a conditioning feature is usually
# asked, which is whether the current regime is unusual for this asset.

# %% [markdown]
# ## Across the panel
#
# One symbol gives one persistence. Fitting the panel gives a distribution, and the
# distribution is what says whether the value read from SPY is a property of equity returns
# or of that symbol.

# %%
panel_rows = []
for symbol in symbols:
    series = symbol_returns(symbol)
    if series is None:
        panel_rows.append({"symbol": symbol, "status": "too short"})
        continue
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always", DataScaleWarning)
        try:
            fit = arch_model(series, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
                disp="off", show_warning=False
            )
        except (ValueError, np.linalg.LinAlgError) as failure:
            panel_rows.append({"symbol": symbol, "status": type(failure).__name__})
            continue
    panel_rows.append(
        {
            "symbol": symbol,
            "status": "fitted",
            "alpha": fit.params["alpha[1]"],
            "beta": fit.params["beta[1]"],
            "persistence": fit.params["alpha[1]"] + fit.params["beta[1]"],
            "rescaled": any(issubclass(entry.category, DataScaleWarning) for entry in raised),
        }
    )

panel = pd.DataFrame(panel_rows)
fitted_panel = panel[panel["status"] == "fitted"]

print(f"Symbols fitted: {len(fitted_panel)} of {len(panel)}")
print(
    "Persistence across the panel: "
    f"median {fitted_panel['persistence'].median():.4f}, "
    f"range {fitted_panel['persistence'].min():.4f} to {fitted_panel['persistence'].max():.4f}"
)
print(f"Fits above 0.99: {(fitted_panel['persistence'] > 0.99).sum()}")
print(
    f"Fits at or above 1, where the variance has no long-run average: {(fitted_panel['persistence'] >= 1.0).sum()}"
)
print(f"Symbols whose returns `arch` considered badly scaled: {fitted_panel['rescaled'].sum()}")

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.hist(fitted_panel["persistence"], bins=30, color=COLORS["blue"])
ax.axvline(1.0, color=COLORS["negative"], linestyle="--", linewidth=0.8)
ax.annotate(
    "One: a shock that never decays",
    xy=(1.0, 0.98),
    xycoords=("data", "axes fraction"),
    xytext=(-4, 0),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=7,
    color=COLORS["negative"],
)
ax.set_xlabel("Persistence, alpha plus beta")
ax.set_ylabel("Symbols")
ax.set_title("Volatility persistence sits just below one across the whole panel")
show_with_alt(
    fig,
    "A histogram of the persistence parameter across every fitted symbol in the panel, with "
    "a dashed line at one. The mass sits in a narrow band just below the line, with a short "
    "tail reaching down and nothing above it.",
)

# %% [markdown]
# The concentration just below one is the finding, and it is why persistence is rarely worth
# carrying as a cross-sectional feature: every symbol has nearly the same value, so a column
# holding it separates almost nothing. What varies across symbols and over time is the
# *level* of the conditional volatility, which is the column to keep.
#
# The boundary at one is a different matter, and the panel reaches it. Be precise about what
# fails there, because it is narrower than it sounds. The conditional variance recursion is
# well defined at any persistence: given a starting value it produces one number per
# session, and the forecast $h$ steps ahead is finite for every finite $h$. What
# $\omega/(1-\alpha-\beta)$ is, is the limit those forecasts approach as the horizon grows,
# and that limit exists only while the persistence is below one. At exactly one the expected
# forecast grows without settling anywhere: each further step adds $\omega$ and nothing pulls
# it back.
#
# So a fit on that boundary does not invalidate the conditional volatility column, and does
# not invalidate a one-step forecast either. What it invalidates is every statement about a
# long-run level, including this notebook's own long-run volatility figure and its
# half-life, and any forecast far enough ahead that mean reversion was doing the work.

# %% [markdown]
# ## The features this notebook produces
#
# | Column | What it is | Where the timing guarantee holds |
# |---|---|---|
# | filtered conditional volatility | the recursion run forward under parameters fitted on the training block | on the held-out block only |
# | standardized residual | the return divided by that session's estimate | wherever the estimate dividing it does |
# | volatility percentile rank | where the current estimate sits in a trailing window | throughout |
# | volatility of volatility | how unstable the estimate itself has been | throughout |
# | persistence | a fitted parameter, one per symbol, not a per-session column | on a block after the one it was fitted on |
#
# The right-hand column is the one to read, because "causal" is a property of a value's
# position rather than of the construction that produced it. The filtered series carries the
# guarantee only where its parameters preceded it, which is the block after the training
# split; inside the training block the same values were produced by parameters estimated
# partly from them. The standardized residuals plotted earlier come from the whole-sample
# fit and are a diagnostic rather than a feature, and the panel's persistence is estimated
# over each symbol's entire history and is descriptive for the same reason. The cell below
# marks the part of the filtered series that carries the guarantee.

# %%
held_out = np.zeros(len(returns), dtype=bool)
held_out[split:] = True
print(f"Filtered sessions with the timing guarantee: {held_out.sum():,} of {len(returns):,}")
print(f"Their conditional volatility: mean {walk_forward[held_out].mean():.4f} percent per session")

# %% [markdown]
# ## Key takeaways
#
# 1. **Volatility is the part of a return series that is predictable**, and a GARCH fit is
#    the standard way to turn that into one number per session. The mean is not, which is
#    the previous notebook's finding.
# 2. **The two parameters are the model.** How much a new move moves the estimate, and how
#    much of the old estimate carries into the next one. Their sum decides how long a shock
#    takes to decay, and it sits just below one for essentially every liquid asset.
# 3. **Check the standardized residuals, and expect two different answers.** The clustering
#    goes away, which says the variance model worked; the tails do not, which says the
#    innovation distribution is wrong and is a separate repair.
# 4. **A risk number is a claim that can be counted.** Applying the value-at-risk rule at
#    every session and counting the exceedances is the check, and under a normal assumption
#    on fat-tailed data it comes out short.
# 5. **A feature has to be filtered, not fitted in place.** Estimating parameters on a
#    training block and running the recursion forward gives a column no later return
#    touched, and on this data it agrees closely with the in-sample version anyway.
#
# **Known limitations.** Every fit here uses a normal innovation distribution, which the
# residual diagnostics reject; a Student-t fit would change the value-at-risk numbers and
# little else. The exceedance counts are in-sample for the whole-sample fit. The panel's
# symbols overlap heavily, so its distribution of persistence is not a hundred independent
# draws. And nothing here forecasts more than one session ahead, where the mean-reversion
# of the variance towards its long-run level starts to dominate.
#
# **Previous**: `07_arima_features`, which modelled the mean.
# **Next**: `09_har_rough_volatility` for volatility models built on longer memory than a
# single decay rate can express.
