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
# # Kalman Filters for Financial Feature Extraction
#
# **Chapter 9 | Section 9.2**
#
# **Docker image**: `ml4t`
#
# A moving average answers "what is the price, ignoring the noise" by averaging a fixed
# number of past observations with fixed weights. The **Kalman filter** answers the same
# question by carrying an explicit model of what the price is doing and how noisily it is
# observed, and it decides how much to trust each new observation from those two
# quantities rather than from a window length. Where a moving average has one setting, the
# filter has a model, and every new observation updates both the estimate and how uncertain
# that estimate is.
#
# **Learning objectives**
#
# - Write down a two-part model of a price series, one part saying how the underlying level
#   and its rate of change evolve and one saying how noisily the price is observed, and
#   implement the recursion that updates both as observations arrive.
# - Read the four quantities the recursion produces as features: the estimated level, its
#   rate of change, the surprise in each new observation, and how uncertain the estimate is.
# - Fit the two noise settings to data by maximum likelihood instead of choosing them by
#   hand, and refit them forward through the sample so no feature reads a setting estimated
#   from its own future.
# - Estimate a hedge ratio between two assets that is allowed to change over time, and say
#   what the filter does that a rolling regression does not.
#
# **Book reference**
#
# Chapter 9, Section 9.2 (Transforming signals to uncover hidden structure).
#
# **Prerequisites**
#
# Matrix multiplication and inversion. `01_visual_diagnostics` for what a stationary series
# is, which is what the innovation below is supposed to be.

# %% [markdown]
# ## Setup

# %%
"""Kalman Filters - state-space feature extraction for trading systems."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from scipy.optimize import minimize
from scipy.stats import linregress

from data import load_etfs
from utils.style import COLORS, FIGSIZE, show_with_alt

# %% tags=["parameters"]
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

# %% [markdown]
# ## The series
#
# Six years of daily SPY closes, and later the same window of QQQ for the pair. The filter
# is applied to the price level rather than to returns, because the quantity it is built to
# estimate is a level that moves slowly under an observation that jumps around.

# %%
etfs = load_etfs()

spy = (
    etfs.filter(pl.col("symbol") == "SPY")
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .sort("timestamp")
)

prices = spy["close"].to_numpy()
sessions = spy["timestamp"].to_list()
print(f"SPY: {len(prices):,} sessions ({sessions[0]} to {sessions[-1]})")

# %% [markdown]
# ## The model
#
# The filter needs two statements. The first says what the thing being estimated does when
# nobody is looking, the second says how the observation relates to it.
#
# **The state** is what we are tracking, and here it has two parts: the level of the price
# and the amount that level moves per session.
#
# $$\mathbf{x}_t = [\text{level}_t, \; \text{slope}_t]^\top$$
#
# **The transition** says the level advances by its own slope and the slope stays where it
# was, both disturbed by noise. This is the *local linear trend* model: a level that drifts
# and a drift that itself drifts.
#
# $$\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t, \qquad
# \mathbf{F} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \qquad
# \mathbf{w}_t \sim N(\mathbf{0}, \mathbf{Q})$$
#
# **The observation** says the closing price is the level plus noise. The slope is never
# observed; it is inferred entirely from how the level has been moving.
#
# $$y_t = \mathbf{H}\mathbf{x}_t + v_t, \qquad \mathbf{H} = [1, \; 0], \qquad
# v_t \sim N(0, R)$$
#
# $\mathbf{Q}$ and $R$ are the two settings that decide everything the filter does.
# $\mathbf{Q}$ says how much the state is allowed to move on its own; $R$ says how noisy
# the observation is. Their ratio is the whole behaviour of the filter: a large $R$ next to
# a small $\mathbf{Q}$ means observations are noise and the state is steady, so the estimate
# barely moves; the reverse means the observation is nearly the truth and the estimate
# tracks it.

# %% [markdown]
# ## The recursion
#
# Each session runs three steps.
#
# **Predict.** Advance the state and its covariance through the transition, which moves the
# estimate forward and makes it less certain, because $\mathbf{Q}$ is added to the
# covariance.
#
# **Innovate.** The **innovation** is the part of the new observation the prediction did not
# account for, $y_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1}$. If the model is right this is
# unpredictable noise, so it is both the filter's own diagnostic and a feature: a large
# innovation is a session that surprised the model.
#
# **Update.** The **Kalman gain** $\mathbf{K}$ decides how much of the innovation to
# believe. It is the ratio of the state's uncertainty to the total uncertainty of the
# prediction, so it is large when the state is poorly pinned down and small when the
# observation is noisy. The estimate moves by the gain times the innovation, and the
# covariance shrinks.
#
# The function also accumulates the **log-likelihood** of the observations under these
# settings, which is what the next section maximises. Its first terms are dominated by the
# deliberately vague starting covariance rather than by the settings, so `burn_in` drops
# them.


# %%
def kalman_local_linear(
    prices: np.ndarray,
    observation_noise: float | np.ndarray,
    level_noise: float | np.ndarray,
    slope_noise: float | np.ndarray,
    burn_in: int = 0,
) -> dict:
    """Local linear trend filter. Noise settings may be scalars or per-session arrays."""
    n = len(prices)
    obs_noise = np.broadcast_to(np.asarray(observation_noise, dtype=float), (n,))
    lvl_noise = np.broadcast_to(np.asarray(level_noise, dtype=float), (n,))
    slp_noise = np.broadcast_to(np.asarray(slope_noise, dtype=float), (n,))

    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([[1.0, 0.0]])

    state = np.array([prices[0], 0.0])
    covariance = np.eye(2) * 10.0

    out = {name: np.zeros(n) for name in ("level", "slope", "innovation", "uncertainty")}
    log_likelihood = 0.0

    for t in range(n):
        state_pred = transition @ state
        cov_pred = transition @ covariance @ transition.T + np.diag([lvl_noise[t], slp_noise[t]])

        innovation = prices[t] - observation @ state_pred
        innovation_var = observation @ cov_pred @ observation.T + obs_noise[t]

        if t >= burn_in:
            log_likelihood += -0.5 * (
                np.log(2 * np.pi * innovation_var[0, 0]) + innovation[0] ** 2 / innovation_var[0, 0]
            )

        gain = cov_pred @ observation.T @ np.linalg.inv(innovation_var)
        state = state_pred + gain @ innovation
        covariance = (np.eye(2) - gain @ observation) @ cov_pred

        out["level"][t], out["slope"][t] = state
        out["innovation"][t] = innovation[0]
        out["uncertainty"][t] = covariance[0, 0]

    out["log_likelihood"] = log_likelihood
    return out


# %% [markdown]
# ## What the filter produces
#
# Run it once with settings chosen by hand, deliberately in the smoothing direction: an
# observation variance well above the level variance, so the filter treats each close as
# noisy and moves its estimate slowly. The four panels are the four features.

# %%
HAND_SET = {"observation_noise": 1.0, "level_noise": 0.01, "slope_noise": 0.001}
LIKELIHOOD_BURN_IN = 20  # sessions the vague starting covariance dominates

kf_hand = kalman_local_linear(prices, **HAND_SET, burn_in=LIKELIHOOD_BURN_IN)

# %%
fig, axes = plt.subplots(4, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(sessions, prices, linewidth=0.5, alpha=0.5, color=COLORS["neutral"], label="Close")
ax.plot(sessions, kf_hand["level"], linewidth=1, color=COLORS["blue"], label="Estimated level")
ax.set_ylabel("US dollars")
ax.set_title("Close and estimated level")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(sessions, kf_hand["slope"], linewidth=0.8, color=COLORS["amber"])
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Dollars per session")
ax.set_title("Slope: how fast the level is moving")

ax = axes[2]
ax.plot(sessions, kf_hand["innovation"], linewidth=0.5, alpha=0.8, color=COLORS["copper"])
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("US dollars")
ax.set_title("Innovation: what each close did that the prediction did not expect")

ax = axes[3]
ax.plot(sessions, kf_hand["uncertainty"], linewidth=0.8, color=COLORS["slate"])
ax.set_ylabel("Variance")
ax.set_xlabel("Session")
ax.set_title("Uncertainty settles once the filter has seen enough")

fig.suptitle("One pass of the filter produces four series")
show_with_alt(
    fig,
    "Four stacked panels sharing a time axis. The first draws the SPY close in grey with "
    "the estimated level over it; the two are indistinguishable at this vertical scale. "
    "The second is "
    "the slope, oscillating around zero and turning negative through 2022. The third is "
    "the innovation, a band of noise around zero with the widest excursions in early 2020. "
    "The fourth is the level variance, which falls steeply over the first few sessions and "
    "is then flat for the rest of the sample.",
)

# %% [markdown]
# The top panel is drawn over a range of hundreds of dollars, and the estimate is within a
# few dollars of the close throughout, so the two lines lie on top of each other. How far
# apart they are is the thing the settings control, and the next section measures it rather
# than asking the reader to see it.
#
# The bottom panel is worth pausing on. The starting covariance says the filter knows
# nothing, so its first estimates are uncertain; within a few sessions the uncertainty
# reaches a level and stays there. That level is a property of the settings and not of the
# data: with a constant $\mathbf{Q}$ and $R$, the covariance recursion converges to a fixed
# point and the gain converges with it. After convergence the filter is a fixed linear
# filter, and the uncertainty feature stops carrying information until the settings change.

# %% [markdown]
# ## Fitting the two settings instead of choosing them
#
# The settings were picked by hand, and the panel above shows the consequence: the level
# lags because the hand-set observation noise is a hundred times the level noise, which is
# an assertion about SPY nobody checked. **Maximum likelihood** replaces the assertion with
# an estimate. The filter already computes the likelihood of the observed prices under a
# given set of noise variances, so maximising it over those variances picks the settings
# that make the data most probable.
#
# The optimiser searches over logs of the variances so that any value it tries is positive.
# It is run here on a **training cut** rather than the whole sample, because the settings
# are a fitted object and every use of them below is a claim about data they were not fitted
# to.

# %%
TRAIN_FRACTION = 0.6
train_end = int(len(prices) * TRAIN_FRACTION)
LOG_START = np.log([1.0, 0.01, 0.001])


def negative_log_likelihood(log_params: np.ndarray, series: np.ndarray) -> float:
    """The filter's log-likelihood, negated, as a function of the log noise variances."""
    result = kalman_local_linear(series, *np.exp(log_params), burn_in=LIKELIHOOD_BURN_IN)
    return -result["log_likelihood"]


fit = minimize(
    negative_log_likelihood,
    LOG_START,
    args=(prices[:train_end],),
    method="Nelder-Mead",
    options={"maxiter": 500},
)
fitted = np.exp(fit.x)

display(
    pd.DataFrame(
        [
            {
                "setting": "observation variance R",
                "hand set": HAND_SET["observation_noise"],
                "fitted": fitted[0],
            },
            {
                "setting": "level variance, Q",
                "hand set": HAND_SET["level_noise"],
                "fitted": fitted[1],
            },
            {
                "setting": "slope variance, Q",
                "hand set": HAND_SET["slope_noise"],
                "fitted": fitted[2],
            },
        ]
    )
)

# %% [markdown]
# The variances themselves are hard to read against each other, so compare what the two
# settings make the filter do. The gap between the level estimate and the close says
# whether the filter smooths or tracks, in dollars, and the log-likelihood says which
# settings the data prefer.

# %%
behaviour = []
for label, params in [("hand set", tuple(HAND_SET.values())), ("fitted", tuple(fitted))]:
    run = kalman_local_linear(prices, *params, burn_in=LIKELIHOOD_BURN_IN)
    behaviour.append(
        {
            "settings": label,
            "mean gap to the close, dollars": np.abs(run["level"] - prices).mean(),
            "log-likelihood": run["log_likelihood"],
        }
    )
display(pd.DataFrame(behaviour))

# %% [markdown]
# The fit moves decisively away from smoothing. Both the likelihood and the gap to the
# close say the hand-set settings were wrong about SPY in the same direction: at daily
# frequency the close is close to an exact observation of a level that moves a great deal
# on its own, which is another way of saying the series looks like a random walk. The
# filter that follows tracks the close, and its slope is near the last session's price
# change rather than a trend over weeks.
#
# Which of the two state variances absorbs that movement is much less determined than the
# fact that one of them must. The level may move because the level itself is noisy, or
# because the slope is noisy and drags it; no observation distinguishes the two, since
# neither is observed. The likelihood surface is therefore nearly flat along that trade,
# and an unbounded search started elsewhere reaches a different corner with a comparable
# likelihood. That weak identification is the reason the walk-forward fit below runs a
# bounded optimiser rather than this one.

# %% [markdown]
# ## Refitting forward
#
# One fit on one training cut is still one set of numbers applied to years it never saw.
# Refitting on a moving window gives each part of the sample settings estimated from the
# window that ended before it. `REFIT_WINDOW` is how much history each fit reads and
# `REFIT_STEP` is how often the fit is repeated; the bounds keep the optimiser away from
# the flat corner the unbounded fit walked into.

# %%
REFIT_WINDOW = 504  # sessions each fit reads: about two years
REFIT_STEP = 63  # sessions between fits: about one quarter
LOG_BOUNDS = [(-5, 5), (-5, 5), (-15, 5)]  # log variances; the third may go much smaller

refits = []
for start in range(0, len(prices) - REFIT_WINDOW, REFIT_STEP):
    window = prices[start : start + REFIT_WINDOW]
    fit_window = minimize(
        negative_log_likelihood,
        LOG_START,
        args=(window,),
        method="L-BFGS-B",
        bounds=LOG_BOUNDS,
        options={"maxiter": 200},
    )
    obs, level, slope = np.exp(fit_window.x)
    refits.append(
        {
            "effective_from": sessions[start + REFIT_WINDOW],
            "observation_variance": obs,
            "level_variance": level,
            "slope_variance": slope,
            "level_over_observation": level / obs,
        }
    )

refit_df = pd.DataFrame(refits).set_index("effective_from")
display(refit_df.head())

# %%
fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

ax = axes[0]
ax.plot(
    refit_df.index, refit_df["observation_variance"], linewidth=1, color=COLORS["blue"], label="R"
)
ax.plot(
    refit_df.index,
    refit_df["level_variance"],
    linewidth=1,
    color=COLORS["amber"],
    label="Q for the level",
)
ax.set_yscale("log")
ax.set_ylabel("Variance, log scale")
ax.set_title("Both variances move with the window")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(refit_df.index, refit_df["level_over_observation"], linewidth=1, color=COLORS["copper"])
ax.axhline(1, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_yscale("log")
ax.set_ylabel("Ratio, log scale")
ax.set_xlabel("Session the settings take effect")
ax.set_title("Above the line the filter tracks; below it the filter smooths")

fig.suptitle(f"Settings refitted every {REFIT_STEP} sessions, on {REFIT_WINDOW} sessions")
show_with_alt(
    fig,
    "Two stacked panels over the refit dates. The top panel plots the observation variance "
    "and the level variance on a log scale: the level variance rises steadily while the "
    "observation variance falls away to a floor in the last few windows. The bottom panel "
    "plots their ratio, also on a log scale, against a dashed reference at one; it stays "
    "above the line throughout and climbs by three orders of magnitude at the end.",
)

# %% [markdown]
# The ratio is the readable quantity: above one the filter believes the observation more
# than the model and tracks the price, below one it believes the model more and smooths.
# It moves by orders of magnitude across the sample, which is the argument for refitting.
# One set of settings applied to six years asserts that the balance between how much the
# level moves and how noisily it is measured did not change, and the panel says it did.
#
# The late windows are also where the bound starts doing work. Count how many refits leave
# the observation variance sitting on the floor the bounds impose: at those windows the
# likelihood would go on shrinking it, which is the model saying the close is measured
# without error and the filter should simply copy it. The bound is the only thing standing
# between this filter and the identity function, and it is a modelling choice rather than a
# result.

# %%
at_lower_bound = np.isclose(np.log(refit_df["observation_variance"]), LOG_BOUNDS[0][0])
print(
    f"Refits with the observation variance on its lower bound: {at_lower_bound.sum()} of {len(refit_df)}"
)

# %% [markdown]
# ## Features from settings the feature did not see
#
# The refits are turned into features by running the filter once with the settings carried
# forward from the most recent fit, so each session is filtered with settings estimated on a
# window that closed before it. Sessions before the first refit have no fitted settings and
# are excluded from everything below rather than being filtered with a guess.


# %%
def schedule_from_refits(refit_frame: pd.DataFrame, index: list, column: str) -> np.ndarray:
    """One value per session, held from the refit that most recently took effect."""
    series = pd.Series(refit_frame[column].to_numpy(), index=refit_frame.index)
    return series.reindex(pd.DatetimeIndex(index), method="ffill").to_numpy()


schedule = {
    name: schedule_from_refits(refit_df, sessions, column)
    for name, column in [
        ("observation_noise", "observation_variance"),
        ("level_noise", "level_variance"),
        ("slope_noise", "slope_variance"),
    ]
}
fitted_from = ~np.isnan(schedule["observation_noise"])
filled = {name: np.nan_to_num(values, nan=1.0) for name, values in schedule.items()}

kf_walk = kalman_local_linear(prices, **filled)
print(
    f"Sessions filtered with settings fitted before them: {fitted_from.sum():,} of {len(prices):,}"
)

# %% [markdown]
# ## Against a moving average and a rolling regression
#
# Two baselines, both fixed-window. The exponential moving average smooths the level with a
# fixed decay; the rolling ordinary least squares slope estimates the trend by fitting a
# straight line to the last `BASELINE_WINDOW` closes. Both apply the same weights in every
# market, and neither has any notion of how uncertain its answer is; the filter's weights
# follow the settings, and the settings follow the window they were fitted on.
#
# The figures below cover only the sessions a refit reaches, for the same reason the scores
# further down do: before the first refit there are no fitted settings to filter with.

# %%
BASELINE_WINDOW = 20

ema = pd.Series(prices).ewm(span=BASELINE_WINDOW, adjust=False).mean().to_numpy()

rolling_slope = np.full(len(prices), np.nan)
steps = np.arange(BASELINE_WINDOW)
for t in range(BASELINE_WINDOW, len(prices)):
    rolling_slope[t] = linregress(steps, prices[t - BASELINE_WINDOW : t]).slope

# %%
shown = np.where(fitted_from, 1.0, np.nan)  # blanks the sessions no refit reaches

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)

ax = axes[0]
ax.plot(
    sessions,
    (kf_walk["level"] - prices) * shown,
    linewidth=0.8,
    color=COLORS["blue"],
    label="Filter level",
)
ax.plot(
    sessions,
    (ema - prices) * shown,
    linewidth=0.8,
    color=COLORS["amber"],
    alpha=0.8,
    label=f"{BASELINE_WINDOW}-session EMA",
)
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("US dollars")
ax.set_title("Distance from the close: how much each estimate smooths")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(
    sessions,
    kf_walk["slope"] * shown / np.nanstd(kf_walk["slope"][fitted_from]),
    linewidth=0.8,
    color=COLORS["blue"],
    label="Filter slope",
)
ax.plot(
    sessions,
    rolling_slope * shown / np.nanstd(rolling_slope[fitted_from]),
    linewidth=0.8,
    color=COLORS["amber"],
    alpha=0.8,
    label=f"{BASELINE_WINDOW}-session OLS slope",
)
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("Standard deviations")
ax.set_xlabel("Session")
ax.set_title("The two trend estimates are not equally smooth")
ax.legend(fontsize=7)

fig.suptitle("Adaptive weights against fixed weights")
show_with_alt(
    fig,
    "Two stacked panels covering the sessions the refits reach. The top plots how far each "
    "level estimate sits from the close: the filter stays within a few dollars of zero "
    "throughout while the moving average swings much further either side, most widely "
    "during 2022. The bottom plots the two trend estimates in standard-deviation units: "
    "the rolling regression slope crosses zero repeatedly while the filter slope is much "
    "smoother and stays on one side for long stretches.",
)

# %% [markdown]
# ## Do they predict anything
#
# The **information coefficient** is the correlation between a feature read at a session and
# the return over the following sessions. It answers whether the feature carries any signal
# about what comes next, and its sign says in which direction.
#
# Only sessions filtered with settings fitted beforehand are scored, and the forward return
# is the return the feature could not have seen.

# %%
FORWARD_SESSIONS = 5

forward_return = np.full(len(prices), np.nan)
forward_return[:-FORWARD_SESSIONS] = (
    prices[FORWARD_SESSIONS:] - prices[:-FORWARD_SESSIONS]
) / prices[:-FORWARD_SESSIONS]

candidates = {
    "filter slope": kf_walk["slope"],
    "filter innovation": kf_walk["innovation"],
    f"{BASELINE_WINDOW}-session OLS slope": rolling_slope,
}

ic_rows = []
for name, feature in candidates.items():
    scored = fitted_from & ~np.isnan(feature) & ~np.isnan(forward_return)
    ic_rows.append(
        {
            "feature": name,
            "sessions scored": int(scored.sum()),
            "information coefficient": np.corrcoef(feature[scored], forward_return[scored])[0, 1],
        }
    )

display(pd.DataFrame(ic_rows))

# %% [markdown]
# Read the sign before the size. A negative information coefficient says a rising estimate
# is followed by a falling price, which is short-horizon reversal rather than momentum, and
# it is the expected sign at this horizon for a broad index.
#
# Read neither as a significance test. The forward windows overlap: consecutive sessions
# share four of their five days, so the scored sessions are nowhere near that many
# independent observations, and a standard error computed as though they were would be
# several times too small. `01_visual_diagnostics` closes on the same problem for its
# rolling stationarity columns, which are refitted on overlapping windows for the same
# reason and are dependent in the same way.
#
# The size invites a comparison the previous figure has already ruled out. The bounded fit
# puts almost all of the state's movement in the level and almost none in the slope, so the
# filter's slope changes far more slowly than a four-week regression line does: the two
# panels above show one crossing zero repeatedly while the other holds a sign for months.
# They are trend estimates over different horizons, scored against a five-session forward
# return, so whichever comes out ahead the comparison is between horizons rather than
# between estimators.
#
# Making it a comparison between estimators means fixing the horizon first: either
# constrain the filter's settings so that its level estimate has the same smoothness as the
# rolling window implies, or score each estimate against the forward window its own horizon
# covers. Both are more work than reading two numbers off a table, which is the point.

# %% [markdown]
# ## A hedge ratio that is allowed to move
#
# A pairs trade holds one asset against a multiple of another, and that multiple is the
# **hedge ratio**. Estimating it by regression over a fixed window forces a choice nobody
# can make well: a short window is noisy and a long one is slow, and the ratio is not
# constant either way.
#
# The same filter estimates it with a different reading of the same equations. The state is
# the pair of regression coefficients, the transition says they follow a random walk, and
# the observation row carries the explanatory asset's price, so the model is
# $y_t = \beta_t x_t + \alpha_t + \varepsilon_t$ with both coefficients free to drift.
#
# Two settings control it. `DELTA` sets how much the coefficients may move per session,
# through $\mathbf{Q} = \frac{\delta}{1-\delta}\mathbf{I}$: at the value used here the
# coefficients move very little per session and a great deal over a year. `PAIR_OBS_VAR`
# is the variance of the pricing error, in squared dollars, and it is set from the size of
# the residual a fixed regression leaves rather than assumed.


# %%
def kalman_hedge_ratio(x: np.ndarray, y: np.ndarray, delta: float, obs_var: float) -> dict:
    """Time-varying regression of *y* on *x* with random-walk coefficients."""
    n = len(x)
    state = np.array([0.0, 0.0])
    covariance = np.eye(2)
    drift = delta / (1 - delta) * np.eye(2)

    out = {name: np.zeros(n) for name in ("beta", "alpha", "innovation", "uncertainty")}
    for t in range(n):
        observation = np.array([[x[t], 1.0]])
        cov_pred = covariance + drift

        innovation = y[t] - (observation @ state)[0]
        innovation_var = (observation @ cov_pred @ observation.T)[0, 0] + obs_var

        gain = (cov_pred @ observation.T / innovation_var).flatten()
        state = state + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, observation)) @ cov_pred

        out["beta"][t], out["alpha"][t] = state
        out["innovation"][t] = innovation
        out["uncertainty"][t] = covariance[0, 0]
    return out


# %%
PAIR_SYMBOL = "QQQ"
DELTA = 1e-3
HEDGE_WINDOW = 60  # sessions in the rolling regression the filter is compared against

pair = (
    etfs.filter(pl.col("symbol") == PAIR_SYMBOL)
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .select(["timestamp", "close"])
    .rename({"close": "pair_close"})
)
paired = (
    spy.select(["timestamp", "close"]).join(pair, on="timestamp", how="inner").sort("timestamp")
)

spy_px = paired["close"].to_numpy()
pair_px = paired["pair_close"].to_numpy()
pair_sessions = paired["timestamp"].to_list()

# The observation variance is the squared residual of one fixed regression over the whole
# pair, which is the scale of pricing error the filter should not chase.
static = linregress(pair_px, spy_px)
PAIR_OBS_VAR = float(np.var(spy_px - (static.slope * pair_px + static.intercept)))

hedge = kalman_hedge_ratio(pair_px, spy_px, delta=DELTA, obs_var=PAIR_OBS_VAR)

rolling_hedge = np.full(len(spy_px), np.nan)
for t in range(HEDGE_WINDOW, len(spy_px)):
    rolling_hedge[t] = linregress(pair_px[t - HEDGE_WINDOW : t], spy_px[t - HEDGE_WINDOW : t]).slope

print(f"Static regression slope over the whole pair: {static.slope:.3f}")
print(f"Observation variance set from its residual: {PAIR_OBS_VAR:,.1f} squared dollars")

# %% [markdown]
# The spread is what is left after the hedge: the price of one asset minus the estimated
# multiple of the other. Its band is drawn from an **expanding** standard deviation, so the
# band on any session is computed from that session and the ones before it, and a spread
# marked as extreme was extreme against the history available at the time.

# %%
spread = spy_px - hedge["beta"] * pair_px - hedge["alpha"]
expanding_std = pd.Series(spread).expanding(min_periods=HEDGE_WINDOW).std().to_numpy()

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(pair_sessions, hedge["beta"], linewidth=1, color=COLORS["blue"], label="Filter")
ax.plot(
    pair_sessions,
    rolling_hedge,
    linewidth=1,
    linestyle="--",
    color=COLORS["amber"],
    alpha=0.8,
    label=f"{HEDGE_WINDOW}-session regression",
)
ax.set_ylabel("Hedge ratio")
ax.set_title(f"SPY against {PAIR_SYMBOL}, two ways of letting the ratio move")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(pair_sessions, spread, linewidth=0.8, color=COLORS["blue"])
ax.plot(pair_sessions, 2 * expanding_std, linewidth=0.6, linestyle=":", color=COLORS["negative"])
ax.plot(pair_sessions, -2 * expanding_std, linewidth=0.6, linestyle=":", color=COLORS["negative"])
ax.axhline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_ylabel("US dollars")
ax.set_title("Spread, against two expanding standard deviations")

ax = axes[2]
ax.plot(pair_sessions, np.sqrt(hedge["uncertainty"]), linewidth=0.8, color=COLORS["slate"])
ax.set_ylabel("Standard deviations")
ax.set_xlabel("Session")
ax.set_title("How uncertain the hedge ratio is")

fig.suptitle("A hedge ratio the filter revises as the pair moves")
show_with_alt(
    fig,
    "Three stacked panels. The top compares the filter's hedge ratio with a sixty-session "
    "rolling regression estimate: both drift over the sample and the regression estimate "
    "is the more jagged of the two. The middle draws the residual spread against dotted "
    "bands at plus and minus two expanding standard deviations, which widen early and "
    "settle. The bottom draws the standard deviation of the hedge-ratio estimate, which "
    "falls sharply at the start and then holds roughly level.",
)

# %% [markdown]
# The two estimates disagree in a way worth naming. The rolling regression jumps whenever a
# large observation enters or leaves its window, so it moves for reasons that have nothing
# to do with the pair; the filter has no window to leave and moves only when the innovation
# says the old ratio is not explaining the new prices. What the filter does not give is a
# free lunch on the setting: `DELTA` plays the role the window length played, and the
# estimate is as responsive as that setting makes it.

# %% [markdown]
# ## The features this notebook produces
#
# | Column | What it is | What it is read for |
# |---|---|---|
# | `level` | the filtered level of the price | a smoothed price whose smoothing follows the data |
# | `slope` | the filtered rate of change | a trend estimate with no window length |
# | `innovation` | the part of each close the prediction missed | how surprising a session was |
# | `uncertainty` | the variance of the level estimate | how much to trust the other three |
# | `beta` | the filtered hedge ratio between two assets | what to hold against what, and its spread |
#
# Every one is **filtered** rather than smoothed: the value at a session uses observations
# up to that session and no later. A smoother, which revises earlier states using later
# observations, produces better estimates of the past and cannot be used as a feature at
# all.

# %% [markdown]
# ## Key takeaways
#
# 1. **The settings are the model, and their ratio is the behaviour.** How much the state
#    may move against how noisily it is observed decides everything the filter does; a
#    window length is what a fixed-weight method has instead.
# 2. **Fitting the settings by maximum likelihood is what makes them an estimate**, and
#    fitting them forward on a moving window is what stops the estimate from reading its own
#    future. On this sample the fit reverses the hand-set assumption entirely.
# 3. **A slope with no window still has a horizon**, and it is set by the fitted settings.
#    Comparing it against a fixed-window slope compares two horizons unless the horizons are
#    fixed first.
# 4. **Filtered, never smoothed.** Every quantity here uses observations up to its own
#    session. The smoothed version of each is a better estimate and is not a feature.
# 5. **A time-varying hedge ratio moves when the pair moves**, where a rolling regression
#    also moves when an old observation leaves the window.
#
# **Known limitations.** The local linear trend model assumes normal noise with constant
# variance inside each refit window, and the innovation panel shows the variance is not
# constant; a filter with time-varying observation noise, which is what the volatility
# models later in the chapter estimate, is the natural repair. The information coefficients
# here are single numbers over one symbol and one horizon and carry no confidence interval.
# And the uncertainty feature converges to a constant under fixed settings, so it carries
# information only across refits.
#
# **Next**: `05_spectral_features` for describing a series by its frequencies, and
# `06_path_signatures` for features of the path rather than of the level.
