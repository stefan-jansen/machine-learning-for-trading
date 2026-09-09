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
# # Regime as Feature
#
# **Chapter 9 | Section 9.5**
#
# **Docker image**: `ml4t`
#
# The last three notebooks built regime labels and probabilities. This one spends them, and
# the question is not whether the regime is real but how a predictive model should receive it.
# Two designs are available and they differ in what happens when the regime call is wrong.
#
# Give the model the regime probability as one more column and it stays one model, fitted on
# every session, free to use the column or ignore it. Split the training data by regime and
# fit one model per state, and each model sees only its own regime's sessions and a
# misclassified session at prediction time is routed to a model that never saw its kind.
#
# The comparison only means something if the regime feature is built the way it would be in
# production, which is the constraint that shapes the whole notebook: the hidden Markov model
# is refit inside every fold, on the training block alone, and the probabilities it produces
# come from a forward recursion that conditions on the past.
#
# **Learning objectives**
#
# - Build a regime feature inside a walk-forward loop, so that no fold's feature was fitted
#   on that fold's test block.
# - Fit the same predictive model with and without that feature and read the difference
#   against the spread across folds rather than as a single number.
# - Build the mixture-of-experts alternative and see what splitting the training data costs.
# - Read a feature importance that comes out near zero, and say what it does and does not
#   tell you about the feature.
#
# **Book reference**
#
# Chapter 9, Section 9.5 (Regime features).
#
# **Prerequisites**
#
# `11_hmm_regimes` for the hidden Markov model and for the filtered-against-smoothed
# distinction. `12_wasserstein_regimes` for the clustering alternative.
# `06_strategy_definition/02_cv_foundations` for why the folds below carry a gap.

# %% [markdown]
# ## Setup

# %%
"""Regime as feature - regime probabilities inside a walk-forward predictive model."""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from ml4t.engineer.features.ml import regime_conditional_features, rolling_entropy
from ml4t.engineer.features.regime import (
    choppiness_index,
    hurst_exponent,
    market_regime_classifier,
)
from ml4t.engineer.features.volatility import (
    garch_forecast,
    realized_volatility,
    volatility_regime_probability,
)
from ml4t.engineer.logging import setup_logging
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from case_studies.utils.temporal import (
    filtered_state_probs,
    fit_hmm_restarts,
    relabel_states,
    sort_states_by_variance,
)
from data import load_etfs, load_macro
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

setup_logging(level=logging.ERROR)  # per-call timing notices from the indicator library

# %% tags=["parameters"]
START_DATE = "2005-01-01"
END_DATE = "2024-06-30"
FORECAST_HORIZON = 5
VOLATILITY_WINDOW = 21
MOMENTUM_WINDOWS = (21, 63)
N_STATES = 2
N_RESTARTS = 3
N_SPLITS = 5
MINIMUM_REGIME_SESSIONS = 20
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## The data and the target
#
# SPY daily closes and the volatility index, joined on the session. The features are the ones
# a reader has already met: the return, a rolling standard deviation of it, two momentum sums
# over different lengths, and the volatility index measured against its own recent level so
# the column is comparable across two decades in which the index's average moved.
#
# The target is the sum of the next `FORECAST_HORIZON` returns. It reaches that many sessions
# into the future, which is what makes the gap in the folds below necessary: a training block
# ending at session $t$ contains targets built from returns up to $t + 4$, so a test block
# starting at $t + 1$ would be predicting sessions its own training labels already saw.

# %%
SESSIONS_PER_YEAR = 252

prices = load_etfs(symbols=["SPY"]).select(["timestamp", "close"]).sort("timestamp")
volatility_index = (
    load_macro().select(["timestamp", "vixcls"]).rename({"vixcls": "volatility_index"})
)

panel = (
    prices.join(volatility_index, on="timestamp", how="inner")
    .drop_nulls()
    .sort("timestamp")
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .with_columns(returns=pl.col("close").log().diff() * 100)
    .with_columns(
        volatility=pl.col("returns").rolling_std(VOLATILITY_WINDOW) * np.sqrt(SESSIONS_PER_YEAR),
        momentum_short=pl.col("returns").rolling_sum(MOMENTUM_WINDOWS[0]),
        momentum_long=pl.col("returns").rolling_sum(MOMENTUM_WINDOWS[1]),
        index_level=pl.col("volatility_index").rolling_mean(VOLATILITY_WINDOW),
    )
    .with_columns(
        index_deviation=(pl.col("volatility_index") - pl.col("index_level"))
        / pl.col("volatility_index").rolling_std(MOMENTUM_WINDOWS[1])
    )
    .with_columns(target=pl.col("returns").rolling_sum(FORECAST_HORIZON).shift(-FORECAST_HORIZON))
    .drop_nulls()
)

frame = panel.to_pandas().set_index("timestamp")
frame.index = pd.DatetimeIndex(frame.index)

BASE_FEATURES = ["returns", "volatility", "momentum_short", "momentum_long", "index_deviation"]
REGIME_FEATURE = "probability_of_the_volatile_state"

print(f"Sessions: {len(frame):,}, {frame.index.min().date()} to {frame.index.max().date()}")
print(f"Target: sum of the next {FORECAST_HORIZON} returns, in percent")
print(f"Target mean {frame['target'].mean():.4f}, standard deviation {frame['target'].std():.4f}")
display(frame[[*BASE_FEATURES, "target"]].tail(3))

# %% [markdown]
# ## The regime feature, built one fold at a time
#
# This is the cell the notebook exists for. `TimeSeriesSplit` gives five expanding training
# blocks, each followed by a test block, and a gap of `FORECAST_HORIZON` sessions between them
# so no training target overlaps a test session. Inside each fold:
#
# 1. The hidden Markov model is fitted on the training block's returns and volatility, from
#    several starts, keeping the fit with the highest training likelihood. Nothing from the
#    test block enters the fit.
# 2. Its states are renumbered by fitted variance, so state one is the more volatile one in
#    every fold. Expectation maximisation returns states in an arbitrary order, and a feature
#    named for one of them means different things across folds without this step.
# 3. Filtered probabilities are computed by forward recursion over every session up to the end
#    of the test block. The recursion at session $t$ conditions on sessions up to $t$, and the
#    parameters it uses were fitted on the training block, so a test-block value is a quantity
#    that could have been computed on the day.
#
# What this costs is that the feature is not one column but five, one per fold, and they
# disagree with each other on the sessions they share. That is the honest situation: a regime
# probability is a function of what had been fitted at the time, and the number a reader sees
# for a given session depends on which fold asked.

# %%
splits = list(TimeSeriesSplit(n_splits=N_SPLITS, gap=FORECAST_HORIZON).split(frame))
observations = frame[["returns", "volatility"]].to_numpy()

VOLATILE_STATE = N_STATES - 1

fold_probability: list[np.ndarray] = []
fold_state: list[np.ndarray] = []
fold_rows = []

for fold, (train_index, test_index) in enumerate(splits, start=1):
    fitted = fit_hmm_restarts(
        observations[train_index], n_states=N_STATES, random_state=SEED, n_restarts=N_RESTARTS
    )
    prefix = observations[: test_index[-1] + 1]
    raw = filtered_state_probs(fitted.model, prefix)
    states, probabilities = relabel_states(
        raw.argmax(axis=1), raw, sort_states_by_variance(fitted.model)
    )
    fold_probability.append(probabilities[:, VOLATILE_STATE])
    fold_state.append(states)

    volatile = probabilities[test_index, VOLATILE_STATE]
    fold_rows.append(
        {
            "fold": fold,
            "training sessions": len(train_index),
            "test sessions": len(test_index),
            "test block starts": frame.index[test_index[0]].date(),
            "training log likelihood": fitted.log_likelihood,
            "restarts rejected or failed": fitted.n_rejected + fitted.n_failed,
            "mean probability of the volatile state, test block": float(volatile.mean()),
            "test sessions called volatile": int((volatile > 0.5).sum()),
        }
    )

display(pd.DataFrame(fold_rows).set_index("fold").T)

# %% [markdown]
# The mean probability of the volatile state differs across the test blocks, which is what a
# regime feature is supposed to do, and the fold covering 2020 is the one to check that
# against. The restart column should be read too: a fold where restarts failed is a fold whose
# feature came from fewer starting points than the others.

# %% [markdown]
# ## Three designs, one loop
#
# Every design sees the same folds, the same features and the same fold-local regime feature.
# They differ only in what they do with it.
#
# - **Baseline** fits one model on `BASE_FEATURES` and never sees the regime.
# - **Regime as feature** fits one model on `BASE_FEATURES` plus the filtered probability of
#   the volatile state.
# - **Mixture of experts** fits one model per hard state on `BASE_FEATURES` alone, using the
#   training block's states to split the training rows, and routes each test session to the
#   model for its own state. A state with fewer than `MINIMUM_REGIME_SESSIONS` training rows
#   gets no model of its own, and its test sessions fall back to the other one.
#
# The scaler is refitted on each fold's training block and applied to the test block, which
# matters more than it looks: a scaler fitted on the whole sample carries the test block's
# mean and variance into the training data, and does it silently.


# %%
def evaluate_single_model(estimator, features: list[str], regime_column: bool) -> pd.DataFrame:
    """Fit one model per fold on `features`, optionally with the fold's regime probability."""
    rows = []
    for fold, ((train_index, test_index), probability) in enumerate(
        zip(splits, fold_probability, strict=True), start=1
    ):
        design = frame[features].to_numpy()
        if regime_column:
            # The fold's recursion stops at the end of its test block, so the column is only
            # defined that far; every index either loop uses lies inside it.
            column = np.full(len(frame), np.nan)
            column[: len(probability)] = probability
            design = np.column_stack([design, column])

        scaler = StandardScaler()
        model = clone(estimator)
        model.fit(
            scaler.fit_transform(design[train_index]), frame["target"].to_numpy()[train_index]
        )
        predicted = model.predict(scaler.transform(design[test_index]))
        actual = frame["target"].to_numpy()[test_index]
        rows.append(
            {
                "fold": fold,
                "root mean squared error": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r squared": float(r2_score(actual, predicted)),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def evaluate_mixture_of_experts(estimator, features: list[str]) -> pd.DataFrame:
    """One model per state, fitted on that state's training rows and routed to by state."""
    rows = []
    for fold, ((train_index, test_index), states) in enumerate(
        zip(splits, fold_state, strict=True), start=1
    ):
        design = frame[features].to_numpy()
        target = frame["target"].to_numpy()

        scaler = StandardScaler()
        train_design = scaler.fit_transform(design[train_index])
        test_design = scaler.transform(design[test_index])

        experts = {}
        for state in range(N_STATES):
            inside = states[train_index] == state
            if inside.sum() < MINIMUM_REGIME_SESSIONS:
                continue
            expert = clone(estimator)
            expert.fit(train_design[inside], target[train_index][inside])
            experts[state] = expert

        predicted = np.empty(len(test_index))
        fallback = next(iter(experts.values()))
        routed = states[test_index]
        for state in range(N_STATES):
            selected = routed == state
            if selected.any():
                predicted[selected] = experts.get(state, fallback).predict(test_design[selected])

        actual = target[test_index]
        rows.append(
            {
                "fold": fold,
                "root mean squared error": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r squared": float(r2_score(actual, predicted)),
                "experts fitted": len(experts),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


# %%
ESTIMATORS = {
    "ridge": Ridge(alpha=1.0),
    "gradient boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=3, random_state=SEED
    ),
}

per_fold = {}
for name, estimator in ESTIMATORS.items():
    per_fold[(name, "baseline")] = evaluate_single_model(estimator, BASE_FEATURES, False)
    per_fold[(name, "regime as feature")] = evaluate_single_model(estimator, BASE_FEATURES, True)
    per_fold[(name, "mixture of experts")] = evaluate_mixture_of_experts(estimator, BASE_FEATURES)

summary = pd.DataFrame(
    [
        {
            "estimator": name,
            "design": design,
            "mean error across folds": table["root mean squared error"].mean(),
            "spread of the error across folds": table["root mean squared error"].std(),
            "worst fold's error": table["root mean squared error"].max(),
            "mean r squared": table["r squared"].mean(),
        }
        for (name, design), table in per_fold.items()
    ]
).set_index(["estimator", "design"])

display(summary)

# %% [markdown]
# Read the spread column before the mean. The five folds cover different market conditions, and
# the error moves between folds by roughly an order of magnitude more than it moves between
# designs. That is the first thing the table says, and it is the reason a single averaged number
# with no spread beside it would mislead.
#
# The r squared column is the other thing to read, and it is negative or close to zero
# throughout. A negative r squared means the model predicts the test block worse than that
# block's own mean would, which is the normal outcome for a five-session return forecast from
# five features and is why the chapter treats regime information as conditioning rather than
# as a signal.

# %% [markdown]
# ## The paired difference, and what five folds can support
#
# The designs share their folds, so a fold-by-fold difference removes whatever made a fold
# hard. That is worth doing and it is not a test. A paired difference has variance
# $\mathrm{Var}(A) + \mathrm{Var}(B) - 2\,\mathrm{Cov}(A, B)$, so pairing helps in proportion to
# the covariance, and with five folds that are nested inside one another the covariance is not
# something five numbers can estimate. What the column below supports is a direction and a
# magnitude, not a claim of significance.

# %%
paired = pd.DataFrame(
    {
        f"{name}: {design} minus baseline": (
            per_fold[(name, design)]["root mean squared error"]
            - per_fold[(name, "baseline")]["root mean squared error"]
        )
        for name in ESTIMATORS
        for design in ("regime as feature", "mixture of experts")
    }
)
display(paired.T.assign(mean=paired.mean(), folds_improved=(paired < 0).sum()))

# %% [markdown]
# ## The last fold, drawn
#
# Every panel below is drawn from the final fold, whose training block is the longest and whose
# test block is the most recent. The model is the gradient-boosted regime-as-feature design,
# refitted here so its predictions can be looked at rather than only scored.

# %%
final_train, final_test = splits[-1]
final_probability = fold_probability[-1]
final_state = fold_state[-1]

regime_column = np.full(len(frame), np.nan)
regime_column[: len(final_probability)] = final_probability
final_design = np.column_stack([frame[BASE_FEATURES].to_numpy(), regime_column])

final_scaler = StandardScaler()
final_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)
final_model.fit(
    final_scaler.fit_transform(final_design[final_train]),
    frame["target"].to_numpy()[final_train],
)
final_predicted = final_model.predict(final_scaler.transform(final_design[final_test]))

test_dates = frame.index[final_test]
test_actual = frame["target"].to_numpy()[final_test]
test_probability = final_probability[final_test]
test_state = final_state[final_test]

print(f"Final fold test block: {test_dates[0].date()} to {test_dates[-1].date()}")
print(
    f"Test sessions in the volatile state: {int((test_state == VOLATILE_STATE).sum())} of {len(test_state)}"
)

# %%
fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)

ax = axes[0]
ax.plot(test_dates, frame["close"].to_numpy()[final_test], linewidth=0.9, color=COLORS["blue"])
ax.fill_between(
    test_dates,
    frame["close"].to_numpy()[final_test].min(),
    frame["close"].to_numpy()[final_test].max(),
    where=test_state == VOLATILE_STATE,
    alpha=0.2,
    color=COLORS["copper"],
)
ax.set_ylabel("US dollars")
ax.set_title("The volatile state falls on the drawdowns of the test block", fontsize=9)

ax = axes[1]
ax.fill_between(test_dates, 0, test_probability, alpha=0.6, color=COLORS["copper"])
ax.axhline(0.5, color=COLORS["recede"], linestyle="--", linewidth=0.7)
ax.set_ylabel("Probability")
ax.set_title("The filtered probability is continuous, not a switch", fontsize=9)

ax = axes[2]
for state, color, name in (
    (0, COLORS["blue"], "calm"),
    (VOLATILE_STATE, COLORS["copper"], "volatile"),
):
    selected = test_state == state
    ax.scatter(
        test_dates[selected],
        (final_predicted - test_actual)[selected],
        s=5,
        alpha=0.6,
        color=color,
        label=name,
    )
ax.axhline(0, color=COLORS["recede"], linestyle="--", linewidth=0.7)
ax.set_ylabel("Percent")
ax.set_xlabel("Session")
ax.set_title("The errors are wider in the volatile state", fontsize=9)
ax.legend(fontsize=7)

fig.suptitle("The final fold: where the regime feature was high and what it bought")
show_with_alt(
    fig,
    "Three stacked panels over the final fold's test block. The top plots the closing price "
    "with shaded bands where the hard state is the volatile one, and the bands cover the "
    "declines. The middle fills the filtered probability of that state against a dashed line "
    "at one half; it rises and falls smoothly rather than stepping. The bottom scatters the "
    "prediction error coloured by state, and the volatile points spread further from zero "
    "than the calm ones.",
)

# %%
errors = pd.DataFrame(
    [
        {
            "state": name,
            "sessions": int((test_state == state).sum()),
            "root mean squared error": float(
                np.sqrt(
                    mean_squared_error(
                        test_actual[test_state == state], final_predicted[test_state == state]
                    )
                )
            ),
            "standard deviation of the target": float(test_actual[test_state == state].std()),
        }
        for state, name in ((0, "calm"), (VOLATILE_STATE, "volatile"))
    ]
).set_index("state")

display(errors)

# %% [markdown]
# The error is larger in the volatile state, and the last column is what stops that being read
# as a finding about the model: the target itself is more dispersed there by about as much, so
# most of the difference is the question rather than the answer.
#
# What is left after that comparison is the part worth reading. In the volatile state the error
# is a little larger than the target's own standard deviation, which is what a negative r
# squared looks like at the level of one state: the model would have done better there by
# predicting that state's mean. In the calm state it is a little smaller. Reporting the error
# alone would have shown neither.

# %% [markdown]
# ## What the importance says, and what it does not
#
# The importance below is impurity-based: for each feature, the total reduction in squared
# error over the splits that used it. It measures what the fitted trees did, and that is
# narrower than what a reader usually wants it to mean.

# %%
importance = (
    pd.DataFrame(
        {
            "feature": [*BASE_FEATURES, REGIME_FEATURE],
            "importance": final_model.feature_importances_,
        }
    )
    .sort_values("importance")
    .set_index("feature")
)

fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.barh(importance.index, importance["importance"], color=COLORS["blue"])
ax.set_xlabel("Reduction in squared error over the splits that used it")
ax.set_title("Volatility carries the regime information the probability would add", fontsize=9)
show_with_alt(
    fig,
    "A horizontal bar chart of six features ordered by impurity-based importance. The "
    "rolling volatility and the two momentum sums take the largest bars, and the regime "
    "probability takes one of the smallest.",
)

display(importance.T)

# %% [markdown]
# A near-zero importance here does not say the regime is uninformative. It says this model, on
# these features, found little to split on that `volatility` did not already offer, and
# `volatility` is one of the two series the hidden Markov model was fitted on. The regime
# probability is a nonlinear summary of the same information with a persistence assumption
# attached, and a tree that can already split on the raw series reaches most of it directly.
#
# Two things follow. Where the base features do not contain what the regime model reads, the
# same column can rank far higher; the case studies from Chapter 16 use regime features
# alongside cross-sectional signals that carry no volatility of their own. And an importance
# split between two correlated columns is divided arbitrarily between them, so a low value on
# one of a correlated pair is not evidence about either.

# %% [markdown]
# ## The same idea as a feature catalog
#
# The regime feature above was built by hand from a fitted model. `ml4t-engineer` supplies a
# set of regime and volatility features as Polars expressions, composable in one
# `with_columns()`, and this is what a downstream pipeline in Chapters 11 and 12 receives.
#
# One argument convention is worth stating because getting it wrong produces a plausible
# column rather than an error. These expressions differ in what they take:
# `realized_volatility`, `garch_forecast` and `rolling_entropy` read a **return** column, while
# `hurst_exponent`, `choppiness_index`, `market_regime_classifier` and
# `volatility_regime_probability` read **prices** and difference them internally. Handing
# returns to one of the price expressions gives a column of numbers with the right shape and
# the wrong meaning.

# %%
CATALOG_WINDOW = 20
HURST_WINDOW = 100
CHOPPINESS_WINDOW = 14
ENTROPY_WINDOW = 50

catalog = (
    load_etfs(symbols=["SPY"])
    .select(["timestamp", "open", "high", "low", "close", "volume"])
    .sort("timestamp")
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .with_columns(returns=pl.col("close").pct_change())
    .drop_nulls()
    .with_columns(
        hurst=hurst_exponent("close", period=HURST_WINDOW),
        choppiness=choppiness_index("high", "low", "close", period=CHOPPINESS_WINDOW),
        trend_regime=market_regime_classifier("high", "low", "close", "volume"),
        realized=realized_volatility("returns", period=CATALOG_WINDOW),
        garch=garch_forecast("returns", horizon=1, alpha=0.1, beta=0.85),
        entropy=rolling_entropy("returns", window=ENTROPY_WINDOW, n_bins=10),
    )
    .with_columns(**volatility_regime_probability("close", period=CATALOG_WINDOW))
)

print(f"Catalog: {catalog.height:,} sessions, {catalog.width} columns")
display(
    catalog.select(
        ["timestamp", "hurst", "choppiness", "trend_regime", "realized", "garch", "entropy"]
    ).tail(3)
)

# %% [markdown]
# ### Interactions without writing them out
#
# `regime_conditional_features` multiplies one feature by an indicator for each value of a
# regime column, so each output column is the feature inside one regime and zero everywhere
# else. A linear model then fits a separate coefficient per regime without any branching, and a
# tree model gets a column that is already the interaction it would otherwise have to
# discover through two splits.
#
# The regime values below are the three `market_regime_classifier` emits: minus one for a
# downtrend, zero for a market with no trend the classifier will name, and one for an uptrend.
# The helper names each output column after its regime rather than after its position in the
# list, so the pairing between value and column is spelled out and checked below.

# %%
TREND_REGIMES = [-1, 0, 1]
# The helper names its columns for the regime value, so the pairing is stated rather than sorted.
REGIME_COLUMNS = ["feat_bear", "feat_neutral", "feat_bull"]

conditional = regime_conditional_features("returns", "trend_regime", regime_values=TREND_REGIMES)
assert sorted(conditional) == sorted(REGIME_COLUMNS), sorted(conditional)
with_interactions = catalog.with_columns(**conditional)

display(
    pl.DataFrame(
        [
            {
                "regime": regime,
                "column": name,
                "sessions in this regime": int((with_interactions["trend_regime"] == regime).sum()),
                "sessions where the column is not zero": int((with_interactions[name] != 0).sum()),
            }
            for regime, name in zip(TREND_REGIMES, REGIME_COLUMNS, strict=True)
        ]
    )
)

# %% [markdown]
# Two things to read here. The session counts partition the sample, since every session gets
# exactly one trend regime, and the classifier names a trend on only about a tenth of them: the
# neutral column carries almost everything and the two directional columns are sparse. Each of
# those is estimated from its own share of the data alone, which is the mixture-of-experts
# problem in a milder form.
#
# And the two counts per row differ. A session inside a regime whose return happens to be zero
# gives a zero in its own column, so counting non-zero values undercounts the regime. That makes
# no difference to a model, which reads the value and not the count, but it is the reason the
# two columns are shown side by side rather than one being taken for the other.

# %% [markdown]
# ## Takeaways
#
# 1. **A regime feature has to be refit inside the fold that uses it.** Fitting the hidden
#    Markov model once on the whole sample and then splitting the folds puts every fold's test
#    block into the parameters of its own feature. The cost of doing it correctly is one fit per
#    fold and a feature that disagrees with itself across folds on the sessions they share.
# 2. **Filtered probabilities, not smoothed ones.** `predict_proba` and Viterbi decoding both
#    condition on the whole sequence. The forward recursion conditions on the past, which is
#    what a value computed on the day can know.
# 3. **Read the spread across folds before the mean.** The error moves more between folds than
#    between designs here, and an averaged number with no spread beside it would have hidden
#    that. Five nested folds cannot support a significance claim about the difference, however
#    the pairing is done.
# 4. **The mixture of experts pays for its sharpness in sample size.** Each expert sees only its
#    own regime's training rows, and a misrouted test session reaches a model that never saw its
#    kind. The single model with a regime column degrades instead of switching.
# 5. **A low impurity importance on a regime column is a statement about the other columns.**
#    Where `volatility` is already a feature, a probability fitted on volatility adds little that
#    a tree cannot reach directly. The same column ranks higher where the base features carry no
#    volatility of their own.
# 6. **Check what each library expression consumes.** Half of the catalog above reads prices and
#    differences them internally; handing those a return column produces a plausible number and
#    no error.
#
# **Previous**: `11_hmm_regimes` fits the model whose probabilities this notebook spends, and
# `12_wasserstein_regimes` the clustering alternative. **Next**: `14_panel_features` moves from
# one series to a cross-section.
