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
# # Regime Detection with Hidden Markov Models
#
# **Docker image**: `ml4t`
#
# This notebook provides a thorough introduction to HMMs for financial regime
# detection, from first principles through production considerations.
#
# **Learning Objectives**:
# - Understand observable threshold baselines (VIX, moving average cross)
# - Implement the forward algorithm on a toy example
# - Compare filtered vs smoothed probabilities and identify look-ahead bias
# - Address EM estimation pitfalls: local optima, initialization, model selection
# - Prevent label switching across estimation windows
# - Fit Markov-Switching AR and compare with HMM
#
# **Book Reference**: Chapter 9, Section 9.5 (Regime Features)
#
# **Prerequisites**: `01_visual_diagnostics` for stationarity concepts;
# basic probability and matrix operations.

# %%
"""Regime Detection with Hidden Markov Models - thorough tutorial."""

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hmmlearn.hmm import GaussianHMM
from ml4t.engineer.features.regime import (
    choppiness_index,
    fractal_efficiency,
    hurst_exponent,
    market_regime_classifier,
    trend_intensity_index,
)
from scipy import stats
from sklearn.cluster import KMeans
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from data import load_etfs, load_macro
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

# %% tags=["parameters"]
# Production defaults - Papermill injects overrides for CI
N_INITS = 10
N_ITER = 200
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Load Data

# %%
CASE_DIR = get_case_study_dir("etfs")

etfs = load_etfs(symbols=["SPY"])
spy = (
    etfs.select(["timestamp", "open", "high", "low", "close", "volume"])
    .sort("timestamp")
    .with_columns(
        returns=pl.col("close").log().diff() * 100,
        volatility=pl.col("close").log().diff().rolling_std(window_size=21) * 100 * np.sqrt(252),
    )
    .drop_nulls()
)

spy_pd = spy.to_pandas().set_index("timestamp")
spy_pd.index = pd.DatetimeIndex(spy_pd.index)

print(
    f"SPY: {len(spy_pd):,} observations ({spy_pd.index.min().date()} to {spy_pd.index.max().date()})"
)

# %% [markdown]
# # Part 1 - Observable Threshold Baselines
#
# Before fitting latent-variable models, establish simple baselines using
# observable quantities. These are widely used in practice because they are
# transparent, deterministic, and free from estimation risk.

# %% [markdown]
# ### Baseline 1: VIX Threshold
#
# VIX > 20 is the classic "fear threshold." It requires no estimation and is
# available in real time from CBOE.

# %%
macro = load_macro()
vix = macro.select(["timestamp", "vixcls"]).drop_nulls().rename({"vixcls": "vix"}).sort("timestamp")

# Merge VIX into SPY data
vix_pd = vix.to_pandas().set_index("timestamp")
vix_pd.index = pd.DatetimeIndex(vix_pd.index)
spy_pd = spy_pd.join(vix_pd, how="left").ffill()

spy_pd["vix_regime"] = np.where(spy_pd["vix"] > 20, 1, 0)

pct_high = spy_pd["vix_regime"].mean()
print(f"VIX > 20: {pct_high:.1%} of trading days classified as high-vol")

# %% [markdown]
# ### Baseline 2: 200-Day Moving Average Cross
#
# Price above 200-day MA = uptrend; below = downtrend. A trend/momentum
# regime indicator used by CTAs and systematic macro funds.

# %%
spy_pd["ma_200"] = spy_pd["close"].rolling(200).mean()
spy_pd["trend_regime"] = np.where(spy_pd["close"] > spy_pd["ma_200"], 0, 1)

pct_downtrend = spy_pd["trend_regime"].dropna().mean()
print(f"Below 200-day MA: {pct_downtrend:.1%} of trading days classified as downtrend")

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax = axes[0]
ax.plot(spy_pd.index, spy_pd["close"], linewidth=0.8)
ax.plot(spy_pd.index, spy_pd["ma_200"], linewidth=1, color=COLORS["amber"], label="200-day MA")
ax.fill_between(
    spy_pd.index,
    spy_pd["close"].min(),
    spy_pd["close"].max(),
    where=spy_pd["trend_regime"] == 1,
    alpha=0.1,
    color=COLORS["negative"],
    label="Downtrend",
)
ax.set_title("SPY with 200-Day MA Regime")
ax.set_ylabel("Price")
ax.legend(loc="upper left")

ax = axes[1]
ax.fill_between(spy_pd.index, 0, spy_pd["vix"], alpha=0.3, color=COLORS["amber"])
ax.axhline(20, color=COLORS["neutral"], linestyle="--", linewidth=0.5, label="VIX = 20")
ax.set_title("VIX with Threshold Regime")
ax.set_ylabel("VIX")
ax.legend()

ax = axes[2]
combined = spy_pd["vix_regime"] + spy_pd["trend_regime"]
ax.fill_between(spy_pd.index, 0, combined.fillna(0), alpha=0.5, color=COLORS["negative"])
ax.set_title("Combined Stress (VIX>20 + Below 200-MA)")
ax.set_ylabel("Stress Count (0-2)")
ax.set_yticks([0, 1, 2])

plt.tight_layout()
plt.show()

# %% [markdown]
# **Takeaway**: Observable baselines are transparent and require zero estimation.
# They serve as the benchmark that any statistical model should beat.

# %% [markdown]
# # Part 2 - HMM Tutorial: Forward Algorithm
#
# Before fitting HMMs to financial data, we work through the mechanics on a
# small toy example. This builds intuition for what the model is actually
# computing.

# %% [markdown]
# ### HMM Setup
#
# An HMM has:
# - $K$ hidden states (e.g., bull/bear)
# - Transition matrix $A$ where $A_{ij} = P(\text{state}_t = j \mid \text{state}_{t-1} = i)$
# - Emission parameters: $P(\text{observation} \mid \text{state})$
# - Initial state distribution $\pi$
#
# The **forward algorithm** computes $P(\text{state}_t \mid \text{observations}_{1:t})$
# - the filtered probability given *only past and current* observations.

# %%
# Toy HMM: 2 states, 10 observations
# State 0: "Calm" (low vol, positive mean)
# State 1: "Stressed" (high vol, negative mean)
K = 2
T_toy = 10

# Transition matrix: states are sticky
A = np.array(
    [
        [0.95, 0.05],  # Calm → Calm: 95%, Calm → Stressed: 5%
        [0.10, 0.90],
    ]
)  # Stressed → Calm: 10%, Stressed → Stressed: 90%

# Emission parameters (Gaussian). The distributions overlap: a single
# observation near the boundary is genuinely ambiguous, which is precisely
# where filtered and smoothed estimates diverge (the look-ahead effect below).
means = np.array([0.05, -0.10])  # Calm: +5 bps, Stressed: -10 bps
stds = np.array([0.06, 0.08])  # Calm: lower vol, Stressed: higher vol

# Initial distribution
pi = np.array([0.8, 0.2])


# Sample a path from the true HMM. Draw with a local RNG and take the first
# seed whose path visits BOTH states - a path that never switches has no
# look-ahead gap to illustrate. This is deterministic and isolated from the
# global RNG (every model fit below sets its own random_state).
def sample_toy_path(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample (states, observations) of length T_toy from the true HMM."""
    states = np.zeros(T_toy, dtype=int)
    obs = np.zeros(T_toy)
    states[0] = rng.choice(K, p=pi)
    obs[0] = rng.normal(means[states[0]], stds[states[0]])
    for t in range(1, T_toy):
        states[t] = rng.choice(K, p=A[states[t - 1]])
        obs[t] = rng.normal(means[states[t]], stds[states[t]])
    return states, obs


for toy_seed in range(500):
    true_states, observations = sample_toy_path(np.random.default_rng(toy_seed))
    if true_states.min() == 0 and true_states.max() == 1:
        break

print("True states: ", true_states)
print("Observations:", np.round(observations, 4))

# %% [markdown]
# ### Forward Algorithm Implementation
#
# The forward variable $\alpha_t(k) = P(\text{obs}_{1:t}, \text{state}_t = k)$
# is computed recursively:
#
# $$\alpha_1(k) = \pi_k \cdot b_k(o_1)$$
# $$\alpha_t(k) = b_k(o_t) \sum_{j=1}^{K} \alpha_{t-1}(j) \cdot A_{jk}$$
#
# where $b_k(o_t) = P(o_t \mid \text{state} = k)$ is the emission probability.


# %%
def forward_algorithm(
    obs: np.ndarray, A: np.ndarray, means: np.ndarray, stds: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    """Compute forward probabilities (filtered) for a Gaussian HMM.

    Returns alpha[t, k] = P(obs_{1:t}, state_t = k).
    """
    T = len(obs)
    K = len(pi)
    alpha = np.zeros((T, K))

    # Initialization
    for k in range(K):
        alpha[0, k] = pi[k] * stats.norm.pdf(obs[0], means[k], stds[k])

    # Recursion
    for t in range(1, T):
        for k in range(K):
            emission = stats.norm.pdf(obs[t], means[k], stds[k])
            alpha[t, k] = emission * np.sum(alpha[t - 1, :] * A[:, k])

    return alpha


alpha = forward_algorithm(observations, A, means, stds, pi)

# Normalize to get filtered probabilities P(state_t | obs_{1:t})
filtered_probs = alpha / alpha.sum(axis=1, keepdims=True)

print("=== Forward Algorithm Results ===")
print(f"{'t':>3} {'Obs':>8} {'True':>5} {'P(Calm)':>10} {'P(Stress)':>10} {'Pred':>5}")
print("-" * 45)
for t in range(T_toy):
    pred = np.argmax(filtered_probs[t])
    print(
        f"{t:>3} {observations[t]:>8.4f} {true_states[t]:>5} "
        f"{filtered_probs[t, 0]:>10.4f} {filtered_probs[t, 1]:>10.4f} {pred:>5}"
    )

accuracy = (np.argmax(filtered_probs, axis=1) == true_states).mean()
print(f"\nFiltered accuracy: {accuracy:.1%}")

# %% [markdown]
# ### Filtered vs Smoothed Probabilities
#
# The **filtered** probability $P(\text{state}_t \mid \text{obs}_{1:t})$ uses only
# past and current data - it is **causal** and safe for trading.
#
# The **smoothed** probability $P(\text{state}_t \mid \text{obs}_{1:T})$ uses the
# entire sample including *future* data - it has **look-ahead bias**.
#
# Using smoothed probabilities as features in a backtest is a common mistake
# that inflates performance.

# %%
# Compute smoothed probabilities using hmmlearn for comparison
toy_hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=1)
# Set parameters manually to match our toy model
toy_hmm.startprob_ = pi
toy_hmm.transmat_ = A
toy_hmm.means_ = means.reshape(-1, 1)
toy_hmm.covars_ = (stds**2).reshape(-1, 1)

# hmmlearn's predict_proba returns smoothed by default
obs_2d = observations.reshape(-1, 1)
smoothed_probs = toy_hmm.predict_proba(obs_2d)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

t_range = np.arange(T_toy)

ax = axes[0]
ax.plot(
    t_range,
    filtered_probs[:, 0],
    color=COLORS["blue"],
    marker="o",
    linestyle="-",
    label="Filtered P(Calm)",
    markersize=6,
)
ax.plot(
    t_range,
    smoothed_probs[:, 0],
    color=COLORS["copper"],
    marker="s",
    linestyle="--",
    label="Smoothed P(Calm)",
    markersize=6,
)
ax.set_title("Filtered vs Smoothed Probabilities - Toy Example")
ax.set_ylabel("P(Calm)")
# Pin to [0, 1] and turn off the scientific-notation offset that matplotlib
# auto-applies when both series saturate near 1.0 - the +1 offset makes the
# probability differences look like ~1e-5 noise.
ax.set_ylim(-0.05, 1.05)
ax.ticklabel_format(useOffset=False, axis="y")
ax.legend()

# Highlight look-ahead bias
ax = axes[1]
bias = smoothed_probs[:, 0] - filtered_probs[:, 0]
ax.bar(
    t_range,
    bias,
    color=[
        COLORS["negative"] if b > 0.01 else COLORS["positive"] if b < -0.01 else COLORS["neutral"]
        for b in bias
    ],
)
ax.axhline(0, color=COLORS["neutral"], linewidth=0.5)
ax.set_title("Look-Ahead Bias (Smoothed - Filtered)")
ax.set_ylabel("Probability Difference")
ax.set_xlabel("Time Step")

plt.tight_layout()
plt.show()

print(f"Mean absolute bias: {np.mean(np.abs(bias)):.4f}")
print("Smoothed probabilities use future data → look-ahead bias in backtests!")

# %% [markdown]
# # Part 3 - HMM on Financial Data
#
# Now we apply HMMs to SPY returns, addressing practical challenges:
# initialization, model selection, and label switching.

# %% [markdown]
# ### Multiple Random Initializations
#
# EM is sensitive to initialization. Running multiple starts and selecting
# the best log-likelihood reduces the risk of local optima.

# %%
X = spy_pd[["returns", "volatility"]].dropna().values

n_inits = N_INITS
best_ll = -np.inf
best_model = None
log_likelihoods = []

for seed in range(n_inits):
    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(X)
    ll = model.score(X)
    log_likelihoods.append(ll)
    if ll > best_ll:
        best_ll = ll
        best_model = model

print(f"=== EM Initialization: {n_inits} Random Starts ===")
print(f"Log-likelihoods: {[f'{ll:.1f}' for ll in log_likelihoods]}")
print(f"Range: {max(log_likelihoods) - min(log_likelihoods):.1f}")
print(f"Best: {best_ll:.1f} (seed {np.argmax(log_likelihoods)})")

# %% [markdown]
# ### K-Means Seeded Initialization
#
# Instead of random initialization, use k-means clustering to provide
# better starting values for the emission parameters.


# %%
def fit_hmm_kmeans_init(X: np.ndarray, n_states: int, random_state: int = 42) -> GaussianHMM:
    """Fit HMM with k-means-seeded initialization."""
    # K-means for initial emission parameters
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    kmeans.fit(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=random_state,
        init_params="st",  # Only init startprob and transmat
    )

    # Set emission parameters from k-means
    model.means_ = kmeans.cluster_centers_
    model.covars_ = np.array(
        [np.cov(X[kmeans.labels_ == k].T) + np.eye(X.shape[1]) * 1e-6 for k in range(n_states)]
    )

    model.fit(X)
    return model


hmm_kmeans = fit_hmm_kmeans_init(X, n_states=2)
ll_kmeans = hmm_kmeans.score(X)
print(f"K-means init log-likelihood: {ll_kmeans:.1f}")
print(f"Best random init log-likelihood: {best_ll:.1f}")
print(f"K-means {'better' if ll_kmeans > best_ll else 'worse'} by {abs(ll_kmeans - best_ll):.1f}")

# Use the better model
hmm2 = hmm_kmeans if ll_kmeans > best_ll else best_model
assert hmm2 is not None

# %% [markdown]
# ### BIC for Number of States
#
# Compare 2, 3, and 4 states using BIC (Bayesian Information Criterion).
# More states always improve log-likelihood but may overfit.

# %%
bic_results = []
models = {}

for k in [2, 3, 4]:
    model = fit_hmm_kmeans_init(X, n_states=k)
    ll = model.score(X)

    # BIC = -2*LL + n_params * log(n_obs)
    # model.score(X) returns the TOTAL log-likelihood, not per-sample
    n_obs = len(X)
    d = X.shape[1]
    # Parameters: (k-1) initial + k*(k-1) transition + k*d means + k*d*(d+1)/2 covariances
    n_params = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2
    bic = -2 * ll + n_params * np.log(n_obs)

    bic_results.append({"K": k, "LL": ll, "n_params": n_params, "BIC": bic})
    models[k] = model
    print(f"K={k}: LL={ll:.1f}, params={n_params}, BIC={bic:.0f}")

# Select best by BIC
best_k = min(bic_results, key=lambda x: x["BIC"])["K"]
print(f"\nBest K by BIC: {best_k}")

# %% [markdown]
# BIC decreases monotonically through K=4, suggesting the data supports more
# than two states in-sample. However, BIC is known to over-select states for
# HMMs on long financial series - more states always capture more volatility
# clustering patterns, but these additional regimes often fail to persist
# out-of-sample. We proceed with **K=2** (calm/stressed) for interpretability
# and robustness, consistent with the common finding that equity returns are
# well-described by two volatility regimes.

# %% [markdown]
# ### Label Switching Prevention
#
# HMM states are unordered - "State 0" might be high-vol in one estimation
# and low-vol in another. Sort states by a consistent property (variance)
# to prevent label switching.


# %%
def sort_states_by_variance(model: GaussianHMM) -> np.ndarray:
    """Sort HMM states by variance (ascending) for consistent labeling.

    Returns (sorted_means, sorted_covars, state_order).
    """
    # Compute total variance for each state
    variances = np.array([np.trace(model.covars_[k]) for k in range(model.n_components)])
    order = np.argsort(variances)  # Low vol first

    return order


# %%
def relabel_states(states: np.ndarray, probs: np.ndarray, order: np.ndarray) -> tuple:
    """Relabel states according to the given order."""
    inv_order = np.argsort(order)
    new_states = inv_order[states]
    new_probs = probs[:, order]
    return new_states, new_probs


# Apply to 2-state model
order_2 = sort_states_by_variance(hmm2)
states_2 = hmm2.predict(X)
probs_2 = hmm2.predict_proba(X)

states_sorted, probs_sorted = relabel_states(states_2, probs_2, order_2)

# Verify: State 0 should have lower volatility
for k in range(2):
    mask = states_sorted == k
    label = "Low-Vol" if k == 0 else "High-Vol"
    mean_vol = spy_pd.loc[spy_pd.index[: len(mask)][mask], "volatility"].mean()
    mean_ret = spy_pd.loc[spy_pd.index[: len(mask)][mask], "returns"].mean()
    pct = mask.mean()
    print(f"State {k} ({label}): {pct:.1%} of time, mean ret={mean_ret:.3f}%, vol={mean_vol:.1f}%")

# %% [markdown]
# ### Filtered Probabilities for Production
#
# In production, use **filtered** (not smoothed) probabilities to avoid
# look-ahead bias. hmmlearn's `predict_proba` returns smoothed by default.
# We implement the forward algorithm directly using hmmlearn's internal
# `_compute_log_likelihood` for per-observation emission probabilities.
# This is a private API - if it changes between hmmlearn versions, the
# forward pass logic itself (below) remains correct with any emission source.


# %%
def compute_filtered_probs(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Compute filtered probabilities P(state_t | obs_{1:t}).

    Uses the forward algorithm internally, then normalizes.
    """
    # hmmlearn provides _compute_log_likelihood for emissions
    framelogprob = model._compute_log_likelihood(X)

    n_samples = X.shape[0]
    n_components = model.n_components

    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    # Forward pass (log-domain for numerical stability)
    fwdlattice = np.zeros((n_samples, n_components))

    # Initialization
    fwdlattice[0] = log_startprob + framelogprob[0]

    # Recursion
    for t in range(1, n_samples):
        for j in range(n_components):
            fwdlattice[t, j] = framelogprob[t, j] + np.logaddexp.reduce(
                fwdlattice[t - 1] + log_transmat[:, j]
            )

    # Normalize to get probabilities
    log_normalizer = np.logaddexp.reduce(fwdlattice, axis=1, keepdims=True)
    filtered = np.exp(fwdlattice - log_normalizer)

    return filtered


filtered = compute_filtered_probs(hmm2, X)
smoothed = hmm2.predict_proba(X)

# Apply label sorting
filtered_sorted = filtered[:, order_2]
smoothed_sorted = smoothed[:, order_2]

# %% [markdown]
# ### Filtered vs Smoothed on Real Data
#
# The difference is most visible around regime transitions - smoothed
# probabilities "know" the transition is coming before it happens.

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

dates = spy_pd.index[: len(filtered)]

ax = axes[0]
ax.plot(dates, spy_pd["close"].values[: len(filtered)], linewidth=0.5)
ax.set_title("SPY Price")
ax.set_ylabel("Price")

ax = axes[1]
ax.plot(dates, filtered_sorted[:, 1], linewidth=0.8, label="Filtered P(High-Vol)", alpha=0.7)
ax.plot(dates, smoothed_sorted[:, 1], linewidth=0.8, label="Smoothed P(High-Vol)", alpha=0.7)
ax.set_title("High-Vol State Probability: Filtered vs Smoothed")
ax.set_ylabel("Probability")
ax.legend()

# Show bias in a focused window around COVID
ax = axes[2]
covid_mask = (dates >= "2019-06-01") & (dates <= "2020-12-31")
if covid_mask.sum() > 0:
    ax.plot(
        dates[covid_mask], filtered_sorted[covid_mask, 1], linewidth=1.5, label="Filtered (causal)"
    )
    ax.plot(
        dates[covid_mask],
        smoothed_sorted[covid_mask, 1],
        linewidth=1.5,
        label="Smoothed (look-ahead)",
    )
    ax.set_title("COVID Period: Smoothed Anticipates Transition")
    ax.set_ylabel("P(High-Vol)")
    ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# # Part 4 - Regime Features
#
# Extract features from the fitted HMM for downstream ML models.

# %%
spy_pd["hmm_state"] = states_sorted[: len(spy_pd)]
spy_pd["regime_prob_high"] = filtered_sorted[: len(spy_pd), 1]
spy_pd["regime_prob_low"] = filtered_sorted[: len(spy_pd), 0]

# Transition probabilities as features
transmat = hmm2.transmat_[order_2][:, order_2]
spy_pd["transition_low_to_high"] = transmat[0, 1]
spy_pd["transition_high_to_low"] = transmat[1, 0]

# Expected duration in current state
# E[duration in state k] = 1 / (1 - A_{kk})
spy_pd["expected_duration"] = np.where(
    spy_pd["hmm_state"] == 0,
    1 / (1 - transmat[0, 0]),
    1 / (1 - transmat[1, 1]),
)

# Regime entropy: uncertainty about current state
# High entropy = uncertain, low entropy = confident
eps = 1e-10
p = filtered_sorted[: len(spy_pd)]
spy_pd["regime_entropy"] = -np.sum(p * np.log(p + eps), axis=1)

print("=== Regime Feature Summary ===")
print("Transition matrix (sorted):")
print(f"  P(Low→Low):  {transmat[0, 0]:.4f}  P(Low→High):  {transmat[0, 1]:.4f}")
print(f"  P(High→Low): {transmat[1, 0]:.4f}  P(High→High): {transmat[1, 1]:.4f}")
print("\nExpected durations:")
print(f"  Low-vol regime:  {1 / (1 - transmat[0, 0]):.0f} days")
print(f"  High-vol regime: {1 / (1 - transmat[1, 1]):.0f} days")

# %% [markdown]
# # Part 5 - Markov-Switching AR (Hamilton)
#
# MS-AR provides a complementary approach with different advantages:
# - Explicitly models AR dynamics within each regime
# - Provides maximum likelihood with analytical gradients
# - Includes regime-specific AR coefficients and variances

# %%
returns_clean = spy_pd["returns"].dropna()

# statsmodels warns that the DatetimeIndex carries no frequency; the index is
# used only for later alignment, so suppress it locally (statsmodels resets the
# module-level filter, so a catch_warnings block is needed).
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    msar = MarkovAutoregression(
        returns_clean,
        k_regimes=2,
        order=1,
        switching_ar=False,
        switching_variance=True,
    )
    msar_result = msar.fit(disp=False)

print("=== MS-AR(1) Model ===")
print(f"Regime 0 variance: {msar_result.params['sigma2[0]']:.4f}")
print(f"Regime 1 variance: {msar_result.params['sigma2[1]']:.4f}")

# Extract both filtered and smoothed probabilities
# Filtered: conditions only on past/current observations given the fitted model (causal)
# Smoothed: uses the full sample (for diagnostics/comparison only)
msar_filtered_0 = msar_result.filtered_marginal_probabilities[0]
msar_filtered_1 = msar_result.filtered_marginal_probabilities[1]
msar_smoothed_0 = msar_result.smoothed_marginal_probabilities[0]
msar_smoothed_1 = msar_result.smoothed_marginal_probabilities[1]

# Align indices - MS-AR drops observations due to AR lag
spy_pd["msar_filtered_0"] = np.nan
spy_pd["msar_filtered_1"] = np.nan
spy_pd["msar_smoothed_0"] = np.nan
spy_pd["msar_smoothed_1"] = np.nan

common_idx = spy_pd.index.intersection(msar_filtered_0.index)
spy_pd.loc[common_idx, "msar_filtered_0"] = msar_filtered_0.loc[common_idx].values
spy_pd.loc[common_idx, "msar_filtered_1"] = msar_filtered_1.loc[common_idx].values
spy_pd.loc[common_idx, "msar_smoothed_0"] = msar_smoothed_0.loc[common_idx].values
spy_pd.loc[common_idx, "msar_smoothed_1"] = msar_smoothed_1.loc[common_idx].values

# Identify which MS-AR regime is high-vol
var_0 = msar_result.params["sigma2[0]"]
var_1 = msar_result.params["sigma2[1]"]
high_vol_regime = 0 if var_0 > var_1 else 1
print(f"High-vol regime: {high_vol_regime} (σ²={max(var_0, var_1):.4f})")

# %% [markdown]
# **Filtered vs smoothed MS-AR**: Like HMMs, MS-AR provides both filtered and
# smoothed marginal probabilities. The filtered probabilities use only past and
# current observations - these are the production-safe features. Smoothed
# probabilities incorporate the full sample and are useful only for historical
# analysis.

# %% [markdown]
# ### Method Comparison

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

ax = axes[0]
ax.plot(spy_pd.index, spy_pd["close"], linewidth=0.5)
ax.set_title("SPY Price")
ax.set_ylabel("Price")

ax = axes[1]
ax.fill_between(
    spy_pd.index, 0, spy_pd["vix_regime"], alpha=0.5, color=COLORS["amber"], label="VIX > 20"
)
ax.set_title("Baseline: VIX Threshold")
ax.set_ylabel("Regime")
ax.legend()

ax = axes[2]
ax.fill_between(spy_pd.index, 0, spy_pd["regime_prob_high"], alpha=0.7, color=COLORS["negative"])
ax.set_title("HMM: Filtered P(High-Vol)")
ax.set_ylabel("Probability")

ax = axes[3]
msar_high = spy_pd[f"msar_filtered_{high_vol_regime}"].fillna(0)
ax.fill_between(spy_pd.index, 0, msar_high, alpha=0.7, color=COLORS["amber"])
ax.set_title("MS-AR: Filtered P(High-Vol)")
ax.set_ylabel("Probability")

plt.tight_layout()
plt.show()

# %%
# Agreement analysis
valid = spy_pd[["vix_regime", "regime_prob_high"]].dropna()
hmm_high = (valid["regime_prob_high"] > 0.5).astype(int)
vix_high = valid["vix_regime"]

agreement = (hmm_high == vix_high).mean()
print(f"HMM vs VIX agreement: {agreement:.1%}")

# Confusion analysis
tp = ((hmm_high == 1) & (vix_high == 1)).sum()
fp = ((hmm_high == 1) & (vix_high == 0)).sum()
fn = ((hmm_high == 0) & (vix_high == 1)).sum()
tn = ((hmm_high == 0) & (vix_high == 0)).sum()

print("\nHMM vs VIX Confusion (VIX as 'truth'):")
print(f"  True Positive:  {tp:>5}  False Positive: {fp:>5}")
print(f"  False Negative: {fn:>5}  True Negative:  {tn:>5}")

# %% [markdown]
# **Interpretation**: HMM and VIX-based regimes agree on most days but diverge
# meaningfully at transitions. False positives (HMM detects stress that VIX
# misses) often correspond to elevated realized volatility before VIX catches up.
# False negatives (VIX elevated but HMM calm) may reflect VIX overshooting
# during event-driven spikes. Neither is "correct" - they measure different
# aspects of the volatility regime.

# %% [markdown]
# # Part 6 - Indicator-Based Regime Detection (ml4t-engineer)
#
# HMMs are probabilistic and data-driven - they learn regime structure from
# the data. An alternative approach uses **deterministic indicators** from
# technical analysis that classify regime based on fixed rules applied to
# price dynamics.
#
# `ml4t-engineer` provides five regime indicators as Polars expressions,
# plus a composite `market_regime_classifier` that combines them.

# %%
spy_indicators = spy.with_columns(
    chop=choppiness_index("high", "low", "close", period=14),
    hurst=hurst_exponent("close", period=100),
    fractal_eff=fractal_efficiency("close", period=20),
    trend_intensity=trend_intensity_index("close", period=30),
    regime=market_regime_classifier("high", "low", "close", "volume"),
).drop_nulls(subset=["chop"])

print("=== Indicator-Based Regime Features ===")
for col in ["chop", "hurst", "fractal_eff", "trend_intensity", "regime"]:
    vals = spy_indicators[col].drop_nulls()
    print(f"  {col:<18}: mean={vals.mean():.3f}, std={vals.std():.3f}")

# %% [markdown]
# ### HMM vs Indicator Comparison
#
# We compare the HMM high-volatility state with indicator-based regimes.
# These capture **different concepts**: the HMM identifies volatility regimes
# (calm vs stressed) from return dynamics, while `market_regime_classifier`
# produces trend regimes (bearish/range-bound = -1, neutral = 0, bullish = 1)
# from price structure. High volatility and bearish trend often co-occur
# during sell-offs but can diverge - recovery rallies are high-vol + bullish,
# and slow grinds can be low-vol + bearish. The comparison measures overlap,
# not equivalence.

# %%
# Align indicator regimes with HMM period
ind_pd = (
    spy_indicators.select(
        ["timestamp", "chop", "hurst", "fractal_eff", "trend_intensity", "regime"]
    )
    .to_pandas()
    .set_index("timestamp")
)
ind_pd.index = pd.DatetimeIndex(ind_pd.index)

# Align to common dates
common = spy_pd.index.intersection(ind_pd.index)
hmm_high_vol = (spy_pd.loc[common, "regime_prob_high"] > 0.5).astype(int)
ind_bearish = (ind_pd.loc[common, "regime"] == -1).astype(int)  # -1 = bearish/range-bound

agreement = (hmm_high_vol == ind_bearish).mean()
print(f"HMM high-vol vs Indicator bearish overlap: {agreement:.1%}")
print("(These are different regime definitions - overlap, not equivalence)")

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

ax = axes[0]
ax.plot(common, spy_pd.loc[common, "close"], linewidth=0.5)
ax.set_title("SPY Price")
ax.set_ylabel("Price")

ax = axes[1]
ax.fill_between(
    common, 0, spy_pd.loc[common, "regime_prob_high"], alpha=0.7, color=COLORS["negative"]
)
ax.set_title("HMM: Filtered P(High-Vol) - Probabilistic")
ax.set_ylabel("Probability")

ax = axes[2]
ax.fill_between(
    common,
    0,
    ind_pd.loc[common, "chop"] / 100,
    alpha=0.5,
    color=COLORS["blue"],
    label="Choppiness Index / 100",
)
ax.plot(common, ind_pd.loc[common, "hurst"], linewidth=0.8, color=COLORS["positive"], label="Hurst")
ax.axhline(0.5, color=COLORS["neutral"], linestyle="--", linewidth=0.5)
ax.set_title("Indicator Regime Features - Deterministic")
ax.set_ylabel("Value")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Comparison**: HMM captures regime switches through latent-state inference -
# it adapts to the data but requires estimation and is sensitive to initialization.
# Indicator-based regimes are transparent and deterministic but use fixed rules
# that may not adapt to structural changes. In practice, combining both (e.g.,
# using indicators as additional HMM features) often works best.

# %% [markdown]
# ## Save Regime Features
#
# Save regime states as a chapter demonstration artifact (see the caveat below
# for why case-study pipelines must not read it).
# The cross-join below broadcasts SPY-derived regime states to all symbols
# in the case study universe - a simplification appropriate for market-level
# regime features. Per-asset regime models would require individual HMM fits.
#
# Both the probability columns and the hard `regime_hmm`/`vol_regime` labels are
# **filtered** (causal): each date uses only past and current observations. The
# hard label is the argmax of the filtered probabilities, not `hmm3.predict`
# (Viterbi), which would look ahead. One honest caveat remains: the HMM
# **parameters** are estimated by EM over the whole sample, so these features are
# point-in-time in their state inference but not in their parameter estimation. A
# strictly live pipeline re-fits the model walk-forward; here we keep a single fit
# for clarity, and the filtered inference is the part that most affects backtest
# realism.
#
# Because of that parameter-level look-ahead, the artifact filename carries a
# `_fullsample_demo` suffix: it is a chapter demonstration and **must not be
# consumed by case-study or holdout-evaluated pipelines** — its EM parameters
# saw the sealed holdout period. The lookahead-safe version of these features
# (HMM refit per CV fold, filtered probabilities extracted per fold) is built
# in `case_studies/etfs/04_model_based_features.py`; downstream chapters should
# read that per-fold artifact instead.

# %%
MODEL_DIR = CASE_DIR / "models" / "time_series"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 3-state HMM for richer regime information
hmm3 = fit_hmm_kmeans_init(X, n_states=3)
order_3 = sort_states_by_variance(hmm3)

# Filtered (causal) probabilities: the point-in-time regime evidence at each date,
# using only past and current observations (see the filtered-vs-smoothed discussion
# above). These are the columns we save as features.
filtered_3 = compute_filtered_probs(hmm3, X)
filtered_3_sorted = filtered_3[:, order_3]

# Hard regime label = argmax of the FILTERED probabilities, so the saved state
# feature is causal and consistent with the probability columns. We deliberately do
# NOT use hmm3.predict(X): that is the Viterbi global decode, which conditions on the
# entire sample (including the future) and would embed look-ahead bias in a feature
# meant for backtesting.
states_3_causal = filtered_3_sorted.argmax(axis=1)

# %%
# Build output with all symbols
all_symbols = load_etfs().select("symbol").unique().sort("symbol").get_column("symbol")

vol_regime_map = {0: "low", 1: "normal", 2: "high"}

base_df = pl.DataFrame(
    {
        "timestamp": spy_pd.index[: len(states_3_causal)].values,
        "regime_hmm": states_3_causal,
        "regime_prob_0": filtered_3_sorted[:, 0],
        "regime_prob_1": filtered_3_sorted[:, 1],
        "regime_prob_2": filtered_3_sorted[:, 2],
        "vol_regime": [vol_regime_map[s] for s in states_3_causal],
        "trend_regime": spy_pd["trend_regime"].values[: len(states_3_causal)],
    }
)

symbols_df = pl.DataFrame({"symbol": all_symbols})
regime_df = base_df.join(symbols_df, how="cross")
regime_df = regime_df.select(
    [
        "timestamp",
        "symbol",
        "regime_hmm",
        "regime_prob_0",
        "regime_prob_1",
        "regime_prob_2",
        "vol_regime",
        "trend_regime",
    ]
)

# `_fullsample_demo`: EM parameters saw the full sample (see caveat above) —
# not for case-study/holdout-evaluated consumption.
output_path = MODEL_DIR / "regime_states_fullsample_demo.parquet"
regime_df.write_parquet(output_path)
print(f"Saved regime states (full-sample demo): {regime_df.shape}")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Observable baselines** (VIX > 20, 200-day MA) are transparent and
#    require no estimation - they set the bar any statistical model must clear
# 2. **The forward algorithm** computes filtered probabilities using only past
#    data - essential for avoiding look-ahead bias in trading
# 3. **Smoothed probabilities use future data** - never use them as features
#    in a backtest or live trading system
# 4. **EM is sensitive to initialization** - use multiple random starts or
#    k-means seeding to find better optima
# 5. **BIC selects the number of states** - more states always improve fit
#    but risk overfitting; 2-3 states usually suffice for financial data
# 6. **Sort states by variance** to prevent label switching across
#    estimation windows
# 7. **Regime probabilities are better features than hard classifications** -
#    they preserve uncertainty and degrade gracefully
# 8. **Indicator-based regime detection** (choppiness, Hurst, fractal efficiency)
#    via ml4t-engineer provides transparent, deterministic alternatives to HMM -
#    combine both approaches for robust regime features
#
# **Next**: See `12_wasserstein_regimes` for distribution-based clustering
# and `13_regime_as_feature` for integrating regime features into ML pipelines.
