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
# # Wasserstein Regime Clustering
#
# **Chapter 9 | Section 9.5**
#
# **Docker image**: `ml4t`
#
# The previous notebook fitted a model that says which of two states generated each return.
# This one asks a different question of the same data. Cut the return series into windows,
# treat each window as a distribution in its own right, and group the windows that look
# alike. Nothing is assumed about how a regime behaves or how it switches; what is assumed
# is a way of measuring how far one distribution is from another.
#
# That measure is the **Wasserstein distance**, and in one dimension it is the distance
# between quantile functions. It reads the whole shape rather than a mean and a variance,
# which is what makes it worth the extra machinery: two windows can agree on both moments
# and disagree about where their losses sit.
#
# **Learning objectives**
#
# - Cut a return series into windows and treat each one as an empirical distribution.
# - Compute the one-dimensional Wasserstein distance and the barycenter it implies, and see
#   why both reduce to operations on sorted returns.
# - Cluster the windows with it, and measure against a ground truth what a two-moment
#   summary of the same windows misses.
# - Turn the result into columns that a session's own history could have produced, which
#   requires fitting the centroids on a first block and assigning the rest forward.
#
# **Book reference**
#
# Chapter 9, Section 9.5 (Regime features).
#
# **Prerequisites**
#
# `11_hmm_regimes` for the filtered-against-smoothed distinction, which decides everything
# about how the features below are built. Quantiles and k-means.

# %% [markdown]
# ## Setup

# %%
"""Wasserstein regime clustering - clustering return windows as distributions."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display
from matplotlib.patches import Patch
from ml4t.diagnostic.evaluation.drift import compute_psi, compute_wasserstein_distance
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from data import load_sp500_index
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, FIGSIZE, show_with_alt

# %% tags=["parameters"]
START_DATE = "1980-01-01"
END_DATE = "2024-12-31"
WINDOW_LEN = 21
OVERLAP = 5
N_CLUSTERS = 2
WASSERSTEIN_P = 1.0
N_MOMENTS = 4
N_INIT = 10
MAX_ITER = 100
TRAIN_FRACTION = 0.6
N_STEPS = 2000
N_SWITCHES = 6
MMD_BOOTSTRAPS = 200
MMD_SAMPLE_SIZE = 200
SEED = 42

# %%
set_global_seeds(SEED)

# %% [markdown]
# ## Cutting the series into distributions
#
# The **stream lift** is the step that turns one series into many distributions. A window of
# `WINDOW_LEN` sessions slides along the returns in steps of `WINDOW_LEN - OVERLAP`, and
# each position of the window is one sample of that many returns. Sorting the returns inside
# a window loses their order, which is the point: what is kept is the distribution the
# window drew from, and two windows with the same distribution and different orderings are
# meant to look identical.
#
# The two lengths trade against each other. A long window estimates the distribution better
# and reacts to a change later. Overlap buys more windows out of the same data without
# buying more information, since consecutive windows share `OVERLAP` sessions.

# %%
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class LiftedStream:
    """Windows of a return series, raw and sorted, with the index each one starts at."""

    segments: FloatArray
    sorted_segments: FloatArray
    starts: IntArray
    window_len: int
    step: int


def lift_stream(returns: FloatArray, window_len: int, overlap: int) -> LiftedStream:
    """Cut a return series into overlapping windows of length `window_len`."""
    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if window_len < 2:
        raise ValueError("window_len must be at least 2")
    if not 0 <= overlap < window_len:
        raise ValueError("overlap must be at least 0 and less than window_len")
    if returns.shape[0] < window_len:
        raise ValueError("returns must be at least window_len long")

    step = window_len - overlap
    windows = np.lib.stride_tricks.sliding_window_view(returns, window_shape=window_len)[::step]
    segments = np.ascontiguousarray(windows, dtype=np.float64)

    return LiftedStream(
        segments=segments,
        sorted_segments=np.sort(segments, axis=1),
        starts=np.arange(0, segments.shape[0] * step, step, dtype=np.int64),
        window_len=window_len,
        step=step,
    )


# %% [markdown]
# ## The distance, and the average it implies
#
# For two samples of equal size the $p$-Wasserstein distance is an average over matched
# quantiles:
#
# $$W_p(a, b) = \left(\frac{1}{n}\sum_{i=1}^{n} |a_{(i)} - b_{(i)}|^p\right)^{1/p}$$
#
# where $a_{(i)}$ is the $i$-th smallest value. The optimal transport plan between two equal
# sized samples on the line is the one that matches them in order, so sorting is the whole
# computation. Every quantile contributes, which is why a difference confined to the worst
# few returns registers here and not in a variance.
#
# Clustering needs an average as well as a distance, and the average that goes with this
# distance is the **barycenter**: the sample minimising the sum of $W_p^p$ to the members of
# a group. Because the distance decomposes across matched quantiles, so does the
# minimisation, and it has a closed form. For $p = 1$ the barycenter is the quantile-wise
# median of the members; for $p = 2$ it is the quantile-wise mean.


# %%
def wasserstein_distance_1d(sorted_a: FloatArray, sorted_b: FloatArray, p: float) -> float:
    """The p-Wasserstein distance between two equal-sized samples, both already sorted."""
    if sorted_a.shape != sorted_b.shape:
        raise ValueError("both samples must have the same shape")
    return float((np.abs(sorted_a - sorted_b) ** p).mean() ** (1.0 / p))


def distances_to_centroid(
    sorted_segments: FloatArray, centroid: FloatArray, p: float
) -> FloatArray:
    """The distance from every window to one centroid."""
    return (np.abs(sorted_segments - centroid[None, :]) ** p).mean(axis=1) ** (1.0 / p)


def wasserstein_barycenter_1d(sorted_members: FloatArray, p: float) -> FloatArray:
    """The quantile-wise median (p equal to one) or mean (p equal to two) of the members."""
    if sorted_members.ndim != 2:
        raise ValueError("sorted_members must be two-dimensional")
    if sorted_members.shape[0] == 0:
        raise ValueError("a barycenter needs at least one member")
    if p == 1.0:
        return np.median(sorted_members, axis=0).astype(np.float64)
    if p == 2.0:
        return sorted_members.mean(axis=0).astype(np.float64)
    raise ValueError("p must be 1 or 2, the two exponents with a closed-form barycenter")


# %% [markdown]
# ## Lloyd's algorithm with a different distance
#
# With a distance and an average in hand, k-means needs nothing else. Assign every window to
# its nearest centroid, replace each centroid with the barycenter of its members, and repeat
# until the centroids stop moving. The initialisation is the k-means++ rule with the
# Wasserstein distance in place of the Euclidean one, and the whole fit is repeated `N_INIT`
# times because Lloyd's algorithm finds a local optimum and which one depends on where it
# started.


# %%
@dataclass(frozen=True)
class ClusteringResult:
    """Labels and centroids of one fit, with what the fit did to get there."""

    labels: IntArray
    centroids: FloatArray
    inertia: float
    n_iter: int
    converged: bool


class WassersteinKMeans1D:
    """k-means over one-dimensional samples under the p-Wasserstein distance."""

    def __init__(
        self,
        n_clusters: int,
        p: float,
        n_init: int,
        max_iter: int,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.p = p
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, sorted_segments: FloatArray) -> ClusteringResult:
        """Run Lloyd's algorithm `n_init` times and keep the fit with the lowest inertia."""
        if sorted_segments.shape[0] < self.n_clusters:
            raise ValueError("there must be at least as many windows as clusters")

        rng = np.random.default_rng(self.random_state)
        best = min(
            (
                self._lloyd(sorted_segments, self._seed_centroids(sorted_segments, rng))
                for _ in range(self.n_init)
            ),
            key=lambda result: result.inertia,
        )
        return best

    def predict(self, sorted_segments: FloatArray, centroids: FloatArray) -> IntArray:
        """Assign windows to the nearest of centroids fitted somewhere else."""
        return self._distance_matrix(sorted_segments, centroids).argmin(axis=1).astype(np.int64)

    def _distance_matrix(self, sorted_segments: FloatArray, centroids: FloatArray) -> FloatArray:
        return np.column_stack(
            [distances_to_centroid(sorted_segments, centroid, self.p) for centroid in centroids]
        )

    def _seed_centroids(self, sorted_segments: FloatArray, rng: np.random.Generator) -> FloatArray:
        n_samples = sorted_segments.shape[0]
        centroids = np.empty((self.n_clusters, sorted_segments.shape[1]), dtype=np.float64)
        centroids[0] = sorted_segments[int(rng.integers(0, n_samples))]

        closest = distances_to_centroid(sorted_segments, centroids[0], self.p) ** 2
        for index in range(1, self.n_clusters):
            centroids[index] = sorted_segments[
                int(rng.choice(n_samples, p=closest / closest.sum()))
            ]
            closest = np.minimum(
                closest, distances_to_centroid(sorted_segments, centroids[index], self.p) ** 2
            )
        return centroids

    def _lloyd(self, sorted_segments: FloatArray, centroids: FloatArray) -> ClusteringResult:
        centroids = centroids.copy()
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            distances = self._distance_matrix(sorted_segments, centroids)
            labels = distances.argmin(axis=1)
            previous = centroids.copy()

            for index in range(self.n_clusters):
                members = sorted_segments[labels == index]
                if members.shape[0] == 0:
                    centroids[index] = sorted_segments[int(distances.min(axis=1).argmax())]
                else:
                    centroids[index] = wasserstein_barycenter_1d(members, p=self.p)

            movement = sum(
                wasserstein_distance_1d(previous[index], centroids[index], p=self.p)
                for index in range(self.n_clusters)
            )
            if movement < self.tol:
                converged = True
                break

        distances = self._distance_matrix(sorted_segments, centroids)
        return ClusteringResult(
            labels=distances.argmin(axis=1).astype(np.int64),
            centroids=centroids,
            inertia=float(distances.min(axis=1).sum()),
            n_iter=iteration,
            converged=converged,
        )


# %% [markdown]
# ## What a two-moment summary of the same windows can do
#
# The comparison the section is built around replaces the distribution with a short list of
# moments and clusters those instead. Each window becomes its first `n_moments` raw moments,
# scaled by the reciprocal factorial so the list is a truncated series expansion rather than
# a set of numbers on incomparable scales, and the moments are standardised because k-means
# in Euclidean space is not scale-free.
#
# This is the baseline the distributional method has to beat, and it is not a straw man: the
# first two moments are a mean and a variance, which is what most regime models distinguish
# states by, and `N_MOMENTS` of them carry skewness and kurtosis as well.
#
# Two models are fitted on those features rather than one, because the comparison would
# otherwise confound the features with the algorithm. k-means in that space assigns a window
# to the nearest centre, which draws spherical clusters of equal size; a Gaussian mixture
# fits a covariance per component and can draw elongated ones. Whatever separates the results
# of those two is the geometry, since the features they read are identical.


# %%
def moment_features(segments: FloatArray, n_moments: int) -> FloatArray:
    """The first `n_moments` raw moments of each window, scaled by the reciprocal factorial."""
    return np.column_stack(
        [
            (segments**order).mean(axis=1) / math.factorial(order)
            for order in range(1, n_moments + 1)
        ]
    )


@dataclass(frozen=True)
class MomentClustering:
    """A fit on moment features, kept together with what it needs to assign more windows."""

    labels: IntArray
    scaler: StandardScaler
    model: KMeans | GaussianMixture

    def predict(self, segments: FloatArray, n_moments: int) -> IntArray:
        """Assign windows the fit never saw, standardising them the way the fit was."""
        features = self.scaler.transform(moment_features(segments, n_moments))
        return self.model.predict(features).astype(np.int64)


def fit_moment_kmeans(
    segments: FloatArray, n_clusters: int, n_moments: int, random_state: int | None
) -> MomentClustering:
    """k-means on standardised moment features."""
    scaler = StandardScaler()
    features = scaler.fit_transform(moment_features(segments, n_moments))
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    return MomentClustering(
        labels=model.fit_predict(features).astype(np.int64), scaler=scaler, model=model
    )


def fit_moment_mixture(
    segments: FloatArray, n_components: int, n_moments: int, random_state: int | None
) -> MomentClustering:
    """A Gaussian mixture on the same standardised moment features."""
    scaler = StandardScaler()
    features = scaler.fit_transform(moment_features(segments, n_moments))
    model = GaussianMixture(n_components=n_components, random_state=random_state, reg_covar=1e-6)
    model.fit(features)
    return MomentClustering(
        labels=model.predict(features).astype(np.int64), scaler=scaler, model=model
    )


# %% [markdown]
# ## Reordering the clusters so the labels mean something
#
# k-means returns cluster numbers in whatever order the initialisation produced them, and
# nothing ties cluster zero of one fit to cluster zero of another. Two fits are compared
# below and their labels go into a feature, so both need a rule that fixes the numbering from
# the data. The rule here is the same one `11_hmm_regimes` uses: order the clusters by the
# dispersion of the windows assigned to them, so the higher-numbered cluster is always the
# more volatile one.


# %%
def order_by_dispersion(segments: FloatArray, labels: IntArray, n_clusters: int) -> IntArray:
    """A relabelling that puts the clusters in increasing order of their members' spread."""
    spreads = [
        segments[labels == index].std() if np.any(labels == index) else np.inf
        for index in range(n_clusters)
    ]
    order = np.argsort(spreads)
    mapping = np.empty(n_clusters, dtype=np.int64)
    mapping[order] = np.arange(n_clusters, dtype=np.int64)
    return mapping


# %% [markdown]
# ## A second opinion that does not depend on the distance used to cluster
#
# Inertia cannot compare the two methods, because each reports it in its own geometry. The
# **maximum mean discrepancy** gives a number that neither method optimises: it embeds two
# samples through a kernel and measures the distance between their mean embeddings, so a
# small value says the two samples look like draws from one distribution.
#
# It is computed here three times per method, between the members of each cluster and between
# the two clusters, over bootstrap resamples because the estimator is biased upward at small
# sample sizes and the median across resamples is more stable than one value.
#
# The kernel needs a width, and the answer is not a fixed number: a width far from the scale
# of the data drives every kernel value to one or to zero and the discrepancy to nothing. The
# median heuristic sets it from the data, at the median distance between pairs of windows,
# and the value it picks is printed so it is not a hidden choice.


# %%
def gaussian_kernel(x: FloatArray, y: FloatArray, sigma: float) -> FloatArray:
    """The Gaussian kernel matrix between two sets of vectors."""
    squared = np.sum(x * x, axis=1, keepdims=True) + np.sum(y * y, axis=1) - 2.0 * (x @ y.T)
    return np.exp(-np.maximum(squared, 0.0) / (2.0 * sigma * sigma))


def median_kernel_width(sample: FloatArray, rng: np.random.Generator, n_pairs: int = 2000) -> float:
    """The median heuristic: half the median squared distance between random pairs, rooted."""
    left = rng.integers(0, sample.shape[0], size=n_pairs)
    right = rng.integers(0, sample.shape[0], size=n_pairs)
    squared = np.sum((sample[left] - sample[right]) ** 2, axis=1)
    return float(np.sqrt(np.median(squared[squared > 0.0]) / 2.0))


def maximum_mean_discrepancy(x: FloatArray, y: FloatArray, sigma: float) -> float:
    """The biased estimator of the maximum mean discrepancy under a Gaussian kernel."""
    squared = (
        gaussian_kernel(x, x, sigma).mean()
        - 2.0 * gaussian_kernel(x, y, sigma).mean()
        + gaussian_kernel(y, y, sigma).mean()
    )
    return math.sqrt(max(float(squared), 0.0))


def bootstrap_discrepancy(
    first: FloatArray,
    second: FloatArray | None,
    sigma: float,
    n_bootstrap: int,
    sample_size: int,
    random_state: int | None,
) -> float:
    """The median discrepancy over resamples; with `second` unset, both draws come from `first`."""
    rng = np.random.default_rng(random_state)
    other = first if second is None else second
    values = [
        maximum_mean_discrepancy(
            first[rng.choice(first.shape[0], size=sample_size, replace=True)],
            other[rng.choice(other.shape[0], size=sample_size, replace=True)],
            sigma,
        )
        for _ in range(n_bootstrap)
    ]
    return float(np.median(values))


# %% [markdown]
# ## A series whose regimes are known
#
# The benchmark needs a ground truth, so the first data is simulated. Two sets of parameters
# alternate at fixed points: one with a small positive drift and low volatility, one with a
# small negative drift and volatility more than twice as high. Each block is a geometric
# Brownian motion, which means the returns inside a block are independent draws from one
# normal distribution and the only thing that changes at a switch is which normal.
#
# That makes the experiment favourable to any method that reads the variance, and the point
# of running it is not to show that the distributional method works. It is to see how much of
# the truth a two-moment summary recovers when the truth is entirely contained in two
# moments.


# %%
def simulate_gbm_log_returns(
    n_steps: int, mu: float, sigma: float, random_state: int
) -> FloatArray:
    """Log returns of a geometric Brownian motion over `n_steps` unit intervals."""
    rng = np.random.default_rng(random_state)
    return (mu - 0.5 * sigma * sigma) + sigma * rng.standard_normal(n_steps)


def simulate_two_regime_stream(
    n_steps: int,
    calm: Mapping[str, float],
    stressed: Mapping[str, float],
    switch_points: Sequence[int],
    random_state: int,
) -> tuple[FloatArray, IntArray]:
    """A return series that alternates between two parameter sets at the given indices."""
    rng = np.random.default_rng(random_state)
    cuts = sorted({int(point) for point in switch_points if 0 < point < n_steps})
    boundaries = [0, *cuts, n_steps]
    returns = np.empty(n_steps, dtype=np.float64)
    regimes = np.empty(n_steps, dtype=np.int64)

    for block, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        regime = block % 2
        parameters = calm if regime == 0 else stressed
        returns[start:end] = simulate_gbm_log_returns(
            end - start,
            mu=parameters["mu"],
            sigma=parameters["sigma"],
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        regimes[start:end] = regime

    return returns, regimes


# %%
CALM = {"mu": 0.0005, "sigma": 0.01}
STRESSED = {"mu": -0.0003, "sigma": 0.025}

switch_points = [int(N_STEPS * (index + 1) / (N_SWITCHES + 1)) for index in range(N_SWITCHES)]
simulated_returns, true_regime = simulate_two_regime_stream(
    N_STEPS, CALM, STRESSED, switch_points, random_state=SEED
)

simulated = lift_stream(simulated_returns, window_len=WINDOW_LEN, overlap=OVERLAP)
window_truth = np.array(
    [
        int(true_regime[start : start + WINDOW_LEN].mean() > 0.5)
        for start in simulated.starts.tolist()
    ],
    dtype=np.int64,
)

print(f"Simulated sessions: {N_STEPS:,}, switches: {len(switch_points)}")
print(f"Sessions in the calm regime: {int((true_regime == 0).sum()):,}")
print(f"Windows: {simulated.segments.shape[0]}, each {WINDOW_LEN} sessions, step {simulated.step}")
print(f"Windows whose majority regime is the stressed one: {int(window_truth.sum())}")

# %% [markdown]
# A window that straddles a switch contains returns from both regimes, so its ground truth is
# whichever regime supplied more of its sessions. With a window of `WINDOW_LEN` sessions and
# six switches, a handful of windows are mixtures and no method can label them correctly by
# any definition. That sets a ceiling below one on every score in the table.

# %%
simulated_wasserstein = WassersteinKMeans1D(
    n_clusters=N_CLUSTERS,
    p=WASSERSTEIN_P,
    n_init=N_INIT,
    max_iter=MAX_ITER,
    random_state=SEED,
).fit(simulated.sorted_segments)

simulated_moments = fit_moment_kmeans(
    simulated.segments, n_clusters=N_CLUSTERS, n_moments=N_MOMENTS, random_state=SEED
)
simulated_mixture = fit_moment_mixture(
    simulated.segments, n_components=N_CLUSTERS, n_moments=N_MOMENTS, random_state=SEED
)


def relabelled(labels: IntArray) -> IntArray:
    """The same assignment, renumbered so the more volatile cluster is the higher number."""
    return order_by_dispersion(simulated.segments, labels, N_CLUSTERS)[labels]


methods = {
    "Wasserstein k-means": relabelled(simulated_wasserstein.labels),
    "k-means on moments": relabelled(simulated_moments.labels),
    "Gaussian mixture on moments": relabelled(simulated_mixture.labels),
}

print(f"Wasserstein k-means converged: {simulated_wasserstein.converged}")
print(f"Iterations of the best of {N_INIT} initialisations: {simulated_wasserstein.n_iter}")
for name, labels in methods.items():
    print(f"Windows per cluster, {name}: {np.bincount(labels).tolist()}")

# %% [markdown]
# The **adjusted Rand index** scores an assignment against the truth without caring which
# cluster got which number: it counts pairs of windows the two agree to put together or
# apart, and subtracts what agreement a random assignment of the same cluster sizes would
# reach. One is a perfect match and zero is chance.
#
# The table reports it alongside the discrepancy within each cluster and between the two.
# A silhouette score is not reported: it would be computed in a chosen space, and each of
# these methods clusters in a different one, so whichever space is picked flatters the method
# that optimises in it.

# %%
sigma = median_kernel_width(simulated.sorted_segments, np.random.default_rng(SEED))
print(f"Kernel width from the median heuristic: {sigma:.4f}")

comparison_rows = []
for name, labels in methods.items():
    members = [simulated.sorted_segments[labels == index] for index in range(N_CLUSTERS)]
    comparison_rows.append(
        {
            "method": name,
            "adjusted Rand index against the truth": adjusted_rand_score(window_truth, labels),
            "discrepancy within the calmer cluster": bootstrap_discrepancy(
                members[0], None, sigma, MMD_BOOTSTRAPS, MMD_SAMPLE_SIZE, SEED
            ),
            "discrepancy within the more volatile cluster": bootstrap_discrepancy(
                members[1], None, sigma, MMD_BOOTSTRAPS, MMD_SAMPLE_SIZE, SEED
            ),
            "discrepancy between the clusters": bootstrap_discrepancy(
                members[0], members[1], sigma, MMD_BOOTSTRAPS, MMD_SAMPLE_SIZE, SEED
            ),
        }
    )

display(pd.DataFrame(comparison_rows).set_index("method"))

# %% [markdown]
# Read the three rows against each other rather than the first against the truth. The
# simulation differs only in a drift and a volatility, so the moment features carry every
# quantity that matters; the two fits on those features nonetheless land far apart, and the
# mixture lands close to the Wasserstein result. What separates the two moment fits is not
# the information available to them but the shape of cluster each can draw, so most of the
# gap between the first row and the second is the equal-size spherical geometry that k-means
# imposes on standardised moments and not a limitation of moments as such.
#
# The discrepancies say something the Rand index does not. Every method produces clusters
# whose members resemble each other far more than they resemble the other cluster's, so all
# three found a real division. Only the comparison against the truth says which division.
# That is the general shape of the problem: an unsupervised method can always report a clean
# separation, and on real data there is no column to check it against.

# %% [markdown]
# ## Which session carries a window's label
#
# A window covering sessions $t$ to $t + h - 1$ is only complete at the close of session
# $t + h - 1$, so that is the first session whose feature can carry the window's label. The
# obvious alternative, writing the label back across every session the window covers, would
# put a label on session $t$ that was computed from returns up to $t + h - 1$, and any model
# reading it would be reading its own future.
#
# The function below therefore stamps each label on the last session of its window and holds
# it until the next window closes. Two consequences follow and both are real. The sessions
# before the first window closes have no label at all. And a regime change is visible only
# once enough of the new regime has entered a window to move it across the boundary, which is
# a delay of up to a window plus a step.


# %%
def label_at_window_close(
    starts: IntArray, window_len: int, labels: IntArray, n_sessions: int
) -> FloatArray:
    """Stamp each window's label on the session it closes on, and hold it until the next."""
    closes = starts + window_len - 1
    inside = closes < n_sessions
    stamped = pd.Series(np.nan, index=np.arange(n_sessions))
    stamped.iloc[closes[inside]] = labels[inside]
    return stamped.ffill().to_numpy()


# %%
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["grid_2x2"])
sessions = np.arange(N_STEPS)

ax = axes[0, 0]
for regime, color, name in ((0, COLORS["blue"], "calm"), (1, COLORS["copper"], "stressed")):
    mask = true_regime == regime
    ax.scatter(sessions[mask], simulated_returns[mask], s=1, alpha=0.5, color=color, label=name)
ax.axhline(0, color=COLORS["recede"], linestyle="--", linewidth=0.6)
ax.set_xlabel("Session")
ax.set_ylabel("Log return")
ax.set_title("The simulated series, coloured by the regime that drew it", fontsize=9)
ax.legend(fontsize=7, markerscale=4)

ax = axes[0, 1]
window_means = simulated.segments.mean(axis=1)
window_spreads = simulated.segments.std(axis=1)
for regime, color in ((0, COLORS["blue"]), (1, COLORS["copper"])):
    mask = methods["Wasserstein k-means"] == regime
    ax.scatter(window_spreads[mask], window_means[mask], s=10, alpha=0.7, color=color)
ax.set_xlabel("Standard deviation within the window")
ax.set_ylabel("Mean within the window")
ax.set_title("The clusters divide the windows along the spread", fontsize=9)

ax = axes[1, 0]
inferred = label_at_window_close(
    simulated.starts, WINDOW_LEN, methods["Wasserstein k-means"], N_STEPS
)
ax.step(sessions, true_regime, where="post", linewidth=1, color=COLORS["neutral"], label="drawn")
ax.step(sessions, inferred, where="post", linewidth=1.2, color=COLORS["blue"], label="assigned")
ax.set_yticks([0, 1])
ax.set_yticklabels(["calm", "stressed"])
ax.set_xlabel("Session")
ax.set_title("The assignment follows the switch a window late", fontsize=9)
ax.legend(fontsize=7)

ax = axes[1, 1]
quantiles = np.linspace(0, 1, WINDOW_LEN)
ordered = simulated_wasserstein.centroids[
    np.argsort(order_by_dispersion(simulated.segments, simulated_wasserstein.labels, N_CLUSTERS))
]
for position, color, name in (
    (0, COLORS["blue"], "calmer"),
    (1, COLORS["copper"], "more volatile"),
):
    ax.plot(quantiles, ordered[position], linewidth=1.6, color=color, label=name)
ax.axhline(0, color=COLORS["recede"], linestyle="--", linewidth=0.6)
ax.set_xlabel("Quantile")
ax.set_ylabel("Log return")
ax.set_title("The two barycenters differ at every quantile", fontsize=9)
ax.legend(fontsize=7)

fig.suptitle("What the clustering recovers from a series whose regimes are known")
show_with_alt(
    fig,
    "Four panels on simulated data. The top left scatters returns against session with the "
    "two regimes in different colours, the stressed blocks visibly wider. The top right plots "
    "each window's mean against its standard deviation, coloured by cluster, and the split "
    "runs vertically along the standard deviation with the means overlapping. The bottom left "
    "steps the drawn regime and the assigned one against session; they agree except for a "
    "short lag after each switch. The bottom right draws the two cluster barycenters as "
    "quantile functions, one flatter and one steeper, separated across the whole range.",
)

# %% [markdown]
# The top right panel is the reason the method is worth its cost, read in reverse. The split
# runs along the spread and the means overlap, which says the clustering has divided the
# windows on their volatility and taken no view on their drift. On this simulation that is the
# correct division and a two-moment summary could have found it. The bottom right panel shows
# the same division as the object the algorithm actually manipulates: two quantile functions,
# separated across the whole range rather than at one summary number.

# %% [markdown]
# ## The same construction on the index, fitted forward
#
# Everything above is fitted on the whole sample, which is what a benchmark against a known
# truth needs and what a feature must not be. The real-data section splits the windows in
# time: the centroids are fitted on the first `TRAIN_FRACTION` of them and every window is
# then assigned to the nearest of those fixed centroids. A window after the split is scored
# against centroids that no session inside it contributed to.
#
# The split is a single one rather than a rolling refit, and that is a simplification worth
# naming. A rolling refit would face the problem `11_hmm_regimes` describes, that cluster
# numbers move between fits and have to be tied down by a characteristic of the members; the
# ordering rule above is what would do it.

# %%
index_prices = (
    load_sp500_index()
    .select(["timestamp", "close"])
    .filter(pl.col("timestamp") >= pl.lit(START_DATE).str.to_date())
    .filter(pl.col("timestamp") <= pl.lit(END_DATE).str.to_date())
    .sort("timestamp")
)

index_returns = np.log(index_prices["close"].to_numpy()[1:] / index_prices["close"].to_numpy()[:-1])
index_dates = pd.DatetimeIndex(index_prices["timestamp"].to_list()[1:])

index_windows = lift_stream(index_returns, window_len=WINDOW_LEN, overlap=OVERLAP)
n_train_windows = int(index_windows.segments.shape[0] * TRAIN_FRACTION)
train_windows = index_windows.sorted_segments[:n_train_windows]

print(
    f"S&P 500 index: {len(index_returns):,} daily returns, {index_dates[0].date()} to {index_dates[-1].date()}"
)
print(f"Windows: {index_windows.segments.shape[0]}, of which fitted on: {n_train_windows}")
print(
    f"First session the centroids may be used on: {index_dates[n_train_windows * index_windows.step + WINDOW_LEN - 1].date()}"
)

# %%
estimator = WassersteinKMeans1D(
    n_clusters=N_CLUSTERS, p=WASSERSTEIN_P, n_init=N_INIT, max_iter=MAX_ITER, random_state=SEED
)
index_fit = estimator.fit(train_windows)
index_order = order_by_dispersion(
    index_windows.segments[:n_train_windows], index_fit.labels, N_CLUSTERS
)
index_centroids = index_fit.centroids[np.argsort(index_order)]
index_labels = estimator.predict(index_windows.sorted_segments, index_centroids)

STRESSED_CLUSTER = N_CLUSTERS - 1

print(f"Converged: {index_fit.converged} in {index_fit.n_iter} iterations")
print(f"Windows per cluster over the whole sample: {np.bincount(index_labels).tolist()}")
print(
    f"Windows per cluster over the fitted block: {np.bincount(index_labels[:n_train_windows]).tolist()}"
)
print(f"Windows per cluster after it: {np.bincount(index_labels[n_train_windows:]).tolist()}")

# %% [markdown]
# The share of windows in the more volatile cluster differs between the two blocks, and that
# is information rather than a defect: the centroids are held fixed, so the share after the
# split says how often the later sample looked like the earlier sample's stressed windows.

# %% [markdown]
# ## What the two clusters were, as a position
#
# The label is known at the close of the session it is stamped on, so the return it can be
# earned against is the next session's. The table below therefore holds each regime's label
# for one session and reads the following return, over the block after the split only, which
# makes it a comparison of two rules rather than a description of a partition. It charges no
# costs and no slippage, so it is an upper bound on what either side is worth.
#
# The drawdown is of an equity curve that holds the index while the label says one thing and
# nothing while it says the other, so the sessions outside the regime contribute a flat
# stretch rather than being spliced out. Compounding only the sessions inside a regime, which
# is the easier thing to write, produces a curve that skips the gaps and a drawdown belonging
# to no position anyone could hold.

# %%
SESSIONS_PER_YEAR = 252

index_label_series = pd.Series(
    label_at_window_close(index_windows.starts, WINDOW_LEN, index_labels, len(index_returns)),
    index=index_dates,
)
held = index_label_series.shift(1)
test_start = index_dates[n_train_windows * index_windows.step + WINDOW_LEN - 1]
evaluated = index_dates > test_start
returns_series = pd.Series(index_returns, index=index_dates)

regime_rows = []
for cluster, name in ((0, "calmer"), (STRESSED_CLUSTER, "more volatile")):
    inside = evaluated & (held == cluster).to_numpy()
    while_inside = returns_series[inside]
    invested = np.where(inside, index_returns, 0.0)[evaluated]
    curve = np.exp(np.cumsum(invested))
    regime_rows.append(
        {
            "regime": name,
            "sessions": int(inside.sum()),
            "share of the evaluated block": inside.sum() / int(evaluated.sum()),
            "annualised mean return": SESSIONS_PER_YEAR * while_inside.mean(),
            "annualised volatility": np.sqrt(SESSIONS_PER_YEAR) * while_inside.std(),
            "worst single session": while_inside.min(),
            "deepest drawdown of holding only here": float(
                (curve / np.maximum.accumulate(curve) - 1.0).min()
            ),
        }
    )

display(pd.DataFrame(regime_rows).set_index("regime"))

# %% [markdown]
# The two rows differ in volatility by construction, since that is what the clusters were
# ordered by, so the volatility column is a check that the ordering did what it claims rather
# than a result. The mean return is the result, and it is the asymmetry the chapter's
# downstream notebooks use: the two regimes are not two draws from one distribution with
# different spreads.
#
# Read the drawdown column against the volatility one rather than on its own. The two regimes
# hold the index for different numbers of sessions, and a curve with fewer sessions has fewer
# chances to fall, so the deeper drawdown falls to whichever regime combines wide sessions
# with enough of them.

# %% [markdown]
# ## The distance between the two clusters' returns
#
# The clustering worked on windows. A separate question is how far apart the two clusters'
# individual returns are as distributions, and `ml4t-diagnostic` answers it with two measures
# built for monitoring a model in production. `compute_wasserstein_distance` calibrates its
# threshold by permutation, so the distance is compared against a null of one distribution
# rather than against a fixed number. `compute_psi` bins both samples and sums a symmetric
# relative difference per bin, and its thresholds are conventions rather than tests.

# %%
calmer_returns = returns_series[evaluated & (held == 0).to_numpy()].to_numpy()
stressed_returns = returns_series[evaluated & (held == STRESSED_CLUSTER).to_numpy()].to_numpy()

distance = compute_wasserstein_distance(calmer_returns, stressed_returns, random_state=SEED)
stability = compute_psi(calmer_returns, stressed_returns)

print(f"Wasserstein distance between the two clusters' returns: {distance.distance:.6f}")
print(f"Permutation threshold at the default level: {distance.threshold:.6f}")
print(
    f"Population stability index: {stability.psi:.4f}, which the library calls {stability.alert_level}"
)

# %% [markdown]
# Both numbers are large, and neither is a surprise: the two samples were separated by a
# procedure that reads their distributions. What the permutation threshold adds is a scale,
# since a distance is otherwise uninterpretable in the units of a daily log return.

# %%
fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)
stressed_mask = (held == STRESSED_CLUSTER).to_numpy() & evaluated

ax = axes[0]
cumulative = np.exp(np.cumsum(index_returns))
ax.semilogy(index_dates, cumulative, linewidth=0.6, color=COLORS["blue"])
ax.fill_between(
    index_dates,
    cumulative.min(),
    cumulative.max(),
    where=stressed_mask,
    alpha=0.2,
    color=COLORS["copper"],
)
ax.axvline(test_start, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
ax.set_ylabel("Index, log scale")
ax.set_title("The shaded windows fall on the declines, not around them", fontsize=9)
ax.legend(
    handles=[
        Patch(facecolor=COLORS["copper"], alpha=0.3, label="assigned to the volatile cluster")
    ],
    fontsize=7,
    loc="upper left",
)

ax = axes[1]
rolling = pd.Series(index_returns, index=index_dates).rolling(WINDOW_LEN).std() * np.sqrt(
    SESSIONS_PER_YEAR
)
ax.plot(index_dates, rolling, linewidth=0.6, color=COLORS["blue"])
ax.fill_between(
    index_dates, 0, float(rolling.max()), where=stressed_mask, alpha=0.2, color=COLORS["copper"]
)
ax.axvline(test_start, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
ax.set_ylabel("Annualised")
ax.set_title(f"Realised volatility over the same {WINDOW_LEN} sessions", fontsize=9)

ax = axes[2]
ax.plot(index_dates, index_label_series, linewidth=0.6, color=COLORS["neutral"])
ax.axvline(test_start, color=COLORS["neutral"], linestyle="--", linewidth=0.8)
ax.set_yticks([0, 1])
ax.set_yticklabels(["calmer", "volatile"])
ax.set_xlabel("Session")
ax.set_title("The label itself, held between window closes", fontsize=9)

fig.suptitle("Centroids fitted before the dashed line, assigned forward after it")
show_with_alt(
    fig,
    "Three stacked panels over the index history with a dashed vertical line at the end of "
    "the fitted block. The top plots the cumulative index on a log scale with shaded bands "
    "where the label is the volatile cluster; the bands sit on the sharp declines. The middle "
    "plots rolling realised volatility with the same shading, and the shaded stretches line "
    "up with the peaks. The bottom draws the label as a two-level line, flat for long "
    "stretches and switching in short bursts.",
)

# %% [markdown]
# The middle panel is the one to be suspicious of. The label and a rolling standard deviation
# of the same window agree closely, which raises the question of what the clustering bought
# over a threshold on that standard deviation. On this series and with two clusters, not much:
# the two disagree on the sessions near the boundary and nowhere else, and the disagreement is
# what the next notebook has to earn its keep on.
#
# What the clustering does have is a construction that extends. Three clusters, a longer
# window, or a distribution that differs in skewness rather than spread are all the same code
# with a different argument, and none of them has a threshold to choose.

# %% [markdown]
# ## The three columns
#
# The chapter's feature catalog asks for three, and each one is stamped at a window's close
# and held, so every value is a function of returns up to and including the session it sits
# on.
#
# - `wasserstein_cluster` is the assignment, ordered so the higher number is the more
#   volatile cluster.
# - `cluster_distance` is the window's distance to the centroid it was assigned to. A window
#   is assigned to its nearest centroid whether or not it resembles it, so this column is what
#   says whether the assignment means anything: a large value is an environment the fitted
#   block did not contain.
# - `tail_divergence` is one on the sessions where the Wasserstein assignment and a
#   two-moment assignment of the same window disagree. What the two read differs by
#   everything a mean and a variance omit, so a disagreement locates a window whose shape and
#   whose first two moments point different ways. Which part of the shape did it is not
#   something this column records.

# %%
N_MOMENT_BASELINE = 2

moment_fit = fit_moment_kmeans(
    index_windows.segments[:n_train_windows],
    n_clusters=N_CLUSTERS,
    n_moments=N_MOMENT_BASELINE,
    random_state=SEED,
)
moment_order = order_by_dispersion(
    index_windows.segments[:n_train_windows], moment_fit.labels, N_CLUSTERS
)
moment_labels_index = moment_order[
    moment_fit.predict(index_windows.segments, n_moments=N_MOMENT_BASELINE)
]

window_distances = np.array(
    [
        wasserstein_distance_1d(window, index_centroids[label], p=WASSERSTEIN_P)
        for window, label in zip(index_windows.sorted_segments, index_labels.tolist(), strict=True)
    ]
)
disagreement = (index_labels != moment_labels_index).astype(np.int64)

features = (
    pl.DataFrame(
        {
            "timestamp": index_dates,
            "wasserstein_cluster": label_at_window_close(
                index_windows.starts, WINDOW_LEN, index_labels, len(index_returns)
            ),
            "cluster_distance": label_at_window_close(
                index_windows.starts, WINDOW_LEN, window_distances, len(index_returns)
            ),
            "tail_divergence": label_at_window_close(
                index_windows.starts, WINDOW_LEN, disagreement, len(index_returns)
            ),
        }
    )
    .with_columns(pl.exclude("timestamp").fill_nan(None))
    .drop_nulls()
)

print(f"Feature rows: {features.height:,} of {len(index_returns):,} sessions")
print(
    f"Windows where the two assignments disagree: {int(disagreement.sum())} of {disagreement.size}"
)
display(features.describe())
display(features.tail(3))

# %% [markdown]
# The row count is short of the session count by the sessions before the first window closed,
# and those rows are dropped rather than filled. A distributional feature has a warm-up and
# saying so in the row count is cheaper than explaining a filled value later.
#
# The disagreement rate is the number to carry into `13_regime_as_feature`. A rate near zero
# would say the column is constant and worthless; a rate near half would say the two methods
# have nothing to do with each other and one of them is wrong. The rate here is between them,
# which is the only case in which the column can carry anything.

# %% [markdown]
# ## Takeaways
#
# 1. **The stream lift turns one series into many samples, and sorting is the whole
#    computation.** The one-dimensional Wasserstein distance between two equal-sized samples
#    is an average over matched quantiles, and its barycenter is the quantile-wise median or
#    mean, so k-means needs no optimiser it did not already have.
# 2. **A method that reads the whole distribution beat one that read four moments of it, and
#    the reason was the geometry.** A Gaussian mixture on the same four moments came close to
#    the distributional result while k-means on them did not, which puts most of the gap in
#    the cluster shapes k-means can draw rather than in what moments omit.
# 3. **A window's label belongs at the window's close.** Writing it back over the sessions the
#    window covers is the natural thing to code and it puts a session's own future into its
#    feature. The cost of doing it correctly is a warm-up at the start and a lag of up to a
#    window plus a step after every switch.
# 4. **Centroids fitted on the whole sample are not a feature either.** Fit them on a first
#    block and assign forward; the share of later windows landing in each cluster then means
#    something, because the clusters were not drawn to accommodate them.
# 5. **An unsupervised method always reports a clean separation.** Every method here produced
#    clusters far apart in maximum mean discrepancy, including the one that recovered least of
#    the truth. Without a column to check against, a separation statistic says the algorithm
#    ran and nothing about whether the division is the one that matters.
#
# **Reference**: Horvath, Issa and Muguruza (2021), "Clustering Market Regimes Using the
# Wasserstein Distance".
#
# **Previous**: `11_hmm_regimes` fits a model of how regimes switch instead of clustering
# windows. **Next**: `13_regime_as_feature` takes regime columns into a downstream model.
