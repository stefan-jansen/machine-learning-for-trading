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
# # Instrumented PCA (IPCA) in the Two-Step Framework
#
# **Docker image**: `ml4t`
#
# **Chapter 14: Latent Factors**
#
# This notebook implements Instrumented PCA from Kelly, Pruitt, and Su (2019)
# "Characteristics are Covariances: A Unified Model of Risk and Return", and
# slots the implementation into the chapter's two-step factor-forecasting
# framework. IPCA is the first estimator in the chapter where the framework
# becomes operational — the loadings are characteristics-driven and the factor
# series can be forecast forward via any model in our Stage 2 catalog.
#
# ## The Two-Step Framework (recap)
#
# Asset return forecasting decomposes into three stages (see **Figure 14.9**):
#
# 1. **Stage 1 — Factor Model.** Compresses an $(T \times N)$ returns panel
#    to a $(T \times K)$ factor history $F$. IPCA realises this stage as
#    $r_{i,t+1} = \beta_{i,t}' f_{t+1} + \varepsilon$ with
#    $\beta_{i,t} = z_{i,t} \Gamma$.
# 2. **Stage 2 — Factor Premium Forecaster.** Predicts $\hat\lambda_{t+1}$
#    from the training factor history $F_{1:t}$. The simplest choice is the
#    sample mean (KPS 2019's implicit forecaster); we also demonstrate AR(1)
#    and EWMA (see **Figure 14.10** for the full Tier-1/2/3 catalog).
# 3. **Stage 3 — Asset Mapper.** Translates the factor forecast back to a
#    per-asset signal: $\hat\mu_{i,t+1} = \beta_{i,t} \cdot \hat\lambda_{t+1}$.
#
# ## Why IPCA?
#
# | PCA | IPCA |
# |-----|------|
# | Static loadings $B$ | Time-varying loadings $\beta_{i,t} = z_{i,t}\,\Gamma$ |
# | Loadings unexplained | Loadings tied to observable characteristics |
# | Stage-3 mapper is constant per asset | Stage-3 mapper updates with $z_{i,t}$ |
#
# IPCA preserves the same Stage 2 / Stage 3 plumbing as PCA — the only
# difference is *how* Stage 1 produces $F$ and how Stage 3 builds $\beta_{i,t}$.
# That is the unifying claim of the framework: changing the factor model
# changes one component, not the pipeline.
#
# ## Learning Objectives
#
# - LO1: Understand characteristic-driven factor loadings as Stage 1 output
# - LO2: Implement alternating least squares (ALS) estimation of $\Gamma$
# - LO3: Swap the Stage 2 forecaster (constant / AR(1) / EWMA) and observe
#   how the asset-prediction signal changes
# - LO4: Recognise the null result on i.i.d. synthetic factors — when factor
#   premia are unpredictable, no Stage 2 forecaster can help
#
# **Why synthetic data?** This notebook generates data with a *known* $\Gamma$
# matrix to verify that ALS estimation recovers the true characteristic-to-factor
# mapping. This parameter-recovery validation is impossible with real data
# where $\Gamma$ is unobservable. For real-data application with walk-forward
# IC evaluation, see [`08_latent_factors`](../case_studies/us_firm_characteristics/08_latent_factors.ipynb).
#
# **Prerequisites**: Complete [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb).
# **Next**: See [`05_rp_pca`](05_rp_pca.ipynb) for risk-premium-aware factor
# extraction (same framework, different Stage 1 objective).

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""IPCA in the two-step framework — characteristic-driven Stage 1, swappable Stage 2 forecasters."""

import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import FancyBboxPatch
from ml4t.diagnostic.metrics import cross_sectional_ic, cross_sectional_ic_series

from utils.predictions_cache import predictions_cache_key, save_predictions
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")


# %% [markdown]
# ### Universe-Size Scaling
#
# IPCA fit time scales as $O(\text{MAX\_ITER} \cdot N_{\text{periods}} \cdot
# N_{\text{assets}} \cdot N_{\text{chars}}^2)$ for the Γ update plus an $N_K^3$
# eigendecomposition per iteration. The defaults below (500 periods × 100 assets
# × 10 characteristics × 3 factors) finish a full fit in seconds on synthetic
# data; production case studies (5–10× more assets, 50+ characteristics) take
# minutes to tens of minutes per fold. Halve `N_ASSETS` or `MAX_ITER` for a
# faster smoke run; doubling either is roughly linear in wall-time.

# %% tags=["parameters"]
# Production defaults (Papermill overrides for CI testing)
N_PERIODS = 500
N_ASSETS = 100
N_CHARACTERISTICS = 10
N_TRUE_FACTORS = 3
N_IPCA_FACTORS = 3
MAX_ITER = 100
TRAIN_FRAC = 0.7
SEED = 42

# %%
set_global_seeds(SEED)
RANDOM_SEED = SEED
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 2. Stage 1 — IPCA Factor Extraction
#
# ### Alternating Least Squares (ALS) Estimation
#
# IPCA is estimated via ALS, which alternates between two steps:
#
# 1. **Given Γ, estimate factors**: For each period $t$, compute
#    $\beta_{i,t} = Z_{i,t} \Gamma$ and solve the cross-sectional regression
#    $r_t = B_t f_t$ for $f_t$.
#
# 2. **Given factors, update Γ**: Pool all observations across time and assets,
#    forming $r_{i,t} = (z_{i,t} \otimes f_t)' \text{vec}(\Gamma)$, and solve
#    for $\Gamma$ via OLS on the Kronecker-product design matrix.
#
# The loop converges when the mean squared error stabilizes. Stage 1's outputs
# — $\Gamma$, the factor series $F$, and the betas $\beta_{i,t} = z_{i,t}\Gamma$
# — are exactly the inputs Stages 2 and 3 will consume.

# %%
fig, ax = plt.subplots(figsize=(8.5, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")

ax.add_patch(
    FancyBboxPatch(
        (8, 22),
        32,
        14,
        boxstyle="round,pad=0.4",
        facecolor="#D6E5FF",
        edgecolor="navy",
    )
)
ax.text(
    24, 31, "Step A: given $\\Gamma$, estimate $F$", ha="center", fontweight="bold", fontsize=10
)
ax.text(24, 26, r"$f_t = (B_t^\top B_t)^{-1} B_t^\top r_t$ per period", ha="center", fontsize=9)

ax.add_patch(
    FancyBboxPatch(
        (60, 22),
        32,
        14,
        boxstyle="round,pad=0.4",
        facecolor="#FFE0B0",
        edgecolor="darkorange",
    )
)
ax.text(76, 31, "Step B: given $F$, update $\\Gamma$", ha="center", fontweight="bold", fontsize=10)
ax.text(76, 26, r"pooled vec-form OLS on $z \otimes f$", ha="center", fontsize=9)

ax.annotate("", xy=(60, 30), xytext=(40, 30), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(40, 28), xytext=(60, 28), arrowprops=dict(arrowstyle="->", lw=2))

ax.add_patch(
    FancyBboxPatch(
        (32, 4),
        36,
        10,
        boxstyle="round,pad=0.3",
        facecolor="#E0F4E0",
        edgecolor="darkgreen",
    )
)
ax.text(
    50,
    9,
    r"converged when $\max(|\Delta\Gamma|, |\Delta f|) < 10^{-6}$",
    ha="center",
    fontsize=9,
    fontweight="bold",
)
ax.annotate(
    "",
    xy=(50, 14),
    xytext=(50, 22),
    arrowprops=dict(arrowstyle="->", lw=1, color="darkgreen", alpha=0.6),
)

ax.set_title("IPCA Stage 1: alternating least squares")
fig.show()


# %% [markdown]
# ### IPCA Class
#
# The IPCA model maps characteristics to time-varying factor loadings via
# $\beta_{i,t} = Z_{i,t} \Gamma$. The class stores the estimated $\Gamma$
# matrix and latent factors. Three methods follow `__init__`: `fit` runs
# the ALS loop and applies the KPS ΘY identification described below,
# `estimate_factors` projects new periods through the trained $\Gamma$ to
# recover realised factors on the test window, and the static helper
# `_normalize_theta_y` carries out the post-fit rotation.


# %%
class IPCA:
    """
    Instrumented Principal Component Analysis.

    Kelly, Pruitt, Su (2019) "Characteristics are Covariances"

    The model assumes:
        r_i,t+1 = beta_i,t' * f_t+1 + eps_i,t+1
        beta_i,t = Z_i,t * Gamma

    Parameters
    ----------
    n_factors : int
        Number of latent factors K
    max_iter : int
        Maximum ALS iterations
    tol : float
        Convergence tolerance
    """

    def __init__(self, n_factors: int = 5, max_iter: int = 100, tol: float = 1e-6):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = None  # L x K matrix
        self.factors = None  # T x K matrix (training-period factor history)
        self.fitted = False

    def fit(self, returns: np.ndarray, characteristics: np.ndarray) -> "IPCA":
        """Fit IPCA model via Alternating Least Squares.

        Each iteration: (1) computes betas from current Gamma, solves
        cross-sectional regressions for factors f_t; (2) pools all
        observations into a Kronecker-product design matrix and solves
        for updated Gamma via OLS. Stopping is on first-order Γ + factor
        deltas; after convergence the factor history is recomputed under
        the final Γ and rotated into the KPS ΘY identification set.
        """
        T, N = returns.shape
        L = characteristics.shape[2]
        K = self.n_factors

        print(f"IPCA fitting: T={T}, N={N}, L={L}, K={K}")

        # Initialize Γ via the leading-K eigenvectors of Σ_t (Z_t' r_t)(Z_t' r_t)'.
        # Matches the KPS / library initialization; makes the ALS recovery less
        # seed-sensitive than a small Gaussian start.
        moment = np.zeros((L, L))
        for t in range(T):
            r_t = returns[t]
            valid = ~np.isnan(r_t)
            if not valid.any():
                continue
            cross_moment = characteristics[t, valid].T @ r_t[valid]
            moment += np.outer(cross_moment, cross_moment)
        init_eigvals, init_eigvecs = np.linalg.eigh(moment)
        init_order = np.argsort(init_eigvals)[::-1][:K]
        self.gamma = init_eigvecs[:, init_order]

        factors = np.zeros((T, K))
        prev_gamma = self.gamma.copy()
        prev_factors = factors.copy()

        for iteration in range(self.max_iter):
            # Step 1: Given Gamma, compute betas and estimate factors
            betas = np.einsum("tnl,lk->tnk", characteristics, self.gamma)
            for t in range(T):
                B_t = betas[t]
                r_t = returns[t]
                BtB = B_t.T @ B_t + 1e-6 * np.eye(K)
                Btr = B_t.T @ r_t
                try:
                    factors[t] = np.linalg.solve(BtB, Btr)
                except np.linalg.LinAlgError:
                    factors[t] = np.linalg.lstsq(B_t, r_t, rcond=None)[0]

            # Step 2: Given factors, update Gamma via pooled OLS
            # Accumulate normal equations to avoid materializing the full Kronecker product.
            XtX = 1e-6 * np.eye(L * K)
            Xty = np.zeros(L * K)

            for t in range(T):
                r_t = returns[t]
                valid = ~np.isnan(r_t)
                if not valid.any():
                    continue
                ZF_t = (characteristics[t, :, :, np.newaxis] * factors[t, np.newaxis, :]).reshape(
                    N, L * K
                )
                ZF_v = ZF_t[valid]
                r_v = r_t[valid]
                XtX += ZF_v.T @ ZF_v
                Xty += ZF_v.T @ r_v

            gamma_vec = np.linalg.solve(XtX, Xty)
            self.gamma = gamma_vec.reshape(L, K)

            if iteration % 10 == 0:
                predicted = np.einsum("tnk,tk->tn", betas, factors)
                loss = np.mean((returns - predicted) ** 2)
                print(f"  Iter {iteration:3d}: MSE = {loss:.6f}")

            # KPS first-order stopping: largest absolute change in Γ or f.
            gamma_delta = float(np.max(np.abs(self.gamma - prev_gamma)))
            factor_delta = float(np.max(np.abs(factors - prev_factors)))
            if max(gamma_delta, factor_delta) < self.tol:
                print(f"  Converged at iteration {iteration}")
                break
            prev_gamma = self.gamma.copy()
            prev_factors = factors.copy()

        # Final factor history under the converged Γ — fixes the half-step
        # mismatch between the stored Γ (post-update) and the loop-carried
        # factor history (pre-update) when stopping is `max_iter`-bound.
        betas = np.einsum("tnl,lk->tnk", characteristics, self.gamma)
        for t in range(T):
            B_t = betas[t]
            r_t = returns[t]
            BtB = B_t.T @ B_t + 1e-6 * np.eye(K)
            Btr = B_t.T @ r_t
            try:
                factors[t] = np.linalg.solve(BtB, Btr)
            except np.linalg.LinAlgError:
                factors[t] = np.linalg.lstsq(B_t, r_t, rcond=None)[0]

        # KPS ΘY identification: orthonormal Γ with descending factor covariance.
        self.gamma, factors = self._normalize_theta_y(self.gamma, factors)

        self.factors = factors
        self.fitted = True
        return self

    @staticmethod
    def _normalize_theta_y(
        gamma: np.ndarray, factor_history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate (Γ, f) into the KPS ΘY identification set.

        KPS (2020) appendix C.4: orthonormalize Γ so that Γ'Γ = I_K and
        rotate the factor covariance to descending eigen-order. A
        deterministic sign convention pins the ± ambiguity of each
        eigenvector to a non-negative max-magnitude entry of Γ.
        """
        chol = np.linalg.cholesky(gamma.T @ gamma)
        chol_inv = np.linalg.inv(chol)
        gamma_orthonormal = gamma @ chol_inv.T
        f_intermediate = factor_history @ chol

        f_cov = (f_intermediate.T @ f_intermediate) / factor_history.shape[0]
        eigvals, eigvecs = np.linalg.eigh(f_cov)
        order = np.argsort(eigvals)[::-1]
        rotation = eigvecs[:, order]

        gamma_final = gamma_orthonormal @ rotation
        f_final = f_intermediate @ rotation
        for k in range(gamma_final.shape[1]):
            argmax = int(np.argmax(np.abs(gamma_final[:, k])))
            if gamma_final[argmax, k] < 0:
                gamma_final[:, k] *= -1.0
                f_final[:, k] *= -1.0
        return gamma_final, f_final

    def estimate_factors(self, returns: np.ndarray, characteristics: np.ndarray) -> np.ndarray:
        """Estimate factors for new data using fitted Gamma.

        Given a trained Gamma, compute betas from new characteristics
        and solve the cross-sectional regression r_t = B_t f_t for each
        period to extract latent factors. Produces the realised factor
        series on the test window — used to evaluate Stage 2 forecasters
        against ground truth.
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        T = len(returns)
        K = self.n_factors
        betas = np.einsum("tnl,lk->tnk", characteristics, self.gamma)

        factors = np.zeros((T, K))
        for t in range(T):
            B_t = betas[t]
            r_t = returns[t]
            BtB = B_t.T @ B_t + 1e-6 * np.eye(K)
            Btr = B_t.T @ r_t
            factors[t] = np.linalg.solve(BtB, Btr)
        return factors


# %% [markdown]
# ### ΘY Identification
#
# The ALS optimum is unique only up to a $K \times K$ invertible rotation $R$:
# any pair $(\Gamma R,\; R^{-1} f_t)$ reproduces the cross-sectional
# residuals identically. KPS (2020) appendix C.4 resolves this with the
# ΘY normalization: rotate so that $\Gamma' \Gamma = I_K$ and
# $(1/T)\sum_t f_t f_t'$ is diagonal with descending entries. The
# `_normalize_theta_y` helper applies this rotation at the end of `fit`,
# so reported loadings and factor histories are comparable across runs
# and consistent with the library implementation. A deterministic sign
# convention — flip each column of $\Gamma$ so its max-magnitude entry is
# non-negative — pins the remaining $\pm$ ambiguity of the eigendecomposition.


# %% [markdown]
# ## 3. Generate Synthetic Data with Known IPCA Structure
#
# We use synthetic data with a known $\Gamma$ matrix so we can verify recovery.
# **Important caveat**: the synthetic factors are i.i.d. draws — by construction
# nothing predicts $f_{t+1}$ from $f_{1:t}$. This sets up a clean null
# experiment for Stage 2: we expect *every* forecaster to fail at OOS
# prediction here, regardless of sophistication. For real-data application
# where characteristics carry forward-looking information, see the
# `us_firm_characteristics` case study.


# %%
def generate_ipca_data(
    n_periods: int = 500,
    n_assets: int = 100,
    n_characteristics: int = 10,
    n_true_factors: int = 3,
) -> dict:
    """Generate data with IPCA structure for testing."""
    rng = np.random.default_rng(RANDOM_SEED)

    true_gamma = rng.standard_normal((n_characteristics, n_true_factors)) * 0.5

    characteristics = rng.standard_normal((n_periods, n_assets, n_characteristics))
    for t in range(1, n_periods):
        characteristics[t] = 0.9 * characteristics[t - 1] + 0.1 * rng.standard_normal(
            (n_assets, n_characteristics)
        )

    for t in range(n_periods):
        for l_idx in range(n_characteristics):
            chars_tl = characteristics[t, :, l_idx]
            characteristics[t, :, l_idx] = (chars_tl - chars_tl.mean()) / (chars_tl.std() + 1e-8)

    return _build_ipca_returns(
        characteristics, true_gamma, n_periods, n_assets, n_characteristics, rng
    )


# %% [markdown]
# Given the standardised characteristics and the true $\Gamma$, the helper
# below draws i.i.d. factor returns, computes the implied asset returns
# under the IPCA generating equation, and packages the train/test split.


# %%
def _build_ipca_returns(characteristics, true_gamma, n_periods, n_assets, n_characteristics, rng):
    """Compute returns from IPCA structure and package results."""
    n_true_factors = true_gamma.shape[1]

    # True factor returns — i.i.d. by construction (null for Stage 2 forecasters)
    true_factors = rng.standard_normal((n_periods, n_true_factors)) * 0.02

    true_betas = np.einsum("tnl,lk->tnk", characteristics, true_gamma)
    returns = np.einsum("tnk,tk->tn", true_betas, true_factors)
    returns += rng.standard_normal((n_periods, n_assets)) * 0.01

    forward_returns = np.roll(returns, -1, axis=0)
    forward_returns[-1] = np.nan

    train_size = int(n_periods * TRAIN_FRAC)

    return {
        "returns": returns,
        "characteristics": characteristics,
        "forward_returns": forward_returns,
        "true_gamma": true_gamma,
        "true_factors": true_factors,
        "train_size": train_size,
        "n_assets": n_assets,
        "n_characteristics": n_characteristics,
    }


# %%
data = generate_ipca_data(
    n_periods=N_PERIODS,
    n_assets=N_ASSETS,
    n_characteristics=N_CHARACTERISTICS,
    n_true_factors=N_TRUE_FACTORS,
)

n_chars = data["n_characteristics"]
pl.DataFrame(
    {
        "Parameter": [
            "Periods",
            "Assets",
            "Characteristics",
            "True Factors",
            "Train Size",
            "Test Size",
        ],
        "Value": [
            N_PERIODS,
            N_ASSETS,
            n_chars,
            N_TRUE_FACTORS,
            data["train_size"],
            N_PERIODS - data["train_size"],
        ],
    }
)

# %% [markdown]
# ## 4. Stage 1 Output: Train IPCA on the Training Window

# %%
train_size = data["train_size"]
train_returns = data["returns"][:train_size]
train_chars = data["characteristics"][:train_size]

test_returns = data["returns"][train_size:-1]
test_chars = data["characteristics"][train_size:-1]
test_fwd = data["forward_returns"][train_size:-1]

print("Training IPCA model (Stage 1)...")
start_time = time()

ipca = IPCA(n_factors=N_IPCA_FACTORS, max_iter=MAX_ITER)
ipca.fit(train_returns, train_chars)

print(f"\nTraining time: {time() - start_time:.1f}s")
print(f"Stage 1 outputs: Γ ({ipca.gamma.shape}), F_train ({ipca.factors.shape})")

# %% [markdown]
# ## 5. Validation — Γ Recovery
#
# Before evaluating forecasters, sanity-check that ALS recovered the true
# Γ up to rotation. On real data Γ is unobservable; here we constructed
# the data with a known Γ specifically to validate the estimator.

# %%
true_gamma = data["true_gamma"]
est_gamma = ipca.gamma

correlations = []
for k in range(ipca.n_factors):
    best_corr = 0
    for k_true in range(true_gamma.shape[1]):
        corr = np.abs(np.corrcoef(true_gamma[:, k_true], est_gamma[:, k])[0, 1])
        best_corr = max(best_corr, corr)
    correlations.append(best_corr)

pl.DataFrame(
    {
        "Estimated Factor": [f"Factor {k + 1}" for k in range(ipca.n_factors)],
        "Rotation-Aligned Correlation": [f"{c:.3f}" for c in correlations],
    }
)

# %% [markdown]
# **Interpretation**: The first two factors recover with high rotation-aligned
# correlation (0.97 and 0.85) and the third only partially (0.63), confirming
# that ALS recovers the dominant characteristic-to-factor mapping up to the
# expected rotation ambiguity, with the weakest factor least well identified.
# Stage 1 is doing its job. Whatever happens next at Stages 2 and 3 cannot
# be blamed on a broken estimator.

# %% [markdown]
# ## 6. Stage 2 — Factor Premium Forecasters
#
# Stage 2 takes the training factor history $F_{1:t}$ and emits a one-step-ahead
# factor-premium forecast $\hat\lambda_{t+1} \in \mathbb{R}^K$. We implement
# three Tier-1 forecasters (see **Figure 14.10** for the full catalog).
#
# **Constant** (the implicit choice in KPS 2019, = Engel 2025's IID-BS):
# $$\hat\lambda_{t+1} = \frac{1}{T}\sum_{s=1}^{T} f_s$$
#
# **AR(1)** (per-factor first-order autoregression):
# $$\hat\lambda_{k,t+1} = c_k + \phi_k\, f_{k,t}$$
#
# **EWMA** (exponentially weighted average, half-life $h$):
# $$\hat\lambda_{k,t+1} = (1 - \alpha)\sum_{s=1}^{T} \alpha^{T-s}\, f_{k,s},\quad
# \alpha = \exp(-\ln 2 / h)$$


# %%
class ConstantForecaster:
    """Sample mean of training factors (Engel 2025 IID-BS)."""

    def fit(self, F: np.ndarray) -> "ConstantForecaster":
        self.lam_ = F.mean(axis=0)
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return np.tile(self.lam_, (n_steps, 1))


# %% [markdown]
# **Note: AR(1) Stage 2 forecast is open-loop.** The recursive path
# below is generated from the last training factor, not updated with
# realised test factors as they arrive. This isolates parameter
# estimation noise from rolling re-estimation noise; rolling one-step
# variants are explored elsewhere in the chapter.


# %%
class AR1Forecaster:
    """Per-factor AR(1): λ̂_{k,t+1} = c_k + φ_k · f_{k,t} (open-loop iteration)."""

    def fit(self, F: np.ndarray) -> "AR1Forecaster":
        K = F.shape[1]
        self.c_ = np.zeros(K)
        self.phi_ = np.zeros(K)
        for k in range(K):
            x = F[:-1, k]
            y = F[1:, k]
            phi, c = np.polyfit(x, y, deg=1)
            self.phi_[k] = phi
            self.c_[k] = c
        self.last_ = F[-1]
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        # Iterate the AR(1) recursion forward
        out = np.zeros((n_steps, len(self.last_)))
        prev = self.last_.copy()
        for s in range(n_steps):
            prev = self.c_ + self.phi_ * prev
            out[s] = prev
        return out


# %% [markdown]
# EWMA collapses the training factor history into a single weighted
# average with half-life $h$; like Constant it emits a flat path across
# the test window, but recent factors carry more weight than distant
# ones. Default $h = 12$ periods.


# %%
class EWMAForecaster:
    """EWMA with half-life h periods (default 12)."""

    def __init__(self, half_life: float = 12.0):
        self.half_life = half_life

    def fit(self, F: np.ndarray) -> "EWMAForecaster":
        T = len(F)
        alpha = np.exp(-np.log(2) / self.half_life)
        weights = alpha ** np.arange(T - 1, -1, -1)
        weights /= weights.sum()
        self.lam_ = (weights[:, None] * F).sum(axis=0)
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return np.tile(self.lam_, (n_steps, 1))


# %% [markdown]
# ### Fit each forecaster on the training-window factor history

# %%
F_train = ipca.factors  # (T_train, K) — Stage 1 output
n_test = len(test_returns)

forecasters = {
    "Constant": ConstantForecaster().fit(F_train),
    "AR(1)": AR1Forecaster().fit(F_train),
    "EWMA(h=12)": EWMAForecaster(half_life=12.0).fit(F_train),
}

# Stage 2 predictions: one-step-ahead λ̂ for each test period
# (For Constant and EWMA these are constant across the test window;
#  for AR(1) they evolve with the iterated recursion.)
lam_predictions = {name: f.predict(n_test) for name, f in forecasters.items()}

for name, lam in lam_predictions.items():
    print(f"  {name:14s}: λ̂ shape {lam.shape}, first row = {lam[0].round(5)}")

# %% [markdown]
# ## 7. Stage 3 — Asset Prediction Across Forecasters
#
# Stage 3 turns each forecaster's $\hat\lambda_{t+1}$ into per-asset signals
# via $\hat\mu_{i,t+1} = \beta_{i,t} \cdot \hat\lambda_{t+1}$ where
# $\beta_{i,t} = z_{i,t} \Gamma$ from Stage 1.

# %%
test_betas = np.einsum("tnl,lk->tnk", test_chars, ipca.gamma)


# %% [markdown]
# The Stage 3 mapper itself is a single einsum: contract the
# $(T, N, K)$ beta tensor with the $(T, K)$ factor-forecast matrix to
# produce $(T, N)$ asset-level forecasts.


# %%
def stage3_predictions(lam: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Map factor forecast to asset forecast: μ̂ = β · λ̂."""
    return np.einsum("tnk,tk->tn", betas, lam)


# %% [markdown]
# To evaluate forecasters with the chapter's IC machinery, reshape the
# wide $(T, N)$ prediction and realised-return arrays into a long-form
# Polars frame keyed by `(date, symbol)`.


# %%
def _long_form(mu_hat: np.ndarray, fwd: np.ndarray) -> pl.DataFrame:
    """Reshape (T, N) prediction/return arrays into long-form (date, symbol, y_pred, y_true)."""
    T, N = mu_hat.shape
    return pl.DataFrame(
        {
            "date": np.repeat(np.arange(T), N),
            "symbol": np.tile(np.arange(N), T),
            "y_pred": mu_hat.flatten(),
            "y_true": fwd.flatten(),
        }
    )


# %% [markdown]
# `compute_period_ic` returns one Spearman cross-sectional IC per
# period; we use it to build the IC distributions plotted later.


# %%
def compute_period_ic(mu_hat: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Per-date Spearman cross-sectional IC series (one IC per period)."""
    df = _long_form(mu_hat, fwd)
    series = cross_sectional_ic_series(
        predictions=df,
        returns=df,
        pred_col="y_pred",
        ret_col="y_true",
        date_col="date",
        entity_col="symbol",
        method="spearman",
        min_obs=5,
    )
    return series.drop_nulls("ic")["ic"].to_numpy()


# %% [markdown]
# `evaluate` summarises a single forecaster: OOS $R^2$ plus the
# library's IC summary (mean, IR, t-stat, p-value, period count).


# %%
def evaluate(name: str, mu_hat: np.ndarray, fwd: np.ndarray) -> dict:
    """OOS R² + per-date Spearman IC summary (mean, IR, t-stat, p-value, n_periods)."""
    mask = ~np.isnan(fwd.flatten())
    pf = mu_hat.flatten()[mask]
    af = fwd.flatten()[mask]
    ss_res = np.sum((af - pf) ** 2)
    ss_tot = np.sum((af - af.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    df = _long_form(mu_hat, fwd)
    summary = cross_sectional_ic(
        predictions=df,
        returns=df,
        pred_col="y_pred",
        ret_col="y_true",
        date_col="date",
        entity_col="symbol",
        method="spearman",
        min_obs=5,
    )
    return {
        "Forecaster": name,
        "R²": f"{r2:.4f}",
        "Mean IC": f"{summary['ic_mean']:.4f}",
        "IC IR": f"{summary['ic_ir']:.4f}",
        "t-stat": f"{summary['ic_t']:.2f}",
        "p-value": f"{summary['p_value']:.3f}",
        "Periods": int(summary["n_periods"]),
    }


# %%
results = []
mu_hat_by_forecaster = {}
for name, lam in lam_predictions.items():
    mu_hat = stage3_predictions(lam, test_betas)
    mu_hat_by_forecaster[name] = mu_hat
    results.append(evaluate(name, mu_hat, test_fwd))

pl.DataFrame(results)

# %% [markdown]
# ### Cache Stage 3 Predictions
#
# Save the stacked long-form prediction frame so re-running the visualisation
# and K-sensitivity cells below doesn't re-fit IPCA. Cache key includes the
# data dimensions, IPCA hyperparameters, and the forecaster set; see
# `utils/predictions_cache.py` for the contract.

# %%
predictions_spec = {
    "notebook": "ipca",
    "data": {
        "source": "synthetic",
        "n_periods": N_PERIODS,
        "n_assets": N_ASSETS,
        "n_characteristics": N_CHARACTERISTICS,
        "n_true_factors": N_TRUE_FACTORS,
        "train_frac": TRAIN_FRAC,
        "random_seed": RANDOM_SEED,
    },
    "model": {
        "name": "ipca",
        "n_factors": N_IPCA_FACTORS,
        "max_iter": MAX_ITER,
    },
    "forecasters": list(lam_predictions.keys()),
}

stacked = pl.concat(
    [
        _long_form(mu_hat, test_fwd).with_columns(pl.lit(name).alias("forecaster"))
        for name, mu_hat in mu_hat_by_forecaster.items()
    ]
)
save_predictions(chapter=14, notebook_id="ipca", spec=predictions_spec, frame=stacked)
print(f"Cached {len(stacked):,} rows under key {predictions_cache_key(predictions_spec)}")

# %% [markdown]
# **Interpretation**: All three forecasters produce ICs that are not
# distinguishable from zero — IC means around 0.04–0.06 over 149 test
# periods, with t-statistics of 0.90–1.62 (p-values 0.107–0.370). OOS $R^2$
# is essentially zero across the board (Constant +0.0003, AR(1) +0.0002,
# EWMA -0.0175). This is the **correct** result — the factor series was
# constructed as i.i.d. draws, so $f_{t+1}$ is unpredictable from $f_{1:t}$
# regardless of forecaster sophistication. Stage 1 is sound (Γ recovery is
# strong — rotation-aligned correlations 0.97 / 0.85 / 0.63, Section 5);
# Stage 3 is mechanically correct
# ($\hat\mu = \beta \cdot \hat\lambda$); the only path through which a
# Stage 2 choice could matter is if the factor series carried a forecastable
# component, and here, by construction, it does not.
#
# This null is pedagogically important: it isolates *what Stage 2 forecasters
# can and cannot do*. They can extract structure from a forecastable factor
# series (real macro/risk-premium dynamics); they cannot create signal where
# none exists. The case study notebook demonstrates the contrast on real
# characteristics-driven data.

# %% [markdown]
# ## 8. K-Sensitivity Analysis (Constant forecaster)
#
# Within the framework, Stage 1 hyperparameters (here, the number of factors
# $K$) interact with Stage 2 capacity. We fix Stage 2 to Constant and sweep
# $K$ from 1 to 6, showing that even with the simplest forecaster, choosing
# $K$ above the true factor count (3) inflates noise without adding signal.

# %%
k_values = range(1, 7)
k_results = []

for k in k_values:
    model_k = IPCA(n_factors=k, max_iter=MAX_ITER)
    model_k.fit(train_returns, train_chars)

    # Stage 2 = Constant on the K-factor model's training factors
    lam_k = ConstantForecaster().fit(model_k.factors).predict(n_test)
    betas_k = np.einsum("tnl,lk->tnk", test_chars, model_k.gamma)
    mu_hat_k = np.einsum("tnk,tk->tn", betas_k, lam_k)

    m = ~np.isnan(test_fwd.flatten())
    pf = mu_hat_k.flatten()[m]
    af = test_fwd.flatten()[m]
    ss_r = np.sum((af - pf) ** 2)
    ss_t = np.sum((af - af.mean()) ** 2)
    r2_k = 1 - ss_r / ss_t if ss_t > 0 else 0

    ic_k = compute_period_ic(mu_hat_k, test_fwd)
    ic_k_mean = float(np.mean(ic_k)) if len(ic_k) else 0.0

    k_results.append({"K": k, "R²": r2_k, "Mean IC": ic_k_mean})
    print(f"  K={k}: R²={r2_k:.4f}, IC={ic_k_mean:.4f}")

pl.DataFrame(k_results)

# %% [markdown]
# **Interpretation**: With i.i.d. factors no $K$ delivers a meaningful OOS
# R² — every entry sits within $\pm 5 \times 10^{-4}$ of zero, indistinguishable
# from sampling noise. The Stage-1 fit, by contrast, shows its in-sample MSE
# flattening once $K$ reaches the true factor count: iter-0 MSE falls steeply
# from $K=1$ (0.00067) through $K=2$ (0.00028) to $K=3$ (0.00010), then
# plateaus for $K\geq 3$ (≈0.0001 at $K=4$–6, converging in a handful of ALS
# iterations). The elbow at $K=3$ marks where extra factors stop reducing the
# fit error materially. On real data where characteristics carry genuine predictive
# information, $K$-sweeps typically show OOS improvement up to the optimal
# factor count. Always report results across a range of $K$ values rather
# than selecting a single "optimal" $K$.

# %% [markdown]
# ## 9. Visualization

# %% [markdown]
# Build the 2x2 figure: $\Gamma$ heatmap (top-left), factor time series
# with Stage 2 overlay (top-right), Stage 3 IC distributions
# (bottom-left), and the K-sensitivity sweep (bottom-right).

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: estimated Gamma heatmap
ax1 = axes[0, 0]
im = ax1.imshow(ipca.gamma, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax1.set_xlabel("Factor", fontsize=11)
ax1.set_ylabel("Characteristic", fontsize=11)
ax1.set_xticks(range(ipca.n_factors))
ax1.set_xticklabels([f"Factor {k + 1}" for k in range(ipca.n_factors)])
ax1.set_yticks(range(n_chars))
ax1.set_yticklabels([f"Char {c + 1}" for c in range(n_chars)], fontsize=9)
ax1.set_title("Stage 1: Estimated Γ Matrix", fontsize=12)
_ = plt.colorbar(im, ax=ax1)

# %%
# Panel 2: factor time series (train + realised test) + Stage 2 Constant overlay
ax2 = axes[0, 1]
T_train_plot = ipca.factors.shape[0]
test_factors_actual = ipca.estimate_factors(test_returns, test_chars)
T_total = T_train_plot + len(test_factors_actual)

for k in range(min(2, ipca.n_factors)):
    ax2.plot(
        np.arange(T_train_plot),
        ipca.factors[:, k],
        alpha=0.6,
        label=f"f_{k + 1} (train)" if k < 2 else None,
    )
    ax2.plot(
        np.arange(T_train_plot, T_total),
        test_factors_actual[:, k],
        alpha=0.6,
        linestyle="-",
        label=f"f_{k + 1} (realised test)" if k == 0 else None,
    )

ax2.axhline(
    lam_predictions["Constant"][0, 0],
    color=COLORS["amber"],
    linestyle="--",
    linewidth=1.5,
    label="λ̂ Constant (factor 1)",
)
ax2.axvline(T_train_plot, color="gray", linestyle=":", linewidth=1)
ax2.set_xlabel("Time")
ax2.set_ylabel("Factor Value")
ax2.set_title("Stage 1 Factor Series + Stage 2 Forecast")
_ = ax2.legend(fontsize=8, loc="upper right")

# %%
# Panel 3: IC distribution per forecaster
ax3 = axes[1, 0]
for name, mu_hat in mu_hat_by_forecaster.items():
    ics = compute_period_ic(mu_hat, test_fwd)
    ax3.hist(ics, bins=20, alpha=0.45, label=name, edgecolor="white")
ax3.axvline(0, color="gray", linestyle=":", linewidth=1)
ax3.set_xlabel("Information Coefficient")
ax3.set_ylabel("Frequency")
ax3.set_title("Stage 3 IC Distribution per Forecaster")
ax3.legend()

# Panel 4: K-sensitivity curve
ax4 = axes[1, 1]
ks = [r["K"] for r in k_results]
r2s = [r["R²"] for r in k_results]
ics = [r["Mean IC"] for r in k_results]
ax4.plot(ks, r2s, marker="o", color=COLORS["blue"], label="R²")
ax4_twin = ax4.twinx()
ax4_twin.plot(ks, ics, marker="s", color=COLORS["amber"], label="Mean IC")
ax4.axvline(N_TRUE_FACTORS, color="gray", linestyle=":", linewidth=1)
ax4.set_xlabel("K (number of factors)")
ax4.set_ylabel("R²", color=COLORS["blue"])
ax4_twin.set_ylabel("Mean IC", color=COLORS["amber"])
ax4.set_title("K-Sensitivity (Stage 2 = Constant)")
ax4.legend(loc="upper left")
ax4_twin.legend(loc="upper right")

fig.subplots_adjust(left=0.08, right=0.94, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
fig.show()

# %% [markdown]
# **Interpretation**: The Γ heatmap (top-left) shows Stage 1's
# characteristic-to-factor mapping. The factor time series (top-right) shows
# Stage 1's training-window factor history with the Constant Stage 2 forecast
# overlaid as a horizontal dashed line — visualising that Constant collapses
# the entire history into one number. The IC histograms (bottom-left) show
# all three forecasters bunching around zero, as expected on i.i.d. factors.
# K-sensitivity (bottom-right) confirms that even with a fixed Stage 2,
# Stage 1 hyperparameter choices propagate through the pipeline.

# %% [markdown]
# ## 10. Key Takeaways
#
# 1. **IPCA fits Stage 1 of the framework cleanly**: $\beta_{i,t} = z_{i,t}\Gamma$
#    produces time-varying loadings whose dynamics come from observable
#    characteristics, and the factor history $F$ feeds Stage 2 just like any
#    other LF estimator.
# 2. **Stage 2 is swappable**: Constant, AR(1), and EWMA all plug into the same
#    pipeline; the only difference is how $\hat\lambda_{t+1}$ is computed from
#    $F_{1:t}$. KPS 2019's predictive formula corresponds to the Constant
#    forecaster.
# 3. **Parameter recovery validates Stage 1**: ALS correctly identifies the
#    true Γ up to rotation, isolating the source of any prediction failure.
# 4. **Synthetic null is correct**: i.i.d. factors are unpredictable from the
#    factor history, so all forecasters must produce near-zero IC. Real data
#    with persistent characteristic-driven premia is where Stage 2 choice
#    starts to matter.
# 5. **K-sensitivity propagates through the pipeline**: even with the simplest
#    Stage 2, choosing $K$ above the true factor count degrades OOS quality.
#    Always report a sweep, not a single value.
#
# **Next**: See [`05_rp_pca`](05_rp_pca.ipynb) for risk-premium-aware Stage 1
# (Lettau-Pelger 2020) under the same framework, and
# [`08_latent_factors`](../case_studies/us_firm_characteristics/08_latent_factors.ipynb)
# for real-data application with walk-forward IC evaluation across all five
# LF estimators and the full forecaster catalog.
