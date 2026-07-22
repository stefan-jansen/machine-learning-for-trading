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
# # Risk-Premium PCA (RP-PCA) in the Two-Step Framework
#
# **Docker image**: `ml4t`
#
# **Chapter 14: Latent Factors**
#
# This notebook implements Risk-Premium PCA from Lettau & Pelger (2020)
# "Estimating Latent Asset-Pricing Factors", and slots the implementation
# into the chapter's two-step factor-forecasting framework. RP-PCA's
# innovation lives at **Stage 1** — the factor model is selected to
# balance variance explanation with cross-sectional pricing — while
# Stages 2 and 3 are shared with PCA and IPCA.
#
# ## The Two-Step Framework (recap)
#
# 1. **Stage 1 — Factor Model.** Compresses an $(T \times N)$ returns
#    panel to a $(T \times K)$ factor history $F$. RP-PCA realises this
#    stage as the eigendecomposition of $\Sigma + \gamma\,\bar{X}\bar{X}'$
#    rather than the plain covariance $\Sigma$.
# 2. **Stage 2 — Factor Premium Forecaster.** Predicts $\hat\lambda_{t+1}$
#    from the training factor history $F_{1:t}$. Lettau & Pelger's predictive
#    formula uses the sample mean (= our `Constant` forecaster); we also
#    demonstrate AR(1) and EWMA (Figure 14.10).
# 3. **Stage 3 — Asset Mapper.** Translates the factor forecast back to a
#    per-asset signal: $\hat\mu_{i,t+1} = B_i \cdot \hat\lambda_{t+1}$,
#    where $B$ are the (static within-fold) RP-PCA loadings.
#
# ## What RP-PCA changes
#
# Standard PCA finds factors that explain **variance**. RP-PCA modifies the
# objective so that factors must also explain **cross-sectional expected
# returns**:
#
# $$\mathcal{L} = \frac{1}{NT}\sum_{i,t}(X_{ti} - F'_t \Lambda_i)^2
# + \gamma\,\frac{1}{N}\sum_i(\bar{X}_i - \bar{F}'\Lambda_i)^2$$
#
# The second term penalises pricing errors. The estimator is equivalent to
# applying PCA to the modified second-moment matrix
# $\Sigma + \gamma\,\bar{X}\bar{X}'$.
#
# | $\gamma$ | Behaviour |
# |---|---|
# | $\gamma = 0$ | Standard covariance PCA |
# | $\gamma > 0$ | Overweight expected-return information |
# | Large $\gamma$ | Better at detecting **weak high-Sharpe** factors that PCA misses |
#
# This is most valuable when the variance-dominant factors and the
# pricing-dominant factors **diverge** — which happens in large heterogeneous
# panels (the paper's 370 anomaly portfolios) but not in a diversified ETF
# universe.
#
# ## Important caveat — RP-PCA assumes stable loadings
#
# Lettau & Pelger find that RP-PCA performs poorly out-of-sample on
# **individual stocks** because firm-level loadings are unstable. RP-PCA
# is best suited to **characteristic-sorted portfolios**, not individual
# names. For stock-level prediction, the natural extension is to combine
# RP-PCA's factor extraction with IPCA's characteristic-driven loadings —
# see [`04_ipca`](04_ipca.ipynb) and the case study notebooks.
#
# ## Learning Objectives
#
# - LO1: Understand the variance-vs-pricing tradeoff and the role of $\gamma$
# - LO2: Implement RP-PCA via the modified eigendecomposition
# - LO3: Decompose factors by role (comovement vs priced) following the
#   paper's empirical taxonomy
# - LO4: Combine the RP-PCA Stage 1 with the Stage 2 forecaster catalog
#   (Constant / AR(1) / EWMA) and evaluate Stage 3 asset predictions
# - LO5: Recognise when RP-PCA helps (heterogeneous panels with weak priced
#   factors) and when it doesn't (homogeneous diversified portfolios)
#
# **Data**: ETF daily returns (~100 ETFs, ~18 years). The ETF universe is
# diversified, so we *do not* expect a large RP-PCA improvement here — the
# null is itself the lesson.
#
# **Prerequisites**: Complete [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb) and [`04_ipca`](04_ipca.ipynb).
# **Next**: [`06_conditional_autoencoder`](06_conditional_autoencoder.ipynb) for non-linear Stage 1.

# %% [markdown]
# ## 1. Setup and Imports

# %%
"""RP-PCA in the two-step framework — pricing-aware Stage 1, swappable Stage 2 forecasters."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ml4t.diagnostic.metrics import cross_sectional_ic, cross_sectional_ic_series
from scipy.linalg import eigh

from data import load_etfs
from utils.predictions_cache import predictions_cache_key, save_predictions
from utils.style import COLORS

warnings.filterwarnings("ignore")


# %% [markdown]
# ### Universe-Size Scaling
#
# RP-PCA decomposes a $T \times N$ panel; the dominant cost is the
# $N \times N$ eigendecomposition for each $\gamma$ (so wall-time is roughly
# linear in `len(GAMMAS)` and cubic in `N`). The default ETF universe (~100
# symbols × 3,346 trading days) fits all seven $\gamma$ values in seconds.
# Setting `MAX_SYMBOLS = 30` halves runtime for a smoke run; scaling to a
# 500-stock US-equity panel grows the eigendecomposition ~125× and pushes a
# single $\gamma$ to a few seconds.

# %% tags=["parameters"]
# Production defaults (Papermill overrides for CI testing)
N_FACTORS = 5
GAMMAS = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
TRAIN_FRAC = 0.7
START_DATE = "2006-01-01"
END_DATE = "2024-12-31"
MAX_SYMBOLS = 0  # 0 = all
MIN_OBSERVATIONS = 252

# %% [markdown]
# ## 2. Stage 1 — RP-PCA Factor Extraction
#
# RP-PCA decomposes the matrix
# $$M_\gamma = \Sigma + \gamma\,\bar{r}\bar{r}^\top,$$
# where $\Sigma$ is the **centered** sample covariance of returns and
# $\bar{r}$ is the cross-sectional mean. This is the centered-covariance
# convention; an alternative formulation uses the uncentered second moment
# $\frac{1}{T}X^\top X$ in place of $\Sigma$, which produces nearly
# identical estimates for daily returns whose means are small relative to
# their standard deviations.
#
# $M_\gamma$ is symmetric by construction; we extract its top-$K$
# eigenvectors via `scipy.linalg.eigh` (faster and more stable than the
# general `eig` for symmetric matrices). The eigenvectors define the
# factor portfolio weights $\Lambda$; factor returns are $F = X\Lambda$.
# Mean returns are estimated only from the training window — leakage check
# per gpt-01 §"Leakage warnings".


# %%
class RPPCA:
    """
    Risk-Premium Principal Component Analysis.

    Lettau & Pelger (2020) "Estimating Latent Asset-Pricing Factors".

    Stage 1 of the two-step framework. Outputs:
        loadings: (N, K) factor portfolio weights (eigenvectors of M_γ)
        factors:  (T, K) factor return series F = X · W
        risk_premia: (K,) sample-mean factor premia

    Parameters
    ----------
    n_factors : int
        Number of latent factors K
    gamma : float
        Weight on cross-sectional pricing error (γ=0 ⇒ standard PCA)
    """

    def __init__(self, n_factors: int = 5, gamma: float = 1.0):
        self.n_factors = n_factors
        self.gamma = gamma
        self.loadings = None
        self.factors = None
        self.risk_premia = None
        self.fitted = False

    def fit(self, returns: np.ndarray) -> "RPPCA":
        """Fit RP-PCA via eigendecomposition of Σ + γ·μμ' (centered covariance)."""
        T, N = returns.shape
        K = self.n_factors

        print(f"RP-PCA Stage 1: T={T}, N={N}, K={K}, γ={self.gamma}")

        mu = returns.mean(axis=0)
        sigma = np.cov(returns.T)
        rp_matrix = sigma + self.gamma * np.outer(mu, mu)

        eigenvalues, eigenvectors = eigh(rp_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.loadings = eigenvectors[:, :K]
        self.eigenvalues = eigenvalues[:K]
        self.factors = returns @ self.loadings
        self.risk_premia = self.factors.mean(axis=0)

        self.fitted = True
        self._mu = mu
        return self

    def extract_factors(self, returns: np.ndarray) -> np.ndarray:
        """Project new returns onto the trained factor space (F = X · Λ)."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return returns @ self.loadings

    def get_factor_sharpes(self) -> np.ndarray:
        """Annualised Sharpe ratio per factor (assumes daily returns)."""
        if self.factors is None:
            raise ValueError("Model not fitted.")
        return np.array(
            [
                self.factors[:, k].mean() / self.factors[:, k].std() * np.sqrt(252)
                for k in range(self.n_factors)
            ]
        )


# %% [markdown]
# ### Method summary
#
# `RPPCA.fit(returns)` constructs $M_\gamma$ from the training window and
# extracts the top-$K$ eigenvectors as loadings $\Lambda$, returning a
# fitted instance.
#
# `RPPCA.extract_factors(returns)` projects new returns onto the trained
# factor space ($F_{\text{test}} = X_{\text{test}}\Lambda$), producing the
# *realised* factor history on the test window — used to evaluate Stage 2
# forecasters against ground truth.
#
# `RPPCA.get_factor_sharpes()` reports the annualised Sharpe ratio per
# latent factor — $\sqrt{252}\,\bar{f}_k / s_{f_k}$. This is **not** a
# tradeable strategy's Sharpe; the factor is a linear combination of the
# panel's columns (a "factor portfolio") and its SR measures how much
# risk-adjusted return that combination *would* earn if it were directly
# investable. Factor SR is the key diagnostic for whether $\gamma$ is
# elevating high-SR directions over high-variance ones (priced vs
# comoving factors).


# %% [markdown]
# ## 3. Load ETF Return Data
#
# 100 diversified ETFs, daily returns, 2006-2024. The temporal split
# preserves the leakage discipline: $\bar{X}$ and $\Sigma$ are estimated
# on the training window only.

# %%
etf_data = load_etfs(start_date=START_DATE, end_date=END_DATE)

etf_returns = (
    etf_data.sort("timestamp")
    .with_columns(pl.col("close").pct_change().over("symbol").alias("return"))
    .drop_nulls(subset=["return"])
)

symbol_counts = etf_returns.group_by("symbol").len()
valid_symbols = symbol_counts.filter(pl.col("len") >= MIN_OBSERVATIONS)["symbol"].to_list()
if MAX_SYMBOLS > 0:
    valid_symbols = valid_symbols[:MAX_SYMBOLS]

etf_returns = etf_returns.filter(pl.col("symbol").is_in(valid_symbols))

return_wide = etf_returns.pivot(on="symbol", index="timestamp", values="return").sort("timestamp")
dates = return_wide["timestamp"].to_numpy()
symbols = [c for c in return_wide.columns if c != "timestamp"]
returns_np = return_wide.select(symbols).to_numpy().astype(np.float64)
returns_np = np.nan_to_num(returns_np, nan=0.0)

T, N = returns_np.shape
train_size = int(T * TRAIN_FRAC)

print(f"ETF panel: T={T}, N={N}, train={train_size}, test={T - train_size}")
print(f"Date range: {dates[0]} to {dates[-1]}")

# %%
pl.DataFrame(
    {
        "Parameter": ["Trading Days", "ETFs", "Factors", "Train Size", "Test Size"],
        "Value": [T, N, N_FACTORS, train_size, T - train_size],
    }
)

# %% [markdown]
# ## 4. Stage 1 Output: RP-PCA Sweep over γ
#
# Fit RP-PCA across the γ grid. At γ=0 we get standard PCA; increasing γ
# progressively shifts factor extraction toward pricing-relevant
# dimensions. Stage 1's outputs ($\Lambda$, $F$, factor Sharpes) are
# what Stages 2 and 3 will consume.

# %%
train_returns = returns_np[:train_size]
test_returns = returns_np[train_size:]

# Forward returns for prediction (Stage 3 evaluation target)
forward_returns = np.roll(returns_np, -1, axis=0)
forward_returns[-1] = np.nan
test_fwd = forward_returns[train_size:-1]

results = {}

for gamma in GAMMAS:
    name = "PCA" if gamma == 0 else f"RP-PCA (γ={gamma})"
    print(f"\nFitting {name}...")
    model = RPPCA(n_factors=N_FACTORS, gamma=gamma)
    model.fit(train_returns)
    factor_sharpes = model.get_factor_sharpes()

    results[name] = {
        "gamma": gamma,
        "model": model,
        "factor_sharpes": factor_sharpes,
        "avg_factor_sharpe": float(np.mean(np.abs(factor_sharpes))),
    }
    print(f"  Factor Sharpes: {factor_sharpes.round(2)}")

# %% [markdown]
# ## 5. Factor Role Decomposition (paper's pedagogical taxonomy)
#
# Lettau & Pelger argue that latent factors fall into three economic
# roles, distinguishable by their *first* and *second* moments:
#
# | Role | Variance | Mean (Sharpe) | Use |
# |---|---|---|---|
# | **Comovement** | High | Low | Risk modelling — explains correlations |
# | **Priced** | Low | High | Expected-return premia — explains cross-section |
# | **Both** | High | High | Doubly important; rare |
#
# RP-PCA's contribution is detecting the **Priced** role, which standard
# PCA may miss because such factors have low variance and would otherwise
# be ranked below noise-dominated PCs.

# %%
# Decompose factors by role for the largest meaningful γ
roles_gamma = 100.0
roles_model = results[f"RP-PCA (γ={roles_gamma})"]["model"]
factor_means = roles_model.risk_premia
factor_vols = roles_model.factors.std(axis=0)
factor_sharpes = roles_model.get_factor_sharpes()

# Heuristic role assignment: median split on |mean| × Sharpe / vol
abs_means = np.abs(factor_means)
abs_sharpes = np.abs(factor_sharpes)
mean_threshold = np.median(abs_means)
vol_threshold = np.median(factor_vols)

roles = []
for k in range(roles_model.n_factors):
    high_mean = abs_means[k] > mean_threshold
    high_var = factor_vols[k] > vol_threshold
    if high_mean and high_var:
        role = "Both (rare)"
    elif high_mean and not high_var:
        role = "Priced (RP-PCA's target)"
    elif high_var and not high_mean:
        role = "Comovement (risk model)"
    else:
        role = "Marginal"
    roles.append(role)

pl.DataFrame(
    {
        "Factor": [f"F_{k + 1}" for k in range(roles_model.n_factors)],
        "|Mean|": [f"{m:.4f}" for m in abs_means],
        "Vol": [f"{v:.4f}" for v in factor_vols],
        "Sharpe (ann.)": [f"{s:.2f}" for s in factor_sharpes],
        "Role": roles,
    }
)

# %% [markdown]
# **Interpretation**: On a diversified ETF panel the heuristic median split
# tends to label the first two factors "Both (rare)" — they have variance
# and Sharpe magnitude above the median because the panel is dominated by a
# single high-variance market factor with non-trivial sample mean — and the
# remaining factors fall into "Marginal" (below the median on both axes).
# True "Priced" factors (low variance, high Sharpe) — the ones RP-PCA was
# designed to elevate — do not surface here. Their absence reflects the
# homogeneity of the asset universe rather than any defect in the estimator.
# Lettau & Pelger find Priced factors abundantly in 370-anomaly-portfolio
# panels, where the cross-section spans a much wider Sharpe range.

# %% [markdown]
# ## 6. Stage 2 — Factor Premium Forecasters
#
# Stage 2 takes the training factor history $F_{1:T}$ and emits
# $\hat\lambda_{t+1}\in\mathbb{R}^K$. Three Tier-1 forecasters from Figure
# 14.10 — same classes as in `04_ipca` — let us swap forecasters without
# touching Stages 1 or 3.


# %% [markdown]
# **Constant forecaster.** The sample mean of training factors —
# corresponds to Lettau & Pelger's predictive default and serves as the
# baseline for the head-to-head comparison.


# %%
class ConstantForecaster:
    """Sample mean of training factors (= Lettau-Pelger predictive default)."""

    def fit(self, F: np.ndarray) -> "ConstantForecaster":
        self.lam_ = F.mean(axis=0)
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        return np.tile(self.lam_, (n_steps, 1))


# %% [markdown]
# **AR(1) forecaster.** A per-factor AR(1) model
# $\hat\lambda_{k,t+1} = c_k + \phi_k\,f_{k,t}$ fit by OLS on the training
# factor history.
#
# **Open-loop semantics.** The AR(1) Stage 2 forecast here is *open-loop*:
# the recursive multi-step path is generated from the last *training*
# factor value and propagated forward without updating from realised test
# factors. This isolates parameter-estimation noise (how well $\phi_k$ is
# learned) from rolling re-estimation noise (how well it adapts after each
# new observation). Production deployments rebalance and re-fit on every
# new bar; the open-loop path is the pedagogical lower bound.


# %%
class AR1Forecaster:
    """Per-factor AR(1): λ̂_{k,t+1} = c_k + φ_k · f_{k,t} (open-loop)."""

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
        out = np.zeros((n_steps, len(self.last_)))
        prev = self.last_.copy()
        for s in range(n_steps):
            prev = self.c_ + self.phi_ * prev
            out[s] = prev
        return out


# %% [markdown]
# **EWMA forecaster.** Exponentially weighted mean of training factors
# with half-life $h$ periods (default 60 daily observations ≈ 12-week
# half-life). Down-weights older history relative to the Constant
# forecaster.


# %%
class EWMAForecaster:
    """EWMA with half-life h periods (default 60 daily ≈ 12-week half-life)."""

    def __init__(self, half_life: float = 60.0):
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
# ### Fit each forecaster on Stage 1 output for one chosen γ
#
# We pick γ=10 (the paper's recommended weight for detecting weak priced
# factors) for the head-to-head Stage 2 comparison. The Constant baseline
# below corresponds to Lettau & Pelger's predictive formula.

# %%
focus_gamma = 10.0
focus_model = results[f"RP-PCA (γ={focus_gamma})"]["model"]
F_train = focus_model.factors  # (T_train, K)
n_test = len(test_fwd)

forecasters = {
    "Constant": ConstantForecaster().fit(F_train),
    "AR(1)": AR1Forecaster().fit(F_train),
    "EWMA(h=60)": EWMAForecaster(half_life=60.0).fit(F_train),
}

lam_predictions = {name: f.predict(n_test) for name, f in forecasters.items()}

for name, lam in lam_predictions.items():
    print(f"  {name:14s}: λ̂ shape {lam.shape}, first row = {lam[0].round(6)}")

# %% [markdown]
# ## 7. Stage 3 — Asset Prediction Across Forecasters
#
# Stage 3 turns each Stage 2 output into a per-asset signal:
# $$\hat\mu_{i,t+1} = \Lambda_i \cdot \hat\lambda_{t+1}$$
# where $\Lambda$ is the (static) RP-PCA loading matrix from Stage 1.
# This contrasts with IPCA's $\beta_{i,t} = z_{i,t}\Gamma$ — RP-PCA's
# loadings do not vary with characteristics, which is exactly the
# limitation the paper warns about for individual stocks.


# %% [markdown]
# **`stage3_predictions`.** Map a factor-premium forecast back to a
# per-asset signal via the (static) RP-PCA loading matrix.


# %%
def stage3_predictions(lam: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """μ̂_{t,i} = Σ_k Λ_{i,k} · λ̂_{t,k}."""
    return lam @ loadings.T


# %% [markdown]
# **`_long_form`.** Reshape paired (T, N) prediction and realised-return
# arrays into a long-form Polars frame keyed by (date, symbol) — the
# format consumed by the `ml4t.diagnostic` IC helpers.


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
# **`compute_period_ic`.** Return the per-date Spearman cross-sectional IC
# series — one IC per period — used downstream for IC-distribution plots.


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
# **`evaluate`.** Aggregate Stage 3 quality into a one-row summary: pooled
# OOS R² plus per-date Spearman IC mean, IR, t-statistic, p-value, and
# number of test periods.


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
focus_loadings = focus_model.loadings  # (N, K)

stage3_results = []
mu_hat_by_forecaster = {}
for name, lam in lam_predictions.items():
    mu_hat = stage3_predictions(lam, focus_loadings)
    mu_hat_by_forecaster[name] = mu_hat
    stage3_results.append(evaluate(name, mu_hat, test_fwd))

pl.DataFrame(stage3_results)

# %% [markdown]
# ### Cache Stage 3 Predictions
#
# We cache the stacked long-form prediction frame (date, symbol, forecaster,
# y_pred, y_true) so re-running the visualisation / interpretation cells below
# does not retrain Stage 1 or recompute Stages 2 and 3. The cache is keyed on
# the data window, RP-PCA hyperparameters, and the forecaster set — change any
# of those and `predictions_cache_key(spec)` shifts to a new file. See
# `utils/predictions_cache.py` for the contract.

# %%
predictions_spec = {
    "notebook": "rp_pca",
    "data": {
        "source": "etfs",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "max_symbols": MAX_SYMBOLS,
        "min_obs": MIN_OBSERVATIONS,
    },
    "model": {
        "name": "rp_pca",
        "n_factors": N_FACTORS,
        "focus_gamma": focus_gamma,
        "train_frac": TRAIN_FRAC,
    },
    "forecasters": list(lam_predictions.keys()),
}

stacked = pl.concat(
    [
        _long_form(mu_hat, test_fwd).with_columns(pl.lit(name).alias("forecaster"))
        for name, mu_hat in mu_hat_by_forecaster.items()
    ]
)
save_predictions(chapter=14, notebook_id="rp_pca", spec=predictions_spec, frame=stacked)
print(f"Cached {len(stacked):,} rows under key {predictions_cache_key(predictions_spec)}")

# %% [markdown]
# **Interpretation**: On daily ETF data the three Stage 2 forecasters
# deliver mean per-date IC around 0.021 with t-statistics of 2.13 (Constant /
# AR(1)) and 2.23 (EWMA) over 1,433 test periods (p-values 0.026–0.033) —
# small in magnitude but statistically distinguishable from zero. OOS R²
# is essentially zero (-0.0001 / -0.0001 / 0.000) because the long
# right-tail of asset returns dominates the squared-error denominator.
# Two forces shape the result:
#
# 1. **Signal-to-noise floor.** Daily mean returns are extraordinarily noisy
#    (signal-to-noise ≈ 0.04). The IC sits in the same magnitude band as
#    individual-day cross-sectional variation; it is large enough on average
#    to reject the null but too small to support a daily-rebalancing strategy
#    after costs.
# 2. **Static loadings on individual ETFs.** Stage 3's static $\Lambda$
#    cannot differentiate ETFs whose true factor exposures vary over time.
#    This is the loadings-stability limitation Lettau & Pelger flag for
#    individual securities.
#
# The pedagogical takeaway is structural: the framework correctly isolates
# *where* the limitation lives. Stage 1 (factor extraction) is sound;
# Stage 2 forecasters are interchangeable on this panel; the practical
# bottleneck is the daily horizon plus static Stage 3 loadings on a
# homogeneous universe. The case study notebooks move to longer horizons
# and to characteristic-driven Stage 3 (IPCA / CAE), which is where the
# framework starts to deliver economically meaningful predictions.

# %% [markdown]
# ## 8. Stage 1 Risk-Model Diagnostics (γ-sweep)
#
# Even when Stage 3 prediction is weak, RP-PCA's Stage 1 output retains
# its risk-modelling value: factor Sharpes, eigenvalue spectra, and
# loading rotations as $\gamma$ varies.

# %%
diag_results = []
for name, res in results.items():
    model = res["model"]
    test_recon = (test_returns @ model.loadings) @ model.loadings.T
    recon_flat = test_recon.flatten()
    actual_flat = test_returns.flatten()
    ss_res = np.sum((actual_flat - recon_flat) ** 2)
    ss_tot = np.sum((actual_flat - actual_flat.mean()) ** 2)
    recon_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    diag_results.append(
        {
            "Model": name,
            "γ": res["gamma"],
            "Avg |Factor SR|": f"{res['avg_factor_sharpe']:.2f}",
            "Recon R² (test)": f"{recon_r2:.4f}",
        }
    )

pl.DataFrame(diag_results)

# %% [markdown]
# **Finding**: Average |Factor SR| stays near 0.41 from $\gamma=0$ to
# $\gamma=100$, then drops at $\gamma=500$ as the first factor collapses
# onto the mean-return portfolio. Reconstruction $R^2$ degrades
# monotonically with $\gamma$ as the modified objective trades variance
# fit for pricing fit. This is the **expected** behaviour on a homogeneous
# panel — RP-PCA is operating correctly; the asset universe simply lacks
# the weak-but-priced factors that would reward heavy $\gamma$ weighting.

# %% [markdown]
# ## 9. Visualization
#
# Four panels summarise the γ-sweep diagnostics: factor Sharpe vs γ,
# eigenvalue spectra (PCA vs RP-PCA γ=100), Stage 3 IC distributions
# across forecasters, and the leading-loading rotation scatter.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Avg |Factor Sharpe| by γ
ax1 = axes[0, 0]
gammas_x = [res["gamma"] for res in results.values()]
sharpes_y = [res["avg_factor_sharpe"] for res in results.values()]
ax1.bar(range(len(gammas_x)), sharpes_y, color=COLORS["blue"], alpha=0.7)
ax1.set_xticks(range(len(gammas_x)))
ax1.set_xticklabels([f"γ={g}" for g in gammas_x], fontsize=9)
ax1.set_ylabel("Avg |Factor SR| (ann.)")
ax1.set_title("Stage 1: Factor SR by γ")
ax1.axhline(0, color="gray", linestyle="--")

# Panel 2: Eigenvalue spectrum at γ=0 vs γ=100
ax2 = axes[0, 1]
pca_eig = results["PCA"]["model"].eigenvalues
rppca_eig = results["RP-PCA (γ=100.0)"]["model"].eigenvalues
x_eig = np.arange(1, len(pca_eig) + 1)
ax2.bar(x_eig - 0.2, pca_eig, 0.4, label="PCA (γ=0)", color=COLORS["blue"], alpha=0.7)
ax2.bar(x_eig + 0.2, rppca_eig, 0.4, label="RP-PCA (γ=100)", color=COLORS["amber"], alpha=0.7)
ax2.set_xlabel("Factor")
ax2.set_ylabel("Eigenvalue")
ax2.set_title("Stage 1: Eigenvalue Spectrum")
ax2.legend()

# Panel 3: Stage 3 IC distributions per forecaster
ax3 = axes[1, 0]
for name, mu_hat in mu_hat_by_forecaster.items():
    ics = compute_period_ic(mu_hat, test_fwd)
    ax3.hist(ics, bins=30, alpha=0.45, label=name, edgecolor="white")
ax3.axvline(0, color="gray", linestyle=":", linewidth=1)
ax3.set_xlabel("Information Coefficient")
ax3.set_ylabel("Frequency")
ax3.set_title("Stage 3 IC Distribution (γ=10)")
ax3.legend()

# Panel 4: Loadings comparison PCA vs RP-PCA factor 1
ax4 = axes[1, 1]
pca_loadings = results["PCA"]["model"].loadings[:, 0]
rppca_loadings = results["RP-PCA (γ=100.0)"]["model"].loadings[:, 0]
ax4.scatter(pca_loadings, rppca_loadings, alpha=0.5, s=30)
ax4.plot([-1, 1], [-1, 1], "r--", label="45° line")
ax4.set_xlabel("PCA Loading (Factor 1)")
ax4.set_ylabel("RP-PCA Loading (Factor 1, γ=100)")
ax4.set_title("Loading Rotation: PCA vs RP-PCA")
ax4.legend()

fig.tight_layout()
fig.show()

# %% [markdown]
# **Interpretation**: The factor Sharpe ratios (top-left) are flat across
# γ ∈ [0, 100] at ~0.41 and only drop at γ=500 (to ~0.27) as the leading
# direction collapses onto the mean-return portfolio — the diversified ETF
# panel does not reward pricing-aware extraction at moderate γ. The
# eigenvalue spectrum (top-right) shows γ=100 inflating the first eigenvalue
# without reordering the spectrum. The IC histograms (bottom-left) cluster
# slightly to the right of zero, consistent with the per-date mean ≈ 0.021
# and t-statistic above 2 (CI excludes zero) and indistinguishable across
# forecasters. The loadings scatter (bottom-right) sits on the 45° line —
# even at γ=100, the first factor barely rotates.

# %% [markdown]
# ## 10. Key Takeaways
#
# 1. **RP-PCA is a Stage 1 estimator** that adds a pricing-error penalty
#    to the PCA objective. The estimator reduces to PCA at γ=0 and
#    progressively elevates high-Sharpe directions as γ grows.
# 2. **The framework slots in cleanly**: Stage 2 forecasters (Constant,
#    AR(1), EWMA) and Stage 3 mapping ($\hat\mu = \Lambda\hat\lambda$) are
#    inherited from the unified pipeline. Lettau & Pelger's predictive
#    formula corresponds to the Constant forecaster.
# 3. **Factor roles** decompose into Comovement (high var, low Sharpe),
#    Priced (low var, high Sharpe), and Both. RP-PCA's value is detecting
#    Priced factors that PCA misses — but only when the asset universe
#    contains them.
# 4. **The ETF result is structural, not vacuous**: on a diversified daily
#    ETF panel, RP-PCA Stage 1 produces near-identical results to PCA for
#    any γ ≤ 100; Stage 3 IC is small (~0.021) but statistically significant
#    (t ≈ 2.1–2.2 on 1,433 periods) and roughly flat across Constant, AR(1),
#    and EWMA. The framework runs cleanly; the magnitude is set by the
#    daily horizon and the homogeneous panel.
# 5. **Loading stability matters for Stage 3**: RP-PCA's static loadings
#    work for portfolio-level data but degrade on individual stocks. The
#    natural extension is to combine RP-PCA Stage 1 with IPCA's
#    characteristic-driven Stage 3.
#
# **Next**: See [`06_conditional_autoencoder`](06_conditional_autoencoder.ipynb) for non-linear Stage 1 with
# neural-network-based Stage 3 mapping (Gu, Kelly & Xiu 2021).
