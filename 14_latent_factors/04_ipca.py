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
# # Instrumented PCA: Conditional Betas and Factor Forecasts
#
# **Docker image**: `ml4t`
#
# **Chapter 14: Latent Factor Models**
#
# Instrumented PCA (IPCA) makes an asset's factor loadings linear functions of
# characteristics observed before the return:
#
# $$r^e_{i,t+1}=z_{i,t}^{\top}\Gamma f_{t+1}+\varepsilon_{i,t+1}.$$
#
# This notebook uses a synthetic panel with known $\Gamma$ for two separate
# checks. First, it verifies that alternating least squares (ALS) recovers the
# loading *subspace*. Second, it passes the estimated factors through the
# chapter's three-stage forecasting adapter without assuming that the factors
# are predictable.
#
# **Learning objectives**
#
# - implement the two ALS updates for IPCA;
# - evaluate recovery with rotation-invariant subspace diagnostics;
# - map genuinely walk-forward factor forecasts back to asset returns; and
# - report forecast uncertainty and sensitivity to the factor count.
#
# **Evaluation contract**: the synthetic characteristics at $t$ generate only
# $r_{t+1}$. The structural model uses 349 training pairs, followed by a
# one-period embargo and 150 evaluation pairs. The evaluation window is a
# teaching demonstration, not a sealed final holdout or a model-selection set.
#
# **Prerequisite**: [`01_pca_equity_sectors`](01_pca_equity_sectors.ipynb)
#
# **Book section**: Section 14.5, "Dynamic betas with instrumented PCA"
#
# **Next**: [`05_rp_pca`](05_rp_pca.ipynb) changes the Stage 1 objective to
# emphasize priced variation.

# %% [markdown]
# ## 1. Setup

# %%
"""Recover an IPCA loading subspace and evaluate walk-forward factor forecasts."""

from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from ml4t.diagnostic.metrics.uncertainty import compute_ic_uncertainty
from scipy.linalg import orthogonal_procrustes

from utils.reproducibility import set_global_seeds
from utils.style import (
    COLORS,
    FIGSIZE,
    add_message_title,
    ml4t_diverging,
    ml4t_palette,
    zero_line,
)

# %% tags=["parameters"]
N_PERIODS = 500
N_ASSETS = 100
N_CHARACTERISTICS = 10
N_TRUE_FACTORS = 3
N_IPCA_FACTORS = 3
TRAIN_BOUNDARY = 350
EMBARGO = 1
MAX_ITER = 100
N_BOOTSTRAP = 2_000
EWMA_HALF_LIFE = 12
SEED = 42

set_global_seeds(SEED)

# %% [markdown]
# The adapter keeps three responsibilities separate. IPCA estimates the
# conditional loadings and realized factor history in Stage 1. Stage 2 uses
# only factor realizations available at each decision. Stage 3 combines the
# current characteristics with the fixed, train-only loading map.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single_wide"])
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")

boxes = [
    (0.4, "Stage 1", "Fit Γ and realized\nfactor history", COLORS["silver_muted"]),
    (4.4, "Stage 2", "Forecast next\nfactor premium", COLORS["amber_light"]),
    (8.4, "Stage 3", "Map through\ncurrent betas", COLORS["silver"]),
]
for x_pos, stage, detail, facecolor in boxes:
    patch = FancyBboxPatch(
        (x_pos, 0.9),
        3.2,
        1.7,
        boxstyle="round,pad=0.08",
        facecolor=facecolor,
        edgecolor=COLORS["blue"],
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x_pos + 1.6, 1.95, stage, ha="center", weight="semibold", color=COLORS["blue"])
    ax.text(x_pos + 1.6, 1.42, detail, ha="center", fontsize=8.5, color=COLORS["neutral"])
for x_pos in (3.75, 7.75):
    ax.annotate(
        "",
        xy=(x_pos + 0.55, 1.75),
        xytext=(x_pos, 1.75),
        arrowprops={"arrowstyle": "->", "color": COLORS["amber"], "lw": 1.5},
    )
add_message_title(ax, "Conditional factor models separate structure from forecasting")
plt.show()

# %% [markdown]
# ## 2. Alternating least squares
#
# For fixed $\Gamma$, each date is a small cross-sectional least-squares
# problem for $f_{t+1}$. A small ridge term protects the solve when the
# conditional beta matrix is nearly singular.


# %%
def estimate_factor_history(
    returns: np.ndarray,
    characteristics: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Estimate one realized factor vector per return cross-section."""
    n_periods = returns.shape[0]
    n_factors = gamma.shape[1]
    factors = np.empty((n_periods, n_factors))
    ridge = 1e-8 * np.eye(n_factors)
    for period in range(n_periods):
        betas = characteristics[period] @ gamma
        factors[period] = np.linalg.solve(
            betas.T @ betas + ridge,
            betas.T @ returns[period],
        )
    return factors


# %% [markdown]
# For fixed factors, stack the characteristic-factor interactions into one
# pooled regression. The coefficient vector reshapes directly into $\Gamma$.


# %%
def update_gamma(
    returns: np.ndarray,
    characteristics: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Update the characteristic loading map in one pooled regression."""
    n_characteristics = characteristics.shape[2]
    n_factors = factors.shape[1]
    gram = np.zeros((n_characteristics * n_factors,) * 2)
    score = np.zeros(n_characteristics * n_factors)
    for period, factor in enumerate(factors):
        design = np.einsum("nl,k->nlk", characteristics[period], factor).reshape(
            returns.shape[1], -1
        )
        gram += design.T @ design
        score += design.T @ returns[period]
    ridge = 1e-8 * np.eye(gram.shape[0])
    return np.linalg.solve(gram + ridge, score).reshape(n_characteristics, n_factors)


# %% [markdown]
# IPCA is unchanged by invertible rotations of $\Gamma$ and the factors.
# Orthonormalizing the loading map and ordering directions by factor variance
# chooses a stable representation without changing fitted returns.


# %%
def normalize_ipca(
    gamma: np.ndarray,
    factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose an orthonormal, variance-ordered representation."""
    gamma_orthogonal, transform = np.linalg.qr(gamma)
    factor_history = factors @ transform.T
    covariance = np.atleast_2d(np.cov(factor_history, rowvar=False))
    eigenvalues, rotation = np.linalg.eigh(covariance)
    rotation = rotation[:, np.argsort(eigenvalues)[::-1]]
    gamma_normalized = gamma_orthogonal @ rotation
    factors_normalized = factor_history @ rotation
    anchors = np.argmax(np.abs(gamma_normalized), axis=0)
    signs = np.sign(gamma_normalized[anchors, np.arange(gamma.shape[1])])
    signs[signs == 0] = 1
    return gamma_normalized * signs, factors_normalized * signs


# %% [markdown]
# The initializer applies PCA to characteristic-managed returns. ALS then
# alternates the two closed-form updates until both objects stabilize.


# %%
def fit_ipca(
    returns: np.ndarray,
    characteristics: np.ndarray,
    n_factors: int,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Fit IPCA by alternating least squares."""
    managed = np.einsum("tnl,tn->tl", characteristics, returns) / returns.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(managed.T @ managed)
    gamma = eigenvectors[:, np.argsort(eigenvalues)[-n_factors:]]
    previous_factors = np.zeros((returns.shape[0], n_factors))
    converged = False
    for iteration in range(1, max_iter + 1):
        factors = estimate_factor_history(returns, characteristics, gamma)
        updated_gamma = update_gamma(returns, characteristics, factors)
        gamma_delta = np.max(np.abs(updated_gamma - gamma))
        factor_delta = np.max(np.abs(factors - previous_factors))
        gamma, previous_factors = updated_gamma, factors
        if max(gamma_delta, factor_delta) < tolerance:
            converged = True
            break
    factors = estimate_factor_history(returns, characteristics, gamma)
    gamma, factors = normalize_ipca(gamma, factors)
    fitted = np.einsum("tnk,tk->tn", characteristics @ gamma, factors)
    return {
        "gamma": gamma,
        "factors": factors,
        "converged": converged,
        "iterations": iteration,
        "mse": float(np.mean((returns - fitted) ** 2)),
    }


# %% [markdown]
# ## 3. A timing-correct synthetic panel
#
# Characteristics follow persistent AR(1) processes and are standardized
# within each cross-section. The return paired with $z_t$ is generated from
# the independent factor shock at $t+1$, exactly matching the model equation.


# %%
def generate_ipca_panel(seed: int = SEED) -> dict[str, np.ndarray]:
    """Generate lagged characteristics and their next-period returns."""
    rng = np.random.default_rng(seed)
    characteristics = np.empty((N_PERIODS, N_ASSETS, N_CHARACTERISTICS))
    characteristics[0] = rng.normal(size=(N_ASSETS, N_CHARACTERISTICS))
    for period in range(1, N_PERIODS):
        innovation = rng.normal(size=(N_ASSETS, N_CHARACTERISTICS))
        characteristics[period] = 0.8 * characteristics[period - 1] + 0.6 * innovation
    means = characteristics.mean(axis=1, keepdims=True)
    scales = characteristics.std(axis=1, keepdims=True)
    characteristics = (characteristics - means) / scales
    true_gamma, _ = np.linalg.qr(rng.normal(size=(N_CHARACTERISTICS, N_TRUE_FACTORS)))
    factor_scales = np.array([0.040, 0.025, 0.015])
    true_factors = rng.normal(size=(N_PERIODS + 1, N_TRUE_FACTORS)) * factor_scales
    betas = characteristics @ true_gamma
    next_returns = np.einsum("tnk,tk->tn", betas, true_factors[1:])
    next_returns += rng.normal(scale=0.010, size=next_returns.shape)
    return {
        "characteristics": characteristics,
        "next_returns": next_returns,
        "true_gamma": true_gamma,
        "true_factors": true_factors,
    }


# %%
panel = generate_ipca_panel()
train_stop = TRAIN_BOUNDARY - EMBARGO
test_start = TRAIN_BOUNDARY

train_characteristics = panel["characteristics"][:train_stop]
train_returns = panel["next_returns"][:train_stop]
embargo_characteristics = panel["characteristics"][train_stop:test_start]
embargo_returns = panel["next_returns"][train_stop:test_start]
test_characteristics = panel["characteristics"][test_start:]
test_returns = panel["next_returns"][test_start:]

print(
    f"Pairs: train={len(train_returns)}, embargo={len(embargo_returns)}, "
    f"evaluation={len(test_returns)}"
)

# %% [markdown]
# ## 4. Stage 1: recover the loading subspace
#
# Individual columns of $\Gamma$ are not identified: rotations and sign flips
# leave fitted returns unchanged. Principal-angle cosines and the distance
# between projection matrices therefore test the estimable object.

# %%
started = perf_counter()
ipca = fit_ipca(
    train_returns,
    train_characteristics,
    n_factors=N_IPCA_FACTORS,
    max_iter=MAX_ITER,
)
elapsed = perf_counter() - started
print(
    f"ALS: converged={ipca['converged']}, iterations={ipca['iterations']}, "
    f"train MSE={ipca['mse']:.6f}, elapsed={elapsed:.2f}s"
)

# %%
true_basis, _ = np.linalg.qr(panel["true_gamma"])
estimated_basis, _ = np.linalg.qr(ipca["gamma"])
principal_cosines = np.linalg.svd(true_basis.T @ estimated_basis, compute_uv=False)
projector_distance = np.linalg.norm(
    true_basis @ true_basis.T - estimated_basis @ estimated_basis.T,
    ord="fro",
)
alignment, _ = orthogonal_procrustes(estimated_basis, true_basis)
aligned_basis = estimated_basis @ alignment
alignment_rmse = float(np.sqrt(np.mean((true_basis - aligned_basis) ** 2)))

print(
    "Subspace: cosines="
    f"{np.round(principal_cosines, 4).tolist()}, "
    f"projector distance={projector_distance:.4f}, aligned RMSE={alignment_rmse:.4f}"
)

# %% [markdown]
# Both panels below use the same color scale. Similar vertical patterns after
# orthogonal alignment indicate recovery of the loading subspace, not recovery
# of arbitrarily labeled columns.

# %%
gamma_cmap = LinearSegmentedColormap.from_list("ml4t_diverging", ml4t_diverging())
limit = float(np.max(np.abs(np.concatenate([true_basis, aligned_basis], axis=1))))
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE["dual_h_tall"], sharey=True)
for ax, basis, label in zip(
    axes,
    (true_basis, aligned_basis),
    ("Known basis", "Estimated basis after alignment"),
    strict=True,
):
    image = ax.imshow(basis, aspect="auto", cmap=gamma_cmap, vmin=-limit, vmax=limit)
    ax.set_xlabel(f"{label}\nLatent direction")
    ax.set_xticks(range(N_TRUE_FACTORS), [f"F{k + 1}" for k in range(N_TRUE_FACTORS)])
axes[0].set_ylabel("Characteristic index")
fig.colorbar(image, ax=axes, label="Orthonormal loading coefficient", shrink=0.82)
add_message_title(
    axes[0],
    "ALS recovers the known loading subspace",
    subtitle=f"Minimum principal-angle cosine: {principal_cosines.min():.3f}",
)
plt.show()

# %% [markdown]
# ## 5. Stage 2: one-step factor forecasts
#
# Each forecast is computed before the matching evaluation factor is appended
# to history. The embargo-period return is observable at the first evaluation
# decision, so it updates the factor history but never enters the Stage 1 fit.


# %%
def constant_factor_forecast(history: np.ndarray) -> np.ndarray:
    """Forecast each factor with its expanding historical mean."""
    return history.mean(axis=0)


# %%
def ar1_factor_forecast(history: np.ndarray) -> np.ndarray:
    """Fit one AR(1) per factor and forecast from the latest realization."""
    forecasts = np.empty(history.shape[1])
    design = np.column_stack([np.ones(len(history) - 1), history[:-1]])
    for factor in range(history.shape[1]):
        coefficients = np.linalg.lstsq(design[:, [0, factor + 1]], history[1:, factor], rcond=None)[
            0
        ]
        forecasts[factor] = coefficients[0] + coefficients[1] * history[-1, factor]
    return forecasts


# %%
def ewma_factor_forecast(
    history: np.ndarray,
    half_life: int = EWMA_HALF_LIFE,
) -> np.ndarray:
    """Forecast with an exponentially weighted mean of available factors."""
    ages = np.arange(len(history) - 1, -1, -1)
    weights = np.exp(-np.log(2) * ages / half_life)
    weights /= weights.sum()
    return weights @ history


# %%
def walk_forward_factor_forecasts(
    initial_history: np.ndarray,
    realized_factors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Forecast first, then reveal and append each realized factor vector."""
    forecasters = {
        "Expanding mean": constant_factor_forecast,
        "AR(1)": ar1_factor_forecast,
        "EWMA": ewma_factor_forecast,
    }
    predictions = {name: np.empty_like(realized_factors) for name in forecasters}
    history = initial_history.copy()
    for step, realized in enumerate(realized_factors):
        for name, forecaster in forecasters.items():
            predictions[name][step] = forecaster(history)
        history = np.vstack([history, realized])
    return predictions


# %%
embargo_factors = estimate_factor_history(
    embargo_returns,
    embargo_characteristics,
    ipca["gamma"],
)
initial_factor_history = np.vstack([ipca["factors"], embargo_factors])
realized_test_factors = estimate_factor_history(
    test_returns,
    test_characteristics,
    ipca["gamma"],
)
factor_forecasts = walk_forward_factor_forecasts(
    initial_factor_history,
    realized_test_factors,
)

# %% [markdown]
# ## 6. Stage 3: asset forecasts and uncertainty
#
# The mapper uses only current characteristics and the train-only $\Gamma$.
# Forecast $R^2$ uses the economically neutral zero-return forecast as its
# denominator. Cross-sectional IC uncertainty uses a Newey-West standard error;
# the displayed interval is not an independence-based shortcut.


# %%
def map_asset_forecasts(
    characteristics: np.ndarray,
    gamma: np.ndarray,
    factor_forecasts: np.ndarray,
) -> np.ndarray:
    """Map factor forecasts through current conditional betas."""
    return np.einsum("tnk,tk->tn", characteristics @ gamma, factor_forecasts)


# %%
def as_long_panel(values: np.ndarray, value_name: str) -> pl.DataFrame:
    """Convert a period-by-asset matrix to the canonical long schema."""
    timestamps = pl.datetime_range(
        datetime(2000, 1, 1),
        datetime(2000, 1, 1) + pl.duration(days=values.shape[0] - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame(
        {
            "timestamp": np.repeat(timestamps.to_numpy(), values.shape[1]),
            "symbol": np.tile([f"A{i:03d}" for i in range(values.shape[1])], values.shape[0]),
            value_name: values.ravel(),
        }
    )


# %%
def evaluate_asset_forecast(
    name: str,
    predictions: np.ndarray,
    realized_returns: np.ndarray,
    seed: int,
) -> dict[str, float | str]:
    """Compute zero-benchmark error and HAC IC uncertainty."""
    pred_frame = as_long_panel(predictions, "prediction")
    return_frame = as_long_panel(realized_returns, "forward_return")
    ic_frame = cross_sectional_ic_series(
        pred_frame,
        return_frame,
        date_col="timestamp",
        entity_col="symbol",
    )
    uncertainty = compute_ic_uncertainty(
        ic_frame,
        horizon=1,
        n_boot=N_BOOTSTRAP,
        seed=seed,
    )
    mse = float(np.mean((realized_returns - predictions) ** 2))
    zero_mse = float(np.mean(realized_returns**2))
    return {
        "name": name,
        "mse_ratio": mse / zero_mse,
        "r2_zero": 1 - mse / zero_mse,
        "mean_ic": uncertainty["mean_ic"],
        "ci_low": uncertainty["ci_hac_lower"],
        "ci_high": uncertainty["ci_hac_upper"],
        "p_hac": uncertainty["p_hac"],
    }


# %%
forecast_results = []
for index, (name, forecast) in enumerate(factor_forecasts.items()):
    asset_predictions = map_asset_forecasts(
        test_characteristics,
        ipca["gamma"],
        forecast,
    )
    result = evaluate_asset_forecast(name, asset_predictions, test_returns, SEED + index)
    forecast_results.append(result)
    print(
        f"{name}: MSE ratio={result['mse_ratio']:.4f}, "
        f"IC={result['mean_ic']:.4f} "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}], "
        f"HAC p={result['p_hac']:.3f}"
    )

# %% [markdown]
# The zero-return benchmark is deliberately hard to beat when latent premia
# are independent draws with mean zero. The IC intervals show whether any
# apparent cross-sectional ordering survives time-series uncertainty.

# %%
names = [result["name"] for result in forecast_results]
ratios = np.array([result["mse_ratio"] for result in forecast_results])
means = np.array([result["mean_ic"] for result in forecast_results])
lower = np.array([result["ci_low"] for result in forecast_results])
upper = np.array([result["ci_high"] for result in forecast_results])
colors = ml4t_palette(len(names), categorical=True)

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)
axes[0].bar(names, ratios, color=colors)
zero_line(axes[0], at=1.0)
axes[0].set_ylabel("MSE ratio vs zero")
add_message_title(axes[0], "Factor timing does not improve the zero-return benchmark")

errors = np.vstack([means - lower, upper - means])
axes[1].errorbar(names, means, yerr=errors, fmt="o", color=COLORS["blue"], capsize=4)
zero_line(axes[1])
axes[1].set_ylabel("Mean rank IC")
axes[1].set_xlabel("Walk-forward Stage 2 forecaster")
add_message_title(axes[1], "Every HAC interval includes zero")
plt.show()

# %% [markdown]
# ## 7. Factor-count sensitivity
#
# This is a sensitivity analysis, not hyperparameter selection. Every value of
# $K$ is fit on the same training panel and evaluated with the expanding-mean
# forecaster. The evaluation window never chooses the reported factor count.


# %%
def evaluate_factor_count(n_factors: int) -> dict[str, float]:
    """Fit one K and evaluate its expanding-mean forecast."""
    candidate = fit_ipca(
        train_returns,
        train_characteristics,
        n_factors=n_factors,
        max_iter=MAX_ITER,
    )
    embargo_history = estimate_factor_history(
        embargo_returns, embargo_characteristics, candidate["gamma"]
    )
    initial_history = np.vstack([candidate["factors"], embargo_history])
    realized = estimate_factor_history(test_returns, test_characteristics, candidate["gamma"])
    forecasts = walk_forward_factor_forecasts(initial_history, realized)["Expanding mean"]
    predictions = map_asset_forecasts(test_characteristics, candidate["gamma"], forecasts)
    result = evaluate_asset_forecast(
        f"K={n_factors}", predictions, test_returns, SEED + 100 + n_factors
    )
    return {
        "k": float(n_factors),
        "train_mse": float(candidate["mse"]),
        "mse_ratio": float(result["mse_ratio"]),
        "mean_ic": float(result["mean_ic"]),
        "ci_low": float(result["ci_low"]),
        "ci_high": float(result["ci_high"]),
    }


# %%
k_results = [evaluate_factor_count(n_factors) for n_factors in range(1, 7)]
for result in k_results:
    print(
        f"K={int(result['k'])}: train MSE={result['train_mse']:.6f}, "
        f"evaluation MSE ratio={result['mse_ratio']:.4f}, "
        f"IC={result['mean_ic']:.4f} "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )

# %%
k_values = np.array([result["k"] for result in k_results], dtype=int)
train_mse = np.array([result["train_mse"] for result in k_results])
k_ratios = np.array([result["mse_ratio"] for result in k_results])
k_ic = np.array([result["mean_ic"] for result in k_results])
k_low = np.array([result["ci_low"] for result in k_results])
k_high = np.array([result["ci_high"] for result in k_results])

fig, axes = plt.subplots(3, 1, figsize=FIGSIZE["grid_3x2"], sharex=True)
axes[0].plot(k_values, train_mse, marker="o", color=COLORS["blue"])
axes[0].set_ylabel("Training MSE")
add_message_title(axes[0], "Extra factors keep reducing in-sample reconstruction error")
axes[1].plot(k_values, k_ratios, marker="o", color=COLORS["amber"])
zero_line(axes[1], at=1.0)
axes[1].set_ylabel("MSE ratio vs zero")
add_message_title(axes[1], "Lower reconstruction error does not create forecastability")
axes[2].errorbar(
    k_values,
    k_ic,
    yerr=np.vstack([k_ic - k_low, k_high - k_ic]),
    fmt="o-",
    color=COLORS["copper"],
    capsize=3,
)
zero_line(axes[2])
axes[2].set_ylabel("Mean rank IC")
axes[2].set_xlabel("Assumed factor count K")
add_message_title(axes[2], "IC uncertainty spans zero throughout the sensitivity range")
plt.show()

# %% [markdown]
# ## 8. Takeaways
#
# 1. **Timing defines the model.** Characteristics at $t$ are paired with the
#    return and factor realization at $t+1$; shifting a contemporaneously
#    generated return would test a different data-generating process.
# 2. **The loading subspace is identified, not its labels.** Principal angles
#    and projection distance remain valid under rotations and sign changes.
# 3. **Walk-forward means forecast, then reveal.** Each evaluation factor is
#    appended only after its prediction, while the one-period embargo protects
#    the structural fit at the boundary.
# 4. **Good reconstruction is not factor timing.** ALS recovers the synthetic
#    structure, but all Stage 2 forecasters remain indistinguishable from the
#    zero-return baseline when factor premia are independent.
# 5. **Sensitivity is not selection.** The $K$ sweep documents how the result
#    changes across plausible dimensions; it does not tune on the evaluation
#    window.
