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
# # Conditional Autoencoders: Nonlinear Factor Loadings
#
# **Docker image**: `ml4t-gpu`
#
# **Chapter 14: Latent Factor Models**
#
# The conditional autoencoder (CAE) of Gu, Kelly, and Xiu replaces IPCA's
# linear characteristic map with a neural beta network:
#
# $$r_{i,t}=g_\theta(z_{i,t-1})^{\top}f_t+\varepsilon_{i,t}.$$
#
# A factor network extracts $f_t$ from characteristic-managed portfolio
# returns. Stage 1 is trained as a contemporaneous reconstruction model. A
# separate walk-forward adapter then uses information observed at $t$ to
# forecast $f_{t+1}$ and maps that forecast through $g_\theta(z_{i,t})$ to
# predict $r_{i,t+1}$.
#
# **Learning objectives**
#
# - construct managed portfolios with the joint cross-sectional least-squares
#   solve from GKX Equation 16;
# - train a validation-selected CAE ensemble without test-window state;
# - align current characteristics and factors with next-day stock returns; and
# - distinguish reconstruction quality from forward rank IC and squared error.
#
# **Evaluation contract**: liquidity selection, return clipping thresholds,
# network parameters, and early stopping use training/validation data only.
# The test window is used once for a one-trading-day walk-forward demonstration.
# It does not select the ensemble, factor count, or Stage 2 forecaster.
#
# **Universe limitation**: the source includes delisted firms, but it does not
# encode historical index membership or every stock's investability state. The
# notebook demonstrates model mechanics, not an investable historical strategy.
#
# **GPU note**: the production configuration trains five members for at most
# 200 epochs. On an RTX 3090, early stopping usually completes in a few minutes.
#
# **Prerequisites**: [`04_ipca`](04_ipca.ipynb) and
# [`05_rp_pca`](05_rp_pca.ipynb)
#
# **Book sections**: Sections 14.6-14.7, conditional autoencoders and the
# implementation workshop
#
# **Next**: [`07_stochastic_discount_factor`](07_stochastic_discount_factor.ipynb)
# learns the pricing object directly instead of using the three-stage adapter.

# %% [markdown]
# ## 1. Setup

# %%
"""Train a conditional autoencoder and evaluate next-day factor forecasts."""

from copy import deepcopy
from datetime import timedelta
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from ml4t.diagnostic.metrics.uncertainty import compute_ic_uncertainty
from scipy.stats import spearmanr

from data import load_us_equities
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
TOP_N_STOCKS = 500
N_FACTORS = 5
N_EPOCHS = 200
BATCH_SIZE = 10_000
ENSEMBLE_SIZE = 5
LEARNING_RATE = 0.001
LAMBDA_L1 = 0.0001
EARLY_STOPPING_PATIENCE = 30
RETURN_CLIP_QUANTILES = (0.001, 0.999)
EWMA_HALF_LIFE = 60
N_BOOTSTRAP = 2_000
SEED = 42

# %%
set_global_seeds(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(
    f"Device={device}, stocks={TOP_N_STOCKS}, factors={N_FACTORS}, "
    f"epochs={N_EPOCHS}, ensemble={ENSEMBLE_SIZE}"
)

# %% [markdown]
# ## 2. Point-in-time panel construction
#
# Calendar boundaries are fixed before model fitting. The last 180 calendar
# days form the test window, the preceding 185 days form validation, and all
# earlier observations form training.

# %%
equities_raw = load_us_equities(start_date="2005-01-01", end_date="2018-03-27")
sample_end = equities_raw["timestamp"].max()
train_end = sample_end - timedelta(days=365)
valid_end = sample_end - timedelta(days=180)

# %% [markdown]
# Liquidity is estimated only before the validation boundary. This avoids the
# full-sample top-500 selection that would let future volume and price
# information determine the historical universe.

# %%
liquidity = (
    equities_raw.filter(pl.col("timestamp") < train_end)
    .group_by("symbol")
    .agg(
        (pl.col("close") * pl.col("volume")).mean().alias("dollar_volume"),
        pl.len().alias("n_days"),
    )
    .filter(pl.col("n_days") >= 252)
    .sort(["dollar_volume", "symbol"], descending=[True, False])
)
selected_symbols = liquidity.head(TOP_N_STOCKS)["symbol"].to_list()
equities = equities_raw.filter(pl.col("symbol").is_in(selected_symbols))
print(f"Training-defined universe: {len(selected_symbols)} stocks")

# %% [markdown]
# Five price-derived characteristics are available at the close of each day.
# Stage 1 aligns them with the following trading day's return, matching the
# information set used by the forward adapter.

# %%
characteristic_names = ["LME", "Variance", "ST_REV", "r12_2", "AvgRet21"]
feature_panel = (
    equities.sort(["symbol", "timestamp"])
    .with_columns(
        pl.col("adj_close").pct_change().over("symbol").alias("return"),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
    )
    .with_columns(
        pl.when(pl.col("dollar_volume") > 0)
        .then(pl.col("dollar_volume").log())
        .otherwise(None)
        .rolling_mean(21)
        .over("symbol")
        .alias("LME"),
        pl.col("return").rolling_std(21).over("symbol").alias("Variance"),
        pl.col("adj_close").pct_change(21).over("symbol").alias("ST_REV"),
        (
            pl.col("adj_close") / pl.col("adj_close").shift(252)
            - 1
            - pl.col("adj_close").pct_change(21)
        )
        .over("symbol")
        .alias("r12_2"),
        pl.col("return").rolling_mean(21).over("symbol").alias("AvgRet21"),
    )
    .drop_nulls(subset=["return", *characteristic_names])
)

# %% [markdown]
# Cross-sectional ranks are timestamp-local. A global trading-date map then
# joins $z_{i,t-1}$ to $r_{i,t}$ for the same symbol, excluding gaps rather
# than carrying stale characteristics forward. Return clipping limits isolated
# corporate-action artifacts; both thresholds come from training returns.

# %%
rank_expressions = [
    (pl.col(name).rank("average").over("timestamp") / pl.len().over("timestamp") - 0.5).alias(name)
    for name in characteristic_names
]
feature_panel = feature_panel.with_columns(rank_expressions)
available_dates = feature_panel["timestamp"].unique().sort().to_list()
lag_pairs = pl.DataFrame(
    {"feature_timestamp": available_dates[:-1], "timestamp": available_dates[1:]}
)
model_panel = (
    feature_panel.select(
        pl.col("timestamp").alias("feature_timestamp"), "symbol", *characteristic_names
    )
    .join(lag_pairs, on="feature_timestamp", how="inner")
    .join(feature_panel.select("timestamp", "symbol", "return"), on=["timestamp", "symbol"])
    .sort(["timestamp", "symbol"])
)
training_returns_raw = model_panel.filter(pl.col("timestamp") < train_end)["return"]
clip_lower = float(training_returns_raw.quantile(RETURN_CLIP_QUANTILES[0]))
clip_upper = float(training_returns_raw.quantile(RETURN_CLIP_QUANTILES[1]))
feature_panel = feature_panel.with_columns(pl.col("return").clip(clip_lower, clip_upper))
model_panel = model_panel.with_columns(pl.col("return").clip(clip_lower, clip_upper))

# %%
splits = {
    "train": model_panel.filter(pl.col("timestamp") < train_end),
    "valid": model_panel.filter(
        (pl.col("timestamp") >= train_end) & (pl.col("timestamp") < valid_end)
    ),
    "test": model_panel.filter(pl.col("timestamp") >= valid_end),
}
for name, frame in splits.items():
    print(
        f"{name}: observations={frame.height:,}, dates={frame['timestamp'].n_unique()}, "
        f"symbols={frame['symbol'].n_unique()}"
    )
print(f"Training return clip: [{clip_lower:.4f}, {clip_upper:.4f}]")

# %% [markdown]
# ## 3. Joint managed portfolios
#
# For each date, append a market column to the lagged characteristic matrix and
# solve the joint cross-sectional system
#
# $$x_t=(Z_t^{\top}Z_t)^{-1}Z_t^{\top}r_t.$$
#
# A joint least-squares solve is essential: dividing each numerator by its own
# squared-characteristic sum ignores correlation among characteristics.

# %%
portfolio_names = [f"mp_{name}" for name in characteristic_names] + ["mp_market"]


def managed_portfolios(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute the joint characteristic-managed return vector at each timestamp."""
    timestamps = []
    portfolio_rows = []
    for cross_section in frame.sort(["timestamp", "symbol"]).partition_by(
        "timestamp", maintain_order=True
    ):
        characteristics = cross_section.select(characteristic_names).to_numpy()
        design = np.column_stack([characteristics, np.ones(len(cross_section))])
        realized_returns = cross_section["return"].to_numpy()
        coefficients = np.linalg.lstsq(design, realized_returns, rcond=None)[0]
        timestamps.append(cross_section.item(0, "timestamp"))
        portfolio_rows.append(coefficients)
    values = np.asarray(portfolio_rows)
    return pl.DataFrame(
        {"timestamp": timestamps}
        | {name: values[:, index] for index, name in enumerate(portfolio_names)}
    )


# %%
portfolios = {name: managed_portfolios(frame) for name, frame in splits.items()}
model_frames = {
    name: frame.join(portfolios[name], on="timestamp", how="inner").sort(["timestamp", "symbol"])
    for name, frame in splits.items()
}
for name in splits:
    print(f"{name} managed portfolios: {portfolios[name].shape}")

# %% [markdown]
# ## 4. CAE architecture
#
# The beta network maps characteristics to conditional loadings. The factor
# network maps managed portfolios to contemporaneous factor realizations. Their
# row-wise dot product reconstructs the observed return.


# %%
class BetaNetwork(nn.Module):
    """Map characteristics to nonlinear factor loadings."""

    def __init__(self, n_characteristics: int, n_factors: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_characteristics, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, n_factors),
        )

    def forward(self, characteristics: torch.Tensor) -> torch.Tensor:
        return self.network(characteristics)


# %% [markdown]
# The factor network uses the six jointly estimated portfolio returns as
# instruments for the latent realization on each date.


# %%
class FactorNetwork(nn.Module):
    """Map managed portfolios to latent factor realizations."""

    def __init__(self, n_instruments: int, n_factors: int):
        super().__init__()
        self.linear = nn.Linear(n_instruments, n_factors, bias=False)

    def forward(self, portfolios: torch.Tensor) -> torch.Tensor:
        return self.linear(portfolios)


# %% [markdown]
# The complete model combines both subnetworks through the pricing equation's
# row-wise inner product.


# %%
class ConditionalAutoencoder(nn.Module):
    """Join conditional betas and factor realizations through a dot product."""

    def __init__(self, n_characteristics: int, n_instruments: int, n_factors: int):
        super().__init__()
        self.beta_net = BetaNetwork(n_characteristics, n_factors)
        self.factor_net = FactorNetwork(n_instruments, n_factors)

    def forward(self, characteristics: torch.Tensor, portfolios: torch.Tensor) -> torch.Tensor:
        betas = self.beta_net(characteristics)
        factors = self.factor_net(portfolios)
        return (betas * factors).sum(dim=1)


# %% [markdown]
# ## 5. Train a validation-selected ensemble
#
# L1 regularization is normalized by the number of beta-network weights so its
# magnitude does not grow mechanically with architecture size. Early stopping
# stores a deep copy of the best parameters.


# %%
def prepare_tensors(frame: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move one long panel and its aligned portfolios to the selected device."""
    characteristics = torch.tensor(
        frame.select(characteristic_names).to_numpy(), dtype=torch.float32, device=device
    )
    portfolio_values = torch.tensor(
        frame.select(portfolio_names).to_numpy(), dtype=torch.float32, device=device
    )
    realized_returns = torch.tensor(frame["return"].to_numpy(), dtype=torch.float32, device=device)
    return characteristics, portfolio_values, realized_returns


# %% [markdown]
# Normalizing the sparsity penalty makes its scale independent of the number of
# trainable loading parameters.


# %%
def normalized_l1_penalty(model: ConditionalAutoencoder) -> torch.Tensor:
    """Return the mean absolute beta-network weight multiplied by lambda."""
    weights = [
        parameter.reshape(-1)
        for name, parameter in model.beta_net.named_parameters()
        if "weight" in name
    ]
    return LAMBDA_L1 * torch.cat(weights).abs().mean()


# %% [markdown]
# Mini-batch updates shuffle observations but preserve the timestamp-aligned
# managed-portfolio vector attached to every row.


# %%
def train_epoch(
    model: ConditionalAutoencoder,
    optimizer: optim.Optimizer,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    """Run one shuffled reconstruction epoch."""
    characteristics, portfolio_values, returns = tensors
    model.train()
    order = torch.randperm(len(returns), device=device)
    losses = []
    for start in range(0, len(returns), BATCH_SIZE):
        index = order[start : start + BATCH_SIZE]
        optimizer.zero_grad()
        prediction = model(characteristics[index], portfolio_values[index])
        mse = nn.functional.mse_loss(prediction, returns[index])
        loss = mse + normalized_l1_penalty(model)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


# %% [markdown]
# Validation evaluates the unpenalized reconstruction error used for checkpoint
# selection.


# %%
def reconstruction_mse(
    model: ConditionalAutoencoder,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    """Evaluate reconstruction MSE without changing network state."""
    characteristics, portfolio_values, returns = tensors
    model.eval()
    with torch.no_grad():
        prediction = model(characteristics, portfolio_values)
    return float(nn.functional.mse_loss(prediction, returns))


# %% [markdown]
# Each ensemble member starts from a distinct deterministic seed and restores
# the deep-copied state with the lowest validation error.


# %%
def train_single_model(
    member: int,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    valid_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[ConditionalAutoencoder, dict[str, list[float] | int]]:
    """Train one member and restore its minimum-validation-MSE state."""
    torch.manual_seed(SEED + member)
    model = ConditionalAutoencoder(len(characteristic_names), len(portfolio_names), N_FACTORS).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: dict[str, list[float] | int] = {"train_loss": [], "valid_mse": [], "best_epoch": 0}
    best_mse = float("inf")
    best_state = deepcopy(model.state_dict())
    patience = 0
    started = perf_counter()
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, train_tensors)
        valid_mse = reconstruction_mse(model, valid_tensors)
        history["train_loss"].append(train_loss)
        history["valid_mse"].append(valid_mse)
        if valid_mse < best_mse - 1e-9:
            best_mse, best_state, patience = valid_mse, deepcopy(model.state_dict()), 0
            history["best_epoch"] = epoch
        else:
            patience += 1
        if epoch % 20 == 0:
            print(
                f"member={member} epoch={epoch:3d} train={train_loss:.6f} "
                f"valid={valid_mse:.6f} elapsed={perf_counter() - started:.1f}s"
            )
        if patience >= EARLY_STOPPING_PATIENCE:
            break
    model.load_state_dict(best_state)
    print(f"member={member} best_epoch={history['best_epoch']} best_valid_mse={best_mse:.6f}")
    return model, history


# %%
train_tensors = prepare_tensors(model_frames["train"])
valid_tensors = prepare_tensors(model_frames["valid"])
models = []
histories = []
for member in range(1, ENSEMBLE_SIZE + 1):
    model, history = train_single_model(member, train_tensors, valid_tensors)
    models.append(model)
    histories.append(history)

# %% [markdown]
# The first member illustrates optimization without turning wall-clock time
# into a stable claim. The selected epoch comes from validation reconstruction,
# not test IC.

# %%
fig, ax = plt.subplots(figsize=FIGSIZE["single"])
ax.plot(histories[0]["train_loss"], color=COLORS["blue"], label="Training objective")
ax.plot(histories[0]["valid_mse"], color=COLORS["amber"], label="Validation MSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("Return squared error")
ax.legend()
add_message_title(ax, "Validation selects the reconstruction checkpoint before test evaluation")
fig.show()

# %% [markdown]
# ## 6. Construct the next-day evaluation panel
#
# Current close characteristics are joined to the same symbol's return on the
# next test trading date. The last date has no forward label and is excluded.

# %%
test_dates = splits["test"]["timestamp"].unique().sort().to_list()
date_pairs = pl.DataFrame({"timestamp": test_dates[:-1], "target_timestamp": test_dates[1:]})
test_features = feature_panel.filter(pl.col("timestamp") >= valid_end)
targets = test_features.select(
    pl.col("timestamp").alias("target_timestamp"),
    "symbol",
    pl.col("return").alias("forward_return"),
)
forecast_frame = (
    test_features.join(date_pairs, on="timestamp", how="inner")
    .join(targets, on=["target_timestamp", "symbol"], how="inner")
    .sort(["timestamp", "symbol"])
)
decision_dates = forecast_frame["timestamp"].unique().sort().to_list()
date_to_period = {timestamp: period for period, timestamp in enumerate(decision_dates)}
observation_periods = np.array(
    [date_to_period[timestamp] for timestamp in forecast_frame["timestamp"].to_list()]
)
print(
    f"Forward panel: observations={forecast_frame.height:,}, "
    f"decision_dates={len(decision_dates)}, horizon=1 trading day"
)

# %% [markdown]
# ## 7. Walk-forward factor-premium forecasts
#
# All training and validation factors are observable before the test window.
# At each test decision, the current managed-portfolio vector produces $f_t$;
# Stage 2 appends it and forecasts $f_{t+1}$.


# %%
def expanding_mean_forecast(history: np.ndarray) -> np.ndarray:
    """Forecast the next factor vector with its expanding mean."""
    return history.mean(axis=0)


# %% [markdown]
# An AR(1) adapter allows each latent premium to depend on its latest observed
# realization.


# %%
def ar1_forecast(history: np.ndarray) -> np.ndarray:
    """Refit one AR(1) with intercept per factor."""
    forecasts = np.empty(history.shape[1])
    for factor in range(history.shape[1]):
        design = np.column_stack([np.ones(len(history) - 1), history[:-1, factor]])
        coefficients = np.linalg.lstsq(design, history[1:, factor], rcond=None)[0]
        forecasts[factor] = coefficients[0] + coefficients[1] * history[-1, factor]
    return forecasts


# %% [markdown]
# An exponentially weighted mean offers a smoother recency-sensitive adapter
# without fitting coefficients.


# %%
def ewma_forecast(history: np.ndarray, half_life: int = EWMA_HALF_LIFE) -> np.ndarray:
    """Forecast with an exponentially weighted mean of available factors."""
    ages = np.arange(len(history) - 1, -1, -1)
    weights = np.exp(-np.log(2) * ages / half_life)
    weights /= weights.sum()
    return weights @ history


# %% [markdown]
# Walk-forward evaluation refits or updates each adapter using only the factor
# history available at that decision time.


# %%
def walk_forward_forecasts(
    initial_history: np.ndarray,
    current_factors: np.ndarray,
) -> dict[str, np.ndarray]:
    """Append each current factor before forecasting the following one."""
    forecasters = {
        "Expanding mean": expanding_mean_forecast,
        "AR(1)": ar1_forecast,
        "EWMA": ewma_forecast,
    }
    predictions = {name: np.empty_like(current_factors) for name in forecasters}
    history = initial_history.copy()
    for step, current_factor in enumerate(current_factors):
        history = np.vstack([history, current_factor])
        for name, forecaster in forecasters.items():
            predictions[name][step] = forecaster(history)
    return predictions


# %% [markdown]
# Each member retains its own latent coordinate system. We therefore combine
# asset predictions, not raw factors or betas, across the ensemble.


# %%
def portfolio_tensor(frame: pl.DataFrame) -> torch.Tensor:
    """Convert a unique managed-portfolio sequence to a device tensor."""
    return torch.tensor(
        frame.select(portfolio_names).to_numpy(), dtype=torch.float32, device=device
    )


# %%
past_portfolios = pl.concat([portfolios["train"], portfolios["valid"]]).sort("timestamp")
current_portfolios = portfolios["test"].filter(pl.col("timestamp").is_in(decision_dates))
past_portfolio_tensor = portfolio_tensor(past_portfolios)
current_portfolio_tensor = portfolio_tensor(current_portfolios)
forecast_characteristics = torch.tensor(
    forecast_frame.select(characteristic_names).to_numpy(), dtype=torch.float32, device=device
)

member_predictions = {name: [] for name in ("Expanding mean", "AR(1)", "EWMA")}
for model in models:
    model.eval()
    with torch.no_grad():
        initial_factors = model.factor_net(past_portfolio_tensor).cpu().numpy()
        current_factors = model.factor_net(current_portfolio_tensor).cpu().numpy()
        current_betas = model.beta_net(forecast_characteristics).cpu().numpy()
    factor_predictions = walk_forward_forecasts(initial_factors, current_factors)
    for name, period_prediction in factor_predictions.items():
        per_observation = period_prediction[observation_periods]
        member_predictions[name].append((current_betas * per_observation).sum(axis=1))

ensemble_predictions = {
    name: np.mean(predictions, axis=0) for name, predictions in member_predictions.items()
}

# %% [markdown]
# ## 8. Forward metrics with time-series uncertainty
#
# Rank IC is computed within each decision-date cross-section. MSE uses the
# zero-return forecast as its benchmark. HAC intervals reflect serial
# dependence in the daily IC sequence.

# %%
# %% [markdown]
# The evaluator derives a daily cross-sectional IC series before computing HAC
# uncertainty, rather than treating all stock-day rows as independent.


# %%
def evaluate_forward_prediction(
    name: str,
    prediction: np.ndarray,
    seed: int,
) -> dict[str, float | str]:
    """Evaluate one ensemble forecast on the aligned next-day panel."""
    metric_frame = pl.DataFrame(
        {
            "timestamp": forecast_frame["timestamp"],
            "symbol": forecast_frame["symbol"],
            "prediction": prediction,
            "forward_return": forecast_frame["forward_return"],
        }
    )
    ic_frame = cross_sectional_ic_series(
        metric_frame,
        metric_frame,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
        min_obs=20,
    )
    uncertainty = compute_ic_uncertainty(ic_frame, horizon=1, n_boot=N_BOOTSTRAP, seed=seed)
    realized = forecast_frame["forward_return"].to_numpy()
    mse_ratio = float(np.mean((realized - prediction) ** 2) / np.mean(realized**2))
    return {
        "name": name,
        "mse_ratio": mse_ratio,
        "mean_ic": uncertainty["mean_ic"],
        "ci_low": uncertainty["ci_hac_lower"],
        "ci_high": uncertainty["ci_hac_upper"],
        "p_hac": uncertainty["p_hac"],
    }


# %%
forecast_results = []
for index, (name, prediction) in enumerate(ensemble_predictions.items()):
    result = evaluate_forward_prediction(name, prediction, SEED + index)
    forecast_results.append(result)
    print(
        f"{name}: MSE ratio={result['mse_ratio']:.5f}, "
        f"IC={result['mean_ic']:.4f} "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}], "
        f"HAC p={result['p_hac']:.3f}"
    )

# %%
names = [result["name"] for result in forecast_results]
mse_ratios = np.array([result["mse_ratio"] for result in forecast_results])
mean_ics = np.array([result["mean_ic"] for result in forecast_results])
ci_low = np.array([result["ci_low"] for result in forecast_results])
ci_high = np.array([result["ci_high"] for result in forecast_results])
colors = ml4t_palette(len(names), categorical=True)

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["dual_v"], sharex=True)
axes[0].scatter(names, mse_ratios, color=colors, s=55)
zero_line(axes[0], at=1.0)
axes[0].set_ylabel("MSE ratio vs zero")
maximum_mse_deviation = 100 * np.max(np.abs(mse_ratios - 1))
add_message_title(
    axes[0],
    f"All MSE ratios remain within {maximum_mse_deviation:.1f}% of zero",
)
axes[1].errorbar(
    names,
    mean_ics,
    yerr=np.vstack([mean_ics - ci_low, ci_high - mean_ics]),
    fmt="o",
    color=COLORS["blue"],
    capsize=4,
)
zero_line(axes[1])
axes[1].set_ylabel("Mean next-day rank IC")
axes[1].set_xlabel("Walk-forward Stage 2 forecaster")
add_message_title(axes[1], "Every next-day rank-IC interval includes zero")
fig.show()

# %% [markdown]
# The forward test does not support a predictive claim. The expanding mean's
# 0.99968 error ratio is effectively tied with predicting zero. EWMA has the
# largest mean rank IC at 0.0116, but its 95% HAC interval of
# $[-0.0130, 0.0362]$ includes zero. The CAE can reconstruct the panel while
# its simple factor-premium adapters do not deliver statistically resolved
# next-day cross-sectional forecasts.

# %% [markdown]
# ## 9. Ensemble and loading diagnostics
#
# Ensemble averaging is valid at the asset-prediction surface. Averaging raw
# beta columns across members is not: each latent model has its own rotation
# and scale. We therefore compare member prediction ICs and inspect the first
# member's loading map as a representative coordinate system.


# %%
def mean_rank_ic(prediction: np.ndarray) -> float:
    """Return the mean decision-date Spearman IC for one prediction surface."""
    values = []
    realized = forecast_frame["forward_return"].to_numpy()
    for period in range(len(decision_dates)):
        mask = observation_periods == period
        correlation = spearmanr(prediction[mask], realized[mask]).statistic
        if np.isfinite(correlation):
            values.append(correlation)
    return float(np.mean(values))


# %%
member_ics = [mean_rank_ic(prediction) for prediction in member_predictions["Expanding mean"]]
ensemble_ic = mean_rank_ic(ensemble_predictions["Expanding mean"])
print(
    f"Expanding-mean member IC range=[{min(member_ics):.4f}, {max(member_ics):.4f}], "
    f"ensemble IC={ensemble_ic:.4f}"
)

# %%
representative = models[0]
representative.eval()
with torch.no_grad():
    representative_betas = representative.beta_net(forecast_characteristics).cpu().numpy()

characteristic_values = forecast_frame.select(characteristic_names).to_numpy()
loading_correlations = np.empty((len(characteristic_names), N_FACTORS))
for characteristic in range(len(characteristic_names)):
    for factor in range(N_FACTORS):
        loading_correlations[characteristic, factor] = spearmanr(
            characteristic_values[:, characteristic], representative_betas[:, factor]
        ).statistic

loading_min = loading_correlations.min()
loading_max = loading_correlations.max()

# %%
correlation_cmap = LinearSegmentedColormap.from_list("ml4t_diverging", ml4t_diverging())
fig, ax = plt.subplots(figsize=FIGSIZE["single_tall"])
image = ax.imshow(loading_correlations, cmap=correlation_cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(N_FACTORS), [f"F{factor + 1}" for factor in range(N_FACTORS)])
ax.set_yticks(range(len(characteristic_names)), characteristic_names)
ax.set_xlabel("Representative latent factor")
ax.set_ylabel("Current characteristic rank")
for row in range(loading_correlations.shape[0]):
    for column in range(loading_correlations.shape[1]):
        ax.text(column, row, f"{loading_correlations[row, column]:.2f}", ha="center", va="center")
fig.colorbar(image, ax=ax, label="Spearman correlation")
add_message_title(
    ax,
    f"Within-member loading correlations span {loading_min:.2f} to {loading_max:.2f}",
)
fig.show()

# %% [markdown]
# ## 10. Takeaways
#
# 1. **Managed portfolios require a joint solve.** The factor network receives
#    the full cross-sectional least-squares coefficients, not marginal ratios.
# 2. **Structural learned state is pre-test.** Liquidity selection and clipping
#    are training-only; network checkpoints use validation reconstruction only.
# 3. **Reconstruction and forecasting are different tasks.** The CAE learns
#    contemporaneous conditional factors, then a separate adapter forecasts
#    their next realization.
# 4. **Walk-forward timing is explicit.** Current managed returns update the
#    factor history before the next trading day's return is predicted.
# 5. **Ensemble predictions are identified.** Raw latent columns may rotate and
#    rescale across members, so the notebook averages asset predictions and
#    keeps loading diagnostics inside one representative coordinate system.
#
# See Sections 14.6-14.7 for the model derivation. The next notebook,
# [`07_stochastic_discount_factor`](07_stochastic_discount_factor.ipynb),
# learns a pricing kernel instead of forecasting latent factor premia.
