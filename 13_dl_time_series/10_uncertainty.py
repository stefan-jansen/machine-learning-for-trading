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
# # Prediction Uncertainty: MC Dropout and Deep Ensembles
#
# **Docker image**: `ml4t-gpu`
#
# This notebook implements two complementary approaches to uncertainty estimation:
# **MC Dropout** (Gal and Ghahramani, 2016) and **Deep Ensembles** (Lakshminarayanan
# et al., 2017). Both produce candidate uncertainty estimates around point
# predictions; the empirical work below shows those raw estimates require
# calibration before they can be used in position sizing.
#
# **Learning Objectives**:
# - Implement MC Dropout by keeping dropout active at inference time
# - Train Deep Ensembles with independently initialized models
# - Compare calibration: does predicted uncertainty track actual error?
# - Apply **split-conformal calibration** to convert raw spreads into intervals
#   with controlled empirical coverage
# - Decompose epistemic vs aleatoric uncertainty
#
# **Book Reference**: Chapter 13, Section 13.8 (Prediction Uncertainty)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Prediction Uncertainty - implement MC Dropout and Deep Ensembles for confidence estimation."""

import warnings

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from dl_sequences import create_sequences_multi_asset, load_dl_dataset
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from plotly.subplots import make_subplots
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.reproducibility import set_global_seeds
from utils.style import COLORS  # activates the ml4t Plotly template on import

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
MAX_SYMBOLS = 0
LOOKBACK = 60
HIDDEN_SIZE = 32
DROPOUT = 0.2
MC_SAMPLES = 50
N_ENSEMBLE = 5
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.001

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

set_global_seeds(SEED)
# Extra determinism flags for CUDA training in this notebook
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# ## Data Loading
#
# Both methods share identical data: ETF features and forward return labels from
# the case study pipeline. Loading once eliminates the duplication that separate
# notebooks would require.

# %%
mds = load_dl_dataset("etfs", max_symbols=MAX_SYMBOLS)

FEATURE_COLS = mds.feature_names[:8]
TARGET_COL = mds.label_col

df = mds.dataset.drop_nulls(subset=FEATURE_COLS + [TARGET_COL])
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")
print(f"Rows after dropna: {len(df):,}")

# %% [markdown]
# ## Sequence Creation and Temporal Split

# %%
X, y, timestamps, symbols = create_sequences_multi_asset(
    df,
    FEATURE_COLS,
    TARGET_COL,
    LOOKBACK,
    timestamp_col=mds.date_col,
    symbol_col=mds.entity_cols[0],
)

# Replace NaN/inf with zero - acceptable here because features are already
# standardized returns where NaN typically indicates missing data at series
# boundaries. Forward-fill is an alternative but risks lookahead in panel data.
n_nan = np.isnan(X).sum() + np.isinf(X).sum()
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = np.nan_to_num(y, nan=0.0).astype(np.float32)

print(f"Sequences: {X.shape[0]:,}, shape: {X.shape}")
print(f"NaN/inf values replaced with 0: {n_nan:,}")

# %%
# Date-based 60/20/20 temporal split
unique_dates = np.sort(np.unique(timestamps))
train_end_date = unique_dates[int(len(unique_dates) * 0.6)]
val_end_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = timestamps < train_end_date
val_mask = (timestamps >= train_end_date) & (timestamps < val_end_date)
test_mask = timestamps >= val_end_date

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates, test_symbols = timestamps[test_mask], symbols[test_mask]

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# %% [markdown]
# **Cross-sectional IC helper**: every method below scores predictions with the
# same daily Spearman correlation between forecast and forward return, averaged
# over the test window. Defining it once keeps the comparison consistent.


# %%
def cross_sectional_ic_mean(y_true, y_pred, dates, syms):
    pred_df = pl.DataFrame({"timestamp": dates, "symbol": syms, "prediction": y_pred})
    ret_df = pl.DataFrame({"timestamp": dates, "symbol": syms, "forward_return": y_true})
    ic_per_date = cross_sectional_ic_series(
        pred_df,
        ret_df,
        pred_col="prediction",
        ret_col="forward_return",
        date_col="timestamp",
        entity_col="symbol",
    )
    ic_clean = ic_per_date.drop_nulls("ic")
    return float(ic_clean["ic"].mean()) if ic_clean.height else float("nan")


# %% [markdown]
# ## Ridge Baseline
#
# Both uncertainty methods need a point-prediction baseline for comparison.
# We train Ridge once and reuse it throughout.

# %%
X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_ridge = ridge.predict(X_test_scaled)

ridge_mse = np.mean((y_ridge - y_test) ** 2)
ridge_ic = cross_sectional_ic_mean(y_test, y_ridge, test_dates, test_symbols)

print(f"Ridge baseline - MSE: {ridge_mse:.6f}, IC: {ridge_ic:+.4f}")

# %% [markdown]
# ---
# ## Part A: Monte Carlo Dropout
#
# MC Dropout treats dropout as approximate Bayesian inference. By keeping dropout
# **active at test time** and running multiple forward passes, we get a distribution
# of predictions whose spread estimates model uncertainty.
#
# The key insight from Gal and Ghahramani (2016): a neural network with dropout
# applied before every weight layer is mathematically equivalent to an approximation
# of a deep Gaussian process.

# %% [markdown]
# ### LSTM with MC Dropout
#
# This model has dropout in two places:
# 1. Between LSTM layers (via the `dropout` parameter)
# 2. Before the output head (explicit `nn.Dropout`)
#
# At inference time, we call `model.train()` to keep both dropout layers active,
# then run `MC_SAMPLES` forward passes to generate a prediction distribution.


# %%
class LSTMWithDropout(nn.Module):
    """LSTM with dropout that stays active at inference for MC estimation."""

    def __init__(self, input_size, hidden_size=32, n_layers=2, dropout=0.2):
        super().__init__()
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :]).squeeze(-1)


# %% [markdown]
# ### Training


# %%
def train_lstm(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_v: np.ndarray,
    y_v: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
) -> float:
    """Train LSTM with mini-batch SGD and early stopping. Returns best val loss."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_v_t = torch.FloatTensor(X_v).to(DEVICE)
    y_v_t = torch.FloatTensor(y_v).to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr_t))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            optimizer.zero_grad()
            preds = model(X_tr_t[batch_idx])
            loss = criterion(preds, y_tr_t[batch_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = torch.cat(
                [model(X_v_t[i : i + batch_size]) for i in range(0, len(X_v_t), batch_size)]
            )
            val_loss = criterion(val_preds, y_v_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


# %%
mc_model = LSTMWithDropout(
    input_size=len(FEATURE_COLS),
    hidden_size=HIDDEN_SIZE,
    n_layers=2,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in mc_model.parameters())
print(f"MC Dropout LSTM parameters: {n_params:,}")

mc_val_loss = train_lstm(mc_model, X_train, y_train, X_val, y_val, EPOCHS, LR, BATCH_SIZE)
print(f"Best validation loss: {mc_val_loss:.6f}")

# %% [markdown]
# ### MC Inference
#
# The critical step: switch to `model.train()` to keep dropout active, then
# run `MC_SAMPLES` stochastic forward passes. Each pass samples a different
# subset of neurons, approximating a draw from the posterior over model weights.

# %%
X_test_t = torch.FloatTensor(X_test).to(DEVICE)

mc_model.train()  # Keep dropout active
mc_preds = np.zeros((MC_SAMPLES, len(X_test)))

with torch.no_grad():
    for i in range(MC_SAMPLES):
        batches = [
            mc_model(X_test_t[j : j + BATCH_SIZE]).cpu().numpy()
            for j in range(0, len(X_test_t), BATCH_SIZE)
        ]
        mc_preds[i] = np.concatenate(batches)

mc_mean = mc_preds.mean(axis=0)
mc_std = mc_preds.std(axis=0)

mc_mse = np.mean((mc_mean - y_test) ** 2)
mc_ic = cross_sectional_ic_mean(y_test, mc_mean, test_dates, test_symbols)

print(f"MC Dropout ({MC_SAMPLES} samples) - MSE: {mc_mse:.6f}, IC: {mc_ic:+.4f}")
print(f"Mean uncertainty (std): {mc_std.mean():.6f}")
print(f"Ridge baseline IC for reference: {ridge_ic:+.4f}")

# %% [markdown]
# **Interpretation**: averaging $T$ stochastic forward passes gives the MC
# Dropout point estimate above. With only two LSTM layers at 0.2 dropout, the
# stochastic passes produce highly correlated outputs and the spread of the
# resulting distribution is small in absolute terms. Whether the MC mean beats
# Ridge here is rerun-sensitive on this 8-feature ETF panel (see catalog
# `rerun_history`); what is stable is that the dropout-spread magnitude is far
# smaller than the scale of 21-day forward returns. Both observations matter
# for the calibration check below.

# %% [markdown]
# ### Calibration Helper
#
# Both MC Dropout and Deep Ensembles need the same calibration diagnostic:
# bin predictions by uncertainty quartile and check whether error tracks
# uncertainty. We extract this into a reusable function.


# %%
def compute_calibration_table(std: np.ndarray, abs_error: np.ndarray) -> tuple[pl.DataFrame, float]:
    """Compute uncertainty calibration by quartile. Returns (table, Spearman correlation)."""
    quartile_bounds = np.percentile(std, [25, 50, 75])
    bins = np.digitize(std, quartile_bounds)
    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]

    rows = []
    for i in range(4):
        mask = bins == i
        if mask.sum() > 0:
            rows.append(
                {
                    "Quartile": labels[i],
                    "Mean Std": round(float(std[mask].mean()), 6),
                    "Mean |Error|": round(float(abs_error[mask].mean()), 6),
                    "Samples": int(mask.sum()),
                }
            )

    corr, _ = spearmanr(std, abs_error)
    table = pl.DataFrame(rows)
    return table, float(corr)


# %% [markdown]
# ### Calibration Analysis
#
# Well-calibrated uncertainty means: when the model says "I'm uncertain about
# this prediction," the actual error should be proportionally larger. We bin
# predictions by uncertainty quartile and check whether error tracks uncertainty.

# %%
mc_abs_error = np.abs(mc_mean - y_test)
mc_cal_table, unc_err_corr_mc = compute_calibration_table(mc_std, mc_abs_error)

print(f"MC Dropout - Uncertainty-Error correlation (Spearman): {unc_err_corr_mc:+.3f}")
mc_cal_table

# %% [markdown]
# **Interpretation**: the Spearman correlation printed above is the headline
# calibration diagnostic - it asks whether the model's predicted std orders
# absolute errors. The quartile table makes the same question concrete: does
# mean |error| rise monotonically as you move from the low-std quartile (Q1) to
# the high-std quartile (Q4)? With dropout active on only two LSTM layers, the
# std spread is small in absolute terms; a near-zero correlation here means
# that small spread carries little information about which predictions will be
# wrong. The Q1→Q4 |error| gradient in the table is the visual version of the
# same diagnostic.

# %% [markdown]
# ### MC Dropout Visualization

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Uncertainty vs Absolute Error",
        "Confidence Interval (100-sample slice)",
    ),
    horizontal_spacing=0.15,
)

# Scatter: uncertainty vs error (subsample for readability)
n_scatter = min(2000, len(mc_std))
idx = np.random.choice(len(mc_std), n_scatter, replace=False)
fig.add_trace(
    go.Scattergl(
        x=mc_std[idx],
        y=mc_abs_error[idx],
        mode="markers",
        marker=dict(size=2, opacity=0.3),
        name="Predictions",
    ),
    row=1,
    col=1,
)
fig.update_xaxes(title_text="Predicted Uncertainty (std)", row=1, col=1)
fig.update_yaxes(title_text="Absolute Error", row=1, col=1)

# Time slice: 100 test points with confidence band
slice_idx = slice(0, 100)
x_range = np.arange(100)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=mc_mean[slice_idx],
        mode="lines",
        name="MC Mean",
        line=dict(color=COLORS["blue"]),
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.concatenate([x_range, x_range[::-1]]),
        y=np.concatenate(
            [
                mc_mean[slice_idx] + 2 * mc_std[slice_idx],
                (mc_mean[slice_idx] - 2 * mc_std[slice_idx])[::-1],
            ]
        ),
        fill="toself",
        fillcolor=COLORS["blue_light"],
        line=dict(width=0),
        name="95% CI",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_test[slice_idx],
        mode="markers",
        marker=dict(size=3, color=COLORS["copper"]),
        name="Actual",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="Test Sample Index", row=1, col=2)
fig.update_yaxes(title_text="Forward Return", row=1, col=2)
fig.update_layout(
    width=900,
    height=400,
    title_text="MC Dropout's spread is tiny relative to the scale of forward returns",
    showlegend=True,
)
fig.show()

# %% [markdown]
# ---
# ## Part B: Deep Ensembles
#
# Deep Ensembles train $M$ independently initialized models and use their
# disagreement as an uncertainty estimate. Unlike MC Dropout (which approximates
# weight uncertainty within a single model), ensembles capture **functional
# diversity** arising from different loss landscape minima.

# %% [markdown]
# ### Ensemble Member Architecture
#
# Each member is a simpler LSTM than the MC Dropout version: dropout only within
# the LSTM layers (no extra dropout head). The diversity comes from random
# initialization, not from stochastic regularization.


# %%
class LSTMRegressor(nn.Module):
    """Standard LSTM regressor for ensemble members."""

    def __init__(self, input_size, hidden_size=32, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :]).squeeze(-1)


# %% [markdown]
# ### Training Ensemble Members
#
# Each member receives a different random seed for weight initialization.
# The training procedure is identical; only the starting point differs.


# %%
def train_member(
    seed: int,
    member_id: int,
) -> nn.Module:
    """Train a single ensemble member with the given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMRegressor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        n_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    val_loss = train_lstm(model, X_train, y_train, X_val, y_val, EPOCHS, LR, BATCH_SIZE)
    print(f"  Member {member_id}: val_loss={val_loss:.6f}")
    return model


# %%
print(f"Training {N_ENSEMBLE} ensemble members...")
ensemble = []
for m in range(N_ENSEMBLE):
    seed = SEED + m * 1000
    member = train_member(seed, m)
    ensemble.append(member)

print(f"\nTrained {len(ensemble)} members successfully")

# %% [markdown]
# ### Ensemble Predictions
#
# Each member produces an independent point prediction. The ensemble mean is
# the forecast; the ensemble standard deviation is the uncertainty estimate.

# %%
member_preds = np.zeros((N_ENSEMBLE, len(X_test)))

for i, model in enumerate(ensemble):
    model.eval()
    with torch.no_grad():
        batches = [
            model(X_test_t[j : j + BATCH_SIZE]).cpu().numpy()
            for j in range(0, len(X_test_t), BATCH_SIZE)
        ]
        member_preds[i] = np.concatenate(batches)

ens_mean = member_preds.mean(axis=0)
ens_std = member_preds.std(axis=0)

ens_mse = np.mean((ens_mean - y_test) ** 2)
ens_ic = cross_sectional_ic_mean(y_test, ens_mean, test_dates, test_symbols)

print(f"Deep Ensemble ({N_ENSEMBLE} members) - MSE: {ens_mse:.6f}, IC: {ens_ic:+.4f}")
print(f"Mean uncertainty (std): {ens_std.mean():.6f}")

# Report individual member ICs for context
for i in range(N_ENSEMBLE):
    m_ic = cross_sectional_ic_mean(y_test, member_preds[i], test_dates, test_symbols)
    print(f"  Member {i} IC: {m_ic:+.4f}")

# %% [markdown]
# **Interpretation**: The ensemble mean typically outperforms individual members
# through error averaging - the diversity among members cancels correlated
# errors. The spread of member-level ICs reflects the functional diversity that
# makes ensemble uncertainty informative: wider spread means members learned
# genuinely different functions from the same data.

# %% [markdown]
# ### Calibration Analysis

# %%
ens_abs_error = np.abs(ens_mean - y_test)
ens_cal_table, unc_err_corr_ens = compute_calibration_table(ens_std, ens_abs_error)

print(f"Deep Ensemble - Uncertainty-Error correlation (Spearman): {unc_err_corr_ens:+.3f}")
print(f"  vs MC Dropout above: {unc_err_corr_mc:+.3f}")
ens_cal_table

# %% [markdown]
# **Interpretation**: compare the ensemble correlation above against the MC
# Dropout figure. Each ensemble member converges to a genuinely different local
# optimum, so ensemble disagreement captures more functional diversity than the
# subnetwork sampling inside a single MC Dropout model. On this run the
# ensemble's correlation is positive while MC Dropout's is near zero - the
# expected ordering - but neither is large enough to be a strong calibration
# signal on its own. The split-conformal section below converts these raw
# spreads into intervals with controlled coverage.

# %% [markdown]
# ### Epistemic vs Aleatoric Decomposition
#
# Deep ensembles naturally decompose total predictive uncertainty into two components:
#
# $$\underbrace{\text{Var}[\hat{y}]}_{\text{total}} = \underbrace{\text{Var}_{\theta}[\mathbb{E}[y \mid x, \theta]]}_{\text{epistemic}} + \underbrace{\mathbb{E}_{\theta}[\text{Var}[y \mid x, \theta]]}_{\text{aleatoric}}$$
#
# - **Epistemic uncertainty** (model uncertainty): variance of the ensemble means.
#   Reducible with more data -- members disagree because they learned different
#   functions from finite data.
# - **Aleatoric uncertainty** (data uncertainty): average variance within each
#   member. Irreducible -- inherent noise in the data generating process.
#
# For regression with MSE loss (no per-member variance head), we approximate
# epistemic uncertainty as ensemble disagreement and note that aleatoric
# uncertainty would require heteroscedastic outputs.

# %%
# Epistemic: variance across member means (what we can estimate)
epistemic = ens_std**2

# Total variance of predictions
total_var = member_preds.var(axis=0)

print(f"Mean epistemic variance: {epistemic.mean():.8f}")
print(f"Mean total variance:     {total_var.mean():.8f}")

# %% [markdown]
# **Note**: Without heteroscedastic output heads (where each member predicts both
# $\mu$ and $\sigma^2$), total variance equals epistemic variance. A production
# model would add per-member variance heads to separate aleatoric noise - the
# irreducible uncertainty inherent in the data generating process. See section
# 13.8 for the full variance decomposition equation.

# %% [markdown]
# ### Ensemble Member Agreement Visualization

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Member Predictions (50-sample slice)",
        "Ensemble Uncertainty Distribution",
    ),
    horizontal_spacing=0.15,
)

# Member predictions slice
slice_range = np.arange(50)
for i in range(N_ENSEMBLE):
    fig.add_trace(
        go.Scatter(
            x=slice_range,
            y=member_preds[i, :50],
            mode="lines",
            name=f"Member {i}",
            opacity=0.5,
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=slice_range,
        y=ens_mean[:50],
        mode="lines",
        name="Ensemble Mean",
        line=dict(color=COLORS["slate"], width=2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=slice_range,
        y=y_test[:50],
        mode="markers",
        name="Actual",
        marker=dict(size=4, color=COLORS["copper"]),
    ),
    row=1,
    col=1,
)
fig.update_xaxes(title_text="Test Sample Index", row=1, col=1)
fig.update_yaxes(title_text="Forward Return", row=1, col=1)

# Uncertainty distribution
fig.add_trace(
    go.Histogram(
        x=ens_std, nbinsx=40, name="Ensemble Std", marker_color=COLORS["blue"], showlegend=False
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="Ensemble Std Dev", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_layout(
    width=900, height=400, title_text="Ensemble members disagree, giving a spread of uncertainties"
)
fig.show()

# %% [markdown]
# ---
# ## Part C: MC Dropout vs Deep Ensembles
#
# We now compare both uncertainty methods head-to-head on the same test set,
# alongside the Ridge baseline which provides no uncertainty estimate.

# %%
results = pl.DataFrame(
    {
        "Method": ["Ridge (baseline)", "MC Dropout", "Deep Ensemble"],
        "MSE": [ridge_mse, mc_mse, ens_mse],
        "Spearman IC": [ridge_ic, mc_ic, ens_ic],
        "Mean Uncertainty": [float("nan"), mc_std.mean(), ens_std.mean()],
        "Unc-Error Corr": [float("nan"), unc_err_corr_mc, unc_err_corr_ens],
        "Forward Passes": [1, MC_SAMPLES, N_ENSEMBLE],
    }
)
results

# %% [markdown]
# ### Side-by-Side Calibration

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "MC Dropout: Uncertainty vs Error",
        "Deep Ensemble: Uncertainty vs Error",
    ),
    horizontal_spacing=0.15,
)

n_scatter = min(2000, len(mc_std))
idx = np.random.choice(len(mc_std), n_scatter, replace=False)

fig.add_trace(
    go.Scattergl(
        x=mc_std[idx],
        y=mc_abs_error[idx],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color=COLORS["blue"]),
        name="MC Dropout",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scattergl(
        x=ens_std[idx],
        y=ens_abs_error[idx],
        mode="markers",
        marker=dict(size=2, opacity=0.3, color=COLORS["amber"]),
        name="Deep Ensemble",
    ),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Predicted Uncertainty (std)", row=1, col=1)
fig.update_xaxes(title_text="Predicted Uncertainty (std)", row=1, col=2)
fig.update_yaxes(title_text="Absolute Error", row=1, col=1)
fig.update_yaxes(title_text="Absolute Error", row=1, col=2)
fig.update_layout(
    width=900,
    height=400,
    title_text="Both raw spreads track error weakly; the ensemble's is the wider, more informative one",
)
fig.show()

# %% [markdown]
# ### Coverage Probability
#
# A complementary calibration diagnostic: what fraction of actual values fall
# within the predicted confidence intervals? We compute empirical coverage at
# 50%, 80%, and 95% nominal levels for both methods using the Gaussian
# assumption ($\mu \pm z_\alpha \cdot \sigma$).

# %%
nominal_levels = [0.50, 0.80, 0.95]
coverage_rows = []

for level in nominal_levels:
    z = norm.ppf(0.5 + level / 2)

    # MC Dropout coverage
    mc_lower = mc_mean - z * mc_std
    mc_upper = mc_mean + z * mc_std
    mc_cov = np.mean((y_test >= mc_lower) & (y_test <= mc_upper))

    # Deep Ensemble coverage
    ens_lower = ens_mean - z * ens_std
    ens_upper = ens_mean + z * ens_std
    ens_cov = np.mean((y_test >= ens_lower) & (y_test <= ens_upper))

    coverage_rows.append(
        {
            "Nominal": f"{level:.0%}",
            "MC Dropout": f"{mc_cov:.1%}",
            "Deep Ensemble": f"{ens_cov:.1%}",
        }
    )

coverage_df = pl.DataFrame(coverage_rows)
coverage_df

# %% [markdown]
# **Interpretation**: both methods severely underestimate uncertainty under the
# Gaussian assumption. The dropout-spread is tiny in absolute terms, so MC
# Dropout's intervals are essentially zero-width and capture almost none of
# the actual returns. The ensemble does better - its members genuinely
# disagree - but the resulting intervals are still far too narrow at every
# nominal level. The pattern is consistent across reruns: raw LSTM
# uncertainty estimates are **not usable for risk management without
# post-hoc calibration**. The next section uses split-conformal prediction to
# convert these raw spreads into intervals with controlled empirical coverage.

# %% [markdown]
# ### Split-Conformal Calibration
#
# Split-conformal prediction (Vovk et al.) turns any point predictor into a
# distribution-free interval estimator: under exchangeability of validation
# and test residuals, the resulting test intervals have at least $1-\alpha$
# marginal coverage in finite samples.
#
# We implement two variants on the held-out validation set:
#
# - **Plain conformal**: $q_\alpha = \mathrm{Quantile}_{1-\alpha}\big(|y_{\text{val}} - \hat\mu_{\text{val}}|\big)$,
#   test interval $[\hat\mu_{\text{test}} - q_\alpha,\ \hat\mu_{\text{test}} + q_\alpha]$.
#   Width is constant across test points.
# - **Normalized (locally adaptive) conformal**: $q_\alpha = \mathrm{Quantile}_{1-\alpha}\big(|y_{\text{val}} - \hat\mu_{\text{val}}| / \hat\sigma_{\text{val}}\big)$,
#   test interval $[\hat\mu_{\text{test}} - q_\alpha\,\hat\sigma_{\text{test}},\ \hat\mu_{\text{test}} + q_\alpha\,\hat\sigma_{\text{test}}]$.
#   Width scales with the model's own uncertainty estimate, so well-calibrated
#   $\hat\sigma$ yields tighter intervals where the model is confident.
#
# We need validation predictions from MC Dropout and the ensemble. We re-run
# inference on the val split.
#
# **Caveat - calibration set overlaps the early-stopping set.** The conformal
# quantiles below are computed on the same `X_val` split that `train_lstm` used
# for early stopping. The models have therefore been tuned to minimize error on
# exactly these points, so the residuals $|y_{\text{val}} - \hat\mu_{\text{val}}|$
# are biased *downward* and the resulting intervals are mildly optimistic - they
# slightly under-cover relative to a strict split-conformal guarantee, which
# requires a calibration set exchangeable with the test set and untouched during
# training. A production pipeline carves a third, dedicated calibration split
# before training; we reuse the validation split here to keep the notebook to a
# single train/val/test partition. Read the coverage numbers as an upper bound
# on real-world calibration quality, not a finite-sample guarantee.

# %%
X_val_t = torch.FloatTensor(X_val).to(DEVICE)

mc_model.train()  # keep dropout active for MC inference
mc_preds_val = np.zeros((MC_SAMPLES, len(X_val)))

with torch.no_grad():
    for i in range(MC_SAMPLES):
        batches = [
            mc_model(X_val_t[j : j + BATCH_SIZE]).cpu().numpy()
            for j in range(0, len(X_val_t), BATCH_SIZE)
        ]
        mc_preds_val[i] = np.concatenate(batches)

mc_mean_val = mc_preds_val.mean(axis=0)
mc_std_val = mc_preds_val.std(axis=0)
mc_resid_val = np.abs(mc_mean_val - y_val)

member_preds_val = np.zeros((N_ENSEMBLE, len(X_val)))
for i, model in enumerate(ensemble):
    model.eval()
    with torch.no_grad():
        batches = [
            model(X_val_t[j : j + BATCH_SIZE]).cpu().numpy()
            for j in range(0, len(X_val_t), BATCH_SIZE)
        ]
        member_preds_val[i] = np.concatenate(batches)

ens_mean_val = member_preds_val.mean(axis=0)
ens_std_val = member_preds_val.std(axis=0)
ens_resid_val = np.abs(ens_mean_val - y_val)

print(
    f"Val residuals - MC: mean |error| = {mc_resid_val.mean():.5f}, "
    f"Ensemble: mean |error| = {ens_resid_val.mean():.5f}"
)


# %% [markdown]
# #### Calibration: plain and normalized conformal quantiles
#
# `eps` guards the normalized variant against division by zero where the model
# reports near-zero predicted std on the validation set.


# %%
def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample (1-α)·(n+1)/n quantile of |y − μ| - split-conformal width."""
    n = len(residuals)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(k, n)
    return float(np.sort(residuals)[k - 1])


EPS = 1e-8
nominal_levels_conf = [0.50, 0.80, 0.95]
conf_rows = []

for level in nominal_levels_conf:
    alpha = 1 - level

    # Plain conformal: constant-width intervals
    q_mc_plain = conformal_quantile(mc_resid_val, alpha)
    q_ens_plain = conformal_quantile(ens_resid_val, alpha)

    mc_cov_plain = np.mean((y_test >= mc_mean - q_mc_plain) & (y_test <= mc_mean + q_mc_plain))
    ens_cov_plain = np.mean((y_test >= ens_mean - q_ens_plain) & (y_test <= ens_mean + q_ens_plain))

    # Normalized conformal: width scales with predicted std
    q_mc_norm = conformal_quantile(mc_resid_val / (mc_std_val + EPS), alpha)
    q_ens_norm = conformal_quantile(ens_resid_val / (ens_std_val + EPS), alpha)

    mc_lo = mc_mean - q_mc_norm * mc_std
    mc_hi = mc_mean + q_mc_norm * mc_std
    ens_lo = ens_mean - q_ens_norm * ens_std
    ens_hi = ens_mean + q_ens_norm * ens_std

    mc_cov_norm = np.mean((y_test >= mc_lo) & (y_test <= mc_hi))
    ens_cov_norm = np.mean((y_test >= ens_lo) & (y_test <= ens_hi))

    mc_width_norm = float((mc_hi - mc_lo).mean())
    ens_width_norm = float((ens_hi - ens_lo).mean())

    conf_rows.append(
        {
            "Nominal": f"{level:.0%}",
            "MC plain cov": f"{mc_cov_plain:.1%}",
            "MC plain width": f"{2 * q_mc_plain:.5f}",
            "MC norm cov": f"{mc_cov_norm:.1%}",
            "MC norm width": f"{mc_width_norm:.5f}",
            "Ens plain cov": f"{ens_cov_plain:.1%}",
            "Ens plain width": f"{2 * q_ens_plain:.5f}",
            "Ens norm cov": f"{ens_cov_norm:.1%}",
            "Ens norm width": f"{ens_width_norm:.5f}",
        }
    )

conformal_df = pl.DataFrame(conf_rows)
conformal_df

# %% [markdown]
# **Interpretation**: split-conformal calibration **restores coverage**. The plain
# variant reaches its nominal rate at every level - the raw Gaussian intervals were
# under-covered by an order of magnitude - and it assigns the same width to every
# test point. The normalized variant scales width by the model's own $\hat\sigma$,
# which pays off only when $\hat\sigma$ carries real, well-conditioned information.
# For the Deep Ensemble, whose member disagreement gives a genuine spread of predicted
# std, normalized conformal holds coverage near nominal at roughly the plain average
# width while adapting locally. For MC Dropout it is a cautionary case: its predicted
# std collapses toward zero, so the normalized variant has almost nothing to adapt to,
# and dividing validation residuals by a near-degenerate $\hat\sigma$ is numerically
# fragile - the normalized quantile is dominated by the smallest std values, so the MC
# normalized width is unstable from run to run (sometimes close to the plain width,
# sometimes inflated by orders of magnitude when a single validation std lands near
# zero). Either way it buys nothing over plain conformal, whose coverage is already at
# nominal. The lesson is that normalized (locally adaptive) conformal needs a
# non-degenerate uncertainty estimate; when the base spread is uninformative, plain
# conformal is the safer choice.
#
# The marginal coverage guarantee is distribution-free under exchangeability
# (Vovk et al. 2005) - the only assumption is that validation and test residuals
# come from the same distribution, which is the same assumption underlying any
# held-out evaluation.

# %% [markdown]
# ### Uncertainty-Based Prediction Filtering
#
# The practical payoff: does removing the most uncertain predictions improve
# forecast quality? We exclude the highest-uncertainty quartile and compare IC
# on the filtered subset against the full test set.

# %%
q75_mc = np.percentile(mc_std, 75)
q75_ens = np.percentile(ens_std, 75)

mc_keep = mc_std <= q75_mc
ens_keep = ens_std <= q75_ens

mc_ic_full = cross_sectional_ic_mean(y_test, mc_mean, test_dates, test_symbols)
mc_ic_filtered = cross_sectional_ic_mean(
    y_test[mc_keep], mc_mean[mc_keep], test_dates[mc_keep], test_symbols[mc_keep]
)

ens_ic_full = cross_sectional_ic_mean(y_test, ens_mean, test_dates, test_symbols)
ens_ic_filtered = cross_sectional_ic_mean(
    y_test[ens_keep], ens_mean[ens_keep], test_dates[ens_keep], test_symbols[ens_keep]
)

filter_rows = [
    {
        "Method": "MC Dropout",
        "IC (full)": f"{mc_ic_full:+.4f}",
        "IC (filtered Q1-Q3)": f"{mc_ic_filtered:+.4f}",
        "Samples removed": f"{(~mc_keep).sum():,}",
    },
    {
        "Method": "Deep Ensemble",
        "IC (full)": f"{ens_ic_full:+.4f}",
        "IC (filtered Q1-Q3)": f"{ens_ic_filtered:+.4f}",
        "Samples removed": f"{(~ens_keep).sum():,}",
    },
]

filter_df = pl.DataFrame(filter_rows)
filter_df

# %% [markdown]
# **Interpretation**: the filtering table above answers a concrete question -
# does excluding the most uncertain quartile improve IC? The expected direction
# is yes for a well-calibrated method; the realised direction and magnitude
# vary across reruns at this signal-to-noise ratio. The result that does
# survive reruns is the calibration story from the uncertainty-error
# correlation: the ensemble's disagreement tracks error more reliably than MC
# Dropout's dropout-spread. The ensemble result supports position sizing
# (Chapter 19): scale exposure inversely with ensemble uncertainty, even when
# simple quartile filtering moves IC only a few thousandths.

# %% [markdown]
# ### Connections to Other Chapters
#
# - **Calibration methods** (section 11.3): Platt scaling and isotonic regression
#   can post-process the raw uncertainty estimates above into better-calibrated
#   probabilities. The split-conformal section here implements the
#   distribution-free analog for regression intervals; see section 11.5 for the
#   general framework.
# - **Conformal allocation** (Chapter 17 case studies): the
#   `conformal_weighted` allocator (`case_studies/utils/allocation.py`) consumes
#   per-prediction Mondrian widths produced by `case_studies/utils/conformal.py`
#   - the same construction shown here, generalized to (symbol, fold)-stratified
#   calibration on the registry's stored predictions.
# - **Foundation model calibration**: The base architectures in `01_core_architectures`
#   and `04_transformers` can be wrapped with these uncertainty methods. TSFMs
#   (section 13.6) face additional calibration challenges from pretraining
#   distribution mismatch - see section 13.8 for discussion.
# - **Position sizing** (Chapter 19): Scale exposure inversely with ensemble
#   uncertainty - when ensemble members strongly agree, increase allocation.
# - **Model saving**: For production use, save ensemble members via
#   `torch.save(model.state_dict(), path)` and reload with matching architecture.

# %% [markdown]
# ## Key Takeaways
#
# 1. **MC Dropout is cheap**: Run $T$ forward passes through a single trained model.
#    No extra training cost - just keep dropout active via `model.train()` at inference.
#
# 2. **Deep Ensembles are better-calibrated**: Training $M$ independent models from
#    different initializations captures functional diversity. Ensemble disagreement
#    tracks absolute error more reliably than MC Dropout's dropout-spread on this
#    dataset, but costs $M \times$ training time.
#
# 3. **Both capture epistemic uncertainty**: Uncertainty is highest where the model
#    has seen less data or where patterns are ambiguous. This maps directly to
#    **position sizing** - reduce exposure when the model is uncertain.
#
# 4. **Raw uncertainty is not calibrated**: Gaussian intervals from either method
#    are severely under-covered (~0-20% empirical against 50%/80%/95% nominal).
#    The qualitative ordering - ensemble correlation positive, MC Dropout near
#    zero - is stable across reruns; the precise magnitudes are not.
#
# 5. **Split-conformal restores coverage**: Calibrating either method's spreads
#    against validation residuals produces intervals that achieve their nominal
#    coverage rate on the held-out test set in finite samples, with no
#    distributional assumption beyond exchangeability. This is the operational
#    fix for using uncertainty estimates in risk-aware position sizing.
#
# 6. **Aleatoric uncertainty requires architecture changes**: Separating data noise
#    from model uncertainty needs heteroscedastic output heads (predicting both
#    $\mu$ and $\sigma^2$), which neither baseline method provides.
#
# 7. **Practical guidance**: Use MC Dropout when compute is limited or the model
#    already has dropout. Use Deep Ensembles when calibration quality matters
#    for risk management. In both cases, layer split-conformal calibration on
#    top before using intervals downstream.
#
# **Next**: See `11_library_landscape` for high-level forecasting libraries
# (Darts, NeuralForecast, sktime) that wrap these architectures.
#
# **Book**: Section 13.8 discusses uncertainty quantification theory and the
# connection to Bayesian deep learning.
