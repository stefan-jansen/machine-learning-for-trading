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
# # Transformer Architectures for Time Series
#
# **Docker image**: `ml4t-gpu`
#
# This notebook compares two modern Transformer variants for time series:
# **PatchTST** (patching along time) and **iTransformer** (attention over features).
# Both address limitations identified in the Great Debate (Section 13.4).
#
# **Learning Objectives**:
# - Implement PatchTST with channel-independent patching
# - Implement iTransformer with inverted attention (features as tokens)
# - Compare both approaches on ETF return prediction
# - Understand when each architecture excels
#
# **Book Reference**: Chapter 13, Section 13.5 (The Transformer's Evolution).
# PatchTST: Nie et al. (2023); iTransformer: Liu et al. (2024).
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %%
"""Compare PatchTST and iTransformer on ETF returns."""

import os
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from dl_sequences import (
    create_sequences_multi_asset,
    load_dl_dataset,
)
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from case_studies.config.patchtst.patchtst import PatchTST
from utils.reproducibility import set_global_seeds
from utils.style import COLORS

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
SEED = 42
LOOKBACK = 60
PATCH_SIZE = 6
D_MODEL = 32
N_HEADS = 2
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.0005
LABEL_HORIZON = 21

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

set_global_seeds(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# %% [markdown]
# ## Data Loading
#
# Load ETF features and labels from the case study pipeline.

# %%
mds = load_dl_dataset("etfs")

FEATURE_COLS = [
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_42d",
    "ret_63d",
    "ret_126d",
    "ret_189d",
    "ret_252d",
]
TARGET_COL = mds.label_col

missing_features = sorted(set(FEATURE_COLS) - set(mds.feature_names))
if missing_features:
    raise ValueError(f"Missing required ETF momentum features: {missing_features}")

df = mds.dataset.drop_nulls(subset=FEATURE_COLS + [TARGET_COL])
print(f"Features: {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")
print(f"Rows after dropna: {len(df):,}")

# %% [markdown]
# ## Sequence Creation
#
# Both models consume standard `(batch, lookback, n_features)` tensors. PatchTST does
# its patching internally - overlapping stride, channel-independent embedding, and
# RevIN - rather than requiring pre-patched inputs. iTransformer transposes the same
# tensor to treat each feature as a token.

# %%
X_reg, y_reg, timestamps, symbols = create_sequences_multi_asset(
    df,
    FEATURE_COLS,
    TARGET_COL,
    LOOKBACK,
    timestamp_col=mds.date_col,
    symbol_col=mds.entity_cols[0],
)
sequence_order = np.lexsort((symbols.astype(str), timestamps))
X_reg = np.nan_to_num(X_reg[sequence_order], nan=0.0, posinf=0.0, neginf=0.0)
y_reg = np.nan_to_num(y_reg[sequence_order], nan=0.0)
timestamps = timestamps[sequence_order]
symbols = symbols[sequence_order]
print(f"Sequences: {X_reg.shape}")

# %%
# Date-based 60/20/20 temporal split. The target is a 21-day forward return,
# so training and validation labels whose outcome windows cross the next
# boundary are purged. Input windows may use earlier observations, as they
# would at inference time.
unique_dates = np.sort(np.unique(timestamps))
train_boundary_idx = int(len(unique_dates) * 0.6)
val_boundary_idx = int(len(unique_dates) * 0.8)
train_end_date = unique_dates[train_boundary_idx]
val_end_date = unique_dates[val_boundary_idx]
train_label_cutoff = unique_dates[train_boundary_idx - LABEL_HORIZON]
val_label_cutoff = unique_dates[val_boundary_idx - LABEL_HORIZON]

train_mask = timestamps < train_label_cutoff
val_mask = (timestamps >= train_end_date) & (timestamps < val_label_cutoff)
test_mask = timestamps >= val_end_date

X_train, y_train = X_reg[train_mask], y_reg[train_mask]
X_val, y_val = X_reg[val_mask], y_reg[val_mask]
X_test, y_test = X_reg[test_mask], y_reg[test_mask]
test_dates, test_symbols = timestamps[test_mask], symbols[test_mask]

print(f"Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")
print(
    f"Purged {LABEL_HORIZON} target dates before each boundary: "
    f"validation starts {train_end_date}, test starts {val_end_date}"
)


# %% [markdown]
# ### Cross-sectional IC helper
#
# We evaluate every model on the same cross-sectional metric: per-date Spearman
# rank correlation between predictions and forward returns, then averaged across
# dates. The helper takes flat NumPy arrays so it can be called identically
# for the Transformer variants and the Ridge baseline.


# %%
def cross_sectional_ic_mean(y_true, y_pred, dates, syms):
    """Mean cross-sectional Spearman IC across dates."""
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
# > **Note**: This fixed 60/20/20 split is a pedagogical simplification. Its
# > 21-day purge keeps forward-label windows disjoint, but production deployment
# > still requires the expanding walk-forward protocol from Chapter 6.

# %% [markdown]
# ## PatchTST
#
# The paper's PatchTST has three structural properties that must be preserved, or
# the model loses its inductive bias and collapses to a generic Transformer on
# tokens:
#
# 1. **Channel-independent patching.** Each feature channel is treated as its own
#    univariate sequence and passed through the same shared Transformer weights.
#    No cross-channel mixing inside the encoder. This is the central regularizer.
# 2. **Overlapping patches.** Stride is smaller than patch length (typically
#    `stride = patch_len / 2`), so adjacent patches share timesteps. This keeps
#    local temporal context across the patch boundary.
# 3. **RevIN (Reversible Instance Normalization).** Per-sample per-channel mean/std
#    are removed before the backbone and added back after the prediction head,
#    making the model robust to distribution shift.
#
# We use the paper authors' reference implementation
# (`case_studies.config.patchtst.PatchTST`) directly, wired via a thin scalar-
# regression head. This matches the case-study pipelines, so findings in this
# chapter and in Section 13.9 use the same model.


# %%
# PatchTST imported at the top; instantiate with teaching-scale dimensions.
# The backbone does patching internally - input is raw `(batch, lookback, n_features)`.


# %% [markdown]
# ## iTransformer
#
# Inverts the attention dimension: treats each **feature** as a token
# (rather than each timestep). This lets attention capture cross-variate
# dependencies directly.


# %%
class iTransformer(nn.Module):
    """Teaching-scale iTransformer with a scalar regression adapter."""

    def __init__(self, lookback, n_features, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.n_features = n_features
        # Each feature's complete history is one variate token. The paper
        # intentionally omits positional embeddings: temporal order lives in
        # the neurons of this projection, not in the token order.
        self.input_proj = nn.Linear(lookback, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.token_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.feature_head = nn.Linear(n_features, 1)

    def forward(self, x):
        # x: (batch, lookback, n_features)
        means = x.mean(dim=1, keepdim=True).detach()
        variances = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - means) / torch.sqrt(variances + 1e-5)
        x = x.permute(0, 2, 1)  # (batch, n_features, lookback)
        x = self.input_proj(x)  # (batch, n_features, d_model)
        x = self.encoder(x)
        per_feature = self.token_head(x).squeeze(-1)
        return self.feature_head(per_feature).squeeze(-1)


# %% [markdown]
# ## Training
#
# A chunked validation forward keeps full-channel PatchTST in particular from
# OOMing on the whole val tensor at once; we reuse the same helper at test
# time.


# %%
def _chunked_forward(model, X_t, batch_size):
    """Run `model` over `X_t` in batches and concatenate outputs."""
    out = []
    for i in range(0, len(X_t), batch_size):
        out.append(model(X_t[i : i + batch_size]))
    return torch.cat(out, dim=0)


# %% [markdown]
# ### Training loop
#
# AdamW + cosine schedule + gradient clipping. Early stopping on val loss
# with patience=5; we restore the best state before returning. Memory is
# released after training so the next model's allocations don't stack on
# stale tensors.


# %%
def train_model(model, X_tr, y_tr, X_v, y_v, epochs, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_v_t = torch.FloatTensor(X_v).to(DEVICE)
    y_v_t = torch.FloatTensor(y_v).to(DEVICE)
    best_val, best_state, patience_counter = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_tr_t))
        epoch_loss, n_seen = 0.0, 0
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(_chunked_forward(model, X_v_t, batch_size), y_v_t).item()
        avg_train = epoch_loss / n_seen
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val, patience_counter = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: train={avg_train:.6f}, val={val_loss:.6f}")
        if patience_counter >= 5:
            break
    if best_state:
        model.load_state_dict(best_state)
    del X_tr_t, y_tr_t, X_v_t, y_v_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, history


# %% [markdown]
# ## Train PatchTST

# %%
set_global_seeds(SEED)
patchtst = PatchTST(
    n_features=len(FEATURE_COLS),
    lookback=LOOKBACK,
    patch_size=PATCH_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    revin=True,
).to(DEVICE)
n_params = sum(p.numel() for p in patchtst.parameters())
print(f"PatchTST: lookback={LOOKBACK}, patch_size={PATCH_SIZE}, stride={PATCH_SIZE // 2}")
print(f"Parameters: {n_params:,}")

patchtst, hist_patch = train_model(patchtst, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE, LR)

# %% [markdown]
# ## Train iTransformer

# %%
print(f"\niTransformer: {len(FEATURE_COLS)} feature tokens, lookback={LOOKBACK}")

set_global_seeds(SEED)
itrans = iTransformer(LOOKBACK, len(FEATURE_COLS), D_MODEL, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
n_params_i = sum(p.numel() for p in itrans.parameters())
print(f"Parameters: {n_params_i:,}")

itrans, hist_itrans = train_model(itrans, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE, LR)

# %% [markdown]
# ### Training Convergence
#
# Comparing validation loss curves reveals how quickly each architecture learns
# from the same data. PatchTST's reduced sequence length (patches vs full lookback)
# can affect convergence speed.

# %%
fig = go.Figure()
for name, hist in [("PatchTST", hist_patch), ("iTransformer", hist_itrans)]:
    fig.add_trace(
        go.Scatter(
            y=hist["val_loss"], mode="lines", name=f"{name} ({len(hist['val_loss'])} epochs)"
        )
    )

fig.update_layout(
    title=f"Both architectures stop within {max(len(hist_patch['val_loss']), len(hist_itrans['val_loss']))} epochs",
    xaxis_title="Epoch",
    yaxis_title="MSE (validation)",
)
fig.show()

# %% [markdown]
# ## Evaluation
#
# We compare both Transformer variants against a Ridge regression baseline
# that flattens the lookback window into a single feature vector.

# %%
# Transformer predictions
patchtst.eval()
itrans.eval()


def _predict_chunked(model, X, batch_size=BATCH_SIZE):
    """NumPy-array variant of `_chunked_forward` for the test-time evaluator."""
    model.eval()
    X_t = torch.FloatTensor(X).to(DEVICE)
    with torch.no_grad():
        pred = _chunked_forward(model, X_t, batch_size).cpu().numpy()
    del X_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


pred_patch = _predict_chunked(patchtst, X_test)
pred_itrans = _predict_chunked(itrans, X_test)

# %%
# Ridge baseline on flattened sequences. Its penalty is scale-sensitive, so the
# scaler is fit on training inputs only and applied unchanged to the test set.
X_flat_train = X_train.reshape(len(X_train), -1)
X_flat_test = X_test.reshape(len(X_test), -1)
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_flat_train, y_train)
pred_ridge = ridge.predict(X_flat_test)

# %%
# Compute metrics
results = {}
for name, pred in [("PatchTST", pred_patch), ("iTransformer", pred_itrans), ("Ridge", pred_ridge)]:
    mse = np.mean((pred - y_test) ** 2)
    ic = cross_sectional_ic_mean(y_test, pred, test_dates, test_symbols)
    results[name] = {"mse": mse, "ic": ic}

zero_mse = float(np.mean(y_test**2))
for result in results.values():
    result["mse_ratio"] = result["mse"] / zero_mse

best_ic_name = max(results, key=lambda name: results[name]["ic"])
both_transformers_beat_ridge = all(
    results[name]["ic"] > results["Ridge"]["ic"] for name in ("PatchTST", "iTransformer")
)
comparison_title = (
    "Transformers edge Ridge on IC; only iTransformer beats zero-return MSE"
    if both_transformers_beat_ridge
    else f"{best_ic_name} leads rank IC; the single split does not favor both Transformers"
)
print(
    f"Best cross-sectional rank IC: {best_ic_name} ({results[best_ic_name]['ic']:.3f}). "
    f"Zero-return test MSE: {zero_mse:.6f}."
)

# %% [markdown]
# **Interpretation**: Cross-sectional IC asks whether a model ranks ETFs well
# within each decision date; MSE asks whether its return levels are calibrated.
# The two panels can therefore disagree. Treat this purged single split as an
# architectural demonstration, not an architecture ranking. The authoritative
# comparison is the walk-forward evaluation in Section 13.9.

# %%
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Cross-sectional rank skill", "Squared error versus zero return"),
)
bar_palette = {
    "PatchTST": COLORS["blue"],
    "iTransformer": COLORS["amber"],
    "Ridge": COLORS["slate"],
}
for name, r in results.items():
    fig.add_trace(
        go.Bar(
            x=[name],
            y=[r["ic"]],
            name=name,
            marker_color=bar_palette.get(name, COLORS["blue"]),
            text=[f"{r['ic']:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[name],
            y=[r["mse_ratio"]],
            marker_color=bar_palette.get(name, COLORS["blue"]),
            text=[f"{r['mse_ratio']:.2f}x"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

fig.update_layout(
    title=comparison_title,
)
fig.update_yaxes(title_text="Mean daily Spearman IC", row=1, col=1)
fig.update_yaxes(title_text="Test MSE relative to zero forecast", row=1, col=2)
fig.add_hline(y=0, line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(y=1, line_dash="dot", line_color=COLORS["neutral"], row=1, col=2)
fig.show()

# %% [markdown]
# Both Transformer variants share the same raw lookback tensor; they differ in
# which axis attention is applied over - features (iTransformer) or time-patch
# tokens (PatchTST). The scaled Ridge baseline on the flattened sequence is the
# null model: a linear map with no attention structure at all.

# %% [markdown]
# ### iTransformer Attention Weights
#
# Because iTransformer applies attention over features (not time steps), the
# attention matrix reveals which feature pairs the model considers related.
# We extract weights from the first encoder layer and compare them with the
# uniform 1/N reference. Attention weights are a model diagnostic, not a causal
# feature-importance measure.

# %%
# Extract first-layer attention from iTransformer's self_attn module. Calling
# self_attn with need_weights=True is the supported way to read attention from
# the standard PyTorch TransformerEncoderLayer; we feed the input the encoder
# itself sees after instance normalization and input projection, so the weights correspond to the same
# computation the model is performing during evaluation. Evenly spaced holdout
# rows cover the panel rather than taking one contiguous asset block.
itrans.eval()
sample_idx = np.linspace(0, len(X_test) - 1, num=min(512, len(X_test)), dtype=int)
X_sample = torch.FloatTensor(X_test[sample_idx]).to(DEVICE)
with torch.no_grad():
    means = X_sample.mean(dim=1, keepdim=True)
    variances = X_sample.var(dim=1, keepdim=True, unbiased=False)
    x_normalized = (X_sample - means) / torch.sqrt(variances + 1e-5)
    x_inv = x_normalized.permute(0, 2, 1)  # (batch, n_features, lookback)
    x_proj = itrans.input_proj(x_inv)
    first_layer = itrans.encoder.layers[0]
    _, attn = first_layer.self_attn(x_proj, x_proj, x_proj, need_weights=True)
avg_attn = attn.mean(dim=0).cpu().numpy()  # (n_features, n_features)
uniform_attention = 1 / len(FEATURE_COLS)
attention_deviation_pp = 100 * (avg_attn - uniform_attention)

fig = go.Figure(
    go.Heatmap(
        z=attention_deviation_pp,
        x=FEATURE_COLS,
        y=FEATURE_COLS,
        zmid=0,
        colorscale=[
            [0, COLORS["negative"]],
            [0.5, COLORS["silver"]],
            [1, COLORS["positive"]],
        ],
        colorbar_title="Deviation<br>from uniform<br>(percentage points)",
        hovertemplate=(
            "Query: %{y}<br>Key: %{x}<br>Deviation from uniform: %{z:.2f} pp<extra></extra>"
        ),
    )
)
max_attention_deviation = float(np.max(np.abs(attention_deviation_pp)))
fig.update_layout(
    title=f"Attention stays within {max_attention_deviation:.2f} pp of uniform across momentum horizons",
    xaxis_title="Key feature",
    yaxis_title="Query feature",
    width=700,
    height=520,
)
fig.show()

# %% [markdown]
# Near-uniform attention means this first layer does not sharply favor a few
# momentum horizons on the sampled holdout panel. That is a useful negative
# result. It does not prove that the features are unrelated, and the weights
# should not be read as causal importance.

# %% [markdown]
# ## Key Takeaways
#
# 1. **PatchTST** patches along the time axis, reducing attention complexity from
#    $O(L^2)$ to $O((L/P)^2)$ - a dramatic speedup enabling longer lookback windows
# 2. **iTransformer** inverts the attention dimension: features become tokens, enabling
#    direct cross-variate dependency modeling without positional embeddings
# 3. **Rank IC and MSE answer different questions**: compare the architecture panel
#    with both Ridge and the zero-return squared-error reference
# 4. **One split cannot rank architectures**: the purged holdout illustrates the
#    mechanics; Section 13.9 supplies the walk-forward comparison
# 5. Both represent improvements over vanilla Transformer designs by incorporating
#    temporal inductive biases (patching) or sidestepping the temporal order problem
#    entirely (inverted attention)
# 6. **Attention is descriptive, not causal**: near-uniform weights here do not
#    establish feature importance or the absence of dependence
# 7. This notebook evaluates a single ETF universe; cross-dataset comparison in
#    `12_case_study_insights` tests whether these patterns generalize
#
# PyTorch deterministic algorithms and a fixed cuBLAS workspace make repeated
# executions reproducible on the same software and GPU stack; another environment
# may still produce small floating-point differences.
#
# **Next**: See `05_tcn` for temporal convolutional networks.
