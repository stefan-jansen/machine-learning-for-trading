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
# `03_great_debate` left a Transformer that was not reading position. Attention
# compares every token to every other and has no built-in notion of order, so what a
# token *is* decides what attention can find. The vanilla design makes one day one
# token, which gives attention 60 nearly interchangeable scalars to relate.
#
# Both architectures here answer that by changing the token. **PatchTST** makes a token
# a short run of consecutive days, so each one carries a piece of local shape rather
# than a single number. **iTransformer** goes further and makes a token an entire
# feature's history, so attention relates whole variables to each other and temporal
# order lives inside a token rather than between tokens. Neither is a bigger
# Transformer; both are a different answer to what should be compared with what.
#
# **Learning objectives**:
# - Build a patched Transformer over one feature at a time, and say what patching buys
#   in attention cost as the window grows.
# - Build an inverted Transformer whose tokens are features, and explain why it needs
#   no positional encoding when the vanilla design does.
# - Score both against a ridge regression on the same flattened window, and against
#   forecasting zero, so a difference between architectures is read against a
#   difference from nothing.
# - Read an attention matrix per head, and say what such a matrix does and does not
#   establish about which inputs matter.
#
# **Book Reference**: Chapter 13, Section 13.5 (Modern transformer variants).
# PatchTST: Nie et al. (2023); iTransformer: Liu et al. (2024).
#
# **Prerequisites**: `03_great_debate`; ETF features from `case_studies/etfs/`.

# %%
"""Compare PatchTST and iTransformer on ETF returns."""

import os

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
from utils.style import COLORS, show_plotly_with_alt

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
# ## The data
#
# This is the first notebook in the chapter to read the ETF **case study** rather than
# raw prices: a panel of momentum features and a forward-return label, built by the
# pipeline in `case_studies/etfs/` and shared with the Chapter 11 and 12 notebooks that
# fit tabular models to the same target. Using it means the comparison here can be read
# against those.
#
# The eight features are trailing returns over horizons from a week to a year. They are
# already the kind of input a Transformer is supposed to relate to each other, which is
# what makes the inverted design applicable at all - a token has to be something worth
# comparing, and "this fund's momentum over 252 days" is. The label is the forward
# 21-day return, so this is a monthly-horizon problem on a daily grid.

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
# ### What is in the panel
#
# A row per fund per date, so the panel has two extents worth knowing before anything
# is fitted: how far back it goes, and how many funds are quoting on a given date. The
# second one moves - funds launch - and it matters here because the ranking metric is
# computed across whatever funds exist on each date. A date with six funds and a date
# with ninety contribute equally to the average, and a rank correlation over six names
# is a much noisier number.
#
# The table gives each input feature its horizon in trading days and its coverage, and
# the label its spread. The horizons span a week to a year over the same window, which
# is what gives attention over features something to relate.

# %%
horizon_days = {c: int(c.removeprefix("ret_").removesuffix("d")) for c in FEATURE_COLS}
profile = pl.DataFrame(
    {
        "Feature": FEATURE_COLS,
        "Trailing horizon (days)": [horizon_days[c] for c in FEATURE_COLS],
        "Standard deviation": [round(float(df[c].std()), 4) for c in FEATURE_COLS],
    }
)
per_date = df.group_by(mds.date_col).len().sort(mds.date_col)
print(
    f"{df[mds.date_col].min()} to {df[mds.date_col].max()}, "
    f"{df[mds.entity_cols[0]].n_unique()} funds; funds per date "
    f"{per_date['len'].min()} to {per_date['len'].max()}, median {per_date['len'].median():.0f}"
)
print(
    f"Label {TARGET_COL}: mean {df[TARGET_COL].mean():+.5f}, "
    f"standard deviation {df[TARGET_COL].std():.5f}"
)
profile

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

# %% [markdown]
# The split is by **date**, at fixed fractions of the trading days, and an example is
# placed by the date it carries. The label is a 21-day forward return, so an example
# within `LABEL_HORIZON` days of a boundary resolves on the far side of it; those are
# dropped. Input windows may still reach back across a boundary, and should - at
# decision time a model has all of the past available to it.

# %%
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
# ### Scoring: the information coefficient
#
# Every model is scored the same way. On each date, rank the funds by prediction, rank
# them by what they actually returned, and correlate the two rankings - the Spearman
# rank correlation, averaged over dates. It measures whether the ordering was useful,
# which is what a cross-sectional strategy acts on, and it is a different question from
# whether the predicted return levels were close, which squared error measures. Both
# are reported below because a model can do well on one and badly on the other.
#
# The helper takes flat arrays so the Transformers and the ridge baseline go through
# exactly the same scoring path.


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
# The label is a 21-day forward return, so an example dated within 21 trading days of a
# boundary has an outcome that lands on the far side of it. Those examples are dropped -
# the purge printed above - which is what keeps a training label from being resolved by
# days the validation set is being asked about. Input windows may still reach back
# across a boundary, and should: at decision time a model has all of the past.
#
# One chronological cut is a simplification. Chapter 6 sets out the expanding
# walk-forward protocol that a deployment estimate needs.

# %% [markdown]
# ## PatchTST
#
# The paper's PatchTST has three structural properties that must be preserved, or
# the model loses the structure that distinguishes it and reduces to a generic
# Transformer over tokens:
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


# %% [markdown]
# ## iTransformer
#
# The other answer to what a token should be: make it a whole **feature**, not a day
# and not a patch of days. Each of the eight momentum horizons becomes one token
# carrying its entire `LOOKBACK`-day history, and attention then relates horizons to
# horizons rather than moments to moments.
#
# Two consequences follow from that one choice. There is no positional encoding,
# because features have no natural order to encode - temporal order lives inside a
# token, handled by the projection that maps a history to a vector. And attention now
# has eight tokens instead of sixty, which is a far smaller matrix and a far more
# interpretable one, since each row and column names something a reader can identify.


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
# with a patience of five epochs, and the weights from the lowest-scoring epoch are
# restored before returning. GPU memory is released after training so the next
# model's allocations do not stack on stale tensors.


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
    title="Validation error per epoch, with early stopping deciding the length",
    xaxis_title="Epoch",
    yaxis_title="Mean squared error on the validation windows",
)
show_plotly_with_alt(
    fig,
    "Two lines, one per architecture, plotting validation mean squared error against "
    "training epoch. Each line ends where early stopping halted that architecture, so "
    "the lines are of different lengths and the legend names each one's epoch count.",
)

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

# %% [markdown]
# The ridge baseline flattens each window into one long vector and fits a single
# penalised linear map. Ridge shrinks coefficients towards zero by a penalty on their
# size, which makes it sensitive to the scale of each input, so the pipeline
# standardises first - fitted on the training inputs alone and applied unchanged to the
# held-back ones, so no test statistic reaches the transform.

# %%
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

summary_df = pl.DataFrame(
    [
        {
            "Model": name,
            "Mean daily rank IC": round(r["ic"], 4),
            "Test MSE": round(float(r["mse"]), 6),
            "MSE / zero forecast": round(float(r["mse_ratio"]), 3),
        }
        for name, r in results.items()
    ]
)
summary_df

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
    title="Ranking skill and calibration are scored separately",
)
fig.update_yaxes(title_text="Mean daily Spearman IC", row=1, col=1)
fig.update_yaxes(title_text="Test MSE relative to zero forecast", row=1, col=2)
fig.add_hline(y=0, line_color=COLORS["neutral"], row=1, col=1)
fig.add_hline(
    y=1,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text="zero forecast",
    annotation_position="bottom right",
    row=1,
    col=2,
)
show_plotly_with_alt(
    fig,
    "Two bar charts over PatchTST, iTransformer and the ridge baseline. The left gives "
    "each one's mean daily cross-sectional rank correlation with a line at zero; the "
    "right gives its test mean squared error as a multiple of the zero forecast's, "
    "with a dotted line at one.",
)

# %% [markdown]
# The two panels answer the two questions separately, and they can disagree: ranking
# the funds correctly on each date and predicting the right return levels are not the
# same achievement, and a model that shrinks its predictions towards zero improves one
# while doing nothing for the other.
#
# The ridge regression is the reference that matters. It sees the identical window,
# flattened into one long vector, and fits a single penalised linear map - no attention,
# no patches, no tokens. Whatever the two Transformers do differently has to show up as
# a difference from that, not merely as a difference from each other. The zero forecast
# on the right panel plays the same role for the error column.

# %% [markdown]
# ### What the iTransformer's attention looks like
#
# Because a token here is a whole feature, the attention matrix is
# feature-by-feature: row $i$, column $j$ is how much weight the model put on
# feature $j$ while producing feature $i$'s output. Each row is a softmax and so sums
# to one, which means the reference point is not zero but $1/N$ - the weight a row
# would give every column if it favoured none of them. The heatmaps below plot the
# distance from that uniform reference, in percentage points.
#
# **One matrix per attention head.** A layer with `N_HEADS` heads computes that many
# separate attention patterns and concatenates their outputs, and PyTorch's
# `need_weights=True` returns the mean across heads *by default*. That average is not
# what any head did: two heads attending to complementary halves of the feature set
# average to something flatter than either, so a head-averaged matrix can look
# near-uniform when no head is. `average_attn_weights=False` returns them separately,
# which is what is drawn here.

# %% [markdown]
# The encoder is fed the input it actually sees: instance-normalised, transposed so
# features become tokens, then projected. These layers have `norm_first` set to False,
# so `self_attn` receives that projection unnormalised, which is what the cell passes.

# %%
itrans.eval()
sample_idx = np.linspace(0, len(X_test) - 1, num=min(512, len(X_test)), dtype=int)
X_sample = torch.FloatTensor(X_test[sample_idx]).to(DEVICE)
with torch.no_grad():
    means = X_sample.mean(dim=1, keepdim=True)
    variances = X_sample.var(dim=1, keepdim=True, unbiased=False)
    x_normalized = (X_sample - means) / torch.sqrt(variances + 1e-5)
    x_proj = itrans.input_proj(x_normalized.permute(0, 2, 1))
    first_layer = itrans.encoder.layers[0]
    _, attn = first_layer.self_attn(
        x_proj, x_proj, x_proj, need_weights=True, average_attn_weights=False
    )
# (batch, heads, features, features) -> averaged over the sampled windows only.
per_head_attn = attn.mean(dim=0).cpu().numpy()
uniform_attention = 1 / len(FEATURE_COLS)
head_deviation_pp = 100 * (per_head_attn - uniform_attention)
span = float(np.max(np.abs(head_deviation_pp)))
print(
    f"{head_deviation_pp.shape[0]} heads, {len(FEATURE_COLS)} feature tokens; "
    f"uniform weight is {100 * uniform_attention:.1f}%"
)

# %%
fig = make_subplots(
    rows=1,
    cols=head_deviation_pp.shape[0],
    subplot_titles=[f"Head {h + 1}" for h in range(head_deviation_pp.shape[0])],
    shared_yaxes=True,
)
for h in range(head_deviation_pp.shape[0]):
    fig.add_trace(
        go.Heatmap(
            z=head_deviation_pp[h],
            x=FEATURE_COLS,
            y=FEATURE_COLS,
            zmid=0,
            zmin=-span,
            zmax=span,
            colorscale=[
                [0, COLORS["negative"]],
                [0.5, COLORS["silver"]],
                [1, COLORS["positive"]],
            ],
            showscale=h == head_deviation_pp.shape[0] - 1,
            colorbar_title="Deviation<br>from uniform<br>(pp)",
            hovertemplate=(
                "Query: %{y}<br>Key: %{x}<br>Deviation from uniform: %{z:.2f} pp<extra></extra>"
            ),
        ),
        row=1,
        col=h + 1,
    )
fig.update_layout(
    title="Each head's attention over the momentum horizons, against uniform",
    height=460,
)
fig.update_xaxes(title_text="Key feature")
fig.update_yaxes(title_text="Query feature", row=1, col=1)
show_plotly_with_alt(
    fig,
    "One heatmap per attention head of the iTransformer's first encoder layer, on a "
    "shared colour scale. Each cell is how far that query-key feature pair's attention "
    "weight sits from the uniform weight, in percentage points, with the momentum "
    "horizons on both axes.",
)

# %% [markdown]
# Read the heads separately and compare the patterns, keeping the claims to the
# patterns. Heads that concentrate weight on different feature pairs are attending
# differently; a head close to uniform is spreading its weight evenly across the
# tokens. Both are observations about these matrices on these windows, and neither
# extends to what the heads contribute.
#
# It is tempting to read two similar matrices as one head being redundant, and that
# does not follow: each head multiplies its weights into its **own** learned value
# projection, so two heads with the same attention pattern can still write
# complementary things into the output. Establishing that a head contributes little
# means removing it and measuring what the predictions do, which this notebook does not
# do.
#
# Three further limits. It is the **first** layer of two, so it is not the model's
# overall view of the features. It is averaged over the sampled holdout windows, so a
# head that behaves differently in different market conditions shows up here as its
# average behaviour. And attention weight is not importance: a
# feature can receive little attention and still dominate the output through the
# residual path around the attention block, which is why these matrices are a
# description of one internal computation and not a feature-importance ranking, causal
# or otherwise.

# %% [markdown]
# ## Key takeaways
#
# 1. **What a token is decides what attention can do.** Self-attention relates tokens
#    to each other and has no notion of order beyond what the tokens carry, so the
#    vanilla choice of one day per token gives it 60 interchangeable scalars. Patching
#    puts local shape inside a token; inverting puts a whole variable inside one. Both
#    architectures are that one decision, not a larger model.
# 2. **Patching cuts the cost of attention quadratically in the patch size.** Attention
#    is $O(L^2)$ in the number of tokens, so tokens of `PATCH_SIZE` days over a window
#    of $L$ make it $O((L/P)^2)$ - which is what makes a long window affordable, and the
#    reason to reach for patching before reaching for a smaller window.
# 3. **The inverted design needs no positional encoding, and that is a consequence
#    rather than a saving.** Its tokens are features, and features have no natural
#    order to encode; temporal order lives inside a token, where the projection handles
#    it. Ask what a design's tokens are before asking what it does about position.
# 4. **Read an attention matrix per head, and read it as a description.** PyTorch
#    averages heads by default, and an average across heads is not what any head did.
#    Even per head, attention weight is not importance - the residual path routes
#    around the attention block - so these matrices say what one internal computation
#    did and not which inputs mattered.
# 5. **Score ranking and calibration separately, and both against something that is
#    not a Transformer.** A ridge regression on the same flattened window is the
#    comparison that decides whether the architecture bought anything; the two
#    Transformers against each other cannot answer that.
#
# **Known limitations.** One chronological split of one ETF panel, one label horizon,
# one seed, and teaching-scale dimensions throughout - both models here are small
# enough to train in minutes and neither is the size its paper used. Nothing here ranks
# the architectures; `12_case_study_insights` is where the same families are compared
# across case studies under walk-forward validation. The attention read-out is the
# first of two encoder layers, averaged over sampled holdout windows.
#
# **Next**: `05_tcn` drops attention entirely and gets a long receptive field from
# dilated causal convolutions instead.
