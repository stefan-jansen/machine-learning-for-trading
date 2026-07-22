# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.001638, "end_time": "2026-05-25T01:24:21.965925+00:00", "exception": false, "start_time": "2026-05-25T01:24:21.964287+00:00", "status": "completed"}
# # CNN with Time Series Image Encoding (GAF + MTF)
#
# **Docker image**: `ml4t-gpu`
#
# This notebook converts time series windows into 2D images using **Gramian Angular
# Fields** (GAF) and **Markov Transition Fields** (MTF), then trains a standard CNN
# to regress on forward returns. The approach reformulates forecasting as image
# regression, allowing convolutional networks to detect visual patterns in the
# encoded representations.
#
# **Learning Objectives**:
# - Implement Gramian Angular Summation Fields (GASF) that encode temporal
#   correlations as angular differences
# - Implement Markov Transition Fields (MTF) that encode transition probabilities
#   between discretized states
# - Stack GAF + MTF as multi-channel images and train a CNN regressor on
#   forward returns
# - Compare CNN-on-images against a Ridge+PCA baseline on the same flattened
#   image pixels, evaluated by MSE and Spearman IC
#
# **Book Reference**: Chapter 13, Section 13.6 (The Full Practitioner Toolkit)
#
# **Prerequisites**: ETF features (`case_studies/etfs/`)

# %% papermill={"duration": 7.012912, "end_time": "2026-05-25T01:24:28.980445+00:00", "exception": false, "start_time": "2026-05-25T01:24:21.967533+00:00", "status": "completed"}
"""CNN with Time Series Image Encoding — convert time series to GAF/MTF images for forward-return regression."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from ml4t.diagnostic.metrics import cross_sectional_ic_series
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

from dl_sequences import create_sequences_multi_asset, load_dl_dataset, train_model

# %% papermill={"duration": 0.005775, "end_time": "2026-05-25T01:24:28.988178+00:00", "exception": false, "start_time": "2026-05-25T01:24:28.982403+00:00", "status": "completed"} tags=["parameters"]
SEED = 42
LOOKBACK = 20
IMAGE_SIZE = 32
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
DROPOUT = 0.5
MAX_TRAIN_SAMPLES = 40_000
MAX_VAL_SAMPLES = 10_000
MAX_TEST_SAMPLES = 10_000
INFER_BATCH_SIZE = 1_024

# %% papermill={"duration": 0.066581, "end_time": "2026-05-25T01:24:29.056244+00:00", "exception": false, "start_time": "2026-05-25T01:24:28.989663+00:00", "status": "completed"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

set_global_seeds(SEED)

# %% [markdown] papermill={"duration": 0.00132, "end_time": "2026-05-25T01:24:29.059105+00:00", "exception": false, "start_time": "2026-05-25T01:24:29.057785+00:00", "status": "completed"}
# ## Data Loading
#
# We use ETF data from the case study pipeline, providing diverse cross-asset
# time series for testing image encoding approaches.

# %% papermill={"duration": 0.564624, "end_time": "2026-05-25T01:24:29.624988+00:00", "exception": false, "start_time": "2026-05-25T01:24:29.060364+00:00", "status": "completed"}
mds = load_dl_dataset("etfs")

# We load six features for completeness but only the first feature column is
# encoded as a GAF + MTF image (see `create_image_dataset`). The other five
# features are retained in the panel only so it stays compatible with shared
# sequence builders; production usage would encode all features into a
# (2 × F)-channel image.
FEATURE_COLS = mds.feature_names[:6]
TARGET_COL = mds.label_col

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")

# %% [markdown] papermill={"duration": 0.001679, "end_time": "2026-05-25T01:24:29.628160+00:00", "exception": false, "start_time": "2026-05-25T01:24:29.626481+00:00", "status": "completed"}
# ## Sequence Creation and Temporal Split

# %% papermill={"duration": 3.434698, "end_time": "2026-05-25T01:24:33.064153+00:00", "exception": false, "start_time": "2026-05-25T01:24:29.629455+00:00", "status": "completed"}
df = mds.dataset.drop_nulls(subset=FEATURE_COLS + [TARGET_COL])
print(f"Rows after dropping nulls: {len(df):,}")

X, y, timestamps, symbols = create_sequences_multi_asset(
    df,
    FEATURE_COLS,
    TARGET_COL,
    LOOKBACK,
    timestamp_col=mds.date_col,
    symbol_col=mds.entity_cols[0],
)
print(f"Sequences: {X.shape[0]:,}, shape: {X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = np.nan_to_num(y, nan=0.0).astype(np.float32)

# %% papermill={"duration": 0.064152, "end_time": "2026-05-25T01:24:33.131125+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.066973+00:00", "status": "completed"}
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


# %% [markdown] papermill={"duration": 0.002412, "end_time": "2026-05-25T01:24:33.136275+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.133863+00:00", "status": "completed"}
# ### Pedagogical subsampling
#
# The GAF/MTF encoding loop produces one (image_size × image_size × 2) tensor
# per sample, so the full ETF panel takes considerable wall-clock to encode.
# We cap each split at a few tens of thousands of sequences for tractable
# runtime — the cap is row-based here, which means the last batch can start
# mid-date and leave a partial cross-section; the per-date Spearman IC
# downstream handles partial cross-sections via `min_obs` rather than
# refusing to compute.


# %% papermill={"duration": 0.005966, "end_time": "2026-05-25T01:24:33.144744+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.138778+00:00", "status": "completed"}
if len(X_train) > MAX_TRAIN_SAMPLES:
    X_train = X_train[-MAX_TRAIN_SAMPLES:]
    y_train = y_train[-MAX_TRAIN_SAMPLES:]
if len(X_val) > MAX_VAL_SAMPLES:
    X_val = X_val[-MAX_VAL_SAMPLES:]
    y_val = y_val[-MAX_VAL_SAMPLES:]
if len(X_test) > MAX_TEST_SAMPLES:
    X_test = X_test[-MAX_TEST_SAMPLES:]
    y_test = y_test[-MAX_TEST_SAMPLES:]
    test_dates = test_dates[-MAX_TEST_SAMPLES:]
    test_symbols = test_symbols[-MAX_TEST_SAMPLES:]

print(f"Capped split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")


# %% [markdown] papermill={"duration": 0.001346, "end_time": "2026-05-25T01:24:33.147928+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.146582+00:00", "status": "completed"}
# ### Cross-sectional IC helper
#
# Mean cross-sectional Spearman IC by date — same metric used in the rest of
# Chapter 13 so the Image-CNN result is comparable with the other
# architectures.


# %% papermill={"duration": 0.004392, "end_time": "2026-05-25T01:24:33.153689+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.149297+00:00", "status": "completed"}
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


# %% [markdown] papermill={"duration": 0.001546, "end_time": "2026-05-25T01:24:33.156648+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.155102+00:00", "status": "completed"}
# ## Gramian Angular Summation Field (GASF)
#
# The GASF encodes a time series as a matrix of trigonometric sums. Given a
# normalized series $\tilde{x}_i \in [-1, 1]$, we compute angles
# $\phi_i = \arccos(\tilde{x}_i)$ and form:
#
# $$\text{GASF}_{i,j} = \cos(\phi_i + \phi_j)$$
#
# This preserves temporal ordering. The diagonal evaluates to
# $\cos(2\phi_i) = 2\tilde{x}_i^2 - 1$ — a deterministic function of the
# normalized value, not the value itself — while off-diagonal entries capture
# pairwise temporal correlations between time steps $i$ and $j$.


# %% papermill={"duration": 0.004659, "end_time": "2026-05-25T01:24:33.162691+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.158032+00:00", "status": "completed"}
def gramian_angular_field(series: np.ndarray, image_size: int) -> np.ndarray:
    """Compute the Gramian Angular Summation Field (GASF).

    Args:
        series: 1D array of shape (T,) -- raw feature values for one window
        image_size: Output image dimension (image_size x image_size)

    Returns:
        GASF matrix of shape (image_size, image_size) with values in [-1, 1]
    """
    # Resample to image_size via linear interpolation
    target_positions = np.linspace(0, len(series) - 1, image_size)
    source_positions = np.arange(len(series))
    resampled = np.interp(target_positions, source_positions, series)

    # Min-max scale to [-1, 1]
    s_min, s_max = resampled.min(), resampled.max()
    if s_max - s_min < 1e-8:
        return np.zeros((image_size, image_size), dtype=np.float32)
    scaled = 2.0 * (resampled - s_min) / (s_max - s_min) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)

    # Compute angular representation
    phi = np.arccos(scaled)

    # GASF: cos(phi_i + phi_j)
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf.astype(np.float32)


# %% [markdown] papermill={"duration": 0.001389, "end_time": "2026-05-25T01:24:33.165497+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.164108+00:00", "status": "completed"}
# ## Markov Transition Field (MTF)
#
# The MTF discretizes a time series into $Q$ quantile bins and builds a
# transition matrix $W$ where $W_{q_i, q_j}$ is the probability of
# transitioning from bin $q_i$ to bin $q_j$. The full MTF matrix is:
#
# $$\text{MTF}_{i,j} = W_{q_i, q_j}$$
#
# where $q_i$ is the quantile bin of the $i$-th timestep. This captures the
# dynamic transition structure of the series.


# %% papermill={"duration": 0.004854, "end_time": "2026-05-25T01:24:33.171761+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.166907+00:00", "status": "completed"}
def markov_transition_field(series: np.ndarray, image_size: int, n_bins: int = 8) -> np.ndarray:
    """Compute the Markov Transition Field (MTF).

    Args:
        series: 1D array of shape (T,) -- raw feature values for one window
        image_size: Output image dimension (image_size x image_size)
        n_bins: Number of quantile bins for discretization

    Returns:
        MTF matrix of shape (image_size, image_size) with values in [0, 1]
    """
    # Resample to image_size via linear interpolation
    target_positions = np.linspace(0, len(series) - 1, image_size)
    source_positions = np.arange(len(series))
    resampled = np.interp(target_positions, source_positions, series)

    # Discretize into quantile bins
    bin_edges = np.percentile(resampled, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # ensure max value is included
    bin_ids = np.digitize(resampled, bin_edges[1:-1])

    # Build transition matrix (n_bins x n_bins)
    transition = np.zeros((n_bins, n_bins), dtype=np.float32)
    for t in range(len(bin_ids) - 1):
        transition[bin_ids[t], bin_ids[t + 1]] += 1

    # Normalize rows to get probabilities
    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition /= row_sums

    # Build MTF: entry (i, j) = transition probability from bin at time i to bin at time j
    # Vectorized via advanced indexing (equivalent to the nested loop but ~100x faster)
    mtf = transition[np.ix_(bin_ids, bin_ids)]

    return mtf.astype(np.float32)


# %% [markdown] papermill={"duration": 0.001404, "end_time": "2026-05-25T01:24:33.174717+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.173313+00:00", "status": "completed"}
# ## Create Multi-Channel Image Dataset
#
# For each sequence, we encode the first feature as a GASF and an MTF,
# then stack them as a 2-channel image. This gives the CNN both angular
# correlation (GASF) and transition dynamics (MTF) as complementary views.
#
# > **Simplification**: This demo encodes only the first feature column. A production
# > system would encode all features, stacking GASF+MTF per feature to produce a
# > $(2 \times F)$-channel image (e.g., 12 channels for 6 features). We use one
# > feature here to keep encoding time manageable and focus on the method itself.


# %% papermill={"duration": 0.004637, "end_time": "2026-05-25T01:24:33.180781+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.176144+00:00", "status": "completed"}
def create_image_dataset(X_sequences: np.ndarray, image_size: int) -> np.ndarray:
    """Convert feature sequences into stacked GAF + MTF image tensors.

    For each sample, takes the first feature column and computes both GASF
    and MTF encodings, returning a (N, 2, H, W) tensor.

    Args:
        X_sequences: Array of shape (N, lookback, n_features)
        image_size: Target image dimension (H = W = image_size)

    Returns:
        Image tensor of shape (N, 2, image_size, image_size)
    """
    n_samples = X_sequences.shape[0]
    images = np.zeros((n_samples, 2, image_size, image_size), dtype=np.float32)

    for i in range(n_samples):
        series = X_sequences[i, :, 0]  # first feature
        images[i, 0] = gramian_angular_field(series, image_size)
        images[i, 1] = markov_transition_field(series, image_size)

        if (i + 1) % 10000 == 0 or i == n_samples - 1:
            print(f"  Encoded {i + 1:,}/{n_samples:,} samples")

    return images


# %% [markdown] papermill={"duration": 0.001384, "end_time": "2026-05-25T01:24:33.183619+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.182235+00:00", "status": "completed"}
# ## Encode Training, Validation, and Test Sets

# %% papermill={"duration": 6.424907, "end_time": "2026-05-25T01:24:39.609924+00:00", "exception": false, "start_time": "2026-05-25T01:24:33.185017+00:00", "status": "completed"}
print("Encoding training images...")
X_train_img = create_image_dataset(X_train, IMAGE_SIZE)
print(f"Train images: {X_train_img.shape}")

print("Encoding validation images...")
X_val_img = create_image_dataset(X_val, IMAGE_SIZE)
print(f"Val images: {X_val_img.shape}")

print("Encoding test images...")
X_test_img = create_image_dataset(X_test, IMAGE_SIZE)
print(f"Test images: {X_test_img.shape}")

# %% [markdown] papermill={"duration": 0.001544, "end_time": "2026-05-25T01:24:39.613199+00:00", "exception": false, "start_time": "2026-05-25T01:24:39.611655+00:00", "status": "completed"}
# ## Visualize Sample Encodings
#
# Inspecting the GASF and MTF channels for a few training samples to verify
# the encoding produces visually distinct patterns.

# %% papermill={"duration": 0.51172, "end_time": "2026-05-25T01:24:40.126405+00:00", "exception": false, "start_time": "2026-05-25T01:24:39.614685+00:00", "status": "completed"}
fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)

for row in range(3):
    idx = row * 1000  # spread samples across the dataset
    if idx >= len(X_train_img):
        idx = row

    # Raw feature series
    axes[row, 0].plot(X_train[idx, :, 0], linewidth=0.8)
    axes[row, 0].set_title(f"Sample {idx}: Raw Series" if row == 0 else f"Sample {idx}")
    axes[row, 0].set_xlabel("Timestep")

    # GASF channel
    im1 = axes[row, 1].imshow(X_train_img[idx, 0], cmap="RdBu_r", aspect="auto")
    axes[row, 1].set_title("GASF" if row == 0 else "")
    plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

    # MTF channel
    im2 = axes[row, 2].imshow(X_train_img[idx, 1], cmap="YlOrRd", aspect="auto")
    axes[row, 2].set_title("MTF" if row == 0 else "")
    plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

fig.suptitle("Time Series Image Encodings: GASF and MTF Channels")
fig.show()

# %% [markdown] papermill={"duration": 0.001929, "end_time": "2026-05-25T01:24:40.130368+00:00", "exception": false, "start_time": "2026-05-25T01:24:40.128439+00:00", "status": "completed"}
# ## CNN Architecture
#
# Three convolutional blocks (Conv2d $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool)
# followed by adaptive average pooling and a linear regression head.
# The architecture is intentionally simple to isolate the contribution of
# the image encoding from the model complexity.


# %% papermill={"duration": 0.00512, "end_time": "2026-05-25T01:24:40.137310+00:00", "exception": false, "start_time": "2026-05-25T01:24:40.132190+00:00", "status": "completed"}
class CNNBlock(nn.Module):
    """Single CNN building block: Conv2d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# %% [markdown] papermill={"duration": 0.001805, "end_time": "2026-05-25T01:24:40.140951+00:00", "exception": false, "start_time": "2026-05-25T01:24:40.139146+00:00", "status": "completed"}
# ### Full Image CNN
#
# Three CNN blocks downsample the spatial dimensions by $2\times$ each,
# followed by adaptive average pooling to a fixed-size vector regardless
# of the input image size. Dropout before the final linear layer provides
# regularization.


# %% papermill={"duration": 0.00506, "end_time": "2026-05-25T01:24:40.147770+00:00", "exception": false, "start_time": "2026-05-25T01:24:40.142710+00:00", "status": "completed"}
class ImageCNN(nn.Module):
    """CNN for regression on GAF+MTF encoded time series images.

    Architecture:
        Input (N, 2, H, W)
        -> CNNBlock(2, 32)   -> (N, 32, H/2, W/2)
        -> CNNBlock(32, 64)  -> (N, 64, H/4, W/4)
        -> CNNBlock(64, 128) -> (N, 128, H/8, W/8)
        -> AdaptiveAvgPool2d(1, 1) -> (N, 128)
        -> Dropout -> Linear -> (N, 1)
    """

    def __init__(self, n_channels: int = 2, dropout: float = 0.5):
        super().__init__()
        self.block1 = CNNBlock(n_channels, 32)
        self.block2 = CNNBlock(32, 64)
        self.block3 = CNNBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten to (N, 128)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)  # (N,)


# %% [markdown] papermill={"duration": 0.001816, "end_time": "2026-05-25T01:24:40.151421+00:00", "exception": false, "start_time": "2026-05-25T01:24:40.149605+00:00", "status": "completed"}
# ## Train the Image CNN

# %% papermill={"duration": 11.158235, "end_time": "2026-05-25T01:24:51.311457+00:00", "exception": false, "start_time": "2026-05-25T01:24:40.153222+00:00", "status": "completed"}
model = ImageCNN(n_channels=2, dropout=DROPOUT).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"ImageCNN parameters: {n_params:,}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, channels: 2 (GASF + MTF)")

history = train_model(model, X_train_img, y_train, X_val_img, y_val, EPOCHS, LR, BATCH_SIZE, DEVICE)

# %% [markdown] papermill={"duration": 0.001951, "end_time": "2026-05-25T01:24:51.315532+00:00", "exception": false, "start_time": "2026-05-25T01:24:51.313581+00:00", "status": "completed"}
# ## Evaluate on Test Set

# %% papermill={"duration": 0.061413, "end_time": "2026-05-25T01:24:51.378866+00:00", "exception": false, "start_time": "2026-05-25T01:24:51.317453+00:00", "status": "completed"}
model.eval()
with torch.no_grad():
    preds = []
    for i in range(0, len(X_test_img), INFER_BATCH_SIZE):
        X_test_t = torch.FloatTensor(X_test_img[i : i + INFER_BATCH_SIZE]).to(DEVICE)
        preds.append(model(X_test_t).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)

test_mse = np.mean((y_pred - y_test) ** 2)
test_ic = cross_sectional_ic_mean(y_test, y_pred, test_dates, test_symbols)

print("\nImage CNN Test Results:")
print(f"  MSE: {test_mse:.6f}")
print(f"  Spearman IC: {test_ic:.4f}")

# %% [markdown] papermill={"duration": 0.002011, "end_time": "2026-05-25T01:24:51.383036+00:00", "exception": false, "start_time": "2026-05-25T01:24:51.381025+00:00", "status": "completed"}
# ## Ridge + PCA Baseline
#
# Flatten the 2-channel images into vectors, reduce dimensionality with PCA
# (100 components), then fit a Ridge regression. This tests whether the CNN
# learns spatial structure beyond what a linear model can extract from the
# same pixel representation.

# %% papermill={"duration": 4.207208, "end_time": "2026-05-25T01:24:55.592181+00:00", "exception": false, "start_time": "2026-05-25T01:24:51.384973+00:00", "status": "completed"}
X_train_flat = X_train_img.reshape(len(X_train_img), -1)
X_test_flat = X_test_img.reshape(len(X_test_img), -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

n_components = min(100, X_train_scaled.shape[1], X_train_scaled.shape[0])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_pca, y_train)
y_ridge_pred = ridge.predict(X_test_pca)

ridge_mse = np.mean((y_ridge_pred - y_test) ** 2)
ridge_ic = cross_sectional_ic_mean(y_test, y_ridge_pred, test_dates, test_symbols)

print(f"\nRidge + PCA ({n_components} components) Baseline:")
print(f"  MSE: {ridge_mse:.6f}")
print(f"  Spearman IC: {ridge_ic:.4f}")

# %% [markdown] papermill={"duration": 0.009176, "end_time": "2026-05-25T01:24:55.604622+00:00", "exception": false, "start_time": "2026-05-25T01:24:55.595446+00:00", "status": "completed"}
# ## Summary

# %% papermill={"duration": 0.010541, "end_time": "2026-05-25T01:24:55.618329+00:00", "exception": false, "start_time": "2026-05-25T01:24:55.607788+00:00", "status": "completed"}
results_df = pl.DataFrame(
    {
        "Model": ["Image CNN (GASF+MTF)", f"Ridge + PCA ({n_components})"],
        "Spearman IC": [test_ic, ridge_ic],
        "MSE": [test_mse, ridge_mse],
    }
)
results_df

# %% [markdown] papermill={"duration": 0.005255, "end_time": "2026-05-25T01:24:55.629681+00:00", "exception": false, "start_time": "2026-05-25T01:24:55.624426+00:00", "status": "completed"}
# **Interpretation**: treat this as a method illustration, not an
# architectural claim. In this run the Image CNN's cross-sectional Spearman
# IC (0.008) falls well short of the Ridge+PCA baseline on the same flattened
# pixels (0.043 — roughly 5× higher; see `results_df`), so the CNN's spatial
# inductive bias does **not** extract patterns from the GASF + MTF image
# encoding that a linear projection of the same pixels cannot. That gap is
# itself the instructive result: adding convolutional capacity on top of a
# single-channel-derived image buys nothing over a linear model on the same
# pixels here, and the image representation as a class does not beat
# sequence-native models in this chapter.
# The encoding maps a single feature channel into a 2-channel image, so
# the representation is necessarily impoverished compared to the
# multi-feature sequence inputs used by other architectures here.
# Section 13.6 reviews published comparisons that encode all features and
# evaluate against sequence-native models — those are the right
# benchmarks for the GAF/MTF representation as a class.

# %% [markdown] papermill={"duration": 0.002, "end_time": "2026-05-25T01:24:55.633744+00:00", "exception": false, "start_time": "2026-05-25T01:24:55.631744+00:00", "status": "completed"}
# ## Key Takeaways
#
# 1. **GAF encodes temporal correlation**: The Gramian Angular Field maps
#    pairwise temporal relationships into a symmetric matrix, preserving the
#    original time ordering along the diagonal
# 2. **MTF captures transition dynamics**: The Markov Transition Field
#    discretizes the series into quantile bins and encodes state-to-state
#    transition probabilities, complementing the angular view
# 3. **Multi-channel images provide richer signal**: Stacking GASF and MTF
#    as separate channels gives the CNN two complementary views of the same
#    window, similar to RGB channels in natural images
# 4. **Encoding cost is non-trivial**: The per-sample image generation loop
#    adds significant preprocessing overhead compared to feeding raw sequences
#    directly to an LSTM or Transformer
# 5. **Representation differs from sequence-native models**: this notebook
#    does not evaluate the CNN against LSTM/Transformer baselines or test
#    ensembling — Section 13.6 reviews the published comparisons
# 6. **Single-feature encoding is a teaching simplification**: encoded
#    representations come from one feature channel only, so single-split
#    IC results are method illustrations rather than empirical claims
#    about GAF/MTF + CNN versus Ridge as architectures in general
#
# **Next**: See `09_foundation_models` for pre-trained time series foundation
# models that skip manual feature engineering entirely.
# **Book**: Section 13.6 discusses image encoding alongside other alternative
# representations in the full practitioner toolkit.
