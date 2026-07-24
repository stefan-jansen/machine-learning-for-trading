# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Autoencoder on Crypto Returns
#
# **Chapter 14: Latent Factors**
#
# This notebook applies a vanilla autoencoder to crypto hourly returns,
# using reconstruction error as an anomaly signal and visualizing the latent space.
#
# **Why Crypto for Autoencoders**:
# - High-frequency data (35K+ hourly observations)
# - Multiple correlated assets (BTC, ETH, SOL, etc.)
# - Clear regime structure for anomaly detection
#
# **Key Concepts**:
# - Reconstruction error as anomaly/regime indicator
# - Latent space visualization (2D embedding)
# - Relationship between reconstruction error and volatility
#
# **Learning Outcomes**:
# - LO1: Apply autoencoder to multi-asset returns
# - LO2: Use reconstruction error for anomaly detection
# - LO3: Visualize latent representations
#
# **Cross-References**:
# - Chapter 14: `conditional_autoencoder.py` (GKX model)
# - Chapter 13: Deep learning fundamentals
# - Chapter 11: `garch_crypto_vol.py` (volatility comparison)

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ML4T configuration
from data import load_crypto_perps
from utils import DATA_DIR

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
# Production defaults — Papermill injects overrides for CI

# %%
# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Date ranges
START_DATE = "2021-01-01"
END_DATE = "2024-12-01"
TEST_START = "2023-06-01"

# Crypto symbols (top by volume/liquidity)
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "SUIUSDT",
]

# Autoencoder parameters
LATENT_DIM = 2  # For visualization
HIDDEN_DIM = 32
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001

print("Autoencoder Crypto Configuration:")
print(f"  Symbols: {SYMBOLS}")
print(f"  Date range: {START_DATE} to {END_DATE}")
print(f"  Latent dim: {LATENT_DIM}")

# %% [markdown]
# ## 2. Load Crypto Hourly Data

# %%
print("Loading crypto hourly data...")

crypto = load_crypto_perps("1h")

# Filter symbols and date range
crypto = (
    crypto.filter(
        (pl.col("symbol").is_in(SYMBOLS))
        & (pl.col("timestamp") >= pl.lit(START_DATE).str.to_datetime().dt.replace_time_zone("UTC"))
        & (pl.col("timestamp") <= pl.lit(END_DATE).str.to_datetime().dt.replace_time_zone("UTC"))
    )
    .sort(["symbol", "timestamp"])
    .select(["timestamp", "symbol", "close"])
)

# Pivot to wide format
crypto_wide = crypto.pivot(on="symbol", index="timestamp", values="close").sort("timestamp")

# Convert to pandas (strip timezone — not needed for autoencoder analysis)
df = crypto_wide.to_pandas()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df.set_index("timestamp")

# Calculate hourly returns (scaled for stability)
returns = df.pct_change().dropna() * 100  # Percentage returns

# Drop any remaining NaN columns
returns = returns.dropna(axis=1, how="all")
available_symbols = returns.columns.tolist()

print(f"  Observations: {len(returns):,}")
print(f"  Symbols: {available_symbols}")
print(f"  Date range: {returns.index.min()} to {returns.index.max()}")

# %% [markdown]
# ## 3. Autoencoder Architecture


# %%
class CryptoAutoencoder(nn.Module):
    """
    Vanilla Autoencoder for Crypto Returns.

    Architecture:
    - Encoder: Input → Hidden → Latent
    - Decoder: Latent → Hidden → Output (reconstruction)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 2):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


print("Autoencoder architecture defined")

# %% [markdown]
# ## 4. Train/Test Split and Preparation

# %%
# Split data
test_start_dt = pd.Timestamp(TEST_START)
train = returns[returns.index < test_start_dt].copy()
test = returns[returns.index >= test_start_dt].copy()

print(f"Train: {len(train):,} observations ({train.index.min()} to {train.index.max()})")
print(f"Test:  {len(test):,} observations ({test.index.min()} to {test.index.max()})")

# Standardize
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Convert to tensors
train_tensor = torch.FloatTensor(train_scaled).to(device)
test_tensor = torch.FloatTensor(test_scaled).to(device)

# DataLoader
train_dataset = TensorDataset(train_tensor, train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Input dimension: {train_scaled.shape[1]}")

# %% [markdown]
# ## 5. Training

# %%
print("\nTraining autoencoder...")

# Initialize model
input_dim = train_scaled.shape[1]
model = CryptoAutoencoder(input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training loop
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        x_hat, _ = model(batch_x)
        loss = criterion(x_hat, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Test loss
    model.eval()
    with torch.no_grad():
        test_hat, _ = model(test_tensor)
        test_loss = criterion(test_hat, test_tensor).item()
        test_losses.append(test_loss)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

print(f"\nFinal Test Loss: {test_losses[-1]:.4f}")

# Plot training curve
fig = go.Figure()
fig.add_trace(go.Scatter(y=train_losses, name="Train Loss"))
fig.add_trace(go.Scatter(y=test_losses, name="Test Loss"))
fig.update_layout(title="Autoencoder Training", xaxis_title="Epoch", yaxis_title="MSE Loss")
fig.show()

# %% [markdown]
# ## 6. Reconstruction Error Analysis

# %%
print("\nComputing reconstruction errors...")

model.eval()
with torch.no_grad():
    # Get reconstructions
    train_hat, train_z = model(train_tensor)
    test_hat, test_z = model(test_tensor)

    # Per-sample reconstruction error (MSE)
    train_recon_error = ((train_tensor - train_hat) ** 2).mean(dim=1).cpu().numpy()
    test_recon_error = ((test_tensor - test_hat) ** 2).mean(dim=1).cpu().numpy()

# Add to DataFrames
train_results = train.copy()
train_results["recon_error"] = train_recon_error
train_results["is_test"] = False

test_results = test.copy()
test_results["recon_error"] = test_recon_error
test_results["is_test"] = True

# Combine
all_results = pd.concat([train_results, test_results])

print("\nReconstruction Error Statistics:")
print(f"  {'Split':<10} {'Mean':<12} {'Std':<12} {'95th pct':<12}")
print("  " + "-" * 46)
print(
    f"  {'Train':<10} {train_recon_error.mean():<12.4f} "
    f"{train_recon_error.std():<12.4f} {np.percentile(train_recon_error, 95):<12.4f}"
)
print(
    f"  {'Test':<10} {test_recon_error.mean():<12.4f} "
    f"{test_recon_error.std():<12.4f} {np.percentile(test_recon_error, 95):<12.4f}"
)

# %% [markdown]
# ## 7. Reconstruction Error vs Volatility

# %%
# Calculate realized volatility (rolling 24h std)
btc_col = [c for c in returns.columns if "BTC" in c][0]
all_results["volatility"] = all_results[btc_col].rolling(24).std()

# Correlation
valid_idx = ~all_results["volatility"].isna()
vol_corr = spearmanr(
    all_results.loc[valid_idx, "recon_error"], all_results.loc[valid_idx, "volatility"]
)[0]

print("\nReconstruction Error vs Volatility:")
print(f"  Spearman correlation: {vol_corr:.3f}")

# Visualization
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("BTC Returns", "Reconstruction Error", "24h Rolling Volatility"),
    vertical_spacing=0.08,
)

# Sample for plot
plot_df = all_results.iloc[-2000:]

fig.add_trace(
    go.Scatter(x=plot_df.index, y=plot_df[btc_col], name="BTC Return", line=dict(width=0.5)),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["recon_error"],
        name="Recon Error",
        line=dict(width=1, color="red"),
    ),
    row=2,
    col=1,
)

# Add anomaly threshold (95th percentile from train)
threshold = np.percentile(train_recon_error, 95)
fig.add_hline(y=threshold, line_dash="dash", line_color="gray", row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["volatility"],
        name="Volatility",
        line=dict(width=1, color="orange"),
    ),
    row=3,
    col=1,
)

fig.update_layout(height=700, title_text="Reconstruction Error vs Market Conditions")
fig.show()

# %% [markdown]
# ## 8. Latent Space Visualization

# %%
print("\nLatent Space Analysis...")

# Get latent representations
train_latent = train_z.cpu().numpy()
test_latent = test_z.cpu().numpy()

# Create latent DataFrame
latent_train = pd.DataFrame(train_latent, columns=["z1", "z2"], index=train.index)
latent_train["split"] = "Train"
latent_train["recon_error"] = train_recon_error

latent_test = pd.DataFrame(test_latent, columns=["z1", "z2"], index=test.index)
latent_test["split"] = "Test"
latent_test["recon_error"] = test_recon_error

latent_all = pd.concat([latent_train, latent_test])

# Add volatility regime
latent_all["volatility"] = all_results["volatility"]
vol_median = latent_all["volatility"].median()
latent_all["regime"] = np.where(latent_all["volatility"] > vol_median, "High Vol", "Low Vol")

# Sample for visualization
plot_latent = latent_all.dropna().iloc[::10]  # Subsample

fig = px.scatter(
    plot_latent,
    x="z1",
    y="z2",
    color="regime",
    opacity=0.5,
    title="Latent Space Colored by Volatility Regime",
    color_discrete_map={"High Vol": "red", "Low Vol": "blue"},
)
fig.update_layout(height=500)
fig.show()

# Latent space by reconstruction error
fig = px.scatter(
    plot_latent,
    x="z1",
    y="z2",
    color="recon_error",
    color_continuous_scale="Reds",
    opacity=0.5,
    title="Latent Space Colored by Reconstruction Error",
)
fig.update_layout(height=500)
fig.show()

# %% [markdown]
# ## 9. Anomaly Detection

# %%
print("\nAnomaly Detection using Reconstruction Error...")

# Define anomaly threshold (95th percentile of train)
anomaly_threshold = np.percentile(train_recon_error, 95)
print(f"  Anomaly threshold (95th pct): {anomaly_threshold:.4f}")

# Identify anomalies
test_results["is_anomaly"] = test_results["recon_error"] > anomaly_threshold
n_anomalies = test_results["is_anomaly"].sum()
anomaly_rate = n_anomalies / len(test_results)

print(f"  Test anomalies: {n_anomalies:,} ({anomaly_rate:.1%})")

# Analyze anomaly characteristics
print("\nAnomaly Characteristics:")
normal_mask = ~test_results["is_anomaly"]
anomaly_mask = test_results["is_anomaly"]

print(f"  {'Metric':<20} {'Normal':<15} {'Anomaly':<15}")
print("  " + "-" * 50)

for col in available_symbols[:3]:
    normal_vol = test_results.loc[normal_mask, col].std()
    anomaly_vol = test_results.loc[anomaly_mask, col].std()
    print(f"  {col[:15]:<20} {normal_vol:<15.3f} {anomaly_vol:<15.3f}")

# %% [markdown]
# ## 10. Per-Asset Reconstruction Quality

# %%
print("\nPer-Asset Reconstruction Quality...")

model.eval()
with torch.no_grad():
    test_hat_np = test_hat.cpu().numpy()

# Inverse transform to original scale
test_recon = scaler.inverse_transform(test_hat_np)
test_orig = test.values

# Per-asset MSE
asset_mse = {}
for i, col in enumerate(test.columns):
    mse = np.mean((test_orig[:, i] - test_recon[:, i]) ** 2)
    asset_mse[col] = mse

# Sort
asset_mse_sorted = sorted(asset_mse.items(), key=lambda x: x[1])

print(f"\n  {'Asset':<12} {'MSE':<12} {'Quality':<10}")
print("  " + "-" * 34)
for asset, mse in asset_mse_sorted:
    quality = "Good" if mse < np.median(list(asset_mse.values())) else "Poor"
    print(f"  {asset:<12} {mse:<12.4f} {quality:<10}")

# %% [markdown]
# ## 11. Summary

# %%
print("\n" + "=" * 60)
print("AUTOENCODER CRYPTO - KEY FINDINGS")
print("=" * 60)

print("\n1. MODEL PERFORMANCE:")
print(f"   Final train loss: {train_losses[-1]:.4f}")
print(f"   Final test loss:  {test_losses[-1]:.4f}")
print(f"   Latent dimension: {LATENT_DIM}")

print("\n2. RECONSTRUCTION ERROR:")
print(f"   Mean (train): {train_recon_error.mean():.4f}")
print(f"   Mean (test):  {test_recon_error.mean():.4f}")
print(f"   Correlation with volatility: {vol_corr:.3f}")

print("\n3. ANOMALY DETECTION:")
print(f"   Threshold (95th pct): {anomaly_threshold:.4f}")
print(f"   Anomaly rate in test: {anomaly_rate:.1%}")
print("   High reconstruction error = unusual market conditions")

print("\n4. LATENT SPACE:")
print("   - 2D latent space captures volatility regime structure")
print("   - High-vol periods cluster separately from low-vol")
print("   - Reconstruction error increases with market stress")

print("\n5. PRACTICAL APPLICATIONS:")
print("   - Use reconstruction error as risk indicator")
print("   - Anomaly threshold for regime change detection")
print("   - Latent factors for portfolio construction")
print("   - Compare to conditional autoencoder for factor estimation")
print("=" * 60)

print("\n[OK] Autoencoder crypto analysis complete")
