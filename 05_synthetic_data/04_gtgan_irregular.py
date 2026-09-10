# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
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
# # Chapter 5: GT-GAN - Neural ODEs for Irregular Time Series
#
# **Docker image**: `ml4t-gpu`
#
# This notebook implements a **GT-GAN-inspired** model (based on Jeon et al., NeurIPS 2022)
# using **Neural ODEs** to handle time series with **naturally irregular timestamps**.
#
# **Learning Objectives**:
# - Understand how Neural ODEs handle irregular time intervals via continuous dynamics
# - Implement a GRU-ODE encoder that processes observations at arbitrary timestamps
# - Train a GAN with ODE-based generator and discriminator for irregular series
# - Evaluate interpolation quality at unobserved timestamps (GT-GAN's unique capability)
# - Compare GT-GAN's irregular-data approach with fixed-grid generators (TimeGAN, Sig-CWGAN)
#
# **Book Reference**: Chapter 5, Section 5.5 (GANs for financial time series) - GT-GAN
# is treated here as a hybrid GAN + continuous-time generator for irregular observations.
#
# **Prerequisites**: Familiarity with GANs (`01_timegan_pytorch.py`) and Chapter 3 bar data.
#
# ## Key Innovation: Genuine Irregularity
#
# Unlike other notebooks that use daily returns on regular grids, GT-GAN is designed
# for **naturally irregular** data. We use Chapter 3's information-theoretic bars:
#
# - **Tick bars**: Fixed number of trades → variable time between bars
# - **Volume bars**: Fixed volume threshold → variable time between bars
# - **Dollar bars**: Fixed dollar volume → variable time between bars
#
# These bars have genuine irregular timestamps driven by market activity, not
# artificial random masking.
#
# ## Neural ODEs: The Key Innovation
#
# Instead of discrete recurrence:
#
# $$h_{t+1} = \text{RNN}(h_t, x_t)$$
#
# Neural ODEs define continuous dynamics:
#
# $$\frac{dh}{dt} = f_\theta(h, t)$$
#
# The hidden state evolves continuously, and we can query it at any time point.
# This naturally handles:
# - Variable time intervals between observations
# - Interpolation to any desired time grid
# - Generation at arbitrary timestamps
#
# ## Model Architecture (Simplified GT-GAN)
#
# GT-GAN uses Neural ODEs to handle continuous-time dynamics:
# 1. **GRU-ODE Encoder**: Maps irregular observations to latent space
# 2. **ODE Generator**: Generates synthetic latent trajectories
# 3. **ODE Discriminator**: Classifies real vs synthetic in latent space
# 4. **ODE Decoder**: Reconstructs observations from latent space
#
# See `figures/figure_5_12_gtgan_architecture.png` for the full architecture diagram.
#
# ## References
#
# - **GT-GAN Paper**: Jeon et al. (2022). "GT-GAN: General Purpose Time Series
#   Synthesis with Generative Adversarial Networks." NeurIPS 2022.
# - **Neural ODEs**: Chen et al. (2018). "Neural Ordinary Differential Equations."
# - **Information Bars**: López de Prado (2018). "Advances in Financial Machine Learning."
#
# ---
#
# ## WARNING: IMPORTANT: Data Source Requirement
#
# **GT-GAN requires naturally irregular data** - that's its core design goal.
#
# | Data Source | Natural Irregularity? | Suitability |
# |-------------|----------------------|-------------|
# | Chapter 3 bar data | [OK] Yes (activity-driven) | [OK] Ideal |
# | Synthetic fallback | WARNING: Simulated | WARNING: Demo only |
# | Daily returns + masking | [FAIL] Artificial | [FAIL] Not recommended |
#
# **If Chapter 3 bar data is unavailable**, this notebook falls back to generating
# synthetic bars with log-normal inter-arrival times. While this demonstrates the
# Neural ODE architecture, the statistical properties are artificial.
#
# **For production use**: Ensure Chapter 3's microstructure bar data is available.
# Run Chapter 3 notebooks first to generate tick/volume/dollar bars.
#
# ### What this run reports
#
# Three numbers say how the run went, all printed in the evaluation cells below:
# reconstruction MSE for the autoencoder round-trip, the KS statistic against the
# real marginal, and the smoothness ratio of the ODE interpolation against the real
# bars. This is a short training pass over a small sample, so read them as a
# demonstration of the architecture rather than as achievable quality.
#
# ---
#
# ## Per-Sample Time Integration
#
# `ode_method` selects one of two fixed-step solvers, and both advance each row of a
# batch by that row's own interval:
#
# - **Euler**: one evaluation per step, `h_new = h + dt * f(h)`.
# - **RK4**: four evaluations per step, combined as `h + (dt / 6)(k1 + 2 k2 + 2 k3 + k4)`.
#
# Neither drift network takes the time as an input, so a step depends only on the width
# of the interval and a column of `dt` values is enough to give every row its own step.
# That is what `ode_evolve` below does. `torchdiffeq.odeint` is still used for the
# generator, whose whole batch genuinely shares one time span; it takes a single grid
# for a batched initial state, so it cannot be used where the rows differ.
#
# An adaptive solver would need per-sample step control. The GT-GAN paper used
# `torchode` (Lienen & Günnemann, 2022) for that, but its `torchtyping` dependency is
# incompatible with current PyTorch, so the config cell rejects any method other than
# the two above.
#
# **Reference**: Lienen, M. & Günnemann, S. (2022). "torchode: A Parallel ODE
# Solver for PyTorch." https://arxiv.org/abs/2210.12375

# %%
"""GT-GAN — Neural ODE-based generative model for irregular time series."""

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import Image, display
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
from tqdm import tqdm

from utils.paths import get_chapter_dir, get_output_dir
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, plot_fidelity_comparison, show_plotly_with_alt, show_with_alt

# %% [markdown]
# ## GT-GAN Architecture
#
# GT-GAN uses Neural ODEs to handle continuous-time dynamics with genuine irregularity.
# Unlike discrete RNNs, the hidden state evolves continuously and can be queried at
# any time point.
# %%
ASSETS_DIR = get_chapter_dir(5) / "assets"
if (ASSETS_DIR / "gtgan_architecture.jpeg").exists():
    display(Image(ASSETS_DIR / "gtgan_architecture.jpeg", width=800))


# Checkpoint path for model persistence
CHECKPOINT_PATH = get_output_dir(5, "gtgan") / "checkpoints" / "gtgan_model.pt"

# %% tags=["parameters"]
# Paper-faithful defaults (Jeon et al., NeurIPS 2022)
SEQ_LENGTH = 32
LATENT_DIM = 32
HIDDEN_DIM = 24
ODE_HIDDEN = 24
MAX_STEPS = 2000
BATCH_SIZE = 128
ODE_METHOD = "rk4"  # "euler" or "rk4"; both step each sample by its own dt
N_BARS = 2000  # Fallback synthetic bar count (when Ch3 data unavailable)
RETRAIN = False  # Set True to retrain even if checkpoint exists
SEED = 42

# The reconstruction MSE this notebook treats as a well-behaved autoencoder. It is a
# reading aid for the printed number, not a value the paper reports.
GOOD_RECONSTRUCTION_MSE = 0.01

# Progress bars write to stderr and papermill records every repaint, so they are off by
# default and the training loop prints the same numbers to stdout instead. Set True to
# watch a long run interactively.
PROGRESS_BARS = False

# %%
set_global_seeds(SEED)

# %%
# Configuration for naturally irregular Chapter 3 bar data
CONFIG = {
    "bar_type": "dollar",
    "seq_length": SEQ_LENGTH,
    "features": ["close", "volume"],
    "latent_dim": LATENT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "ode_hidden": ODE_HIDDEN,
    "max_steps": MAX_STEPS,
    "batch_size": BATCH_SIZE,
    "learning_rate": 1e-3,
    "ode_method": ODE_METHOD,
    "holdout_fraction": 0.2,
    # Bump when the architecture changes: nothing else in a checkpoint says what the
    # weights were fitted into, so older ones load whenever the shapes line up.
    # 2.1.0: the decoder integrates each sequence along its own timestamps.
    "weights_version": "2.2.0",
}

if CONFIG["ode_method"] not in {"euler", "rk4"}:
    raise ValueError(
        f"ode_method must be 'euler' or 'rk4', got {CONFIG['ode_method']!r}. An "
        "adaptive solver would pick its step sizes from one grid for the whole "
        "batch, which is what the irregular timestamps here are meant to avoid."
    )

print(f"GT-GAN: Steps={CONFIG['max_steps']}, ODE={CONFIG['ode_method']}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %% [markdown]
# ## Data: Chapter 3 Information-Theoretic Bars
#
# The key insight is that GT-GAN's Neural ODE architecture is designed for
# **genuinely irregular timestamps**. Information bars from Chapter 3 provide this:
#
# | Bar Type | Trigger Condition | Inter-arrival Time |
# |----------|------------------|-------------------|
# | Tick bars | Every N trades | Variable (activity-driven) |
# | Volume bars | Every V shares | Variable (liquidity-driven) |
# | Dollar bars | Every $D traded | Variable (value-driven) |
#
# This is fundamentally different from daily returns with artificial masking.


# %%
def generate_synthetic_irregular_bars(n_bars: int = 1000, seed: int = 42) -> pl.DataFrame:
    """
    Generate synthetic bars with naturally irregular timestamps.

    Used as fallback when Chapter 3 bar data is not available.
    Simulates the statistical properties of information-theoretic bars.

    WARNING: This is DEMO MODE ONLY. Synthetic bars have artificial
    statistical properties. For production use, run Chapter 3 notebooks
    first to generate real tick/volume/dollar bars from market data.
    """
    print("\n" + "=" * 70)
    print("WARNING: DEMO MODE: Using synthetic bars (Chapter 3 data unavailable)")
    print("    Results are illustrative only. For production:")
    print("    1. Run Chapter 3 microstructure notebooks first")
    print("    2. Generate tick/volume/dollar bars from real market data")
    print("=" * 70 + "\n")
    rng = np.random.default_rng(seed)

    # Simulate irregular inter-arrival times (log-normal, like real bars)
    dt_mean = 60  # mean 60 seconds between bars
    dt_std = 30  # high variance (realistic for dollar bars)
    dt = rng.lognormal(np.log(dt_mean), np.log(1 + dt_std / dt_mean), n_bars)

    # Cumulative timestamps
    timestamps = np.cumsum(dt)
    base_time = datetime(2024, 1, 2, 9, 30, tzinfo=UTC)  # Market open
    timestamps_dt = [base_time + timedelta(seconds=int(t)) for t in timestamps]

    # Simulate OHLCV (geometric Brownian motion)
    returns = rng.normal(0.0001, 0.001, n_bars)  # Small returns per bar
    prices = 100 * np.exp(np.cumsum(returns))

    # OHLC from close prices with noise
    close = prices
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n_bars))

    # Volume (correlated with price movement)
    volume = rng.exponential(1000, n_bars) * (1 + 10 * np.abs(returns))

    df = pl.DataFrame(
        {
            "timestamp": timestamps_dt,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(int),
        }
    )
    print(f"Generated {len(df)} synthetic irregular bars (Chapter 3 data unavailable)")
    return df


# %% [markdown]
# ### Load Chapter 3 Bar Data
#
# Loads information-theoretic bars generated in Chapter 3's microstructure notebooks.
# Falls back to synthetic bars if the data is unavailable.


# %%
def load_chapter3_bars(bar_type: str = "dollar") -> pl.DataFrame:
    """
    Load information-theoretic bars from Chapter 3 output.

    Args:
        bar_type: One of 'tick', 'volume', 'dollar'

    Returns:
        DataFrame with OHLCV and timestamp columns
    """
    if bar_type not in ("tick", "volume", "dollar"):
        raise ValueError(f"bar_type must be one of 'tick', 'volume', 'dollar' (got {bar_type!r})")
    # Chapter 3 Databento notebook saves bars to get_output_dir(3, "databento")
    bar_dir = get_output_dir(3, "databento")

    bar_path = bar_dir / f"NVDA_{bar_type}_bars.parquet"
    if not bar_path.exists():
        # Fallback to synthetic data when Chapter 3 hasn't been run
        return generate_synthetic_irregular_bars(n_bars=N_BARS)

    df = pl.read_parquet(bar_path)
    print(f"Loaded {len(df)} {bar_type} bars from Chapter 3")
    print(f"Columns: {df.columns}")

    return df


# %% [markdown]
# ### Compute Inter-Arrival Times
#
# Normalizes bar timestamps to the [0, 1] range for ODE integration. The variable
# spacing between bars is the key signal that GT-GAN's Neural ODE exploits.


# %%
def compute_inter_arrival_times(df: pl.DataFrame) -> np.ndarray:
    """
    Compute normalized inter-arrival times from bar timestamps.

    Returns times in [0, 1] range for ODE integration.
    """
    # Get timestamps in nanoseconds
    timestamps = df.select("timestamp").to_numpy().flatten()

    # Convert to numeric (nanoseconds since epoch)
    if hasattr(timestamps[0], "value"):
        # datetime64 objects
        times_ns = np.array([t.value for t in timestamps], dtype=np.float64)
    else:
        # Already numeric
        times_ns = timestamps.astype(np.float64)

    # Normalize to [0, 1] for ODE integration
    times_norm = (times_ns - times_ns.min()) / (times_ns.max() - times_ns.min() + 1e-10)

    return times_norm.astype(np.float32)


# %% [markdown]
# ### Create Irregular Sequences
#
# Slides a window over the bar data to create training sequences, preserving
# the per-sequence normalized timestamps that encode irregular spacing.


# %%
def create_irregular_sequences(
    df: pl.DataFrame, features: list[str], seq_length: int, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with their associated irregular timestamps.

    Args:
        df: DataFrame with bar data
        features: Column names to use as features
        seq_length: Number of bars per sequence
        times: Normalized timestamps for each bar

    Returns:
        sequences: Shape (n_seq, seq_length, n_features)
        seq_times: Shape (n_seq, seq_length) - normalized times per sequence
    """
    # Extract feature matrix
    data = df.select(features).to_numpy().astype(np.float32)

    n_sequences = len(data) - seq_length + 1
    n_features = len(features)

    sequences = np.zeros((n_sequences, seq_length, n_features), dtype=np.float32)
    seq_times = np.zeros((n_sequences, seq_length), dtype=np.float32)

    for i in range(n_sequences):
        sequences[i] = data[i : i + seq_length]
        # Normalize times within each sequence to [0, 1]
        t_seq = times[i : i + seq_length]
        t_norm = (t_seq - t_seq.min()) / (t_seq.max() - t_seq.min() + 1e-10)
        # Ensure strictly increasing (add small epsilon for duplicate timestamps)
        for j in range(1, len(t_norm)):
            if t_norm[j] <= t_norm[j - 1]:
                t_norm[j] = t_norm[j - 1] + 1e-6
        seq_times[i] = t_norm

    return sequences, seq_times


# %% [markdown]
# ### Prepare Training Data
#
# Load bars, compute inter-arrival times, split into train/holdout, and
# create normalized sequences for model input.

# %%
# Load data
df = load_chapter3_bars(CONFIG["bar_type"])
global_times = compute_inter_arrival_times(df)

# Compute inter-arrival statistics
dt = np.diff(global_times)
print("\nInter-arrival time statistics (normalized [0,1]):")
print(f"  Mean: {dt.mean():.6f}, Std: {dt.std():.6f}")
print(f"  Min: {dt.min():.6f}, Max: {dt.max():.6f}")
print(f"  CV (std/mean): {dt.std() / dt.mean():.2f}")

# Train/holdout split (temporal)
n_total = len(df)
n_holdout = int(n_total * CONFIG["holdout_fraction"])
n_train = n_total - n_holdout

df_train = df.head(n_train)
df_holdout = df.tail(n_holdout)
times_train = global_times[:n_train]
times_holdout = global_times[n_train:]

print("\nTrain/Holdout split:")
print(f"  Training: {n_train} bars")
print(f"  Holdout: {n_holdout} bars")

# Create sequences from training data
sequences, seq_times = create_irregular_sequences(
    df_train, CONFIG["features"], CONFIG["seq_length"], times_train
)
n_features = len(CONFIG["features"])

# Normalize features to [0, 1] for stable training
seq_min = sequences.min(axis=(0, 1), keepdims=True)
seq_max = sequences.max(axis=(0, 1), keepdims=True)
sequences_norm = (sequences - seq_min) / (seq_max - seq_min + 1e-8)

print(f"\nCreated {len(sequences)} training sequences")
print(f"Sequence shape: {sequences.shape}")
print(f"Time shape: {seq_times.shape}")

# Store normalization params
norm_params = {"min": seq_min.squeeze(), "max": seq_max.squeeze()}

# CONFIG names the features and the window, not the bars themselves, and the bars
# come from Chapter 3 when it has been run and from a synthetic fallback when it
# has not. A digest of them is what tells those two runs apart.
_data_digest = hashlib.sha256(np.ascontiguousarray(sequences))
_data_digest.update(np.ascontiguousarray(seq_times))
CONFIG["data_digest"] = _data_digest.hexdigest()[:16]


# %% [markdown]
# ## Neural ODE Components
#
# The core building block is a neural network that defines the derivative:
#
# $$\frac{dh}{dt} = f_\theta(h)$$
#
# Given an initial state $h(t_0)$, the ODE solver integrates forward to compute:
#
# $$h(t) = h(t_0) + \int_{t_0}^{t} f_\theta(h(s)) \, ds$$
#
# This allows querying the hidden state at **any** time point, not just discrete steps.


# %% [markdown]
# ### ODEFunc: Dynamics Network
#
# Defines the derivative $dh/dt = f_\theta(h)$ used by `torchdiffeq`. This is the
# core "physics" that governs how latent states evolve continuously in time.


# %%
class ODEFunc(nn.Module):
    """
    Neural network defining the ODE dynamics: dh/dt = f(h, t).

    This is the "physics" of our latent space - how states evolve over time.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute derivative dh/dt (torchdiffeq interface)."""
        return self.net(h)


# %% [markdown]
# ### Advancing a batch by its own inter-arrival times
#
# `torchdiffeq.odeint` integrates one time grid for the whole batch, so a batch of
# sequences observed at different times has to be reduced to a single grid before it
# can be passed. That reduction is what the irregular timestamps are supposed to
# carry, so the step below takes a `dt` per row instead. Both drift networks ignore
# $t$, which is what makes a per-row step size well defined: over an interval the
# state change depends on the width of the interval and not on where it sits.


# %%
def ode_evolve(func: nn.Module, h: torch.Tensor, dt: torch.Tensor, method: str) -> torch.Tensor:
    """
    Advance each row of ``h`` by its own ``dt``.

    Args:
        func: Drift network with the ``(t, y)`` signature; must ignore ``t``
        h: State, shape (batch, hidden_dim)
        dt: Step width per row, shape (batch,)
        method: "euler" (one derivative evaluation) or "rk4" (four)

    Returns:
        State after the step, same shape as ``h``
    """
    dt = dt.unsqueeze(-1)
    zero = torch.zeros((), device=h.device)

    if method == "euler":
        return h + dt * func(zero, h)

    k1 = func(zero, h)
    k2 = func(zero, h + 0.5 * dt * k1)
    k3 = func(zero, h + 0.5 * dt * k2)
    k4 = func(zero, h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# %% [markdown]
# ### GRU-ODE Cell: Continuous-Time GRU
#
# Combines discrete GRU updates (at observation times) with continuous ODE evolution
# (between observations). Each sample's hidden state evolves according to its own
# inter-arrival time -- no batch averaging. Based on De Brouwer et al. (2019).


# %%
class GRUODECell(nn.Module):
    """
    GRU-ODE: A continuous-time version of GRU with per-sample dt handling.

    When an observation arrives, update the hidden state using GRU gates.
    Between observations, evolve the state using ``ode_evolve``, so each sample
    steps by its own inter-arrival time.

    Reference: De Brouwer et al. (2019) "GRU-ODE-Bayes"
    """

    def __init__(self, input_dim: int, hidden_dim: int, ode_method: str = "euler"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_method = ode_method

        # GRU gates for observation updates
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # ODE function for continuous evolution
        self.ode_func = ODEFunc(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process one timestep: ODE evolution + GRU update.

        Args:
            x: Observation, shape (batch, input_dim)
            h: Previous hidden state, shape (batch, hidden_dim)
            dt_per_sample: Time since last step for each sample, shape (batch,)

        Returns:
            Updated hidden state
        """
        # A row whose dt is zero is left where it is: every term of the step
        # carries a dt factor.
        h = ode_evolve(self.ode_func, h, dt_per_sample, self.ode_method)

        # GRU update at observation time
        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        h_tilde = torch.tanh(self.W_h(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * h_tilde

        return h_new


# %% [markdown]
# ## Encoder and Decoder


# %%
class ODEEncoder(nn.Module):
    """
    Encode irregular time series to latent representation using GRU-ODE.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, ode_method: str = "euler"):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru_ode = GRUODECell(input_dim, hidden_dim, ode_method=ode_method)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode irregular sequence to latent distribution.

        Args:
            x: Observations, shape (batch, seq_len, input_dim)
            times: Observation times, shape (batch, seq_len)

        Returns:
            mu, logvar: Latent distribution parameters
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_dim, device=device)

        for t in range(seq_len):
            # Per-sample dt (no batch averaging)
            if t > 0:
                dt_per_sample = times[:, t] - times[:, t - 1]
            else:
                dt_per_sample = torch.zeros(batch_size, device=device)
            # Clamp to non-negative
            dt_per_sample = torch.clamp(dt_per_sample, min=0.0)
            h = self.gru_ode(x[:, t], h, dt_per_sample)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


# %% [markdown]
# ### ODE Decoder
#
# Decodes latent representations back to time series observations at given
# query times using a Neural ODE to evolve hidden states continuously.


# %%
class ODEDecoder(nn.Module):
    """
    Decode latent representation to time series using Neural ODE.
    """

    def __init__(
        self, latent_dim: int, hidden_dim: int, output_dim: int, ode_method: str = "euler"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_method = ode_method

        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to observations at given times.

        Args:
            z: Latent code, shape (batch, latent_dim)
            times: Query times, shape (batch, n_times)

        Returns:
            Observations, shape (batch, n_times, output_dim)
        """
        h = torch.tanh(self.fc_init(z))

        # Each row is integrated along its own query times from a fixed origin at zero,
        # where create_irregular_sequences puts the start of every window; anchoring at
        # the first query leaves a midpoint decode incomparable with an observation one.
        prev = torch.zeros_like(times[:, 0])

        states = []
        for k in range(times.shape[1]):
            dt = torch.clamp(times[:, k] - prev, min=0.0)
            h = ode_evolve(self.ode_func, h, dt, self.ode_method)
            states.append(h)
            prev = times[:, k]

        return self.fc_out(torch.stack(states, dim=1))


# %% [markdown]
# ## Generator and Discriminator
#
# The GAN components operate in latent space. The generator maps noise through a
# Neural ODE to produce latent codes, while the discriminator uses the same GRU-ODE
# architecture to classify sequences as real or synthetic.

# %% [markdown]
# ### ODE Generator
#
# Maps random noise to latent codes by evolving initial states through
# a Neural ODE. The ODE trajectory acts as a learned nonlinear transformation.


# %%
class ODEGenerator(nn.Module):
    """Generate latent trajectories from noise using Neural ODE."""

    def __init__(self, noise_dim: int, latent_dim: int, hidden_dim: int, ode_method: str = "euler"):
        super().__init__()
        self.noise_dim = noise_dim
        self.ode_method = ode_method

        self.fc_init = nn.Linear(noise_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate latent code from noise."""
        device = z.device

        h0 = torch.tanh(self.fc_init(z))
        t_span = torch.linspace(0, 1, 5, device=device)
        h_trajectory = odeint(self.ode_func, h0, t_span, method=self.ode_method)
        h_final = h_trajectory[-1]

        return self.fc_out(h_final)


# %% [markdown]
# ### ODE Discriminator
#
# Processes sequences through GRU-ODE (respecting irregular timestamps) and
# outputs a real/fake classification. The time-awareness is what distinguishes
# this from a standard GRU discriminator.


# %%
class ODEDiscriminator(nn.Module):
    """Discriminate real vs fake sequences using ODE-based processing."""

    def __init__(self, input_dim: int, hidden_dim: int, ode_method: str = "euler"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_ode = GRUODECell(input_dim, hidden_dim, ode_method=ode_method)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Classify sequence as real or fake."""
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.hidden_dim, device=device)

        for t in range(seq_len):
            # Per-sample dt (no batch averaging)
            if t > 0:
                dt_per_sample = times[:, t] - times[:, t - 1]
            else:
                dt_per_sample = torch.zeros(batch_size, device=device)
            dt_per_sample = torch.clamp(dt_per_sample, min=0.0)
            h = self.gru_ode(x[:, t], h, dt_per_sample)

        return self.fc_out(h)


# %% [markdown]
# ## GT-GAN Model


# %%
class GTGAN(nn.Module):
    """
    GT-GAN-inspired model for time series with irregular timestamps.

    Key difference from TimeGAN: explicitly models variable dt between observations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        noise_dim: int,
        ode_method: str = "euler",
    ):
        super().__init__()

        self.encoder = ODEEncoder(input_dim, hidden_dim, latent_dim, ode_method=ode_method)
        self.decoder = ODEDecoder(latent_dim, hidden_dim, input_dim, ode_method=ode_method)
        self.generator = ODEGenerator(noise_dim, latent_dim, hidden_dim, ode_method=ode_method)
        self.discriminator = ODEDiscriminator(input_dim, hidden_dim, ode_method=ode_method)

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

    def encode(self, x: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, times)

    def decode(self, z: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, times)

    def generate(self, batch_size: int, times: torch.Tensor, device: torch.device) -> torch.Tensor:
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        z = self.generator(noise)
        return self.decode(z, times)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# %% [markdown]
# ### Reusing a checkpoint, and when not to
#
# Training is the expensive step, so a saved checkpoint is loaded when one matches. What
# counts as a match is the question. `CHECKPOINT_IDENTITY` below names every setting that
# moves the weights, including `data_digest`, because `CONFIG` names the features and the
# window and not the bars they are cut from.
#
# The digest matters most on the path a reader is most likely to take. This notebook falls
# back to synthetic bars when the Chapter 3 outputs are absent, so a first run trains on
# the fallback and a later run, after Chapter 3 has been executed, sees real bars. Without
# the digest that second run loads the fallback-trained weights. It also takes the
# checkpoint's scaler, while `sequences_norm` above was scaled from the new bars, which
# would leave the holdout comparison and the training data on different scales. A digest
# mismatch retrains instead, which is the only answer that keeps the two consistent.


# %%
# Initialize model
model = GTGAN(
    input_dim=n_features,
    hidden_dim=CONFIG["hidden_dim"],
    latent_dim=CONFIG["latent_dim"],
    noise_dim=CONFIG["latent_dim"],
    ode_method=CONFIG["ode_method"],
).to(device)

print(f"GT-GAN parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check for existing checkpoint
SKIP_TRAINING = False
_saved = (
    torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if CHECKPOINT_PATH.exists() and not RETRAIN
    else None
)
if _saved is not None and "history" not in _saved:
    print(
        f"\nCheckpoint at {CHECKPOINT_PATH} predates loss-history saving; retraining so "
        "the training-progress section has its figure."
    )
    _saved = None

# Every entry here changes the weights, so a checkpoint fitted under a different value
# is not this run's model. Several are papermill parameters, so without this check a
# second setting would load the first one's weights whenever the shapes still matched.
CHECKPOINT_IDENTITY = (
    "seq_length",
    "features",
    "latent_dim",
    "hidden_dim",
    "ode_hidden",
    "max_steps",
    "batch_size",
    "learning_rate",
    "ode_method",
    "holdout_fraction",
    "weights_version",
    "data_digest",
)
if _saved is not None:
    saved_config = _saved.get("config", {})
    mismatched = {
        key: (saved_config.get(key), CONFIG[key])
        for key in CHECKPOINT_IDENTITY
        if saved_config.get(key) != CONFIG[key]
    }
    if mismatched:
        print(f"\nCheckpoint at {CHECKPOINT_PATH} was fitted under different settings:")
        for key, (was, now) in mismatched.items():
            print(f"  {key}: checkpoint {was!r}, this run {now!r}")
        print("Retraining.")
        _saved = None
if _saved is not None:
    print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = _saved
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.generator.load_state_dict(checkpoint["generator"])
    model.discriminator.load_state_dict(checkpoint["discriminator"])
    norm_params = {
        "min": np.array(checkpoint["scaler"]["min"]),
        "max": np.array(checkpoint["scaler"]["max"]),
    }
    training_losses = checkpoint["history"]
    print("Checkpoint loaded successfully - skipping training")
    SKIP_TRAINING = True
else:
    if RETRAIN:
        print("\nRETRAIN=True: Training from scratch")
    else:
        print(f"\nNo checkpoint found at: {CHECKPOINT_PATH}")
        print("Training from scratch...")


# %% [markdown]
# ## Training Loop


# %%
def train_gtgan(
    model: GTGAN,
    sequences: np.ndarray,
    times: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Train GT-GAN model on irregular time series.

    Per Jeon et al. (NeurIPS 2022): Step-based training (10000 steps)
    instead of epoch-based.
    """
    opt_ae = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=config["learning_rate"],
    )
    opt_g = optim.Adam(model.generator.parameters(), lr=config["learning_rate"])
    opt_d = optim.Adam(model.discriminator.parameters(), lr=config["learning_rate"])

    seq_tensor = torch.FloatTensor(sequences)
    time_tensor = torch.FloatTensor(times)
    dataset = TensorDataset(seq_tensor, time_tensor)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    losses = {"recon": [], "d_real": [], "d_fake": [], "g": []}

    # Step-based training (not epoch-based) per Jeon et al.
    total_steps = config.get("max_steps", 10000)

    print(f"\nTraining GT-GAN for {total_steps} steps...")
    print(f"  Sequences: {len(sequences)}, Features: {sequences.shape[-1]}")
    print(f"  Batch size: {config['batch_size']}, Learning rate: {config['learning_rate']}")

    # Create infinite iterator over batches
    def infinite_dataloader():
        while True:
            yield from dataloader

    data_iter = infinite_dataloader()
    pbar = tqdm(
        range(total_steps), desc=f"Training ({total_steps} steps)", disable=not PROGRESS_BARS
    )

    for step in pbar:
        batch_seq, batch_time = next(data_iter)
        batch_seq = batch_seq.to(device)
        batch_time = batch_time.to(device)
        batch_size = batch_seq.shape[0]

        # Phase 1: Autoencoder
        opt_ae.zero_grad()
        mu, logvar = model.encode(batch_seq, batch_time)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z, batch_time)
        recon_loss = nn.functional.mse_loss(recon, batch_seq)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        ae_loss = recon_loss + 0.1 * kl_loss
        ae_loss.backward()
        opt_ae.step()

        # Phase 2: Discriminator
        opt_d.zero_grad()
        d_real = model.discriminator(batch_seq, batch_time)
        loss_d_real = nn.functional.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        )
        with torch.no_grad():
            fake_seq = model.generate(batch_size, batch_time, device)
        d_fake = model.discriminator(fake_seq, batch_time)
        loss_d_fake = nn.functional.binary_cross_entropy_with_logits(
            d_fake, torch.zeros_like(d_fake)
        )
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_d.step()

        # Phase 3: Generator
        opt_g.zero_grad()
        fake_seq = model.generate(batch_size, batch_time, device)
        d_fake = model.discriminator(fake_seq, batch_time)
        loss_g = nn.functional.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        loss_g.backward()
        opt_g.step()

        # Record losses
        losses["recon"].append(recon_loss.item())
        losses["d_real"].append(loss_d_real.item())
        losses["d_fake"].append(loss_d_fake.item())
        losses["g"].append(loss_g.item())

        if step % max(1, total_steps // 10) == 0:
            report = {
                "Recon": f"{recon_loss.item():.4f}",
                "D_real": f"{loss_d_real.item():.3f}",
                "G": f"{loss_g.item():.3f}",
            }
            pbar.set_postfix(report)
            # The progress bar is the only place set_postfix shows up, so with bars off
            # the same numbers go to stdout and reach the render.
            if not PROGRESS_BARS:
                fields = "  ".join(f"{k}: {v}" for k, v in report.items())
                print(f"  Step {step}/{total_steps}  {fields}", flush=True)

    print("Training complete!")
    return losses


# %% [markdown]
# ### Run Training
#
# Execute step-based training. Progress is logged at regular intervals.

# %%
if not SKIP_TRAINING:
    training_losses = train_gtgan(model, sequences_norm, seq_times, CONFIG, device)

    # Save checkpoint after training
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
        "generator": model.generator.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "scaler": {
            "min": norm_params["min"].tolist(),
            "max": norm_params["max"].tolist(),
        },
        "config": CONFIG,
        # The training-progress figure plots these. Without them a checkpoint-loading
        # run renders that section with no figure at all.
        "history": training_losses,
    }
    torch.save(checkpoint_data, CHECKPOINT_PATH)
    print(f"\nCheckpoint saved to: {CHECKPOINT_PATH}")


# %% [markdown]
# ## Training Progress


# %%
if training_losses["recon"]:  # Only plot if we have training losses
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Reconstruction Loss", "Adversarial Losses"]
    )

    fig.add_trace(
        go.Scatter(y=training_losses["recon"], mode="lines", name="Reconstruction"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(y=training_losses["d_real"], mode="lines", name="D (real)"), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=training_losses["d_fake"], mode="lines", name="D (fake)"), row=1, col=2
    )
    fig.add_trace(go.Scatter(y=training_losses["g"], mode="lines", name="Generator"), row=1, col=2)

    fig.update_layout(
        title_text="GT-GAN Training Progress (Naturally Irregular Timestamps)",
        template="ml4t",
        height=400,
    )
    show_plotly_with_alt(
        fig,
        "Two panels of GT-GAN training curves against step. The left panel shows the "
        "reconstruction loss dropping steeply over the first few hundred steps and "
        "then running flat near zero. The right panel shows the discriminator's losses "
        "on real and on fake sequences together with the generator loss; all three "
        "spike against one another in bursts through roughly the first half of training "
        "and then settle onto a common flat level that holds to the end.",
    )

# %% [markdown]
# **Interpretation**: The reconstruction loss should decrease steadily, indicating the
# autoencoder learns to compress and reconstruct irregular sequences through the ODE
# bottleneck. The adversarial losses (D real, D fake, Generator) should oscillate and
# roughly balance -- if the discriminator dominates (D losses near 0), the generator
# receives no useful gradient. The notebook prints the reconstruction MSE against
# `GOOD_RECONSTRUCTION_MSE` below; when it clears that bar, the GAN component is
# refining a latent space the autoencoder has already learned to invert.

# %% [markdown]
# ## Generate Synthetic Irregular Sequences
#
# Training draws random numbers and loading a checkpoint draws none, so the two paths
# reach this point with different torch RNG states. The latent noise the generator is
# fed, and the initializations inside the evaluation protocol, would then differ between
# the run that fitted the weights and a later run that loaded the same weights back:
# same model, same data, same seed, different reported scores. That was measured, not
# assumed: before this reseed the training path and the loading path returned different
# TSTR ratios from one set of weights. Reseeding here is what makes them agree.

# %%
set_global_seeds(SEED)


# %%
def generate_synthetic(
    model: GTGAN, n_samples: int, times: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Generate synthetic sequences at given time points."""
    model.eval()
    with torch.no_grad():
        synthetic = model.generate(n_samples, times, device)
    model.train()
    return synthetic.cpu().numpy()


# %% [markdown]
# ### Choosing the sequences to evaluate against
#
# The sequences are rolling windows at stride one, so the first N of them start on N
# consecutive bars: one stretch of the tape repeated, not N samples of it, and a quiet
# or a violent week would stand in for the whole period. Everything below compares
# against `real_eval`, drawn at random, and the synthetic sequences are generated on
# that same draw's time grids so the two stay paired.

# %%
N_SYNTHETIC = min(200, len(sequences_norm))
eval_rng = np.random.default_rng(SEED)
eval_idx = eval_rng.choice(len(sequences_norm), size=N_SYNTHETIC, replace=False)
real_eval = sequences_norm[eval_idx]
sample_times = torch.FloatTensor(seq_times[eval_idx]).to(device)
synthetic_sequences = generate_synthetic(model, N_SYNTHETIC, sample_times, device)

print(f"\nGenerated {len(synthetic_sequences)} synthetic sequences")
print(f"Shape: {synthetic_sequences.shape}")


# %% [markdown]
# ## Evaluation
#
# ### Fidelity: Visual Comparison with PCA and t-SNE
#
# We project both real and synthetic sequences into 2D to assess whether the
# generator covers the same regions of the data manifold.

# %%
fig = plot_fidelity_comparison(
    real_eval,
    synthetic_sequences,
    title="GT-GAN: Real vs Synthetic Distribution",
    n_samples=min(200, N_SYNTHETIC),
    flatten_method="flatten",  # Flatten for irregular sequence comparison
)
show_with_alt(
    fig,
    "Two scatter panels of the same flattened sequences, PCA on the left and t-SNE on "
    "the right, each overlaying real points and synthetic ones. In both panels the "
    "synthetic points lie along a narrow band while the real points fall away from it. "
    "In the PCA panel the band is horizontal, at one height and covering part of the "
    "first component's range, with the real points scattered across the whole panel. In "
    "the t-SNE panel it is a diagonal band falling from left to right, and the real "
    "points form separate clusters above and to the left of it and one cluster above "
    "its right end; almost none sit below it.",
)

# %% [markdown]
# **Interpretation**: the two point clouds do not cover the same region. The synthetic
# sequences project onto a narrow band in both panels while the real ones fall elsewhere,
# which is what a generator producing a family of similar sequences looks like under a
# projection. Neither panel is a measurement: a projection into two dimensions can
# separate sets a model cannot, and can hide a difference a model finds easily. The
# discriminative score reported later in the notebook is the measured version of the
# same question, and the interpolation section immediately below asks a different one -
# whether the decoder returns sensible values at times it was not given.

# %% [markdown]
# ### Interpolation at Arbitrary Timestamps
#
# GT-GAN's key capability: query the model at any timestamp, not just observed ones.


# %%
def evaluate_interpolation(
    model: GTGAN,
    sequences: np.ndarray,
    times: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    """
    Evaluate model's ability to interpolate at arbitrary timestamps.

    We encode a sequence, then decode at:
    1. Original timestamps (reconstruction)
    2. Midpoint timestamps (interpolation)
    """
    model.eval()

    n_test = min(50, len(sequences))
    test_idx = rng.choice(len(sequences), size=n_test, replace=False)
    test_seq = torch.FloatTensor(sequences[test_idx]).to(device)
    test_time = torch.FloatTensor(times[test_idx]).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(test_seq, test_time)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z, test_time)

        # Interpolate at midpoints
        midpoint_times = (test_time[:, :-1] + test_time[:, 1:]) / 2
        interp = model.decode(z, midpoint_times)

    recon = recon.cpu().numpy()
    interp = interp.cpu().numpy()
    test_seq_np = test_seq.cpu().numpy()

    # Reconstruction error
    recon_mse = np.mean((recon - test_seq_np) ** 2)

    # Interpolation should be "between" adjacent values (smoothness check)
    interp_smoothness = np.mean(np.abs(np.diff(interp, axis=1)))
    real_smoothness = np.mean(np.abs(np.diff(test_seq_np, axis=1)))

    model.train()

    return {
        "reconstruction_mse": recon_mse,
        "interpolation_smoothness": interp_smoothness,
        "real_smoothness": real_smoothness,
        "smoothness_ratio": interp_smoothness / (real_smoothness + 1e-8),
    }


interp_results = evaluate_interpolation(model, sequences_norm, seq_times, device, eval_rng)

print("\n=== Interpolation Evaluation ===")
recon = interp_results["reconstruction_mse"]
print(f"Reconstruction MSE: {recon:.6f}")
print(
    f"  {'clears' if recon < GOOD_RECONSTRUCTION_MSE else 'above'} the "
    f"GOOD_RECONSTRUCTION_MSE bar of {GOOD_RECONSTRUCTION_MSE}"
)
print(f"Interpolation smoothness: {interp_results['interpolation_smoothness']:.6f}")
print(f"Real data smoothness: {interp_results['real_smoothness']:.6f}")
print(f"Smoothness ratio (interp/real): {interp_results['smoothness_ratio']:.2f}")

# %% [markdown]
# **Interpretation**: a smoothness ratio near one means the ODE-based interpolation
# varies from step to step about as much as the real data does, so the model has learned
# continuous dynamics rather than averaging adjacent points. A ratio well below one is
# over-smoothing, the ODE being too rigid to follow the data; a ratio above one is noisy
# interpolation. The reconstruction MSE measures how faithfully the encode-decode
# round-trip recovers the input, and is printed above beside the bar it is judged against.

# %% [markdown]
# ## Statistical Comparison


# %%
def evaluate_statistics(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """Compare statistical properties."""
    real_flat = real.reshape(-1, real.shape[-1])
    syn_flat = synthetic.reshape(-1, synthetic.shape[-1])

    # KS test per feature
    ks_stats = []
    for i in range(real.shape[-1]):
        ks, _ = stats.ks_2samp(real_flat[:, i], syn_flat[:, i])
        ks_stats.append(ks)

    # Correlation structure (if multiple features)
    if real.shape[-1] > 1:
        real_corr = np.corrcoef(real_flat.T)
        syn_corr = np.corrcoef(syn_flat.T)
        corr_error = np.mean(np.abs(real_corr - syn_corr))
    else:
        corr_error = 0.0

    return {
        "mean_ks": np.mean(ks_stats),
        "max_ks": np.max(ks_stats),
        "correlation_error": corr_error,
    }


stats_results = evaluate_statistics(real_eval, synthetic_sequences)

print("\n=== Statistical Evaluation ===")
print(f"Mean KS statistic: {stats_results['mean_ks']:.4f}")
print(f"Max KS statistic: {stats_results['max_ks']:.4f}")
print(f"Correlation error: {stats_results['correlation_error']:.4f}")

# %% [markdown]
# **Interpretation**: the KS statistic measures distributional divergence, from zero for
# identical distributions to one for completely separated ones. A middling value is
# typical of small-sample generative models and indicates a partial distributional match.
# A high value is expected when training on a few hundred bars and does not on its own
# mean the model has failed - it reflects how little distributional structure a short
# irregular series carries. The correlation error measures how well the cross-feature
# covariance is preserved.

# %% [markdown]
# ## Visualization: Real vs Synthetic with Irregular Timestamps


# %%
def plot_irregular_sequences(
    real_seq: np.ndarray,
    real_times: np.ndarray,
    synthetic_seq: np.ndarray,
    feature_idx: int = 0,
) -> go.Figure:
    """Visualize sequences with their actual irregular timestamps."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Real (Irregular Timestamps)", "Synthetic (Generated)"],
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Scatter(
            x=real_times,
            y=real_seq[:, feature_idx],
            mode="lines+markers",
            name="Real",
            line=dict(color=COLORS["blue"]),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=real_times,  # Same timestamps for comparison
            y=synthetic_seq[:, feature_idx],
            mode="lines+markers",
            name="Synthetic",
            line=dict(color=COLORS["copper"]),
            marker=dict(size=4),
        ),
        row=2,
        col=1,
    )

    # Share y-axis between paired panels so visual spread is comparable.
    real_vals = real_seq[:, feature_idx]
    synth_vals = synthetic_seq[:, feature_idx]
    y_lo = float(min(real_vals.min(), synth_vals.min()))
    y_hi = float(max(real_vals.max(), synth_vals.max()))
    pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)
    shared_range = [y_lo - pad, y_hi + pad]

    fig.update_layout(
        title_text="GT-GAN: Naturally Irregular Time Series Generation",
        template="ml4t",
        height=500,
        xaxis2_title="Normalized Time",
        yaxis_title=CONFIG["features"][feature_idx],
        yaxis2_title=CONFIG["features"][feature_idx],
        yaxis=dict(range=shared_range),
        yaxis2=dict(range=shared_range),
    )
    # The x zeroline is the template's navy, the same color as the real series, and it
    # renders inside the plotting area as a vertical rule that reads as data.
    fig.update_xaxes(zeroline=False)

    return fig


# `synthetic_sequences[i]` was generated on the time grid of `eval_idx[i]`, so the
# real window and the timestamps have to be taken through the same index.
sample_idx = 0
fig = plot_irregular_sequences(
    real_eval[sample_idx],
    seq_times[eval_idx[sample_idx]],
    synthetic_sequences[sample_idx],
)
show_plotly_with_alt(
    fig,
    "Two stacked panels sharing a time axis and a vertical scale, both plotted at the "
    "same irregular observation times so their markers line up column for column. The "
    "upper panel is one real window of the close feature, a walk that reverses "
    "direction repeatedly within a narrow band. The lower panel is the synthetic "
    "sequence decoded on that same grid, a smooth curve rising from the first "
    "observation to the last, steeply at first and then flattening, with none of the "
    "reversals above it and covering far more of the shared vertical scale.",
)

# %% [markdown]
# **Interpretation**: both panels carry the same observation times, because the time
# grid is an argument to `generate_synthetic` rather than something the model produces.
# So nothing in this figure can say whether GT-GAN reproduces the arrival process of
# information-driven bars: it is handed that process, not asked for it. What the two
# panels compare is the values the decoder returns when it is given a real window's
# timestamps, against the values that window actually took, on a shared vertical scale.
#
# On this window the two differ in both. The real series stays inside a narrow band and
# reverses direction repeatedly between observations; the synthetic one rises across most
# of the shared vertical scale without a single reversal. A decoder
# whose output is the state of an ODE evolving between query points is smooth wherever
# the fitted dynamics are smooth, so one window raises that question rather than
# settling it. The interpolation section earlier in the notebook is the measured
# version: it decodes at the observed times and again at the midpoints between them,
# and reports the smoothness of each beside the real series.

# %% [markdown]
# ## Evaluation: GT-GAN Protocol
#
# **Reference**: Jeon et al. (2022) "GT-GAN: General Purpose Time Series Synthesis" [NeurIPS 2022]
#
# GT-GAN's evaluation follows the TimeGAN protocol with additional focus on:
# 1. **Discriminative Score**: Can a classifier distinguish real from synthetic?
# 2. **Predictive Score (TSTR)**: Train on synthetic, predict on real holdout
# 3. **Interpolation Quality**: Unique to GT-GAN - measure reconstruction at irregular timestamps
#
# The key GT-GAN advantage is handling naturally irregular data. We evaluate whether
# the model preserves both statistical properties AND irregular timing patterns.


# %%
def gtgan_paper_evaluation(
    model: GTGAN,
    train_sequences: np.ndarray,
    train_times: np.ndarray,
    holdout_df: pl.DataFrame,
    holdout_times: np.ndarray,
    norm_params: dict,
    config: dict,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    """
    Evaluation for GT-GAN following Jeon et al. (2022).

    Evaluates:
    1. Discriminative: Real/fake classification accuracy
    2. Predictive (TSTR): Train on synthetic, predict on real holdout
    3. Interpolation: Reconstruction at irregular timestamps (GT-GAN specific)

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {}
    model.eval()

    # Create holdout sequences for evaluation
    holdout_data = holdout_df.select(config["features"]).to_numpy().astype(np.float32)
    holdout_seq, holdout_seq_times = create_irregular_sequences(
        holdout_df, config["features"], config["seq_length"], holdout_times
    )

    # Normalize holdout using training normalization params
    holdout_seq_norm = (holdout_seq - norm_params["min"]) / (
        norm_params["max"] - norm_params["min"] + 1e-8
    )

    # Generate synthetic sequences
    # Drawn at random for the same reason as above: consecutive rolling windows are
    # one stretch of the tape, and both arms of every score below read these slices.
    n_eval = min(len(holdout_seq_norm), len(train_sequences), 100)
    train_idx = rng.choice(len(train_sequences), size=n_eval, replace=False)
    holdout_idx = rng.choice(len(holdout_seq_norm), size=n_eval, replace=False)
    sample_times_tensor = torch.FloatTensor(train_times[train_idx]).to(device)

    with torch.no_grad():
        synthetic_eval = model.generate(n_eval, sample_times_tensor, device).cpu().numpy()

    # Flatten for sklearn
    real_flat = train_sequences[train_idx].reshape(n_eval, -1)
    syn_flat = synthetic_eval.reshape(n_eval, -1)
    holdout_flat = holdout_seq_norm[holdout_idx].reshape(n_eval, -1)

    # --- 1. Discriminative Score ---
    # Train classifier to distinguish real vs synthetic
    X_disc = np.vstack([real_flat, syn_flat])
    y_disc = np.array([1] * len(real_flat) + [0] * len(syn_flat))

    # Shuffle and split
    idx = np.random.permutation(len(X_disc))
    X_disc, y_disc = X_disc[idx], y_disc[idx]
    split = int(0.8 * len(X_disc))

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_disc[:split], y_disc[:split])

    disc_acc = accuracy_score(y_disc[split:], clf.predict(X_disc[split:]))
    disc_auc = roc_auc_score(y_disc[split:], clf.predict_proba(X_disc[split:])[:, 1])

    results["discriminative"] = {
        "accuracy": float(disc_acc),
        "auc": float(disc_auc),
        "interpretation": "Closer to 0.5 = better (indistinguishable)",
    }

    print("\n  Discriminative Score:")
    print(f"    Accuracy: {disc_acc:.3f} (target: ~0.5)")
    print(f"    AUC: {disc_auc:.3f} (target: ~0.5)")

    # --- 2. Predictive Score (TSTR) ---
    # Task: Predict next-step value (following TimeGAN protocol)
    def create_prediction_data(sequences):
        X = sequences[:, :-1, :].reshape(len(sequences), -1)
        y = sequences[:, -1, 0]  # Predict first feature at last timestep
        return X, y

    X_real_train, y_real_train = create_prediction_data(train_sequences[train_idx])
    X_syn, y_syn = create_prediction_data(synthetic_eval)
    X_holdout, y_holdout = create_prediction_data(holdout_seq_norm[holdout_idx])

    # TRTR: Train Real, Test Real (baseline)
    reg_trtr = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    reg_trtr.fit(X_real_train, y_real_train)
    pred_trtr = reg_trtr.predict(X_holdout)
    mae_trtr = np.mean(np.abs(pred_trtr - y_holdout))

    # TSTR: Train Synthetic, Test Real
    reg_tstr = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    reg_tstr.fit(X_syn, y_syn)
    pred_tstr = reg_tstr.predict(X_holdout)
    mae_tstr = np.mean(np.abs(pred_tstr - y_holdout))

    mae_ratio = mae_tstr / (mae_trtr + 1e-8)

    results["predictive"] = {
        "mae_trtr": float(mae_trtr),
        "mae_tstr": float(mae_tstr),
        "mae_ratio": float(mae_ratio),
        "interpretation": "Ratio ~1.0 = good (synthetic preserves predictive patterns)",
    }

    print("\n  Predictive Score (TSTR):")
    print(f"    MAE (TRTR baseline): {mae_trtr:.4f}")
    print(f"    MAE (TSTR): {mae_tstr:.4f}")
    print(f"    Ratio: {mae_ratio:.2f}x (target: ~1.0)")

    # --- 3. Interpolation Quality (GT-GAN specific) ---
    # Encode real, decode at same times - measure reconstruction
    test_seq = torch.FloatTensor(train_sequences[train_idx]).to(device)
    test_times = torch.FloatTensor(train_times[train_idx]).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(test_seq, test_times)
        z = model.reparameterize(mu, logvar)
        recon = model.decode(z, test_times)

    recon_mse = float(torch.mean((recon - test_seq) ** 2).cpu())

    # Interpolation at midpoints (unique to irregular data)
    midpoint_times = (test_times[:, :-1] + test_times[:, 1:]) / 2
    with torch.no_grad():
        interp = model.decode(z, midpoint_times)

    # Smoothness: interpolated values should be "between" adjacent real values
    interp_np = interp.cpu().numpy()
    test_np = test_seq.cpu().numpy()

    # Check if interpolated values are bounded by adjacent real values (soft check)
    lower = np.minimum(test_np[:, :-1], test_np[:, 1:])
    upper = np.maximum(test_np[:, :-1], test_np[:, 1:])
    bounded_fraction = np.mean((interp_np >= lower - 0.1) & (interp_np <= upper + 0.1))

    results["interpolation"] = {
        "reconstruction_mse": recon_mse,
        "bounded_fraction": float(bounded_fraction),
        "interpretation": "Higher bounded_fraction = smoother interpolation",
    }

    print("\n  Interpolation Quality (GT-GAN specific):")
    print(f"    Reconstruction MSE: {recon_mse:.6f}")
    print(f"    Bounded fraction: {bounded_fraction:.1%} (interpolations within adjacent bounds)")

    model.train()
    return results


# %% [markdown]
# ### Run Evaluation
#
# Execute the full GT-GAN evaluation protocol: discriminative score,
# predictive score (TSTR), and interpolation quality.

# %%
# Run evaluation
print("=" * 70)
print("EVALUATION (GT-GAN Protocol)")
print("=" * 70)
print("\nReference: Jeon et al. (2022), NeurIPS 2022")
print("Following TimeGAN protocol + GT-GAN interpolation metrics")

paper_results = gtgan_paper_evaluation(
    model=model,
    train_sequences=sequences_norm,
    train_times=seq_times,
    holdout_df=df_holdout,
    holdout_times=times_holdout,
    norm_params=norm_params,
    config=CONFIG,
    device=device,
    rng=eval_rng,
)

# Summary
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print(f"""
| Metric                | Value    | Target   | Status |
|-----------------------|----------|----------|--------|
| Discriminative Acc    | {paper_results["discriminative"]["accuracy"]:.3f}    | ~0.50    | {"[OK]" if abs(paper_results["discriminative"]["accuracy"] - 0.5) < 0.15 else "WARNING"} |
| Discriminative AUC    | {paper_results["discriminative"]["auc"]:.3f}    | ~0.50    | {"[OK]" if abs(paper_results["discriminative"]["auc"] - 0.5) < 0.15 else "WARNING"} |
| TSTR MAE Ratio        | {paper_results["predictive"]["mae_ratio"]:.2f}x    | ~1.0x    | {"[OK]" if 0.7 < paper_results["predictive"]["mae_ratio"] < 1.5 else "WARNING"} |
| Interpolation Bounded | {paper_results["interpolation"]["bounded_fraction"]:.1%}   | >70%     | {"[OK]" if paper_results["interpolation"]["bounded_fraction"] > 0.7 else "WARNING"} |
""")

# %% [markdown]
# **Interpretation**: read the four rows together rather than choosing among them. The
# discriminative accuracy and AUC sit at their maximum, so the classifier separates
# synthetic from real without error: whatever the sequences preserve is not enough to
# fool a model that is looking for the difference. The TSTR ratio misses the band the
# protocol allows, so a predictor fitted on the synthetic sequences carries more error on
# real ones than a predictor fitted on real data does, by the margin the ratio above
# gives. Those two rows are not the same finding: a classifier separating two sets
# perfectly says nothing about how much of the predictive structure the second set
# carries. The
# bounded fraction clears its threshold, but it checks something narrower than its name
# suggests. It encodes a real window, decodes at the midpoint between each pair of
# observation times, and asks whether that value lands between the two neighbouring real
# values with a fixed tolerance added on either side. That tolerance is wider than the
# mean gap between adjacent observations printed in the interpolation section above, so
# most of the band being cleared is tolerance rather than data. And the latent it decodes
# comes from encoding a real window, which makes it the one row here that never runs the
# generator. The passing row and the failing rows are therefore not in tension: they ask
# different questions, and discarding the failing ones to keep the passing one would be
# choosing a metric by its answer.
#
# `MAX_STEPS` is set to two thousand, which is a short adversarial run, and the
# training-progress figure shows the three adversarial losses settling onto a common
# flat level well before the end. Lengthening that run is the change to make before any
# of these numbers is quoted as a property of GT-GAN rather than of this pass.

# %% [markdown]
# ## Save Outputs


# %%
output_dir = get_output_dir(5, "gtgan")
checkpoint_dir = output_dir / "checkpoints" / "gtgan" / f"nvda_{CONFIG['bar_type']}_bars"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Save model checkpoint
checkpoint = {
    "model": model.state_dict(),
    "norm_min": norm_params["min"].tolist(),
    "norm_max": norm_params["max"].tolist(),
}
torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")

# %%
# Build training metadata
metadata = {
    "version": "2.0",
    "generator": {
        "name": "gtgan",
        "version": CONFIG["weights_version"],
        "paper": "Jeon et al., GT-GAN: General Purpose Time Series Synthesis, NeurIPS 2022",
        "data_type": "naturally_irregular",
    },
    "training": {
        "created_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "random_seed": 42,
    },
    "data": {
        "source": "chapter3_information_bars",
        "bar_type": CONFIG["bar_type"],
        "symbol": "NVDA",
        "features": CONFIG["features"],
        "n_features": n_features,
        "seq_length": CONFIG["seq_length"],
        "n_train_bars": n_train,
        "n_holdout_bars": n_holdout,
        "n_sequences": len(sequences),
        "n_synthetic_samples": N_SYNTHETIC,
    },
    "hyperparameters": {
        "latent_dim": CONFIG["latent_dim"],
        "hidden_dim": CONFIG["hidden_dim"],
        "max_steps": CONFIG["max_steps"],  # Step-based per Jeon et al.
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "ode_method": CONFIG["ode_method"],
    },
}

# %%
# Add evaluation metrics and save metadata
metadata["evaluation"] = {
    "reconstruction_mse": float(interp_results["reconstruction_mse"]),
    "smoothness_ratio": float(interp_results["smoothness_ratio"]),
    "mean_ks": float(stats_results["mean_ks"]),
    "correlation_error": float(stats_results["correlation_error"]),
    # Metrics (GT-GAN protocol)
    "discriminative_accuracy": float(paper_results["discriminative"]["accuracy"]),
    "discriminative_auc": float(paper_results["discriminative"]["auc"]),
    "predictive_mae_ratio": float(paper_results["predictive"]["mae_ratio"]),
    "interpolation_bounded_fraction": float(paper_results["interpolation"]["bounded_fraction"]),
}

with open(checkpoint_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# %%
# Save synthetic samples (denormalized)
samples_denorm = (
    synthetic_sequences * (norm_params["max"] - norm_params["min"]) + norm_params["min"]
)
np.save(checkpoint_dir / "samples.npy", samples_denorm.astype(np.float32))
np.save(checkpoint_dir / "sample_times.npy", seq_times[eval_idx].astype(np.float32))

print(f"\nSaved outputs to: {checkpoint_dir}/")
print("  - checkpoint.pt (model weights)")
print("  - metadata.json (training info)")
print(f"  - samples.npy ({N_SYNTHETIC} synthetic sequences)")
print("  - sample_times.npy (irregular timestamps)")


# %% [markdown]
# ## Key Takeaways
#
# **What GT-GAN Offers for Irregular Data**:
#
# 1. **Natural Irregularity**: Uses Chapter 3's information-theoretic bars with genuine
#    variable timestamps (not artificial masking)
# 2. **Continuous Dynamics**: Neural ODEs model evolution between observations
# 3. **Flexible Generation**: Generate at any timestamps, not just training grid
# 4. **Time-Aware Discrimination**: Discriminator accounts for irregular spacing
#
# **Comparison with Other Generators**:
#
# | Generator | Time Grid | Best For |
# |-----------|-----------|----------|
# | TimeGAN | Fixed | Regular daily/hourly data |
# | Tail-GAN | Fixed | Tail risk scenarios |
# | Sig-CWGAN | Fixed | Multi-asset correlations |
# | Diffusion-TS | Fixed | High-dimensional data |
# | **GT-GAN** | **Variable** | **Naturally irregular** (tick/volume bars) |
#
# **Use Cases**:
#
# - High-frequency data with variable sampling (tick bars, volume bars)
# - Multi-asset data with different trading hours
# - Generating scenarios for backtesting irregularly-sampled strategies
