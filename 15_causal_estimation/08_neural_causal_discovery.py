# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Causal Discovery: Beyond Constraint-Based Methods
#
# **Chapter 15: Causal Estimation with ML**
# **Docker image**: `ml4t`
#
# **Book Reference**: Chapter 15, §15.4 (Neural Causal Discovery)
#
# This notebook implements **neural approaches to causal discovery**, moving beyond
# traditional constraint-based methods (like PCMCI) to differentiable DAG learning.
#
# **Why Neural Causal Discovery?**
#
# Traditional methods (PC, FCI, PCMCI) test conditional independence to build DAGs.
# They have limitations:
# 1. **Combinatorial explosion**: Testing all possible edges is expensive
# 2. **Discrete decisions**: No gradient-based optimization
# 3. **Limited scalability**: Struggles with many variables
#
# **Neural approaches** reformulate DAG learning as continuous optimization:
# - **NOTEARS**: Uses trace exponential constraint for acyclicity
# - **DAG-GNN**: Graph neural networks for structure learning
# - **VAR-LiNGAM**: Combines VAR with non-Gaussianity for identification
#
# **Learning Outcomes**:
# - LO1: Understand the NOTEARS formulation for DAG learning
# - LO2: Apply neural causal discovery to financial time series
# - LO3: Compare neural vs constraint-based methods with proper multiple testing
# - LO4: Assess edge stability via bootstrap analysis
#
# **Prerequisites**: [`07_tigramite_time_series`](07_tigramite_time_series.ipynb) for constraint-based
# discovery; ETF OHLCV data from Ch2 data pipeline

# %% [markdown]
# ## 1. Setup and Configuration

# %%
"""Neural Causal Discovery — differentiable DAG learning with NOTEARS and VAR-LiNGAM."""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from scipy import linalg, optimize
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from data import load_etfs
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
N_SAMPLES = 2000
MAX_LAG = 3
N_BOOTSTRAP = 100
SEED = 42
BLOCK_SIZE = 20
SYNTHETIC_SAMPLE_SIZE = 0
RETURN_SAMPLE_LIMIT = 0
NOTEARS_LAMBDA1 = 0.05
NOTEARS_MAX_ITER = 100
START_DATE = "2015-01-01"

# %%

set_global_seeds(SEED)

print("Neural Causal Discovery for Financial Time Series")
print(f"Bootstrap iterations: {N_BOOTSTRAP}")

# %% [markdown]
# ## 2. The NOTEARS Algorithm
#
# **NOTEARS** (Zheng et al., 2018) reformulates DAG learning as:
#
# $$\min_W \frac{1}{2n} \|X - XW\|_F^2 + \lambda \|W\|_1$$
# $$\text{subject to } h(W) = \text{tr}(e^{W \circ W}) - d = 0$$
#
# Where:
# - $W$ is the weighted adjacency matrix (W[i,j] = edge i→j)
# - $h(W) = 0$ is the **acyclicity constraint** (trace exponential trick)
# - The $\ell_1$ penalty induces sparsity
#
# The key insight: $h(W) = 0$ iff $W$ represents a DAG. This makes the
# combinatorial problem continuous and differentiable.


# %%
def _notears_objectives(X, n, d):
    """Build the NOTEARS least-squares loss and acyclicity constraint.

    Each returns ``(value, gradient)`` so the augmented-Lagrangian subproblem
    can be handed to a smooth optimizer (Zheng et al., 2018).
    """

    def loss(W):
        R = X - X @ W
        value = 0.5 / n * np.sum(R**2)
        grad = -1.0 / n * X.T @ R
        return value, grad

    def h(W):
        # h(W) = tr(exp(W∘W)) - d, which is zero iff W encodes a DAG.
        E = linalg.expm(W * W)
        value = np.trace(E) - d
        grad = E.T * W * 2
        return value, grad

    return loss, h


# %% [markdown]
# Use the NOTEARS objectives in an augmented-Lagrangian optimization loop.


# %%
def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
) -> np.ndarray:
    """NOTEARS linear DAG learning via the augmented Lagrangian (Zheng et al., 2018).

    The L1-penalized least-squares objective is minimized under the smooth
    acyclicity constraint ``h(W) = 0``. Each subproblem is solved with L-BFGS-B
    over the ``(W+, W-)`` positive-part split that makes the L1 term
    differentiable, then ``rho`` is escalated until the constraint is met.
    """
    n, d = X.shape
    loss, h = _notears_objectives(X, n, d)
    rho, alpha, h_val = 1.0, 0.0, np.inf

    def _adj(w):
        return (w[: d * d] - w[d * d :]).reshape(d, d)

    def _func(w):
        W = _adj(w)
        loss_val, g_loss = loss(W)
        h_cur, g_h = h(W)
        obj = loss_val + 0.5 * rho * h_cur * h_cur + alpha * h_cur + lambda1 * w.sum()
        g_smooth = g_loss + (rho * h_cur + alpha) * g_h
        g_obj = np.concatenate((g_smooth + lambda1, -g_smooth + lambda1), axis=None)
        return obj, g_obj

    w_est = np.zeros(2 * d * d)
    # Fix the diagonal to zero (no self-loops); off-diagonal parts stay non-negative.
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = w_est, h_val
        while rho < rho_max:
            sol = optimize.minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new = h(_adj(w_new))[0]
            if h_new > 0.25 * h_val:
                rho *= 10
            else:
                break
        w_est, h_val = w_new, h_new
        alpha += rho * h_val
        if h_val <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# %% [markdown]
# ## 3. VAR-LiNGAM: Combining VAR with Non-Gaussianity
#
# **VAR-LiNGAM** extends the Linear Non-Gaussian Acyclic Model to time series:
# 1. Fit a VAR model to capture lagged relationships
# 2. Apply ICA to residuals to identify instantaneous causal order
# 3. Use non-Gaussianity for identification (unlike Gaussian methods)
#
# We first build a simplified version from scratch to expose the mechanics,
# then show the production implementation via `causal-learn`.

# %% [markdown]
# ### From-Scratch Implementation
#
# The core idea: fit a VAR model to capture lagged effects, then apply ICA
# to the residuals — non-Gaussian structure in the residuals identifies the
# instantaneous causal ordering that Gaussian methods cannot recover.


# %%
def var_lingam_scratch(
    X: np.ndarray,
    max_lag: int = 1,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplified VAR-LiNGAM: VAR for lagged effects + ICA for instantaneous."""
    n, d = X.shape
    Y = X[max_lag:]
    X_lagged = X[max_lag - 1 : -1]

    # Fit VAR via Ridge regression
    B_lag = np.zeros((d, d))
    residuals = np.zeros_like(Y)
    for j in range(d):
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        model.fit(X_lagged, Y[:, j])
        B_lag[j, :] = model.coef_
        residuals[:, j] = Y[:, j] - model.predict(X_lagged)

    # ICA on residuals to identify instantaneous effects (LiNGAM)
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=d, random_state=SEED, max_iter=500)
    ica.fit_transform(residuals)
    try:
        B0 = np.eye(d) - np.linalg.inv(ica.mixing_)
    except np.linalg.LinAlgError:
        B0 = np.zeros((d, d))

    B0[np.abs(B0) < threshold] = 0
    B_lag[np.abs(B_lag) < threshold] = 0
    return B0, B_lag


# %% [markdown]
# ### Library Implementation: causal-learn
#
# The `causal-learn` library provides a production-grade `VARLiNGAM` that uses
# proper VAR estimation with BIC lag selection, DirectLiNGAM for the instantaneous
# ordering, and optional pruning — all in three lines.


# %%
from causallearn.search.FCMBased.lingam import VARLiNGAM


def var_lingam_library(
    X: np.ndarray,
    lags: int = 1,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """VAR-LiNGAM via causal-learn: proper VAR + DirectLiNGAM on residuals."""
    model = VARLiNGAM(lags=lags, criterion=None, prune=True, random_state=SEED)
    model.fit(X)

    B0 = model.adjacency_matrices_[0].copy()
    B_lag = model.adjacency_matrices_[1].copy()

    B0[np.abs(B0) < threshold] = 0
    B_lag[np.abs(B_lag) < threshold] = 0
    return B0, B_lag


# %% [markdown]
# ## 4. Block Bootstrap for Stability Analysis
#
# Neural methods are sensitive to noise. We assess edge stability via block
# bootstrap (preserving autocorrelation) to see how often each edge appears.


# %%
def block_bootstrap_indices(n: int, block_size: int = 20) -> np.ndarray:
    """
    Generate block bootstrap indices preserving temporal dependence.

    Args:
        n: Length of time series
        block_size: Size of contiguous blocks

    Returns:
        Bootstrapped indices
    """
    n_blocks = n // block_size + 1
    # Sample blocks with replacement
    block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
    # Concatenate blocks
    indices = np.concatenate([np.arange(start, start + block_size) for start in block_starts])
    return indices[:n]  # Trim to original length


# %% [markdown]
# Convert weighted adjacency matrices into edge sets for evaluation and stability counting.


# %%
def extract_edges(W: np.ndarray, labels: list, threshold: float = 0.0) -> set:
    """Extract edges from adjacency matrix as set of tuples."""
    edges = set()
    for i in range(len(labels)):
        for j in range(len(labels)):
            if abs(W[i, j]) > threshold:
                edges.add((labels[i], labels[j]))
    return edges


# %% [markdown]
# ## 5. Synthetic Validation with Known Ground Truth
#
# Before applying to real data, we validate methods on synthetic data
# with known causal structure to measure precision and sensitivity.

# %%
# Generate synthetic data with known DAG: X0→X1→X2, X0→X2, X2→X3→X4
print("\n=== SYNTHETIC VALIDATION ===\n")

n_synthetic = SYNTHETIC_SAMPLE_SIZE if SYNTHETIC_SAMPLE_SIZE > 0 else N_SAMPLES
TRUE_EDGES = {(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)}

# Equal-variance noise (unit scale) with accumulating coefficients, the linear
# SEM that NOTEARS targets (Zheng et al. 2018): downstream variables inherit
# their parents' variance, so the variance ordering aligns with the causal
# order and the orientation is identifiable from observational data alone.
np.random.seed(SEED)
X_syn = np.zeros((n_synthetic, 5))
X_syn[:, 0] = np.random.randn(n_synthetic)
X_syn[:, 1] = 0.8 * X_syn[:, 0] + np.random.randn(n_synthetic)
X_syn[:, 2] = 0.7 * X_syn[:, 0] + 0.8 * X_syn[:, 1] + np.random.randn(n_synthetic)
X_syn[:, 3] = 0.9 * X_syn[:, 2] + np.random.randn(n_synthetic)
X_syn[:, 4] = 0.8 * X_syn[:, 3] + np.random.randn(n_synthetic)

# %%
# Run NOTEARS and compute precision/sensitivity against known truth
W_syn = notears_linear(X_syn, lambda1=0.05, max_iter=50)
discovered_edges = extract_edges(W_syn, list(range(5)))

tp = len(discovered_edges & TRUE_EDGES)
fp = len(discovered_edges - TRUE_EDGES)
fn = len(TRUE_EDGES - discovered_edges)
precision = tp / max(len(discovered_edges), 1)
sensitivity = tp / len(TRUE_EDGES)
f1_score = 2 * precision * sensitivity / max(precision + sensitivity, 1e-6)

print(f"  Discovered: {discovered_edges}")
print(f"  TP={tp}, FP={fp}, FN={fn}")
print(f"  Precision: {precision:.1%}, Sensitivity: {sensitivity:.1%}, F1: {f1_score:.2f}")

SYNTHETIC_RESULTS = {"precision": precision, "sensitivity": sensitivity, "f1": f1_score}

# %% [markdown]
# ## 6. Load Financial Time Series Data
#
# We use multi-asset ETF returns to discover causal structure.

# %%
print("\n=== LOADING DATA ===\n")

# Configuration
ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM", "XLF"]
END_DATE = "2024-06-01"

# Load ETF data
etf_data = load_etfs()

# Filter to assets and date range
from datetime import datetime

start_dt = datetime.fromisoformat(START_DATE)
end_dt = datetime.fromisoformat(END_DATE)

etf_data = etf_data.filter(
    (pl.col("symbol").is_in(ASSETS))
    & (pl.col("timestamp") >= start_dt)
    & (pl.col("timestamp") <= end_dt)
)


# %%
# Pivot to wide format
prices = (
    etf_data.select(["timestamp", "symbol", "close"])
    .pivot(on="symbol", index="timestamp", values="close")
    .sort("timestamp")
)

# Convert to pandas
prices_pd = prices.to_pandas()
prices_pd["timestamp"] = pd.to_datetime(prices_pd["timestamp"])
prices_pd = prices_pd.set_index("timestamp")

# Compute returns
returns = prices_pd.pct_change().dropna()

if RETURN_SAMPLE_LIMIT > 0:
    returns = returns.tail(RETURN_SAMPLE_LIMIT)

# Update ASSETS to only those present in data (test data may have subset)
ASSETS = [a for a in ASSETS if a in returns.columns]
returns = returns[ASSETS]
print(f"Assets: {ASSETS}")
print(f"Sample period: {returns.index.min()} to {returns.index.max()}")
print(f"Observations: {len(returns)}")

# Standardize for causal discovery
scaler = StandardScaler()
X = scaler.fit_transform(returns.values)

print(f"\nData shape: {X.shape}")

# %% [markdown]
# ## 7. Apply NOTEARS with Bootstrap Stability

# %%
print("\n=== NOTEARS CAUSAL DISCOVERY WITH STABILITY ===\n")

# Apply NOTEARS to full data
lambda1 = NOTEARS_LAMBDA1
W_notears = notears_linear(X, lambda1=lambda1, max_iter=NOTEARS_MAX_ITER)

# Create adjacency dataframe
adj_df = pd.DataFrame(W_notears, index=ASSETS, columns=ASSETS)

print("Discovered Adjacency Matrix (NOTEARS):")
print(adj_df.round(3).to_string())

# Count edges
n_edges = np.sum(np.abs(W_notears) > 0)
print(f"\nTotal edges discovered: {n_edges}")

# Bootstrap stability analysis
print(f"\nBootstrap stability ({N_BOOTSTRAP} iterations)...")
edge_counts = defaultdict(int)
for b in range(N_BOOTSTRAP):
    boot_indices = block_bootstrap_indices(len(X), BLOCK_SIZE)
    W_boot = notears_linear(X[boot_indices], lambda1=lambda1, max_iter=30)
    for i, source in enumerate(ASSETS):
        for j, target in enumerate(ASSETS):
            if abs(W_boot[i, j]) > 0:
                edge_counts[(source, target)] += 1

# %%
# Report edge stability
stable_edges = []
for edge, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
    freq = count / N_BOOTSTRAP
    if freq >= 0.3:
        print(f"  {edge[0]} → {edge[1]}: {freq:.0%} ({'STABLE' if freq >= 0.5 else 'unstable'})")
        stable_edges.append(
            {"Source": edge[0], "Target": edge[1], "Frequency": freq, "Stable": freq >= 0.5}
        )

n_stable = sum(1 for e in stable_edges if e["Stable"])
print(f"\nStable edges (>=50%): {n_stable} / {n_edges}")

# %% [markdown]
# ## 8. Apply VAR-LiNGAM for Time Series Structure
#
# We run both the from-scratch and `causal-learn` library implementations on the
# same data and compare their discovered edges. The library version uses
# DirectLiNGAM (a more principled ordering algorithm than raw ICA) and proper
# VAR estimation, so differences are expected — and instructive.

# %%
print("\n=== VAR-LiNGAM: FROM-SCRATCH IMPLEMENTATION ===\n")

B0_scratch, B_lag_scratch = var_lingam_scratch(X, max_lag=1, threshold=0.1)

print("Instantaneous Effects (B0):")
print(pd.DataFrame(B0_scratch, index=ASSETS, columns=ASSETS).round(3).to_string())
print("\nLagged Effects (B1, lag-1):")
print(pd.DataFrame(B_lag_scratch, index=ASSETS, columns=ASSETS).round(3).to_string())

# %%
print("\n=== VAR-LiNGAM: CAUSAL-LEARN LIBRARY ===\n")

B0, B_lag = var_lingam_library(X, lags=1, threshold=0.1)

inst_df = pd.DataFrame(B0, index=ASSETS, columns=ASSETS)
lag_df = pd.DataFrame(B_lag, index=ASSETS, columns=ASSETS)

print("Instantaneous Effects (B0):")
print(inst_df.round(3).to_string())
print("\nLagged Effects (B1, lag-1):")
print(lag_df.round(3).to_string())

# %% [markdown]
# Compare edge agreement between implementations.

# %%
scratch_lag_edges = set()
for i in range(len(ASSETS)):
    for j in range(len(ASSETS)):
        if abs(B_lag_scratch[j, i]) > 0:
            scratch_lag_edges.add((ASSETS[i], ASSETS[j]))

library_lag_edges = set()
for i in range(len(ASSETS)):
    for j in range(len(ASSETS)):
        if abs(B_lag[j, i]) > 0:
            library_lag_edges.add((ASSETS[i], ASSETS[j]))

shared = scratch_lag_edges & library_lag_edges
print(f"Scratch-only edges: {len(scratch_lag_edges - library_lag_edges)}")
print(f"Library-only edges: {len(library_lag_edges - scratch_lag_edges)}")
print(f"Shared edges: {len(shared)}")

# %%
# Identify lagged causal edges (from library implementation)
lag_edges = []
for i, source in enumerate(ASSETS):
    for j, target in enumerate(ASSETS):
        if abs(B_lag[j, i]) > 0:  # B_lag[j,i] means i at t-1 → j at t
            lag_edges.append(
                {
                    "Source": f"{source}(t-1)",
                    "Target": f"{target}(t)",
                    "Weight": B_lag[j, i],
                }
            )

if lag_edges:
    lag_edges_df = pd.DataFrame(lag_edges).sort_values("Weight", key=abs, ascending=False)
    print("\nDiscovered Lagged Causal Edges (causal-learn VARLiNGAM):")
    print(lag_edges_df.head(10).to_string(index=False))

# %% [markdown]
# ## 9. PCMCI on the Same Universe
#
# `07_tigramite_time_series` runs PCMCI on 4 assets (SPY, IEF, GLD, VIX).
# For a fair comparison, we run PCMCI here on the same 7-asset panel used
# by NOTEARS, VAR-LiNGAM, and Granger above.

# %%
print("\n=== PCMCI ON 7-ASSET PANEL ===\n")

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

dataframe = pp.DataFrame(X, var_names=ASSETS)
parcorr = ParCorr(significance="analytic")
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)
pcmci_results = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=0.05)

# Count significant lagged links at multiple thresholds
p_matrix = pcmci_results["p_matrix"]
val_matrix = pcmci_results["val_matrix"]
n_vars = len(ASSETS)
total_possible = n_vars * n_vars * MAX_LAG

pcmci_sig_links = 0
pcmci_link_details = []
effect_sizes = []

for i, source in enumerate(ASSETS):
    for j, target in enumerate(ASSETS):
        for tau in range(1, MAX_LAG + 1):
            if p_matrix[i, j, tau] < 0.05:
                pcmci_sig_links += 1
                effect_sizes.append(abs(val_matrix[i, j, tau]))
                pcmci_link_details.append(
                    f"  {source}(t-{tau}) → {target}: val={val_matrix[i, j, tau]:.3f}, p={p_matrix[i, j, tau]:.4f}"
                )

print(f"Significant lagged links (p<0.05): {pcmci_sig_links}/{total_possible}")
for detail in sorted(pcmci_link_details):
    print(detail)

if effect_sizes:
    effect_sizes = np.array(effect_sizes)
    print(
        f"\nEffect sizes (|partial corr|): min={effect_sizes.min():.3f}, "
        f"median={np.median(effect_sizes):.3f}, max={effect_sizes.max():.3f}"
    )
    print(f"Links with |val| > 0.10: {(effect_sizes > 0.10).sum()}")

# %% [markdown]
# ## 10. Compare with Granger Causality (FDR-Corrected)
#
# **Multiple Testing Correction**:
# - 7 assets → 42 directed pairs (i→j where i≠j)
# - At α=0.05, expected false positives ≈ 2.1 without correction
# - We apply Benjamini-Hochberg FDR to control false discovery rate

# %%
# Pairwise Granger causality tests
print("\n=== GRANGER CAUSALITY WITH FDR CORRECTION ===\n")

all_granger_tests = []
from statsmodels.tsa.stattools import grangercausalitytests

max_lag_granger = 5
n_pairs = len(ASSETS) * (len(ASSETS) - 1)

for i, source in enumerate(ASSETS):
    for j, target in enumerate(ASSETS):
        if i != j:
            try:
                data = returns[[target, source]].dropna()
                result = grangercausalitytests(data, maxlag=max_lag_granger, verbose=False)
                p_vals = [result[lag + 1][0]["ssr_ftest"][1] for lag in range(max_lag_granger)]
                min_p = min(p_vals)
                all_granger_tests.append(
                    {
                        "Source": source,
                        "Target": target,
                        "P_value_raw": min_p,
                        "Best_Lag": p_vals.index(min_p) + 1,
                    }
                )
            except Exception:
                # Rare pair-specific VAR failures (singular, non-stationary) are left out of
                # the comparison table rather than taking down the whole Granger sweep.
                pass

print(f"Computed {len(all_granger_tests)} pairwise Granger tests")

# %%
# Apply FDR correction (Benjamini-Hochberg)
granger_edges = []
if all_granger_tests:
    granger_df = pd.DataFrame(all_granger_tests)
    rejected, p_corrected, _, _ = multipletests(
        granger_df["P_value_raw"].values, method="fdr_bh", alpha=0.05
    )
    granger_df["P_value_FDR"] = p_corrected
    granger_df["Significant_FDR"] = rejected

    n_raw = sum(granger_df["P_value_raw"] < 0.05)
    n_fdr = sum(rejected)
    print(f"  Uncorrected significant: {n_raw}, FDR-corrected: {n_fdr}")
    print(f"  Expected false positives (uncorrected): ~{0.05 * len(all_granger_tests):.1f}")

    significant_edges = granger_df[granger_df["Significant_FDR"]].sort_values("P_value_FDR")
    if len(significant_edges) > 0:
        print("\nFDR-Significant Granger Edges:")
        print(
            significant_edges[
                ["Source", "Target", "P_value_raw", "P_value_FDR", "Best_Lag"]
            ].to_string(index=False)
        )
        granger_edges = significant_edges.to_dict("records")
    else:
        print("\nNo edges survive FDR correction.")

# %% [markdown]
# ## 11. Method Summary and Comparison

# %%
print("\n=== METHOD SUMMARY ===\n")

methods_summary = {
    "Method": ["NOTEARS", "VAR-LiNGAM", "Granger (FDR)", "PCMCI"],
    "Type": ["Neural/Continuous", "ICA-based", "Statistical Test", "Constraint-based"],
    "Edges_Found": [
        f"{n_edges} ({n_stable} stable)",
        np.sum(np.abs(B_lag) > 0),
        len(granger_edges) if "granger_edges" in dir() else "N/A",
        pcmci_sig_links,
    ],
    "Key_Assumption": [
        "Linearity, acyclicity",
        "Non-Gaussianity",
        "Stationarity",
        "Faithfulness",
    ],
    "Strengths": [
        "Differentiable, scalable",
        "ICA identification",
        "Simple, well-understood",
        "Rigorous CI tests",
    ],
}

summary_df = pd.DataFrame(methods_summary)
print(summary_df.to_string(index=False))

# %% [markdown]
# ## 12. Visualize Discovered Causal Graph


# %%
def _add_directed_edge(fig, x0, y0, x1, y1, mid_x, mid_y, color, width, hover):
    """Render one directed edge with a curved segment and arrow annotation."""
    fig.add_trace(
        go.Scatter(
            x=[x0, mid_x, x1],
            y=[y0, mid_y, y1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            hovertext=hover,
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=mid_x,
        ay=mid_y,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=width,
        arrowcolor=color,
    )


# %% [markdown]
# Iterate through adjacency entries and draw only non-zero directed edges.


# %%
def _add_graph_edges(fig, W, labels, x_nodes, y_nodes, stability=None):
    """Add directed edges with optional stability coloring."""
    for i in range(len(labels)):
        for j in range(len(labels)):
            if abs(W[i, j]) <= 0:
                continue
            weight = W[i, j]
            edge_key = (labels[i], labels[j])
            has_stability = stability and edge_key in stability
            if has_stability:
                stability_value = stability[edge_key]
                color = "green" if stability_value >= 0.5 else "orange"
            else:
                stability_value = None
                color = "green" if weight > 0 else "red"
            width = min(abs(weight) * 5, 5)
            mid_x = (x_nodes[i] + x_nodes[j]) / 2 + 0.1 * (y_nodes[j] - y_nodes[i])
            mid_y = (y_nodes[i] + y_nodes[j]) / 2 - 0.1 * (x_nodes[j] - x_nodes[i])
            hover = f"{labels[i]} → {labels[j]}: {weight:.3f}"
            if stability_value is not None:
                hover += f" ({stability_value:.0%})"
            _add_directed_edge(
                fig,
                x_nodes[i],
                y_nodes[i],
                x_nodes[j],
                y_nodes[j],
                mid_x,
                mid_y,
                color,
                width,
                hover,
            )


# %% [markdown]
# Build a circular network view for the discovered weighted adjacency matrix.


# %%
def create_causal_graph_viz(
    W: np.ndarray,
    labels: list,
    title: str,
    stability: dict = None,
) -> go.Figure:
    """Create circular network visualization of a causal graph."""
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    x_nodes, y_nodes = np.cos(angles), np.sin(angles)

    fig = go.Figure()
    _add_graph_edges(fig, W, labels, x_nodes, y_nodes, stability)

    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker=dict(size=40, color="#1f77b4", line=dict(width=2, color="white")),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hoverinfo="text",
            hovertext=labels,
        )
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        height=500,
        width=720,
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# %%
# Create stability dictionary for visualization
edge_stability_dict = {(e["Source"], e["Target"]): e["Frequency"] for e in stable_edges}

# Visualize NOTEARS graph with stability coloring
fig1 = create_causal_graph_viz(
    W_notears,
    ASSETS,
    "NOTEARS Discovered Causal Graph (green=stable, orange=unstable)",
    stability=edge_stability_dict,
)
fig1.show()

# Visualize VAR-LiNGAM lagged effects (causal-learn library)
fig2 = create_causal_graph_viz(B_lag.T, ASSETS, "VAR-LiNGAM Lagged Causal Graph (causal-learn)")
fig2.show()

# %% [markdown]
# ## 13. Interpretation: Hypotheses for Further Investigation
#
# **CRITICAL**: Discovered edges are **HYPOTHESES**, not proven causation.
#
# ### Validation Steps Before Any Trading Use
#
# 1. **Out-of-sample testing**: Split data temporally, discover on training, validate on test
# 2. **Bootstrap stability**: Only consider edges found in ≥50% of bootstraps
# 3. **Multiple method agreement**: Edges confirmed by NOTEARS, VAR-LiNGAM, AND Granger
# 4. **Economic rationale**: Does the relationship make economic sense?
# 5. **DML/BSTS validation**: Use proper causal inference to estimate effect magnitude
#
# ### Why Edges May Be Spurious
#
# - **Omitted confounders**: Common macro factors drive both assets
# - **Non-stationarity**: Regime changes invalidate constant edge weights
# - **Sample dependence**: Results sensitive to time period
# - **Hyperparameter sensitivity**: Different λ gives different graphs

# %%
print("\n=== INTERPRETATION: HYPOTHESES FOR INVESTIGATION ===\n")

# Identify most robust relationships
print("Most Robust Findings:")
print("=" * 50)

# Edges that are both stable AND have large weights
robust_edges = []
for e in stable_edges:
    if e["Stable"]:
        source, target = e["Source"], e["Target"]
        i, j = ASSETS.index(source), ASSETS.index(target)
        weight = W_notears[i, j]
        robust_edges.append(
            {
                "Edge": f"{source} → {target}",
                "Weight": weight,
                "Bootstrap_Freq": e["Frequency"],
            }
        )

if robust_edges:
    robust_df = pd.DataFrame(robust_edges).sort_values("Bootstrap_Freq", ascending=False)
    print(robust_df.to_string(index=False))
else:
    print("No edges meet stability threshold (≥50% bootstrap frequency).")
    print("This suggests high sensitivity to sample - proceed with caution.")

print("\nValidation Checklist Before Trading:")
print("  [ ] Validate on out-of-sample period (e.g., most recent 6 months)")
print("  [ ] Confirm with DML effect estimation (see 03_econml_dml.py)")
print("  [ ] Consider transaction costs and capacity constraints")
print("  [ ] Monitor for regime changes that may invalidate relationships")

# %% [markdown]
# ## 14. Key Takeaways
#
# ### What We Learned
# - **NOTEARS** reformulates DAG learning as continuous optimization
# - **VAR-LiNGAM** uses non-Gaussianity for time series identification
# - **Bootstrap stability** reveals which edges are robust to sampling noise
# - **FDR correction** controls false discoveries in Granger causality
#
# ### Practical Guidance
# - Use neural methods for **exploration** and **hypothesis generation**
# - Validate discoveries with **rigorous statistical tests** (PCMCI, DML)
# - Never trade on discovered edges without **out-of-sample validation**
# - Consider **ensemble of methods** - edges confirmed by multiple approaches are more credible
#
# ### Limitations
# - Sensitive to hyperparameters (λ, thresholds)
# - Assume linearity (real relationships may be nonlinear)
# - Require stationarity (financial markets are non-stationary)
# - Bootstrap stability is computationally expensive

# %%
print("\n" + "=" * 60)
print("SUMMARY: NEURAL CAUSAL DISCOVERY")
print("=" * 60)

print(f"""
DATA:
  Assets: {", ".join(ASSETS)}
  Period: {returns.index.min().date()} to {returns.index.max().date()}
  Observations: {len(returns)}

SYNTHETIC VALIDATION:
  Precision: {SYNTHETIC_RESULTS["precision"]:.1%}
  Sensitivity: {SYNTHETIC_RESULTS["sensitivity"]:.1%}
  F1 Score: {SYNTHETIC_RESULTS["f1"]:.2f}

NOTEARS RESULTS:
  Edges discovered: {n_edges}
  Stable edges (≥50% bootstrap): {n_stable}
  Sparsity: {1 - n_edges / (len(ASSETS) * (len(ASSETS) - 1)):.1%}

VAR-LiNGAM RESULTS (causal-learn):
  Lagged edges discovered: {np.sum(np.abs(B_lag) > 0)}

PCMCI (7-asset panel):
  Significant lagged links (p<0.05): {pcmci_sig_links}

GRANGER CAUSALITY (FDR-corrected):
  Total pairs tested: {n_pairs}
  Significant after FDR: {len(granger_edges)}

KEY INSIGHT:
  Discovered edges are HYPOTHESES for further investigation.
  Only edges with high bootstrap stability AND economic rationale
  should be considered for trading applications.

NEXT STEPS:
  1. Validate stable edges on held-out period
  2. Estimate effect magnitudes with DML (see 03_econml_dml.py)
  3. Run BSTS event studies for specific relationships
  4. Apply López de Prado correction if testing multiple strategies
""")
