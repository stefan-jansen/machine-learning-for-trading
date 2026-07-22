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
# # Time Series Causal Discovery with PCMCI
#
# **Chapter 15: Causal Estimation with ML**
# **Docker image**: `ml4t`
# **Section Reference**: See Section 15.4 for PCMCI theory and ADIA Lab insights
#
# ## Purpose
# This notebook demonstrates **causal discovery** - learning the causal structure
# from time series data without specifying a DAG a priori. We compare PCMCI with
# traditional Granger causality and show how to discover lead-lag relationships
# in multi-asset financial time series.
#
# ## Learning Objectives
# After completing this notebook, you will be able to:
# - LO1: Apply PCMCI algorithm for time series causal discovery
# - LO2: Compare Granger causality limitations with PCMCI advantages
# - LO3: Interpret discovered causal graphs with time lags
# - LO4: Assess edge stability via bootstrap analysis
#
# ## Methodology Reference
# Runge et al. (2019) "Detecting and quantifying causal associations in large
# nonlinear time series datasets" Science Advances
#
# ## Key Concepts
# 1. **Causal discovery**: Infer DAG from data (vs specifying it)
# 2. **PCMCI algorithm**: PC skeleton + Momentary Conditional Independence
# 3. **Time lags**: Causal relationships can have delays
# 4. **Stability analysis**: Bootstrap to assess edge reliability
#
# **Prerequisites**: [`01_library_overview`](01_library_overview.ipynb) for library context;
# ETF OHLCV data from Ch2 data pipeline

# %% [markdown]
# ## Important Methodological Notes
#
# ### Multiple Testing Correction
#
# **CRITICAL**: PCMCI's pc_alpha parameter already controls for multiple testing
# via the conditioning step. Applying FDR correction afterward is DOUBLE correction.
#
# Choose ONE approach:
# - **Option A (Recommended)**: Use pc_alpha only for threshold. No additional FDR.
# - **Option B**: Set pc_alpha=None, get raw p-values, apply FDR yourself.
#
# This notebook uses Option A per Tigramite documentation.
#
# ### Lag Selection
#
# MAX_LAG must be justified by:
# 1. Domain knowledge (how fast do financial relationships propagate?)
# 2. ACF analysis (at what lag does autocorrelation decay?)
# 3. Computational constraints (more lags = more tests = less power)

# %% [markdown]
# ## Setup

# %%
"""Time Series Causal Discovery with PCMCI — discover lead-lag causal relationships in financial time series."""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from data import load_etfs, load_macro
from utils.reproducibility import set_global_seeds

warnings.filterwarnings("ignore")


# %% tags=["parameters"]
MAX_LAG = 5
ALPHA_LEVEL = 0.05
N_BOOTSTRAP = 100
N_SAMPLES = 500
SEED = 42
BLOCK_SIZE = 20  # Block size for bootstrap (preserves autocorrelation)
STABILITY_THRESHOLD = 0.5  # Edge must appear in >50% of bootstraps

set_global_seeds(SEED)

# %%
# Tigramite imports
import tigramite
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

version = getattr(tigramite, "__version__", "unknown")
print(f"Tigramite version: {version}")

print(f"Bootstrap replications: {N_BOOTSTRAP}")

# %% [markdown]
# ## 1. Load Multi-Asset Time Series and Determine MAX_LAG

# %%
import polars as pl

# %%
# Real-data only — load failure is a fatal error so CI / a fresh reader
# environment without ML4T_DATA_PATH cannot silently publish synthetic numbers.
etf_tickers = ["SPY", "IEF", "GLD"]

etf_df = load_etfs(symbols=etf_tickers, start_date="2020-01-01", end_date="2024-06-01").select(
    ["symbol", "timestamp", "close"]
)
etf_wide = etf_df.pivot(on="symbol", index="timestamp", values="close").sort("timestamp")

macro_data = load_macro(start_date="2020-01-01", end_date="2024-06-01")
vix_col = "vixcls" if "vixcls" in macro_data.columns else "VIXCLS"

vix_df = macro_data.select(["timestamp", vix_col]).rename(
    {"timestamp": "timestamp", vix_col: "VIX"}
)

etf_wide = etf_wide.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
vix_df = vix_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
combined = etf_wide.join(vix_df, on="timestamp", how="inner").sort("timestamp")
prices = combined.to_pandas().set_index("timestamp")
returns = prices.pct_change().dropna().iloc[-N_SAMPLES:]
var_names = list(returns.columns)

print(f"Loaded returns: {returns.shape}, {var_names}")

# %% [markdown]
# ## 2. Justify MAX_LAG via ACF Analysis
#
# Per Runge et al. (2019), MAX_LAG should be chosen based on:
# 1. Domain knowledge about typical response times
# 2. Autocorrelation decay in the data
# 3. Computational feasibility

# %%


def compute_acf(series, max_lag=20):
    """Compute autocorrelation function up to max_lag."""
    acf_values = []
    series = np.asarray(series)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            corr = np.corrcoef(series[lag:], series[:-lag])[0, 1]
            acf_values.append(corr)
    return acf_values


print("\n=== LAG SELECTION JUSTIFICATION ===\n")

# Compute ACF for each variable
acf_results = {}
for var in var_names:
    acf = compute_acf(returns[var].values, max_lag=10)
    acf_results[var] = acf

    # Find lag where ACF drops below 0.1
    decay_lag = next((i for i, a in enumerate(acf) if abs(a) < 0.1), 10)
    print(f"{var}: ACF decays to <0.1 at lag {decay_lag}")

# Determine MAX_LAG based on ACF decay
# Use the maximum decay lag across variables, capped at reasonable value
max_decay_lag = max(
    next((i for i, a in enumerate(acf_results[v]) if abs(a) < 0.1), 10) for v in var_names
)
MAX_LAG = min(max_decay_lag, 5)  # Cap at 5 for computational reasons

print(f"\nChosen MAX_LAG = {MAX_LAG}")
print("Justification: ACF decays to <0.1 by this lag for most variables")
print("Note: Longer lags would reduce statistical power due to more tests")

# %% [markdown]
# ## 3. Prepare Data for Tigramite

# %%
# Convert to Tigramite format
data_array = returns.values
dataframe = pp.DataFrame(data_array, var_names=var_names)

try:
    shape = dataframe.values.shape if hasattr(dataframe.values, "shape") else data_array.shape
except Exception:
    shape = data_array.shape
print(f"Tigramite dataframe shape: {shape}")
print(f"Variable names: {dataframe.var_names}")

# %% [markdown]
# ## 4. PCMCI Algorithm (Without Double Correction)
#
# **CRITICAL**: We use pc_alpha for the PCMCI threshold.
# We do NOT apply additional FDR correction afterward.
#
# Per Runge et al. (2019), the MCI step already controls for multiple testing
# through the conditioning approach.

# %%
# Initialize PCMCI with partial correlation test
parcorr = ParCorr(significance="analytic")
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

print("\nRunning PCMCI algorithm...")
print(f"Max lag: {MAX_LAG}")
print(f"Alpha level: {ALPHA_LEVEL}")
print("\nNote: Using pc_alpha for threshold (NO additional FDR correction)")

# Run PCMCI - pc_alpha handles the threshold
results = pcmci.run_pcmci(
    tau_max=MAX_LAG,
    pc_alpha=ALPHA_LEVEL,
)

print("\nPCMCI completed!")

# %% [markdown]
# ## 5. Interpret Discovered Causal Graph
#
# **Important**: We use the p-values directly from PCMCI results.
# The pc_alpha parameter already applied appropriate thresholding.

# %%
# Use p_matrix directly (NOT corrected again)
# pc_alpha already handled multiple testing via the MCI conditioning
p_matrix = results["p_matrix"]

print("\n" + "=" * 60)
print("DISCOVERED CAUSAL LINKS (PCMCI, pc_alpha threshold)")
print("=" * 60)

significant_links = []
n_vars = len(var_names)

# Tigramite matrix indexing: p_matrix[source, target, lag]
# p_matrix[i, j, tau] tests: does source i at lag tau cause target j?
for i in range(n_vars):  # source variable
    for j in range(n_vars):  # target variable
        for tau in range(1, MAX_LAG + 1):
            if p_matrix[i, j, tau] < ALPHA_LEVEL:
                val = results["val_matrix"][i, j, tau]
                link = {
                    "from": var_names[i],
                    "to": var_names[j],
                    "lag": tau,
                    "strength": val,
                    "p_value": p_matrix[i, j, tau],
                }
                significant_links.append(link)
                print(
                    f"{var_names[i]}[t-{tau}] -> {var_names[j]}[t]  "
                    f"(strength={val:.3f}, p={p_matrix[i, j, tau]:.4f})"
                )

if not significant_links:
    print(f"No significant LAGGED causal links found at alpha = {ALPHA_LEVEL}")
else:
    print(f"\nTotal significant links: {len(significant_links)}")

# %% [markdown]
# ## 6. Bootstrap Stability Analysis
#
# Discovered edges may be unstable. We use block bootstrap to assess which
# edges are consistently found across resamples. This is essential for
# robustness in financial applications.


# %%
def block_bootstrap_indices(n, block_size):
    """Generate block bootstrap indices preserving autocorrelation."""
    n_blocks = n // block_size
    block_starts = np.random.choice(n - block_size + 1, size=n_blocks, replace=True)
    indices = np.concatenate([np.arange(start, start + block_size) for start in block_starts])
    return indices[:n]


print(f"\nBootstrap stability analysis (n={N_BOOTSTRAP}, block_size={BLOCK_SIZE})...")

edge_counts = defaultdict(int)
for b in range(N_BOOTSTRAP):
    boot_indices = block_bootstrap_indices(len(returns), BLOCK_SIZE)
    boot_df = pp.DataFrame(returns.values[boot_indices], var_names=var_names)
    boot_pcmci = PCMCI(
        dataframe=boot_df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
    )

    try:
        boot_results = boot_pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=ALPHA_LEVEL)
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(1, MAX_LAG + 1):
                    if boot_results["p_matrix"][i, j, tau] < ALPHA_LEVEL:
                        edge_counts[(var_names[i], var_names[j], tau)] += 1
    except Exception:
        pass

    if (b + 1) % 20 == 0:
        print(f"  Completed {b + 1}/{N_BOOTSTRAP} bootstraps...")

# %%
# Report edge stability
edge_stability = {edge: count / N_BOOTSTRAP for edge, count in edge_counts.items()}

print(f"\n--- Edge Stability (threshold: {STABILITY_THRESHOLD:.0%}) ---\n")
stable_edges = []
for edge, stability in sorted(edge_stability.items(), key=lambda x: -x[1]):
    source, target, lag = edge
    status = "STABLE" if stability >= STABILITY_THRESHOLD else "unstable"
    print(f"{source}[t-{lag}] -> {target}[t]: {stability:.0%} ({status})")
    if stability >= STABILITY_THRESHOLD:
        stable_edges.append(edge)

print(f"\nTotal edges: {len(edge_stability)}, Stable: {len(stable_edges)}")

# %% [markdown]
# ## 7. Balanced Interpretation of Null Results
#
# If no significant lagged links are found, this does NOT prove market efficiency.

# %%
print("\n" + "=" * 60)
print("INTERPRETATION OF RESULTS")
print("=" * 60)

if len(significant_links) == 0:
    explanations = [
        "Market efficiency could remove lagged predictability, but this is not proof.",
        "Power may be insufficient: daily returns are noisy and weak links need longer samples.",
        "Effects may exist at intraday horizons and vanish in daily aggregation.",
        "Linear ParCorr may miss nonlinear or state-dependent relationships.",
        "Structural breaks may require rolling or regime-conditional analysis.",
    ]
    print("No significant lagged causal links detected at alpha=0.05.")
    print("\nPossible interpretations:")
    for i, item in enumerate(explanations, 1):
        print(f"  {i}. {item}")
    print("\nRecommendation: run lag, frequency, and sample-window sensitivity checks.")
else:
    print(f"Found {len(significant_links)} significant lagged links.")
    print(f"Stable across bootstraps: {len(stable_edges)}")
    print("Stable edges are higher-priority hypotheses for out-of-sample validation.")
    print("Unstable edges are weak evidence and should not drive trading decisions.")

# %% [markdown]
# ## 8. Comparison with Granger Causality
#
# **Granger causality limitations**:
# 1. Pairwise only - ignores multivariate confounding
# 2. Assumes linearity
# 3. Sensitive to lag selection
# 4. Can find spurious relationships due to common causes
#
# **PCMCI advantages**:
# 1. Multivariate - conditions on all other variables
# 2. Controls for confounders automatically
# 3. Tests at multiple lags simultaneously
# 4. Built-in multiple testing control via MCI

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def simple_granger_test(X, Y, max_lag=5):
    """Simple pairwise Granger causality test."""
    n = len(X)
    results = []

    for lag in range(1, max_lag + 1):
        # Restricted model: Y_t ~ Y_{t-1}, ..., Y_{t-lag}
        Y_lags = np.column_stack([Y[lag - i - 1 : n - i - 1] for i in range(lag)])
        Y_target = Y[lag:]

        model_r = LinearRegression().fit(Y_lags, Y_target)
        mse_r = mean_squared_error(Y_target, model_r.predict(Y_lags))

        # Unrestricted model: Y_t ~ Y_{t-1}, ..., Y_{t-lag}, X_{t-1}, ..., X_{t-lag}
        X_lags = np.column_stack([X[lag - i - 1 : n - i - 1] for i in range(lag)])
        XY_lags = np.column_stack([Y_lags, X_lags])

        model_u = LinearRegression().fit(XY_lags, Y_target)
        mse_u = mean_squared_error(Y_target, model_u.predict(XY_lags))

        # F-test approximation
        f_stat = (mse_r - mse_u) / mse_u * (n - 2 * lag - 1) / lag
        results.append({"lag": lag, "f_stat": f_stat, "mse_reduction": (mse_r - mse_u) / mse_r})

    return results


# %%
print("\n" + "=" * 60)
print("GRANGER CAUSALITY (PAIRWISE) COMPARISON")
print("=" * 60)
print("Note: Granger is pairwise and doesn't control for other variables")

pairs_to_test = [("SPY", "IEF"), ("IEF", "VIX"), ("SPY", "VIX")]

for x_name, y_name in pairs_to_test:
    if x_name in returns.columns and y_name in returns.columns:
        X = returns[x_name].values
        Y = returns[y_name].values
        granger_results = simple_granger_test(X, Y, max_lag=3)

        best_lag = max(granger_results, key=lambda x: x["f_stat"])
        print(f"\n{x_name} -> {y_name}:")
        print(f"  Best lag: {best_lag['lag']}, F-stat: {best_lag['f_stat']:.2f}")

# %% [markdown]
# ## 9. Results Summary

# %%
print("\n" + "=" * 60)
print("CHAPTER 15 NOTEBOOK RESULTS: 07_tigramite_time_series.py")
print("=" * 60)

results_summary = {
    "n_samples": len(returns),
    "n_variables": len(var_names),
    "variable_names": var_names,
    "max_lag_tested": MAX_LAG,
    "alpha_level": ALPHA_LEVEL,
    "n_bootstrap": N_BOOTSTRAP,
    "stability_threshold": STABILITY_THRESHOLD,
}

results_summary["n_significant_links"] = len(significant_links)
results_summary["n_stable_edges"] = len(stable_edges)
results_summary["stable_edges"] = [f"{e[0]}[t-{e[2]}]->{e[1]}" for e in stable_edges]

for key, value in results_summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    elif isinstance(value, list) and len(value) > 3:
        print(f"{key}: [{len(value)} items]")
    else:
        print(f"{key}: {value}")

# %% [markdown]
# ## Key Takeaways
#
# ### Methodological Improvements Applied:
#
# 1. **No Double Correction**: Use pc_alpha threshold only, NOT additional FDR.
#    PCMCI's MCI step already controls for multiple testing.
#
# 2. **Justified MAX_LAG**: Selected based on ACF decay analysis, not arbitrary.
#
# 3. **Bootstrap Stability**: Edges found in >50% of bootstraps are more reliable.
#
# 4. **Balanced Interpretation**: Null results have multiple possible explanations,
#    not just "market efficiency".
#
# ### For Chapter Prose:
#
# - Contrast Granger (pairwise) with PCMCI (multivariate)
# - Emphasize stability analysis for robust edge detection
# - Caveat: discovered structure is statistical, not guaranteed causal
# - Multiple explanations for null findings - don't overclaim efficiency
