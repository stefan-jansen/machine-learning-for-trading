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

# %% [markdown] tags=[]
# # Multiple Testing and Selection Bias
#
# **Docker image**: `ml4t`
#
# **Chapter 7: Defining the Learning Task**
# **Section Reference**: 7.4 - Search Accounting and Multiple Testing
#
# ## Purpose
#
# This notebook addresses the **factor zoo problem**: when testing many signals,
# even with proper inference, the "best" will be inflated by selection bias.
# We cover FDR control and complexity-aware corrections.
#
# ## Learning Objectives
#
# 1. Understand why selecting the highest IC inflates the estimate
# 2. Apply Benjamini-Hochberg FDR for discovery control
# 3. Use Rademacher complexity (RAS) for correlated factors
# 4. Build a practical pipeline: HAC p-values → BH → discoveries
#
# ## The Factor Zoo Problem
#
# Harvey, Liu and Zhu (2016) counted several hundred factors published in the academic
# literature by 2015. Testing that many candidates at a conventional significance level
# means a double-digit number of false discoveries is the *expected* outcome even if not
# one of the factors is real - the arithmetic is the candidate count times the level, and
# it is printed later in this notebook against the parameters used here.
#
# Their recommendation is a materially stricter t-statistic threshold for declaring a new
# factor; the thresholds they propose are printed under **Harvey et al. (2016)
# Thresholds** below.
#
# ## Prerequisites
#
# - `06_ic_inference` - provides per-factor HAC inference whose p-values feed
#   the BH/Holm/Rademacher corrections here.
# - Familiarity with the family-wise error rate (FWER) and false discovery
#   rate (FDR), and with the Sharpe ratio's distribution under selection.

# %% tags=[]
"""Multiple Testing - Bonferroni, FDR, and deflated Sharpe corrections for strategy evaluation."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from IPython.display import display
from ml4t.diagnostic.evaluation.stats import (
    benjamini_hochberg_fdr,
    compute_min_trl,
    compute_pbo,
    deflated_sharpe_ratio,
    holm_bonferroni,
    min_trl_fwer,
    multiple_testing_summary,
    rademacher_complexity,
    ras_ic_adjustment,
)
from ml4t.diagnostic.metrics import compute_ic_hac_stats, pooled_ic
from plotly.subplots import make_subplots
from scipy import stats

from data import load_etfs
from utils.paths import get_chapter_dir
from utils.reproducibility import set_global_seeds
from utils.style import (  # importing utils.style activates the ml4t Plotly template
    COLORS,
    show_plotly_with_alt,
)

# %% tags=["parameters"]
SEED = 42
# Resolved from the chapter, not the working directory: the runner sets cwd to the chapter
# dir, so a repo-relative literal writes the publication artifact one level too deep and
# the book figure pipeline keeps reading an older copy at the intended path.
OUTPUT_DIR = get_chapter_dir(7) / "output"
N_FACTORS = 100
N_PERIODS = 252
N_ASSETS = 50
N_FIGURE_SIMS = 200
N_RAD_SIMS = 5000
N_FACTORS_ZOO = 300
N_TRUE_ZOO = 15
N_PERIODS_ZOO = 1260
N_ASSETS_ZOO = 100
ETF_START_DATE = "2010-01-01"
ETF_LABEL_HORIZON = 5  # drives both the fwd return and its HAC truncation
# The synthetic panels draw each period independently, so their labels do not overlap.
# Declaring a one-period horizon states that, rather than leaving the library to guess.
NON_OVERLAPPING = 1
N_RAD_ETF = 5000
N_STRATEGIES_DSR = 50
N_DAYS_DSR = 756
N_STRAT_PBO = 20
N_COMBOS_PBO = 50

# %% tags=[]
set_global_seeds(SEED)


# %% [markdown] tags=[]
# ## The Selection Bias Problem
#
# When testing N factors and keeping the highest-scoring one:
# - **Observed IC**: max(IC₁, IC₂, ..., ICₙ)
# - **True IC**: Often much lower
#
# Under the null (all factors are noise), the expected maximum is:
#
# $$E[\max IC] \approx \sqrt{2 \ln N} \times \sigma_{IC}$$
#
# This is the "expected best by chance" - the selection bias.

# %% tags=[]
# Simulate the selection bias problem with synthetic factors
rng = np.random.default_rng(42)

n_factors = N_FACTORS
n_periods = N_PERIODS
n_assets = N_ASSETS

# Generate factors - ALL are noise (no true predictive power)
factor_signals = rng.standard_normal((n_periods, n_assets, n_factors))
forward_returns = rng.standard_normal((n_periods, n_assets)) * 0.02

# Compute IC for each factor
observed_ics = []
ic_series_all = []

for f in range(n_factors):
    ics = []
    for t in range(n_periods):
        ic = pooled_ic(factor_signals[t, :, f], forward_returns[t, :], method="spearman")
        ics.append(ic)

    ic_series_all.append(ics)
    observed_ics.append(np.mean(ics))

observed_ics = np.array(observed_ics)
ic_series_all = np.array(ic_series_all)

# %% tags=[]
# The "best" factor by IC
best_idx = np.argmax(observed_ics)
best_ic = observed_ics[best_idx]

# Expected max under null: sqrt(2 * ln(N)) * std(IC)
# Note: We use the observed IC std as an estimate of sigma_IC
ic_std = np.std(observed_ics)
expected_max_null = np.sqrt(2 * np.log(n_factors)) * ic_std

print(
    pl.DataFrame(
        {
            "metric": [
                "Factors tested",
                "Sample (days)",
                "True IC",
                "IC mean",
                "IC std",
                "IC max (selected)",
                "IC min",
                "E[max] under null",
            ],
            "value": [
                f"{n_factors}",
                f"{n_periods}",
                "0.0000",
                f"{np.mean(observed_ics):.4f}",
                f"{ic_std:.4f}",
                f"{best_ic:.4f}",
                f"{np.min(observed_ics):.4f}",
                f"{expected_max_null:.4f}",
            ],
        }
    )
)

# %% tags=[]
# Visualize selection bias
fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=observed_ics,
        nbinsx=25,
        name="Factor ICs",
        marker_color=COLORS["blue"],
        opacity=0.7,
    )
)

fig.add_vline(
    x=best_ic,
    line_dash="dash",
    line_color=COLORS["amber"],
    annotation_text=f"Selected: {best_ic:.4f}",
)

fig.add_vline(
    x=0,
    line_dash="dot",
    line_color=COLORS["neutral"],
    annotation_text="True IC = 0",
)

fig.update_layout(
    title=f"Mean IC of {N_FACTORS} pure-noise factors, with the selected one marked",
    xaxis_title="Mean IC",
    yaxis_title="Count",
    height=350,
)

show_plotly_with_alt(
    fig,
    alt=(
        "A histogram of the mean IC of a hundred factors built entirely from noise, so "
        "every one of them has a true IC of zero. The distribution is roughly symmetric "
        "about the dotted line marking that true value, spanning about minus 0.02 to plus "
        "0.02. An amber dashed line marks the factor with the highest IC, standing at the "
        "extreme right edge of the distribution, well clear of the bulk and of the true "
        "value the whole sample was drawn from."
    ),
)

# %% [markdown] tags=[]
# The distribution is centred on zero because that is the truth about every factor in it.
# What the marked line shows is the maximum of a hundred draws from that distribution, and
# a maximum is not an estimate of the thing being maximised over. Reporting the selected
# factor's IC as its IC is the whole of the selection-bias problem: nothing was
# mismeasured, and the number is still wrong, because the selection step is not in it.

# %% [markdown] tags=[]
# ### Publication Figure Artifact
#
# The book figure for this section reads a compact NumPy artifact so formatting
# changes do not rerun the null simulation.

# %% tags=[]


def _vectorized_rank_ic(signals_3d: np.ndarray, returns_2d: np.ndarray) -> np.ndarray:
    sig_ranks = stats.rankdata(signals_3d, axis=1)
    ret_ranks = stats.rankdata(returns_2d, axis=1)
    sig_ranks -= sig_ranks.mean(axis=1, keepdims=True)
    ret_ranks -= ret_ranks.mean(axis=1, keepdims=True)
    numer = (sig_ranks * ret_ranks[:, :, np.newaxis]).sum(axis=1)
    denom_sig = np.sqrt((sig_ranks**2).sum(axis=1))
    denom_ret = np.sqrt((ret_ranks**2).sum(axis=1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(numer / (denom_sig * denom_ret), nan=0.0).mean(axis=0)


def _vectorized_ic_with_pvals(
    signals_3d: np.ndarray, returns_2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sig_ranks = stats.rankdata(signals_3d, axis=1)
    ret_ranks = stats.rankdata(returns_2d, axis=1)
    sig_ranks -= sig_ranks.mean(axis=1, keepdims=True)
    ret_ranks -= ret_ranks.mean(axis=1, keepdims=True)
    numer = (sig_ranks * ret_ranks[:, :, np.newaxis]).sum(axis=1)
    denom_sig = np.sqrt((sig_ranks**2).sum(axis=1))
    denom_ret = np.sqrt((ret_ranks**2).sum(axis=1, keepdims=True))
    with np.errstate(divide="ignore", invalid="ignore"):
        ics = np.nan_to_num(numer / (denom_sig * denom_ret), nan=0.0)
    mean_ics = ics.mean(axis=0)
    se_ics = ics.std(axis=0) / np.sqrt(ics.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.nan_to_num(mean_ics / se_ics, nan=0.0)
    p_values_figure = 2 * stats.norm.sf(np.abs(t_stats))
    return mean_ics, p_values_figure


def write_figure_7_6_artifact() -> Path:
    figure_rng = np.random.default_rng(SEED)
    figure_signals = figure_rng.standard_normal((N_PERIODS, N_ASSETS, N_FACTORS))
    figure_returns = figure_rng.standard_normal((N_PERIODS, N_ASSETS)) * 0.02
    figure_observed_ics = _vectorized_rank_ic(figure_signals, figure_returns)
    figure_best_ic = np.max(figure_observed_ics)
    figure_expected_max = np.sqrt(2 * np.log(N_FACTORS)) * np.std(figure_observed_ics)

    best_ics = np.empty(N_FIGURE_SIMS)
    n_naive_reject = np.empty(N_FIGURE_SIMS, dtype=int)
    n_bh_reject = np.empty(N_FIGURE_SIMS, dtype=int)
    bh_threshold = np.arange(1, N_FACTORS + 1) / N_FACTORS * 0.05

    for sim in range(N_FIGURE_SIMS):
        sim_signals = figure_rng.standard_normal((N_PERIODS, N_ASSETS, N_FACTORS))
        sim_returns = figure_rng.standard_normal((N_PERIODS, N_ASSETS)) * 0.02
        sim_ics, sim_pvals = _vectorized_ic_with_pvals(sim_signals, sim_returns)
        best_ics[sim] = np.max(sim_ics)
        n_naive_reject[sim] = int(np.sum(sim_pvals < 0.05))
        sorted_p = np.sort(sim_pvals)
        reject_idx = np.where(sorted_p <= bh_threshold)[0]
        n_bh_reject[sim] = int(reject_idx[-1] + 1) if len(reject_idx) else 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT_DIR / "figure_7_6_multiple_testing.npz"
    np.savez(
        artifact,
        observed_ics=figure_observed_ics,
        best_ic=figure_best_ic,
        expected_max=figure_expected_max,
        best_ics=best_ics,
        n_naive_reject=n_naive_reject,
        n_bh_reject=n_bh_reject,
        n_factors=N_FACTORS,
        n_sims=N_FIGURE_SIMS,
    )
    return artifact


figure_7_6_artifact = write_figure_7_6_artifact()
print(f"Wrote publication figure artifact: {figure_7_6_artifact}")

# %% [markdown] tags=[]
# ## Benjamini-Hochberg FDR Control
#
# **False Discovery Rate (FDR)** controls the expected proportion of false
# discoveries among rejections:
#
# $$FDR = E\left[\frac{\text{False Positives}}{\text{All Discoveries}}\right]$$
#
# The BH procedure:
# 1. Sort p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₙ₎
# 2. Find largest k where p₍ₖ₎ ≤ (k/n) × α
# 3. Reject hypotheses 1, 2, ..., k

# %% [markdown] tags=[]
# The p-values below are HAC-adjusted rather than naive. The correction procedures that
# follow take p-values as given, so feeding them naive ones would leave the dependence
# problem from `06_ic_inference` untouched and simply carry it through the correction.
#
# Every HAC call on a synthetic panel passes `NON_OVERLAPPING`, because these panels draw
# each period independently and their labels therefore do not overlap. That is a claim
# about the data, and it is worth making explicitly: omitting the argument leaves the
# library to infer the bandwidth from sample size alone and to warn that it may be
# anti-conservative, which is the right warning for real overlapping labels and the wrong
# one here. The ETF search later in the notebook passes its actual horizon instead.

# %% tags=[]
p_values = []
for f in range(n_factors):
    hac_stats = compute_ic_hac_stats(ic_series_all[f], label_horizon=NON_OVERLAPPING)
    p_values.append(hac_stats["p_value"])

p_values = np.array(p_values)

# Apply Benjamini-Hochberg
bh_result = benjamini_hochberg_fdr(p_values, alpha=0.05, return_details=True)

# Count discoveries
naive_significant = np.sum(p_values < 0.05)
bh_significant = np.sum(bh_result["rejected"])

print(
    pl.DataFrame(
        {
            "method": ["Naive (p < 0.05)", "BH-FDR (alpha=0.05)"],
            "discoveries": [naive_significant, bh_significant],
            "expected_fp": [int(n_factors * 0.05), 0],
        }
    )
)

# %% tags=[]
# Visualize BH procedure
fig = make_subplots(rows=1, cols=2, subplot_titles=["P-Value Distribution", "BH Procedure"])

# P-value histogram (should be uniform under null)
fig.add_trace(
    go.Histogram(
        x=p_values,
        nbinsx=20,
        name="P-values",
        marker_color=COLORS["blue"],
    ),
    row=1,
    col=1,
)

fig.add_vline(x=0.05, line_dash="dash", line_color=COLORS["amber"], row=1, col=1)

# BH procedure visualization
sorted_idx = np.argsort(p_values)
sorted_p = p_values[sorted_idx]
ranks = np.arange(1, n_factors + 1)
bh_threshold = (ranks / n_factors) * 0.05

fig.add_trace(
    go.Scatter(
        x=ranks,
        y=sorted_p,
        mode="markers",
        name="Sorted p-values",
        marker=dict(size=5, color=COLORS["blue"]),
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=ranks,
        y=bh_threshold,
        mode="lines",
        name="BH threshold",
        line=dict(dash="dash", color=COLORS["amber"]),
    ),
    row=1,
    col=2,
)

fig.update_layout(
    height=350,
    title_text="P-values of the noise factors, and the Benjamini-Hochberg threshold",
)
fig.update_xaxes(title_text="P-value", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_xaxes(title_text="Rank", row=1, col=2)
fig.update_yaxes(title_text="P-value", row=1, col=2)

show_plotly_with_alt(
    fig,
    alt=(
        "Two panels. The left panel is a histogram of the p-values across the hundred "
        "noise factors: roughly flat between zero and one, which is what a uniform "
        "distribution under a true null looks like, with a dashed amber line at the "
        "significance level near the left edge. The right panel plots the sorted p-values "
        "against their rank as a rising curve from near zero to one, with the "
        "Benjamini-Hochberg threshold drawn as a dashed amber line rising almost flat "
        "along the bottom. The sorted curve sits above the threshold line everywhere "
        "except at the very lowest ranks, so no p-value is far enough below it to be "
        "declared a discovery."
    ),
)

# %% [markdown] tags=[]
# ### Holm-Bonferroni FWER Control
#
# BH controls **FDR** (the expected proportion of false discoveries among rejections).
# Holm-Bonferroni controls **FWER** (the probability of making *any* false discovery).
#
# Use FWER when even one false positive is unacceptable - e.g., deploying a new
# strategy that incurs real capital risk.

# %% tags=[]
# Apply Holm-Bonferroni to the same p-values
holm_result = holm_bonferroni(p_values, alpha=0.05)
holm_significant = np.sum(holm_result["rejected"])

print(
    pl.DataFrame(
        {
            "method": ["Naive (p < 0.05)", "BH-FDR (alpha=0.05)", "Holm-Bonferroni (alpha=0.05)"],
            "discoveries": [naive_significant, bh_significant, holm_significant],
            "controls": ["Nothing", "FDR", "FWER"],
            "guarantee": [
                "None",
                "E[FP/discoveries] <= alpha",
                "P(any FP) <= alpha",
            ],
        }
    )
)

# %% [markdown] tags=[]
# ## Rademacher Complexity (RAS)
#
# When factors are correlated, Rademacher complexity provides a sharper bound
# than assuming independence. The RAS (Rademacher Anti-Serum) adjustment
# accounts for the actual complexity of the hypothesis class.
#
# Key insight: Testing 100 variants of the same factor is less risky than
# testing 100 truly independent factors.

# %% [markdown] tags=[]
# ### Two scales, and why the distinction matters
#
# Rademacher complexity is estimated from a matrix of per-period performances, and
# **it comes out in whatever units that matrix is in**. Two different questions here
# need two different scales, and mixing them is a silent error:
#
# - *How correlated is this candidate set?* Compare $\hat{R}$ to Massart's bound
#   $\sqrt{2\log N / T}$. Massart bounds the maximum of $N$ **standardized** means,
#   so this comparison is only meaningful on standardized ICs.
# - *How much do I deduct from an observed IC?* The RAS bound
#   $\theta_N \ge \hat{\theta}_N - 2\hat{R} - 2\kappa\sqrt{\log(2/\delta)/T}$
#   subtracts $2\hat{R}$ from an IC, so here $\hat{R}$ must be in **IC units**.
#
# Standardizing divides each factor's column by its own **per-period** IC standard
# deviation - the dispersion of that factor's IC across dates, not the dispersion of
# the averaged ICs across factors. The two differ by more than an order of magnitude
# here, and the cell below prints both so the conversion is checkable rather than
# asserted. Feeding the standardized $\hat{R}$ into the adjustment would deduct a
# penalty many times larger than any IC in the set - the bound would reject
# everything, and would do so no matter what the data said.

# %% tags=[]
# Compute Rademacher complexity on both scales
ic_matrix = ic_series_all.T  # Shape: (T, N)
ic_matrix_norm = (ic_matrix - np.mean(ic_matrix, axis=0)) / np.std(ic_matrix, axis=0, ddof=1)

n_rad_sims = N_RAD_SIMS
# Standardized: comparable to Massart, answers "how correlated is the candidate set?"
R_hat_norm = rademacher_complexity(ic_matrix_norm, n_simulations=n_rad_sims, random_state=42)
# Raw IC units: the scale the RAS deduction below is applied on
R_hat = rademacher_complexity(ic_matrix, n_simulations=n_rad_sims, random_state=42)

# Massart's bound (theoretical max for independent factors)
massart_bound = np.sqrt(2 * np.log(n_factors) / n_periods)

print(
    pl.DataFrame(
        {
            "metric": [
                "R-hat (standardized)",
                "Massart bound",
                "Ratio",
                "R-hat (IC units, used by RAS)",
                "per-period IC std (mean over factors, the divisor)",
                "implied scale: R-hat raw / R-hat standardized",
                "std of the averaged ICs across factors (NOT the divisor)",
            ],
            "value": [
                f"{R_hat_norm:.4f}",
                f"{massart_bound:.4f}",
                f"{R_hat_norm / massart_bound:.1%}",
                f"{R_hat:.6f}",
                f"{np.mean(np.std(ic_matrix, axis=0, ddof=1)):.6f}",
                f"{R_hat / R_hat_norm:.6f}",
                f"{np.std(observed_ics, ddof=1):.6f}",
            ],
        }
    )
)

# %% [markdown] tags=[]
# A ratio near one means the factors are nearly independent and the full multiple-testing
# penalty applies. A ratio well below one signals correlation among the candidates, so the
# effective hypothesis count is lower than the nominal count.
#
# The RAS penalty is an absolute deduction - twice the Rademacher average plus the
# estimation term - rather than a proportional shrinkage, so the cells below report both
# components and the resulting lower bound. Significance follows the library's own
# convention: an adjusted IC above zero.

# %% [markdown] tags=[]
# `kappa` is the bound the concentration (Hoeffding) term needs, and it bounds the
# *per-period* IC observations that get averaged - not the averaged IC. That is the same
# units confusion as the complexity term above, one term to the right, and it is easy to
# make because the averaged ICs are tiny: their magnitudes invite a small kappa, while a
# per-period Spearman IC is supported on the whole interval from minus one to one.
#
# **The reported bound uses the full Spearman support.** Hoeffding needs a bound fixed
# *before* the data is seen. The observed sample maximum is a function of the same sample
# the bound is being computed on, so substituting it does not give a conservative bound
# with a smaller constant - it gives no valid coverage guarantee at all, and the
# "significant" flag downstream would then mean nothing.
#
# The empirical maximum is computed too and shown beside it as a **sensitivity calculation
# only**: the size of the estimation term if one were willing to assume the observed range
# persists. No significance claim is read off that row. It is here because the gap between
# the two is the honest cost of a distribution-free bound on one year of data, and that
# cost is invisible if only one value is shown.

# %% tags=[]
KAPPA = 1.0  # Spearman IC support: valid without assumptions, and used for inference
kappa_empirical = float(np.max(np.abs(ic_matrix)))  # sensitivity only, data-dependent

ras = ras_ic_adjustment(
    observed_ic=observed_ics,
    complexity=R_hat,
    n_samples=n_periods,
    delta=0.05,
    kappa=KAPPA,
    return_result=True,
)
ras_sensitivity = ras_ic_adjustment(
    observed_ic=observed_ics,
    complexity=R_hat,
    n_samples=n_periods,
    delta=0.05,
    kappa=kappa_empirical,
    return_result=True,
)
print(
    pl.DataFrame(
        {
            "kappa": [
                f"{KAPPA:.4f}  (Spearman support)",
                f"{kappa_empirical:.4f}  (observed per-period |IC| max)",
            ],
            "role": ["REPORTED bound", "sensitivity only (data-dependent)"],
            "best adjusted IC": [
                f"{np.max(ras.adjusted_values):+.4f}",
                f"{np.max(ras_sensitivity.adjusted_values):+.4f}",
            ],
            "significant": [
                f"{int(np.sum(ras.adjusted_values > 0))}/{n_factors}",
                "not a valid claim",
            ],
        }
    )
)
adjusted_ics = ras.adjusted_values

# Both components are printed: which dominates depends on kappa, N and T.
n_positive_raw = int(np.sum(observed_ics > 0))

print(
    pl.DataFrame(
        {
            "metric": [
                "Positive IC before RAS",
                "Significant after RAS (adj IC > 0)",
                "Best observed IC",
                "Data-snooping penalty (2 R-hat)",
                "Estimation error",
                "Best adjusted IC (lower bound)",
            ],
            "value": [
                f"{n_positive_raw}/{n_factors}",
                f"{ras.n_significant}/{n_factors}",
                f"{best_ic:.4f}",
                f"{ras.data_snooping_penalty:.4f}",
                f"{ras.estimation_error:.4f}",
                f"{adjusted_ics[best_idx]:.4f}",
            ],
        }
    )
)

# %% [markdown] tags=[]
# All 100 factors are pure noise, and the RAS lower bound reflects that: no factor's
# conservative lower bound clears zero, though a good many show a positive raw IC by
# chance. The penalty is an absolute deduction in IC units, not a percentage haircut
# on each IC, and it has two parts. The data-snooping term $2\hat{R}$ is the price of
# having searched the candidate set at all; the estimation term
# $2\kappa\sqrt{\log(2/\delta)/T}$ is the price of a finite sample.
#
# Both terms are now on the IC scale, so their sizes can be compared and the
# comparison means something - read them off the table above rather than from here,
# because which one dominates is not a fixed fact about the method. It moves with
# $\kappa$, with $N$ and with $T$: the search term scales with the number and
# correlation of the candidates, the estimation term with $\kappa/\sqrt{T}$. On this
# panel - a distribution-free $\kappa$ and a single year of data - the finite sample
# is much the more expensive of the two. Lengthen the sample or widen the candidate
# set and that ordering changes.
#
# This paragraph has now been written wrong twice, in both directions, which is the
# argument for printing the components instead of narrating them: an ordering asserted in
# prose outlives the re-run that invalidates it.
#
# Read against the largest observed IC, the total deduction is many times that IC -
# enough to sink every candidate, which is correct, because every candidate here is noise
# by construction.
#
# The point of putting the complexity in IC units is that the comparison becomes a
# statement about the data at all. On the standardized scale the search penalty alone was
# an order of magnitude larger than the largest IC in the set, printed above, and it would
# have rejected everything whatever the ICs were. A bound that returns the same answer for
# every input is not measuring anything.

# %% [markdown] tags=[]
# ## Harvey et al. (2016) Thresholds
#
# Based on the factor zoo of several hundred published factors, Harvey et al. recommend
# stricter t-statistic thresholds for declaring a new factor than the one a single test
# would use. They are declared and printed below rather than typed into prose, because the
# cells that follow apply them and the two should not be able to drift apart.

# %% tags=["results"]
HARVEY_THRESHOLDS = (
    ("traditional", 2.0, "the level a single test would use"),
    ("modern", 3.0, "accounts for the factors already searched"),
    ("strict", 3.5, "for a paper claiming a new factor"),
)
SINGLE_TEST_ALPHA = 0.05

print(f"{'context':<14}{'t >':>6}   rationale")
print("-" * 66)
for label, threshold, rationale in HARVEY_THRESHOLDS:
    print(f"{label:<14}{threshold:>6.1f}   {rationale}")

print(
    f"\nAt a {SINGLE_TEST_ALPHA:.0%} level, searching {N_FACTORS_ZOO} candidates that are "
    f"all noise still yields\n{N_FACTORS_ZOO * SINGLE_TEST_ALPHA:.0f} expected "
    f"'discoveries' - which is the reason the threshold moves."
)

# %% tags=[]
# Simulate factor zoo scenario
n_factors_zoo = N_FACTORS_ZOO
n_true = N_TRUE_ZOO
n_periods_zoo = N_PERIODS_ZOO
n_assets_zoo = N_ASSETS_ZOO

true_ic = 0.03

print(
    f"Factor zoo: {n_factors_zoo} factors ({n_true} true, IC={true_ic}), "
    f"{n_periods_zoo} days ({n_periods_zoo // 252} years), {n_assets_zoo} assets"
)

# %% tags=[]
# Generate factor signals and returns
rng_zoo = np.random.default_rng(123)
factor_signals_zoo = rng_zoo.standard_normal((n_periods_zoo, n_assets_zoo, n_factors_zoo))

# Returns = base noise + contribution from true factors only
base_returns = rng_zoo.standard_normal((n_periods_zoo, n_assets_zoo)) * 0.02
true_signal = factor_signals_zoo[:, :, :n_true].sum(axis=2) * true_ic * 0.01
forward_returns_zoo = base_returns + true_signal

# Compute IC and HAC p-values
zoo_results = []
is_true_factor = np.array([f < n_true for f in range(n_factors_zoo)])

for f in range(n_factors_zoo):
    ics = []
    for t in range(n_periods_zoo):
        ic = pooled_ic(factor_signals_zoo[t, :, f], forward_returns_zoo[t, :], method="spearman")
        ics.append(ic)

    hac = compute_ic_hac_stats(ics, label_horizon=NON_OVERLAPPING)

    zoo_results.append(
        {
            "factor": f"Factor_{f + 1:03d}",
            "mean_ic": np.mean(ics),
            "t_stat_hac": hac["t_stat"],
            "p_value_hac": hac["p_value"],
            "is_true": f < n_true,
        }
    )

zoo_df = pl.DataFrame(zoo_results)

# %% tags=[]
# Apply different thresholds
alpha = 0.05

# Naive: t > 2.0
naive_sig = zoo_df.filter(pl.col("t_stat_hac").abs() > 2.0)
naive_tp = naive_sig.filter(pl.col("is_true")).height
naive_fp = naive_sig.filter(~pl.col("is_true")).height

# Harvey threshold: t > 3.0
harvey_sig = zoo_df.filter(pl.col("t_stat_hac").abs() > 3.0)
harvey_tp = harvey_sig.filter(pl.col("is_true")).height
harvey_fp = harvey_sig.filter(~pl.col("is_true")).height

# BH-FDR
p_values_zoo = zoo_df["p_value_hac"].to_numpy()
bh_zoo = benjamini_hochberg_fdr(p_values_zoo, alpha=0.05, return_details=True)
bh_significant = bh_zoo["rejected"]
bh_tp = np.sum(bh_significant & is_true_factor)
bh_fp = np.sum(bh_significant & ~is_true_factor)

# Holm-Bonferroni (FWER control)
holm_zoo = holm_bonferroni(p_values_zoo, alpha=0.05)
holm_significant_zoo = np.array(holm_zoo["rejected"])
holm_tp = np.sum(holm_significant_zoo & is_true_factor)
holm_fp = np.sum(holm_significant_zoo & ~is_true_factor)


def _fdr(fp, total):
    """Compute realized false discovery rate as false positives / total discoveries."""
    return round(fp / total, 3) if total > 0 else 0.0


methods_data = [
    ("Naive (t > 2.0)", naive_sig.height, naive_tp, naive_fp),
    ("Harvey (t > 3.0)", harvey_sig.height, harvey_tp, harvey_fp),
    ("BH-FDR (alpha=0.05)", int(np.sum(bh_significant)), bh_tp, bh_fp),
    ("Holm-Bonf (alpha=0.05)", int(np.sum(holm_significant_zoo)), holm_tp, holm_fp),
]

print(
    pl.DataFrame(
        {
            "method": [m[0] for m in methods_data],
            "discoveries": [m[1] for m in methods_data],
            "true_pos": [m[2] for m in methods_data],
            "false_pos": [m[3] for m in methods_data],
            "realized_fdr": [_fdr(m[3], m[1]) for m in methods_data],
            "power": [round(m[2] / n_true, 3) for m in methods_data],
        }
    )
)

# %% tags=[]
# Visualize factor zoo results
fig = make_subplots(
    rows=1, cols=2, subplot_titles=["t-Statistic Distribution", "Method Comparison"]
)

# t-stat distribution
t_stats = zoo_df["t_stat_hac"].to_numpy()
true_mask = is_true_factor

fig.add_trace(
    go.Histogram(
        x=t_stats[~true_mask],
        name="Noise factors",
        marker_color=COLORS["neutral"],
        opacity=0.6,
        nbinsx=30,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Histogram(
        x=t_stats[true_mask],
        name="True factors",
        marker_color=COLORS["blue"],
        opacity=0.8,
        nbinsx=15,
    ),
    row=1,
    col=1,
)

# Threshold lines with annotations
for thresh, label, color in [(2.0, "t=2.0", COLORS["amber"]), (3.0, "t=3.0", COLORS["blue"])]:
    for sign in [1, -1]:
        fig.add_vline(x=sign * thresh, line_dash="dash", line_color=color, row=1, col=1)
    fig.add_annotation(
        x=thresh,
        y=1,
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=10, color=color),
        xanchor="left",
        yanchor="top",
        xshift=3,
        row=1,
        col=1,
    )

# Method comparison - colorblind-safe blue/orange
methods = ["Naive (t>2)", "Harvey (t>3)", "BH-FDR", "Holm-Bonf"]
tp_counts = [naive_tp, harvey_tp, bh_tp, holm_tp]
fp_counts = [naive_fp, harvey_fp, bh_fp, holm_fp]

fig.add_trace(
    go.Bar(x=methods, y=tp_counts, name="True Positives", marker_color=COLORS["blue"]),
    row=1,
    col=2,
)
fig.add_trace(
    go.Bar(x=methods, y=fp_counts, name="False Positives", marker_color=COLORS["amber"]),
    row=1,
    col=2,
)

fig.add_hline(
    y=n_true,
    line_dash="dot",
    line_color=COLORS["neutral"],
    row=1,
    col=2,
    annotation_text=f"N true = {n_true}",
    annotation_position="top left",
)

fig.update_layout(
    height=400,
    barmode="stack",
    title_text="Factor-zoo t-statistics, and discoveries by selection rule",
)
fig.update_xaxes(title_text="t-statistic (HAC)", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)

show_plotly_with_alt(
    fig,
    alt=(
        "Two panels. The left panel overlays the HAC t-statistics of the true factors and "
        "the noise factors; the noise distribution is a tall bell centred on zero, the "
        "true factors a low scatter reaching out to the right past a t of five, and "
        "dashed vertical lines mark the several candidate thresholds. The right panel is "
        "a stacked bar for each of four selection rules, splitting that rule's "
        "discoveries into true positives and false positives, with a dotted line at the "
        "number of factors that are genuinely non-null. The naive rule stands well above "
        "that line with a large false-positive block on top. The Harvey threshold keeps "
        "only a sliver of false positives, and the two correction procedures show none at "
        "all - the strictest of them landing below the line, having given up several "
        "genuine factors to get there."
    ),
)

# %% [markdown] tags=[]
# ## Practical Pipeline
#
# The recommended workflow for evaluating many factors:
#
# 1. **Compute HAC-adjusted p-values** for each factor
# 2. **Apply BH-FDR** to control false discovery rate
# 3. **Report adjusted p-values** alongside discoveries
# 4. **Consider RAS** if factors are correlated (e.g., parameter variants)

# %% tags=[]
# Build discovery table
discovery_df = zoo_df.with_columns(
    [
        pl.Series("bh_rejected", bh_zoo["rejected"]),
        pl.Series("adjusted_p", bh_zoo["adjusted_p_values"]),
    ]
)

# Filter to discoveries
discoveries = discovery_df.filter(pl.col("bh_rejected")).sort("mean_ic", descending=True)

print(f"BH-FDR discoveries: {discoveries.height}/{n_factors_zoo} factors")

if discoveries.height > 0:
    print(
        discoveries.select(
            ["factor", "mean_ic", "t_stat_hac", "p_value_hac", "adjusted_p", "is_true"]
        )
    )
else:
    print("No discoveries at α=0.05")
    print("\nTop 5 candidates by IC:")
    print(
        discovery_df.sort("mean_ic", descending=True)
        .head(5)
        .select(["factor", "mean_ic", "t_stat_hac", "p_value_hac", "adjusted_p", "is_true"])
    )

# %% [markdown] tags=[]
# ### Exploration vs. Confirmation Pass
#
# The chapter's *Separate exploration from confirmation* section recommends splitting
# evaluation into two passes:
#
# 1. **Exploration**: screen all candidates on the first portion of data,
#    promote based on fold stability rather than peak performance.
# 2. **Confirmation**: re-evaluate only promoted candidates on held-out data
#    with a reduced comparison set.
#
# The confirmation pass controls FDR more tightly because the search set
# shrinks to only the promoted candidates.

# %% tags=[]
# Split zoo data into exploration (first 80%) and confirmation (last 20%)
n_explore = int(n_periods_zoo * 0.8)

explore_signals = factor_signals_zoo[:n_explore]
explore_returns = forward_returns_zoo[:n_explore]
confirm_signals = factor_signals_zoo[n_explore:]
confirm_returns = forward_returns_zoo[n_explore:]

# Exploration pass: compute IC and HAC p-values for ALL factors
explore_p_values = np.zeros(n_factors_zoo)
explore_ics = np.zeros(n_factors_zoo)

for f in range(n_factors_zoo):
    ics = [
        pooled_ic(explore_signals[t, :, f], explore_returns[t, :], method="spearman")
        for t in range(n_explore)
    ]
    hac = compute_ic_hac_stats(ics, label_horizon=NON_OVERLAPPING)
    explore_p_values[f] = hac["p_value"]
    explore_ics[f] = np.mean(ics)

# BH on exploration pass (full search set)
explore_bh = benjamini_hochberg_fdr(explore_p_values, alpha=0.10, return_details=True)
promoted_idx = np.where(explore_bh["rejected"])[0]

# %% tags=[]
# Confirmation pass: re-evaluate ONLY promoted candidates
if len(promoted_idx) > 0:
    confirm_p_values = np.zeros(len(promoted_idx))
    confirm_ics = np.zeros(len(promoted_idx))

    for i, f in enumerate(promoted_idx):
        ics = [
            pooled_ic(confirm_signals[t, :, f], confirm_returns[t, :], method="spearman")
            for t in range(len(confirm_returns))
        ]
        hac = compute_ic_hac_stats(ics, label_horizon=NON_OVERLAPPING)
        confirm_p_values[i] = hac["p_value"]
        confirm_ics[i] = np.mean(ics)

    # BH on confirmation pass (reduced search set)
    confirm_bh = benjamini_hochberg_fdr(confirm_p_values, alpha=0.05, return_details=True)
    confirmed_mask = confirm_bh["rejected"]
    confirmed_idx = promoted_idx[confirmed_mask]

    # Count true/false positives at each stage
    explore_tp = np.sum(is_true_factor[promoted_idx])
    explore_fp = len(promoted_idx) - explore_tp
    confirm_tp = np.sum(is_true_factor[confirmed_idx])
    confirm_fp = len(confirmed_idx) - confirm_tp

    results = pl.DataFrame(
        {
            "stage": ["Exploration (all factors)", "Confirmation (promoted only)"],
            "search_set": [n_factors_zoo, len(promoted_idx)],
            "discoveries": [len(promoted_idx), len(confirmed_idx)],
            "true_positives": [int(explore_tp), int(confirm_tp)],
            "false_positives": [int(explore_fp), int(confirm_fp)],
            "realized_fdr": [
                round(explore_fp / len(promoted_idx), 3),
                round(confirm_fp / len(confirmed_idx), 3) if len(confirmed_idx) > 0 else 0.0,
            ],
        }
    )
    print(results)
else:
    print("No candidates promoted from exploration pass")

# %% [markdown] tags=[]
# The confirmation pass operates on a smaller search set (only the promoted
# candidates), so BH corrections are less aggressive. At the same time, using
# held-out data prevents the double-dipping that inflates exploration-pass
# discovery rates. This two-pass workflow is the practical implementation of
# the "separate exploration from confirmation" principle the chapter sets out.

# %% [markdown] tags=[]
# ### Applied Example: ETF Feature Search
#
# The HAC call in the search below passes `label_horizon`, because the forward return is
# sampled daily over a multi-day window and this IC series is therefore overlapping.
# Without it the library picks the truncation from the sample size alone, which is the
# defect `06_ic_inference` corrects. The synthetic factor zoos earlier in this notebook
# draw their periods independently, so the automatic rule is right for those and only for
# those.
#
# The synthetic simulations above use known ground truth to verify the
# corrections work. Now we apply the same pipeline to real features on
# the ETF universe - the same data used in notebooks 05, 06, and 08.
#
# We compute 13 candidate features (momentum at 6 lookbacks, reversal
# at 3 horizons, realized volatility, and volume ratios) and test each
# for IC significance with HAC inference. After BH-FDR correction for
# 13 simultaneous tests, how many survive?

# %% tags=[]
# Load ETF data and compute candidate features
etfs_real = load_etfs()
etf_start = datetime.strptime(ETF_START_DATE, "%Y-%m-%d")
etf_end = datetime(2024, 1, 1)

TREASURY_SYMS = ["IEF", "TLT", "SHY", "AGG", "BND", "TIP", "GOVT", "BNDX", "VGSH"]

panel = (
    etfs_real.filter(
        (pl.col("timestamp") >= etf_start)
        & (pl.col("timestamp") < etf_end)
        & ~pl.col("symbol").is_in(TREASURY_SYMS)
    )
    .sort(["symbol", "timestamp"])
    .with_columns(
        (pl.col("close").shift(-ETF_LABEL_HORIZON).over("symbol") / pl.col("close"))
        .log()
        .alias("fwd_5d"),
        pl.col("close").pct_change().shift(1).over("symbol").alias("ret_lag1"),
    )
    .with_columns(
        *[
            (pl.col("close") / pl.col("close").shift(lb).over("symbol") - 1).alias(f"mom_{lb}d")
            for lb in [5, 10, 20, 40, 60, 120]
        ],
        (-pl.col("ret_lag1")).alias("rev_1d"),
        (-pl.col("ret_lag1").rolling_mean(3).over("symbol")).alias("rev_3d"),
        (-pl.col("ret_lag1").rolling_mean(5).over("symbol")).alias("rev_5d"),
        (pl.col("ret_lag1").rolling_std(10).over("symbol") * np.sqrt(252)).alias("rvol_10d"),
        (pl.col("ret_lag1").rolling_std(20).over("symbol") * np.sqrt(252)).alias("rvol_20d"),
        (
            pl.col("volume").rolling_mean(5).over("symbol")
            / pl.col("volume").rolling_mean(20).over("symbol")
        ).alias("vratio_5_20"),
        (
            pl.col("volume").rolling_mean(20).over("symbol")
            / pl.col("volume").rolling_mean(60).over("symbol")
        ).alias("vratio_20_60"),
    )
)

# %% tags=[]
FEAT_COLS = {
    "mom_5d": "Momentum 5d",
    "mom_10d": "Momentum 10d",
    "mom_20d": "Momentum 20d",
    "mom_40d": "Momentum 40d",
    "mom_60d": "Momentum 60d",
    "mom_120d": "Momentum 120d",
    "rev_1d": "Reversal 1d",
    "rev_3d": "Reversal 3d",
    "rev_5d": "Reversal 5d",
    "rvol_10d": "RVol 10d",
    "rvol_20d": "RVol 20d",
    "vratio_5_20": "Vol Ratio 5/20",
    "vratio_20_60": "Vol Ratio 20/60",
}

panel = panel.drop_nulls(subset=["fwd_5d"] + list(FEAT_COLS.keys()))

print(
    f"ETF panel: {len(panel):,} rows, {panel['symbol'].n_unique()} symbols, "
    f"{panel['timestamp'].n_unique():,} dates"
)
print(f"Candidate features: {len(FEAT_COLS)}")

# %% tags=[]
# Cross-sectional IC with HAC inference for each feature
groups = panel.partition_by("timestamp", as_dict=True)

etf_test_results = []
all_ic_series = []

for col, name in FEAT_COLS.items():
    ics = []
    for _key, grp in groups.items():
        if len(grp) < 20:
            continue
        f, y = grp[col].to_numpy(), grp["fwd_5d"].to_numpy()
        if np.std(f) < 1e-10 or np.std(y) < 1e-10:
            continue
        rho, _ = stats.spearmanr(f, y)
        if not np.isnan(rho):
            ics.append(rho)

    # Horizon-aware truncation; see the markdown above this cell.
    hac = compute_ic_hac_stats(ics, label_horizon=ETF_LABEL_HORIZON)
    etf_test_results.append(
        {"feature": name, "ic": hac["mean_ic"], "t_hac": hac["t_stat"], "p_hac": hac["p_value"]}
    )
    all_ic_series.append(ics)

# %% tags=[]
# Apply BH-FDR, Holm-Bonferroni, and Rademacher analysis
etf_p = np.array([r["p_hac"] for r in etf_test_results])
etf_bh = benjamini_hochberg_fdr(etf_p, alpha=0.05, return_details=True)
etf_holm = holm_bonferroni(etf_p, alpha=0.05)

# Rademacher complexity on the correlated IC matrix
min_len = min(len(s) for s in all_ic_series) if all_ic_series else 0
if min_len > 1:
    ic_mat = np.column_stack([s[:min_len] for s in all_ic_series])
    ic_z = (ic_mat - ic_mat.mean(axis=0)) / (ic_mat.std(axis=0, ddof=1) + 1e-10)
    n_rad_etf = N_RAD_ETF
    R_etf = rademacher_complexity(ic_z, n_simulations=n_rad_etf, random_state=42)
    massart_etf = np.sqrt(2 * np.log(len(FEAT_COLS)) / min_len)
else:
    print("Warning: Insufficient data for Rademacher complexity (min_len=0), skipping")
    R_etf = float("nan")
    massart_etf = float("nan")

etf_results_df = pl.DataFrame(
    {
        "feature": [r["feature"] for r in etf_test_results],
        "IC": [round(r["ic"], 4) for r in etf_test_results],
        "HAC_t": [round(r["t_hac"], 2) for r in etf_test_results],
        "p_HAC": [round(r["p_hac"], 4) for r in etf_test_results],
        "BH": list(etf_bh["rejected"]),
        "Holm": list(etf_holm["rejected"]),
    }
).sort("p_HAC")
display(etf_results_df)

n_naive_etf = int(np.sum(etf_p < 0.05))
n_bh_etf = int(np.sum(etf_bh["rejected"]))
n_holm_etf = int(np.sum(etf_holm["rejected"]))

print(f"\nNaive (p < 0.05): {n_naive_etf}/{len(FEAT_COLS)}")
print(f"BH-FDR (alpha=0.05):  {n_bh_etf}/{len(FEAT_COLS)}")
print(f"Holm-Bonf (alpha=0.05): {n_holm_etf}/{len(FEAT_COLS)}")
print(
    f"\nRademacher: R_hat={R_etf:.4f}, Massart={massart_etf:.4f}, ratio={R_etf / massart_etf:.0%}"
)

# %% tags=[]
# IC by feature, colored by BH-FDR significance
sort_idx = np.argsort(etf_p)
sorted_names = [etf_test_results[i]["feature"] for i in sort_idx]
sorted_ics = [etf_test_results[i]["ic"] for i in sort_idx]
sorted_bh = [bool(etf_bh["rejected"][i]) for i in sort_idx]
colors_etf = [COLORS["amber"] if b else COLORS["neutral"] for b in sorted_bh]

fig = go.Figure(
    go.Bar(
        y=sorted_names,
        x=sorted_ics,
        orientation="h",
        marker_color=colors_etf,
    )
)
fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
fig.update_layout(
    title="Mean IC of the searched ETF features, ordered by p-value",
    xaxis_title="Mean IC (HAC)",
    height=400,
)
show_plotly_with_alt(
    fig,
    alt=(
        "A horizontal bar chart of the mean HAC IC of each searched ETF feature, ordered "
        "so the least significant sits at the top and the most significant at the bottom. "
        "The bars are all the same neutral colour, because none of them was declared a "
        "discovery. Magnitudes run from about minus 0.008 for a five-day momentum feature "
        "to about plus 0.026 for a ten-day realized-volatility feature, and both signs "
        "appear among the smallest bars at the top. A dashed line marks zero."
    ),
)

# %% [markdown] tags=[]
# Every bar is drawn in the same colour because nothing cleared the threshold: the search
# found no feature it could call a discovery at this false-discovery rate. The two largest
# ICs are realized-volatility features and are not small in absolute terms, which is the
# point worth sitting with - a respectable-looking IC on a searched set is not evidence,
# and the correction is what says so.

# %% [markdown] tags=[]
# The Rademacher ratio printed above is well below one, reflecting the high correlation
# among the momentum variants - testing six lookbacks is not six independent trials. Even
# with that milder effective penalty, the features do not clear BH-FDR correction after
# HAC inference.
#
# Across the correlated momentum, reversal, volatility and volume features in this scan,
# and with HAC inference, no feature clears BH-FDR at the level set above.
# Univariate bivariate-IC screening is one filter; Chapters 11-12
# evaluate the same features in a multivariate setting where the relevant
# question is conditional contribution to a fitted model, not single-feature
# significance. The corrections here ensure that features *selected* for
# that pipeline have not been promoted purely by selection bias.

# %% [markdown] tags=[]
# ## Output: Discovery Report
#
# The JSON structure below is a template for production logging. Recording
# the search-set size, correction method, and per-method discovery counts
# alongside the Rademacher analysis makes the report self-contained and
# auditable.

# %% tags=[]
# Build structured output - base report with naive and Harvey thresholds
discovery_report = {
    "n_factors_tested": n_factors_zoo,
    "n_true_factors": n_true,
    "sample_periods": n_periods_zoo,
    "methods": {
        "naive_t2": {
            "threshold": "t > 2.0",
            "discoveries": int(naive_sig.height),
            "true_positives": int(naive_tp),
            "false_positives": int(naive_fp),
            "fdr": round(naive_fp / naive_sig.height, 3) if naive_sig.height > 0 else 0,
            "power": round(naive_tp / n_true, 3),
        },
        "harvey_t3": {
            "threshold": "t > 3.0",
            "discoveries": int(harvey_sig.height),
            "true_positives": int(harvey_tp),
            "false_positives": int(harvey_fp),
            "fdr": round(harvey_fp / harvey_sig.height, 3) if harvey_sig.height > 0 else 0,
            "power": round(harvey_tp / n_true, 3),
        },
    },
}

# %% tags=[]
# Add FDR, Holm-Bonferroni, and Rademacher analysis
discovery_report["methods"]["bh_fdr"] = {
    "alpha": 0.05,
    "discoveries": int(np.sum(bh_significant)),
    "true_positives": int(bh_tp),
    "false_positives": int(bh_fp),
    "fdr": round(bh_fp / np.sum(bh_significant), 3) if np.sum(bh_significant) > 0 else 0,
    "power": round(bh_tp / n_true, 3),
}
discovery_report["methods"]["holm_bonferroni"] = {
    "alpha": 0.05,
    "discoveries": int(np.sum(holm_significant_zoo)),
    "true_positives": int(holm_tp),
    "false_positives": int(holm_fp),
    "fdr": round(holm_fp / np.sum(holm_significant_zoo), 3)
    if np.sum(holm_significant_zoo) > 0
    else 0,
    "power": round(holm_tp / n_true, 3),
}
discovery_report["rademacher_analysis"] = {
    # Standardized scale: the one comparable to Massart's bound
    "empirical_complexity_standardized": round(float(R_hat_norm), 4),
    "massart_bound": round(float(massart_bound), 4),
    "complexity_ratio": round(float(R_hat_norm / massart_bound), 3),
    # IC units: the scale the RAS deduction is applied on
    "empirical_complexity_ic_units": round(float(R_hat), 6),
}

# %% tags=[]
print(json.dumps(discovery_report, indent=2))

# %% [markdown] tags=[]
# ## Deflated Sharpe Ratio (DSR)
#
# When the outcome is a strategy Sharpe ratio (not factor IC), the **Deflated Sharpe
# Ratio** (Bailey & López de Prado, 2014) adjusts for selection bias among
# multiple strategies tested.
#
# DSR answers: "Given that I tested N strategies and kept the highest Sharpe, what is the
# probability that this Sharpe is genuinely positive?"
#
# $$DSR = P\left[\hat{SR} > E\left[\max_{k \in K} SR_k\right] \mid H_0\right]$$

# %% tags=[]
# Simulate noise strategy return streams
rng_dsr = np.random.default_rng(99)
n_strategies = N_STRATEGIES_DSR
n_days = N_DAYS_DSR

strategy_returns = [rng_dsr.standard_normal(n_days) * 0.01 for _ in range(n_strategies)]

# Apply DSR to all strategies - picks best and adjusts
dsr_result = deflated_sharpe_ratio(strategy_returns, frequency="daily")

print(
    pl.DataFrame(
        {
            "metric": [
                "Strategies tested",
                "Sample (days)",
                "Best Sharpe (ann.)",
                "E[max] under null (ann.)",
                "Excess over E[max] (ann.)",
                "DSR probability",
                "Significant (95%)",
            ],
            "value": [
                f"{n_strategies}",
                f"{n_days}",
                f"{dsr_result.sharpe_ratio_annualized:.2f}",
                # expected_max_sharpe and deflated_sharpe are per-period, like
                # dsr_result.sharpe_ratio; annualize them so this column is one scale
                f"{dsr_result.expected_max_sharpe * np.sqrt(252):.2f}",
                f"{dsr_result.deflated_sharpe * np.sqrt(252):.2f}",
                f"{dsr_result.probability:.1%}",
                f"{dsr_result.is_significant}",
            ],
        }
    )
)

# %% [markdown] tags=[]
# **Every row above is on the annualized scale.** That matters more than it sounds: the
# library returns `sharpe_ratio`, `expected_max_sharpe` and `deflated_sharpe` per period
# and `sharpe_ratio_annualized` already annualized, so printing them in one column without
# converting puts a factor of the square root of the trading year between two adjacent
# rows. The comparison the table invites - the highest Sharpe against the null's expected
# maximum - is only meaningful once they are on the same scale.
#
# Read that way, the table above is stark. The highest-scoring of the pure-noise
# strategies posts a respectable annualized Sharpe, and the expected maximum *under the
# null* is barely below it. Almost the entire apparent performance is selection. What is
# left after deflation is a small excess, and the DSR probability falls well short of any
# conventional confidence level. `expected_max_sharpe` is the number that makes this
# legible: it says how good the highest-scoring strategy would look *even if none of them
# had any skill at all*.

# %% [markdown] tags=[]
# ## Probability of Backtest Overfitting (PBO)
#
# PBO (Bailey et al., 2017) estimates the probability that the strategy ranked first in
# sample lands in the bottom half out of sample. A PBO above one half is conventionally
# read as severe overfitting - with a caveat this notebook measures rather than states.
#
# The method uses combinatorial purged cross-validation (CPCV): split the data
# into S groups, choose half as in-sample, the rest as out-of-sample, and
# repeat across all $\binom{S}{S/2}$ combinations.
#
# Full implementation with CPCV splitting is covered in Chapter 16. Here we
# demonstrate the `compute_pbo()` function on pre-computed IS/OOS performance.

# %% tags=[]
# Simulate IS/OOS performance for strategies
rng_pbo = np.random.default_rng(77)
n_strat_pbo = N_STRAT_PBO
n_combos = N_COMBOS_PBO

# Under null: IS/OOS are independent noise
is_performance = rng_pbo.standard_normal((n_combos, n_strat_pbo))
oos_performance = rng_pbo.standard_normal((n_combos, n_strat_pbo))

# Add slight IS advantage to one strategy (overfitting)
is_performance[:, 0] += 1.5  # Looks great in-sample

pbo_result = compute_pbo(is_performance, oos_performance)

print(
    pl.DataFrame(
        {
            "metric": [
                "Strategies",
                "IS/OOS combinations",
                "PBO",
                "IS-best median OOS rank",
                "Degradation (mean +/- std)",
            ],
            "value": [
                f"{n_strat_pbo}",
                f"{n_combos}",
                f"{pbo_result.pbo:.1%}",
                f"{pbo_result.is_best_rank_oos_median:.1f} / {n_strat_pbo}",
                f"{pbo_result.degradation_mean:.2f} +/- {pbo_result.degradation_std:.2f}",
            ],
        }
    )
)

# %% [markdown] tags=[]
# The number to compare against here is **one half, not zero**. Strategy 0 was handed a
# large in-sample advantage and nothing else - out of sample it is the same standard normal
# as the other nineteen. So the strategy selected in sample is selected on noise, and its
# out-of-sample rank is uniform: it lands below the median about half the time. A PBO near
# one half is the *correct* reading of a selection that carries no real edge, and the
# median out-of-sample rank printed above - dead centre of the field - says the same thing
# a second way.
#
# This is why the "PBO above one half means severe overfitting" rule of thumb needs care.
# It is not a pass mark with a comfortable margin below it. A strategy whose edge is
# entirely an artifact of selection sits *at* one half, and the sampling error on a PBO
# estimated from this many combinations is wide enough that a point estimate somewhat
# below it is fully consistent with a strategy that has no edge at all. What would
# actually be reassuring is a PBO close to zero, together with a strategy that ranks
# first in sample and stays near the top of the out-of-sample ranking.
#
# PBO is a powerful complement to DSR. While DSR focuses on Sharpe inflation,
# PBO directly measures whether the in-sample best-performing configuration *degrades* out-of-sample.
# See Chapter 16 for applying PBO with actual CPCV backtest splits.

# %% [markdown] tags=[]
# The FWER adjustment in the table below is driven by how much the trial Sharpes disagree
# with each other: if every candidate scored identically, searching more of them would
# tell you nothing new. Setting the trial variance to zero therefore switches the
# correction off and every column would print the single-test answer. The dispersion used
# is the one the Deflated Sharpe search above actually exhibited, so the table reports the
# cost of that search rather than of a hypothetical one.

# %% [markdown] tags=[]
# ## Minimum Track Record Length (MinTRL)
#
# How long must a track record be before we trust a Sharpe ratio?
# `compute_min_trl()` gives the minimum number of observations needed for
# statistical significance, adjusted for non-normal returns (skewness, kurtosis,
# autocorrelation). `min_trl_fwer()` additionally adjusts for the number of
# strategies tested.

# %% tags=[]
# MinTRL table: varying Sharpe and number of strategies
sharpes = [0.5, 1.0, 1.5, 2.0]
n_trials_list = [1, 10, 100]

# Dispersion taken from the DSR search above; see the markdown ahead of this cell.
variance_trials_observed = dsr_result.variance_trials
print(
    f"Sharpe dispersion across the {n_strategies} strategies searched for the DSR: "
    f"variance={variance_trials_observed:.6f} (per-period sd={np.sqrt(variance_trials_observed):.4f})"
)

rows = []
for sr in sharpes:
    row = {"sharpe": sr}
    for n in n_trials_list:
        if n == 1:
            result = compute_min_trl(
                observed_sharpe=sr / np.sqrt(252),
                target_sharpe=0.0,
                frequency="daily",
            )
        else:
            result = min_trl_fwer(
                observed_sharpe=sr / np.sqrt(252),
                n_trials=n,
                variance_trials=variance_trials_observed,
                target_sharpe=0.0,
                frequency="daily",
            )
        years = result.min_trl_years
        row[f"N={n}"] = "never" if years == float("inf") else f"{years:.1f}y"
    rows.append(row)

mintrl_df = pl.DataFrame(rows)
display(mintrl_df)

# %% [markdown] tags=[]
# **Interpretation**: read across a row, not down a column. Every row lengthens as the
# search widens, and the rows do not lengthen at the same rate.
#
# The lowest-Sharpe row already needs more than a decade of daily data with no search at
# all, and once even a handful of candidates have been tried it cannot be confirmed at any
# length - which is what `never` in the table means, the required record growing faster
# than the evidence a longer record supplies. The next higher Sharpe is confirmable in a
# couple of years unsearched, needs longer than a career after a handful of candidates,
# and reaches `never` after a hundred. Only the highest-Sharpe row stays inside a working
# career all the way across.
#
# The columns differ only because the trial Sharpes differ. That is the whole mechanism:
# the FWER correction prices the *search*, so a search over candidates that all score
# alike costs nothing while a search over dispersed candidates is expensive. A higher
# Sharpe buys back room, but the ordering across a row never reverses.
#
# This connects to NB06's track record planning for IC: both IC and Sharpe
# require longer records than practitioners typically assume.

# %% [markdown] tags=[]
# ## One-Call Production Alternative
#
# `multiple_testing_summary()` wraps the manual HAC → BH/Holm pipeline into
# a single call. Use it after you've computed per-factor test results.

# %% tags=[]
# Build test results from the zoo simulation's pre-computed HAC statistics
test_results = [
    {
        "name": r["factor"],
        "p_value": r["p_value_hac"],
        "t_stat": r["t_stat_hac"],
    }
    for r in zoo_results
]

summary_bh = multiple_testing_summary(test_results, method="benjamini_hochberg", alpha=0.05)

summary_df = pl.DataFrame(
    {
        "metric": [
            "Tests",
            "Significant (uncorrected)",
            "Significant (BH-FDR)",
            "Correction method",
        ],
        "value": [
            str(summary_bh["n_tests"]),
            str(summary_bh["n_significant_uncorrected"]),
            str(summary_bh["n_significant_corrected"]),
            summary_bh["correction_method"],
        ],
    }
)
display(summary_df)

# %% [markdown] tags=[]
# ## Summary
#
# ### Key Concepts
#
# | Concept | Description |
# |---------|-------------|
# | **Selection Bias** | Best IC/Sharpe is inflated when testing many factors |
# | **BH-FDR** | Controls expected proportion of false discoveries |
# | **Holm-Bonferroni** | Controls probability of *any* false discovery (FWER) |
# | **Rademacher Complexity** | Sharper bound for correlated hypotheses |
# | **Deflated Sharpe Ratio** | Adjusts best Sharpe for selection among N strategies |
# | **PBO** | Probability that the IS best-performing config degrades worst OOS (Ch16 deep dive) |
# | **MinTRL** | Minimum track record for Sharpe significance |
# | **Harvey Threshold** | A raised t-statistic bar for factor discovery in published literature |
#
# ### When to Use What
#
# | Situation | Tool |
# |-----------|------|
# | Screening many factors (IC) | BH-FDR or Holm-Bonferroni |
# | Correlated factor variants | Rademacher (RAS) adjustment |
# | Selecting best strategy (Sharpe) | Deflated Sharpe Ratio |
# | Validating backtest results | PBO + DSR |
# | Planning data collection | MinTRL |
#
# ### Next Notebooks
#
# - [`08_causal_sanity_checks`](08_causal_sanity_checks.ipynb) - Causal falsification tests
# - Chapter 8 notebooks for factor orthogonality and robustness analysis
# - Chapter 16 for full PBO with CPCV backtest splits
