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
# # S&P 500 Options — Volatility Risk Premium: Causal DML Estimation
#
# Does the volatility risk premium (VRP) *cause* lower hold-to-maturity
# straddle returns (`ret_to_expiry`), or is the cross-sectional VRP
# signal confounded by spread quality and volatility regime? This
# notebook applies DML to `vrp_21d` across the S&P 500 equity straddle
# panel.
#
# **Treatment rationale**: VRP — the gap between implied and realized
# volatility — represents compensation for bearing volatility risk.
# Realized volatility (21d) is the dominant confounder: it is
# mechanically correlated with both VRP (which is defined relative to
# it) and `ret_to_expiry` (since straddles realize their volatility
# over the holding period). VRP momentum (5d) and spread percentile
# capture persistence and liquidity effects.
#
# **Learning Objectives**:
# - Read a panel that clears both gates at conventional levels: HAC
#   significance + block-permutation refutation passes
# - Understand confounding bias — naive OLS understates the absolute
#   effect by about 50 % because RV/VRP momentum/spread absorb most of
#   the cross-sectional signal
# - See VRP as a causal compensation channel for volatility risk
#
# **Book Reference**: Chapter 15, Section 15.6 (Cross-Dataset Causal Evidence)
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb), [`04_model_based_features`](04_model_based_features.ipynb)

# %% [markdown]
# ## Identifying Assumptions
#
# DML estimates a causal effect under three assumptions:
# 1. **Conditional ignorability**: No unobserved confounders — all backdoor paths
#    between treatment and outcome are blocked by the observed confounders.
# 2. **Overlap (positivity)**: Every unit has a nonzero probability of receiving
#    any treatment level, conditional on confounders.
# 3. **SUTVA**: One unit's treatment doesn't affect another's outcome.
#
# These are untestable. The refutation test below provides indirect evidence
# but cannot prove the assumptions hold.

# %%
"""Causal DML — walk-forward estimation with refutation tests."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from case_studies.utils.causal import (
    classify_refutation,
    embargo_from_buffer,
    format_dml_summary,
    register_causal_run,
    run_dml_analysis,
)
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_options"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
RANDOM_SEED = 42
CV_FOLDS = 5
MAX_SAMPLES = 50000
N_PLACEBO = 100

# %%
# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)

# Resolve label from setup.yaml if not overridden
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())
if not PRIMARY_LABEL:
    PRIMARY_LABEL = setup["labels"]["primary"]

# Load DML config and apply Papermill overrides
causal_configs = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "causal_dml")
dml_cfg = causal_configs[0]

for key, val in [
    ("n_folds", CV_FOLDS),
    ("n_placebo", N_PLACEBO),
    ("max_samples", MAX_SAMPLES),
    ("seed", RANDOM_SEED),
]:
    if dml_cfg.get(key, val) != val:
        dml_cfg[key] = val

CV_FOLDS = dml_cfg.get("n_folds", 5)
N_PLACEBO = dml_cfg.get("n_placebo", 100)
MAX_SAMPLES = dml_cfg.get("max_samples", 50000)
RANDOM_SEED = dml_cfg.get("seed", 42)

print(f"DML config: {dml_cfg['config_name']}")

# Read causal config from setup.yaml
causal_cfg = setup.get("causal", {})
TREATMENT_COL = causal_cfg.get("treatment", "")
CONFOUNDER_COLS = causal_cfg.get("confounders", [])
DML_METHOD = causal_cfg.get("method", "walk_forward_dml")

if not TREATMENT_COL:
    raise ValueError(
        f"No causal.treatment in {CASE_STUDY_ID}/setup.yaml. "
        "Add a 'causal:' section with treatment and confounders."
    )

np.random.seed(RANDOM_SEED)

print("Causal DML Configuration:")
print(f"  Case study: {CASE_STUDY_ID}")
print(f"  Treatment: {TREATMENT_COL}")
print(f"  Outcome: {PRIMARY_LABEL}")
print(f"  Confounders: {CONFOUNDER_COLS}")
print(f"  CV folds: {CV_FOLDS}")
print(f"  Placebo reps: {N_PLACEBO}")

# %% [markdown]
# ## 1. Load Artifacts
#
# Load pre-computed features, temporal features, and labels using the
# shared modeling infrastructure.

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)

dataset = mds.dataset
feature_names = mds.feature_names
label_col = mds.label_col
date_col = mds.date_col
entity_cols = mds.entity_cols

# Verify treatment and confounders are in features
available = set(dataset.columns)
assert TREATMENT_COL in available, (
    f"Treatment '{TREATMENT_COL}' not found in features: {sorted(available)}"
)

missing_conf = [c for c in CONFOUNDER_COLS if c not in available]
if missing_conf:
    print(f"WARNING: Missing confounders {missing_conf}, dropping them")
    CONFOUNDER_COLS = [c for c in CONFOUNDER_COLS if c in available]

assert len(CONFOUNDER_COLS) >= 1, "Need at least 1 confounder for DML"

print(f"\nDataset: {len(dataset):,} rows x {len(feature_names)} features")
print(f"Label: {label_col} | Date: {date_col} | Entities: {entity_cols}")

# %% [markdown]
# ## 2. Prepare Analysis Data
#
# Convert to pandas, sort by time, and subset if needed.

# %%
# Select analysis columns
analysis_cols = [date_col] + entity_cols + [TREATMENT_COL, label_col] + CONFOUNDER_COLS
analysis_cols = list(dict.fromkeys(analysis_cols))  # deduplicate

merged_clean = (
    dataset.select([c for c in analysis_cols if c in available])
    .drop_nulls()
    .sort(date_col)
    .to_pandas()
)

EMBARGO_PERIODS = embargo_from_buffer(mds.label_buffer)

BLOCK_SIZE = EMBARGO_PERIODS

# Temporal subset if too large
if len(merged_clean) > MAX_SAMPLES:
    print(f"Taking most recent {MAX_SAMPLES:,} from {len(merged_clean):,}")
    merged_clean = merged_clean.iloc[-MAX_SAMPLES:]

# %% [markdown]
# > **Note**: With `MAX_SAMPLES` capping, results reflect recent-period effects.
# > For large panels, this covers a narrow time window and should not be interpreted
# > as stable long-run causal relationships.

# %%
print(f"\nAnalysis data: {len(merged_clean):,} rows")
print(f"Date range: {merged_clean[date_col].min()} to {merged_clean[date_col].max()}")
print(f"Embargo: {EMBARGO_PERIODS} periods | Block size: {BLOCK_SIZE}")
if entity_cols:
    print(f"Entities: {merged_clean[entity_cols[0]].nunique()}")

# %% [markdown]
# ## 3. Run DML Analysis
#
# Full pipeline: naive OLS baseline, DML with walk-forward CV + embargo,
# and block permutation refutation test. HAC bandwidth is set from the label
# horizon as $\max(h-1,\ \lfloor n^{1/3} \rfloor)$: the ~35-day hold-to-expiry
# label overlaps heavily, so the bandwidth must be at least ~34 or the
# second-stage standard error is understated.

# %%
results = run_dml_analysis(
    merged_clean,
    treatment_col=TREATMENT_COL,
    outcome_col=label_col,
    confounder_cols=CONFOUNDER_COLS,
    n_folds=CV_FOLDS,
    embargo=EMBARGO_PERIODS,
    n_placebo=N_PLACEBO,
    block_size=BLOCK_SIZE,
    seed=RANDOM_SEED,
    horizon=EMBARGO_PERIODS,
)

print(format_dml_summary(results))

# %% [markdown]
# **SP500 Options VRP — Interpretation**: The DML effect of VRP on
# `ret_to_expiry` is −0.123 (HAC SE 0.026, t≈−4.7, p<0.001) — high VRP
# causally predicts lower hold-to-maturity straddle returns, consistent
# with VRP-as-compensation: when implied volatility exceeds realized,
# the long straddle position loses on average and the seller earns
# the premium. The confounding bias is moderate (about 50%): naive OLS
# gives −0.061, so RV/VRP-momentum/spread together absorb roughly half
# of the absolute effect.
#
# The block-permutation refutation passes (empirical p=0.030): placebo
# permutations of `vrp_21d` rarely reproduce the observed magnitude.
# SP500 Options is one of four panels in the cross-dataset trial where
# both gates clear (alongside ETFs, US Firms, and US Equities). The
# negative direction is consistent with the standard volatility-risk-
# premium story: insurance buyers (long straddles) pay implied vol
# above what gets realized; sellers earn that premium as compensation
# for bearing volatility risk.

# %% [markdown]
# ## 4. Statistical Assessment
#
# Confounding bias is defined as:
#
# $$\text{Bias \%} = \frac{\hat{\theta}_{\text{naive}} - \hat{\theta}_{\text{DML}}}{|\hat{\theta}_{\text{DML}}|} \times 100$$
#
# Positive values mean naive OLS overstates the absolute effect.

# %%
dml_result = results["dml_result"]
naive_effect = results["naive_effect"]
dml_effect = dml_result["theta"]
se_hac = dml_result["se_hac"]
bias_pct = results["confounding_bias_pct"]

# p-value computed in run_dml_analysis (no duplication)
p_value = results["p_value_hac"]

ref = results.get("refutation", {})
p_value_perm = ref.get("empirical_p", 1.0)
ref_class = ref.get("refutation_class", classify_refutation(p_value_perm))

print("Statistical significance:")
print(f"  p-value (HAC): {p_value:.4f}")
print(f"  Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
if ref:
    print(f"  Refutation: {ref_class} (p={p_value_perm:.4f})")

# %% [markdown]
# > **When should you be suspicious of large DML corrections?** A naive-to-DML
# > amplification exceeding 5x warrants scrutiny. Possible causes:
# > (1) nuisance models overfitting and stripping outcome-relevant variation,
# > (2) weak instrument-like behavior where the treatment residual has low variance,
# > (3) genuine massive confounding that naive OLS entirely misses.
# > The refutation test helps distinguish (3) from (1-2): if placebos also show
# > inflated effects, the DML correction may be unreliable.

# %% [markdown]
# ### Permutation Distribution
#
# The observed DML effect (red line) against the distribution of placebo
# effects under block permutation of the treatment.

# %%
placebo_arr = np.array(ref.get("placebo_effects", []))
if len(placebo_arr) > 0:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(placebo_arr, bins=30, alpha=0.7, label="Placebo effects")
    ax.axvline(dml_effect, color="red", linewidth=2, label=f"Observed ({dml_effect:.6f})")
    ax.set_xlabel("Treatment Effect")
    ax.set_ylabel("Count")
    ax.set_title("Block Permutation Refutation")
    ax.legend()
    fig.show()

# %% [markdown]
# ## 5. Save Results

# %%

# Per-observation treatment contributions from DML residuals
T_res = dml_result.get("T_res", np.full(len(merged_clean), np.nan))
Y_res = dml_result.get("Y_res", np.full(len(merged_clean), np.nan))

# Residuals are NaN for training-only obs; align to merged_clean length
# (residuals cover the full analysis data before subsampling)
n_analysis = len(merged_clean)
if len(T_res) > n_analysis:
    # Residuals from full data; take tail matching our subsample
    T_res = T_res[-n_analysis:]
    Y_res = Y_res[-n_analysis:]

symbol_col = entity_cols[0] if entity_cols else None

predictions = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(merged_clean[date_col]),
        "symbol": merged_clean[symbol_col].values if symbol_col else "ALL",
        "y_true": merged_clean[label_col].values,
        "treatment_value": merged_clean[TREATMENT_COL].values,
        "treatment_residual": T_res,
        "outcome_residual": Y_res,
        "treatment_contribution": T_res * dml_effect,
        "ate": dml_effect,
        "ate_se": se_hac,
    }
)

# Write standardized results JSON
summary = {
    "treatment": TREATMENT_COL,
    "outcome": label_col,
    "confounders": CONFOUNDER_COLS,
    "n_observations": dml_result["n_obs"],
    "naive_effect": float(naive_effect),
    "dml_effect": float(dml_effect),
    "dml_se_iid": float(dml_result["se_iid"]),
    "dml_se_hac": float(se_hac),
    "confounding_bias_pct": float(bias_pct),
    "p_value_hac": float(p_value),
    "hac_maxlags": int(results.get("hac_maxlags", 0)),
    "refutation_p_value": float(p_value_perm),
    "refutation_class": ref_class,
}

register_causal_run(
    case_study_id=CASE_STUDY_ID,
    label=PRIMARY_LABEL,
    results=results,
    predictions=predictions,
    treatment_col=TREATMENT_COL,
    confounder_cols=CONFOUNDER_COLS,
    n_folds=CV_FOLDS,
    embargo=EMBARGO_PERIODS,
    notebook="10_causal_dml",
)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Both gates clear**: The DML effect of `vrp_21d` on `ret_to_expiry`
#    is −0.123 (HAC SE 0.026, t≈−4.7, p<0.001) and block-permutation
#    refutation passes (empirical p=0.030). SP500 Options joins ETFs,
#    US Firms, and US Equities as one of four panels in the trial where
#    both parametric and placebo evidence support a causal channel.
#
# 2. **Moderate confounding bias**: The naive OLS estimate (−0.061)
#    understates the absolute DML effect (−0.123) by about 50%.
#    Realized volatility, VRP momentum, and spread percentile together
#    absorb roughly half of the cross-sectional VRP signal in the raw
#    data — high-VRP names cluster in particular volatility regimes
#    and liquidity profiles, and the naive estimate captures only the
#    residual after those confounders contaminate the treatment.
#
# 3. **Standard volatility-risk-premium story**: Negative DML effect
#    on `ret_to_expiry` means high VRP causally predicts lower
#    hold-to-maturity straddle returns. Insurance buyers (long
#    straddles) pay implied vol above what gets realized; sellers earn
#    that premium as compensation for bearing volatility risk. The
#    cross-sectional dispersion in VRP carries information beyond the
#    aggregate VRP-realized gap.
#
# 4. **Causal channel for the GBM predictive signal**: GBM achieves
#    IC +0.068 by exploiting VRP and related features. The DML
#    decomposition shows that a substantial portion of that
#    predictability has a direct causal interpretation, not just a
#    confounded correlation. The Ch16 backtest evaluates whether the
#    signal survives execution costs.
#
# **Next**: Ch16 ([`12_backtest`](12_backtest.ipynb)) translates the signal into a backtest.
#
# **Book**: Section 15.6 uses SP500 Options as one of four panels
# where both parametric and placebo evidence support a causal channel.

# %%
print("\n" + "=" * 60)
print(f"CHAPTER 15 RESULTS: {CASE_STUDY_ID} causal DML")
print("=" * 60)
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

print(f"\n[OK] Causal DML analysis complete for {CASE_STUDY_ID}")
