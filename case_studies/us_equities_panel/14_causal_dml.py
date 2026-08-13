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
# # Causal DML — US Equities 12-1 Momentum
#
# Does the classic Jegadeesh--Titman 12-1 momentum factor *cause* future daily
# returns, or is the observed premium confounded by volatility and liquidity?
# On the broadest panel in the book (3,199 stocks, 1962--2018), DML separates
# the causal effect of `past_ret_12m_skip` from correlated risk factors. With
# 53.8% confounding and a borderline p-value (0.122), the answer is nuanced:
# momentum's daily effect is not statistically significant after deconfounding.
#
# **Treatment rationale**: 12-month skip-month return is the canonical momentum
# signal. Volatility (21d) and volume ratio confound the relationship --- illiquid
# stocks with high past returns appear to have stronger momentum, but part of this
# reflects liquidity risk rather than a genuine behavioral effect.
#
# **Learning Objectives**:
# - Quantify confounding bias in the momentum--return relationship (53.8%)
# - Understand why broad panels weaken causal identification (regime diversity)
# - Contrast daily reversal (negative sign) with monthly continuation
#
# **Book Reference**: Chapter 15, Section 15.6 (Cross-Dataset Causal Evidence)
#
# **Prerequisites**: `03_financial_features.py`, `04_temporal.py`

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
CASE_STUDY_ID = "us_equities_panel"
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
# and block permutation refutation test. HAC bandwidth is set automatically
# using the cube-root rule: $\lfloor n^{1/3} \rfloor$.

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
)

print(format_dml_summary(results))

# %% [markdown]
# **US Equities Momentum — Interpretation**: The classic 12-1 momentum
# effect on next-day returns shows substantial confounding bias.
# Naive OLS estimates a daily momentum coefficient of −0.0003, but
# DML reveals −0.0010 after orthogonalization — volatility (21d) and
# volume ratio absorb roughly two-thirds of the absolute effect.
# Amihud illiquidity was specified as a confounder but dropped due to
# data availability, leaving a 2-confounder model.
#
# Both evidence gates clear: HAC t≈−2.2 (p=0.031) and the
# block-permutation refutation passes decisively (empirical p<0.01).
# US Equities is one of four panels in the cross-dataset trial where
# both gates support a causal channel (alongside ETFs, US Firms, and
# SP500 Options). The negative direction is consistent with short-term
# reversal at the daily horizon — high past 12-1 momentum causally
# predicts lower next-day returns even after orthogonalizing the main
# microstructure confounders.

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
    notebook="14_causal_dml",
)

# %% [markdown]
# ## Key Takeaways
#
# 1. **Both gates clear**: The DML effect on next-day returns is −0.0010
#    (HAC SE 0.0005, t≈−2.2, p=0.031) and the block-permutation
#    refutation passes decisively (empirical p<0.01). US Equities is
#    one of four panels in the trial where both parametric and placebo
#    evidence support a causal channel (alongside ETFs, US Firms, and
#    SP500 Options).
#
# 2. **Two-thirds confounding**: Volatility (21d) and volume ratio
#    together absorb roughly two-thirds of the signal in raw OLS — the
#    naive estimate of −0.0003 understates the absolute DML effect of
#    −0.0010 by about 67% (confounding bias +67%). Amihud illiquidity,
#    originally specified as a third confounder, was dropped for data
#    availability.
#
# 3. **Daily reversal versus monthly continuation**: The negative sign
#    at the daily horizon contrasts with the positive monthly momentum
#    in `us_firm_characteristics`. Both panels clear both gates in
#    HistGBR DML. The directional pattern is consistent with short-term
#    reversal turning into intermediate-horizon continuation as the
#    holding window lengthens.
#
# **Next**: See `10_case_study_insights` for comparison across all 9 case studies.
#
# **Book**: Section 15.6 discusses US Equities as one of four panels
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
