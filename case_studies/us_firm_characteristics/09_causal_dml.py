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
# # Causal DML for US Firm Characteristics
#
# This notebook asks whether 12-to-2-month momentum predicts next-month returns after conditioning
# on four observed risk characteristics. It uses expanding-window double machine learning (DML),
# keeps every firm at a decision month on the same side of each fold boundary, and leaves the final
# 2016 holdout sealed. The corrected estimate is positive but clears neither the panel-robust
# significance gate nor the entity-aware placebo gate.
#
# **Learning objectives**
#
# - Residualize momentum and returns with nuisance models fit on earlier decision months.
# - Compare the adjusted effect with naive OLS on the exact same out-of-fold sample.
# - Use panel-robust inference and an entity-aware block-permutation refutation.
# - Distinguish a conditional causal estimate from proof that all confounding is observed.
#
# **Book reference:** Chapter 15, cross-dataset causal evidence.
#
# **Prerequisites:** `03_financial_features` and `04_evaluation`.

# %%
"""Panel-aware causal DML with a sealed final holdout."""

import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from IPython.display import Markdown, display

from case_studies.utils.causal import (
    classify_refutation,
    embargo_from_buffer,
    format_dml_summary,
    register_causal_run,
    run_dml_analysis,
)
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir
from utils.style import COLORS, add_message_title

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
RANDOM_SEED = 42
CV_FOLDS = 5
MAX_SAMPLES = 50_000
N_PLACEBO = 100

# %% [markdown]
# ## Resolve the declared design
#
# The setup file supplies the label, holdout boundary, treatment, confounders, and annual frequency.
# The training preset supplies the fixed DML budget. No holdout result or strategy metric enters
# these choices.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
if not PRIMARY_LABEL:
    PRIMARY_LABEL = setup["labels"]["primary"]

dml_cfg = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "causal_dml")[0]
for key, value in (
    ("n_folds", CV_FOLDS),
    ("n_placebo", N_PLACEBO),
    ("max_samples", MAX_SAMPLES),
    ("seed", RANDOM_SEED),
):
    dml_cfg[key] = value

causal_cfg = setup["causal"]
treatment_col = causal_cfg["treatment"]
confounder_cols = causal_cfg["confounders"]
holdout_start = setup["evaluation"]["holdout_start"]
periods_per_year = int(setup["evaluation"]["periods_per_year"])
embargo_periods = embargo_from_buffer(
    setup["labels"]["buffer"],
    periods_per_year=periods_per_year,
)
development_cutoff = (
    pl.Series([holdout_start]).str.to_date().dt.offset_by(f"-{embargo_periods}mo")[0]
)

print(f"Configuration: {dml_cfg['config_name']} | folds={CV_FOLDS} | placebos={N_PLACEBO}")
print(f"Treatment: {treatment_col} | confounders: {', '.join(confounder_cols)}")
print(f"Holdout starts: {holdout_start}")

# %% [markdown]
# ## Build the development sample
#
# The label buffer closes development one month before the holdout so no forward return reaches
# into 2016. The unchanged 50,000-row cap then selects the most recent complete monthly
# cross-sections that fit below the limit. Conversion to pandas occurs only at the causal utility
# boundary.

# %%
mds = load_modeling_dataset(CASE_STUDY_ID, PRIMARY_LABEL, max_symbols=MAX_SYMBOLS)
date_col = mds.date_col
entity_col = mds.entity_cols[0]
label_col = mds.label_col

analysis_cols = list(
    dict.fromkeys([date_col, entity_col, treatment_col, label_col, *confounder_cols])
)
eligible = (
    mds.dataset.select(analysis_cols)
    .drop_nulls()
    .filter(pl.col(date_col) < development_cutoff)
    .sort(date_col, entity_col)
)
date_counts = (
    eligible.group_by(date_col)
    .len()
    .sort(date_col, descending=True)
    .with_columns(pl.col("len").cum_sum().alias("cumulative_rows"))
)
selected_dates = date_counts.filter(pl.col("cumulative_rows") <= MAX_SAMPLES)[date_col]
if selected_dates.is_empty():
    raise ValueError(f"MAX_SAMPLES={MAX_SAMPLES:,} cannot fit one complete decision month")
analysis = eligible.filter(pl.col(date_col).is_in(selected_dates)).sort(date_col, entity_col)
if analysis.is_empty() or analysis[date_col].max() >= development_cutoff:
    raise ValueError("Development sample is empty or crosses the sealed holdout boundary")

block_size = embargo_periods
schema_bytes = "|".join(f"{name}:{dtype}" for name, dtype in analysis.schema.items()).encode()
row_hash_bytes = analysis.hash_rows(seed=0).to_numpy().tobytes()
input_digest = hashlib.sha256(schema_bytes + row_hash_bytes).hexdigest()

sample_audit = pl.DataFrame(
    {
        "rows": [analysis.height],
        "decision_months": [analysis[date_col].n_unique()],
        "firms": [analysis[entity_col].n_unique()],
        "start": [analysis[date_col].min()],
        "end": [analysis[date_col].max()],
        "development_cutoff": [development_cutoff],
        "embargo_months": [embargo_periods],
        "input_digest": [input_digest[:16]],
    }
)
sample_audit

# %% [markdown]
# ## Estimate on complete decision-month folds
#
# Each fold trains on earlier months and scores later months after a one-month embargo. Driscoll-Kraay
# inference aggregates the second-stage score by month, allowing general dependence across firms at
# the same decision time. The naive comparison uses exactly the rows with out-of-fold DML residuals.

# %%
analysis_pd = analysis.to_pandas()
results = run_dml_analysis(
    analysis_pd,
    treatment_col=treatment_col,
    outcome_col=label_col,
    confounder_cols=confounder_cols,
    n_folds=CV_FOLDS,
    embargo=embargo_periods,
    n_placebo=N_PLACEBO,
    block_size=block_size,
    seed=RANDOM_SEED,
    horizon=embargo_periods,
    time_col=date_col,
    entity_col=entity_col,
)

dml_result = results["dml_result"]
if dml_result["covariance_type"] != "driscoll_kraay":
    raise ValueError(f"Expected panel-robust covariance, found {dml_result['covariance_type']}")
if results["naive_n_obs"] != dml_result["n_obs"]:
    raise ValueError("Naive and adjusted estimates use different comparison samples")
print(format_dml_summary(results))

# %% [markdown]
# ## Adjusted and naive estimates use the same sample
#
# A difference between the two estimates measures sensitivity to the observed confounder set. It
# does not establish that conditional ignorability, overlap, or no-interference assumptions hold.

# %%
refutation = results["refutation"]
effect_table = pl.DataFrame(
    {
        "estimate": ["Naive OLS", "DML adjusted"],
        "monthly_effect": [results["naive_effect"], dml_result["theta"]],
        "basis_points": [results["naive_effect"] * 10_000, dml_result["theta"] * 10_000],
        "observations": [results["naive_n_obs"], dml_result["n_obs"]],
        "decision_months": [dml_result["n_periods"], dml_result["n_periods"]],
    }
)
effect_table

# %% [markdown]
# ## The placebo distribution tests temporal specificity
#
# Treatment histories are permuted in time within each firm, preserving the entity histories used by
# the original design. The observed line should be interpreted alongside the empirical p-value, not
# as proof that unobserved confounding is absent.

# %%
placebo_effects = np.asarray(refutation.get("placebo_effects", []))
if placebo_effects.size < 10:
    raise ValueError("Insufficient successful placebo estimates")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(placebo_effects, bins=20, color=COLORS["silver_muted"], edgecolor=COLORS["neutral"])
ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=1)
ax.axvline(
    dml_result["theta"],
    color=COLORS["negative"],
    linewidth=2,
    label=f"Observed DML ({dml_result['theta']:+.4f})",
)
ax.set(xlabel="Monthly treatment effect (decimal return)", ylabel="Placebo runs")
ax.legend(frameon=False)
add_message_title(
    ax,
    "Every entity-aware placebo exceeds the observed effect",
    subtitle=f"{N_PLACEBO} block permutations within firms; empirical p={refutation['empirical_p']:.3f}",
)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.show()

# %% [markdown]
# ## Register the corrected causal diagnostic
#
# Causal DML has its own registry table because it estimates a treatment effect rather than a
# cross-sectional prediction score. The corrected identity records the one-month group embargo.

# %%
causal_hash = register_causal_run(
    case_study_id=CASE_STUDY_ID,
    label=PRIMARY_LABEL,
    results=results,
    treatment_col=treatment_col,
    confounder_cols=confounder_cols,
    n_folds=CV_FOLDS,
    embargo=embargo_periods,
    notebook="09_causal_dml",
)
print(f"Registered corrected causal identity: {causal_hash}")

# %% [markdown]
# ## Takeaways
#
# The adjusted estimate is a robustness diagnostic for the momentum signal, while prediction and
# strategy selection remain downstream tasks.

# %%
effect_bps = dml_result["theta"] * 10_000
naive_bps = results["naive_effect"] * 10_000
refutation_class = refutation.get(
    "refutation_class",
    classify_refutation(refutation["empirical_p"]),
)
display(
    Markdown(
        f"""
- Across **{dml_result["n_periods"]}** out-of-fold decision months, the adjusted momentum estimate is
  **{effect_bps:+.1f} basis points per month**, compared with **{naive_bps:+.1f} basis points** for
  naive OLS on the same **{dml_result["n_obs"]:,}** observations.
- Driscoll-Kraay inference gives standard error **{dml_result["se_hac"]:.4f}** and two-sided
  **p={results["p_value_hac"]:.3f}**. The entity-aware placebo refutation **{refutation_class.lower()}**
  at empirical **p={refutation["empirical_p"]:.3f}**.
- The positive point estimate clears neither inferential gate. These data therefore do not support
  a nonzero causal momentum effect after conditioning on Beta, IdioVol, LME, and Variance; any causal
  interpretation also depends on the observed-confounding, overlap, and no-interference assumptions.
- All folds operate on complete months with a **{embargo_periods}-month embargo**. The 2016 holdout
  remains sealed for the final selected strategy.
"""
    )
)
