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
# 2016 holdout out of the analysis entirely. What the estimate and the two uncertainty checks come
# to is read off the summary the notebook prints.
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
"""Panel-aware causal DML that leaves the final holdout out of the analysis."""

import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from case_studies.utils.causal import (
    classify_refutation,
    embargo_from_buffer,
    format_dml_summary,
    observation_step,
    register_causal_run,
    run_dml_analysis,
)
from utils.modeling import load_configs, load_modeling_dataset
from utils.paths import get_case_study_dir
from utils.style import COLORS, add_message_title

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
MAX_SYMBOLS = 0
# Each of these is zero or empty for "take the declared value". A run that passes one overrides
# the declaration; a run that passes none reproduces the published analysis.
LABEL = ""
SEED = 0
N_FOLDS = 0
MAX_SAMPLES = 0
N_PLACEBO = 0

# %% [markdown]
# ## Resolve the declared design
#
# The setup file supplies the label, holdout boundary, treatment, confounders, and annual frequency.
# The training preset supplies the fixed DML budget. No holdout result or strategy metric enters
# these choices.
#
# The parameters above are the request; the values the analysis runs on are resolved below and
# carry different names, so neither can quietly overwrite the other. Precedence is: an injected
# parameter wins, otherwise the case study's declaration. The declaration is read with `[...]`
# rather than `.get(key, literal)`, because a literal default would substitute a number the case
# study never declared, silently and only on the configurations that omit the key.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
PRIMARY_LABEL = LABEL or setup["labels"]["primary"]

dml_cfg = load_configs(CASE_STUDY_ID, PRIMARY_LABEL, "causal_dml")[0]
CV_FOLDS = N_FOLDS or int(dml_cfg["n_folds"])
PLACEBO_REPS = N_PLACEBO or int(dml_cfg["n_placebo"])
ROW_CAP = MAX_SAMPLES or int(dml_cfg["max_samples"])
RANDOM_SEED = SEED or int(dml_cfg["seed"])

causal_cfg = setup["causal"]
treatment_col = causal_cfg["treatment"]
confounder_cols = causal_cfg["confounders"]
holdout_start = setup["evaluation"]["holdout_start"]
periods_per_year = int(setup["evaluation"]["periods_per_year"])
label_horizon = embargo_from_buffer(
    setup["labels"]["buffer"],
    periods_per_year=periods_per_year,
)
embargo_periods = label_horizon
development_cutoff = (
    pl.Series([holdout_start]).str.to_date().dt.offset_by(f"-{embargo_periods}mo")[0]
)

print(f"Configuration: {dml_cfg['config_name']} | folds={CV_FOLDS} | placebos={PLACEBO_REPS}")
print(f"Row cap: {ROW_CAP:,} | seed: {RANDOM_SEED}")
print(f"Treatment: {treatment_col} | confounders: {', '.join(confounder_cols)}")
print(f"Holdout starts: {holdout_start}")

# %% [markdown]
# ## Build the development sample
#
# The label buffer closes development one month before the holdout so no forward return reaches
# into 2016. The declared row cap then selects the most recent complete monthly cross-sections that
# fit below the limit. Conversion to pandas occurs only at the causal utility boundary.
#
# The cap is denominated in rows and spent on whole decision months, so the history it buys is the
# cap divided by the width of the cross-section. That matters because the inference below runs
# along the time axis rather than the row axis: the panel-robust standard error clusters by
# decision month, and the placebo permutes each firm's months in blocks. Adding firms widens the
# cross-section without adding a single month to either. A panel this wide would get a short window
# out of the cap the narrower case studies share, so this one declares a preset of its own.

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
selected_dates = date_counts.filter(pl.col("cumulative_rows") <= ROW_CAP)[date_col]
if selected_dates.is_empty():
    raise ValueError(f"ROW_CAP={ROW_CAP:,} cannot fit one complete decision month")
analysis = eligible.filter(pl.col(date_col).is_in(selected_dates)).sort(date_col, entity_col)
if analysis.is_empty() or analysis[date_col].max() >= development_cutoff:
    raise ValueError("Development sample is empty or crosses the holdout boundary")

# The folds are built from complete decision months, not rows, and the row cap decides how many
# months survive. Below the minimum every fold's test window comes back empty, the second stage
# fits nothing, and the run reports a NaN effect over zero observations rather than failing - so
# the count of months is checked here, before an hour of placebo fits, rather than read off the
# result afterwards.
decision_months = analysis[date_col].n_unique()
min_decision_months = CV_FOLDS + embargo_periods + 1
if decision_months < min_decision_months:
    raise ValueError(
        f"{decision_months} decision months cannot fill {CV_FOLDS} expanding folds with a "
        f"{embargo_periods}-month embargo; {min_decision_months} is the minimum. The row cap of "
        f"{ROW_CAP:,} selects whole months, so raising it is what buys more of them."
    )

block_size = label_horizon
cadence = observation_step(analysis, date_col)
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
        "block_months": [block_size],
        "input_digest": [input_digest[:16]],
    }
)
sample_audit

# %% [markdown]
# ### The embargo and the permutation block answer different questions
#
# The **embargo** separates a fold's training months from its test months, so a label still running
# at the end of training cannot also be scored in test. The **permutation block** is the scale of
# serial dependence the placebo has to leave intact: shuffling in blocks shorter than the label
# horizon pulls overlapping labels apart, and the placebo degrades towards an independent draw,
# which is the permutation that is too easy to pass and therefore establishes nothing.
#
# Both come to one month here because the same `1M` label buffer sets both, and they are assigned
# separately so that stays a fact about this case study rather than an assumption buried in a
# shared name. At a one-month horizon consecutive forward returns do not overlap, so there is no
# label-induced dependence for a longer block to preserve. The treatment is a 12-to-2-month
# momentum measure and does carry month-to-month dependence of its own, which a one-month block
# breaks; the placebo therefore tests the timing of the treatment against the outcome, and not
# whether a slowly moving characteristic could have produced the estimate.
#
# The observed cadence is passed as well, which is what lets a hole in a firm's history end a
# block rather than be permuted across. The boundary is a gap wider than `GAP_TOLERANCE_CADENCES`
# cadences, four by declaration, and the cadence here is one month - so a firm absent for a few
# months is still permuted as one continuous history and only a longer gap splits it. That
# threshold is the shared default; `run_dml_analysis` takes no argument to tighten it, so it is
# reported here rather than chosen here.

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
    n_placebo=PLACEBO_REPS,
    block_size=block_size,
    seed=RANDOM_SEED,
    horizon=label_horizon,
    time_col=date_col,
    entity_col=entity_col,
    expected_step=cadence,
)

dml_result = results["dml_result"]
if dml_result["covariance_type"] != "driscoll_kraay":
    raise ValueError(f"Expected panel-robust covariance, found {dml_result['covariance_type']}")
if results["naive_n_obs"] != dml_result["n_obs"]:
    raise ValueError("Naive and adjusted estimates use different comparison samples")
# An empty second stage satisfies both checks above - the covariance type is still the one that
# was asked for, and zero equals zero - so what the estimate has to be is asserted directly.
if dml_result["n_obs"] == 0 or not np.isfinite([dml_result["theta"], dml_result["se_hac"]]).all():
    raise ValueError("DML produced no out-of-fold observations or no finite effect")
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
    "Permuting a firm's own months moves the estimated effect",
    subtitle=(
        f"{PLACEBO_REPS} within-firm block permutations at a {block_size}-month block; "
        "monthly effect in decimal return"
    ),
)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.show()

# %% [markdown]
# ## Register the corrected causal diagnostic
#
# Causal DML has its own registry table because it estimates a treatment effect rather than a
# cross-sectional prediction score. Everything that would change the estimate is named in the
# identity: the fold and embargo geometry, the placebo design and its seed, the label horizon, and
# the row cap and cutoff that decide which months are in the sample. Two runs that differ in any of
# them are two identities rather than one row overwriting another.

# %%
causal_hash = register_causal_run(
    case_study_id=CASE_STUDY_ID,
    label=PRIMARY_LABEL,
    results=results,
    treatment_col=treatment_col,
    confounder_cols=confounder_cols,
    n_folds=CV_FOLDS,
    embargo=embargo_periods,
    time_col=date_col,
    block_size=block_size,
    n_placebo=PLACEBO_REPS,
    seed=RANDOM_SEED,
    horizon=label_horizon,
    max_samples=ROW_CAP,
    development_end=str(development_cutoff),
    notebook="09_causal_dml",
)
print(f"Registered causal identity: {causal_hash}")

# %% [markdown]
# ## What the run produced
#
# One cell, so every number this notebook publishes is in one place and moves together on a re-run.
# `placebo_p_floor` is one over the number of successful placebo draws plus one, the smallest
# p-value a permutation test of this size can report. `placebo_z` is how many placebo standard
# deviations separate the observed estimate from the placebo mean, and it is printed beside the
# p-value because the p-value alone cannot show how far apart the two distributions are.

# %% tags=["results"]
refutation_class = refutation.get(
    "refutation_class",
    classify_refutation(refutation["empirical_p"]),
)
pl.DataFrame(
    {
        "decision_months": [dml_result["n_periods"]],
        "observations": [dml_result["n_obs"]],
        "dml_bps_per_month": [dml_result["theta"] * 10_000],
        "naive_bps_per_month": [results["naive_effect"] * 10_000],
        "se_hac": [dml_result["se_hac"]],
        "p_hac": [results["p_value_hac"]],
        "placebo_p": [refutation["empirical_p"]],
        "placebo_p_floor": [1.0 / (refutation["n_successful"] + 1)],
        "placebo_z": [refutation["z_score"]],
        "refutation_class": [refutation_class],
    }
)

# %% [markdown]
# ## Takeaways
#
# - This is a development-stage robustness diagnostic for the momentum signal. Prediction and
#   strategy selection are downstream tasks and read different evidence.
# - Read the coefficient against its Driscoll-Kraay standard error rather than against zero: that
#   standard error allows for dependence along time and across firms at the same decision month,
#   which an IID standard error on a panel this wide does not.
# - Naive OLS and the adjusted estimate are computed on the same out-of-fold rows, so the gap
#   between them measures sensitivity to the observed confounder set rather than two estimators
#   reading different numbers of observations.
# - The permutation p-value cannot fall below its floor however extreme the estimate is, so a value
#   at the floor is a bound and not a measurement. A block permutation disturbs the timing of the
#   treatment; it does not disturb confounding.
# - A permutation p at the top of its range is compatible with two different situations, and this
#   design cannot separate them. The estimate may be unremarkable against the null the placebo
#   draws, or the null the placebo draws may not be the one the question is about - which is what a
#   block too short for the treatment's own dependence would produce. `placebo_z` says how far
#   apart the observed estimate and the placebo mean are; it does not say which of the two
#   situations put them there. Separating them needs a block sized to the dependence being
#   preserved, and at a one-month block this notebook does not have one, so read the refutation
#   here as uninformative rather than as evidence in either direction.
# - Read the refutation against the coefficient's own uncertainty rather than on its own. Where
#   the estimate is not separable from zero by its Driscoll-Kraay standard error, a placebo test
#   of that estimate could not have said much whichever way it came out. The sentence the evidence
#   supports is that this design cannot tell, which is not the same as a statement about whether
#   an effect is there.
# - Conditioning on Beta, IdioVol, LME and Variance does not establish that conditional
#   ignorability, overlap or no-interference hold. Every step above ends before the 2016 holdout.
