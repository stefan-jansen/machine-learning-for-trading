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
# # Causal DML - FX Momentum
#
# Double machine learning estimates the effect associated with a specified momentum treatment after
# flexible adjustment for the configured volatility and price-state variables. It is a causal
# estimand, not a predictive model configuration, so its result remains separate from prediction
# catalogs and backtest selection.
#
# The shared request verifies that the treatment and every confounder exist, excludes outcomes whose
# horizon reaches the holdout, keeps complete timestamp panels when applying a sample cap, cross-fits
# nuisance models with an embargo, and records the block-placebo design in the causal identity.
#
# **Learning objectives**
#
# - Inspect the treatment, confounders, timing, and refutation design before estimation.
# - Run the same causal request used by direct Python callers.
# - Keep causal evidence separate from predictive-family comparison and selection.
#
# **Book reference**: Chapter 15, Section 15.4
#
# **Prerequisites**: `02_labels`, `03_financial_features`, and `04_model_based_features`.

# %%
"""Estimate and register the configured FX causal DML specification."""

import polars as pl
import yaml

from case_studies.research import ExecutionTier, open_study, supersedes_for
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
RANDOM_SEED = 42
CV_FOLDS = 0
MAX_SAMPLES = 0
N_PLACEBO = 0
FORCE_RETRAIN = False
# The tier is a parameter, not something inferred from whether a reduction happens to be set.
# Inferring it meant a reduced run still opened the case study's own artifacts in place, which
# is the production path. WORKSPACE is the other half: a preview has nowhere else to write.
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
SUPERSEDES_CAUSAL: str = ""

# %% [markdown]
# ## Resolve the estimand and analysis population
#
# Nonzero fold, sample, symbol, or placebo limits declare a preview. They are recorded in identity
# and excluded from canonical evidence.

# %%
set_global_seeds(RANDOM_SEED)
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
labels = (
    [PRIMARY_LABEL]
    if PRIMARY_LABEL
    else [setup["labels"]["primary"], *setup["labels"].get("variants", [])]
)

if FORCE_RETRAIN:
    raise ValueError("an identical complete causal request is reused; change the request to refit")

REDUCTION_PARAMETERS = {
    "max_symbols": MAX_SYMBOLS or None,
    "n_folds": CV_FOLDS or None,
    "max_samples": MAX_SAMPLES or None,
    "n_placebo": N_PLACEBO or None,
}
reductions = {key: value for key, value in REDUCTION_PARAMETERS.items() if value is not None}
tier = ExecutionTier(EXECUTION_TIER)
if tier is ExecutionTier.PREVIEW and not reductions:
    raise ValueError("preview execution must declare at least one reduction")
if tier is ExecutionTier.CANONICAL and reductions:
    raise ValueError(f"canonical execution cannot carry reductions: {sorted(reductions)}")
study = open_study(CASE_STUDY_ID, execution_tier=tier, workspace=WORKSPACE or None)
requests = {
    label: study.causal(
        method="dml",
        label=label,
        config_name="dml",
        execution_tier=tier,
        preview_reductions=reductions,
        overrides={},
        supersedes=supersedes_for(SUPERSEDES_CAUSAL, label, labels=list(labels)),
    )
    for label in labels
}
resolutions = {label: request.resolve() for label, request in requests.items()}
computations = {label: resolved.spec["computation"] for label, resolved in resolutions.items()}
computation = computations[labels[0]]

estimand = computation["estimand"]
print(f"Labels: {', '.join(labels)}")
print(f"Treatment: {estimand['treatment']}")
print(f"Confounders: {', '.join(estimand['confounders'])}")
print(f"Execution tier: {tier.value}")
for horizon, values in computations.items():
    print(
        f"  {horizon}: outcome horizon {values['estimand']['outcome_horizon']}, "
        f"last admissible endpoint {values['estimand']['holdout_endpoint_cutoff']}"
    )

# %% [markdown]
# ## Inspect temporal controls
#
# Cross-fitting groups observations by complete decision timestamps, and the embargo is at least the
# outcome horizon.
#
# The placebo permutes contiguous blocks within each currency pair rather than shuffling rows, and
# the block spans the longer of the two scales that make neighbouring rows dependent. Overlapping
# labels tie rows together across the outcome horizon. The treatment ties them together across its
# own construction window, which for `mom_skip_recent` is the 252 sessions its return is measured
# over - consecutive values share 251 of them - and that is what `causal.treatment_window` in
# `setup.yaml` declares. A block shorter than the longer scale breaks the dependence the
# permutation exists to preserve, which narrows the placebo distribution and makes the p-value read
# stronger than the evidence is. Both scales are printed below.

# %%
pl.DataFrame(
    {
        "label": list(computations),
        "cross_fitting_folds": [c["cv"]["n_folds"] for c in computations.values()],
        "embargo_periods": [c["cv"]["embargo_periods"] for c in computations.values()],
        "fold_unit": [c["cv"]["fold_unit"] for c in computations.values()],
        "treatment_window": [setup["causal"]["treatment_window"] for _ in computations],
        "placebo_method": [c["refutation"]["method"] for c in computations.values()],
        "placebo_block_size": [c["refutation"]["block_size"] for c in computations.values()],
        "analysis_rows": [c["analysis_population"]["n_rows"] for c in computations.values()],
        "analysis_timestamps": [
            c["analysis_population"]["n_timestamps"] for c in computations.values()
        ],
        "analysis_key_digest": [
            c["analysis_population"]["key_digest"] for c in computations.values()
        ],
    }
)

# %% [markdown]
# ## Estimate or reload the causal result
#
# Registration happens only after the fit returns a finite effect and HAC standard error. A missing
# treatment, confounder, or valid analysis population fails before any causal row is written.

# %% tags=["results"]
results = {}
for label, resolved in resolutions.items():
    result = resolved.run()
    if not result.complete:
        raise RuntimeError(f"the causal result for {label} is incomplete")
    # Compare the identity-bearing computation, not the whole specification. `provenance` records
    # the git commit of the run that registered the result, so a full-spec comparison fails on any
    # re-run made after any commit - it would assert that nothing had been committed since, which is
    # not a property of the causal estimate.
    if result.spec["computation"] != resolved.spec["computation"]:
        raise RuntimeError(f"the registered causal computation for {label} differs")
    reloaded = requests[label].resolve().run()
    if reloaded.hash != result.hash:
        raise RuntimeError(f"reloading the causal request for {label} changed its identity")
    results[label] = result

if len({result.hash for result in results.values()}) != len(labels):
    raise RuntimeError("two configured labels resolved to one causal identity")

for label, result in results.items():
    print(f"Registered causal identity, {label}: {result.hash}")
print("Effect estimates and their interpretation are reported in 12_model_analysis.")

# %% [markdown]
# ## Key takeaways
#
# - Treatment, confounders, temporal geometry, and refutation settings are part of the estimand.
# - Invalid specifications fail before registry mutation.
# - Causal results remain distinct from predictive checkpoints and backtest candidates.
