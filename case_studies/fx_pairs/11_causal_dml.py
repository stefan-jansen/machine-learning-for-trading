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

from case_studies.research import ExecutionTier, causal_supersedes, open_study
from utils.paths import get_case_study_dir
from utils.reproducibility import set_global_seeds

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
CV_FOLDS = 0
MAX_SAMPLES = 0
N_PLACEBO = 0
FORCE_RETRAIN = False
# The tier is a parameter, not something inferred from whether a reduction happens to be set.
# Inferring it meant a reduced run still opened the case study's own artifacts in place, which
# is the production path. WORKSPACE is the other half: a preview has nowhere else to write.
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
# The three identities this run retires, one per label. Sizing the placebo block by the
# treatment window rather than the label buffer moves `refutation.block_size`, which is part
# of the causal identity, so every label resolves to a new hash and the registry refuses a
# second current identity per label without being told which one it replaces. These are the
# blocks-of-1/5/21 fits from 2026-08-21, whose p-values this run supersedes rather than
# corrects: they measured a placebo that had already destroyed the dependence it was meant
# to preserve.
#
# A reader's clone holds no causal rows at all, so `causal_supersedes` withholds these
# against a registry that does not have them and the reader registers a first identity.
# That resolution has to happen against the registry rather than by leaving the default
# empty: `run-production-notebook.sh` executes with no parameter overrides, so a value
# supplied only at run time could never be stamped.
SUPERSEDES_CAUSAL: str = (
    '{"fwd_ret_1d": "25f8bdd775de", "fwd_ret_5d": "c797f741134a", "fwd_ret_21d": "3547657669ca"}'
)

# %% [markdown]
# ## Resolve the estimand and analysis population
#
# Nonzero fold, sample, symbol, or placebo limits declare a preview. They are recorded in identity
# and excluded from canonical evidence.

# %%
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
        supersedes=causal_supersedes(
            study, SUPERSEDES_CAUSAL, label, labels=list(labels), execution_tier=tier.value
        ),
    )
    for label in labels
}
resolutions = {label: request.resolve() for label, request in requests.items()}

# Seeded from the RESOLVED specification rather than from a notebook constant. The seed the
# estimate depends on is `dml.yaml`'s, and it is inside the identity hash - so a constant here
# that only reached `set_global_seeds` was a dial that turned without moving the number: change
# it and the identity is unchanged, the cached result fit under the configured seed is served
# back, and nothing says so. Seeding from the resolved value makes the seed this notebook
# announces the seed its results were actually fit under.
RANDOM_SEED = int(resolutions[labels[0]].spec["seed"])
if any(int(resolved.spec["seed"]) != RANDOM_SEED for resolved in resolutions.values()):
    raise RuntimeError(
        "the configured labels resolved to different seeds, so one global seed cannot describe "
        f"them: {sorted({int(r.spec['seed']) for r in resolutions.values()})}"
    )
set_global_seeds(RANDOM_SEED)

computations = {label: resolved.spec["computation"] for label, resolved in resolutions.items()}
computation = computations[labels[0]]

estimand = computation["estimand"]
print(f"Labels: {', '.join(labels)}")
print(f"Treatment: {estimand['treatment']}")
print(f"Confounders: {', '.join(estimand['confounders'])}")
print(f"Execution tier: {tier.value}")
print(f"Seed (from the resolved identity): {RANDOM_SEED}")
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
# The placebo permutes contiguous blocks within each currency pair rather than shuffling rows,
# because two separate things make neighbouring rows dependent and a block has to span both.
# Overlapping labels tie rows together across the outcome horizon. The treatment ties them together
# across its own construction: `mom_skip_recent` is the return from `t-252` to `t-21`, so one value
# covers 231 sessions and consecutive values overlap in 230 of them. `causal.treatment_window` in
# `setup.yaml` declares 252, the oldest price any single value reads, which is the larger of the
# three quantities in that expression and therefore spans the 231-session dependence whichever way
# the arithmetic is read. The block takes the longer of that window and the label buffer.
#
# The table below prints the block the run used beside both scales it has to span, so the choice
# can be checked rather than taken on trust. Until public #623 the block was sized from the label
# buffer alone - 1, 5 and 21 sessions against a treatment whose dependence runs 231 - which is
# close enough to an independent shuffle that it destroyed the dependence the permutation exists
# to preserve, narrowing the placebo distribution and making `refutation_p` read stronger than the
# evidence was.

# %%
pl.DataFrame(
    {
        "label": list(computations),
        "cross_fitting_folds": [c["cv"]["n_folds"] for c in computations.values()],
        "embargo_periods": [c["cv"]["embargo_periods"] for c in computations.values()],
        "fold_unit": [c["cv"]["fold_unit"] for c in computations.values()],
        # The two scales the block has to span, and which of them decided it. Read from the
        # resolved specification rather than from setup.yaml, so the table shows what the run
        # used rather than what the file asks for. `label_horizon_steps` is the outcome
        # horizon and is what the HAC bandwidth is sized by; `label_buffer_steps` is the CV
        # gap and is declared deliberately longer. They are different quantities and the
        # block is sized by neither on its own.
        "label_buffer_steps": [
            c["refutation"]["label_buffer_steps"] for c in computations.values()
        ],
        "label_horizon_steps": [
            c["refutation"]["label_horizon_steps"] for c in computations.values()
        ],
        "treatment_window_steps": [
            c["refutation"]["treatment_window_steps"] for c in computations.values()
        ],
        "placebo_method": [c["refutation"]["method"] for c in computations.values()],
        "placebo_block_size": [c["refutation"]["block_size"] for c in computations.values()],
        "block_size_basis": [c["refutation"]["block_size_basis"] for c in computations.values()],
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
