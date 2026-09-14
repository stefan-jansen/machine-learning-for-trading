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
# # US equities panel: refitting the selected configuration for the holdout
#
# Every number this case study has produced so far was measured on the validation folds, and
# every choice was made by looking at them: which model family, which configuration, how many
# names to hold on each side of the book, how to size them, which risk control to overlay, what
# to charge for a trade. A result selected that way cannot also be the evidence that the
# selection was sound, because the ranking and the evidence would be the same measurement.
#
# The holdout is the window nothing has been selected on: `evaluation.holdout_start` through
# `evaluation.holdout_end` in `config/setup.yaml`, which no notebook from
# [`16_backtest`](16_backtest.ipynb) through [`19_costs`](19_costs.ipynb) reads. This notebook
# fits the selected configuration again on the history available before that window opens and
# writes its predictions over it. It publishes predictions and nothing else:
# [`21_holdout_backtest`](21_holdout_backtest.ipynb) turns them into a return series, and
# [`22_strategy_analysis`](22_strategy_analysis.ipynb) reads both back with intervals.
#
# **A holdout prediction is not the validation model scored on a later window.** Section 2 fits
# again, over a training interval that ends before the window opens, and the new training
# identity is what makes the refit visible rather than asserted: the identity covers the CV
# interval, so a run that came back with the validation training hash would mean no refit
# happened. Section 3 checks exactly that, and raises.
#
# **Learning objectives**
#
# - Derive a holdout retraining interval from declarations rather than choosing one, and say
#   which declaration supplies each boundary.
# - Explain why the interval's upper bound is a label horizon below the window's start, and what
#   a zero gap would leak.
# - Read a holdout prediction set as a measurement of one configuration rather than a comparison
#   among several.
# - Say why a notebook that evaluates a second configuration on this window refuses instead of
#   replacing the first.
#
# **Book reference**: Chapter 20, Section 20.2
#
# **Prerequisites**: [`19_costs`](19_costs.ipynb) is the last stage that selects; the
# configuration it prices is the one refitted here.
#
# **What it writes**: one training run and one prediction set, both at `split='holdout'`. No
# backtest, no selection, no comparison.

# %%
"""US equities panel: refit the selected configuration and predict the holdout window."""

import polars as pl

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.models import reconstruct_locked_model_request
from case_studies.utils.registry import training_hash_from_spec
from case_studies.utils.strategy_analysis import (
    holdout_generations_to_retire,
    registered_holdout_generations,
    resolve_solvent_carrier,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)

# %% [markdown]
# ## 1. Which configuration the holdout runs
#
# The holdout runs the configuration this case study reports, resolved through the same
# `resolve_solvent_carrier` that [`19_costs`](19_costs.ipynb) prices. Resolving it again here
# rather than passing a hash along is deliberate: the two notebooks then agree by construction,
# and a hash written down in one and read in the other agrees only until the sweep is rebuilt.
#
# Nothing about the holdout enters this choice. The selected configuration is the cross-stage
# validation rank-1 over the baseline, allocation and risk-overlay stages, and it was fixed
# before this notebook ran. Which stage it comes from is printed rather than assumed, because on
# this panel the leader is not always the risk-overlay run.

# %%
carrier = resolve_solvent_carrier(CASE_STUDY_ID)
print(
    f"Selected configuration: {carrier['val_backtest_hash']}  stage={carrier['val_stage']}  "
    f"family={carrier['family']}  config={carrier['config_name']}  "
    f"label={carrier['label']}"
)
print(
    f"  validation Sharpe {carrier['val_sharpe']:.3f}, max drawdown {carrier['max_drawdown']:.3f}"
)
print(f"  fitted by training run {carrier['training_hash']}")

# %% [markdown]
# The checkpoint is part of the configuration. Where a family publishes a prediction set per
# checkpoint on a declared schedule, the selected configuration's prediction set names one of
# them, and refitting without it would produce the model at the end of training rather than the
# one that was ranked. A family with no checkpoint dimension stores NULL in both columns, and
# that NULL is carried through unchanged.

# %%
validation_prediction = study.results.open(carrier["val_prediction_hash"])
prediction_record = validation_prediction.registry_record()
CHECKPOINT_KIND = prediction_record["checkpoint_kind"]
CHECKPOINT_VALUE = prediction_record["checkpoint_value"]
print(f"Checkpoint: {CHECKPOINT_KIND}={CHECKPOINT_VALUE}")

# %% [markdown]
# ## 2. The window, and the model that is allowed to see it
#
# The holdout window is not a choice made here. It is `evaluation.holdout_start` and
# `evaluation.holdout_end` from this case study's own `setup.yaml`, read through the same
# `canonical_window` the fold derivation and the backtest slice both go through, so the three
# cannot disagree.
#
# The training interval is everything available before that window, bounded above by a label
# buffer. **The size of that buffer depends on which label the selection landed on.** This panel
# declares three - `fwd_ret_1d`, `fwd_ret_5d` and `fwd_ret_21d` - with buffers of one, five and
# twenty-one sessions, and the derivation reads the buffer of the selected label rather than
# defaulting one. A row dated `t` records an outcome that is not known until `t` plus the
# horizon, so a training set running to the day the window opens would be fitted on labels whose
# returns resolve inside it. Both boundaries are printed below so the gap can be read rather
# than taken on the derivation's word.
#
# Everything else about the configuration is carried across unchanged, and the fields that
# cannot be - the eligibility manifest, and any parameter this family resolves from a fold's own
# training rows - are recomputed against the holdout fold. Carrying those forward would fit a
# model keyed to the validation folds and call it a retrain.

# %%
observation_timeline = (
    pl.read_parquet(study.root / "labels" / f"{carrier['label']}.parquet")
    .get_column("timestamp")
    .unique()
    .sort()
    .to_list()
)
validation_spec = study.results.open(carrier["training_hash"]).spec()
holdout_spec = build_holdout_training_spec(
    study,
    validation_spec,
    timeline=observation_timeline,
    case_study=CASE_STUDY_ID,
)

fold = holdout_spec["computation"]["cv"]["folds"][0]
print(f"Holdout fold {fold['fold']}")
print(f"  trains   {fold['train_start']} -> {fold['train_end']}")
print(f"  predicts {fold['val_start']} -> {fold['val_end']}")
print(f"  label buffer: {holdout_spec['computation']['cv']['request']['label_buffer']}")

# The validation folds are what the buffer is measured against, and the last of them ends before
# the holdout opens. Printing both is what lets a reader check the gap.
validation_folds = validation_spec["computation"]["cv"]["folds"]
latest_validation_end = max(str(entry["val_end"]) for entry in validation_folds)
print(f"Validation folds: {len(validation_folds)}, latest evaluation end {latest_validation_end}")
print(f"Holdout training ends {fold['train_end']}, holdout opens {fold['val_start']}")

# %% [markdown]
# ## 3. Fit, and register the predictions
#
# `reconstruct_locked_model_request` builds the request from the specification above. Its name
# comes from a locked holdout path this case study does not use; it takes a training
# specification and a checkpoint, not a lock, and it is used here because it is the one call
# that refuses a request that is not exactly the specification it was handed - the training
# identity, the checkpoint schedule, the feature lineage and the runtime parameters are all
# checked before anything is fitted.
#
# The training identity below is new. It has to be: it covers the CV interval, and the holdout
# fold is not one of the sixteen validation folds. A run that came back with the validation
# training hash would mean the refit did not happen, so that is checked rather than assumed.
#
# **The window carries one configuration, and this notebook has no way past that.** The check
# below is on the selected configuration rather than on the notebook, and it has exactly two
# outcomes. With the selected configuration unchanged this is an idempotent replay: the
# derivation is deterministic and the training identity covers it, so the same identity comes
# back and the fit is served from the registry, which is why re-running is free. With the
# selected configuration changed it refuses, names both configurations, and stops.
#
# It refuses rather than offering a replacement switch, and the reason is that a replacement
# would not be one. Deleting the earlier generation's rows does not undo having observed its
# result: the selection that produced the new configuration may have been informed by the old
# holdout number, and no deletion reaches that. A switch here would let the case study take a
# second look at the window while leaving a registry that shows only one, which is the specific
# thing that would make the out-of-sample claim false rather than merely weak.

# %%
holdout_training_hash = training_hash_from_spec(holdout_spec)
this_generation = (holdout_training_hash, (CHECKPOINT_KIND, CHECKPOINT_VALUE))
retire = holdout_generations_to_retire(CASE_DIR, this_generation=this_generation)
# A row whose training run records no CV split cannot be shown either way, and deleting on that
# would discard a result nothing has established is wrong. It stops the run instead.
if retire.unattributable:
    raise RuntimeError(
        "the holdout window carries prediction sets whose training runs record no CV split, "
        "so whether they were refitted for the holdout cannot be established: "
        + ", ".join(
            f"{row['prediction_hash']} (training {row['training_hash']})"
            for row in retire.unattributable
        )
        + ". Establish what produced them before registering another evaluation on the same "
        "window; this notebook will not delete a row it cannot show is not a holdout result."
    )
# A row whose training run declares a non-holdout CV may not be reported as a holdout result,
# and it is also not something to delete unattended: `generate_holdout` refits on a holdout fold
# and then registers the predictions under the VALIDATION training identity, so this record
# covers both a validation-fitted model published over the window and a real refit filed under
# the wrong identity.
if retire.not_out_of_sample:
    raise RuntimeError(
        "the holdout window carries prediction sets whose training runs declare a CV split "
        "other than the holdout: "
        + ", ".join(
            f"{row['prediction_hash']} ({row['config_name']}, training {row['training_hash']})"
            for row in retire.not_out_of_sample
        )
        + ". Each is either a validation-fitted model published over the window, which is not "
        "an out-of-sample result, or a refit registered under its validation training identity, "
        "which the retired `20_strategy_synthesis/holdout.py::generate_holdout` wrote until it "
        "was deleted on 2026-09-12 - and the registry cannot tell those apart. This notebook has no way past that: establish which it is and "
        "resolve it through the registry's own lifecycle, which records that the row was retired."
    )
superseded = list(retire.superseded)
if superseded:
    raise RuntimeError(
        "the holdout window already carries a refit of a different configuration: "
        + ", ".join(
            f"{row['prediction_hash']} ({row['config_name']}, training {row['training_hash']})"
            for row in superseded
        )
        + f". This run would evaluate {carrier['config_name']} (training "
        f"{holdout_training_hash}, checkpoint {CHECKPOINT_KIND}={CHECKPOINT_VALUE}) on the same "
        "window, which would be a second configuration measured on a period this case study "
        "reports as unseen. This notebook has no way past that: deleting the earlier generation "
        "would not undo having observed it, and the selection bias it introduces is not removed "
        "by removing the rows. Either leave the selection where it was, or retire the earlier "
        "evaluation through the registry's own lifecycle, which records that a second look was "
        "taken."
    )

# %% tags=["results"]
request = reconstruct_locked_model_request(
    study,
    holdout_spec,
    checkpoint_kind=CHECKPOINT_KIND,
    checkpoint_value=CHECKPOINT_VALUE,
)
model_run = request.run()
holdout_prediction = model_run.predictions[0]

if model_run.training.hash == carrier["training_hash"]:
    raise RuntimeError(
        "the holdout refit produced the validation training identity "
        f"{carrier['training_hash']}, which means it did not refit"
    )
print(f"Holdout training run:   {model_run.training.hash}")
print(f"Holdout prediction set: {holdout_prediction.hash}")

# %% [markdown]
# What the prediction set covers, read back from the registry rather than from the request. The
# two agree only if the fit published what it declared, and the counts are what a reader checks
# the window against: this is a daily panel, so the session count is trading days in the window
# and the row count is those sessions times the names eligible on each.

# %% tags=["results"]
record = holdout_prediction.registry_record()
predictions = holdout_prediction.load()
print(
    f"split={record['split']}  checkpoint={record['checkpoint_kind']}={record['checkpoint_value']}"
)
print(f"rows={predictions.height:,}  sessions={predictions['timestamp'].n_unique():,}")
print(
    f"  {predictions['timestamp'].min()} -> {predictions['timestamp'].max()}, "
    f"{predictions['symbol'].n_unique():,} names"
)

# %% [markdown]
# Every holdout prediction set the registry holds, and whether the model behind it was refitted
# for the window. All of them are listed rather than one silently preferred, because the
# registry is immutable and a reader looking at it later will see whatever is there. A row
# marked VALIDATION-FITTED is not an out-of-sample result whatever its numbers say.

# %% tags=["results"]
for row in registered_holdout_generations(CASE_DIR):
    note = (
        "refitted for the holdout" if row["refitted"] else "VALIDATION-FITTED - not out of sample"
    )
    print(
        f"  {row['prediction_hash']}  training={row['training_hash']}  {row['config_name']}  {note}"
    )

# %% [markdown]
# ## What this notebook establishes, and what it does not
#
# It establishes one thing: a prediction set over the holdout window, produced by the
# configuration this case study selected, fitted on data that ends a full label horizon before
# the window opens. That is a precondition for an out-of-sample claim, not the claim itself.
# Nothing here says whether the predictions are any good - they have not been scored, sized or
# traded.
#
# It does not make the holdout a fresh test in the strict sense. The configuration reached this
# notebook through a selection made on the validation folds, and this window is being used once
# per configuration that gets here. What it does remove is the specific circularity of scoring a
# validation-fitted model on the period meant to judge it.
#
# Re-running this notebook is free: the same configuration re-derives the same training identity
# and the fit is served from the registry. Evaluating a different configuration is not, and is
# refused above.
#
# **Next:** [`21_holdout_backtest`](21_holdout_backtest.ipynb).
