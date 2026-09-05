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
# # NASDAQ-100 Microstructure: Holdout Predictions
#
# **Chapter 20 - Out-of-sample evaluation**
#
# Every number in this case study so far was measured on the validation folds, and every
# choice was made by looking at them: which model family, which configuration, how many names
# to hold at each decision, how to size them, which risk control to overlay, what to charge for
# crossing the spread. A result selected that way cannot also be evidence that the selection was
# sound - the ranking and the evidence would be the same measurement.
#
# The holdout is the window nothing has been selected on. This notebook fits the selected
# configuration on the history available before that window opens and writes its
# predictions over it. [`19_holdout_backtest`](19_holdout_backtest.ipynb) turns those
# predictions into a return series with the sizing and the overlay the case study settled
# on, and [`20_strategy_analysis`](20_strategy_analysis.ipynb) reads both back.
#
# **What this notebook is careful about**
#
# A holdout prediction is not the validation model scored on a later window. Section 2
# fits again, over a training interval that ends before the window opens, and the new
# training identity is what makes the refit visible rather than asserted: the identity
# covers the CV interval, so a run that came back with the validation training hash would
# mean no refit happened. The check is in section 3 and it raises.
#
# **Prerequisites:** [`17_costs`](17_costs.ipynb), which is the last stage that selects.
#
# **Scope:** one training run and one prediction set. No backtest, no selection, no
# comparison - those are 19 and 20.

# %%
"""NASDAQ-100 Microstructure: Holdout Predictions."""

import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.models import reconstruct_locked_model_request
from case_studies.utils.registry import training_hash_from_spec
from case_studies.utils.strategy_analysis import (
    resolve_solvent_carrier,
    training_run_fitted_for_the_holdout,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "nasdaq100_microstructure"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)


def _registered_holdout_generations(case_dir):
    """Every holdout prediction set in the registry, and whether its model was refitted.

    ``refitted`` is read from the training run's own CV rather than from the prediction set's
    split: the split says where the predictions land, and a model fitted on the validation
    folds can publish predictions over the holdout window. That is the distinction the whole
    notebook turns on, and it is the same predicate the canonical lineage resolver applies.
    """
    with sqlite3.connect(str(case_dir / "run_log" / "registry.db")) as conn:
        rows = conn.execute(
            """
            SELECT p.prediction_hash, p.training_hash, p.checkpoint_kind, p.checkpoint_value,
                   t.config_name, t.spec_json
            FROM prediction_sets p
            JOIN training_runs t ON t.training_hash = p.training_hash
            WHERE p.split = 'holdout'
            ORDER BY p.prediction_hash
            """
        ).fetchall()
    return [
        {
            "prediction_hash": prediction_hash,
            "training_hash": training_hash,
            # The checkpoint is part of the configuration, not a detail of it: one training run
            # publishes one prediction set per declared checkpoint, and moving the selection
            # from one checkpoint to another is a different configuration evaluated on the same
            # window. Identity on the training hash alone would see that as the same generation
            # and let both stand.
            "checkpoint": (checkpoint_kind, checkpoint_value),
            "config_name": config_name,
            "refitted": training_run_fitted_for_the_holdout(training_spec_json),
        }
        for (
            prediction_hash,
            training_hash,
            checkpoint_kind,
            checkpoint_value,
            config_name,
            training_spec_json,
        ) in rows
    ]


# %% [markdown]
# ## 1. Which configuration the holdout runs
#
# The holdout runs the configuration the case study reports, resolved through the same
# `resolve_solvent_carrier` [`17_costs`](17_costs.ipynb) prices. Resolving it again here
# rather than passing it along is deliberate: the two notebooks must agree by construction,
# and a hash written down in one and read in the other agrees only until the sweep is
# rebuilt.
#
# Nothing about the holdout enters this choice. The carrier is the cross-stage validation
# rank-1, resolved across every declared label rather than per label, and it was fixed before
# this notebook ran. Which stage it comes from is printed below rather than asserted here.

# %%
carrier = resolve_solvent_carrier(CASE_STUDY_ID)
print(
    f"Carrier: {carrier['val_backtest_hash']}  stage={carrier['val_stage']}  "
    f"family={carrier['family']}  config={carrier['config_name']}  "
    f"label={carrier['label']}"
)
print(
    f"  validation Sharpe {carrier['val_sharpe']:.3f}, max drawdown {carrier['max_drawdown']:.3f}"
)
print(f"  fitted by training run {carrier['training_hash']}")

# %% [markdown]
# The checkpoint is part of the configuration. Where a family publishes a prediction set per
# checkpoint on a declared schedule, the carrier's prediction set names one of them, and
# refitting without it would produce a model at the end of training rather than the one that
# was ranked. A family with no checkpoint dimension stores NULL in both columns and carries
# that NULL through unchanged.

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
# buffer - and **the buffer is not the selected label's.** `build_holdout_cv` takes the widest
# buffer any of this case study's labels declares, which here is `61min` from `fwd_ret_60m`, not
# the primary `fwd_ret_15m`'s `16min`. The reason is that the holdout fold is one fold: a
# fold-scoped temporal artifact carries a single set of boundaries, every label's holdout model
# is fitted on features carrying them, so the geometry has to be label-independent and the
# widest is the only choice that leaks for no label. A fold built on the sixteen-minute buffer
# and handed to the sixty-minute model would give it training rows whose features saw
# forty-five minutes past its own `train_end` - the leak the buffer exists to prevent, arriving
# through the feature rather than the label.
#
# **The width is minutes and it is doing the same work a long one does.** Sixty-one minutes
# against a window opening on 2021-07-01 looks like nothing beside the ETF study's twenty-one
# sessions, and it is the same leak if dropped: a training set running to the first bar of the
# window would be fitted on labels that resolve inside it. Each label declares its own
# (`fwd_ret_5m: 6min`, `fwd_ret_15m` and `fwd_dir_15m: 16min`, `fwd_ret_60m: 61min`) rather than
# inheriting the primary's, and each is a horizon plus one bar because the horizon alone does
# not describe the width of the window the outcome resolves over. The derivation refuses to
# default any of them.
#
# Everything else about the configuration is carried across unchanged, and the fields that
# cannot be - the eligibility manifest, and any parameter this family resolves from a fold's
# own training rows - are recomputed against the holdout fold. Carrying those forward would fit
# a model keyed to the validation folds and call it a retrain.

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
print(f"  trains  {fold['train_start']} -> {fold['train_end']}")
print(f"  predicts {fold['val_start']} -> {fold['val_end']}")
print(f"  label buffer: {holdout_spec['computation']['cv']['request']['label_buffer']}")

# The validation folds are what the buffer is measured against, and the last of them ends
# before the holdout opens. Printing both is what lets a reader check the gap rather than take
# it on the derivation's word.
validation_folds = validation_spec["computation"]["cv"]["folds"]
latest_validation_end = max(str(entry["val_end"]) for entry in validation_folds)
print(f"Validation folds: {len(validation_folds)}, latest evaluation end {latest_validation_end}")
print(f"Holdout training ends {fold['train_end']}, holdout opens {fold['val_start']}")

# %% [markdown]
# ## 3. Fit, and register the predictions
#
# `reconstruct_locked_model_request` builds the request from the spec above. Its name comes
# from a locked holdout path this case study does not use; it takes a training specification
# and a checkpoint, not a lock, and it is used here because it is the one call that refuses a
# request that is not exactly the spec it was handed - the training identity, the checkpoint
# schedule, the feature lineage and the runtime parameters are all checked before anything is
# fitted.
#
# The training identity below is new. It has to be: it covers the CV interval, and the holdout
# fold is not one of the validation folds. A run that came back with the validation training
# hash would mean the refit did not happen, so that is checked rather than assumed.
#
# **The window carries one configuration, and this notebook has no way past that.** The check
# below is on the carrier rather than on the notebook, and it has exactly two outcomes. With
# the carrier unchanged this is an idempotent replay: the derivation is deterministic and the
# training identity covers it, so the same identity comes back and the fit is served from the
# registry, which is why re-running the notebook is free and safe. With the carrier changed it
# refuses, names both configurations, and stops.
#
# It refuses rather than offering a replacement switch, and the reason is that a replacement
# would not be one. Deleting the earlier generation's rows does not undo having observed its
# result: the selection that produced the new carrier may have been informed by the old
# holdout number, and no deletion reaches that. A switch here would let the case study take a
# second look at the window while leaving a registry that shows only one, which is the
# specific thing that would make the out-of-sample claim false rather than merely weak.

# %%
holdout_training_hash = training_hash_from_spec(holdout_spec)
this_generation = (holdout_training_hash, (CHECKPOINT_KIND, CHECKPOINT_VALUE))
superseded = [
    row
    for row in _registered_holdout_generations(CASE_DIR)
    if row["refitted"] and (row["training_hash"], row["checkpoint"]) != this_generation
]
if superseded:
    raise RuntimeError(
        "the holdout window already carries a refit of a different configuration: "
        + ", ".join(
            f"{row['prediction_hash']} ({row['config_name']}, training {row['training_hash']})"
            for row in superseded
        )
        + f". This run would evaluate {carrier['config_name']} (training "
        f"{holdout_training_hash}, checkpoint {CHECKPOINT_KIND}={CHECKPOINT_VALUE}) on the "
        "same window, which would be a second configuration measured on a period this case "
        "study reports as unseen. This notebook has no way past that: deleting the earlier "
        "generation would not undo having observed it, and the selection bias it introduces "
        "is not removed by removing the rows. Either leave the selection where it was, or "
        "retire the earlier evaluation through the registry's own lifecycle, which records "
        "that a second look was taken."
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
print(f"Holdout training run:  {model_run.training.hash}")
print(f"Holdout prediction set: {holdout_prediction.hash}")

# %% [markdown]
# What the prediction set covers, read back from the registry rather than from the request.
# The two agree only if the fit published what it declared, and the counts are what a reader
# checks the window against. This is a MINUTE panel evaluated on a per-label decision cadence,
# so the row count is decision timestamps times the symbols eligible at each - not sessions
# times symbols. The session count is printed separately because it is the number that lines up
# with `evaluation.holdout_start` and `holdout_end`, and the two are easy to conflate on an
# intraday panel.

# %% tags=["results"]
record = holdout_prediction.registry_record()
predictions = holdout_prediction.load()
print(
    f"split={record['split']}  checkpoint={record['checkpoint_kind']}={record['checkpoint_value']}"
)
print(
    f"rows={predictions.height:,}  "
    f"decision timestamps={predictions['timestamp'].n_unique():,}  "
    f"sessions={predictions['timestamp'].dt.date().n_unique():,}"
)
print(
    f"  {predictions['timestamp'].min()} -> {predictions['timestamp'].max()}, "
    f"{predictions['symbol'].n_unique():,} symbols"
)

# %% [markdown]
# Every holdout prediction set the registry holds, and whether the model behind it was
# refitted for the window. All of them are listed rather than one silently preferred, because
# the registry is immutable and a reader looking at it later will see whatever is there. A row
# marked VALIDATION-FITTED is not an out-of-sample result whatever its numbers say.

# %% tags=["results"]
for row in _registered_holdout_generations(CASE_DIR):
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
# per configuration that gets here. What it does remove is the specific circularity of scoring
# a validation-fitted model on the period meant to judge it.
#
# Re-running this notebook is free: the same carrier re-derives the same training identity and
# the fit is served from the registry. Evaluating a DIFFERENT configuration is not, and is
# refused here. If a later pass finds the selection was wrong, that is a question for the
# registry's lifecycle, which records that a second look was taken - not something to settle by
# deleting rows until the registry agrees.
#
# **Next:** [`19_holdout_backtest`](19_holdout_backtest.ipynb).
