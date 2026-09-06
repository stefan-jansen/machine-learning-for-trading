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
# # Crypto Perpetuals Funding: Holdout Predictions
#
# **Chapter 20 - Out-of-sample evaluation**
#
# Every number this case study has reported was measured on the validation folds, and every
# choice was made by looking at them: the label, the model family, the entry rule, the
# allocator, the risk control. A result selected that way cannot also be the evidence that
# the selection was sound, because the ranking and the evidence would be the same
# measurement.
#
# The holdout is the window nothing has been selected on. This notebook fits the selected
# configuration on the history that ends before that window opens and writes its predictions
# over it. [`18_holdout_backtest`](18_holdout_backtest.ipynb) turns those predictions into a
# return series with the sizing and the overlay the case study settled on, and
# [`19_strategy_analysis`](19_strategy_analysis.ipynb) reads both back.
#
# **What this notebook is careful about**
#
# A holdout prediction is not the validation model scored on a later window. Section 3 fits
# again, and the new training identity is what makes the refit visible rather than asserted:
# a run that came back with the validation training hash would mean no refit happened, and
# it raises.
#
# **Prerequisites:** [`16_costs`](16_costs.ipynb), which is the last stage that could still
# move the selection.
#
# **Scope:** one training run and one prediction set. No backtest, no selection, no
# comparison - those are 18 and 19.

# %%
"""Crypto Perpetuals Funding: Holdout Predictions."""

import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.models import reconstruct_locked_model_request
from case_studies.utils.registry import training_hash_from_spec
from case_studies.utils.registry.maintenance import delete_prediction_generation
from case_studies.utils.strategy_analysis import (
    resolve_solvent_carrier,
    training_run_fitted_for_the_holdout,
)
from utils.paths import get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "crypto_perps_funding"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
# Whether a holdout generation for a DIFFERENT configuration may be superseded by this run.
# Off by default: see section 3.
#
# The flag exists because the holdout is not a one-shot resource. What the rule against
# consulting the holdout forbids is SELECTING on it: the configuration evaluated here is
# chosen by validation Sharpe across the signal, allocation and risk stages, and no holdout
# number feeds back into that choice. It says nothing about how many times the evaluation
# may be computed, and a wrong result is deleted and re-run rather than left standing because
# it was observed. The guard below is against something narrower and real: two generations
# readable at once, so nobody downstream has to choose between them and nobody can quote
# whichever number they prefer.
REPLACE_HOLDOUT = False

# %%
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)


def _registered_holdout_generations(case_dir):
    """Every holdout prediction set in the registry, and whether its model was refitted.

    ``refitted`` is read from the training run's own CV rather than from the prediction
    set's split: the split says where the predictions land, and a model fitted on the
    validation folds can publish predictions over the holdout window. Reading the split
    alone would call that a holdout evaluation.
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
            # The checkpoint is part of the configuration, not a detail of it. Identity on
            # the training hash alone would read two checkpoints of one run as the same
            # generation and let both stand.
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
# The holdout runs the configuration the case study reports. `resolve_solvent_carrier`
# applies the selection rule this case study's funnel already implements: compare the
# registered validation backtests across the signal, allocation and risk stages, and take
# the highest Sharpe. It is resolved here rather than passed in from
# [`16_costs`](16_costs.ipynb), so the two agree by construction rather than by a hash
# copied between them.
#
# The two routes to that configuration were checked against each other rather than assumed
# to agree: this resolver and the `crypto-final-validation-{label}` candidate sets that
# [`15_risk_management`](15_risk_management.ipynb) freezes return the same backtest.
#
# Nothing about the holdout enters this choice. The carrier was fixed before this notebook
# ran.

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
# The checkpoint is part of the configuration. Families that checkpoint through training
# publish one prediction set per declared iteration, and the carrier's prediction set names
# one of them - so refitting without it would produce a model at the end of training rather
# than the one that was ranked. A family that checkpoints once carries nulls here, and
# passing them through unchanged is what keeps the lookup exact either way.

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
# `evaluation.holdout_end` from this case study's own `setup.yaml` - 2024 and 2025 - read
# through the same `canonical_window` the fold derivation and the backtest slice both go
# through, so the three cannot disagree.
#
# The training interval is everything available before that window, bounded above by a label
# buffer. The buffer is what stops the last training label's outcome from resolving inside
# the holdout, and here it is a real horizon rather than a formality: these labels are
# forward returns over 8 and 24 hours, so a row observed at the last training timestamp is
# still unrealised for a full horizon after it. The derivation takes the widest declared
# horizon across the case study's labels and refuses to default it - a zero gap would be a
# leak, not a conservative choice.
#
# Everything else about the configuration is carried across unchanged, and the fields that
# cannot be - the eligibility manifest, and any parameter this family resolves from a fold's
# own training rows - are recomputed against the holdout fold. Carrying those forward would
# fit a model keyed to the validation folds and call it a retrain.

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
# before the holdout opens. Printing both is what lets a reader check the gap rather than
# take it on the derivation's word.
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
# The training identity below is new. It has to be: it covers the CV interval, and the
# holdout fold is not one of the validation folds. A run that came back with the validation
# training hash would mean the refit did not happen, and the check after it raises.
#
# **The window carries one configuration at a time.** The holdout is re-runnable, and that is
# not the same as free: every configuration evaluated on it is another look at a period the
# case study reports as unseen, and two evaluated quietly would make that report false.
#
# So the check below is on the carrier rather than on the notebook, and it has exactly two
# outcomes. With the carrier unchanged this is an idempotent replay: the derivation is
# deterministic and the training identity covers it, so the same identity comes back and the
# fit is served from the registry. With the carrier changed it refuses, names both
# configurations, and stops.
#
# `REPLACE_HOLDOUT` is the only way past that, and it is a replacement rather than an
# addition: the superseded generation's rows are deleted, so the registry never holds two
# refits of the holdout window and no downstream resolver has to choose between them.

# %%
holdout_training_hash = training_hash_from_spec(holdout_spec)
this_generation = (holdout_training_hash, (CHECKPOINT_KIND, CHECKPOINT_VALUE))
superseded = [
    row
    for row in _registered_holdout_generations(CASE_DIR)
    if row["refitted"] and (row["training_hash"], row["checkpoint"]) != this_generation
]
if superseded and not REPLACE_HOLDOUT:
    raise RuntimeError(
        "the holdout window already carries a refit of a different configuration: "
        + ", ".join(
            f"{row['prediction_hash']} ({row['config_name']}, training {row['training_hash']})"
            for row in superseded
        )
        + f". This run would evaluate {carrier['config_name']} (training "
        f"{holdout_training_hash}, checkpoint {CHECKPOINT_KIND}={CHECKPOINT_VALUE}) on the "
        "same window. Set REPLACE_HOLDOUT=True to discard the earlier generation, or leave "
        "the selection where it was."
    )
for row in superseded:
    print(f"REPLACING holdout generation {row['prediction_hash']} ({row['config_name']})")
    # The rows go rather than being marked: a superseded holdout evaluation that is still
    # readable is still a number someone can quote, and the point of replacing it is that it
    # should not be one. `delete_prediction_generation` derives the child tables from
    # `PRAGMA foreign_key_list` rather than listing them, so a table added to the schema
    # later is covered without an edit, and it enables foreign keys on its own connection -
    # SQLite leaves them off per connection, which is the only reason a delete that misses a
    # child table appears to succeed.
    removed = delete_prediction_generation(
        CASE_DIR / "run_log" / "registry.db", row["prediction_hash"]
    )
    print(f"  removed {sum(removed.values())} rows: {removed}")

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
# can check the window against: perpetual funding settles every eight hours, so two years is
# on the order of two thousand decision timestamps, and the row count is those timestamps
# times the names eligible at each. The panel is unbalanced - assets enter at listing - so
# the name count is an upper bound rather than a constant.

# %% tags=["results"]
record = holdout_prediction.registry_record()
predictions = holdout_prediction.load()
print(
    f"split={record['split']}  checkpoint={record['checkpoint_kind']}={record['checkpoint_value']}"
)
print(f"rows={predictions.height:,}  timestamps={predictions['timestamp'].n_unique():,}")
print(
    f"  {predictions['timestamp'].min()} -> {predictions['timestamp'].max()}, "
    f"{predictions['symbol'].n_unique():,} names"
)

# %% [markdown]
# Every holdout prediction set the registry holds, and whether the model behind it was
# fitted for this window. Both are listed rather than one silently preferred: the registry is
# immutable, and a reader looking at it later sees whatever is there.

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
# configuration this case study selected, fitted on data that ends before the window opens.
# That is a precondition for an out-of-sample claim, not the claim itself. Nothing here says
# whether the predictions are any good - they have not been scored, sized or traded.
#
# It does not make the holdout a fresh test in the strict sense. The configuration reached
# this notebook through a selection made on the validation folds, and this window is being
# used once per configuration that gets here. What it does remove is the specific
# circularity of scoring a validation-fitted model on the period meant to judge it.
#
# The holdout is re-runnable. If a later pass finds the selection was wrong, the answer is to
# delete this generation and produce another, not to treat the first as spent.
#
# **Next:** [`18_holdout_backtest`](18_holdout_backtest.ipynb).
