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
# # Holdout Predictions - FX Pairs
#
# This notebook refits the one configuration validation selected, on the holdout interval, and
# registers its predictions. It writes predictions and nothing else: the backtest is
# `18_holdout_backtest`, and what any of it is worth is `19_strategy_analysis`.
#
# Selection is not a parameter and is not made here. `resolve_solvent_carrier` reads the
# highest-Sharpe registered validation backtest across the baseline, allocation and risk-overlay
# stages, restricted to runs that stayed solvent, so this notebook cannot select a configuration
# the validation stages did not already rank first.
#
# The holdout is not a one-shot transaction and nothing here pre-registers it. There is no lock,
# no seal and no gate: the whole rule is retrain the selected configuration on everything up to
# the holdout window, predict, and backtest that same configuration on the result. Re-running is
# therefore ordinary. A reader who runs it five hundred times and quotes the best number has
# produced something uninterpretable, and that is a property of what they did rather than
# something the software can prevent - the earlier design tried to, and bought
# unfixability: a holdout found to be wrong after a bug fix could not be corrected, because the
# lock was by construction the one artifact that could not be revised.
#
# **Learning objectives**
#
# - Derive a holdout interval from the panel's observation grid rather than the calendar.
# - Reconstruct one training identity from an immutable validation specification.
# - Register holdout predictions without giving them any influence over selection.
#
# **Book reference**: Chapters 16-20
#
# **Prerequisite**: `16_costs`. The carrier is resolved from the registered validation
# backtests, so every stage that registers one must have run.

# %%
"""Refit the validation-selected FX configuration on the holdout interval."""

import polars as pl

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.models import reconstruct_locked_model_request
from case_studies.utils.strategy_analysis import resolve_solvent_carrier

# %% tags=["parameters"]
CASE_STUDY_ID = "fx_pairs"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
CANDIDATE_SET_NAME = "fx_pairs:holdout-candidates"

# %% [markdown]
# ## Resolve the selection and the holdout interval it determines
#
# The label the selection was made on decides which observation grid the holdout interval is
# stepped back along, so it is read from the selected lineage rather than assumed. FX carries
# three labels on one daily grid, which is exactly the coincidence that would let an assumption
# here survive untested.
#
# The training window ends a whole label buffer, counted in observations, before the holdout
# opens, so the last training label's outcome cannot resolve inside the holdout. The buffer is
# the widest this case study configures rather than the primary label's: a 21-day forward return
# resolves three weeks after the session it is stamped on, and a 1-day buffer would leave three
# weeks of holdout outcomes reachable from the training window.

# %% tags=["results"]
study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER, workspace=WORKSPACE or None)

# The carrier is the highest-Sharpe registered VALIDATION backtest across the baseline,
# allocation and risk-overlay stages, restricted to runs that stayed solvent. It is resolved
# from the registry rather than named here, so this notebook cannot select a configuration that
# the validation stages did not rank first.
carrier = resolve_solvent_carrier(CASE_STUDY_ID)
validation_prediction = study.results.open(carrier["val_prediction_hash"])
prediction_record = validation_prediction.registry_record()
CHECKPOINT_KIND = prediction_record["checkpoint_kind"]
CHECKPOINT_VALUE = prediction_record["checkpoint_value"]

# The label the carrier was fitted on decides which observation grid the holdout interval is
# stepped back along, so it is read from the carrier rather than assumed. FX carries three
# labels on one daily grid, which is exactly the coincidence that would let an assumption here
# survive untested.
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
holdout_fold = holdout_spec["computation"]["cv"]["folds"][0]

pl.DataFrame(
    {
        "field": [
            "carrier backtest",
            "carrier stage",
            "validation Sharpe",
            "family",
            "configuration",
            "label",
            "checkpoint",
            "validation training",
            "validation prediction",
            "holdout train window",
            "holdout evaluation window",
        ],
        "value": [
            carrier["val_backtest_hash"],
            str(carrier["val_stage"]),
            f"{carrier['val_sharpe']:.4f}",
            str(carrier["family"]),
            str(carrier["config_name"]),
            str(carrier["label"]),
            f"{CHECKPOINT_KIND}={CHECKPOINT_VALUE}",
            str(carrier["training_hash"]),
            validation_prediction.hash,
            f"{holdout_fold['train_start']} to {holdout_fold['train_end']}",
            f"{holdout_fold['val_start']} to {holdout_fold['val_end']}",
        ],
    }
)

# %% [markdown]
# ## Refit the carrier on the holdout fold
#
# The request is reconstructed from the immutable validation specification with only the fold
# geometry re-keyed, so the holdout model differs from the validation model in what it was
# fitted on and in nothing else. It publishes the selected checkpoint alone: a holdout refit
# that published its whole checkpoint schedule would hand the next notebook a choice, and
# choosing among holdout checkpoints is selection on the holdout under another name.
#
# The one thing checked afterwards that a specification cannot state about itself is that this
# is a refit at all. A holdout training identity equal to the validation one means the fold
# re-keying changed nothing, and the model is a validation fit predicting forward over a later
# window rather than a model trained up to it.

# %% tags=["results"]
request = reconstruct_locked_model_request(
    study,
    holdout_spec,
    checkpoint_kind=CHECKPOINT_KIND,
    checkpoint_value=CHECKPOINT_VALUE,
)
model_run = request.run()
if model_run.training.hash == carrier["training_hash"]:
    raise RuntimeError(
        f"the holdout refit produced the validation training identity "
        f"{carrier['training_hash']}, so it did not refit"
    )
if len(model_run.predictions) != 1:
    raise RuntimeError(
        f"the holdout refit published {len(model_run.predictions)} prediction sets; "
        "only the selected checkpoint may be published"
    )
prediction = model_run.predictions[0]

record = prediction.registry_record()
if record["split"] != "holdout":
    raise RuntimeError(f"the holdout refit published a {record['split']!r} prediction")
if record["checkpoint_kind"] != CHECKPOINT_KIND or record["checkpoint_value"] != CHECKPOINT_VALUE:
    raise RuntimeError(
        f"the holdout prediction is at checkpoint {record['checkpoint_kind']}="
        f"{record['checkpoint_value']}, not the carrier's {CHECKPOINT_KIND}={CHECKPOINT_VALUE}"
    )
if not prediction.complete:
    raise RuntimeError("the holdout prediction is incomplete")

print(f"Holdout training run:   {model_run.training.hash}")
print(f"Holdout prediction set: {prediction.hash}")

# %% [markdown]
# ## What the holdout refit covers
#
# The coverage is printed rather than assumed: a holdout prediction set that silently covers a
# shorter window than the fold declares would make every number downstream a measurement of a
# different interval than the one this notebook says it measured.

# %% tags=["results"]
frame = prediction.load()
coverage = pl.DataFrame(
    {
        "field": ["prediction", "rows", "symbols", "sessions", "first session", "last session"],
        "value": [
            prediction.hash,
            str(frame.height),
            str(frame.get_column("symbol").n_unique()),
            str(frame.get_column("timestamp").n_unique()),
            str(frame.get_column("timestamp").min()),
            str(frame.get_column("timestamp").max()),
        ],
    }
)
coverage

# %% [markdown]
# ## Key takeaways
#
# - The configuration refitted here was selected on validation, from an immutable set, upstream.
# - The holdout training window stops a full label buffer short of the holdout, counted in
#   observations rather than calendar days.
# - Only the selected checkpoint is published, so no choice among holdout results remains to be
#   made downstream.
