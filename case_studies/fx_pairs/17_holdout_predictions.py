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
# Selection is not a parameter. The candidate set frozen by `15_risk_management` is immutable, and
# its highest validation backtest Sharpe is read from that set rather than supplied here, so this
# notebook cannot select a different configuration than the one already on record.
#
# The holdout is not a one-shot transaction. An earlier design took an authorization lock at this
# point and spent it, which made the holdout unrepeatable by construction and therefore unfixable
# whenever anything upstream turned out to be wrong: the fix needed a retrain the lock forbade.
# What protects the holdout is that selection happens upstream against an immutable set, so
# re-running this notebook re-derives the same training identity and reuses the registered result
# instead of producing a second one.
#
# **Learning objectives**
#
# - Derive a holdout interval from the panel's observation grid rather than the calendar.
# - Reconstruct one training identity from an immutable validation specification.
# - Register holdout predictions without giving them any influence over selection.
#
# **Book reference**: Chapters 16-20
#
# **Prerequisite**: `16_costs`, and the candidate set `15_risk_management` freezes.

# %%
"""Refit the validation-selected FX configuration on the holdout interval."""

import polars as pl

from case_studies.research import open_study
from case_studies.research.holdout import resolve_holdout_selection
from case_studies.research.models import (
    reconstruct_locked_model_request,
    validate_locked_model_run,
)

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
selection = resolve_holdout_selection(study, candidate_set_name=CANDIDATE_SET_NAME)

holdout_cv = selection.holdout_training_spec["computation"]["cv"]
holdout_fold = holdout_cv["folds"][-1]

pl.DataFrame(
    {
        "field": [
            "candidate set",
            "candidate count",
            "selected validation backtest",
            "selected prediction",
            "selected training",
            "label",
            "family",
            "configuration",
            "checkpoint",
            "holdout training identity",
            "holdout train window",
            "holdout evaluation window",
        ],
        "value": [
            selection.candidate_set.hash,
            str(len(selection.candidate_set.members)),
            selection.validation_backtest.hash,
            selection.validation_prediction.hash,
            selection.validation_training.hash,
            selection.label,
            str(selection.validation_training.spec()["family"]),
            str(selection.validation_training.spec()["config_name"]),
            f"{selection.checkpoint_kind}={selection.checkpoint_value}",
            selection.holdout_training_hash,
            f"{holdout_fold['train_start']} to {holdout_fold['train_end']}",
            f"{holdout_fold['val_start']} to {holdout_fold['val_end']}",
        ],
    }
)

# %% [markdown]
# ## Refit the selected configuration on the holdout fold
#
# The request is reconstructed from the immutable validation specification with only the fold
# geometry re-keyed, so the holdout model differs from the validation model in what it was fitted
# on and in nothing else. The runner publishes the selected checkpoint alone: a holdout refit that
# published its whole checkpoint schedule would hand the next notebook a choice, and choosing
# among holdout checkpoints is selection on the holdout under another name.
#
# The fitted state is validated on every run, not only on the run that produced it. A second run
# reuses the registered identity rather than refitting, and accepting that on the strength of the
# registry row alone would skip the one check worth keeping: that the weights on disk are what
# this specification produces, with fold state that agrees.

# %% tags=["results"]
# The request is reconstructed whether or not the refit has to run, because the fitted state is
# validated on both paths. A reused prediction is persisted state from an earlier process, and
# accepting it because a row exists would trust exactly the thing worth checking: that the
# weights on disk are the ones this specification produces, with fold state that agrees. The
# runner itself reuses a complete identity rather than refitting, so reconstructing costs a
# resolution, not a training run.
request = reconstruct_locked_model_request(
    study,
    selection.holdout_training_spec,
    checkpoint_kind=selection.checkpoint_kind,
    checkpoint_value=selection.checkpoint_value,
)
existing = selection.holdout_prediction
model_run = request.run()
if model_run.training.hash != selection.holdout_training_hash:
    raise RuntimeError(
        f"the holdout refit produced training {model_run.training.hash}, "
        f"not the derived identity {selection.holdout_training_hash}"
    )
if len(model_run.predictions) != 1:
    raise RuntimeError(
        f"the holdout refit published {len(model_run.predictions)} predictions; "
        "only the selected checkpoint may be published"
    )
prediction = model_run.predictions[0]
fitted_state_digest = validate_locked_model_run(request, model_run)
if not fitted_state_digest:
    raise RuntimeError("the holdout model produced no fitted-state digest")
if existing is not None and existing.hash != prediction.hash:
    raise RuntimeError(
        f"holdout prediction {existing.hash} was already registered for training "
        f"{selection.holdout_training_hash}, but this run resolved {prediction.hash}"
    )
print(
    f"Holdout prediction {prediction.hash} "
    f"{'reused' if existing is not None else 'refitted and registered'}; "
    f"fitted state {fitted_state_digest[:12]}."
)

record = prediction.registry_record()
if record["split"] != "holdout":
    raise RuntimeError(f"the holdout refit published a {record['split']!r} prediction")
if (
    record["checkpoint_kind"] != selection.checkpoint_kind
    or record["checkpoint_value"] != selection.checkpoint_value
):
    raise RuntimeError(
        f"the holdout prediction is at checkpoint {record['checkpoint_kind']}="
        f"{record['checkpoint_value']}, not the selected "
        f"{selection.checkpoint_kind}={selection.checkpoint_value}"
    )
if not prediction.complete:
    raise RuntimeError("the holdout prediction is incomplete")
if prediction.lineage()["training_spec"] != selection.holdout_training_spec:
    raise RuntimeError("the holdout prediction was fitted against a different training spec")

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
