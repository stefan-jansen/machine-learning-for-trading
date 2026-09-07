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
# # S&P 500 Equity+Options: Holdout Predictions
#
# Every measurement in notebooks 14 through 17 came from validation. This one
# refits the single configuration those four stages arrived at, on all the
# history available before 2021, and predicts the 2021 holdout. It publishes
# predictions and nothing else: what they are worth as a strategy is
# [`19_holdout_backtest`](19_holdout_backtest.ipynb), and what the whole case
# study concludes is [`20_strategy_analysis`](20_strategy_analysis.ipynb).
#
# **Learning objectives**
#
# 1. Derive a holdout retraining interval from declarations rather than choosing
#    one, and see which declaration supplies each boundary.
# 2. Understand why the interval's start is bounded by what the features reach
#    and not by what the calendar allows.
# 3. Read a holdout prediction set as a measurement of one configuration, not as
#    a comparison among several.
#
# **Book reference:** Chapter 20, Section 20.2.
#
# **Prerequisites:** [`17_costs`](17_costs.ipynb), which stresses the
# configuration this notebook refits. Signals form after Friday's close and
# execute at the next available open. The current-constituent universe retains
# survivorship bias.

# %% [markdown]
# ### The holdout is used deliberately, not once
#
# An earlier version of this case study spent the holdout through a lifecycle
# lock that could be finalized a single time, and read a mis-derived result back
# forever after. That is the wrong trade for a book: a holdout evaluated on a
# configuration the pipeline no longer selects is not out-of-sample evidence
# about anything, and machinery whose purpose is to prevent a re-run is
# machinery whose purpose is to preserve a stale answer.
#
# So this notebook re-runs like any other stage. What replaces the lock is that
# nothing here is chosen by hand: the configuration is read from the registry by
# the same rule `17_costs` applies, and every boundary of the retraining
# interval is derived from `config/setup.yaml`, the validation fold set, and the
# feature artifact. The discipline that matters - that the holdout window is
# never consulted while selecting - is enforced by notebooks 14 through 17
# ranking on validation only, which is where it belongs.
#
# One thing does not change and cannot: a holdout read many times, with the
# selection revised after each read, is validation under another name. The
# reader's protection is that the selection above this notebook is reproducible
# from the registry, so a changed answer here is traceable to a changed
# selection rather than to a preference.

# %%
"""S&P 500 Equity+Options: refit the selected configuration and predict the holdout."""

import hashlib
import sqlite3
import warnings
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import CandidateSet, open_selection_field, open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.models import (
    reconstruct_locked_model_request,
    validate_locked_model_run,
)
from case_studies.utils.backtest_loaders import get_backtest_config
from case_studies.utils.backtest_presets import strategy_view
from case_studies.utils.notebook_contracts import prediction_members_in_force
from case_studies.utils.registry import resolve_best_backtest_runs
from case_studies.utils.registry.specs import training_hash_from_spec
from case_studies.utils.strategy_analysis import resolve_solvent_carrier

# %% tags=["parameters"]
CASE_STUDY_ID = "sp500_equity_option_analytics"
EXECUTION_TIER = "canonical"
WORKSPACE: str = ""
LABEL = ""

# %% [markdown]
# ### What is asked for, and what it resolves to
#
# The parameters above are the request; the values this notebook runs on are resolved here under
# different names, so a resolved value cannot overwrite the request that produced it. An injected
# parameter wins; otherwise the case study's own declaration does.
#
# This notebook writes to the registry, so it opens the study rather than reading one.
# `open_study` activates the tier, which is what lets a fit publish a training run and a
# prediction set under it; the read-only `Study.at` the preceding four notebooks use cannot.
#
# **Nothing is resolved before this call.** Activation is what decides which case directory the
# run reads and writes, so a path, a registry handle or a configuration read beforehand names
# the released case study while everything after it names the workspace. Every resolved value
# below is therefore derived from the opened study, and the registry it resolved is printed.

# %%
study = open_study(
    CASE_STUDY_ID,
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="18_holdout_predictions",
)
CASE_DIR = study.root
REGISTRY_DB = CASE_DIR / "run_log" / "registry.db"
bt_config = get_backtest_config(CASE_STUDY_ID)
# The label this stage runs under is a property of what the selection chose, so it is resolved
# below rather than here. `labels.primary` was the winner's label only by coincidence: the field
# spans every declared label, and the stages after the selection - price windows, schedule
# thinning, the return contract - have to be keyed to the label that won.
REQUESTED_LABEL = LABEL
print(f"Case study: {CASE_STUDY_ID}")
print(f"Registry: {REGISTRY_DB}")
# %% [markdown]
# ## 1. Read the selection out of the frozen candidate set
#
# The configuration this notebook refits is not chosen here and is not a
# parameter. [`16_risk_management`](16_risk_management.ipynb) is the last stage
# that ranks, and it writes the field it ranked over as an immutable candidate
# set. This reads the highest validation Sharpe out of that set.
#
# The distinction matters more than it looks. Re-deriving the ranking here would
# also give the right answer today, and would keep giving an answer after
# something upstream moved - silently refitting a different configuration than
# the one the case study reported selecting. A frozen set cannot do that. If it
# no longer describes the pipeline, the resolution fails and says so, and the
# repair is to re-run 16 and look at what changed.

# %%
CANDIDATE_SET_NAME = f"{CASE_STUDY_ID}:holdout-candidates"
# The frozen set where it exists, and the same construction applied live where it does not.
# 16_risk_management writes it by opening the study, which canonical regeneration refuses
# wherever the generated directories are not symlinks - a reader's clean clone and the test
# fixtures both - so the set is in the published run log and absent everywhere else. Reading it
# is the stronger path: it is immutable, so it cannot follow an upstream change. Re-deriving is
# the same rule applied live, and cannot notice that something moved. Which one ran is printed.
#
# Both paths go through `open_selection_field`, which is also what 16 freezes with. They used to
# be separate copies and they disagreed: the freeze spanned every declared label and this
# fallback spanned one, so which configuration a reader selected depended on whether their
# registry held a `candidate_sets` table.
FIELD = open_selection_field(
    study,
    case_study=CASE_STUDY_ID,
    name=CANDIDATE_SET_NAME,
    prediction_hashes=prediction_members_in_force(study)[0],
    resolve_best_backtest_runs=resolve_best_backtest_runs,
)
CANDIDATES = FIELD.candidate_set
FIELD_HASHES = list(FIELD.members)
FIELD_NAME = f"frozen candidate set {CANDIDATES.hash}" if CANDIDATES is not None else "live ranking"
SELECTION_SOURCE = FIELD.source
# The field says which backtests may be chosen from; the resolver says which one is chosen,
# and it is handed the field rather than the whole registry. `SelectionField.selected` is
# `CandidateSet.best_validation_sharpe`, which ranks the stored Sharpe column with a hash
# tie-break and applies nothing else - no re-ranking onto the timestamps every candidate
# prices, no label or universe restriction, no refusal of a run whose equity reached zero.
# This case study's field holds a conformal allocator that sits out its warm-up and books it
# as returns of exactly zero, so the two rankings read two different samples: they name the
# same backtest on this registry and value it at 2.6087 stored against 2.6329 over the shared
# sessions. `20_strategy_analysis` resolves the same way, so all three notebooks describe one
# configuration.
CARRIER = resolve_solvent_carrier(CASE_STUDY_ID, admitted=frozenset(FIELD_HASHES))
SELECTED = study.results.open(CARRIER["val_backtest_hash"])
print(
    f"{FIELD_NAME}: {len(FIELD_HASHES)} members, resolver picks {SELECTED.hash}, "
    f"stored-Sharpe pick {FIELD.selected.hash}"
)

# The label the stages after the selection run under is the winner's, not the case study's
# primary. An injected LABEL is a request to run a different one, and it has to agree with what
# was selected or the holdout refit would be keyed to a contract the selection does not name.
HOLDOUT_LABEL = CARRIER["label"]
if REQUESTED_LABEL and REQUESTED_LABEL != HOLDOUT_LABEL:
    raise RuntimeError(
        f"LABEL={REQUESTED_LABEL!r} was requested but the selection carried forward is "
        f"{SELECTED.hash} on {HOLDOUT_LABEL!r}. Running the holdout refit under another "
        "label's contract would report a different strategy from the one selected."
    )
print(f"Label carried by the selection: {HOLDOUT_LABEL}")
print(f"Selection read from the {SELECTION_SOURCE}")

_why = SELECTED.completeness()
if _why is not None:
    raise RuntimeError(
        f"the selected validation backtest {SELECTED.hash} is incomplete: {_why}. "
        f"It was chosen from the {SELECTION_SOURCE}."
    )
if SELECTED.execution_tier != "canonical":
    raise RuntimeError(f"the selected validation backtest {SELECTED.hash} is not canonical")

selected_record = SELECTED.registry_record()
selected_prediction = study.results.open(selected_record["prediction_hash"])
validation_training = study.results.open(selected_prediction.registry_record()["training_hash"])
VALIDATION_SPEC = validation_training.spec()
_prediction_record = selected_prediction.registry_record()
CHECKPOINT_KIND = _prediction_record["checkpoint_kind"]
CHECKPOINT_VALUE = _prediction_record["checkpoint_value"]

_strategy = strategy_view(SELECTED.spec())
print(
    f"{FIELD_NAME} with {len(FIELD_HASHES)} members selects "
    f"{VALIDATION_SPEC['family']}/{VALIDATION_SPEC.get('config_name')} with "
    f"{(_strategy.get('allocation') or {}).get('method', 'equal_weight')} allocation, "
    f"top-{(_strategy.get('signal') or {}).get('top_k')}, "
    + (
        f"risk overlay {(_strategy.get('risk') or {}).get('name')}"
        if (_strategy.get("risk") or {}).get("name")
        else "no risk overlay"
    )
)

# %% [markdown]
# ### The artifacts the selected fit pinned must be the artifacts on disk
#
# A training identity pins its input artifacts by content hash. A candidate set
# frozen against a retired feature artifact still resolves, and the refit then
# fails several minutes later with a message about a specification mismatch
# rather than about a stale file. Checking here costs one hash per artifact and
# names the role that moved.


# %%
def _sha256(path: Path) -> str:
    """The content hash a training identity records for one input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


# Role-agnostic on purpose. Mapping a recorded role back to a path needs the resolver of
# whichever family produced the fit, and this notebook carries whichever family won. What is
# checkable without that mapping is the weaker, sufficient statement: every hash the fit pinned
# is the hash of some artifact currently on disk. A retired artifact fails it, which is the case
# this exists for.
_on_disk = {
    _sha256(_path): _path
    for _pattern in ("features/*.parquet", "labels/*.parquet", "config/setup.yaml")
    for _path in sorted(CASE_DIR.glob(_pattern))
    if _path.is_file()
}
_pinned = {
    entry["role"]: entry["sha256"]
    for entry in VALIDATION_SPEC["computation"].get("feature_artifacts", [])
}
_moved = [
    f"{_role} (pinned {_sha[:19]}...)" for _role, _sha in _pinned.items() if _sha not in _on_disk
]
if _moved:
    raise RuntimeError(
        f"the selected fit {validation_training.hash} pins artifacts that are not among the "
        f"{len(_on_disk)} on disk, so a refit would not be the configuration that was "
        "selected. Re-run the stage that produces them, then 16_risk_management to re-freeze "
        "the candidate set:\n  " + "\n  ".join(_moved)
    )
print(f"All {len(_pinned)} pinned input artifacts match the files on disk")

# %%
pl.DataFrame(
    {
        "field": [
            "candidate set",
            "candidate count",
            "selected validation backtest",
            "family",
            "configuration",
            "label",
            "validation training",
            "validation prediction",
            "checkpoint",
        ],
        "value": [
            FIELD_NAME,
            str(len(FIELD_HASHES)),
            SELECTED.hash,
            str(VALIDATION_SPEC["family"]),
            str(VALIDATION_SPEC.get("config_name") or ""),
            str(VALIDATION_SPEC["label"]),
            validation_training.hash,
            selected_prediction.hash,
            f"{CHECKPOINT_KIND}={CHECKPOINT_VALUE}",
        ],
    }
)

# %% [markdown]
# ## 2. Where the retraining interval comes from
#
# Nothing below is chosen here. `build_holdout_training_spec` takes the
# validation training specification and returns the same computation over a
# different interval, and each of that interval's four boundaries has a source:
#
# - **The holdout window** is `evaluation.holdout_start` and
#   `evaluation.holdout_end` in `config/setup.yaml`, read through the same
#   `canonical_window` a backtest is sliced to, so the derivation and the slice
#   cannot disagree.
# - **The training end** is one label buffer before the window opens, counted in
#   *observations* along the panel's own dates. Counted as calendar time, `10D`
#   is about seven sessions rather than ten, and the last training label's
#   outcome would resolve inside the holdout - short, silent, and in the
#   direction that looks fine.
# - **The training start** is the earliest start across the validation folds,
#   which is the longest history the configuration could have had, bounded below
#   by what the features actually reach. That bound is the part worth reading
#   twice, and the next cell prints it.
#
# The fold set the validation configuration was fitted over runs newest first,
# so its *first* entry carries the latest start. Taking it would hand the
# holdout the shortest history rather than the longest.

# %%
OBSERVATIONS = (
    pl.read_parquet(study.root / "labels" / f"{VALIDATION_SPEC['label']}.parquet")
    .get_column("timestamp")
    .unique()
    .sort()
    .to_list()
)
HOLDOUT_SPEC = build_holdout_training_spec(
    study, VALIDATION_SPEC, timeline=OBSERVATIONS, case_study=CASE_STUDY_ID
)
HOLDOUT_TRAINING_HASH = training_hash_from_spec(HOLDOUT_SPEC)
holdout_cv = HOLDOUT_SPEC["computation"]["cv"]
holdout_fold = holdout_cv["folds"][0]
cv_request = holdout_cv["request"]

validation_folds = VALIDATION_SPEC["computation"]["cv"]["folds"]
pl.DataFrame(
    {
        "boundary": ["train_start", "train_end", "holdout_start", "holdout_end"],
        "value": [
            holdout_fold["train_start"][:10],
            holdout_fold["train_end"][:10],
            holdout_fold["val_start"][:10],
            holdout_fold["val_end"][:10],
        ],
        "derived from": [
            "earliest validation fold start, bounded by the feature artifact",
            f"{cv_request['label_buffer']} "
            f"({cv_request['label_buffer_steps']} observations) before the window opens",
            "evaluation.holdout_start in config/setup.yaml",
            "evaluation.holdout_end in config/setup.yaml",
        ],
    }
)

# %% [markdown]
# ### Why the training start is not simply the earliest date available
#
# "The whole history available" is a claim about the features, not about the
# calendar. The model-based features this configuration reads are produced per
# fold over a rolling window, so before the fold the holdout is joined against,
# there is no feature history at all - the columns exist and are null. Fitting
# those dates would train an estimator on rows every validation fold saw fully
# populated, which is not the configuration that was ranked.
#
# The family answers where its features start and the derivation takes the later
# of the two boundaries. Where it applies, the clamp is recorded in the
# specification rather than applied silently, because it changes the interval
# and a reader has to be able to see that it did.

# %%
earliest_validation_start = min(str(fold["train_start"])[:10] for fold in validation_folds)
# Cast rather than compare: the label parquet dates a session as a `date`, the derived
# boundaries arrive as ISO strings, and `<` between the two raises rather than coercing.
observation_dates = pl.Series("timestamp", OBSERVATIONS).cast(pl.Date)
if "train_start_floor" in cv_request:
    _unclamped = pl.Series([earliest_validation_start]).str.to_date().item()
    _clamped = pl.Series([holdout_fold["train_start"][:10]]).str.to_date().item()
    _lost = int(((observation_dates >= _unclamped) & (observation_dates < _clamped)).sum())
    print(
        f"Calendar would allow {earliest_validation_start}; the feature artifact starts "
        f"{cv_request['train_start_floor'][:10]}, so the fit begins there and "
        f"{_lost} observation(s) the features do not cover are excluded"
    )
else:
    print(
        f"No feature floor applies: the fit begins at {earliest_validation_start}, the earliest "
        "start across the validation folds"
    )
print(
    f"Holdout training identity {HOLDOUT_TRAINING_HASH}, distinct from the validation fit "
    f"{validation_training.hash} because the interval differs"
)

# %% [markdown]
# ## 3. Refit and predict
#
# The refit publishes the selected checkpoint and no other. A model whose
# schedule writes ten checkpoints would otherwise leave ten holdout prediction
# sets for a later notebook to choose between, and choosing among them on the
# holdout is the thing this window exists to prevent.
#
# A re-run that derives the same specification resolves the registered result
# rather than fitting again. That is a cache, not a seal: change the selection
# above and the identity changes with it, and this fits. The fitted state is
# validated on both paths, because a reused prediction is state an earlier
# process left on disk and the registry row is the thing being checked, not the
# evidence for it.
#
# ### What a second holdout lineage would mean, and why nothing here prevents one
#
# A holdout window read once, against a selection made without seeing it, is out
# of sample. A window read again after the first result was observed, against a
# selection revised in the light of it, is not - it has become a validation set
# with extra steps, and no amount of care in this notebook repairs that.
#
# The earlier design took an authorization lock here and spent it, so that the
# second read was impossible. That failed in the direction nobody wants: it made
# a *wrong* holdout permanent too. When the fit turned out to be on a superseded
# training identity, the repair needed a retrain the lock forbade, and the case
# study was left publishing a result it knew to be stale.
#
# So this notebook is repeatable, and the discipline is placed where it can
# actually hold: the selection is frozen upstream in
# [`16_risk_management`](16_risk_management.ipynb) and read from that set, never
# chosen here, and the count below makes a second lineage visible rather than
# impossible. A reader who sees more than one holdout training identity for this
# label is looking at a window that has been read more than once, and should
# discount the out-of-sample claim accordingly. That is a judgement the page
# hands to the reader with the evidence, rather than one a lock makes for them.
#
# Which is why the superseded lineages stay in the registry. Deleting them would
# clear the warning without restoring anything: the window would still have been
# read, and the only record that it was would be gone. A registry holding three
# holdout fits for one label is telling the truth about what happened to that
# window, and that is worth more than a page that looks clean.

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    _lineages = db.execute(
        "SELECT DISTINCT p.training_hash FROM prediction_sets p "
        "JOIN training_runs t USING(training_hash) "
        "WHERE p.split = 'holdout' AND t.label = ?",
        (VALIDATION_SPEC["label"],),
    ).fetchall()
HOLDOUT_LINEAGES = {row[0] for row in _lineages} | {HOLDOUT_TRAINING_HASH}
if len(HOLDOUT_LINEAGES) == 1:
    print(f"One holdout training identity for {VALIDATION_SPEC['label']}: this one")
else:
    print(
        f"{len(HOLDOUT_LINEAGES)} holdout training identities exist for "
        f"{VALIDATION_SPEC['label']}: {', '.join(sorted(HOLDOUT_LINEAGES))}. The 2021 window has "
        "been fitted against more than one selection. Every result below is therefore a "
        "second-or-later read of a window that has already been seen, and none of them is a "
        "clean out-of-sample number. Report it as what it is, and do not delete the other "
        "lineages to make this message go away - they are the only record that the window was "
        "read more than once."
    )

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    existing = db.execute(
        "SELECT prediction_hash FROM prediction_sets WHERE training_hash = ? AND split = 'holdout'",
        (HOLDOUT_TRAINING_HASH,),
    ).fetchall()
if len(existing) > 1:
    raise RuntimeError(
        f"{HOLDOUT_TRAINING_HASH} carries {len(existing)} holdout prediction sets; a holdout fit "
        "publishes exactly the selected checkpoint, so this registry has rows from an older rule"
    )

# Reconstructed on both paths, not only when the refit has to run. A reused prediction is
# persisted state from an earlier process, and accepting it because a registry row exists would
# skip the one check worth keeping: that the weights on disk are what this specification
# produces. The runner reuses a complete identity rather than refitting, so this costs a
# resolution and not a training run.
request = reconstruct_locked_model_request(
    study,
    HOLDOUT_SPEC,
    checkpoint_kind=CHECKPOINT_KIND,
    checkpoint_value=CHECKPOINT_VALUE,
)
model_run = request.run()
if model_run.training.hash != HOLDOUT_TRAINING_HASH:
    raise RuntimeError(
        f"the refit produced training {model_run.training.hash}, not the derived identity "
        f"{HOLDOUT_TRAINING_HASH}"
    )
if len(model_run.predictions) != 1:
    raise RuntimeError(
        f"the refit published {len(model_run.predictions)} prediction sets; the holdout fit "
        "must publish exactly the selected checkpoint"
    )
HOLDOUT_PREDICTION = model_run.predictions[0]
FITTED_NOW = not existing
FITTED_STATE_DIGEST = validate_locked_model_run(request, model_run)
if not FITTED_STATE_DIGEST:
    raise RuntimeError("the holdout model produced no fitted-state digest")
if existing and existing[0][0] != HOLDOUT_PREDICTION.hash:
    raise RuntimeError(
        f"holdout prediction {existing[0][0]} was already registered for training "
        f"{HOLDOUT_TRAINING_HASH}, but this run resolved {HOLDOUT_PREDICTION.hash}"
    )
print(
    f"Holdout prediction {HOLDOUT_PREDICTION.hash} "
    + ("fitted and registered" if FITTED_NOW else "reused")
    + f"; fitted state {FITTED_STATE_DIGEST[:12]}"
)

holdout_record = HOLDOUT_PREDICTION.registry_record()
if holdout_record["training_hash"] != HOLDOUT_TRAINING_HASH:
    raise RuntimeError("the holdout prediction does not belong to the derived training identity")
if holdout_record["split"] != "holdout":
    raise RuntimeError(f"the refit published a {holdout_record['split']!r} prediction set")
if (holdout_record["checkpoint_kind"], holdout_record["checkpoint_value"]) != (
    CHECKPOINT_KIND,
    CHECKPOINT_VALUE,
):
    raise RuntimeError(
        "the holdout prediction is at a different checkpoint from the one selection was made on"
    )
if not HOLDOUT_PREDICTION.complete:
    raise RuntimeError("the holdout prediction set is registered but incomplete")

# %% [markdown]
# ## 4. What was produced
#
# Two things are worth reading before the backtest runs on them: whether the
# predictions cover the window they claim to, and whether the estimator's rank
# ordering survives out of sample at all. The second is not a strategy result -
# it says nothing about what the portfolio earns, and a configuration can hold
# its information coefficient and still lose money once concentration, turnover
# and costs apply. `19_holdout_backtest` is that measurement.

# %%
predictions = HOLDOUT_PREDICTION.load()
print(
    f"{predictions.height:,} rows, {predictions['symbol'].n_unique()} symbols, "
    f"{predictions['timestamp'].n_unique()} sessions from "
    f"{predictions['timestamp'].min()} to {predictions['timestamp'].max()}"
)

with sqlite3.connect(REGISTRY_DB) as db:
    coverage = db.execute(
        "SELECT n_expected, n_actual, n_missing, n_extra, n_null, status "
        "FROM prediction_coverage WHERE prediction_hash = ?",
        (HOLDOUT_PREDICTION.hash,),
    ).fetchone()
if coverage is None:
    raise RuntimeError("the holdout prediction registered no coverage record")
n_expected, n_actual, n_missing, n_extra, n_null, coverage_status = coverage
if coverage_status != "complete" or n_missing or n_extra:
    raise RuntimeError(
        f"holdout coverage is {coverage_status!r}: {n_missing} missing, {n_extra} extra "
        f"against {n_expected:,} expected keys"
    )
print(
    f"Coverage {coverage_status}: {n_actual:,} of {n_expected:,} expected keys, "
    f"{n_null:,} null prediction(s)"
)

# %% [markdown]
# The information coefficient is the rank correlation between the prediction and
# the realized label. Reading the holdout's beside the validation figure for the
# same configuration says whether the signal decayed, and by how much; it does
# not license a claim about either number on its own. The configuration was
# selected on validation backtest Sharpe and never on an information coefficient
# (`reference/CASE_STUDY_PIPELINE.md` section 5), so the validation row below is the
# selected configuration's IC rather than the quantity the selection ranked - and it
# is still optimistic, because the configuration it describes is the maximum of a
# search.

# %%
with sqlite3.connect(REGISTRY_DB) as db:
    ic_rows = db.execute(
        "SELECT prediction_hash, ic_mean, ic_t, n_folds FROM prediction_metrics "
        "WHERE prediction_hash IN (?, ?)",
        (HOLDOUT_PREDICTION.hash, selected_prediction.hash),
    ).fetchall()
ic_by_hash = {row[0]: row[1:] for row in ic_rows}
_holdout_ic = ic_by_hash.get(HOLDOUT_PREDICTION.hash)
_validation_ic = ic_by_hash.get(selected_prediction.hash)
if _holdout_ic is None:
    raise RuntimeError("the holdout prediction registered no information coefficient")

split_table = pl.DataFrame(
    {
        "split": ["validation (selected configuration)", "holdout (2021)"],
        "prediction": [selected_prediction.hash, HOLDOUT_PREDICTION.hash],
        "ic_mean": [
            None if _validation_ic is None else _validation_ic[0],
            _holdout_ic[0],
        ],
        "ic_t": [
            None if _validation_ic is None else _validation_ic[1],
            _holdout_ic[1],
        ],
        "folds": [
            None if _validation_ic is None else _validation_ic[2],
            _holdout_ic[2],
        ],
    }
)
split_table

# %% [markdown]
# ## Key takeaways
#
# 1. The configuration refitted here was selected by notebooks 14 through 17 on
#    validation alone. This notebook reads that selection back out of the
#    registry rather than being told what it is.
# 2. Every boundary of the retraining interval comes from a declaration: the
#    window from `config/setup.yaml`, the gap from the label's own buffer
#    counted in observations, the start from the validation fold set bounded by
#    what the features reach.
# 3. The holdout fit is a different training run from the validation fit, by
#    construction - a different interval is a different computation - so the two
#    share a configuration and not an identity.
# 4. Only the selected checkpoint is published. A holdout with ten checkpoints
#    to choose between is a validation set.
# 5. An information coefficient is not a strategy result. Whether this signal is
#    worth trading is settled by the next notebook, not this one.
#
# **Next:** [`19_holdout_backtest`](19_holdout_backtest.ipynb) runs the selected
# strategy specification - allocator, concentration and risk overlay unchanged -
# on these predictions.
