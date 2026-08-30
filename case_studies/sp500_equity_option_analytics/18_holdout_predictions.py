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

import json
import sqlite3
import warnings

import polars as pl

warnings.filterwarnings("ignore")

from case_studies.research import open_study
from case_studies.research.holdout import build_holdout_training_spec
from case_studies.research.models import reconstruct_locked_model_request
from case_studies.utils.backtest_loaders import get_backtest_config
from case_studies.utils.backtest_presets import strategy_view
from case_studies.utils.notebook_contracts import prediction_members_in_force
from case_studies.utils.registry import model_source, resolve_best_backtest_runs
from case_studies.utils.registry.specs import training_hash_from_spec
from case_studies.utils.sweep_config import get_allocators
from utils.paths import get_case_study_dir

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

# %%
CASE_DIR = get_case_study_dir(CASE_STUDY_ID)
REGISTRY_DB = CASE_DIR / "run_log" / "registry.db"
bt_config = get_backtest_config(CASE_STUDY_ID)
HOLDOUT_LABEL = LABEL or bt_config.primary_label
print(f"Case study: {CASE_STUDY_ID}; label: {HOLDOUT_LABEL}")

# %% [markdown]
# This notebook writes to the registry, so it opens the study rather than
# reading one. `open_study` activates the tier, which is what lets a fit publish
# a training run and a prediction set under it; the read-only `Study.at` the
# preceding four notebooks use cannot.

# %%
study = open_study(
    CASE_STUDY_ID,
    execution_tier=EXECUTION_TIER,
    workspace=WORKSPACE or None,
    entry_point="18_holdout_predictions",
)
CURRENT_MEMBERS, _population_notes = prediction_members_in_force(study)
for _note in _population_notes:
    print(_note)

# %% [markdown]
# ## 1. Which configuration the pipeline arrived at
#
# The rule is the one [`17_costs`](17_costs.ipynb) applies, applied again rather
# than carried across in a variable: rank the risk-overlay rows together with
# the allocation rows they were overlaid on, and take the highest validation
# Sharpe. Reading it from the registry a second time is what makes the two
# notebooks agree by construction. If they ever disagreed, the assertion below
# would say so rather than this notebook silently refitting something else.
#
# The union matters for the same reason it does there. The risk stage registers
# a row per named control and none for the un-overlaid strategy, so a pool drawn
# from `stage="risk_overlay"` alone would force an overlay onto the carrier even
# where every control hurt it.

# %%
active_allocators = {item["method"] for item in get_allocators(CASE_STUDY_ID)}
candidate_pool = pl.concat(
    [
        resolve_best_backtest_runs(
            CASE_STUDY_ID,
            HOLDOUT_LABEL,
            split="validation",
            stage=stage,
            top_n=9999,
            prediction_hashes=CURRENT_MEMBERS,
        )
        for stage in ("risk_overlay", "allocation")
    ],
    how="diagonal_relaxed",
).unique("backtest_hash")
if candidate_pool.is_empty():
    raise RuntimeError(
        "No full-coverage risk-overlay or allocation candidates: notebooks 15 and 16 have not "
        "run against the populations in force, so there is no selection to carry to the holdout"
    )

candidate_hashes = candidate_pool["prediction_hash"].unique().to_list()
with sqlite3.connect(REGISTRY_DB) as db:
    source_rows = db.execute(
        f"""
        SELECT p.prediction_hash, t.family, t.config_name
        FROM prediction_sets p
        JOIN training_runs t ON p.training_hash = t.training_hash
        WHERE p.prediction_hash IN ({",".join("?" for _ in candidate_hashes)})
        """,
        candidate_hashes,
    ).fetchall()
source_by_hash = {
    prediction_hash: model_source(family, config_name)
    for prediction_hash, family, config_name in source_rows
}

eligible_rows = []
for row in candidate_pool.iter_rows(named=True):
    strategy = strategy_view(json.loads(row["spec_json"]))
    allocator = strategy.get("allocation", {}).get("method", "equal_weight")
    if allocator == "equal_weight" or allocator in active_allocators:
        eligible_rows.append(
            {
                **row,
                "source": source_by_hash[row["prediction_hash"]],
                "allocator": allocator,
                "top_k": strategy.get("signal", {}).get("top_k"),
                "risk": (strategy.get("risk") or {}).get("name"),
            }
        )
if not eligible_rows:
    raise RuntimeError("No eligible strategy lineage in the risk-overlay or allocation pools")

CARRIER = pl.DataFrame(eligible_rows).sort("sharpe", descending=True).row(0, named=True)
print(
    f"Selected {CARRIER['source']} with {CARRIER['allocator']} allocation, "
    f"top-{CARRIER['top_k']}, "
    + (f"risk overlay {CARRIER['risk']}" if CARRIER["risk"] else "no risk overlay")
    + f", validation Sharpe {CARRIER['sharpe']:.3f}"
)

# %% [markdown]
# The strategy specification travels to [`19_holdout_backtest`](19_holdout_backtest.ipynb)
# through the registry rather than through this notebook. What 18 needs from the
# carrier is narrower: the model that produced its signal, and which checkpoint
# of that model the selection was made on.

# %%
selected_prediction = study.results.open(CARRIER["prediction_hash"])
selected_record = selected_prediction.registry_record()
validation_training = study.results.open(selected_record["training_hash"])
VALIDATION_SPEC = validation_training.spec()
CHECKPOINT_KIND = selected_record["checkpoint_kind"]
CHECKPOINT_VALUE = selected_record["checkpoint_value"]

pl.DataFrame(
    {
        "field": [
            "family",
            "configuration",
            "label",
            "validation training",
            "validation prediction",
            "checkpoint",
        ],
        "value": [
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
# A re-run that derives the same specification finds the prediction already in
# the registry and reads it rather than fitting again. That is a cache, not a
# seal: change the selection above and the identity changes with it, and this
# fits.

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

request = reconstruct_locked_model_request(
    study,
    HOLDOUT_SPEC,
    checkpoint_kind=CHECKPOINT_KIND,
    checkpoint_value=CHECKPOINT_VALUE,
)
if existing:
    HOLDOUT_PREDICTION = study.results.open(existing[0][0])
    FITTED_NOW = False
    print(f"Holdout prediction {HOLDOUT_PREDICTION.hash} already registered; read, not refitted")
else:
    model_run = request.run()
    if len(model_run.predictions) != 1:
        raise RuntimeError(
            f"the refit published {len(model_run.predictions)} prediction sets; the holdout fit "
            "must publish exactly the selected checkpoint"
        )
    HOLDOUT_PREDICTION = model_run.predictions[0]
    FITTED_NOW = True
    print(f"Holdout prediction {HOLDOUT_PREDICTION.hash} fitted and registered")

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
# not license a claim about either number on its own, because the validation
# figure is the one the configuration was selected on and is optimistic by
# construction.

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
        "split": ["validation (selected on)", "holdout (2021)"],
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
