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
# # US equities panel: a model that carries state across the window
#
# [`06_linear`](06_linear.ipynb), [`07_gbm`](07_gbm.ipynb) and
# [`08_tabular_dl`](08_tabular_dl.ipynb) all read the same flat table: one row per stock per
# session, one column per feature, and nothing in the representation saying the rows are ordered
# in time. A model on that table sees the past only through columns somebody computed in advance -
# a 21-session momentum, a rolling volatility. It never sees the sequence itself.
#
# A **sequence model** is handed the sequence. Each training example here is a **window**: the 60
# most recent sessions of one stock's features, in order, as a matrix of sessions by features -
# about three months. The model reads the window and emits one number, the predicted return.
#
# **A window has to be 60 consecutive sessions of the same stock, and on this panel that binds.**
# A stock that lists part-way through a fold, halts, or delists leaves a gap, and a window
# spanning a gap would treat the two sides as consecutive sessions and read the jump across it as
# one day's move. Windows are therefore built only where the sessions are unbroken, which is why
# the number of training examples is far smaller than the number of rows and differs between
# folds.
#
# [`09_dl_nlinear`](09_dl_nlinear.ipynb) read each window as one fixed-length input and applied a
# single linear map to it. An **LSTM** reads the window one session at a time instead, carrying a
# **hidden state** forward: a vector summarising everything it has seen so far, updated at each
# step and used to predict at the end.
#
# **What the architecture adds is a decision about what to keep.** At each step three learned
# gates decide how much of the new session to admit, how much of the existing state to discard,
# and how much of the state to expose. That is what lets a recurrent model carry something from
# the start of a window to its end without it being diluted at every step, which a plain recurrent
# network cannot do.
#
# The configuration here stacks two such layers with a hidden state of 64. Two layers rather than
# one so the second reads a sequence of summaries rather than a sequence of observations; 64
# rather than more because the training examples are windows, not rows, and the window count is
# what bounds how much capacity can be estimated from them.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Describe what a hidden state is and what the gates of an LSTM decide at each step.
# - Say what a recurrent model can represent that a single linear map over the same window cannot.
# - Explain why the number of eligible windows, rather than the number of panel rows, is what
#   bounds the capacity worth fitting.
# - Read the epoch schedule out of a declared configuration and say how many scoreable models the
#   run publishes for it.
#
# **A neural fit has a meaningful state at every epoch**, in the way a boosted model has one at
# every iteration and a linear fit does not. An **epoch** is one pass over the training windows.
# Each configuration here trains for 100 of them and saves its weights every 5, so it publishes
# twenty scoreable models rather than one, each registered with its own identity. The count that
# matters downstream is configurations times checkpoints.
#
# **Book reference**: Chapter 13. Chapter 6, Section 6.7 (Search accounting and run logging)
# introduces the run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices, and
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds.
#
# **What it writes**: one training run per configuration and one complete validation prediction set
# per configuration and epoch checkpoint, in `run_log/registry.db` and under `run_log/training/`
# and `run_log/predictions/`, grouped under a named population.
# [`15_model_analysis`](15_model_analysis.ipynb) compares that population against the other
# families and [`16_backtest`](16_backtest.ipynb) backtests every member and selects on validation
# backtest Sharpe. **Selection happens there, not here.**

# %%
"""Generate LSTM validation predictions through the shared research interface."""

import os
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import Study, open_study, plan_models
from utils.modeling import load_configs
from utils.paths import REPO_ROOT, get_case_study_dir

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = ""
CONFIG_NAMES = []
COMMON_OVERRIDES = {}
CONFIG_OVERRIDES = {}
DEVICE = "cuda"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
FOLD_IDS = []
MAX_TRAIN_SEQUENCES = 0
PREVIEW_N_EPOCHS = 0

# %% [markdown]
# ## 1. Which configurations, and on which label
#
# The menu at `config/training/{label}.yaml` lists the sequence configurations declared for a
# label, and this notebook takes the ones whose architecture is `lstm`. Each name resolves to a
# preset holding the full parameter set - here a 60-session lookback, 100 epochs, a checkpoint
# every 5, and a dropout of 0.1.
#
# What each setting a run may pass decides:
#
# - **`CONFIG_NAMES`** empty fits every declared `lstm` configuration. A named subset fits only
#   those, which is what to do first: at panel scale a full run is hours, and the point of a first
#   pass is to find out whether the plumbing works.
# - **`COMMON_OVERRIDES`** changes a parameter for every selected configuration, and
#   **`CONFIG_OVERRIDES`** changes one named configuration and takes precedence. An override moves
#   a training identity, so an overridden run registers beside the published one rather than
#   replacing it.
# - **`EXECUTION_TIER`** is `canonical` or `preview`. A canonical run fits every eligible window on
#   every fold at the published epoch schedule. A preview run has to declare at least one
#   reduction and carries it in the identity, so its results can never be compared against
#   canonical ones or reach a holdout decision.
# - **`PREVIEW_N_EPOCHS`** shortens the schedule for a preview. It is part of the identity rather
#   than a runtime detail, because a model trained for fewer epochs is a different model rather
#   than the same one measured sooner.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
label = PRIMARY_LABEL or setup["labels"]["primary"]

all_sequence_configs = load_configs(CASE_STUDY_ID, label, family="deep_learning")
published_configs = [
    config
    for config in all_sequence_configs
    if config.get("params", {}).get("architecture") == "lstm"
]
published_names = [str(config["config_name"]) for config in published_configs]
selected_names = list(CONFIG_NAMES) if CONFIG_NAMES else published_names
unknown_names = sorted(set(selected_names) - set(published_names))
unknown_overrides = sorted(set(CONFIG_OVERRIDES) - set(selected_names))
if not published_names:
    raise ValueError("The published training menu has no LSTM configuration")
if unknown_names:
    raise ValueError(f"Unknown LSTM configurations: {unknown_names}")
if unknown_overrides:
    raise ValueError(f"Overrides supplied for unselected configurations: {unknown_overrides}")
if len(selected_names) != len(set(selected_names)):
    raise ValueError("CONFIG_NAMES contains duplicates")

menu = pl.DataFrame(
    {
        "config_name": [config["config_name"] for config in published_configs],
        "architecture": [config["params"]["architecture"] for config in published_configs],
        "published_params": [str(config.get("params") or {}) for config in published_configs],
        "n_epochs": [config.get("n_epochs") for config in published_configs],
        "checkpoint_interval": [config.get("checkpoint_interval") for config in published_configs],
        "selected": [config["config_name"] in selected_names for config in published_configs],
    }
)
menu

# %%
preview_reductions = {}
if MAX_SYMBOLS:
    preview_reductions["max_symbols"] = int(MAX_SYMBOLS)
if FOLD_IDS:
    preview_reductions["folds"] = [int(fold) for fold in FOLD_IDS]
if MAX_TRAIN_SEQUENCES:
    preview_reductions["max_train_sequences"] = int(MAX_TRAIN_SEQUENCES)

# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if preview_reductions or PREVIEW_N_EPOCHS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if not preview_reductions:
        raise ValueError("Preview execution requires a data or fold reduction")
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError("EXECUTION_TIER must be 'canonical' or 'preview'")

# %% [markdown]
# ## 2. Binding the declarations to the data
#
# Each selected LSTM configuration becomes one request with the declared sequence reductions.

# %%
requests = []
for config_name in selected_names:
    overrides = {
        "device": DEVICE,
        **COMMON_OVERRIDES,
        **dict(CONFIG_OVERRIDES.get(config_name, {})),
    }
    if PREVIEW_N_EPOCHS:
        overrides["n_epochs"] = int(PREVIEW_N_EPOCHS)
    requests.append(
        study.model(
            family="deep_learning",
            label=label,
            config_name=config_name,
            overrides=overrides,
            execution_tier=EXECUTION_TIER,
            preview_reductions=preview_reductions,
        )
    )
requests = tuple(requests)

request_table = pl.DataFrame(
    {
        "family": [request.family for request in requests],
        "label": [request.label for request in requests],
        "config_name": [request.config_name for request in requests],
        "overrides": [str(request.overrides) for request in requests],
        "execution_tier": [request.execution_tier.value for request in requests],
        "preview_reductions": [str(request.preview_reductions) for request in requests],
    }
)
request_table

# %% [markdown]
# ## 3. Planning, then fitting
#
# The planner resolves every training and epoch-checkpoint identity before fitting and writes the
# canonical checkpoint population first. Execution builds only sequences that follow the declared
# observation calendar and excludes
# windows that cross missing expected periods. Each epoch checkpoint stores the fitted
# preprocessing state, model weights, predictions, and exact eligible-key evidence. A retry reuses
# valid candidate-fold checkpoints and recomputes incomplete work.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-lstm-checkpoints-v1",
    )

planned_population = pl.DataFrame(
    {
        "family": [member.family for member in plan.members],
        "config_name": [member.config_name for member in plan.members],
        "checkpoint_kind": [member.checkpoint_kind for member in plan.members],
        "checkpoint_value": [member.checkpoint_value for member in plan.members],
        "training_hash": [member.training_hash for member in plan.members],
        "prediction_hash": [member.prediction_hash for member in plan.members],
    }
)
planned_population

# %%
execution = plan.run()

# %% [markdown]
# ## 4. What was actually fitted
#
# These rows expose the feature, fold, sequence, runtime, model, and checkpoint settings used by
# the runner, including defaults that were not repeated in the notebook parameters.

# %%
resolved_rows = []
for run in execution.runs:
    spec = run.training.spec()
    computation = spec["computation"]
    model = computation["model"]
    resolved_rows.append(
        {
            "config_name": spec["config_name"],
            "architecture": model["params"]["architecture"],
            "features": len(computation["feature_names"]),
            "folds": computation["expected_prediction_keys"]["n_folds"],
            "eligible_rows": computation["expected_prediction_keys"]["n_rows"],
            "lookback": model["params"]["lookback"],
            "device": computation["numerics"]["device"],
            "n_epochs": model["params"]["n_epochs"],
            "checkpoints": [item["value"] for item in computation["checkpoint_schedule"]],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("config_name")
resolved_table

# %% [markdown]
# ## 5. What came out
#
# Each catalog row is one complete validation prediction set for one training identity and epoch.
# Downstream notebooks filter these rows with Polars and pass the selected table directly to
# backtesting. The hashes remain visible for exact provenance and artifact reads.

# %% tags=["results"]
catalog_columns = [
    "family",
    "config_name",
    "label",
    "split",
    "checkpoint_kind",
    "checkpoint_value",
    "execution_tier",
    "complete",
    "ic_mean",
    "training_hash",
    "prediction_hash",
]
catalog_rows = execution.catalog_rows.select(
    column for column in catalog_columns if column in execution.catalog_rows.columns
).sort("config_name", "checkpoint_value", "prediction_hash")
catalog_rows

# %%
# %% [markdown]
# A prediction set can be registered complete and still have scored no dates: cross-sectional IC
# needs `min_obs` names on a date, so a reduced universe whose symbols do not overlap in time
# yields `ic_n_days = 0` and a null IC for every checkpoint while coverage stays complete. That is
# a run that reports nothing and passes. `11_dl_tsmixer` carried a pinned symbol whitelist to avoid
# it, which repaired one panel and left the condition unchecked; this asserts it instead.

# %% tags=["results"]
scored = execution.catalog_rows.select("config_name", "checkpoint_value", "ic_mean", "ic_n_days")
unscored = scored.filter(pl.col("ic_n_days").is_null() | (pl.col("ic_n_days") <= 0))
if not unscored.is_empty():
    raise RuntimeError(f"prediction sets scored no dates: {unscored.to_dicts()}")
scored

# %%
coverage_rows = []
for run in execution.runs:
    if not run.training.complete:
        raise RuntimeError(f"Incomplete training result: {run.training.hash}")
    for prediction in run.predictions:
        record = prediction.registry_record()
        coverage = prediction.coverage()
        if not prediction.complete or coverage is None or coverage["status"] != "complete":
            raise RuntimeError(f"Incomplete prediction result: {prediction.hash}")
        coverage_rows.append(
            {
                "config_name": run.training.spec()["config_name"],
                "checkpoint": record["checkpoint_value"],
                "training_hash": run.training.hash,
                "prediction_hash": prediction.hash,
                "coverage_status": coverage["status"],
                "expected_rows": coverage["n_expected"],
                "actual_rows": coverage["n_actual"],
                "training_artifacts": len(run.training.artifacts()),
                "prediction_artifacts": len(prediction.artifacts()),
            }
        )

coverage_table = pl.DataFrame(coverage_rows).sort("config_name", "checkpoint")
if official_population is not None:
    official_population.require_complete()
coverage_table

# %%
execution_diagnostics = pl.DataFrame(execution.diagnostics)
execution_diagnostics

# %% [markdown]
# ## 6. Naming the set the later notebooks open
#
# A canonical default CUDA run freezes every returned LSTM prediction row under a stable name. The
# same bounded family set supplies raw diagnostics because this notebook has one published
# configuration. Preview and customized canonical requests do not publish an official set.

# %% tags=["results"]
set_rows = []
is_published_population = (
    EXECUTION_TIER == "canonical"
    and selected_names == published_names
    and not COMMON_OVERRIDES
    and not CONFIG_OVERRIDES
    and DEVICE == "cuda"
)
if is_published_population:
    label_name = label.replace("_", "-")
    full_set = study.predictions.freeze(
        execution.catalog_rows,
        name=f"us-equities-{label_name}-lstm-v1",
    )
    set_rows = [
        {
            "role": "backtest and diagnostic population",
            "set_name": full_set.name,
            "members": len(full_set.members),
        }
    ]
compatible_sets = pl.DataFrame(
    set_rows,
    schema={"role": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `15_model_analysis.py` reopens the named set for descriptive analysis. `16_backtest.py` passes
# every catalog row directly to the shared backtest runner. Model metrics do not choose a
# configuration or checkpoint.

# %% [markdown]
# ## What to notice
#
# **Read this against [`09_dl_nlinear`](09_dl_nlinear.ipynb) rather than on its own.** The two read
# identical windows over identical folds, so the difference between them is what recurrence and
# the gating bought on this panel and nothing else.
#
# **More capacity is not free at this window count.** The training examples are windows of
# consecutive sessions, not panel rows, and a two-layer recurrent network has far more parameters
# to estimate from them than a single linear map does. Where the learning curve turns over is the
# evidence about whether that capacity was supportable here.
#
# **A checkpoint is part of a configuration.** Twenty per configuration, each registered
# separately, for the reason `09_dl_nlinear` gives.
#
# **Known limitations.** Windows are built only where sessions are unbroken, so the training set
# is not a uniform sample of the panel. Everything here is ranking accuracy on validation folds
# read many times over, and none of it is a statement about tradability.
#
# **Next**: [`11_dl_tsmixer`](11_dl_tsmixer.ipynb) drops recurrence entirely and mixes along the
# two axes of the window in turn.
