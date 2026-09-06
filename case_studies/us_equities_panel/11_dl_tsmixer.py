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
# # US equities panel: mixing along time and across features instead of recurring
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
# [`10_dl_lstm`](10_dl_lstm.ipynb) walked the window one session at a time. **TSMixer** does not
# walk it at all. The window is a matrix of sessions by features, and the model alternates two
# operations on it: a small network applied down each feature's column, mixing **across time**,
# and a small network applied across each session's row, mixing **across features**. Stacking a
# few of those blocks lets information travel along both axes without any recurrence.
#
# **The reason to try it is the shape of this particular panel.** Recurrence processes a window
# sequentially, so its cost grows with the window length and its gradient has to survive the whole
# walk; mixing touches the whole window at once.
#
# The other two models do combine features - the LSTM's gates read the whole feature vector at
# every step, and NLinear's final layer is a linear combination across features - but each does it
# in one place and one way. TSMixer makes feature mixing an explicit, nonlinear operation that
# happens once per block and alternates with mixing along time, so a combination of features can
# itself be mixed across time and then recombined. On a feature set this dense in the
# cross-section, that repetition is what is being tested.
#
# The configuration here stacks two blocks at a hidden dimension of 32, smaller than the recurrent
# model's state for the same reason: what bounds the capacity worth fitting is the number of
# eligible windows.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Describe the two mixing operations, say which axis of the window each acts on, and say what
#   information can travel where after two blocks.
# - Say how TSMixer's feature mixing differs from the way the other two models combine features,
#   given that both of them also do.
# - Explain why an architecture without recurrence can be preferable on long windows, in terms of
#   cost and of what has to survive training.
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
"""Generate TSMixer validation predictions through the shared research interface."""

import os
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import open_study, plan_models
from utils.modeling import load_configs
from utils.paths import get_case_study_dir

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
# label, and this notebook takes the ones whose architecture is `tsmixer`. Each name resolves to a
# preset holding the full parameter set - here a 60-session lookback, 100 epochs, a checkpoint
# every 5, and a dropout of 0.1.
#
# What each setting a run may pass decides:
#
# - **`CONFIG_NAMES`** empty fits every declared `tsmixer` configuration. A named subset fits only
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
    if config.get("params", {}).get("architecture") == "tsmixer"
]
published_names = [str(config["config_name"]) for config in published_configs]
selected_names = list(CONFIG_NAMES) if CONFIG_NAMES else published_names
unknown_names = sorted(set(selected_names) - set(published_names))
unknown_overrides = sorted(set(CONFIG_OVERRIDES) - set(selected_names))
if not published_names:
    raise ValueError("The published training menu has no TSMixer configuration")
if unknown_names:
    raise ValueError(f"Unknown TSMixer configurations: {unknown_names}")
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

# Both tiers resolve the study through `open_study`. It reads the labels and features in place and
# redirects only writes, so a preview run scores the same inputs a canonical one does and cannot
# publish over it.
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
# Each selected TSMixer configuration becomes one request with the declared sequence reductions.

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
        name="us-equities-tsmixer-checkpoints-v1",
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

# %% [markdown]
# A prediction set can be registered complete and still have scored no dates. Cross-sectional
# information coefficient needs a minimum number of names quoted on a date before the ranking on
# that date means anything, so a universe whose stocks do not overlap in time yields no scorable
# dates and a null IC at every checkpoint while every coverage check passes. That is a run which
# reports nothing and looks successful, so it is asserted on rather than left to be noticed.

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
# A canonical default CUDA run freezes every returned TSMixer prediction row under a stable name.
# The same bounded family set supplies raw diagnostics because this notebook has one published
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
        name=f"us-equities-{label_name}-tsmixer-v1",
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
# **Three architectures, one window, one set of folds.** NLinear applies a single linear map, the
# LSTM walks the window carrying state, and this mixes along both axes. Any difference between
# them is the architecture, because nothing else varies - which is what makes the comparison worth
# having, and is also why none of the three chooses anything here.
#
# **What is being tested is repeated, explicit feature mixing, not feature mixing as such.** Both
# of the other models combine features once. If alternating that with temporal mixing block after
# block is worth anything on this panel, this is where it shows, and the place to look is against
# the LSTM rather than against NLinear.
#
# **A checkpoint is part of a configuration.** Twenty per configuration, each registered
# separately, for the reason `09_dl_nlinear` gives.
#
# **Known limitations.** Windows are built only where sessions are unbroken. Everything measured
# here is ranking accuracy on validation folds read many times over, and the comparison between
# the three families is settled on validation backtest Sharpe in
# [`16_backtest`](16_backtest.ipynb), not here.
#
# **Next**: [`12_dl_weekly`](12_dl_weekly.ipynb) keeps these models and changes the sampling grid.
