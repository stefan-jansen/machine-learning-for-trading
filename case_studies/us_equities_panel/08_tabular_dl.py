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
# # US equities panel: a neural network on the same flat table, and what an ensemble of them costs
#
# [`06_linear`](06_linear.ipynb) and [`07_gbm`](07_gbm.ipynb) read the same design matrix: one row
# per stock per session, one column per feature, nothing in the representation saying the rows are
# ordered in time. They differ in what they can express. A penalized linear model gives each
# feature one coefficient, and where several columns carry nearly the same information it can
# spread weight across all of them. A tree ensemble can express an interaction - a condition on one
# feature evaluated inside a region another feature defines - but it gets there by picking one
# column at each split, and among near-duplicate columns which one gets picked is close to
# arbitrary.
#
# A neural network on that same table is a third answer. Its first layer is a weighted sum of every
# feature, so like the linear model it never has to choose between correlated columns; the
# nonlinearity after it lets those sums combine into interactions the linear model cannot write
# down. That is the reason to fit one here, rather than a general preference for neural networks:
# the two properties that pulled against each other in the previous two notebooks are not obviously
# in conflict in this architecture.
#
# **TabM is an ensemble, and the ensemble is the point.** Averaging several independently
# initialized networks is a standard way to make a neural fit less erratic, and the ordinary cost
# is training that many networks. TabM trains most of one. A two-layer network - the **backbone** -
# is shared by every member. Each member owns two small things of its own: a vector carrying one
# number per hidden unit, which scales the backbone's output element by element, and its own final
# linear layer turning that scaled output into a prediction. The members' predictions are averaged.
# What differs between members is therefore one vector and one output layer each, set against a
# backbone as wide as the hidden size - which is why adding members grows the model far more slowly
# than training that many separate networks would.
#
# **The three declared configurations move both dials at once.** `tabm_s`, `tabm_m` and `tabm_l`
# pair a hidden width of 64, 128 and 256 with 4, 8 and 16 members. So this grid is a capacity
# ladder rather than an experiment separating width from ensemble size: a difference between two
# rungs cannot be attributed to either dial on its own.
#
# **A neural fit has a meaningful state at every epoch**, the way a boosted model has one at every
# iteration and a linear fit does not. An **epoch** is one pass over the training rows. Each
# configuration here trains for 200 of them and saves its weights every 25, so it produces eight
# scoreable models rather than one, and each is registered with its own identity. The count that
# matters downstream is configurations times checkpoints - three times eight - not three.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Describe what a weight-sharing ensemble holds in common between its members and what it keeps
#   separate, and say why *k* members cost far less than *k* networks.
# - Read the epoch schedule out of a declared configuration and say how many scoreable models the
#   run publishes for it.
# - Say why a grid that moves width and member count together cannot attribute a difference to
#   either one, and what a grid that separated them would have to hold fixed.
# - Recognise that a model can predict nearly the same value for every stock on a date, why that
#   date then contributes nothing to a ranking measure, and why a run can be registered complete
#   and still have scored no dates at all.
# - Locate where a configuration and a stopping point are actually chosen in this case study, and
#   say why that is not here.
#
# **Book reference**: Chapter 12, Section 12.3 (Deep Learning Alternatives). Chapter 6, Section 6.7
# (Search accounting and run logging) introduces the run log this notebook writes to.
#
# **Prerequisites**: [`03_financial_features`](03_financial_features.ipynb) and
# [`04_model_based_features`](04_model_based_features.ipynb) have written the feature matrices,
# [`05_evaluation`](05_evaluation.ipynb) has established the walk-forward folds, and
# [`06_linear`](06_linear.ipynb) and [`07_gbm`](07_gbm.ipynb) fitted the two populations this one
# sits beside.
#
# **What it writes**: one training run per configuration and one complete validation prediction set
# per configuration and epoch checkpoint, in `run_log/registry.db` and under `run_log/training/`
# and `run_log/predictions/`, grouped under a named population.
# [`15_model_analysis`](15_model_analysis.ipynb) compares that population against the other
# families and [`16_backtest`](16_backtest.ipynb) backtests every member and selects on validation
# backtest Sharpe. **Selection happens there, not here.**

# %%
"""Generate TabM validation predictions through the shared research interface."""

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
DIAGNOSTIC_CONFIG_NAMES = ["tabm_s"]
DEVICE = "cuda"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
MAX_FOLDS = 0
PREVIEW_N_EPOCHS = 0
PREVIEW_CHECKPOINT_INTERVAL = 0

# %% [markdown]
# ## 1. Which configurations, and on which label
#
# The menu at `config/training/{label}.yaml` lists the TabM configurations declared for a label,
# and each name resolves to a preset in `case_studies/config/tabm/` holding the full parameter set.
# The table below shows the whole menu with a column marking which entries this run selected.
#
# What each setting a run may pass decides:
#
# - **`CONFIG_NAMES`** empty fits the whole declared menu. A list such as `['tabm_s']` fits that
#   subset, which is what to do first: at panel scale the full menu is hours, and the point of a
#   first pass is to find out whether the plumbing works.
# - **`COMMON_OVERRIDES`** changes a model or runner parameter for every selected configuration and
#   **`CONFIG_OVERRIDES`** changes one named configuration, taking precedence. An override moves a
#   training identity, so an overridden run registers beside the published one rather than
#   replacing it.
# - **`DIAGNOSTIC_CONFIG_NAMES`** names the small subset [`15_model_analysis`](15_model_analysis.ipynb)
#   compares predictions across. It is bounded on purpose: that comparison holds every member's
#   prediction frame in memory at once and correlates them pairwise, so its cost grows with the
#   square of the membership.
# - **`EXECUTION_TIER`** is `canonical` or `preview`. A canonical run fits the whole panel on every
#   fold at the published epoch schedule. A preview run has to declare at least one reduction, and
#   its results carry that reduction in their identity so they can never be compared against
#   canonical ones or reach a holdout decision.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
label = PRIMARY_LABEL or setup["labels"]["primary"]

published_configs = load_configs(CASE_STUDY_ID, label, family="tabular_dl")
published_names = [str(config["config_name"]) for config in published_configs]
selected_names = list(CONFIG_NAMES) if CONFIG_NAMES else published_names
unknown_names = sorted(set(selected_names) - set(published_names))
unknown_overrides = sorted(set(CONFIG_OVERRIDES) - set(selected_names))
if unknown_names:
    raise ValueError(f"Unknown TabM configurations: {unknown_names}")
if unknown_overrides:
    raise ValueError(f"Overrides supplied for unselected configurations: {unknown_overrides}")
if len(selected_names) != len(set(selected_names)):
    raise ValueError("CONFIG_NAMES contains duplicates")
unknown_diagnostics = sorted(set(DIAGNOSTIC_CONFIG_NAMES) - set(selected_names))
if not DIAGNOSTIC_CONFIG_NAMES or unknown_diagnostics:
    raise ValueError(f"Invalid diagnostic configurations: {unknown_diagnostics}")

menu = pl.DataFrame(
    {
        "config_name": [config["config_name"] for config in published_configs],
        "library": [config["library"] for config in published_configs],
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
if MAX_FOLDS:
    preview_reductions["folds"] = list(range(int(MAX_FOLDS)))
if PREVIEW_N_EPOCHS:
    preview_reductions["n_epochs"] = int(PREVIEW_N_EPOCHS)
if PREVIEW_CHECKPOINT_INTERVAL:
    preview_reductions["checkpoint_interval"] = int(PREVIEW_CHECKPOINT_INTERVAL)

# Both tiers resolve the study through `open_study`. It reads the labels and features in place and
# redirects only writes, so a preview run scores the same inputs a canonical one does and cannot
# publish over it.
if EXECUTION_TIER == "canonical":
    if preview_reductions:
        raise ValueError("Canonical execution cannot declare preview reductions")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if not preview_reductions:
        raise ValueError("Preview execution requires at least one declared reduction")
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
# A **request** is one configuration bound to one label and one execution tier, with its overrides
# resolved. It is what the planner reads, and it holds no data - so the table below can be
# inspected before anything is loaded.

# %%
requests = []
for config_name in selected_names:
    overrides = {
        "device": DEVICE,
        **COMMON_OVERRIDES,
        **dict(CONFIG_OVERRIDES.get(config_name, {})),
    }
    requests.append(
        study.model(
            family="tabular_dl",
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
# **Planning works out every identity before any fitting starts.** Each configuration-and-epoch
# pair gets its training and prediction hash from the declarations and the fold boundaries alone,
# and the list of them is written down as a **population**: a named, immutable membership that the
# run then has to fill completely. Declaring the membership first is what makes the downstream
# comparison well defined, because a population that came out short would otherwise look like a
# smaller experiment rather than a failed one.
#
# The plan walks folds on the outside and configurations on the inside, so one prepared fold is
# resident at a time however many configurations were declared - which on a three-thousand-name
# panel across sixteen ten-year training windows is the difference between fitting and running out
# of memory. Every declared epoch saves its preprocessing, its weights and its predictions before
# the fold is released, so an interrupted run resumes from the last completed fold instead of
# starting over.

# %%
plan = plan_models(study, requests=requests)
official_population = None
if EXECUTION_TIER == "canonical":
    official_population = plan.create_population(
        name="us-equities-tabular-dl-checkpoints-v1",
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
# A configuration names a preset, and a preset leaves most settings to a default. The table below
# is the fully resolved specification the runner used - every feature count, fold count, device,
# epoch schedule and batch size, including the defaults nothing above restated. This is the record
# a reader checks a result against, and it is what the training hash is computed from.

# %%
resolved_rows = []
for run in execution.runs:
    spec = run.training.spec()
    computation = spec["computation"]
    model = computation["model"]
    resolved_rows.append(
        {
            "config_name": spec["config_name"],
            "task": computation["task"]["type"],
            "features": len(computation["feature_names"]),
            "folds": computation["expected_prediction_keys"]["n_folds"],
            "device": computation["numerics"]["device"],
            "n_epochs": model["params"]["n_epochs"],
            "batch_size": model["params"]["batch_size"],
            "checkpoints": [item["value"] for item in computation["checkpoint_schedule"]],
            "training_hash": run.training.hash,
        }
    )

resolved_table = pl.DataFrame(resolved_rows).sort("config_name")
resolved_table

# %% [markdown]
# ## 5. What came out
#
# One row per configuration and epoch checkpoint. Each is one complete set of validation
# predictions, with the hash of the training run that produced it and the hash of the predictions
# themselves, so any row can be traced back to the exact fitted state behind it.
#
# `ic_mean` is the **information coefficient**: on each validation date, rank the stocks by the
# model's prediction, rank them by the return they went on to earn, correlate the two rankings, and
# average that daily correlation over the validation period. It measures whether predictions order
# the cross-section correctly, and nothing about what a strategy trading them would earn.

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
# ## 6. Naming the sets the later notebooks open
#
# `16_backtest` never opens the population. It opens **named prediction sets**, one per label and
# family, because a comparison is only meaningful within one label's protocol.
# `15_model_analysis` opens both - the population, to confirm the run filled every member it
# promised, and the named sets, to make the comparison. Freezing is what creates those names.
#
# Only an unnarrowed canonical run publishes them, and for the same reason a narrowed run may not
# publish the canonical population: a name must not mean two different member sets at two different
# times. A run that overrides a parameter, selects a subset, or runs on a device other than the
# declared one keeps its rows and publishes no name.

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
        name=f"us-equities-{label_name}-tabular-dl-v1",
    )
    diagnostic_set = study.predictions.freeze(
        execution.catalog_rows.filter(pl.col("config_name").is_in(DIAGNOSTIC_CONFIG_NAMES)),
        name=f"us-equities-{label_name}-tabular-dl-diagnostics-v1",
    )
    set_rows = [
        {
            "role": "backtest population",
            "set_name": full_set.name,
            "members": len(full_set.members),
        },
        {
            "role": "bounded diagnostics",
            "set_name": diagnostic_set.name,
            "members": len(diagnostic_set.members),
        },
    ]
compatible_sets = pl.DataFrame(
    set_rows,
    schema={"role": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `15_model_analysis.py` reopens the named compatible and diagnostic sets. `16_backtest.py` passes
# every full-set catalog row directly to the shared backtest runner. Model metrics do not choose a
# configuration or checkpoint.

# %% [markdown]
# ## What to notice
#
# **A checkpoint is part of a configuration, not a detail of how it was fitted.** Saving weights
# every 25 epochs turns three configurations into twenty-four scoreable models. Treating that as
# three candidates while quietly keeping each one's best epoch would report the maximum of eight
# numbers as though it were one, which is why every checkpoint is registered separately and
# compared as its own candidate.
#
# **A complete run is not the same as a scored one.** Cross-sectional information coefficient needs
# a minimum number of names on a date before that date can be ranked at all. A universe whose
# stocks do not overlap in time can satisfy every coverage check and still score zero dates,
# leaving a null IC under a status that reads complete. The assertion above refuses that rather
# than reporting it.
#
# **This grid measures capacity and nothing finer.** Width and member count move together across
# the three rungs, so a difference between them is a difference in capacity as a whole. Separating
# the two would need a grid holding one fixed while the other varies, which this case study does
# not declare.
#
# **Known limitations.** The features are the same point-in-time columns the previous two
# notebooks read, so anything absent from them is absent here too; the architecture finds
# interactions among the columns it is given and does not create information. Validation results
# say how the fits ranked on folds that have been read many times over by the time a case study
# reaches this notebook, and say nothing about behaviour under a changed feature distribution.
#
# **Next**: [`09_dl_nlinear`](09_dl_nlinear.ipynb) drops the flat-table representation and gives a
# model the ordered window instead.
