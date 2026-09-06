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
# # US Equities Panel: Equal-Weight Baseline
#
# Every complete model configuration and checkpoint enters the equal-weight baseline. The notebook
# opens immutable prediction sets by name, selects their catalog rows with Polars, plans every
# backtest identity before execution, and passes each selected row directly to the shared runner.
# Predictive metrics do not filter or rank this population.
#
# **Learning objectives**
#
# - Construct the equal-weight signal baseline from the complete prediction population.
# - Plan every backtest identity before execution and preserve unsuccessful members for restart.
# - Validate complete result coverage and publish compatible label sets.
#
# **Book reference**: Chapter 16, Sections 16.4-16.8
#
# **Prerequisites**: `15_model_analysis.py` validates the complete model populations consumed here.

# %%
"""Generate the complete US-equities equal-weight validation baseline."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from case_studies.research import (
    CandidateSet,
    OfficialPopulation,
    Study,
    open_study,
    plan_backtests,
    run_backtests,
)
from case_studies.utils.backtest_loaders import get_backtest_config, load_backtest_prices_for
from case_studies.utils.sweep_config import get_entry_schemes_for
from utils.paths import REPO_ROOT
from utils.style import COLORS, show_with_alt, zero_line

# %% [markdown]
# ### Which prediction sets enter the pool
#
# Every model notebook froze its results under a name per label and family, and the list below is
# the pool this baseline ranks. A name missing from it is a family that never competes, so the
# list is the declaration of what the strategy chain is allowed to choose from.
#
# **The weekly experiment is deliberately absent.** [`12_dl_weekly`](12_dl_weekly.ipynb) fits
# `fwd_ret_5d` on a Friday-only grid, and its predictions carry the same label as the daily models
# below. Ranking them together would compare series scored on different sets of decision dates -
# one on every session, one on about a fifth of them - and read the difference as a difference
# between models. It runs and registers; it is not a candidate here.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PREDICTION_SET_NAMES = [
    "us-equities-fwd-ret-1d-linear-v1",
    "us-equities-fwd-ret-5d-linear-v1",
    "us-equities-fwd-ret-21d-linear-v1",
    "us-equities-fwd-ret-1d-gbm-v1",
    "us-equities-fwd-ret-5d-gbm-v1",
    "us-equities-fwd-ret-21d-gbm-v1",
    "us-equities-fwd-ret-1d-tabular-dl-v1",
    "us-equities-fwd-ret-1d-nlinear-v1",
    "us-equities-fwd-ret-1d-lstm-v1",
    "us-equities-fwd-ret-1d-tsmixer-v1",
    "us-equities-fwd-ret-1d-pca-v1",
    "us-equities-fwd-ret-1d-ipca-v1",
    "us-equities-fwd-ret-5d-pca-v1",
    "us-equities-fwd-ret-5d-ipca-v1",
    "us-equities-fwd-ret-21d-pca-v1",
    "us-equities-fwd-ret-21d-ipca-v1",
]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_FAMILIES = []
PREVIEW_CONFIG_NAMES = []
PREVIEW_MAX_PREDICTIONS = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## Declare the prediction population
#
# Canonical execution requires the complete named model population. A reduced proof uses preview
# ancestry, names at least one label, family, or configuration, limits the prediction count, and
# limits the real-data universe. Preview rows remain outside official populations and candidate
# sets.

# %%
preview_filters = bool(PREVIEW_LABELS or PREVIEW_FAMILIES or PREVIEW_CONFIG_NAMES)
# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if preview_filters or PREVIEW_MAX_PREDICTIONS or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not PREDICTION_SET_NAMES or len(PREDICTION_SET_NAMES) != len(set(PREDICTION_SET_NAMES)):
        raise ValueError("Canonical execution requires unique named prediction sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if not preview_filters or PREVIEW_MAX_PREDICTIONS < 1 or MAX_SYMBOLS < 1:
        raise ValueError(
            "Preview execution requires a catalog filter, prediction limit, and symbol limit"
        )
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

# %% [markdown]
# ## Select eligible prediction rows
#
# Canonical rows come from the named immutable sets. Preview rows use only the visible catalog
# filters and limits declared above.

# %%
prediction_catalog = study.predictions.table(include_preview=True)
if EXECUTION_TIER == "canonical":
    prediction_sets = tuple(CandidateSet.one(study, name=name) for name in PREDICTION_SET_NAMES)
    if any(result_set.member_kind != "prediction" for result_set in prediction_sets):
        raise ValueError("Every declared input set must contain predictions")
    prediction_members = tuple(
        member for result_set in prediction_sets for member in result_set.members
    )
    if len(prediction_members) != len(set(prediction_members)):
        raise ValueError("Declared prediction sets overlap")
    predictions = prediction_catalog.filter(pl.col("prediction_hash").is_in(prediction_members))
    if predictions.height != len(prediction_members):
        raise ValueError("The prediction catalog does not contain every declared member")
else:
    predictions = prediction_catalog.filter(pl.col("execution_tier") == "preview")
    if PREVIEW_LABELS:
        predictions = predictions.filter(pl.col("label").is_in(PREVIEW_LABELS))
    if PREVIEW_FAMILIES:
        predictions = predictions.filter(pl.col("family").is_in(PREVIEW_FAMILIES))
    if PREVIEW_CONFIG_NAMES:
        predictions = predictions.filter(pl.col("config_name").is_in(PREVIEW_CONFIG_NAMES))
    predictions = predictions.sort("prediction_hash").head(PREVIEW_MAX_PREDICTIONS)

ineligible = predictions.filter(
    (pl.col("split") != "validation")
    | (pl.col("execution_tier") != EXECUTION_TIER)
    | ~pl.col("complete")
)
if predictions.is_empty() or not ineligible.is_empty():
    raise ValueError(
        "The baseline requires complete validation predictions from one execution tier"
    )

# %% [markdown]
# The selected rows expose the model family, configuration, label, checkpoint, and exact result
# identities that enter the baseline.

# %% tags=["results"]
prediction_population = predictions.select(
    "family",
    "config_name",
    "label",
    "checkpoint_kind",
    "checkpoint_value",
    "training_hash",
    "prediction_hash",
).sort("label", "family", "config_name", "checkpoint_value", "prediction_hash")
prediction_population

# %% [markdown]
# ## Plan every baseline member
#
# The shared planner resolves the exact backtest identity without fitting or writing a result. The
# canonical population snapshot is written before the first backtest so an unsuccessful member
# remains visible as missing rather than disappearing from the cohort.

# %%
backtest_config = get_backtest_config(CASE_STUDY_ID)
planned_requests = []
plan_rows = []


def plan_baseline_member(label, prices, scheme, prediction_row):
    selected_prediction = predictions.filter(
        pl.col("prediction_hash") == prediction_row["prediction_hash"]
    )
    signal = {key: value for key, value in scheme.items() if key != "name"}
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        prices=prices,
        chapter="ch16",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One prediction and signal request must plan one backtest")
    expected_hash = plan.expected_hashes[0]
    request = {
        "label": label,
        "prediction_hash": prediction_row["prediction_hash"],
        "selection": selected_prediction,
        "signal_name": scheme["name"],
        "signal": signal,
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "family": prediction_row["family"],
        "config_name": prediction_row["config_name"],
        "checkpoint_kind": prediction_row["checkpoint_kind"],
        "checkpoint_value": prediction_row["checkpoint_value"],
        "signal": scheme["name"],
        "prediction_hash": prediction_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
for label in predictions.get_column("label").unique().sort().to_list():
    label_predictions = predictions.filter(pl.col("label") == label)
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
    )
    n_assets = prices.get_column("symbol").n_unique()
    schemes = get_entry_schemes_for(
        CASE_STUDY_ID,
        label,
        n_assets,
        long_short=backtest_config.long_short,
    )
    if not schemes or any(scheme["method"] != "equal_weight_top_k" for scheme in schemes):
        raise ValueError(f"{label} does not declare only equal-weight baseline schemes")
    for prediction_row in label_predictions.iter_rows(named=True):
        for scheme in schemes:
            request, row = plan_baseline_member(label, prices, scheme, prediction_row)
            planned_requests.append(request)
            plan_rows.append(row)
    del prices

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "family", "config_name", "checkpoint_value", "signal", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The baseline plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-baseline-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## Execute the planned members
#
# Each call receives one selected Polars catalog row. A failure is recorded and the remaining
# members continue, while completed results are reusable on restart. The notebook raises after the
# pass if any member is still unsuccessful.

# %%
execution_rows = []
failure_rows = []


def execute_baseline_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        prices=prices,
        chapter="ch16",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Baseline execution changed its planned identity")
    return {
        "label": request["label"],
        "prediction_hash": request["prediction_hash"],
        "signal": request["signal_name"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in predictions.get_column("label").unique().sort().to_list():
    prices = load_backtest_prices_for(
        CASE_STUDY_ID,
        label,
        split="validation",
        max_symbols=MAX_SYMBOLS,
    )
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            execution_rows.append(execute_baseline_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "prediction_hash": request["prediction_hash"],
                    "signal": request["signal_name"],
                    "backtest_hash": request["expected_hash"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    del prices

# %% tags=["results"]
execution_diagnostics = pl.DataFrame(
    execution_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "signal": pl.String,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "signal": pl.String,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Baseline population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## Freeze the reader-facing baseline sets
#
# Each label gets one immutable compatible set. The allocation notebook reopens these names and
# derives its shortlist from complete validation backtest Sharpe rows.

# %% tags=["results"]
set_rows = []
completed = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed.height != planned_population.height
    or completed.filter(~pl.col("complete")).height
    or completed.filter(pl.col("stage") != "signal").height
    or completed.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The baseline catalog is incomplete or mis-staged")
if EXECUTION_TIER == "canonical":
    for label in completed.get_column("label").unique().sort().to_list():
        label_name = label.replace("_", "-")
        # No comparison_contract, matching cme_futures/research_workflow.py:811, which builds the
        # same per-label pool across the full funnel and declares nothing. An empty contract makes
        # every protocol field required-constant, which is the guard: if two members disagree on
        # `cv` they measured their Sharpe on different folds and ranking them is not a comparison,
        # and this field is the only thing checking that. Latent-factor members will refuse on
        # `feature_artifacts` when they enter this pool - latent builds it from a different object
        # than the other five families (latent_factors/case_study.py:337-383, carrying the label
        # digest and setup.yaml bytes). That refusal is a known adapter defect surfacing, not a
        # property to declare around; report it rather than adding the field here.
        result_set = study.backtests.freeze(
            completed.filter(pl.col("label") == label),
            name=f"us-equities-{label_name}-baseline-v1",
        )
        set_rows.append(
            {"label": label, "set_name": result_set.name, "members": len(result_set.members)}
        )

compatible_sets = pl.DataFrame(
    set_rows,
    schema={"label": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# ## Inspect the baseline result population
#
# Every point below is one complete equal-weight validation backtest. The figure displays the
# distribution by label and model family without filtering the population or changing the
# downstream selection rule.

# %% tags=["results"]
groups = completed.select("label", "family").unique().sort("label", "family").rows(named=True)
fig, ax = plt.subplots(figsize=(9, 5))
for position, group in enumerate(groups):
    values = completed.filter(
        (pl.col("label") == group["label"]) & (pl.col("family") == group["family"])
    ).get_column("sharpe")
    jitter = (values.arg_sort().to_numpy() % 9 - 4) / 40
    ax.scatter(
        position + jitter,
        values.to_numpy(),
        alpha=0.45,
        s=18,
        color=COLORS["blue"],
        edgecolors="none",
    )
zero_line(ax)
ax.set_xticks(
    range(len(groups)),
    [f"{group['label']}\n{group['family']}" for group in groups],
)
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.set_ylabel("Validation Sharpe")
ax.set_title("Equal-weight baseline Sharpe by label and model family")
fig.tight_layout()
show_with_alt(
    fig,
    "A strip plot with one column per label and model family. Each point is one complete "
    "equal-weight validation backtest, placed at its Sharpe ratio on the vertical axis and "
    "spread horizontally within its column so overlapping points stay visible. A dashed "
    "horizontal reference line marks zero.",
)

# %% [markdown]
# The allocation notebook consumes these complete named baseline sets. Selection remains based on
# validation backtest Sharpe, with the checkpoint retained as part of the selected configuration.

# %% [markdown]
# ## Key takeaways and limitations
#
# - Every complete model configuration and checkpoint receives the same equal-weight signal-stage
#   treatment.
# - The official population snapshot makes a missing or failed backtest visible instead of silently
#   narrowing the cohort.
# - Validation Sharpe becomes the downstream strategy-selection metric; predictive IC remains
#   descriptive.
# - Equal weighting is the common signal baseline. Allocation, costs, risk controls, and the
#   holdout replay are evaluated through separate requests.
