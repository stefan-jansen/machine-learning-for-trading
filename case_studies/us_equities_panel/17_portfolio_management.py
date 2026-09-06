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
# # US equities panel: the same names, sized differently
#
# [`16_backtest`](16_backtest.ipynb) put the same amount of money in every position. That is the
# plainest rule there is, and it embeds an assumption worth naming: that a stock the model ranked
# first and a stock it ranked fiftieth deserve the same capital, and that a quiet stock and a
# violent one do too.
#
# This notebook keeps the names and changes only the money. The model, the checkpoint, the
# rebalancing dates and which stocks are held are all held fixed; what varies is how much goes into
# each. Every allocator declared in `config/setup.yaml` is applied, and they answer the question
# in three different ways:
#
# - **From the prediction.** `score_weighted` gives more capital to the names the model was more
#   confident about, so it trusts the magnitude of a prediction and not only its order.
#   `conformal_weighted` reads the prediction's uncertainty rather than its size: it weights each
#   name by one over the width of its prediction interval, so a name the model is less sure about
#   gets less capital. That is the same width whose calibration
#   [`15_model_analysis`](15_model_analysis.ipynb) checked, which is why the check there matters
#   here.
# - **From each stock's own volatility.** `inverse_vol` puts less into a stock that moves more, so
#   each position contributes a similar amount of variation rather than a similar amount of money.
#   `risk_parity` as implemented here is the same idea with a steeper exponent - weight
#   proportional to one over volatility raised to 1.5 - which approximates equal risk contribution
#   without estimating how the stocks move together.
# - **From how the stocks move together.** `mvo_ledoit_wolf` and `hrp` read a covariance matrix, so
#   they alone can tell that two names which always move together are one bet held twice. That is
#   the property none of the rules above can see, and it is the one that has to be estimated. These
#   two need history before they can decide anything, and how much is declared per allocator rather
#   than assumed.
#
# **Equal weight is excluded here, and not because it lost.** It is the baseline every row is
# measured against, and re-running it would produce the identical backtest under a new name.
#
# **A shortlist is taken first, and that is a real decision.** Applying every allocator to every
# member of the whole model population would multiply an already large grid by seven. So the
# highest validation Sharpe per distinct model configuration is carried forward, which means the
# allocator comparison is made on strategies the equal-weight rule already liked. An allocator that
# rescues a model equal weight buried is not something this design can find.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Name the assumption an equal-weight book makes about its positions, and say what each family
#   of allocator replaces it with.
# - Say what a covariance-reading allocator can see that a per-stock one cannot, what it needs in
#   exchange, and which of the declared allocators actually read one.
# - Explain why a lookback window is declared per allocator rather than shared, and what a shared
#   one would silently do to the ones that need less.
# - State what a shortlist taken on baseline Sharpe makes it impossible for this comparison to
#   discover.
#
# **Book reference**: Chapter 17, Sections 17.2 to 17.8.
#
# **Prerequisites**: [`16_backtest`](16_backtest.ipynb) has frozen the equal-weight baseline sets
# this notebook draws from.
#
# **What it writes**: one validation backtest per surviving configuration and allocator, in
# `run_log/registry.db`, frozen as one named allocation set per label.
# [`18_risk_management`](18_risk_management.ipynb) reads them next.

# %%
"""Generate the US-equities allocation-stage validation population."""

import json
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
from case_studies.research.strategy import strategy_warmup_periods
from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices_for,
)
from case_studies.utils.sweep_config import (
    get_allocators,
    get_checkpoints_per_config,
    get_top_n_predictions,
)
from utils.paths import REPO_ROOT

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
BASELINE_SET_NAMES = [
    "us-equities-fwd-ret-1d-baseline-v1",
    "us-equities-fwd-ret-5d-baseline-v1",
    "us-equities-fwd-ret-21d-baseline-v1",
]
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_MAX_BASELINE_ROWS = 0
PREVIEW_MAX_ALLOCATORS = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## 2. The baseline this notebook varies
#
# The equal-weight sets are opened and checked complete. Everything below changes one thing about
# them, so a gap here would silently narrow what the allocator comparison is made over.

# %%
# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS or PREVIEW_MAX_BASELINE_ROWS or PREVIEW_MAX_ALLOCATORS or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not BASELINE_SET_NAMES or len(BASELINE_SET_NAMES) != len(set(BASELINE_SET_NAMES)):
        raise ValueError("Canonical execution requires unique named baseline sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if (
        not PREVIEW_LABELS
        or PREVIEW_MAX_BASELINE_ROWS < 1
        or PREVIEW_MAX_ALLOCATORS < 1
        or MAX_SYMBOLS < 1
    ):
        raise ValueError(
            "Preview execution requires labels and explicit row, allocator, and symbol limits"
        )
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

# %% [markdown]
# ## 3. Which baseline rows can be re-sized
#
# Complete, validation-split, and produced under this run's tier. A row failing any of those is
# refused rather than dropped, so the shortlist below is taken from a population that means what it
# says.

# %%
backtest_catalog = study.backtests.table(include_preview=True)
if EXECUTION_TIER == "canonical":
    baseline_sets = tuple(CandidateSet.one(study, name=name) for name in BASELINE_SET_NAMES)
    if any(result_set.member_kind != "backtest" for result_set in baseline_sets):
        raise ValueError("Every declared baseline set must contain backtests")
    baseline_members = tuple(
        member for result_set in baseline_sets for member in result_set.members
    )
    if len(baseline_members) != len(set(baseline_members)):
        raise ValueError("Declared baseline sets overlap")
    baseline = backtest_catalog.filter(pl.col("backtest_hash").is_in(baseline_members))
    if baseline.height != len(baseline_members):
        raise ValueError("The backtest catalog does not contain every baseline member")
else:
    baseline = (
        backtest_catalog.filter(
            (pl.col("execution_tier") == "preview")
            & (pl.col("stage") == "signal")
            & pl.col("label").is_in(PREVIEW_LABELS)
        )
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(PREVIEW_MAX_BASELINE_ROWS)
    )

ineligible = baseline.filter(
    (pl.col("split") != "validation")
    | (pl.col("execution_tier") != EXECUTION_TIER)
    | (pl.col("stage") != "signal")
    | ~pl.col("complete")
    | pl.col("sharpe").is_null()
    | ~pl.col("sharpe").is_finite()
)
if baseline.is_empty() or not ineligible.is_empty():
    raise ValueError("Allocation requires complete finite equal-weight validation rows")

# %% [markdown]
# ## 4. The shortlist, and what it costs
#
# One row per distinct model configuration, taken on baseline Sharpe. Without it every allocator
# would be applied to every member of the whole model population, multiplying an already large grid
# by the number of allocators.
#
# **What that makes invisible is worth stating plainly.** The allocators are compared only on
# strategies the equal-weight rule already ranked highly. An allocator whose value is precisely
# that it rescues a model equal weight buried cannot be discovered by this design, and no result
# below is evidence against one existing.

# %% tags=["results"]
top_n = get_top_n_predictions(CASE_STUDY_ID, "allocation")
checkpoints_per_config = get_checkpoints_per_config(CASE_STUDY_ID)
if checkpoints_per_config != 1:
    raise ValueError(
        "backtest.sweep.checkpoints_per_config is "
        f"{checkpoints_per_config}; this notebook advances one checkpoint per "
        "model configuration"
    )
shortlist_parts = []
for label in baseline.get_column("label").unique().sort().to_list():
    ranked = baseline.filter(pl.col("label") == label).sort(
        "sharpe", "backtest_hash", descending=[True, False]
    )
    shortlist_parts.append(
        ranked.unique(
            subset=["family", "config_name"],
            keep="first",
            maintain_order=True,
        ).head(top_n)
    )
shortlist = pl.concat(shortlist_parts).sort("label", "sharpe", descending=[False, True])
if shortlist.is_empty():
    raise RuntimeError("The equal-weight baseline produced no allocation survivors")

shortlist.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "prediction_hash",
    "backtest_hash",
    "sharpe",
)

# %% [markdown]
# ## 5. Planning one backtest per allocator
#
# Each surviving configuration crossed with each declared allocator, every identity written down
# before the first runs.
#
# **The history each allocator needs is declared per allocator, not shared.** The methods that read
# a covariance matrix cannot decide anything until they have enough bars to estimate one, and the
# amount differs between them - the mean-variance method here declares a longer window than the
# others because shrinkage on a matrix estimated from too few observations collapses toward its
# target and hands back something close to equal weight under a different name. Giving every
# allocator the longest window instead would change what the cheap ones are measured on.

# %%
# Prices are cached by (label, warmup) rather than loaded once per label. Strategy._build_spec
# (research/strategy.py:389) digests exactly the frame it is handed, and strategy_warmup_periods
# (:201-211) resolves a different prefix per allocator: 0 for the non-moment methods, vol_window
# for inverse_vol / risk_parity / hrp, lookback for mvo and mvo_ledoit_wolf. Handing every member
# of a label the same 126-bar frame stamps a price digest that 20_strategy_analysis recomputes at
# the member's own warmup (20:157-169) and then rejects as "does not use canonical validation
# prices" - and lifecycle.evaluate_holdout (lifecycle.py:342-368) applies the same rule, so the
# holdout inherits it. cme_futures/research_workflow.py:674-682 caches on the same key.
_price_cache: dict[tuple[str, int], object] = {}


def prices_for(label, warmup_periods):
    key = (str(label), int(warmup_periods))
    if key not in _price_cache:
        _price_cache[key] = load_backtest_prices_for(
            CASE_STUDY_ID,
            label,
            split="validation",
            max_symbols=MAX_SYMBOLS,
            warmup_periods=int(warmup_periods),
        )
    return _price_cache[key]


allocators = [
    config for config in get_allocators(CASE_STUDY_ID) if config["method"] != "equal_weight"
]
if EXECUTION_TIER == "preview":
    allocators = allocators[:PREVIEW_MAX_ALLOCATORS]
if not allocators or any(config["method"] == "equal_weight" for config in allocators):
    raise ValueError("Allocation requires at least one non-baseline sizing method")

prediction_catalog = study.predictions.table(include_preview=True)
backtest_config = get_backtest_config(CASE_STUDY_ID)
planned_requests = []
plan_rows = []


# %%
def plan_allocation_member(label, prices, allocation, baseline_row):
    selected_prediction = prediction_catalog.filter(
        pl.col("prediction_hash") == baseline_row["prediction_hash"]
    )
    if selected_prediction.height != 1:
        raise ValueError("A baseline survivor must resolve one prediction catalog row")
    baseline_spec = json.loads(baseline_row["spec_json"])
    signal = dict(baseline_spec["strategy"]["signal"])
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        allocation=allocation,
        prices=prices,
        chapter="ch17",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One allocation request must plan one backtest")
    expected_hash = plan.expected_hashes[0]
    request = {
        "label": label,
        "selection": selected_prediction,
        "signal": signal,
        "allocation": allocation,
        "prediction_hash": baseline_row["prediction_hash"],
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "family": baseline_row["family"],
        "config_name": baseline_row["config_name"],
        "checkpoint_kind": baseline_row["checkpoint_kind"],
        "checkpoint_value": baseline_row["checkpoint_value"],
        "allocation": allocation["method"],
        "prediction_hash": baseline_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
for label in shortlist.get_column("label").unique().sort().to_list():
    for baseline_row in shortlist.filter(pl.col("label") == label).iter_rows(named=True):
        for allocation in allocators:
            prices = prices_for(
                label, strategy_warmup_periods({"strategy": {"allocation": allocation}})
            )
            request, row = plan_allocation_member(label, prices, allocation, baseline_row)
            planned_requests.append(request)
            plan_rows.append(row)

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "family", "config_name", "checkpoint_value", "allocation", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The allocation plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-allocation-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## 6. Running them
#
# Independent per member, so a failure costs that allocator on that configuration and leaves the
# rest usable.

# %%
execution_rows = []
failure_rows = []


def execute_allocation_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        allocation=request["allocation"],
        prices=prices,
        chapter="ch17",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Allocation execution changed its planned identity")
    return {
        "label": request["label"],
        "prediction_hash": request["prediction_hash"],
        "allocation": request["allocation"]["method"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in shortlist.get_column("label").unique().sort().to_list():
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            prices = prices_for(
                label,
                strategy_warmup_periods({"strategy": {"allocation": request["allocation"]}}),
            )
            execution_rows.append(execute_allocation_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "prediction_hash": request["prediction_hash"],
                    "allocation": request["allocation"]["method"],
                    "backtest_hash": request["expected_hash"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

# %% tags=["results"]
execution_diagnostics = pl.DataFrame(
    execution_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "allocation": pl.String,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "prediction_hash": pl.String,
        "allocation": pl.String,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Allocation population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## 7. Naming the allocation sets
#
# One frozen set per label, published only by an unnarrowed canonical run, for the reason
# [`16_backtest`](16_backtest.ipynb) gives.

# %% tags=["results"]
set_rows = []
completed = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed.height != planned_population.height
    or completed.filter(~pl.col("complete")).height
    or completed.filter(pl.col("stage") != "allocation").height
    or completed.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The allocation catalog is incomplete or mis-staged")
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
            name=f"us-equities-{label_name}-allocation-v1",
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
# ## 8. What came out
#
# Each allocator against the equal-weight row it was built from. The comparison is like-for-like:
# same model, same checkpoint, same names, same dates, different money.
#
# **A small difference is a result.** Equal weight is a strong baseline on a broad cross-section
# precisely because it makes no estimate that can be wrong, and an allocator that reads a
# covariance matrix has to estimate one well enough to beat that. Where the differences are small,
# what that says is that the estimation was not worth its error here - not that sizing does not
# matter.
#
# **Still gross of costs.** The allocators differ in how much they trade, and turnover is charged
# in [`19_costs`](19_costs.ipynb), so an allocator that looks better here may not survive it.

# %% tags=["results"]
allocation_results = planned_population.select("label", "allocation", "backtest_hash").join(
    completed.select("backtest_hash", "sharpe"),
    on="backtest_hash",
    how="inner",
    validate="1:1",
)
if allocation_results.height != planned_population.height:
    raise RuntimeError("The plotted allocation population differs from the planned population")

fig, ax = plt.subplots(figsize=(10, 5))
for label in allocation_results.get_column("label").unique().sort().to_list():
    label_rows = allocation_results.filter(pl.col("label") == label)
    ax.scatter(
        label_rows["allocation"],
        label_rows["sharpe"],
        alpha=0.5,
        s=20,
        label=label,
    )
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Allocator")
ax.set_ylabel("Validation Sharpe")
ax.set_title("Alternative-sizing validation Sharpe by allocator")
ax.tick_params(axis="x", rotation=25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.show()

# %% [markdown]
# The cost and risk notebooks reopen these names together with the matching equal-weight baseline.
# No model is retrained, and the selected checkpoint remains part of every downstream identity.

# %% [markdown]
# ## What to notice
#
# **Every row here differs from its baseline in exactly one thing.** Same model, same checkpoint,
# same names on the same dates, different money. That is what makes a difference attributable to
# the sizing rule.
#
# **Equal weight is hard to beat on a broad cross-section, and the reason is estimation.** The
# allocators that read a covariance matrix have to estimate one from a finite window, and a
# three-thousand-name cross-section gives far fewer observations per parameter than a small
# universe does. An allocator that does not beat equal weight here has not shown that sizing is
# irrelevant; it has shown that the estimate it needed was not accurate enough to pay for itself.
#
# **The shortlist bounds what this can find.** Allocators are compared only on strategies equal
# weight already ranked highly, so nothing here can discover one whose value is rescuing a model
# equal weight buried.
#
# **Still gross of costs, and the allocators differ in turnover.** A rule that reweights more
# aggressively trades more, so an ordering established here can change once
# [`19_costs`](19_costs.ipynb) charges for it.
#
# **Known limitations.** The covariance-reading allocators are sensitive to their lookback, and one
# window per allocator is declared rather than swept, so nothing here separates an allocator's
# method from its window. Validation folds have been read many times over by this point.
#
# **Next**: [`18_risk_management`](18_risk_management.ipynb) lays rules on top that can close a
# position before the next rebalance.
