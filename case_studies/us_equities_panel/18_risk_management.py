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
# # US equities panel: rules that close a position early
#
# The two notebooks before this one decided which stocks to hold and how much to put in each. Both
# hold every position from one rebalancing date to the next, whatever it does in between. A **risk
# overlay** is a rule that can close a position before then.
#
# Three kinds are declared, and they differ in what they watch:
#
# - A **stop loss** closes a position that has lost more than a fixed fraction from where it was
#   opened. It watches the loss from entry.
# - A **trailing stop** closes a position that has fallen more than a fixed fraction from its own
#   best level. It watches the give-back, so a position that rose and then reversed is closed even
#   while it is still ahead of entry.
# - A **time exit** closes a position after a fixed number of bars whatever it has done. It watches
#   nothing about the price, which makes it the control: it tests whether the holding period itself
#   was the problem, separately from any threshold.
#
# **An overlay only ever removes.** It cannot enter a position the strategy did not take, so it
# can only cut a loss short or cut a gain short, and which of the two it does more of is exactly
# what the sweep measures. Fourteen controls are declared across the three kinds, spanning stops
# from 3% to 15% and trailing stops from 1% to 20%, so the sweep says how the effect moves with the
# threshold rather than whether one chosen threshold helped.
#
# **The thing to check first is whether the overlays changed anything at all.** A control that
# never fires returns the unprotected book unchanged in every digit, and so does a control that was
# declared but never reached the engine. One result cannot tell those apart, and the second has
# happened in this repository: fourteen controls, from a 3% stop to a 40-bar time exit, whose
# Sharpe, drawdown and trade count all matched the unprotected book exactly, because the
# configuration declared them in one shape and the engine read another. Fourteen different rules
# declining to act on one book and agreeing to the last digit is not a result about risk control.
#
# So the results section compares each overlay against the strategy it was laid on rather than
# reporting its performance alone. **A difference proves the control acted; matching statistics
# prove nothing about whether it fired** - a stop can close a position the next rebalance would
# have closed anyway and leave both numbers where they were. Matching across *every* declared
# setting is a reason to confirm the controls reach the engine, not a conclusion.
#
# **Learning objectives.** By the end of this notebook you will be able to:
#
# - Describe what each of the three kinds of overlay watches, and say which one is the control and
#   why.
# - Say why an overlay can only remove, and what that implies about how it can change a return
#   distribution.
# - Say what a result matching its unprotected baseline on two summary statistics does and does
#   not establish about whether a control fired, and what evidence would settle it.
# - Say why a threshold sweep is more informative than a single chosen threshold.
#
# **Book reference**: Chapter 19, Sections 19.3 to 19.6.
#
# **Prerequisites**: [`16_backtest`](16_backtest.ipynb) and
# [`17_portfolio_management`](17_portfolio_management.ipynb) have frozen the baseline and
# allocation sets this notebook draws from.
#
# **What it writes**: one validation backtest per fixed configuration and declared control, in
# `run_log/registry.db`, frozen as one named risk-overlay set per label, plus the union of the
# three stages as the population validation selection is made over.
# [`19_costs`](19_costs.ipynb) then charges the chosen strategy for trading.

# %%
"""Generate risk overlays and freeze the official US-equities validation set."""

import json
import os
from pathlib import Path

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
from case_studies.utils.backtest_loaders import load_backtest_prices_for, warmup_periods_for
from case_studies.utils.sweep_config import (
    get_portfolio_risk_controls,
    get_position_risk_controls,
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
ALLOCATION_SET_NAMES = [
    "us-equities-fwd-ret-1d-allocation-v1",
    "us-equities-fwd-ret-5d-allocation-v1",
    "us-equities-fwd-ret-21d-allocation-v1",
]
VALIDATION_SET_NAME_TEMPLATE = "us-equities-{label}-validation-strategies-v1"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
PREVIEW_LABELS = []
PREVIEW_MAX_SOURCE_ROWS = 0
PREVIEW_MAX_RISK_CONTROLS = 0
MAX_SYMBOLS = 0

# %% [markdown]
# ## 2. The strategies an overlay may be laid on
#
# The baseline and allocation sets, opened and checked complete. Both stages are eligible: an
# overlay is a rule about when to close a position, and it applies whether the position was sized
# equally or by an allocator.

# %%
declared_set_names = [*BASELINE_SET_NAMES, *ALLOCATION_SET_NAMES]
# Both tiers resolve the study through `open_study`, never `Study.open`/`Study.regenerate`
# directly. In a maintainer worktree the generated directories are symlinks to shared data, and
# `open_study` handles that by reading inputs in place - `root` stays the release case directory
# and only writes are redirected to the workspace. `Study.open(workspace=...)` instead puts `root`
# inside the workspace, so `source = self.root / "labels"` (workspace.py:274) resolves somewhere
# else and `_ensure_input_link` rejects the link a sibling notebook already made. Two notebooks in
# one session then cannot both open a preview workspace.
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS or PREVIEW_MAX_SOURCE_ROWS or PREVIEW_MAX_RISK_CONTROLS or MAX_SYMBOLS:
        raise ValueError("Canonical execution cannot declare preview reductions")
    if not declared_set_names or len(declared_set_names) != len(set(declared_set_names)):
        raise ValueError("Canonical execution requires unique named strategy sets")
    study = open_study(CASE_STUDY_ID, execution_tier=EXECUTION_TIER)
elif EXECUTION_TIER == "preview":
    if (
        not PREVIEW_LABELS
        or PREVIEW_MAX_SOURCE_ROWS < 1
        or PREVIEW_MAX_RISK_CONTROLS < 1
        or MAX_SYMBOLS < 1
    ):
        raise ValueError(
            "Preview execution requires labels and explicit row, risk, and symbol limits"
        )
    study = open_study(
        CASE_STUDY_ID,
        execution_tier=EXECUTION_TIER,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

# %% [markdown]
# ## 3. Which rows can carry an overlay
#
# Complete, validation-split, and produced under this run's tier, with a finite Sharpe. A row
# failing any of those is refused rather than dropped.

# %%
backtest_catalog = study.backtests.table(include_preview=True)
if EXECUTION_TIER == "canonical":
    declared_sets = tuple(CandidateSet.one(study, name=name) for name in declared_set_names)
    if any(result_set.member_kind != "backtest" for result_set in declared_sets):
        raise ValueError("Every declared input set must contain backtests")
    source_members = tuple(member for result_set in declared_sets for member in result_set.members)
    if len(source_members) != len(set(source_members)):
        raise ValueError("Declared baseline and allocation sets overlap")
    eligible = backtest_catalog.filter(pl.col("backtest_hash").is_in(source_members))
    if eligible.height != len(source_members):
        raise ValueError("The backtest catalog does not contain every declared strategy member")
else:
    eligible = (
        backtest_catalog.filter(
            (pl.col("execution_tier") == "preview")
            & pl.col("stage").is_in(["signal", "allocation"])
            & pl.col("label").is_in(PREVIEW_LABELS)
        )
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(PREVIEW_MAX_SOURCE_ROWS)
    )

ineligible = eligible.filter(
    (pl.col("split") != "validation")
    | (pl.col("execution_tier") != EXECUTION_TIER)
    | ~pl.col("stage").is_in(["signal", "allocation"])
    | ~pl.col("complete")
    | pl.col("sharpe").is_null()
    | ~pl.col("sharpe").is_finite()
)
if eligible.is_empty() or not ineligible.is_empty():
    raise ValueError("Risk overlays require complete finite selection-eligible validation rows")

# %% [markdown]
# ## 4. Fixing the strategy the overlays are applied to
#
# One configuration per label, taken on validation Sharpe across both earlier stages. Everything
# identifying the strategy is then held - the model, the checkpoint, the signal, the sizing - so
# every row below differs from every other only in the overlay laid on it.
#
# **Sweeping overlays across several strategies at once would confound the two.** A table in which
# both the strategy and the control vary cannot say whether a difference came from the rule or from
# the book it was applied to.

# %% tags=["results"]
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


top_n = get_top_n_predictions(CASE_STUDY_ID, "risk_overlay")
selected_parts = []
for label in eligible.get_column("label").unique().sort().to_list():
    selected_parts.append(
        eligible.filter(pl.col("label") == label)
        .sort("sharpe", "backtest_hash", descending=[True, False])
        .head(top_n)
    )
selected_sources = pl.concat(selected_parts).sort("label", "backtest_hash")
if selected_sources.is_empty():
    raise RuntimeError("No risk-overlay source configuration was selected")

selected_sources.select(
    "label",
    "family",
    "config_name",
    "checkpoint_kind",
    "checkpoint_value",
    "stage",
    "prediction_hash",
    "backtest_hash",
    "sharpe",
)

# %% [markdown]
# ## 5. The controls, and the check that they bound
#
# Fourteen declared controls across the three kinds, planned as one backtest each against the fixed
# strategy.
#
# **Whether the overlays changed anything is checked after execution**, in the section that reports
# results: each one against the strategy it was laid on, on the trade count and the Sharpe. A
# difference establishes that the control acted. Matching values establish only that those two
# statistics did not move, and leave every other outcome undetermined.

# %%
risk_requests = []
for control in get_position_risk_controls(CASE_STUDY_ID):
    if control["type"] == "time_exit":
        rule = {"type": control["type"], "bars": control["bars"]}
    else:
        rule = {"type": control["type"], "threshold": control["threshold"]}
    risk_requests.append(
        {
            "name": control["name"],
            "spec": {"name": control["name"], "position_rules": [rule]},
        }
    )
for control in get_portfolio_risk_controls(CASE_STUDY_ID):
    risk_requests.append(
        {
            "name": control["name"],
            "spec": {
                "name": control["name"],
                "portfolio_limits": [{"type": control["type"], "threshold": control["threshold"]}],
            },
        }
    )
if EXECUTION_TIER == "preview":
    risk_requests = risk_requests[:PREVIEW_MAX_RISK_CONTROLS]
if not risk_requests or len({request["name"] for request in risk_requests}) != len(risk_requests):
    raise ValueError("Risk controls must be non-empty and uniquely named")

prediction_catalog = study.predictions.table(include_preview=True)
planned_requests = []
plan_rows = []


# %%
def risk_member_records(
    label, source_row, selection, signal, allocation, risk_request, expected_hash
):
    request = {
        "label": label,
        "selection": selection,
        "signal": signal,
        "allocation": allocation,
        "risk": risk_request["spec"],
        "risk_name": risk_request["name"],
        "prediction_hash": source_row["prediction_hash"],
        "source_backtest_hash": source_row["backtest_hash"],
        "expected_hash": expected_hash,
    }
    row = {
        "label": label,
        "source_stage": source_row["stage"],
        "source_backtest_hash": source_row["backtest_hash"],
        "risk": risk_request["name"],
        "prediction_hash": source_row["prediction_hash"],
        "backtest_hash": expected_hash,
    }
    return request, row


# %%
def plan_risk_member(label, prices, risk_request, source_row):
    selected_prediction = prediction_catalog.filter(
        pl.col("prediction_hash") == source_row["prediction_hash"]
    )
    if selected_prediction.height != 1:
        raise ValueError("A risk source must resolve one prediction catalog row")
    source_spec = json.loads(source_row["spec_json"])
    signal = dict(source_spec["strategy"]["signal"])
    allocation = source_spec["strategy"].get("allocation")
    plan = plan_backtests(
        study,
        predictions=selected_prediction,
        signal=signal,
        allocation=allocation,
        risk=risk_request["spec"],
        prices=prices,
        chapter="ch19",
    )
    if len(plan.members) != 1:
        raise RuntimeError("One risk request must plan one backtest")
    return risk_member_records(
        label,
        source_row,
        selected_prediction,
        signal,
        allocation,
        risk_request,
        plan.expected_hashes[0],
    )


# %%
for label in selected_sources.get_column("label").unique().sort().to_list():
    for source_row in selected_sources.filter(pl.col("label") == label).iter_rows(named=True):
        for risk_request in risk_requests:
            prices = prices_for(
                label,
                # The source's allocation lives in its spec_json, not as a catalog column, so
                # source_row.get("allocation") is always None and would silently warm up 0 bars
                # for every allocation-stage source.
                strategy_warmup_periods(json.loads(source_row["spec_json"])),
            )
            request, row = plan_risk_member(label, prices, risk_request, source_row)
            planned_requests.append(request)
            plan_rows.append(row)

# %%
planned_population = pl.DataFrame(plan_rows).sort(
    "label", "risk", "source_backtest_hash", "backtest_hash"
)
if planned_population.get_column("backtest_hash").n_unique() != planned_population.height:
    raise ValueError("The risk plan contains duplicate backtest identities")

official_population = None
if EXECUTION_TIER == "canonical":
    official_population = OfficialPopulation.create(
        study,
        name="us-equities-risk-overlay-v1",
        member_kind="backtest",
        members=tuple(planned_population.get_column("backtest_hash")),
    )

planned_population

# %% [markdown]
# ## 6. Running them
#
# Independent per control, so a failure costs that control and leaves the rest usable.

# %%
execution_rows = []
failure_rows = []


def execute_risk_member(prices, request):
    execution = run_backtests(
        study,
        predictions=request["selection"],
        signal=request["signal"],
        allocation=request["allocation"],
        risk=request["risk"],
        prices=prices,
        chapter="ch19",
    )
    if len(execution.results) != 1 or execution.results[0].hash != request["expected_hash"]:
        raise RuntimeError("Risk execution changed its planned identity")
    return {
        "label": request["label"],
        "source_backtest_hash": request["source_backtest_hash"],
        "risk": request["risk_name"],
        "backtest_hash": execution.results[0].hash,
        "status": execution.diagnostics[0]["status"],
    }


# %% tags=["results"]
for label in selected_sources.get_column("label").unique().sort().to_list():
    for request in (item for item in planned_requests if item["label"] == label):
        try:
            prices = prices_for(
                label,
                strategy_warmup_periods({"strategy": {"allocation": request["allocation"]}}),
            )
            execution_rows.append(execute_risk_member(prices, request))
        except Exception as error:
            failure_rows.append(
                {
                    "label": label,
                    "source_backtest_hash": request["source_backtest_hash"],
                    "risk": request["risk_name"],
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
        "source_backtest_hash": pl.String,
        "risk": pl.String,
        "backtest_hash": pl.String,
        "status": pl.String,
    },
)
failures = pl.DataFrame(
    failure_rows,
    schema={
        "label": pl.String,
        "source_backtest_hash": pl.String,
        "risk": pl.String,
        "backtest_hash": pl.String,
        "error_type": pl.String,
        "error": pl.String,
    },
)
if not failures.is_empty():
    raise RuntimeError(f"Risk population has {failures.height} unsuccessful members")

if official_population is not None:
    official_population.require_complete()

execution_diagnostics

# %% [markdown]
# ## 7. Naming the overlay sets, and the population selection is made over
#
# One frozen risk-overlay set per label, plus the union of all three stages - baseline, allocation
# and overlay - as the population validation selection is made over.
#
# **The union is the point.** A strategy may legitimately come from any of the three stages, so
# ranking only the last of them would exclude an un-overlaid book that was better than every
# overlaid one. [`19_costs`](19_costs.ipynb) derives its pool from the same stage sequence for the
# same reason.

# %% tags=["results"]
completed_risk = study.backtests.table(include_preview=True).filter(
    pl.col("backtest_hash").is_in(planned_population.get_column("backtest_hash"))
)
if (
    completed_risk.height != planned_population.height
    or completed_risk.filter(~pl.col("complete")).height
    or completed_risk.filter(pl.col("stage") != "risk_overlay").height
    or completed_risk.filter(pl.col("execution_tier") != EXECUTION_TIER).height
    or completed_risk.filter(pl.col("sharpe").is_null() | ~pl.col("sharpe").is_finite()).height
):
    raise RuntimeError("The risk catalog is incomplete or mis-staged")
# Did the overlay change anything? Each result is compared against the strategy it was laid on, on
# the two axes the catalog carries: the trade count and the Sharpe. A row that differs on either
# acted - there is no other way for the numbers to move. A row identical on both changed no
# neither of the two statistics compared here - which is weaker than "changed nothing", since two
# different return paths can share a Sharpe and a trade count while differing in total return or in
# drawdown, and weaker still than "never fired": a stop can close a position the next rebalance
# would have closed anyway, replacing one exit with an earlier one and leaving the count where it
# was. What would settle whether a control fired is a per-control trigger count, which the backtest
# does not surface into the catalog today.
overlay_effect = (
    completed_risk.select("label", "backtest_hash", "sharpe", "num_trades")
    .join(
        planned_population.select("backtest_hash", "risk", "source_backtest_hash"),
        on="backtest_hash",
        how="inner",
    )
    .join(
        backtest_catalog.select(
            pl.col("backtest_hash").alias("source_backtest_hash"),
            pl.col("num_trades").alias("source_num_trades"),
            pl.col("sharpe").alias("source_sharpe"),
        ),
        on="source_backtest_hash",
        how="left",
    )
    .with_columns(
        # Null on either side is unknown rather than unmoved: a source whose trade count was never
        # registered cannot answer the question, and reading it as "did not move" would
        # manufacture the very signature this check exists to detect.
        trades_moved=pl.when(pl.col("num_trades").is_null() | pl.col("source_num_trades").is_null())
        .then(None)
        .otherwise(pl.col("num_trades") != pl.col("source_num_trades")),
        sharpe_moved=pl.when(pl.col("sharpe").is_null() | pl.col("source_sharpe").is_null())
        .then(None)
        .otherwise(pl.col("sharpe") != pl.col("source_sharpe")),
    )
    .select("label", "risk", "num_trades", "source_num_trades", "trades_moved", "sharpe_moved")
    .sort("label", "risk")
)
# Three outcomes, not two. A row is CHANGED when either comparison is true, because one difference
# is enough to establish the control acted. It is UNCHANGED only when both are false and both were
# comparable. Anything else is UNKNOWN - a comparison that could not be made is not evidence of
# sameness, and collapsing it into one would manufacture the signature this check exists to detect.
_changed = overlay_effect.get_column("trades_moved").fill_null(False) | overlay_effect.get_column(
    "sharpe_moved"
).fill_null(False)
_comparable_both = (
    overlay_effect.get_column("trades_moved").is_not_null()
    & overlay_effect.get_column("sharpe_moved").is_not_null()
)
_unchanged = _comparable_both & ~_changed
n_changed, n_unchanged = int(_changed.sum()), int(_unchanged.sum())
n_unknown = overlay_effect.height - n_changed - n_unchanged
print(
    f"{n_changed} of {overlay_effect.height} overlay results differ from the strategy they were "
    f"laid on; {n_unchanged} match it on both compared statistics; {n_unknown} could not be "
    "fully compared"
)
if n_unchanged and not n_changed and not n_unknown:
    print(
        "  No declared control moved either the trade count or the Sharpe. Neither statistic "
        "moving is possible on a calm book, and is also what a control the engine never installed "
        "looks like, so confirm the controls reach the engine before reading it either way."
    )
print(overlay_effect)

set_rows = []
if EXECUTION_TIER == "canonical":
    for label in completed_risk.get_column("label").unique().sort().to_list():
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
            completed_risk.filter(pl.col("label") == label),
            name=f"us-equities-{label_name}-risk-overlay-v1",
        )
        set_rows.append(
            {"label": label, "set_name": result_set.name, "members": len(result_set.members)}
        )

        # The selection pool this label's holdout is chosen from: its baseline and allocation
        # members plus the risk overlays just published. No contract - one label means one
        # label_artifact, and every other protocol field being required-constant is the guard.
        validation_candidates = pl.concat(
            [
                eligible.filter(pl.col("label") == label),
                completed_risk.filter(pl.col("label") == label),
            ]
        ).sort("backtest_hash")
        if (
            validation_candidates.get_column("backtest_hash").n_unique()
            != validation_candidates.height
        ):
            raise ValueError(f"Selection-eligible strategy sets overlap for {label}")
        validation_set = study.backtests.freeze(
            validation_candidates,
            name=VALIDATION_SET_NAME_TEMPLATE.format(label=label_name),
        )
        set_rows.append(
            {
                "label": label,
                "set_name": validation_set.name,
                "members": len(validation_set.members),
            }
        )


compatible_sets = pl.DataFrame(
    set_rows,
    schema={"label": pl.String, "set_name": pl.String, "members": pl.Int64},
)
compatible_sets

# %% [markdown]
# `20_strategy_analysis.py` reopens one of these per-label sets -
# `us-equities-<label>-validation-strategies-v1` - and applies the one official rule: highest
# validation backtest Sharpe with the backtest hash as deterministic tie-break, within that label.
# The holdout follows from that selection with nothing in between: retrain the selected
# configuration on everything up to the holdout start, predict the holdout window, and run the
# same backtest configuration on those predictions.

# %% [markdown]
# ## What to notice
#
# **Check that the overlays moved something before reading what they did.** A result matching the
# unprotected book on both compared statistics has not been shown to change anything, and matching
# across every declared setting is a reason to confirm the controls reach the engine rather than a
# finding about risk control. Neither question is settled by these two columns; a per-control
# trigger count would settle the first, and the backtest does not surface one today.
#
# **An overlay can only remove, so it reshapes a return distribution rather than shifting it.** It
# truncates the left tail by closing losers early and truncates the right by closing winners early,
# and which effect dominates is a property of how the strategy's returns actually arrive. A book
# whose gains come from a few positions running a long way is one an early exit hurts.
#
# **A threshold sweep says more than a chosen threshold.** A control that helps at one setting and
# hurts either side of it has found a feature of this sample. One that improves monotonically as it
# loosens is saying the overlay is worth nothing and the loosest version is closest to not having
# it.
#
# **The time exit is the control worth reading first.** It watches no price, so where it moves the
# result as much as a stop does, what the stops were doing was shortening the holding period rather
# than responding to losses.
#
# **Known limitations.** Overlays are evaluated on one fixed configuration per label, so nothing
# here says whether the same control would help a different strategy. The controls act on bar
# closes, so an intra-bar breach is not seen. And this remains gross of costs, while an overlay
# that fires often adds trades - which is charged in [`19_costs`](19_costs.ipynb).
#
# **Next**: [`19_costs`](19_costs.ipynb) asks how much friction the chosen strategy absorbs.
