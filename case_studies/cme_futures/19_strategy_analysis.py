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
# # CME Futures: Strategy Analysis
#
# The four preceding notebooks each produced a registered, complete population of validation
# backtests: the equal-weight signal baseline over every model configuration and checkpoint, the
# alternative allocators over the signal shortlist, the transaction-cost grid, and the position-risk
# overlays. This notebook reads those registered results and names one case-study configuration
# from them.
#
# A case-study configuration is a model configuration - family, settings, and the training
# checkpoint the predictions came from - together with the signal, sizing, and risk rules applied to
# it. The configuration with the highest validation Sharpe across the signal, allocation, and
# risk-overlay stages is the one selected. Cost-sensitivity backtests vary the friction assumption
# on a configuration already chosen, so they describe it rather than compete with it and are not in
# the pool.
#
# Both return horizons are selected from together. The horizon is part of the configuration, so it
# is read off the row that wins rather than fixed before looking.
#
# The holdout period is a later date range that no notebook up to this point has touched. It is
# evaluated on the selected configuration alone, one time, and it may disagree with the validation
# result. That disagreement is an outcome to report, not a reason to select again.
#
# Prerequisites: `13_backtest`, `14_portfolio_management`, `15_risk_management`, and `16_costs`.

# %%
"""Select and describe one CME futures case-study configuration."""

import plotly.express as px
import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    final_selection_candidate_set,
    final_validation_candidate_set,
    final_validation_results,
    open_study,
    product_universe_table,
    rank_by_validation_sharpe,
    selection_catalog,
)
from case_studies.research import OfficialPopulation, Result
from case_studies.utils.strategy_analysis import select_holdout_self_backtest
from utils.style import COLORS

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []

# %% [markdown]
# ## The pool the configuration is selected from
#
# Each per-label pool opens the three stage populations and fails if any member is missing,
# incomplete, or produced under a preview identity. Combining the two horizons into one immutable
# set records exactly which results were compared, so the selection can be repeated later against
# the same members rather than against whatever the registry holds at the time.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS:
        raise ValueError("canonical execution cannot declare preview reductions")
    labels = ALL_LABELS
elif EXECUTION_TIER == "preview":
    if WORKSPACE is None or not PREVIEW_LABELS:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_LABELS")
    unknown = sorted(set(PREVIEW_LABELS) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"preview labels this case study does not declare: {unknown}")
    labels = tuple(PREVIEW_LABELS)
else:
    raise ValueError(f"unsupported execution tier: {EXECUTION_TIER!r}")
universe = product_universe_table()
universe

# %% [markdown]
# Only a canonical pool is an immutable set. A preview run publishes no candidate set - one cannot
# hold a preview member - so its pool is the rows its own reduced execution produced and the
# `candidate_set_hash` column below is null. Everything downstream, the ranking rule included, is
# the same either way; what differs is whether the pool can be reopened later by name.

# %%
if EXECUTION_TIER == "canonical":
    per_label = {label: final_validation_candidate_set(study, label=label) for label in labels}
    per_label_results = {
        label: tuple(Result.open(study, value) for value in pool_set.members)
        for label, pool_set in per_label.items()
    }
    candidates = final_selection_candidate_set(study)
    pool_results = tuple(Result.open(study, value) for value in candidates.members)
    pool_identity = candidates.hash
    per_label_identity = {label: pool_set.hash for label, pool_set in per_label.items()}
else:
    per_label_results = {
        label: final_validation_results(study, label=label, execution_tier=EXECUTION_TIER)
        for label in labels
    }
    pool_results = tuple(result for results in per_label_results.values() for result in results)
    pool_identity = None
    per_label_identity = dict.fromkeys(labels)
pool = selection_catalog(study, (result.hash for result in pool_results))
pool_size = pl.DataFrame(
    [
        {
            "label": label,
            "candidates": len(results),
            "candidate_set_hash": per_label_identity[label],
        }
        for label, results in per_label_results.items()
    ],
    schema={"label": pl.String, "candidates": pl.Int64, "candidate_set_hash": pl.String},
).sort("label")

# %% tags=["results"]
pool_size

# %% [markdown]
# ## What each selection stage contributed
#
# The three stages run in sequence, each on the survivors of the one before. The baseline stage
# carries every configuration and checkpoint at equal weight. Allocation runs on the strongest
# distinct configurations from that stage, and the risk overlay on the strongest result so far for
# each horizon. Later stages therefore hold far fewer candidates than the first, and the spread
# within a stage shows how much of the outcome the sizing and risk rules decide once the model is
# fixed.
#
# **The medians cannot be read across stages.** Each stage runs on the survivors of the one before,
# chosen on the same validation Sharpe the table reports, so the pool shrinks from 496 to 60 to 14
# by selecting on the quantity being summarized. On `fwd_ret_21d` the median rises from -0.392 to
# 0.192 to 1.010 along that shrinking pool, and almost all of that movement is the selection, not
# the position sizing methods or the risk rules. What the stages do support is the comparison
# within a row: the fourteen risk overlays share one model, one signal and one sizing rule,
# and they still span 0.322 to 1.274, which is the range the position rule alone is
# responsible for.

# %%
stage_summary = (
    pool.group_by("label", "stage")
    .agg(
        pl.len().alias("candidates"),
        pl.col("sharpe").min().alias("min_sharpe"),
        pl.col("sharpe").median().alias("median_sharpe"),
        pl.col("sharpe").max().alias("max_sharpe"),
    )
    .sort("label", "stage")
)

# %% tags=["results"]
stage_summary

# %% tags=["results"]
fig = px.strip(
    pool.to_pandas(),
    x="stage",
    y="sharpe",
    color="label",
    stripmode="overlay",
    category_orders={"stage": ["signal", "allocation", "risk_overlay"]},
    color_discrete_sequence=[COLORS["blue"], COLORS["amber"]],
    labels={"stage": "Selection stage", "sharpe": "Validation Sharpe", "label": "Return horizon"},
)
fig.update_layout(
    title="Validation Sharpe by selection stage and return horizon",
    height=420,
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# ## The selected configuration
#
# `best_validation_sharpe` ranks the immutable set by Sharpe and breaks a tie on the backtest
# identity, so the same pool always names the same row. The prediction checkpoint is part of that
# identity: two rows from the same trained model at different checkpoints are different
# configurations. A holdout matched on the trained model alone can therefore land on a different
# checkpoint from the one selected, and report a different result for what looks like the same
# model, which is why the checkpoint travels with the selection into the lock below.

# %%
selected = rank_by_validation_sharpe(study, pool_results)[0]
selected_row = pool.filter(pl.col("backtest_hash") == selected.hash)
selected_label = selected_row.item(0, "label")
selected_strategy = selected.spec()["strategy"]

# %% tags=["results"]
selected_row

# %% tags=["results"]
pl.DataFrame(
    [
        {
            "candidate_set_hash": pool_identity,
            "candidates_compared": len(pool_results),
            "label": selected_label,
            "signal": str(selected_strategy["signal"]),
            "allocation": str(selected_strategy.get("allocation")),
            "risk": str(selected_strategy.get("risk")),
        }
    ]
)

# %% [markdown]
# ## What friction costs this configuration
#
# The cost grid was run on the per-horizon leader of the signal and allocation stages, holding the
# model, sizing, and contract specification fixed and varying only the all-in cost assumption.
# Commission and slippage each take half of the quoted figure. The curve for the selected horizon
# shows how the validation result changes as the assumed fill gets worse.

# %%
if EXECUTION_TIER == "canonical":
    cost_population = OfficialPopulation.one(study, name="cme_futures-cost-validation-v1")
    cost_members = list(cost_population.require_complete())
else:
    cost_members = (
        study.backtests.table(include_preview=True)
        .filter(
            (pl.col("execution_tier") == "preview")
            & (pl.col("stage") == "cost_sensitivity")
            & pl.col("complete")
        )
        .get_column("backtest_hash")
        .to_list()
    )
cost_curve = (
    study.backtests.table(include_preview=True)
    .filter(pl.col("backtest_hash").is_in(cost_members) & (pl.col("label") == selected_label))
    .with_columns(
        (
            pl.col("spec_json")
            .str.json_path_match("$.decision_artifact.parameters.costs.commission_bps")
            .cast(pl.Float64)
            + pl.col("spec_json")
            .str.json_path_match("$.decision_artifact.parameters.costs.slippage_bps")
            .cast(pl.Float64)
        ).alias("total_cost_bps")
    )
    .select("total_cost_bps", "sharpe", "total_return", "num_trades", "backtest_hash")
    .sort("total_cost_bps")
)
if cost_curve.is_empty():
    raise RuntimeError(f"the cost population contains no member for {selected_label!r}")
# `json_path_match` returns null for a path that is not in the document rather than raising, and
# null + null is null, so reading the grid value from the wrong place yields a full-height frame
# whose cost axis is entirely missing. The emptiness check above passes on such a frame and the
# curve below plots against nothing. Refuse it here instead.
if cost_curve.get_column("total_cost_bps").null_count():
    raise RuntimeError(
        "cost members record no all-in cost at "
        "$.decision_artifact.parameters.costs; the cost curve has no axis"
    )

# %% tags=["results"]
cost_curve

# %% tags=["results"]
fig = px.line(
    cost_curve.to_pandas(),
    x="total_cost_bps",
    y="sharpe",
    markers=True,
    labels={"total_cost_bps": "All-in cost (bps per trade)", "sharpe": "Validation Sharpe"},
    color_discrete_sequence=[COLORS["copper"]],
)
fig.update_layout(
    title="Validation Sharpe across the all-in transaction-cost grid",
    height=380,
)
fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
fig.show()

# %% [markdown]
# ## The holdout
#
# The holdout evaluates one configuration: the one the validation backtests selected. That is the
# highest validation backtest Sharpe across the baseline, position-sizing, allocation and
# risk-management stages, and it is fixed before any holdout artifact exists.
#
# What keeps the holdout from becoming an axis to search over is the direction of that rule, not a
# gate. The ranking reads validation rows only, and the holdout row below is found by matching the
# selected strategy specification - never by taking whichever holdout backtest scored best. A
# holdout number therefore cannot change which configuration is reported here.
#
# Nothing about it is one-shot. A holdout result that turns out to be wrong is deleted and produced
# again; what would make the number uninterpretable is evaluating many configurations on the window
# and reporting the best, which is the thing the selection rule rules out. The holdout notebooks
# produce the row; this one reads it. Where they have not run, the table is empty and the validation
# result above stands on its own.

# %% tags=["results"]
# `select_holdout_self_backtest` is the shared resolver every strategy-analysis notebook uses.
# It takes the selection this notebook already made and finds the holdout backtest replaying that
# same strategy specification, at the same configuration and checkpoint, over a training run whose
# own CV declares the holdout fold. It returns None where no such run exists, and raises rather
# than choosing where two of them do.
#
# Calling it rather than re-deriving the lineage here is deliberate. A second implementation
# living beside the first agrees with it on the registry it was written against and diverges on
# the next one, and a divergence in this particular lookup is a holdout number attributed to the
# wrong configuration.
holdout_backtest_hash = select_holdout_self_backtest("cme_futures", selected.hash)
print(
    f"Selected validation backtest: {selected.hash}  ({selected_label})\n"
    f"Holdout replay: {holdout_backtest_hash or 'not produced yet'}"
)

# %% tags=["results"]
if holdout_backtest_hash is None:
    comparison = pl.DataFrame()
else:
    evaluated = study.backtests.table(include_preview=True).filter(
        pl.col("backtest_hash") == holdout_backtest_hash
    )
    comparison = pl.concat(
        [
            selected_row.select("label", "sharpe", "max_drawdown", "num_trades").with_columns(
                pl.lit("validation").alias("split")
            ),
            evaluated.select("label", "sharpe", "max_drawdown", "num_trades").with_columns(
                pl.lit("holdout").alias("split")
            ),
        ]
    ).select("split", "label", "sharpe", "max_drawdown", "num_trades")
comparison
