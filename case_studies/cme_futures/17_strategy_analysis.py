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
# Prerequisites: `13_backtest`, `14_portfolio_management`, `15_costs`, and `16_risk_management`.

# %%
"""Select and describe one CME futures case-study configuration."""

import plotly.express as px
import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    final_selection_candidate_set,
    final_validation_candidate_set,
    holdout_evidence,
    open_study,
    product_universe_table,
    selection_catalog,
)
from case_studies.research import OfficialPopulation
from utils.style import COLORS

# %% [markdown]
# ## The pool the configuration is selected from
#
# Each per-label pool opens the three stage populations and fails if any member is missing,
# incomplete, or produced under a preview identity. Combining the two horizons into one immutable
# set records exactly which results were compared, so the selection can be repeated later against
# the same members rather than against whatever the registry holds at the time.

# %%
study = open_study(execution_tier="canonical")
universe = product_universe_table()
universe

# %%
per_label = {label: final_validation_candidate_set(study, label=label) for label in ALL_LABELS}
candidates = final_selection_candidate_set(study)
pool = selection_catalog(study, candidates)
pool_size = pl.DataFrame(
    [
        {"label": label, "candidates": len(pool_set.members), "candidate_set_hash": pool_set.hash}
        for label, pool_set in per_label.items()
    ]
).sort("label")

# %% tags=["results"]
pool_size

# %% [markdown]
# ## What each selection stage contributed
#
# The three stages run in sequence, each on the survivors of the one before. The signal stage
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
# the allocators or the risk rules. What the stages do support is the comparison within a row: the
# fourteen risk overlays share one model, one signal, and one sizing rule, and they still span
# 0.322 to 1.274, which is the range the position rule alone is responsible for.

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
selected = candidates.best_validation_sharpe()
selected_row = pool.filter(pl.col("backtest_hash") == selected.hash)
selected_label = selected_row.item(0, "label")
selected_strategy = selected.spec()["strategy"]

# %% tags=["results"]
selected_row

# %% tags=["results"]
pl.DataFrame(
    [
        {
            "candidate_set_hash": candidates.hash,
            "candidates_compared": len(candidates.members),
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
cost_population = OfficialPopulation.one(study, name="cme_futures-cost-validation-v1")
cost_members = cost_population.require_complete()
cost_curve = (
    study.backtests.table()
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
# The research lock is what allows the holdout to be used at all. It records the candidate set, the
# selected validation backtest, and the retraining contract - the same specification, differing only
# in the cross-validation interval that extends through the holdout period - before any holdout
# artifact exists. The lifecycle then admits one evaluation against that lock and refuses a second,
# so the holdout cannot become an axis to search over.
#
# The lock and its single evaluation are created by the lifecycle path, not here. This notebook
# reads what that path recorded and reports the selected configuration's holdout result beside its
# validation result. Where the lifecycle has not yet been locked, the table below is empty and the
# validation result above stands on its own.

# %%
holdout = holdout_evidence(study)
if holdout.height > 1:
    raise RuntimeError("the lifecycle holds more than one research lock")
if not holdout.is_empty():
    locked = holdout.item(0, "validation_backtest_hash")
    if locked != selected.hash:
        raise RuntimeError(
            f"the research lock was created from backtest {locked}, "
            f"not the configuration this pool selects, {selected.hash}"
        )
    if holdout.item(0, "label") != selected_label:
        raise RuntimeError("the research lock records a different return horizon")

# %% tags=["results"]
holdout

# %% tags=["results"]
if holdout.is_empty() or holdout.item(0, "holdout_backtest_hash") is None:
    comparison = pl.DataFrame()
else:
    evaluated = study.backtests.table().filter(
        pl.col("backtest_hash") == holdout.item(0, "holdout_backtest_hash")
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
