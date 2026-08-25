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
# # CME Futures: Equal-Weight Signal Backtests
#
# This notebook sends every complete model configuration and checkpoint through the same signal
# baseline. At each weekly decision, the signal ranks products by the selected prediction row and
# holds equal-weight long and short groups for each configured concentration. This is
# `stage='signal'`; equal weight is not an allocation method in the next stage.
#
# Reader-facing prices and decisions use `product`. The shared boundary records the front-contract
# position, raw-to-adjusted roll identity, cumulative-ratio transitions, expiry reference, contract
# specifications, prediction lineage, and state-transition policy before converting `product` to
# the existing engine's internal `symbol` key.

# %%
"""Run the complete CME futures equal-weight validation baseline."""

import polars as pl

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    MODEL_POPULATION_NAMES,
    create_label_candidate_sets,
    load_futures_price_path,
    official_prediction_catalog,
    open_study,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import get_top_k_values_for

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []
PREVIEW_MAX_PREDICTIONS = 0

# %% [markdown]
# ## Futures data used by the strategy
#
# The model predicts a continuous front-contract return. `raw_close` is the traded contract level;
# `adj_close` is the multiplicatively back-adjusted level used for continuous returns. A change in
# `cum_ratio` identifies a roll transition. The backtest consumes adjusted OHLC while retaining this
# audit table and the product expiry rules in its identity.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS or PREVIEW_MAX_PREDICTIONS:
        raise ValueError("canonical execution cannot declare preview reductions")
    labels = ALL_LABELS
elif EXECUTION_TIER == "preview":
    if WORKSPACE is None or not PREVIEW_LABELS or PREVIEW_MAX_PREDICTIONS < 1:
        raise ValueError(
            "preview execution requires WORKSPACE, PREVIEW_LABELS and PREVIEW_MAX_PREDICTIONS"
        )
    unknown = sorted(set(PREVIEW_LABELS) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"preview labels this case study does not declare: {unknown}")
    labels = tuple(PREVIEW_LABELS)
else:
    raise ValueError(f"unsupported execution tier: {EXECUTION_TIER!r}")
price_paths = {label: load_futures_price_path(label) for label in labels}
market_rows = []
for label, path in price_paths.items():
    roll_counts = path.roll_transitions.group_by("product").len().rename({"len": "rolls"})
    market_rows.append(
        path.audit.group_by("product")
        .agg(
            pl.col("timestamp").min().alias("first_session"),
            pl.col("timestamp").max().alias("last_session"),
        )
        .join(roll_counts, on="product", how="left")
        .join(path.expiry_rules, on="product", how="left")
        .with_columns(pl.lit(label).alias("label"), pl.col("rolls").fill_null(0))
    )
market_contract = pl.concat(market_rows).sort("label", "product")

# %% tags=["results"]
market_contract

# %% [markdown]
# ## Complete baseline requests
#
# The source rows come from the six official model populations. Each population must be complete
# before this cell can construct a request. No registry ordering, row cap, cached metric, or caught
# failure can remove a candidate.

# %%
if EXECUTION_TIER == "canonical":
    predictions = official_prediction_catalog(study, MODEL_POPULATION_NAMES)
else:
    predictions = (
        study.predictions.table(include_preview=True)
        .filter(
            (pl.col("execution_tier") == "preview")
            & (pl.col("split") == "validation")
            & pl.col("complete")
            & pl.col("label").is_in(list(labels))
        )
        .sort("label", "family", "config_name", "checkpoint_kind", "checkpoint_value")
        .head(PREVIEW_MAX_PREDICTIONS)
    )
    if predictions.is_empty():
        raise RuntimeError("preview execution found no complete validation predictions to backtest")
request_rows = []
for label in labels:
    label_catalog = predictions.filter(pl.col("label") == label)
    n_products = price_paths[label].prices.get_column("product").n_unique()
    for row in label_catalog.iter_rows(named=True):
        for top_k in get_top_k_values_for("cme_futures", label, n_products):
            request_rows.append(
                {
                    "request_name": f"{row['prediction_hash']}-equal-weight-k{top_k}",
                    "prediction_hash": row["prediction_hash"],
                    "label": label,
                    "signal": {"method": "equal_weight_top_k", "top_k": top_k},
                    "allocation": None,
                    "risk": None,
                    "costs": None,
                    "chapter": "ch16",
                }
            )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "signal")

# %% [markdown]
# ## Execute and freeze the comparable sets
#
# Target weights are canonical typed decisions with unique `product,timestamp` keys and exact
# prediction eligibility. Expected backtest identities are snapshotted before the engine runs.
# Every member must finish before the per-label validation candidate sets are created.
#
# ### What happens at a fold boundary, and what it costs to read
#
# The five validation folds are consecutive stretches of calendar time, and this backtest runs
# through them as one series of weekly decisions rather than as five separate simulations. At a
# boundary the position **carries**; it is not flattened. The declared policy is
# `StateTransitionPolicy(fold_boundary="continue")`.
#
# Two reasons, and the second is the harder one. Nothing happens in the market on the four dates
# that separate the folds - they are an index this case study cut for evaluation, not events - so
# flattening there would be an artifact of how the sample was divided. And the liquidation could
# not be executed here in any case: the schedule decides on Friday's close and fills at Monday's
# open, so there is no weight row for the engine to snap a reset onto, and it refuses to snap one
# forward rather than carry the old state across the boundary and then charge a round trip for no
# change in exposure.
#
# **So a per-fold number in this pipeline is not computed from a flat start.** A fold inherits at
# most one week of exposure from the fold before it. That is four of roughly 260 weekly decisions,
# about 1.5% of them, and it is the reason not to read a per-fold Sharpe here or downstream as
# though the fold were a standalone track record. The alternative - same-bar execution, which would
# buy the flat start - is what makes a futures backtest implausible, and it is not a trade worth
# making for four decisions.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name=("cme_futures-signal-validation-v1" if EXECUTION_TIER == "canonical" else None),
)
# A candidate set is canonical too - `CandidateSet.create` refuses a preview member
# (research/comparison.py:50-51) - so a preview run leaves the funnel's named pools alone and
# the notebooks downstream read its backtest catalog directly instead.
candidate_sets = (
    create_label_candidate_sets(study, execution, stage="signal")
    if EXECUTION_TIER == "canonical"
    else {}
)

# %% [markdown]
# `source` says whether each member was computed by this run or served from the registry because
# an identical identity was already recorded. A re-run of a registered sweep is entirely `reused`
# and completes in seconds; without the column that is indistinguishable from having computed
# every row.

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# `14_portfolio_management` ranks each immutable per-label set by validation backtest Sharpe.
# `17_strategy_analysis` interprets the validated strategy results.
