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
# # CME Futures: Transaction-Cost Sensitivity
#
# For each return horizon, this notebook selects the highest validation Sharpe from the immutable
# union of equal-weight signal and allocation results. It then applies the declared all-in cost grid
# to that fixed configuration. Commission and slippage each receive half of the grid value.
#
# Cost sensitivity is not a selection stage. Its rows are excluded from the final selection pool.
# Contract multipliers, tick sizes, margin rates, front-contract position, roll adjustment, and
# product identity remain unchanged across the grid.

# %%
"""Run CME futures transaction-cost sensitivity on fixed configurations."""

import json

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    open_study,
    product_universe_table,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.strategy_analysis import resolve_solvent_carrier
from case_studies.utils.sweep_config import get_cost_grid_bps

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []

# %% [markdown]
# ## Fixed inputs
#
# There is one configuration to price, and the shared carrier selector below decides which
# label it is on. `PREVIEW_LABELS` is still validated rather than ignored: a preview run
# that names a label the case study does not declare is a mistake worth stopping, even
# though nothing here loops over the set.

# %%
study = open_study(execution_tier=EXECUTION_TIER, workspace=WORKSPACE)
if EXECUTION_TIER == "canonical":
    if PREVIEW_LABELS:
        raise ValueError("canonical execution cannot declare preview reductions")
elif EXECUTION_TIER == "preview":
    if WORKSPACE is None or not PREVIEW_LABELS:
        raise ValueError("preview execution requires WORKSPACE and PREVIEW_LABELS")
    unknown = sorted(set(PREVIEW_LABELS) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"preview labels this case study does not declare: {unknown}")
else:
    raise ValueError(f"unsupported execution tier: {EXECUTION_TIER!r}")
universe = product_universe_table()
universe

# %%
cost_grid = get_cost_grid_bps("cme_futures")
if not cost_grid:
    raise ValueError("the configured cost grid is empty")

# The configuration this case study reports, resolved once by the shared selector rather
# than ranked again here. `resolve_solvent_carrier` goes through
# `resolve_canonical_rank1_lineage`, which ranks across the signal, allocation and
# risk-overlay stages together, re-ranks conformal candidates on exact common timestamp
# support and applies LABEL_RESTRICTIONS, UNIVERSE_RESTRICTIONS and CARRIER_PINS. A plain
# Sharpe ranking beside it does none of those, and where the two disagree this notebook
# prices a strategy the chapter does not report while `19_strategy_analysis` finds no cost
# rows for the carrier it selected. It also refuses a carrier whose equity reached zero,
# whose Sharpe would be computed on a balance that no longer exists.
#
# This replaces a per-label loop that ran a cost grid for each label off the pre-overlay
# allocation results. That was wrong twice over: there is one strategy, not one per label,
# and it read the stage before risk management, so the ladder priced the pre-overlay winner
# rather than the configuration that is actually shipped.
carrier = resolve_solvent_carrier("cme_futures")
strategy = json.loads(carrier["spec_json"])["strategy"]
selected_label = carrier["label"]
prediction_hash = carrier["val_prediction_hash"]
print(
    f"Pricing the canonical validation rank-1: {carrier['val_backtest_hash']} "
    f"({carrier['family']}/{carrier['config_name']}) on {selected_label}, "
    f"from the {carrier['val_stage']} stage, validation Sharpe {carrier['val_sharpe']:.3f}, "
    f"max drawdown {carrier['max_drawdown']:.3f}."
)
print(
    f"  It carries {'a' if strategy.get('risk') else 'no'} risk overlay, and that block "
    "travels with it below rather than being cleared."
)

request_rows = []
for total_cost_bps in cost_grid:
    request_rows.append(
        {
            "request_name": f"{carrier['val_backtest_hash']}-cost-{total_cost_bps:g}",
            "prediction_hash": prediction_hash,
            "label": selected_label,
            "signal": strategy["signal"],
            "allocation": strategy.get("allocation"),
            # Carried, not cleared. The configuration being priced is the one that came out
            # of risk management, and dropping its overlay here would price a strategy
            # nobody selected.
            "risk": strategy.get("risk"),
            "costs": {
                "commission_bps": total_cost_bps / 2,
                "slippage_bps": total_cost_bps / 2,
            },
            "chapter": "ch18",
        }
    )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "costs")

# %% [markdown]
# ## Execute the declared grid
#
# Expected identities are snapshotted before execution. An empty input, failed grid member, missing
# sidecar, or incomplete lineage fails the notebook instead of reporting a smaller population.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name="cme_futures-cost-validation-v1" if EXECUTION_TIER == "canonical" else None,
)

# %% [markdown]
# `source` says whether each member was computed by this run or served from the registry because
# an identical identity was already recorded. A re-run of a registered sweep is entirely `reused`
# and completes in seconds; without the column that is indistinguishable from having computed
# every row.

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# `19_strategy_analysis` may describe the cost curve, but these backtests do not participate in
# configuration selection.
