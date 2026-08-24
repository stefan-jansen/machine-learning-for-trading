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

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    open_study,
    pre_overlay_candidate_set,
    product_universe_table,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import get_cost_grid_bps

# %% [markdown]
# ## Fixed per-label inputs
#
# The union set fails to open if either the signal or allocation population is missing or partial.
# The selected backtest supplies its exact prediction checkpoint, signal, and allocation settings.

# %%
study = open_study(execution_tier="canonical")
universe = product_universe_table()
universe

# %%
cost_grid = get_cost_grid_bps("cme_futures")
if not cost_grid:
    raise ValueError("the configured cost grid is empty")

request_rows = []
for label in ALL_LABELS:
    selected = pre_overlay_candidate_set(study, label=label).best_validation_sharpe()
    strategy = selected.spec()["strategy"]
    prediction_hash = selected.registry_record()["prediction_hash"]
    for total_cost_bps in cost_grid:
        request_rows.append(
            {
                "request_name": f"{selected.hash}-cost-{total_cost_bps:g}",
                "prediction_hash": prediction_hash,
                "label": label,
                "signal": strategy["signal"],
                "allocation": strategy.get("allocation"),
                "risk": None,
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
    population_name="cme_futures-cost-validation-v1",
)

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# `17_strategy_analysis` may describe the cost curve, but these backtests do not participate in
# configuration selection.
