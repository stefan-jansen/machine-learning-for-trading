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
# # CME Futures: Risk Overlays
#
# For each return horizon, this notebook selects the highest validation Sharpe from the immutable
# union of signal and allocation results, then applies every position-level risk rule declared in
# the case-study configuration. Stop-loss, trailing-stop, and time-exit parameters are fixed before
# the validation backtest. They are not calibrated from the same validation price path they assess.
#
# Risk rules execute inside the existing futures engine after product-keyed target decisions cross
# the typed boundary. Every declared rule must finish, and the resulting per-label candidate sets
# remain eligible for final validation selection.

# %%
"""Run the declared CME futures risk-overlay population."""

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    create_label_candidate_sets,
    open_study,
    pre_overlay_candidate_set,
    product_universe_table,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import get_position_risk_controls

# %% [markdown]
# ## Fixed per-label inputs and risk rules
#
# No candidate cap or runtime-dependent skip is allowed. The configured list is the population.

# %%
study = open_study(execution_tier="canonical")
universe = product_universe_table()
universe

# %%
risk_controls = get_position_risk_controls("cme_futures")
if not risk_controls:
    raise ValueError("the configured position-risk population is empty")

request_rows = []
for label in ALL_LABELS:
    selected = pre_overlay_candidate_set(study, label=label).best_validation_sharpe()
    strategy = selected.spec()["strategy"]
    prediction_hash = selected.registry_record()["prediction_hash"]
    for control in risk_controls:
        rule = {key: value for key, value in control.items() if key != "name"}
        request_rows.append(
            {
                "request_name": f"{selected.hash}-risk-{control['name']}",
                "prediction_hash": prediction_hash,
                "label": label,
                "signal": strategy["signal"],
                "allocation": strategy.get("allocation"),
                "risk": {"position_rules": [rule]},
                "costs": None,
                "chapter": "ch19",
            }
        )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "risk")

# %% [markdown]
# ## Execute and freeze risk candidates
#
# Each request carries the fitted prediction checkpoint, product decisions, fold-transition policy,
# contract and roll inputs, and one risk rule. Missing members fail before the candidate set exists.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name="cme_futures-risk-validation-v1",
)
candidate_sets = create_label_candidate_sets(
    study,
    execution,
    name_prefix="cme-risk",
)

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# Final selection in `17_strategy_analysis` uses the union of signal, allocation, and risk-overlay
# results. Cost-sensitivity rows are excluded.
