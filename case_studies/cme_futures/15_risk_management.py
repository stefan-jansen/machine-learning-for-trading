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
    pre_overlay_results,
    product_universe_table,
    rank_by_validation_sharpe,
    run_official_backtest_requests,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import get_position_risk_controls

# %% tags=["parameters"]
EXECUTION_TIER = "canonical"
WORKSPACE: str | None = None
PREVIEW_LABELS: list[str] = []

# %% [markdown]
# ## Fixed per-label inputs and risk rules
#
# No candidate cap or runtime-dependent skip is allowed. The configured list is the population.

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

# %%
risk_controls = get_position_risk_controls("cme_futures")
if not risk_controls:
    raise ValueError("the configured position-risk population is empty")

request_rows = []
for label in labels:
    selected = rank_by_validation_sharpe(
        study, pre_overlay_results(study, label=label, execution_tier=EXECUTION_TIER)
    )[0]
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
    population_name="cme_futures-risk-validation-v1" if EXECUTION_TIER == "canonical" else None,
)
candidate_sets = (
    create_label_candidate_sets(study, execution, stage="risk")
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
# Final selection in `19_strategy_analysis` uses the union of signal, allocation, and risk-overlay
# results. Cost-sensitivity rows are excluded.
