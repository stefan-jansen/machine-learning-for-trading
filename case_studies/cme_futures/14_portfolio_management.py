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
# # CME Futures: Portfolio Allocation
#
# The signal stage ranks complete configurations by equal-weight validation backtest Sharpe. For
# each label, this notebook retains the strongest checkpoint and signal concentration for each of
# the configured number of distinct model configurations, then evaluates the declared alternative
# allocators. Equal weight is not among them: it is the baseline stage itself, and because
# `stage` is not part of `backtest_hash`, running it again here produces a row hashing
# identically to its baseline parent, so one of the two is silently lost. Measured in this
# case study's own pre-rebuild store: 48 rows stamped `stage='signal'` while carrying
# `allocation.method='equal_weight'`, and no allocation-stage equal-weight rows at all.
#
# All allocator lookbacks come from the case-study configuration. The official population is fixed
# before execution; machine speed and caught failures cannot change which allocators run.

# %%
"""Run the declared CME futures allocation population."""

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    create_label_candidate_sets,
    open_study,
    product_universe_table,
    run_official_backtest_requests,
    shortlist_signal_configurations,
    strategy_request_frame,
)
from case_studies.utils.sweep_config import get_allocators, get_top_n_predictions

# %% [markdown]
# ## Select signal configurations by validation Sharpe
#
# The shortlist is deterministic. It scans the immutable signal candidate set in descending Sharpe
# order with the backtest identity as tie-break, and keeps one exact checkpoint and strategy per
# distinct `(family, config_name)` pair.

# %%
study = open_study(execution_tier="canonical")
universe = product_universe_table()
universe

# %%
shortlist_size = get_top_n_predictions("cme_futures", "allocation")
allocators = get_allocators("cme_futures")
if not allocators:
    raise ValueError("the configured allocator population is empty")
if any(allocation.get("method") == "equal_weight" for allocation in allocators):
    raise ValueError(
        "equal_weight is the baseline stage, not an allocator: `stage` is not part of "
        "`backtest_hash`, so an equal-weight reweight hashes identically to its baseline "
        "parent and one of the two rows is lost. Remove it from the configured menu."
    )

request_rows = []
for label in ALL_LABELS:
    for baseline in shortlist_signal_configurations(
        study,
        label=label,
        limit=shortlist_size,
    ):
        prediction_hash = baseline.registry_record()["prediction_hash"]
        signal = baseline.spec()["strategy"]["signal"]
        for allocation in allocators:
            method = allocation["method"]
            request_rows.append(
                {
                    "request_name": f"{baseline.hash}-{method}",
                    "prediction_hash": prediction_hash,
                    "label": label,
                    "signal": signal,
                    "allocation": allocation,
                    "risk": None,
                    "costs": None,
                    "chapter": "ch17",
                }
            )
requests = strategy_request_frame(request_rows)
requests.select("request_name", "prediction_hash", "label", "signal", "allocation")

# %% [markdown]
# ## Execute and freeze allocation candidates
#
# Moment-based allocators receive only price history before each decision. Product-keyed typed
# decisions retain the selected prediction, roll audit, expiry reference, and allocation settings.

# %%
execution = run_official_backtest_requests(
    study,
    requests,
    population_name="cme_futures-allocation-validation-v1",
)
candidate_sets = create_label_candidate_sets(
    study,
    execution,
    name_prefix="cme-allocation",
)

# %% tags=["results"]
execution.catalog_rows.sort("label", "request_name")

# %% [markdown]
# The next two execution notebooks select the highest validation Sharpe from the union of signal and
# allocation results for each label. Cost sensitivity is diagnostic; risk overlays remain eligible
# for final selection.
