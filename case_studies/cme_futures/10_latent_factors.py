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
# # CME Futures: Latent-Factor Requests
#
# The latent-factor stage contains two declared configurations. `10a_pca` fits principal components
# within each training fold. `10b_stochastic_discount_factor` estimates the neural stochastic
# discount factor within the same fold contract. Neither notebook selects by IC.
#
# This index exposes the complete request population without launching either computation. The two
# execution notebooks publish disjoint official populations that `13_backtest` later combines with
# the other predictive families.

# %%
"""Show the declared CME futures latent-factor requests."""

from case_studies.cme_futures.research_workflow import (
    ALL_LABELS,
    model_request_catalog,
    product_universe_table,
)

# %%
requests = model_request_catalog("latent_factors", labels=ALL_LABELS)
universe = product_universe_table()
universe

# %%
requests.sort("label", "config_name")
