# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Latent Factor Model Suite for CME Futures
#
# The CME futures case study now runs its latent-factor models in separate
# notebooks:
#
# - `10a_pca`
# - `10b_stochastic_discount_factor`
#
# Cross-model comparison remains in `12_model_analysis`.

# %%
"""Latent factor notebook index for the CME futures case study."""

import warnings

from case_studies.utils.analytics import load_best_ic_per_family

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "cme_futures"

# %%
best = load_best_ic_per_family(
    families=["latent_factors"],
    case_studies=[CASE_STUDY_ID],
)

if best.is_empty():
    print("No latent-factor results are registered yet for this case study.")
else:
    print(best)
