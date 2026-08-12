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
# # Latent Factor Model Suite for the US Equities Panel
#
# The US equities panel now runs its latent-factor case study in separate
# notebooks:
#
# - `13a_pca`
# - `13b_ipca`
#
# Cross-model comparison remains in `15_model_analysis`.

# %%
"""Latent factor notebook index for the US equities panel case study."""

import warnings

from case_studies.utils.analytics import load_best_ic_per_family

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"

# %%
best = load_best_ic_per_family(
    families=["latent_factors"],
    case_studies=[CASE_STUDY_ID],
)

if best.is_empty():
    print("No latent-factor results are registered yet for this case study.")
else:
    print(best)
