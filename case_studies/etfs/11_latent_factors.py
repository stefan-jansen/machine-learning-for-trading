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
# # Latent Factor Model Suite for ETF Rotation
#
# The ETF case study now runs the Chapter 14 latent-factor models in separate
# notebooks so each model can be inspected on its own terms:
#
# - `11a_pca`
# - `11b_ipca`
# - `11c_conditional_autoencoder`
# - `11d_stochastic_discount_factor`
# - `11e_supervised_autoencoder`
#
# The model-specific notebooks all use the shared `ml4t-models` library path.
# Cross-model comparison stays in `13_model_analysis`.

# %%
"""Latent factor notebook index for the ETF case study."""

import warnings

from case_studies.utils.analytics import load_best_ic_per_family

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"

# %% [markdown]
# ## Current Registered Result
#
# If latent-factor runs have already been registered for the ETF case study,
# the best validation IC is shown here.

# %%
best = load_best_ic_per_family(
    families=["latent_factors"],
    case_studies=[CASE_STUDY_ID],
)

if best.is_empty():
    print("No latent-factor results are registered yet for this case study.")
else:
    print(best)
