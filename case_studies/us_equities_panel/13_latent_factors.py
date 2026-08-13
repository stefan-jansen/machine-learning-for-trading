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
# # Latent Factor Requests for the US Equities Panel
#
# This index separates the two latent-factor computations while keeping their outputs under the
# same public result contract. [`13a_pca`](13a_pca.ipynb) estimates statistical factors from the
# finalized feature panel. [`13b_ipca`](13b_ipca.ipynb) estimates characteristic-conditioned
# factors. Each notebook constructs its own resolved request, validates fitted state and prediction
# coverage, and prints the immutable training and prediction hashes it produces.
#
# Predictive interpretation belongs in [`15_model_analysis`](15_model_analysis.ipynb). This index
# does not compare metrics or choose between PCA and IPCA.

# %%
"""Reference index for the latent-factor execution notebooks."""

from case_studies.research import Study

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"

# %% [markdown]
# ## Result references
#
# The released study is read-only here. A hash printed by either execution notebook can be opened
# with `study.results.open(result_hash)`. The returned result exposes its specification, protocol,
# lineage, coverage, and artifacts without a registry-wide metric query.

# %%
study = Study.open(CASE_STUDY_ID)
result_catalog = study.results
print(f"Result catalog ready for {study.case_study}: {result_catalog.__class__.__name__}")

# %% [markdown]
# ## Handoff
#
# Run the PCA and IPCA execution notebooks to create or reuse their exact result identities. Pass
# their prediction hashes to the model-analysis notebook as an explicit compatible set. Strategy
# evaluation later receives every complete configuration and checkpoint; this index makes no
# predictive or portfolio decision.
