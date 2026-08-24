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
# coverage, and returns the corresponding catalog rows.
#
# Predictive interpretation belongs in [`15_model_analysis`](15_model_analysis.ipynb). This index
# does not compare metrics or choose between PCA and IPCA.
#
# **Learning objectives**
#
# - Distinguish the PCA and IPCA requests used by this case study.
# - Select complete latent-factor results through the shared catalog boundary.
# - Trace labels, checkpoints, and cross-validation identities without copying hashes.
#
# **Book reference**: Chapter 13
#
# **Prerequisites**: `13a_pca.py` and `13b_ipca.py` publish the latent-factor results indexed here.

# %%
"""Reference index for the latent-factor execution notebooks."""

import os
from pathlib import Path

import polars as pl

from case_studies.research import Study

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"

# %% [markdown]
# ## Result catalog
#
# Canonical execution reads the released study. Preview execution reads an isolated workspace.
# Ordinary Polars filters select complete latent prediction rows by label, configuration,
# checkpoint, and protocol. Hashes remain visible for provenance, but readers pass selected rows or
# named compatible sets to downstream code rather than copying them.

# %%
if EXECUTION_TIER == "canonical":
    study = Study.open(CASE_STUDY_ID)
elif EXECUTION_TIER == "preview":
    study = Study.open(
        CASE_STUDY_ID,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
    )
else:
    raise ValueError(f"Unsupported execution tier: {EXECUTION_TIER!r}")

latent_results = (
    study.predictions.table(include_preview=EXECUTION_TIER == "preview")
    .filter(
        (pl.col("family") == "latent_factors")
        & (pl.col("split") == "validation")
        & (pl.col("execution_tier") == EXECUTION_TIER)
        & pl.col("complete")
    )
    .select(
        "label",
        "config_name",
        "checkpoint_kind",
        "checkpoint_value",
        "cv_identity",
        "training_hash",
        "prediction_hash",
    )
    .sort("label", "config_name", "checkpoint_kind", "checkpoint_value")
)
latent_results

# %% [markdown]
# ## Handoff
#
# Run the PCA and IPCA execution notebooks to create or reuse their exact result identities. Model
# analysis reopens the named compatible sets for each label protocol. Strategy evaluation receives
# every selected catalog row and checkpoint; this index makes no predictive or portfolio decision.

# %% [markdown]
# ## Key takeaways and limitations
#
# - PCA and IPCA expose the same result and catalog interfaces while fitting different factor
#   structures.
# - Catalog fields provide the reader-facing selection boundary; hashes remain provenance keys.
# - This index summarizes available result identities. Estimation details and fitted-state checks
#   remain in the two execution notebooks.
