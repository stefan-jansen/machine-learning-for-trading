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
# # PCA for ETF Rotation
#
# PCA is the persistent-ID baseline in the latent-factor suite. For ETFs that
# assumption is valid, so we can evaluate whether the return-panel factor
# baseline adds anything over direct predictors.

# %%
"""ETF PCA case-study run via the shared latent-factor library path."""

import warnings

from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
    run_case_study_variants,
)

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "etfs"
PRIMARY_LABEL = ""
MAX_SYMBOLS = 0
N_FACTORS = 5
N_EPOCHS = 50
USE_CACHE = True
FORCE_RETRAIN = False
MAX_FOLDS = 0
MAX_VARIANT_LABELS = -1
RUN_VARIANTS = True
USE_MACRO = False
MODEL_NAME = "pca"

# %% tags=["injected-parameters"]
# Parameters
USE_CACHE = False
FORCE_RETRAIN = True


# %%
context = load_case_study_context(
    CASE_STUDY_ID,
    primary_label=PRIMARY_LABEL,
    max_symbols=MAX_SYMBOLS,
    max_folds=MAX_FOLDS,
    max_variant_labels=MAX_VARIANT_LABELS,
    use_macro=USE_MACRO,
)
available_models = configured_models(context)
if MODEL_NAME not in available_models:
    raise ValueError(f"{MODEL_NAME!r} is not configured for {CASE_STUDY_ID}")

print(f"Case study: {CASE_STUDY_ID}")
print(f"Model: {MODEL_NAME}")
print(f"Primary label: {context.primary_label}")
print(f"Variant labels: {context.variant_labels}")
print(f"Dataset rows: {len(context.dataset):,}")
print(f"Features: {len(context.feature_names)}")
print(f"Splits: {len(context.splits)}")

# %% [markdown]
# ## Run Walk-Forward CV

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="11a_pca",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
)

print(result["model_results"])
print(result["fold_metrics"][MODEL_NAME])

# %% [markdown]
# ## Variant Labels

# %%
variant_results = {}
if RUN_VARIANTS and context.variant_labels:
    variant_results = run_case_study_variants(
        context,
        model_name=MODEL_NAME,
        notebook="11a_pca",
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        use_cache=USE_CACHE,
        force_retrain=FORCE_RETRAIN,
    )
    for label, variant_result in variant_results.items():
        print(label, variant_result["model_results"])
