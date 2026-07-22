# ---
# jupyter:
#   jupytext:
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
# # Conditional Autoencoder for ETF Rotation
#
# The CAE notebook runs the structural extractor and then maps checkpointed
# factor states through the shared factor-premium baseline.

# %%
"""ETF CAE case-study run via the shared latent-factor library path."""

import warnings

from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
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
USE_MACRO = False
MODEL_NAME = "cae"
REPORTING_CHECKPOINT = N_EPOCHS

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
print(f"Dataset rows: {len(context.dataset):,}")
print(f"Features: {len(context.feature_names)}")
print(f"Splits: {len(context.splits)}")
print(f"Model kwargs: {context.model_kwargs.get(MODEL_NAME, {})}")

# %% [markdown]
# ## Run Walk-Forward CV

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="11c_conditional_autoencoder",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
    reporting_epoch=REPORTING_CHECKPOINT,
)

if result["model_results"][0]["best_epoch"] != REPORTING_CHECKPOINT:
    raise AssertionError("CAE result did not use the fixed reporting checkpoint")
print(result["model_results"])
print(result["fold_metrics"][MODEL_NAME])
