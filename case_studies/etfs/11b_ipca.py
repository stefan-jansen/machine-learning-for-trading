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

# %% [markdown] papermill={"duration": 0.003667, "end_time": "2026-04-28T02:25:47.448243+00:00", "exception": false, "start_time": "2026-04-28T02:25:47.444576+00:00", "status": "completed"}
# # IPCA for ETF Rotation
#
# IPCA uses the ETF characteristic panel directly and forecasts the training
# sample factor premia through the shared latent-factor pipeline.

# %% papermill={"duration": 2.79495, "end_time": "2026-04-28T02:25:50.246752+00:00", "exception": false, "start_time": "2026-04-28T02:25:47.451802+00:00", "status": "completed"}
"""ETF IPCA case-study run via the shared latent-factor library path."""

import warnings

from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
    run_case_study_variants,
)

warnings.filterwarnings("ignore")

# %% papermill={"duration": 0.004137, "end_time": "2026-04-28T02:25:50.251631+00:00", "exception": false, "start_time": "2026-04-28T02:25:50.247494+00:00", "status": "completed"} tags=["parameters"]
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
MODEL_NAME = "ipca"

# %% papermill={"duration": 0.002856, "end_time": "2026-04-28T02:25:50.255004+00:00", "exception": false, "start_time": "2026-04-28T02:25:50.252148+00:00", "status": "completed"} tags=["injected-parameters"]
# Parameters
USE_CACHE = False
FORCE_RETRAIN = True


# %% papermill={"duration": 0.558196, "end_time": "2026-04-28T02:25:50.813714+00:00", "exception": false, "start_time": "2026-04-28T02:25:50.255518+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001567, "end_time": "2026-04-28T02:25:50.816548+00:00", "exception": false, "start_time": "2026-04-28T02:25:50.814981+00:00", "status": "completed"}
# ## Run Walk-Forward CV

# %% papermill={"duration": 165.593686, "end_time": "2026-04-28T02:28:36.411098+00:00", "exception": false, "start_time": "2026-04-28T02:25:50.817412+00:00", "status": "completed"}
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="11b_ipca",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
)

print(result["model_results"])
print(result["fold_metrics"][MODEL_NAME])

# %% [markdown] papermill={"duration": 0.001585, "end_time": "2026-04-28T02:28:36.413700+00:00", "exception": false, "start_time": "2026-04-28T02:28:36.412115+00:00", "status": "completed"}
# ## Variant Labels

# %% papermill={"duration": 167.290035, "end_time": "2026-04-28T02:31:23.704487+00:00", "exception": false, "start_time": "2026-04-28T02:28:36.414452+00:00", "status": "completed"}
variant_results = {}
if RUN_VARIANTS and context.variant_labels:
    variant_results = run_case_study_variants(
        context,
        model_name=MODEL_NAME,
        notebook="11b_ipca",
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        use_cache=USE_CACHE,
        force_retrain=FORCE_RETRAIN,
    )
    for label, variant_result in variant_results.items():
        print(label, variant_result["model_results"])
