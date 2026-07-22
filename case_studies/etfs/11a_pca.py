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

# %% [markdown] papermill={"duration": 0.005298, "end_time": "2026-04-28T01:35:53.756099+00:00", "exception": false, "start_time": "2026-04-28T01:35:53.750801+00:00", "status": "completed"}
# # PCA for ETF Rotation
#
# PCA is the persistent-ID baseline in the latent-factor suite. For ETFs that
# assumption is valid, so we can evaluate whether the return-panel factor
# baseline adds anything over direct predictors.

# %% papermill={"duration": 2.489411, "end_time": "2026-04-28T01:35:56.249594+00:00", "exception": false, "start_time": "2026-04-28T01:35:53.760183+00:00", "status": "completed"}
"""ETF PCA case-study run via the shared latent-factor library path."""

import warnings

from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
    run_case_study_variants,
)

warnings.filterwarnings("ignore")

# %% papermill={"duration": 0.004288, "end_time": "2026-04-28T01:35:56.254629+00:00", "exception": false, "start_time": "2026-04-28T01:35:56.250341+00:00", "status": "completed"} tags=["parameters"]
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

# %% papermill={"duration": 0.002969, "end_time": "2026-04-28T01:35:56.258121+00:00", "exception": false, "start_time": "2026-04-28T01:35:56.255152+00:00", "status": "completed"} tags=["injected-parameters"]
# Parameters
USE_CACHE = False
FORCE_RETRAIN = True


# %% papermill={"duration": 0.675417, "end_time": "2026-04-28T01:35:56.934056+00:00", "exception": false, "start_time": "2026-04-28T01:35:56.258639+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.001759, "end_time": "2026-04-28T01:35:56.936577+00:00", "exception": false, "start_time": "2026-04-28T01:35:56.934818+00:00", "status": "completed"}
# ## Run Walk-Forward CV

# %% papermill={"duration": 84.102746, "end_time": "2026-04-28T01:37:21.039875+00:00", "exception": false, "start_time": "2026-04-28T01:35:56.937129+00:00", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.000685, "end_time": "2026-04-28T01:37:21.041498+00:00", "exception": false, "start_time": "2026-04-28T01:37:21.040813+00:00", "status": "completed"}
# ## Variant Labels

# %% papermill={"duration": 82.781912, "end_time": "2026-04-28T01:38:43.824070+00:00", "exception": false, "start_time": "2026-04-28T01:37:21.042158+00:00", "status": "completed"}
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
