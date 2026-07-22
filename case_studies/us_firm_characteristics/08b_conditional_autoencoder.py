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
# # Conditional Autoencoder for US Firm Characteristics
#
# This notebook tests whether nonlinear characteristic-conditioned factor loadings improve
# one-month cross-sectional return forecasts.
#
# **Learning objectives**
#
# - Fit a conditional autoencoder through sealed walk-forward folds.
# - Distinguish physical training checkpoints from validation-selected aliases.
# - Evaluate checkpoint stability with daily rank IC and HAC uncertainty.
#
# **Book reference:** Chapter 14, latent factor models and conditional autoencoders.
#
# **Prerequisites:** `03_financial_features`, `04_evaluation`, and the split contract used by
# `05_linear` through `07_tabular_dl`.

# %%
"""US firm characteristics CAE case-study run via the shared latent-factor library path."""

import warnings

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import Markdown, display

from case_studies.utils import registry
from case_studies.utils.latent_factors.case_study import (
    configured_models,
    load_case_study_context,
    run_case_study_model,
)
from utils.reproducibility import set_global_seeds
from utils.style import COLORS, add_message_title

warnings.filterwarnings("ignore")

# %% tags=["parameters"]
CASE_STUDY_ID = "us_firm_characteristics"
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
SEED = 42

# %% [markdown]
# ## Load the dated modeling surface
#
# The shared context fixes the feature order, ten expanding-window splits, continuous target, and
# explicit CUDA runtime. Persistent firm identifiers keep each company stable across months.

# %%
set_global_seeds(SEED)
context = load_case_study_context(
    CASE_STUDY_ID,
    primary_label=PRIMARY_LABEL,
    max_symbols=MAX_SYMBOLS,
    max_folds=MAX_FOLDS,
    max_variant_labels=MAX_VARIANT_LABELS,
    use_macro=USE_MACRO,
)
if MODEL_NAME not in configured_models(context):
    raise ValueError(f"{MODEL_NAME!r} is not configured for {CASE_STUDY_ID}")

print(f"Observations: {len(context.dataset):,}")
print(f"Characteristics: {len(context.feature_names)}")
print(f"Walk-forward folds: {len(context.splits)}")
print(f"Runtime: {context.device}, deterministic={context.deterministic_algorithms}")

# %% [markdown]
# ## Run or replay walk-forward training
#
# The fixed reporting rule uses the final physical epoch. Checkpoint zero is not part of the
# registry surface because the model library reserves it for a state selected with validation loss.
# Cached execution requires an exact match on input digest, split boundaries, runtime, checkpoint
# coverage, prediction keys, and reconstructed daily IC metrics.

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="08b_conditional_autoencoder",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
)
model_result = result["model_results"][0]
fold_metrics = result["fold_metrics"][MODEL_NAME]

print(f"Reporting epoch: {model_result['best_epoch']}")
print(f"Mean daily rank IC: {model_result['mean_ic']:+.4f}")
print(f"Completed folds: {model_result['n_folds']}")

# %% [markdown]
# ## Confirm the registered physical checkpoint surface
#
# The registry is the reader-facing source of truth. The query below binds the displayed curve to
# the one complete CAE training identity produced by this notebook.

# %%
training_runs = registry.load_training_runs(
    CASE_STUDY_ID,
    family="latent_factors",
    label=context.primary_label,
).filter(
    (pl.col("config_name") == MODEL_NAME) & (pl.col("entry_point") == "08b_conditional_autoencoder")
)
if training_runs.height != 1:
    raise ValueError(f"Expected one complete CAE identity, found {training_runs.height}")
training_hash = training_runs["training_hash"][0]

prediction_sets = registry.load_prediction_sets(
    CASE_STUDY_ID,
    training_hash=training_hash,
    split="validation",
)
prediction_metrics = registry.load_prediction_metrics(CASE_STUDY_ID)
checkpoint_summary = (
    prediction_sets.join(prediction_metrics, on="prediction_hash", how="inner")
    .sort("checkpoint_value")
    .select(
        pl.col("checkpoint_value").alias("epoch"),
        "ic_mean_daily",
        "ic_ci_lo",
        "ic_ci_hi",
        "ic_p_hac",
        "ic_n_days",
    )
)
checkpoint_summary

# %% [markdown]
# ## Training longer does not rescue the CAE signal
#
# Fold dispersion shows whether a mean result reflects a stable cross-period effect. The final
# epoch is fixed before inspecting its IC, while the full curve remains diagnostic.

# %%
curve = (
    fold_metrics.group_by("epoch")
    .agg(
        pl.col("ic_mean").mean().alias("mean_ic"),
        pl.col("ic_mean").std().alias("fold_std"),
    )
    .sort("epoch")
)
selected_epoch = int(model_result["best_epoch"])
selected = curve.filter(pl.col("epoch") == selected_epoch).row(0, named=True)

# %% [markdown]
# The path below marks the predeclared final checkpoint and uses fold dispersion to show how much
# the cross-period estimates vary around each mean.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    curve["epoch"],
    curve["mean_ic"],
    yerr=curve["fold_std"],
    color=COLORS["blue"],
    marker="o",
    capsize=3,
    label="Mean IC with fold dispersion",
)
ax.scatter(
    [selected_epoch],
    [selected["mean_ic"]],
    color=COLORS["amber"],
    edgecolor=COLORS["blue"],
    s=80,
    zorder=3,
    label=f"Fixed epoch {selected_epoch}",
)
ax.axhline(0, color=COLORS["neutral"], linewidth=1, linestyle="--")
ax.set(xlabel="Training epoch", ylabel="Mean daily rank IC")
ax.legend(loc="best")
direction = "negative" if selected["mean_ic"] < 0 else "positive"
add_message_title(
    ax,
    f"CAE remains {direction} at the fixed physical checkpoint",
    subtitle="Ten sealed walk-forward folds; bars show one standard deviation across folds",
)
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Takeaways
#
# The final checkpoint and its HAC interval determine the interpretation; no checkpoint is chosen
# after inspecting validation IC.

# %%
fixed = checkpoint_summary.filter(pl.col("epoch") == selected_epoch).row(0, named=True)
interval_verdict = "excludes" if fixed["ic_ci_lo"] * fixed["ic_ci_hi"] > 0 else "includes"
display(
    Markdown(
        f"""
- The fixed epoch-{selected_epoch} CAE reaches daily rank IC **{fixed["ic_mean_daily"]:+.4f}**
  across **{int(fixed["ic_n_days"])}** validation months.
- Its HAC 95% interval is **[{fixed["ic_ci_lo"]:+.4f}, {fixed["ic_ci_hi"]:+.4f}]** and
  {interval_verdict} zero, so the model does not establish a stable forecasting edge.
- The checkpoint path stays negative on average, indicating that additional reconstruction
  training does not repair the weak forward-return mapping.
- `08c_stochastic_discount_factor` next tests an explicitly asset-pricing-driven latent model;
  Chapter 14 explains why reconstruction and pricing objectives can imply different signals.
"""
    )
)
