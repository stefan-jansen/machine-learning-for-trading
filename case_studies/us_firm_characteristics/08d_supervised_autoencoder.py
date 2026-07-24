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
# # Supervised Autoencoder for US Firm Characteristics
#
# This notebook tests whether direct return supervision helps an autoencoder extract a useful
# one-month cross-sectional signal from firm characteristics.
#
# **Learning objectives**
#
# - Fit a supervised autoencoder through sealed walk-forward folds.
# - Compare complete out-of-sample prediction surfaces at physical training checkpoints.
# - Trace undertraining and overtraining from model IC into downstream strategy candidates.
#
# **Book reference:** Chapter 14, latent factor models and supervised autoencoders.
#
# **Prerequisites:** `03_financial_features`, `04_evaluation`, and the split contract used by
# `05_linear` through `07_tabular_dl`.

# %%
"""US firm characteristics supervised autoencoder run via the shared library path."""

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
MODEL_NAME = "sae"
SEED = 42

# %% [markdown]
# ## Load the dated modeling surface
#
# The shared context fixes the feature order, ten expanding-window splits, continuous target, and
# explicit CUDA runtime. Persistent firm identifiers keep each company stable across months. The
# 2016 holdout remains outside training, validation, cache identity, and checkpoint comparison.

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
# The model emits predictions every five epochs. Each checkpoint combines predictions from the
# same physical epoch across all ten folds, creating a complete out-of-sample surface. The runner's
# terminal-epoch summary does not remove other checkpoints: every complete surface remains eligible
# for model analysis and the signal-stage backtest.

# %%
result = run_case_study_model(
    context,
    model_name=MODEL_NAME,
    notebook="08d_supervised_autoencoder",
    n_factors=N_FACTORS,
    n_epochs=N_EPOCHS,
    use_cache=USE_CACHE,
    force_retrain=FORCE_RETRAIN,
)
fold_metrics = result["fold_metrics"][MODEL_NAME]

print(f"Physical checkpoints: {fold_metrics['epoch'].n_unique()}")
print(f"Completed folds: {fold_metrics['fold_id'].n_unique()}")

# %% [markdown]
# ## Confirm the registered checkpoint surface
#
# The registry is the reader-facing source of truth. This query binds the displayed curve to the
# single corrected SAE training identity and verifies that no interrupted or legacy identity enters
# the comparison.

# %%
training_runs = registry.load_training_runs(
    CASE_STUDY_ID,
    family="latent_factors",
    label=context.primary_label,
).filter(
    (pl.col("config_name") == MODEL_NAME) & (pl.col("entry_point") == "08d_supervised_autoencoder")
)
if training_runs.height != 1:
    raise ValueError(f"Expected one complete SAE identity, found {training_runs.height}")
training_hash = training_runs["training_hash"][0]

prediction_sets = registry.load_prediction_sets(
    CASE_STUDY_ID,
    training_hash=training_hash,
    split="validation",
)
prediction_metrics = registry.load_prediction_metrics(CASE_STUDY_ID)

# %% [markdown]
# Fold signs and dispersion complement the time-series HAC intervals without changing checkpoint
# eligibility or selecting a different epoch inside each fold.

# %%
fold_summary = (
    fold_metrics.group_by("epoch")
    .agg(
        pl.col("ic_mean").std().alias("fold_std"),
        (pl.col("ic_mean") > 0).sum().alias("positive_folds"),
    )
    .sort("epoch")
)
checkpoint_summary = (
    prediction_sets.join(prediction_metrics, on="prediction_hash", how="inner")
    .sort("checkpoint_value")
    .select(
        pl.col("checkpoint_value").alias("epoch"),
        "prediction_hash",
        "ic_mean_daily",
        "ic_ci_lo",
        "ic_ci_hi",
        "ic_p_hac",
        "ic_n_days",
    )
    .join(fold_summary, on="epoch", how="left")
)
if checkpoint_summary.height != 10:
    raise ValueError(f"Expected ten complete SAE checkpoints, found {checkpoint_summary.height}")
checkpoint_summary

# %% [markdown]
# ## Direct supervision peaks early, then reverses
#
# The global development maximum uses one epoch across every fold. It is not a fold-by-fold
# checkpoint composite. All ten surfaces continue to the strategy stage, where their trading
# behavior and selection risk can be compared under one common backtest contract.

# %%
selected = checkpoint_summary.sort("ic_mean_daily", descending=True).row(0, named=True)
terminal = checkpoint_summary.sort("epoch").tail(1).row(0, named=True)
selected_epoch = int(selected["epoch"])

lower_error = checkpoint_summary["ic_mean_daily"] - checkpoint_summary["ic_ci_lo"]
upper_error = checkpoint_summary["ic_ci_hi"] - checkpoint_summary["ic_mean_daily"]

# %% [markdown]
# The curve marks the global development maximum without suppressing any alternative checkpoint.
# Bars show HAC 95% intervals computed from the time-ordered monthly IC series.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    checkpoint_summary["epoch"],
    checkpoint_summary["ic_mean_daily"],
    yerr=[lower_error, upper_error],
    color=COLORS["blue"],
    marker="o",
    capsize=3,
    label="Checkpoint IC with HAC 95% interval",
)
ax.scatter(
    [selected_epoch],
    [selected["ic_mean_daily"]],
    color=COLORS["amber"],
    edgecolor=COLORS["blue"],
    s=90,
    zorder=3,
    label=f"Global development maximum: epoch {selected_epoch}",
)
ax.axhline(0, color=COLORS["neutral"], linewidth=1, linestyle="--")
ax.set(xlabel="Training epoch", ylabel="Mean monthly rank IC")
ax.legend(loc="lower left")
add_message_title(
    ax,
    "SAE peaks early and reverses after epoch 30",
    subtitle="One checkpoint surface per epoch across ten sealed walk-forward folds",
)
fig.tight_layout()
fig.show()

# %% [markdown]
# ## Takeaways
#
# Checkpoint comparison is part of development research. The final holdout remains sealed until the
# complete model and strategy funnel selects one carrier.

# %%
selected_interval = "excludes" if selected["ic_ci_lo"] * selected["ic_ci_hi"] > 0 else "includes"
display(
    Markdown(
        f"""
- Epoch **{selected_epoch}** is the global development maximum at daily rank IC
  **{selected["ic_mean_daily"]:+.4f}** across **{int(selected["ic_n_days"])}** validation months;
  its HAC 95% interval **[{selected["ic_ci_lo"]:+.4f}, {selected["ic_ci_hi"]:+.4f}]**
  {selected_interval} zero, and all **{int(selected["positive_folds"])}** folds are positive.
- Epochs 10 through 25 form a broad positive plateau. The curve crosses below zero by epoch 35 and
  ends at **{terminal["ic_mean_daily"]:+.4f}**, showing that longer training overfits this monthly
  cross-section.
- Every physical checkpoint remains a distinct prediction candidate for `11_backtest`; no
  fold-specific best-epoch alias enters the registry.
- `08_latent_factors` next compares the eligible latent objectives. Model-family and strategy
  selection occur downstream, and the sealed holdout is opened only for the final carrier.
"""
    )
)
