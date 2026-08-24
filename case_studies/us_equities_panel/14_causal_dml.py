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
# # Causal DML - US Equities Panel
#
# This notebook estimates the configured treatment effect through the shared causal request and
# result boundary. Readers choose the outcome label, nuisance-model overrides, and execution tier.
# Shared code owns artifact loading, treatment and confounder validation, temporal geometry,
# nuisance fitting, refutation, identity construction, recovery, and registry publication.
#
# The notebook is an execution client. It shows what was requested, what the request resolved to,
# and which immutable result was produced. Dedicated causal and model-analysis material interprets
# results across case studies.
#
# **Learning objectives**
#
# - Define a treatment, outcome, confounder set, and temporal nuisance-fitting design.
# - Inspect the resolved estimand and its identity-bearing inputs before fitting.
# - Validate causal estimates, HAC uncertainty, refutation results, and registry persistence.
#
# **Book reference**: Chapter 15, Section 15.6 (Cross-Dataset Causal Evidence)
#
# **Prerequisites**: `03_financial_features.py`, `04_model_based_features.py`, and the finalized
# label artifacts.

# %%
"""Estimate the configured causal effect through the shared DML boundary."""

import os
from pathlib import Path

import polars as pl
import yaml

from case_studies.research import Study
from utils.modeling import load_configs
from utils.paths import REPO_ROOT, get_case_study_dir

# %% [markdown]
# ## What the estimator assumes
#
# DML first predicts the outcome and treatment from the declared confounders, then estimates the
# treatment effect from the two residual series. Walk-forward nuisance fits and an embargo keep
# future outcomes out of earlier estimates. HAC uncertainty addresses serial dependence, while the
# block-permutation refutation checks how often a similarly large estimate appears after disrupting
# the treatment assignment within each symbol.
#
# The estimate still requires conditional ignorability, overlap, and no interference between
# entities. Those assumptions are not established by a low p-value or a successful refutation. The
# resolved request therefore records the treatment, outcome, complete confounder list, temporal
# design, and refutation policy rather than treating DML as an automatic causal conclusion.

# %% tags=["parameters"]
CASE_STUDY_ID = "us_equities_panel"
PRIMARY_LABEL = ""
CONFIG_NAME = "dml"
NUISANCE_OVERRIDES = {}
EXECUTION_TIER = "canonical"
WORKSPACE = "experiments"
MAX_SYMBOLS = 0
PREVIEW_MAX_SAMPLES = 0
PREVIEW_N_FOLDS = 0
PREVIEW_N_PLACEBO = 0

# %% [markdown]
# ## Configure the estimand and execution
#
# The treatment and complete confounder list live in `config/setup.yaml`; the outcome is the label
# selected here. `CONFIG_NAME` chooses a published DML configuration. `NUISANCE_OVERRIDES` changes
# validated `HistGradientBoostingRegressor` parameters without duplicating the remaining defaults.
#
# Canonical execution uses the complete declared pre-holdout population. A reduced pipeline check
# must use `EXECUTION_TIER = 'preview'` and declare at least one reduction. Preview reductions are
# part of the immutable identity and cannot enter canonical comparisons or conclusions.

# %%
case_dir = get_case_study_dir(CASE_STUDY_ID)
setup = yaml.safe_load((case_dir / "config" / "setup.yaml").read_text())
label = PRIMARY_LABEL or setup["labels"]["primary"]

published_configs = load_configs(CASE_STUDY_ID, label, family="causal_dml")
published_names = [str(config["config_name"]) for config in published_configs]
if CONFIG_NAME not in published_names:
    raise ValueError(f"Unknown DML configuration: {CONFIG_NAME!r}")

causal_config = setup.get("causal") or {}
treatment = causal_config.get("treatment")
confounders = list(causal_config.get("confounders") or [])
if not treatment:
    raise ValueError("config/setup.yaml must declare causal.treatment")
if not confounders:
    raise ValueError("config/setup.yaml must declare at least one causal.confounder")

config_menu = pl.DataFrame(
    {
        "config_name": published_names,
        "selected": [name == CONFIG_NAME for name in published_names],
        "treatment": [str(treatment)] * len(published_names),
        "outcome": [label] * len(published_names),
    }
)
config_menu

# %%
preview_reductions = {}
if MAX_SYMBOLS:
    preview_reductions["max_symbols"] = int(MAX_SYMBOLS)
if PREVIEW_MAX_SAMPLES:
    preview_reductions["max_samples"] = int(PREVIEW_MAX_SAMPLES)
if PREVIEW_N_FOLDS:
    preview_reductions["n_folds"] = int(PREVIEW_N_FOLDS)
if PREVIEW_N_PLACEBO:
    preview_reductions["n_placebo"] = int(PREVIEW_N_PLACEBO)

if EXECUTION_TIER == "canonical":
    if preview_reductions:
        raise ValueError("Canonical execution cannot declare preview reductions")
    study = Study.regenerate(CASE_STUDY_ID, release_root=REPO_ROOT)
elif EXECUTION_TIER == "preview":
    if not preview_reductions:
        raise ValueError("Preview execution requires at least one declared reduction")
    study = Study.open(
        CASE_STUDY_ID,
        workspace=Path(os.environ.get("ML4T_OUTPUT_DIR") or WORKSPACE),
        release_root=REPO_ROOT,
    )
else:
    raise ValueError("EXECUTION_TIER must be 'canonical' or 'preview'")

request = study.causal(
    method="dml",
    label=label,
    config_name=CONFIG_NAME,
    overrides={"nuisance_params": dict(NUISANCE_OVERRIDES)} if NUISANCE_OVERRIDES else {},
    execution_tier=EXECUTION_TIER,
    preview_reductions=preview_reductions,
)
resolved = request.resolve()

# %% [markdown]
# ## Inspect the resolved request
#
# Resolution fails before fitting if the finalized artifacts omit the treatment or any configured
# confounder. The table exposes the exact population, temporal design, nuisance estimator, and input
# identities used by the run.

# %% tags=["results"]
spec = resolved.spec
computation = spec["computation"]
feature_artifacts = computation["feature_artifacts"]
artifact_names = (
    sorted(feature_artifacts)
    if isinstance(feature_artifacts, dict)
    else [str(item) for item in feature_artifacts]
)
resolved_table = pl.DataFrame(
    [
        {
            "config_name": spec["config_name"],
            "label": spec["label"],
            "treatment": computation["estimand"]["treatment"],
            "confounders": computation["estimand"]["confounders"],
            "feature_artifacts": artifact_names,
            "features": len(computation["feature_names"]),
            "analysis_rows": computation["analysis_population"]["n_rows"],
            "decision_times": computation["analysis_population"]["n_timestamps"],
            "folds": computation["cv"]["n_folds"],
            "placebos": computation["refutation"]["n_placebo"],
            "execution_tier": spec["execution_tier"],
            "causal_hash": resolved.identity,
        }
    ]
)
resolved_table

# %% [markdown]
# ## Execute and validate the result
#
# The runner reopens an exact complete result on retry. A new request produces one causal registry
# record only after finite effect and HAC uncertainty estimates exist. Causal results remain
# separate from predictive candidate sets and strategy selection.

# %%
result = resolved.run()
if not result.complete:
    raise RuntimeError(f"Incomplete causal result: {result.hash}")
if result.hash != resolved.identity:
    raise RuntimeError("Causal result identity differs from the resolved request")
if result.execution_tier != EXECUTION_TIER:
    raise RuntimeError("Causal result execution tier differs from the request")

# %% tags=["results"]
result_table = pl.DataFrame(
    [
        {
            "causal_hash": result.hash,
            "observations": result.metrics["n_obs"],
            "dml_effect": result.metrics["dml_effect"],
            "hac_standard_error": result.metrics["dml_se_hac"],
            "hac_p_value": result.metrics["p_value_hac"],
            "naive_effect": result.metrics["naive_effect"],
            "confounding_bias_pct": result.metrics["confounding_bias_pct"],
            "refutation_p_value": result.metrics["refutation_p"],
            "complete": result.complete,
            "execution_tier": result.execution_tier,
        }
    ]
)
result_table

# %% [markdown]
# ## Downstream handoff
#
# `15_model_analysis.py` can open this exact causal result separately from its predictive result
# sets. The resolved specification retains the label, treatment, complete confounder list, nuisance
# defaults and overrides, temporal design, analysis population, finalized input digests, source,
# runtime, and execution tier needed to reproduce the estimate.

# %% [markdown]
# ## Key takeaways and limitations
#
# - The resolved request makes the treatment, outcome, confounders, folds, and nuisance models part
#   of one reproducible estimand.
# - Walk-forward nuisance fits and the embargo preserve temporal ordering in the observed panel.
# - HAC uncertainty and block-permutation refutation address specified sampling concerns; causal
#   interpretation still depends on conditional ignorability, overlap, and limited interference.
# - The causal result remains separate from predictive model selection and strategy selection.
