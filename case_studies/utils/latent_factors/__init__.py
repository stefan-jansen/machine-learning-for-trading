"""Shared latent factor utilities for the case studies."""

from __future__ import annotations

# Import torch before the ml4t.* libraries below. Torch's wheel ships a
# CUDA runtime that exports `cudaGetDriverEntryPointByVersion`, but the
# older system `libcudart.so.12` reachable through the dependency
# graph (or `LD_LIBRARY_PATH`) does not. If a sibling import in the
# latent-factor stack causes the system cudart to load first, torch's
# `from torch._C import *` later fails with an undefined-symbol error
# that surfaces as an ImportError in the SDF/CAE/SAE notebooks. Loading
# torch first ensures its bundled cudart wins the resolution.
import torch  # noqa: F401

from case_studies.utils.latent_factors.cae import run_cae_fold
from case_studies.utils.latent_factors.cv import load_fold_extras, run_latent_factor_cv
from case_studies.utils.latent_factors.ipca import run_ipca_fold
from case_studies.utils.latent_factors.panel import (
    compute_managed_portfolios,
    prepare_panel_data,
    prepare_ragged_panel_data,
    rank_normalize_cross_section,
)
from case_studies.utils.latent_factors.pca import run_pca_fold
from case_studies.utils.latent_factors.sae import run_sae_fold
from case_studies.utils.latent_factors.sdf import run_sdf_fold

EXPENSIVE_MODELS = frozenset({"sdf", "cae", "sae"})

__all__ = [
    "EXPENSIVE_MODELS",
    "compute_managed_portfolios",
    "load_fold_extras",
    "prepare_panel_data",
    "prepare_ragged_panel_data",
    "rank_normalize_cross_section",
    "run_cae_fold",
    "run_ipca_fold",
    "run_latent_factor_cv",
    "run_pca_fold",
    "run_sae_fold",
    "run_sdf_fold",
]
