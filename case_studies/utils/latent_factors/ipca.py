"""Instrumented PCA on dated cross-sections."""

from __future__ import annotations

from typing import Any

import numpy as np

from case_studies.utils.latent_factors.library_bridge import run_ipca_fold_with_library


def run_ipca_fold(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    n_factors: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    factor_ridge: float = 1e-6,
    gamma_ridge: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit IPCA on unbalanced dated cross-sections and forecast with train premiums."""
    return run_ipca_fold_with_library(
        chars_train,
        returns_train,
        chars_val,
        returns_val,
        n_factors=n_factors,
        max_iter=max_iter,
        tol=tol,
        factor_ridge=factor_ridge,
        gamma_ridge=gamma_ridge,
    )
