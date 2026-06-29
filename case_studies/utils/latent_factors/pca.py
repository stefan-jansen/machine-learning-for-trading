"""Return-panel PCA baseline for persistent-ID datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from case_studies.utils.latent_factors.library_bridge import run_pca_fold_with_library


def run_pca_fold(
    chars_train: np.ndarray,
    returns_train: np.ndarray,
    chars_val: np.ndarray,
    returns_val: np.ndarray,
    n_factors: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit PCA on the training return panel and emit expected-return forecasts."""
    del chars_train, chars_val
    return run_pca_fold_with_library(
        returns_train,
        returns_val,
        n_factors=n_factors,
    )
