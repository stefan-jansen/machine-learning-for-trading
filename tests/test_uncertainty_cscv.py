"""Tests for case_studies/utils/uncertainty.py CSCV partition + PBO smoke.

Covers two pieces P2.5 added:

1. ``_cscv_split_pairs`` — IS/OOS partition shape and balance for
   ``n_folds`` in {2, 3, 4}, including the asymmetric odd-fold case.
2. ``compute_cohort_metrics`` end-to-end with a ``fold_returns_by_hash``
   argument, asserting that ``pbo`` / ``pbo_median_oos_rank`` /
   ``pbo_mean_degradation`` come back populated (i.e. the
   ``compute_pbo`` field-name and partition wiring is intact).
"""

from __future__ import annotations

from math import comb

import numpy as np
import polars as pl
import pytest


@pytest.mark.parametrize(
    "n_folds, is_half, oos_half",
    [
        (2, 1, 1),  # balanced
        (3, 1, 2),  # asymmetric — OOS gets the extra fold
        (4, 2, 2),  # balanced
    ],
)
def test_cscv_split_pairs_partition_shape(n_folds: int, is_half: int, oos_half: int) -> None:
    from case_studies.utils.uncertainty import _cscv_split_pairs

    rng = np.random.default_rng(0)
    k_variants = 5
    fold_sharpes = rng.normal(size=(n_folds, k_variants))

    is_perf, oos_perf = _cscv_split_pairs(fold_sharpes)

    expected_n = comb(n_folds, n_folds // 2)
    assert is_perf.shape == (expected_n, k_variants)
    assert oos_perf.shape == (expected_n, k_variants)

    # Every row must be the mean of `is_half` folds (IS) and
    # `oos_half` folds (OOS) of the original matrix — verified by
    # reconstructing the underlying sums.
    for row_is, row_oos in zip(is_perf, oos_perf, strict=True):
        # IS mean × is_half + OOS mean × oos_half == sum of all folds
        total = fold_sharpes.sum(axis=0)
        reconstructed = row_is * is_half + row_oos * oos_half
        np.testing.assert_allclose(reconstructed, total, atol=1e-12)


def test_cscv_split_pairs_single_fold_returns_empty() -> None:
    from case_studies.utils.uncertainty import _cscv_split_pairs

    is_perf, oos_perf = _cscv_split_pairs(np.array([[1.0, 2.0, 3.0]]))
    assert is_perf.shape == (0, 3)
    assert oos_perf.shape == (0, 3)


def test_compute_cohort_metrics_populates_pbo_with_fold_returns() -> None:
    """End-to-end smoke: PBO fields must come back non-null when
    ``fold_returns_by_hash`` is supplied for >=2 variants with >=2 folds.

    The pre-P2.5 code called ``compute_pbo(fs, fs)`` and read the wrong
    PBOResult attribute names — both bugs would surface here as NULLs.
    """
    from case_studies.utils.uncertainty import compute_cohort_metrics

    rng = np.random.default_rng(7)
    n_periods = 252
    timestamps = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2020, 12, 31),
        interval="1d",
        eager=True,
    ).head(n_periods)

    def _make_frame(mu: float) -> pl.DataFrame:
        ret = rng.normal(loc=mu / 252, scale=0.01, size=n_periods)
        return pl.DataFrame({"timestamp": timestamps, "ret": ret})

    # Three "variants" with hash-shaped keys (32 hex chars satisfies any
    # downstream FK convention; here we just need stable dict keys).
    returns_by_hash = {f"{i:032x}": _make_frame(mu=mu) for i, mu in enumerate([0.05, 0.08, 0.12])}

    n_folds = 4
    fold_returns_by_hash = {
        h: rng.normal(loc=0.0, scale=1.0, size=n_folds) for h in returns_by_hash
    }

    out = compute_cohort_metrics(
        returns_by_hash,
        periods_per_year=252.0,
        fold_returns_by_hash=fold_returns_by_hash,
        rademacher_n_simulations=50,
        rademacher_seed=0,
    )

    assert out, "compute_cohort_metrics returned empty dict — alignment failed"
    assert out["leader_hash"] in returns_by_hash
    assert out["k_variants"] == 3

    # PBO fields must be populated (the bug-surface check).
    assert out["pbo"] is not None
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["pbo_n_combinations"] == float(comb(n_folds, n_folds // 2))
    assert out["pbo_median_oos_rank"] is not None
    assert out["pbo_mean_degradation"] is not None
    assert out["pbo_n_folds"] == float(n_folds)
