"""Shared causal inference utilities for Ch15 notebooks and case study DML.

Provides:
- block_permute(): Block permutation preserving autocorrelation
- manual_dml_timeseries(): Walk-forward DML with embargo
- run_dml_analysis(): Full DML pipeline (naive + DML + refutation)

Used by teaching notebooks (02-04, 07) and case study 09_causal_dml.py.
"""

from __future__ import annotations

import os

# HistGradientBoostingRegressor uses OMP-parallel histogram construction whose
# floating-point reduction order is non-deterministic across threads, so the placebo
# loop is only bit-reproducible at a fixed thread count.
#
# Setting OMP_NUM_THREADS here does NOT achieve that on its own. Every notebook imports
# case_studies.research (and through it sklearn) before this module, so the OpenMP and
# OpenBLAS runtimes have already read the variable by the time this line runs: measured
# with threadpoolctl, the notebook import order leaves the openmp pool at 16 and openblas
# at 24 despite os.environ reporting 1. The env pin is kept because it does work when this
# module is imported first, but DML_THREAD_LIMIT below is the mechanism that holds, and the
# limit is recorded in the resolved specification so two runs at different counts cannot
# share an identity.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Fixed rather than derived from the host: a value like -1 varies with the machine, so a
# result would not be identity-stable across the readers' hardware.
DML_THREAD_LIMIT = 1

import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.regression.linear_model import OLS
from threadpoolctl import threadpool_limits

from utils.modeling import RANDOM_SEED, seed_everything

if TYPE_CHECKING:
    from case_studies.research.workspace import Study


_DML_PREVIEW_FIELDS = {"max_samples", "max_symbols", "n_folds", "n_placebo"}


@dataclass(frozen=True)
class DMLResearchContext:
    analysis: pd.DataFrame
    treatment_col: str
    outcome_col: str
    confounder_cols: tuple[str, ...]
    time_col: str
    entity_col: str
    n_folds: int
    embargo: int
    n_placebo: int
    block_size: int
    seed: int
    horizon: int
    expected_step: pd.Timedelta
    nuisance_params: dict[str, Any]
    runtime_provenance: dict[str, Any]


def observation_step(frame: Any, date_col: str = "timestamp") -> pd.Timedelta:
    """Measure the spacing between consecutive decision times in *frame*.

    Returns the most common gap between distinct sorted values of ``date_col``,
    which is the observation grid the panel is actually recorded on. Sessions,
    weekends and holidays introduce larger gaps; taking the mode rather than the
    minimum or the mean makes those irrelevant.

    Accepts a Polars or pandas frame, or anything exposing the column through
    ``__getitem__``.
    """
    column = frame[date_col]
    values = pd.Series(column.to_list() if hasattr(column, "to_list") else list(column))
    stamps = pd.to_datetime(values).drop_duplicates().sort_values()
    if len(stamps) < 2:
        raise ValueError(f"{date_col!r} has fewer than two distinct values; no grid to measure")
    gaps = stamps.diff().dropna()
    return pd.Timedelta(gaps.mode().iloc[0])


def embargo_from_buffer(
    label_buffer: str,
    *,
    periods_per_year: int | None = None,
    observed_step: str | pd.Timedelta | None = None,
) -> int:
    """Convert a label buffer string to an integer embargo period count.

    The embargo is counted in *observation periods*, so converting a duration
    into one requires knowing how long an observation period is. Pass
    ``observed_step`` — from :func:`observation_step` on the frame being
    analysed — and the conversion is exact: the number of periods spanning the
    buffer, rounded up, at least one.

    Without ``observed_step`` the conversion falls back to a fixed assumption
    about bar size per unit, which is correct only when that assumption holds:

    - D: one period per ``value`` days, correct on a daily grid
    - H/h: the number of ``value``-hour bars in one day, so "8H" gives a
      one-day embargo on 8-hour bars
    - M: ``value`` monthly groups when ``periods_per_year=12``, else
      ``value`` months x 21 trading days
    - T/min: the number of ``value``-minute spans in 15 minutes, which assumes
      the panel is recorded in 15-minute bars

    That last assumption is the one that bites, because it is wrong by exactly
    the ratio between the assumed bar and the real one, and nothing in the result
    shows it. A "15min" buffer resolves to a single period whatever the panel is
    recorded at, so on a one-minute grid it yields a one-minute embargo against a
    fifteen-minute label. Pass ``observed_step`` on any sub-daily panel rather
    than relying on the declared bar size, which can disagree with the artifacts.

    A month buffer has no fixed length and is rejected when ``observed_step`` is
    supplied; use the ``periods_per_year`` branch for it.
    """
    import math
    import re

    if observed_step is not None:
        step = pd.Timedelta(observed_step)
        if step <= pd.Timedelta(0):
            raise ValueError(f"observed_step must be positive, got {observed_step!r}")
        # pandas deprecated the uppercase hour alias; the buffers are authored by
        # hand in setup.yaml and still use it.
        normalized = re.sub(r"(?<=\d)H\b", "h", label_buffer.strip())
        if re.match(r"\d+\s*M\b", normalized):
            raise ValueError(
                f"A month buffer ({label_buffer!r}) has no fixed length, so it cannot be "
                f"divided by an observation step. Use the periods_per_year branch by "
                f"omitting observed_step."
            )
        span = pd.Timedelta(normalized)
        return max(1, math.ceil(span / step))

    match = re.match(r"(\d+)(D|H|h|M|T|min)", label_buffer.strip())
    if not match:
        raise ValueError(f"Cannot parse label_buffer: {label_buffer}")
    value, unit = int(match.group(1)), match.group(2)
    return {
        "D": value,
        "H": max(1, 24 // value),
        "h": max(1, 24 // value),
        "M": value if periods_per_year == 12 else value * 21,
        "T": max(1, value // 15),
        "min": max(1, value // 15),
    }[unit]


# A segment boundary is a hole in the observation series, not a step the calendar
# always takes. Splitting wherever the gap differs from the cadence at all cut a
# daily series at every weekend, leaving five-row segments that could not hold two
# blocks of any useful size; the short-segment path then shuffled them. Four
# cadences clears a weekend (three) and a long weekend (four) on a daily series and
# still catches a real hole.
GAP_TOLERANCE_CADENCES = 4


def block_permute(
    arr: np.ndarray,
    block_size: int,
    rng: np.random.Generator | None = None,
    groups: np.ndarray | None = None,
    units: np.ndarray | None = None,
    expected_step: str | pd.Timedelta | None = None,
    gap_tolerance: str | pd.Timedelta | None = None,
) -> np.ndarray:
    """Permute array in blocks to preserve autocorrelation structure.

    Essential for refutation tests on time series data. Random permutation
    destroys autocorrelation, making placebo tests too easy to pass.

    Parameters
    ----------
    arr : array-like
        Array to permute.
    block_size : int
        Size of blocks to preserve.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    groups : array-like, optional
        Ordered decision time for each row. For a single time series, this
        validates that each row is one decision time.
    units : array-like, optional
        Panel entity for each row. When supplied with ``groups``, treatment is
        block-permuted within each entity, so ``block_size`` counts that
        entity's ordered decision times rather than flattened panel rows.

    Returns
    -------
    np.ndarray
        Block-permuted array.
    """
    arr = np.asarray(arr)
    if rng is None:
        rng = np.random.default_rng()

    segments = _permutation_segments(len(arr), groups, units, expected_step, gap_tolerance)
    result = np.array(arr, copy=True)
    for idx in segments:
        result[idx] = _permute_one_segment(arr[idx], block_size, rng)
    return result


def _permutation_segments(
    n: int,
    groups: np.ndarray | None,
    units: np.ndarray | None,
    expected_step: str | pd.Timedelta | None,
    gap_tolerance: str | pd.Timedelta | None,
) -> list[np.ndarray]:
    """The maximal stretches of rows a block permutation may reorder within.

    One entity's uninterrupted run of decision times is one segment. Blocks never cross
    a segment boundary, because the rows on either side are not adjacent in time - they
    belong to different entities, or to the same entity either side of a gap.

    This is the single definition of where the series is cut. ``block_permute`` permutes
    within these segments and ``_immobile_masks`` counts what they leave standing, so
    the two can never disagree about the segmentation.
    """
    if units is not None:
        if groups is None:
            raise ValueError("groups are required when units are supplied")
        group_arr = np.asarray(groups)
        unit_arr = np.asarray(units)
        if len(group_arr) != n or len(unit_arr) != n:
            raise ValueError("groups and units must have the same length as arr")
        segments: list[np.ndarray] = []
        for unit in pd.unique(unit_arr):
            idx = np.flatnonzero(unit_arr == unit)
            unit_groups = group_arr[idx]
            if len(unit_groups) > 1 and np.any(unit_groups[1:] <= unit_groups[:-1]):
                raise ValueError("groups must be strictly increasing within each unit")
            segments.extend(
                idx[bounds]
                for bounds in _contiguous_runs(unit_groups, expected_step, gap_tolerance)
            )
        return segments

    if groups is not None:
        group_arr = np.asarray(groups)
        if len(group_arr) != n:
            raise ValueError("groups must have the same length as arr")
        if len(np.unique(group_arr)) != n:
            raise ValueError("units are required when decision times contain multiple rows")
        if n > 1 and np.any(group_arr[1:] <= group_arr[:-1]):
            raise ValueError("groups must be strictly increasing")
        positions = np.arange(n)
        return [
            positions[bounds]
            for bounds in _contiguous_runs(group_arr, expected_step, gap_tolerance)
        ]

    return [np.arange(n)]


def _contiguous_runs(
    group_arr: np.ndarray,
    expected_step: str | pd.Timedelta | None,
    gap_tolerance: str | pd.Timedelta | None,
) -> list[slice]:
    """Split one entity's ordered decision times wherever the gap exceeds the tolerance."""
    n = len(group_arr)
    if expected_step is None or n <= 1:
        return [slice(0, n)]
    cadence = pd.Timedelta(expected_step)
    tolerance = (
        pd.Timedelta(gap_tolerance)
        if gap_tolerance is not None
        else GAP_TOLERANCE_CADENCES * cadence
    )
    timestamps = pd.to_datetime(group_arr, utc=True)
    steps = np.asarray(timestamps[1:] - timestamps[:-1])
    boundaries = np.flatnonzero(steps > tolerance) + 1
    if not boundaries.size:
        return [slice(0, n)]
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, n]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def _permute_one_segment(
    values: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Reorder whole blocks within one uninterrupted stretch."""
    n = len(values)
    n_blocks = n // block_size
    if n_blocks < 2:
        # Not enough room for two blocks, so there is no permutation to make at this
        # block size. Returning the segment intact preserves the dependence the
        # caller asked to keep; shuffling it would destroy exactly that, which is
        # what the old `rng.permutation(arr)` did to every weekend-bounded segment
        # of a daily series. A caller that permutes nothing at all is caught by
        # `_assert_placebo_permutation_possible`, not here: inside a panel, one short unit staying
        # put while the others move is correct.
        return np.array(values, copy=True)

    pieces = [
        values[idx * block_size : (idx + 1) * block_size] for idx in rng.permutation(n_blocks)
    ]

    # The trailing rows that do not fill a whole block stay where they are. This is a
    # property of block permutation itself, not of the data: it happens to any segment
    # whose length is not a multiple of the block size, however long the segment is.
    remainder_start = n_blocks * block_size
    if remainder_start < n:
        pieces.append(values[remainder_start:])

    return np.concatenate(pieces)


def _immobile_masks(
    n: int, block_size: int, segments: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Which rows no draw can move, split by the two unrelated reasons.

    The first mask is rows in a segment too short to hold two blocks. Those are frozen
    because of how the data is shaped against the block size, they carry their observed
    values into every placebo, and a large share of them makes the refutation
    uninformative - which is worth telling the caller about.

    The second is the trailing remainder of a segment whose length is not a multiple of
    the block size. Those rows also never move, but for a reason intrinsic to block
    permutation that no choice of gap tolerance can remove, and shrinking the block size
    to chase them is the wrong response. They are reported, not warned about.
    """
    short = np.zeros(n, dtype=bool)
    remainder = np.zeros(n, dtype=bool)
    for idx in segments:
        n_blocks = len(idx) // block_size
        if n_blocks < 2:
            short[idx] = True
        elif n_blocks * block_size < len(idx):
            remainder[idx[n_blocks * block_size :]] = True
    return short, remainder


def _walk_forward_indices(
    n_rows: int,
    n_folds: int,
    embargo: int,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window folds in rows or complete decision-time groups."""
    if groups is None:
        fold_size = n_rows // (n_folds + 1)
        folds = []
        for fold in range(n_folds):
            train_end = (fold + 1) * fold_size
            test_start = train_end + embargo
            test_end = min(test_start + fold_size, n_rows)
            folds.append((np.arange(0, train_end), np.arange(test_start, test_end)))
        return folds

    group_arr = np.asarray(groups)
    if len(group_arr) != n_rows:
        raise ValueError("groups must have the same length as the input arrays")
    group_starts = np.flatnonzero(np.r_[True, group_arr[1:] != group_arr[:-1]])
    ordered_groups = group_arr[group_starts]
    if len(np.unique(ordered_groups)) != len(ordered_groups) or (
        len(ordered_groups) > 1 and np.any(ordered_groups[1:] < ordered_groups[:-1])
    ):
        raise ValueError("groups must be sorted and contiguous")

    fold_size = len(ordered_groups) // (n_folds + 1)
    folds = []
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, len(ordered_groups))
        train_groups = ordered_groups[:train_end]
        test_groups = ordered_groups[test_start:test_end]
        folds.append(
            (
                np.flatnonzero(np.isin(group_arr, train_groups)),
                np.flatnonzero(np.isin(group_arr, test_groups)),
            )
        )
    return folds


def manual_dml_timeseries(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    embargo: int = 21,
    model_y=None,
    model_t=None,
    return_residuals: bool = False,
    hac_maxlags: int | None = None,
    horizon: int | None = None,
    groups: np.ndarray | None = None,
    thread_limit: int = DML_THREAD_LIMIT,
) -> dict:
    """Walk-forward DML with embargo for temporal data.

    Follows Chernozhukov et al. (2017) and de Prado (2018):
    1. Split data into K temporal folds (not random)
    2. For each fold, train on earlier data, predict on later
    3. Embargo gap between train and test prevents autocorrelation leakage
    4. HAC standard errors account for residual autocorrelation

    Parameters
    ----------
    Y : array
        Outcome variable.
    T : array
        Treatment variable.
    X : array
        Confounder matrix.
    n_folds : int
        Number of temporal folds.
    embargo : int
        Gap periods between train and test sets.
    model_y, model_t : sklearn estimator, optional
        Nuisance models for E[Y|X] and E[T|X].
    return_residuals : bool
        If True, include residual arrays in result dict.
    hac_maxlags : int or None
        HAC (Newey-West) bandwidth. If given, used verbatim. If None, resolved
        from `horizon` (see below).
    horizon : int or None
        Label horizon in observation periods. Overlapping h-period forward
        returns induce MA(h-1) structure, so the HAC bandwidth must satisfy
        L >= h - 1. When `hac_maxlags` is None, the bandwidth is
        `max(horizon - 1, cube-root-of-n)` if `horizon` is given, else the
        cube-root rule alone. Pass this for any overlapping label of horizon
        >= ~10 periods, or the standard error is understated and the
        t-statistic overstated.
    groups : array-like or None
        Ordered decision-time group for each observation. For panel data,
        supply the timestamp column so folds and embargoes operate on complete
        decision times rather than arbitrary rows.

    Returns
    -------
    dict
        Keys: theta, se_iid, se_hac, t_stat_iid, t_stat_hac, p_value_hac,
        n_obs, n_periods, hac_maxlags, covariance_type. If return_residuals:
        also Y_res, T_res.
    """
    # Pinned here, where the nuisance models are actually fitted, so that every caller is
    # covered: run_dml_analysis, the six case-study DML stages that call it directly, and
    # the chapter-15 notebooks that call this function themselves and rely on the default
    # HistGradientBoostingRegressor. The module-level OMP_NUM_THREADS setdefault does not
    # bind for any of them, because sklearn is imported first.
    with threadpool_limits(limits=thread_limit):
        seed_everything(RANDOM_SEED)

        n = len(Y)

        # Initialize residual arrays
        Y_res = np.full(n, np.nan)
        T_res = np.full(n, np.nan)

        folds = _walk_forward_indices(n, n_folds, embargo, groups=groups)

        for train_idx, test_idx in folds:
            if len(test_idx) == 0:
                continue

            if len(train_idx) < 50 or len(test_idx) < 10:
                continue

            # Fit nuisance models on training data (clone to avoid mutation)
            from sklearn.base import clone

            _default_y = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42)
            _default_t = HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42)
            my = clone(model_y) if model_y is not None else _default_y
            mt = clone(model_t) if model_t is not None else _default_t

            my.fit(X[train_idx], Y[train_idx])
            mt.fit(X[train_idx], T[train_idx])

            Y_res[test_idx] = Y[test_idx] - my.predict(X[test_idx])
            T_res[test_idx] = T[test_idx] - mt.predict(X[test_idx])

        # Drop observations without residuals
        valid = ~np.isnan(Y_res) & ~np.isnan(T_res)
        Y_v = Y_res[valid]
        T_v = T_res[valid]
        n_valid = len(Y_v)
        valid_groups = np.asarray(groups)[valid] if groups is not None else None
        n_periods = len(np.unique(valid_groups)) if valid_groups is not None else n_valid

        empty = {
            "theta": np.nan,
            "se_iid": np.nan,
            "se_hac": np.nan,
            "t_stat_iid": np.nan,
            "t_stat_hac": np.nan,
            "p_value_hac": np.nan,
            "n_obs": n_valid,
            "n_periods": n_periods,
            "hac_maxlags": 0,
            "covariance_type": "driscoll_kraay" if groups is not None else "newey_west",
        }
        if n_valid < 50:
            if return_residuals:
                empty["Y_res"] = Y_res
                empty["T_res"] = T_res
            return empty

        # Final stage: Y_res = alpha + theta * T_res + epsilon
        # Must include intercept: cross-fitting residuals may have non-zero mean
        # when training data varies across folds (expanding window).
        if hac_maxlags is None:
            auto = max(1, int(n_periods ** (1 / 3)))
            # Overlapping h-period labels need L >= h-1; the cube-root rule is
            # horizon-blind and under-lags long-horizon overlapping returns.
            hac_maxlags = max(horizon - 1, auto) if horizon else auto
            hac_maxlags = min(hac_maxlags, max(1, n_periods // 2))

        T_const = sm.add_constant(T_v)
        ols_iid = OLS(Y_v, T_const).fit()
        theta = ols_iid.params[1]

        # HC0 standard error
        se_iid = np.sqrt(ols_iid.cov_HC0[1, 1])

        # Serial-correlation-robust standard error with frequency-adaptive bandwidth.
        # Panel rows share decision times, so ordinary row-wise Newey-West treats
        # cross-sectional observations as extra time periods and understates risk.
        # Driscoll-Kraay aggregates the score by decision time and remains robust to
        # general cross-sectional dependence.
        se_hac = se_iid
        try:
            if valid_groups is not None:
                time_codes = pd.factorize(valid_groups, sort=False)[0]
                robust = ols_iid.get_robustcov_results(
                    cov_type="hac-groupsum",
                    time=time_codes,
                    maxlags=hac_maxlags,
                    use_correction="hac",
                    df_correction=False,
                )
            else:
                robust = ols_iid.get_robustcov_results(
                    cov_type="HAC",
                    maxlags=hac_maxlags,
                    use_correction=True,
                )
            cov = robust.cov_params()
            se_hac = np.sqrt(cov.iloc[1, 1] if hasattr(cov, "iloc") else cov[1, 1])
        except Exception:
            pass  # Fall back to HC0 standard errors on numerical failure

        t_stat_hac = theta / se_hac if se_hac > 0 else np.nan
        p_value_hac = (
            2 * stats.t.sf(abs(t_stat_hac), df=max(n_periods - 2, 1))
            if not np.isnan(t_stat_hac)
            else np.nan
        )

        result = {
            "theta": theta,
            "se_iid": se_iid,
            "se_hac": se_hac,
            "t_stat_iid": theta / se_iid if se_iid > 0 else np.nan,
            "t_stat_hac": t_stat_hac,
            "p_value_hac": p_value_hac,
            "n_obs": n_valid,
            "n_periods": n_periods,
            "hac_maxlags": hac_maxlags,
            "covariance_type": "driscoll_kraay" if groups is not None else "newey_west",
        }

        if return_residuals:
            result["Y_res"] = Y_res
            result["T_res"] = T_res

        return result


REFUTATION_ALPHA = 0.05


def _placebo_is_unchanged(original: np.ndarray, permuted: np.ndarray) -> bool:
    """Whether a placebo draw returned the observed treatment.

    ``np.array_equal`` calls two arrays different wherever either holds a NaN, so a
    frame the resolver's ``drop_nulls()`` never touched - which is every frame the
    case-study notebooks pass to ``run_dml_analysis`` directly - would report every
    identity draw as a real permutation. Compare the non-null positions and require
    the null positions to agree.
    """
    return not _placebo_moved_mask(original, permuted).any()


def _placebo_moved_fraction(original: np.ndarray, permuted: np.ndarray) -> float:
    """The share of comparable rows this draw actually moved.

    Separate from `_placebo_moved_mask` on purpose. That one answers "are these the same
    series at all", so a disagreement in null positions means every row counts as moved;
    here that convention would report a frozen panel as fully permuted the moment one
    draw shifted a missing value. Rows that are null on either side are simply not
    comparable, so they are left out of both the count and the denominator.
    """
    original = np.asarray(original)
    permuted = np.asarray(permuted)
    if original.shape != permuted.shape:
        return 0.0
    if not np.issubdtype(original.dtype, np.floating):
        comparable = np.ones(original.shape, dtype=bool)
    else:
        comparable = ~np.isnan(original) & ~np.isnan(permuted)
    if not comparable.any():
        return 0.0
    return float(np.mean(original[comparable] != permuted[comparable]))


def _placebo_moved_mask(original: np.ndarray, permuted: np.ndarray) -> np.ndarray:
    """Which comparable rows the permutation actually moved.

    ``True`` where the row differs, ``False`` where it sits still. Rows whose original
    value is missing are never comparable, so they are reported as unmoved rather than
    counted as evidence either way. A shape or null-position mismatch means the two are
    not the same series at all, so every row counts as moved.
    """
    original = np.asarray(original)
    permuted = np.asarray(permuted)
    if original.shape != permuted.shape:
        return np.ones(original.shape, dtype=bool)
    if not np.issubdtype(original.dtype, np.floating):
        return original != permuted
    missing = np.isnan(original)
    if not np.array_equal(missing, np.isnan(permuted)):
        return np.ones(original.shape, dtype=bool)
    moved = np.zeros(original.shape, dtype=bool)
    moved[~missing] = original[~missing] != permuted[~missing]
    return moved


# Below this many successful placebo draws the permutation test is not computed at all:
# the plus-one correction floors the empirical p at 1/(n+1), so under ten draws no data
# could produce a pass and a number would be reported that no test earned. When it is not
# computed, `refutation` stays empty and `refutation_p` is registered NULL - a missing
# measurement, which is what it is.
MIN_PLACEBO_DRAWS = 10


def placebo_request_is_on_the_boundary(n_placebo: int) -> bool:
    """Would one failed draw take the whole refutation with it?

    This is not enforced at run time, and the attempt to do so is worth recording. Nine
    tests in tests/test_causal_adapter.py request two to six draws on purpose, to
    exercise the block-span and permutation-feasibility logic without paying for a
    refutation they never read; refusing every small request turned all nine red for a
    property they are not about. The boundary is a property of a *declared reduction* -
    a config that says how a real run should be made cheap - not of every call.

    Zero means "do not refute", which is a different statement from "refute with too
    few draws to say anything".
    """
    return 0 < n_placebo < MIN_PLACEBO_DRAWS + PLACEBO_REQUEST_MARGIN


# Enough that one draw failing does not take the whole test with it. Small on purpose:
# a placebo draw is a full nuisance refit, so this is the least that makes the boundary
# unreachable by a single failure rather than a comfortable cushion.
PLACEBO_REQUEST_MARGIN = 5


def _assert_placebo_permutation_possible(
    unchanged_draws: int, n_draws: int, block_size: int, short_segment_fraction: float = 0.0
) -> None:
    """A refutation whose every placebo equals the observed treatment measures nothing.

    ``block_permute`` leaves a segment intact when it cannot hold two blocks of the
    requested size, which is the right thing to do to one short unit in a panel. If it
    happens to *every* segment - the block is larger than the longest uninterrupted
    stretch of observations - then the "permuted" treatment is the observed treatment,
    every placebo effect equals the observed effect, and the refutation reports p = 1
    while looking like it ran. Fail instead of publishing that.

    The test is over the whole set of draws, not each one. ``rng.permutation(n_blocks)``
    can return the identity by chance - one time in two at two blocks, one in six at
    three - so failing on a single unchanged draw aborts runs that are structurally
    fine, with a message asserting something false about the data. Every draw coming
    back unchanged is the structural condition; the chance of that happening to a
    series that can be permuted falls off as the draws multiply.
    """
    if n_draws and unchanged_draws == n_draws:
        raise ValueError(
            f"block permutation with block_size={block_size} left the treatment "
            f"unchanged on all {n_draws} placebo draws: no uninterrupted segment of "
            "the series holds two blocks of that size. Either the block size exceeds "
            "the data's contiguous runs or the gap tolerance is splitting the series "
            "too finely."
        )
    if n_draws and short_segment_fraction > 0:
        warnings.warn(
            f"block permutation with block_size={block_size} cannot move "
            f"{short_segment_fraction:.1%} of the treatment rows: they sit in segments "
            "too short to hold two blocks, so the placebo distribution holds them at "
            "their observed values and the refutation p-value is biased toward 1. Read "
            "placebo_frozen_fraction alongside the p-value, and lower block_size or "
            "widen gap_tolerance if the frozen share is large.",
            UserWarning,
            stacklevel=3,
        )


def empirical_permutation_p(placebo_effects: np.ndarray, observed_effect: float) -> float:
    """Two-sided Monte Carlo p-value for the block-permutation refutation.

    The observed statistic is itself one draw the permutation distribution can
    produce, so both the count and the denominator take the plus-one correction
    (Davison and Hinkley 1997; Phipson and Smyth 2010). Without it, a run in
    which no placebo reaches the observed effect reports ``p = 0.000`` - a claim
    no finite number of permutations can support. With ``n`` placebo draws the
    smallest p-value the test can report is ``1 / (n + 1)``.

    Parameters
    ----------
    placebo_effects : np.ndarray
        Treatment effects from the successful placebo permutations.
    observed_effect : float
        The treatment effect estimated on the unpermuted data.

    Returns
    -------
    float
        The fraction of the permutation distribution at least as extreme as the
        observed effect in absolute value, in ``(0, 1]``.
    """
    placebo = np.asarray(placebo_effects, dtype=float)
    at_least_as_extreme = int(np.sum(np.abs(placebo) >= abs(observed_effect)))
    return (1.0 + at_least_as_extreme) / (1.0 + placebo.size)


def classify_refutation(empirical_p: float, n_successful: int | None = None) -> str:
    """Pass, fail, or too few draws to tell, at the 5 % level.

    Returns "Passes" if the empirical placebo p-value is below 5 % (the observed effect
    cannot be reproduced by permutation in most placebo runs), and "Fails" otherwise.

    "Underpowered" is the third answer, and it is the honest one whenever the number of
    successful draws puts "Passes" out of reach. The plus-one correction floors the
    reported p-value at ``1 / (n + 1)``, so at 19 successful draws or fewer the smallest
    value the test can produce is already at or above 5 % and "Fails" would be published
    whatever the data said - untrue by construction in the same way ``p = 0.000`` was at
    the other end. The preview tier runs ten draws, so this is the ordinary case there,
    not an exotic one.

    ``n_successful`` is optional so that a caller holding only a p-value keeps the
    two-way answer; pass it wherever the draw count is known.
    """
    if n_successful is not None and 1.0 / (n_successful + 1) >= REFUTATION_ALPHA:
        return "Underpowered"
    return "Passes" if empirical_p < REFUTATION_ALPHA else "Fails"


def _resolve_panel_columns(
    df: pd.DataFrame,
    time_col: str | None,
    entity_col: str | None,
) -> tuple[str | None, str | None]:
    """Resolve canonical panel columns while preserving single-series inputs."""
    if time_col is None and "timestamp" in df.columns:
        time_col = "timestamp"
    if entity_col is None:
        entity_col = next((name for name in ("symbol", "product") if name in df.columns), None)
    if entity_col is not None and time_col is None:
        raise ValueError("time_col is required when an entity column is present")
    if time_col is not None and df[time_col].duplicated().any() and entity_col is None:
        raise ValueError("entity_col is required when decision times contain multiple rows")
    return time_col, entity_col


def run_dml_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
    n_folds: int = 5,
    embargo: int = 21,
    n_placebo: int = 100,
    block_size: int = 21,
    seed: int = 42,
    hac_maxlags: int | None = None,
    horizon: int | None = None,
    time_col: str | None = None,
    entity_col: str | None = None,
    model_y=None,
    model_t=None,
    expected_step: str | pd.Timedelta | None = None,
    thread_limit: int = DML_THREAD_LIMIT,
) -> dict:
    """Full DML analysis pipeline: naive OLS, DML, and refutation tests.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset sorted by time.
    treatment_col : str
        Treatment variable column name.
    outcome_col : str
        Outcome variable column name.
    confounder_cols : list[str]
        Confounder column names.
    n_folds : int
        Number of walk-forward CV folds.
    embargo : int
        Gap periods between train and test.
    n_placebo : int
        Number of block permutation replications.
    block_size : int
        Block size for permutation test.
    seed : int
        Random seed.
    hac_maxlags : int or None
        HAC bandwidth passed through to the second stage. If None, resolved
        from `horizon`.
    horizon : int or None
        Label horizon in observation periods, forwarded to the second-stage
        HAC regression so the Newey-West bandwidth satisfies L >= horizon - 1.
        Pass it for overlapping labels (horizon >= ~10). It is the outcome
        horizon, read with
        `resolve_label_horizon(case_study_id, label, setup)`, and not the CV
        buffer, which bounds a different quantity and can be longer. When both
        `horizon` and `hac_maxlags` are None, the bandwidth falls back to the
        horizon-blind cube-root rule and a warning is emitted.
    time_col : str or None
        Ordered decision-time column. Inferred from canonical ``timestamp``
        when omitted. Required for panel data so cross-fitting, embargoes, and
        placebo blocks keep each decision time intact.
    entity_col : str or None
        Panel entity column. Inferred from canonical ``symbol`` or ``product``
        when omitted. Supply with ``time_col`` for non-canonical panels so
        placebo blocks are permuted within entity histories.

    Returns
    -------
    dict
        Comprehensive results with keys: naive_effect, naive_n_obs, dml_result,
        confounding_bias, confounding_bias_pct, refutation (z_score,
        empirical_p, placebo_mean, placebo_std, placebo_effects,
        refutation_class), p_value_hac, hac_maxlags, and n_obs.
    """
    # Also wrapped here, not only inside manual_dml_timeseries. The naive-OLS comparison
    # below runs np.linalg.lstsq after that call returns, and LAPACK's dgelsd reaches
    # threaded BLAS on a tall design - so naive_effect, and confounding_bias_pct which is a
    # difference between it and the pinned theta, would otherwise vary with the ambient
    # pool while the spec records deterministic_reduction: True. The inner context nests
    # harmlessly and still covers callers that use manual_dml_timeseries directly.
    with threadpool_limits(limits=thread_limit):
        # Input validation
        time_col, entity_col = _resolve_panel_columns(df, time_col, entity_col)
        n = len(df)
        min_rows = (n_folds + 1) * 50 + n_folds * embargo
        if n < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows for {n_folds}-fold CV with embargo={embargo}, got {n}"
            )
        if df[treatment_col].std() < 1e-10:
            raise ValueError(f"Treatment '{treatment_col}' has near-zero variance")
        if df[outcome_col].std() < 1e-10:
            raise ValueError(f"Outcome '{outcome_col}' has near-zero variance")

        if hac_maxlags is None and horizon is None:
            import warnings

            warnings.warn(
                "run_dml_analysis: no horizon or hac_maxlags given; the second-stage "
                "HAC bandwidth falls back to the horizon-blind cube-root rule, which "
                "under-lags overlapping labels of horizon >= ~10 and overstates the "
                "t-statistic. Pass the outcome horizon in observation periods: read the "
                "horizon with resolve_label_horizon(case_study_id, label, setup) - not the "
                "CV buffer, which can be longer - and convert it against the panel's own "
                "cadence. embargo_from_buffer without observed_step applies per-unit "
                "defaults instead, which read 24H as one period on an eight-hour panel.",
                stacklevel=2,
            )

        _dml_started_at = datetime.now(UTC).isoformat()
        _dml_t0 = time.perf_counter()

        rng = np.random.default_rng(seed)

        T = df[treatment_col].values
        Y = df[outcome_col].values
        X = df[confounder_cols].values
        groups = df[time_col].values if time_col is not None else None
        units = df[entity_col].values if entity_col is not None else None

        # DML estimate
        dml = manual_dml_timeseries(
            Y,
            T,
            X,
            n_folds=n_folds,
            embargo=embargo,
            return_residuals=True,
            hac_maxlags=hac_maxlags,
            horizon=horizon,
            groups=groups,
            model_y=model_y,
            model_t=model_t,
            thread_limit=thread_limit,
        )

        # Compare the adjusted estimate with naive OLS on the exact second-stage
        # population. Earlier walk-forward dates have no out-of-fold residuals and
        # cannot enter only one side of the comparison.
        valid = np.isfinite(dml["Y_res"]) & np.isfinite(dml["T_res"])
        naive_n_obs = int(valid.sum())
        T_const = np.column_stack([np.ones(naive_n_obs), T[valid]])
        naive_coef = np.linalg.lstsq(T_const, Y[valid], rcond=None)[0]
        naive_effect = naive_coef[1]

        # Confounding bias
        dml_effect = dml["theta"]
        bias = naive_effect - dml_effect
        bias_pct = bias / abs(dml_effect) * 100 if dml_effect != 0 else 0.0

        # Block permutation refutation
        placebo_effects = []
        placebo_n_obs = []
        unchanged_draws = 0
        moved_fractions: list[float] = []
        # What no draw can move is a property of the segmentation and the block size, so
        # it is computed from them rather than inferred from what the draws happened to
        # do. The two reasons a row never moves are unrelated and only one of them is
        # worth a warning: see `_immobile_masks`.
        short_frozen, remainder_frozen = _immobile_masks(
            len(T),
            block_size,
            _permutation_segments(len(T), groups, units, expected_step, None),
        )
        for _ in range(n_placebo):
            T_perm = block_permute(
                T,
                block_size,
                rng=rng,
                groups=groups,
                units=units,
                expected_step=expected_step,
            )
            moved_fractions.append(_placebo_moved_fraction(T, T_perm))
            unchanged_draws += _placebo_is_unchanged(T, T_perm)
            perm_result = manual_dml_timeseries(
                Y,
                T_perm,
                X,
                n_folds=n_folds,
                embargo=embargo,
                hac_maxlags=hac_maxlags,
                horizon=horizon,
                groups=groups,
                model_y=model_y,
                model_t=model_t,
                thread_limit=thread_limit,
            )
            if not np.isnan(perm_result["theta"]):
                if perm_result["n_obs"] != dml["n_obs"]:
                    raise RuntimeError(
                        "Observed and placebo DML statistics use different second-stage samples"
                    )
                placebo_effects.append(perm_result["theta"])
                placebo_n_obs.append(int(perm_result["n_obs"]))

        frozen_fraction = float(short_frozen.mean()) if short_frozen.size else 0.0
        remainder_fraction = float(remainder_frozen.mean()) if remainder_frozen.size else 0.0
        _assert_placebo_permutation_possible(
            unchanged_draws, n_placebo, block_size, frozen_fraction
        )

        refutation = {}
        if len(placebo_effects) >= MIN_PLACEBO_DRAWS:
            placebo_arr = np.array(placebo_effects)
            p_mean = np.mean(placebo_arr)
            p_std = np.std(placebo_arr)
            z = (dml_effect - p_mean) / p_std if p_std > 0 else np.inf
            emp_p = empirical_permutation_p(placebo_arr, dml_effect)
            ref_class = classify_refutation(emp_p, len(placebo_effects))
            refutation = {
                "z_score": z,
                "empirical_p": emp_p,
                "placebo_mean": p_mean,
                "placebo_std": p_std,
                "n_successful": len(placebo_effects),
                "placebo_frozen_fraction": frozen_fraction,
                "placebo_remainder_fraction": remainder_fraction,
                "placebo_moved_fraction": float(np.mean(moved_fractions))
                if moved_fractions
                else 0.0,
                "n_folds": n_folds,
                "placebo_n_obs": placebo_n_obs,
                "placebo_effects": placebo_effects,
                "refutation_class": ref_class,
            }

        return {
            "naive_effect": naive_effect,
            "naive_n_obs": naive_n_obs,
            "dml_result": dml,
            "confounding_bias": bias,
            "confounding_bias_pct": bias_pct,
            "refutation": refutation,
            "p_value_hac": dml.get("p_value_hac", np.nan),
            "hac_maxlags": dml.get("hac_maxlags", 0),
            "n_obs": len(df),
            "started_at": _dml_started_at,
            "elapsed_s": time.perf_counter() - _dml_t0,
        }


def format_dml_summary(results: dict) -> str:
    """Format DML analysis results for display."""
    dml = results["dml_result"]
    p_hac = results.get("p_value_hac", dml.get("p_value_hac", np.nan))
    hac_lags = results.get("hac_maxlags", dml.get("hac_maxlags", "?"))
    lines = [
        "=" * 60,
        "DML ANALYSIS SUMMARY",
        "=" * 60,
        f"Analysis rows: {results['n_obs']:,}",
        f"Second-stage rows: {dml.get('n_obs', results['n_obs']):,}",
        f"Second-stage decision times: {dml.get('n_periods', results['n_obs']):,}",
        f"Covariance: {dml.get('covariance_type', 'newey_west').replace('_', '-').title()}",
        f"HAC bandwidth: {hac_lags} lags (max of horizon-1 and cube-root)",
        "",
        f"Naive OLS rows:    {results.get('naive_n_obs', results['n_obs']):,}",
        f"Naive OLS effect:  {results['naive_effect']:.6f}",
        f"DML effect:        {dml['theta']:.6f}",
        f"  SE (IID):        {dml['se_iid']:.6f}",
        f"  SE (HAC):        {dml['se_hac']:.6f}",
        f"  t-stat (HAC):    {dml['t_stat_hac']:.2f}",
        f"  p-value (HAC):   {p_hac:.4f}",
        "",
        f"Confounding bias:  {results['confounding_bias']:.6f} ({results['confounding_bias_pct']:+.1f}%)",
    ]

    ref = results.get("refutation", {})
    if ref:
        ref_class = ref.get("refutation_class", classify_refutation(ref["empirical_p"]))
        lines += [
            "",
            "Refutation (block permutation):",
            f"  Z-score:      {ref['z_score']:.2f}",
            f"  Empirical p:  {ref['empirical_p']:.4f}",
            f"  Classification: {ref_class}",
            f"  Placebos:     {ref['n_successful']}",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reader-facing causal request adapter
# ---------------------------------------------------------------------------


def _causal_source_identity() -> dict[str, str]:
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def _causal_runtime_identity() -> dict[str, str]:
    return {
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "statsmodels": importlib.metadata.version("statsmodels"),
    }


def _causal_runtime_provenance(study: Study) -> dict[str, Any]:
    return {
        "entry_point": "case_studies.utils.causal",
        "packages": _causal_runtime_identity(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_commit": study.manifest.get("baseline_source_commit", "unknown"),
    }


def _whole_timestamp_tail(
    frame,
    *,
    timestamp: str,
    entity: str,
    max_rows: int,
):
    import polars as pl

    ordered = frame.sort([timestamp, entity])
    if max_rows <= 0 or ordered.height <= max_rows:
        return ordered
    counts = ordered.group_by(timestamp).len().sort(timestamp)
    counts = counts.with_columns(pl.col("len").reverse().cum_sum().reverse().alias("suffix_n"))
    keep = counts.filter(pl.col("suffix_n") <= max_rows).select(timestamp)
    if keep.is_empty():
        raise ValueError("max_samples is smaller than the final timestamp panel")
    return ordered.join(keep, on=timestamp, how="semi").sort([timestamp, entity])


def _observed_cadence(frame, timestamp: str) -> pd.Timedelta:
    values = pd.DatetimeIndex(frame.get_column(timestamp).unique().sort().to_list())
    if len(values) < 2:
        raise ValueError("causal analysis needs at least two decision timestamps")
    differences = pd.Series(values[1:] - values[:-1])
    cadence = differences.mode().iloc[0]
    if cadence <= pd.Timedelta(0):
        raise ValueError("causal observation cadence must be positive")
    return pd.Timedelta(cadence)


def _treatment_persistence_steps(setup: dict[str, Any], treatment: str) -> int | None:
    """Bars the treatment's own construction window spans, from `features.windows`.

    A 14-day z-score is autocorrelated over its whole window whatever the label
    horizon is, so a placebo that permutes in blocks shorter than that window
    destroys the dependence the permutation exists to preserve. `setup.yaml`
    already declares every window once, keyed by the suffix the emitted column
    carries, so the number is read from there rather than parsed out of the
    column name. Returns None when the register declares no window for this
    treatment, which the caller records rather than papers over.
    """
    windows = (setup.get("features") or {}).get("windows") or {}
    for family, suffixes in windows.items():
        prefix = f"{family}_"
        if not treatment.startswith(prefix):
            continue
        # Only the suffix-keyed mapping says which window *this* column carries.
        # The register also holds bare ints and lists of bar counts for families
        # whose columns are named some other way (etfs `skip_recent: 21`, and
        # `momentum: [5, 10, 21, ...]`), and guessing which element a treatment
        # was built from would put a wrong block size behind a right-looking
        # number. Those return None, and the caller warns.
        if not isinstance(suffixes, dict):
            continue
        declared = suffixes.get(treatment[len(prefix) :])
        if declared is not None:
            return max(1, int(declared))
    return None


def _resolve_nuisance_params(config: dict[str, Any], overrides: dict[str, Any], seed: int):
    configured = dict(config.get("params") or {})
    supplied = dict(overrides.get("nuisance_params") or {})
    estimator = HistGradientBoostingRegressor(random_state=seed)
    unknown = (set(configured) | set(supplied)) - set(estimator.get_params(deep=True))
    if unknown:
        raise ValueError(f"unsupported DML nuisance parameters: {sorted(unknown)}")
    estimator.set_params(**configured, **supplied)
    return estimator.get_params(deep=True)


def resolve_causal_request(study: Study, request: dict[str, Any]):
    import polars as pl
    import yaml

    from case_studies.research.contracts import ExecutionTier
    from case_studies.research.identity import ResolvedSpec
    from case_studies.utils.artifact_digest import value_digest
    from utils.artifact_specs import resolve_label_horizon
    from utils.modeling import load_configs, load_modeling_dataset

    tier = ExecutionTier(request["execution_tier"])
    reductions = dict(request["preview_reductions"])
    unknown_reductions = set(reductions) - _DML_PREVIEW_FIELDS
    if unknown_reductions:
        raise ValueError(f"unsupported DML preview reductions: {sorted(unknown_reductions)}")
    # Every field, not merely a non-empty dict. This path no longer falls back to the shared
    # preset's max_samples, so a preview that omits it would silently resolve the full
    # population - an uncapped preview, which is the opposite of what the tier is for.
    if tier is ExecutionTier.PREVIEW:
        missing_reductions = _DML_PREVIEW_FIELDS - set(reductions)
        if missing_reductions:
            raise ValueError(
                "preview causal requests must declare every reduction; missing "
                f"{sorted(missing_reductions)}"
            )
    allowed_overrides = {"nuisance_params"}
    unknown_overrides = set(request["overrides"]) - allowed_overrides
    if unknown_overrides:
        raise ValueError(f"unsupported DML overrides: {sorted(unknown_overrides)}")

    study.require_writable()
    study.activate(tier)
    label_ref = study.labels.get(request["label"], execution_tier=tier)
    mds = load_modeling_dataset(
        study.case_study,
        label_ref.name,
        max_symbols=int(reductions.get("max_symbols", 0)),
    )
    if mds.date_col != "timestamp" or not mds.entity_cols:
        raise ValueError("DML runner requires timestamp and an entity key")
    if mds.entity_cols[0] not in {"product", "symbol"}:
        raise ValueError(f"DML runner does not support entity key {mds.entity_cols[0]!r}")
    configs = {
        config["config_name"]: config
        for config in load_configs(study.case_study, label_ref.name, "causal_dml")
    }
    try:
        config = configs[request["config_name"]]
    except KeyError as error:
        raise ValueError(f"unknown DML configuration {request['config_name']!r}") from error
    setup = yaml.safe_load((study.root / "config" / "setup.yaml").read_text()) or {}
    causal = setup.get("causal") or {}
    treatment = str(causal["treatment"])
    confounders = tuple(str(value) for value in causal["confounders"])
    seed = int(config.get("seed", RANDOM_SEED))
    n_folds = int(reductions.get("n_folds", config.get("n_folds", 5)))
    n_placebo = int(reductions.get("n_placebo", config.get("n_placebo", 100)))
    # The shared preset still declares max_samples for the six case-study DML stages that have
    # not migrated to this path and read it as their own default. This path ignores it: a
    # canonical run uses the full declared population, and a reduction reaches it only through
    # preview_reductions, which research/causal.py refuses for a canonical request. So the
    # preset cannot cap a canonical sample, and the resolved spec records max_samples: 0.
    max_samples = int(reductions.get("max_samples", 0))
    if n_folds < 2 or n_placebo < 0 or max_samples < 0:
        raise ValueError("DML folds, placebos, and sample cap are invalid")

    columns = [mds.date_col, mds.entity_cols[0], treatment, mds.label_col, *confounders]
    missing = sorted(set(columns) - set(mds.dataset.columns))
    if missing:
        raise ValueError(f"DML analysis columns are missing: {missing}")
    holdout = pd.Timestamp(setup["evaluation"]["holdout_start"], tz="UTC")
    # Two quantities, and one name used to stand for both. The buffer keeps a fold's training
    # rows clear of its validation labels and bounds the pre-holdout window; the outcome horizon
    # is how long one outcome stays open, which is what makes successive outcomes overlap and
    # therefore what the Newey-West bandwidth has to cover. `resolve_label_horizon` reads the
    # second and its docstring says it may be shorter than the first. They agree in most case
    # studies here, but by configuration rather than by construction, so deriving the bandwidth
    # from the buffer is right by luck wherever it is right at all.
    buffer_delta = pd.Timedelta(str(mds.label_buffer).replace("H", "h"))
    outcome_delta = pd.Timedelta(
        str(
            resolve_label_horizon(study.case_study, label_ref.name, setup) or mds.label_buffer
        ).replace("H", "h")
    )
    # The seal and the placebo block below both take the buffer, on the grounds that it is the
    # longer of the two. Nothing enforces that: both values are hand-authored in setup.yaml and
    # `resolve_label_horizon` promises only that the horizon *may* be shorter. A case study
    # declaring a horizon longer than its buffer would get a permutation block shorter than the
    # dependence it exists to hold fixed, and a seal that leaves outcomes reaching into the
    # holdout, both silently.
    if outcome_delta > buffer_delta:
        raise ValueError(
            f"outcome horizon {outcome_delta} exceeds the CV buffer {buffer_delta} for "
            f"{label_ref.name!r}. The buffer bounds the pre-holdout window and sizes the "
            "placebo block, so it cannot be shorter than the outcome it is holding clear. "
            "Raise labels.buffer (or labels.variant_buffers) in setup.yaml, or correct "
            "labels.horizons."
        )
    date_dtype = mds.dataset.schema[mds.date_col]
    # The cutoff steps back a count of observations, not a calendar duration. The horizons
    # these buffers describe are counted in the panel's own observations, so subtracting the
    # buffer as calendar time answers a different question on any gapped calendar. A 21D
    # buffer is roughly fifteen sessions on a five-session week, which left the last few
    # retained rows resolving their returns inside the holdout, silently. The two
    # constructions agree only on a continuous panel, which is why this held wherever anyone
    # had checked it.
    #
    # The cadence therefore has to be measured before the cutoff rather than after it, and on
    # the pre-holdout observations alone: `_observed_cadence` takes the mode of the gaps, so a
    # panel whose spacing changes across the holdout boundary would otherwise size the buffer
    # and the embargo from rows the analysis never touches.
    entity_col = mds.entity_cols[0]
    populated = mds.dataset.select(columns).drop_nulls()
    pre_holdout = populated.filter(
        pl.col(mds.date_col) < pl.lit(holdout.to_pydatetime()).cast(date_dtype, strict=False)
    ).select(entity_col, mds.date_col)
    if pre_holdout.is_empty():
        raise ValueError("DML request resolved an empty pre-holdout analysis frame")
    cadence = _observed_cadence(pre_holdout, mds.date_col)
    buffer_steps = max(1, int(np.ceil(buffer_delta / cadence)))
    outcome_horizon_steps = max(1, int(np.ceil(outcome_delta / cadence)))
    # The step-back is counted within each entity, not across the panel's distinct
    # timestamps. A label advances by that entity's own observations, so on a panel where
    # one product is missing some of the final sessions, a global count reaches back fewer
    # of that product's observations than the buffer names and leaves its last rows
    # resolving inside the holdout. The panel-wide count is only correct when every entity
    # trades every session, which is a property of the data rather than of this function.
    #
    # Each entity is then sealed against its own cutoff rather than against a panel-wide
    # collapse of them. Taking the earliest and applying it everywhere would let one
    # entity whose history ends early - a delisted firm, a contract that stopped trading -
    # drag the boundary back to its own exit date and silently truncate every other
    # entity to an early slice of the panel.
    #
    # An entity that does not hold more than `buffer_steps` observations before the
    # holdout has no row whose outcome resolves before it, so it contributes no cutoff and
    # drops out entirely.
    per_entity = (
        pre_holdout.unique()
        .sort(entity_col, mds.date_col)
        .group_by(entity_col)
        .agg(
            pl.col(mds.date_col).len().alias("observations"),
            pl.col(mds.date_col).alias("timestamps"),
        )
        .filter(pl.col("observations") > buffer_steps)
        .with_columns(
            pl.col("timestamps")
            .list.get(pl.col("observations") - buffer_steps)
            .alias("entity_cutoff")
        )
    )
    if per_entity.is_empty():
        raise ValueError(
            f"no entity holds more than {buffer_steps} observations before "
            f"{holdout.date()}, so none can absorb the buffer and leave anything to analyse"
        )
    analysis = (
        populated.join(per_entity.select(entity_col, "entity_cutoff"), on=entity_col, how="inner")
        .filter(pl.col(mds.date_col) < pl.col("entity_cutoff"))
        .select(columns)
    )
    if analysis.is_empty():
        raise ValueError("DML request resolved an empty pre-holdout analysis frame")
    # The estimand records one boundary, and with a per-entity seal the honest scalar is
    # the loosest of them: no entity retains a row at or after its own cutoff, and none
    # of those cutoffs is later than this one.
    endpoint_cutoff = pd.Timestamp(per_entity.get_column("entity_cutoff").max())
    analysis = _whole_timestamp_tail(
        analysis,
        timestamp=mds.date_col,
        entity=mds.entity_cols[0],
        max_rows=max_samples,
    )
    if analysis.is_empty():
        raise ValueError("DML request resolved an empty pre-holdout analysis frame")
    # The cadence is measured on the pre-holdout observations above, so the embargo
    # is counted against it rather than against an assumed bar size. Leaving the
    # fallback here would make the embargo short by the ratio between the two on
    # any panel whose real cadence differs from its declared one, while the
    # horizon computed from that same cadence stayed right. A month buffer has no
    # fixed length and keeps the calendar branch.
    try:
        embargo = embargo_from_buffer(mds.label_buffer, observed_step=cadence)
    except ValueError:
        embargo = embargo_from_buffer(mds.label_buffer)
    # Two separate scales create the serial dependence the placebo has to preserve, and
    # the block spans the longer of them: the overlapping labels span the label horizon,
    # and the treatment's own construction window spans itself. Sizing by the horizon
    # alone permutes a 42-bar rolling z-score in blocks of one on an 8-hour panel, which
    # is the iid shuffle `block_permute` exists to avoid.
    treatment_window_steps = _treatment_persistence_steps(setup, treatment)
    if treatment_window_steps is None:
        # A canonical run refuses rather than warns. The warning was a strict improvement
        # on the silent one-bar block that shipped before it, but it is still a registered
        # result whose refutation is weaker than it reads: fx_pairs' `mom_skip_recent`
        # spans observations 21 through 252, resolves to None here, and would take a
        # one-bar block against its one-day label - a full within-symbol shuffle, which is
        # exactly the dependence the placebo exists to preserve. Nothing downstream can
        # tell that result from one whose block was sized correctly.
        #
        # Preview keeps the warning. The tier exists to run reduced and be thrown away, and
        # failing it would block CI on every case study that has not yet declared a window
        # without protecting any registered number.
        if tier is not ExecutionTier.PREVIEW:
            raise ValueError(
                f"{study.case_study}: features.windows declares no construction window for "
                f"treatment {treatment!r}, so the placebo block would span only the label "
                f"buffer ({buffer_steps} bars). Declare the treatment's window in "
                "setup.yaml under `features.windows`; a canonical refutation will not be "
                "registered against a block that cannot be shown to span the treatment."
            )
        warnings.warn(
            f"{study.case_study}: features.windows declares no construction window for "
            f"treatment {treatment!r}, so the placebo block spans only the label buffer "
            f"({buffer_steps} bars). If the treatment is a rolling statistic, declare its "
            "window so the block can span it.",
            UserWarning,
            stacklevel=2,
        )
    block_size = max(buffer_steps, treatment_window_steps or 1)
    block_size_basis = (
        "treatment_window" if block_size == treatment_window_steps else "label_buffer"
    )
    nuisance_params = _resolve_nuisance_params(config, request["overrides"], seed)
    key_frame = analysis.select(mds.entity_cols[0], mds.date_col)
    if key_frame.n_unique([mds.entity_cols[0], mds.date_col]) != key_frame.height:
        raise ValueError("DML analysis keys are not unique")

    computation = {
        "label_artifact": {"digest": label_ref.digest, "name": label_ref.name},
        "feature_artifacts": mds.input_lineage["artifacts"],
        "feature_names": list(mds.feature_names),
        "estimand": {
            "method": "walk_forward_dml",
            "outcome": mds.label_col,
            "treatment": treatment,
            "confounders": list(confounders),
            "treatment_observed_at": "decision_timestamp",
            "outcome_horizon": str(outcome_delta),
            "holdout_endpoint_cutoff": endpoint_cutoff.isoformat(),
        },
        "cv": {
            "n_folds": n_folds,
            "embargo_periods": embargo,
            "fold_unit": "complete_timestamp_panel",
        },
        "model": {
            "class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "implementation": "scikit-learn",
            "nuisance_params": nuisance_params,
        },
        "numerics": {
            "thread_limit": DML_THREAD_LIMIT,
            "deterministic_reduction": True,
        },
        "refutation": {
            "method": "within_symbol_contiguous_block_permutation",
            "n_placebo": n_placebo,
            "block_size": block_size,
            "block_size_basis": block_size_basis,
            "label_buffer_steps": buffer_steps,
            "treatment_window_steps": treatment_window_steps,
            "seed": seed,
            "temporal_gap_policy": "reset",
            "temporal_gap_tolerance_cadences": GAP_TOLERANCE_CADENCES,
            "observation_cadence": str(cadence),
        },
        "analysis_population": {
            "key_digest": value_digest(key_frame, (mds.entity_cols[0], mds.date_col)),
            "n_rows": analysis.height,
            "n_timestamps": analysis.get_column(mds.date_col).n_unique(),
            "max_samples": max_samples,
        },
        "input_data_spec": mds.input_lineage,
        "source_identity": _causal_source_identity(),
        "runtime_identity": _causal_runtime_identity(),
    }
    if tier is ExecutionTier.PREVIEW:
        computation["preview_reductions"] = reductions
    provenance = _causal_runtime_provenance(study)
    spec = ResolvedSpec.create(
        family="causal_dml",
        label=label_ref.name,
        seed=seed,
        computation=computation,
        provenance=provenance,
        config_name=config["config_name"],
        execution_tier=tier.value,
    ).as_dict()
    context = DMLResearchContext(
        analysis=analysis.to_pandas(),
        treatment_col=treatment,
        outcome_col=mds.label_col,
        confounder_cols=confounders,
        time_col=mds.date_col,
        entity_col=mds.entity_cols[0],
        n_folds=n_folds,
        embargo=embargo,
        n_placebo=n_placebo,
        # Main's split stands: the block takes the buffer, the bandwidth takes the
        # outcome horizon. The buffer measures the dependence the fold structure already
        # keeps clear; the bandwidth corrects for outcomes that overlap, and how far one
        # outcome reaches is the horizon. Newey-West is not monotonic in its bandwidth, so
        # a bandwidth taken from the wrong quantity is not a known direction of error.
        #
        # The treatment's own construction window is a third scale, and the block spans it
        # too. A 14-day z-score stays autocorrelated over its whole window whatever the
        # buffer is, so a block sized by the buffer alone permutes it in blocks of one on
        # an 8-hour panel - the iid shuffle `block_permute` exists to avoid.
        # `block_size_basis` records which of the two set it.
        block_size=block_size,
        seed=seed,
        horizon=outcome_horizon_steps,
        expected_step=cadence,
        nuisance_params=nuisance_params,
        runtime_provenance=provenance,
    )
    return spec, context


def run_resolved_causal_request(
    study: Study,
    spec: dict[str, Any],
    context: DMLResearchContext,
    *,
    supersedes: str | None = None,
):
    import math

    from case_studies.research.causal import CausalResult
    from case_studies.utils.registry.registration import (
        declare_causal_supersedes,
    )
    from case_studies.utils.registry.registration import register_causal_run as register_record
    from case_studies.utils.registry.specs import canonical_json, training_hash_from_spec

    causal_hash = training_hash_from_spec(spec)
    try:
        cached = CausalResult.open(
            study,
            causal_hash,
            include_preview=spec["execution_tier"] == "preview",
        )
    except KeyError:
        cached = None
    if cached is not None:
        if training_hash_from_spec(cached.spec) != causal_hash or not cached.complete:
            raise ValueError(f"causal cache is incomplete or conflicts with {causal_hash}")
        if supersedes is not None:
            # The declaration has to land even when the fit does not re-run, because
            # that is the shape of the repair. A registry already holding two
            # undeclared identities tells the author to re-register the newer one
            # naming the older; doing so reproduces the same causal_hash, so the cache
            # answers and the fit is skipped. Returning here without writing would drop
            # the declaration silently and leave the label unresolvable, with the
            # notebook reporting success. register_causal_run fills the column
            # once and validates what it is given.
            declare_causal_supersedes(
                study.case_study,
                causal_hash,
                supersedes_hash=supersedes,
                label=context.outcome_col,
                tier=str(spec["execution_tier"]),
                case_dir=study.storage_root(spec["execution_tier"]),
            )
        return cached

    nuisance_y = HistGradientBoostingRegressor(**context.nuisance_params)
    nuisance_t = HistGradientBoostingRegressor(**context.nuisance_params)
    thread_limit = int(
        spec["computation"].get("numerics", {}).get("thread_limit", DML_THREAD_LIMIT)
    )
    results = run_dml_analysis(
        context.analysis,
        context.treatment_col,
        context.outcome_col,
        list(context.confounder_cols),
        n_folds=context.n_folds,
        embargo=context.embargo,
        n_placebo=context.n_placebo,
        block_size=context.block_size,
        seed=context.seed,
        horizon=context.horizon,
        time_col=context.time_col,
        entity_col=context.entity_col,
        model_y=nuisance_y,
        model_t=nuisance_t,
        expected_step=context.expected_step,
        thread_limit=thread_limit,
    )
    dml = results["dml_result"]
    if not all(math.isfinite(float(dml[name])) for name in ("theta", "se_hac")):
        raise ValueError("DML fit did not produce a finite effect and HAC standard error")
    refutation = results.get("refutation", {})
    refutation_p = refutation.get("empirical_p")
    refutation_n = refutation.get("n_successful")
    case_dir = study.storage_root(spec["execution_tier"])
    register_record(
        study.case_study,
        causal_hash,
        # The identity this run retires, from the notebook's SUPERSEDES_CAUSAL. A refit
        # under a changed version of this file produces a second canonical identity for
        # the same label, and a reader resolves a label to exactly one; register_record
        # refuses the second without a declaration.
        supersedes_hash=supersedes,
        label=context.outcome_col,
        treatment=context.treatment_col,
        confounders_json=json.dumps(list(context.confounder_cols)),
        embargo=context.embargo,
        n_folds=context.n_folds,
        n_obs=int(dml["n_obs"]),
        dml_effect=float(dml["theta"]),
        dml_se_hac=float(dml["se_hac"]),
        p_value_hac=float(results["p_value_hac"]),
        naive_effect=float(results["naive_effect"]),
        confounding_bias_pct=float(results["confounding_bias_pct"]),
        refutation_p=float(refutation_p) if refutation_p is not None else None,
        refutation_n_successful=int(refutation_n) if refutation_n is not None else None,
        spec_json=canonical_json(spec),
        notebook="case_studies.utils.causal",
        started_at=results.get("started_at"),
        elapsed_s=results.get("elapsed_s"),
        case_dir=case_dir,
    )
    return CausalResult.open(
        study,
        causal_hash,
        include_preview=spec["execution_tier"] == "preview",
    )


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def register_causal_run(
    case_study_id: str,
    label: str,
    results: dict,
    predictions=None,
    *,
    treatment_col: str = "",
    confounder_cols: list[str] | None = None,
    n_folds: int = 5,
    embargo: int = 0,
    time_col: str | None = None,
    block_size: int | None = None,
    n_placebo: int | None = None,
    seed: int | None = None,
    horizon: int | None = None,
    max_samples: int | None = None,
    max_symbols: int | None = None,
    development_end: str | None = None,
    notebook: str = "causal_dml",
    case_dir=None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
    supersedes_hash: str | None = None,
) -> str:
    """Register a causal DML run in the dedicated `causal_runs` table.

    Causal DML estimates a treatment effect rather than a cross-sectional
    score, so it lives in its own table, distinct from `training_runs`,
    `prediction_sets`, and `prediction_metrics` which serve predictive
    families. The `predictions` argument (per-row residuals + ATE) is
    accepted for backward compatibility but no longer persisted: it has
    no downstream readers, and re-running the case study notebook is the
    canonical way to regenerate diagnostics.
    """
    import json

    # Alias the registration helper to avoid shadowing this wrapper's own name.
    # a future refactor that hoists this import to module level would otherwise
    # turn the call below into infinite recursion.
    from case_studies.utils.registry.registration import (
        register_causal_run as _register_causal_run,
    )
    from case_studies.utils.registry.specs import (
        build_training_spec,
        canonical_json,
        training_hash_from_spec,
    )

    dml_result = results.get("dml_result", {})
    ref = results.get("refutation", {})

    causal_params = {"treatment": treatment_col, "embargo": embargo}
    if confounder_cols:
        causal_params["confounders"] = confounder_cols
    if time_col is not None:
        causal_params["time_col"] = time_col
    if block_size is not None:
        causal_params["block_size"] = block_size
    if n_placebo is not None:
        causal_params["n_placebo"] = n_placebo
    if seed is not None:
        causal_params["seed"] = seed
    if horizon is not None:
        causal_params["horizon"] = horizon
    if max_samples is not None:
        causal_params["max_samples"] = max_samples
    if max_symbols is not None:
        causal_params["max_symbols"] = max_symbols
    if development_end is not None:
        causal_params["development_end"] = development_end

    spec = build_training_spec(
        "causal_dml",
        "dml",
        label,
        n_folds=n_folds,
        causal_params=causal_params,
    )
    causal_hash = training_hash_from_spec(spec)

    # Preserve NULLs for unknown p-values rather than silently coercing them
    # to 1.0. A HAC p-value that underflows to exactly 0.0 is a strongly
    # significant result, and ``or 1.0`` would flip its meaning.
    p_value_hac = results.get("p_value_hac")
    refutation_p = ref.get("empirical_p")
    refutation_n = ref.get("n_successful")

    _register_causal_run(
        case_study_id,
        causal_hash,
        label=label,
        treatment=treatment_col,
        confounders_json=json.dumps(confounder_cols or []),
        embargo=embargo,
        n_folds=n_folds,
        n_obs=int(dml_result.get("n_obs", 0)),
        dml_effect=float(dml_result.get("theta", 0.0)),
        dml_se_hac=float(dml_result.get("se_hac", 0.0)),
        p_value_hac=float(p_value_hac) if p_value_hac is not None else None,
        naive_effect=float(results.get("naive_effect", 0.0)),
        confounding_bias_pct=float(results.get("confounding_bias_pct", 0.0)),
        refutation_p=float(refutation_p) if refutation_p is not None else None,
        refutation_n_successful=int(refutation_n) if refutation_n is not None else None,
        spec_json=canonical_json(spec),
        notebook=notebook,
        started_at=started_at or results.get("started_at"),
        elapsed_s=elapsed_s if elapsed_s is not None else results.get("elapsed_s"),
        # The causal identity this run retires. A refit produces a second canonical
        # identity for the same label and `CausalResult.one` resolves a label to exactly
        # one, so the chain is declared here rather than guessed from `created_at`.
        # Notebooks pass their SUPERSEDES_CAUSAL parameter through.
        supersedes_hash=supersedes_hash,
        case_dir=case_dir,
    )

    p_hac_display = f"{float(p_value_hac):.4f}" if p_value_hac is not None else "n/a"
    print(f"  -> registered causal_dml (causal_hash={causal_hash}, p_hac={p_hac_display})")
    return causal_hash
