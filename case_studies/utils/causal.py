"""Shared causal inference utilities for Ch15 notebooks and case study DML.

Provides:
- block_permute(): Block permutation preserving autocorrelation
- manual_dml_timeseries(): Walk-forward DML with embargo
- run_dml_analysis(): Full DML pipeline (naive + DML + refutation)

Used by teaching notebooks (02-04, 07) and case study 09_causal_dml.py.
"""

from __future__ import annotations

import os

# Pin OpenMP threading before sklearn imports — HistGradientBoostingRegressor
# uses OMP-parallel histogram construction whose floating-point reduction order
# is non-deterministic across threads. With OMP_NUM_THREADS=1 the placebo loop
# is bit-reproducible across runs at the same seed/spec/data.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.regression.linear_model import OLS

from utils.modeling import RANDOM_SEED, seed_everything


def embargo_from_buffer(label_buffer: str) -> int:
    """Convert a label buffer string to an integer embargo period count.

    Supports all pandas duration units (D, H/h, M, T/min).
    Returns the number of observation periods to skip between train and test.

    Bar-frequency assumptions baked into the conversion:
    - D: one period per `value` days (e.g. "5D" → 5)
    - H/h: the buffer is interpreted as the bar cadence; the result is the
      number of `value`-hour bars in one trading day (24 // value), so "8H"
      → 3 bars (a one-day embargo on 8-hour bars)
    - M: `value` months × 21 trading days
    - T/min: the buffer is the cadence; the result is the bars in 15 minutes
      (`value` // 15), assuming 15-minute base bars
    """
    import re

    match = re.match(r"(\d+)(D|H|h|M|T|min)", label_buffer.strip())
    if not match:
        raise ValueError(f"Cannot parse label_buffer: {label_buffer}")
    value, unit = int(match.group(1)), match.group(2)
    return {
        "D": value,
        "H": max(1, 24 // value),
        "h": max(1, 24 // value),
        "M": value * 21,
        "T": max(1, value // 15),
        "min": max(1, value // 15),
    }[unit]


def block_permute(
    arr: np.ndarray, block_size: int, rng: np.random.Generator | None = None
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

    Returns
    -------
    np.ndarray
        Block-permuted array.
    """
    arr = np.asarray(arr)
    n = len(arr)
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = n // block_size
    if n_blocks < 2:
        return rng.permutation(arr)

    block_indices = rng.permutation(n_blocks)

    result = []
    for idx in block_indices:
        start = idx * block_size
        result.append(arr[start : start + block_size])

    # Handle remainder
    remainder_start = n_blocks * block_size
    if remainder_start < n:
        result.append(arr[remainder_start:])

    return np.concatenate(result)


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

    Returns
    -------
    dict
        Keys: theta, se_iid, se_hac, t_stat_iid, t_stat_hac, p_value_hac,
        n_obs, hac_maxlags. If return_residuals: also Y_res, T_res.
    """
    seed_everything(RANDOM_SEED)

    n = len(Y)

    # Initialize residual arrays
    Y_res = np.full(n, np.nan)
    T_res = np.full(n, np.nan)

    fold_size = n // (n_folds + 1)

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

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

    empty = {
        "theta": np.nan,
        "se_iid": np.nan,
        "se_hac": np.nan,
        "t_stat_iid": np.nan,
        "t_stat_hac": np.nan,
        "p_value_hac": np.nan,
        "n_obs": n_valid,
        "hac_maxlags": 0,
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
        auto = max(1, int(n_valid ** (1 / 3)))
        # Overlapping h-period labels need L >= h-1; the cube-root rule is
        # horizon-blind and under-lags long-horizon overlapping returns.
        hac_maxlags = max(horizon - 1, auto) if horizon else auto
        hac_maxlags = min(hac_maxlags, max(1, n_valid // 2))

    T_const = sm.add_constant(T_v)
    ols_iid = OLS(Y_v, T_const).fit()
    theta = ols_iid.params[1]

    # HC0 standard error
    se_iid = np.sqrt(ols_iid.cov_HC0[1, 1])

    # HAC (Newey-West) standard error with frequency-adaptive bandwidth
    se_hac = se_iid
    try:
        ols_hac = OLS(Y_v, T_const).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
        cov = ols_hac.cov_params()
        se_hac = np.sqrt(cov.iloc[1, 1] if hasattr(cov, "iloc") else cov[1, 1])
    except Exception:
        pass  # Fall back to HC0 standard errors on numerical failure

    t_stat_hac = theta / se_hac if se_hac > 0 else np.nan
    p_value_hac = (
        2 * (1 - stats.t.cdf(abs(t_stat_hac), df=n_valid - 2))
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
        "hac_maxlags": hac_maxlags,
    }

    if return_residuals:
        result["Y_res"] = Y_res
        result["T_res"] = T_res

    return result


REFUTATION_ALPHA = 0.05


def classify_refutation(empirical_p: float) -> str:
    """Binary pass/fail of the block-permutation refutation test at 5 %.

    Returns "Passes" if the empirical placebo p-value is below 5 %
    (the observed effect cannot be reproduced by permutation in
    most placebo runs); "Fails" otherwise. Always read the raw
    `empirical_p` alongside the label.
    """
    return "Passes" if empirical_p < REFUTATION_ALPHA else "Fails"


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
        Pass it for overlapping labels (horizon >= ~10) — e.g.
        `horizon=embargo_from_buffer(mds.label_buffer)`. When both `horizon`
        and `hac_maxlags` are None, the bandwidth falls back to the
        horizon-blind cube-root rule and a warning is emitted.

    Returns
    -------
    dict
        Comprehensive results with keys: naive_effect, dml_result,
        confounding_bias, confounding_bias_pct, refutation (z_score,
        empirical_p, placebo_mean, placebo_std, placebo_effects,
        refutation_class), p_value_hac, hac_maxlags, and n_obs.
    """
    # Input validation
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
            "t-statistic. Pass horizon=embargo_from_buffer(mds.label_buffer).",
            stacklevel=2,
        )

    _dml_started_at = datetime.now(UTC).isoformat()
    _dml_t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    T = df[treatment_col].values
    Y = df[outcome_col].values
    X = df[confounder_cols].values

    # Naive OLS
    T_const = np.column_stack([np.ones(len(T)), T])
    naive_coef = np.linalg.lstsq(T_const, Y, rcond=None)[0]
    naive_effect = naive_coef[1]

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
    )

    # Confounding bias
    dml_effect = dml["theta"]
    bias = naive_effect - dml_effect
    bias_pct = bias / abs(dml_effect) * 100 if dml_effect != 0 else 0.0

    # Block permutation refutation
    placebo_effects = []
    for _ in range(n_placebo):
        T_perm = block_permute(T, block_size, rng=rng)
        perm_result = manual_dml_timeseries(
            Y,
            T_perm,
            X,
            n_folds=min(3, n_folds),
            embargo=embargo,
        )
        if not np.isnan(perm_result["theta"]):
            placebo_effects.append(perm_result["theta"])

    refutation = {}
    if len(placebo_effects) >= 10:
        placebo_arr = np.array(placebo_effects)
        p_mean = np.mean(placebo_arr)
        p_std = np.std(placebo_arr)
        z = (dml_effect - p_mean) / p_std if p_std > 0 else np.inf
        emp_p = np.mean(np.abs(placebo_arr) >= np.abs(dml_effect))
        ref_class = classify_refutation(emp_p)
        refutation = {
            "z_score": z,
            "empirical_p": emp_p,
            "placebo_mean": p_mean,
            "placebo_std": p_std,
            "n_successful": len(placebo_effects),
            "placebo_effects": placebo_effects,
            "refutation_class": ref_class,
        }

    return {
        "naive_effect": naive_effect,
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
        f"Observations: {results['n_obs']:,}",
        f"HAC bandwidth: {hac_lags} lags (max of horizon-1 and cube-root)",
        "",
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
    notebook: str = "causal_dml",
    case_dir=None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
) -> str:
    """Register a causal DML run in the dedicated `causal_runs` table.

    Causal DML estimates a treatment effect rather than a cross-sectional
    score, so it lives in its own table — distinct from `training_runs`,
    `prediction_sets`, and `prediction_metrics` which serve predictive
    families. The `predictions` argument (per-row residuals + ATE) is
    accepted for backward compatibility but no longer persisted: it has
    no downstream readers, and re-running the case study notebook is the
    canonical way to regenerate diagnostics.
    """
    import json

    # Alias the registration helper to avoid shadowing this wrapper's own name —
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

    spec = build_training_spec(
        "causal_dml",
        "dml",
        label,
        n_folds=n_folds,
        causal_params=causal_params,
    )
    causal_hash = training_hash_from_spec(spec)

    # Preserve NULLs for unknown p-values rather than silently coercing them
    # to 1.0 — a HAC p-value that underflows to exactly 0.0 is a strongly
    # significant result, and ``or 1.0`` would flip its meaning.
    p_value_hac = results.get("p_value_hac")
    refutation_p = ref.get("empirical_p")

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
        spec_json=canonical_json(spec),
        notebook=notebook,
        started_at=started_at or results.get("started_at"),
        elapsed_s=elapsed_s if elapsed_s is not None else results.get("elapsed_s"),
    )

    p_hac_display = f"{float(p_value_hac):.4f}" if p_value_hac is not None else "n/a"
    print(f"  -> registered causal_dml (causal_hash={causal_hash}, p_hac={p_hac_display})")
    return causal_hash
