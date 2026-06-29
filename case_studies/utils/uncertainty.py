"""Backtest performance uncertainty: block bootstrap, HAC SE, PSR/DSR, paired comparisons.

Operates on the daily out-of-sample strategy return series persisted at
``run_log/backtest/{hash}/daily_returns.parquet``. Series-level CIs are computed
via stationary block bootstrap; Sharpe SE uses the López de Prado (2025) closed
form with autocorrelation, skew, and kurtosis corrections; mean-return SE uses
Newey-West HAC. Selection bias across K variants uses the library's
:func:`deflated_sharpe_ratio`. Challenger-vs-baseline uncertainty uses paired
stationary block bootstrap on daily-return differences.

All bootstraps share a single seed so the same call is reproducible. Default
``n_boot=2000``; tune down for sweeps if needed.

Block-length policy
-------------------
``resolve_block_length(case_study, label, returns)`` picks the block length:

1. ``setup.yaml.labels.{label}.rebalance_step`` if present (canonical).
2. Falls back to :func:`ml4t.diagnostic.evaluation.stats._optimal_block_size`.
3. Floored at the label's forward-return horizon (``ret_5d`` → ≥5,
   ``ret_to_expiry`` → keeps the floor at 1, since horizon is variable).

Baseline registry
-----------------
:data:`STAGE_BASELINE` declares the natural baseline for each backtest stage,
used by the Ch20 paired-bootstrap synthesis:

- ``signal``  → equal-weight benchmark (per case study, registered separately)
- ``allocation``    → ``signal`` leader of the same (label, family)
- ``cost_sensitivity`` → ``signal`` leader (no costs)
- ``risk_overlay``  → ``cost_sensitivity`` leader (with costs, no risk overlay)

Per-case-study baselines for the signal stage live in
:data:`SIGNAL_BASELINE_BY_CASE_STUDY`; populate this when the equal-weight
benchmark name in the registry is non-default.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def periods_per_year_from_setup(case_study: str) -> int:
    """Resolve periods-per-year from the case study's setup.yaml.

    Reads ``evaluation.periods_per_year`` from
    ``case_studies/{case_study}/config/setup.yaml``. This is the
    annualization convention of the **daily_returns** grid that the
    backtester actually writes (NYSE-like 5d/wk → 252, 7d/wk crypto →
    365, genuinely monthly us_firm → 12). It is NOT the trade cadence
    or rebalance frequency.

    Raises FileNotFoundError if the case study has no setup.yaml, or
    KeyError if `evaluation.periods_per_year` is not declared — both
    are programming errors that callers should fix upstream rather
    than silently absorb.
    """
    import yaml

    from utils.paths import get_case_study_dir

    setup_path = get_case_study_dir(case_study) / "config" / "setup.yaml"
    with setup_path.open() as f:
        setup = yaml.safe_load(f)
    evaluation = setup.get("evaluation", {}) if isinstance(setup, dict) else {}
    if "periods_per_year" not in evaluation:
        raise KeyError(
            f"{setup_path} is missing evaluation.periods_per_year; add it "
            "(252 for daily 5d/wk markets, 365 for 7d/wk crypto, 12 for "
            "genuinely monthly us_firm)."
        )
    return int(evaluation["periods_per_year"])


# ---------------------------------------------------------------------------
# Block-length resolver
# ---------------------------------------------------------------------------


_LABEL_HORIZON_RE = re.compile(r"(\d+)\s*d\b")


def _label_horizon_floor(label: str | None) -> int:
    """Best-effort horizon (days) implied by a label name; 1 if unknown."""
    if not label:
        return 1
    match = _LABEL_HORIZON_RE.search(label)
    if match:
        return max(int(match.group(1)), 1)
    return 1


def resolve_block_length(
    case_study: str | None,
    label: str | None,
    returns: np.ndarray,
    *,
    explicit: int | None = None,
) -> int:
    """Resolve block length: rebalance_step → optimal → floor at label horizon."""
    if explicit is not None and explicit > 0:
        return int(explicit)

    rebalance_step: int | None = None
    if case_study and label:
        try:
            from case_studies.utils.backtest_loaders import get_rebalance_step

            rebalance_step = int(get_rebalance_step(case_study, label))
        except Exception:
            rebalance_step = None

    floor = _label_horizon_floor(label)

    if rebalance_step and rebalance_step > 0:
        return max(rebalance_step, floor)

    from ml4t.diagnostic.evaluation.stats import _optimal_block_size

    optimal = int(round(float(_optimal_block_size(returns))))
    return max(optimal, floor, 1)


# ---------------------------------------------------------------------------
# Baseline registry (Ch20 paired-bootstrap synthesis)
# ---------------------------------------------------------------------------


STAGE_BASELINE: dict[str, str] = {
    "signal": "equal_weight",
    "allocation": "signal_leader",
    "cost_sensitivity": "signal_leader",
    "risk_overlay": "cost_sensitivity_leader",
}


SIGNAL_BASELINE_BY_CASE_STUDY: dict[str, str] = {
    "etfs": "equal_weight",
    "nasdaq100_microstructure": "equal_weight",
    "sp500_equity_option_analytics": "equal_weight",
    "sp500_options": "equal_weight",
    "us_firm_characteristics": "equal_weight",
    "us_equities_panel": "equal_weight",
    "fx_pairs": "equal_weight",
    "crypto_perps_funding": "equal_weight",
    "cme_futures": "equal_weight",
}


# ---------------------------------------------------------------------------
# Series-level uncertainty
# ---------------------------------------------------------------------------


@dataclass
class _Stats:
    sharpe: float
    sortino: float
    ann_return: float
    volatility: float
    max_drawdown: float
    calmar: float


def _sample_stats(returns: np.ndarray, periods_per_year: int) -> _Stats:
    """Point-estimate statistics on a return series."""
    if len(returns) < 2:
        return _Stats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    sharpe = (mu / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    downside = returns[returns < 0]
    if len(downside) > 1:
        dsd = float(np.sqrt(np.mean(downside**2)))
        sortino = (mu / dsd * np.sqrt(periods_per_year)) if dsd > 0 else 0.0
    else:
        sortino = 0.0
    cum = np.cumprod(1.0 + returns)
    total_return = float(cum[-1] - 1.0)
    n_years = len(returns) / periods_per_year
    # Guard the negative-cumulative-return case: if the strategy lost more
    # than 100 % (cumulative growth ≤ 0), the geometric mean would be
    # complex. Report `ann_return = -1` (total wipeout) instead.
    base = 1.0 + total_return
    if n_years <= 0:
        ann_return = 0.0
    elif base <= 0.0:
        ann_return = -1.0
    else:
        ann_return = float(base ** (1.0 / n_years) - 1.0)
    vol = sd * np.sqrt(periods_per_year)
    running_max = np.maximum.accumulate(cum)
    # Avoid div-by-zero / negative running_max (post-bankruptcy paths).
    safe_max = np.where(running_max > 0, running_max, np.nan)
    dd = (cum - running_max) / safe_max
    max_dd_raw = float(np.nanmin(dd)) if np.any(np.isfinite(dd)) else 0.0
    max_dd = max_dd_raw if np.isfinite(max_dd_raw) else 0.0
    calmar = (ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    return _Stats(sharpe, sortino, ann_return, vol, max_dd, calmar)


def _newey_west_mean_se(returns: np.ndarray, lag: int) -> float:
    """Newey-West HAC standard error of the sample mean."""
    n = len(returns)
    if n < 3:
        return float("nan")
    r = returns - np.mean(returns)
    gamma0 = float(np.dot(r, r) / n)
    s = gamma0
    for h in range(1, min(lag, n - 1) + 1):
        gamma_h = float(np.dot(r[:-h], r[h:]) / n)
        w = 1.0 - h / (lag + 1.0)
        s += 2.0 * w * gamma_h
    s = max(s, 0.0)
    return float(np.sqrt(s / n))


def _sharpe_se_lo(returns: np.ndarray, periods_per_year: int) -> float:
    """LdP-2025 Sharpe SE with autocorrelation, skewness, kurtosis."""
    from ml4t.diagnostic.evaluation.stats import compute_sharpe_variance

    n = len(returns)
    if n < 4:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return float("nan")
    sr = mu / sd  # native frequency
    centered = returns - mu
    m2 = float(np.mean(centered**2))
    if m2 == 0:
        return float("nan")
    skew = float(np.mean(centered**3) / m2**1.5)
    kurt = float(np.mean(centered**4) / m2**2)  # Pearson convention (normal=3)
    rho = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    if not np.isfinite(rho) or abs(rho) >= 0.999:
        rho = 0.0
    var = compute_sharpe_variance(
        sharpe=sr,
        n_samples=n,
        skewness=skew,
        kurtosis=kurt,
        autocorrelation=rho,
        n_trials=1,
    )
    if var <= 0 or not np.isfinite(var):
        return float("nan")
    se_native = float(np.sqrt(var))
    return se_native * np.sqrt(periods_per_year)


def _stationary_bootstrap_metrics(
    returns: np.ndarray,
    *,
    periods_per_year: int,
    block_length: int,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run a stationary bootstrap and return arrays of resampled metrics."""
    from ml4t.diagnostic.evaluation.stats import _stationary_bootstrap_indices

    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_boot)
    sortinos = np.empty(n_boot)
    ann_rets = np.empty(n_boot)
    vols = np.empty(n_boot)
    max_dds = np.empty(n_boot)
    calmars = np.empty(n_boot)

    np_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        for i in range(n_boot):
            idx = _stationary_bootstrap_indices(len(returns), float(block_length))
            sample = returns[idx]
            stats = _sample_stats(sample, periods_per_year)
            sharpes[i] = stats.sharpe
            sortinos[i] = stats.sortino
            ann_rets[i] = stats.ann_return
            vols[i] = stats.volatility
            max_dds[i] = stats.max_drawdown
            calmars[i] = stats.calmar
    finally:
        np.random.set_state(np_state)

    return {
        "sharpe": sharpes,
        "sortino": sortinos,
        "ann_return": ann_rets,
        "volatility": vols,
        "max_drawdown": max_dds,
        "calmar": calmars,
    }


def _percentile_ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(arr, 100 * alpha / 2)),
        float(np.percentile(arr, 100 * (1 - alpha / 2))),
    )


def compute_backtest_uncertainty(
    daily_returns: np.ndarray | pl.Series | pl.DataFrame,
    *,
    periods_per_year: int = 252,
    block_length: int | None = None,
    case_study: str | None = None,
    label: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute series-level uncertainty for one backtest.

    Returns a flat dict suitable for upsert into ``backtest_metrics``:

    - ``sharpe_se_lo``                — Lo / LdP-2025 closed-form SE
    - ``sharpe_ci95_lo`` / ``_hi``    — block-bootstrap percentile CI
    - ``sortino_ci95_lo`` / ``_hi``
    - ``ann_return_hac_se``           — Newey-West HAC SE of mean return (annualized)
    - ``ann_return_ci95_lo`` / ``_hi``
    - ``max_dd_ci95_lo`` / ``_hi``
    - ``calmar_ci95_lo`` / ``_hi``
    - ``psr_pvalue``                  — 1 − P(true SR > 0) under PSR
    - ``bootstrap_block_length``
    - ``bootstrap_n``
    """
    arr = _coerce_returns(daily_returns)
    if arr.size < 4:
        return {}

    block = resolve_block_length(case_study, label, arr, explicit=block_length)
    boot = _stationary_bootstrap_metrics(
        arr,
        periods_per_year=periods_per_year,
        block_length=block,
        n_boot=n_boot,
        seed=seed,
    )

    point = _sample_stats(arr, periods_per_year)
    sharpe_se = _sharpe_se_lo(arr, periods_per_year)

    # NW lag: at least the bootstrap block (≈ rebalance step)
    nw_lag = max(block - 1, int(np.floor(4 * (len(arr) / 100) ** (2 / 9))))
    mean_se_native = _newey_west_mean_se(arr, lag=nw_lag)
    ann_return_hac_se = (
        mean_se_native * periods_per_year if np.isfinite(mean_se_native) else float("nan")
    )

    # PSR
    psr_pvalue = float("nan")
    try:
        from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

        psr = deflated_sharpe_ratio(arr, periods_per_year=periods_per_year)
        psr_pvalue = float(psr.p_value)
    except Exception:
        psr_pvalue = float("nan")

    sh_lo, sh_hi = _percentile_ci(boot["sharpe"])
    so_lo, so_hi = _percentile_ci(boot["sortino"])
    ar_lo, ar_hi = _percentile_ci(boot["ann_return"])
    md_lo, md_hi = _percentile_ci(boot["max_drawdown"])
    cl_lo, cl_hi = _percentile_ci(boot["calmar"])

    return {
        "sharpe_se_lo": _to_float(sharpe_se),
        "sharpe_ci95_lo": _to_float(sh_lo),
        "sharpe_ci95_hi": _to_float(sh_hi),
        "sortino_ci95_lo": _to_float(so_lo),
        "sortino_ci95_hi": _to_float(so_hi),
        "ann_return_hac_se": _to_float(ann_return_hac_se),
        "ann_return_ci95_lo": _to_float(ar_lo),
        "ann_return_ci95_hi": _to_float(ar_hi),
        "max_dd_ci95_lo": _to_float(md_lo),
        "max_dd_ci95_hi": _to_float(md_hi),
        "calmar_ci95_lo": _to_float(cl_lo),
        "calmar_ci95_hi": _to_float(cl_hi),
        "psr_pvalue": _to_float(psr_pvalue),
        "bootstrap_block_length": float(block),
        "bootstrap_n": float(n_boot),
    }


# ---------------------------------------------------------------------------
# Paired uncertainty: challenger vs baseline
# ---------------------------------------------------------------------------


def compute_paired_uncertainty(
    challenger: np.ndarray | pl.Series,
    baseline: np.ndarray | pl.Series,
    *,
    periods_per_year: int = 252,
    block_length: int | None = None,
    case_study: str | None = None,
    label: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired stationary bootstrap on daily-return differences.

    Inputs must be the same length and aligned by date. Returns a flat dict for
    upsert into ``backtest_paired_metrics``.
    """
    from ml4t.diagnostic.evaluation.stats import _stationary_bootstrap_indices

    c = _coerce_returns(challenger)
    b = _coerce_returns(baseline)
    # Caller's contract: pre-aligned by timestamp via inner-join. If the
    # per-side leading-zero strip leaves the two arrays at different
    # lengths, head/tail-truncation would misalign them (challenger
    # position i and baseline position i would correspond to different
    # original timestamps). Refuse rather than bootstrap a misaligned
    # pair silently — callers must pre-align if they bypass _joint_coerce.
    if c.size != b.size:
        return {}
    if c.size < 4:
        return {}

    diff = c - b
    block = resolve_block_length(case_study, label, diff, explicit=block_length)

    point_c = _sample_stats(c, periods_per_year)
    point_b = _sample_stats(b, periods_per_year)
    sharpe_diff = point_c.sharpe - point_b.sharpe
    ret_diff = point_c.ann_return - point_b.ann_return
    max_dd_diff = point_c.max_drawdown - point_b.max_drawdown

    # Information ratio on the diff series; require sd above a real-world floor
    # (1bp/day) — degenerate near-constant differences are not informative
    sd_diff = float(np.std(diff, ddof=1))
    info_ratio = (
        float(np.mean(diff) / sd_diff * np.sqrt(periods_per_year))
        if sd_diff > 1e-6
        else float("nan")
    )

    # Paired bootstrap: same indices applied to both series
    rng = np.random.default_rng(seed)
    sharpe_diffs = np.empty(n_boot)
    ret_diffs = np.empty(n_boot)
    max_dd_diffs = np.empty(n_boot)
    irs = np.empty(n_boot)
    wins = 0

    np_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        for i in range(n_boot):
            idx = _stationary_bootstrap_indices(c.size, float(block))
            cs = _sample_stats(c[idx], periods_per_year)
            bs = _sample_stats(b[idx], periods_per_year)
            sharpe_diffs[i] = cs.sharpe - bs.sharpe
            ret_diffs[i] = cs.ann_return - bs.ann_return
            max_dd_diffs[i] = cs.max_drawdown - bs.max_drawdown
            d = c[idx] - b[idx]
            sd = float(np.std(d, ddof=1))
            irs[i] = (
                float(np.mean(d) / sd * np.sqrt(periods_per_year)) if sd > 1e-6 else float("nan")
            )
            if cs.sharpe > bs.sharpe:
                wins += 1
    finally:
        np.random.set_state(np_state)

    sd_lo, sd_hi = _percentile_ci(sharpe_diffs)
    rd_lo, rd_hi = _percentile_ci(ret_diffs)
    mdd_lo, mdd_hi = _percentile_ci(max_dd_diffs)
    ir_lo, ir_hi = _percentile_ci(irs)

    # Two-sided bootstrap p-value for sharpe_diff != 0 (centered around the bootstrap mean)
    centered = sharpe_diffs - np.mean(sharpe_diffs)
    p_value = float(np.mean(np.abs(centered) >= abs(sharpe_diff)))

    return {
        "sharpe_diff": _to_float(sharpe_diff),
        "sharpe_diff_ci95_lo": _to_float(sd_lo),
        "sharpe_diff_ci95_hi": _to_float(sd_hi),
        "ret_diff": _to_float(ret_diff),
        "ret_diff_ci95_lo": _to_float(rd_lo),
        "ret_diff_ci95_hi": _to_float(rd_hi),
        "max_dd_diff": _to_float(max_dd_diff),
        "max_dd_diff_ci95_lo": _to_float(mdd_lo),
        "max_dd_diff_ci95_hi": _to_float(mdd_hi),
        "info_ratio": _to_float(info_ratio),
        "info_ratio_ci95_lo": _to_float(ir_lo),
        "info_ratio_ci95_hi": _to_float(ir_hi),
        "prob_challenger_wins": float(wins) / float(n_boot),
        "p_value": p_value,
        "bootstrap_block_length": float(block),
        "bootstrap_n": float(n_boot),
    }


def compute_independent_diff_uncertainty(
    challenger: np.ndarray | pl.Series,
    baseline: np.ndarray | pl.Series,
    *,
    periods_per_year: int = 252,
    block_length: int | None = None,
    case_study: str | None = None,
    label: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Independent-bootstrap difference CI for two disjoint return series.

    Use when challenger and baseline come from non-overlapping windows
    (e.g. holdout vs validation of the same lineage). Bootstraps each
    series over its full window separately, then forms the difference
    distribution from independent draws.

    Returns the same dict shape as :func:`compute_paired_uncertainty`
    so registry callers are interchangeable. ``info_ratio`` columns are
    NaN — there is no diff *series* to ratio when the windows are
    disjoint. Block length is resolved once from ``(case_study, label)``
    and applied to both bootstraps.
    """
    from ml4t.diagnostic.evaluation.stats import _stationary_bootstrap_indices

    c = _coerce_returns(challenger)
    b = _coerce_returns(baseline)
    if c.size < 4 or b.size < 4:
        return {}

    # Resolve block length per side: disjoint windows can have different
    # autocorrelation structure (different volatility regimes / sample
    # sizes), so a block tuned to one side would under- or over-state
    # bootstrap variance on the other.
    block_c = resolve_block_length(case_study, label, c, explicit=block_length)
    block_b = resolve_block_length(case_study, label, b, explicit=block_length)

    point_c = _sample_stats(c, periods_per_year)
    point_b = _sample_stats(b, periods_per_year)
    sharpe_diff = point_c.sharpe - point_b.sharpe
    ret_diff = point_c.ann_return - point_b.ann_return
    max_dd_diff = point_c.max_drawdown - point_b.max_drawdown

    rng = np.random.default_rng(seed)
    sharpes_c = np.empty(n_boot)
    sharpes_b = np.empty(n_boot)
    rets_c = np.empty(n_boot)
    rets_b = np.empty(n_boot)
    mdds_c = np.empty(n_boot)
    mdds_b = np.empty(n_boot)

    np_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        for i in range(n_boot):
            idx_c = _stationary_bootstrap_indices(c.size, float(block_c))
            idx_b = _stationary_bootstrap_indices(b.size, float(block_b))
            cs = _sample_stats(c[idx_c], periods_per_year)
            bs = _sample_stats(b[idx_b], periods_per_year)
            sharpes_c[i] = cs.sharpe
            sharpes_b[i] = bs.sharpe
            rets_c[i] = cs.ann_return
            rets_b[i] = bs.ann_return
            mdds_c[i] = cs.max_drawdown
            mdds_b[i] = bs.max_drawdown
    finally:
        np.random.set_state(np_state)

    sharpe_diffs = sharpes_c - sharpes_b
    ret_diffs = rets_c - rets_b
    max_dd_diffs = mdds_c - mdds_b
    wins = float(np.sum(sharpes_c > sharpes_b))

    sd_lo, sd_hi = _percentile_ci(sharpe_diffs)
    rd_lo, rd_hi = _percentile_ci(ret_diffs)
    mdd_lo, mdd_hi = _percentile_ci(max_dd_diffs)

    centered = sharpe_diffs - np.mean(sharpe_diffs)
    p_value = float(np.mean(np.abs(centered) >= abs(sharpe_diff)))

    return {
        "sharpe_diff": _to_float(sharpe_diff),
        "sharpe_diff_ci95_lo": _to_float(sd_lo),
        "sharpe_diff_ci95_hi": _to_float(sd_hi),
        "ret_diff": _to_float(ret_diff),
        "ret_diff_ci95_lo": _to_float(rd_lo),
        "ret_diff_ci95_hi": _to_float(rd_hi),
        "max_dd_diff": _to_float(max_dd_diff),
        "max_dd_diff_ci95_lo": _to_float(mdd_lo),
        "max_dd_diff_ci95_hi": _to_float(mdd_hi),
        "info_ratio": float("nan"),
        "info_ratio_ci95_lo": float("nan"),
        "info_ratio_ci95_hi": float("nan"),
        "prob_challenger_wins": wins / float(n_boot),
        "p_value": p_value,
        # Schema column is single-valued; report the larger block so the
        # value is conservative w.r.t. autocorrelation. n_c / n_b expose
        # the actual per-side post-coerce sample sizes for callers that
        # need an accurate "n_overlap"-equivalent on the disjoint path.
        "bootstrap_block_length": float(max(block_c, block_b)),
        "bootstrap_block_length_c": float(block_c),
        "bootstrap_block_length_b": float(block_b),
        "bootstrap_n": float(n_boot),
        "n_c": float(c.size),
        "n_b": float(b.size),
    }


# ---------------------------------------------------------------------------
# Selection adjustment across K variants (DSR + reality check + PBO + MinTRL)
# ---------------------------------------------------------------------------


def compute_selection_adjustment(
    returns_by_variant: dict[str, np.ndarray | pl.Series],
    *,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Selection-bias adjustment across K candidate strategies — **raw-K only**.

    .. deprecated::
        Returns ``dsr / dsr_pvalue / expected_max_sharpe / min_trl_periods``
        using raw trial counts (no Marchenko-Pastur or effective-rank
        correlation correction), which overcounts trials when variants are
        correlated. The recommended replacement is
        :func:`compute_cohort_metrics`, which surfaces raw / MP / ER DSR
        alongside RAS and is persisted to the ``cohort_metrics`` table.
        Consumers should read ``cohort_metrics`` (e.g. via
        :func:`case_studies.utils.notebook_render.selection_adjusted_leader_table`
        or :meth:`BacktestExplorer.deflated_sharpe`) rather than
        recomputing from this helper.

    Combines:

    - **DSR** for the best-of-K leader (Sharpe haircut for selection bias)
    - **Expected max Sharpe** under the null
    - **MinTRL** — periods needed for leader to reach significance at α=0.05
    - **White's reality check** — bootstrap p-value against the best benchmark
    - **PBO** — Probability of Backtest Overfitting across folds (caller must
      pass per-fold returns by variant via ``returns_by_variant`` keyed
      ``"{variant}__fold{i}"``); skipped if no fold-level keys are present.
    """
    from ml4t.diagnostic.evaluation.stats import (
        compute_min_trl,
        deflated_sharpe_ratio,
    )

    # Series prep
    arrays = {}
    for name, ret in returns_by_variant.items():
        a = _coerce_returns(ret)
        if a.size >= 4 and float(np.std(a, ddof=1)) > 1e-10:
            arrays[name] = a
    if not arrays:
        return {}

    names = list(arrays.keys())
    arr_list = [arrays[n] for n in names]
    sharpes = {n: _sample_stats(arrays[n], periods_per_year).sharpe for n in names}
    leader = max(sharpes, key=sharpes.get)

    out: dict[str, Any] = {
        "leader": leader,
        "leader_sharpe": float(sharpes[leader]),
        "k_variants": float(len(arr_list)),
    }

    # DSR
    try:
        dsr = deflated_sharpe_ratio(arr_list, periods_per_year=periods_per_year)
        out["dsr"] = float(dsr.deflated_sharpe)
        out["dsr_pvalue"] = float(dsr.p_value)
        out["expected_max_sharpe"] = float(dsr.expected_max_sharpe)
        out["min_trl_periods"] = float(dsr.min_trl)
        out["dsr_significant"] = bool(dsr.is_significant)
    except Exception:
        pass

    # MinTRL standalone (for the leader, against SR=0 benchmark)
    try:
        leader_arr = arrays[leader]
        mtrl = compute_min_trl(
            leader_arr,
            periods_per_year=periods_per_year,
        )
        out["leader_min_trl"] = float(mtrl.min_trl)
    except Exception:
        pass

    return out


def compute_reality_check(
    challenger_returns: dict[str, np.ndarray | pl.Series],
    benchmark_returns: np.ndarray | pl.Series,
    *,
    block_size: int | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """White's reality check: do any of K challengers beat the benchmark?

    Returns ``{p_value, test_statistic, best_strategy, k_strategies}``.
    """
    from ml4t.diagnostic.evaluation.stats import whites_reality_check

    bench = _coerce_returns(benchmark_returns)
    names = list(challenger_returns.keys())
    arrs: list[np.ndarray] = []
    keep_names: list[str] = []
    for n in names:
        a = _coerce_returns(challenger_returns[n])
        if a.size == bench.size and float(np.std(a, ddof=1)) > 1e-10:
            arrs.append(a)
            keep_names.append(n)
    if not arrs:
        return {}
    strategies = np.column_stack(arrs)
    rc = whites_reality_check(
        returns_benchmark=bench,
        returns_strategies=strategies,
        bootstrap_samples=n_bootstrap,
        block_size=block_size,
        random_state=seed,
    )
    best_idx = int(np.argmax(np.mean(strategies - bench.reshape(-1, 1), axis=0)))
    return {
        "reality_check_pvalue": float(rc.get("p_value", float("nan"))),
        "reality_check_statistic": float(rc.get("test_statistic", float("nan"))),
        "reality_check_best": keep_names[best_idx],
        "k_strategies": float(len(keep_names)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_returns(x: np.ndarray | pl.Series | pl.DataFrame) -> np.ndarray:
    if isinstance(x, pl.DataFrame):
        for col in ("daily_return", "ret", "return", "value"):
            if col in x.columns:
                arr = x[col].to_numpy()
                break
        else:
            arr = x[x.columns[-1]].to_numpy()
    elif isinstance(x, pl.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x).flatten()
    arr = arr.astype(np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    # Engine-mode parquets often carry leading zero rows from bars before the
    # first signal. Including them dilates uncertainty by underestimating
    # variance and overstating effective sample size.
    if arr.size > 0:
        nonzero = np.flatnonzero(arr != 0.0)
        if nonzero.size > 0:
            arr = arr[nonzero[0] :]
    return arr


def _to_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(f):
        return float("nan")
    return f


def load_daily_returns(case_study: str, backtest_hash: str) -> np.ndarray | None:
    """Load persisted daily returns for a backtest hash; None if missing."""
    from utils.paths import get_case_study_dir

    path = (
        get_case_study_dir(case_study)
        / "run_log"
        / "backtest"
        / backtest_hash
        / "daily_returns.parquet"
    )
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    return _coerce_returns(df)


def load_daily_returns_with_timestamp(case_study: str, backtest_hash: str) -> pl.DataFrame | None:
    """Load persisted daily returns as a (timestamp, ret) frame.

    Cohort selection statistics that pass ``correlation_method`` to
    :func:`ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio` need
    an equal-length N×K matrix across variants; this helper preserves the
    timestamp so the caller can inner-join on it before stacking.

    Returns ``None`` if the parquet is missing. Unlike :func:`load_daily_returns`,
    leading zero rows are NOT stripped here — that strip happens after
    cross-variant alignment (otherwise variants land on different windows).
    """
    from utils.paths import get_case_study_dir

    path = (
        get_case_study_dir(case_study)
        / "run_log"
        / "backtest"
        / backtest_hash
        / "daily_returns.parquet"
    )
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    ret_col = next(
        (c for c in ("daily_return", "ret", "return", "value") if c in df.columns),
        df.columns[-1],
    )
    if "timestamp" not in df.columns:
        return None
    return df.select(
        pl.col("timestamp"),
        pl.col(ret_col).cast(pl.Float64).alias("ret"),
    ).drop_nulls()


def _align_variants_on_timestamp(
    returns_by_hash: dict[str, pl.DataFrame],
) -> tuple[np.ndarray, list[str]] | None:
    """Inner-join per-hash return frames on timestamp; return (T×K matrix, hashes).

    Variants with too-few observations after alignment are dropped. Returns
    ``None`` if fewer than 2 variants survive or fewer than 4 timestamps remain.

    Daily-returns parquets across stages/case-studies write the timestamp
    column with inconsistent dtypes (``Date`` for monthly-rebalance
    aggregations, ``Datetime[ms]`` for some engine paths, ``Datetime[μs]``
    for others). The polars inner-join refuses to match across dtypes, so
    every frame is normalized to ``Datetime[μs]`` before joining. Any
    timezone is stripped — these are calendar-day rebalance stamps, not
    instants — so the join is a pure key match.
    """
    frames: dict[str, pl.DataFrame] = {}
    for name, frame in returns_by_hash.items():
        if frame is None or frame.is_empty():
            continue
        if "timestamp" not in frame.columns or "ret" not in frame.columns:
            continue
        ts_dtype = frame.schema["timestamp"]
        ts_expr = pl.col("timestamp")
        if ts_dtype == pl.Date:
            ts_expr = ts_expr.cast(pl.Datetime("us"))
        elif isinstance(ts_dtype, pl.Datetime):
            if getattr(ts_dtype, "time_zone", None) is not None:
                ts_expr = ts_expr.dt.replace_time_zone(None)
            if getattr(ts_dtype, "time_unit", "us") != "us":
                ts_expr = ts_expr.cast(pl.Datetime("us"))
        frames[name] = frame.select(
            ts_expr.alias("timestamp"),
            pl.col("ret").cast(pl.Float64).alias(name),
        )
    if len(frames) < 2:
        return None
    names = list(frames.keys())
    joined = frames[names[0]]
    for name in names[1:]:
        joined = joined.join(frames[name], on="timestamp", how="inner")
    if joined.height < 4:
        return None
    matrix = joined.select(names).to_numpy().astype(np.float64, copy=False)
    finite_rows = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite_rows]
    if matrix.shape[0] < 4:
        return None
    return matrix, names


def compute_cohort_metrics(
    returns_by_hash: dict[str, pl.DataFrame],
    *,
    periods_per_year: float,
    baseline_returns: pl.DataFrame | np.ndarray | None = None,
    fold_returns_by_hash: dict[str, np.ndarray] | None = None,
    rademacher_n_simulations: int = 2000,
    rademacher_seed: int = 0,
) -> dict[str, Any]:
    """Compute the full cohort selection-bias bundle for a set of variants.

    Returns a flat dict matching the ``cohort_metrics`` table schema (minus
    identity columns ``cohort_type / stage / label / family``, which the
    caller adds). Empty dict if alignment fails or too few variants survive.

    The ``leader_hash`` value emitted in the result is one of the dict keys
    of ``returns_by_hash`` — the contract is that those keys ARE the
    ``backtest_runs.backtest_hash`` strings used as the natural identifier
    everywhere downstream. The ``cohort_metrics`` table has a
    ``leader_hash REFERENCES backtest_runs(backtest_hash) NOT NULL`` FK,
    so passing non-hash dict keys (synthetic variant names, family labels,
    …) will fail at insert with a foreign-key violation. Callers compose
    the dict from ``load_daily_returns_with_timestamp(case_study, hash)``
    keyed on the backtest hash — do not key on family/method names.

    Estimators
    ----------
    - Raw-K DSR (no correlation adjustment) via
      :func:`ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio`.
    - MP-K DSR (``correlation_method="marchenko_pastur"``).
    - ER-K DSR (``correlation_method="effective_rank"``).
    - Rademacher Adjusted Sharpe (RAS) — lower bound on leader Sharpe.
    - White's Reality Check vs ``baseline_returns`` (optional).
    - Probability of Backtest Overfitting (CSCV) on
      ``fold_returns_by_hash`` (optional). The per-fold Sharpe matrix is
      partitioned into all C(S, S/2) IS/OOS half-fold combinations; the
      library's :func:`compute_pbo` then operates on the IS / OOS mean
      Sharpe matrices.

    Parameters
    ----------
    returns_by_hash
        Mapping ``backtest_hash → pl.DataFrame[timestamp, ret]`` (use
        :func:`load_daily_returns_with_timestamp`). Dict keys MUST be
        registered ``backtest_runs.backtest_hash`` values — see contract
        note above.
    periods_per_year
        Annualization factor (use :func:`periods_per_year_from_setup`).
    baseline_returns
        If provided, used as Reality Check benchmark. Same frame layout
        as a variant, or a numpy array of returns aligned to the variant
        intersection.
    fold_returns_by_hash
        Mapping ``backtest_hash → per-fold Sharpe ratios (1D)``. All
        variants must share fold cardinality. Skipped if not provided.
    """
    from ml4t.diagnostic.evaluation.stats import (
        compute_min_trl,
        deflated_sharpe_ratio,
        effective_number_of_trials,
        rademacher_complexity,
        ras_sharpe_adjustment,
    )

    aligned = _align_variants_on_timestamp(returns_by_hash)
    if aligned is None:
        return {}
    matrix, names = aligned
    n_periods, k_variants = matrix.shape
    if k_variants < 2:
        return {}

    sharpes = _sharpe_per_column(matrix, periods_per_year)
    if np.all(np.isnan(sharpes)):
        return {}
    leader_idx = int(np.nanargmax(sharpes))
    leader_hash = names[leader_idx]
    leader_arr = matrix[:, leader_idx]

    out: dict[str, Any] = {
        "leader_hash": leader_hash,
        "k_variants": int(k_variants),
        "periods_per_year": float(periods_per_year),
        "leader_sharpe": float(sharpes[leader_idx]),
    }

    # Per-estimator failures are caught narrowly and surfaced via warnings
    # so a regression in the library API or a degenerate input shape shows
    # up as a one-line emission rather than a silent NULL in the registry.
    _ESTIMATOR_ERRORS = (ValueError, TypeError, np.linalg.LinAlgError, ZeroDivisionError)

    # Leader Sortino + MinTRL
    try:
        out["leader_sortino"] = float(_sortino(leader_arr, periods_per_year))
    except _ESTIMATOR_ERRORS as exc:
        warnings.warn(f"leader_sortino failed for {leader_hash}: {exc}", stacklevel=2)
        out["leader_sortino"] = None
    try:
        mtrl = compute_min_trl(leader_arr, periods_per_year=periods_per_year)
        out["leader_min_trl"] = float(mtrl.min_trl)
    except _ESTIMATOR_ERRORS as exc:
        warnings.warn(f"leader_min_trl failed for {leader_hash}: {exc}", stacklevel=2)
        out["leader_min_trl"] = None

    # Effective trials — MP and ER
    try:
        et_mp = effective_number_of_trials(matrix, method="marchenko_pastur")
        out["n_trials_effective_mp"] = float(et_mp.k_eff)
    except _ESTIMATOR_ERRORS as exc:
        warnings.warn(f"n_trials_effective_mp failed for {leader_hash}: {exc}", stacklevel=2)
        out["n_trials_effective_mp"] = None
    try:
        et_er = effective_number_of_trials(matrix, method="effective_rank")
        out["n_trials_effective_er"] = float(et_er.k_eff)
    except _ESTIMATOR_ERRORS as exc:
        warnings.warn(f"n_trials_effective_er failed for {leader_hash}: {exc}", stacklevel=2)
        out["n_trials_effective_er"] = None

    # DSR — raw, MP, ER (three calls; library handles K correctly per method)
    arr_list = [matrix[:, i] for i in range(k_variants)]
    for suffix, kwargs in (
        ("raw", {}),
        ("mp", {"correlation_method": "marchenko_pastur"}),
        ("er", {"correlation_method": "effective_rank"}),
    ):
        try:
            if "correlation_method" in kwargs:
                dsr = deflated_sharpe_ratio(matrix, periods_per_year=periods_per_year, **kwargs)
            else:
                dsr = deflated_sharpe_ratio(arr_list, periods_per_year=periods_per_year)
            out[f"dsr_{suffix}"] = float(dsr.deflated_sharpe)
            out[f"dsr_{suffix}_pvalue"] = float(dsr.p_value)
            out[f"expected_max_sharpe_{suffix}"] = float(dsr.expected_max_sharpe)
            out[f"min_trl_periods_{suffix}"] = float(dsr.min_trl)
        except _ESTIMATOR_ERRORS as exc:
            warnings.warn(f"dsr_{suffix} failed for {leader_hash}: {exc}", stacklevel=2)
            out[f"dsr_{suffix}"] = None
            out[f"dsr_{suffix}_pvalue"] = None
            out[f"expected_max_sharpe_{suffix}"] = None
            out[f"min_trl_periods_{suffix}"] = None

    # RAS — Rademacher Adjusted Sharpe lower bound on leader Sharpe
    try:
        complexity = rademacher_complexity(
            matrix,
            n_simulations=rademacher_n_simulations,
            random_state=rademacher_seed,
        )
        annualized_sharpes = sharpes  # already annualized
        ras_result = ras_sharpe_adjustment(
            annualized_sharpes,
            complexity=complexity,
            n_samples=n_periods,
            n_strategies=k_variants,
            return_result=True,
        )
        out["ras_complexity"] = float(complexity)
        out["ras_n_strategies"] = float(k_variants)
        out["ras_leader"] = float(ras_result.adjusted_values[leader_idx])
        # The library's RASResult doesn't expose a p-value directly; surface
        # the standardized leader-vs-zero z-score as a downstream proxy if
        # later wired. Leave None for now.
        out["ras_pvalue"] = None
    except _ESTIMATOR_ERRORS as exc:
        warnings.warn(f"ras_sharpe_adjustment failed for {leader_hash}: {exc}", stacklevel=2)
        out["ras_complexity"] = None
        out["ras_n_strategies"] = float(k_variants)
        out["ras_leader"] = None
        out["ras_pvalue"] = None

    # Reality Check vs baseline (optional). compute_reality_check is
    # defined above in the same module — no self-import needed.
    out["reality_check_pvalue"] = None
    out["reality_check_statistic"] = None
    out["reality_check_k"] = None
    if baseline_returns is not None:
        try:
            challenger_returns = {name: matrix[:, i] for i, name in enumerate(names)}
            rc = compute_reality_check(challenger_returns, baseline_returns)
            if rc:
                out["reality_check_pvalue"] = float(rc.get("reality_check_pvalue", float("nan")))
                out["reality_check_statistic"] = float(
                    rc.get("reality_check_statistic", float("nan"))
                )
                out["reality_check_k"] = float(rc.get("k_strategies", k_variants))
        except _ESTIMATOR_ERRORS as exc:
            warnings.warn(f"reality_check failed for {leader_hash}: {exc}", stacklevel=2)

    # PBO — CSCV on per-fold Sharpe matrix (optional). The library's
    # compute_pbo expects a pair of (n_combinations, K) IS/OOS Sharpe
    # matrices; with only an OOS-fold matrix in hand we synthesize the
    # CSCV partition here: for each balanced split of folds into IS/OOS
    # halves, the IS row is the mean Sharpe over the IS folds and the
    # OOS row is the mean over the complement.
    out["pbo"] = None
    out["pbo_n_combinations"] = None
    out["pbo_median_oos_rank"] = None
    out["pbo_mean_degradation"] = None
    out["pbo_n_folds"] = None
    if fold_returns_by_hash is not None:
        try:
            from ml4t.diagnostic.evaluation.stats import compute_pbo

            fold_names = [n for n in names if n in fold_returns_by_hash]
            if len(fold_names) >= 2:
                fold_sharpes = np.array(
                    [fold_returns_by_hash[n] for n in fold_names], dtype=np.float64
                ).T  # (n_folds, K)
                n_folds = fold_sharpes.shape[0]
                if fold_sharpes.ndim == 2 and n_folds >= 2:
                    is_perf, oos_perf = _cscv_split_pairs(fold_sharpes)
                    if is_perf.shape[0] >= 1:
                        pbo_result = compute_pbo(is_perf, oos_perf)
                        out["pbo"] = float(pbo_result.pbo)
                        out["pbo_n_combinations"] = float(pbo_result.n_combinations)
                        out["pbo_median_oos_rank"] = float(pbo_result.is_best_rank_oos_median)
                        out["pbo_mean_degradation"] = float(pbo_result.degradation_mean)
                        out["pbo_n_folds"] = float(n_folds)
        except _ESTIMATOR_ERRORS as exc:
            warnings.warn(f"pbo failed for {leader_hash}: {exc}", stacklevel=2)

    return out


def _cscv_split_pairs(
    fold_sharpes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (IS, OOS) Sharpe matrices for CSCV from an (n_folds, K) input.

    Enumerates every ``C(n_folds, n_folds // 2)`` IS-side choice and pairs
    it with its complement. For each split, the IS row is the mean Sharpe
    across the IS folds and the OOS row is the mean across the complement.
    Returns two arrays each of shape ``(n_combinations, K)`` suitable for
    :func:`ml4t.diagnostic.evaluation.stats.compute_pbo`.

    Partition sizes:

    - Even ``n_folds``: IS and OOS both have ``n_folds // 2`` folds — the
      canonical CSCV setup with balanced halves.
    - Odd ``n_folds``: IS has ``n_folds // 2`` folds and OOS has
      ``n_folds - n_folds // 2`` (one more) — the partition is *not*
      balanced. PBO interpretation still holds (overfitting probability
      that the IS-best variant underperforms OOS-median) but the OOS
      half averages more folds, so it has lower sampling variance than
      the IS half. Callers preferring strictly balanced halves should
      drop one fold before calling.

    For ``n_folds=1`` returns empty arrays — PBO is not defined.
    """
    n_folds, _ = fold_sharpes.shape
    if n_folds < 2:
        return np.empty((0, fold_sharpes.shape[1])), np.empty((0, fold_sharpes.shape[1]))
    half = n_folds // 2
    fold_ids = np.arange(n_folds)
    is_rows = []
    oos_rows = []
    for is_idx in combinations(fold_ids, half):
        is_mask = np.zeros(n_folds, dtype=bool)
        is_mask[list(is_idx)] = True
        is_rows.append(fold_sharpes[is_mask].mean(axis=0))
        oos_rows.append(fold_sharpes[~is_mask].mean(axis=0))
    return np.asarray(is_rows), np.asarray(oos_rows)


def _sharpe_per_column(matrix: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Annualized Sharpe per column of an (n_periods, k_variants) matrix."""
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0, ddof=1)
    sigma = np.where(sigma > 1e-12, sigma, np.nan)
    return mu / sigma * np.sqrt(periods_per_year)


def _sortino(arr: np.ndarray, periods_per_year: float) -> float:
    mu = float(np.mean(arr))
    downside = arr[arr < 0]
    if downside.size < 2:
        return float("nan")
    d_std = float(np.std(downside, ddof=1))
    if d_std <= 1e-12:
        return float("nan")
    return mu / d_std * float(np.sqrt(periods_per_year))


__all__ = [
    "STAGE_BASELINE",
    "SIGNAL_BASELINE_BY_CASE_STUDY",
    "resolve_block_length",
    "compute_backtest_uncertainty",
    "compute_paired_uncertainty",
    "compute_independent_diff_uncertainty",
    "compute_selection_adjustment",
    "compute_cohort_metrics",
    "load_daily_returns_with_timestamp",
    "compute_reality_check",
    "load_daily_returns",
]
