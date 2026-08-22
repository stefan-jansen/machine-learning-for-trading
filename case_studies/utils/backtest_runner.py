"""Core backtest execution — engine-first, used by BOTH demo and sweep notebooks.

This module provides a single ``run_backtest()`` function that:
1. Converts predictions to target weights via strategy_spec["signal"]
2. Dispatches to engine or vectorized path
3. Optionally registers the result in registry.db
4. Returns a unified result object

The key invariant is that **sweep notebooks call the same function as demo
notebooks**. There is no separate vectorized reimplementation for sweeps.

Usage::

    from case_studies.utils.backtest_runner import run_backtest

    result = run_backtest(
        case_study="etfs",
        prediction_hash="abc123",
        strategy_spec=spec,
        prices=prices,
        predictions=predictions,
    )
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from case_studies.utils.backtest_loaders import BacktestConfig, get_backtest_config
from case_studies.utils.backtest_presets import (
    apply_calendar_session_enforcement,
    ensure_backtest_spec,
    is_backtest_spec,
    runtime_backtest_config,
    strategy_view,
)
from case_studies.utils.signals import build_target_weights_from_config

# ---------------------------------------------------------------------------
# Periods per year for Sharpe annualization
# ---------------------------------------------------------------------------

# NOTE: a `cadence -> periods_per_year` table used to live here. It was the
# source of a real defect: it assumed each return observation is one rebalance
# period, which is false for every case study that rebalances slowly and marks
# to market daily. Measured grid frequencies are 252/yr for sp500_options and
# nasdaq100_microstructure despite weekly cohorts and intraday signals, and
# 252/yr for monthly-rebalanced etfs. Annualize with
# `resolve_periods_per_year` below, which reads the declared
# `evaluation.periods_per_year` and reconciles it against the grid on hand.


def observed_periods_per_year(daily_returns) -> float:
    """Sampling rate of a return frame, or 0.0 if it cannot be measured.

    Rows divided by elapsed span would answer a different question - how densely
    the frame covers its own window - and would read a daily series with a
    six-month hole in it as a twice-weekly one. What the annualization needs is
    the spacing between consecutive observations, so the rate is measured over
    the intervals that are typical of the series and gaps far longer than the
    typical spacing are dropped from both the count and the span.

    A weekday grid has intervals of one day and three across a weekend, both
    within the cut, and comes out near 252. A grid thinned to every fifth
    session has intervals near seven days throughout and comes out near 52. A
    delisting hole or a suspended market drops out of the span instead of
    dragging the rate down with it.
    """
    n = len(daily_returns)
    if n <= 2:
        return 0.0
    all_ts = daily_returns["timestamp"].unique().sort()
    gaps = all_ts.diff().drop_nulls().dt.total_seconds().to_numpy().astype(float)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return 0.0

    typical = float(np.median(gaps))
    if typical <= 0:
        return 0.0
    kept = gaps[gaps <= 5.0 * typical]
    if kept.size == 0:
        return 0.0

    span_years = float(kept.sum()) / (365.25 * 86400)
    return kept.size / span_years if span_years > 0.01 else 0.0


def reconcile_periods_per_year(declared: int, daily_returns, *, case_study: str = "") -> int:
    """Keep the declared factor unless the frame on hand contradicts it outright.

    The declared `evaluation.periods_per_year` is the authority: it describes
    the convention of the registered return grid, and using it keeps metrics on
    the round factor the rest of the pipeline shares. But it is a case-study
    constant, and one path writes a grid that does not match it. The vectorized
    path thins predictions to rebalance dates before aggregating, so with
    `sp500_options`'s rebalance_step of 5 a daily prediction grid leaves about
    74 rows a year against a declared 252.

    The override is one-directional, because thinning is the only mechanism by
    which the written grid departs from the declared convention, and it only
    coarsens. A frame that reads *finer* than declared is therefore not evidence
    that the declared value is wrong - an intraday grid whose case study still
    declares 252 is a setup.yaml that has not caught up, and inflating Sharpe by
    the square root of a factor of 139 is not the way to report that. It stays on
    the declared value and says so in the log.

    A factor of two of headroom on the coarse side keeps holidays and partial
    years on the declared value, and still catches thinning, which coarsens by
    the rebalance step and so lands well outside it.
    """
    import logging

    observed = observed_periods_per_year(daily_returns)
    if not observed or not declared:
        return declared or int(observed) or 252

    ratio = observed / declared
    if ratio > 2.0:
        logging.getLogger(__name__).warning(
            "%s: return grid is %.1f obs/yr, finer than the %d declared in setup.yaml. "
            "Annualizing at the declared value; declare the grid's own frequency if "
            "this case study writes intraday returns.",
            case_study or "backtest",
            observed,
            declared,
        )
        return declared
    if ratio >= 0.5:
        return declared

    logging.getLogger(__name__).warning(
        "%s: return grid is %.1f obs/yr but setup.yaml declares %d; annualizing at "
        "the observed rate. Thinning to rebalance dates is the usual cause.",
        case_study or "backtest",
        observed,
        declared,
    )
    return int(observed)


def resolve_periods_per_year(case_study: str, daily_returns) -> int:
    """Annualization factor for a return series, resolved once per backtest.

    Every path that annualizes a registered return grid goes through here, so
    the overall metrics, the per-fold metrics and the backfill cannot disagree
    about what the same frame is.
    """
    from case_studies.utils.uncertainty import periods_per_year_from_setup

    try:
        declared = int(periods_per_year_from_setup(case_study))
    except (FileNotFoundError, KeyError):
        declared = 0

    return reconcile_periods_per_year(declared, daily_returns, case_study=case_study)


def overall_periods_per_year(case_study: str | None, calendar: str, daily_returns) -> int:
    """Annualization for a whole-window backtest, whichever engine produced it.

    The engine path used to read the exchange calendar, which returns ~252 for
    every equity venue and so disagreed with a monthly case study by a factor of
    21, and with its own per-fold metrics. A named case study now resolves the
    same way everywhere; the calendar remains for a call that has no case study
    and therefore no declared value to read.
    """
    if case_study:
        return resolve_periods_per_year(case_study, daily_returns)
    return calendar_periods_per_year(calendar)


# Calendar name → exchange_calendars MIC code
_CALENDAR_TO_XCAL: dict[str, str] = {
    "NYSE": "XNYS",
    "CME": "us_futures",
    "FX": "24/5",
    "crypto": "24/7",
}

# Cache for calendar session counts
_calendar_ppy_cache: dict[str, int] = {}


def calendar_periods_per_year(calendar: str) -> int:
    """Get trading days per year for a calendar using exchange_calendars.

    Computes the average number of sessions over a 10-year window
    (2015-2024) and caches the result.
    """
    if calendar in _calendar_ppy_cache:
        return _calendar_ppy_cache[calendar]

    xcal_name = _CALENDAR_TO_XCAL.get(calendar, calendar)

    try:
        import exchange_calendars as xcals

        cal = xcals.get_calendar(xcal_name)
        total = sum(
            len(cal.sessions_in_range(f"{y}-01-01", f"{y}-12-31")) for y in range(2015, 2025)
        )
        ppy = round(total / 10)
    except Exception:
        # Fallback if exchange_calendars unavailable or calendar unknown
        ppy = 252

    _calendar_ppy_cache[calendar] = ppy
    return ppy


# ---------------------------------------------------------------------------
# Portfolio metrics via ml4t-diagnostic
# ---------------------------------------------------------------------------


def compute_portfolio_metrics(
    returns: np.ndarray,
    *,
    periods_per_year: int = 252,
    case_study: str | None = None,
    label: str | None = None,
    uncertainty: bool = True,
    uncertainty_n_boot: int = 1000,
    uncertainty_seed: int = 0,
    trim_leading_zeros: bool = False,
) -> dict[str, float]:
    """Compute portfolio metrics using ml4t-diagnostic.

    Replaces hand-rolled Sharpe/drawdown/etc. with the library's
    validated implementation. When ``uncertainty=True`` (default) the returned
    dict is extended with block-bootstrap CIs, Lo/LdP-2025 Sharpe SE,
    Newey-West HAC SE for annualized return, and PSR p-value — driven by
    :func:`case_studies.utils.uncertainty.compute_backtest_uncertainty`.

    Parameters
    ----------
    returns : np.ndarray
        Array of period returns (daily or per-rebalance).
    periods_per_year : int
        Annualization factor (252 for daily, 52 for weekly, etc.).
    case_study, label : optional
        Used by the block-length resolver to pick rebalance_step from setup.yaml.
    uncertainty : bool, default True
        If False, skip the bootstrap (fast path for sweep inner loops).
    uncertainty_n_boot, uncertainty_seed : int
        Bootstrap configuration.
    trim_leading_zeros : bool, default False
        Legacy first-non-zero strip. Kept for callers that pass pre-canonical
        return series (e.g., raw engine output without canonical-window slice).
        Production callers (``_run_engine`` and the retrofit pipeline) pass
        ``False`` because they slice to the canonical (cs, label, split) window
        first, which preserves real "no-trade" days at the start of the window
        as legitimate zero-return periods rather than stripping them.

    Returns
    -------
    dict[str, float]
        Metric name → value. Keys match the existing backtest_metrics schema,
        plus uncertainty columns when ``uncertainty=True``.
    """
    from ml4t.diagnostic.evaluation import PortfolioAnalysis

    if trim_leading_zeros and len(returns) > 0:
        nonzero = np.flatnonzero(np.asarray(returns) != 0.0)
        if len(nonzero) > 0:
            returns = returns[nonzero[0] :]

    if len(returns) < 2:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "cagr": 0.0,
            "calmar": 0.0,
            "volatility": 0.0,
            "win_rate": 0.0,
            "omega": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "stability": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "tail_ratio": 0.0,
            "n_periods": int(len(returns)),
        }

    analysis = PortfolioAnalysis(returns=returns, periods_per_year=periods_per_year)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation",
            category=RuntimeWarning,
        )
        pm = analysis.compute_summary_stats()

    def _safe(v: float) -> float:
        """Sanitize metric value: handle complex, inf, nan."""
        if isinstance(v, complex):
            v = v.real
        if not np.isfinite(v):
            return 0.0
        return float(v)

    out = {
        "sharpe": _safe(pm.sharpe_ratio),
        "sortino": _safe(pm.sortino_ratio),
        "total_return": _safe(pm.total_return),
        "max_drawdown": _safe(pm.max_drawdown),
        "cagr": _safe(pm.annual_return),
        "calmar": _safe(pm.calmar_ratio),
        "volatility": _safe(pm.annual_volatility),
        "win_rate": _safe(pm.win_rate),
        "omega": _safe(pm.omega_ratio),
        "var_95": _safe(pm.var_95),
        "cvar_95": _safe(pm.cvar_95),
        "stability": _safe(pm.stability),
        "skewness": _safe(pm.skewness),
        "kurtosis": _safe(pm.kurtosis),
        "tail_ratio": _safe(pm.tail_ratio),
        "n_periods": int(len(returns)),
    }

    if uncertainty and len(returns) >= 4:
        try:
            from case_studies.utils.uncertainty import compute_backtest_uncertainty

            unc = compute_backtest_uncertainty(
                returns,
                periods_per_year=periods_per_year,
                case_study=case_study,
                label=label,
                n_boot=uncertainty_n_boot,
                seed=uncertainty_seed,
            )
            out.update(unc)
        except Exception as exc:  # pragma: no cover - never block point estimates
            warnings.warn(
                f"compute_backtest_uncertainty failed: {exc}; point metrics returned without CIs",
                stacklevel=2,
            )

    return out


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestRunResult:
    """Unified result from both engine and vectorized paths."""

    daily_returns: pl.DataFrame  # [timestamp, daily_return]
    metrics: dict[str, float]
    strategy_spec: dict
    prediction_hash: str
    backtest_hash: str | None = None
    # Engine-only fields
    engine_result: Any = None  # BacktestResult from ml4t-backtest
    weights: pl.DataFrame | None = None
    execution_mode: str = "engine"


def _target_weights_by_timestamp(
    weights: pl.DataFrame,
) -> dict[date | datetime, dict[str, float]]:
    """Build deterministic timestamp and symbol ordered engine targets."""
    duplicate_count = weights.select(pl.struct("timestamp", "symbol").is_duplicated().sum()).item()
    if duplicate_count:
        raise ValueError(
            f"Target weights contain {duplicate_count} duplicate timestamp-symbol rows"
        )
    targets: dict[date | datetime, dict[str, float]] = {}
    for row in weights.sort("timestamp", "symbol").iter_rows(named=True):
        timestamp = row["timestamp"]
        if timestamp not in targets:
            targets[timestamp] = {}
        targets[timestamp][row["symbol"]] = row["weight"]
    return targets


def _engine_timestamp(
    value: object,
    *,
    feed_is_date: bool,
    feed_timezone: str | None,
    configured_timezone: str,
) -> date | datetime:
    if feed_is_date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        raise TypeError(f"engine target timestamp must be date-like, got {type(value).__name__}")
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, date):
        timestamp = datetime.combine(value, time.min)
    else:
        raise TypeError(f"engine target timestamp must be date-like, got {type(value).__name__}")
    if feed_timezone is not None:
        zone = ZoneInfo(feed_timezone)
        return (
            timestamp.replace(tzinfo=zone)
            if timestamp.tzinfo is None
            else timestamp.astimezone(zone)
        )
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(ZoneInfo(configured_timezone)).replace(tzinfo=None)
    return timestamp


# ---------------------------------------------------------------------------
# Weight precomputation (for risk sweep reuse)
# ---------------------------------------------------------------------------


def precompute_weights(
    predictions: pl.DataFrame,
    strategy_spec: dict,
    prices: pl.DataFrame,
    *,
    label: str = "",
    case_study: str = "",
    prediction_hash: str | None = None,
    conformal_widths: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute allocation weights from a strategy spec, without running the engine.

    Use this to avoid redundant MVO/HRP computation in Ch19 risk sweeps
    where the same allocation weights are tested with different risk overlays.

    ``prediction_hash`` is required for the ``conformal_weighted`` allocation
    method (it loads that prediction's conformal widths); other methods ignore it.

    Returns
    -------
    pl.DataFrame
        Weights [timestamp, symbol, weight] ready for ``run_backtest(precomputed_weights=...)``.
    """
    predictions = normalize_prediction_columns(predictions)
    strategy = strategy_view(strategy_spec)
    signal_config = strategy["signal"]
    rebal_spec = strategy.get("rebalance", {})
    weights = build_target_weights_from_config(predictions, signal_config)
    alloc_spec = strategy.get("allocation")
    if alloc_spec:
        cadence = strategy.get("rebalance", {}).get("cadence", "")
        weights = _apply_allocation(
            weights,
            predictions,
            prices,
            alloc_spec,
            cadence=cadence,
            label=label,
            case_study=case_study,
            prediction_hash=prediction_hash,
            conformal_widths=conformal_widths,
        )
    return weights


# ---------------------------------------------------------------------------
# Strategy spec construction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prediction normalization
# ---------------------------------------------------------------------------


def normalize_prediction_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize prediction columns to canonical [timestamp, symbol, y_score, ...]."""
    renames = {}

    # Time column: date → timestamp
    if "timestamp" not in df.columns and "date" in df.columns:
        renames["date"] = "timestamp"

    # Entity column: asset/product/stock_id/entity → symbol
    if "symbol" not in df.columns:
        for col in ("asset", "product", "stock_id", "entity"):
            if col in df.columns:
                renames[col] = "symbol"
                break

    # Score column
    if "y_score" not in df.columns:
        if "prediction" in df.columns:
            renames["prediction"] = "y_score"

    if "y_true" not in df.columns and "actual" in df.columns:
        renames["actual"] = "y_true"
    if "fold_id" not in df.columns and "fold" in df.columns:
        renames["fold"] = "fold_id"

    if renames:
        df = df.rename(renames)

    # Cast types
    if "timestamp" in df.columns:
        ts_dtype = df.schema["timestamp"]
        if ts_dtype == pl.Date:
            df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        elif ts_dtype in (pl.String, pl.Utf8):
            df = df.with_columns(pl.col("timestamp").str.to_datetime().cast(pl.Datetime("us")))
        elif hasattr(ts_dtype, "time_zone") and ts_dtype.time_zone:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    if "symbol" in df.columns and df.schema["symbol"] != pl.String:
        df = df.with_columns(pl.col("symbol").cast(pl.String))

    return df


# Tolerant-by-design cap on (timestamp, symbol) join misses between a
# classification label and its continuous-return counterpart. >10% null
# rate indicates a regeneration mismatch between the two label parquets
# (not source-data sparsity), and is escalated to a hard error rather
# than silently dropping rows from the backtest. Callers operating in
# a legitimately high-null regime can override via the ``max_null_rate``
# parameter on ``substitute_continuous_return_for_classification``.
_MAX_NULL_RATE = 0.10

# Polars integer dtypes for symbol id columns (e.g., us_firm ``stock_id``).
# Used by ``_align_symbol_dtype`` to detect numeric-vs-string mismatches.
_INT_SYMBOL_DTYPES = (
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
)


def _align_symbol_dtype(
    target: pl.DataFrame,
    other: pl.DataFrame,
    *,
    case_study: str,
    target_side: str,
    other_side: str,
) -> pl.DataFrame:
    """Cast ``other['symbol']`` to ``target['symbol'].dtype``, failing loudly.

    Polars will silently raise ``InvalidOperationError`` when a string
    column with real tickers (e.g. ``"AAPL"``) is cast to integer — the
    error message names neither the case study nor the column origin,
    making diagnostics painful. This helper detects the pl.Utf8 ↔
    integer mismatch and surfaces a context-rich error before the cast,
    keeping the same behavior for compatible cases (same dtype, or
    same-kind cast).
    """
    target_dtype = target["symbol"].dtype
    other_dtype = other["symbol"].dtype
    if other_dtype == target_dtype:
        return other
    target_is_int = target_dtype in _INT_SYMBOL_DTYPES
    other_is_str = other_dtype in (pl.Utf8, pl.String)
    other_is_int = other_dtype in _INT_SYMBOL_DTYPES
    target_is_str = target_dtype in (pl.Utf8, pl.String)
    if target_is_int and other_is_str:
        # Probe: every value must parse as the target integer dtype.
        try:
            return other.with_columns(pl.col("symbol").cast(target_dtype))
        except Exception as exc:  # noqa: BLE001 — surface Polars's opaque error
            raise TypeError(
                f"_align_symbol_dtype: incompatible symbol representations for "
                f"case_study={case_study!r}: {target_side}.symbol is "
                f"{target_dtype} (numeric ids) but {other_side}.symbol is "
                f"{other_dtype} (likely tickers, not parseable as integer). "
                f"Underlying Polars error: {exc}"
            ) from exc
    if other_is_int and target_is_str:
        return other.with_columns(pl.col("symbol").cast(target_dtype))
    # Same-kind cast (e.g., Int32 → Int64, Utf8 → String alias).
    return other.with_columns(pl.col("symbol").cast(target_dtype))


def substitute_continuous_return_for_classification(
    predictions: pl.DataFrame,
    case_study: str,
    label: str,
    *,
    max_null_rate: float = _MAX_NULL_RATE,
) -> pl.DataFrame:
    """Replace binary y_true with the underlying continuous return for classification labels.

    The vectorized backtest computes ``gross_ret = weight * y_true``. For
    regression labels y_true is the forward return; for classification
    labels (fwd_class_*, fwd_dir_*) it is the binary class indicator, so
    the product collapses into a position-weighted accuracy proxy rather
    than economic P&L. We substitute y_true with the continuous return
    declared in setup.yaml::labels.classification_eval_label.

    Returns predictions unchanged when ``label`` is not registered as a
    classification target (i.e., regression labels pass through).
    """
    if not label:
        return predictions
    from pathlib import Path as _Path

    import yaml as _yaml

    from utils import CASE_STUDIES_DIR

    setup_path = _Path(CASE_STUDIES_DIR) / case_study / "config" / "setup.yaml"
    if not setup_path.exists():
        return predictions
    setup = _yaml.safe_load(setup_path.read_text())
    mapping = (setup.get("labels") or {}).get("classification_eval_label") or {}
    if label not in mapping:
        return predictions

    eval_label = str(mapping[label])
    eval_path = _Path(CASE_STUDIES_DIR) / case_study / "labels" / f"{eval_label}.parquet"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Continuous-return label {eval_label!r} expected at {eval_path} "
            f"for classification label {label!r} but not found. Required so the "
            f"vectorized backtest can compute economic P&L instead of weight × binary."
        )
    eval_df = pl.read_parquet(eval_path).select(["timestamp", "symbol", eval_label])

    # Dedupe-assert eval_df on the join key before the left join. A duplicate
    # (timestamp, symbol) row in the continuous-return parquet would fan out
    # ``predictions`` silently, inflating downstream weight × y_true into a
    # wrong-but-plausible P&L (the very failure mode this function is meant
    # to prevent on the classification path).
    eval_h0 = eval_df.height
    eval_h_uniq = eval_df.unique(subset=["timestamp", "symbol"]).height
    if eval_h_uniq != eval_h0:
        raise ValueError(
            f"substitute_continuous_return_for_classification: continuous-return "
            f"label parquet at {eval_path} has {eval_h0 - eval_h_uniq} duplicate "
            f"(timestamp, symbol) rows ({eval_h_uniq} unique). Re-run the upstream "
            f"label step for case_study={case_study!r} to produce a unique-keyed "
            f"parquet."
        )
    eval_df = eval_df.unique(subset=["timestamp", "symbol"], keep="first")

    # Harmonize join-key dtypes to match the (already-normalized) predictions frame.
    if eval_df["timestamp"].dtype != predictions["timestamp"].dtype:
        if eval_df["timestamp"].dtype == pl.Date:
            eval_df = eval_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        eval_df = eval_df.cast({"timestamp": predictions["timestamp"].dtype})
    eval_df = _align_symbol_dtype(
        predictions,
        eval_df,
        case_study=case_study,
        target_side="predictions",
        other_side=f"labels/{eval_label}.parquet",
    )

    pred_h0 = predictions.height
    joined = (
        predictions.drop("y_true")
        .join(eval_df, on=["timestamp", "symbol"], how="left")
        .rename({eval_label: "y_true"})
    )
    # Height-assert: ``left`` should never produce more rows than the left frame
    # carried in. Belt-and-suspenders for the dedupe assertion above.
    if joined.height != pred_h0:
        raise RuntimeError(
            f"substitute_continuous_return_for_classification: left join "
            f"changed row count {pred_h0} -> {joined.height} for case_study="
            f"{case_study!r} label={label!r}. eval_df keys are not unique "
            f"on (timestamp, symbol) after dedupe — internal invariant broken."
        )

    n_null = int(joined["y_true"].null_count())
    if n_null > 0:
        n_total = joined.height
        null_rate = n_null / n_total
        # Tolerant-by-design caps at ``max_null_rate`` (default
        # ``_MAX_NULL_RATE`` = 10%); above that, raise. >10% null rate
        # indicates a regeneration mismatch between the classification and
        # continuous-return parquets, not source-data sparsity.
        if null_rate > max_null_rate:
            raise ValueError(
                f"substitute_continuous_return_for_classification: "
                f"{n_null}/{n_total} ({null_rate:.2%}) predictions for "
                f"classification label {label!r} have no matching {eval_label!r} "
                f"value after join on (timestamp, symbol); exceeds "
                f"max_null_rate={max_null_rate:.2%}. Null rate above "
                f"{max_null_rate:.0%} indicates a regeneration mismatch "
                f"between the classification and continuous-return label "
                f"parquets; re-run the upstream label step for {case_study!r}."
            )
        print(
            f"  WARN substitute_continuous_return_for_classification: "
            f"{n_null}/{n_total} ({null_rate:.4%}) predictions for "
            f"classification label {label!r} have no matching {eval_label!r} "
            f"value after join on (timestamp, symbol); dropping those rows."
        )
        joined = joined.filter(pl.col("y_true").is_not_null())
    return joined


def _apply_cost_feasible_filter(
    predictions: pl.DataFrame,
    case_study: str,
    prediction_hash: str | None,
) -> pl.DataFrame:
    """Restrict predictions to the frozen, per-split cost-feasible universe.

    The split is resolved from the prediction set's registry entry; the
    symbol list is read from ``setup.yaml::universe.cost_feasible.{split}``.
    Raises if the split cannot be resolved or the list is absent — a silent
    full-universe fallback would change the registered result.
    """
    from pathlib import Path as _Path

    import yaml as _yaml

    from case_studies.utils.cv_window import lookup_split
    from utils import CASE_STUDIES_DIR

    if not prediction_hash:
        raise ValueError(
            "universe_filter='cost_feasible' requires a prediction_hash to "
            f"resolve the split for case_study={case_study!r}; got none."
        )
    split = lookup_split(case_study, prediction_hash)
    if split not in ("validation", "holdout"):
        raise ValueError(
            f"universe_filter='cost_feasible' could not resolve split for "
            f"prediction_hash={prediction_hash!r} (case_study={case_study!r}); "
            f"lookup_split returned {split!r}. The prediction set must be "
            f"registered with a 'validation' or 'holdout' split first."
        )
    setup = _yaml.safe_load(
        (_Path(CASE_STUDIES_DIR) / case_study / "config" / "setup.yaml").read_text()
    )
    symbols = (((setup.get("universe") or {}).get("cost_feasible")) or {}).get(split)
    if not symbols:
        raise KeyError(
            f"setup.yaml::universe.cost_feasible.{split} missing/empty for "
            f"case_study={case_study!r}; required when "
            f"signal.universe_filter='cost_feasible'."
        )
    filtered = predictions.filter(pl.col("symbol").is_in(list(symbols)))
    if filtered.is_empty() and not predictions.is_empty():
        raise ValueError(
            f"universe_filter='cost_feasible' produced an empty frame for "
            f"case_study={case_study!r} split={split!r}: the prediction set's "
            f"symbols do not intersect the frozen cost-feasible list (e.g. a "
            f"point-in-time ticker mismatch like FB/META). Refusing to run a "
            f"zero-row backtest — same 'no silent fallback' intent as above."
        )
    return filtered


def apply_universe_filter(
    predictions: pl.DataFrame,
    prices: pl.DataFrame,
    case_study: str,
    signal_config: dict | None,
    prediction_hash: str | None = None,
) -> pl.DataFrame:
    """Apply spec-declared universe restriction to predictions before backtest.

    When ``signal_config["universe_filter"] == "liquid"`` (sp500_options
    rung-3 in the O'Donovan-Yu / Muravyev-Pearson HTM cost cascade), the
    backtest must restrict each rebalance date to the tightest-quoted
    subset of the universe. The quantile lives in
    ``setup.yaml::backtest.sweep.htm_cost_cascade.liquid_quantile``; the
    spread column is ``instr_rel_spread`` on the prices frame.

    When ``signal_config["universe_filter"] == "cost_feasible"``
    (nasdaq100_microstructure), the backtest restricts predictions to a
    FROZEN, per-split symbol list committed under
    ``setup.yaml::universe.cost_feasible.{validation,holdout}``. The list is
    the cost-feasible universe — the cheapest-to-trade names by round-trip
    cost, profiled strictly before each window (no look-ahead — see
    ``build_cost_feasible_universe.py``); the split is resolved from the
    prediction set's registry entry via ``lookup_split``. Like ``liquid``,
    only the filter *name* enters the backtest hash, not the resolved symbols.

    Returns predictions unchanged when no filter applies. Built into
    ``run_backtest`` so any caller — sweep notebooks, ``generate_holdout``,
    ad-hoc scripts — gets the same filter as the bespoke sp500_options
    pipeline, driven purely by the strategy spec.
    """
    if not signal_config:
        return predictions
    uf = str(signal_config.get("universe_filter", "")).strip().lower()
    if uf in ("", "full", "none"):
        return predictions
    if uf == "cost_feasible":
        return _apply_cost_feasible_filter(predictions, case_study, prediction_hash)
    if uf != "liquid":
        raise ValueError(
            f"universe_filter={uf!r} not supported. Allowed: 'liquid', 'cost_feasible', or 'full'."
        )
    if "instr_rel_spread" not in prices.columns:
        raise ValueError(
            f"universe_filter='liquid' requires 'instr_rel_spread' on the prices "
            f"frame for case_study={case_study!r}; got columns={list(prices.columns)}."
        )
    # Deliberately pinned to the repository copy rather than resolved through
    # get_case_study_dir/get_htm_cost_cascade: liquid_quantile is NOT part of the
    # backtest spec and so never enters backtest_hash. Letting an experiment copy
    # vary it would let two different quantiles collide on one hash against the
    # registry the experiment inherits. Making it experiment-editable therefore
    # requires plumbing it into strategy.signal and the hash first, and dropping
    # the hardcoded LIQUID_QUANTILE prefilters in sp500_options 12_backtest.py
    # and 13_portfolio_management.py - a hash-identity change, not a config change.
    from pathlib import Path as _Path

    import yaml as _yaml

    from utils import CASE_STUDIES_DIR

    setup = _yaml.safe_load(
        (_Path(CASE_STUDIES_DIR) / case_study / "config" / "setup.yaml").read_text()
    )
    cascade = (((setup.get("backtest") or {}).get("sweep") or {}).get("htm_cost_cascade")) or {}
    if "liquid_quantile" not in cascade:
        raise KeyError(
            f"setup.yaml::backtest.sweep.htm_cost_cascade.liquid_quantile missing for "
            f"case_study={case_study!r}; required when signal.universe_filter='liquid'."
        )
    liquid_quantile = float(cascade["liquid_quantile"])

    # Daily quantile of relative half-spread; ties broken with rank('min').
    # Collapse timestamp to the date grain before grouping so any caller
    # supplying sub-daily or unnormalized intraday bars still produces a
    # within-date rank rather than a within-bar rank (mirrors the bespoke
    # sp500_options sweep). Dedupe ``(date, symbol)`` to one row per
    # (date, symbol) — taking the min half-spread when multiple bars share
    # a date — so the rank denominator is symbol-count, not bar-count.
    half = (
        prices.select(
            pl.col("timestamp").cast(pl.Date).alias("_date"),
            pl.col("symbol"),
            (pl.col("instr_rel_spread") / 2).alias("_hs"),
        )
        .group_by(["_date", "symbol"])
        .agg(pl.col("_hs").min())
    )
    liquid_keys = (
        half.with_columns(
            (pl.col("_hs").rank("min").over("_date") / pl.col("_hs").count().over("_date")).alias(
                "_q"
            )
        )
        .filter(pl.col("_q") <= liquid_quantile)
        .select([pl.col("_date").alias("timestamp"), pl.col("symbol")])
    )
    if liquid_keys["timestamp"].dtype != predictions["timestamp"].dtype:
        # Predictions stamps are typically Datetime("us") at midnight; cast
        # back from Date so the semi-join key types match exactly.
        if predictions["timestamp"].dtype == pl.Datetime("us"):
            liquid_keys = liquid_keys.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        else:
            liquid_keys = liquid_keys.cast({"timestamp": predictions["timestamp"].dtype})
    liquid_keys = _align_symbol_dtype(
        predictions,
        liquid_keys,
        case_study=case_study,
        target_side="predictions",
        other_side="prices",
    )
    return predictions.join(liquid_keys, on=["timestamp", "symbol"], how="semi")


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------


def resolved_allow_short_selling(
    strategy_spec: dict,
    precomputed_weights: pl.DataFrame | None = None,
) -> bool:
    """Return the account short-selling flag implied by a resolved strategy spec.

    Identity-defining, so it has exactly one implementation: callers that hash a
    spec before handing it to :func:`run_backtest` must derive the flag through
    this function, or the spec they hashed and the spec that runs will differ.
    """
    signal_config = strategy_view(strategy_spec)["signal"]
    allow_short = bool(
        strategy_spec["backtest_config"]["account"].get("allow_short_selling", False)
    ) or bool(signal_config.get("long_short", False))
    allow_short = allow_short or (
        str(signal_config.get("direction", "long_only")).strip().lower() == "short_only"
    )
    if precomputed_weights is not None:
        allow_short = allow_short or bool(precomputed_weights.filter(pl.col("weight") < 0).height)
    return allow_short


def run_backtest(
    case_study: str,
    prediction_hash: str,
    strategy_spec: dict,
    *,
    prices: pl.DataFrame,
    predictions: pl.DataFrame,
    label: str = "",
    register: bool = True,
    initial_cash: float = 1_000_000.0,
    calendar: str = "NYSE",
    precomputed_weights: pl.DataFrame | None = None,
    funding_rates: pl.DataFrame | None = None,
    force_rebacktest: bool = False,
    resolved_spec_only: bool = False,
    contract_specs: dict | None = None,
    option_lifecycle: pl.DataFrame | None = None,
) -> BacktestRunResult:
    """Core backtest: predictions -> weights -> engine/vectorized -> result.

    This is the SINGLE entry point for ALL backtests — demo, sweep, and
    downstream chapters. Sweep notebooks call this in a loop with different
    strategy_specs; they never contain backtest math themselves.

    Parameters
    ----------
    case_study : str
        Case study identifier (e.g., "etfs").
    prediction_hash : str
        Hash of the prediction set being backtested.
    strategy_spec : dict
        Identity-defining configuration with signal, execution, costs sections.
    prices : pl.DataFrame
        Price data [timestamp, symbol, open, high, low, close, volume].
    predictions : pl.DataFrame
        Predictions [timestamp, symbol, y_score, y_true, ...].
    label : str
        Label name (used for thinning in vectorized mode).
    register : bool
        Whether to register the result in registry.db.
    initial_cash : float
        Starting portfolio value.
    calendar : str
        Trading calendar for daily return aggregation.
    precomputed_weights : pl.DataFrame, optional
        Pre-computed allocation weights [timestamp, symbol, weight].
        When provided, skips signal computation and allocation — goes
        straight to engine/vectorized with these weights. Use this in
        Ch19 risk sweeps where allocation is identical across risk
        variants (avoids re-running expensive MVO/HRP per variant).
    contract_specs : dict, optional
        Per-asset contract specifications (futures multipliers, tick sizes).
        Pass for futures case studies to get correct P&L scaling.
    funding_rates : pl.DataFrame, optional
        Official position-signed perpetual-futures funding settlements.

    Returns
    -------
    BacktestRunResult
        Unified result with daily_returns, metrics, and optional engine_result.
    """
    import time
    from datetime import UTC

    _bt_started_at = datetime.now(UTC).isoformat()
    _bt_t0 = time.perf_counter()

    # 0. Normalize prediction columns to canonical schema, then for
    # classification labels replace the binary y_true with the underlying
    # continuous return so weight × y_true produces economic P&L rather
    # than a position-weighted accuracy proxy (see
    # ``substitute_continuous_return_for_classification`` docstring).
    predictions = normalize_prediction_columns(predictions)
    predictions = substitute_continuous_return_for_classification(predictions, case_study, label)
    if resolved_spec_only:
        # The caller (the locked-holdout producer) hashed this exact spec and must
        # get that identity back. Re-deriving it here would reread the mutable
        # preset that ensure_backtest_spec consults, so an unrelated setup.yaml
        # edit could silently move a locked backtest's identity.
        if not is_backtest_spec(strategy_spec):
            raise ValueError("resolved_spec_only requires an already-canonical backtest spec")
        declared = strategy_spec.get("backtest_config", {}).get("metadata", {})
        if declared.get("prediction_hash") != prediction_hash:
            raise ValueError(
                "resolved backtest specification does not declare the prediction being run"
            )
        # ensure_backtest_spec would have filled these from the preset. Skipping it means
        # a spec that omits them reaches the rebalance logic and dies on a bare KeyError
        # far from the cause, so require them here instead of defaulting them back in.
        rebalance = strategy_spec.get("strategy", {}).get("rebalance") or {}
        missing = {"min_weight_change", "min_trade_value"} - set(rebalance)
        if missing:
            raise ValueError(f"resolved backtest specification omits rebalance {sorted(missing)}")
        strategy_spec = deepcopy(strategy_spec)
    else:
        strategy_spec = ensure_backtest_spec(
            case_study,
            get_backtest_config(case_study),
            strategy_spec,
            prices=prices,
            prediction_hash=prediction_hash,
            initial_cash=initial_cash,
        )
    from case_studies.utils.conformal import ensure_conformal_calibration_identity

    strategy_spec = ensure_conformal_calibration_identity(strategy_spec)
    # Re-source initial_cash from the canonical spec. ensure_backtest_spec's
    # idempotent-canonical branch preserves an existing backtest_config.cash.initial
    # (typically $100K from setup.yaml) without overwriting it from the function
    # arg. The broker starts at that spec value; the RiskManager must initialize
    # its high-water-mark from the same number or it sees a fictitious 90%
    # drawdown on bar 1 when the function-arg default ($1M) diverges from the
    # spec ($100K) — halting the strategy before any trade is placed.
    initial_cash = float(strategy_spec["backtest_config"]["cash"]["initial"])
    strategy = strategy_view(strategy_spec)
    signal_config = strategy["signal"]
    allow_short = resolved_allow_short_selling(strategy_spec, precomputed_weights)
    strategy_spec["backtest_config"]["account"]["allow_short_selling"] = allow_short

    # Apply spec-declared universe restriction (e.g., sp500_options rung-3
    # 'liquid' subset). Driven purely by strategy.signal.universe_filter so
    # the bespoke sweep notebooks and generic generate_holdout share the
    # same code path.
    predictions = apply_universe_filter(
        predictions,
        prices,
        case_study,
        strategy.get("signal") or {},
        prediction_hash=prediction_hash,
    )

    # Skip-if-complete: if the backtest_hash already has complete artifacts,
    # return the cached result instead of re-running (unless force_rebacktest).
    if register and not force_rebacktest:
        from case_studies.utils.registry import backtest_dir as _bt_dir_fn
        from case_studies.utils.registry import backtest_run_status
        from case_studies.utils.registry.store import _case_dir, _open_registry

        _bt_status = backtest_run_status(case_study, prediction_hash, strategy_spec)
        if _bt_status.complete:
            _cached_dir = _bt_dir_fn(case_study, _bt_status.backtest_hash)
            _cached_returns = _cached_dir / "daily_returns.parquet"
            if _cached_returns.exists():
                print(f"  SKIP backtest ({_bt_status.summary()}) — reusing cached result")
                cached_df = pl.read_parquet(_cached_returns)
                # Load cached metrics from registry
                _db = _open_registry(_case_dir(case_study))
                try:
                    _metric_cols = [
                        r[1] for r in _db.execute("PRAGMA table_info(backtest_metrics)").fetchall()
                    ]
                    _metric_cols = [
                        c for c in _metric_cols if c not in ("backtest_hash", "computed_at")
                    ]
                    if _metric_cols:
                        _q = f"SELECT {', '.join(_metric_cols)} FROM backtest_metrics WHERE backtest_hash = ?"
                        _row = _db.execute(_q, (_bt_status.backtest_hash,)).fetchone()
                        cached_metrics = dict(zip(_metric_cols, _row, strict=True)) if _row else {}
                    else:
                        cached_metrics = {}
                finally:
                    _db.close()
                return BacktestRunResult(
                    daily_returns=cached_df,
                    metrics=cached_metrics,
                    strategy_spec=strategy_spec,
                    prediction_hash=prediction_hash,
                    backtest_hash=_bt_status.backtest_hash,
                    engine_result=None,
                    weights=precomputed_weights,
                    execution_mode=strategy.get("rebalance", {}).get("mode", "unknown"),
                )

    rebal_spec = strategy.get("rebalance", {})

    if precomputed_weights is not None:
        # Skip signal + allocation — use provided weights directly
        weights = precomputed_weights
    elif signal_config.get("method") == "slot_persistent_signal_exit":
        # Slot selection IS the allocation — Ch17 allocator stage is skipped
        # because the slot mechanism's `weight_per_slot` plays the role of
        # the cheap allocator. See case_studies/utils/slot_strategy.py for
        # the mechanism and rules/standards docs for the design call.
        from case_studies.utils.slot_strategy import build_persistent_slot_weights_hybrid

        slot_kwargs: dict = {}
        for required in ("long_q", "lookback_days", "bars_per_day", "max_slots", "hold_bars"):
            # Reject both absent and explicit-None values so a hand-wired
            # signal_config surfaces here rather than as a far-off TypeError
            # inside build_persistent_slot_weights_hybrid.
            if signal_config.get(required) is None:
                msg = (
                    f"signal method 'slot_persistent_signal_exit' requires a "
                    f"non-null {required!r} in signal_config; got {sorted(signal_config)}"
                )
                raise KeyError(msg)
            slot_kwargs[required] = signal_config[required]
        for opt in (
            "weight_per_slot",
            "exit_signal_q",
            "take_profit",
            "stop_loss",
            "pred_freshness_max_min",
            "direction",
        ):
            if signal_config.get(opt) is not None:
                slot_kwargs[opt] = signal_config[opt]
        weights, _slot_stats = build_persistent_slot_weights_hybrid(
            predictions,
            prices,
            **slot_kwargs,
        )
    else:
        # 1. Convert predictions to target weights
        weights = build_target_weights_from_config(predictions, signal_config)

        # 1b. Apply allocation method if specified (Ch17+)
        alloc_spec = strategy.get("allocation")
        if alloc_spec:
            weights = _apply_allocation(
                weights,
                predictions,
                prices,
                alloc_spec,
                cadence=rebal_spec.get("cadence", ""),
                label=label,
                case_study=case_study,
                prediction_hash=prediction_hash,
            )

    # 2. Dispatch to engine or vectorized
    bt_cfg = strategy_spec["backtest_config"]
    commission_block = bt_cfg["commission"]
    slippage_block = bt_cfg["slippage"]
    if commission_block.get("model") == "per_share":
        cost_spec = {
            "model": "per_share_plus_spread",
            "per_share": float(commission_block["per_share"]),
            "default_half_spread_usd": float(slippage_block.get("spread", 0.0)),
            "asset_spreads": dict(slippage_block.get("spread_by_asset", {}) or {}),
            "spread_convention": slippage_block.get("spread_convention", "half_spread"),
        }
    else:
        cost_spec = {
            "model": "percentage",
            "commission_bps": float(commission_block["rate"]) * 10_000.0,
            "slippage_bps": float(slippage_block["rate"]) * 10_000.0,
        }

    if rebal_spec["mode"] == "vectorized":
        if funding_rates is not None:
            raise ValueError("vectorized backtests cannot execute funding cashflows")
        if (
            "_state_transition" in weights.columns
            and weights.filter(pl.col("_state_transition")).height
        ):
            raise ValueError("vectorized backtests cannot sequence state transitions")
        # sp500_options HTM short-straddle uses a dedicated multi-cohort daily-MTM
        # backtest path: overlapping 5-cohort book, per-cohort daily premium + hedge
        # P&L, entry-spread + hedge-rebalance transaction costs. The simple
        # weights × y_true vectorized path cannot express this strategy because
        # y_true is a single 30-day return, not a daily P&L series.
        if case_study == "sp500_options" and label == "ret_to_expiry":
            decision_kind = (strategy_spec.get("decision_artifact") or {}).get("kind")
            result = _run_htm_daily_mtm(
                case_study=case_study,
                predictions=predictions,
                signal_config=signal_config,
                initial_cash=initial_cash,
                risk_spec=strategy.get("risk", {}),
                allocation_spec=strategy.get("allocation", {}),
                label=label,
                prediction_hash=prediction_hash,
                option_decisions=weights if decision_kind == "short_straddles" else None,
                option_lifecycle=option_lifecycle,
                option_accounting=strategy_spec.get("options_accounting"),
            )
        else:
            result = _run_vectorized(
                weights=weights,
                predictions=predictions,
                prices=prices,
                cost_spec=cost_spec,
                cadence=rebal_spec.get("cadence", ""),
                label=label,
                case_study=case_study,
                initial_cash=initial_cash,
                risk_spec=strategy.get("risk", {}),
                prediction_hash=prediction_hash,
            )
    else:
        result = _run_engine(
            weights=weights,
            prices=prices,
            predictions=predictions,
            strategy_spec=strategy_spec,
            rebalance_spec=rebal_spec,
            risk_spec=strategy.get("risk", {}),
            allow_short=allow_short,
            initial_cash=initial_cash,
            calendar=calendar,
            contract_specs=contract_specs,
            case_study=case_study,
            label=label,
            funding_rates=funding_rates,
        )

    # Build metrics dict
    metrics = result["metrics"]

    # Build daily returns DataFrame
    daily_returns = result["daily_returns"]

    # Extract trade log, fills, equity, portfolio state from engine result
    # (all None for vectorized path)
    trades_df = result.get("trades_df")
    fills_df = result.get("fills_df")
    equity_df = result.get("equity_df")
    portfolio_state_df = result.get("portfolio_state_df")

    # 3. Register
    backtest_hash = None
    if register:
        from case_studies.utils.registry import (
            compute_backtest_fold_metrics,
            register_backtest_fold_metrics,
            register_backtest_run,
        )

        _bt_elapsed_s = time.perf_counter() - _bt_t0
        backtest_hash = register_backtest_run(
            case_study,
            prediction_hash,
            strategy_spec,
            returns=daily_returns,
            trades=trades_df,
            fills=fills_df,
            equity=equity_df,
            portfolio_state=portfolio_state_df,
            weights=weights,
            metrics=metrics,
            started_at=_bt_started_at,
            elapsed_s=_bt_elapsed_s,
        )

        # Compute and register per-fold backtest metrics.
        # Annualize at the frequency of the *return series*, not the rebalance
        # cadence. `daily_returns` is one row per session regardless of how often
        # the strategy trades, so a monthly-rebalanced strategy on daily returns
        # was being annualized at 12 instead of 252. Sharpe, Sortino and
        # volatility scale with the square root of the factor, so those came out
        # low by exactly sqrt(252/12) = 4.583. CAGR carries the factor in a
        # compounding exponent rather than a multiplier, so its error is not a
        # fixed ratio, and Calmar inherits whatever CAGR did because it divides
        # CAGR by max drawdown. total_return and max_drawdown are unaffected.
        # Same resolver and same frame as the overall metrics, so the two agree.
        fold_metrics = compute_backtest_fold_metrics(
            daily_returns,
            case_study,
            label=label,
            periods_per_year=resolve_periods_per_year(case_study, daily_returns),
        )
        if fold_metrics:
            register_backtest_fold_metrics(case_study, backtest_hash, fold_metrics)

    return BacktestRunResult(
        daily_returns=daily_returns,
        metrics=metrics,
        strategy_spec=strategy_spec,
        prediction_hash=prediction_hash,
        backtest_hash=backtest_hash,
        engine_result=result.get("engine_result"),
        weights=weights,
        execution_mode=rebal_spec["mode"],
    )


# ---------------------------------------------------------------------------
# Engine path
# ---------------------------------------------------------------------------


def _run_engine(
    weights: pl.DataFrame,
    prices: pl.DataFrame,
    predictions: pl.DataFrame,
    strategy_spec: dict,
    rebalance_spec: dict,
    risk_spec: dict,
    allow_short: bool,
    initial_cash: float,
    calendar: str,
    contract_specs: dict | None = None,
    *,
    case_study: str | None = None,
    label: str | None = None,
    funding_rates: pl.DataFrame | None = None,
) -> dict:
    """Run backtest via ml4t-backtest Engine."""
    from ml4t.backtest import DataFeed, Engine, RebalanceConfig, Strategy, TargetWeightExecutor

    from case_studies.utils.backtest_loaders import (
        extract_daily_returns_frame,
        infer_session_alignment,
    )

    config = runtime_backtest_config(strategy_spec)
    profile_rebalance_mode = config.rebalance_mode

    # Session enforcement — drop bars outside trading sessions (e.g., CME
    # Saturdays). Idempotent with the same mutation applied in
    # ``ensure_backtest_spec``; kept here as belt-and-suspenders so the
    # engine always sees the right value even if a spec is passed in raw.
    apply_calendar_session_enforcement(config, calendar)

    # Pre-compute weight dict from DataFrame
    price_timestamp_dtype = prices.schema["timestamp"]
    feed_is_date = price_timestamp_dtype == pl.Date
    feed_timezone = (
        price_timestamp_dtype.time_zone if isinstance(price_timestamp_dtype, pl.Datetime) else None
    )
    raw_weight_dict = _target_weights_by_timestamp(weights)
    weight_dict = {
        _engine_timestamp(
            timestamp,
            feed_is_date=feed_is_date,
            feed_timezone=feed_timezone,
            configured_timezone=config.resolved_timezone,
        ): targets
        for timestamp, targets in raw_weight_dict.items()
    }

    # Resolve calendar-aware rebalance schedule, then thin by the label's
    # non-overlapping step from setup.yaml::labels.rebalance_step. Mirrors
    # the same two-step thinning that thin_to_rebalance_dates() applies on
    # the vectorized path (see backtest_loaders.thin_to_rebalance_dates).
    # Without this, multi-step labels (e.g. fwd_ret_60m on a 15m cadence
    # with step=4) over-rebalance by step×.
    #
    # The schedule is derived from the canonical *prediction* timeline, not
    # from weight_dict.keys(). Allocation-class methods (score_weighted,
    # HRP, MVO, inverse_vol, risk_parity) pre-thin to non-overlapping
    # rebalance dates inside _apply_allocation via thin_to_rebalance_dates;
    # if we resolved the schedule from those already-sparse weight keys and
    # applied gather_every(step) again, we'd thin by step² and trade
    # ~step× too rarely. The on_data callback already gates on
    # ``timestamp in weight_dict``, so dates without weights are skipped.
    from case_studies.utils.backtest_loaders import (
        get_rebalance_step,
        resolve_rebalance_timestamps,
    )

    cadence = rebalance_spec.get("cadence", "monthly_month_end")
    all_pred_ts = pl.Series("ts", predictions["timestamp"].unique().sort().to_list())
    schedule_dates = resolve_rebalance_timestamps(all_pred_ts, cadence, calendar)
    if case_study and label:
        step = get_rebalance_step(case_study, label)
        if step > 1:
            schedule_dates = schedule_dates.gather_every(step)
    rebalance_schedule = {
        _engine_timestamp(
            timestamp,
            feed_is_date=feed_is_date,
            feed_timezone=feed_timezone,
            configured_timezone=config.resolved_timezone,
        )
        for timestamp in schedule_dates.to_list()
    }
    transition_timestamps = {
        _engine_timestamp(
            timestamp,
            feed_is_date=feed_is_date,
            feed_timezone=feed_timezone,
            configured_timezone=config.resolved_timezone,
        )
        for timestamp in (
            weights.filter(pl.col("_state_transition")).get_column("timestamp").unique().to_list()
            if "_state_transition" in weights.columns
            else []
        )
    }
    rebalance_schedule.update(transition_timestamps)
    if transition_timestamps and config.execution_mode.value != "same_bar":
        raise ValueError("state-transition sequencing requires same-bar engine execution")

    # Build risk components from spec (Ch19)
    position_rules = _build_position_rules(risk_spec)
    risk_manager = _build_risk_manager(risk_spec, initial_cash)

    # Rebalance thresholds are sourced from setup.yaml::backtest.rebalance and
    # always present in the canonical strategy.rebalance block (populated by
    # ensure_backtest_spec()).
    min_weight_change = float(rebalance_spec["min_weight_change"])
    min_trade_value = float(rebalance_spec["min_trade_value"])

    funding_ledger = None
    if funding_rates is not None:
        from case_studies.crypto_perps_funding.funding_backtest import FundingSettlementLedger

        funding_ledger = FundingSettlementLedger(funding_rates)

    # Build strategy
    class _PrecomputedStrategy(Strategy):
        def __init__(self):
            self._rules_set = False
            self.executor = TargetWeightExecutor(
                config=RebalanceConfig(
                    min_trade_value=min_trade_value,
                    min_weight_change=min_weight_change,
                    allow_fractional=None,  # Defer to broker.share_type (profile)
                    allow_short=allow_short,
                    rebalance_mode=profile_rebalance_mode,
                )
            )

        def on_start(self, broker):
            if funding_ledger is not None:
                funding_ledger.install(broker)

        def on_data(self, timestamp, data, context, broker):
            # Set position rules on broker (once, first bar)
            if not self._rules_set:
                if position_rules:
                    broker.set_position_rules(position_rules)
                self._rules_set = True

            if timestamp in transition_timestamps:
                broker.flatten_all_positions(reason="declared state transition")
                broker._process_orders()

            # Check portfolio-level limits (each bar)
            if risk_manager:
                positions = {a: p.market_value for a, p in broker.positions.items()}
                risk_results = risk_manager.update(
                    equity=broker.get_account_value(),
                    positions=positions,
                    timestamp=timestamp,
                    broker=broker,
                )
                # Two guards on purpose: the liquidate check catches a bar
                # where the manager flattened but left is_halted=False, while
                # is_halted catches a prior-bar halt; neither subsumes the other.
                if any(result.action == "liquidate" for result in risk_results):
                    return
                if risk_manager.is_halted:
                    return

            # Calendar-aware schedule: only rebalance on resolved dates
            if timestamp not in rebalance_schedule:
                return

            if timestamp in weight_dict:
                targets = {a: w for a, w in weight_dict[timestamp].items() if a in data}
                if targets:
                    self.executor.execute(targets, data, broker)

    # Resolve the canonical (cs, label, split) window — same window for every
    # strategy on the same (cs, label, split). Callers pre-window `prices` via
    # load_backtest_prices_for(cs, label, split=...) so the parquet read is
    # row-group-pruned; the engine asserts the price range stays within the
    # canonical window. Falls back to predictions.min/max only when no
    # canonical window can be derived (label without CV folds, sentinel
    # prediction_hash, etc.).
    from case_studies.utils.cv_window import canonical_window, lookup_split

    prices_ts_dtype = prices.schema["timestamp"]
    window = None
    if case_study and label:
        prediction_hash = (
            strategy_spec.get("backtest_config", {}).get("metadata", {}).get("prediction_hash")
        )
        split = lookup_split(case_study, prediction_hash) if prediction_hash else None
        if split is not None:
            window = canonical_window(case_study, label, split=split)
        # If split is unknown (no prediction_hash in metadata or unrecognized
        # split label) we deliberately fall through to the predictions.min/max
        # branch below rather than silently mis-windowing a holdout backtest
        # against the validation window.

    if window is not None:
        win_start, win_end = window
        # Compare on the date component so calendar-edge drift (parquet starts
        # 2024-01-02 when win_start=2024-01-01 because Jan 1 is a holiday) is
        # tolerated. The upper-bound assertion fires when prices EXTEND past
        # the canonical window — i.e. the caller forgot to pre-window the
        # right edge. The lower bound is intentionally NOT asserted: callers
        # may load earlier-than-canonical prefix history when a rolling-vol
        # allocator (inverse_vol / risk_parity / hrp / mvo_ledoit_wolf) needs
        # warmup so the first rebalance has data-driven (not median-imputed)
        # weights. The daily_returns frame is sliced to [win_start, win_end]
        # below regardless of how wide the load was.
        prices_dates = prices["timestamp"].dt.date()
        prices_min_date = cast(date | None, prices_dates.min())
        prices_max_date = cast(date | None, prices_dates.max())
        if prices_min_date is None or prices_max_date is None:
            raise RuntimeError(
                f"Empty prices frame for cs={case_study} label={label} "
                f"split={split} — canonical window [{win_start}, {win_end}]."
            )
        if prices_max_date > win_end:
            raise AssertionError(
                f"Prices not pre-windowed for cs={case_study} label={label} "
                f"split={split}: canonical window [{win_start}, {win_end}], "
                f"prices range [{prices_min_date}, {prices_max_date}] — "
                f"upper bound exceeded. Pass end_date to load_backtest_prices() "
                f"or call load_backtest_prices_for(cs, label, split=split)."
            )
    elif predictions.height > 0:
        # Fallback when canonical window unavailable: still slice to the
        # predictions' span so demo notebooks with sentinel prediction_hash
        # don't process pre-history.
        pred_ts = predictions["timestamp"]
        if pred_ts.dtype != prices_ts_dtype:
            pred_ts = pred_ts.cast(prices_ts_dtype)
        prices = prices.filter(
            (pl.col("timestamp") >= pred_ts.min()) & (pl.col("timestamp") <= pred_ts.max())
        )

    # signals_df is intentionally omitted: _PrecomputedStrategy reads
    # weight_dict directly, so routing predictions through the bar iterator
    # would waste hot-path memory.
    feed = DataFeed(prices_df=prices, feed_spec=config.feed_spec)
    strategy = _PrecomputedStrategy()
    engine = Engine.from_config(feed, strategy, config, contract_specs=contract_specs)
    engine_result = engine.run()

    funding_metrics = funding_ledger.metrics() if funding_ledger is not None else {}

    # Extract daily returns
    session_aligned = infer_session_alignment(calendar)
    daily_df = extract_daily_returns_frame(
        engine_result,
        calendar=calendar,
        session_aligned=session_aligned,
    )

    # Slice the persisted daily-returns frame to the canonical (cs, label,
    # split) window — same window as the price-trim above, so every
    # (cs, label, split) produces a daily_returns parquet covering the same
    # dates regardless of which strategy was run. Date-component compare so
    # intraday bars on win_end aren't dropped by midnight promotion.
    if window is not None:
        daily_df = daily_df.filter(
            (pl.col("timestamp").dt.date() >= window[0])
            & (pl.col("timestamp").dt.date() <= window[1])
        )
    returns_arr = daily_df["daily_return"].to_numpy()

    ppy = overall_periods_per_year(case_study, calendar, daily_df)
    metrics = compute_portfolio_metrics(returns_arr, periods_per_year=ppy, trim_leading_zeros=False)
    metrics.update(funding_metrics)

    # Engine-specific metrics (execution details not derivable from returns)
    m = engine_result.metrics
    metrics["num_trades"] = m.get("num_trades", 0)
    metrics["total_commission"] = m.get("total_commission", 0.0)
    metrics["total_slippage"] = m.get("total_slippage", 0.0)

    # avg_turnover: target-weight semantics (sum_i |Δw_i| averaged over the daily
    # timeline, 0 on non-rebalance days). Same formula as the vectorized path so
    # the registry column has consistent meaning across both engines. Skipping the
    # engine's own `m["avg_turnover"]` (notional/equity) — that value is unbounded
    # for leveraged products (cme_futures multipliers inflate it 10⁴–10⁵×) and
    # mixes incompatibly with vectorized-path rows on the same column.
    if weights.height > 0:
        turnover_weights = weights.with_columns(pl.lit(1).alias("_event_order"))
        if "_state_transition" in weights.columns:
            flat_states = []
            transition_timestamps = (
                weights.filter(pl.col("_state_transition"))
                .get_column("timestamp")
                .unique()
                .sort()
                .to_list()
            )
            for transition_timestamp in transition_timestamps:
                previous_state = (
                    weights.filter(pl.col("timestamp") < transition_timestamp)
                    .sort("symbol", "timestamp")
                    .group_by("symbol", maintain_order=True)
                    .last()
                )
                if previous_state.height > 0:
                    flat_states.append(
                        previous_state.with_columns(
                            pl.lit(transition_timestamp)
                            .cast(weights.schema["timestamp"])
                            .alias("timestamp"),
                            pl.lit(0.0).cast(weights.schema["weight"]).alias("weight"),
                            pl.lit(0).alias("_event_order"),
                        ).select(turnover_weights.columns)
                    )
            if flat_states:
                turnover_weights = pl.concat([turnover_weights, *flat_states])
        weights_sorted = turnover_weights.sort("symbol", "timestamp", "_event_order").with_columns(
            abs_change=(
                pl.col("weight") - pl.col("weight").shift(1).over("symbol").fill_null(0.0)
            ).abs(),
        )
        turnover_by_ts = (
            weights_sorted.with_columns(pl.col("timestamp").cast(daily_df.schema["timestamp"]))
            .group_by("timestamp")
            .agg(turnover=pl.col("abs_change").sum())
        )
        # Align to daily timeline so non-rebalance days contribute 0 to the mean
        # (matches port_ret.join(turnover) in the vectorized path).
        turnover_aligned = daily_df.join(
            turnover_by_ts,
            on="timestamp",
            how="left",
        ).with_columns(pl.col("turnover").fill_null(0.0))
        mean_turnover = cast(float | None, turnover_aligned["turnover"].mean())
        metrics["avg_turnover"] = float(mean_turnover) if mean_turnover is not None else 0.0
    else:
        metrics["avg_turnover"] = 0.0

    # Extract trade log
    trades_df = None
    if engine_result.trades:
        try:
            trades_df = engine_result.to_trades_dataframe()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("Trade extraction failed: %s", e)

    # Extract fill-level execution records (quote-aware since backtest b11)
    fills_df = None
    if engine_result.fills:
        try:
            fills_df = engine_result.to_fills_dataframe()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("Fills extraction failed: %s", e)

    # Extract equity curve and portfolio state (bar-level resolution)
    equity_df = None
    portfolio_state_df = None
    try:
        equity_df = engine_result.to_equity_dataframe()
        portfolio_state_df = engine_result.to_portfolio_state_dataframe()
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("Equity/portfolio state extraction failed: %s", e)

    return {
        "daily_returns": daily_df,
        "metrics": metrics,
        "engine_result": engine_result,
        "trades_df": trades_df,
        "fills_df": fills_df,
        "equity_df": equity_df,
        "portfolio_state_df": portfolio_state_df,
    }


# ---------------------------------------------------------------------------
# Hold-to-expiry daily-MTM path (sp500_options / ret_to_expiry)
# ---------------------------------------------------------------------------


def _run_htm_daily_mtm(
    case_study: str,
    predictions: pl.DataFrame,
    signal_config: dict,
    initial_cash: float,
    risk_spec: dict | None = None,
    allocation_spec: dict | None = None,
    label: str | None = None,
    prediction_hash: str | None = None,
    option_decisions: pl.DataFrame | None = None,
    option_lifecycle: pl.DataFrame | None = None,
    option_accounting: dict | None = None,
) -> dict:
    """Dispatch wrapper for the hold-to-expiry daily-MTM short-straddle backtest.

    Delegates to ``case_studies.sp500_options._htm_backtest.run_htm_daily_mtm``,
    which implements the multi-cohort daily-MTM accounting:

    - Friday entry of top-K short straddles, ~30-day DTE.
    - Daily delta hedge via the underlying stock (threshold rehedging).
    - Cash-settle at expiry (no market exit, no exit bid-ask).
    - Full transaction costs: entry option spread (bid-ask on both legs) on
      cohort entry day; hedge-trade spread + equity commission on every
      hedge rebalance day.
    - Book size = 5 concurrent cohorts × 1/5 capital each (fully invested).

    The entry-and-weighting scheme is read from ``signal_config`` (same shape
    as the vectorized signal dispatcher): ``method`` + ``top_k`` / ``percentile``.

    Returns the same shape as ``_run_vectorized``: ``{daily_returns, metrics}``
    where ``daily_returns`` has columns ``[timestamp, daily_return]`` so the
    registry write path treats it identically to any other backtest.
    """
    from case_studies.sp500_options._htm_backtest import run_htm_daily_mtm
    from utils import CASE_STUDIES_DIR, ML4T_DATA_PATH

    cs_dir = CASE_STUDIES_DIR / case_study
    labels_dir = cs_dir / "labels"
    raw_options_dir = ML4T_DATA_PATH / "equities" / "market" / "sp500" / "options_straddles_raw"

    method = str(signal_config.get("method", "equal_weight_top_k"))
    top_k = int(signal_config.get("top_k", 20))
    percentile = float(signal_config.get("percentile", 90.0))
    if risk_spec:
        raise ValueError("the specialized option path does not support risk overlays")
    required_accounting = {
        "n_roll",
        "delta_hedge",
        "delta_threshold",
        "hedge_spread_bps",
        "equity_commission_per_share",
        "option_commission_per_contract",
        "option_contract_multiplier",
        "option_spread_fraction",
        "exit_at_max_days",
    }
    missing_accounting = required_accounting - set(option_accounting or {})
    if missing_accounting:
        raise ValueError(
            f"specialized option accounting is missing fields: {sorted(missing_accounting)}"
        )
    assert option_accounting is not None

    result = run_htm_daily_mtm(
        case_study=case_study,
        predictions=predictions,
        labels_dir=labels_dir,
        raw_options_dir=raw_options_dir,
        method=method,
        top_k=top_k,
        percentile=percentile,
        exit_at_max_days=option_accounting["exit_at_max_days"],
        n_roll=int(option_accounting["n_roll"]),
        delta_hedge=bool(option_accounting["delta_hedge"]),
        delta_threshold=float(option_accounting["delta_threshold"]),
        hedge_spread_bps=float(option_accounting["hedge_spread_bps"]),
        equity_commission_per_share=float(option_accounting["equity_commission_per_share"]),
        option_commission_per_contract=float(option_accounting["option_commission_per_contract"]),
        option_contract_multiplier=int(option_accounting["option_contract_multiplier"]),
        allocation_spec=allocation_spec,
        decisions=option_decisions,
        option_lifecycle=option_lifecycle,
        option_spread_fraction=float(option_accounting["option_spread_fraction"]),
    )
    port = result["daily_returns"]
    metrics = result["metrics"]

    # Slice port to canonical (cs, label, split) window so daily_returns and
    # the aux cost-accounting metrics (cumulative_entry_cost, n_rebalance_dates,
    # etc.) all reflect the same canonical window. Mirrors _run_engine and
    # _run_vectorized — same drifting-parquet bug otherwise. The cohort fields
    # (entry_cost_day, n_open, etc.) are filtered with the same `date` slice
    # because the multi-cohort daily-MTM book emits one row per holding date.
    sliced = False
    if prediction_hash and case_study and label:
        from case_studies.utils.cv_window import canonical_window, lookup_split

        split = lookup_split(case_study, prediction_hash)
        if split is not None:
            window = canonical_window(case_study, label, split=split)
            if window is not None:
                win_start, win_end = window
                port_filtered = port.filter(
                    (pl.col("date").cast(pl.Date) >= win_start)
                    & (pl.col("date").cast(pl.Date) <= win_end)
                )
                if port_filtered.is_empty():
                    raise RuntimeError(
                        f"Canonical window [{win_start}, {win_end}] for "
                        f"cs={case_study} label={label} split={split} produced "
                        f"empty port (HTM daily-MTM; port span "
                        f"{port['date'].min()} → {port['date'].max()})."
                    )
                if port_filtered.height != port.height:
                    sliced = True
                port = port_filtered

    # Shape the return like _run_vectorized so the registry writer is agnostic.
    daily_returns = port.select(
        pl.col("date").cast(pl.Datetime("us")).alias("timestamp"),
        pl.col("portfolio_ret").alias("daily_return"),
    )

    # When the canonical-window slice actually trimmed rows, recompute the
    # returns-based metric set so the registry Sharpe/CAGR/volatility/etc.
    # reflect the sliced daily_returns rather than the inner function's
    # pre-slice values. Aux cohort metrics (cumulative_entry_cost, etc.) are
    # recomputed from sliced port below regardless of slice.
    if sliced:
        from case_studies.sp500_options._htm_backtest import _compute_metrics

        metrics.update(_compute_metrics(port))

    # Final unified metric pass: replace HTM-internal Sharpe/Sortino/etc. with
    # the canonical ml4t.diagnostic.PortfolioAnalysis values so HTM metrics are
    # comparable to engine/vectorized paths AND include the uncertainty bands
    # (sharpe_se_lo, sharpe_ci95_lo/hi, sortino_ci95_*, ann_return_hac_se +
    # ci95, max_dd_ci95_*, calmar_ci95_*, psr_pvalue, bootstrap_block_length/n).
    # HTM uses daily MTM on NYSE sessions → periods_per_year = 252. Operates on
    # the FINAL daily_returns (post-slice, post-risk-overlay) so the persisted
    # parquet and the registered metrics are derived from the same series.
    returns_arr = daily_returns["daily_return"].to_numpy()
    metrics.update(
        compute_portfolio_metrics(
            returns_arr,
            periods_per_year=252,
            case_study=case_study,
            label=label,
            uncertainty=True,
        )
    )

    # Number of distinct rebalance events (= entry days with any new cohort).
    # `n_open.sum()` is the count of cohort-days, kept under a distinct key.
    metrics["n_rebalance_dates"] = int((port["entry_cost_day"] > 0).sum())
    metrics["cohort_days_open"] = int(port["n_open"].sum())
    metrics["avg_cohorts_open"] = float(port["n_open"].mean())
    metrics["cumulative_entry_cost"] = float(port["entry_cost_day"].sum())
    metrics["cumulative_hedge_cost"] = float(port["hedge_cost_day"].sum())
    if "exit_cost_day" in port.columns:
        metrics["cumulative_exit_cost"] = float(port["exit_cost_day"].sum())

    return {
        "daily_returns": daily_returns,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Vectorized path (for 3 special case studies)
# ---------------------------------------------------------------------------


def _run_vectorized(
    weights: pl.DataFrame,
    predictions: pl.DataFrame,
    prices: pl.DataFrame,
    cost_spec: dict,
    cadence: str,
    label: str,
    case_study: str,
    initial_cash: float,
    risk_spec: dict | None = None,
    prediction_hash: str | None = None,
) -> dict:
    """Run vectorized backtest (weight × forward return - costs).

    Used for us_firm_characteristics, sp500_options, nasdaq100_microstructure.

    Cost dispatch supports two models:
      * percentage — fractional drag = turnover × (commission_bps + slippage_bps) / 1e4
      * per_share_plus_spread — fractional drag = sum_i(|Δw_i| × (per_share + half_spread_i) / price_i),
        which models per-share commission and per-asset half-spread slippage. The
        |Δshares_i| × cost_per_share_i / NAV identity reduces to the form above
        because |Δshares_i| = |Δw_i| × NAV / price_i and NAV cancels.

    Portfolio-level risk overlays are applied post-hoc via
    ``_apply_vectorized_risk``. Only ``max_drawdown`` is supported — it models
    an intraday exit at the threshold with explicit slippage. ``daily_loss``
    is refused on this path: an honest per-bar halt requires intraday
    position tracking that the close-to-close return series cannot express.
    Position-level rules (stop-loss, trailing stop) likewise cannot be
    applied in vectorized mode.
    """
    from case_studies.utils.backtest_loaders import get_rebalance_step, thin_to_rebalance_dates

    # Thin predictions to non-overlapping periods. Step is declared per-label
    # in the case study's setup.yaml under labels.rebalance_step.
    step = get_rebalance_step(case_study, label)
    thinned = thin_to_rebalance_dates(predictions, cadence=cadence, step=step)

    # Re-compute weights on thinned predictions
    # (The weights were computed on full predictions; we need to recompute
    # or filter to thinned timestamps)
    rebalance_dates = thinned["timestamp"].unique()
    # Semi-join to filter — avoids Polars is_in precision mismatch
    rebal_df = pl.DataFrame({"timestamp": rebalance_dates})
    if rebal_df["timestamp"].dtype != weights["timestamp"].dtype:
        rebal_df = rebal_df.cast({"timestamp": weights["timestamp"].dtype})
    weights_thinned = weights.join(rebal_df, on="timestamp", how="semi")

    # Harmonize timestamp dtypes before join
    thinned_sel = thinned.select(["timestamp", "symbol", "y_true"])
    if weights_thinned["timestamp"].dtype != thinned_sel["timestamp"].dtype:
        thinned_sel = thinned_sel.cast({"timestamp": weights_thinned["timestamp"].dtype})

    # Join weights with forward returns
    bt = weights_thinned.join(
        thinned_sel,
        on=["timestamp", "symbol"],
        how="inner",
    )

    # Portfolio returns per period
    port_ret = (
        bt.group_by("timestamp")
        .agg(
            gross_ret=(pl.col("weight") * pl.col("y_true")).sum(),
            n_positions=pl.len(),
        )
        .sort("timestamp")
    )

    # Compute per-symbol weight changes and aggregate turnover for diagnostics
    weights_sorted = weights_thinned.sort("timestamp", "symbol").with_columns(
        abs_change=(
            pl.col("weight") - pl.col("weight").shift(1).over("symbol").fill_null(0.0)
        ).abs(),
    )
    turnover = weights_sorted.group_by("timestamp").agg(turnover=pl.col("abs_change").sum())

    port_ret = port_ret.join(turnover, on="timestamp", how="left").with_columns(
        pl.col("turnover").fill_null(0.0)
    )

    # Apply costs — dispatch on cost_spec.model
    cost_model = cost_spec.get("model", "percentage")
    if cost_model == "per_share_plus_spread":
        per_share = float(cost_spec["per_share"])
        default_hs = float(cost_spec.get("default_half_spread_usd", 0.0))
        asset_spreads = cost_spec.get("asset_spreads", {}) or {}

        if "close" in prices.columns:
            price_col = "close"
        elif "mid" in prices.columns:
            price_col = "mid"
        else:
            raise ValueError(
                "per_share_plus_spread cost model requires a 'close' or 'mid' "
                f"column on the prices frame; got columns={list(prices.columns)}"
            )
        prices_sel = prices.select(["timestamp", "symbol", pl.col(price_col).alias("_px")])
        if prices_sel["timestamp"].dtype != weights_sorted["timestamp"].dtype:
            prices_sel = prices_sel.cast({"timestamp": weights_sorted["timestamp"].dtype})

        wc_priced = weights_sorted.join(prices_sel, on=["timestamp", "symbol"], how="left")
        # Per-asset half-spread map; default fallback for symbols not in map
        if asset_spreads:
            wc_priced = wc_priced.with_columns(
                _hs=pl.col("symbol").replace_strict(
                    asset_spreads, default=default_hs, return_dtype=pl.Float64
                )
            )
        else:
            wc_priced = wc_priced.with_columns(_hs=pl.lit(default_hs, dtype=pl.Float64))

        # Fractional cost drag per period: sum_i(|Δw_i| × (per_share + hs_i) / price_i)
        # Skip rows where price is null (symbol not in prices frame); they contribute 0.
        cost_drag = (
            wc_priced.with_columns(
                _drag=pl.when(pl.col("_px").is_not_null() & (pl.col("_px") > 0))
                .then(pl.col("abs_change") * (per_share + pl.col("_hs")) / pl.col("_px"))
                .otherwise(0.0)
            )
            .group_by("timestamp")
            .agg(cost_drag=pl.col("_drag").sum())
        )
        port_ret = port_ret.join(cost_drag, on="timestamp", how="left").with_columns(
            pl.col("cost_drag").fill_null(0.0),
            net_ret=pl.col("gross_ret") - pl.col("cost_drag"),
        )
    else:
        cost_rate = (
            float(cost_spec.get("commission_bps", 0.0)) + float(cost_spec.get("slippage_bps", 0.0))
        ) / 10_000
        port_ret = port_ret.with_columns(
            net_ret=pl.col("gross_ret") - pl.col("turnover") * cost_rate,
        )

    # Slice port_ret to canonical (cs, label, split) window so every strategy
    # on the same (cs, label, split) produces a daily_returns parquet covering
    # the same dates regardless of which predictions span which dates. Mirrors
    # _run_engine slice at lines 740-786 + 805-816. Without this, vectorized
    # daily_returns drift by the prediction window's left/right edges and
    # cross-config comparisons aren't apples-to-apples. Slice happens BEFORE
    # the risk overlay so the drawdown breaker only fires on canonical-window
    # losses (not on stale pre-canonical drawdowns).
    if prediction_hash and case_study and label:
        from case_studies.utils.cv_window import canonical_window, lookup_split

        split = lookup_split(case_study, prediction_hash)
        if split is not None:
            window = canonical_window(case_study, label, split=split)
            if window is not None:
                win_start, win_end = window
                port_ret_filtered = port_ret.filter(
                    (pl.col("timestamp").cast(pl.Date) >= win_start)
                    & (pl.col("timestamp").cast(pl.Date) <= win_end)
                )
                if port_ret_filtered.is_empty():
                    raise RuntimeError(
                        f"Canonical window [{win_start}, {win_end}] for "
                        f"cs={case_study} label={label} split={split} produced "
                        f"empty port_ret (vectorized path; port_ret span "
                        f"{port_ret['timestamp'].min()} → {port_ret['timestamp'].max()})."
                    )
                port_ret = port_ret_filtered

    # Apply portfolio-level risk overlays (post-hoc on return series)
    if risk_spec:
        port_ret = _apply_vectorized_risk(port_ret, risk_spec)

    # Daily returns DataFrame
    daily_returns = port_ret.select(
        pl.col("timestamp"),
        pl.col("net_ret").alias("daily_return"),
    )

    # Portfolio metrics via ml4t-diagnostic
    returns_arr = daily_returns["daily_return"].to_numpy()
    n = len(returns_arr)

    # Annualize at the frequency of the grid this function writes, which is
    # neither the rebalance cadence nor always the declared value. This path
    # thins predictions to rebalance dates first, so its grid can be coarser
    # than the registered convention. `resolve_periods_per_year` handles that
    # reconciliation, and `run_backtest` calls it on this same frame for the
    # per-fold metrics, so overall and fold annualization cannot diverge.
    periods_per_year = resolve_periods_per_year(case_study, daily_returns)

    metrics = compute_portfolio_metrics(returns_arr, periods_per_year=periods_per_year or 252)

    # Vectorized-specific metrics (not derivable from returns alone)
    avg_turnover = cast(float, port_ret["turnover"].mean()) if n > 0 else 0.0
    metrics["avg_turnover"] = avg_turnover
    metrics["n_periods"] = n

    return {
        "daily_returns": daily_returns,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Vectorized risk overlays (Ch19) — portfolio-level limits only
# ---------------------------------------------------------------------------


def _apply_vectorized_risk(port_ret: pl.DataFrame, risk_spec: dict) -> pl.DataFrame:
    """Apply portfolio-level risk limits to a close-to-close return series.

    Used by the vectorized + HTM dispatch paths (us_firm_characteristics,
    sp500_options/ret_to_expiry). Engine-path case studies use
    ``ml4t.backtest.risk.RiskManager`` via ``_build_risk_manager`` and never
    enter this function.

    Supported limits:
        ``max_drawdown``: model an intraday exit at the drawdown threshold.
        Find the first close where cumulative drawdown crosses ``-threshold``;
        replace that bar's return with the equity move from the prior close
        down to ``peak * (1 - threshold) * (1 - breach_slippage)``; zero every
        subsequent bar. Default ``breach_slippage`` = 50 bps; configurable
        via ``risk_spec['breach_slippage']``.

    Refused (raises ``ValueError``):
        ``daily_loss``: a vectorized close-to-close series cannot implement
        a per-bar daily-loss halt without lookahead — zeroing the breach
        bar's loss while keeping every winning bar inflates Sharpe to
        infinity in the limit. Use ``max_drawdown`` or move the CS to the
        engine path (which has proper ``DailyLossLimit`` halt-on-update
        semantics through ``ml4t.backtest.risk``).
    """
    limits = risk_spec.get("portfolio_limits", [])
    if not limits:
        return port_ret

    dd_threshold = None
    for lc in limits:
        ltype = lc["type"]
        if ltype == "max_drawdown":
            dd_threshold = lc["threshold"]
        elif ltype == "daily_loss":
            raise ValueError(
                "daily_loss portfolio limit is not supported on the "
                "vectorized/HTM path: the only honest implementation needs "
                "intraday position tracking (engine path's "
                "ml4t.backtest.risk.DailyLossLimit). Drop it from the sweep "
                "config or move the case study to the engine path."
            )

    returns = port_ret["net_ret"].to_numpy().copy()

    if dd_threshold is not None:
        breach_slippage = float(risk_spec.get("breach_slippage", 0.005))
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        drawdowns = cum / peak - 1.0
        breach_idx = np.where(drawdowns < -abs(dd_threshold))[0]
        if len(breach_idx) > 0:
            i = int(breach_idx[0])
            prior_eq = float(cum[i - 1]) if i > 0 else 1.0
            # Exit at peak * (1 - threshold), then take breach_slippage on the
            # exit. Equity at exit = peak[i] * (1 - threshold) * (1 - slip).
            exit_eq = float(peak[i]) * (1.0 - abs(dd_threshold)) * (1.0 - breach_slippage)
            returns[i] = exit_eq / prior_eq - 1.0
            returns[i + 1 :] = 0.0

    return port_ret.with_columns(pl.Series("net_ret", returns))


# ---------------------------------------------------------------------------
# Allocation dispatch (Ch17)
# ---------------------------------------------------------------------------


def _apply_allocation(
    weights: pl.DataFrame,
    predictions: pl.DataFrame,
    prices: pl.DataFrame,
    alloc_spec: dict,
    *,
    cadence: str = "",
    label: str = "",
    case_study: str = "",
    prediction_hash: str | None = None,
    conformal_widths: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Post-process signal weights with an allocation method.

    Dispatches to utils.allocation functions based on alloc_spec["method"].
    The signal weights determine asset SELECTION (which assets are in the
    portfolio); the allocation method determines SIZING (how much weight
    each gets).
    """
    method = alloc_spec.get("method", "equal_weight")
    top_k = int(alloc_spec.get("top_k", weights["symbol"].n_unique()))
    long_short = bool(alloc_spec.get("long_short", False))

    if method == "equal_weight":
        return weights

    from case_studies.utils.allocation import (
        _cap_weights,
        compute_conformal_weights,
        compute_hrp_weights,
        compute_inverse_vol_weights,
        compute_mvo_weights,
        compute_risk_parity_weights,
    )

    # Harmonize timestamp + symbol dtypes before joins. us_firm predictions
    # carry symbol as UInt32 (stock_id) while prices use String; without this,
    # downstream is_in/join on symbol silently produces empty results. The
    # symbol cast is routed through ``_align_symbol_dtype`` so a real
    # ticker-vs-id mismatch surfaces with case-study context instead of an
    # opaque Polars ``InvalidOperationError``.
    ts_dtype = weights["timestamp"].dtype
    if predictions["timestamp"].dtype != ts_dtype:
        predictions = predictions.cast({"timestamp": ts_dtype})
    predictions = _align_symbol_dtype(
        weights,
        predictions,
        case_study=case_study,
        target_side="weights",
        other_side="predictions",
    )
    if prices["timestamp"].dtype != ts_dtype:
        prices = prices.cast({"timestamp": ts_dtype})
    prices = _align_symbol_dtype(
        weights,
        prices,
        case_study=case_study,
        target_side="weights",
        other_side="prices",
    )

    # Filter predictions to only the assets selected by the signal step
    selected_keys = weights.select(["timestamp", "symbol"]).unique()
    filtered_preds = predictions.join(selected_keys, on=["timestamp", "symbol"], how="inner")

    # Allocation only matters on actual rebalance dates. Without cadence-aware
    # thinning, covariance-based allocators solve the same optimization on every
    # prediction timestamp even when the engine only rebalances weekly/monthly.
    from case_studies.utils.backtest_loaders import get_rebalance_step, thin_to_rebalance_dates

    if not case_study or not label:
        raise ValueError(
            "_apply_allocation requires both case_study and label to look up "
            "labels.rebalance_step from setup.yaml. Pass them from the caller."
        )
    step = get_rebalance_step(case_study, label)
    rebal_preds = thin_to_rebalance_dates(filtered_preds, cadence=cadence, step=step)

    # Max weight cap — applied after all covariance-based allocators
    max_weight = float(alloc_spec.get("max_weight", 0.0))

    if method == "score_weighted":
        from case_studies.utils.signals import build_target_weights

        result = build_target_weights(
            rebal_preds,
            method="score_weighted_top_k",
            top_k=top_k,
            long_short=long_short,
        )
        if max_weight > 0:
            result = _cap_weights(result, max_weight)
        return result

    if method == "conformal_weighted":
        if not prediction_hash:
            raise ValueError(
                "conformal_weighted allocation requires prediction_hash; "
                "caller must pass it through _apply_allocation."
            )
        from case_studies.utils.conformal import load_conformal_widths

        alpha = float(alloc_spec.get("alpha", 0.20))
        min_calibration_n = int(alloc_spec.get("min_calibration_n", 30))
        calibration_version = str(alloc_spec.get("calibration_version", "walk_forward_v2"))
        widths = conformal_widths
        if widths is None:
            widths = load_conformal_widths(
                case_study,
                prediction_hash,
                alpha=alpha,
                min_calibration_n=min_calibration_n,
                calibration_version=calibration_version,
            )
        # Conformal widths are keyed by the timestamps stored in predictions.parquet,
        # which keep their original time zone; `normalize_prediction_columns` has
        # already made the prediction side tz-naive. Harmonize to the weights dtype
        # (as predictions and prices are above) or the join raises SchemaError on
        # every case study whose predictions are tz-aware, e.g. crypto_perps_funding.
        if widths["timestamp"].dtype != ts_dtype:
            widths = widths.cast({"timestamp": ts_dtype})
        supported_timestamps = widths.select("timestamp").unique()
        rebal_preds = rebal_preds.join(supported_timestamps, on="timestamp", how="inner")
        if rebal_preds.is_empty():
            raise ValueError(
                "conformal_weighted: no prediction timestamps have prior-only calibration"
            )
        floor_q = float(alloc_spec.get("floor_quantile", 0.01))
        result = compute_conformal_weights(
            rebal_preds,
            widths,
            top_k,
            long_short=long_short,
            floor_quantile=floor_q,
        )
        if max_weight > 0:
            result = _cap_weights(result, max_weight)
        return result

    vol_window = int(alloc_spec.get("vol_window", alloc_spec.get("lookback", 63)))

    if method == "inverse_vol":
        result = compute_inverse_vol_weights(
            rebal_preds, prices, top_k, vol_window=vol_window, long_short=long_short
        )
    elif method == "risk_parity":
        result = compute_risk_parity_weights(
            rebal_preds, prices, top_k, vol_window=vol_window, long_short=long_short
        )
    elif method in ("mvo", "mvo_ledoit_wolf"):
        lookback = int(alloc_spec.get("lookback", 126))
        mvo_max_weight = max_weight if max_weight > 0 else 1.0
        result = compute_mvo_weights(
            rebal_preds,
            prices,
            top_k,
            lookback=lookback,
            max_weight=mvo_max_weight,
            long_short=long_short,
        )
    elif method == "hrp":
        result = compute_hrp_weights(
            rebal_preds, prices, top_k, vol_window=vol_window, long_short=long_short
        )
    else:
        import logging

        logging.getLogger(__name__).warning(
            "Unknown allocation method '%s', returning signal weights", method
        )
        return weights

    if max_weight > 0:
        result = _cap_weights(result, max_weight)
    return result


# ---------------------------------------------------------------------------
# Risk rules (Ch19) — engine-level integration
# ---------------------------------------------------------------------------


def _build_position_rules(risk_spec: dict):
    """Create ml4t-backtest PositionRule objects from risk spec.

    Supports: stop_loss, trailing_stop, time_exit.
    Returns a RuleChain (multiple rules) or single rule, or None.
    """
    rules_config = risk_spec.get("position_rules", [])
    if not rules_config:
        return None

    from ml4t.backtest.risk import RuleChain, StopLoss, TimeExit, TrailingStop

    rules = []
    for rc in rules_config:
        rtype = rc["type"]
        if rtype == "stop_loss":
            rules.append(StopLoss(pct=rc["threshold"]))
        elif rtype == "trailing_stop":
            rules.append(TrailingStop(pct=rc["threshold"]))
        elif rtype == "time_exit":
            rules.append(TimeExit(max_bars=rc["bars"]))

    if not rules:
        return None
    return RuleChain(rules) if len(rules) > 1 else rules[0]


def _build_risk_manager(risk_spec: dict, initial_cash: float):
    """Create RiskManager with portfolio-level limits from risk spec.

    Supports: max_drawdown, daily_loss.
    Returns initialized RiskManager, or None.
    """
    limits_config = risk_spec.get("portfolio_limits", [])
    if not limits_config:
        return None

    from ml4t.backtest.risk import DailyLossLimit, MaxDrawdownLimit, RiskManager

    limits = []
    for lc in limits_config:
        ltype = lc["type"]
        if ltype == "max_drawdown":
            limits.append(MaxDrawdownLimit(max_drawdown=lc["threshold"]))
        elif ltype == "daily_loss":
            limits.append(DailyLossLimit(max_daily_loss_pct=lc["threshold"]))

    if not limits:
        return None
    rm = RiskManager(limits=limits)
    rm.initialize(initial_cash)
    return rm


# ---------------------------------------------------------------------------
# Convenience: run random-signal plumbing test
# ---------------------------------------------------------------------------


def run_plumbing_test(
    case_study: str,
    prices: pl.DataFrame,
    strategy_spec: dict,
    *,
    predictions: pl.DataFrame | None = None,
    label: str | None = None,
    n_assets: int | None = None,
    top_k: int = 20,
    seed: int = 42,
    initial_cash: float = 1_000_000.0,
    calendar: str = "NYSE",
    contract_specs: dict | None = None,
) -> float:
    """Run a random-signal backtest. Returns Sharpe ratio (should be ~0).

    This validates the backtest pipeline produces no spurious alpha
    from random inputs.
    """
    backtest_config = get_backtest_config(case_study)
    strategy_spec = ensure_backtest_spec(
        case_study,
        backtest_config,
        strategy_spec,
        prices=prices,
        prediction_hash="plumbing_test",
        initial_cash=initial_cash,
    )
    strategy = strategy_view(strategy_spec)
    rebal_spec = strategy.get("rebalance", {})

    if rebal_spec["mode"] == "vectorized":
        prediction_hash = "plumbing_test"
        label = label or backtest_config.primary_label
        if predictions is None:
            from case_studies.utils.registry import load_prediction_index, read_predictions

            prediction_index = load_prediction_index(
                case_study,
                label=label,
                split="validation",
            )
            if prediction_index.is_empty():
                raise ValueError(
                    f"Vectorized plumbing test found no validation predictions for "
                    f"{case_study}/{label}"
                )
            prediction_hash = prediction_index.row(0, named=True)["prediction_hash"]
            predictions = read_predictions(case_study, prediction_hash)

        random_predictions = normalize_prediction_columns(predictions)
        rng = np.random.default_rng(seed)
        random_predictions = random_predictions.with_columns(
            pl.Series("y_score", rng.standard_normal(random_predictions.height))
        )
        result = run_backtest(
            case_study,
            prediction_hash,
            strategy_spec,
            prices=prices,
            predictions=random_predictions,
            label=label,
            register=False,
            initial_cash=initial_cash,
            calendar=calendar,
            contract_specs=contract_specs,
        )
        return float(result.metrics["sharpe"])

    # Engine plumbing test
    from ml4t.backtest import DataFeed, Engine, RebalanceConfig, Strategy, TargetWeightExecutor

    config = runtime_backtest_config(strategy_spec)
    signal_config = strategy["signal"]
    long_short = bool(signal_config.get("long_short", False))
    signal_direction = str(signal_config.get("direction", "long_only")).strip().lower()
    allow_short = long_short or signal_direction == "short_only"

    # Calendar-aware rebalance schedule for random signal
    from case_studies.utils.backtest_loaders import resolve_rebalance_timestamps

    cadence = rebal_spec.get("cadence", "monthly_month_end")
    price_ts = prices["timestamp"].unique().sort()
    plumbing_schedule = set(resolve_rebalance_timestamps(price_ts, cadence, calendar).to_list())

    asset_list = sorted(prices["symbol"].unique().to_list())
    k = min(top_k, len(asset_list))
    rng = np.random.default_rng(seed)

    class _RandomStrategy(Strategy):
        def __init__(self):
            self.executor = TargetWeightExecutor(
                config=RebalanceConfig(
                    min_trade_value=100.0,
                    min_weight_change=0.005,
                    allow_fractional=None,  # Defer to broker.share_type (profile)
                    allow_short=allow_short,
                )
            )

        def on_data(self, timestamp, data, context, broker):
            if timestamp not in plumbing_schedule:
                return

            available = [a for a in asset_list if a in data]
            if not available:
                return

            if signal_direction == "short_only":
                selected = rng.choice(available, size=min(k, len(available)), replace=False)
                weight = -1.0 / len(selected)
                targets = {a: weight for a in selected}
            elif long_short:
                side_k = min(k, len(available) // 2)
                if side_k == 0:
                    return
                selected = rng.choice(available, size=side_k * 2, replace=False).tolist()
                longs = selected[:side_k]
                shorts = selected[side_k:]
                long_weight = 1.0 / len(longs)
                short_weight = -1.0 / len(shorts)
                targets = {a: long_weight for a in longs}
                targets.update({a: short_weight for a in shorts})
            else:
                selected = rng.choice(available, size=min(k, len(available)), replace=False)
                weight = 1.0 / len(selected)
                targets = {a: weight for a in selected}

            self.executor.execute(targets, data, broker)

    feed = DataFeed(prices_df=prices, feed_spec=config.feed_spec)
    strategy = _RandomStrategy()
    engine = Engine.from_config(feed, strategy, config, contract_specs=contract_specs)
    result = engine.run()

    return result.metrics.get("sharpe", 0.0)
