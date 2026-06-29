"""Registry completeness checks and skip-if-exists logic.

Provides a single entry point for "should I train this config?" decisions
across all model families (linear, gbm, tabular_dl, deep_learning,
latent_factors, causal_dml) and for backtests.

Contract
--------
Every training notebook should guard each config:

    spec = build_training_spec(...)
    status = training_run_status(CASE_STUDY_ID, spec)
    if status.complete and not FORCE_RETRAIN:
        print(f"  {cfg['config_name']}: SKIP — {status.summary()}")
        continue
    if status.partial:
        print(f"  {cfg['config_name']}: RETRAIN — {status.summary()}")
    # ... train and register

Every backtest sweep should guard each variant:

    strategy_spec = build_backtest_spec(...)
    status = backtest_run_status(CASE_STUDY_ID, pred_hash, strategy_spec)
    if status.complete and not FORCE_REBACKTEST:
        print(f"  {variant_name}: SKIP — backtest already complete")
        continue
    # ... run backtest

Rationale
---------
Large sweeps (GBM on 9.2M-row us_equities_panel, nasdaq100 microstructure,
DL families) can take hours. Re-running from scratch after a correctness
fix, partial interruption, or added configs wastes compute. Re-running
training where the training_hash already has complete artifacts is pure
waste — the hash IS the identity. If the hash exists and has all expected
artifacts, the result is reproducible and can be reused.

The only legitimate reasons to retrain:
1. The fix or config change produces a NEW hash (handled automatically).
2. The existing artifacts are corrupt or partially written.
3. FORCE_RETRAIN=True (explicit opt-in for debugging).

Partial state handling
----------------------
If some artifacts exist but not all (e.g., training_runs row but no
predictions.parquet), report the partial state and retrain. We NEVER
silently reuse a partial state because the result would be misleading
(the ic_mean might exist while the predictions are gone).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .specs import backtest_hash_from_parts, canonical_json, training_hash_from_spec
from .store import (
    _backtest_dir,
    _case_dir,
    _open_registry,
    _prediction_dir,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingRunStatus:
    """Completeness status of a training run in the registry.

    Fields
    ------
    training_hash : str
        Canonical identity hash from the spec.
    exists : bool
        True if the training_runs row exists.
    has_predictions : bool
        True if at least one prediction_sets row exists.
    has_predictions_file : bool
        True if at least one predictions.parquet file exists on disk.
    has_metrics : bool
        True if the prediction has an ic_mean value.
    complete : bool
        True if all required artifacts are present.
    partial : bool
        True if the run exists but some artifacts are missing.
    missing : tuple[str, ...]
        Names of missing artifacts.
    """

    training_hash: str
    exists: bool
    has_predictions: bool
    has_predictions_file: bool
    has_metrics: bool
    missing: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing and self.exists

    @property
    def partial(self) -> bool:
        return self.exists and bool(self.missing)

    def summary(self) -> str:
        if not self.exists:
            return f"no training_run for hash {self.training_hash[:12]}"
        if self.complete:
            return f"complete (hash={self.training_hash[:12]})"
        return f"partial (hash={self.training_hash[:12]}, missing: {', '.join(self.missing)})"


@dataclass(frozen=True)
class BacktestRunStatus:
    """Completeness status of a backtest run in the registry."""

    backtest_hash: str
    exists: bool
    has_returns: bool
    has_metrics: bool
    missing: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing and self.exists

    @property
    def partial(self) -> bool:
        return self.exists and bool(self.missing)

    def summary(self) -> str:
        if not self.exists:
            return f"no backtest_run for hash {self.backtest_hash[:12]}"
        if self.complete:
            return f"complete (hash={self.backtest_hash[:12]})"
        return f"partial (hash={self.backtest_hash[:12]}, missing: {', '.join(self.missing)})"


# ---------------------------------------------------------------------------
# Training run completeness
# ---------------------------------------------------------------------------


def training_run_status(
    case_study: str,
    spec: dict,
    *,
    require_metrics: bool = True,
    require_predictions_file: bool = True,
    case_dir: Path | None = None,
) -> TrainingRunStatus:
    """Inspect the registry for a training run matching the given spec.

    Parameters
    ----------
    case_study : str
        Case study id.
    spec : dict
        Complete training spec (same structure build_training_spec produces).
    require_metrics : bool
        Whether ic_mean must be non-NULL for the run to count as complete.
        Default True. Causal DML runs are tracked in `causal_runs`, not
        through this path.
    require_predictions_file : bool
        Whether predictions.parquet must exist on disk. Default True.
    case_dir : Path, optional
        Override case study directory.

    Returns
    -------
    TrainingRunStatus
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    t_hash = training_hash_from_spec(spec)

    db = _open_registry(case_dir)
    try:
        row = db.execute(
            "SELECT training_hash FROM training_runs WHERE training_hash = ?",
            (t_hash,),
        ).fetchone()
        exists = row is not None
        if not exists:
            return TrainingRunStatus(
                training_hash=t_hash,
                exists=False,
                has_predictions=False,
                has_predictions_file=False,
                has_metrics=False,
                missing=("training_run",),
            )

        # Prediction sets
        pred_hashes = [
            r[0]
            for r in db.execute(
                "SELECT prediction_hash FROM prediction_sets WHERE training_hash = ?",
                (t_hash,),
            ).fetchall()
        ]
        has_predictions = len(pred_hashes) > 0

        # Metrics on the prediction(s)
        has_metrics = False
        if has_predictions:
            # Get any prediction with non-null ic_mean
            q = (
                f"SELECT prediction_hash FROM prediction_metrics "
                f"WHERE prediction_hash IN ({','.join('?' * len(pred_hashes))}) "
                f"AND ic_mean IS NOT NULL"
            )
            m_rows = db.execute(q, tuple(pred_hashes)).fetchall()
            has_metrics = len(m_rows) > 0
    finally:
        db.close()

    # Check predictions.parquet files on disk
    has_predictions_file = False
    if has_predictions:
        for ph in pred_hashes:
            f = _prediction_dir(case_dir, ph) / "predictions.parquet"
            if f.exists():
                has_predictions_file = True
                break

    missing = []
    if not has_predictions:
        missing.append("prediction_sets")
    if require_predictions_file and not has_predictions_file:
        missing.append("predictions.parquet")
    if require_metrics and not has_metrics:
        missing.append("ic_mean")

    return TrainingRunStatus(
        training_hash=t_hash,
        exists=exists,
        has_predictions=has_predictions,
        has_predictions_file=has_predictions_file,
        has_metrics=has_metrics,
        missing=tuple(missing),
    )


def skip_training_if_complete(
    case_study: str,
    spec: dict,
    *,
    force_retrain: bool = False,
    verbose: bool = True,
    **kwargs,
) -> TrainingRunStatus:
    """Convenience wrapper for the "should I train?" decision.

    Returns the status. Caller should check ``status.complete`` and
    ``force_retrain`` to decide whether to skip.

    When ``verbose=True``, prints a one-line status for partial/complete runs
    so interactive runs get visible feedback.

    Example
    -------
        status = skip_training_if_complete(CASE_STUDY_ID, spec,
                                          force_retrain=FORCE_RETRAIN)
        if status.complete and not FORCE_RETRAIN:
            print(f"  {cfg_name}: SKIP ({status.summary()})")
            continue
    """
    status = training_run_status(case_study, spec, **kwargs)
    if verbose:
        if status.complete and not force_retrain:
            return status  # caller prints
        if status.partial:
            print(f"  WARNING: partial run detected, will retrain: {status.summary()}")
    return status


# ---------------------------------------------------------------------------
# Backtest run completeness
# ---------------------------------------------------------------------------


def backtest_run_status(
    case_study: str,
    prediction_hash: str,
    strategy_spec: dict,
    *,
    require_metrics: bool = True,
    require_returns_file: bool = True,
    case_dir: Path | None = None,
) -> BacktestRunStatus:
    """Inspect the registry for a backtest run matching prediction_hash + strategy_spec."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    b_hash = backtest_hash_from_parts(prediction_hash, strategy_spec)

    db = _open_registry(case_dir)
    try:
        row = db.execute(
            "SELECT backtest_hash FROM backtest_runs WHERE backtest_hash = ?",
            (b_hash,),
        ).fetchone()
        exists = row is not None
        if not exists:
            return BacktestRunStatus(
                backtest_hash=b_hash,
                exists=False,
                has_returns=False,
                has_metrics=False,
                missing=("backtest_run",),
            )

        has_metrics = False
        if require_metrics:
            m_row = db.execute(
                "SELECT sharpe FROM backtest_metrics WHERE backtest_hash = ? AND sharpe IS NOT NULL",
                (b_hash,),
            ).fetchone()
            has_metrics = m_row is not None
    finally:
        db.close()

    # Check returns.parquet on disk
    has_returns = (_backtest_dir(case_dir, b_hash) / "daily_returns.parquet").exists()

    missing = []
    if require_returns_file and not has_returns:
        missing.append("daily_returns.parquet")
    if require_metrics and not has_metrics:
        missing.append("sharpe")

    return BacktestRunStatus(
        backtest_hash=b_hash,
        exists=exists,
        has_returns=has_returns,
        has_metrics=has_metrics,
        missing=tuple(missing),
    )


def skip_backtest_if_complete(
    case_study: str,
    prediction_hash: str,
    strategy_spec: dict,
    *,
    force_rebacktest: bool = False,
    verbose: bool = True,
    **kwargs,
) -> BacktestRunStatus:
    """Convenience wrapper for the "should I backtest?" decision.

    Example
    -------
        status = skip_backtest_if_complete(CASE_STUDY_ID, pred_hash, spec,
                                          force_rebacktest=FORCE_REBACKTEST)
        if status.complete and not FORCE_REBACKTEST:
            print(f"  {variant_name}: SKIP ({status.summary()})")
            continue
    """
    status = backtest_run_status(case_study, prediction_hash, strategy_spec, **kwargs)
    if verbose:
        if status.partial:
            print(f"  WARNING: partial backtest detected, will re-run: {status.summary()}")
    return status


__all__ = [
    "TrainingRunStatus",
    "BacktestRunStatus",
    "training_run_status",
    "skip_training_if_complete",
    "backtest_run_status",
    "skip_backtest_if_complete",
]
