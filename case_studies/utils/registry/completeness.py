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

import json
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


@dataclass(frozen=True)
class PredictionCoverage:
    """Exact expected-versus-actual prediction coverage evidence."""

    expected_key_digest: str
    actual_key_digest: str
    n_expected: int
    n_actual: int
    n_duplicates: int
    n_missing: int
    n_extra: int
    n_null: int
    n_non_finite: int
    n_folds_expected: int
    n_folds_actual: int
    schema_json: str
    status: str

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def as_dict(self) -> dict[str, str | int]:
        return {
            "expected_key_digest": self.expected_key_digest,
            "actual_key_digest": self.actual_key_digest,
            "n_expected": self.n_expected,
            "n_actual": self.n_actual,
            "n_duplicates": self.n_duplicates,
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "n_null": self.n_null,
            "n_non_finite": self.n_non_finite,
            "n_folds_expected": self.n_folds_expected,
            "n_folds_actual": self.n_folds_actual,
            "schema_json": self.schema_json,
            "status": self.status,
        }


def _prediction_key_columns(frame) -> tuple[str, ...]:
    columns = set(frame.columns)
    entities = [name for name in ("symbol", "product") if name in columns]
    if len(entities) != 1:
        raise ValueError("prediction coverage requires exactly one of symbol or product")
    return (
        entities[0],
        *(("position",) if "position" in columns else ()),
        "timestamp",
        "fold_id",
    )


def _canonical_key_frame(frame, key_columns: tuple[str, ...] | None = None):
    import polars as pl

    if not isinstance(frame, pl.DataFrame):
        frame = pl.from_pandas(frame)
    if "fold" in frame.columns and "fold_id" not in frame.columns:
        frame = frame.rename({"fold": "fold_id"})
    if key_columns is None:
        key_columns = _prediction_key_columns(frame)
    required = set(key_columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"prediction coverage requires columns {sorted(required)}; missing {missing}"
        )
    return frame.select(
        *(
            pl.col(name).cast(pl.Int64) if name == "fold_id" else pl.col(name).cast(pl.String)
            for name in key_columns
        )
    )


def _key_digest(frame, key_columns: tuple[str, ...]) -> str:
    from case_studies.utils.artifact_digest import value_digest

    return value_digest(frame, key_columns)


def evaluate_prediction_coverage(expected_keys, predictions) -> PredictionCoverage:
    """Compare exact prediction keys and finite scores without mutating storage."""
    import polars as pl

    expected = _canonical_key_frame(expected_keys)
    key_columns = tuple(expected.columns)
    actual = _canonical_key_frame(predictions, key_columns)
    if expected.n_unique(key_columns) != expected.height:
        raise ValueError("expected prediction coverage keys must be unique")

    unique_actual = actual.unique(key_columns)
    n_duplicates = actual.height - unique_actual.height
    n_missing = expected.join(unique_actual, on=key_columns, how="anti").height
    n_extra = unique_actual.join(expected, on=key_columns, how="anti").height

    if not isinstance(predictions, pl.DataFrame):
        predictions = pl.from_pandas(predictions)
    score_col = "y_score" if "y_score" in predictions.columns else "prediction"
    if score_col not in predictions.columns:
        raise ValueError("prediction coverage requires y_score or prediction")
    score = predictions.get_column(score_col).cast(pl.Float64, strict=False)
    n_null = score.null_count()
    n_non_finite = (score.is_not_null() & ~score.is_finite()).sum()
    expected_digest = _key_digest(expected, key_columns)
    actual_digest = _key_digest(unique_actual, key_columns)
    complete = not any((n_duplicates, n_missing, n_extra, n_null, n_non_finite)) and (
        expected_digest == actual_digest
    )
    return PredictionCoverage(
        expected_key_digest=expected_digest,
        actual_key_digest=actual_digest,
        n_expected=expected.height,
        n_actual=actual.height,
        n_duplicates=n_duplicates,
        n_missing=n_missing,
        n_extra=n_extra,
        n_null=n_null,
        n_non_finite=int(n_non_finite),
        n_folds_expected=expected.get_column("fold_id").n_unique(),
        n_folds_actual=actual.get_column("fold_id").n_unique(),
        schema_json=json.dumps(
            {name: str(dtype) for name, dtype in predictions.schema.items()},
            sort_keys=True,
            separators=(",", ":"),
        ),
        status="complete" if complete else "partial",
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
