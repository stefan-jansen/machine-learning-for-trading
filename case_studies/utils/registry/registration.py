"""Registration functions for training runs, prediction sets, and backtest runs."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .specs import (
    SUPPORTED_IDENTITY_VERSIONS,
    _hashable_strategy_spec,
    _validate_spec,
    backtest_hash_from_parts,
    build_training_spec,
    canonical_json,
    prediction_hash_from_parts,
    project_training_identity,
    training_hash_from_spec,
)
from .store import (
    _backtest_dir,
    _case_dir,
    _git_hash,
    _infer_stage,
    _open_registry,
    _prediction_dir,
    _save_json,
    _save_parquet,
    _training_dir,
    _upsert_wide_metrics,
    _utc_now,
)

logger = logging.getLogger(__name__)

VALID_PREDICTION_SPLITS = frozenset({"validation", "holdout"})
MAX_PREDICTION_STD_RATIO = 100.0


def _atomic_save_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _save_json(temporary, data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_prediction_dispersion(predictions) -> None:
    """Reject a prediction set with an implausible score scale on any fold.

    The bound is deliberately wide. Across 8,090 finite folds in the nine
    production registries, the largest fold below the failure population was
    72.72. The known divergent folds started at 187.41 and extended to
    9.22e39. Rank correlation cannot detect this failure because it is invariant
    to score scale.
    """
    import math

    import polars as pl

    if not isinstance(predictions, pl.DataFrame):
        predictions = pl.from_pandas(predictions)

    fold_col = _detect_fold_col(predictions)
    y_true_col, y_score_col = _detect_score_cols(predictions)
    if fold_col is None or not {y_true_col, y_score_col}.issubset(predictions.columns):
        return

    typed = predictions.lazy().select(
        pl.col(fold_col).alias("fold"),
        pl.col(y_true_col).cast(pl.Float64, strict=False).alias("actual"),
        pl.col(y_score_col).cast(pl.Float64, strict=False).alias("score"),
    )
    fold_health = (
        typed.group_by("fold")
        .agg(
            pl.len().alias("n_total"),
            pl.col("score").is_finite().sum().alias("n_finite"),
        )
        .collect()
    )
    invalid_folds = []
    for row in fold_health.iter_rows(named=True):
        n_invalid = row["n_total"] - row["n_finite"]
        if n_invalid:
            invalid_folds.append(f"fold {row['fold']}: {n_invalid} non-finite score(s)")
    if invalid_folds:
        raise ValueError(
            "Refusing to register predictions with a non-finite fold: " + "; ".join(invalid_folds)
        )

    dispersion = (
        typed.filter(pl.col("actual").is_finite() & pl.col("score").is_finite())
        .group_by("fold")
        .agg(
            pl.len().alias("n"),
            pl.col("actual").std().alias("actual_std"),
            pl.col("score").std().alias("score_std"),
        )
        .collect()
    )

    violations = []
    for row in dispersion.iter_rows(named=True):
        actual_std = row["actual_std"]
        score_std = row["score_std"]
        if row["n"] < 2 or actual_std is None or actual_std <= 0 or score_std is None:
            continue
        ratio = float(score_std / actual_std)
        if not math.isfinite(ratio) or ratio > MAX_PREDICTION_STD_RATIO:
            violations.append(
                f"fold {row['fold']}: prediction dispersion ratio {ratio:.6g} "
                f"(score std {score_std:.6g} / target std {actual_std:.6g})"
            )

    if violations:
        raise ValueError(
            "Refusing to register predictions with a diverged fold; "
            f"the maximum allowed per-fold dispersion ratio is "
            f"{MAX_PREDICTION_STD_RATIO:g}: " + "; ".join(violations)
        )


def clear_prediction_sets(
    case_study: str,
    training_hash: str,
    *,
    split: str = "validation",
    case_dir: Path | None = None,
) -> dict[str, int]:
    """Remove one training run's existing prediction surface and descendants.

    Forced retraining reuses the deterministic training hash. Clearing first
    prevents checkpoints that are absent from the replacement run from leaking
    back into registry-backed leaderboards.
    """
    if split not in VALID_PREDICTION_SPLITS:
        raise ValueError(f"invalid prediction split: {split!r}")
    if case_dir is None:
        case_dir = _case_dir(case_study)

    db = _open_registry(case_dir)
    try:
        prediction_hashes = [
            row[0]
            for row in db.execute(
                "SELECT prediction_hash FROM prediction_sets WHERE training_hash = ? AND split = ?",
                (training_hash, split),
            ).fetchall()
        ]
        if not prediction_hashes:
            return {"prediction_sets": 0, "backtest_runs": 0}

        placeholders = ",".join("?" for _ in prediction_hashes)
        backtest_hashes = [
            row[0]
            for row in db.execute(
                f"SELECT backtest_hash FROM backtest_runs "
                f"WHERE prediction_hash IN ({placeholders})",
                prediction_hashes,
            ).fetchall()
        ]
        if backtest_hashes:
            bt_placeholders = ",".join("?" for _ in backtest_hashes)
            db.execute(
                f"DELETE FROM backtest_paired_metrics WHERE challenger_hash IN ({bt_placeholders}) "
                f"OR benchmark_hash IN ({bt_placeholders})",
                (*backtest_hashes, *backtest_hashes),
            )
            db.execute(
                f"DELETE FROM cohort_metrics WHERE leader_hash IN ({bt_placeholders})",
                backtest_hashes,
            )
            db.execute(
                f"DELETE FROM backtest_fold_metrics WHERE backtest_hash IN ({bt_placeholders})",
                backtest_hashes,
            )
            db.execute(
                f"DELETE FROM backtest_metrics WHERE backtest_hash IN ({bt_placeholders})",
                backtest_hashes,
            )
            db.execute(
                f"DELETE FROM backtest_runs WHERE backtest_hash IN ({bt_placeholders})",
                backtest_hashes,
            )

        db.execute(
            f"DELETE FROM fold_metrics WHERE prediction_hash IN ({placeholders})",
            prediction_hashes,
        )
        db.execute(
            f"DELETE FROM prediction_metrics WHERE prediction_hash IN ({placeholders})",
            prediction_hashes,
        )
        db.execute(
            f"DELETE FROM prediction_coverage WHERE prediction_hash IN ({placeholders})",
            prediction_hashes,
        )
        db.execute(
            f"DELETE FROM prediction_sets WHERE prediction_hash IN ({placeholders})",
            prediction_hashes,
        )
        db.commit()
    finally:
        db.close()

    for backtest_hash in backtest_hashes:
        shutil.rmtree(_backtest_dir(case_dir, backtest_hash), ignore_errors=True)
    for prediction_hash in prediction_hashes:
        shutil.rmtree(_prediction_dir(case_dir, prediction_hash), ignore_errors=True)

    return {
        "prediction_sets": len(prediction_hashes),
        "backtest_runs": len(backtest_hashes),
    }


# ---------------------------------------------------------------------------
# Registration: Training Runs
# ---------------------------------------------------------------------------


def register_training_run(
    case_study: str,
    spec: dict,
    *,
    entry_point: str | None = None,
    case_dir: Path | None = None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
    runtime_provenance: dict | None = None,
) -> str:
    """Register a training run. Returns training_hash.

    Parameters
    ----------
    case_study : str
        Case study ID (e.g. "etfs").
    spec : dict
        Identity-defining config (hashed). Must contain at least
        ``family``, ``label``, and ``seed``. If ``seed`` is omitted,
        DEFAULT_SEED (42) is injected automatically.
    entry_point : str, optional
        Notebook or script path that produced this run.
    case_dir : Path, optional
        Override case study directory.
    started_at : str, optional
        ISO timestamp when training started.
    elapsed_s : float, optional
        Wall-clock seconds for the training run.
    runtime_provenance : dict, optional
        Execution environment recorded separately from the portable hash.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    spec = _validate_spec(spec)
    t_hash = training_hash_from_spec(spec)
    spec_json_str = canonical_json(spec)

    if spec.get("identity_version") in SUPPORTED_IDENTITY_VERSIONS:
        identity_version = int(spec["identity_version"])
        tier = spec.get("execution_tier")
        if tier not in {"canonical", "preview"}:
            raise ValueError("versioned training spec requires execution_tier canonical or preview")
        db = _open_registry(case_dir)
        try:
            existing = db.execute(
                "SELECT spec_json, identity_version, execution_tier FROM training_runs "
                "WHERE training_hash = ?",
                (t_hash,),
            ).fetchone()
            if existing is not None:
                existing_spec = json.loads(existing[0])
                if project_training_identity(existing_spec) != project_training_identity(
                    spec
                ) or existing[1:] != (identity_version, tier):
                    raise ValueError(f"immutable training identity conflict for {t_hash}")
                spec_path = _training_dir(case_dir, t_hash) / "spec.json"
                try:
                    artifact_spec = json.loads(spec_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"immutable training spec artifact conflict for {t_hash}"
                    ) from exc
                if artifact_spec != existing_spec:
                    raise ValueError(f"immutable training spec artifact conflict for {t_hash}")
                return t_hash

            train_dir = _training_dir(case_dir, t_hash)
            spec_path = train_dir / "spec.json"
            if spec_path.exists():
                try:
                    orphaned_spec = json.loads(spec_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"immutable training spec artifact conflict for {t_hash}"
                    ) from exc
                if orphaned_spec != spec:
                    raise ValueError(f"immutable training spec artifact conflict for {t_hash}")
            else:
                _atomic_save_json(spec_path, spec)
            if runtime_provenance is not None:
                runtime_path = train_dir / "runtime.json"
                if runtime_path.exists():
                    try:
                        orphaned_runtime = json.loads(runtime_path.read_text())
                    except (OSError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"immutable training runtime artifact conflict for {t_hash}"
                        ) from exc
                    if orphaned_runtime != runtime_provenance:
                        raise ValueError(
                            f"immutable training runtime artifact conflict for {t_hash}"
                        )
                else:
                    _atomic_save_json(runtime_path, runtime_provenance)
            try:
                db.execute(
                    """
                    INSERT INTO training_runs
                    (training_hash, family, label, config_name, spec_json, created_at,
                     git_commit, entry_point, started_at, elapsed_s, runtime_json,
                     identity_version, execution_tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        t_hash,
                        spec["family"],
                        spec["label"],
                        spec.get("config_name"),
                        spec_json_str,
                        _utc_now(),
                        _git_hash(),
                        entry_point,
                        started_at,
                        elapsed_s,
                        canonical_json(runtime_provenance)
                        if runtime_provenance is not None
                        else None,
                        identity_version,
                        tier,
                    ),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
        finally:
            db.close()
        return t_hash

    # Write spec.json (authoritative identity artifact)
    train_dir = _training_dir(case_dir, t_hash)
    _save_json(train_dir / "spec.json", spec)
    if runtime_provenance is not None:
        _save_json(train_dir / "runtime.json", runtime_provenance)

    # Insert into DB
    db = _open_registry(case_dir)
    try:
        db.execute(
            """
            INSERT OR REPLACE INTO training_runs
            (training_hash, family, label, config_name,
             spec_json, created_at, git_commit, entry_point,
             started_at, elapsed_s, runtime_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t_hash,
                spec["family"],
                spec["label"],
                spec.get("config_name"),
                spec_json_str,
                _utc_now(),
                _git_hash(),
                entry_point,
                started_at,
                elapsed_s,
                canonical_json(runtime_provenance) if runtime_provenance is not None else None,
            ),
        )
        db.commit()
    finally:
        db.close()

    return t_hash


def record_training_runtime(
    case_study: str,
    training_hash: str,
    *,
    case_dir: Path | None = None,
    measured: dict,
) -> None:
    """Record what a completed training run cost, against the row it produced.

    A training run is registered before it is fitted, because the identity has to exist before
    anything can be written under it. Nothing then came back to say what the fit cost, so
    ``training_runs.elapsed_s`` was NULL on every row the current path produced while the value
    sat in the run's ``runtime.json`` where no query looks. Scheduling the next run from recorded
    cost - which is what ``reference/case-study-runtimes.md`` exists to do - needs the column.

    ``measured`` carries the resource capture: wall seconds, CPU seconds, cores actually used and
    peak resident memory. ``elapsed_s`` is promoted to its own column because that is what is
    queried; the rest is merged into ``runtime_json``.

    The ``runtime.json`` artifact beside the row is deliberately left alone. It records what the
    run *was* and is compared byte for byte when the same identity is registered again, so
    writing a measurement into it would turn a legitimate re-run into an identity conflict.
    """
    if not measured:
        return
    if case_dir is None:
        case_dir = _case_dir(case_study)
    elapsed = measured.get("elapsed_s")

    db = _open_registry(case_dir)
    try:
        row = db.execute(
            "SELECT runtime_json FROM training_runs WHERE training_hash = ?",
            (training_hash,),
        ).fetchone()
        if row is None:
            raise ValueError(f"no training run registered for {training_hash}")
        try:
            runtime = json.loads(row[0]) if row[0] else {}
        except json.JSONDecodeError:
            runtime = {}
        if not isinstance(runtime, dict):
            runtime = {}
        # Measurements are namespaced so they cannot collide with a declared provenance field,
        # and so a reader can tell what the run declared from what it turned out to cost.
        resources = dict(runtime.get("resources") or {})
        resources.update(measured)
        runtime["resources"] = resources
        db.execute(
            "UPDATE training_runs SET elapsed_s = ?, runtime_json = ? WHERE training_hash = ?",
            (
                float(elapsed) if elapsed is not None else None,
                canonical_json(runtime),
                training_hash,
            ),
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def register_epoch_checkpoint(
    case_study: str,
    *,
    family: str,
    library: str,
    config_name: str,
    label: str,
    n_folds: int,
    n_epochs: int | None,
    best_epoch: int,
    ic_mean: float,
    predictions,
    extra_params: dict | None = None,
    learning_curves=None,
    feature_sets: list[str] | None = None,
    entry_point: str | None = None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
    prediction_split: str = "validation",
    checkpoint_interval: int | None = None,
    spec_extra_params: dict | None = None,
    task_type: str = "regression",
    class_values: list | None = None,
    eval_col: str | None = None,
    training_spec: dict | None = None,
) -> str:
    """Shared 'one-config-per-epoch-checkpoint' registration path.

    Used by the ``tabular_dl`` and ``deep_learning`` families, both of
    which run a walk-forward CV per config, score each checkpointed
    epoch on the validation slice, and register the best-checkpoint
    predictions as a single prediction_set under a training_hash keyed
    on the *configured* n_epochs (not the discovered best epoch).

    Other families use their own registration paths:
    - GBM has ``register_gbm_result`` (multi-checkpoint with
      fold_metrics.parquet and a checkpoint_interval hash field).
    - Latent factors register inline from ``run_latent_factor_cv`` via
      ``_register_model_predictions`` (multi-model, multi-epoch).
    - Causal DML registers treatment effect metrics, not IC predictions.

    Parameters
    ----------
    case_study : str
        Case study ID (e.g. "etfs").
    family : str
        Model family — must be "deep_learning" or "tabular_dl". Drives
        the registry ``family`` column and the ``build_training_spec``
        family-specific field population.
    library : str
        Library used to train the model ("pytorch" for deep_learning,
        "tabm" for tabular_dl). Written to ``spec["library"]`` in the
        FileNotFoundError fallback branch.
    config_name : str
        Preset name (e.g. "lstm_h128", "tabm_h256_m16").
    label : str
        Target label (e.g. "fwd_ret_21d").
    n_folds : int
        Number of CV folds executed.
    n_epochs : int | None
        The CONFIGURED total training budget (from the preset YAML).
        This goes into the training_hash so the hash identifies the
        configuration, not the outcome. Pass None to omit from the
        hash (same training_hash as historical runs without the field).
    best_epoch : int
        The DISCOVERED best checkpoint from early-stopping evaluation.
        Used only as ``checkpoint_value`` on the prediction_set — never
        in the training_hash, because two runs of the same config with
        the same seed can (in principle) pick different best epochs
        and must still hash to the same training identity.
    ic_mean : float
        Mean cross-sectional IC of the best-checkpoint predictions.
        Written as a metric on the prediction_set.
    predictions : pandas.DataFrame | polars.DataFrame
        Best-checkpoint predictions for the validation slice. Schema
        per ``register_prediction_set``.
    extra_params : dict, optional
        Params used ONLY in the fallback (no-preset-file) branch to
        build a hand-crafted spec when ``load_preset`` would raise
        ``FileNotFoundError``. When the preset file is found on disk,
        these values are ignored — the preset's own params are
        authoritative and adding ``extra_params`` to the main-path
        spec would change existing training_hashes. Used by the
        deep_learning family to carry ``architecture`` and
        ``lookback`` as fallback-only hints.
    learning_curves : polars.DataFrame, optional
        Per-epoch IC curves for this config. Written to
        ``<training_dir>/learning_curves.parquet`` via ``_save_parquet``
        (handles pl.Object columns).
    feature_sets : list[str], optional
        Override for the spec's ``feature_sets`` field. Default is
        ``["financial", "model_based"]`` in ``build_training_spec``.
    entry_point : str, optional
        Notebook label for provenance.
    started_at : str, optional
        ISO timestamp when this config's training started.
    elapsed_s : float, optional
        Wall-clock seconds for this config's training.
    training_spec : dict, optional
        Prebuilt identity used by cache lookup. When supplied, registration
        validates its family, config, label, and fold count and stores this
        exact spec instead of rebuilding it.

    Returns
    -------
    str
        The ``training_hash`` for the registered run.
    """
    assert family in ("deep_learning", "tabular_dl"), (
        f"register_epoch_checkpoint: family must be 'deep_learning' or 'tabular_dl', got {family!r}"
    )

    if training_spec is not None:
        spec = dict(training_spec)
        expected_identity = {
            "family": family,
            "config_name": config_name,
            "label": label,
            "n_folds": n_folds,
        }
        mismatches = {
            key: (spec.get(key), value)
            for key, value in expected_identity.items()
            if spec.get(key) != value
        }
        if mismatches:
            raise ValueError(f"training_spec disagrees with registration inputs: {mismatches}")
    else:
        spec = None

    try:
        # Main path: preset loaded from disk is authoritative.
        # extra_params is deliberately NOT passed here — doing so would
        # merge architecture/lookback into spec["params"] and change the
        # training_hash vs. historical runs that already populated the
        # preset's own params from disk.
        if spec is None:
            spec = build_training_spec(
                family,
                config_name,
                label,
                n_folds=n_folds,
                n_epochs=n_epochs,
                feature_sets=feature_sets,
                checkpoint_interval=checkpoint_interval,
                extra_params=spec_extra_params,
            )
    except FileNotFoundError:
        # Fallback for unknown config_name (no preset on disk).
        spec = {
            "config_name": config_name,
            "family": family,
            "feature_sets": feature_sets or ["financial", "model_based"],
            "label": label,
            "library": library,
            "n_folds": n_folds,
            "params": {
                **(dict(extra_params) if extra_params else {}),
                **(dict(spec_extra_params) if spec_extra_params else {}),
            },
            "seed": 42,
        }
        if n_epochs is not None:
            spec["n_epochs"] = n_epochs

    t_hash = register_training_run(
        case_study,
        spec=spec,
        entry_point=entry_point,
        started_at=started_at,
        elapsed_s=elapsed_s,
    )

    # Save learning curves using _save_parquet (handles pl.Object columns).
    if (
        learning_curves is not None
        and hasattr(learning_curves, "height")
        and learning_curves.height > 0
    ):
        from .store import get_training_dir as _get_training_dir

        train_dir = _get_training_dir(case_study, spec)
        _save_parquet(train_dir / "learning_curves.parquet", learning_curves)

    register_prediction_set(
        case_study,
        t_hash,
        checkpoint_value=best_epoch,
        checkpoint_kind="epoch",
        split=prediction_split,
        predictions=predictions,
        metrics={"ic_mean": ic_mean},
        task_type=task_type,
        class_values=class_values,
        eval_col=eval_col,
        label=label,
    )
    return t_hash


# ---------------------------------------------------------------------------
# Registration: Prediction Sets
# ---------------------------------------------------------------------------


def _declared_label_buffer(case_study: str, label: str | None) -> str | None:
    """The holding period the case study declares for ``label``."""
    if not label:
        return None
    try:
        from utils.artifact_specs import load_setup_config, resolve_label_buffer

        return resolve_label_buffer(case_study, label, load_setup_config(case_study))
    except Exception:  # noqa: BLE001 - a missing declaration is not a registration failure
        return None


def _sibling_direction_labels(case_study: str, case_dir, label: str | None):
    """The binary direction label cut from ``label``, loaded, or ``(None, None)``.

    Scoring a regression model by AUC needs the direction label derived from the same return,
    which ``labels.classification_eval_label`` already declares in the other direction. A label
    with more than two levels is skipped rather than collapsed: ``fwd_dir_8h_3c`` has a neutral
    band, so "up" in it is "up beyond the band", a different event from the plain direction
    label's "up", and storing the two under one column is how a distinction gets lost. Where a
    case study declares both, the strictly binary one is used.

    A missing or unreadable label is not a registration failure - the AUC is a secondary
    reading, and the run that produced the predictions is what matters.
    """
    if not label:
        return None, None
    try:
        import polars as pl

        from utils.modeling import get_direction_labels

        for name in get_direction_labels(case_study, label):
            path = case_dir / "labels" / f"{name}.parquet"
            if not path.exists():
                continue
            frame = pl.read_parquet(path)
            if name not in frame.columns:
                continue
            if frame.get_column(name).drop_nulls().n_unique() == 2:
                return frame, name
    except Exception as exc:  # noqa: BLE001
        logger.debug("no sibling direction label for %s/%s: %s", case_study, label, exc)
    return None, None


def register_prediction_set(
    case_study: str,
    training_hash: str,
    *,
    checkpoint_value: int | None = None,
    checkpoint_kind: str | None = None,
    split: str = "validation",
    predictions=None,
    metrics: dict[str, float | dict] | None = None,
    task_type: str = "regression",
    class_values: list | None = None,
    eval_col: str | None = None,
    label: str | None = None,
    case_dir: Path | None = None,
    expected_keys=None,
    allow_partial: bool = False,
) -> str:
    """Register a prediction set. Returns prediction_hash.

    Parameters
    ----------
    case_study : str
        Case study ID.
    training_hash : str
        Parent training run hash.
    checkpoint_value : int, optional
        Checkpoint number (e.g. 150 trees, 50 epochs). None for final-only.
    checkpoint_kind : str, optional
        Checkpoint type: "tree_limit", "epoch", "final".
    split : str
        "validation" or "holdout".
    predictions : DataFrame, optional
        Predictions to save as parquet.
    metrics : dict, optional
        Convenience: metrics to register in the same call.
        Keys are metric names, values are floats.
    task_type : str
        "regression" or "classification". Controls which metrics are computed.
    class_values : list, optional
        Sorted unique class values for classification (e.g. [0, 1] or [-1, 0, 1]).
        Required when task_type="classification".
    eval_col : str, optional
        For classification predictions, the column name in ``predictions``
        holding the continuous return that the binary/categorical label was
        derived from. IC is computed against this column; AUC/accuracy/log_loss
        use the binary label. Required when ``task_type="classification"``.
    case_dir : Path, optional
        Override case study directory.
    """
    from .metrics import compute_prediction_fold_metrics

    if split not in VALID_PREDICTION_SPLITS:
        raise ValueError(
            f"prediction_split={split!r} is not one of {sorted(VALID_PREDICTION_SPLITS)}. "
            "Typo guard: papermill PREDICTION_SPLIT params are free-form strings; "
            f"only {sorted(VALID_PREDICTION_SPLITS)} produce valid pred_sets."
        )

    if case_dir is None:
        case_dir = _case_dir(case_study)

    if predictions is not None:
        _validate_prediction_dispersion(predictions)

    db = _open_registry(case_dir)
    try:
        parent = db.execute(
            "SELECT identity_version, execution_tier FROM training_runs WHERE training_hash = ?",
            (training_hash,),
        ).fetchone()
    finally:
        db.close()
    if parent is None:
        raise ValueError(f"unknown training_hash {training_hash}")
    identity_version, _execution_tier = parent
    coverage = None
    if identity_version in SUPPORTED_IDENTITY_VERSIONS:
        if predictions is None or expected_keys is None:
            raise ValueError(
                "versioned prediction registration requires predictions and expected_keys"
            )
        from .completeness import evaluate_prediction_coverage

        coverage = evaluate_prediction_coverage(expected_keys, predictions)
        if not coverage.complete and not allow_partial:
            raise ValueError(f"prediction coverage is partial: {coverage.as_dict()}")
        db = _open_registry(case_dir)
        try:
            existing_schema = db.execute(
                """
                SELECT c.schema_json
                FROM prediction_coverage c
                JOIN prediction_sets p ON p.prediction_hash = c.prediction_hash
                WHERE p.training_hash = ? AND p.split = ?
                LIMIT 1
                """,
                (training_hash, split),
            ).fetchone()
        finally:
            db.close()
        if existing_schema is not None and existing_schema[0] != coverage.schema_json:
            raise ValueError("prediction schema differs from an existing checkpoint")

    p_hash = prediction_hash_from_parts(
        training_hash,
        checkpoint_value,
        split,
        checkpoint_kind=checkpoint_kind,
        identity_version=identity_version,
    )

    if identity_version in SUPPORTED_IDENTITY_VERSIONS:
        assert coverage is not None
        import polars as pl

        from case_studies.utils.artifact_digest import value_digest

        normalized_predictions = (
            predictions if isinstance(predictions, pl.DataFrame) else pl.from_pandas(predictions)
        )
        prediction_artifact_digest = value_digest(normalized_predictions)
        pred_dir = _prediction_dir(case_dir, p_hash)
        pred_path = pred_dir / "predictions.parquet"
        temporary = pred_dir / f".predictions.{uuid.uuid4().hex}.tmp"
        db = _open_registry(case_dir)
        created_artifact = False
        try:
            existing = db.execute(
                "SELECT training_hash, checkpoint_value, checkpoint_kind, split "
                "FROM prediction_sets WHERE prediction_hash = ?",
                (p_hash,),
            ).fetchone()
            expected_row = (training_hash, checkpoint_value, checkpoint_kind, split)
            if existing is not None:
                if existing != expected_row:
                    raise ValueError(f"immutable prediction identity conflict for {p_hash}")
                if (
                    not pred_path.exists()
                    or value_digest(pl.read_parquet(pred_path)) != prediction_artifact_digest
                ):
                    raise ValueError(f"immutable prediction artifact conflict for {p_hash}")
                recorded_digest = db.execute(
                    "SELECT artifact_digest FROM prediction_coverage WHERE prediction_hash = ?",
                    (p_hash,),
                ).fetchone()
                if recorded_digest is None:
                    raise ValueError(f"prediction {p_hash} has no coverage record")
                if recorded_digest[0] not in (None, prediction_artifact_digest):
                    raise ValueError(f"immutable prediction digest conflict for {p_hash}")
                if recorded_digest[0] is None:
                    db.execute(
                        "UPDATE prediction_coverage SET artifact_digest = ? "
                        "WHERE prediction_hash = ? AND artifact_digest IS NULL",
                        (prediction_artifact_digest, p_hash),
                    )
                    db.commit()
            else:
                if pred_path.exists():
                    try:
                        orphaned_digest = value_digest(pl.read_parquet(pred_path))
                    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
                        raise ValueError(
                            f"immutable prediction artifact conflict for {p_hash}"
                        ) from exc
                    if orphaned_digest != prediction_artifact_digest:
                        raise ValueError(f"immutable prediction artifact conflict for {p_hash}")
                else:
                    _save_parquet(temporary, predictions)
                try:
                    db.execute(
                        """
                        INSERT INTO prediction_sets
                        (prediction_hash, training_hash, checkpoint_value, checkpoint_kind,
                         split, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (p_hash, *expected_row, _utc_now()),
                    )
                    values = coverage.as_dict()
                    values["artifact_digest"] = prediction_artifact_digest
                    columns = ", ".join(("prediction_hash", *values))
                    placeholders = ", ".join("?" for _ in range(len(values) + 1))
                    db.execute(
                        f"INSERT INTO prediction_coverage ({columns}) VALUES ({placeholders})",
                        (p_hash, *values.values()),
                    )
                    if metrics:
                        _upsert_wide_metrics(
                            db, "prediction_metrics", {"prediction_hash": p_hash}, metrics
                        )
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    if not pred_path.exists():
                        os.replace(temporary, pred_path)
                        created_artifact = True
                    db.commit()
                except Exception:
                    db.rollback()
                    if created_artifact:
                        pred_path.unlink(missing_ok=True)
                    raise
        finally:
            temporary.unlink(missing_ok=True)
            db.close()
    else:
        # Save predictions
        if predictions is not None:
            pred_dir = _prediction_dir(case_dir, p_hash)
            _save_parquet(pred_dir / "predictions.parquet", predictions)

        # Insert into DB
        db = _open_registry(case_dir)
        try:
            db.execute(
                """
                INSERT OR REPLACE INTO prediction_sets
                (prediction_hash, training_hash, checkpoint_value, checkpoint_kind,
                 split, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    p_hash,
                    training_hash,
                    checkpoint_value,
                    checkpoint_kind,
                    split,
                    _utc_now(),
                ),
            )

            if metrics:
                _upsert_wide_metrics(db, "prediction_metrics", {"prediction_hash": p_hash}, metrics)

            db.commit()
        finally:
            db.close()

    # Auto-compute fold metrics when predictions are provided
    if predictions is not None and _has_fold_column(predictions):
        try:
            fold_col = _detect_fold_col(predictions)
            assert fold_col is not None
            y_true_col, y_score_col = _detect_score_cols(predictions)
            prediction_columns = set(predictions.columns)
            metric_entity_col = "product" if "product" in prediction_columns else "symbol"
            # Resolve label from training_runs if caller didn't supply it.
            resolved_label = label
            if not resolved_label:
                try:
                    db_lookup = _open_registry(case_dir)
                    row = db_lookup.execute(
                        "SELECT label FROM training_runs WHERE training_hash = ?",
                        (training_hash,),
                    ).fetchone()
                    if row and row[0]:
                        resolved_label = row[0]
                    db_lookup.close()
                except Exception:  # noqa: BLE001
                    pass
            direction_frame, direction_name = (
                _sibling_direction_labels(case_study, case_dir, resolved_label)
                if task_type != "classification"
                else (None, None)
            )
            headline, fold_m = compute_prediction_fold_metrics(
                predictions,
                y_true_col=y_true_col,
                y_score_col=y_score_col,
                fold_col=fold_col,
                entity_col=metric_entity_col,
                task_type=task_type,
                class_values=class_values,
                eval_col=eval_col,
                label=resolved_label,
                label_buffer=_declared_label_buffer(case_study, resolved_label),
                direction_labels=direction_frame,
                direction_col=direction_name,
            )
            # Merge auto-computed headline with caller-provided metrics
            merged = {**headline, **(metrics or {})}
            register_prediction_metrics(case_study, p_hash, merged, case_dir=case_dir)
            # Store per-fold metrics
            register_fold_metrics(case_study, p_hash, fold_m, case_dir=case_dir)
        except Exception as exc:
            if identity_version in SUPPORTED_IDENTITY_VERSIONS:
                raise
            logger.warning("Could not compute fold metrics for %s: %s", p_hash, exc)

    return p_hash


def _has_fold_column(predictions) -> bool:
    """Return True iff the predictions frame carries a fold column.

    Single-fold frames (i.e. holdout retrains, where ``fold_id`` is a constant
    0) still need the SSOT metrics path: daily-pooled IC + HAC inference on
    the holdout window is the canonical signal-quality readout, not the
    per-fold ``ic_mean`` alone.
    """
    fold_col = _detect_fold_col(predictions)
    return fold_col is not None


def _detect_fold_col(predictions) -> str | None:
    """Detect fold column name (fold_id or fold)."""
    cols = predictions.columns
    if "fold_id" in cols:
        return "fold_id"
    if "fold" in cols:
        return "fold"
    return None


def _detect_score_cols(predictions) -> tuple[str, str]:
    """Detect (y_true_col, y_score_col) from column names."""
    cols = set(predictions.columns)
    y_true = "y_true" if "y_true" in cols else "actual" if "actual" in cols else "y_true"
    y_score = (
        "y_score" if "y_score" in cols else "prediction" if "prediction" in cols else "y_score"
    )
    return y_true, y_score


# ---------------------------------------------------------------------------
# Registration: Prediction Metrics (standalone)
# ---------------------------------------------------------------------------


def register_prediction_metrics(
    case_study: str,
    prediction_hash: str,
    metrics: dict[str, float | dict],
    *,
    case_dir: Path | None = None,
) -> None:
    """Register metrics for an existing prediction set.

    Parameters
    ----------
    metrics : dict
        Keys are metric names (e.g. "ic_mean", "ic_std").
        Values are floats (scalar) or dicts (with "value" key and extra detail).
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    db = _open_registry(case_dir)
    try:
        _upsert_wide_metrics(
            db, "prediction_metrics", {"prediction_hash": prediction_hash}, metrics
        )
        db.commit()
    finally:
        db.close()


def register_fold_metrics(
    case_study: str,
    prediction_hash: str,
    fold_metrics: dict[int, dict[str, float]],
    *,
    case_dir: Path | None = None,
) -> None:
    """Register per-fold metrics for a prediction set.

    Parameters
    ----------
    fold_metrics : dict[int, dict[str, float]]
        Outer key = fold_id, inner key = metric name, value = metric value.
        Example: {0: {"ic": 0.03, "rmse": 0.05}, 1: {"ic": 0.01, "rmse": 0.06}}
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    now = _utc_now()
    db = _open_registry(case_dir)
    try:
        for fold_id, metrics in fold_metrics.items():
            _upsert_wide_metrics(
                db,
                "fold_metrics",
                {"prediction_hash": prediction_hash, "fold_id": int(fold_id)},
                metrics,
                computed_at=now,
            )
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Registration: Backtest Runs
# ---------------------------------------------------------------------------


def register_backtest_run(
    case_study: str,
    prediction_hash: str,
    strategy_spec: dict,
    *,
    stage: str | None = None,
    returns=None,
    trades=None,
    fills=None,
    equity=None,
    portfolio_state=None,
    weights=None,
    metrics: dict[str, float | dict] | None = None,
    case_dir: Path | None = None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
) -> str:
    """Register a backtest run. Returns backtest_hash.

    Parameters
    ----------
    prediction_hash : str
        Input prediction set hash.
    strategy_spec : dict
        Identity-defining strategy config (hashed).
    stage : str, optional
        Pipeline stage: "signal", "allocation", "cost_sensitivity",
        "risk_overlay".  If None, inferred from strategy_spec content.
    returns : DataFrame, optional
        Daily portfolio returns to save as parquet.
    trades : DataFrame, optional
        Trade log (entry/exit/pnl/fees) to save as parquet.
    fills : DataFrame, optional
        Per-fill execution records (quote-aware) to save as parquet.
    equity : DataFrame, optional
        Bar-level equity curve [timestamp, equity, return, drawdown, ...].
    portfolio_state : DataFrame, optional
        Bar-level portfolio state [timestamp, equity, cash, gross_exposure, ...].
    weights : DataFrame, optional
        Target weights [timestamp, symbol, weight] to save as parquet.
    metrics : dict, optional
        Convenience: metrics to register in the same call.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)
    if stage is None:
        stage = _infer_stage(strategy_spec, case_dir=case_dir, prediction_hash=prediction_hash)

    identity_version = strategy_spec.get("identity_version")
    if identity_version in SUPPORTED_IDENTITY_VERSIONS:
        db = _open_registry(case_dir)
        try:
            ancestry = db.execute(
                """
                SELECT t.execution_tier, c.status
                FROM prediction_sets p
                JOIN training_runs t ON t.training_hash = p.training_hash
                LEFT JOIN prediction_coverage c ON c.prediction_hash = p.prediction_hash
                WHERE p.prediction_hash = ?
                """,
                (prediction_hash,),
            ).fetchone()
        finally:
            db.close()
        if ancestry is None or ancestry[1] != "complete":
            raise ValueError("backtest requires complete prediction coverage")
        requested_tier = strategy_spec.get("execution_tier", "canonical")
        if requested_tier not in {"canonical", "preview"}:
            raise ValueError("backtest execution_tier must be canonical or preview")
        if ancestry[0] != requested_tier:
            raise ValueError("backtest execution tier conflicts with prediction ancestry")

    b_hash = backtest_hash_from_parts(
        prediction_hash, strategy_spec, identity_version=identity_version
    )
    stored_strategy_spec = dict(strategy_spec)
    stored_strategy_spec.pop("_runtime_backtest_config", None)
    spec_json_str = canonical_json(stored_strategy_spec)
    existing_strategy_spec: dict | None = None
    existing_backtest = False
    bt_dir = _backtest_dir(case_dir, b_hash)
    spec_path = bt_dir / "spec.json"

    db = _open_registry(case_dir)
    try:
        existing = db.execute(
            "SELECT prediction_hash, spec_json, "
            "EXISTS(SELECT 1 FROM backtest_metrics m WHERE m.backtest_hash = ?), "
            "artifact_digests_json FROM backtest_runs WHERE backtest_hash = ?",
            (b_hash, b_hash),
        ).fetchone()
    finally:
        db.close()
    if existing is not None:
        existing_backtest = True
        existing_spec = json.loads(existing[1] or "{}")
        if identity_version in SUPPORTED_IDENTITY_VERSIONS:
            same_identity = canonical_json(
                _hashable_strategy_spec(existing_spec)
            ) == canonical_json(_hashable_strategy_spec(strategy_spec))
            if existing[0] != prediction_hash or not same_identity:
                raise ValueError(f"immutable backtest identity conflict for {b_hash}")
        existing_strategy_spec = existing_spec
        if spec_path.is_file():
            try:
                artifact_spec = json.loads(spec_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"immutable backtest spec artifact conflict for {b_hash}") from exc
            artifact_spec.pop("_runtime_backtest_config", None)
            if canonical_json(artifact_spec) != canonical_json(existing_spec):
                raise ValueError(f"immutable backtest spec artifact conflict for {b_hash}")
        import polars as pl

        from case_studies.utils.artifact_digest import value_digest

        artifact_values = {
            "daily_returns.parquet": returns,
            "trades.parquet": trades,
            "fills.parquet": fills,
            "equity.parquet": equity,
            "portfolio_state.parquet": portfolio_state,
            "weights.parquet": weights,
        }
        for filename, value in artifact_values.items():
            if value is None:
                continue
            new_frame = value if isinstance(value, pl.DataFrame) else pl.from_pandas(value)
            existing_path = _backtest_dir(case_dir, b_hash) / filename
            if existing_path.exists() and value_digest(
                pl.read_parquet(existing_path)
            ) != value_digest(new_frame):
                raise ValueError(f"immutable backtest artifact conflict for {b_hash}")
        existing_returns = _backtest_dir(case_dir, b_hash) / "daily_returns.parquet"
        try:
            recorded_digests = json.loads(existing[3] or "")
        except (json.JSONDecodeError, TypeError):
            recorded_digests = None
        digests_valid = (
            isinstance(recorded_digests, dict) and "daily_returns.parquet" in recorded_digests
        )
        if digests_valid:
            for filename, expected_digest in recorded_digests.items():
                path = existing_returns.parent / filename
                if not path.is_file() or value_digest(pl.read_parquet(path)) != expected_digest:
                    digests_valid = False
                    break
        if (
            identity_version in SUPPORTED_IDENTITY_VERSIONS
            and existing_returns.exists()
            and existing[2]
            and digests_valid
        ):
            return b_hash

    # Write spec.json
    expected_artifact_spec = (
        existing_strategy_spec if existing_strategy_spec is not None else stored_strategy_spec
    )
    if spec_path.exists():
        try:
            artifact_spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"immutable backtest spec artifact conflict for {b_hash}") from exc
        artifact_spec.pop("_runtime_backtest_config", None)
        if canonical_json(artifact_spec) != canonical_json(expected_artifact_spec):
            raise ValueError(f"immutable backtest spec artifact conflict for {b_hash}")
    else:
        _save_json(
            spec_path,
            expected_artifact_spec,
        )

    import polars as pl

    from case_studies.utils.artifact_digest import value_digest

    artifact_values = {
        "daily_returns.parquet": returns,
        "trades.parquet": trades,
        "fills.parquet": fills,
        "equity.parquet": equity,
        "portfolio_state.parquet": portfolio_state,
        "weights.parquet": weights,
    }
    for filename, value in artifact_values.items():
        if value is None:
            continue
        frame = value if isinstance(value, pl.DataFrame) else pl.from_pandas(value)
        path = bt_dir / filename
        if path.exists() and value_digest(pl.read_parquet(path)) != value_digest(frame):
            raise ValueError(f"immutable backtest artifact conflict for {b_hash}")

    # Save returns
    if returns is not None and not (
        existing_backtest and (bt_dir / "daily_returns.parquet").exists()
    ):
        _save_parquet(bt_dir / "daily_returns.parquet", returns)

    # Save trade log
    if trades is not None and not (existing_backtest and (bt_dir / "trades.parquet").exists()):
        _save_parquet(bt_dir / "trades.parquet", trades)

    # Save fill-level execution records
    if fills is not None and not (existing_backtest and (bt_dir / "fills.parquet").exists()):
        _save_parquet(bt_dir / "fills.parquet", fills)

    # Save bar-level equity curve
    if equity is not None and not (existing_backtest and (bt_dir / "equity.parquet").exists()):
        _save_parquet(bt_dir / "equity.parquet", equity)

    # Save bar-level portfolio state
    if portfolio_state is not None and not (
        existing_backtest and (bt_dir / "portfolio_state.parquet").exists()
    ):
        _save_parquet(bt_dir / "portfolio_state.parquet", portfolio_state)

    # Save target weights
    if weights is not None and not (existing_backtest and (bt_dir / "weights.parquet").exists()):
        _save_parquet(bt_dir / "weights.parquet", weights)

    artifact_digests = {
        path.name: value_digest(pl.read_parquet(path)) for path in sorted(bt_dir.glob("*.parquet"))
    }
    if "daily_returns.parquet" not in artifact_digests:
        raise ValueError("registered backtest has no daily-returns artifact")
    artifact_digests_json = canonical_json(artifact_digests)

    # Defensive: compute per-backtest uncertainty inline from daily
    # returns when the caller didn't pre-compute it. The canonical
    # backtest path (case_studies.utils.backtest_runner.run_backtest)
    # already populates uncertainty via compute_portfolio_metrics, so
    # this branch only fires for callers that bypass that path. Catches
    # the stale-CI class of bugs where a code path writes point
    # estimates without the uncertainty pack.
    needs_uncertainty = returns is not None and (metrics is None or "sharpe_se_lo" not in metrics)
    if needs_uncertainty:
        assert returns is not None
        from case_studies.utils.uncertainty import (
            compute_backtest_uncertainty,
            periods_per_year_from_setup,
        )

        try:
            ppy = periods_per_year_from_setup(case_study)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning(
                "periods_per_year_from_setup failed for %s; defaulting to 252: %s",
                case_study,
                exc,
            )
            ppy = 252
        try:
            uncertainty = compute_backtest_uncertainty(
                returns,
                periods_per_year=ppy,
                case_study=case_study,
            )
        except Exception as exc:
            logger.warning(
                "compute_backtest_uncertainty failed for %s/%s: %s",
                case_study,
                b_hash,
                exc,
            )
            uncertainty = {}
        if uncertainty:
            metrics = dict(metrics) if metrics else {}
            metrics.update(uncertainty)
            if "n_periods" not in metrics:
                n = returns.height if hasattr(returns, "height") else len(returns)
                metrics["n_periods"] = float(n)

    # Insert into DB without replacing an existing parent row. Metric UPSERTs
    # below update only supplied columns and preserve prior headline and fold data.
    db = _open_registry(case_dir)
    try:
        if identity_version in SUPPORTED_IDENTITY_VERSIONS:
            existing = db.execute(
                "SELECT prediction_hash, spec_json FROM backtest_runs WHERE backtest_hash = ?",
                (b_hash,),
            ).fetchone()
            if existing is not None:
                existing_spec = json.loads(existing[1] or "{}")
                same_identity = canonical_json(
                    _hashable_strategy_spec(existing_spec)
                ) == canonical_json(_hashable_strategy_spec(strategy_spec))
                if existing[0] != prediction_hash or not same_identity:
                    raise ValueError(f"immutable backtest identity conflict for {b_hash}")
                stored_digest_json = db.execute(
                    "SELECT artifact_digests_json FROM backtest_runs WHERE backtest_hash = ?",
                    (b_hash,),
                ).fetchone()[0]
                if stored_digest_json not in (None, artifact_digests_json):
                    raise ValueError(f"immutable backtest artifact digest conflict for {b_hash}")
        db.execute(
            """
            INSERT OR IGNORE INTO backtest_runs
            (backtest_hash, prediction_hash, spec_json, stage, created_at, git_commit,
             started_at, elapsed_s, artifact_digests_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                b_hash,
                prediction_hash,
                spec_json_str,
                stage,
                _utc_now(),
                _git_hash(),
                started_at,
                elapsed_s,
                artifact_digests_json,
            ),
        )
        db.execute(
            "UPDATE backtest_runs SET artifact_digests_json = ? "
            "WHERE backtest_hash = ? AND artifact_digests_json IS NULL",
            (artifact_digests_json, b_hash),
        )

        if metrics:
            _upsert_wide_metrics(db, "backtest_metrics", {"backtest_hash": b_hash}, metrics)

        db.commit()
    finally:
        db.close()

    return b_hash


# ---------------------------------------------------------------------------
# Registration: Backtest Metrics (standalone)
# ---------------------------------------------------------------------------


def register_backtest_metrics(
    case_study: str,
    backtest_hash: str,
    metrics: dict[str, float | dict],
    *,
    case_dir: Path | None = None,
) -> None:
    """Register metrics for an existing backtest run."""
    if case_dir is None:
        case_dir = _case_dir(case_study)

    db = _open_registry(case_dir)
    try:
        _upsert_wide_metrics(db, "backtest_metrics", {"backtest_hash": backtest_hash}, metrics)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Registration: Backtest Fold Metrics
# ---------------------------------------------------------------------------


def register_backtest_fold_metrics(
    case_study: str,
    backtest_hash: str,
    fold_metrics: dict[int, dict[str, float]],
    *,
    case_dir: Path | None = None,
) -> None:
    """Register per-fold backtest metrics (Sharpe, max_dd, etc. per CV fold).

    Parameters
    ----------
    fold_metrics : dict[int, dict[str, float]]
        Outer key = fold_id, inner key = metric name, value = metric value.
        Example: {0: {"sharpe": 0.5, "max_drawdown": -0.1}, 1: {...}}
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    now = _utc_now()
    db = _open_registry(case_dir)
    try:
        for fold_id, metrics in fold_metrics.items():
            _upsert_wide_metrics(
                db,
                "backtest_fold_metrics",
                {"backtest_hash": backtest_hash, "fold_id": int(fold_id)},
                metrics,
                computed_at=now,
            )
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Registration: Paired-bootstrap comparison (challenger vs baseline)
# ---------------------------------------------------------------------------


def register_paired_metrics(
    case_study: str,
    challenger_hash: str,
    benchmark_hash: str,
    metrics: dict[str, float],
    *,
    benchmark_kind: str | None = None,
    periods_per_year: int | None = None,
    case_dir: Path | None = None,
) -> None:
    """Register paired-bootstrap comparison metrics for a challenger vs baseline.

    ``metrics`` is the dict returned by
    :func:`case_studies.utils.uncertainty.compute_paired_uncertainty`.
    ``benchmark_kind`` is one of ``"equal_weight"``, ``"signal_leader"``,
    ``"cost_sensitivity_leader"``, or any caller-defined label.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    columns = (
        "challenger_hash, benchmark_hash, benchmark_kind, periods_per_year, "
        "bootstrap_block_length, bootstrap_n, sharpe_diff, sharpe_diff_ci95_lo, "
        "sharpe_diff_ci95_hi, ret_diff, ret_diff_ci95_lo, ret_diff_ci95_hi, "
        "max_dd_diff, max_dd_diff_ci95_lo, max_dd_diff_ci95_hi, info_ratio, "
        "info_ratio_ci95_lo, info_ratio_ci95_hi, prob_challenger_wins, p_value, "
        "computed_at"
    )
    placeholders = ", ".join(["?"] * 21)

    def _f(key: str) -> float | None:
        v = metrics.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _i(v: Any) -> int | None:
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    row = (
        challenger_hash,
        benchmark_hash,
        benchmark_kind,
        periods_per_year,
        _i(metrics.get("bootstrap_block_length")),
        _i(metrics.get("bootstrap_n")),
        _f("sharpe_diff"),
        _f("sharpe_diff_ci95_lo"),
        _f("sharpe_diff_ci95_hi"),
        _f("ret_diff"),
        _f("ret_diff_ci95_lo"),
        _f("ret_diff_ci95_hi"),
        _f("max_dd_diff"),
        _f("max_dd_diff_ci95_lo"),
        _f("max_dd_diff_ci95_hi"),
        _f("info_ratio"),
        _f("info_ratio_ci95_lo"),
        _f("info_ratio_ci95_hi"),
        _f("prob_challenger_wins"),
        _f("p_value"),
        _utc_now(),
    )

    update_clause = ", ".join(
        f"{col.strip()} = excluded.{col.strip()}"
        for col in columns.split(",")
        if col.strip() not in {"challenger_hash", "benchmark_hash"}
    )
    db = _open_registry(case_dir)
    try:
        db.execute(
            f"INSERT INTO backtest_paired_metrics ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(challenger_hash, benchmark_hash) DO UPDATE SET {update_clause}",
            row,
        )
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Registration: Selection-bias cohort metrics (cohort_metrics table)
# ---------------------------------------------------------------------------


# Identity + meta columns that the caller supplies out-of-band; every other
# key in a cohort's ``metrics`` dict is a bare cohort_metrics column name
# (matching the flat dict returned by ``compute_cohort_metrics``).
_COHORT_META_COLS = ("leader_hash", "k_variants", "periods_per_year", "computed_at")


def register_cohort_metrics(
    case_study: str,
    cohorts: list[dict],
    *,
    prune_dangling: bool = True,
    replace_all: bool = False,
    case_dir: Path | None = None,
) -> int:
    """Persist selection-bias cohort rows to ``cohort_metrics``.

    Each entry in ``cohorts`` is a dict with keys ``cohort_type``, ``stage``,
    ``label``, ``family`` and ``metrics`` — the last being the flat dict
    returned by :func:`case_studies.utils.uncertainty.compute_cohort_metrics`,
    which carries ``leader_hash``, ``k_variants``, ``periods_per_year`` and the
    DSR / RAS / reality-check / PBO columns. Rows are keyed by the
    ``(cohort_type, stage, label, family)`` identity (matching
    ``idx_cohort_unique``); an existing row with that identity is replaced.

    When ``replace_all`` is set, the case-study cohort table is cleared in the
    same transaction before the supplied complete snapshot is inserted. This
    removes identities that cease to qualify after an eligibility-rule change.

    When ``prune_dangling`` is set (default), cohort rows whose ``leader_hash``
    no longer maps to a ``backtest_runs`` row are removed after the writes, so a
    post-rerun leader shift cannot leave a stale leader row behind.

    Returns the number of dangling rows pruned.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    db = _open_registry(case_dir)
    try:
        if replace_all:
            db.execute("DELETE FROM cohort_metrics")
        for c in cohorts:
            metrics = dict(c["metrics"])  # copy — we pop meta keys below
            leader_hash = metrics.pop("leader_hash")
            k_variants = metrics.pop("k_variants")
            ppy = metrics.pop("periods_per_year")
            computed_at = metrics.pop("computed_at", None) or _utc_now()

            cohort_type = c["cohort_type"]
            stage = c.get("stage")
            label = c["label"]
            family = c.get("family")

            cols = [
                "cohort_type",
                "stage",
                "label",
                "family",
                "leader_hash",
                "k_variants",
                "periods_per_year",
                "computed_at",
            ]
            vals: list[object] = [
                cohort_type,
                stage,
                label,
                family,
                leader_hash,
                k_variants,
                ppy,
                computed_at,
            ]
            for k, v in metrics.items():
                cols.append(k)
                vals.append(v)

            # Explicit REPLACE semantics on the identity index (DELETE + INSERT).
            db.execute(
                """
                DELETE FROM cohort_metrics
                WHERE cohort_type = ?
                  AND COALESCE(stage, '') = COALESCE(?, '')
                  AND label = ?
                  AND COALESCE(family, '') = COALESCE(?, '')
                """,
                (cohort_type, stage, label, family),
            )
            placeholders = ",".join("?" * len(vals))
            db.execute(
                f"INSERT INTO cohort_metrics ({','.join(cols)}) VALUES ({placeholders})",
                vals,
            )

        n_pruned = 0
        if prune_dangling:
            cur = db.execute(
                "DELETE FROM cohort_metrics "
                "WHERE leader_hash NOT IN (SELECT backtest_hash FROM backtest_runs)"
            )
            n_pruned = cur.rowcount or 0
        db.commit()
        return n_pruned
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Registration: Causal-DML runs (dedicated causal_runs table)
# ---------------------------------------------------------------------------


def register_causal_run(
    case_study: str,
    causal_hash: str,
    *,
    label: str,
    treatment: str,
    confounders_json: str,
    embargo: int | None,
    n_folds: int | None,
    n_obs: int,
    dml_effect: float,
    dml_se_hac: float,
    p_value_hac: float | None,
    naive_effect: float | None,
    confounding_bias_pct: float | None,
    refutation_p: float | None,
    spec_json: str,
    notebook: str | None,
    started_at: str | None,
    elapsed_s: float | None,
    case_dir: Path | None = None,
) -> None:
    """Persist one causal-DML run to ``causal_runs``.

    Causal-DML estimation lives in its own table because the predictive
    completeness contract (``ic_mean`` non-null, etc.) does not apply to
    treatment-effect estimates. Callers compute the spec and result fields
    upstream; this function owns the SQL row write.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)
    import json

    version = (json.loads(spec_json) or {}).get("identity_version")
    immutable = version in SUPPORTED_IDENTITY_VERSIONS
    db = _open_registry(case_dir)
    try:
        existing = db.execute(
            "SELECT label, treatment, confounders_json, embargo, n_folds, n_obs, "
            "dml_effect, dml_se_hac, p_value_hac, naive_effect, confounding_bias_pct, "
            "refutation_p, spec_json, notebook FROM causal_runs WHERE causal_hash = ?",
            (causal_hash,),
        ).fetchone()
        expected = (
            label,
            treatment,
            confounders_json,
            embargo,
            n_folds,
            n_obs,
            dml_effect,
            dml_se_hac,
            p_value_hac,
            naive_effect,
            confounding_bias_pct,
            refutation_p,
            spec_json,
            notebook,
        )
        if immutable and existing is not None:
            if existing != expected:
                raise ValueError(f"immutable causal result conflict for {causal_hash}")
            return
        # ON CONFLICT DO UPDATE rather than INSERT OR REPLACE — consistent with
        # register_paired_metrics, avoids the implicit DELETE that triggers
        # FK cascades and loses the original created_at timestamp.
        db.execute(
            """
            INSERT INTO causal_runs (
                causal_hash, label, treatment, confounders_json, embargo,
                n_folds, n_obs, dml_effect, dml_se_hac, p_value_hac,
                naive_effect, confounding_bias_pct, refutation_p,
                spec_json, notebook, started_at, elapsed_s, git_commit, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(causal_hash) DO UPDATE SET
                label=excluded.label,
                treatment=excluded.treatment,
                confounders_json=excluded.confounders_json,
                embargo=excluded.embargo,
                n_folds=excluded.n_folds,
                n_obs=excluded.n_obs,
                dml_effect=excluded.dml_effect,
                dml_se_hac=excluded.dml_se_hac,
                p_value_hac=excluded.p_value_hac,
                naive_effect=excluded.naive_effect,
                confounding_bias_pct=excluded.confounding_bias_pct,
                refutation_p=excluded.refutation_p,
                spec_json=excluded.spec_json,
                notebook=excluded.notebook,
                started_at=excluded.started_at,
                elapsed_s=excluded.elapsed_s,
                git_commit=excluded.git_commit
            WHERE causal_runs.label IS NOT excluded.label
               OR causal_runs.treatment IS NOT excluded.treatment
               OR causal_runs.confounders_json IS NOT excluded.confounders_json
               OR causal_runs.embargo IS NOT excluded.embargo
               OR causal_runs.n_folds IS NOT excluded.n_folds
               OR causal_runs.n_obs IS NOT excluded.n_obs
               OR causal_runs.dml_effect IS NOT excluded.dml_effect
               OR causal_runs.dml_se_hac IS NOT excluded.dml_se_hac
               OR causal_runs.p_value_hac IS NOT excluded.p_value_hac
               OR causal_runs.naive_effect IS NOT excluded.naive_effect
               OR causal_runs.confounding_bias_pct IS NOT excluded.confounding_bias_pct
               OR causal_runs.refutation_p IS NOT excluded.refutation_p
               OR causal_runs.spec_json IS NOT excluded.spec_json
               OR causal_runs.notebook IS NOT excluded.notebook
            """,
            (
                causal_hash,
                label,
                treatment,
                confounders_json,
                embargo,
                n_folds,
                n_obs,
                dml_effect,
                dml_se_hac,
                p_value_hac,
                naive_effect,
                confounding_bias_pct,
                refutation_p,
                spec_json,
                notebook,
                started_at,
                elapsed_s,
                _git_hash(),
                _utc_now(),
            ),
        )
        db.commit()
    finally:
        db.close()
