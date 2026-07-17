"""Registration functions for training runs, prediction sets, and backtest runs."""

from __future__ import annotations

import logging
from pathlib import Path

from .specs import (
    _validate_spec,
    backtest_hash_from_parts,
    build_training_spec,
    canonical_json,
    prediction_hash_from_parts,
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
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    spec = _validate_spec(spec)
    t_hash = training_hash_from_spec(spec)
    spec_json_str = canonical_json(spec)

    # Write spec.json (authoritative identity artifact)
    train_dir = _training_dir(case_dir, t_hash)
    _save_json(train_dir / "spec.json", spec)

    # Insert into DB
    db = _open_registry(case_dir)
    try:
        db.execute(
            """
            INSERT OR REPLACE INTO training_runs
            (training_hash, family, label, config_name,
             spec_json, created_at, git_commit, entry_point,
             started_at, elapsed_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        db.commit()
    finally:
        db.close()

    return t_hash


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

    Returns
    -------
    str
        The ``training_hash`` for the registered run.
    """
    assert family in ("deep_learning", "tabular_dl"), (
        f"register_epoch_checkpoint: family must be 'deep_learning' or 'tabular_dl', got {family!r}"
    )

    try:
        # Main path: preset loaded from disk is authoritative.
        # extra_params is deliberately NOT passed here — doing so would
        # merge architecture/lookback into spec["params"] and change the
        # training_hash vs. historical runs that already populated the
        # preset's own params from disk.
        spec = build_training_spec(
            family,
            config_name,
            label,
            n_folds=n_folds,
            n_epochs=n_epochs,
            feature_sets=feature_sets,
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
            "params": dict(extra_params) if extra_params else {},
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
    )
    return t_hash


# ---------------------------------------------------------------------------
# Registration: Prediction Sets
# ---------------------------------------------------------------------------


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

    p_hash = prediction_hash_from_parts(training_hash, checkpoint_value, split)

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
            y_true_col, y_score_col = _detect_score_cols(predictions)
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
            headline, fold_m = compute_prediction_fold_metrics(
                predictions,
                y_true_col=y_true_col,
                y_score_col=y_score_col,
                fold_col=fold_col,
                task_type=task_type,
                class_values=class_values,
                eval_col=eval_col,
                label=resolved_label,
            )
            # Merge auto-computed headline with caller-provided metrics
            merged = {**headline, **(metrics or {})}
            register_prediction_metrics(case_study, p_hash, merged, case_dir=case_dir)
            # Store per-fold metrics
            register_fold_metrics(case_study, p_hash, fold_m, case_dir=case_dir)
        except Exception as exc:
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

    b_hash = backtest_hash_from_parts(prediction_hash, strategy_spec)
    spec_json_str = canonical_json(strategy_spec)

    # Write spec.json
    bt_dir = _backtest_dir(case_dir, b_hash)
    _save_json(bt_dir / "spec.json", strategy_spec)

    # Save returns
    if returns is not None:
        _save_parquet(bt_dir / "daily_returns.parquet", returns)

    # Save trade log
    if trades is not None:
        _save_parquet(bt_dir / "trades.parquet", trades)

    # Save fill-level execution records
    if fills is not None:
        _save_parquet(bt_dir / "fills.parquet", fills)

    # Save bar-level equity curve
    if equity is not None:
        _save_parquet(bt_dir / "equity.parquet", equity)

    # Save bar-level portfolio state
    if portfolio_state is not None:
        _save_parquet(bt_dir / "portfolio_state.parquet", portfolio_state)

    # Save target weights
    if weights is not None:
        _save_parquet(bt_dir / "weights.parquet", weights)

    # Defensive: compute per-backtest uncertainty inline from daily
    # returns when the caller didn't pre-compute it. The canonical
    # backtest path (case_studies.utils.backtest_runner.run_backtest)
    # already populates uncertainty via compute_portfolio_metrics, so
    # this branch only fires for callers that bypass that path. Catches
    # the stale-CI class of bugs where a code path writes point
    # estimates without the uncertainty pack.
    needs_uncertainty = returns is not None and (metrics is None or "sharpe_se_lo" not in metrics)
    if needs_uncertainty:
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

    # Insert into DB — clean child tables first to avoid FK violations
    # on INSERT OR REPLACE (which is DELETE + INSERT under the hood)
    db = _open_registry(case_dir)
    try:
        db.execute("DELETE FROM backtest_fold_metrics WHERE backtest_hash = ?", (b_hash,))
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = ?", (b_hash,))
        db.execute(
            """
            INSERT OR REPLACE INTO backtest_runs
            (backtest_hash, prediction_hash, spec_json, stage, created_at, git_commit,
             started_at, elapsed_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
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

    def _i(v: object) -> int | None:
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

    When ``prune_dangling`` is set (default), cohort rows whose ``leader_hash``
    no longer maps to a ``backtest_runs`` row are removed after the writes, so a
    post-rerun leader shift cannot leave a stale leader row behind.

    Returns the number of dangling rows pruned.
    """
    if case_dir is None:
        case_dir = _case_dir(case_study)

    db = _open_registry(case_dir)
    try:
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
    db = _open_registry(case_dir)
    try:
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
