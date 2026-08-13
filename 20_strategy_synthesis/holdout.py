"""Holdout prediction and backtest generation for case studies.

Provides the core logic for out-of-sample validation: select the best
model from the validation registry, retrain on all pre-holdout data,
generate predictions on the holdout window, and backtest.

Usage::

    from holdout import generate_holdout

    result = generate_holdout("etfs")
    print(f"Holdout Sharpe: {result['holdout_sharpe']:.3f}")
"""

from __future__ import annotations

import contextlib
import gc
import json
import sqlite3
import time
from datetime import date as dt_date
from datetime import timedelta
from pathlib import Path
from typing import Any

# lightgbm has to be imported before anything that loads scikit-learn, which
# `ml4t.diagnostic` does transitively. Both ship their own OpenMP runtime and
# the first one loaded wins for the whole process; on macOS ARM64 the loser's
# first multithreaded fit dies in `__kmp_suspend_initialize_thread`, killing
# the kernel with no traceback. `_train_gbm` re-imports it locally for reading;
# this one is here only to lose no race. Plain `import` statements sort ahead
# of `from ... import` ones, so isort keeps this above the ml4t line.
import lightgbm  # noqa: F401
import numpy as np
import polars as pl
import torch  # win the cudart-resolution race vs ml4t.diagnostic  # noqa: F401
import yaml
from ml4t.diagnostic.metrics import cross_sectional_ic

# polars.PanicException inherits from BaseException, not Exception, so the
# fallback loop's `except Exception` would let panics escape and abort the
# whole holdout run. Catch it alongside Exception in the train-and-predict
# attempt so a single config's panic only rejects that candidate.
try:
    from polars.exceptions import PanicException as _PolarsPanicException
except ImportError:  # pragma: no cover - polars-version drift safety
    _PolarsPanicException = type("PanicException", (Exception,), {})

from case_studies.utils.backtest_explorer import BacktestExplorer
from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.backtest_runner import run_backtest
from case_studies.utils.carrier_pins import prioritize_carrier_hash
from case_studies.utils.conformal import (
    compute_holdout_conformal_widths,
    holdout_conformal_embargo_steps,
)
from case_studies.utils.registry import (
    register_backtest_run,
    register_prediction_metrics,
    register_prediction_set,
    register_training_run,
)
from utils.modeling import (
    RANDOM_SEED,
    load_configs,
    load_modeling_dataset,
    prepare_single_fold,
    resolve_linear_params,
    seed_everything,
)
from utils.paths import get_case_study_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_buffer_to_days(buffer: str) -> int:
    """Convert label buffer string (e.g., '21D', '8H') to days."""
    buf = buffer.strip()
    if buf.endswith(("D", "d")):
        return int(buf[:-1])
    if buf.upper().endswith("M") and not buf.endswith("min"):
        return int(buf[:-1]) * 30
    if buf.endswith(("H", "h")):
        return max(1, int(buf[:-1]) // 24)
    if buf.endswith(("T", "min")):
        n = int(buf.replace("min", "").replace("T", ""))
        return 1  # intraday → 1 day minimum
    return 0


def _delete_pre_registered_prediction(cs_id: str, prediction_hash: str) -> None:
    """Rollback a pre-registered conformal holdout pred_set after backtest failure.

    Removes (1) the prediction_metrics row, (2) the prediction_sets row, and
    (3) the on-disk predictions dir under ``run_log/predictions/<hash>/``.
    Used by the conformal_weighted fallback path in ``generate_holdout``.
    """
    import shutil

    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if db_path.exists():
        with contextlib.closing(sqlite3.connect(str(db_path))) as db:
            db.execute(
                "DELETE FROM prediction_metrics WHERE prediction_hash = ?",
                (prediction_hash,),
            )
            db.execute(
                "DELETE FROM fold_metrics WHERE prediction_hash = ?",
                (prediction_hash,),
            )
            db.execute(
                "DELETE FROM prediction_sets WHERE prediction_hash = ?",
                (prediction_hash,),
            )
            db.commit()
    pred_dir = case_dir / "run_log" / "predictions" / prediction_hash
    if pred_dir.exists():
        shutil.rmtree(pred_dir, ignore_errors=True)


EXCLUDED_FROM_HOLDOUT_SELECTION = frozenset({"benchmark"})


def _candidate_from_row(cs_id: str, row: dict) -> dict:
    """Hydrate a backtest_explorer row into a full candidate dict.

    Pulls training_hash, checkpoint_value, training_spec, and strategy_spec
    from the registry. Used by select_best_models to assemble each rank.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    pred_row = db.execute(
        "SELECT training_hash, checkpoint_value, checkpoint_kind, split "
        "FROM prediction_sets WHERE prediction_hash = ?",
        (row["prediction_hash"],),
    ).fetchone()

    train_row = db.execute(
        "SELECT spec_json, family, config_name FROM training_runs WHERE training_hash = ?",
        (pred_row["training_hash"],),
    ).fetchone()

    bt_row = db.execute(
        "SELECT spec_json FROM backtest_runs WHERE backtest_hash = ?",
        (row["backtest_hash"],),
    ).fetchone()

    db.close()

    return {
        "backtest_hash": row["backtest_hash"],
        "prediction_hash": row["prediction_hash"],
        "training_hash": pred_row["training_hash"],
        "checkpoint_value": pred_row["checkpoint_value"],
        "checkpoint_kind": pred_row["checkpoint_kind"],
        "family": train_row["family"],
        "config_name": train_row["config_name"],
        "training_spec": json.loads(train_row["spec_json"]),
        "strategy_spec": json.loads(bt_row["spec_json"]),
        "val_sharpe": row["sharpe"],
        "label": row.get("label") or dict(train_row).get("label", ""),
    }


HOLDOUT_SELECTION_STAGES: tuple[str, ...] = ("signal", "allocation", "risk_overlay")
"""Stages eligible for holdout rank-1 selection.

``cost_sensitivity`` is excluded: those rows are perturbation analyses (what
happens if costs are 2× our model?), not alternative strategies. ``benchmark``
family is excluded separately via ``EXCLUDED_FROM_HOLDOUT_SELECTION``.

This mirrors the trial-count basis the label-cohort ``cohort_metrics`` uses
for DSR / RAS / Reality Check / PBO selection-bias adjustments - see
``scripts/backfill_cohort_metrics.py::_list_label_cohorts``.
"""


def select_best_models(
    cs_id: str,
    *,
    top_n: int = 5,
    min_ic: float | None = None,
    families: list[str] | None = None,
    labels: list[str] | None = None,
) -> list[dict]:
    """Query registry for the top-N models by validation Sharpe across stages.

    Pools validation backtests across ``HOLDOUT_SELECTION_STAGES``
    (signal + allocation + risk_overlay) so the rank-1 reflects the best
    end-to-end strategy spec - selection, allocator, and risk overlay
    combined - not just the equal-weight baseline spec. Returns candidate dicts
    deduplicated by ``prediction_hash`` so each rank corresponds to a
    distinct trained model. For a given prediction_hash, the dedupe keeps
    the highest-Sharpe backtest row (i.e., its winning strategy_spec).

    The dedupe matters for the holdout-retrain fallback in ``generate_holdout``:
    if the rank-1 retrain produces degenerate (constant) predictions, the
    fallback must select a *different* trained model rather than a different
    strategy spec on the same model.

    Filters mirror ``select_best_model``: benchmark families excluded;
    optional families/labels/min_ic restrictions applied before ranking.
    """
    explorer = BacktestExplorer(cs_id)
    # Pull a generous pool from each stage so dedupe-by-prediction_hash
    # still yields top_n after the cross-stage merge. When ``labels``
    # restricts the eligible label set, the per-stage top_n must be much
    # larger because high-Sharpe rows on excluded labels (e.g.
    # sp500_options' inflated fwd_ret_10d) otherwise dominate the
    # explorer.best() top-N and crowd out the restricted-label rows
    # before the label filter applies post-concat.
    per_stage_n = max(top_n * 20, 5000 if labels else 200)
    stage_frames = []
    for stage in HOLDOUT_SELECTION_STAGES:
        df = explorer.best(stage=stage, top_n=per_stage_n)
        if not df.is_empty():
            stage_frames.append(df)
    if not stage_frames:
        raise ValueError(
            f"No validation backtests in stages {HOLDOUT_SELECTION_STAGES} for {cs_id}"
        )
    best_df = pl.concat(stage_frames, how="diagonal_relaxed").sort("sharpe", descending=True)

    best_df = best_df.filter(~pl.col("family").is_in(list(EXCLUDED_FROM_HOLDOUT_SELECTION)))
    if best_df.is_empty():
        raise ValueError(
            f"No non-benchmark backtests in stages {HOLDOUT_SELECTION_STAGES} for {cs_id}"
        )

    if families:
        best_df = best_df.filter(pl.col("family").is_in(families))
        if best_df.is_empty():
            raise ValueError(
                f"No backtests in stages {HOLDOUT_SELECTION_STAGES} for {cs_id} with families {families}"
            )

    if labels:
        best_df = best_df.filter(pl.col("label").is_in(labels))
        if best_df.is_empty():
            raise ValueError(
                f"No backtests in stages {HOLDOUT_SELECTION_STAGES} for {cs_id} with labels {labels}"
            )

    universe_pin = UNIVERSE_RESTRICTIONS.get(cs_id)
    if universe_pin and "universe_filter" in best_df.columns:
        best_df = best_df.filter(pl.col("universe_filter") == universe_pin)
        if best_df.is_empty():
            raise ValueError(
                f"No backtests in stages {HOLDOUT_SELECTION_STAGES} for {cs_id} "
                f"with universe_filter='{universe_pin}'"
            )

    if min_ic is not None:
        case_dir = get_case_study_dir(cs_id)
        db_path = case_dir / "run_log" / "registry.db"
        db = sqlite3.connect(str(db_path))
        ic_rows = db.execute(
            "SELECT prediction_hash, ic_mean FROM prediction_metrics WHERE ic_mean IS NOT NULL"
        ).fetchall()
        db.close()
        ic_map = {ph: val for ph, val in ic_rows}

        viable_rows = []
        for i in range(len(best_df)):
            ph = best_df[i, "prediction_hash"]
            ic = ic_map.get(ph)
            if ic is not None and ic > min_ic:
                viable_rows.append(i)

        if viable_rows:
            best_df = best_df[viable_rows]

    best_df = prioritize_carrier_hash(best_df, cs_id)

    seen_phashes: set[str] = set()
    candidates: list[dict] = []
    for row in best_df.iter_rows(named=True):
        ph = row["prediction_hash"]
        if ph in seen_phashes:
            continue
        seen_phashes.add(ph)
        candidates.append(_candidate_from_row(cs_id, row))
        if len(candidates) >= top_n:
            break

    if not candidates:
        raise ValueError(
            f"No viable candidates for {cs_id} after dedupe across stages "
            f"{HOLDOUT_SELECTION_STAGES}"
        )
    return candidates


def select_best_model(
    cs_id: str,
    *,
    min_ic: float | None = None,
    families: list[str] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Return the rank-1 equal-weight baseline model. Thin wrapper around select_best_models.

    Kept for backward compatibility with the Ch20 nb00 preview cell and any
    external callers. The holdout pipeline uses ``select_best_models`` to
    enable degeneracy-driven fallback to rank-2 / rank-3.
    """
    return select_best_models(cs_id, top_n=1, min_ic=min_ic, families=families, labels=labels)[0]


def _is_degenerate_predictions(
    predictions: pl.DataFrame, *, std_threshold: float = 1e-6
) -> tuple[bool, str]:
    """Return (is_degenerate, reason). Used by the holdout fallback loop.

    Degenerate cases:
    - ``y_score`` column has near-zero std (regression collapsed to constant)
    - ``y_score`` is all-NaN (training failed silently)
    - fewer than 2 distinct values across the full holdout window

    The threshold defaults to 1e-6 (one part per million of std). Below that,
    cross-sectional ranking ties everything; the resulting backtest is a
    machine-precision artifact rather than a real out-of-sample test.
    """
    s = predictions["y_score"]
    if s.null_count() == s.len():
        return True, "all_null"
    finite = s.drop_nulls()
    if finite.len() < 2:
        return True, f"insufficient_finite_values (n={finite.len()})"
    arr = finite.to_numpy()
    if not np.isfinite(arr).any():
        return True, "all_non_finite"
    pred_std = float(np.nanstd(arr))
    if pred_std < std_threshold:
        return True, f"constant_predictions (std={pred_std:.2e} < {std_threshold:.0e})"
    n_unique = int(pl.Series(arr).n_unique())
    if n_unique < 2:
        return True, f"single_unique_value (n_unique={n_unique})"
    return False, ""


def build_holdout_split(mds, setup: dict) -> dict:
    """Create holdout split matching the validation window length.

    The train window uses the same duration as the CV folds (rolling window),
    ending just before the holdout start (minus label buffer). This ensures
    the holdout retrain produces a model consistent with the validation
    protocol - same window size, same effective training distribution.
    """
    eval_cfg = setup["evaluation"]
    holdout_start = str(eval_cfg["holdout_start"])
    holdout_end = str(eval_cfg["holdout_end"])

    buffer = setup.get("labels", {}).get("buffer", "0D")
    buffer_days = _parse_buffer_to_days(buffer)

    hs = dt_date.fromisoformat(holdout_start)
    train_end = str(hs - timedelta(days=buffer_days))

    # Match the validation fold window length
    if mds.splits:
        first_split = mds.splits[0]
        fold_start = dt_date.fromisoformat(str(first_split["train_start"])[:10])
        fold_end = dt_date.fromisoformat(str(first_split["train_end"])[:10])
        window_days = (fold_end - fold_start).days
        te = dt_date.fromisoformat(train_end)
        train_start = str(te - timedelta(days=window_days))
    else:
        # Fallback: use all available data
        date_col = mds.date_col
        all_dates = mds.dataset[date_col].unique().sort()
        train_start = str(all_dates[0])[:10]

    return {
        "fold": 0,
        "train_start": train_start,
        "train_end": train_end,
        "val_start": holdout_start,
        "val_end": holdout_end,
    }


def load_existing_holdout(cs_id: str) -> dict:
    """Load existing holdout results from registry without retraining.

    Returns a result dict matching generate_holdout's return format,
    so the summary table always shows meaningful results.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return {"cs_id": cs_id, "skipped": True}

    with contextlib.closing(sqlite3.connect(str(db_path))) as db:
        db.row_factory = sqlite3.Row

        pred = db.execute(
            "SELECT ps.prediction_hash, ps.training_hash, ps.checkpoint_value, "
            "ps.checkpoint_kind, pm.ic_mean "
            "FROM prediction_sets ps "
            "LEFT JOIN prediction_metrics pm ON ps.prediction_hash = pm.prediction_hash "
            "WHERE ps.split = 'holdout' "
            "ORDER BY ps.created_at DESC LIMIT 1"
        ).fetchone()

        if not pred:
            return {"cs_id": cs_id, "skipped": True}

        train = db.execute(
            "SELECT family, config_name, spec_json FROM training_runs WHERE training_hash=?",
            (pred["training_hash"],),
        ).fetchone()

        bt = db.execute(
            "SELECT bm.sharpe, bm.cagr, bm.max_drawdown, br.backtest_hash "
            "FROM backtest_runs br "
            "JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash "
            "WHERE br.prediction_hash = ?",
            (pred["prediction_hash"],),
        ).fetchone()

        # Validation Sharpe for the SAME checkpoint that produced this
        # holdout (matched on training_hash + checkpoint_kind + checkpoint_value).
        # Training-only matching can drift to a sibling checkpoint of the
        # holdout's lineage; the strict scope keeps val_sharpe on the same
        # rung as the holdout. Linear/GBM rows have NULL checkpoint dims -
        # build the WHERE clause to honor that.
        ck = pred["checkpoint_kind"]
        cv = pred["checkpoint_value"]
        ckind_clause = "ps.checkpoint_kind IS NULL" if ck is None else "ps.checkpoint_kind = ?"
        cval_clause = "ps.checkpoint_value IS NULL" if cv is None else "ps.checkpoint_value = ?"
        params: list = [pred["training_hash"]]
        if ck is not None:
            params.append(ck)
        if cv is not None:
            params.append(cv)
        val_bt = db.execute(
            f"SELECT bm.sharpe FROM prediction_sets ps "
            f"JOIN backtest_runs br ON ps.prediction_hash = br.prediction_hash "
            f"JOIN backtest_metrics bm ON br.backtest_hash = bm.backtest_hash "
            f"WHERE ps.training_hash = ? AND ps.split = 'validation' "
            f"AND {ckind_clause} AND {cval_clause} "
            f"ORDER BY bm.sharpe DESC LIMIT 1",
            params,
        ).fetchone()
        val_sharpe = val_bt["sharpe"] if val_bt else float("nan")

        spec = json.loads(train["spec_json"]) if train else {}
        label = spec.get("label", "")

        return {
            "cs_id": cs_id,
            "family": train["family"] if train else "",
            "config_name": train["config_name"] if train else "",
            "label": label,
            "val_sharpe": val_sharpe,
            "holdout_ic": pred["ic_mean"] if pred["ic_mean"] is not None else float("nan"),
            "holdout_sharpe": bt["sharpe"] if bt else float("nan"),
            "holdout_cagr": bt["cagr"] if bt else float("nan"),
            "holdout_maxdd": bt["max_drawdown"] if bt else float("nan"),
            "prediction_hash": pred["prediction_hash"],
            "backtest_hash": bt["backtest_hash"] if bt else "",
            "elapsed_s": 0,
        }


def has_holdout_predictions(cs_id: str, *, top_n: int = 5) -> bool:
    """Check if any of the top-N validation candidates has holdout predictions.

    Returns True iff at least one of the top-N validation candidates'
    training_hash values appears in `prediction_sets` with split='holdout'.
    Returns False if a holdout exists but none of the top-N candidates'
    training_hashes match it - that signals the holdout is stale relative
    to the current validation sweep (e.g., the rank-1 reshuffled out of
    the top-N) and should be regenerated.

    The top-N envelope (rather than rank-1 only) accommodates
    `generate_holdout`'s degeneracy fallback: when rank-1's holdout retrain
    collapses, the accepted holdout's training_hash matches a rank-2/3/...
    candidate. Without the envelope, every subsequent run would see "no
    rank-1 holdout" and stack another holdout backtest in the registry.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return False

    try:
        candidates = select_best_models(cs_id, top_n=top_n)
    except ValueError:
        return False
    candidate_hashes = [c["training_hash"] for c in candidates]
    if not candidate_hashes:
        return False

    db = sqlite3.connect(str(db_path))
    placeholders = ",".join("?" for _ in candidate_hashes)
    count = db.execute(
        f"SELECT COUNT(*) FROM prediction_sets "
        f"WHERE split = 'holdout' AND training_hash IN ({placeholders})",
        candidate_hashes,
    ).fetchone()[0]
    db.close()
    return count > 0


def delete_holdout_predictions(cs_id: str) -> int:
    """Delete all existing holdout predictions and their backtests.

    Deletion order respects FK constraints: child tables first, then parents.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return 0

    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA foreign_keys=ON")
    rows = db.execute(
        "SELECT prediction_hash FROM prediction_sets WHERE split = 'holdout'"
    ).fetchall()
    deleted = 0
    for (ph,) in rows:
        bts = db.execute(
            "SELECT backtest_hash FROM backtest_runs WHERE prediction_hash=?", (ph,)
        ).fetchall()
        for (bh,) in bts:
            # Child tables first. backtest_paired_metrics has FK on
            # challenger_hash; we also clean rows where the holdout backtest
            # is referenced as benchmark_hash (no FK but logically stale).
            db.execute(
                "DELETE FROM backtest_paired_metrics WHERE challenger_hash=? OR benchmark_hash=?",
                (bh, bh),
            )
            db.execute("DELETE FROM backtest_fold_metrics WHERE backtest_hash=?", (bh,))
            db.execute("DELETE FROM backtest_metrics WHERE backtest_hash=?", (bh,))
            db.execute("DELETE FROM backtest_runs WHERE backtest_hash=?", (bh,))
        # Child tables of prediction_sets
        db.execute("DELETE FROM fold_metrics WHERE prediction_hash=?", (ph,))
        db.execute("DELETE FROM prediction_metrics WHERE prediction_hash=?", (ph,))
        db.execute("DELETE FROM prediction_sets WHERE prediction_hash=?", (ph,))
        deleted += 1
    db.commit()
    db.close()
    return deleted


# ---------------------------------------------------------------------------
# Family-specific training
# ---------------------------------------------------------------------------


def _train_linear(cs_id, mds, holdout_split, best) -> pl.DataFrame:
    """Train linear model on holdout split."""
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    seed_everything(RANDOM_SEED)
    configs = load_configs(cs_id, mds.label_col, family="linear")
    config = next(c for c in configs if c["config_name"] == best["config_name"])

    model_class_name = config.get("model_class", "Ridge")
    model_classes = {"Ridge": Ridge, "Lasso": Lasso, "ElasticNet": ElasticNet}
    model_class = model_classes[model_class_name]
    entity_col = mds.entity_cols[0] if mds.entity_cols else None
    fold_data = prepare_single_fold(
        mds.dataset,
        holdout_split,
        mds.feature_names,
        mds.label_col,
        mds.date_col,
        entity_col,
        temporal_by_fold=mds.temporal_by_fold,
        temporal_keys=mds.temporal_keys,
        temporal_feature_names=mds.temporal_feature_names,
    )
    if fold_data is None:
        raise RuntimeError(f"Empty holdout fold for {cs_id}")

    params = resolve_linear_params(config, fold_data["X_train"], fold_data["y_train"])
    model = model_class(**params)
    model.fit(fold_data["X_train"], fold_data["y_train"])
    y_pred = model.predict(fold_data["X_val"])

    return _assemble_predictions(fold_data, y_pred, mds, entity_col)


def _train_gbm(cs_id, mds, holdout_split, best) -> pl.DataFrame:
    """Train GBM model on holdout split.

    Handles regression, binary, and multiclass labels. For classification,
    labels are remapped to 0-indexed for LightGBM and predictions are
    converted to expected-value scores via _extract_gbm_score.
    """
    import lightgbm as lgb

    from case_studies.utils.gbm import (
        _extract_gbm_score,
        _make_lgb_native_params,
        _remap_labels_for_lgb,
    )
    from utils.modeling import detect_label_type

    seed_everything(RANDOM_SEED)
    # Use training spec from registry (not load_configs) to handle
    # cross-label models (e.g., regression objective on classification labels)
    spec = best["training_spec"]
    params = dict(spec.get("params", {}))
    n_trees = best["checkpoint_value"] or spec.get("max_iterations", 500)

    entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
    dataset_pd = mds.dataset.to_pandas()

    # Detect if classification label (multiclass needs remapping + score extraction)
    label_series = mds.dataset[mds.label_col]
    task_type, num_classes, class_values = detect_label_type(mds.label_col, label_series)
    is_classification = task_type == "classification" and bool(class_values)

    # For multiclass, ensure num_class is in params
    if is_classification and num_classes > 2:
        params["num_class"] = num_classes

    # Use native LightGBM API (handles classification correctly)
    native_params = _make_lgb_native_params(params, "cpu")
    native_params["metric"] = "None"
    native_params["verbosity"] = -1

    dates_series = dataset_pd[mds.date_col]
    split = holdout_split

    train_mask = (dates_series >= split["train_start"]) & (dates_series <= split["train_end"])
    val_mask = (dates_series >= split["val_start"]) & (dates_series <= split["val_end"])

    X_train = np.nan_to_num(
        dataset_pd.loc[train_mask, mds.feature_names].values.astype(np.float32), nan=0.0
    )
    y_train = np.nan_to_num(
        dataset_pd.loc[train_mask, mds.label_col].values.astype(np.float32), nan=0.0
    )
    X_val = np.nan_to_num(
        dataset_pd.loc[val_mask, mds.feature_names].values.astype(np.float32), nan=0.0
    )
    y_val = np.nan_to_num(
        dataset_pd.loc[val_mask, mds.label_col].values.astype(np.float32), nan=0.0
    )

    train_valid = ~np.isnan(dataset_pd.loc[train_mask, mds.label_col].values)
    val_valid = ~np.isnan(dataset_pd.loc[val_mask, mds.label_col].values)
    X_train, y_train = X_train[train_valid], y_train[train_valid]
    X_val, y_val = X_val[val_valid], y_val[val_valid]

    # Classification: remap labels to 0-indexed for LightGBM
    if is_classification:
        y_train_lgb, _ = _remap_labels_for_lgb(y_train.astype(int), class_values)
        y_val_lgb, _ = _remap_labels_for_lgb(y_val.astype(int), class_values)
    else:
        y_train_lgb = y_train
        y_val_lgb = y_val

    dtrain = lgb.Dataset(X_train, label=y_train_lgb, free_raw_data=True)
    model = lgb.train(native_params, dtrain, num_boost_round=n_trees)
    raw_pred = model.predict(X_val, num_iteration=n_trees)

    # Convert multiclass probabilities to expected value (continuous score)
    if is_classification and class_values:
        y_pred = _extract_gbm_score(raw_pred, class_values, len(X_val))
    else:
        y_pred = raw_pred

    val_meta = dataset_pd.loc[val_mask].iloc[val_valid.nonzero()[0]]
    val_dates = val_meta[mds.date_col].values
    val_entities = val_meta[entity_col].values if entity_col in dataset_pd.columns else None

    predictions = pl.DataFrame(
        {
            "timestamp": val_dates,
            "symbol": val_entities if val_entities is not None else ["unknown"] * len(y_val),
            "y_true": y_val.astype(np.float64),
            "y_score": y_pred.astype(np.float64),
            "fold_id": np.zeros(len(y_val), dtype=np.int32),
        }
    )

    del dataset_pd, X_train, y_train, X_val, y_val, model
    gc.collect()
    return predictions


def _train_dl(cs_id, mds, holdout_split, best) -> pl.DataFrame:
    """Train DL model on holdout split."""
    import tempfile

    from case_studies.utils.deep_learning import run_dl_cv

    seed_everything(RANDOM_SEED)
    configs = load_configs(cs_id, mds.label_col, family="deep_learning")
    config = next(c for c in configs if c["config_name"] == best["config_name"])

    target_epoch = best["checkpoint_value"]
    entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
    dataset_pd = mds.dataset.to_pandas()

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    with tempfile.TemporaryDirectory(prefix="holdout_dl_") as tmpdir:
        save_dir = Path(tmpdir) / "dl_output"
        result = run_dl_cv(
            dataset_pd,
            [holdout_split],
            configs=[config],
            n_features=len(mds.feature_names),
            feature_names=mds.feature_names,
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=entity_col,
            device=device,
            save_dir=save_dir,
            register=False,
            case_study=cs_id,
        )

    all_preds = result.get("all_predictions", pl.DataFrame())
    if all_preds.is_empty():
        raise RuntimeError(f"DL produced no predictions for {cs_id}")

    if "epoch" in all_preds.columns:
        available_epochs = sorted(all_preds["epoch"].unique().to_list())
        epoch = (
            target_epoch
            if target_epoch in available_epochs
            else result.get("best_epoch", available_epochs[-1])
        )
        predictions = all_preds.filter(pl.col("epoch") == epoch)
    else:
        predictions = result.get("predictions", all_preds)

    col_renames = {}
    if "timestamp" not in predictions.columns and mds.date_col in predictions.columns:
        col_renames[mds.date_col] = "timestamp"
    if "symbol" not in predictions.columns and entity_col in predictions.columns:
        col_renames[entity_col] = "symbol"
    if col_renames:
        predictions = predictions.rename(col_renames)

    if "fold_id" not in predictions.columns:
        predictions = predictions.with_columns(pl.lit(0).cast(pl.Int32).alias("fold_id"))

    select_cols = [
        c
        for c in ["timestamp", "symbol", "y_true", "y_score", "fold_id"]
        if c in predictions.columns
    ]
    predictions = predictions.select(select_cols)

    del dataset_pd
    gc.collect()
    return predictions


def _assemble_predictions(fold_data, y_pred, mds, entity_col) -> pl.DataFrame:
    """Build predictions DataFrame from fold_data and raw predictions."""
    meta_pl = fold_data["meta_pl"]
    timestamp_col = mds.date_col
    entity_col_name = entity_col or "symbol"

    timestamps = meta_pl[timestamp_col].to_list() if meta_pl is not None else fold_data["dates"]
    entities = (
        meta_pl[entity_col_name].to_list()
        if meta_pl is not None and entity_col_name in meta_pl.columns
        else (
            fold_data["entities"].tolist()
            if fold_data["entities"] is not None
            else ["unknown"] * len(y_pred)
        )
    )

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": entities,
            "y_true": fold_data["y_val"].astype(np.float64),
            "y_score": y_pred.astype(np.float64),
            "fold_id": np.zeros(len(y_pred), dtype=np.int32),
        }
    )


def _train_tabular_dl(cs_id, mds, holdout_split, best) -> pl.DataFrame:
    """Train TabM model on holdout split."""
    import tempfile

    from case_studies.utils.tabular_dl import run_tabm_cv

    seed_everything(RANDOM_SEED)
    # Use training spec from registry to find config
    spec = best["training_spec"]
    config = spec  # spec contains all params needed

    target_epoch = best["checkpoint_value"]
    entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
    dataset_pd = mds.dataset.to_pandas()

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    with tempfile.TemporaryDirectory(prefix="holdout_tabm_") as tmpdir:
        save_dir = Path(tmpdir) / "tabm_output"
        result = run_tabm_cv(
            dataset_pd,
            [holdout_split],
            configs=[config],
            n_features=len(mds.feature_names),
            feature_names=mds.feature_names,
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=entity_col,
            device=device,
            save_dir=save_dir,
            register=False,
            case_study=cs_id,
        )

    all_preds = result.get("all_predictions", pl.DataFrame())
    if all_preds.is_empty():
        raise RuntimeError(f"TabM produced no predictions for {cs_id}")

    if "epoch" in all_preds.columns:
        available_epochs = sorted(all_preds["epoch"].unique().to_list())
        epoch = (
            target_epoch
            if target_epoch in available_epochs
            else result.get("best_epoch", available_epochs[-1])
        )
        predictions = all_preds.filter(pl.col("epoch") == epoch)
    else:
        predictions = result.get("predictions", all_preds)

    col_renames = {}
    if "timestamp" not in predictions.columns and mds.date_col in predictions.columns:
        col_renames[mds.date_col] = "timestamp"
    if "symbol" not in predictions.columns and entity_col in predictions.columns:
        col_renames[entity_col] = "symbol"
    if col_renames:
        predictions = predictions.rename(col_renames)

    if "fold_id" not in predictions.columns:
        predictions = predictions.with_columns(pl.lit(0).cast(pl.Int32).alias("fold_id"))

    select_cols = [
        c
        for c in ["timestamp", "symbol", "y_true", "y_score", "fold_id"]
        if c in predictions.columns
    ]

    del dataset_pd
    gc.collect()
    return predictions.select(select_cols)


def _train_latent_factors(cs_id, mds, holdout_split, best) -> pl.DataFrame:
    """Train latent factor model on holdout split."""
    import tempfile

    from case_studies.utils.latent_factors import run_latent_factor_cv

    seed_everything(RANDOM_SEED)
    model_name = best["config_name"]  # e.g., "sae", "pca", "ipca"
    spec = best["training_spec"]
    n_factors = spec.get("params", {}).get("n_factors", 5)

    entity_col = mds.entity_cols[0] if mds.entity_cols else "symbol"
    target_epoch = best["checkpoint_value"] or 0
    persistent_entities = cs_id != "us_firm_characteristics"

    with tempfile.TemporaryDirectory(prefix="holdout_lf_") as tmpdir:
        save_dir = Path(tmpdir) / "lf_output"
        result = run_latent_factor_cv(
            panel_data=None,
            splits=[holdout_split],
            models=[model_name],
            n_factors=n_factors,
            n_epochs=50,
            save_dir=save_dir,
            use_cache=False,
            force_retrain=True,
            random_state=RANDOM_SEED,
            dataset=mds.dataset,
            feature_names=mds.feature_names,
            label_col=mds.label_col,
            date_col=mds.date_col,
            entity_col=entity_col,
            case_study_id=cs_id,
            eval_label_col=mds.eval_label_col,
            task_type=mds.task_type,
            class_values=mds.class_values or None,
            prediction_split="holdout",
            persistent_entities=persistent_entities,
        )

    all_preds = result.get("all_predictions", {}).get(model_name, pl.DataFrame())
    if all_preds.is_empty():
        raise RuntimeError(f"Latent factors produced no predictions for {cs_id}/{model_name}")

    # Filter to target epoch (neural models have multiple, PCA has epoch=0)
    if "epoch" in all_preds.columns:
        available_epochs = sorted(all_preds["epoch"].unique().to_list())
        epoch = target_epoch if target_epoch in available_epochs else available_epochs[-1]
        predictions = all_preds.filter(pl.col("epoch") == epoch)
    else:
        predictions = all_preds

    # Normalize columns
    col_renames = {}
    if "timestamp" not in predictions.columns and mds.date_col in predictions.columns:
        col_renames[mds.date_col] = "timestamp"
    if "symbol" not in predictions.columns and entity_col in predictions.columns:
        col_renames[entity_col] = "symbol"
    if col_renames:
        predictions = predictions.rename(col_renames)

    if "fold_id" not in predictions.columns:
        predictions = predictions.with_columns(pl.lit(0).cast(pl.Int32).alias("fold_id"))

    select_cols = [
        c
        for c in ["timestamp", "symbol", "y_true", "y_score", "fold_id"]
        if c in predictions.columns
    ]
    return predictions.select(select_cols)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


TRAIN_DISPATCH = {
    "linear": _train_linear,
    "gbm": _train_gbm,
    "deep_learning": _train_dl,
    "tabular_dl": _train_tabular_dl,
    "latent_factors": _train_latent_factors,
}


# Per-case-study label restrictions for Ch20 rank-1 selection. sp500_options
# has only one canonical label (`ret_to_expiry`) after the 2026-05-17 purge
# of the legacy fwd_ret_5d/10d/dh_5d/dh_10d sweep entries - those went
# through the vectorized backtest path which treats their 5d/10d forward
# returns as daily returns, inflating Sharpes to non-credible levels
# (e.g. fwd_ret_10d allocation Sharpe ~6.5). Only ret_to_expiry runs
# through the HTM daily-MTM dispatch that honors option bid-ask costs.
# Per-case-study label whitelist. Used by both the holdout retrain selector
# (this module) and the cluster-diagnostics rank-1 selector
# (01_aggregate_synthesis.py). Both consumers must agree, hence the single
# canonical definition here imported by 01_aggregate_synthesis.
LABEL_RESTRICTIONS: dict[str, frozenset[str]] = {
    "sp500_options": frozenset({"ret_to_expiry"}),
}


# Per-CS canonical universe pin: case_study -> strategy.signal.universe_filter
# value eligible to anchor the holdout-retrain rank-1. sp500_options trades only
# the liquid (bottom-quintile half-spread) subset; full-universe rows are kept
# for the Ch18 htm_cost_cascade comparison only and must never be selected as the
# deployed carrier (the full-universe round-trip spread consumes the VRP edge).
# Without this, full-universe allocation backtests from the standard sweep leak
# into the holdout selection by raw Sharpe. Mirrors
# case_studies/utils/strategy_analysis.py::UNIVERSE_RESTRICTIONS - keep in sync.
UNIVERSE_RESTRICTIONS: dict[str, str] = {
    "sp500_options": "liquid",
}


def generate_holdout(
    cs_id: str,
    *,
    force: bool = False,
    verbose: bool = True,
    max_fallback_attempts: int = 5,
) -> dict[str, Any]:
    """Generate holdout predictions and backtest for one case study.

    Selects the rank-1 equal-weight baseline model by validation Sharpe, retrains on
    all pre-holdout data, generates predictions, and runs the holdout
    backtest. If the rank-1's retrained predictions are degenerate (constant
    or all-NaN), the function falls back to rank-2, rank-3, etc., up to
    ``max_fallback_attempts``. The fallback ordinal and the list of
    rejected candidates are returned for the §6 audit trail.

    Parameters
    ----------
    cs_id : str
        Case study identifier (e.g., "etfs", "cme_futures").
    force : bool
        If True, delete existing holdout predictions before generating.
    verbose : bool
        If True, print progress messages.
    max_fallback_attempts : int
        Maximum candidates to try before raising. Defaults to 5.

    Returns
    -------
    dict
        Result with keys: cs_id, family, config_name, label,
        val_sharpe, selection_ordinal, fallback_attempts, holdout_ic,
        holdout_sharpe, holdout_cagr, holdout_maxdd, prediction_hash,
        backtest_hash, elapsed_s.

        ``selection_ordinal`` is 1 if the validation rank-1 produced
        non-degenerate predictions; 2+ if a fallback was used. The
        ``fallback_attempts`` list records each rejected candidate with
        its degeneracy reason.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if has_holdout_predictions(cs_id):
        if force:
            n = delete_holdout_predictions(cs_id)
            _log(f"  Deleted {n} existing holdout prediction(s)")
        else:
            _log("  Holdout predictions exist - loading from registry")
            return load_existing_holdout(cs_id)

    label_filter = LABEL_RESTRICTIONS.get(cs_id)
    candidates = select_best_models(
        cs_id,
        top_n=max_fallback_attempts,
        min_ic=None,
        labels=list(label_filter) if label_filter else None,
    )
    if label_filter:
        _log(f"  Label filter active: {label_filter}")
    _log(
        f"  Top-{len(candidates)} validation Sharpe candidates loaded "
        f"(rank-1: {candidates[0]['family']}/{candidates[0]['config_name']} "
        f"Sharpe={candidates[0]['val_sharpe']:.3f})"
    )

    setup_path = get_case_study_dir(cs_id) / "config" / "setup.yaml"
    setup = yaml.safe_load(setup_path.read_text())

    t0 = time.perf_counter()
    fallback_attempts: list[dict[str, Any]] = []
    mds = None
    last_label: str | None = None
    prices_cache: dict[str, pl.DataFrame] = {}

    def _load_prices_for(label: str, candidate_spec: dict | None = None) -> pl.DataFrame:
        """Load holdout prices, prepending validation-window history when the
        strategy uses a rolling-vol allocator (inverse_vol / risk_parity / MVO /
        HRP). Without burn-in, ``compute_*_weights`` returns no usable
        weights when the holdout window is shorter than ``vol_window``."""
        alloc = (candidate_spec or {}).get("allocation", {})
        vw = int(alloc.get("vol_window") or alloc.get("lookback") or 0)
        cache_key = f"{label}::warmup" if vw > 0 else label
        if cache_key in prices_cache:
            return prices_cache[cache_key]
        p = load_backtest_prices_for(cs_id, label, split="holdout")
        if vw > 0:
            # Prepend validation-window prices so the allocator's rolling
            # window has burn-in. The vectorized path slices port_ret to the
            # canonical holdout window after the fact, so warmup history is
            # only seen by the allocator - not the return aggregation.
            p_val = load_backtest_prices_for(cs_id, label, split="validation")
            # Cast columns to compatible dtypes before concat (some CSes load
            # different schemas for validation vs holdout).
            for col in ("timestamp", "symbol"):
                if col in p_val.columns and col in p.columns and p_val.schema[col] != p.schema[col]:
                    p_val = p_val.cast({col: p.schema[col]})
            common_cols = [c for c in p.columns if c in p_val.columns]
            p = (
                pl.concat(
                    [p_val.select(common_cols), p.select(common_cols)], how="vertical_relaxed"
                )
                .unique(subset=["timestamp", "symbol"])
                .sort(["timestamp", "symbol"])
            )
        price_ts = p.schema["timestamp"]
        if price_ts == pl.Date:
            p = p.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        elif hasattr(price_ts, "time_zone") and price_ts.time_zone:
            p = p.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        prices_cache[cache_key] = p
        return p

    bt_result = None
    prices = None

    for ordinal, candidate in enumerate(candidates, start=1):
        family = candidate["family"]
        label = candidate["training_spec"]["label"]

        if family in ("deep_learning", "tabular_dl", "latent_factors"):
            with contextlib.suppress(ImportError):
                import torch  # noqa: F811

        prefix = "  Trying" if ordinal == 1 else f"  Falling back to rank-{ordinal}"
        _log(
            f"{prefix}: {family}/{candidate['config_name']} "
            f"(val_sharpe={candidate['val_sharpe']:.3f}, label={label})"
        )

        # Reload modeling dataset only if label changed across fallback attempts
        if mds is None or label != last_label:
            if mds is not None:
                del mds
                gc.collect()
            mds = load_modeling_dataset(cs_id, label)
            last_label = label
            _log(f"    Dataset: {len(mds.dataset):,} rows, {len(mds.feature_names)} features")

        holdout_split = build_holdout_split(mds, setup)
        _log(
            f"    Split: train=[{holdout_split['train_start']}..{holdout_split['train_end']}] "
            f"holdout=[{holdout_split['val_start']}..{holdout_split['val_end']}]"
        )

        train_fn = TRAIN_DISPATCH.get(family)
        if train_fn is None:
            _log(f"    SKIP: unsupported family ({family})")
            fallback_attempts.append(
                {
                    "ordinal": ordinal,
                    "family": family,
                    "config_name": candidate["config_name"],
                    "label": label,
                    "val_sharpe": candidate["val_sharpe"],
                    "reason": f"unsupported_family ({family})",
                }
            )
            continue

        try:
            predictions = train_fn(cs_id, mds, holdout_split, candidate)
        except (Exception, _PolarsPanicException) as e:  # noqa: BLE001 - degenerate-config training failure
            _log(f"    REJECT: training raised {type(e).__name__}: {e}")
            fallback_attempts.append(
                {
                    "ordinal": ordinal,
                    "family": family,
                    "config_name": candidate["config_name"],
                    "label": label,
                    "val_sharpe": candidate["val_sharpe"],
                    "reason": f"training_error: {type(e).__name__}: {e}",
                }
            )
            continue

        _log(f"    Predictions: {len(predictions):,} rows")

        is_degenerate, reason = _is_degenerate_predictions(predictions)
        if is_degenerate:
            _log(f"    REJECT: {reason}")
            fallback_attempts.append(
                {
                    "ordinal": ordinal,
                    "family": family,
                    "config_name": candidate["config_name"],
                    "label": label,
                    "val_sharpe": candidate["val_sharpe"],
                    "reason": reason,
                }
            )
            del predictions
            gc.collect()
            continue

        # Backtest gate: even with non-degenerate predictions, the engine may
        # produce an empty result (e.g., monthly + classification empty port_ret).
        # Treat run_backtest failure as a fallback trigger.
        prices = _load_prices_for(
            label, candidate["strategy_spec"].get("strategy", candidate["strategy_spec"])
        )

        # If the candidate's allocator is conformal_weighted, the holdout
        # prediction_hash must be registered AND conformal widths written to
        # disk BEFORE run_backtest - the allocator loads widths keyed on the
        # holdout prediction_hash via load_conformal_widths. On any failure
        # we delete the pre-registered prediction set so the fallback loop
        # can try the next candidate cleanly.
        alloc_spec = candidate["strategy_spec"].get("strategy", {}).get("allocation", {})
        needs_pre_registration = alloc_spec.get("method") == "conformal_weighted"
        pre_registered_hash: str | None = None

        if needs_pre_registration:
            try:
                register_training_run(cs_id, spec=candidate["training_spec"])
                pre_registered_hash = register_prediction_set(
                    cs_id,
                    candidate["training_hash"],
                    checkpoint_value=candidate["checkpoint_value"],
                    checkpoint_kind=candidate["checkpoint_kind"],
                    split="holdout",
                    predictions=predictions,
                    task_type=mds.task_type,
                    class_values=mds.class_values or None,
                )
                embargo = holdout_conformal_embargo_steps(cs_id, label)
                alpha = float(alloc_spec.get("alpha", 0.20))
                widths = compute_holdout_conformal_widths(
                    cs_id,
                    candidate["prediction_hash"],
                    pre_registered_hash,
                    alpha=alpha,
                    embargo_steps=embargo,
                    write=True,
                )
                _log(
                    f"    Pre-registered {pre_registered_hash[:12]} + "
                    f"widths: {widths.height:,} rows, "
                    f"{widths['symbol'].n_unique()} symbols, "
                    f"median width={float(widths['width'].median()):.4f} "
                    f"(alpha={alpha}, embargo={embargo})"
                )
            except (Exception, _PolarsPanicException) as e:  # noqa: BLE001
                _log(f"    REJECT: conformal widths setup raised {type(e).__name__}: {e}")
                # Only the prediction set is rolled back. A training_runs row
                # written by register_training_run before register_prediction_set
                # failed is content-addressed by training_hash and idempotent:
                # with no prediction set or backtest referencing it, it is inert
                # and is reused (not duplicated) if this candidate is retried.
                if pre_registered_hash:
                    _delete_pre_registered_prediction(cs_id, pre_registered_hash)
                fallback_attempts.append(
                    {
                        "ordinal": ordinal,
                        "family": family,
                        "config_name": candidate["config_name"],
                        "label": label,
                        "val_sharpe": candidate["val_sharpe"],
                        "reason": f"conformal_widths_error: {type(e).__name__}: {e}",
                    }
                )
                del predictions
                gc.collect()
                continue

        try:
            bt_result = run_backtest(
                cs_id,
                pre_registered_hash or "",  # empty if not pre-registered
                candidate["strategy_spec"],
                prices=prices,
                predictions=predictions,
                label=label,
                register=False,
            )
        except (RuntimeError, ValueError) as e:
            # ValueError covers the conformal_weighted runtime-check path
            # (missing widths / unregistered prediction_hash) in case the
            # pre-registration branch above misses a corner.
            _log(f"    REJECT: backtest raised {type(e).__name__}: {e}")
            if pre_registered_hash:
                _delete_pre_registered_prediction(cs_id, pre_registered_hash)
            fallback_attempts.append(
                {
                    "ordinal": ordinal,
                    "family": family,
                    "config_name": candidate["config_name"],
                    "label": label,
                    "val_sharpe": candidate["val_sharpe"],
                    "reason": f"backtest_error: {type(e).__name__}: {e}",
                }
            )
            del predictions
            bt_result = None
            gc.collect()
            continue

        # Non-degenerate predictions AND backtest succeeded - use this candidate
        best = candidate
        best["_pre_registered_hash"] = pre_registered_hash
        _log(f"    Accepted: rank-{ordinal} predictions are non-degenerate and backtest succeeded.")
        break
    else:
        # No candidate produced viable predictions
        attempted = "\n".join(
            f"    rank-{a['ordinal']}: {a['family']}/{a['config_name']} "
            f"(val_sharpe={a['val_sharpe']:.3f}) - {a['reason']}"
            for a in fallback_attempts
        )
        raise RuntimeError(
            f"Holdout retrain failed for {cs_id}: all {len(fallback_attempts)} "
            f"top candidates produced degenerate predictions, training errors, "
            f"or backtest failures:\n{attempted}"
        )

    selection_ordinal = ordinal
    if selection_ordinal > 1:
        _log(
            f"  Fallback used: rank-{selection_ordinal} accepted after "
            f"{selection_ordinal - 1} rejection(s)."
        )

    entity_col = mds.entity_cols[0] if mds.entity_cols else None
    _entity = "symbol" if entity_col and "symbol" in predictions.columns else None
    holdout_ic = cross_sectional_ic(
        predictions,
        predictions,
        pred_col="y_score",
        ret_col="y_true",
        date_col="timestamp",
        entity_col=_entity,
        method="spearman",
        min_obs=5,
    )["ic_mean"]
    _log(f"  Holdout IC: {holdout_ic:+.4f}")

    metrics_payload: dict[str, Any] = {"ic_mean": holdout_ic}
    if selection_ordinal > 1:
        # Persist the fallback ordinal in the registry for the §6 audit trail.
        # Stored as a metric so it surfaces in prediction_metrics without a schema change.
        metrics_payload["holdout_selection_ordinal"] = float(selection_ordinal)

    if best.get("_pre_registered_hash"):
        # conformal_weighted path: prediction_set was already registered before
        # the backtest so the allocator could load widths. Just upsert any
        # remaining metrics (the auto-fold-metrics path during pre-registration
        # already populated headline IC).
        prediction_hash = best["_pre_registered_hash"]
        register_prediction_metrics(cs_id, prediction_hash, metrics_payload)
        _log(f"  Registered (pre-registered for conformal): {prediction_hash[:12]}")
    else:
        register_training_run(cs_id, spec=best["training_spec"])
        prediction_hash = register_prediction_set(
            cs_id,
            best["training_hash"],
            checkpoint_value=best["checkpoint_value"],
            checkpoint_kind=best["checkpoint_kind"],
            split="holdout",
            predictions=predictions,
            metrics=metrics_payload,
            task_type=mds.task_type,
            class_values=mds.class_values or None,
        )
        _log(f"  Registered: {prediction_hash[:12]}")

    trades_df = None
    if bt_result.engine_result and hasattr(bt_result.engine_result, "trades"):
        with contextlib.suppress(Exception):
            trades_df = bt_result.engine_result.to_trades_dataframe()

    # Warmup-leakage assertion (closes deferred review #2500 finding #2):
    # ``_load_prices_for`` prepends validation-window prices when the
    # rank-1 strategy uses a rolling-vol allocator so the allocator's
    # burn-in is data-driven, but the registered ``daily_returns`` must
    # only cover the canonical holdout window. The vectorized backtest
    # path in run_backtest slices to the prediction window so this
    # normally holds; the assertion makes the contract explicit and
    # surfaces silent leakage at register-time rather than letting the
    # next downstream consumer (cohort_metrics, paired_metrics) inherit
    # tainted returns.
    if bt_result.daily_returns is not None and not bt_result.daily_returns.is_empty():
        eval_cfg = setup.get("evaluation") or {}
        ho_start_str = eval_cfg.get("holdout_start")
        if not ho_start_str:
            raise AssertionError(
                f"Cannot run warmup-leakage assertion for cs={cs_id}: "
                f"evaluation.holdout_start missing from setup.yaml. The "
                f"assertion guards against silent leakage from the "
                f"warmup-prefix path; misconfiguration must not silently "
                f"disable it."
            )
        holdout_start_date = dt_date.fromisoformat(str(ho_start_str))
        min_ts = bt_result.daily_returns["timestamp"].min()
        if hasattr(min_ts, "date"):
            min_ts_date = min_ts.date()
        else:
            min_ts_date = min_ts
        if min_ts_date is not None and min_ts_date < holdout_start_date:
            raise AssertionError(
                f"Holdout backtest daily_returns leaks validation-window "
                f"data: min(timestamp)={min_ts_date} < "
                f"holdout_start={holdout_start_date} for cs={cs_id} "
                f"prediction_hash={prediction_hash[:12]}. "
                f"The warmup-prefix prices supplied to _load_prices_for "
                f"were not sliced back to the holdout window before "
                f"daily_returns was emitted."
            )

    backtest_hash = register_backtest_run(
        cs_id,
        prediction_hash,
        best["strategy_spec"],
        returns=bt_result.daily_returns,
        trades=trades_df,
        weights=bt_result.weights,
        metrics=bt_result.metrics,
    )

    elapsed = time.perf_counter() - t0
    holdout_sharpe = bt_result.metrics.get("sharpe", float("nan"))
    holdout_cagr = bt_result.metrics.get("cagr", float("nan"))
    holdout_maxdd = bt_result.metrics.get("max_drawdown", float("nan"))

    _log(
        f"  Backtest: Sharpe={holdout_sharpe:.3f}, "
        f"CAGR={holdout_cagr:.3f}, MaxDD={holdout_maxdd:.3f} "
        f"({elapsed:.0f}s)"
    )

    del mds, predictions, prices
    gc.collect()

    return {
        "cs_id": cs_id,
        "family": best["family"],
        "config_name": best["config_name"],
        "label": last_label,
        "val_sharpe": best["val_sharpe"],
        "selection_ordinal": selection_ordinal,
        "fallback_attempts": fallback_attempts,
        "holdout_ic": holdout_ic,
        "holdout_sharpe": holdout_sharpe,
        "holdout_cagr": holdout_cagr,
        "holdout_maxdd": holdout_maxdd,
        "prediction_hash": prediction_hash,
        "backtest_hash": backtest_hash,
        "elapsed_s": elapsed,
    }
