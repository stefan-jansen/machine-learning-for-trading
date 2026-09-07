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
import shutil
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

from case_studies.utils.backtest_loaders import load_backtest_prices_for
from case_studies.utils.backtest_runner import run_backtest
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
from case_studies.utils.registry.completeness import evaluate_prediction_coverage
from case_studies.utils.strategy_analysis import (
    LABEL_RESTRICTIONS,
    SELECTION_STAGES,
    UNIVERSE_RESTRICTIONS,
    NoSelectableCandidates,
    selectable_validation_candidates,
)
from utils.cv_splits import most_recent_split
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


def _candidate_from_row(cs_id: str, row: dict) -> dict:
    """Hydrate a :func:`selectable_validation_candidates` row into a retrain candidate.

    The row already carries the identities and the strategy spec the ranking read. What
    the retrain additionally needs is the checkpoint the prediction set sits on - the
    checkpoint is part of the configuration, so a replay that ignores it replays a
    different model - and the training specification the refit is built from.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        pred_row = db.execute(
            "SELECT checkpoint_value, checkpoint_kind FROM prediction_sets "
            "WHERE prediction_hash = ?",
            (row["prediction_hash"],),
        ).fetchone()
        train_row = db.execute(
            "SELECT spec_json FROM training_runs WHERE training_hash = ?",
            (row["training_hash"],),
        ).fetchone()
    finally:
        db.close()

    return {
        "backtest_hash": row["backtest_hash"],
        "prediction_hash": row["prediction_hash"],
        "training_hash": row["training_hash"],
        "checkpoint_value": pred_row["checkpoint_value"],
        "checkpoint_kind": pred_row["checkpoint_kind"],
        "family": row["family"],
        "config_name": row["config_name"],
        "training_spec": json.loads(train_row["spec_json"]),
        "strategy_spec": json.loads(row["spec_json"]),
        # The Sharpe the selection was made on, which is the common-support one wherever the
        # field was re-ranked. Reporting the stored value instead would have the two entry
        # points agree on the configuration and print different numbers for it, and chapter
        # 20 measures holdout decay against this.
        "val_sharpe": (
            row["comparison_sharpe"] if row["comparison_sharpe"] is not None else row["sharpe"]
        ),
        "label": row["label"] or "",
    }


HOLDOUT_SELECTION_STAGES: tuple[str, ...] = SELECTION_STAGES
"""Stages eligible for holdout rank-1 selection.

An alias for :data:`case_studies.utils.strategy_analysis.SELECTION_STAGES`, kept for
callers that import this name. The stage pool is a property of the selection rule
(``reference/CASE_STUDY_PIPELINE.md`` section 5), not of this module, and declaring it
twice is how the two holdout selectors came to disagree in the first place.
"""


def select_best_models(
    cs_id: str,
    *,
    top_n: int = 5,
    min_ic: float | None = None,
    families: list[str] | None = None,
    labels: list[str] | None = None,
) -> list[dict]:
    """The top-N distinct trained models by validation Sharpe, best first.

    The ranking is :func:`selectable_validation_candidates`, which is also what
    ``resolve_canonical_rank1_lineage`` and therefore every case study's own holdout
    notebook ranks. This function used to build its own pool - ``BacktestExplorer.best``
    per stage, concatenated and re-sorted - with its own eligibility filter and its own
    ordering, and the two were kept together by a comment. They disagreed. Measured on
    ``fx_pairs`` 2026-09-07: ``deep_learning/tcn`` on ``fwd_ret_21d`` carries two
    backtests tied at Sharpe 0.2639142245820113, this path answered ``9402978117e9`` and
    the canonical resolver answered ``56070f34dff1``. Two strategy specifications for one
    model, and the holdout replays the specification exactly, so the choice decided which
    strategy spent the case study's single holdout use.

    What is added here and belongs here, because it is about *retraining* rather than
    about which configuration the case study reports:

    * the dedupe by ``prediction_hash``, keeping each model's best-Sharpe backtest. The
      holdout retrain falls back to rank-2 when rank-1's refit produces degenerate
      predictions, and the fallback has to reach a different *trained model* rather than
      a different strategy spec on the same one;
    * ``min_ic``, a caller-supplied floor that no production path passes. IC selects
      nothing (``reference/CASE_STUDY_PIPELINE.md`` section 5); this only ever narrows a
      pool already ordered by Sharpe.

    ``families`` and ``labels`` narrow the pool and are passed straight through;
    ``labels`` replaces ``LABEL_RESTRICTIONS`` rather than adding to it.
    """
    candidates_df = selectable_validation_candidates(cs_id, labels=labels, families=families)

    if min_ic is not None:
        case_dir = get_case_study_dir(cs_id)
        db_path = case_dir / "run_log" / "registry.db"
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            ic_map = {
                prediction_hash: ic_mean
                for prediction_hash, ic_mean in db.execute(
                    "SELECT prediction_hash, ic_mean FROM prediction_metrics "
                    "WHERE ic_mean IS NOT NULL"
                ).fetchall()
            }
        finally:
            db.close()
        viable = [
            row
            for row in candidates_df
            if (ic := ic_map.get(row["prediction_hash"])) is not None and ic > min_ic
        ]
        if viable:
            candidates_df = viable

    seen_phashes: set[str] = set()
    candidates: list[dict] = []
    for row in candidates_df:
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

    # Match the validation fold window length. Fold windows differ in length, so
    # this reads the fold that ends nearest the holdout rather than a list position.
    if mds.splits:
        latest_split = most_recent_split(mds.splits)
        fold_start = dt_date.fromisoformat(str(latest_split["train_start"])[:10])
        fold_end = dt_date.fromisoformat(str(latest_split["train_end"])[:10])
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


def _align_expected_timestamps(expected: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """Put the expected keys in the predictions' timestamp representation.

    `evaluate_prediction_coverage` compares keys by casting them to strings, so a
    `Date` and a `Datetime("ms")` holding the same day compare as different keys -
    "2016-01-29" against "2016-01-29 00:00:00.000". Every family runner builds its
    expected frame from the frame it predicted into, so the two always agree there and
    this never arises; here the expectation comes from the dataset and the predictions
    come back through pandas, which promotes a date to a millisecond timestamp.

    Only the representation is aligned. A prediction on the wrong day still fails to
    join, which is the comparison this exists to make.
    """
    if "timestamp" not in predictions.columns:
        return expected
    target = predictions.schema["timestamp"]
    if expected.schema["timestamp"] == target:
        return expected
    return expected.with_columns(pl.col("timestamp").cast(target))


def _holdout_expected_keys(mds, holdout_split: dict) -> pl.DataFrame:
    """The prediction keys a holdout retrain is expected to cover.

    Built from the dataset and the split alone, before any model runs, so that
    ``evaluate_prediction_coverage`` compares the produced predictions against a
    declaration made independently of them. Taking the keys off the predictions
    themselves would make coverage complete by construction and record a fact about
    nothing.

    This is the widest set any family could answer for: every row inside the holdout
    window whose label is finite. `_train_linear` and `_train_gbm` predict exactly this
    set - they mask the window by date and drop the null labels. The sequence, latent
    factor and tabular runners answer for less of it by design: a gap-free lookback
    drops the first rows of each entity, an evaluation label may be missing where the
    training label is not, and the latent path can require an entity to persist across
    the window.

    So the caller adjudicates against this frame only what the frame can settle. A key
    outside the window, a key predicted twice, and a null or non-finite score are wrong
    whatever the family. Predicting fewer keys than the window holds is not, and is
    reported rather than charged to the candidate.
    """
    date_col = mds.date_col
    label_col = mds.label_col
    entity_col = mds.entity_cols[0] if mds.entity_cols else None
    day = pl.col(date_col).cast(pl.Utf8).str.slice(0, 10)
    label = pl.col(label_col).cast(pl.Float64, strict=False)
    window = mds.dataset.filter(
        (day >= str(holdout_split["val_start"])[:10])
        & (day <= str(holdout_split["val_end"])[:10])
        & label.is_not_null()
        & label.is_finite()
    )
    if window.is_empty():
        raise ValueError(
            f"holdout window [{holdout_split['val_start']}..{holdout_split['val_end']}] "
            f"holds no rows with a finite {label_col}"
        )
    symbol = (
        pl.col(entity_col).cast(pl.Utf8)
        if entity_col and entity_col in window.columns
        else pl.lit("unknown", dtype=pl.Utf8)
    )
    expected = window.select(
        symbol.alias("symbol"),
        pl.col(date_col).alias("timestamp"),
        pl.lit(int(holdout_split["fold"]), dtype=pl.Int32).alias("fold_id"),
    )
    if expected.n_unique(["symbol", "timestamp", "fold_id"]) != expected.height:
        raise ValueError(
            "holdout window produced duplicate expected prediction keys; the dataset is "
            "not one row per entity and date inside the window"
        )
    return expected


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

    # Both refusals mean the same thing here: nothing is currently selectable, so no
    # holdout can cover the current top-N. `ValueError` is this module's own refusal after
    # the dedupe; `NoSelectableCandidates` is the canonical selector's for an empty pool -
    # an initialised registry with no eligible validation backtest, a population that
    # publishes nothing, a carrier pin left over from an earlier sweep. Answering False
    # sends the caller to `generate_holdout`, which asks the same selector again without a
    # guard and reports whichever refusal applies from inside the driver's own handler.
    try:
        candidates = select_best_models(cs_id, top_n=top_n)
    except (ValueError, NoSelectableCandidates):
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


def _referencing_columns(
    db: sqlite3.Connection, parent_table: str, parent_column: str
) -> list[tuple[str, str]]:
    """Every ``(table, column)`` the schema declares as a foreign key into one column.

    Read from the schema rather than listed here. The hand-written list this replaces named
    `fold_metrics`, `prediction_metrics`, `backtest_metrics`, `backtest_fold_metrics` and
    `backtest_paired_metrics`, and omitted `prediction_coverage` and `cohort_metrics`, both
    of which have been foreign keys into these tables since. Every registered holdout hits
    that: `DELETE FROM prediction_sets` under `PRAGMA foreign_keys=ON` raises
    `sqlite3.IntegrityError: FOREIGN KEY constraint failed` while its coverage row stands.
    Reproduced on a copy of cme_futures' production registry, holdout 18d48c3b9cc2. A list
    that has to be maintained by hand will be wrong again; the schema cannot be.
    """
    tables = [
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    ]
    references = []
    for table in tables:
        for row in db.execute(f"PRAGMA foreign_key_list('{table}')").fetchall():
            if row[2] == parent_table and row[4] == parent_column:
                references.append((table, row[3]))
    return sorted(references)


def delete_holdout_predictions(cs_id: str) -> int:
    """Delete all existing holdout predictions and their backtests, rows and artifacts.

    Deletion order respects FK constraints: child tables first, then parents. Which tables
    those are is read from the schema - see :func:`_referencing_columns`.

    The artifact directories go with the rows. A prediction artifact is immutable per hash:
    `register_prediction_set` refuses a hash whose `predictions.parquet` is already on disk
    with a different digest, and the row it would have compared against is gone. Leaving the
    directories behind therefore turns a regenerated holdout that lands on the same hash and
    different content into an "immutable prediction artifact conflict" with nothing left in
    the registry to explain it - which is exactly the state `force=True` exists to clear.
    The directories are removed after the transaction commits, so a failed delete leaves both
    halves intact rather than the rows behind their files.
    """
    case_dir = get_case_study_dir(cs_id)
    db_path = case_dir / "run_log" / "registry.db"
    if not db_path.exists():
        return 0

    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA foreign_keys=ON")
    prediction_children = _referencing_columns(db, "prediction_sets", "prediction_hash")
    backtest_children = _referencing_columns(db, "backtest_runs", "backtest_hash")
    rows = db.execute(
        "SELECT prediction_hash FROM prediction_sets WHERE split = 'holdout'"
    ).fetchall()
    deleted = 0
    stale_dirs: list[Path] = []
    try:
        for (ph,) in rows:
            stale_dirs.append(case_dir / "run_log" / "predictions" / ph)
            bts = db.execute(
                "SELECT backtest_hash FROM backtest_runs WHERE prediction_hash=?", (ph,)
            ).fetchall()
            for (bh,) in bts:
                stale_dirs.append(case_dir / "run_log" / "backtest" / bh)
                for table, column in backtest_children:
                    if table == "backtest_runs":
                        continue
                    db.execute(f"DELETE FROM {table} WHERE {column}=?", (bh,))
                # No foreign key declares it, but a paired metric benchmarked against a
                # deleted holdout backtest is as stale as one challenging it.
                db.execute("DELETE FROM backtest_paired_metrics WHERE benchmark_hash=?", (bh,))
                db.execute("DELETE FROM backtest_runs WHERE backtest_hash=?", (bh,))
            for table, column in prediction_children:
                if table == "backtest_runs":
                    continue
                db.execute(f"DELETE FROM {table} WHERE {column}=?", (ph,))
            db.execute("DELETE FROM prediction_sets WHERE prediction_hash=?", (ph,))
            deleted += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    for stale in stale_dirs:
        shutil.rmtree(stale, ignore_errors=True)
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

    # `force` deletes unconditionally. Gating the delete on `has_holdout_predictions`
    # skipped it exactly when it was needed: that check looks for a holdout tied to one of
    # the *current* validation top-N, and its own docstring says it returns False when a
    # holdout exists but no top-N candidate matches it - which is the definition of stale.
    # So a holdout whose training fell out of the top-N after a sweep reshuffle survived
    # `force=True`, and the new holdout was written beside it, two holdouts for one case
    # study. Measured on nasdaq100_microstructure: bf76fac27013 (training b8add3b63794)
    # coexisted with 6f95e10992fe and had to be deleted by hand.
    if force:
        n = delete_holdout_predictions(cs_id)
        _log(f"  Deleted {n} existing holdout prediction(s)")
    elif has_holdout_predictions(cs_id):
        _log("  Holdout predictions exist - loading from registry")
        return load_existing_holdout(cs_id)

    # The restriction is applied inside the selection, not passed in here. Passing it
    # made this call site look like the place the rule lives, and `has_holdout_predictions`
    # - which calls the same selector without it - then ranked over a wider label set than
    # the run it is reporting on.
    label_filter = LABEL_RESTRICTIONS.get(cs_id)
    candidates = select_best_models(cs_id, top_n=max_fallback_attempts, min_ic=None)
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

        # Declared before the model runs, and outside every rejection handler below: a
        # failure to state what the holdout should cover is a defect in this driver, not
        # evidence about the candidate, and must not be charged to one as a fallback.
        expected_keys = _holdout_expected_keys(mds, holdout_split)
        _log(f"    Expected holdout keys: {expected_keys.height:,}")

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

        # Coverage is adjudicated here rather than inside `register_prediction_set`,
        # because a candidate whose predictions are malformed is a candidate to fall back
        # from, and reaching registration with one would end the run instead. Registering
        # below can then treat any failure of its own as the infrastructure error it is.
        #
        # Only what the window can settle is charged to the candidate. Predicting fewer
        # keys than the window holds is what the sequence, latent-factor and tabular
        # runners do by design, and rejecting them for it would demote the selection for
        # a reason that is not about the model - the defect this whole path was fixed for,
        # arriving from the other side.
        expected_keys = _align_expected_timestamps(expected_keys, predictions)
        coverage = evaluate_prediction_coverage(expected_keys, predictions)
        malformed = (
            coverage.n_extra or coverage.n_duplicates or coverage.n_null or coverage.n_non_finite
        )
        if malformed:
            _log(f"    REJECT: holdout predictions are malformed: {coverage.as_dict()}")
            fallback_attempts.append(
                {
                    "ordinal": ordinal,
                    "family": family,
                    "config_name": candidate["config_name"],
                    "label": label,
                    "val_sharpe": candidate["val_sharpe"],
                    "reason": (
                        f"malformed_predictions: {coverage.n_extra} keys outside the "
                        f"holdout window, {coverage.n_duplicates} predicted twice, "
                        f"{coverage.n_null} null and {coverage.n_non_finite} non-finite "
                        f"scores, against {coverage.n_expected} eligible keys"
                    ),
                }
            )
            del predictions
            gc.collect()
            continue

        if coverage.n_missing:
            # Not a rejection, and not something to register either.
            #
            # `register_backtest_run` requires `prediction_coverage.status = 'complete'`
            # (registration.py:1176), so a short holdout registered with `allow_partial`
            # produces a prediction set whose backtest is then refused - the run fails
            # later and further from the cause. And falling back would demote the
            # selection on evidence this path cannot read: it cannot tell a family whose
            # runner narrows the window by design - a gap-free lookback, an evaluation
            # label missing where the training label is not, an entity required to
            # persist - from a fit that is genuinely short.
            #
            # The holdout is used once, so an ambiguity about which candidate deserves it
            # stops here and asks, rather than resolving itself quietly. Supporting these
            # families means their runner declaring its own eligible key frame, which is
            # the open half of ml4t/agent-workspace#938.
            raise RuntimeError(
                f"{family}/{candidate['config_name']} predicted "
                f"{coverage.n_actual:,} of the {coverage.n_expected:,} keys eligible in "
                f"the holdout window, {coverage.n_missing:,} short. This path declares the "
                "window itself as the expectation, which is exact for the linear and GBM "
                "runners and too wide for any family that narrows eligibility. It cannot "
                "tell that case from a short fit, and the holdout is spent once, so it "
                "stops rather than falling back to another candidate. The runner has to "
                "declare its own eligible keys - ml4t/agent-workspace#938."
            )

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
            # Registration sits outside the handler below. It failed here once for a
            # reason that was nothing to do with the candidate - the call omitted
            # `expected_keys`, which every versioned training identity requires - and the
            # handler recorded that as two candidates rejected and handed the holdout to
            # rank-3. An error the registry raises about the call is not evidence about
            # the model, and must stop the run rather than demote the selection.
            register_training_run(cs_id, spec=candidate["training_spec"])
            pre_registered_hash = register_prediction_set(
                cs_id,
                candidate["training_hash"],
                checkpoint_value=candidate["checkpoint_value"],
                checkpoint_kind=candidate["checkpoint_kind"],
                split="holdout",
                predictions=predictions,
                expected_keys=expected_keys,
                task_type=mds.task_type,
                class_values=mds.class_values or None,
            )
            try:
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
                # Only the prediction set is rolled back - the one registered just
                # above, which the widths were to be keyed on. The `training_runs` row
                # written with it is content-addressed by training_hash and idempotent:
                # with no prediction set or backtest referencing it, it is inert and is
                # reused rather than duplicated if this candidate is retried.
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
            expected_keys=expected_keys,
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
