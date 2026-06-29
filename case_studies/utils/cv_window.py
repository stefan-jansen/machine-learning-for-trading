"""Canonical (cs, label, split) windows for backtest slicing.

Single source of truth for the date range a backtest's daily-returns parquet
should cover.

Validation window = union of CV fold val_starts/val_ends → (min start, max end).
Holdout window    = setup.yaml.evaluation.{holdout_start, holdout_end}.

Same window for every strategy on the same (cs, label, split). Idempotent —
re-running ``run_backtest`` on a sliced parquet is a no-op.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from utils.paths import get_case_study_dir

Split = Literal["validation", "holdout"]


def _to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return datetime.fromisoformat(str(x)[:19]).date()


@lru_cache(maxsize=32)
def _load_setup_yaml(case_study: str) -> dict | None:
    """Cached parse of ``case_studies/<cs>/config/setup.yaml``.

    Called from both ``_fold_splits`` and ``_holdout_window`` (and
    transitively from ``generate_cv_splits → load_evaluation_config``);
    the file is small but the parse repeated O(K_labels × N_runs) times
    is wasted work. Returns ``None`` when the file is absent.
    """
    setup_path = get_case_study_dir(case_study) / "config" / "setup.yaml"
    if not setup_path.exists():
        return None
    return yaml.safe_load(setup_path.read_text())


@lru_cache(maxsize=128)
def _fold_splits(case_study: str, label: str) -> tuple[tuple[int, date, date], ...] | None:
    """All CV folds as ((fold_id, val_start, val_end), ...) — same source canonical_window uses.

    Returns tuple-of-tuples (immutable, hashable) for lru_cache friendliness.
    ``None`` when the label artifact itself is missing (no folds derivable).
    Misconfiguration — missing ``label_buffer`` for a label that *does*
    have a parquet — raises ``ValueError`` with the same actionable hint
    as :func:`utils.modeling.load_modeling_dataset` (loud-fail contract:
    config drift must surface as an error, not silently degrade to a
    predictions-min/max fallback downstream).

    Reads only the label parquet's time column to derive folds —
    ``generate_cv_splits`` is calendar-aware and uses only unique
    timestamps (see its docstring). Schema is introspected to pick
    ``timestamp`` else ``date`` (matching ``load_us_equities`` /
    ``load_sp500_daily_bars``); the full feature/label/temporal join
    is not needed.
    """
    import logging

    import polars as pl

    from utils.artifact_specs import load_label_spec, resolve_label_buffer, resolve_storage_path
    from utils.cv_splits import generate_cv_splits

    logger = logging.getLogger(__name__)

    try:
        label_spec = load_label_spec(case_study, label)
        label_path = resolve_storage_path(case_study, label_spec, f"labels/{label}.parquet")
    except (KeyError, FileNotFoundError) as e:
        logger.debug("_fold_splits(%s, %s): no folds (%s)", case_study, label, e)
        return None

    if not label_path.exists():
        logger.debug(
            "_fold_splits(%s, %s): label parquet missing at %s",
            case_study,
            label,
            label_path,
        )
        return None

    setup = _load_setup_yaml(case_study)
    label_buffer = resolve_label_buffer(case_study, label, setup)
    if not label_buffer:
        raise ValueError(
            f"No explicit label buffer found for '{label}' in "
            f"case_studies/{case_study}/config/setup.yaml. "
            f"Add buffer to labels.buffer (primary) or labels.variant_buffers (variants)."
        )

    schema_names = pl.scan_parquet(label_path).collect_schema().names()
    if "timestamp" in schema_names:
        date_col = "timestamp"
    elif "date" in schema_names:
        date_col = "date"
    else:
        raise ValueError(
            f"Label parquet for '{label}' in case_study '{case_study}' "
            f"({label_path}) has neither 'timestamp' nor 'date' column. "
            f"Found: {schema_names}. Canonical schema requires 'timestamp'."
        )

    ts_df = pl.scan_parquet(label_path).select(date_col).unique().sort(date_col).collect()
    splits = generate_cv_splits(
        ts_df,
        case_study_id=case_study,
        label_buffer=label_buffer,
        date_col=date_col,
    )
    out = tuple((i, _to_date(s["val_start"]), _to_date(s["val_end"])) for i, s in enumerate(splits))
    return out if out else None


def _validation_window_for_label(case_study: str, label: str) -> tuple[date, date] | None:
    """Validation window from CV folds: min(val_start) → max(val_end)."""
    splits = _fold_splits(case_study, label)
    if splits is None:
        return None
    starts = [s[1] for s in splits]
    ends = [s[2] for s in splits]
    return min(starts), max(ends)


def fold_boundaries(case_study: str, label: str) -> list[dict] | None:
    """Public accessor for fold boundaries: ``[{fold, val_start, val_end}, ...]``.

    Use this instead of calling ``generate_cv_splits(daily_returns, ...)`` when
    daily_returns is val-only — that path fails for case studies whose CV config
    requires a train_size larger than the val window (e.g., ``train_size=10Y``
    on an 8-year validation period). Same source as the canonical_window
    helper, so fold IDs are stable.
    """
    splits = _fold_splits(case_study, label)
    if splits is None:
        return None
    return [{"fold": fold, "val_start": vs, "val_end": ve} for fold, vs, ve in splits]


@lru_cache(maxsize=32)
def _holdout_window(case_study: str) -> tuple[date, date] | None:
    setup = _load_setup_yaml(case_study)
    if setup is None:
        return None
    e = setup.get("evaluation", {})
    hs, he = e.get("holdout_start"), e.get("holdout_end")
    if hs is None or he is None:
        return None
    return _to_date(hs), _to_date(he)


def canonical_window(
    case_study: str,
    label: str,
    *,
    split: Split = "validation",
) -> tuple[date, date] | None:
    """Per-(cs, label, split) date range for the daily_returns parquet.

    Returns None if not derivable (no CV folds for the label, no holdout
    configured, etc.). Callers must handle None as "skip the slice".
    """
    if split == "holdout":
        return _holdout_window(case_study)
    return _validation_window_for_label(case_study, label)


def lookup_split(case_study: str, prediction_hash: str) -> Split | None:
    """Resolve the split a prediction_hash belongs to from the registry.

    Returns None when:
      - the registry DB doesn't exist
      - the prediction_hash isn't found
      - the stored split value isn't one of {"validation", "holdout"}
        (NULL, empty string, or schema-drift values like "oos") — caller must
        decide how to handle, never assume "validation".
    """
    db = get_case_study_dir(case_study) / "run_log" / "registry.db"
    if not db.exists():
        return None
    con = sqlite3.connect(str(db))
    try:
        row = con.execute(
            "SELECT split FROM prediction_sets WHERE prediction_hash = ?",
            (prediction_hash,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    if row[0] == "holdout":
        return "holdout"
    if row[0] == "validation":
        return "validation"
    return None
