"""Canonical (cs, label, split) windows for backtest slicing.

Single source of truth for the date range a backtest's daily-returns parquet
should cover.

Validation window = union of CV fold val_starts/val_ends → (min start, max end).
Holdout window    = setup.yaml.evaluation.{holdout_start, holdout_end}.

Same window for every strategy on the same (cs, label, split). Idempotent —
re-running ``run_backtest`` on a sliced parquet is a no-op.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml

from utils.paths import get_case_study_dir

Split = Literal["validation", "holdout"]

_LEGACY_TEMPORAL_FOLD_ROUTES = {
    "cme_futures": ("primary_label", False),
    "crypto_perps_funding": ("primary_label", True),
    "etfs": ("primary_label", True),
    "fx_pairs": ("primary_label", False),
    "nasdaq100_microstructure": ("primary_label", True),
    "sp500_equity_option_analytics": ("primary_label", True),
    "sp500_options": ("financial", False),
    "us_equities_panel": ("primary_label", True),
}
_TEMPORAL_FOLD_FIELDS = ("fold", "train_start", "train_end", "val_start", "val_end")


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


def _derive_modeling_splits(case_study: str, label: str) -> list[dict] | None:
    """Derive complete canonical modeling folds from the label timeline.

    Returns ``None`` when the label artifact itself is missing (no folds derivable).
    Misconfiguration - missing ``label_buffer`` for a label that *does*
    have a parquet - raises ``ValueError`` with the same actionable hint
    as :func:`utils.modeling.load_modeling_dataset` (loud-fail contract:
    config drift must surface as an error, not silently degrade to a
    predictions-min/max fallback downstream).

    Reads only the label parquet's time column to derive folds.
    ``generate_cv_splits`` is calendar-aware and uses only unique
    timestamps (see its docstring). Schema is introspected to pick
    ``timestamp`` else ``date`` (matching ``load_us_equities`` /
    ``load_sp500_daily_bars``); the full feature/label/temporal join
    is not needed.
    """
    import logging

    import polars as pl

    from utils.artifact_specs import (
        load_label_spec,
        resolve_label_buffer,
        resolve_label_horizon,
        resolve_storage_path,
    )
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

    ts_df = cast(
        pl.DataFrame,
        pl.scan_parquet(label_path).select(date_col).unique().sort(date_col).collect(),
    )
    splits = generate_cv_splits(
        ts_df,
        case_study_id=case_study,
        label_buffer=label_buffer,
        outcome_horizon=resolve_label_horizon(case_study, label, setup),
        date_col=date_col,
    )
    return splits or None


@lru_cache(maxsize=128)
def _fold_splits(case_study: str, label: str) -> tuple[tuple[int, date, date], ...] | None:
    """All CV folds as ``((fold_id, val_start, val_end), ...)``."""
    splits = _derive_modeling_splits(case_study, label)
    if splits is None:
        return None
    return tuple(
        (int(split["fold"]), _to_date(split["val_start"]), _to_date(split["val_end"]))
        for split in splits
    )


def modeling_fold_boundaries(case_study: str, label: str) -> list[dict] | None:
    """Return complete train and validation boundaries for feature producers."""
    splits = _derive_modeling_splits(case_study, label)
    if splits is None:
        return None
    return [
        {
            "fold": int(split["fold"]),
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "val_start": split["val_start"],
            "val_end": split["val_end"],
        }
        for split in splits
    ]


def _validated_temporal_folds(raw_folds: Any, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError(f"{source} has no temporal fold geometry")
    folds: list[dict[str, Any]] = []
    for raw in raw_folds:
        if not isinstance(raw, dict) or any(field not in raw for field in _TEMPORAL_FOLD_FIELDS):
            raise ValueError(f"{source} has invalid temporal fold geometry")
        folds.append(
            {
                field: int(raw[field]) if field == "fold" else raw[field]
                for field in _TEMPORAL_FOLD_FIELDS
            }
        )
    if len({fold["fold"] for fold in folds}) != len(folds):
        raise ValueError(f"{source} has duplicate temporal fold ids")
    return folds


def temporal_artifact_fold_boundaries(
    case_study: str,
    primary_label: str,
    artifact_path: str | Path,
) -> list[dict[str, Any]]:
    """Load producer fold metadata or mirror the locked Stage 04 producer route."""
    from case_studies.utils.artifact_digest import sidecar_path
    from utils.artifact_specs import (
        load_feature_spec,
        load_label_spec,
        resolve_label_buffer,
        resolve_label_horizon,
    )
    from utils.cv_splits import generate_cv_splits

    artifact_path = Path(artifact_path)
    metadata_path = sidecar_path(artifact_path)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if "fold_geometry" in metadata:
            return _validated_temporal_folds(
                metadata["fold_geometry"],
                source=str(metadata_path),
            )

    route = _LEGACY_TEMPORAL_FOLD_ROUTES.get(case_study)
    if route is None:
        raise ValueError(
            f"no verified legacy temporal-fold resolver exists for case study {case_study!r}"
        )
    timeline_source, include_outcome_horizon = route
    setup = _load_setup_yaml(case_study) or {}
    label_buffer = resolve_label_buffer(case_study, primary_label, setup)
    if not label_buffer:
        raise ValueError(f"no label buffer configured for primary label {primary_label!r}")

    if timeline_source == "financial":
        timeline_spec = load_feature_spec(case_study, "financial")
        fallback = "features/financial.parquet"
    else:
        timeline_spec = load_label_spec(case_study, primary_label)
        fallback = f"labels/{primary_label}.parquet"
    relative_path = str((timeline_spec or {}).get("storage", {}).get("path", fallback))
    timeline_path = artifact_path.parent.parent / relative_path
    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)

    import polars as pl

    schema = pl.scan_parquet(timeline_path).collect_schema()
    date_col = "timestamp" if "timestamp" in schema else "date"
    timeline = cast(
        pl.DataFrame,
        pl.scan_parquet(timeline_path).select(date_col).unique().collect(),
    )
    split_kwargs: dict[str, Any] = {
        "case_study_id": case_study,
        "label_buffer": label_buffer,
        "date_col": date_col,
    }
    if include_outcome_horizon:
        split_kwargs["outcome_horizon"] = resolve_label_horizon(case_study, primary_label, setup)
    folds = generate_cv_splits(timeline, **split_kwargs)
    return _validated_temporal_folds(
        folds,
        source=f"legacy Stage 04 route for {case_study}",
    )


def configured_labels(case_study: str) -> list[str]:
    """Every label the case study configures, primary first."""
    setup = _load_setup_yaml(case_study) or {}
    labels = setup.get("labels", {}) or {}
    primary = labels.get("primary")
    variants = list(labels.get("variants", []) or [])
    return ([primary] if primary else []) + [v for v in variants if v != primary]


def assert_variant_folds_are_out_of_sample(
    case_study: str,
    primary_label: str,
    *,
    variants: Sequence[str] | None = None,
) -> list[dict]:
    """Every variant label's fold F must validate after the primary's fold F stops training.

    A stage-04 artifact carries one fold set, built on the primary label's geometry,
    and a model trained on a variant label reads that artifact by ``fold`` id. So the
    values it reads for fold F were fit on data through the **primary** label's
    ``train_end``. The variant's own validation window has to start after that, or the
    model is scored on sessions the features already saw:

        variant.val_start > primary.train_end,  per fold, per configured variant

    Nothing checked this. Measured across the eight stage-04 case studies on
    2026-08-09, 20 variant labels, zero violations, so no artifact leaks today; what
    was missing is anything that would notice if a label's buffer changed. Of the
    two notebooks that assert something, only sp500_equity_option_analytics tests
    this property; fx_pairs compares the outer span
    (``variant.train_start >= primary.train_start and variant.val_end <= primary.val_end``),
    which never reads ``val_start`` and would pass on a geometry that leaks.

    **Compares timestamps, never dates.** At date granularity
    nasdaq100_microstructure fold 0 reads as a violation, 2020-12-29 against
    2020-12-29; the bars are minutes, the fit closes at 15:22 and the variant's
    validation opens at 15:38.

    Returns one row per (label, fold) with the gap, so a notebook can display what it
    just asserted. A variant with no label parquet is skipped, which is what the
    producers do with it too.
    """
    primary = _derive_modeling_splits(case_study, primary_label)
    if primary is None:
        raise ValueError(
            f"No folds derivable for the primary label '{primary_label}' of "
            f"'{case_study}', so no variant geometry can be checked against it."
        )
    by_fold = {int(s["fold"]): s for s in primary}

    if variants is None:
        variants = [x for x in configured_labels(case_study) if x != primary_label]

    rows: list[dict] = []
    violations: list[str] = []
    for label in variants:
        splits = _derive_modeling_splits(case_study, label)
        if splits is None:
            continue
        for split in splits:
            fold = int(split["fold"])
            if fold not in by_fold:
                violations.append(f"{label} fold {fold} has no counterpart in {primary_label}")
                continue
            fit_end = pd.Timestamp(by_fold[fold]["train_end"])
            val_start = pd.Timestamp(split["val_start"])
            rows.append(
                {
                    "label": label,
                    "fold": fold,
                    f"{primary_label} train_end": fit_end,
                    "val_start": val_start,
                    "gap": val_start - fit_end,
                }
            )
            if val_start <= fit_end:
                violations.append(
                    f"{label} fold {fold}: validation opens {val_start}, and the features "
                    f"for that fold were fit through {fit_end}"
                )

    if violations:
        raise AssertionError(
            "the artifact's per-fold estimation windows reach into a variant label's "
            "validation span, so a model reading fold F by id is scored on sessions its "
            "features already saw: " + "; ".join(violations)
        )
    return rows


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
