"""Predictions cache for chapter teaching notebooks.

Cache long-form prediction frames keyed by a content-addressed spec hash so
re-running visualization / interpretation cells doesn't require re-training
the upstream stages. Cache files live under

    {chapter_dir}/output/predictions/{notebook_id}/{spec_hash}.parquet

which is gitignored.

Frame schema (required columns):

    date    -- value identifying the prediction period
    symbol  -- value identifying the asset
    y_pred  -- float, model prediction
    y_true  -- float, realised forward return

Optional column:

    forecaster -- string label, when one notebook stacks multiple forecasters
                  (e.g. Constant / AR(1) / EWMA) into a single frame.

Typical use::

    from utils.predictions_cache import load_predictions, save_predictions

    spec = {
        "data": {"source": "etfs", "start": START_DATE, "end": END_DATE,
                 "max_symbols": MAX_SYMBOLS},
        "model": {"name": "rp_pca", "n_factors": N_FACTORS,
                  "focus_gamma": focus_gamma},
        "forecasters": ["Constant", "AR(1)", "EWMA"],
    }
    cached = load_predictions(chapter=14, notebook_id="rp_pca", spec=spec)
    if cached is None:
        # ... expensive Stage 1-3 work, build long-form `frame` ...
        save_predictions(chapter=14, notebook_id="rp_pca", spec=spec, frame=frame)
        cached = frame

    # downstream code consumes `cached` for plotting / IC summaries.

The spec dict is the contract: any value that materially changes the
predictions must appear in it. Two runs that share a spec hash will share a
cache entry, so changing a hyperparameter without updating the spec will
silently reuse stale predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from utils.paths import get_chapter_dir

REQUIRED_COLUMNS = ("date", "symbol", "y_pred", "y_true")
KEY_LENGTH = 12


def predictions_cache_key(spec: dict) -> str:
    """SHA-1 hash of the canonicalised spec; first 12 hex digits."""
    payload = json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:KEY_LENGTH]


def predictions_cache_path(chapter: int | str, notebook_id: str, spec: dict) -> Path:
    """Deterministic parquet path; does not create the file or its parents."""
    root = get_chapter_dir(chapter) / "output" / "predictions" / notebook_id
    return root / f"{predictions_cache_key(spec)}.parquet"


def load_predictions(chapter: int | str, notebook_id: str, spec: dict) -> pl.DataFrame | None:
    """Return the cached predictions frame, or None if no cache entry exists."""
    path = predictions_cache_path(chapter, notebook_id, spec)
    if not path.exists():
        return None
    return pl.read_parquet(path)


def save_predictions(chapter: int | str, notebook_id: str, spec: dict, frame: pl.DataFrame) -> Path:
    """Write predictions frame to its content-addressed cache location."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"predictions frame missing required columns {missing}; have {list(frame.columns)}"
        )
    path = predictions_cache_path(chapter, notebook_id, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path
