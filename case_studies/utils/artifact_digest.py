"""Value digests for the artifacts one case-study stage hands to the next.

A digest computed over content makes an artifact self-describing at the point it
is written: the ``<artifact>.digest.json`` sidecar records what was written, and
its ``inputs`` field records the digests of the artifacts it was built from.

It is a record, not propagation. Nothing downstream reads these sidecars today -
stage 02 writes them and the chain stops there - so an upstream value change does
not yet flow into any later artifact's digest. Note also that a model run is
separately pinned to its input bytes by ``training_input_identity``, which
hashes the label, feature and setup files directly rather than through a sidecar.

The digest is over content, not file bytes: it is invariant to row order and to
parquet metadata churn, and sensitive to any value change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .registry.specs import canonical_json, compute_hash

DIGEST_LENGTH = 16

# The parquet encoding every identity-defining artifact is written under.
#
# Registry training identity digests these files' *bytes* - `utils/modeling.py`'s
# `_sha256_file` feeds `computation.feature_artifacts` and
# `computation.input_data_spec.artifacts`, both inside the hashed `computation` block. Two
# files holding identical data therefore hash differently if they were written with a
# different compression codec or row-group size, so a polars upgrade that moves a default
# would fork the `training_hash` of every run reading that artifact: cached runs stop
# matching, the sweep refits everything, and no number has moved to say why.
#
# These are polars 1.41.1's own defaults, written out. Verified byte-identical to writing
# with none of them passed, so stating them moves nothing today and pins the encoding
# against a future default change. `tests/test_artifact_digest_encoding.py` records the
# bytes a fixed frame produces under them, so a change arrives as a red test rather than as
# a registry that has quietly grown two identities for one piece of work.
_PARQUET_WRITE_SETTINGS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": None,
    "statistics": True,
    "row_group_size": None,
    "data_page_size": None,
}


def value_digest(df: pl.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Return the content digest of *df* over *columns* (default: all).

    Per-row hashes are computed over name-sorted columns and then sorted, so the
    result does not depend on row order or on column order — only on the set of
    columns and the values in them.
    """
    cols = sorted(df.columns if columns is None else columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    row_hashes = df.select(cols).hash_rows().sort().to_numpy().tobytes()
    content = canonical_json(
        {"columns": cols, "rows": hashlib.sha256(row_hashes).hexdigest()},
    )
    return compute_hash(content, length=DIGEST_LENGTH)


def fold_digests(df: pl.DataFrame, *, fold_column: str = "fold") -> dict[str, str]:
    """Return one content digest per fold id in *df*, keyed by the id as a string.

    A fold-scoped artifact is written once per stage-04 run and read by every stage after
    it, and the reader that matters is the one verifying a holdout lock: the lock pins the
    artifact by whole-file sha256, so appending the holdout fold - which is the whole reason
    stage 04 writes a holdout fold at all - changes the pin and the retrain is refused. What
    the lock actually needs to know is narrower: that the folds it was selected under still
    hold the values they held. One digest per fold answers that; the whole-file digest cannot.

    Each fold's digest is :func:`value_digest` over that fold's rows, so it moves when and
    only when a value in that fold moves, and is invariant to row order exactly as the
    whole-frame digest is.
    """
    if fold_column not in df.columns:
        raise KeyError(f"fold column {fold_column!r} not in frame: {sorted(df.columns)}")
    return {
        str(fold): value_digest(df.filter(pl.col(fold_column) == fold))
        for fold in sorted(df.get_column(fold_column).unique().to_list())
    }


def _json_ready(value: Any) -> Any:
    """Render dates and times as ISO strings, recursively, leaving everything else alone.

    Fold boundaries arrive as ``datetime.date`` or ``pandas.Timestamp`` because that is what
    the producers hold them as, and ``json.dumps`` refuses both. Coercing here rather than
    asking every caller to stringify keeps the sidecar's dates in one format, which is the
    format ``temporal_artifact_fold_boundaries`` already parses.
    """
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):  # pandas.Timestamp and anything else date-like
        return str(value.isoformat())
    return value


def digest_sidecar(
    df: pl.DataFrame,
    *,
    keys: Sequence[str],
    written_by: str,
    inputs: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    fold_column: str | None = None,
) -> dict:
    """Build the sidecar record for *df* without writing it.

    ``metadata`` carries producer facts a reader cannot recover from the frame. The one
    that needs it is ``fold_geometry``: ``temporal_artifact_fold_boundaries`` reads it
    when the sidecar carries it, and otherwise regenerates the fold set by calling
    ``generate_cv_splits`` - which returns the cross-validation folds and nothing else, so
    an artifact carrying a fold that routine does not produce has that fold invisible to
    every consumer. The frame states which fold ids exist and never states what their
    boundaries were.

    ``fold_column`` records a digest per fold beside the whole-frame one - see
    :func:`fold_digests` for what reads it. It is opt-in rather than inferred from the
    presence of a ``fold`` column: the record is what a lock is verified against, so which
    artifacts make a fold-scoped claim is a producer's decision, not a schema coincidence.

    Reserved keys are refused rather than merged. A caller that could overwrite ``digest``
    or ``n_rows`` could make a sidecar describe a file it was not computed from.
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"key columns not in frame: {missing}")
    record = {
        "digest": value_digest(df),
        "n_rows": df.height,
        "columns": sorted(df.columns),
        "keys": list(keys),
        "written_by": written_by,
        "inputs": dict(inputs or {}),
    }
    if fold_column is not None:
        record["fold_digests"] = fold_digests(df, fold_column=fold_column)
    if metadata:
        reserved = sorted(set(metadata) & set(record))
        if reserved:
            raise ValueError(
                f"sidecar metadata may not redefine the record's own fields: {reserved}"
            )
        record.update(_json_ready(dict(metadata)))
    return record


def write_artifact(
    df: pl.DataFrame,
    path: Path | str,
    *,
    keys: Sequence[str],
    written_by: str,
    inputs: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    fold_column: str | None = None,
) -> dict:
    """Write *df* to *path* as parquet with its digest sidecar beside it.

    Returns the sidecar record, whose ``digest`` is what a downstream stage
    passes back as one of its own ``inputs``. ``metadata`` and ``fold_column`` are
    passed through to :func:`digest_sidecar`.
    """
    path = Path(path)
    record = digest_sidecar(
        df,
        keys=keys,
        written_by=written_by,
        inputs=inputs,
        metadata=metadata,
        fold_column=fold_column,
    )
    # Rendered before the parquet is written, not after. A record that cannot be serialized
    # would otherwise raise with the new parquet already on disk beside the previous run's
    # sidecar - a file and a digest that describe different data, which is the one state
    # this pair exists to make impossible.
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, **_PARQUET_WRITE_SETTINGS)
    sidecar_path(path).write_text(serialized)
    return record


def sidecar_path(path: Path | str) -> Path:
    """Return the sidecar path for an artifact path."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".digest.json")


def read_digest(path: Path | str) -> dict:
    """Read the sidecar record written beside an artifact."""
    return json.loads(sidecar_path(path).read_text())
