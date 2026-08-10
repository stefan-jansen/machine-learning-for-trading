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
from pathlib import Path

import polars as pl

from .registry.specs import canonical_json, compute_hash

DIGEST_LENGTH = 16


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


def digest_sidecar(
    df: pl.DataFrame,
    *,
    keys: Sequence[str],
    written_by: str,
    inputs: Mapping[str, str] | None = None,
) -> dict:
    """Build the sidecar record for *df* without writing it."""
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"key columns not in frame: {missing}")
    return {
        "digest": value_digest(df),
        "n_rows": df.height,
        "columns": sorted(df.columns),
        "keys": list(keys),
        "written_by": written_by,
        "inputs": dict(inputs or {}),
    }


def write_artifact(
    df: pl.DataFrame,
    path: Path | str,
    *,
    keys: Sequence[str],
    written_by: str,
    inputs: Mapping[str, str] | None = None,
) -> dict:
    """Write *df* to *path* as parquet with its digest sidecar beside it.

    Returns the sidecar record, whose ``digest`` is what a downstream stage
    passes back as one of its own ``inputs``.
    """
    path = Path(path)
    record = digest_sidecar(df, keys=keys, written_by=written_by, inputs=inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    sidecar_path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def sidecar_path(path: Path | str) -> Path:
    """Return the sidecar path for an artifact path."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".digest.json")


def read_digest(path: Path | str) -> dict:
    """Read the sidecar record written beside an artifact."""
    return json.loads(sidecar_path(path).read_text())
