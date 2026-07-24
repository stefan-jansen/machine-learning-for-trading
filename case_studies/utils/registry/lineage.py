"""Content identities for model-training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def modeling_input_fingerprint(
    case_dir: Path,
    label: str,
    splits: list[dict[str, Any]],
    feature_names: list[str],
    max_symbols: int,
) -> str:
    """Hash the artifacts and assembled contract that determine a model fit."""
    digest = hashlib.sha256()
    relative_paths = (
        "features/financial.parquet",
        "features/model_based.parquet",
        f"labels/{label}.parquet",
        "config/setup.yaml",
    )
    for relative in relative_paths:
        path = case_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing lineage input: {path}")
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

    contract = {
        "contract_version": 1,
        "feature_names": feature_names,
        "max_symbols": max_symbols,
        "splits": splits,
    }
    digest.update(json.dumps(contract, default=str, separators=(",", ":"), sort_keys=True).encode())
    return digest.hexdigest()
