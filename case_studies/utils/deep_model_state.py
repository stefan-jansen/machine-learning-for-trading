from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

SCHEMA_VERSION = 1


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _tensorize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, Mapping):
        return {str(key): _tensorize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_tensorize(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"unsupported checkpoint value {type(value).__name__}")


def _untensorize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, dict):
        return {key: _untensorize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_untensorize(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, list | tuple):
            digest.update(b"sequence")
            for member in item:
                update(member)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def checkpoint_sidecar(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.json")


def write_deep_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    architecture: str,
    model_kwargs: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    """Write one immutable fitted checkpoint and its digest sidecar."""
    path = Path(path)
    sidecar = checkpoint_sidecar(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "model_kwargs": _tensorize(model_kwargs),
        "preprocessing": _tensorize(preprocessing),
        "metadata": _tensorize(metadata),
        "state_dict": _cpu_state_dict(model),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    checkpoint_tmp = path.with_name(f".{path.name}.{token}.tmp")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    try:
        torch.save(payload, checkpoint_tmp)
        record = {
            "schema_version": SCHEMA_VERSION,
            "sha256": _sha256(checkpoint_tmp),
            "content_sha256": _semantic_digest(payload),
            "size": checkpoint_tmp.stat().st_size,
            "architecture": architecture,
            "metadata": _tensorize(metadata),
        }
        sidecar_tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        if path.exists() or sidecar.exists():
            if path.is_file() and sidecar.is_file():
                existing = json.loads(sidecar.read_text())
                if existing.get("content_sha256") == record["content_sha256"] and _sha256(
                    path
                ) == existing.get("sha256"):
                    return path
            raise FileExistsError(f"immutable checkpoint conflict: {path}")
        os.replace(checkpoint_tmp, path)
        try:
            os.replace(sidecar_tmp, sidecar)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    finally:
        checkpoint_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
    return path


def load_deep_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint only after its immutable digest contract passes."""
    path = Path(path)
    sidecar = checkpoint_sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint or sidecar is missing: {path}")
    record = json.loads(sidecar.read_text())
    actual_digest = _sha256(path)
    if record.get("schema_version") != SCHEMA_VERSION or record.get("sha256") != actual_digest:
        raise ValueError(f"checkpoint digest does not match its sidecar: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    return _untensorize(payload)


def restore_deep_model(
    path: Path,
    factory: Callable[[str, Mapping[str, Any]], nn.Module],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Rebuild a fitted model and return its preprocessing and metadata."""
    payload = load_deep_checkpoint(path)
    model = factory(payload["architecture"], payload["model_kwargs"])
    tensor_state = {
        name: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for name, value in payload["state_dict"].items()
    }
    model.load_state_dict(tensor_state, strict=True)
    model.eval()
    return model, payload["preprocessing"], payload["metadata"]
