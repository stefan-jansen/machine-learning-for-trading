"""Point-in-time macro context for latent-factor models."""

from __future__ import annotations

import hashlib
from typing import Any

import polars as pl

from data import load_macro, load_macro_initial_release

MACRO_CONTEXT_ALIGNMENT = "backward_asof"


def load_configured_macro_context(
    config: dict[str, Any] | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load, lag, and identify a configured macro context panel."""
    if not config:
        raise ValueError("macro_context configuration is required")

    policy = str(config.get("policy", ""))
    source = str(config.get("source", ""))
    version = str(config.get("version", ""))
    series = list(config.get("series", []))
    lag_days = int(config.get("availability_lag_days", 0))
    alignment = str(config.get("alignment", ""))

    if not source or not policy or not version:
        raise ValueError("macro_context requires non-empty source, policy, and version")
    if not series or len(series) != len(set(series)):
        raise ValueError("macro_context series must be non-empty and unique")
    if lag_days < 1:
        raise ValueError("macro_context availability_lag_days must be at least one")
    if alignment != MACRO_CONTEXT_ALIGNMENT:
        raise ValueError(
            f"macro_context alignment must be {MACRO_CONTEXT_ALIGNMENT!r}, got {alignment!r}"
        )

    if source == "alfred_initial_release":
        raw = load_macro_initial_release(series=series)
    elif source == "fred_latest":
        raw = load_macro(series=series)
    else:
        raise ValueError(f"Unsupported macro_context source: {source!r}")
    missing = [column for column in series if column not in raw.columns]
    if missing:
        raise ValueError(f"Configured macro series are missing from the data: {missing}")

    panel = (
        raw.select(["timestamp", *series])
        .with_columns(
            (pl.col("timestamp").cast(pl.Date) + pl.duration(days=lag_days)).alias("timestamp")
        )
        .sort("timestamp")
    )
    identity = {
        "source": source,
        "policy": policy,
        "version": version,
        "series": series,
        "availability_lag_days": lag_days,
        "alignment": alignment,
        "input_digest": _macro_context_digest(panel),
        "coverage_start": panel["timestamp"].min().isoformat(),
        "coverage_end": panel["timestamp"].max().isoformat(),
    }
    return panel, identity


def _macro_context_digest(panel: pl.DataFrame) -> str:
    """Return a deterministic content digest for the ordered macro panel."""
    schema = "|".join(f"{name}:{dtype}" for name, dtype in panel.schema.items())
    payload = schema.encode() + b"\n" + panel.write_csv().encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = ["MACRO_CONTEXT_ALIGNMENT", "load_configured_macro_context"]
