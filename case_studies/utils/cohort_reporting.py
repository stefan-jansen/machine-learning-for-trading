"""Fail-closed attribution and presentation helpers for cohort diagnostics."""

from __future__ import annotations

from typing import Any

MIN_PBO_COMBINATIONS = 10


def cohort_metric_attribution(cohort: dict[str, Any], carrier_hash: str) -> dict[str, Any]:
    """Identify whether cohort diagnostics belong to the selected carrier."""
    leader_hash = cohort.get("leader_hash")
    if not leader_hash:
        raise ValueError("cohort_metrics row has no leader_hash")
    is_carrier = leader_hash == carrier_hash
    subject = "pinned carrier" if is_carrier else f"family cohort leader {leader_hash}"
    return {
        "leader_hash": leader_hash,
        "carrier_hash": carrier_hash,
        "applies_to_carrier": is_carrier,
        "subject": subject,
    }


def reportable_pbo(
    pbo: float | None,
    n_combinations: float | int | None,
    *,
    minimum: int = MIN_PBO_COMBINATIONS,
) -> dict[str, Any]:
    """Return a PBO presentation that suppresses underidentified estimates."""
    count = int(n_combinations) if n_combinations is not None else None
    if pbo is None or count is None:
        return {"value": None, "status": "unavailable", "n_combinations": count}
    if count < minimum:
        return {
            "value": None,
            "status": f"insufficient combinations ({count} < {minimum})",
            "n_combinations": count,
        }
    return {"value": float(pbo), "status": "available", "n_combinations": count}
