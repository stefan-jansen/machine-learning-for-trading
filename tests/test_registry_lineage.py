"""Contracts for the audited model-input lineage helper."""

from __future__ import annotations

import hashlib
from pathlib import Path

import case_studies.utils.registry as registry

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITED_LINEAGE_SHA256 = "7b9a8626f2d337d00ad33d992a35f6dca176ad375ed484b75d60737e22f44e0d"


def test_registry_lineage_source_matches_audited_fixture_bytes() -> None:
    """Maintained lineage source must remain byte-identical to the audited fixture."""
    source = REPO_ROOT / "case_studies/utils/registry/lineage.py"

    assert hashlib.sha256(source.read_bytes()).hexdigest() == AUDITED_LINEAGE_SHA256


def test_registry_exports_modeling_input_fingerprint() -> None:
    """The maintained package must expose the audited lineage helper."""
    assert "modeling_input_fingerprint" in registry.__all__
    assert registry.modeling_input_fingerprint.__module__ == "case_studies.utils.registry.lineage"
