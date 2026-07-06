"""Phase 4 — coverage ratchet: every free dataset keeps a live drift smoke.

Phase 1 keeps ``download_all.py``'s registry in lock-step with the on-disk
scripts. Phase 3 adds a live external drift smoke per free source. This test
welds the two together so coverage can't rot as later beats ship more datasets:
a newly shipped free dataset must appear in the free-tier list (Phase 1) *and*
carry a drift test (Phase 3), or CI fails on the PR that adds it.

Deterministic and network-free — it only introspects test modules and the
download registry, so it runs in the fast per-PR ``test-unit`` job.
"""

from __future__ import annotations

import tests.test_external_drift as drift
from tests.test_download_scripts_registry import DATA_DIR, FREE_TIER_SCRIPTS

# Maps each free-tier download script (the Phase 1 SSOT) to the drift-source key
# it must be smoke-tested under in tests/test_external_drift.py. Keep this in
# lock-step with FREE_TIER_SCRIPTS — the tests below fail loudly if it drifts.
FREE_SCRIPT_TO_DRIFT_SOURCE = {
    "etfs/market/download.py": "etf",
    "crypto/market/download.py": "crypto",
    "prediction_markets/download.py": "prediction_markets",
    "futures/positioning/cot_download.py": "cot",
    "factors/ff_download.py": "fama_french",
    "factors/aqr_download.py": "aqr",
    "equities/firm_characteristics/download.py": "firm_characteristics",
    "macro/download.py": "fred",
    "fx/market/download.py": "fx",
}


def _drift_test_names() -> set[str]:
    return {
        name for name in dir(drift) if name.startswith("test_") and callable(getattr(drift, name))
    }


def test_every_free_script_is_mapped_to_a_drift_source():
    """A new free dataset can't ship without declaring its drift source.

    When a later beat adds a free dataset to FREE_TIER_SCRIPTS, this fails until
    the author maps it here (which in turn forces a DRIFT_SOURCES entry and a
    drift test via the checks below).
    """
    mapped = set(FREE_SCRIPT_TO_DRIFT_SOURCE)
    free = set(FREE_TIER_SCRIPTS)
    unmapped = free - mapped
    assert not unmapped, (
        f"free-tier download scripts with no drift-source mapping: {sorted(unmapped)} — "
        f"add them to FREE_SCRIPT_TO_DRIFT_SOURCE and give each a drift test."
    )
    stale = mapped - free
    assert not stale, (
        f"drift-source mappings for scripts no longer in FREE_TIER_SCRIPTS: {sorted(stale)} — "
        f"remove them (the dataset was dropped or renamed)."
    )


def test_mapped_scripts_exist_on_disk():
    """Every mapped free script resolves to a real file (catches a rename)."""
    missing = [rel for rel in FREE_SCRIPT_TO_DRIFT_SOURCE if not (DATA_DIR / rel).is_file()]
    assert not missing, f"mapped free-tier scripts missing on disk: {missing}"


def test_drift_sources_match_mapping():
    """``DRIFT_SOURCES`` and the free-script mapping are bidirectionally consistent."""
    declared = set(drift.DRIFT_SOURCES)
    mapped = set(FREE_SCRIPT_TO_DRIFT_SOURCE.values())
    assert declared == mapped, (
        f"DRIFT_SOURCES {sorted(declared)} != mapped free sources {sorted(mapped)} — "
        f"a source is declared without a script mapping (or vice versa)."
    )


def test_every_drift_source_has_a_test():
    """Each declared drift source is backed by a ``test_<source>*`` function."""
    names = _drift_test_names()
    for source in drift.DRIFT_SOURCES:
        assert any(n.startswith(f"test_{source}") for n in names), (
            f"drift source '{source}' has no test_{source}* function in "
            f"tests/test_external_drift.py — declared but not smoke-tested."
        )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
