"""Coverage-ratchet tests for the download-script registry.

``download_all.py`` dispatches by shelling out to the paths in
``DOWNLOAD_SCRIPTS``. If a downloader is renamed or moved without updating the
map, dispatch silently degrades to "Script not found" at runtime — invisible to
CI until a reader hits it. These tests keep the map and the on-disk scripts in
lock-step, in both directions, and fail when a *new* ``download.py`` is added
without wiring it in (so coverage can't rot as later beats ship more datasets).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import data.download_all as da

DATA_DIR = Path(da.__file__).parent

# The free datasets a reader gets with ``download_all.py --free-only`` (no API
# key, or a free key). Single source of truth: the registry ratchet below pins
# that these ship, and ``tests/test_download_coverage.py`` (Phase 4) pins that
# each keeps a live drift smoke — so a newly shipped free dataset can't land
# without both a registry entry and an external-drift test.
FREE_TIER_SCRIPTS = (
    "etfs/market/download.py",
    "crypto/market/download.py",
    "prediction_markets/download.py",
    "futures/positioning/cot_download.py",
    "factors/ff_download.py",
    "factors/aqr_download.py",
    "equities/firm_characteristics/download.py",
    "macro/download.py",
    "fx/market/download.py",
)


def _discovered_download_scripts() -> set[str]:
    """All download scripts on disk, as paths relative to ``data/``."""
    found = set(DATA_DIR.glob("**/download.py")) | set(DATA_DIR.glob("**/*_download.py"))
    return {str(p.relative_to(DATA_DIR)) for p in found}


@pytest.mark.parametrize("script_name,relative_path", sorted(da.DOWNLOAD_SCRIPTS.items()))
def test_registered_script_exists(script_name, relative_path):
    """Every entry in DOWNLOAD_SCRIPTS resolves to a real file."""
    assert (DATA_DIR / relative_path).is_file(), (
        f"DOWNLOAD_SCRIPTS['{script_name}'] -> {relative_path} does not exist; "
        "a downloader was renamed/moved without updating download_all.py"
    )


def test_every_download_script_is_registered():
    """Coverage ratchet: no download script may be orphaned from the registry."""
    registered = set(da.DOWNLOAD_SCRIPTS.values())
    discovered = _discovered_download_scripts()
    unregistered = discovered - registered
    assert not unregistered, (
        f"download scripts present on disk but missing from DOWNLOAD_SCRIPTS: "
        f"{sorted(unregistered)} — wire them into download_all.py (or rename)."
    )


def test_free_tier_scripts_present():
    """The free datasets a reader gets with `download_all.py --free-only` all ship."""
    missing = [p for p in FREE_TIER_SCRIPTS if not (DATA_DIR / p).is_file()]
    assert not missing, f"free-tier download scripts missing: {missing}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
