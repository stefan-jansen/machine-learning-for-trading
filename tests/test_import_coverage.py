"""Scanner-driven import-coverage test.

Uses ``envs.scan_imports`` to extract every third-party top-level import
that appears anywhere in the book's source code, then verifies each one
that's expected to resolve in the current Docker image actually does.

The point of this test is **drift detection**. When a chapter adds a
new dependency, the scanner picks it up automatically — no hand-edited
list to remember to update. If the new package isn't installed in the
image a reader built (or isn't classified in ``IMAGE_OVERRIDES``), the
test fails loudly instead of waiting for a reader to hit the missing
import mid-notebook.

The current image is taken from the ``ML4T_IMAGE`` environment variable
(defaulting to ``"ml4t"``). Each Docker image's entrypoint should set
``ML4T_IMAGE=<image-id>`` so the right set of imports gets exercised.
"""

from __future__ import annotations

import importlib
import os

import pytest

from envs.scan_imports import (
    IMAGE_OVERRIDES,
    REPO_ROOT,
    VALID_IMAGES,
    classify,
    scan_repo,
    try_import,
)

# The default image for agent / CI runs without an explicit ML4T_IMAGE
_DEFAULT_IMAGE = "ml4t"


@pytest.fixture(scope="module")
def scanned_imports() -> set[str]:
    """Run the scanner once per module — it walks the whole tree."""
    return scan_repo()


@pytest.fixture(scope="module")
def classified(scanned_imports) -> dict[str, set[str]]:
    return classify(scanned_imports)


# -----------------------------------------------------------------------------
# Sanity: the scanner finds things
# -----------------------------------------------------------------------------


def test_scanner_discovers_reasonable_number_of_external_imports(scanned_imports) -> None:
    """A healthy scan finds 50-200 third-party imports across 27 chapters +
    9 case studies. Outside that envelope indicates the filter is broken
    (too few: stdlib/first-party leaking in; too many: everything is being
    called third-party).
    """
    assert 50 <= len(scanned_imports) <= 200, (
        f"Scanner found {len(scanned_imports)} external imports — "
        "outside the plausible range for this repo"
    )


def test_scanner_finds_core_stack(scanned_imports) -> None:
    """The book's core stack must appear in every scan."""
    core = {"numpy", "pandas", "polars", "matplotlib", "scipy", "sklearn", "torch"}
    missing = core - scanned_imports
    assert not missing, f"core stack packages not detected: {missing}"


def test_scanner_excludes_stdlib(scanned_imports) -> None:
    """Basic regression: no stdlib name should appear in external imports."""
    stdlib_leak = scanned_imports & {"os", "sys", "json", "pathlib", "re", "ast"}
    assert not stdlib_leak, f"stdlib names leaked into external set: {stdlib_leak}"


def test_scanner_excludes_first_party(scanned_imports) -> None:
    """First-party names must be auto-filtered by the scanner."""
    first_party_leak = scanned_imports & {
        "utils",
        "data",
        "case_studies",
        "conftest",
    }
    assert not first_party_leak, f"first-party leaked: {first_party_leak}"


# -----------------------------------------------------------------------------
# Classification invariants
# -----------------------------------------------------------------------------


def test_every_image_override_targets_a_valid_image() -> None:
    """Every package in IMAGE_OVERRIDES must map to a recognized image id."""
    invalid = {pkg: img for pkg, img in IMAGE_OVERRIDES.items() if img not in VALID_IMAGES}
    assert not invalid, f"packages mapped to unknown images: {invalid}"


def test_every_override_target_appears_in_at_least_one_source_file(scanned_imports) -> None:
    """Trim dead entries: a package in IMAGE_OVERRIDES should actually be
    imported somewhere. If none of the code uses it, the classification
    entry is stale.
    """
    stale = set(IMAGE_OVERRIDES) - scanned_imports
    assert not stale, (
        f"IMAGE_OVERRIDES entries no longer imported anywhere: {sorted(stale)} — "
        "remove them from envs/scan_imports.py"
    )


def test_classify_groups_partition_the_scanned_set(classified, scanned_imports) -> None:
    """The union of all image buckets must equal the scanned set (no orphans)."""
    union: set[str] = set()
    for bucket in classified.values():
        union |= bucket
    assert union == scanned_imports


# -----------------------------------------------------------------------------
# The real test: every expected import resolves in this image
# -----------------------------------------------------------------------------


def test_every_expected_import_resolves_in_current_image(classified) -> None:
    """Attempt to import every package the scanner classified for this image.

    The image id comes from ``$ML4T_IMAGE`` (default ``ml4t``). Each Docker
    entrypoint should set this variable. Running pytest locally picks up
    the default.
    """
    image = os.environ.get("ML4T_IMAGE", _DEFAULT_IMAGE)
    expected = sorted(classified[image])

    failures: list[tuple[str, str]] = []
    for pkg in expected:
        ok, err = try_import(pkg)
        if not ok:
            failures.append((pkg, err))

    if failures:
        lines = "\n".join(f"  {pkg}: {err[:120]}" for pkg, err in failures)
        pytest.fail(
            f"{len(failures)} of {len(expected)} expected imports failed in "
            f"image={image!r}:\n{lines}"
        )


# -----------------------------------------------------------------------------
# Smoke: first-party packages we ship must import too (readers installed us)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        "utils.paths",
        "utils.modeling",
        "data",
        "case_studies.utils.analytics",
        "case_studies.utils.signals",
        "case_studies.utils.allocation",
        "case_studies.utils.registry.metrics",
    ],
)
def test_first_party_modules_import(module: str) -> None:
    """First-party packages must load cleanly — regression guard for refactors
    that break the data/* or case_studies/* package tree."""
    ok, err = try_import(module)
    assert ok, f"first-party import failed: {module}: {err[:200]}"
