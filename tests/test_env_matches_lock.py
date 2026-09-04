"""The detector that would have caught the 11-day stale image.

`.github/scripts/check_env_matches_lock.py` runs inside `ml4t/ml4t:latest` and
compares its installed distributions against `uv.lock`. This file exercises the
comparison against synthetic inventories, because the check itself can only ever
report on the environment it happens to run in - and a detector that is only
observed passing is not known to be able to fail.

The condition it exists for: the image built 2026-07-16 carried
`ml4t-models==0.1.0a4` while the repository declared `>=0.1.0a6` from 2026-07-21,
so for eleven days public CI measured a dependency set the repository did not
declare, and the notebook that failed on it read as a numerical problem.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_env_matches_lock", REPO_ROOT / ".github" / "scripts" / "check_env_matches_lock.py"
)
assert _spec and _spec.loader
check = importlib.util.module_from_spec(_spec)
sys.modules["check_env_matches_lock"] = check
_spec.loader.exec_module(check)


# --- name normalization ------------------------------------------------------


def test_names_compare_under_pep503_normalization() -> None:
    """`ml4t_diagnostic`, `ML4T-Diagnostic` and `ml4t.diagnostic` are one distribution."""
    assert (
        check.canonical("ML4T_Diagnostic")
        == check.canonical("ml4t.diagnostic")
        == "ml4t-diagnostic"
    )


def test_normalization_makes_a_differently_spelled_name_comparable() -> None:
    locked = {check.canonical("ML4T_Models"): "0.1.2"}
    installed = {check.canonical("ml4t-models"): "0.1.0a4"}
    assert check.drift(locked, installed) == [("ml4t-models", "0.1.2", "0.1.0a4")]


# --- what counts as drift ----------------------------------------------------


def test_an_environment_that_matches_the_lock_reports_nothing() -> None:
    locked = {"polars": "1.20.0", "numpy": "2.3.1"}
    assert check.drift(locked, dict(locked)) == []


def test_a_version_below_the_locked_one_is_drift() -> None:
    """The filed instance, restated: the image lagged the lock by one release."""
    locked = {"ml4t-models": "0.1.0a6"}
    installed = {"ml4t-models": "0.1.0a4"}
    assert check.drift(locked, installed) == [("ml4t-models", "0.1.0a6", "0.1.0a4")]


def test_a_version_above_the_locked_one_is_drift_too() -> None:
    """LightGBM went 4.6.0 -> 4.7.0 under a floating pin and broke the build."""
    locked = {"lightgbm": "4.6.0"}
    installed = {"lightgbm": "4.7.0"}
    assert check.drift(locked, installed) == [("lightgbm", "4.6.0", "4.7.0")]


def test_a_distribution_the_lock_does_not_pin_is_not_drift() -> None:
    assert check.drift({"polars": "1.20.0"}, {"polars": "1.20.0", "ipdb": "0.13.13"}) == []


def test_a_locked_distribution_the_environment_omits_is_not_drift() -> None:
    """The lock resolves extras and platforms an image is not meant to carry."""
    assert check.drift({"polars": "1.20.0", "darts": "0.38.0"}, {"polars": "1.20.0"}) == []


# --- the declared carve-out --------------------------------------------------


def test_the_torch_stack_is_exempt_because_the_image_builds_it_from_the_cuda_index() -> None:
    locked = {"torch": "2.10.0", "torchvision": "0.25.0", "triton": "3.2.0"}
    installed = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128", "triton": "3.5.0"}
    assert check.drift(locked, installed) == []


def test_the_cuda_runtime_wheels_are_exempt_by_prefix() -> None:
    locked = {"nvidia-cublas-cu12": "12.8.3.14"}
    installed = {"nvidia-cublas-cu12": "12.9.0.0"}
    assert check.drift(locked, installed) == []


def test_lightgbm_is_not_exempt() -> None:
    """It was carved out of the Dockerfile's constraint once, and 4.7 broke the build.

    The Dockerfile now installs the lock's exact version; this is what holds it.
    """
    assert not check.is_declared_off_lock("lightgbm")


def test_every_carve_out_records_why_it_is_one() -> None:
    assert all(reason.strip() for reason in check.DECLARED_OFF_LOCK.values())


# --- what pyproject overrides ------------------------------------------------


def test_an_override_is_read_out_of_pyproject() -> None:
    """The repository's own override, so the parsing is checked against real text."""
    overridden = check.overridden_dependencies(REPO_ROOT / "pyproject.toml")
    assert "protobuf" in overridden
    assert "protobuf>=5.0" in overridden["protobuf"]


def test_an_overridden_distribution_is_not_drift() -> None:
    """uv asserts the version by fiat and pip has no equivalent.

    protobuf is overridden to >=5.0 because 4.x has a C-extension metaclass bug on
    Python 3.14; the lock then resolves 7.35.0 while opentelemetry-proto requires
    <7.0, so a pip install into the image resolves 6.33.6 and is right to.
    """
    locked = {"protobuf": "7.35.0"}
    installed = {"protobuf": "6.33.6"}
    assert check.drift(locked, installed) == [("protobuf", "7.35.0", "6.33.6")]
    assert check.drift(locked, installed, {"protobuf": "overridden to protobuf>=5.0"}) == []


def test_an_override_specifier_is_parsed_down_to_the_name() -> None:
    written = {
        "tool": {"uv": {"override-dependencies": ["Some_Pkg[extra]>=1.2", "other!=3"]}},
    }
    import tempfile
    import tomllib as _tomllib  # noqa: F401

    path = Path(tempfile.mkdtemp()) / "pyproject.toml"
    path.write_text('[tool.uv]\noverride-dependencies = ["Some_Pkg[extra]>=1.2", "other!=3"]\n')
    assert set(check.overridden_dependencies(path)) == {"some-pkg", "other"}
    assert written  # the literal above documents the shape being parsed


def test_a_missing_pyproject_overrides_nothing() -> None:
    assert check.overridden_dependencies(Path("/nonexistent/pyproject.toml")) == {}


# --- the lock is read, not assumed -------------------------------------------


def test_the_repository_lock_parses_and_pins_the_ml4t_libraries() -> None:
    locked = check.locked_versions(REPO_ROOT / "uv.lock")
    assert locked["ml4t-diagnostic"]
    assert locked["polars"]


def test_every_declared_ml4t_floor_is_satisfied_by_the_lock() -> None:
    """pyproject states a reason for each ml4t-* floor; the lock has to clear it.

    Compared with `packaging`, not by splitting on dots: the versions that matter
    here are prereleases, and the filed instance was 0.1.0a4 against a 0.1.0a6
    floor - two versions a numeric-component comparison reads as equal.
    """
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    locked = check.locked_versions(REPO_ROOT / "uv.lock")
    requirements = [
        Requirement(dep) for dep in project["project"]["dependencies"] if dep.startswith("ml4t-")
    ]
    assert requirements, "no ml4t-* dependency found in pyproject"
    for requirement in requirements:
        got = locked[check.canonical(requirement.name)]
        assert requirement.specifier.contains(Version(got), prereleases=True), (
            f"uv.lock resolves {requirement.name} {got}, "
            f"which does not satisfy {requirement.specifier}"
        )


def test_the_floor_comparison_can_fail_on_a_prerelease() -> None:
    """The filed instance, restated against the comparison that has to catch it."""
    assert not Requirement("ml4t-models>=0.1.0a6").specifier.contains(
        Version("0.1.0a4"), prereleases=True
    )
