"""A copying builder has to refuse a production tree that is missing what it needs.

The four downloaded-source builders in ``tests/create_test_data.py`` copy whatever
production carries. That is deliberate - a production directory that has gained a file
should ship it rather than be silently trimmed - but it means the build itself proves
nothing about completeness. What proves it is the ``*_REQUIRED`` tuple each builder
checks: the files a loader can name or a notebook opens by path.

A guard that cannot be shown to fire is not a guard. These tests build each dataset
against a shadow source tree with one required file removed, and assert the build
raises and names the file and the script that fetches it.

The source tree is synthesised, not production: these builders copy bytes and parse
nothing, so a file of the right name exercises the guard exactly as the real one would.
That is deliberate rather than convenient. A test that reads production can only run on
the workstation, and in CI it would skip - and a guard whose check never runs is the
same defect as the guard not existing.
"""

from pathlib import Path

import pytest

from tests.create_test_data import (
    AQR_DIR,
    AQR_REQUIRED,
    FF_DIR,
    FF_REQUIRED,
    MACRO_DIR,
    MACRO_REQUIRED,
    ONCHAIN_DIR,
    ONCHAIN_REQUIRED,
    build_aqr_factors,
    build_crypto_onchain,
    build_fama_french_factors,
    build_fred_macro,
)

# One name per source that production carries and no *_REQUIRED tuple names. They are
# here so the success case proves the builders copy more than the minimum, which is the
# behaviour that lets a production directory gain a file and have it ship.
EXTRAS = {
    "factors/fama-french": ("bp_me_monthly.parquet",),
    "factors/aqr": ("esg_frontier.parquet", "metadata.json"),
    "macro": ("fred_macro_dictionary.parquet",),
    "crypto/onchain": (),
}

CASES = [
    pytest.param(build_fama_french_factors, FF_DIR, FF_REQUIRED, "ff_download.py", id="ff"),
    pytest.param(build_aqr_factors, AQR_DIR, AQR_REQUIRED, "aqr_download.py", id="aqr"),
    pytest.param(build_fred_macro, MACRO_DIR, MACRO_REQUIRED, "download_alfred.py", id="macro"),
    pytest.param(build_crypto_onchain, ONCHAIN_DIR, ONCHAIN_REQUIRED, "download.py", id="onchain"),
]


def _source(root: Path, directory: Path, required: tuple[str, ...], omit: str) -> Path:
    """A stand-in production tree holding the required files plus extras, minus ``omit``."""
    staged = root / directory
    staged.mkdir(parents=True)
    for name in (*required, *EXTRAS[directory.as_posix()]):
        if name != omit:
            (staged / name).write_bytes(b"")
    return root


@pytest.mark.parametrize(("build", "directory", "required", "script"), CASES)
def test_a_missing_required_file_fails_the_build(
    build, directory: Path, required: tuple[str, ...], script: str, tmp_path: Path
) -> None:
    """Every required file, one at a time. A tuple entry nothing checks is decoration."""
    for name in required:
        source = _source(tmp_path / name, directory, required, omit=name)
        with pytest.raises(FileNotFoundError) as excinfo:
            build(source, tmp_path / "out" / name)
        message = str(excinfo.value)
        assert name in message, f"{name} removed but not named in: {message}"
        assert script in message, f"{name} failure does not say what fetches it: {message}"


@pytest.mark.parametrize(("build", "directory", "required", "script"), CASES)
def test_a_complete_source_builds(
    build, directory: Path, required: tuple[str, ...], script: str, tmp_path: Path
) -> None:
    """The negative half: with nothing removed the same call succeeds.

    Without this, a builder that raised unconditionally would pass every case above.
    """
    source = _source(tmp_path / "full", directory, required, omit="")
    written = build(source, tmp_path / "out")
    names = {path.name for path in written}
    assert names >= set(required)
    assert names >= set(EXTRAS[directory.as_posix()]), "the builder copied only the minimum"


def test_initial_release_is_required_and_comes_from_its_own_script() -> None:
    """A regression pin, because this one was omitted and the omission was invisible.

    `fred_macro_initial_release.parquet` is written by `download_alfred.py` while every
    other FRED file comes from `download.py`, so a production tree can hold all of them
    and not this one. `07_macro_data_alignment` calls `load_macro_initial_release()`
    unconditionally: without the file the build succeeded and the notebook raised.
    """
    assert "fred_macro_initial_release.parquet" in MACRO_REQUIRED
