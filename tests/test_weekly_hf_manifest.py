"""The HF cache manifest must name exactly what the weekly notebooks pin.

The weekly job's cache key is a digest of `.github/weekly-hf-models.yaml`, and its
prefetch step downloads what that file lists. Both are wrong the moment a notebook
adds a model or moves a pin: the prefetch misses the new checkpoint, the key does
not change, and the weekly run goes back to downloading 1.3 GB on a shared-IP
runner, which is the failure the cache exists to remove.

Nothing in the repository declares these models except the notebooks themselves, so
this test reads them out of the notebook sources and compares. It is the only thing
that stops the manifest going stale silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "hf_prefetch.py"
_spec = importlib.util.spec_from_file_location("hf_prefetch", SCRIPT)
assert _spec and _spec.loader
hf_prefetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf_prefetch)

MODELS = hf_prefetch.load_manifest()
NOTEBOOKS = sorted({nb for m in MODELS for nb in m["notebooks"]})


def test_the_manifest_lists_some_models_and_some_notebooks() -> None:
    """A manifest that emptied out would make every other assertion vacuous."""
    assert MODELS
    assert NOTEBOOKS


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_each_notebook_pins_exactly_the_models_the_manifest_gives_it(notebook: str) -> None:
    source = ROOT / f"{notebook}.py"
    assert source.exists(), f"{notebook} is in the manifest but has no .py"
    pins = hf_prefetch.notebook_pins(source)

    expected_repos = sorted(m["repo_id"] for m in MODELS if notebook in m["notebooks"])
    expected_revisions = sorted({m["revision"] for m in MODELS if notebook in m["notebooks"]})

    # Revisions are compared exactly: a 40-hex constant in a notebook here is a
    # model pin and nothing else, so a new model or a moved pin shows up as a
    # difference. Repo ids are a containment check, because `owner/name` also
    # matches a currency pair, a timezone and a column label, and a notebook
    # gaining one of those is not a cache problem.
    assert pins["revisions"] == expected_revisions, (
        "the notebook pins a revision the manifest does not have: either a model was "
        "added or a pin moved, and the prefetch and cache key both read the manifest"
    )
    assert set(expected_repos) <= set(pins["repo_ids"]), (
        f"the manifest gives {notebook} a model its source never names"
    )


def test_every_weekly_tier_notebook_that_pins_a_model_is_in_the_manifest() -> None:
    """A weekly notebook that starts downloading a checkpoint has to be listed."""
    import yaml

    overrides = yaml.safe_load((ROOT / "tests" / "overrides.yaml").read_text())
    weekly = [k for k, v in overrides.items() if isinstance(v, dict) and v.get("tier") == "weekly"]
    unlisted = []
    for stem in weekly:
        source = ROOT / f"{stem}.py"
        if not source.exists():
            continue
        # A pinned revision is the signal. Every model load in these notebooks passes
        # `revision=`, and unlike a repo id a 40-hex constant has no other meaning.
        pins = hf_prefetch.notebook_pins(source)
        if pins["revisions"] and stem not in NOTEBOOKS:
            unlisted.append((stem, pins["revisions"]))
    assert not unlisted, f"weekly notebooks pin models that the manifest omits: {unlisted}"


def test_the_cache_key_follows_the_pins_and_not_the_ordering() -> None:
    """Keying on the pins is what removes the weekly re-download."""
    key = hf_prefetch.cache_key(MODELS)
    assert hf_prefetch.cache_key(list(reversed(MODELS))) == key
    moved = [{**MODELS[0], "revision": "0" * 40}, *MODELS[1:]]
    assert hf_prefetch.cache_key(moved) != key
