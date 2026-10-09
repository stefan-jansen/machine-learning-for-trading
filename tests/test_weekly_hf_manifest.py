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

MODELS, LOADER_FILES = hf_prefetch.load_manifest()
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
    key = hf_prefetch.cache_key(MODELS, LOADER_FILES)
    assert hf_prefetch.cache_key(list(reversed(MODELS)), LOADER_FILES) == key
    assert hf_prefetch.cache_key(MODELS, list(reversed(LOADER_FILES))) == key
    moved = [{**MODELS[0], "revision": "0" * 40}, *MODELS[1:]]
    assert hf_prefetch.cache_key(moved, LOADER_FILES) != key


def test_the_cache_key_changes_when_the_file_selection_changes() -> None:
    """Otherwise a cache built under a narrower selection restores into a wider run.

    The missing weights would then be discovered by a notebook at load time rather
    than by the prefetch, which is the failure the cache exists to remove.
    """
    narrower = [f for f in LOADER_FILES if f != "model*.safetensors"]
    assert narrower != LOADER_FILES
    assert hf_prefetch.cache_key(MODELS, narrower) != hf_prefetch.cache_key(MODELS, LOADER_FILES)


def test_a_manifest_without_a_file_selection_is_refused(tmp_path: Path) -> None:
    """An empty allow list makes snapshot_download take every weight format."""
    bad = tmp_path / "m.yaml"
    bad.write_text("models:\n  - repo_id: a/b\n    revision: " + "c" * 40 + "\n")
    with pytest.raises(ValueError, match="loader_files"):
        hf_prefetch.load_manifest(bad)


# The file list of BAAI/bge-large-en-v1.5 at the pinned revision
# d4aa6901d3a41ba39fb536a557fa166f842b0e09, from the Hub's own metadata on
# 2026-10-09. The whole repository is 4.019 GB because the same weights ship three
# times; the three ~1.34 GB entries are what the selection has to drop.
BGE_LARGE_FILES = [
    ".gitattributes",
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "onnx/model.onnx",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def test_the_selection_keeps_one_weight_format_and_drops_the_others() -> None:
    """Applied through filter_repo_objects, which is what snapshot_download uses."""
    from huggingface_hub.utils import filter_repo_objects

    kept = sorted(filter_repo_objects(BGE_LARGE_FILES, allow_patterns=LOADER_FILES))
    assert "model.safetensors" in kept, "the loader would have no weights to read"
    assert "pytorch_model.bin" not in kept
    assert "onnx/model.onnx" not in kept


def test_the_selection_keeps_every_file_sentence_transformers_opens() -> None:
    """A missing one of these fails at load time with a warm cache, not at prefetch."""
    from huggingface_hub.utils import filter_repo_objects

    kept = set(filter_repo_objects(BGE_LARGE_FILES, allow_patterns=LOADER_FILES))
    required = {
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "1_Pooling/config.json",
    }
    assert required <= kept, f"the selection drops {sorted(required - kept)}"
