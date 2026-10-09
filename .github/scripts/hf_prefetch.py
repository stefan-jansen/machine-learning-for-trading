#!/usr/bin/env python3
"""Prefetch the weekly notebooks' pinned checkpoints, and name the cache key.

The weekly job set no cache directory and cached nothing, so every Monday
re-downloaded BAAI/bge-large-en-v1.5 (~1.3 GB) inside a fresh container. That
download is what failed: in run 37320316180 it raised "couldn't connect to
huggingface.co" at cell 27 of 02_domain_embeddings_comparison while two ~90 MB
models in the same job downloaded and passed.

Two commands, both reading .github/weekly-hf-models.yaml:

``key``
    A digest of the (repo_id, revision) pairs, for ``actions/cache``. It changes
    when a pin changes and at no other time, so a warm run is warm. A date-bucketed
    key would discard the whole cache on a schedule, which is the cost this removes.

``fetch``
    ``snapshot_download`` each repo at its pinned revision. The notebooks pin the
    same revisions themselves, so a prefetched revision is the one they resolve, and
    a warm cache serves them without a request for file content.

``revisions``
    What each notebook pins, read from its source, for the manifest test.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "weekly-hf-models.yaml"

# A HuggingFace repo id is `owner/name`. The filter has to exclude the paths and
# filenames that also carry a slash, or the manifest test reports a data file as an
# unlisted model.
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_NOT_A_REPO_ID = (".py", ".ipynb", ".json", ".csv", ".parquet", ".txt", ".md", ".yaml", ".yml")


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    models = yaml.safe_load(path.read_text())["models"]
    missing = [m for m in models if not (m.get("repo_id") and m.get("revision"))]
    if missing:
        msg = f"{path} has entries without repo_id and revision: {missing}"
        raise ValueError(msg)
    return models


def cache_key(models: list[dict]) -> str:
    """A digest of the pins, stable across orderings of the manifest."""
    pairs = sorted(f"{m['repo_id']}@{m['revision']}" for m in models)
    return hashlib.sha256("\n".join(pairs).encode()).hexdigest()[:16]


def notebook_pins(notebook_py: Path) -> dict[str, list[str]]:
    """The repo ids and 40-hex revisions a notebook's source names.

    Read as two sets rather than paired up, because the three notebooks spell the
    pairing three different ways: a dict literal, two module constants, and a
    ``revision=`` keyword. Comparing sets catches a new model or a moved pin without
    this having to understand any of those shapes.
    """
    tree = ast.parse(notebook_py.read_text())
    strings = {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    return {
        "repo_ids": sorted(
            s for s in strings if _REPO_ID.match(s) and not s.endswith(_NOT_A_REPO_ID)
        ),
        "revisions": sorted(s for s in strings if _REVISION.match(s)),
    }


def fetch(models: list[dict]) -> int:
    from huggingface_hub import snapshot_download

    for m in models:
        print(f"prefetching {m['repo_id']}@{m['revision']}", flush=True)
        path = snapshot_download(repo_id=m["repo_id"], revision=m["revision"])
        print(f"  -> {path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("key", "fetch", "revisions"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    models = load_manifest(args.manifest)
    if args.command == "key":
        print(cache_key(models))
        return 0
    if args.command == "revisions":
        print(json.dumps(models, indent=2))
        return 0
    return fetch(models)


if __name__ == "__main__":
    sys.exit(main())
