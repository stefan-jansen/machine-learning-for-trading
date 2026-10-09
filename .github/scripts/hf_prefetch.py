#!/usr/bin/env python3
"""Prefetch the weekly notebooks' pinned checkpoints, and name the cache key.

The weekly job set no cache directory and cached nothing, so every Monday
re-downloaded BAAI/bge-large-en-v1.5 (~1.3 GB) inside a fresh container. That
download is what failed: in run 37320316180 it raised "couldn't connect to
huggingface.co" at cell 27 of 02_domain_embeddings_comparison while two ~90 MB
models in the same job downloaded and passed.

Two commands, both reading .github/weekly-hf-models.yaml:

``key``
    A digest of the (repo_id, revision) pairs AND the file selection, for
    ``actions/cache``. It changes when a pin or the selection changes and at no other
    time, so a warm run is warm. A date-bucketed key would discard the whole cache on
    a schedule, which is the cost this removes; a key over the pins alone would serve
    a cache built under a narrower selection to a run that needs a wider one. The key
    being stable is not retention: GitHub deletes an entry nobody has accessed in over
    seven days, which is why weekly-external.yml restores it on three days a week.

``fetch``
    ``snapshot_download`` each repo at its pinned revision, restricted to the files
    the loaders open. Unrestricted, these four repositories are 6.287 GB because each
    ships its weights in three or four formats; the selection is 1.659 GB. Both figures
    fit the repository's 10 GB cache limit beside the 2.655 GB already there, so the
    selection buys headroom and transfer time rather than rescuing a breached cap. The
    notebooks pin the same revisions, so a prefetched revision is the one they
    resolve.

``select``
    List what the selection resolves to at each pinned revision, with sizes, through
    the same discovery path ``snapshot_download`` uses (``list_repo_files`` and
    ``filter_repo_objects``) and without downloading a single model body. Needs
    network; it is for checking a pattern change, not for CI.

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


def load_manifest(path: Path = MANIFEST) -> tuple[list[dict], list[str]]:
    """The models and the file selection, which are one contract and travel together."""
    manifest = yaml.safe_load(path.read_text())
    models = manifest["models"]
    missing = [m for m in models if not (m.get("repo_id") and m.get("revision"))]
    if missing:
        msg = f"{path} has entries without repo_id and revision: {missing}"
        raise ValueError(msg)
    loader_files = manifest.get("loader_files") or []
    if not loader_files:
        # An empty selection means snapshot_download takes everything, which is the
        # 6.287 GB this manifest exists to avoid. Refuse rather than quietly widen.
        msg = f"{path} declares no loader_files, so the prefetch would download every format"
        raise ValueError(msg)
    return models, loader_files


def cache_key(models: list[dict], loader_files: list[str]) -> str:
    """A digest of the pins and the selection, stable across orderings of either.

    The selection is in the key deliberately. Without it, a cache populated under a
    narrower pattern list restores into a run that needs a wider one, and the missing
    files are discovered by a notebook at load time rather than by the prefetch.
    """
    pairs = sorted(f"{m['repo_id']}@{m['revision']}" for m in models)
    return hashlib.sha256("\n".join([*pairs, "--", *sorted(loader_files)]).encode()).hexdigest()[
        :16
    ]


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


def fetch(models: list[dict], loader_files: list[str]) -> int:
    from huggingface_hub import snapshot_download

    for m in models:
        print(
            f"prefetching {m['repo_id']}@{m['revision']} ({len(loader_files)} patterns)", flush=True
        )
        path = snapshot_download(
            repo_id=m["repo_id"],
            revision=m["revision"],
            allow_patterns=loader_files,
        )
        print(f"  -> {path}", flush=True)
    return 0


def select(models: list[dict], loader_files: list[str]) -> int:
    """What the selection resolves to, through the real discovery path, no bodies."""
    import urllib.request

    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi()
    whole = chosen = 0
    for m in models:
        repo, rev = m["repo_id"], m["revision"]
        meta = json.loads(
            urllib.request.urlopen(  # noqa: S310 - the Hub's own metadata endpoint
                f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
            ).read()
        )
        sizes = {s["rfilename"]: (s.get("size") or 0) for s in meta["siblings"]}
        files = api.list_repo_files(repo, revision=rev)
        kept = sorted(filter_repo_objects(files, allow_patterns=loader_files))
        if not any(f.endswith(".safetensors") for f in kept):
            print(f"::error::{repo}@{rev} selection has no weights: {kept}")
            return 1
        whole += sum(sizes.values())
        chosen += sum(sizes.get(f, 0) for f in kept)
        print(f"{repo}@{rev[:8]}")
        print(f"   whole repo {sum(sizes.values()) / 1e6:9.1f} MB ({len(files)} files)")
        print(
            f"   selected   {sum(sizes.get(f, 0) for f in kept) / 1e6:9.1f} MB ({len(kept)} files)"
        )
        print(f"   skipped    {sorted(set(files) - set(kept))}")
    print(
        f"\ntotal {whole / 1e9:.3f} GB -> {chosen / 1e9:.3f} GB (saves {(whole - chosen) / 1e9:.3f} GB)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("key", "fetch", "select", "revisions"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    models, loader_files = load_manifest(args.manifest)
    if args.command == "key":
        print(cache_key(models, loader_files))
        return 0
    if args.command == "revisions":
        print(json.dumps({"models": models, "loader_files": loader_files}, indent=2))
        return 0
    if args.command == "select":
        return select(models, loader_files)
    return fetch(models, loader_files)


if __name__ == "__main__":
    sys.exit(main())
