#!/usr/bin/env python3
"""Record, in a new artifact's sidecar, which older artifact it replaces and what survived.

A stage-04 temporal artifact is pinned into every training identity fitted on it by
whole-file sha256. Appending the holdout fold - which is what a holdout retrain needs the
artifact to carry - changes that digest, and the lock then refuses the retrain even though
every fold the configuration was selected under is unchanged. Whether the file was extended
or rewritten is exactly what a whole-file digest cannot say.

This script says it, once, while both files are still on disk: it verifies the old file
against the sha256 the lock names, digests each file per fold, and writes into the new
file's sidecar the old file's sha256 together with the per-fold digests of the folds that
came through unchanged. The replaced sha256 ties the record to one pin and to no other, so
a supersession recorded against a different vintage cannot be read as covering this one.

It refuses to record anything if a fold the old file held is missing from the new one or
holds different values there. That is a replacement, not an extension, and no result
fitted on the old file may be carried across it.

The record is evidence about a file that will not exist much longer. What consumes it -
which check admits which retrain - is decided where locks are taken and executed, not
here; this script only makes the answer recordable while it is still knowable.

Usage:
    uv run python scripts/record_artifact_supersession.py \
        --superseded ~/ml4t/preserved/.../model_based.parquet \
        --current case_studies/fx_pairs/features/model_based.parquet \
        [--expect-sha256 750be334...] [--fold-column fold] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from case_studies.utils.artifact_digest import (  # noqa: E402
    fold_digests,
    read_digest,
    sidecar_path,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--superseded", type=Path, required=True, help="the artifact being replaced"
    )
    parser.add_argument("--current", type=Path, required=True, help="the artifact now in place")
    parser.add_argument(
        "--expect-sha256",
        help="refuse unless the superseded file hashes to this - paste it from the lock",
    )
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    old_path, new_path = args.superseded.expanduser(), args.current.expanduser()
    for path in (old_path, new_path):
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            return 1
    if old_path.resolve() == new_path.resolve():
        print("error: an artifact cannot supersede itself", file=sys.stderr)
        return 1

    old_sha = sha256_file(old_path)
    if args.expect_sha256 and old_sha != args.expect_sha256:
        print(
            f"error: {old_path} hashes to {old_sha}, not the expected {args.expect_sha256}. "
            "This is not the artifact the lock pins.",
            file=sys.stderr,
        )
        return 1

    old_folds = fold_digests(pl.read_parquet(old_path), fold_column=args.fold_column)
    new_folds = fold_digests(pl.read_parquet(new_path), fold_column=args.fold_column)

    missing = sorted(fold for fold in old_folds if fold not in new_folds)
    changed = sorted(
        fold
        for fold, digest in old_folds.items()
        if fold in new_folds and new_folds[fold] != digest
    )
    if missing or changed:
        if missing:
            print(f"error: folds {missing} are in {old_path.name} and not in {new_path.name}")
        if changed:
            print(f"error: folds {changed} hold different values in {new_path.name}")
        print(
            "This file replaces the pinned artifact rather than extending it, so no lock "
            "fitted on the old one may be reconstructed against it. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    added = sorted(set(new_folds) - set(old_folds), key=lambda fold: (len(fold), fold))
    record = read_digest(new_path)
    record["supersedes"] = {"sha256": old_sha, "fold_digests": old_folds}
    record.setdefault("fold_digests", new_folds)

    print(f"superseded: {old_path}  sha256 {old_sha[:12]}  folds {sorted(old_folds)}")
    print(f"current:    {new_path}  folds {sorted(new_folds)}")
    print(f"unchanged:  {sorted(old_folds)}")
    print(f"added:      {added or 'none'}")
    if args.dry_run:
        print("--dry-run: sidecar not written")
        return 0
    sidecar_path(new_path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {sidecar_path(new_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
