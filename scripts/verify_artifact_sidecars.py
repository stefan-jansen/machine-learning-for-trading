#!/usr/bin/env python
"""Check every case-study artifact against the digest sidecar its producer wrote.

`case_studies/utils/artifact_digest.py` writes `<artifact>.digest.json` beside every
label and feature parquet, recording a hash of the values, the row count, the key
columns, and the digest of each input. Nothing read one until
`utils.modeling.verify_artifact_sidecars`, and that runs only under
`load_modeling_dataset(..., verify_input_digests=True)` because verifying a *content*
digest means hashing every row - a multi-GB re-read on the larger panels, paid on
every load, for a state only an artifact regeneration can create.

This is the other end of that trade: run it once, after the regeneration, over
everything. A run costs what one full load would have cost per artifact, and it
reports the whole set rather than stopping at the first disagreement.

    uv run python scripts/verify_artifact_sidecars.py
    uv run python scripts/verify_artifact_sidecars.py --case-study etfs
    uv run python scripts/verify_artifact_sidecars.py --require-sidecar

Exit 0 means every artifact hashes to what its producer recorded. Exit 1 names each
one that does not. Without `--require-sidecar` an artifact with no sidecar is
reported and does not fail, because artifacts written before the sidecar existed are
in that state and cannot be told apart from a silent change - which is the whole
reason to regenerate them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import polars as pl  # noqa: E402

from case_studies.utils.artifact_digest import read_digest, sidecar_path, value_digest  # noqa: E402
from utils.paths import get_case_study_dir  # noqa: E402


def artifacts_for(case_dir: Path) -> list[Path]:
    """Every label and feature parquet the case study has written."""
    return sorted(
        p
        for sub in ("labels", "features")
        for p in (case_dir / sub).glob("*.parquet")
        if (case_dir / sub).is_dir()
    )


def check(path: Path, *, require_sidecar: bool) -> tuple[str, str]:
    """Return (status, detail) for one artifact. Status is ok, absent or a failure."""
    if not sidecar_path(path).exists():
        return ("missing" if require_sidecar else "absent", "no digest sidecar")
    try:
        record = read_digest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "unreadable", str(exc)
    recorded = record.get("digest")
    if not recorded:
        return "no-digest", "sidecar records no digest"
    try:
        frame = pl.read_parquet(path)
        actual = value_digest(frame)
    except (OSError, KeyError, pl.exceptions.PolarsError) as exc:
        # Reported and carried on rather than raised: the point of this script is
        # that one run tells you about every artifact, and a truncated parquet is
        # exactly the kind of thing you want listed beside the others.
        return "unreadable", f"{type(exc).__name__}: {exc}"
    if actual != recorded:
        return "changed", f"hashes {actual}, sidecar records {recorded}"
    rows = record.get("n_rows")
    if rows is not None and int(rows) != frame.height:
        return "changed", f"{frame.height} rows, sidecar records {rows}"
    return "ok", recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", action="append", dest="case_studies")
    parser.add_argument(
        "--require-sidecar",
        action="store_true",
        help="fail on an artifact whose producer recorded nothing, not only on one that changed",
    )
    args = parser.parse_args()

    root = REPO_ROOT / "case_studies"
    names = args.case_studies or sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "config" / "setup.yaml").exists()
    )

    failures: list[str] = []
    absent = 0
    verified = 0
    checked_any = False
    for name in names:
        # Resolved the way the notebooks resolve it, so a --case-study worktree
        # reads the canonical artifacts through its symlink rather than an empty
        # directory beside the checkout.
        for path in artifacts_for(get_case_study_dir(name, create=False)):
            checked_any = True
            rel = path
            status, detail = check(path, require_sidecar=args.require_sidecar)
            if status == "ok":
                verified += 1
            elif status == "absent":
                absent += 1
                print(f"  no sidecar  {rel}")
            else:
                failures.append(f"{rel}: {detail}")
                print(f"  {status.upper():10s} {rel}: {detail}")

    print(f"\n{verified} verified, {absent} without a sidecar, {len(failures)} failed")
    if not checked_any:
        print(
            "\nNo artifact was found for any of "
            f"{', '.join(names)}. A worktree built without --case-study has no "
            "labels/ or features/ to read; re-create it with --case-study, or run "
            "this from one that has them."
        )
        return 1
    if failures:
        print(
            "\nAn artifact that does not hash to what its producer recorded has changed since it "
            "was written, so anything trained on it carries an identity that does not describe "
            "its inputs. Re-run the notebook that writes it."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
