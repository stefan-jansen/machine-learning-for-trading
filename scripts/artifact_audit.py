#!/usr/bin/env python3
"""Report registered fits whose named feature artifact is not on disk.

A training identity carries ``computation.feature_artifacts.<role>.sha256``, the hash of
the artifact the fit read. Nothing else checks the file is still there, so a fit whose
input is gone resolves, reads as healthy, and is a record of a result rather than a
reproducible one.

**The scope this walks is printed, and that is not decoration.** An earlier run of an
uncommitted version of this audit reported ``crypto_perps_funding/financial`` as missing
for 144 fits while the file sat at the canonical path under its registered hash, untouched
for two months (ml4t/agent-workspace#1176). A case study's ``features/`` is a symlink into
the artifacts root in every checkout that has one, so an audit run from a throwaway
worktree - or one that does not resolve the link - hashes a different tree than the fits
read and reports absence with full confidence. Every path below is resolved and printed,
and a resolved path outside the artifacts root is a hard error rather than a quiet
mis-scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.paths import get_case_study_dir  # noqa: E402

CASE_STUDIES = (
    "cme_futures",
    "crypto_perps_funding",
    "etfs",
    "fx_pairs",
    "nasdaq100_microstructure",
    "sp500_equity_option_analytics",
    "sp500_options",
    "us_equities_panel",
    "us_firm_characteristics",
)
ARTIFACT_DIRS = ("features", "labels")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_artifacts(spec_json: str) -> list[tuple[str, str, int | None]]:
    """``(role, sha256, size)`` for one spec, across both shapes the registry holds.

    A dict keyed by role, and a list of ``{role, sha256}`` whose hashes may carry a
    ``sha256:`` prefix. Matching on the hash alone reaches both, but the role and size are
    what make a report readable, so both shapes are parsed rather than regex-scraped.
    """
    artifacts = json.loads(spec_json).get("computation", {}).get("feature_artifacts")
    out: list[tuple[str, str, int | None]] = []
    if isinstance(artifacts, dict):
        for role, entry in artifacts.items():
            if isinstance(entry, dict) and entry.get("sha256"):
                out.append((role, str(entry["sha256"]).removeprefix("sha256:"), entry.get("size")))
    elif isinstance(artifacts, list):
        for entry in artifacts:
            if isinstance(entry, dict) and entry.get("sha256"):
                out.append(
                    (
                        str(entry.get("role", "?")),
                        str(entry["sha256"]).removeprefix("sha256:"),
                        entry.get("size"),
                    )
                )
    return out


def walk(
    case_study: str, artifacts_root: Path | None
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Hash every parquet the case study's artifact directories hold, after resolving them.

    The third return is the directories that were not there. An absent one is not a small
    gap in an otherwise good answer: every hash it would have contributed is absent from
    ``on_disk``, so every fit that read it is counted as naming a missing artifact. The
    caller has to know, because the two readings are opposite and the output is identical.
    """
    case_dir = get_case_study_dir(case_study)
    on_disk: dict[str, Path] = {}
    scope: list[str] = []
    absent: list[str] = []
    for name in ARTIFACT_DIRS:
        declared = case_dir / name
        if not declared.exists():
            scope.append(f"    {name}/  ABSENT at {declared}")
            absent.append(f"{case_study}/{name} at {declared}")
            continue
        resolved = declared.resolve()
        scope.append(f"    {name}/  -> {resolved}")
        if artifacts_root is not None and artifacts_root not in resolved.parents:
            raise SystemExit(
                f"{case_study}/{name} resolves to {resolved}, which is not under "
                f"{artifacts_root}. Hashing it would compare the registry's fits against a "
                f"tree they never read. Run from a checkout whose artifact directories link "
                f"into the artifacts root, or pass --artifacts-root to name the tree you mean."
            )
        for parquet in sorted(resolved.rglob("*.parquet")):
            on_disk.setdefault(sha256_of(parquet), parquet)
    return on_disk, scope, absent


def default_artifacts_root(case_studies: tuple[str, ...]) -> Path | None:
    """The tree the artifact directories actually live in, taken from the first that exists.

    Derived from a RESOLVED artifact directory, never from the case-study directory: in a
    checkout those are two different trees, because ``case_studies/<cs>/features`` is a
    symlink into the artifacts root and its unresolved parent is the repository.
    """
    for case_study in case_studies:
        case_dir = get_case_study_dir(case_study)
        for name in ARTIFACT_DIRS:
            declared = case_dir / name
            if declared.exists():
                # <root>/<case study>/<features|labels>
                return declared.resolve().parents[1]
    return None


def audit(
    case_studies: tuple[str, ...], artifacts_root: Path | None
) -> tuple[int, list[str], list[str]]:
    """Returns the missing-fit count, the case studies skipped, and the directories unseen.

    The second and third are not details. A partial audit that prints "0 fits name an
    artifact that is not on disk" is indistinguishable from a clean one, and a checkout is
    missing a registry whenever its gitignored ``run_log`` symlink was never created -
    which is the normal state of most worktrees, not an exception.

    The third exists because the second was not enough. An absent *artifact directory*
    inside an audited case study used to print one ABSENT line in the middle of the scope
    block and then contribute every one of that case study's fits to a confident total.
    Measured 2026-09-14 on a checkout whose ``us_equities_panel`` had ``run_log`` but no
    ``features/``: the audit reported 393 fits naming missing artifacts and exited 1, with
    no PARTIAL, while ``financial.parquet`` sat on disk at exactly the 4,478,156,899 bytes
    the MISSING line quoted. A case study that could not be fully seen is now not counted
    at all, because there is no way to tell its real findings from its blind ones.
    """
    total_missing = 0
    unaudited: list[str] = []
    unseen: list[str] = []
    for case_study in case_studies:
        db = get_case_study_dir(case_study) / "run_log" / "registry.db"
        print(f"\n{case_study}")
        if not db.exists():
            print(f"    NOT AUDITED: no registry at {db}")
            unaudited.append(case_study)
            continue
        on_disk, scope, absent = walk(case_study, artifacts_root)
        for line in scope:
            print(line)
        if absent:
            print(
                f"    NOT AUDITED: {len(absent)} artifact directory(ies) absent, so every "
                f"fit here would read as missing. Nothing from this case study is counted."
            )
            unseen.extend(absent)
            continue
        print(f"    {len(on_disk)} distinct parquet hashed")

        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT training_hash, spec_json FROM training_runs")
            missing: Counter[tuple[str, str, int | None]] = Counter()
            fits = 0
            for _, spec_json in rows:
                fits += 1
                for role, sha, size in registered_artifacts(spec_json):
                    if sha not in on_disk:
                        missing[(role, sha, size)] += 1
        finally:
            connection.close()

        if not missing:
            print(f"    {fits} fits, every named artifact present")
            continue
        for (role, sha, size), count in sorted(missing.items(), key=lambda kv: -kv[1]):
            size_text = f"{size:,} bytes" if size else "size not recorded"
            print(f"    MISSING  {role:<14} {sha[:16]}…  {size_text}  named by {count} fits")
            total_missing += count
    return total_missing, unaudited, unseen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study", action="append", choices=CASE_STUDIES)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        help=(
            "Tree every artifact directory must resolve under. Defaults to the parent of the "
            "first case study's resolved directory. Pass 'none' to disable the check, which "
            "is how you audit a deliberately isolated tree."
        ),
    )
    args = parser.parse_args()
    selected = tuple(args.case_study) if args.case_study else CASE_STUDIES

    if args.artifacts_root is not None and str(args.artifacts_root) == "none":
        artifacts_root = None
    elif args.artifacts_root is not None:
        artifacts_root = args.artifacts_root.resolve()
    else:
        artifacts_root = default_artifacts_root(selected)
        if artifacts_root is None:
            raise SystemExit(
                "no case study has a features/ or labels/ directory, so there is no tree to "
                "audit. Pass --artifacts-root to name one explicitly."
            )

    print(f"artifacts root: {artifacts_root if artifacts_root else 'UNCHECKED'}")
    total, unaudited, unseen = audit(selected, artifacts_root)
    if unaudited or unseen:
        # Deliberately no total. A number printed here is read as the answer however it is
        # qualified, and over an incomplete scope it is not one.
        print("\nPARTIAL: this audit did not see the whole tree, so it reports no count.")
        if unaudited:
            print(
                f"  {len(unaudited)} of {len(selected)} case studies had no registry: "
                f"{', '.join(unaudited)}"
            )
        if unseen:
            print(f"  {len(unseen)} artifact directory(ies) absent:")
            for line in unseen:
                print(f"    {line}")
        print("  Run from a checkout where every case study links into the artifacts root.")
        return 2
    print(f"\n{total} fits name an artifact that is not on disk")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
