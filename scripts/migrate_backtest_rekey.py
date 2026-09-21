#!/usr/bin/env python3
"""Check what a registry's backtest addresses reach, before and after a re-key.

Three questions, all read-only::

    scripts/migrate_backtest_rekey.py check --case-study nasdaq100_microstructure
    scripts/migrate_backtest_rekey.py check --case-study nasdaq100_microstructure \
        --baseline /path/to/registry-before.db

``plan`` classifies every row and proves or refuses every collision, and ``check`` reports the columns that hold a backtest address, found by reading the schema
rather than a list; the references that resolve to nothing; and whether every row has an
artifact directory and every artifact directory a row, counted in both directions.

With ``--baseline`` it compares against a copy of the registry taken before a migration and
exits non-zero when a reference that resolved then does not resolve now. That delta is the
gate, not an absolute count: a healthy registry dangles by design, because a superseded
population keeps its old member list as history and a population published before its sweep
finishes names members that do not exist yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from case_studies.utils.registry.backtest_rekey import (
    dangling_references,
    directory_bijection,
    hash_bearing_sites,
    open_readonly,
    plan_backtest_rekey,
    referential_regressions,
)
from utils.paths import get_case_study_dir


def _report(case_dir: Path, baseline: Path | None) -> int:
    registry = case_dir / "run_log" / "registry.db"
    if not registry.exists():
        print(f"{case_dir.name}: no registry.db")
        return 0

    with open_readonly(registry) as db:
        sites = hash_bearing_sites(db)
        dangling = dangling_references(db)

    print(f"=== {case_dir.name}")
    print(f"  sites holding a backtest address: {len(sites)}")
    for site in sorted(sites, key=lambda s: (-s.total, s.table, s.column)):
        shape = "document" if site.is_document else "column"
        print(f"    {site.table}.{site.column:<22} {site.total:>8,}  {shape}")

    bijection = directory_bijection(case_dir)
    print(f"  rows {bijection.rows:,}  directories {bijection.directories:,}")
    print(f"    rows with no directory     {len(bijection.rows_without_directory):>8,}")
    print(f"    directories with no row    {len(bijection.directories_without_row):>8,}")

    if dangling:
        print("  dangling references")
        for (table, column), count in sorted(dangling.items(), key=lambda kv: -kv[1]):
            print(f"    {table}.{column:<22} {count:>8,}")
    else:
        print("  dangling references: none")

    failures = 0
    if not bijection.exact:
        print("  FAIL: the registry and the artifact tree do not name the same addresses")
        failures += 1

    if baseline is not None:
        with open_readonly(baseline) as db:
            before = dangling_references(db)
        worse = referential_regressions(before, dangling)
        if worse:
            print("  FAIL: references that resolved in the baseline and do not now")
            for (table, column), (was, now) in sorted(worse.items(), key=lambda kv: kv[0]):
                print(f"    {table}.{column:<22} {was:,} -> {now:,}")
            failures += 1
        else:
            print("  no reference that resolved in the baseline fails to resolve now")
    return failures


def _plan(case_dir: Path) -> int:
    """Report what a re-key would do to one registry. Writes nothing."""
    registry = case_dir / "run_log" / "registry.db"
    if not registry.exists():
        # A worktree built without `--case-study` carries no `run_log` symlink at all. One
        # absent registry must not abort the pass over the other eight.
        print(f"{case_dir.name}: no registry.db")
        return 0

    plan = plan_backtest_rekey(case_dir)
    verdict = "SKIP" if plan.skipped else "migrate"
    print(
        f"{plan.case_study:<32} rows {plan.rows:>7,}  reachable {plan.reachable:>7,}  "
        f"moved {plan.moved:>6,}  occupied {plan.occupied:>4}  re-keys {len(plan.mapping):>6,}"
        f"  {verdict}"
    )
    proved = sum(1 for merge in plan.merges if merge.proved)
    if plan.merges:
        print(f"    collisions proved {proved} of {len(plan.merges)}")
    for reason in plan.refusals:
        print(f"    REFUSED  {reason}")
    return 1 if plan.skipped else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="report sites, dangling references and the bijection")
    check.add_argument("--case-study", action="append", default=[], help="repeatable")
    check.add_argument("--case-dir", type=Path, action="append", default=[])
    check.add_argument(
        "--baseline",
        type=Path,
        help="a copy of the registry taken before a migration; fails on any new dangle",
    )
    plan = sub.add_parser("plan", help="classify every row and prove every collision")
    plan.add_argument("--case-study", action="append", default=[], help="repeatable")
    plan.add_argument("--case-dir", type=Path, action="append", default=[])
    args = parser.parse_args()

    case_dirs = [get_case_study_dir(name) for name in args.case_study]
    case_dirs.extend(args.case_dir)
    if not case_dirs:
        parser.error("pass at least one --case-study or --case-dir")
    if args.command == "plan":
        sys.exit(1 if sum(_plan(Path(case_dir)) for case_dir in case_dirs) else 0)

    if args.baseline is not None and len(case_dirs) != 1:
        parser.error("--baseline compares one registry against one baseline")

    failures = sum(_report(Path(case_dir), args.baseline) for case_dir in case_dirs)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
