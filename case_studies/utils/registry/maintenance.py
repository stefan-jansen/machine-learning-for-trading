"""Auditable, idempotent maintenance for semantically duplicate backtests."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .specs import backtest_hash_from_parts


@dataclass(frozen=True)
class DuplicateBacktest:
    keep_hash: str
    drop_hashes: tuple[str, ...]


def find_semantic_backtest_duplicates(db_path: Path) -> list[DuplicateBacktest]:
    """Find rows differing only by normalized, identity-neutral spec defaults."""
    with closing(sqlite3.connect(str(db_path))) as db:
        rows = db.execute(
            "SELECT backtest_hash, prediction_hash, stage, spec_json FROM backtest_runs"
        ).fetchall()

    groups: dict[tuple[str, str | None, str], list[str]] = {}
    for stored_hash, prediction_hash, stage, spec_json in rows:
        if not spec_json:
            continue
        spec = json.loads(spec_json)
        semantic_hash = backtest_hash_from_parts(prediction_hash, spec)
        groups.setdefault((prediction_hash, stage, semantic_hash), []).append(stored_hash)

    duplicates = []
    for hashes in groups.values():
        if len(hashes) < 2:
            continue
        ordered = sorted(hashes)
        duplicates.append(DuplicateBacktest(ordered[0], tuple(ordered[1:])))
    return sorted(duplicates, key=lambda item: item.keep_hash)


def deduplicate_semantic_backtests(
    db_path: Path, *, apply: bool = False
) -> list[DuplicateBacktest]:
    """Remove unreferenced semantic duplicates; dry-run unless ``apply`` is true."""
    duplicates = find_semantic_backtest_duplicates(db_path)
    if not apply or not duplicates:
        return duplicates

    drops = [item for group in duplicates for item in group.drop_hashes]
    with closing(sqlite3.connect(str(db_path))) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        placeholders = ",".join("?" for _ in drops)
        references = {
            "backtest_paired_metrics.challenger_hash": db.execute(
                f"SELECT challenger_hash FROM backtest_paired_metrics WHERE challenger_hash IN ({placeholders})",
                drops,
            ).fetchall(),
            "backtest_paired_metrics.benchmark_hash": db.execute(
                f"SELECT benchmark_hash FROM backtest_paired_metrics WHERE benchmark_hash IN ({placeholders})",
                drops,
            ).fetchall(),
            "cohort_metrics.leader_hash": db.execute(
                f"SELECT leader_hash FROM cohort_metrics WHERE leader_hash IN ({placeholders})",
                drops,
            ).fetchall(),
        }
        live_refs = {name: rows for name, rows in references.items() if rows}
        if live_refs:
            raise RuntimeError(f"Refusing to delete referenced backtests: {live_refs}")

        db.execute(
            f"DELETE FROM backtest_fold_metrics WHERE backtest_hash IN ({placeholders})", drops
        )
        db.execute(f"DELETE FROM backtest_metrics WHERE backtest_hash IN ({placeholders})", drops)
        db.execute(f"DELETE FROM backtest_runs WHERE backtest_hash IN ({placeholders})", drops)
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    groups = deduplicate_semantic_backtests(args.registry, apply=args.apply)
    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: {sum(len(group.drop_hashes) for group in groups)} duplicate rows")
    for group in groups:
        print(f"keep={group.keep_hash} drop={','.join(group.drop_hashes)}")


if __name__ == "__main__":
    main()
