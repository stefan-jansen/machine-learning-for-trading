"""What holds a backtest address, and does it still resolve.

A re-key moves ``backtest_runs.backtest_hash`` without recomputing anything, so its
acceptance test is not a number to replay. It is two questions asked of the registry and
the filesystem together:

1. Does every reference that resolved before the migration still resolve after it?
2. Does every row have an artifact directory, and every artifact directory a row?

Both have to be asked, and the second has to be counted in both directions. The
2026-09-14 re-key committed clean over 37,484 broken references because
``PRAGMA foreign_key_check`` is telling the truth about columns and four of the ten sites
holding a backtest address carry no foreign key, two of them being JSON documents that no
column-level check can see. Its restore then put the registries back and left every
artifact directory renamed, which read as a completed rollback under every check that
looked only at the database: 2,365 rows with no directory and 2,365 directories with no
row, a symmetry that says rename rather than loss and that nothing was asking about.

So the site list here is *derived*, not written down. :func:`hash_bearing_sites` reads the
schema and asks every text column whether it holds a registered address, which covers a
table added after this module was written. A list cannot.

Reading a registry is always read-only: these files are shared, and two risk lanes write
them while this runs.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Schema readers, imported rather than copied so the two cannot drift. `fold_renumbering`
# is the migration of this shape the package already carries.
from .fold_renumbering import _table_columns, _tables

__all__ = [
    "Bijection",
    "DanglingCount",
    "HashSite",
    "dangling_references",
    "declared_identities",
    "directory_bijection",
    "hash_bearing_sites",
    "open_readonly",
    "referential_regressions",
    "registered_addresses",
]

#: Every identity this registry family declares is twelve hex characters. The width is
#: load-bearing: ``cohort_metrics.member_digest`` is a sha256 over member addresses, so a
#: pattern that accepted 12 *to* 64 would read that derived value as a reference and call
#: it dangling. Sixty-four hex characters contain no twelve-character token under the
#: boundaries below, which is what keeps the two apart.
IDENTITY_WIDTH = 12

_IDENTITY_TOKEN = re.compile(rf"(?<![0-9a-f])[0-9a-f]{{{IDENTITY_WIDTH}}}(?![0-9a-f])")

#: The columns that *declare* an identity rather than referring to one. A value here is
#: not a reference and cannot dangle.
_DECLARING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("backtest_runs", "backtest_hash"),
    ("training_runs", "training_hash"),
    ("prediction_sets", "prediction_hash"),
    ("candidate_sets", "set_hash"),
    ("official_populations", "population_hash"),
    ("decision_artifacts", "decision_hash"),
    ("causal_runs", "causal_hash"),
)

#: Columns that hold hash-shaped text which is not a registry identity, so a token in them
#: resolving to nothing says nothing. ``official_populations.name`` carries a git commit
#: prefix (``g`` then twelve hex) beside the prediction hash the population was cut from.
#: They are reported like any other column rather than suppressed: the gate is a per-column
#: delta, so a constant over-count cannot hide a reference that broke, and suppressing a
#: column is how a site stops being watched.

#: Artifact subdirectories that are not a backtest address. A quarantined directory is
#: deliberately outside the bijection: it is what a merged collision left behind, and
#: counting it would report the merge as a broken migration.
QUARANTINE_DIRNAME = ".quarantine"


def open_readonly(registry: Path | str) -> sqlite3.Connection:
    """Open a registry for reading and nothing else."""
    return sqlite3.connect(f"file:{Path(registry)}?mode=ro", uri=True)


def declared_identities(db: sqlite3.Connection) -> dict[str, set[str]]:
    """Every identity the registry declares, by the table that owns it.

    A table absent from this registry contributes an empty set rather than raising, because
    the nine canonical registries were created over months and do not all carry every table.
    """
    owned: dict[str, set[str]] = {}
    present = set(_tables(db))
    for table, column in _DECLARING_COLUMNS:
        if table not in present or column not in _table_columns(db, table):
            owned[table] = set()
            continue
        owned[table] = {
            str(row[0]) for row in db.execute(f'SELECT "{column}" FROM "{table}"') if row[0]
        }
    return owned


def registered_addresses(db: sqlite3.Connection) -> set[str]:
    """The addresses ``backtest_runs`` currently holds."""
    return declared_identities(db)["backtest_runs"]


@dataclass(frozen=True)
class HashSite:
    """One ``(table, column)`` that holds at least one registered backtest address."""

    table: str
    column: str
    #: values that are an address and nothing else
    bare: int
    #: values that are a document with an address inside, which no column rewrite reaches
    embedded: int

    @property
    def total(self) -> int:
        return self.bare + self.embedded

    @property
    def is_document(self) -> bool:
        return self.embedded > 0


def hash_bearing_sites(
    db: sqlite3.Connection, addresses: Iterable[str] | None = None
) -> list[HashSite]:
    """Every column holding a registered backtest address, found by asking rather than listing.

    *addresses* defaults to what ``backtest_runs`` declares. Pass a different set to ask the
    question of a subset, such as only the rows a migration plans to move.

    The declaring column itself is excluded: ``backtest_runs.backtest_hash`` is where an
    address lives, not a reference to one.
    """
    wanted = set(addresses) if addresses is not None else registered_addresses(db)
    if not wanted:
        return []
    declared = set(_DECLARING_COLUMNS)
    sites: list[HashSite] = []
    for table in _tables(db):
        for column in _table_columns(db, table):
            if (table, column) in declared:
                continue
            bare = embedded = 0
            for (value,) in _text_values(db, table, column):
                if value in wanted:
                    bare += 1
                elif len(value) > IDENTITY_WIDTH and set(_IDENTITY_TOKEN.findall(value)) & wanted:
                    embedded += 1
            if bare or embedded:
                sites.append(HashSite(table=table, column=column, bare=bare, embedded=embedded))
    return sites


@dataclass(frozen=True)
class DanglingCount:
    """How many values in one column name an identity the registry does not hold."""

    table: str
    column: str
    count: int


def dangling_references(db: sqlite3.Connection) -> dict[tuple[str, str], int]:
    """Per column, how many values name an identity no declaring table holds.

    By value rather than by schema, because two of the ten sites a backtest address reaches
    are JSON documents and none of the four the 2026-09-14 run missed carries a foreign key.

    **The number is only meaningful as a delta.** A healthy registry dangles by design: a
    superseded population generation keeps its old member list as history, and a population
    published before its sweep finishes names members that do not exist yet. So the question
    a migration has to answer is whether a count went up, which is
    :func:`referential_regressions`.
    """
    known: set[str] = set()
    for owned in declared_identities(db).values():
        known |= owned
    declared = set(_DECLARING_COLUMNS)
    counts: dict[tuple[str, str], int] = {}
    for table in _tables(db):
        for column in _table_columns(db, table):
            if (table, column) in declared:
                continue
            dangling = 0
            for (value,) in _text_values(db, table, column):
                if len(value) == IDENTITY_WIDTH:
                    if _IDENTITY_TOKEN.fullmatch(value) and value not in known:
                        dangling += 1
                else:
                    tokens = set(_IDENTITY_TOKEN.findall(value))
                    if tokens and not tokens <= known:
                        dangling += 1
            if dangling:
                counts[(table, column)] = dangling
    return counts


def referential_regressions(
    before: dict[tuple[str, str], int], after: dict[tuple[str, str], int]
) -> dict[tuple[str, str], tuple[int, int]]:
    """The columns that dangle more after than before, as ``{site: (before, after)}``.

    Empty means no reference that resolved before fails to resolve now. That, and not an
    absolute count of zero, is what a migration has to hold.
    """
    return {
        site: (before.get(site, 0), count)
        for site, count in after.items()
        if count > before.get(site, 0)
    }


@dataclass(frozen=True)
class Bijection:
    """Whether ``backtest_runs`` and ``run_log/backtest/`` name the same set of addresses."""

    rows: int
    directories: int
    rows_without_directory: tuple[str, ...]
    directories_without_row: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return not self.rows_without_directory and not self.directories_without_row


def directory_bijection(case_dir: Path | str, *, registry: Path | str | None = None) -> Bijection:
    """Count rows with no artifact directory and directories with no row, both ways.

    Counting one direction is what made the 2026-09-15 breakage invisible: the registries
    passed ``integrity_check`` with zero dangling members while every artifact directory sat
    under a name no row claimed. The symmetry of the two counts is also what said the cause
    was a rename and not data loss, so both numbers are reported even when both are zero.
    """
    root = Path(case_dir)
    db_path = Path(registry) if registry is not None else root / "run_log" / "registry.db"
    with open_readonly(db_path) as db:
        rows = registered_addresses(db)
    backtests = root / "run_log" / "backtest"
    directories = (
        {
            entry.name
            for entry in backtests.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        }
        if backtests.exists()
        else set()
    )
    return Bijection(
        rows=len(rows),
        directories=len(directories),
        rows_without_directory=tuple(sorted(rows - directories)),
        directories_without_row=tuple(sorted(directories - rows)),
    )


def _text_values(db: sqlite3.Connection, table: str, column: str) -> list[tuple[str]]:
    """Every non-null string in one column, or nothing when the column cannot be read."""
    try:
        return [
            (row[0],)
            for row in db.execute(f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL')
            if isinstance(row[0], str)
        ]
    except sqlite3.Error:
        return []
