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

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Schema readers, imported rather than copied so the two cannot drift. `fold_renumbering`
# is the migration of this shape the package already carries.
from .fold_renumbering import _table_columns, _tables
from .specs import (
    _DEFAULT_ELIDED_CONFIG_KEYS,
    _hashable_strategy_spec,
    _is_engine_default,
    backtest_hash_from_hashable,
    backtest_hash_from_parts,
)

__all__ = [
    "Bijection",
    "DanglingCount",
    "HashSite",
    "Merge",
    "RekeyPlan",
    "Row",
    "dangling_references",
    "declared_identities",
    "directory_bijection",
    "hash_bearing_sites",
    "open_readonly",
    "plan_backtest_rekey",
    "prove_merge",
    "pre_elision_hash",
    "read_rows",
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

#: How far two artifacts may drift and still be the same result. A float sum is not
#: associative, so the same computation reduced in a different order lands within an ulp or
#: two; the observed gap in the nine is 5.6e-17 on values of order 0.05. These are wide
#: enough to absorb that and far too tight to absorb a different run: the smallest real
#: disagreement measured is 0.4 on a return.
_ARTIFACT_ATOL = 1e-12
_ARTIFACT_RTOL = 1e-9


@contextmanager
def open_readonly(registry: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a registry for reading and nothing else, and close it on the way out.

    A context manager rather than a bare connection because ``sqlite3.Connection.__exit__``
    ends the transaction and leaves the handle open. These registries are shared and two of
    them are under live writes, so a scan over the nine must not leave eighteen descriptors
    and their WAL mappings open until the garbage collector gets to them.
    """
    db = sqlite3.connect(f"file:{Path(registry)}?mode=ro", uri=True)
    try:
        yield db
    finally:
        db.close()


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

    **Tokens, not rows.** A document contributes one count per broken *occurrence* inside
    it, so an address named twice counts twice. Per row it would contribute one whether one
    member dangled or forty, so a migration that broke thirty-nine more references inside a
    snapshot that already dangled would move no count at all - in the two columns this check
    exists to cover, and nowhere else. Occurrences rather than distinct addresses because
    both sides of the delta are counted the same way, and a repeated member in a snapshot is
    itself a reference that has to keep resolving.

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
                    dangling += sum(
                        1 for token in _IDENTITY_TOKEN.findall(value) if token not in known
                    )
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


# --------------------------------------------------------------------------------------
# Planning a re-key
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One ``backtest_runs`` row, with the address it is stored under and the one it computes."""

    address: str
    computed: str
    prediction_hash: str
    created_at: str
    stage: str | None
    digests: dict[str, str]
    spec: dict

    @property
    def moved(self) -> bool:
        return self.computed != self.address


@dataclass(frozen=True)
class Merge:
    """Two or more rows computing one address, and whether they may be merged.

    ``proved`` is the whole gate. A target is occupied only when two rows carry the same
    spec under today's hasher, and the elision proof says the spec did not move and the
    hasher did, so an occupied target should mean the incumbent is a byte-identical twin.
    Where that is not true, two genuinely different results are claiming one address, which
    is a question about the hasher and not a case this migration decides.
    """

    target: str
    survivor: str
    retired: tuple[str, ...]
    proved: bool
    reason: str


@dataclass(frozen=True)
class RekeyPlan:
    """What a re-key would do to one registry, and whether it may run at all."""

    case_study: str
    rows: int
    reachable: int
    #: survivor address -> the address it computes, for every row that moves and stays
    mapping: dict[str, str]
    merges: tuple[Merge, ...]
    #: addresses neither today's hasher nor the pre-elision one explains
    unexplained: tuple[str, ...]
    #: addresses the re-key would read or write that are already an identity of some other
    #: kind in this registry - the vacated side as well as the minted one
    namespace_clashes: tuple[str, ...] = ()

    @property
    def moved(self) -> int:
        return self.rows - self.reachable

    @property
    def occupied(self) -> int:
        return len(self.merges)

    @property
    def refusals(self) -> tuple[str, ...]:
        """Why this registry cannot be migrated, one line per distinct reason.

        Grouped rather than listed per collision: ``sp500_equity_option_analytics`` refuses
        134 times for one reason and ``us_firm_characteristics`` 81 times, and 215 identical
        lines bury the one registry whose refusal might be different.
        """
        by_reason: dict[str, list[str]] = {}
        for merge in self.merges:
            if not merge.proved:
                by_reason.setdefault(merge.reason, []).append(merge.target)
        reasons = [
            f"{len(targets)} collisions refused, {reason}, first at "
            f"{', '.join(sorted(targets)[:2])}"
            for reason, targets in sorted(by_reason.items())
        ]
        if self.unexplained:
            reasons.append(
                f"{len(self.unexplained)} rows are stored at an address neither today's hasher "
                f"nor the pre-elision one computes, first {self.unexplained[0]}"
            )
        if self.namespace_clashes:
            reasons.append(
                f"{len(self.namespace_clashes)} addresses the re-key would move from or onto "
                f"are already an identity of another kind here, first "
                f"{self.namespace_clashes[0]}"
            )
        return tuple(reasons)

    @property
    def skipped(self) -> bool:
        """A registry with any refusal is skipped whole, not migrated in part."""
        return bool(self.refusals)


def pre_elision_hash(prediction_hash: str, spec: dict) -> str:
    """The address *spec* computed before the engine-default keys stopped being hashed.

    ``_DEFAULT_ELIDED_CONFIG_KEYS`` only ever removes keys whose value equals the engine
    default, so the pre-fix view is today's view with exactly those keys put back. Re-hashing
    the 6,841 moved rows this way reproduces the stored address for 6,841 of them, which is
    what says the spec did not move and the hasher did.
    """
    hashable = _hashable_strategy_spec(spec)
    stored_config = spec.get("backtest_config")
    hashable_config = hashable.get("backtest_config")
    if isinstance(stored_config, dict) and isinstance(hashable_config, dict):
        for (section, key), default in _DEFAULT_ELIDED_CONFIG_KEYS:
            block = stored_config.get(section)
            if isinstance(block, dict) and key in block and _is_engine_default(block[key], default):
                hashable_config.setdefault(section, {})[key] = block[key]
    return backtest_hash_from_hashable(
        prediction_hash, hashable, identity_version=spec.get("identity_version")
    )


def read_rows(db: sqlite3.Connection) -> list[Row]:
    """Every ``backtest_runs`` row, with the address its own stored spec computes today."""
    db.row_factory = sqlite3.Row
    rows: list[Row] = []
    for record in db.execute(
        "SELECT backtest_hash, prediction_hash, spec_json, stage, created_at, "
        "artifact_digests_json FROM backtest_runs"
    ):
        spec = json.loads(record["spec_json"]) if record["spec_json"] else {}
        digests = (
            json.loads(record["artifact_digests_json"]) if record["artifact_digests_json"] else {}
        )
        rows.append(
            Row(
                address=record["backtest_hash"],
                computed=backtest_hash_from_parts(record["prediction_hash"], spec),
                prediction_hash=record["prediction_hash"],
                created_at=record["created_at"],
                stage=record["stage"],
                digests=digests,
                spec=spec,
            )
        )
    return rows


def _frames_agree(left: Path, right: Path) -> bool | None:
    """Whether two artifact files hold the same numbers. ``None`` when they cannot be read.

    Not by value alone: a null is compared as a null and never as a number.

    Bytes are the wrong unit here. In ``us_firm_characteristics`` 47 of the 81 colliding
    pairs differ only in the last bit of a float - 0.05365977120707852 against ...854, on 4
    of 110 values - because a float sum lands differently depending on the order it is
    reduced in. That is the same result written twice, and a digest comparison calls it two
    results. Non-numeric columns still have to match exactly: a different symbol or a
    different timestamp is a different run, however close the weights are.
    """
    try:
        import polars as pl

        first, second = pl.read_parquet(left), pl.read_parquet(right)
    except Exception:
        return None
    if first.schema != second.schema or first.height != second.height:
        return False
    for column, dtype in first.schema.items():
        a, b = first[column], second[column]
        if dtype.is_float():
            # Nulls first, and on their own. `null - 0.49` is null, `null > tolerance` is
            # null, and `.any()` skips nulls, so a tolerance test alone reads a missing
            # value and a present one as agreement - the loudest difference there is.
            missing = a.is_null()
            if not missing.equals(b.is_null()):
                return False
            gap = (a - b).abs()
            tolerance = _ARTIFACT_ATOL + _ARTIFACT_RTOL * b.abs()
            # The comparison is null exactly where both sides are null, which the line above
            # has already established is agreement. Everywhere else it is a real verdict.
            if bool((gap > tolerance).fill_null(False).any()):
                return False
        elif not a.equals(b):
            return False
    return True


def _artifacts_agree(members: list[Row], artifact_root: Path) -> tuple[bool, str]:
    """Compare every recorded artifact of every member by value, not by digest."""
    first = members[0]
    for name in sorted(first.digests):
        reference = artifact_root / first.address / name
        for member in members[1:]:
            verdict = _frames_agree(reference, artifact_root / member.address / name)
            if verdict is None:
                return False, f"the colliding rows hold different artifacts ({name} unreadable)"
            if not verdict:
                return False, f"the colliding rows hold different numbers in {name}"
    return True, ""


def prove_merge(members: list[Row], *, artifact_root: Path | None = None) -> tuple[bool, str]:
    """Whether these rows are the same result under two addresses.

    Two independent equalities, both required, and both compared whole rather than by a
    count of keys: ``us_firm_characteristics`` records two artifact digests where the other
    registries record six, so a check that asserted six would pass it vacuously.

    Identical digests settle it. When they differ and *artifact_root* is given, the files are
    compared by value, because a digest cannot tell one-bit float noise from a different
    result and both appear in the nine. Without *artifact_root* a digest difference refuses,
    which is the conservative reading and what a caller with no artifact tree gets.
    """
    first = members[0]
    if not first.digests or any(not member.digests for member in members):
        return False, "a row in the collision records no artifact digests"
    if any(set(member.digests) != set(first.digests) for member in members):
        return False, "the colliding rows record different artifact names"
    reference = _hashable_strategy_spec(first.spec)
    if any(_hashable_strategy_spec(member.spec) != reference for member in members[1:]):
        return False, "the colliding rows hold different specs after the elided keys are stripped"
    if any(member.digests != first.digests for member in members):
        if artifact_root is None:
            return False, "the colliding rows hold different artifacts"
        agree, reason = _artifacts_agree(members, artifact_root)
        if not agree:
            return False, reason
    return True, ""


def _touched_addresses(mapping: Mapping[str, str], merges: Iterable[Merge]) -> set[str]:
    """Every address the re-key reads from or writes to, both sides of the rewrite."""
    touched = set(mapping) | set(mapping.values())
    for merge in merges:
        if merge.proved:
            touched.add(merge.target)
            touched.update(merge.retired)
    return touched


def plan_backtest_rekey(case_dir: Path | str, *, registry: Path | str | None = None) -> RekeyPlan:
    """Classify every row, and prove or refuse every collision. Writes nothing.

    The survivor of a collision is the earliest ``created_at`` row: it is the one
    ``candidate_set_members``, ``official_population_members`` and the reference tables
    already point at, and the one whose ``notebook`` and ``git_commit`` the published
    results trace to. Which side of the collision it sits on is not fixed - in
    ``nasdaq100_microstructure`` the row already at the target address is the later one in
    106 of 106 pairs, and in the two registries this refuses it is the earlier one - so the
    rule is stated over ``created_at`` and never over which address a row happens to hold.

    An address the re-key would move *from* or *onto* that is already an identity of some
    other kind refuses the registry. ``official_population_members.member_hash`` is
    polymorphic - it holds prediction hashes in ``etfs``, ``nasdaq100_microstructure`` and
    ``us_firm_characteristics``, and 4,882 backtest addresses beside 777 prediction hashes
    in ``crypto_perps_funding`` - and a re-key rewrites a reference site by matching the
    stored value, ``SET col = new WHERE col = old``.

    Both sides are checked, and they fail differently. A *vacated* address that is also a
    prediction hash is the silent one: the rewrite converts a prediction reference into a
    backtest reference and nothing downstream can tell. A *minted* address that is also a
    prediction hash leaves two kinds of identity sharing one value, so the next migration
    reads an alias rather than a reference.

    All six declaring namespaces are disjoint from backtest addresses in the nine registries
    today and neither side clashes. This measures it per run rather than carrying that
    forward: the re-key mints addresses that did not exist when the scan ran, and two lanes
    write these registries live.
    """
    root = Path(case_dir)
    db_path = Path(registry) if registry is not None else root / "run_log" / "registry.db"
    with open_readonly(db_path) as db:
        rows = read_rows(db)
        owned = declared_identities(db)
    foreign = set().union(
        *(taken for table, taken in owned.items() if table != "backtest_runs"), set()
    )

    unexplained = tuple(
        sorted(
            row.address
            for row in rows
            if row.moved and pre_elision_hash(row.prediction_hash, row.spec) != row.address
        )
    )

    by_target: dict[str, list[Row]] = {}
    for row in rows:
        by_target.setdefault(row.computed, []).append(row)

    mapping: dict[str, str] = {}
    merges: list[Merge] = []
    for target, members in sorted(by_target.items()):
        if len(members) == 1:
            row = members[0]
            if row.moved:
                mapping[row.address] = target
            continue
        ordered = sorted(members, key=lambda member: (member.created_at, member.address))
        survivor, retired = ordered[0], ordered[1:]
        proved, reason = prove_merge(ordered, artifact_root=root / "run_log" / "backtest")
        merges.append(
            Merge(
                target=target,
                survivor=survivor.address,
                retired=tuple(member.address for member in retired),
                proved=proved,
                reason=reason,
            )
        )
        if proved and survivor.address != target:
            mapping[survivor.address] = target

    return RekeyPlan(
        case_study=root.name,
        rows=len(rows),
        reachable=sum(1 for row in rows if not row.moved),
        mapping=mapping,
        merges=tuple(merges),
        unexplained=unexplained,
        namespace_clashes=tuple(sorted(_touched_addresses(mapping, merges) & foreign)),
    )
