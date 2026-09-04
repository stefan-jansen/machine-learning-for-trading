from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from case_studies.utils.registry.specs import canonical_json, compute_hash
from case_studies.utils.registry.store import _git_hash, _open_registry, _utc_now

from .results import Result

if TYPE_CHECKING:
    from .workspace import Study


def binding_table(db: sqlite3.Connection) -> str:
    """The table this registry records candidate-set name bindings in.

    ``candidate_set_names`` where the registry has been opened for writing since bindings moved
    off the identity row, and ``candidate_sets`` where it has not - which is every registry a
    reader clones and every one an older version wrote. The fallback resolves one name per
    identity, the only bindings such a registry ever held.
    """
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidate_set_names'"
    ).fetchone()
    return "candidate_set_names" if exists is not None else "candidate_sets"


def name_bindings(db: sqlite3.Connection, name: str) -> list[tuple[str, str | None]]:
    """Every ``(set_hash, supersedes_hash)`` recorded under ``name``.

    A name is a binding onto a candidate set, not part of its identity. ``candidate_set_names``
    holds one row per ``(name, set_hash)``, so one set can carry several names - a union that
    turns out to equal one of its inputs is the case that forces this - and a name can point at
    a set first written under a different one.

    Registries written before that table existed keep the binding in ``candidate_sets.name``,
    one name per identity. Reading it here is what lets :meth:`CandidateSet.one` resolve against
    a registry no writer has opened since, which is every registry a reader clones.
    """
    table = binding_table(db)
    return db.execute(
        f"SELECT set_hash, supersedes_hash FROM {table} WHERE name = ?",  # noqa: S608 - fixed set
        (name,),
    ).fetchall()


def _unsuperseded_hash(db: sqlite3.Connection, name: str) -> str | None:
    """The one generation of ``name`` that no later generation replaces, if it is unique."""
    rows = name_bindings(db, name)
    replaced = {row[1] for row in rows if row[1] is not None}
    heads = [row[0] for row in rows if row[0] not in replaced]
    return heads[0] if len(heads) == 1 else None


def candidate_set_supersedes(study: Study, *, name: str, declared: str | None) -> str | None:
    """Whether a declared candidate-set generation may be offered to :meth:`CandidateSet.create`.

    The same decision :func:`case_studies.research.population.population_supersedes` makes for an
    official population, applied to a candidate set. It exists because the declaration is
    committed source that has to be right in three situations the notebook cannot tell apart:

    - **A clean clone.** ``run_log/`` is gitignored, so a reader starts with an empty registry -
      often with no ``candidate_sets`` table at all, which raises ``OperationalError`` rather
      than ``ValueError``. ``create`` refuses a first generation that claims to supersede
      something, so the declared hash must be withheld and the reader's run publishes generation
      one. **This is the ordinary case for anyone who is not the author**, and offering the hash
      unconditionally is what stops a published notebook running for the people it is published
      for.
    - **The re-run.** The generation in force is the one this declaration produced, so
      ``current.supersedes == declared``, and offering the hash resolves to the set already
      published rather than writing a new one.
    - **The refit.** The declaration names the tip itself, and offering it publishes the next
      generation over that tip.

    Anything else is withheld, and ``create`` then refuses and names the hash it requires, which
    is a better answer than this function guessing.

    ``CandidateSet.create`` already takes and enforces ``supersedes``; what had no shared
    implementation is this decision, so every notebook that declares a lineage either resolved it
    itself or - four in ``crypto_perps_funding`` alone - did not resolve it and would have stopped
    on a reader's first run.
    """
    if not declared:
        return None
    try:
        current = CandidateSet.one(study, name=name)
    except (ValueError, KeyError, sqlite3.OperationalError):
        # No generation under this name, or no table at all: `one` raises `ValueError` when the
        # name resolves to other than exactly one head, `open` raises `KeyError` for a hash the
        # table does not hold, and a clean clone has no `candidate_sets` table for either to read.
        return None
    if declared in (current.supersedes, current.hash):
        return declared
    return None


@dataclass(frozen=True)
class CandidateSet:
    study: Study
    hash: str
    name: str
    member_kind: str
    members: tuple[str, ...]
    comparison_contract: dict[str, Any]
    supersedes: str | None = None

    @classmethod
    def create(
        cls,
        study: Study,
        name: str,
        members: Iterable[Result],
        *,
        comparison_contract: dict[str, Any] | None = None,
        supersedes: str | None = None,
    ) -> CandidateSet:
        study.require_writable()
        study.activate()
        resolved = tuple(members)
        if not resolved:
            raise ValueError("candidate set requires at least one member")
        if any(member.study != study for member in resolved):
            raise ValueError("candidate set member belongs to another study")
        kinds = {member.kind for member in resolved}
        if len(kinds) != 1 or kinds.pop() not in {"prediction", "backtest"}:
            raise ValueError("candidate set members must share prediction or backtest kind")
        member_kind = resolved[0].kind
        if any(member.execution_tier == "preview" for member in resolved):
            raise ValueError("preview results cannot enter a canonical candidate set")
        partial = [
            (member.hash, reason)
            for member in resolved
            if (reason := member.completeness()) is not None
        ]
        if partial:
            detail = "; ".join(f"{member_hash}: {reason}" for member_hash, reason in partial)
            raise ValueError(f"partial results cannot enter a candidate set - {detail}")
        if member_kind == "backtest" and any(
            (member.spec().get("decision_artifact") or {}).get("canonical") is False
            for member in resolved
        ):
            raise ValueError("exploratory decision backtests cannot enter a candidate set")
        ordered = tuple(sorted(resolved, key=lambda member: member.hash))
        protocols = [member.protocol() for member in ordered]
        if any(protocol["split"] != "validation" for protocol in protocols):
            raise ValueError("canonical candidate sets require validation results")

        contract = dict(comparison_contract or {})
        comparable_fields = set(contract.get("comparable_fields") or [])
        base = protocols[0]
        for protocol in protocols[1:]:
            differences = {key for key in base if base.get(key) != protocol.get(key)}
            undeclared = differences - comparable_fields
            if undeclared:
                raise ValueError(
                    f"candidate set contains protocol-incompatible results: {sorted(undeclared)}"
                )
        common_protocol = {
            key: value for key, value in base.items() if key not in comparable_fields
        }
        supplied_protocol = contract.get("protocol")
        if supplied_protocol is not None and supplied_protocol != common_protocol:
            raise ValueError("comparison contract protocol does not match its members")
        contract["protocol"] = common_protocol
        contract["comparable_fields"] = sorted(comparable_fields)
        member_hashes = tuple(member.hash for member in ordered)
        if len(set(member_hashes)) != len(member_hashes):
            raise ValueError("candidate set members must be unique")
        set_hash = compute_hash(
            canonical_json(
                {
                    "member_kind": member_kind,
                    "members": member_hashes,
                    "comparison_contract": contract,
                }
            )
        )
        db = _open_registry(study.root)
        try:
            existing = db.execute(
                "SELECT member_kind, comparison_contract_json FROM candidate_sets WHERE set_hash = ?",
                (set_hash,),
            ).fetchone()
            expected = (member_kind, canonical_json(contract))
            if existing is not None and existing != expected:
                raise ValueError(f"immutable candidate-set conflict for {set_hash}")

            # The identity and the name are written under different conditions, so they are
            # decided apart. A set already stored is not written again; a name not yet bound to
            # it still has to be. Deciding both on the identity alone is what let a set
            # requested under a second name return the stored one and bind nothing, so the
            # caller held an object whose name the registry did not have and the next
            # `one(name=...)` raised with nothing pointing at the cause.
            bound = db.execute(
                "SELECT supersedes_hash FROM candidate_set_names WHERE name = ? AND set_hash = ?",
                (name, set_hash),
            ).fetchone()
            if bound is None:
                # A candidate set is derived, so re-running the stage that freezes it produces
                # a second set under the same name whenever the registry or the admission rule
                # moved. Two live generations make the name unresolvable and every reader of it
                # raises, so a changed set has to say which one it replaces - the same contract
                # OfficialPopulation.create holds, for the same reason.
                head = _unsuperseded_hash(db, name)
                if head is not None and supersedes != head:
                    raise ValueError(
                        f"a changed candidate set named {name!r} must explicitly supersedes {head}"
                    )
                if head is None and supersedes is not None:
                    raise ValueError("first candidate set version cannot supersede another set")
            else:
                supersedes = bound[0]

            if existing is None:
                # `candidate_sets.name` and `.supersedes_hash` record the binding this identity
                # was first written under. `candidate_set_names` is what resolution reads; these
                # two columns stay for registries and readers that predate it.
                db.execute(
                    "INSERT INTO candidate_sets "
                    "(set_hash, name, member_kind, comparison_contract_json, created_at, "
                    "git_commit, supersedes_hash) VALUES (?,?,?,?,?,?,?)",
                    (
                        set_hash,
                        name,
                        member_kind,
                        expected[1],
                        _utc_now(),
                        _git_hash(),
                        supersedes,
                    ),
                )
                db.executemany(
                    "INSERT INTO candidate_set_members (set_hash, member_hash, ordinal) "
                    "VALUES (?,?,?)",
                    [(set_hash, value, ordinal) for ordinal, value in enumerate(member_hashes)],
                )
            if bound is None:
                db.execute(
                    "INSERT INTO candidate_set_names "
                    "(name, set_hash, supersedes_hash, created_at, git_commit) VALUES (?,?,?,?,?)",
                    (name, set_hash, supersedes, _utc_now(), _git_hash()),
                )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return cls(study, set_hash, name, member_kind, member_hashes, contract, supersedes)

    @classmethod
    def one(cls, study: Study, *, name: str) -> CandidateSet:
        """Resolve the generation of a named set that nothing supersedes.

        Earlier generations stay readable by hash, which is what makes recording the lineage
        worth anything: a result registered against a superseded set can still be traced to
        the comparison it was made in.
        """
        with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
            head = _unsuperseded_hash(db, name)
            if head is None:
                count = db.execute(
                    "SELECT count(*) FROM candidate_sets WHERE name = ?", (name,)
                ).fetchone()[0]
                raise ValueError(
                    f"candidate set name {name!r} resolved to {count} unsuperseded identities"
                )
        return cls.open(study, head)

    @classmethod
    def open(cls, study: Study, set_hash: str) -> CandidateSet:
        db_path = study.root / "run_log" / "registry.db"
        with closing(sqlite3.connect(db_path)) as db:
            row = db.execute(
                "SELECT name, member_kind, comparison_contract_json, supersedes_hash "
                "FROM candidate_sets WHERE set_hash = ?",
                (set_hash,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown candidate set {set_hash!r}")
            members = tuple(
                value[0]
                for value in db.execute(
                    "SELECT member_hash FROM candidate_set_members WHERE set_hash = ? "
                    "ORDER BY ordinal",
                    (set_hash,),
                ).fetchall()
            )
        return cls(study, set_hash, row[0], row[1], members, json.loads(row[2]), row[3])

    def extend(
        self, name: str, members: Iterable[Result], *, supersedes: str | None = None
    ) -> CandidateSet:
        """A new set holding this one's members and *members*, under *name*.

        *supersedes* names the generation of *name* this one retires, and is required
        whenever one is already recorded - ``create`` refuses a changed set under a live
        name without it. An extension is exactly where that happens: re-running the stage
        after upstream results moved produces different members under the same name, and
        without a way to pass the retired hash the notebook has no answer to give.
        """
        existing = [Result.open(self.study, member_hash) for member_hash in self.members]
        return self.create(
            self.study,
            name,
            [*existing, *members],
            comparison_contract=self.comparison_contract,
            supersedes=supersedes,
        )

    def best_validation_sharpe(self) -> Result:
        """Select deterministically within this immutable backtest set."""
        hashes = self._ranked_validation_hashes()
        return Result.open(self.study, hashes[0])

    def ranked_validation_sharpe(self, *, limit: int | None = None) -> tuple[Result, ...]:
        """Return complete members ordered by validation Sharpe and identity tie-break."""
        if limit is not None and limit < 1:
            raise ValueError("validation Sharpe ranking limit must be positive")
        hashes = self._ranked_validation_hashes()
        selected = hashes[:limit] if limit is not None else hashes
        return tuple(Result.open(self.study, result_hash) for result_hash in selected)

    def _ranked_validation_hashes(self) -> tuple[str, ...]:
        if self.member_kind != "backtest":
            raise ValueError("validation Sharpe ranking requires backtest members")
        rows = (
            self.study.backtests.table()
            .filter(
                pl.col("backtest_hash").is_in(self.members)
                & (pl.col("split") == "validation")
                & (pl.col("execution_tier") == "canonical")
                & pl.col("stage").is_in(["signal", "allocation", "risk_overlay"])
                & pl.col("sharpe").is_not_null()
            )
            .sort("sharpe", "backtest_hash", descending=[True, False])
        )
        if rows.height != len(self.members):
            raise ValueError("candidate set contains an ineligible selection member")
        if any(not Result.open(self.study, member_hash).complete for member_hash in self.members):
            raise ValueError("candidate set contains an incomplete selection member")
        return tuple(rows.get_column("backtest_hash"))
