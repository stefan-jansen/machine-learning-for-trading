from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from case_studies.utils.registry.specs import canonical_json, compute_hash
from case_studies.utils.registry.store import _open_registry, _utc_now

from .results import Result

if TYPE_CHECKING:
    from .workspace import Study


def _refuse_preview_activation() -> None:
    """A population is written to the canonical registry whatever tier is active.

    That is correct - a population is canonical by definition - but it means a preview run that
    reaches this code writes into the shared registry rather than its own workspace. The member
    check below cannot catch it: a population is snapshotted *before* its members are fitted, so
    every lookup misses and every member passes. On 2026-08-16 a preview notebook test left a
    28-member population in the canonical etfs registry that way, and the next canonical run was
    refused because a population of that name already existed with different members.

    Callers guard this too. This is the guard that does not depend on remembering.
    """
    import os
    from pathlib import Path

    active = os.environ.get("ML4T_OUTPUT_DIR")
    if active and Path(active).name == ".preview":
        raise ValueError(
            "a preview run cannot create an official population: it would be written to the "
            "canonical registry"
        )


@dataclass(frozen=True)
class OfficialPopulation:
    study: Study
    hash: str
    name: str
    member_kind: str
    members: tuple[str, ...]
    supersedes: str | None

    @classmethod
    def create(
        cls,
        study: Study,
        *,
        name: str,
        member_kind: str,
        members: list[str] | tuple[str, ...],
        supersedes: str | None = None,
    ) -> OfficialPopulation:
        study.require_writable()
        _refuse_preview_activation()
        if member_kind not in {"training", "prediction", "backtest"}:
            raise ValueError("official population member_kind is not supported")
        normalized = tuple(dict.fromkeys(str(member) for member in members))
        if not normalized or len(normalized) != len(members):
            raise ValueError("official population members must be non-empty and unique")
        for member_hash in normalized:
            try:
                result = Result.open(study, member_hash, include_preview=True)
            except KeyError:
                continue
            if result.kind != member_kind:
                raise ValueError(
                    f"official population member {member_hash} has kind {result.kind}, "
                    f"not {member_kind}"
                )
            if result.execution_tier == "preview":
                raise ValueError(
                    f"preview result {member_hash} cannot enter an official population"
                )
            if (
                member_kind == "backtest"
                and (result.spec().get("decision_artifact") or {}).get("canonical") is False
            ):
                raise ValueError(
                    f"exploratory decision backtest {member_hash} cannot enter an official population"
                )
        snapshot = {
            "schema_version": 1,
            "name": name,
            "member_kind": member_kind,
            "members": list(normalized),
            "supersedes": supersedes,
        }
        population_hash = compute_hash(canonical_json(snapshot))
        db = _open_registry(study.root)
        try:
            latest = db.execute(
                "SELECT population_hash FROM official_populations WHERE name = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
            if latest is not None:
                if latest[0] == population_hash:
                    return cls.open(study, population_hash)
                if supersedes != latest[0]:
                    raise ValueError(
                        f"a changed population named {name!r} must explicitly supersedes "
                        f"{latest[0]}"
                    )
            elif supersedes is not None:
                raise ValueError("first population version cannot supersede another snapshot")
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO official_populations "
                "(population_hash, name, member_kind, snapshot_json, supersedes_hash, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    population_hash,
                    name,
                    member_kind,
                    canonical_json(snapshot),
                    supersedes,
                    _utc_now(),
                ),
            )
            db.executemany(
                "INSERT INTO official_population_members "
                "(population_hash, member_hash, ordinal) VALUES (?,?,?)",
                [
                    (population_hash, member_hash, ordinal)
                    for ordinal, member_hash in enumerate(normalized)
                ],
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return cls(study, population_hash, name, member_kind, normalized, supersedes)

    @classmethod
    def one(cls, study: Study, *, name: str) -> OfficialPopulation:
        """Resolve the current immutable population by name, without a hash handoff.

        A name accumulates a snapshot per generation: refitting the same configurations under a
        corrected estimator parameter produces different prediction identities, so the second run
        supersedes the first rather than replacing it. Every earlier snapshot stays readable by
        hash, which is what makes the lineage worth writing down, but a caller asking by name
        wants the generation in force. That is the one snapshot in the chain that nothing
        supersedes; two of those under a single name means the chain forked and no answer is
        defensible.
        """
        with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
            rows = db.execute(
                "SELECT population_hash, supersedes_hash FROM official_populations "
                "WHERE name = ? ORDER BY population_hash",
                (name,),
            ).fetchall()
        superseded = {row[1] for row in rows if row[1] is not None}
        current = [row[0] for row in rows if row[0] not in superseded]
        if len(current) != 1:
            raise ValueError(
                f"official population name {name!r} resolved to {len(current)} current "
                f"identities among {len(rows)} snapshots"
            )
        return cls.open(study, current[0])

    @classmethod
    def open(cls, study: Study, population_hash: str) -> OfficialPopulation:
        with sqlite3.connect(study.root / "run_log" / "registry.db") as db:
            row = db.execute(
                "SELECT name, member_kind, snapshot_json, supersedes_hash "
                "FROM official_populations WHERE population_hash = ?",
                (population_hash,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown official population {population_hash!r}")
        snapshot = json.loads(row[2])
        return cls(
            study,
            population_hash,
            row[0],
            row[1],
            tuple(snapshot["members"]),
            row[3],
        )

    def require_complete(self) -> tuple[str, ...]:
        failures = []
        for member_hash in self.members:
            try:
                result = Result.open(self.study, member_hash)
            except KeyError:
                failures.append(f"{member_hash}:missing")
                continue
            if result.kind != self.member_kind:
                failures.append(f"{member_hash}:kind={result.kind}")
            elif not result.complete:
                failures.append(f"{member_hash}:partial")
        if failures:
            raise ValueError(
                f"official population {self.hash} is incomplete: {', '.join(failures)}"
            )
        return self.members
