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
            if result.execution_tier == "preview":
                raise ValueError(
                    f"preview result {member_hash} cannot enter an official population"
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
