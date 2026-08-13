from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from case_studies.utils.registry.specs import canonical_json, compute_hash
from case_studies.utils.registry.store import _git_hash, _open_registry, _utc_now

from .results import Result

if TYPE_CHECKING:
    from .workspace import Study


@dataclass(frozen=True)
class CandidateSet:
    study: Study
    hash: str
    name: str
    member_kind: str
    members: tuple[str, ...]
    comparison_contract: dict[str, Any]

    @classmethod
    def create(
        cls,
        study: Study,
        name: str,
        members: Iterable[Result],
        *,
        comparison_contract: dict[str, Any] | None = None,
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
        if any(not member.complete for member in resolved):
            raise ValueError("partial results cannot enter a candidate set")
        protocols = [member.protocol() for member in resolved]
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
        supplied_protocol = contract.get("protocol")
        if supplied_protocol is not None and supplied_protocol != base:
            raise ValueError("comparison contract protocol does not match its members")
        contract["protocol"] = base
        contract.setdefault("comparable_fields", sorted(comparable_fields))
        member_hashes = tuple(sorted(member.hash for member in resolved))
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
            if existing is None:
                db.execute(
                    "INSERT INTO candidate_sets "
                    "(set_hash, name, member_kind, comparison_contract_json, created_at, git_commit) "
                    "VALUES (?,?,?,?,?,?)",
                    (set_hash, name, member_kind, expected[1], _utc_now(), _git_hash()),
                )
                db.executemany(
                    "INSERT INTO candidate_set_members (set_hash, member_hash, ordinal) "
                    "VALUES (?,?,?)",
                    [(set_hash, value, ordinal) for ordinal, value in enumerate(member_hashes)],
                )
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return cls(study, set_hash, name, member_kind, member_hashes, contract)

    @classmethod
    def open(cls, study: Study, set_hash: str) -> CandidateSet:
        db_path = study.root / "run_log" / "registry.db"
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT name, member_kind, comparison_contract_json FROM candidate_sets "
                "WHERE set_hash = ?",
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
        return cls(study, set_hash, row[0], row[1], members, json.loads(row[2]))

    def extend(self, name: str, members: Iterable[Result]) -> CandidateSet:
        existing = [Result.open(self.study, member_hash) for member_hash in self.members]
        return self.create(
            self.study,
            name,
            [*existing, *members],
            comparison_contract=self.comparison_contract,
        )

    def best_validation_sharpe(self) -> Result:
        """Select deterministically within this immutable backtest set."""
        if self.member_kind != "backtest":
            raise ValueError("validation Sharpe selection requires backtest members")
        placeholders = ",".join("?" for _ in self.members)
        with sqlite3.connect(self.study.root / "run_log" / "registry.db") as db:
            rows = db.execute(
                f"""
                SELECT m.backtest_hash, m.sharpe
                FROM backtest_metrics m
                JOIN backtest_runs b ON b.backtest_hash = m.backtest_hash
                JOIN prediction_sets p ON p.prediction_hash = b.prediction_hash
                JOIN prediction_coverage c ON c.prediction_hash = p.prediction_hash
                JOIN training_runs t ON t.training_hash = p.training_hash
                WHERE m.backtest_hash IN ({placeholders})
                  AND p.split = 'validation'
                  AND c.status = 'complete'
                  AND t.execution_tier = 'canonical'
                  AND b.stage IN ('signal', 'allocation', 'risk_overlay')
                  AND m.sharpe IS NOT NULL
                ORDER BY m.sharpe DESC, m.backtest_hash ASC
                """,
                self.members,
            ).fetchall()
        if len(rows) != len(self.members):
            raise ValueError("candidate set contains an ineligible selection member")
        return Result.open(self.study, rows[0][0])
