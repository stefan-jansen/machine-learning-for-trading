from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from case_studies.utils.registry.specs import canonical_json
from case_studies.utils.registry.store import _open_registry, _utc_now

if TYPE_CHECKING:
    from .workspace import Study


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    """Return the artifact's path within the study's own address space.

    Deliberately does not resolve symlinks. A study whose ``run_log`` is a symlink into
    a shared artifact store resolves its artifacts out of the study root, so resolving
    rejects a file that is inside the study as the study addresses it. A study whose
    ``run_log`` is a real directory - one driven through ``ML4T_OUTPUT_DIR`` - never
    leaves the root and never saw this, which is the whole of why it survived earlier
    runs. The trigger is the layout, not the execution tier.

    Resolving was also wrong for the value's purpose. The result is a study-relative
    label: ``reusable_fold`` compares it against the stored row, and where it is joined
    back to open the artifact - ``scripts/prove_cme_futures_interface.py`` does exactly
    this - it is joined onto the study root, so it has to address the artifact the way
    the study does. A resolved label does not survive that join, and it silently
    depends on where the symlink pointed when it was written, so re-pointing the store
    would stop a completed fold from matching and re-run work that was already done.

    ``..`` is still normalized away, without following symlinks, so a path that climbs
    out of the study root is rejected rather than recorded as a label containing ``..``.
    """
    relative = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    if relative == os.pardir or relative.startswith(f"{os.pardir}{os.sep}"):
        raise ValueError(f"completed-fold artifact is outside the study root: {path}")
    return relative


@dataclass(frozen=True)
class ExecutionAttempt:
    study: Study
    root: Path
    id: str
    scientific_identity: str

    def finish(self, status: str, diagnostics: dict[str, Any]) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("execution attempt status must be completed or failed")
        db = _open_registry(self.root)
        try:
            updated = db.execute(
                "UPDATE execution_attempts SET status = ?, diagnostics_json = ?, "
                "completed_at = ? WHERE attempt_id = ? AND status = 'running'",
                (status, canonical_json(diagnostics), _utc_now(), self.id),
            )
            if updated.rowcount != 1:
                raise ValueError(f"execution attempt {self.id} is not running")
            db.commit()
        finally:
            db.close()


class ExecutionLedger:
    def __init__(self, study: Study, root: Path | None = None) -> None:
        self.study = study
        self.root = root or study.root

    def start(self, scientific_identity: str) -> ExecutionAttempt:
        self.study.require_writable()
        attempt_id = uuid.uuid4().hex
        db = _open_registry(self.root)
        try:
            db.execute(
                "INSERT INTO execution_attempts "
                "(attempt_id, scientific_identity, status, diagnostics_json, started_at) "
                "VALUES (?,?,?,?,?)",
                (attempt_id, scientific_identity, "running", "{}", _utc_now()),
            )
            db.commit()
        finally:
            db.close()
        return ExecutionAttempt(self.study, self.root, attempt_id, scientific_identity)

    def fold_completion_exists(
        self,
        *,
        training_hash: str,
        candidate_identity: str,
        fold_id: int,
    ) -> bool:
        db_path = self.root / "run_log" / "registry.db"
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT 1 FROM candidate_fold_completions WHERE training_hash = ? "
                "AND candidate_identity = ? AND fold_id = ?",
                (training_hash, candidate_identity, fold_id),
            ).fetchone()
        return row is not None

    def reusable_fold(
        self,
        *,
        training_hash: str,
        candidate_identity: str,
        fold_id: int,
        fitted_state: Path,
        prediction_shard: Path,
        resolved_settings: dict[str, Any],
    ) -> bool:
        db_path = self.root / "run_log" / "registry.db"
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "SELECT fitted_state_path, fitted_state_digest, prediction_shard_path, "
                "prediction_shard_digest, resolved_settings_json "
                "FROM candidate_fold_completions WHERE training_hash = ? "
                "AND candidate_identity = ? AND fold_id = ?",
                (training_hash, candidate_identity, fold_id),
            ).fetchone()
        if row is None or not fitted_state.is_file() or not prediction_shard.is_file():
            return False
        expected_paths = (
            _relative(self.root, fitted_state),
            _relative(self.root, prediction_shard),
        )
        return (
            row[0] == expected_paths[0]
            and row[2] == expected_paths[1]
            and row[1] == _sha256(fitted_state)
            and row[3] == _sha256(prediction_shard)
            and json.loads(row[4]) == json.loads(canonical_json(resolved_settings))
        )

    def complete_fold(
        self,
        *,
        training_hash: str,
        candidate_identity: str,
        fold_id: int,
        fitted_state: Path,
        prediction_shard: Path,
        resolved_settings: dict[str, Any],
    ) -> None:
        if not fitted_state.is_file() or not prediction_shard.is_file():
            raise ValueError("completed fold requires fitted state and prediction shard artifacts")
        row = (
            training_hash,
            candidate_identity,
            fold_id,
            _relative(self.root, fitted_state),
            _sha256(fitted_state),
            _relative(self.root, prediction_shard),
            _sha256(prediction_shard),
            canonical_json(resolved_settings),
            _utc_now(),
        )
        db = _open_registry(self.root)
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR REPLACE INTO candidate_fold_completions "
                "(training_hash, candidate_identity, fold_id, fitted_state_path, "
                "fitted_state_digest, prediction_shard_path, prediction_shard_digest, "
                "resolved_settings_json, completed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                row,
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
