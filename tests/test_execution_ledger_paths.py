"""The execution ledger addresses artifacts as the study does, not as the filesystem does.

A study whose ``run_log`` is a symlink into a shared artifact store resolves its
artifacts out of the study root; a study whose ``run_log`` is a real directory does not.
So a containment check that resolved symlinks passed in one layout and failed in the
other, on the same code and the same execution tier. These tests build the symlinked
layout, because a fix exercised only against a real directory cannot fail.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from case_studies.research.recovery import ExecutionLedger, _relative


def _case_study_layout(tmp_path: Path) -> tuple[Path, Path]:
    """A study root whose run_log points at a shared store outside it."""
    root = tmp_path / "worktree" / "case_studies" / "fx_pairs"
    store = tmp_path / "code" / "case_studies" / "fx_pairs" / "run_log"
    (store / "training" / "abc123").mkdir(parents=True)
    root.mkdir(parents=True)
    (root / "run_log").symlink_to(store, target_is_directory=True)
    artifact = root / "run_log" / "training" / "abc123" / "fold_0.bin"
    artifact.write_bytes(b"fitted-state")
    return root, artifact


def test_symlinked_run_log_artifact_is_inside_the_study(tmp_path: Path) -> None:
    root, artifact = _case_study_layout(tmp_path)

    # The precondition that made this fail: the artifact resolves out of the study root.
    assert artifact.is_file()
    assert not str(artifact.resolve()).startswith(str(root.resolve()))

    assert _relative(root, artifact) == os.path.join("run_log", "training", "abc123", "fold_0.bin")


def test_label_is_stable_when_the_shared_store_moves(tmp_path: Path) -> None:
    """A relabelled store must not invalidate a completed fold.

    Resolving made the recorded label depend on where the symlink pointed when it was
    written, so re-pointing the store stopped a completed fold from matching and re-ran
    work that was already done.
    """
    root, artifact = _case_study_layout(tmp_path)
    before = _relative(root, artifact)

    moved = tmp_path / "relocated-code" / "case_studies" / "fx_pairs" / "run_log"
    moved.parent.mkdir(parents=True)
    (tmp_path / "code" / "case_studies" / "fx_pairs" / "run_log").rename(moved)
    (root / "run_log").unlink()
    (root / "run_log").symlink_to(moved, target_is_directory=True)

    assert artifact.is_file()
    assert _relative(root, artifact) == before


def test_artifact_outside_the_study_is_still_rejected(tmp_path: Path) -> None:
    root, _ = _case_study_layout(tmp_path)

    with pytest.raises(ValueError, match="outside the study root"):
        _relative(root, tmp_path / "elsewhere" / "fold_0.bin")


def test_parent_traversal_is_rejected_rather_than_recorded(tmp_path: Path) -> None:
    """Dropping resolve() must not drop the containment guard with it.

    ``relative_to`` alone accepts this lexically and returns a label containing ``..``.
    """
    root, _ = _case_study_layout(tmp_path)
    climbing = root / "run_log" / ".." / ".." / ".." / "etc" / "passwd"

    with pytest.raises(ValueError, match="outside the study root"):
        _relative(root, climbing)


def test_complete_and_reuse_a_fold_under_a_symlinked_run_log(tmp_path: Path) -> None:
    """The end-to-end path that actually failed: record a fold, then reuse it.

    The unit tests above pin the labelling rule; this one drives ExecutionLedger the
    way a canonical 06/07/08 run does, so the regression is caught at the surface where
    it was reported rather than only at the helper it came from.
    """
    root, fitted_state = _case_study_layout(tmp_path)
    shard = fitted_state.parent / "shard_0.parquet"
    shard.write_bytes(b"prediction-shard")

    # candidate_fold_completions references training_runs, so seed the parent row.
    from case_studies.utils.registry.store import _open_registry, _utc_now

    training_hash = "t" * 8
    db = _open_registry(root)
    try:
        db.execute(
            "INSERT INTO training_runs (training_hash, family, label, created_at) VALUES (?,?,?,?)",
            (training_hash, "linear", "fwd_ret_1d", _utc_now()),
        )
        db.commit()
    finally:
        db.close()

    ledger = ExecutionLedger(study=SimpleNamespace(), root=root)
    fold = {
        "training_hash": training_hash,
        "candidate_identity": "ridge",
        "fold_id": 0,
    }
    settings = {"alpha": 1.0}

    ledger.complete_fold(
        **fold,
        fitted_state=fitted_state,
        prediction_shard=shard,
        resolved_settings=settings,
    )

    assert ledger.fold_completion_exists(**fold)
    assert ledger.reusable_fold(
        **fold,
        fitted_state=fitted_state,
        prediction_shard=shard,
        resolved_settings=settings,
    )

    # The recorded label addresses the artifact from the study root, so it opens.
    with sqlite3.connect(root / "run_log" / "registry.db") as db:
        stored = db.execute(
            "SELECT fitted_state_path, prediction_shard_path FROM candidate_fold_completions"
        ).fetchone()
    assert (root / stored[0]).read_bytes() == b"fitted-state"
    assert (root / stored[1]).read_bytes() == b"prediction-shard"
