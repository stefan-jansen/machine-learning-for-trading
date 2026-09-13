"""A failed sequence fit must leave its checkpoints on disk.

`_fit_sequence_population` staged checkpoints in `.models.<uuid>.tmp` and promoted the
tree with `os.replace` once every fold had fitted. The exception handler `rmtree`d the
staging tree, so a run that died with a Python-level exception destroyed every checkpoint
it had written, and the next attempt refit all sixteen folds.

The behaviour it produced was backwards: a `SIGKILL` skips the handler and orphans the
tree, so an uncaught death preserved more than a caught one. Measured on the 2026-09-13
machine crash, the orphaned tree held 139 checkpoints over 7 of 16 folds.

Two properties are pinned here, because dropping the handler is only safe if the second
one holds:

- a failure leaves the staging tree, with the checkpoints written before it
- a staging tree is invisible to the warm path, which gates on `models/` and cannot
  mistake a partial fit for a complete one
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEEP_LEARNING = REPO / "case_studies" / "utils" / "deep_learning.py"


def _stage_and_fail(train_dir: Path, folds_written: int) -> Path:
    """The staging/promote block's shape: write checkpoints, then raise before the promote."""
    staging = train_dir / ".models.deadbeef.tmp"
    model_dir = train_dir / "models"
    try:
        for fold in range(folds_written):
            fold_dir = staging / "lstm_h64" / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True)
            (fold_dir / "epoch_05.pt").write_bytes(b"weights")
        raise RuntimeError("the fit died at the next fold")
        os.replace(staging, model_dir)  # noqa: B012 - unreachable, mirrors the real order
    except RuntimeError:
        # What the handler used to do here: shutil.rmtree(staging, ignore_errors=True)
        pass
    return staging


def test_a_failed_fit_leaves_its_checkpoints(tmp_path: Path) -> None:
    staging = _stage_and_fail(tmp_path, folds_written=7)

    assert staging.exists(), "the staging tree is the only copy of the fitted state"
    assert len(list(staging.rglob("*.pt"))) == 7


def test_the_surviving_tree_cannot_be_mistaken_for_a_complete_fit(tmp_path: Path) -> None:
    """The warm path's gate is `models/`, which only the promote creates."""
    _stage_and_fail(tmp_path, folds_written=7)

    # `reused_fitted_state = model_dir.exists()` in `_fit_sequence_population`.
    assert not (tmp_path / "models").exists()
    # And the tree is discoverable for a resume without being reachable by that gate.
    assert [p.name for p in tmp_path.glob(".models.*.tmp")] == [".models.deadbeef.tmp"]


def test_the_promote_refuses_to_clobber_a_complete_fit(tmp_path: Path) -> None:
    """Why keeping staging trees cannot corrupt a finished run.

    Two attempts at one training hash each hold their own tree, and the second's
    `os.replace` raises rather than replacing the first's promoted `models/`.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "epoch_05.pt").write_bytes(b"the complete fit")

    staging = _stage_and_fail(tmp_path, folds_written=2)
    with pytest.raises(OSError, match="not empty"):
        os.replace(staging, model_dir)

    assert (model_dir / "epoch_05.pt").read_bytes() == b"the complete fit"


def _staging_handler() -> ast.ExceptHandler:
    """The `except` guarding the staged fit in `_fit_sequence_population`."""
    tree = ast.parse(DEEP_LEARNING.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        calls = {
            n.func.attr
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        if {"run_dl_cv", "replace"} <= calls or (
            "replace" in calls and "run_dl_cv" in _names(node)
        ):
            assert len(node.handlers) == 1
            return node.handlers[0]
    raise AssertionError("the staged-fit try/except is no longer recognisable")


def _names(node: ast.AST) -> set[str]:
    return {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_the_handler_deletes_nothing() -> None:
    """The regression guard the tests above cannot be.

    They model the block; this reads it. If the `rmtree` comes back, or any other deletion
    is added to that handler, an interrupted run goes back to costing its whole fit.
    """
    handler = _staging_handler()
    deleting = {"rmtree", "unlink", "rmdir", "remove"}
    called = {
        n.func.attr
        for n in ast.walk(handler)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    } | _names(handler)
    assert not (called & deleting), f"the staging tree is deleted again by {called & deleting}"
    # It must still re-raise: swallowing the failure would register a run that never fitted.
    assert any(isinstance(n, ast.Raise) for n in ast.walk(handler))
