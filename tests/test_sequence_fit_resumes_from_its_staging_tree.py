"""An interrupted sequence fit refits the folds it is missing, and only those.

Keeping the staging tree (`tests/test_killed_sequence_run_keeps_its_checkpoints.py`)
made an interrupted fit's checkpoints survive, but nothing read them: the
reconstruction path validates the full declared fold list and refuses a tree holding
fourteen of sixteen, and the fit path asked `run_dl_cv` for every fold. So a run
killed in fold 15 of 16 still paid for all sixteen.

Three properties are pinned here, because the resume is only safe if all three hold:

- a fold counts as already fitted only when every declared checkpoint for it passes
  exactly the check the promote applies, so a resume can never adopt a fold the
  strict validator would reject
- a staging tree is adopted only when no live run holds its claim, because two runs
  of one training hash can be in flight at once and interleaving them promotes a
  tree that is corrupt and validates
- a resumed run publishes the full declared population, reconstructed from the
  promoted tree, not the subset of folds it happened to fit
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from case_studies.utils.deep_model_state import (  # noqa: E402
    checkpoint_sidecar,
    complete_deep_checkpoint_folds,
    deep_checkpoint_path,
    validate_deep_checkpoint_population,
    write_deep_checkpoint,
)

REPO = Path(__file__).resolve().parents[1]
DEEP_LEARNING = REPO / "case_studies" / "utils" / "deep_learning.py"

CONFIG = "lstm_h64"
ARCHITECTURE = "lstm"
CHECKPOINTS = (5, 10)


def _write_fold(root: Path, fold: int, checkpoints=CHECKPOINTS) -> None:
    for checkpoint in checkpoints:
        write_deep_checkpoint(
            deep_checkpoint_path(root, CONFIG, fold, checkpoint),
            model=nn.Linear(2, 1),
            architecture=ARCHITECTURE,
            model_kwargs={"in_features": 2},
            preprocessing={},
            metadata={
                "config_name": CONFIG,
                "fold": fold,
                "checkpoint_kind": "epoch",
                "checkpoint_value": checkpoint,
            },
        )


def _complete(root: Path, fold_ids=(0, 1, 2, 3)) -> tuple[int, ...]:
    return complete_deep_checkpoint_folds(
        root,
        config_name=CONFIG,
        fold_ids=fold_ids,
        checkpoints=CHECKPOINTS,
        architecture=ARCHITECTURE,
    )


def test_a_half_written_fold_is_not_complete(tmp_path: Path) -> None:
    """The fold a run died inside holds some of its checkpoints, and must be refit."""
    for fold in (0, 1):
        _write_fold(tmp_path, fold)
    _write_fold(tmp_path, 2, checkpoints=(5,))

    assert _complete(tmp_path) == (0, 1)


def test_a_corrupt_checkpoint_disqualifies_its_fold(tmp_path: Path) -> None:
    """A digest that does not match its sidecar is not fitted state, so the fold refits."""
    for fold in (0, 1):
        _write_fold(tmp_path, fold)
    victim = deep_checkpoint_path(tmp_path, CONFIG, 1, 10)
    victim.write_bytes(victim.read_bytes() + b"trailing")

    assert _complete(tmp_path) == (0,)


def test_a_foreign_architecture_disqualifies_its_fold(tmp_path: Path) -> None:
    """Checkpoints from another architecture are present, loadable and useless here."""
    _write_fold(tmp_path, 0)
    write_deep_checkpoint(
        deep_checkpoint_path(tmp_path, CONFIG, 1, 5),
        model=nn.Linear(2, 1),
        architecture="gru",
        model_kwargs={"in_features": 2},
        preprocessing={},
        metadata={
            "config_name": CONFIG,
            "fold": 1,
            "checkpoint_kind": "epoch",
            "checkpoint_value": 5,
        },
    )

    assert _complete(tmp_path) == (0,)


def test_refitting_only_the_named_gaps_promotes(tmp_path: Path) -> None:
    """The whole resume, end to end on the tree: report, refit the gaps, promote.

    The report and the strict validator must not disagree. A fold the report skips is
    a fold the resume refits, and once it has, the tree the promote sees must be the
    one an uninterrupted run would have produced - no gaps and no leftovers.
    """
    declared = (0, 1, 2, 3, 4)
    for fold in (0, 1, 2, 3):
        _write_fold(tmp_path, fold)
    _write_fold(tmp_path, 4, checkpoints=(5,))

    already = _complete(tmp_path, fold_ids=declared)
    assert already == (0, 1, 2, 3)

    with pytest.raises(ValueError, match="incomplete"):
        validate_deep_checkpoint_population(
            tmp_path,
            config_name=CONFIG,
            fold_ids=declared,
            checkpoints=CHECKPOINTS,
            architecture=ARCHITECTURE,
        )

    # What the resumed fit does: clear the gaps, then fit them into the same tree.
    # The clear is not tidiness. `write_deep_checkpoint` is immutable and raises on a
    # content conflict, and a refit's weights differ from the interrupted attempt's,
    # so leaving fold 4's `epoch_0005.pt` in place fails the refit rather than the
    # promote - after the resume has already paid for it.
    from case_studies.utils.deep_learning import _clear_partial_folds

    gaps = [fold for fold in declared if fold not in already]
    assert _clear_partial_folds(tmp_path, CONFIG, gaps) == 2  # the .pt and its sidecar
    for fold in gaps:
        _write_fold(tmp_path, fold)

    assert _complete(tmp_path, fold_ids=declared) == declared
    validate_deep_checkpoint_population(
        tmp_path,
        config_name=CONFIG,
        fold_ids=declared,
        checkpoints=CHECKPOINTS,
        architecture=ARCHITECTURE,
    )


def test_a_declared_checkpoint_that_was_never_written_disqualifies_its_fold(
    tmp_path: Path,
) -> None:
    """A fold complete under one schedule is not complete under a longer one."""
    _write_fold(tmp_path, 0)

    assert (
        complete_deep_checkpoint_folds(
            tmp_path,
            config_name=CONFIG,
            fold_ids=(0,),
            checkpoints=(5, 10, 15),
            architecture=ARCHITECTURE,
        )
        == ()
    )


def test_a_missing_sidecar_disqualifies_its_fold(tmp_path: Path) -> None:
    """The checkpoint file alone is not fitted state; `load_deep_checkpoint` needs both."""
    _write_fold(tmp_path, 0)
    checkpoint_sidecar(deep_checkpoint_path(tmp_path, CONFIG, 0, 10)).unlink()

    assert _complete(tmp_path, fold_ids=(0,)) == ()


def test_an_empty_tree_names_no_folds(tmp_path: Path) -> None:
    """A fresh attempt reports nothing to reuse rather than raising."""
    assert _complete(tmp_path) == ()


class TestStagingClaim:
    """A tree is adopted only when the run that made it is gone."""

    @staticmethod
    def _claim(train_dir: Path):
        from case_studies.utils.deep_learning import _claim_staging_tree

        return _claim_staging_tree(train_dir)

    def test_a_fresh_train_dir_gets_a_new_tree(self, tmp_path: Path) -> None:
        staging, handle = self._claim(tmp_path)
        try:
            assert staging.is_dir()
            assert staging.name.startswith(".models.") and staging.name.endswith(".tmp")
        finally:
            handle.close()

    def test_an_unheld_tree_is_adopted(self, tmp_path: Path) -> None:
        """This is the crashed run's tree: the process is gone, so the lock is free."""
        dead = tmp_path / ".models.deadbeef.tmp"
        dead.mkdir()
        (tmp_path / f"{dead.name}.lock").touch()

        staging, handle = self._claim(tmp_path)
        try:
            assert staging == dead
        finally:
            handle.close()

    def test_a_held_tree_is_left_alone(self, tmp_path: Path) -> None:
        """A second run of the same training hash must not write into the first one's tree."""
        first, first_handle = self._claim(tmp_path)
        try:
            second, second_handle = self._claim(tmp_path)
            try:
                assert second != first
                assert second.is_dir()
            finally:
                second_handle.close()
        finally:
            first_handle.close()

    def test_a_released_tree_becomes_adoptable(self, tmp_path: Path) -> None:
        """Closing the claim is what a dead run's kernel does for it."""
        first, first_handle = self._claim(tmp_path)
        first_handle.close()

        second, second_handle = self._claim(tmp_path)
        try:
            assert second == first
        finally:
            second_handle.close()

    def test_a_tree_with_no_lock_file_is_still_adoptable(self, tmp_path: Path) -> None:
        """Trees orphaned before this change carry no lock file, and are the common case."""
        orphan = tmp_path / ".models.0123456789abcdef.tmp"
        orphan.mkdir()

        staging, handle = self._claim(tmp_path)
        try:
            assert staging == orphan
        finally:
            handle.close()


def _fit_branch() -> ast.AST:
    """The `else` branch of `run_resolved_request`'s warm/cold split."""
    tree = ast.parse(DEEP_LEARNING.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_resolved_request"
    )
    branch = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "reused_fitted_state"
    )
    return ast.Module(body=branch.orelse, type_ignores=[])


def test_the_fit_asks_for_the_missing_folds_only() -> None:
    """`selected_folds` is what makes a resume cost the missing folds rather than all."""
    call = next(
        node
        for node in ast.walk(_fit_branch())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_dl_cv"
    )
    keywords = {kw.arg for kw in call.keywords}
    assert "already_fitted_folds" in keywords, (
        "the staged fit must say which folds are already on disk, "
        "or a resumed run refits every one of them"
    )
    assert "checkpoint_root" in keywords


def test_a_resumed_run_reconstructs_the_full_population() -> None:
    """`run_dl_cv`'s result covers the folds it fitted; publishing that would be narrower.

    A resumed run must republish the whole declared population, which is what
    `_reconstruct_sequence_predictions` returns from the promoted tree.
    """
    names = {
        node.func.id
        for node in ast.walk(_fit_branch())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_reconstruct_sequence_predictions" in names, (
        "a resumed run must rebuild predictions from the promoted tree, "
        "not publish the subset of folds it fitted"
    )


def test_a_partial_fold_is_cleared_before_it_is_refit() -> None:
    """Without the clear, the refit hits the immutable-checkpoint conflict and dies."""
    names = {
        node.func.id
        for node in ast.walk(_fit_branch())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_clear_partial_folds" in names


def test_the_clear_never_touches_a_complete_fold(tmp_path: Path) -> None:
    """It takes the fold list it is given, and the caller gives it the gaps only."""
    for fold in (0, 1):
        _write_fold(tmp_path, fold)
    _write_fold(tmp_path, 2, checkpoints=(5,))
    from case_studies.utils.deep_learning import _clear_partial_folds

    assert _clear_partial_folds(tmp_path, CONFIG, [2]) == 2  # the .pt and its sidecar
    assert _complete(tmp_path) == (0, 1)
    assert not (tmp_path / CONFIG / "fold_02").exists()


def test_the_handler_still_deletes_nothing() -> None:
    """The resume is worth nothing if the failure path goes back to deleting the tree."""
    handler = next(
        node for node in ast.walk(_fit_branch()) if isinstance(node, ast.Try) and node.handlers
    ).handlers[0]
    destructive = {"rmtree", "unlink", "rmdir", "remove"}
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
    }
    assert not (called & destructive), f"the failure handler deletes: {called & destructive}"


def test_the_lock_file_outlives_a_failed_promote() -> None:
    """Removing the lock while the tree is still adoptable breaks the claim.

    A later run would create a fresh file at the same path, flock that instead of the
    one the live run holds, and adopt a tree another run is writing into.
    """
    branch = _fit_branch()
    try_node = next(node for node in ast.walk(branch) if isinstance(node, ast.Try))
    unlinks_in_finally = [
        node
        for node in ast.walk(ast.Module(body=try_node.finalbody, type_ignores=[]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unlink"
    ]
    assert not unlinks_in_finally, (
        "the lock file must be removed only after the promote succeeded, "
        "not in the finally that also runs on failure"
    )


def _function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_the_fold_filter_runs_before_the_darts_dispatch() -> None:
    """Order decides whether a Darts resume works at all.

    `run_dl_cv` dispatches to `run_darts_cv` and returns, so a fold filter below that
    dispatch never runs for a Darts config. The adopted folds would be refit, and the
    first checkpoint written for one raises the immutable-checkpoint conflict.
    """
    body = _function(DEEP_LEARNING, "run_dl_cv")
    dispatch = min(
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_darts_cv"
    )
    filtered = min(
        node.lineno
        for node in ast.walk(body)
        if isinstance(node, ast.Name) and node.id == "already_fitted_folds"
    )
    assert filtered < dispatch, (
        "the already-fitted folds must be removed from `splits` before either backend "
        f"is dispatched (filter at line {filtered}, run_darts_cv at {dispatch})"
    )


def _validation_fold_ids(path: Path, function: str, validator: str) -> str:
    """The `fold_ids=` expression the tree validator is called with."""
    call = next(
        node
        for node in ast.walk(_function(path, function))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == validator
    )
    return next(ast.unparse(kw.value) for kw in call.keywords if kw.arg == "fold_ids")


def test_the_native_path_expects_the_folds_it_adopted() -> None:
    """One set, not two, on the native path - and the reason is the shards.

    A native resume reads its checkpoint metrics from `save_dir/_incremental`, and the
    adopted folds' shards are still there: `clear_fold_predictions` drops one
    `(config, fold)` glob at that fold's start, so only a refit fold loses its own. So
    the adopted folds appear in both populations - the per-checkpoint coverage
    comparison, which would otherwise skip every checkpoint and leave the aggregation
    with nothing to assemble, and the staging-tree validation, which would otherwise
    call them undeclared artifacts.
    """
    body = _function(DEEP_LEARNING, "run_dl_cv")
    seed = next(
        node.value
        for node in ast.walk(body)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "expected_fold_ids"
    )
    assert "adopted" in ast.unparse(seed), (
        f"expected_fold_ids must start from the adopted folds, got {ast.unparse(seed)}"
    )

    fold_ids = _validation_fold_ids(
        DEEP_LEARNING, "run_dl_cv", "validate_deep_checkpoint_population"
    )
    assert "adopted" in fold_ids


def test_the_darts_path_keeps_the_two_populations_apart() -> None:
    """The Darts path aggregates from in-memory slices, so the sets genuinely differ.

    `cfg_slices` holds only what this run predicted, so expecting the adopted folds in
    the coverage comparison skips every checkpoint and `assemble_cv_result` raises. The
    tree validation still has to expect them, because the tree holds them.
    """
    darts = REPO / "case_studies" / "utils" / "darts_forecasting.py"
    signature = _function(darts, "run_darts_cv").args
    names = {arg.arg for arg in signature.args + signature.kwonlyargs}
    assert "already_fitted_folds" in names

    fold_ids = _validation_fold_ids(darts, "run_darts_cv", "validate_darts_checkpoint_population")
    assert "already_fitted_folds" in fold_ids, (
        f"the Darts tree validation must expect the adopted folds, got fold_ids={fold_ids}"
    )

    expected = next(
        node
        for node in ast.walk(_function(darts, "run_darts_cv"))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "expected_fold_ids"
            for target in node.targets
        )
    )
    assert "already_fitted_folds" not in ast.unparse(expected.value), (
        "expected_fold_ids drives the per-checkpoint coverage comparison, and the Darts "
        "path predicts nothing for an adopted fold"
    )


class TestPrunedShardsForceARefit:
    """A fold whose prediction shards have gone cannot be adopted on the native path."""

    @staticmethod
    def _shards(tmp_path: Path, folds, epochs=(5, 10)) -> Path:
        from case_studies.utils.registry.store import incremental_shard_path

        incr = tmp_path / "_incremental"
        incr.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            for epoch in epochs:
                # Named by the producer, so the test cannot drift from the file layout.
                incremental_shard_path(incr, CONFIG, fold, epoch).write_bytes(b"")
        return tmp_path

    def test_a_fold_with_shards_stays_adopted(self, tmp_path: Path) -> None:
        from case_studies.utils.deep_learning import _folds_with_prediction_shards

        self._shards(tmp_path, (0, 1, 2))
        kept = _folds_with_prediction_shards(tmp_path, [{"config_name": CONFIG}], {0, 1, 2})
        assert kept == {0, 1, 2}

    def test_a_fold_whose_shards_were_pruned_is_refit(self, tmp_path: Path) -> None:
        """Adopting it anyway leaves every checkpoint short and nothing to assemble."""
        from case_studies.utils.deep_learning import _folds_with_prediction_shards

        self._shards(tmp_path, (0, 2))
        kept = _folds_with_prediction_shards(tmp_path, [{"config_name": CONFIG}], {0, 1, 2})
        assert kept == {0, 2}

    def test_a_missing_incremental_directory_adopts_nothing(self, tmp_path: Path) -> None:
        from case_studies.utils.deep_learning import _folds_with_prediction_shards

        assert _folds_with_prediction_shards(tmp_path, [{"config_name": CONFIG}], {0, 1}) == set()
        assert _folds_with_prediction_shards(None, [{"config_name": CONFIG}], {0, 1}) == set()


def test_the_native_resume_checks_its_shards_before_adopting() -> None:
    """Without this the resume fails at aggregation, after refitting the other folds."""
    names = {
        node.func.id
        for node in ast.walk(_function(DEEP_LEARNING, "run_dl_cv"))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_folds_with_prediction_shards" in names
