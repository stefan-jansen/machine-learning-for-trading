"""A registered training run records when work on it began.

A training run is registered before it is fitted, because the identity has to exist before
anything can be written under it, and `elapsed_s` is filled in afterwards by
`record_training_cost`. Between those two moments the row said nothing at all: `started_at`
was never passed by any caller on the research path, so a run in flight was indistinguishable
from one that had wedged.

That is not hypothetical. `nasdaq100_microstructure`'s `06_linear` ran 7h23m at 100% of one
core, and the question "is it working?" could not be answered from the registry - it held
thirteen rows with `started_at` and `elapsed_s` both NULL. The run had in fact completed 24
of 26 fold-fits and was stuck on a `liblinear` L1 configuration; that was recoverable only by
reading file modification times off the fold artifacts.

`started_at` is a table column and not part of the spec, so recording it moves no training
hash. That is the property this file pins, because an observability change that reprices the
corpus is not an observability change.

Refs ml4t/agent-workspace#990, #1026.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from case_studies.research import Study
from case_studies.utils.registry.specs import training_hash_from_spec
from tests.test_research_registry import _seed_release, _training_spec


def _study(tmp_path: Path) -> Study:
    return Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )


def _row(study: Study, training_hash: str) -> tuple:
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        return db.execute(
            "SELECT started_at, elapsed_s FROM training_runs WHERE training_hash = ?",
            (training_hash,),
        ).fetchone()


def test_a_registered_run_records_a_start_time_without_being_asked(tmp_path: Path) -> None:
    """Defaulted, so a caller that forgets still leaves a legible row."""
    study = _study(tmp_path)
    before = datetime.now(UTC)
    training = study.results.register_training(_training_spec())
    after = datetime.now(UTC)

    started_at, elapsed_s = _row(study, training.hash)
    assert started_at is not None, "the row cannot say whether the fit has begun"
    assert before <= datetime.fromisoformat(started_at) <= after
    # Still NULL: the fit has not finished, which is exactly the window `started_at` makes
    # legible rather than a gap in it.
    assert elapsed_s is None


def test_an_explicit_start_time_is_kept(tmp_path: Path) -> None:
    """A caller that captured the moment the fit began passes it rather than re-taking it."""
    study = _study(tmp_path)
    stamp = "2026-09-05T04:40:19.123456+00:00"
    training = study.results.register_training(_training_spec(), started_at=stamp)
    assert _row(study, training.hash)[0] == stamp


def test_recording_a_start_time_moves_no_training_hash(tmp_path: Path) -> None:
    """The whole constraint on this change, checked rather than argued.

    `started_at` is a table column. If it ever reaches `spec_json`, every training identity
    in the corpus moves and every stored result is orphaned - so this asserts both that the
    hash matches the spec's own projection and that the stored spec does not carry the field.
    """
    study = _study(tmp_path)
    spec = _training_spec()
    expected = training_hash_from_spec(spec)

    training = study.results.register_training(spec, started_at="2026-09-05T04:40:19+00:00")

    assert training.hash == expected, "recording a start time changed the training identity"
    with closing(sqlite3.connect(study.root / "run_log" / "registry.db")) as db:
        spec_json = db.execute(
            "SELECT spec_json FROM training_runs WHERE training_hash = ?", (training.hash,)
        ).fetchone()[0]
    assert "started_at" not in spec_json, "started_at leaked into the hashed spec"


def test_two_runs_of_one_spec_keep_one_identity_and_differ_in_start(tmp_path: Path) -> None:
    """Re-registering the same identity is not a new identity, whatever the clock says."""
    study = _study(tmp_path)
    spec = _training_spec()
    first = study.results.register_training(spec, started_at="2026-09-05T01:00:00+00:00")
    second = study.results.register_training(spec, started_at="2026-09-05T02:00:00+00:00")
    assert first.hash == second.hash
