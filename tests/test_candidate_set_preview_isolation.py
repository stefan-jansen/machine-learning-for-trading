"""A preview run cannot write a candidate set into the canonical registry.

`OfficialPopulation.create` refuses to activate a preview study. `CandidateSet.create`
performed the same activation and did not, so a preview run could write a canonical
candidate set with nothing raising.

The tier check `create` does carry refuses preview *members*, and that is a different
question. A `Study`'s `root` is the canonical case directory whatever tier is active, so a
preview resolved *canonical* members: every member's `execution_tier` read `canonical`, the
member check passed, and the write went to the shared registry. Nothing downstream
distinguishes that row from one a canonical run wrote.

What made it visible is the case where it is safe. A preview of
`crypto_perps_funding/19_strategy_analysis` failed with

    ValueError: a changed candidate set named 'crypto-final-selection'
                must explicitly supersedes a840029e01ed

and `a840029e01ed` is not in the preview workspace - it is in the canonical crypto registry.
It raised only because the name was already bound. With the name unbound, which is the state
five of six case studies were in on the morning of 2026-09-06, the same call reaches the
write and says nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from case_studies.research.comparison import CandidateSet
from case_studies.research.contracts import ExecutionTier
from case_studies.research.workspace import Study


def _study(tmp_path: Path, tier: ExecutionTier) -> Study:
    case_dir = tmp_path / "cs"
    (case_dir / "run_log").mkdir(parents=True, exist_ok=True)
    (case_dir / "config").mkdir(parents=True, exist_ok=True)
    return Study(
        case_study="cs",
        root=case_dir,
        release_root=tmp_path,
        output_root=tmp_path,
        read_only=False,
        entry_point="test",
        manifest={},
        execution_tier=tier,
    )


def test_a_preview_run_cannot_create_a_candidate_set(tmp_path: Path) -> None:
    """The refusal, and it has to fire before anything is resolved.

    `create` is given no members here on purpose: it must refuse for being a preview rather
    than for the argument being empty, so the check has to sit ahead of the member
    validation it used to sit behind. A preview reaching that validation is a preview that
    has already resolved canonical results.
    """
    study = _study(tmp_path, ExecutionTier.PREVIEW)

    with pytest.raises(ValueError, match="preview run cannot create"):
        CandidateSet.create(study, "the-set", [])


def test_an_empty_member_list_is_still_refused_for_a_canonical_run(tmp_path: Path) -> None:
    """The other side of the ordering, so the preview check has not swallowed a real one."""
    study = _study(tmp_path, ExecutionTier.CANONICAL)

    with pytest.raises(ValueError, match="at least one member"):
        CandidateSet.create(study, "the-set", [])


def test_a_canonical_run_resolves_to_exactly_the_path_it_resolved_before(
    tmp_path: Path,
) -> None:
    """`CandidateSet` has far more callers than the coverage gate did.

    Routing its reads and writes through the tier's storage root is only safe if the
    canonical tier still lands on the case directory itself, which is the one property every
    existing caller depends on.
    """
    from case_studies.research.comparison import _candidate_set_root

    study = _study(tmp_path, ExecutionTier.CANONICAL)

    assert _candidate_set_root(study) == study.root


def test_a_preview_reads_its_own_workspace_rather_than_the_canonical_registry(
    tmp_path: Path,
) -> None:
    """The visible symptom: a preview resolving a hash out of the shared registry.

    The canonical registry holds a set under this name and the preview workspace does not,
    so a lookup that addresses the canonical root finds it and one that addresses the
    preview root does not. `one` raising for an unbound name is the correct preview answer:
    the preview has published nothing.

    The root helper is imported inside the test rather than at module scope, so that a build
    without it fails the refusal above on its behaviour instead of failing the whole file at
    collection.
    """
    from case_studies.research.comparison import _candidate_set_root

    canonical = _study(tmp_path, ExecutionTier.CANONICAL)
    with sqlite3.connect(canonical.root / "run_log" / "registry.db") as db:
        db.execute(
            "CREATE TABLE candidate_sets (set_hash TEXT PRIMARY KEY, name TEXT, "
            "member_kind TEXT, comparison_contract_json TEXT, created_at TEXT, "
            "git_commit TEXT, supersedes_hash TEXT)"
        )
        db.execute(
            "CREATE TABLE candidate_set_members (set_hash TEXT, member_hash TEXT, ordinal INTEGER)"
        )
        db.execute(
            "INSERT INTO candidate_sets VALUES "
            "('a840029e01ed', 'crypto-final-selection', 'backtest', '{}', '2026-09-06', 'abc', NULL)"
        )
        db.execute("INSERT INTO candidate_set_members VALUES ('a840029e01ed', 'bt', 0)")

    assert CandidateSet.one(canonical, name="crypto-final-selection").hash == "a840029e01ed"

    preview = _study(tmp_path, ExecutionTier.PREVIEW)
    preview_root = _candidate_set_root(preview)
    assert preview_root != preview.root
    (preview_root / "run_log").mkdir(parents=True, exist_ok=True)
    with pytest.raises((ValueError, KeyError, sqlite3.OperationalError)):
        CandidateSet.one(preview, name="crypto-final-selection")
