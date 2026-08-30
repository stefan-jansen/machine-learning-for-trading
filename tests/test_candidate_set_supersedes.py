"""Whether a declared candidate-set generation may be offered to `CandidateSet.create`.

A notebook that has published a candidate set and then changed its membership has to name the
generation it replaces, or `create` refuses the write. The hash it names is committed source, so
the same declaration reaches three situations the notebook cannot tell apart, and offering it
unconditionally is wrong in the one that matters most for a published repository - a reader's
clean clone, where `run_log/` is gitignored and there is no generation to supersede.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CandidateSet, candidate_set_supersedes
from case_studies.research.workspace import Study


@pytest.fixture
def study(tmp_path: Path) -> Study:
    case_dir = tmp_path / "cs"
    (case_dir / "run_log").mkdir(parents=True)
    return Study(
        case_study="cs",
        root=case_dir,
        release_root=tmp_path,
        output_root=tmp_path,
        read_only=False,
        entry_point="test",
        manifest={},
    )


def _generations(study: Study, rows: list[tuple[str, str | None]]) -> None:
    """Write a lineage directly, so the test states the registry rather than building one."""
    db = sqlite3.connect(study.root / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE IF NOT EXISTS candidate_sets ("
        "set_hash TEXT PRIMARY KEY, name TEXT, member_kind TEXT, comparison_contract_json TEXT, "
        "created_at TEXT, git_commit TEXT, supersedes_hash TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS candidate_set_members "
        "(set_hash TEXT, member_hash TEXT, ordinal INTEGER)"
    )
    for index, (set_hash, supersedes) in enumerate(rows):
        db.execute(
            "INSERT INTO candidate_sets VALUES (?,?,?,?,?,?,?)",
            (
                set_hash,
                "the-set",
                "backtest",
                '{"protocol": {}}',
                f"2026-01-0{index + 1}",
                "abc",
                supersedes,
            ),
        )
        db.execute("INSERT INTO candidate_set_members VALUES (?, ?, 0)", (set_hash, "member"))
    db.commit()
    db.close()


def test_a_clean_clone_withholds_the_declaration(study: Study) -> None:
    """No `candidate_sets` table at all, which is what a reader starts with.

    `create` refuses a first generation that claims to supersede something, so a notebook that
    offers its committed hash here stops on the reader's first run.
    """
    assert not (study.root / "run_log" / "registry.db").exists()

    assert candidate_set_supersedes(study, name="the-set", declared="aaaaaaaaaaaa") is None


def test_an_empty_lineage_withholds_the_declaration(study: Study) -> None:
    """The table exists and holds no generation of this name."""
    _generations(study, [])

    assert candidate_set_supersedes(study, name="the-set", declared="aaaaaaaaaaaa") is None


def test_the_re_run_offers_the_hash_its_own_generation_replaced(study: Study) -> None:
    """The generation in force is the one this declaration produced."""
    _generations(study, [("aaaaaaaaaaaa", None), ("bbbbbbbbbbbb", "aaaaaaaaaaaa")])

    assert (
        candidate_set_supersedes(study, name="the-set", declared="aaaaaaaaaaaa") == "aaaaaaaaaaaa"
    )


def test_the_refit_offers_a_hash_that_names_the_tip(study: Study) -> None:
    """The declaration names the current head, so the next generation goes over it."""
    _generations(study, [("aaaaaaaaaaaa", None)])

    assert (
        candidate_set_supersedes(study, name="the-set", declared="aaaaaaaaaaaa") == "aaaaaaaaaaaa"
    )


def test_a_hash_naming_neither_the_tip_nor_its_predecessor_is_withheld(study: Study) -> None:
    """`create` then refuses and names the hash it wants, which beats this guessing."""
    _generations(study, [("aaaaaaaaaaaa", None), ("bbbbbbbbbbbb", "aaaaaaaaaaaa")])

    assert candidate_set_supersedes(study, name="the-set", declared="cccccccccccc") is None


def test_no_declaration_is_no_declaration(study: Study) -> None:
    _generations(study, [("aaaaaaaaaaaa", None)])

    assert candidate_set_supersedes(study, name="the-set", declared=None) is None
    assert candidate_set_supersedes(study, name="the-set", declared="") is None


def test_the_resolver_is_exported_beside_create() -> None:
    """It is the decision `CandidateSet.create`'s own `supersedes` parameter needs made."""
    assert "supersedes" in CandidateSet.create.__code__.co_varnames
    assert pl is not None
