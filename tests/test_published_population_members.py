"""What a case study publishes is a membership, not the complement of retirement.

A prediction no population ever listed has not been retired by anyone. A reader that ranks
over "everything not retired" therefore admits experimental results the case study never
published, and does so most easily in the ordinary case: one generation, nothing retired,
the exclusion set empty and the ranking wide open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from case_studies.research.population import published_members_at, superseded_members_at


def _registry(tmp_path: Path, generations, unlisted=()) -> Path:
    """``generations`` are (population_hash, name, supersedes_hash, members)."""
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "CREATE TABLE official_populations (population_hash TEXT, name TEXT, "
        "member_kind TEXT, supersedes_hash TEXT)"
    )
    db.execute("CREATE TABLE official_population_members (population_hash TEXT, member_hash TEXT)")
    for population_hash, name, supersedes, members in generations:
        db.execute(
            "INSERT INTO official_populations VALUES (?,?,?,?)",
            (population_hash, name, "prediction", supersedes),
        )
        db.executemany(
            "INSERT INTO official_population_members VALUES (?,?)",
            [(population_hash, m) for m in members],
        )
    db.commit()
    db.close()
    assert unlisted is not None
    return case_dir


def test_an_unlisted_prediction_is_not_published_though_nothing_retired_it(tmp_path) -> None:
    """The ordinary case: one generation, an empty retirement set, and a stray result."""
    case_dir = _registry(tmp_path, [("gen1", "models", None, ["a", "b"])])

    assert superseded_members_at(case_dir) == frozenset()
    published = published_members_at(case_dir)
    assert published == frozenset({"a", "b"})
    assert "experimental" not in published


def test_a_superseded_generation_is_not_published(tmp_path) -> None:
    case_dir = _registry(
        tmp_path,
        [
            ("gen1", "models", None, ["old"]),
            ("gen2", "models", "gen1", ["new"]),
        ],
    )

    assert published_members_at(case_dir) == frozenset({"new"})
    assert superseded_members_at(case_dir) == frozenset({"old"})


def test_a_downstream_name_cannot_restore_what_its_producer_retired(tmp_path) -> None:
    """Names are independent, so a backtest set can still list a prediction the models refit past.

    Listed by some tip and retired by nobody is the conjunction; the union alone would let the
    downstream listing put the retired identity back in front of a reader.
    """
    case_dir = _registry(
        tmp_path,
        [
            ("gen1", "models", None, ["old"]),
            ("gen2", "models", "gen1", ["new"]),
            ("bt1", "backtests", None, ["old", "new"]),
        ],
    )

    assert published_members_at(case_dir) == frozenset({"new"})


def test_names_are_independent(tmp_path) -> None:
    """One name moving on does not retire another name's members."""
    case_dir = _registry(
        tmp_path,
        [
            ("gen1", "models", None, ["old"]),
            ("gen2", "models", "gen1", ["new"]),
            ("bt1", "backtests", None, ["kept"]),
        ],
    )

    assert published_members_at(case_dir) == frozenset({"new", "kept"})


def test_a_registry_declaring_no_populations_cannot_be_asked(tmp_path) -> None:
    """None, not an empty set: membership is unavailable, so a caller must not narrow to nothing."""
    case_dir = tmp_path / "probe"
    (case_dir / "run_log").mkdir(parents=True)
    sqlite3.connect(case_dir / "run_log" / "registry.db").close()

    assert published_members_at(case_dir) is None
