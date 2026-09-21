"""The acceptance apparatus for a backtest re-key: what holds an address, and does it resolve."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry.backtest_rekey import (
    QUARANTINE_DIRNAME,
    dangling_references,
    declared_identities,
    directory_bijection,
    hash_bearing_sites,
    open_readonly,
    referential_regressions,
    registered_addresses,
)
from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL

#: Addresses are twelve hex characters, so the fixtures spell real ones rather than labels.
ALPHA = "aaaaaaaaaaa1"
BETA = "bbbbbbbbbbb2"
GAMMA = "ccccccccccc3"
ABSENT = "ddddddddddd4"

_TRAINING = "1111111111a0"
_PREDICTION = "2222222222b0"


def _registry(case_dir: Path, addresses: list[str]) -> Path:
    """A registry on the real schema holding *addresses*, with an artifact directory each."""
    run_log = case_dir / "run_log"
    (run_log / "backtest").mkdir(parents=True, exist_ok=True)
    path = run_log / "registry.db"
    with sqlite3.connect(path) as db:
        db.executescript(REGISTRY_SCHEMA_SQL)
        db.execute(
            "INSERT INTO training_runs (training_hash, family, label, created_at) VALUES (?,?,?,?)",
            (_TRAINING, "gbm", "fwd_ret_21d", "2026-09-01T00:00:00+00:00"),
        )
        db.execute(
            "INSERT INTO prediction_sets (prediction_hash, training_hash, split, created_at) "
            "VALUES (?,?,?,?)",
            (_PREDICTION, _TRAINING, "validation", "2026-09-01T00:00:00+00:00"),
        )
        for index, address in enumerate(addresses):
            db.execute(
                "INSERT INTO backtest_runs (backtest_hash, prediction_hash, spec_json, stage, "
                "created_at) VALUES (?,?,?,?,?)",
                (
                    address,
                    _PREDICTION,
                    json.dumps({"strategy": {"signal": {"method": "slot", "rank": index}}}),
                    "signal",
                    f"2026-09-0{index + 1}T00:00:00+00:00",
                ),
            )
            db.execute(
                "INSERT INTO backtest_metrics (backtest_hash, computed_at, sharpe) VALUES (?,?,?)",
                (address, "2026-09-01T00:00:00+00:00", 1.0 + index),
            )
            (run_log / "backtest" / address).mkdir()
    return path


def _add_candidate_set(path: Path, set_hash: str, members: list[str]) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO candidate_sets (set_hash, name, member_kind, comparison_contract_json, "
            "created_at) VALUES (?,?,?,?,?)",
            (set_hash, "a-comparison", "backtest", "{}", "2026-09-01T00:00:00+00:00"),
        )
        db.executemany(
            "INSERT INTO candidate_set_members (set_hash, member_hash, ordinal) VALUES (?,?,?)",
            [(set_hash, member, ordinal) for ordinal, member in enumerate(members)],
        )


def _add_population(path: Path, population_hash: str, members: list[str]) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO official_populations (population_hash, name, member_kind, snapshot_json, "
            "created_at) VALUES (?,?,?,?,?)",
            (
                population_hash,
                "a-population",
                "backtest",
                json.dumps({"members": members}),
                "2026-09-01T00:00:00+00:00",
            ),
        )
        db.executemany(
            "INSERT INTO official_population_members (population_hash, member_hash, ordinal) "
            "VALUES (?,?,?)",
            [(population_hash, member, ordinal) for ordinal, member in enumerate(members)],
        )


def _add_cohort(path: Path, leader: str, members: list[str]) -> None:
    digest = hashlib.sha256("\n".join(sorted(set(members))).encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO cohort_metrics (cohort_type, stage, label, leader_hash, k_variants, "
            "member_digest, members_json, periods_per_year, computed_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "stage",
                "signal",
                "fwd_ret_21d",
                leader,
                len(members),
                digest,
                json.dumps(members),
                252.0,
                "2026-09-01T00:00:00+00:00",
            ),
        )


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    _registry(tmp_path, [ALPHA, BETA, GAMMA])
    return tmp_path


def _registry_path(case_dir: Path) -> Path:
    return case_dir / "run_log" / "registry.db"


def _site(sites, table: str, column: str):
    return next((s for s in sites if s.table == table and s.column == column), None)


# --------------------------------------------------------------------------------------
# The site scan
# --------------------------------------------------------------------------------------


def test_registered_addresses_are_what_backtest_runs_declares(case_dir: Path) -> None:
    with open_readonly(_registry_path(case_dir)) as db:
        assert registered_addresses(db) == {ALPHA, BETA, GAMMA}


def test_a_missing_table_contributes_an_empty_set_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "bare.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY)")
        db.execute("INSERT INTO backtest_runs VALUES (?)", (ALPHA,))
    with open_readonly(path) as db:
        owned = declared_identities(db)
    assert owned["backtest_runs"] == {ALPHA}
    assert owned["candidate_sets"] == set()


def test_the_scan_finds_a_bare_reference_column(case_dir: Path) -> None:
    _add_candidate_set(_registry_path(case_dir), "eeeeeeeeeee5", [ALPHA, BETA])
    with open_readonly(_registry_path(case_dir)) as db:
        site = _site(hash_bearing_sites(db), "candidate_set_members", "member_hash")
    assert site is not None
    assert (site.bare, site.embedded) == (2, 0)
    assert not site.is_document


def test_the_scan_finds_an_address_inside_a_json_document(case_dir: Path) -> None:
    _add_population(_registry_path(case_dir), "fffffffffff6", [ALPHA, GAMMA])
    with open_readonly(_registry_path(case_dir)) as db:
        site = _site(hash_bearing_sites(db), "official_populations", "snapshot_json")
    assert site is not None
    assert (site.bare, site.embedded) == (0, 1)
    assert site.is_document


def test_the_scan_excludes_the_column_that_declares_the_address(case_dir: Path) -> None:
    with open_readonly(_registry_path(case_dir)) as db:
        assert _site(hash_bearing_sites(db), "backtest_runs", "backtest_hash") is None
        assert _site(hash_bearing_sites(db), "backtest_metrics", "backtest_hash") is not None


def test_the_scan_reads_the_schema_so_a_table_added_later_is_covered(case_dir: Path) -> None:
    with sqlite3.connect(_registry_path(case_dir)) as db:
        db.execute("CREATE TABLE a_table_written_after_this_module (note TEXT)")
        db.execute("INSERT INTO a_table_written_after_this_module VALUES (?)", (BETA,))
    with open_readonly(_registry_path(case_dir)) as db:
        site = _site(hash_bearing_sites(db), "a_table_written_after_this_module", "note")
    assert site is not None and site.bare == 1


def test_a_sha256_digest_is_not_read_as_a_reference(case_dir: Path) -> None:
    """``cohort_metrics.member_digest`` is derived from addresses and is not one."""
    _add_cohort(_registry_path(case_dir), ALPHA, [ALPHA, BETA])
    with open_readonly(_registry_path(case_dir)) as db:
        sites = hash_bearing_sites(db)
        dangling = dangling_references(db)
    assert _site(sites, "cohort_metrics", "member_digest") is None
    assert ("cohort_metrics", "member_digest") not in dangling
    assert _site(sites, "cohort_metrics", "members_json").embedded == 1


def test_the_scan_asks_only_about_the_addresses_it_is_given(case_dir: Path) -> None:
    _add_candidate_set(_registry_path(case_dir), "eeeeeeeeeee5", [ALPHA, BETA])
    with open_readonly(_registry_path(case_dir)) as db:
        site = _site(hash_bearing_sites(db, [ALPHA]), "candidate_set_members", "member_hash")
    assert site is not None and site.bare == 1


# --------------------------------------------------------------------------------------
# The referential delta
# --------------------------------------------------------------------------------------


def test_a_reference_to_an_unregistered_address_dangles(case_dir: Path) -> None:
    _add_candidate_set(_registry_path(case_dir), "eeeeeeeeeee5", [ALPHA, ABSENT])
    with open_readonly(_registry_path(case_dir)) as db:
        assert dangling_references(db)[("candidate_set_members", "member_hash")] == 1


def test_a_retired_token_inside_a_document_dangles(case_dir: Path) -> None:
    _add_population(_registry_path(case_dir), "fffffffffff6", [ABSENT])
    with open_readonly(_registry_path(case_dir)) as db:
        assert dangling_references(db)[("official_populations", "snapshot_json")] == 1


def test_a_document_dangles_when_any_one_of_its_tokens_does(case_dir: Path) -> None:
    """One resolving member does not vouch for the rest of the list.

    The 2026-09-14 gate asked whether a document held *any* token that resolved, so a
    snapshot naming one live member and twenty retired ones read as healthy.
    """
    _add_population(_registry_path(case_dir), "fffffffffff6", [ALPHA, ABSENT])
    with open_readonly(_registry_path(case_dir)) as db:
        assert dangling_references(db)[("official_populations", "snapshot_json")] == 1


def test_an_identity_of_another_kind_is_a_declaration_and_does_not_dangle(
    case_dir: Path,
) -> None:
    """``decision_artifacts`` and ``causal_runs`` declare their own identities.

    Left out of the universe they read as 879 dangling references in ``sp500_options``,
    which is every decision the case study has recorded.
    """
    with sqlite3.connect(_registry_path(case_dir)) as db:
        db.execute(
            "INSERT INTO decision_artifacts (decision_hash, decision_kind, spec_json, "
            "artifact_digest, canonical, created_at) VALUES (?,?,?,?,?,?)",
            ("9999999999e9", "target_weights", "{}", "f0" * 8, 1, "2026-09-01T00:00:00+00:00"),
        )
    with open_readonly(_registry_path(case_dir)) as db:
        assert ("decision_artifacts", "decision_hash") not in dangling_references(db)


def test_a_dangling_count_that_did_not_move_is_not_a_regression(case_dir: Path) -> None:
    """A healthy registry dangles by design, so the gate is a delta and not an absolute."""
    _add_population(_registry_path(case_dir), "fffffffffff6", [ABSENT])
    with open_readonly(_registry_path(case_dir)) as db:
        before = dangling_references(db)
        after = dangling_references(db)
    assert before[("official_populations", "snapshot_json")] == 1
    assert referential_regressions(before, after) == {}


def test_a_dangling_count_that_rose_is_a_regression() -> None:
    site = ("candidate_set_members", "member_hash")
    assert referential_regressions({site: 3}, {site: 4}) == {site: (3, 4)}
    assert referential_regressions({}, {site: 1}) == {site: (0, 1)}
    assert referential_regressions({site: 4}, {site: 3}) == {}


def test_breaking_a_reference_shows_up_as_a_regression(case_dir: Path) -> None:
    """The end-to-end shape of the gate: rewrite a hash in one place and not the other."""
    _add_candidate_set(_registry_path(case_dir), "eeeeeeeeeee5", [ALPHA, BETA])
    with open_readonly(_registry_path(case_dir)) as db:
        before = dangling_references(db)
    with sqlite3.connect(_registry_path(case_dir)) as db:
        db.execute(
            "UPDATE backtest_runs SET backtest_hash = ? WHERE backtest_hash = ?", (ABSENT, ALPHA)
        )
    with open_readonly(_registry_path(case_dir)) as db:
        after = dangling_references(db)
    assert referential_regressions(before, after)[("candidate_set_members", "member_hash")] == (
        0,
        1,
    )


# --------------------------------------------------------------------------------------
# The row-to-directory bijection
# --------------------------------------------------------------------------------------


def test_the_bijection_is_exact_when_rows_and_directories_agree(case_dir: Path) -> None:
    result = directory_bijection(case_dir)
    assert result.exact
    assert (result.rows, result.directories) == (3, 3)


def test_the_bijection_reports_a_row_with_no_directory(case_dir: Path) -> None:
    (case_dir / "run_log" / "backtest" / BETA).rmdir()
    result = directory_bijection(case_dir)
    assert not result.exact
    assert result.rows_without_directory == (BETA,)
    assert result.directories_without_row == ()


def test_the_bijection_reports_a_directory_with_no_row(case_dir: Path) -> None:
    (case_dir / "run_log" / "backtest" / ABSENT).mkdir()
    result = directory_bijection(case_dir)
    assert not result.exact
    assert result.rows_without_directory == ()
    assert result.directories_without_row == (ABSENT,)


def test_a_rename_shows_up_as_a_symmetric_pair_of_counts(case_dir: Path) -> None:
    """The symmetry is what says a rename rather than data loss, so both sides are counted."""
    backtests = case_dir / "run_log" / "backtest"
    (backtests / ALPHA).rename(backtests / ABSENT)
    result = directory_bijection(case_dir)
    assert result.rows_without_directory == (ALPHA,)
    assert result.directories_without_row == (ABSENT,)
    assert len(result.rows_without_directory) == len(result.directories_without_row)


def test_the_bijection_ignores_a_quarantined_directory(case_dir: Path) -> None:
    (case_dir / "run_log" / "backtest" / QUARANTINE_DIRNAME).mkdir()
    assert directory_bijection(case_dir).exact
