"""The acceptance apparatus for a backtest re-key: what holds an address, and does it resolve."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.registry.backtest_rekey import (
    QUARANTINE_DIRNAME,
    Row,
    dangling_references,
    declared_identities,
    directory_bijection,
    hash_bearing_sites,
    open_readonly,
    plan_backtest_rekey,
    pre_elision_hash,
    prove_merge,
    referential_regressions,
    registered_addresses,
)
from case_studies.utils.registry.specs import backtest_hash_from_parts
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


def test_the_reader_closes_the_registry_it_opened(case_dir: Path) -> None:
    # `sqlite3.Connection.__exit__` ends the transaction and leaves the handle open, so a
    # scan over the nine shared registries would hold every descriptor until collection.
    with open_readonly(_registry_path(case_dir)) as db:
        assert registered_addresses(db) == {ALPHA, BETA, GAMMA}
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")


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


def test_a_document_dangles_once_per_broken_token_and_not_once_per_row(case_dir: Path) -> None:
    """Counting rows would let a snapshot that already dangles absorb any number of breaks.

    A snapshot naming one live member and one retired one counts 1. Retiring a second member
    has to make it 2, or the delta that gates the migration cannot see the breakage in the
    two columns it exists for.
    """
    _add_population(_registry_path(case_dir), "fffffffffff6", [ALPHA, BETA, ABSENT])
    site = ("official_populations", "snapshot_json")
    with open_readonly(_registry_path(case_dir)) as db:
        before = dangling_references(db)
    assert before[site] == 1
    with sqlite3.connect(_registry_path(case_dir)) as db:
        db.execute("DELETE FROM backtest_metrics WHERE backtest_hash = ?", (BETA,))
        db.execute("DELETE FROM backtest_runs WHERE backtest_hash = ?", (BETA,))
    with open_readonly(_registry_path(case_dir)) as db:
        after = dangling_references(db)
    assert after[site] == 2
    assert referential_regressions(before, after)[site] == (1, 2)


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


# --------------------------------------------------------------------------------------
# Planning a re-key
# --------------------------------------------------------------------------------------

ALL_ELIDED = ("account", "position_sizing", "feed")
DIGESTS = {"daily_returns.parquet": "a1" * 8, "weights.parquet": "b2" * 8}
OTHER_DIGESTS = {"daily_returns.parquet": "c3" * 8, "weights.parquet": "d4" * 8}


def _write_artifacts(case_dir: Path, address: str, returns: list[float | None]) -> None:
    """The two artifacts `DIGESTS` names, so a digest mismatch can be read past."""
    directory = case_dir / "run_log" / "backtest" / address
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, day + 1) for day in range(len(returns))],
            "daily_return": pl.Series(returns, dtype=pl.Float64),
        }
    ).write_parquet(directory / "daily_returns.parquet")
    pl.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.5, 0.5]}).write_parquet(
        directory / "weights.parquet"
    )


def _spec(rank: int, *, elided: tuple[str, ...] = ()) -> dict:
    """A stored spec, written with *elided* naming which engine-default keys it spells out.

    A row that spells any of them out is what a pre-fix engine registered, and it is exactly
    the row whose stored address today's hasher no longer computes. Every variant reduces to
    one hashable view, so they all compute one address and differ only in the pre-fix one -
    which is the shape of a real collision.
    """
    config: dict = {
        "account": {"initial_cash": 100_000},
        "position_sizing": {"max_weight": 0.1},
        "feed": {"price_col": "close"},
    }
    if "account" in elided:
        config["account"]["lock_notional_update_mode"] = "position_legs"
    if "position_sizing" in elided:
        config["position_sizing"]["share_rounding"] = "nearest"
    if "feed" in elided:
        config["feed"]["vwap_col"] = None
    return {
        "strategy": {"signal": {"method": "slot_persistent", "rank": rank}},
        "backtest_config": config,
    }


def _insert_run(
    path: Path,
    address: str,
    spec: dict,
    *,
    created_at: str,
    digests: dict[str, str] | None = DIGESTS,
) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO backtest_runs (backtest_hash, prediction_hash, spec_json, stage, "
            "created_at, artifact_digests_json) VALUES (?,?,?,?,?,?)",
            (
                address,
                _PREDICTION,
                json.dumps(spec),
                "signal",
                created_at,
                json.dumps(digests) if digests is not None else None,
            ),
        )


@pytest.fixture
def empty_case(tmp_path: Path) -> Path:
    _registry(tmp_path, [])
    return tmp_path


def _moved_pair(rank: int) -> tuple[str, str, dict]:
    """``(stored address, address it computes today, spec)`` for a pre-fix row."""
    spec = _spec(rank, elided=ALL_ELIDED)
    return (
        pre_elision_hash(_PREDICTION, spec),
        backtest_hash_from_parts(_PREDICTION, spec),
        spec,
    )


def test_the_elided_keys_are_what_move_a_row(empty_case: Path) -> None:
    stored, computed, _ = _moved_pair(0)
    assert stored != computed
    unmoved = _spec(0)
    assert pre_elision_hash(_PREDICTION, unmoved) == backtest_hash_from_parts(_PREDICTION, unmoved)


def test_a_row_that_computes_its_own_address_is_reachable(empty_case: Path) -> None:
    spec = _spec(0)
    _insert_run(
        _registry_path(empty_case),
        backtest_hash_from_parts(_PREDICTION, spec),
        spec,
        created_at="2026-09-01T00:00:00+00:00",
    )
    plan = plan_backtest_rekey(empty_case)
    assert (plan.rows, plan.reachable, plan.moved) == (1, 1, 0)
    assert plan.mapping == {} and not plan.skipped


def test_a_row_written_with_the_elided_keys_moves_and_is_explained(empty_case: Path) -> None:
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-01T00:00:00+00:00")
    plan = plan_backtest_rekey(empty_case)
    assert (plan.rows, plan.reachable, plan.moved) == (1, 0, 1)
    assert plan.mapping == {stored: computed}
    assert plan.unexplained == () and not plan.skipped


def test_an_address_neither_hasher_computes_refuses_the_whole_registry(empty_case: Path) -> None:
    _, _, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), ABSENT, spec, created_at="2026-09-01T00:00:00+00:00")
    plan = plan_backtest_rekey(empty_case)
    assert plan.unexplained == (ABSENT,)
    assert plan.skipped
    assert "neither today's hasher nor the pre-elision one" in plan.refusals[0]


def test_a_target_that_is_another_kind_of_identity_refuses_the_registry(empty_case: Path) -> None:
    """``official_population_members.member_hash`` holds whichever kind the population was
    cut over - prediction hashes in four of the nine registries and backtest addresses in
    ``crypto_perps_funding`` - so a mapping keyed on backtest addresses alone would rewrite
    a prediction reference the moment the two namespaces meet. They do not meet in any of
    the nine today; this is what says so rather than assuming it.
    """
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-01T00:00:00+00:00")
    _add_candidate_set(_registry_path(empty_case), computed, [stored])

    plan = plan_backtest_rekey(empty_case)
    assert plan.namespace_clashes == (computed,)
    assert plan.skipped
    assert "already an identity of another kind" in "".join(plan.refusals)


def test_an_address_being_vacated_that_is_another_identity_refuses_too(empty_case: Path) -> None:
    """The silent side. A re-key rewrites a reference site by matching the stored value, so
    an old address that is also a prediction hash turns a prediction reference into a
    backtest one and nothing downstream can tell. A minted address that clashes only leaves
    two kinds sharing a value.
    """
    stored, _computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-01T00:00:00+00:00")
    _add_candidate_set(_registry_path(empty_case), stored, [stored])

    plan = plan_backtest_rekey(empty_case)
    assert plan.namespace_clashes == (stored,)
    assert plan.skipped


def test_a_collision_keeps_the_earlier_row_when_it_is_the_mover(empty_case: Path) -> None:
    """nasdaq's shape: the row already at the target address is the later of the two."""
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-12T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-19T00:00:00+00:00",
    )
    plan = plan_backtest_rekey(empty_case)
    assert plan.occupied == 1
    merge = plan.merges[0]
    assert (merge.target, merge.survivor, merge.retired) == (computed, stored, (computed,))
    assert merge.proved and not plan.skipped
    assert plan.mapping == {stored: computed}


def test_a_collision_keeps_the_earlier_row_when_it_is_already_at_the_target(
    empty_case: Path,
) -> None:
    """The other registries' shape, and the same rule: the survivor is chosen on created_at."""
    stored, computed, spec = _moved_pair(0)
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-06T00:00:00+00:00",
    )
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-18T00:00:00+00:00")
    plan = plan_backtest_rekey(empty_case)
    merge = plan.merges[0]
    assert (merge.survivor, merge.retired) == (computed, (stored,))
    assert merge.proved
    assert plan.mapping == {}


def test_a_collision_whose_rows_hold_different_artifacts_refuses(empty_case: Path) -> None:
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=OTHER_DIGESTS,
    )
    _write_artifacts(empty_case, stored, [0.01, 0.02, 0.03])
    _write_artifacts(empty_case, computed, [0.01, 0.02, 0.49])

    plan = plan_backtest_rekey(empty_case)
    assert not plan.merges[0].proved
    assert plan.merges[0].reason == (
        "the colliding rows hold different numbers in daily_returns.parquet"
    )
    assert plan.skipped
    assert plan.mapping == {}


def test_a_collision_differing_only_in_the_last_bit_of_a_float_is_proved(
    empty_case: Path,
) -> None:
    """us_firm_characteristics' shape: 47 of its 81 pairs differ by 5.6e-17 on 4 of 110
    values, because a float sum is not associative. A digest reads that as two results.
    """
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=OTHER_DIGESTS,
    )
    _write_artifacts(empty_case, stored, [0.05365977120707852, 0.02, 0.03])
    _write_artifacts(empty_case, computed, [0.05365977120707854, 0.02, 0.03])

    plan = plan_backtest_rekey(empty_case)
    assert plan.merges[0].proved, plan.merges[0].reason
    assert not plan.skipped


@pytest.mark.parametrize(
    ("left", "right", "proved", "pins"),
    [
        # Only the absolute tolerance can carry this: the relative one is a multiple of a
        # value that is zero.
        (0.0, 1e-17, True, "atol"),
        # Only the relative one can carry this: the gap is a thousand times the absolute
        # tolerance, and a millionth of the value.
        (1.0, 1.0 + 1e-10, True, "rtol"),
        # And neither carries a gap this wide, which is what keeps the window honest.
        (1.0, 1.0 + 1e-7, False, "the upper bound"),
    ],
)
def test_the_artifact_tolerance_window(
    empty_case: Path, left: float, right: float, proved: bool, pins: str
) -> None:
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=OTHER_DIGESTS,
    )
    _write_artifacts(empty_case, stored, [left, 0.02, 0.03])
    _write_artifacts(empty_case, computed, [right, 0.02, 0.03])

    plan = plan_backtest_rekey(empty_case)
    assert plan.merges[0].proved is proved, f"{pins}: {plan.merges[0].reason}"


@pytest.mark.parametrize(
    ("left", "right"),
    [(None, 0.49), (0.49, None)],
)
def test_a_missing_value_against_a_present_one_refuses(
    empty_case: Path, left: float | None, right: float | None
) -> None:
    """The loudest difference there is, and the one a tolerance test cannot see: `null - x`
    is null, `null > tolerance` is null, and polars' `.any()` skips nulls.
    """
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=OTHER_DIGESTS,
    )
    _write_artifacts(empty_case, stored, [left, 0.02, 0.03])
    _write_artifacts(empty_case, computed, [right, 0.02, 0.03])

    plan = plan_backtest_rekey(empty_case)
    assert not plan.merges[0].proved
    assert plan.merges[0].reason == (
        "the colliding rows hold different numbers in daily_returns.parquet"
    )


def test_two_nulls_in_the_same_place_are_agreement(empty_case: Path) -> None:
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=OTHER_DIGESTS,
    )
    _write_artifacts(empty_case, stored, [None, 0.05365977120707852, 0.03])
    _write_artifacts(empty_case, computed, [None, 0.05365977120707854, 0.03])

    plan = plan_backtest_rekey(empty_case)
    assert plan.merges[0].proved, plan.merges[0].reason


def test_a_digest_mismatch_with_no_artifact_to_read_refuses(empty_case: Path) -> None:
    """Conservative when the tree is not there: an unreadable artifact is not a proof."""
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=OTHER_DIGESTS,
    )
    plan = plan_backtest_rekey(empty_case)
    assert not plan.merges[0].proved
    assert "unreadable" in plan.merges[0].reason
    assert plan.skipped


def test_a_collision_whose_rows_record_no_digests_refuses(empty_case: Path) -> None:
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-06T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-18T00:00:00+00:00",
        digests=None,
    )
    assert plan_backtest_rekey(empty_case).merges[0].reason.startswith("a row in the collision")


def test_the_digest_maps_are_compared_whole_rather_than_counted() -> None:
    """``us_firm_characteristics`` records two artifact digests where the others record six."""
    two = {"daily_returns.parquet": "a1" * 8, "weights.parquet": "b2" * 8}
    rows = [
        Row(
            ALPHA,
            GAMMA,
            _PREDICTION,
            "2026-09-01T00:00:00+00:00",
            "signal",
            two,
            _spec(0, elided=ALL_ELIDED),
        ),
        Row(BETA, GAMMA, _PREDICTION, "2026-09-02T00:00:00+00:00", "signal", dict(two), _spec(0)),
    ]
    assert prove_merge(rows) == (True, "")
    rows[1].digests["weights.parquet"] = "ff" * 8
    assert prove_merge(rows)[0] is False


def test_rows_that_differ_after_the_elided_keys_are_stripped_refuse() -> None:
    """The branch a real collision cannot reach without a hash collision."""
    rows = [
        Row(
            ALPHA,
            GAMMA,
            _PREDICTION,
            "2026-09-01T00:00:00+00:00",
            "signal",
            dict(DIGESTS),
            _spec(0, elided=ALL_ELIDED),
        ),
        Row(
            BETA,
            GAMMA,
            _PREDICTION,
            "2026-09-02T00:00:00+00:00",
            "signal",
            dict(DIGESTS),
            _spec(1, elided=ALL_ELIDED),
        ),
    ]
    proved, reason = prove_merge(rows)
    assert not proved
    assert reason == "the colliding rows hold different specs after the elided keys are stripped"


def test_three_rows_at_one_address_keep_only_the_earliest(empty_case: Path) -> None:
    stored, computed, spec = _moved_pair(0)
    _insert_run(_registry_path(empty_case), stored, spec, created_at="2026-09-02T00:00:00+00:00")
    _insert_run(
        _registry_path(empty_case),
        computed,
        _spec(0),
        created_at="2026-09-03T00:00:00+00:00",
    )
    third = _spec(0, elided=("account",))
    _insert_run(
        _registry_path(empty_case),
        pre_elision_hash(_PREDICTION, third),
        third,
        created_at="2026-09-01T00:00:00+00:00",
    )
    plan = plan_backtest_rekey(empty_case)
    merge = plan.merges[0]
    assert merge.survivor == pre_elision_hash(_PREDICTION, third)
    assert sorted(merge.retired) == sorted([stored, computed])
    assert merge.proved and plan.unexplained == ()
