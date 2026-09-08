"""A fixture registry keeps only the training runs its own artifacts back.

The nine CI fixture registries were copied whole from production while the artifacts
beside them were subsampled, so every copied training run pins a file the fixture has
never held. `register_training_run` refuses to add a run fitted on a different vintage
of the same artifact, which is correct and which stopped the 2026-09-07 regeneration at
`06_linear` for two case studies (ml4t/agent-workspace#1082). These tests pin the
pruning that clears the stale rows before the generator registers anything.
"""

import json
import sqlite3

import polars as pl
import pytest

from tests.fixture_registry import (
    capture_backed_prediction_rows,
    delete_training_runs,
    pinned_input_shas,
    prune_stale_training_runs,
    restore_backed_prediction_rows,
    sha256_file,
    shipped_artifact_shas,
    stale_training_runs,
)
from tests.generate_intermediates import registers_training_runs

SCHEMA = """
CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT, spec_json TEXT);
CREATE TABLE prediction_sets (prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT);
CREATE TABLE prediction_metrics (prediction_hash TEXT PRIMARY KEY, ic_mean REAL);
CREATE TABLE prediction_coverage (prediction_hash TEXT PRIMARY KEY, n_expected INTEGER);
CREATE TABLE fold_metrics (prediction_hash TEXT, fold_id INTEGER, ic REAL);
CREATE TABLE backtest_runs (backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT);
CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
CREATE TABLE backtest_fold_metrics (backtest_hash TEXT, fold_id INTEGER, sharpe REAL);
CREATE TABLE cohort_metrics (cohort_type TEXT, stage TEXT, leader_hash TEXT);
CREATE TABLE official_population_members (population_hash TEXT, member_hash TEXT, ordinal INTEGER);
"""


def _spec(**artifacts: str) -> str:
    return json.dumps(
        {
            "computation": {
                "input_data_spec": {
                    "artifacts": {
                        name: {"sha256": sha, "size": 1} for name, sha in artifacts.items()
                    }
                }
            }
        }
    )


@pytest.fixture
def case_dir(tmp_path):
    """A case study shipping one feature panel and one label, with a registry beside it."""
    (tmp_path / "features").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "run_log").mkdir()
    pl.DataFrame({"symbol": ["A"], "x": [1.0]}).write_parquet(
        tmp_path / "features/financial.parquet"
    )
    pl.DataFrame({"symbol": ["A"], "y": [1.0]}).write_parquet(
        tmp_path / "labels/fwd_ret_1d.parquet"
    )
    db = sqlite3.connect(str(tmp_path / "run_log" / "registry.db"))
    db.executescript(SCHEMA)
    db.commit()
    db.close()
    return tmp_path


def _connect(case_dir):
    return sqlite3.connect(str(case_dir / "run_log" / "registry.db"))


def test_shipped_shas_are_keyed_by_digest_not_by_artifact_name(case_dir):
    shipped = shipped_artifact_shas(case_dir)
    assert shipped == {
        sha256_file(case_dir / "features/financial.parquet"): "features/financial.parquet",
        sha256_file(case_dir / "labels/fwd_ret_1d.parquet"): "labels/fwd_ret_1d.parquet",
    }


def test_a_sha256_prefixed_pin_compares_against_the_bare_digest():
    """`feature_artifacts` writes `sha256:<hex>` in one shape and bare hex in the other.

    Reading the prefixed form without stripping it makes every such run look stale, which
    would delete a population the fixture does hold.
    """
    assert pinned_input_shas(_spec(financial="sha256:" + "a" * 64)) == {"financial": "a" * 64}


def test_a_run_fitted_on_a_shipped_artifact_is_not_stale(case_dir):
    on_disk = sha256_file(case_dir / "features/financial.parquet")
    db = _connect(case_dir)
    db.execute(
        "INSERT INTO training_runs VALUES (?,?,?)",
        ("t_local", "fwd_ret_1d", _spec(financial=on_disk)),
    )
    db.commit()
    assert stale_training_runs(db, case_dir) == {}
    db.close()


def test_a_run_pinning_an_artifact_the_fixture_lacks_is_stale_and_names_the_pin(case_dir):
    db = _connect(case_dir)
    db.execute(
        "INSERT INTO training_runs VALUES (?,?,?)",
        ("t_copied", "fwd_ret_1d", _spec(financial="b" * 64)),
    )
    db.commit()
    assert stale_training_runs(db, case_dir) == {"t_copied": {"financial": "b" * 64}}
    db.close()


def test_a_run_pinning_nothing_is_never_stale(case_dir):
    """A spec with no `input_data_spec.artifacts` block makes no claim to check."""
    db = _connect(case_dir)
    db.execute("INSERT INTO training_runs VALUES (?,?,?)", ("t_bare", "fwd_ret_1d", "{}"))
    db.commit()
    assert stale_training_runs(db, case_dir) == {}
    db.close()


def test_deleting_a_training_run_takes_its_whole_downstream_surface(case_dir):
    """SQLite leaves the foreign keys unenforced, so the cascade has to be spelled out.

    Deleting `training_runs` alone leaves prediction sets, metrics and backtests pointing
    at a row that is gone, which reads downstream as a corrupt registry rather than a
    pruned one.
    """
    db = _connect(case_dir)
    db.execute("INSERT INTO training_runs VALUES ('t','fwd_ret_1d','{}')")
    db.execute("INSERT INTO prediction_sets VALUES ('p','t','validation')")
    db.execute("INSERT INTO prediction_metrics VALUES ('p', 0.1)")
    db.execute("INSERT INTO prediction_coverage VALUES ('p', 10)")
    db.execute("INSERT INTO fold_metrics VALUES ('p', 0, 0.1)")
    db.execute("INSERT INTO backtest_runs VALUES ('b','p','signal')")
    db.execute("INSERT INTO backtest_metrics VALUES ('b', 1.0)")
    db.execute("INSERT INTO backtest_fold_metrics VALUES ('b', 0, 1.0)")
    db.execute("INSERT INTO cohort_metrics VALUES ('stagelabel','signal','b')")
    db.execute("INSERT INTO official_population_members VALUES ('pop','p',0)")
    db.commit()

    delete_training_runs(db, {"t"})

    for table in (
        "training_runs",
        "prediction_sets",
        "prediction_metrics",
        "prediction_coverage",
        "fold_metrics",
        "backtest_runs",
        "backtest_metrics",
        "backtest_fold_metrics",
        "cohort_metrics",
        "official_population_members",
    ):
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    db.close()


def test_pruning_keeps_the_runs_the_fixture_backs_and_drops_only_the_rest(case_dir):
    on_disk = sha256_file(case_dir / "features/financial.parquet")
    db = _connect(case_dir)
    db.execute(
        "INSERT INTO training_runs VALUES (?,?,?)",
        ("t_local", "fwd_ret_1d", _spec(financial=on_disk)),
    )
    db.execute(
        "INSERT INTO training_runs VALUES (?,?,?)",
        ("t_copied", "fwd_ret_1d", _spec(financial="b" * 64)),
    )
    db.execute("INSERT INTO prediction_sets VALUES ('p_local','t_local','validation')")
    db.execute("INSERT INTO prediction_sets VALUES ('p_copied','t_copied','validation')")
    db.commit()
    db.close()

    summary = prune_stale_training_runs(case_dir)
    assert summary["training_runs_pruned"] == 1
    assert summary["example"]["training_hash"] == "t_copied"

    db = _connect(case_dir)
    assert [r[0] for r in db.execute("SELECT training_hash FROM training_runs")] == ["t_local"]
    assert [r[0] for r in db.execute("SELECT prediction_hash FROM prediction_sets")] == ["p_local"]
    db.close()


def test_pruning_a_case_study_with_no_registry_yet_is_a_no_op(tmp_path):
    assert prune_stale_training_runs(tmp_path) == {}


def test_pruning_is_idempotent(case_dir):
    db = _connect(case_dir)
    db.execute(
        "INSERT INTO training_runs VALUES (?,?,?)",
        ("t_copied", "fwd_ret_1d", _spec(financial="b" * 64)),
    )
    db.commit()
    db.close()
    assert prune_stale_training_runs(case_dir)["training_runs_pruned"] == 1
    assert prune_stale_training_runs(case_dir)["training_runs_pruned"] == 0


@pytest.mark.parametrize(
    ("stem", "registers"),
    [
        ("01_feasibility_analysis", False),
        ("02_labels", False),
        ("03_financial_features", False),
        ("04_model_based_features", False),
        ("05_evaluation", False),
        ("06_linear", True),
        ("07_gbm", True),
        # us_firm_characteristics has no model-based stage, so its first registering
        # stage is 05, not 06. A boundary read from the stage number would prune after
        # that notebook had already registered against the copied rows.
        ("05_linear", True),
        ("04_evaluation", False),
    ],
)
def test_the_prune_happens_before_the_first_stage_that_registers(tmp_path, stem, registers):
    assert registers_training_runs(tmp_path / f"{stem}.py") is registers


def test_a_resample_keeps_the_rows_that_name_a_shipped_artifact(case_dir):
    """A rebuild from production must not strand the artifacts a generation wrote.

    `sample_registry_for_tests.py` unlinks the registry and rebuilds it from production
    while leaving `run_log/predictions/` in place, so without this every artifact the
    generator produced loses the only row that names it - 39 such directories on
    `sp500_equity_option_analytics` (ml4t/agent-workspace#1081).
    """
    (case_dir / "run_log" / "predictions" / "p_local").mkdir(parents=True)
    (case_dir / "run_log" / "predictions" / "p_local" / "predictions.parquet").write_bytes(b"")
    db_path = case_dir / "run_log" / "registry.db"
    db = _connect(case_dir)
    db.execute("INSERT INTO training_runs VALUES ('t_local','fwd_ret_1d','{}')")
    db.execute("INSERT INTO prediction_sets VALUES ('p_local','t_local','validation')")
    db.execute("INSERT INTO prediction_metrics VALUES ('p_local', 0.5)")
    db.commit()
    db.close()

    captured = capture_backed_prediction_rows(db_path, case_dir)

    # The rebuild: the registry is replaced by one carrying production's rows only.
    db_path.unlink()
    db = _connect(case_dir)
    db.executescript(SCHEMA)
    db.execute("INSERT INTO training_runs VALUES ('t_prod','fwd_ret_1d','{}')")
    db.execute("INSERT INTO prediction_sets VALUES ('p_prod','t_prod','validation')")
    db.commit()
    db.close()

    restore_backed_prediction_rows(db_path, captured)

    db = _connect(case_dir)
    assert sorted(r[0] for r in db.execute("SELECT prediction_hash FROM prediction_sets")) == [
        "p_local",
        "p_prod",
    ]
    assert db.execute(
        "SELECT ic_mean FROM prediction_metrics WHERE prediction_hash='p_local'"
    ).fetchone() == (0.5,)
    db.close()


def test_a_restored_row_replaces_the_production_row_for_the_same_hash(case_dir):
    """Where both registries carry the hash, the row measured on the shipped file wins.

    Production's metrics describe the production artifact; the fixture ships a subsample.
    Keeping production's row would leave the fixture claiming 2104 IC days for a parquet
    holding 66 (ml4t/agent-workspace#286).
    """
    (case_dir / "run_log" / "predictions" / "p").mkdir(parents=True)
    db_path = case_dir / "run_log" / "registry.db"
    db = _connect(case_dir)
    db.execute("INSERT INTO training_runs VALUES ('t','fwd_ret_1d','{}')")
    db.execute("INSERT INTO prediction_sets VALUES ('p','t','validation')")
    db.execute("INSERT INTO prediction_metrics VALUES ('p', 0.5)")
    db.commit()
    db.close()
    captured = capture_backed_prediction_rows(db_path, case_dir)

    db_path.unlink()
    db = _connect(case_dir)
    db.executescript(SCHEMA)
    db.execute("INSERT INTO training_runs VALUES ('t','fwd_ret_1d','{}')")
    db.execute("INSERT INTO prediction_sets VALUES ('p','t','validation')")
    db.execute("INSERT INTO prediction_metrics VALUES ('p', 9.9)")
    db.commit()
    db.close()

    restore_backed_prediction_rows(db_path, captured)
    db = _connect(case_dir)
    assert db.execute(
        "SELECT ic_mean FROM prediction_metrics WHERE prediction_hash='p'"
    ).fetchone() == (0.5,)
    assert db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0] == 1
    db.close()


def test_capturing_from_a_fixture_with_no_prediction_directories_returns_nothing(case_dir):
    assert capture_backed_prediction_rows(case_dir / "run_log" / "registry.db", case_dir) == {}
