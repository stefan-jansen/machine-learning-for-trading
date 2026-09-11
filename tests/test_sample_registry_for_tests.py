"""Tests for the registry sampler's output-root guard.

Step 3 of the regeneration path unlinks each destination registry before opening
its source, and a production registry.db is 43-180 MB and gitignored. A wrong
--output therefore destroys the results source of truth with nothing to restore it
from, which is the one failure in this path that cannot be undone by re-running it.
"""

import ast
import sqlite3
from pathlib import Path

from tests.pm_helpers import REPO_ROOT, get_overrides, invocations_for
from tests.sample_registry_for_tests import (
    CASE_STUDY_IDS,
    CODE_CS_DIR,
    DEFAULT_INTERMEDIATES_DIR,
    PINNED_PREDICTION_CONFIGS,
    _populate_sample_db,
    rejected_output_root,
)


def test_the_intended_output_root_is_accepted() -> None:
    assert rejected_output_root(DEFAULT_INTERMEDIATES_DIR) is None


def test_a_case_studies_tree_is_rejected() -> None:
    """Named-directory rule, not path equality: in a worktree CODE_CS_DIR is the
    worktree's own tree while the canonical registries live in ~/ml4t/code."""
    assert rejected_output_root(CODE_CS_DIR) is not None
    assert rejected_output_root(Path.home() / "ml4t" / "code" / "case_studies") is not None
    assert rejected_output_root(CODE_CS_DIR / "etfs") is not None


def test_a_root_resolving_onto_a_source_registry_is_rejected() -> None:
    """The destination for cs_id is <root>/<cs_id>/run_log/registry.db, so a root one
    level above a case-study tree collides with the source even under another name."""
    assert rejected_output_root(CODE_CS_DIR.parent / "case_studies") is not None


def test_a_symlinked_destination_is_rejected(tmp_path: Path) -> None:
    """The worktree setup symlinks each case study's run_log to the canonical one, so
    a destination that only looks separate is the normal case, not an exotic one."""
    root = tmp_path / "intermediates"
    (root / "etfs").mkdir(parents=True)
    (root / "etfs" / "run_log").symlink_to(CODE_CS_DIR / "etfs" / "run_log")

    assert rejected_output_root(root) is not None


def test_a_symlinked_case_study_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "intermediates"
    root.mkdir(parents=True)
    (root / "etfs").symlink_to(CODE_CS_DIR / "etfs")

    assert rejected_output_root(root) is not None


def test_every_case_study_is_covered_by_the_check(tmp_path: Path) -> None:
    """The guard iterates CASE_STUDY_IDS; an empty list would make it vacuous."""
    assert len(CASE_STUDY_IDS) == 9
    assert rejected_output_root(tmp_path) is None


# --- The tables a sampled registry gets rows for ------------------------------
#
# Step 1 copies every table's CREATE statement, so a table nothing copies rows for
# still exists in the fixture - empty. `causal_runs` and `backtest_paired_metrics`
# were in exactly that state across all nine shipped registries: a notebook reading
# either found no rows and could not distinguish "not computed" from "not sampled",
# and the fixture had quietly replaced a populated table with an empty one.


def _source_registry(path: Path) -> sqlite3.Connection:
    """A registry with two backtests, one paired comparison and one causal run."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT, family TEXT);
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
        );
        CREATE TABLE backtest_runs (
            backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT
        );
        CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
        CREATE TABLE backtest_fold_metrics (backtest_hash TEXT, fold INTEGER);
        CREATE TABLE backtest_paired_metrics (
            challenger_hash TEXT NOT NULL, benchmark_hash TEXT NOT NULL,
            sharpe_diff REAL, computed_at TEXT NOT NULL,
            PRIMARY KEY (challenger_hash, benchmark_hash)
        );
        CREATE TABLE causal_runs (
            causal_hash TEXT PRIMARY KEY, label TEXT NOT NULL,
            dml_effect REAL, created_at TEXT NOT NULL
        );
        INSERT INTO training_runs VALUES ('t1', 'fwd_ret_5d', 'linear');
        INSERT INTO prediction_sets VALUES ('p1', 't1', 'validation');
        INSERT INTO backtest_runs VALUES ('bt_challenger', 'p1', 'signal');
        INSERT INTO backtest_runs VALUES ('bt_benchmark', 'p1', 'signal');
        INSERT INTO backtest_metrics VALUES ('bt_challenger', 1.4);
        INSERT INTO backtest_metrics VALUES ('bt_benchmark', 0.9);
        INSERT INTO backtest_paired_metrics
            VALUES ('bt_challenger', 'bt_benchmark', 0.5, '2026-01-01');
        INSERT INTO backtest_paired_metrics
            VALUES ('bt_challenger', 'bt_absent', 0.7, '2026-01-01');
        INSERT INTO causal_runs VALUES ('c1', 'fwd_ret_5d', 0.003, '2026-01-01');
        """
    )
    connection.commit()
    return connection


def _sampled(tmp_path: Path) -> sqlite3.Connection:
    src_path, dst_path = tmp_path / "src.db", tmp_path / "dst.db"
    src = _source_registry(src_path)
    dst = sqlite3.connect(dst_path)
    _populate_sample_db(src, dst, dst_path)
    src.close()
    return dst


def test_causal_runs_reaches_the_sampled_registry(tmp_path: Path) -> None:
    """Keyed by causal_hash and filtered by nothing, so full is the only answer."""
    dst = _sampled(tmp_path)
    assert dst.execute("SELECT causal_hash FROM causal_runs").fetchall() == [("c1",)]


def test_a_paired_comparison_reaches_it_when_both_sides_do(tmp_path: Path) -> None:
    dst = _sampled(tmp_path)
    pairs = dst.execute(
        "SELECT challenger_hash, benchmark_hash FROM backtest_paired_metrics"
    ).fetchall()
    assert ("bt_challenger", "bt_benchmark") in pairs


def test_a_pair_whose_benchmark_was_not_sampled_is_dropped(tmp_path: Path) -> None:
    """benchmark_hash carries no foreign key, so nothing else would drop it, and a
    comparison against a backtest the fixture does not have is not readable."""
    dst = _sampled(tmp_path)
    benchmarks = {
        row[0] for row in dst.execute("SELECT benchmark_hash FROM backtest_paired_metrics")
    }
    assert "bt_absent" not in benchmarks


def test_a_registry_without_these_tables_still_samples(tmp_path: Path) -> None:
    """A stub or an older canonical registry has neither, and that is not an error."""
    src_path, dst_path = tmp_path / "src.db", tmp_path / "dst.db"
    src = sqlite3.connect(src_path)
    src.executescript(
        """
        CREATE TABLE training_runs (training_hash TEXT PRIMARY KEY, label TEXT, family TEXT);
        CREATE TABLE prediction_sets (
            prediction_hash TEXT PRIMARY KEY, training_hash TEXT, split TEXT
        );
        CREATE TABLE backtest_runs (
            backtest_hash TEXT PRIMARY KEY, prediction_hash TEXT, stage TEXT
        );
        CREATE TABLE backtest_metrics (backtest_hash TEXT PRIMARY KEY, sharpe REAL);
        CREATE TABLE backtest_fold_metrics (backtest_hash TEXT, fold INTEGER);
        INSERT INTO training_runs VALUES ('t1', 'fwd_ret_5d', 'linear');
        """
    )
    src.commit()
    dst = sqlite3.connect(dst_path)

    stats = _populate_sample_db(src, dst, dst_path)

    assert stats["causal_runs"] == 0
    assert stats["status"] == "OK"


def _pinned_parameters_of(notebook_py: Path) -> tuple[str | None, dict[str, str]]:
    """A notebook's CASE_STUDY_ID and the ``*_CONFIG`` names its parameters cell assigns.

    Both are read as module-level string constants, which is what the parameters cell of
    a percent-format notebook compiles to and what papermill replaces at run time.
    """
    tree = ast.parse(notebook_py.read_text(encoding="utf-8"))
    case_study_id: str | None = None
    configs: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        if target.id == "CASE_STUDY_ID":
            case_study_id = value.value
        elif target.id.endswith("_CONFIG"):
            configs[target.id] = value.value
    return case_study_id, configs


def test_every_config_a_pinned_notebook_runs_with_is_declared_for_sampling() -> None:
    """A configuration the sampler does not carry fails only in CI, halfway through a run.

    PINNED_PREDICTION_CONFIGS is what puts a configuration's predictions parquet in the
    fixture. Nothing couples it to what the notebooks actually ask for: _resolve_pinned_hashes
    checks the declaration against production, where every configuration exists, so a name
    the declaration omits resolves a registry row with no artifact behind it and the notebook
    raises mid-run. That has happened once already, to
    26_mlops_governance/02_online_drift_detection.

    Both sources of a name are checked: the ``*_CONFIG`` defaults in the notebook's own
    parameters cell, and whatever `tests/overrides.yaml` replaces them with. A default is
    checked even where an override currently replaces it, because the declaration is meant
    to carry it - that is what lets an override be deleted once a regenerated fixture holds
    every pinned configuration. Runs are enumerated through
    ``invocations_for`` rather than by reading ``parameters``, so an entry moved to the
    ``invocations`` shape is validated rather than silently dropped - and a dropped entry
    looks exactly like a passing one.
    """
    checked: list[str] = []
    for notebook_py in sorted(REPO_ROOT.rglob("*.py")):
        if any(part.startswith(".") for part in notebook_py.parts):
            continue
        text = notebook_py.read_text(encoding="utf-8")
        if "CASE_STUDY_ID" not in text:
            continue
        case_study_id, defaults = _pinned_parameters_of(notebook_py)
        declared = PINNED_PREDICTION_CONFIGS.get(case_study_id or "")
        if not declared:
            continue

        key = notebook_py.relative_to(REPO_ROOT).with_suffix("").as_posix()
        names = {config for _family, _label, config, _split in declared}
        for run in invocations_for(get_overrides(key), key=key):
            named = set(defaults.items()) | {
                (name, value)
                for name, value in run.parameters.items()
                if name.endswith("_CONFIG") and isinstance(value, str)
            }
            if not named:
                continue
            where = f"{key} [{run.id}]" if run.id else key
            missing = {f"{name}={value}" for name, value in named if value not in names}
            assert not missing, (
                f"{where} names a configuration the fixture is not built to carry: "
                f"{sorted(missing)}. Add it to PINNED_PREDICTION_CONFIGS[{case_study_id!r}] "
                "in tests/sample_registry_for_tests.py, or name one already there."
            )
            checked.append(where)

    assert checked, (
        "no notebook names a *_CONFIG for a case study in PINNED_PREDICTION_CONFIGS, "
        "so this check covered nothing"
    )
