"""A seeded registry must be usable by a notebook on the research API.

Two defects, one behind the other, both invisible.

The seeder wrote its own nine-table copy of the registry DDL under a docstring
promising it matched `REGISTRY_SCHEMA_SQL` "exactly". Nothing checked that promise and
it had drifted by thirteen tables and two columns.

Fixing that alone changed nothing on the path the suite takes, because `conftest` copies
a registry out of `test-data` before `seed_results` runs, and `_seed_registry_db` then
returned early on a five-column "fully current schema" check. All nine shipped
registries satisfy those five columns while missing the same thirteen tables, so the
corrected DDL was never reached.

**These tests therefore exercise the copied registry, not a fresh one.** A test that
seeds into an empty directory passes against both the broken and the fixed seeder, which
is how the first fix was measured as working when it was not.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from case_studies.research.catalog import _frame, _registry_rows
from case_studies.utils.registry.specs import IDENTITY_VERSION
from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL
from tests.fixtures.seed_results import seed_results

CASE_STUDY = "fx_pairs"
SHIPPED = Path.home() / "ml4t" / "test-data" / "intermediates"


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    """Reproduce conftest's order: copy the shipped intermediates, then seed."""
    src = SHIPPED / CASE_STUDY
    if not (src / "run_log" / "registry.db").is_file():
        pytest.skip(f"no shipped registry at {src}")
    root = tmp_path_factory.mktemp("seed893")
    shutil.copytree(src, root / CASE_STUDY)
    seed_results(root, [CASE_STUDY])
    return root / CASE_STUDY


def test_the_shipped_registry_is_the_one_under_test(seeded: Path) -> None:
    """Guard the guard: if this stops being the copied registry, the rest proves nothing."""
    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        n_folds = db.execute("SELECT COUNT(*) FROM fold_metrics").fetchone()[0]
    # fold_metrics is what separates the two registries. The shipped fx_pairs one holds
    # 2099 rows; a fresh seed writes exactly two per prediction, so a few dozen. Row
    # counts in training_runs do not separate them - the shipped registry has 19 - which
    # is why this asserts on the table that does.
    assert n_folds > 1000, (
        f"only {n_folds} fold_metrics rows: this is a freshly seeded registry, not the "
        "one shipped in test-data, so these tests are measuring the wrong path"
    )


def test_seeded_predictions_are_usable_by_the_research_catalog(seeded: Path) -> None:
    """The end-to-end symptom a rebuilt notebook hits at its first executable cell."""
    frame = _frame(_registry_rows(seeded, "workspace"))
    assert frame.height > 0, "the registry resolved no prediction rows at all"
    assert frame["identity_status"].unique().to_list() == ["current"]
    assert frame.filter(frame["complete"]).height == frame.height, (
        "every row must be complete; a partial row is what made the catalog look empty"
    )


def test_the_registry_carries_every_canonical_table(seeded: Path) -> None:
    """Drift caused this, so the assertion is on the schema, not on one column."""
    expected = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", REGISTRY_SCHEMA_SQL))
    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        actual = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert expected <= actual, f"missing: {sorted(expected - actual)}"


def test_training_runs_follow_the_current_identity_version(seeded: Path) -> None:
    """Reading the constant is what makes the fixture survive the next bump."""
    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        versions = {r[0] for r in db.execute("SELECT identity_version FROM training_runs")}
    assert versions == {IDENTITY_VERSION}


def test_declining_the_preview_tier_suppresses_the_injection(tmp_path) -> None:
    """`research_preview=False` must suppress the tier injection, not merely be accepted.

    This covers the helper contract only. It passes with or without the two call sites
    that read `overrides["research_preview"]`, which was true when it was first written
    under a name claiming otherwise - the same vacuity this branch exists to remove. The
    call sites are a one-line pass-through whose effect shows up as fx_pairs 13-16
    executing; there is no seam to assert it at without running a notebook.

    The harness injects the preview tier for any notebook declaring both
    `EXECUTION_TIER` and `WORKSPACE`. That pair means "this runs self-contained at
    reduced scale", which is true of a notebook that WRITES at the tier it is handed
    and false of one that only READS at it: the reader is self-contained only if its
    producer ran into the same workspace, and the per-notebook suite runs each
    notebook alone. Without the opt-out such a notebook filters for preview rows
    nothing wrote and stops at its first cell.
    """
    from tests.pm_helpers import injected_parameters

    py_path = tmp_path / "13_backtest.py"
    py_path.write_text(
        '# %% tags=["parameters"]\n'
        'CASE_STUDY_ID = "fx_pairs"\n'
        'EXECUTION_TIER = "canonical"\n'
        'WORKSPACE: str = ""\n'
        "\n"
        "# %%\n"
        "print(EXECUTION_TIER)\n"
    )

    injected = injected_parameters(py_path, {}, tmp_path, research_preview=True)
    assert injected["EXECUTION_TIER"] == "preview", (
        "the pair must still opt a producer into the preview tier; if this stops "
        "holding, the opt-out below is measuring nothing"
    )

    declined = injected_parameters(py_path, {}, tmp_path, research_preview=False) or {}
    assert "EXECUTION_TIER" not in declined, "declining must leave the declared tier alone"
    # But it must NOT decline the isolated workspace. Leaving WORKSPACE empty sends
    # open_study down the workspace=None branch into Study.regenerate, which is the
    # in-place production path: in a maintainer worktree that writes the published
    # registry. Measured 2026-08-24 - a test run put 13 official populations, 1 candidate
    # set and 269 backtest runs into the real fx_pairs registry and froze one population
    # incomplete. The harness must not be able to reach that path at all.
    assert declined.get("WORKSPACE") == str(tmp_path.resolve()), (
        "declining the preview tier must still isolate the workspace, or the run writes "
        "to the maintainer's published artifacts"
    )


def test_every_prediction_has_a_coverage_row(seeded: Path) -> None:
    """The count, because `INSERT OR IGNORE` turns a constraint violation into silence.

    `prediction_coverage.artifact_digest` is `TEXT NOT NULL`. An earlier version of the
    seeder inserted NULL there intending to fill it later; under `INSERT OR IGNORE` every
    row was dropped without error, and the catalog test above still passed because it
    asserts on the rows that exist rather than on how many there are.
    """
    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        n_pred = db.execute("SELECT COUNT(*) FROM prediction_sets").fetchone()[0]
        n_cov = db.execute("SELECT COUNT(*) FROM prediction_coverage").fetchone()[0]
    assert n_pred > 0
    assert n_cov == n_pred, f"{n_pred} predictions but {n_cov} coverage rows"


def test_recorded_artifact_digests_match_the_artifacts(seeded: Path) -> None:
    """`PredictionResult.complete` is stricter than the catalog's `complete` column.

    The catalog reads `coverage.status` and stops; `results.py:419-423` additionally
    verifies `value_digest` of the parquet against the recorded digest. The fixture used to
    record a fabricated 12-character `_make_hash` value where `value_digest` produces 16, so
    no seeded prediction could satisfy the stricter check and a population would freeze
    cleanly from the catalog and then fail its own `require_complete`.
    """
    import polars as pl

    from case_studies.utils.artifact_digest import value_digest

    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        rows = db.execute(
            "SELECT prediction_hash, artifact_digest FROM prediction_coverage"
        ).fetchall()

    checked = 0
    for p_hash, recorded in rows:
        artifact = seeded / "run_log" / "predictions" / p_hash / "predictions.parquet"
        if not artifact.is_file():
            continue
        assert recorded, f"{p_hash} has an artifact but no recorded digest"
        assert value_digest(pl.read_parquet(artifact)) == recorded, (
            f"{p_hash}: recorded {recorded!r} does not describe the artifact on disk"
        )
        checked += 1
    assert checked > 0, "no seeded prediction had an artifact to check"
