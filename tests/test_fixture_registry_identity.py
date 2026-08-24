"""The seeded registry must be usable by a notebook on the research API.

Before #893 the fixture wrote its own nine-table copy of the registry schema under a
docstring promising it matched `REGISTRY_SCHEMA_SQL` "exactly". Nothing checked that
promise, and it had drifted by thirteen tables and two columns. Every seeded prediction
resolved as `identity_status="legacy"` and `complete=False`, so any rebuilt notebook
filtering on the current identity read an empty catalog and stopped at its first cell -
against a fixture that looked fully populated.

These tests fail against that fixture and pass against the canonical schema.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from case_studies.research.catalog import _frame, _registry_rows
from case_studies.utils.registry.specs import IDENTITY_VERSION
from case_studies.utils.registry.store import REGISTRY_SCHEMA_SQL
from tests.fixtures.seed_results import seed_results

CASE_STUDY = "fx_pairs"


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("seed893")
    seed_results(root, [CASE_STUDY])
    return root / CASE_STUDY


def test_seeded_predictions_are_usable_by_the_research_catalog(seeded: Path) -> None:
    """The end-to-end symptom: a rebuilt notebook resolves rows it can actually use."""
    frame = _frame(_registry_rows(seeded, "workspace"))
    assert frame.height > 0, "the fixture seeded no prediction rows at all"
    assert frame["identity_status"].unique().to_list() == ["current"]
    assert frame.filter(frame["complete"]).height == frame.height, (
        "every seeded row must be complete; a partial row is what made the catalog "
        "look empty to a notebook filtering on it"
    )


def test_the_fixture_registry_carries_every_canonical_table(seeded: Path) -> None:
    """Drift is what caused #893, so the test is on the schema, not on one column."""
    expected = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", REGISTRY_SCHEMA_SQL))
    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        actual = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert expected <= actual, f"missing from the seeded registry: {sorted(expected - actual)}"


def test_seeded_training_runs_follow_the_current_identity_version(seeded: Path) -> None:
    """Reading the constant is what makes the fixture survive the next bump."""
    with sqlite3.connect(seeded / "run_log" / "registry.db") as db:
        versions = {row[0] for row in db.execute("SELECT identity_version FROM training_runs")}
    assert versions == {IDENTITY_VERSION}
