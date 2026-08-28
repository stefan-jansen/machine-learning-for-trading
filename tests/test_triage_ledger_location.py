"""The chapter-20 reader and the stage-05 writer resolve one path.

``case_studies/<cs>/05_evaluation`` writes ``evaluation/triage_ledger.parquet`` through
``get_case_study_dir``, which honours ``ML4T_OUTPUT_DIR``. Chapter 20 read it from ``REPO_ROOT``,
which does not - so in every CI job the fixture was seeded at
``/tmp/ml4t-test-output/<cs>/evaluation/`` and the reader looked under the checkout. ``ch18-20``
reported "No triage ledger for [...]" while all nine ledgers were present in the test-data repo.

The bug was invisible locally, where ``ML4T_OUTPUT_DIR`` is unset and the two paths coincide, so
the test that matters is the one that sets it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
import pytest


def _reloaded_analytics():
    import utils.paths

    importlib.reload(utils.paths)
    import case_studies.utils.analytics as analytics

    return importlib.reload(analytics)


@pytest.fixture
def restore_modules():
    yield
    _reloaded_analytics()


def test_the_ledger_is_read_from_the_output_root_the_writer_wrote_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    analytics = _reloaded_analytics()

    seeded = tmp_path / "etfs" / "evaluation"
    seeded.mkdir(parents=True)
    pl.DataFrame({"feature": ["mom_21"], "verdict": ["advance"]}).write_parquet(
        seeded / "triage_ledger.parquet"
    )

    assert analytics.triage_ledger_path("etfs") == seeded / "triage_ledger.parquet"

    ledger = analytics.load_triage_ledger("etfs")
    assert ledger is not None, "the reader did not find the ledger the writer's path produced"
    assert ledger.get_column("case_study").to_list() == ["etfs"]
    assert ledger.get_column("feature").to_list() == ["mom_21"]


def test_a_repo_relative_path_would_not_have_found_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    """The failure direction, stated as the thing the old code did.

    Without this, the test above passes on a reader that happens to resolve to the same place for
    another reason. This pins that the redirect actually moves the path off the checkout.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    analytics = _reloaded_analytics()

    from utils.paths import REPO_ROOT

    resolved = analytics.triage_ledger_path("etfs")
    old_path = REPO_ROOT / "case_studies" / "etfs" / "evaluation" / "triage_ledger.parquet"

    assert resolved != old_path
    assert not resolved.is_relative_to(REPO_ROOT)


def test_an_absent_ledger_reads_as_absent_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_modules: None
) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    analytics = _reloaded_analytics()

    assert analytics.load_triage_ledger("etfs") is None
