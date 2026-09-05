"""What tests/generate_intermediates.py treats as a completed unit of work.

Two ways this script used to report success it had not earned, both of which left
a stale fixture looking freshly built:

- a pipeline stage (01-05) marked ``skip`` in ``tests/overrides.yaml`` cascaded to
  every later stage of that case study, and skips were not counted as failures, so
  the run exited 0 having generated nothing for those stages;
- a ``--case-studies`` value naming nothing never entered ``results``, so the
  failure count stayed zero and a typo produced a green run.

The generator and the timed CI job read the same ``skip`` key and want different
answers from it: the job is protecting a time budget, generation is producing the
artifact that job consumes. ``--ignore-skips`` is that second answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.generate_intermediates import (  # noqa: E402
    CASE_STUDIES,
    FAILED,
    INCOMPLETE,
    NOT_RUN,
    OK,
    SKIPPED,
    exit_code,
    resolve_case_studies,
    seed_configs,
)

# --- requested names ---------------------------------------------------------


def test_resolve_case_studies_returns_the_request_when_every_name_is_known() -> None:
    assert resolve_case_studies(["etfs", "cme_futures"]) == ["etfs", "cme_futures"]


def test_resolve_case_studies_rejects_a_name_that_matches_nothing() -> None:
    with pytest.raises(ValueError, match="cme_future"):
        resolve_case_studies(["etfs", "cme_future"])


def test_resolve_case_studies_names_the_supported_list_in_the_error() -> None:
    with pytest.raises(ValueError, match="us_equities_panel"):
        resolve_case_studies(["typo"])


# --- exit status -------------------------------------------------------------


def test_a_run_where_everything_ran_exits_zero() -> None:
    assert exit_code({"etfs::01_x": OK, "etfs::02_y": OK}) == 0


def test_a_stage_skipped_above_the_pipeline_does_not_fail_the_run() -> None:
    """Stage 17 is a holdout notebook: skipping it leaves nothing else unbuilt."""
    assert exit_code({"etfs::01_x": OK, "etfs::17_holdout": SKIPPED}) == 0


def test_a_failed_stage_exits_nonzero() -> None:
    assert exit_code({"etfs::01_x": FAILED, "etfs::02_y": NOT_RUN}) == 1


def test_an_incomplete_unit_exits_nonzero() -> None:
    """The regression this file exists for: a cascading skip is not a success."""
    assert exit_code({"cme::04_model_based_features": INCOMPLETE, "cme::05_eval": NOT_RUN}) == 1


def test_stages_not_run_behind_an_incomplete_one_do_not_by_themselves_fail() -> None:
    assert exit_code({"etfs::05_eval": NOT_RUN}) == 0


# --- config seeding scope ----------------------------------------------------


def test_seed_configs_writes_only_the_case_studies_it_was_given(tmp_path: Path) -> None:
    """A scoped regeneration must not rewrite eight other case studies' configs."""
    seed_configs(tmp_path, ["cme_futures"])

    seeded = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert "cme_futures" in seeded
    assert seeded & set(CASE_STUDIES) == {"cme_futures"}


def test_seed_configs_defaults_to_every_case_study(tmp_path: Path) -> None:
    seed_configs(tmp_path)

    seeded = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert set(CASE_STUDIES) <= seeded
