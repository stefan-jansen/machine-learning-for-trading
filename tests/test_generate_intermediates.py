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


def test_seed_configs_refreshes_shared_presets_over_an_existing_tree(tmp_path: Path) -> None:
    """Every preset a seeded case study names has to resolve after seeding.

    The shared preset copy used to be guarded on ``not dst.exists()``, so it ran once
    on a tree that had no ``config/`` and never again. A preset added or edited after
    that first generation never reached the fixture, and the notebook that named it
    failed with ``Preset not found`` against a file that had been in the repo since the
    initial release - which is how sp500_options/06_linear went down on 2026-09-06.

    Written against the contract rather than the copy: seed over a tree that already
    holds a stale ``config/``, then require that every preset named by a seeded case
    study's training menus exists. That fails on the old guard and cannot be satisfied
    by any implementation that leaves a preset behind.
    """
    stale = tmp_path / "config" / "lasso"
    stale.mkdir(parents=True)
    (stale / "lasso_a0.1.yaml").write_text("stale: true\n")

    seed_configs(tmp_path, ["sp500_options"])

    named: set[str] = set()
    for menu in (tmp_path / "sp500_options" / "config" / "training").glob("*.yaml"):
        for line in menu.read_text().splitlines():
            entry = line.strip()
            if entry.startswith("- "):
                named.add(entry[2:].strip().strip("\"'"))

    assert named, "no presets named by sp500_options' training menus"
    available = {p.stem for p in (tmp_path / "config").rglob("*.yaml")}
    assert named <= available, f"presets named but not seeded: {sorted(named - available)}"
    assert "stale" not in (stale / "lasso_a0.1.yaml").read_text()
