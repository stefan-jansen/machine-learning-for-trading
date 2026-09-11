"""The pre-run gate must check the registry the run will write, and say so when there is none.

`check_supersedes_declarations` used the checker's own default artifacts root,
`~/ml4t/artifacts/case_studies`, which is where a standard checkout's gitignored `run_log/`
symlink points and nowhere else. Under `ML4T_OUTPUT_DIR` - which is how the test suite and
any redirected run resolve their outputs - that reads a different registry than the one the
run registers into, and when the default location holds nothing it reads no registry at all.
The second case is the expensive one: `check_case_study` records a missing registry as a
`no-registry` finding, which is not `stale`, so the gate reported "no dead declaration" for
a check that never ran.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import pre_run_gate

_SCHEMA = """
CREATE TABLE causal_runs (
    causal_hash TEXT PRIMARY KEY,
    label TEXT,
    spec_json TEXT,
    supersedes_hash TEXT
);
CREATE TABLE official_populations (
    population_hash TEXT PRIMARY KEY,
    name TEXT,
    member_kind TEXT,
    snapshot_json TEXT,
    supersedes_hash TEXT,
    created_at TEXT
);
"""

CASE_STUDY = "redirected_case_study"
NOTEBOOK = "06_linear"


def _registry(root: Path, generations: list[tuple[str, str, str | None]]) -> None:
    """`generations` is (hash, name, supersedes_hash), oldest first."""
    run_log = root / CASE_STUDY / "run_log"
    run_log.mkdir(parents=True)
    db = sqlite3.connect(run_log / "registry.db")
    db.executescript(_SCHEMA)
    db.executemany(
        "INSERT INTO official_populations VALUES (?, ?, 'prediction', '{}', ?, '2026-01-01')",
        generations,
    )
    db.commit()
    db.close()


def _notebook(repo: Path, declared: str) -> None:
    directory = repo / "case_studies" / CASE_STUDY
    directory.mkdir(parents=True)
    (directory / f"{NOTEBOOK}.py").write_text(
        '# %% tags=["parameters"]\n'
        f'SUPERSEDES_POPULATION: str = "{declared}"\n'
        "\n# %%\n"
        'population_name = POPULATION_NAME or "universe"\n'
    )


@pytest.fixture
def redirected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    (repo / "case_studies").mkdir(parents=True)
    artifacts.mkdir()
    monkeypatch.setattr(pre_run_gate, "REPO_ROOT", repo)
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(artifacts))
    return repo, artifacts


def _run(case_study: str = CASE_STUDY, notebook: str | None = NOTEBOOK):
    report = pre_run_gate.Report(case_study=case_study, family="linear", label="fwd_1d")
    pre_run_gate.check_supersedes_declarations(report, case_study, notebook)
    assert len(report.checks) == 1
    return report.checks[0]


def test_a_stale_literal_in_the_redirected_registry_is_found(redirected):
    """`gen1` is two generations back, so the head no longer offers it. Only the
    redirected registry holds that chain; the default artifacts root knows nothing of it."""
    repo, artifacts = redirected
    _registry(
        artifacts,
        [
            ("gen1", "universe", None),
            ("gen2", "universe", "gen1"),
            ("gen3", "universe", "gen2"),
        ],
    )
    _notebook(repo, declared="gen1")

    check = _run()

    assert not check.passed
    assert "gen1" in check.detail


def test_a_live_literal_in_the_redirected_registry_passes(redirected):
    """The control: the same wiring reports green when the declaration names the head."""
    repo, artifacts = redirected
    _registry(
        artifacts,
        [
            ("gen1", "universe", None),
            ("gen2", "universe", "gen1"),
            ("gen3", "universe", "gen2"),
        ],
    )
    _notebook(repo, declared="gen3")

    check = _run()

    assert check.passed
    assert "no dead declaration" in check.detail


def test_no_registry_is_reported_rather_than_passed(redirected):
    """A checker with nothing to read must not report the green it did not earn."""
    repo, artifacts = redirected
    _notebook(repo, declared="gen1")

    check = _run()

    assert not check.passed
    assert "no registry" in check.detail
