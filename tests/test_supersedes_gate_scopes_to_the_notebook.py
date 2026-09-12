"""The launch gate must refuse on the declarations the run reads, not on its siblings'.

`nb-run.sh` calls this checker with `--case-study` alone, so every literal the case study
commits decided the exit status of every notebook's launch. On 2026-09-11 that refused
`fx_pairs/13_backtest` - whose own two declarations had just been swept to the sentinel and
which reads nothing else - by naming eight refused literals in seven sibling notebooks it
never touches. The only way through was `NB_RUN_ALLOW_STALE_SUPERSEDES=1`, which also
switches off the check on the notebook's OWN declarations. A gate whose false refusals are
cleared by a flag that disables the true ones teaches the flag, and the flag is then
load-bearing for every run of that case study.

A sibling's stale literal is still worth saying out loud - it refuses the run that freezes
it, and the operator queueing a chain is the one person for whom fixing it is free - so it
is reported, and only the exit status is scoped.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import check_supersedes_literals as checker

_SCHEMA = """
CREATE TABLE official_populations (
    population_hash TEXT PRIMARY KEY,
    name TEXT,
    member_kind TEXT,
    snapshot_json TEXT,
    supersedes_hash TEXT,
    created_at TEXT
);
"""

CASE_STUDY = "scoped_case_study"


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """One case study, one live lineage two generations deep, and two notebooks."""
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    directory = repo / "case_studies" / CASE_STUDY
    directory.mkdir(parents=True)
    run_log = artifacts / CASE_STUDY / "run_log"
    run_log.mkdir(parents=True)

    db = sqlite3.connect(run_log / "registry.db")
    db.executescript(_SCHEMA)
    db.executemany(
        "INSERT INTO official_populations VALUES (?, ?, 'prediction', '{}', ?, '2026-01-01')",
        [("gen1", "universe", None), ("gen2", "universe", "gen1")],
    )
    db.commit()
    db.close()

    monkeypatch.setattr(checker, "REPO_ROOT", repo)
    return repo, artifacts


def _notebook(repo: Path, stem: str, declared: str) -> None:
    (repo / "case_studies" / CASE_STUDY / f"{stem}.py").write_text(
        '# %% tags=["parameters"]\n'
        f'SUPERSEDES_POPULATION: str = "{declared}"\n'
        "\n# %%\n"
        'population_name = POPULATION_NAME or "universe"\n'
    )


def _run(artifacts: Path, *extra: str) -> int:
    return checker.main(["--case-study", CASE_STUDY, "--artifacts-root", str(artifacts), *extra])


def test_a_siblings_stale_literal_does_not_refuse_this_notebook(corpus, capsys):
    """The reproduction. `06_clean` names the head; `07_stale` names the generation it
    replaced. Launching `06_clean` must not be refused for `07_stale`'s declaration, which
    `06_clean` never reads."""
    repo, artifacts = corpus
    _notebook(repo, "06_clean", declared="live")
    _notebook(repo, "07_stale", declared="gen1")

    assert _run(artifacts, "--notebook", "06_clean") == 0

    err = capsys.readouterr().err
    assert "07_stale" in err, "the sibling is still reported, just not fatal"
    assert "does not block this run" in err


def test_the_notebooks_own_stale_literal_still_refuses(corpus):
    """The negative selftest. Scoping must not make the gate unable to refuse: the same
    literal, in the notebook being launched, still exits non-zero."""
    repo, artifacts = corpus
    _notebook(repo, "06_clean", declared="live")
    _notebook(repo, "07_stale", declared="gen1")

    assert _run(artifacts, "--notebook", "07_stale") == 1


def test_without_the_flag_every_notebook_still_decides(corpus):
    """The control for the control. `--case-study` alone keeps the corpus-wide behaviour the
    scan and CI depend on, so scoping is something the launch path asks for rather than a
    change to what the checker means."""
    repo, artifacts = corpus
    _notebook(repo, "06_clean", declared="live")
    _notebook(repo, "07_stale", declared="gen1")

    assert _run(artifacts) == 1


def test_a_clean_case_study_passes_either_way(corpus):
    """No refusal anywhere: scoped and unscoped must agree, or the scoping is doing something
    besides narrowing."""
    repo, artifacts = corpus
    _notebook(repo, "06_clean", declared="live")
    _notebook(repo, "07_also_clean", declared="live")

    assert _run(artifacts) == 0
    assert _run(artifacts, "--notebook", "06_clean") == 0


def test_an_unknown_notebook_refuses_rather_than_passing_vacuously(corpus):
    """A typo'd stem must not read as "nothing to check". Scoping to a notebook the case study
    does not hold leaves every finding non-fatal, which is the shape of the defect this whole
    checker exists to catch: a check that silently applies to nothing."""
    repo, artifacts = corpus
    _notebook(repo, "06_clean", declared="live")
    _notebook(repo, "07_stale", declared="gen1")

    with pytest.raises(SystemExit) as exit_info:
        _run(artifacts, "--notebook", "99_not_here")
    assert exit_info.value.code == 2
