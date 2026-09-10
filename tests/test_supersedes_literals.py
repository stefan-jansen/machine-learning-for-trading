"""The check that a committed supersedes literal still names a live generation.

Nothing that runs before a production chain can see a stale one. `supersedes_for_run`
returns `None` for any tier but `canonical`, so a preview never resolves a supersedes hash
and never reaches the lineage check registration enforces - and that guard is right: a
preview is discarded with its workspace and has no lineage to extend. On 2026-09-06 twelve
`sp500_options` notebooks passed smoke and three failed the real chain fifteen minutes later
on eleven stale literals.

**CI cannot see it either, and these tests are built around that rather than pretending
otherwise.** `run_log/` is gitignored and no registry is tracked, so a checkout has nothing
to check against. A corpus scan here would skip on every CI run forever, which is the same
absence-reading-as-a-pass that the defect itself is made of. So the logic is tested against
registries these tests build, which runs everywhere, and the corpus scan is a separate test
that skips out loud when there is no registry to read.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from case_studies.utils.registry.specs import IDENTITY_VERSION
from scripts.check_supersedes_literals import check_all, check_case_study

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


def _registry(root: Path, case_study: str, generations: list[tuple[str, str, str | None]]) -> None:
    """`generations` is (hash, name, supersedes_hash), oldest first."""
    run_log = root / case_study / "run_log"
    run_log.mkdir(parents=True)
    db = sqlite3.connect(run_log / "registry.db")
    db.executescript(_SCHEMA)
    db.executemany(
        "INSERT INTO official_populations VALUES (?, ?, 'prediction', '{}', ?, '2026-01-01')",
        [(h, n, s) for h, n, s in generations],
    )
    db.commit()
    db.close()


def _notebook(repo: Path, case_study: str, stem: str, declared: str, name: str) -> None:
    directory = repo / "case_studies" / case_study
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.py").write_text(
        '# %% tags=["parameters"]\n'
        f'SUPERSEDES_POPULATION: str = "{declared}"\n'
        "\n# %%\n"
        f'population_name = POPULATION_NAME or "{name}"\n'
    )


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    (repo / "case_studies").mkdir(parents=True)
    artifacts.mkdir()
    return repo, artifacts


def test_a_literal_naming_the_tip_is_live(tree):
    """The refit: declaring the tip publishes the next generation over it."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["live"]


def test_a_literal_naming_what_the_tip_replaced_is_live(tree):
    """The re-run: the generation in force is the one this declaration produced."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None), ("bbbb", "etfs-gbm-v1", "aaaa")])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["live"]


def test_a_literal_two_generations_behind_is_stale(tree):
    """The failure: neither the tip nor what the tip replaced."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")
    _registry(
        artifacts,
        "etfs",
        [
            ("aaaa", "etfs-gbm-v1", None),
            ("bbbb", "etfs-gbm-v1", "aaaa"),
            ("cccc", "etfs-gbm-v1", "bbbb"),
        ],
    )

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].is_stale
    assert "cccc" in findings[0].detail


def test_a_literal_naming_a_hash_the_registry_lost_is_stale(tree):
    """What a reset leaves behind, and the shape all six live cases take."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "gone", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].is_stale
    assert "no lineage the registry holds" in findings[0].detail


def test_the_check_is_keyed_on_the_hash_not_the_population_name(tree):
    """The reason this is not a name lookup, pinned so it cannot be simplified back.

    Only some notebooks build their name as `POPULATION_NAME or "a-literal"`; the rest use an
    f-string or a module constant, which no static read resolves. A name-keyed scan skips
    those and reports green - on the real corpus that was 8 of 17 literals, including 3 of
    the 6 stale ones. Keying on the hash reads a notebook whose name is unreadable.
    """
    repo, artifacts = tree
    directory = repo / "case_studies" / "fx_pairs"
    directory.mkdir(parents=True)
    (directory / "08_tabular_dl.py").write_text(
        'SUPERSEDES_POPULATION: str = "gone"\n'
        'population_name = POPULATION_NAME or f"{CASE_STUDY_ID}:{label}:tabular_dl"\n'
    )
    _registry(artifacts, "fx_pairs", [("aaaa", "fx:whatever:tabular_dl", None)])

    findings = check_case_study("fx_pairs", repo_root=repo, artifacts_root=artifacts)

    assert len(findings) == 1, "a notebook whose population name is an f-string was skipped"
    # `unresolved`, not `stale`: with the name unreadable and the hash in no lineage, this
    # is either a dead literal or one waiting for a first publication, and the difference
    # cannot be told from here. What matters is that the declaration was READ at all.
    assert findings[0].status == "unresolved"


def test_an_empty_registry_does_not_make_every_declaration_stale(tree):
    """A reset registry publishes generation one; failing there would block valid runs.

    `population_supersedes` withholds a hash it cannot place and `create` then publishes the
    first generation without a predecessor. So an absent hash is only wrong when a generation
    ALREADY exists under the name this notebook publishes under - merely creating the
    registry must not flip this check from passing to blocking.
    """
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert not any(f.is_stale for f in findings), "an empty registry was read as staleness"
    assert findings[0].status == "unresolved"


def test_an_empty_literal_is_not_a_finding(tree):
    """Most notebooks have never superseded anything. That is the ordinary state."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "06_linear", "", "etfs-linear-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-linear-v1", None)])

    assert check_case_study("etfs", repo_root=repo, artifacts_root=artifacts) == []


def test_no_registry_is_reported_and_does_not_fail(tree):
    """A reader's clone. `run_log/` is gitignored, so this is the ordinary case off-machine."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["no-registry"]
    assert not any(f.is_stale for f in findings)


def test_a_forked_lineage_is_reported_rather_than_guessed_at(tree):
    """Two snapshots nothing supersedes has no defensible answer."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None), ("bbbb", "etfs-gbm-v1", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["forked"]


def test_exit_status_is_one_when_anything_is_stale(tree):
    """The gate's contract: a queueing step reads the status, not the prose."""
    from scripts.check_supersedes_literals import main

    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "gone", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None)])

    import scripts.check_supersedes_literals as module

    original = module.REPO_ROOT
    module.REPO_ROOT = repo
    try:
        assert main(["--artifacts-root", str(artifacts)]) == 1
        _notebook(repo, "etfs", "07_gbm", "aaaa", "etfs-gbm-v1")
        assert main(["--artifacts-root", str(artifacts)]) == 0
    finally:
        module.REPO_ROOT = original


def test_the_scan_reads_the_real_corpus_without_falling_over():
    """The scanner against real registries, which the synthetic fixtures cannot stand in for.

    It deliberately does NOT assert the corpus is clean. Six literals are stale as this is
    written, and leaving them is the decision rather than an oversight: editing one makes the
    paired `.ipynb` stale, and the only sanctioned way to commit that drops its outputs. That
    costs a live render in a `done` case study for a fault that bites nothing until those
    notebooks next execute - at which point the run restores the render for free. So the
    corpus is corrected by `scripts/check_supersedes_literals.py` refusing the chain at
    queue time, not by a red test somebody learns to scroll past.

    What this pins is that the scan runs on real data and every finding is classified. A
    scanner that crashed on a real schema, or invented a status, would pass every synthetic
    test above and fail here.
    """
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_root = Path.home() / "ml4t" / "artifacts" / "case_studies"
    if not artifacts_root.is_dir():
        pytest.skip(f"no registries at {artifacts_root}; run_log/ is gitignored")

    findings = check_all(repo_root=repo_root, artifacts_root=artifacts_root)
    if not any(f.status != "no-registry" for f in findings):
        pytest.skip("registries present but none holds an official_populations table")

    known = {"live", "stale", "superseded", "unresolved", "forked", "no-registry"}
    unclassified = [f for f in findings if f.status not in known]
    assert not unclassified, f"unrecognised status: {unclassified}"
    assert all(f.detail for f in findings), "a finding with no explanation is not usable"


def _causal_notebook(repo: Path, case_study: str, stem: str, declared: str) -> None:
    """A bare hash is written double-quoted; a JSON map can only be single-quoted."""
    directory = repo / "case_studies" / case_study
    directory.mkdir(parents=True, exist_ok=True)
    line = (
        f"SUPERSEDES_CAUSAL: str = '{declared}'"
        if declared.startswith("{")
        else f'SUPERSEDES_CAUSAL: str = "{declared}"'
    )
    (directory / f"{stem}.py").write_text(line + "\n")


def _causal_rows(
    root: Path,
    case_study: str,
    rows: list[tuple[str, str, str | None]],
    *,
    identity_version: int = IDENTITY_VERSION,
    tier: str = "canonical",
) -> None:
    """`rows` is (hash, label, supersedes_hash).

    `spec_json` is not decoration: `current_causal_identities` reads `identity_version` and
    `execution_tier` out of it and excludes a row carrying either wrong. A fixture without it
    would let this test pass against a checker that ignores both.
    """
    db = sqlite3.connect(root / case_study / "run_log" / "registry.db")
    db.executemany(
        "INSERT INTO causal_runs VALUES (?, ?, ?, ?)",
        [
            (
                h,
                label,
                json.dumps({"identity_version": identity_version, "execution_tier": tier}),
                sup,
            )
            for h, label, sup in rows
        ],
    )
    db.commit()
    db.close()


def test_a_current_causal_identity_is_live(tree):
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", "aaaa")
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["live"]


def test_a_retired_causal_identity_is_reported_but_does_not_fail(tree):
    """Withheld is not automatically wrong, and this is the case that proves it.

    A notebook that already published its successor and was left unchanged still declares
    the predecessor. `causal_supersedes` withholds it, the runner resolves to the cached
    successor, and nothing fails. Failing here would send that author to declare a hash
    which already records theirs as its own predecessor.
    """
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", "aaaa")
    _registry(artifacts, "etfs", [])
    _causal_rows(
        artifacts, "etfs", [("aaaa", "fwd_ret_21d", None), ("bbbb", "fwd_ret_21d", "aaaa")]
    )

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].status == "superseded"
    assert not findings[0].is_stale, "an unchanged re-run must not be reported as a failure"


def test_a_bare_causal_hash_the_registry_lost_is_unresolved(tree):
    """A bare declaration names no label, so the checker cannot say which it is.

    `etfs/12_causal_dml` is this shape: the hash is gone, but without the label there is no
    way to tell a dead literal from one waiting on a first publication, and refusing would
    block a valid first run.
    """
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", "gone")
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].status == "unresolved"
    assert not findings[0].is_stale


def test_a_per_label_causal_hash_is_stale_when_that_label_already_resolves(tree):
    """The case that IS provable: the label is named and already has a current identity."""
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", '{"fwd_ret_21d": "gone"}')
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].is_stale
    assert "placebo refit" in findings[0].detail
    assert findings[0].label == "fwd_ret_21d"
    assert findings[0].remedy == "aaaa"


def test_an_empty_causal_table_does_not_make_a_declaration_stale(tree):
    """A reset registry accepts the first identity; refusing would block a valid run."""
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", '{"fwd_ret_21d": "gone"}')
    _registry(artifacts, "etfs", [])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["unresolved"]


def test_a_per_label_causal_declaration_is_read(tree):
    """A notebook fitting several labels declares a JSON object, not a bare hash."""
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", '{"fwd_ret_21d": "aaaa", "fwd_ret_5d": "gone"}')
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert len(findings) == 2, "a per-label declaration must be read for every label it names"
    # `fwd_ret_5d` has no rows at all, so its missing hash is unresolved rather than stale.
    assert sorted(f.status for f in findings) == ["live", "unresolved"]


def test_an_outdated_identity_version_is_not_current(tree):
    """`current_causal_identities` excludes it, so approving it would be a false green."""
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", "aaaa")
    _registry(artifacts, "etfs", [])
    _causal_rows(
        artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)], identity_version=IDENTITY_VERSION - 1
    )

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].status == "superseded", "an outdated identity was treated as current"


def test_a_preview_tier_row_is_not_current(tree):
    """The other filter the resolver applies, and the other way to approve a dead hash."""
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", "aaaa")
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)], tier="preview")

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].status == "superseded", "a preview row was treated as canonical"


def test_an_annotation_of_str_or_none_is_read(tree):
    """`cme_futures/09_dl_lstm.py` writes `str | None`; a `: str =` pattern misses it."""
    repo, artifacts = tree
    directory = repo / "case_studies" / "cme_futures"
    directory.mkdir(parents=True)
    (directory / "09_dl_lstm.py").write_text(
        'SUPERSEDES_POPULATION: str | None = "gone"\n'
        'population_name = POPULATION_NAME or "cme-lstm-v1"\n'
    )
    _registry(artifacts, "cme_futures", [("aaaa", "cme-lstm-v1", None)])

    findings = check_case_study("cme_futures", repo_root=repo, artifacts_root=artifacts)

    assert len(findings) == 1, "a `str | None` annotation was skipped"
    assert findings[0].is_stale


def test_a_parenthesised_multiline_declaration_is_read(tree):
    """`fx_pairs/11_causal_dml.py` wraps its JSON in parentheses across two lines."""
    repo, artifacts = tree
    directory = repo / "case_studies" / "fx_pairs"
    directory.mkdir(parents=True)
    (directory / "11_causal_dml.py").write_text(
        'SUPERSEDES_CAUSAL: str = (\n    \'{"fwd_ret_1d": "aaaa", "fwd_ret_5d": "gone"}\'\n)\n'
    )
    _registry(artifacts, "fx_pairs", [])
    _causal_rows(artifacts, "fx_pairs", [("aaaa", "fwd_ret_1d", None)])

    findings = check_case_study("fx_pairs", repo_root=repo, artifacts_root=artifacts)

    assert len(findings) == 2, "a parenthesised multi-line declaration was skipped"
    assert sorted(f.status for f in findings) == ["live", "unresolved"]


def _exit_status(repo: Path, artifacts: Path, *argv: str) -> int:
    import scripts.check_supersedes_literals as module

    original = module.REPO_ROOT
    module.REPO_ROOT = repo
    try:
        return module.main(["--artifacts-root", str(artifacts), *argv])
    finally:
        module.REPO_ROOT = original


def test_the_named_flag_waives_a_stale_literal(tree):
    """For a run whose membership is unchanged, where the literal is never read.

    Named after what it waives rather than `--force`, because a general force flag becomes
    the way people get past every check in the script and the gate turns into a warning
    wearing a different hat.
    """
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "gone", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None)])

    assert _exit_status(repo, artifacts) == 1
    assert _exit_status(repo, artifacts, "--allow-stale-supersedes") == 0


def test_an_unresolved_declaration_never_blocks(tree):
    """Refusing on "I could not resolve this" asserts knowledge the checker does not have."""
    repo, artifacts = tree
    directory = repo / "case_studies" / "fx_pairs"
    directory.mkdir(parents=True)
    (directory / "08_tabular_dl.py").write_text(
        'SUPERSEDES_POPULATION: str = "gone"\n'
        'population_name = POPULATION_NAME or f"{CASE_STUDY_ID}:tabular_dl"\n'
    )
    _registry(artifacts, "fx_pairs", [("aaaa", "some-other-name", None)])

    findings = check_case_study("fx_pairs", repo_root=repo, artifacts_root=artifacts)

    assert [f.status for f in findings] == ["unresolved"]
    assert _exit_status(repo, artifacts) == 0


def test_a_stale_finding_carries_the_hash_to_paste(tree):
    """The message has to say what the fix is, not only that something is wrong."""
    repo, artifacts = tree
    _notebook(repo, "etfs", "07_gbm", "gone", "etfs-gbm-v1")
    _registry(artifacts, "etfs", [("aaaa", "etfs-gbm-v1", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].remedy == "aaaa"


def test_a_per_label_causal_remedy_names_that_labels_current_identity(tree):
    """A per-label declaration knows its label, so the answer can be exact."""
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", '{"fwd_ret_21d": "gone"}')
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("bbbb", "fwd_ret_21d", None)])

    findings = check_case_study("etfs", repo_root=repo, artifacts_root=artifacts)

    assert findings[0].is_stale
    assert findings[0].remedy == "bbbb"


def test_a_multi_label_repair_says_which_entry_to_replace(capsys, tree):
    """Following "set it to <hash>" on a mapping would break the run this message saves.

    `supersedes_for` rejects a bare hash from a notebook that fits several labels, so the
    instruction has to name the entry rather than the declaration.
    """
    repo, artifacts = tree
    _causal_notebook(repo, "etfs", "12_causal_dml", '{"fwd_ret_21d": "gone"}')
    _registry(artifacts, "etfs", [])
    _causal_rows(artifacts, "etfs", [("aaaa", "fwd_ret_21d", None)])

    assert _exit_status(repo, artifacts) == 1
    message = capsys.readouterr().err

    assert "replace the 'fwd_ret_21d' entry in the mapping" in message
    assert "leaving the other entries as they are" in message
