"""Gate: a committed notebook must be its current .py executed in production.

Stamped notebooks carry ``metadata.ml4t_provenance`` recording the git blob of the
paired ``.py`` they were executed from and whether the run used production
parameters. This test fails if any *stamped* notebook is stale (its ``.py`` changed
since execution), was committed from a TEST-mode run, or carries a stamp that
contradicts the ``injected-parameters`` cell in the committed notebook.

Unstamped notebooks are not failed here (adoption is gradual — stamp notebooks as
they are re-run through the canonical path). See
``.github/scripts/notebook_provenance.py`` for the stamp/check tool. To stamp::

    uv run python .github/scripts/notebook_provenance.py stamp <nb.ipynb> \
        --executor <env> --production

The executor declares the parameters (``--production`` or ``--parameters '<json>'``)
because ``metadata.papermill.parameters`` is a fossil: papermill does not clear it
on an unparameterized re-run, and ``jupytext --sync`` deletes the
``injected-parameters`` cell while leaving that metadata behind. The tests below pin
both halves of the replacement — the declaration is what gets recorded, and a
committed injected cell can veto it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

import notebook_provenance  # noqa: E402
from notebook_provenance import (  # noqa: E402
    check_all,
    contradicts_injected_cell,
    injected_parameters,
    production_parameters,
    stamp_notebook,
)


def _notebook(cells: list[dict], metadata: dict | None = None) -> dict:
    return {
        "cells": cells,
        "metadata": metadata or {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _injected_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"tags": ["injected-parameters"]},
        "source": source,
        "outputs": [],
        "execution_count": 1,
    }


def test_production_parameters_allow_only_full_run_cache_bypasses():
    assert production_parameters({})
    assert production_parameters({"FORCE_RETRAIN": True})
    assert production_parameters({"FORCE_RETRAIN": "true", "USE_CACHE": "false"})
    assert production_parameters({"FORCE_REBACKTEST": "1"})

    assert not production_parameters({"FORCE_RETRAIN": False})
    assert not production_parameters({"MAX_FOLDS": 1})
    assert not production_parameters({"TRAIN_SAMPLE_FRAC": 0.1})
    assert not production_parameters({"USE_CACHE": True})


def test_numeric_and_string_bools_are_the_same_override() -> None:
    """``1`` from a JSON declaration and ``"1"`` from ``papermill -p`` are one run."""
    assert production_parameters({"FORCE_REBACKTEST": 1})
    assert production_parameters({"USE_CACHE": 0})
    assert not production_parameters({"USE_CACHE": 1})

    nb = _notebook([_injected_cell("# Parameters\nFORCE_RETRAIN = 1\n")])
    assert contradicts_injected_cell(nb, {"FORCE_RETRAIN": "1"}) is None
    assert contradicts_injected_cell(nb, {"FORCE_RETRAIN": True}) is None
    assert contradicts_injected_cell(nb, {"FORCE_RETRAIN": 0}) is not None


def test_bool_coercion_does_not_reach_ordinary_numeric_parameters() -> None:
    """``MAX_SYMBOLS = 1`` is a count, not a flag, so it must not match ``true``."""
    nb = _notebook([_injected_cell("# Parameters\nMAX_SYMBOLS = 1\n")])
    assert contradicts_injected_cell(nb, {"MAX_SYMBOLS": "1"}) is None
    assert contradicts_injected_cell(nb, {"MAX_SYMBOLS": True}) is not None
    assert contradicts_injected_cell(nb, {"MAX_SYMBOLS": 2}) is not None


def test_large_integers_do_not_collapse_onto_each_other() -> None:
    """Above 2**53 a float round-trip would make two distinct values compare equal."""
    nb = _notebook([_injected_cell("# Parameters\nSEED = 9007199254740993\n")])
    assert contradicts_injected_cell(nb, {"SEED": 9007199254740993}) is None
    assert contradicts_injected_cell(nb, {"SEED": "9007199254740993"}) is None
    assert contradicts_injected_cell(nb, {"SEED": 9007199254740992}) is not None


def test_float_matches_its_decimal_string() -> None:
    nb = _notebook([_injected_cell("# Parameters\nTRAIN_SAMPLE_FRAC = 0.1\n")])
    assert contradicts_injected_cell(nb, {"TRAIN_SAMPLE_FRAC": "0.1"}) is None
    assert contradicts_injected_cell(nb, {"TRAIN_SAMPLE_FRAC": 0.2}) is not None


def test_non_finite_values_reach_one_normal_form() -> None:
    """A Decimal NaN does not equal itself, so these have to compare as text.

    The cell source is what papermill 2.7's ``PythonTranslator`` actually emits for
    a non-finite float: a ``float(...)`` call rather than a literal.
    """
    nb = _notebook([_injected_cell("# Parameters\nCLIP = float('nan')\nCAP = float('inf')\n")])
    assert contradicts_injected_cell(nb, {"CLIP": float("nan"), "CAP": float("inf")}) is None
    assert contradicts_injected_cell(nb, {"CLIP": "NaN", "CAP": "Infinity"}) is None
    assert contradicts_injected_cell(nb, {"CLIP": float("nan"), "CAP": 5}) is not None


def test_papermill_translator_still_spells_non_finite_floats_as_a_call() -> None:
    """Pins the assumption the test above encodes, so it fails if papermill changes."""
    papermill_translators = pytest.importorskip("papermill.translators")

    assert papermill_translators.PythonTranslator.translate(float("nan")) == "float('nan')"
    assert papermill_translators.PythonTranslator.translate(float("inf")) == "float('inf')"


def test_string_parameters_compare_by_value() -> None:
    nb = _notebook([_injected_cell('# Parameters\nSTART_DATE = "2024-06-01"\n')])
    assert contradicts_injected_cell(nb, {"START_DATE": "2024-06-01"}) is None
    assert contradicts_injected_cell(nb, {"START_DATE": "2020-01-01"}) is not None


# -----------------------------------------------------------------------------
# The injected-parameters cell — the record that belongs to the execution
# -----------------------------------------------------------------------------


def test_injected_parameters_reads_the_cell_papermill_wrote() -> None:
    nb = _notebook([_injected_cell('# Parameters\nMAX_SYMBOLS = 5\nSTART_DATE = "2024-06-01"\n')])
    assert injected_parameters(nb) == {"MAX_SYMBOLS": 5, "START_DATE": "2024-06-01"}


def test_injected_parameters_is_none_without_the_cell() -> None:
    """No cell is no evidence — not evidence of a production run."""
    assert injected_parameters(_notebook([])) is None


def test_injected_parameters_accepts_a_source_list() -> None:
    """nbformat stores cell source as a list of lines as often as a string."""
    nb = _notebook([_injected_cell(["# Parameters\n", "MAX_SYMBOLS = 5\n"])])
    assert injected_parameters(nb) == {"MAX_SYMBOLS": 5}


# -----------------------------------------------------------------------------
# The stamp declares; the injected cell can veto
# -----------------------------------------------------------------------------


def test_no_injected_cell_never_contradicts() -> None:
    assert contradicts_injected_cell(_notebook([]), {"MAX_SYMBOLS": 5}) is None


def test_declaration_matching_the_injected_cell_does_not_contradict() -> None:
    nb = _notebook([_injected_cell("# Parameters\nMAX_SYMBOLS = 5\n")])
    assert contradicts_injected_cell(nb, {"MAX_SYMBOLS": 5}) is None


def test_papermill_string_coercion_does_not_look_like_a_contradiction() -> None:
    """``-p FORCE_RETRAIN true`` reaches the cell as a string and the CLI as a bool."""
    nb = _notebook([_injected_cell('# Parameters\nFORCE_RETRAIN = "true"\n')])
    assert contradicts_injected_cell(nb, {"FORCE_RETRAIN": True}) is None


def test_claiming_production_over_an_injected_cell_contradicts() -> None:
    """The false positive the fossil allowed: a TEST run stamped as production."""
    nb = _notebook([_injected_cell("# Parameters\nMAX_SYMBOLS = 5\n")])
    conflict = contradicts_injected_cell(nb, {})
    assert conflict is not None
    assert "MAX_SYMBOLS" in conflict


def test_stamp_refuses_to_contradict_the_injected_cell(tmp_path, monkeypatch) -> None:
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text(json.dumps(_notebook([_injected_cell("# Parameters\nMAX_SYMBOLS = 5\n")])))
    (tmp_path / "demo.py").write_text("# %%\nMAX_SYMBOLS = 0\n")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)

    with pytest.raises(SystemExit, match="refusing to stamp"):
        stamp_notebook(nb_path, executor="local-uv", parameters={})


def test_stamp_records_the_declaration_not_the_papermill_fossil(tmp_path, monkeypatch) -> None:
    """The defect this replaced.

    Papermill leaves ``metadata.papermill.parameters`` in place across an
    unparameterized re-run, and ``jupytext --sync`` keeps that metadata while
    deleting the ``injected-parameters`` cell. A notebook in exactly that state —
    stale TEST parameters in metadata, no injected cell — used to be stamped
    ``production=False`` from the fossil, failing a genuine production run.
    """
    nb_path = tmp_path / "demo.ipynb"
    fossil = {"papermill": {"parameters": {"MAX_SYMBOLS": 5, "START_DATE": "2024-06-01"}}}
    nb_path.write_text(json.dumps(_notebook([], metadata=fossil)))
    (tmp_path / "demo.py").write_text("# %%\nMAX_SYMBOLS = 0\n")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)

    stamp = stamp_notebook(nb_path, executor="local-uv", parameters={})

    assert stamp["production"] is True
    assert stamp["parameters"] == {}
    # The fossil is overwritten, so it cannot outlive the stamp and disagree.
    written = json.loads(nb_path.read_text())
    assert written["metadata"]["papermill"]["parameters"] == {}


def test_stamp_records_declared_overrides_as_test_mode(tmp_path, monkeypatch) -> None:
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text(json.dumps(_notebook([])))
    (tmp_path / "demo.py").write_text("# %%\nMAX_SYMBOLS = 0\n")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)

    stamp = stamp_notebook(nb_path, executor="local-uv", parameters={"MAX_SYMBOLS": 5})

    assert stamp["production"] is False
    assert stamp["parameters"] == {"MAX_SYMBOLS": 5}


# -----------------------------------------------------------------------------
# The gate over the committed tree
# -----------------------------------------------------------------------------


def test_stamped_notebooks_are_current_and_production() -> None:
    stale, testmode, contradicted, _unverified = check_all(strict=False)
    assert not stale and not testmode and not contradicted, (
        "Committed notebooks are out of sync with their source .py:\n"
        + (
            "  STALE (re-run in the canonical env):\n    " + "\n    ".join(stale) + "\n"
            if stale
            else ""
        )
        + (
            "  TEST-MODE (must be a production run):\n    " + "\n    ".join(testmode) + "\n"
            if testmode
            else ""
        )
        + (
            "  CONTRADICTED (stamp disagrees with the injected-parameters cell):\n    "
            + "\n    ".join(contradicted)
            if contradicted
            else ""
        )
    )


# --- Scoped check: a notebook whose source the change deleted ----------------
#
# check_all cannot see this case. Its `paired_py() is None` branch cannot tell a
# notebook that was just orphaned from one that was never paired, and three tracked
# notebooks are deliberately unpaired, so the distinction has to come from the diff.
# These use a real git repo because the whole mechanism is a `git diff` invocation -
# rename detection, -z quoting and --diff-filter are the behaviour under test, and a
# mocked diff would only assert that the mock returns what it was told to.


def _git(repo: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _repo_with_a_paired_notebook(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "chapter").mkdir(parents=True)
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "chapter" / "demo.py").write_text("# %%\nX = 1\n")
    (repo / "chapter" / "demo.ipynb").write_text(json.dumps({"cells": [], "metadata": {}}))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "paired notebook")
    _git(repo, "branch", "-M", "main")
    _git(repo, "checkout", "-qb", "topic")
    return repo


def test_deleting_the_source_and_keeping_the_notebook_is_reported(tmp_path, monkeypatch) -> None:
    repo = _repo_with_a_paired_notebook(tmp_path)
    _git(repo, "rm", "-q", "chapter/demo.py")
    _git(repo, "commit", "-qm", "drop the source, keep the render")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    orphaned = notebook_provenance.notebooks_orphaned_since("main")

    assert len(orphaned) == 1
    assert "chapter/demo.ipynb" in orphaned[0]
    assert "chapter/demo.py" in orphaned[0]


def test_moving_the_source_leaves_the_notebook_orphaned(tmp_path, monkeypatch) -> None:
    """The case --no-renames exists for. With rename detection on, git reports only
    the destination and the notebook rendered from the old path goes unmentioned."""
    repo = _repo_with_a_paired_notebook(tmp_path)
    _git(repo, "mv", "chapter/demo.py", "chapter/renamed.py")
    _git(repo, "commit", "-qm", "move the source out from under the notebook")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    orphaned = notebook_provenance.notebooks_orphaned_since("main")

    assert len(orphaned) == 1
    assert "chapter/demo.ipynb" in orphaned[0]


def test_deleting_both_halves_is_not_an_orphan(tmp_path, monkeypatch) -> None:
    """Retiring a notebook properly must stay merge-able."""
    repo = _repo_with_a_paired_notebook(tmp_path)
    _git(repo, "rm", "-q", "chapter/demo.py", "chapter/demo.ipynb")
    _git(repo, "commit", "-qm", "retire the notebook")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    assert notebook_provenance.notebooks_orphaned_since("main") == []


def test_editing_a_paired_notebook_is_not_an_orphan(tmp_path, monkeypatch) -> None:
    repo = _repo_with_a_paired_notebook(tmp_path)
    (repo / "chapter" / "demo.py").write_text("# %%\nX = 2\n")
    _git(repo, "commit", "-qam", "ordinary edit")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    assert notebook_provenance.notebooks_orphaned_since("main") == []


def test_a_notebook_name_with_a_space_survives_the_diff_parse(tmp_path, monkeypatch) -> None:
    """git quotes such a path under plain --name-only, and splitting on whitespace
    tears it into fragments matching no suffix - the notebook then leaves the gate's
    scope and passes unchecked, which is the one thing a gate must never do."""
    repo = _repo_with_a_paired_notebook(tmp_path)
    (repo / "chapter" / "my demo.py").write_text("# %%\nX = 1\n")
    (repo / "chapter" / "my demo.ipynb").write_text(json.dumps({"cells": [], "metadata": {}}))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a notebook with a space in its name")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    changed = [
        str(p.relative_to(repo)) for p in notebook_provenance.notebooks_changed_since("main")
    ]

    assert "chapter/my demo.ipynb" in changed
