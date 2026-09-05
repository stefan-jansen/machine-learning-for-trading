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
    destamped,
    injected_parameters,
    is_cleared,
    production_parameters,
    stamp_notebook,
    was_executed,
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


def test_an_identity_bearing_override_is_production() -> None:
    """A canonical run may be required to carry a parameter.

    research/population.py refuses a re-run into a changed population unless it names
    the population it supersedes, and that value is set deliberately by a person. Such
    a run is production: the override adds a declaration rather than removing work.
    """
    assert production_parameters({"SUPERSEDES_POPULATION": "342446006141"})
    assert production_parameters({"SUPERSEDES_POPULATION": "342446006141", "FORCE_RETRAIN": True})


def test_waiving_the_value_check_does_not_admit_a_reduced_run() -> None:
    """The waiver is per name, not a general relaxation.

    Only the value of an identity-bearing override goes unchecked - its correctness is
    enforced where it is consumed. A name that is not on the list is still not
    production, including alongside one that is.
    """
    assert not production_parameters({"SUPERSEDES_POPULATION": "abc", "MAX_SYMBOLS": 8})
    assert not production_parameters({"SUPERSEDES_POPULATION": "abc", "FORCE_RETRAIN": False})
    assert not production_parameters({"SUPERSEDES_ARTIFACT": "abc"})


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
    result = check_all(strict=False)
    stale, testmode, contradicted, hollow = (
        result.stale,
        result.testmode,
        result.contradicted,
        result.hollow,
    )
    assert not stale and not testmode and not contradicted and not hollow, (
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
        + (
            "  HOLLOW (stamped over an empty output set):\n    " + "\n    ".join(hollow)
            if hollow
            else ""
        )
    )


def test_no_notebook_loses_a_provenance_stamp_it_already_had() -> None:
    """The hole in "unstamped notebooks are not failed here".

    Not failing an unstamped notebook is deliberate, and it means dropping a stamp
    switches this gate off for that file rather than failing it - a gate that passes
    because its subject is absent, which is the shape the September sign-offs had.
    fx_pairs lost its stamp twice during stage 03 before anything said so. Stamps are
    only ever added, so a notebook stamped at HEAD and unstamped now is a regression.
    """
    lost = destamped()
    assert not lost, "these notebooks had a provenance stamp at HEAD and do not now:\n    " + (
        "\n    ".join(lost)
    )


# -----------------------------------------------------------------------------
# Alt-text-only drift
#
# The stamp records the .py blob, so any edit to the .py moves it and the gate reads
# a corrected figure description as a notebook needing re-execution. For alt text
# that is wrong: show_plotly_with_alt puts the string in output metadata and takes
# the image from fig._repr_mimebundle_(), which never sees it, so no re-run can
# produce different outputs. nasdaq100_microstructure/04 is 90 minutes to restate
# four sentences. These pin the carve-out and its edges.
# -----------------------------------------------------------------------------


def _alt_cell(
    alt: str,
    *,
    png: str = "iVBORw0KGgo=",
    carried: str | None = None,
    call: str = "show_plotly_with_alt",
) -> dict:
    """A code cell calling *call* with one png output carrying *carried*."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": f'fig = build()\n{call}(fig, "{alt}")\n',
        "outputs": [
            {
                "output_type": "display_data",
                "data": {"image/png": png},
                "metadata": {"image/png": {"alt": alt if carried is None else carried}},
            }
        ],
    }


def _drift(tmp_path, monkeypatch, old_src: str, new_src: str, nb: dict) -> bool:
    """Run alt_text_only_drift with *old_src* as the stamped blob and *new_src* on disk."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=tmp_path,
        input=old_src,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    py = tmp_path / "nb.py"
    py.write_text(new_src, encoding="utf-8")
    return notebook_provenance.alt_text_only_drift(blob, py, nb)


def test_correcting_only_the_alt_text_is_not_stale(tmp_path, monkeypatch) -> None:
    old = 'fig = build()\nshow_plotly_with_alt(fig, "the bottom row is the holdout")\n'
    new = 'fig = build()\nshow_plotly_with_alt(fig, "the top row is the holdout")\n'
    nb = _notebook([_alt_cell("the top row is the holdout")])
    assert _drift(tmp_path, monkeypatch, old, new, nb)


def test_correcting_alt_text_on_a_qualified_call_is_not_stale(tmp_path, monkeypatch) -> None:
    """``import utils.style as style`` gives ``style.show_plotly_with_alt(...)``.

    The exception matched only bare ``ast.Name`` calls, so the qualified form fell
    outside it. ``case_studies/etfs/05_evaluation.py`` is the one notebook in the
    corpus written this way, and ``import utils.style as style`` appears 19 times.
    """
    old = 'fig = build()\nstyle.show_plotly_with_alt(fig, "the bottom row is the holdout")\n'
    new = 'fig = build()\nstyle.show_plotly_with_alt(fig, "the top row is the holdout")\n'
    nb = _notebook([_alt_cell("the top row is the holdout", call="style.show_plotly_with_alt")])
    assert _drift(tmp_path, monkeypatch, old, new, nb)


def test_a_code_change_beside_a_qualified_alt_change_is_still_stale(tmp_path, monkeypatch) -> None:
    """Recognising the qualified form must not also forgive a changed constant."""
    old = 'fig = build(n=5)\nstyle.show_plotly_with_alt(fig, "old words")\n'
    new = 'fig = build(n=20)\nstyle.show_plotly_with_alt(fig, "new words")\n'
    nb = _notebook([_alt_cell("new words", call="style.show_plotly_with_alt")])
    assert not _drift(tmp_path, monkeypatch, old, new, nb)


def test_a_code_change_beside_an_alt_change_is_still_stale(tmp_path, monkeypatch) -> None:
    """The whole point of the stamp. One changed constant must not ride along."""
    old = 'fig = build(n=5)\nshow_plotly_with_alt(fig, "old words")\n'
    new = 'fig = build(n=20)\nshow_plotly_with_alt(fig, "new words")\n'
    nb = _notebook([_alt_cell("new words")])
    assert not _drift(tmp_path, monkeypatch, old, new, nb)


def test_alt_text_the_outputs_do_not_carry_is_stale(tmp_path, monkeypatch) -> None:
    """A source-only edit. The outputs still describe the figure the old way."""
    old = 'fig = build()\nshow_plotly_with_alt(fig, "old words")\n'
    new = 'fig = build()\nshow_plotly_with_alt(fig, "new words")\n'
    nb = _notebook([_alt_cell("new words", carried="old words")])
    assert not _drift(tmp_path, monkeypatch, old, new, nb)


def test_a_trailing_semicolon_is_stale(tmp_path, monkeypatch) -> None:
    """A semicolon suppresses a cell's automatic display and leaves the AST identical."""
    old = '# %%\nfig = build()\nshow_plotly_with_alt(fig, "words")\nsummary\n'
    new = '# %%\nfig = build()\nshow_plotly_with_alt(fig, "words")\nsummary;\n'
    nb = _notebook([_alt_cell("words")])
    assert not _drift(tmp_path, monkeypatch, old, new, nb)


def test_moving_a_cell_boundary_is_stale(tmp_path, monkeypatch) -> None:
    """Which code shares a cell decides which value is its last expression."""
    old = "# %%\nfig = build()\n\n# %%\nsummary\n"
    new = "# %%\nfig = build()\nsummary\n"
    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([]))


def test_retagging_a_cell_is_stale(tmp_path, monkeypatch) -> None:
    """The marker carries cell tags, so it is compared even for a markdown cell."""
    old = "# %% [markdown]\n# words\n"
    new = '# %% [markdown] tags=["results"]\n# words\n'
    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([]))


def test_rewrapping_the_alt_literal_is_not_stale(tmp_path, monkeypatch) -> None:
    """ruff format rewraps a long alt string; the value is what matters, not the wrapping."""
    old = '# %%\nshow_plotly_with_alt(fig, "one two three")\n'
    new = '# %%\nshow_plotly_with_alt(\n    fig,\n    "one two "\n    "three",\n)\n'
    nb = _notebook([_alt_cell("one two three")])
    assert _drift(tmp_path, monkeypatch, old, new, nb)


def test_a_comment_only_change_is_not_stale(tmp_path, monkeypatch) -> None:
    """Markdown cells are comments in a jupytext .py and cannot affect outputs."""
    old = "# %% [markdown]\n# the last row is the holdout\n\n# %%\nfig = build()\n"
    new = "# %% [markdown]\n# the top row is the holdout\n\n# %%\nfig = build()\n"
    assert _drift(tmp_path, monkeypatch, old, new, _notebook([]))


def test_removing_papermill_markers_under_the_exact_filter_is_not_stale(
    tmp_path, monkeypatch
) -> None:
    old = (
        "# ---\n# jupyter:\n#   jupytext:\n#     text_representation:\n# ---\n"
        '# %% papermill={"duration": 1.2, "status": "completed"} tags=["parameters"]\n'
        "MAX_FOLDS = 0\n"
    )
    new = (
        "# ---\n# jupyter:\n#   jupytext:\n"
        "#     cell_metadata_filter: tags,-all\n#     text_representation:\n# ---\n"
        '# %% tags=["parameters"]\nMAX_FOLDS = 0\n'
    )
    nb = _notebook([], metadata={"jupytext": {"cell_metadata_filter": "tags,-all"}})
    assert _drift(tmp_path, monkeypatch, old, new, nb)


def test_papermill_cleanup_does_not_forgive_a_tag_change(tmp_path, monkeypatch) -> None:
    old = '# %% papermill={"duration": 1.2} tags=["parameters"]\nMAX_FOLDS = 0\n'
    new = (
        "# ---\n# jupyter:\n#   jupytext:\n#     cell_metadata_filter: tags,-all\n# ---\n"
        '# %% tags=["results"]\nMAX_FOLDS = 0\n'
    )
    nb = _notebook([], metadata={"jupytext": {"cell_metadata_filter": "tags,-all"}})
    assert not _drift(tmp_path, monkeypatch, old, new, nb)


def test_papermill_cleanup_requires_the_exact_filter(tmp_path, monkeypatch) -> None:
    old = '# %% papermill={"duration": 1.2}\nresult = compute()\n'
    new = "# %%\nresult = compute()\n"
    nb = _notebook([], metadata={"jupytext": {"cell_metadata_filter": "tags,-papermill"}})
    assert not _drift(tmp_path, monkeypatch, old, new, nb)


def test_code_appended_into_a_markdown_cell_is_stale(tmp_path, monkeypatch) -> None:
    """A markdown body is only ignorable while it is all comments.

    Measured against the real notebook: its last cell is `# %% [markdown]`, so appending
    `fig;` landed in a markdown body and was forgiven until the body was compared.
    """
    old = "# %% [markdown]\n# words\n"
    new = "# %% [markdown]\n# words\nfig;\n"
    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([]))


def test_editing_markdown_prose_is_still_not_stale(tmp_path, monkeypatch) -> None:
    old = "# %% [markdown]\n# the last row is the holdout\n"
    new = "# %% [markdown]\n# the top row is the holdout\n"
    assert _drift(tmp_path, monkeypatch, old, new, _notebook([]))


def test_an_unparseable_source_is_stale(tmp_path, monkeypatch) -> None:
    old = "fig = build()\n"
    new = "fig = build(\n"
    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([]))


def test_a_stamped_blob_this_repo_does_not_have_is_stale(tmp_path, monkeypatch) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    py = tmp_path / "nb.py"
    py.write_text("fig = build()\n", encoding="utf-8")
    assert not notebook_provenance.alt_text_only_drift("0" * 40, py, _notebook([]))


def test_prose_edit_beside_a_computed_alt_is_not_stale(tmp_path, monkeypatch) -> None:
    """Eight case studies write the leading configuration into their alt with an f-string.

    That alt is not knowable from the source, so it cannot be compared against what the
    output carries. Requiring the comparison anyway made the counts disagree and failed
    the notebook, so the carve-out never applied to the notebooks that read their figures
    off the frame - the ones it was written for.
    """
    old = (
        "# %% [markdown]\n# the grid spans 0.032 to 0.0008\n\n"
        "# %%\nfig = build()\n"
        'show_plotly_with_alt(fig, f"the leader is {leader} at {value:+.3f}")\n'
    )
    new = (
        "# %% [markdown]\n# read best_ic against worst_ic in the frame\n\n"
        "# %%\nfig = build()\n"
        'show_plotly_with_alt(fig, f"the leader is {leader} at {value:+.3f}")\n'
    )
    cell = {
        "cell_type": "code",
        "metadata": {},
        "source": 'fig = build()\nshow_plotly_with_alt(fig, f"the leader is {leader} at {value:+.3f}")\n',
        "outputs": [
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo="},
                "metadata": {"image/png": {"alt": "the leader is ridge at +0.032"}},
            }
        ],
    }
    assert _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


def _computed_alt_cell(source_alt: str, carried: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": f"fig = build()\nshow_plotly_with_alt(fig, {source_alt})\n",
        "outputs": [
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo="},
                "metadata": {"image/png": {"alt": carried}},
            }
        ],
    }


def test_rewording_a_computed_alt_the_output_carries_is_allowed(tmp_path, monkeypatch) -> None:
    """The case #867 was filed for.

    Writing alt text against computed values is what stops a description drifting from
    its figure, and it used to cost a full re-execution to reword: the f-string was left
    whole in the compared dump, so any edit read as stale. The same reword to a plain
    literal next door was accepted as a diff.

    The bargain is the same as the literal branch's: accepted only because the output
    metadata carries the reworded text too.
    """
    old = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the leader is {leader}")\n'
    new = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the winner is {leader}")\n'
    cell = _computed_alt_cell('f"the winner is {leader}"', "the winner is ridge")

    assert _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


def test_rewording_a_computed_alt_the_output_does_not_carry_is_stale(tmp_path, monkeypatch) -> None:
    """Editing the .py and leaving the executed alt saying the old thing is stale.

    Without this the loosening would forgive a notebook whose figure description and
    whose source disagree, which is the state the whole gate exists to refuse.
    """
    old = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the leader is {leader}")\n'
    new = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the winner is {leader}")\n'
    cell = _computed_alt_cell('f"the winner is {leader}"', "the leader is ridge")

    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


def test_changing_what_a_computed_alt_reads_is_stale(tmp_path, monkeypatch) -> None:
    """Only the prose is forgiven; the interpolated expressions are not.

    `{leader}` to `{runner_up}` changes what the alt asserts about the data, so it has
    to force the re-run - and it does, because the expression parts stay in the dump.
    """
    old = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the winner is {leader}")\n'
    new = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the winner is {runner_up}")\n'
    cell = _computed_alt_cell('f"the winner is {runner_up}"', "the winner is ridge")

    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


def test_an_implicit_concatenation_of_a_literal_and_an_f_string_is_handled(
    tmp_path, monkeypatch
) -> None:
    """The shape that broke the first attempt at this.

    `"Boosting curve, " f"{n} below zero"` is one JoinedStr whose first constant is a
    whole quoted literal and whose last is bare prose between the braces. Blanking them
    textually needs a different placeholder for each, and guessing wrong writes source
    that does not parse - which does not fail loudly, it makes the exception
    unavailable for the notebook and reports it as stale. Eight notebooks in the tree
    have this shape.
    """
    old = (
        "# %%\nfig = build()\nshow_plotly_with_alt(\n    fig,\n"
        '    "Boosting curve, "\n    f"{n} lines below zero.",\n)\n'
    )
    new = (
        "# %%\nfig = build()\nshow_plotly_with_alt(\n    fig,\n"
        '    "Boosting curves, "\n    f"{n} lines below zero.",\n)\n'
    )
    cell = _computed_alt_cell(
        '"Boosting curves, " f"{n} lines below zero."',
        "Boosting curves, 3 lines below zero.",
    )

    assert _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


def test_a_computed_alt_the_output_does_not_carry_is_stale(tmp_path, monkeypatch) -> None:
    """An alt added since execution: the image is there and no alt metadata is."""
    old = '# %% [markdown]\n# one\n\n# %%\nfig = build()\nshow_plotly_with_alt(fig, f"{leader}")\n'
    new = '# %% [markdown]\n# two\n\n# %%\nfig = build()\nshow_plotly_with_alt(fig, f"{leader}")\n'
    cell = {
        "cell_type": "code",
        "metadata": {},
        "source": 'fig = build()\nshow_plotly_with_alt(fig, f"{leader}")\n',
        "outputs": [
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo="},
                "metadata": {},
            }
        ],
    }
    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


def test_every_row_the_scan_returns_is_one_of_the_files_it_was_given() -> None:
    """The gate must answer for the files it is given, not for the working tree.

    `check_all` scanning everything is what let one session's dirty notebook block
    every unrelated commit in a shared worktree. This is the contract that fixes it,
    and it holds whatever state the tree is in: no category may name a notebook
    outside `only`. Nothing unstaged can reach main, so the narrower scan gives up
    no protection.
    """
    everything = check_all()
    all_rows = sorted({r for category in everything for r in category})
    assert all_rows, "no notebook in this tree for the scan to report on"

    chosen = {all_rows[0]}
    for category in check_all(only=chosen):
        assert set(category) <= chosen

    excluded = set(all_rows[1:])
    if excluded:
        for category in check_all(only=excluded):
            assert all_rows[0] not in category


def test_the_paired_py_selects_its_notebook() -> None:
    """pre-commit stages the `.py`, so naming it has to reach the `.ipynb` beside it."""
    everything = check_all()
    all_rows = sorted({r for category in everything for r in category})
    assert all_rows, "no notebook in this tree for the scan to report on"
    nb = all_rows[0]
    by_py = check_all(only={nb.removesuffix(".ipynb") + ".py"})
    assert sorted({r for category in by_py for r in category}) == [nb]


def test_an_empty_restriction_still_scans_everything() -> None:
    """`only=None` is the whole tree, which is what CI calls and must not change."""
    assert check_all(only=None) == check_all()


# -----------------------------------------------------------------------------
# The cleared state
#
# Before this existed the gate had no legal way to land a correction to a notebook
# that had already been executed: keeping the stale stamp read as STALE and dropping
# it read as DE-STAMPED, and clearing either one needed the production run that the
# uncommitted correction was a prerequisite for. Work accumulated in worktrees
# instead - 177 uncommitted files across 24 of them on 2026-08-23.
#
# A cleared notebook (no stamp, no outputs) asserts nothing about a run, so there is
# nothing for the gate to catch it lying about. What still fails is the render that
# looks cleared but claims a run: a stamp over an empty output set.
# -----------------------------------------------------------------------------


def _code(source: str, outputs: list | None = None) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": outputs if outputs is not None else [],
        "execution_count": None,
    }


def _stdout(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text}


def test_a_notebook_with_outputs_was_executed() -> None:
    assert was_executed(_notebook([_code("print(1)", [_stdout("1")])]))


def test_a_notebook_whose_code_never_ran_was_not_executed() -> None:
    assert not was_executed(_notebook([_code("print(1)")]))


def test_a_silent_cell_with_an_execution_count_was_executed() -> None:
    """A cell that only assigns or writes a file runs fine and displays nothing.

    The kernel still stamps ``execution_count``, so the notebook is not hollow.
    """
    cell = _code("x = 1")
    cell["execution_count"] = 3
    assert was_executed(_notebook([cell]))


def test_a_blank_code_cell_does_not_count_as_unexecuted_code() -> None:
    """jupytext leaves empty trailing cells; they must not make a live notebook hollow."""
    assert was_executed(_notebook([_code("print(1)", [_stdout("1")]), _code("   ")]))


def test_a_markdown_only_notebook_is_not_hollow() -> None:
    assert was_executed(_notebook([{"cell_type": "markdown", "source": "# hi", "metadata": {}}]))


def test_cleared_is_no_stamp_and_no_outputs(tmp_path) -> None:
    nb = tmp_path / "cleared.ipynb"
    nb.write_text(json.dumps(_notebook([_code("print(1)")])), encoding="utf-8")
    assert is_cleared(nb)


def test_a_notebook_keeping_its_outputs_is_not_cleared(tmp_path) -> None:
    nb = tmp_path / "live.ipynb"
    nb.write_text(json.dumps(_notebook([_code("print(1)", [_stdout("1")])])), encoding="utf-8")
    assert not is_cleared(nb)


def test_a_stamped_notebook_is_not_cleared_even_with_no_outputs(tmp_path) -> None:
    """The hollow render: it claims a production run and has nothing to show for it."""
    nb = tmp_path / "hollow.ipynb"
    nb.write_text(
        json.dumps(
            _notebook(
                [_code("print(1)")],
                metadata={notebook_provenance.STAMP_KEY: {"production": True}},
            )
        ),
        encoding="utf-8",
    )
    assert not is_cleared(nb)


def test_clearing_a_notebook_makes_it_committable(tmp_path) -> None:
    """End to end on the state that used to be unreachable: stamped, edited, cleared."""
    nb = tmp_path / "nb.ipynb"
    nb.write_text(
        json.dumps(
            _notebook(
                [
                    _injected_cell("MAX_SYMBOLS = 5"),
                    _code("print(1)", [_stdout("1")]),
                ],
                metadata={
                    notebook_provenance.STAMP_KEY: {"production": True, "source_py_blob": "abc"},
                    "papermill": {"parameters": {"MAX_SYMBOLS": 5}},
                },
            )
        ),
        encoding="utf-8",
    )
    notebook_provenance._cmd_clear(__import__("argparse").Namespace(notebooks=[str(nb)]))
    written = json.loads(nb.read_text(encoding="utf-8"))
    assert notebook_provenance.STAMP_KEY not in written["metadata"]
    assert "papermill" not in written["metadata"]
    assert not any(
        "injected-parameters" in (c.get("metadata", {}).get("tags") or []) for c in written["cells"]
    )
    assert is_cleared(nb)


def test_stamp_refuses_a_notebook_that_never_ran(tmp_path, monkeypatch) -> None:
    """Fail at the stamp, not two steps later at the commit.

    The way an unexecuted notebook gets stamped is a `jupytext --sync` between the run
    and the stamp: the .py is newer, so the .ipynb is rebuilt from it and the outputs
    go. Refusing here names the step that went wrong.
    """
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "nb.py").write_text("# %%\nprint(1)\n", encoding="utf-8")
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps(_notebook([_code("print(1)")])), encoding="utf-8")
    with pytest.raises(SystemExit, match="nothing in it was executed"):
        stamp_notebook(nb, "local-uv", parameters={})


def test_stamp_accepts_a_notebook_whose_cells_only_have_execution_counts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "nb.py").write_text("# %%\nx = 1\n", encoding="utf-8")
    cell = _code("x = 1")
    cell["execution_count"] = 1
    nb = tmp_path / "nb.ipynb"
    nb.write_text(json.dumps(_notebook([cell])), encoding="utf-8")
    assert stamp_notebook(nb, "local-uv", parameters={})["production"] is True


# --- The merge gate's scope: what a change is answerable for ------------------
#
# `check --since <base>` is what CI runs, so these cover the diff parse the scope is
# built from. They use a real git repo because the mechanism *is* a `git diff`
# invocation - rename detection, -z quoting and --diff-filter are the behaviour under
# test, and a mocked diff would only assert that the mock returns what it was told to.


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
    """check_all cannot see this by construction.

    Its ``paired_py() is None`` branch cannot tell a notebook that was just orphaned
    from one that was never paired, and tracked notebooks are deliberately unpaired,
    so the distinction has to come from the diff.
    """
    repo = _repo_with_a_paired_notebook(tmp_path)
    _git(repo, "rm", "-q", "chapter/demo.py")
    _git(repo, "commit", "-qm", "drop the source, keep the render")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    orphaned = notebook_provenance.notebooks_orphaned_since("main")

    assert len(orphaned) == 1
    assert "chapter/demo.ipynb" in orphaned[0]
    assert "chapter/demo.py" in orphaned[0]


def test_moving_the_source_leaves_the_notebook_orphaned(tmp_path, monkeypatch) -> None:
    """The case --no-renames exists for.

    With rename detection on, git reports only the destination and the notebook
    rendered from the old path goes unmentioned.
    """
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
    """git quotes such a path under plain --name-only.

    Splitting on whitespace then tears it into fragments matching no suffix - the
    notebook leaves the gate's scope and passes unchecked, which is the one thing a
    gate must never do.
    """
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


def test_editing_the_paired_py_puts_the_notebook_in_scope(tmp_path, monkeypatch) -> None:
    """Changing the .py is exactly what makes the rendered notebook stale."""
    repo = _repo_with_a_paired_notebook(tmp_path)
    (repo / "chapter" / "demo.py").write_text("# %%\nX = 2\n")
    _git(repo, "commit", "-qam", "edit the source only")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    changed = [
        str(p.relative_to(repo)) for p in notebook_provenance.notebooks_changed_since("main")
    ]

    assert changed == ["chapter/demo.ipynb"]


def test_a_notebook_nobody_touched_is_out_of_scope(tmp_path, monkeypatch) -> None:
    """The whole point of scoping: one stale notebook elsewhere is somebody else's."""
    repo = _repo_with_a_paired_notebook(tmp_path)
    (repo / "unrelated.txt").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "touch nothing paired")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    assert notebook_provenance.notebooks_changed_since("main") == []
    assert notebook_provenance.notebooks_orphaned_since("main") == []


def test_a_force_push_reverting_a_notebook_is_seen_only_against_the_previous_tip(
    tmp_path, monkeypatch
) -> None:
    """The case ``--no-merge-base`` exists for.

    A pull request asks what a branch adds on top of its base, so it diffs the merge
    base. A push asks what the published tree *becomes*, and a force-push can revert a
    notebook relative to the tip it replaces without the merge base ever seeing it: the
    merge base of the new tip and the old one is their common ancestor, where the revert
    has not happened yet.
    """
    repo = _repo_with_a_paired_notebook(tmp_path)
    _git(repo, "checkout", "-q", "main")
    fork = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "chapter" / "demo.py").write_text("# %%\nX = 2\n")
    _git(repo, "commit", "-qam", "the edit that was published")
    previous_tip = _git(repo, "rev-parse", "HEAD").strip()

    # The force-push: main is rewound past the edit and re-grown, so the new tip does
    # not descend from the old one and carries demo.py back at X = 1.
    _git(repo, "reset", "-q", "--hard", fork)
    (repo / "unrelated.txt").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "rewrite main without the edit")
    monkeypatch.setattr(notebook_provenance, "REPO_ROOT", repo)

    merge_base_scope = [
        str(p.relative_to(repo)) for p in notebook_provenance.notebooks_changed_since(previous_tip)
    ]
    assert merge_base_scope == [], "the revert is invisible from the common ancestor"

    against_the_tip = [
        str(p.relative_to(repo))
        for p in notebook_provenance.notebooks_changed_since(previous_tip, merge_base=False)
    ]
    assert against_the_tip == ["chapter/demo.ipynb"]
