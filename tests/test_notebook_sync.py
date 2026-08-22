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
    stale, testmode, contradicted, _unverified, _alt_only = check_all(strict=False)
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


def test_editing_the_literal_part_of_a_computed_alt_is_stale(tmp_path, monkeypatch) -> None:
    """A computed alt is not blanked, so its literal parts stay in the compared AST dump."""
    old = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the leader is {leader}")\n'
    new = '# %%\nfig = build()\nshow_plotly_with_alt(fig, f"the winner is {leader}")\n'
    cell = {
        "cell_type": "code",
        "metadata": {},
        "source": 'fig = build()\nshow_plotly_with_alt(fig, f"the winner is {leader}")\n',
        "outputs": [
            {
                "output_type": "display_data",
                "data": {"image/png": "iVBORw0KGgo="},
                "metadata": {"image/png": {"alt": "the winner is ridge"}},
            }
        ],
    }
    assert not _drift(tmp_path, monkeypatch, old, new, _notebook([cell]))


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
