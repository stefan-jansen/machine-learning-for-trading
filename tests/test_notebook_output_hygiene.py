"""Guards on what committed notebooks expose to readers.

Three hygiene defects have reached readers from committed ``.ipynb`` files:

* machine-specific absolute paths (``/home/<user>/...``) baked into cell
  outputs and papermill metadata,
* an empty ``tags: []`` stamped on every cell by papermill, which desynced the
  notebook from its jupytext-paired ``.py`` and made JupyterLab refuse to open
  it (public issue #372), and
* a plotly figure serialized as a bare ``application/json`` payload with no
  ``image/png`` (and no ``text/html``) sibling, which GitHub's notebook viewer
  cannot render: the reader sees a collapsed JSON tree instead of the chart.
  This happens when a notebook is executed with ``PLOTLY_RENDERER=json`` (the
  headless/CI recipe) instead of the default ``plotly_mimetype+png`` renderer.

Each test scans every tracked ``.ipynb`` and names the script that fixes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

from sanitize_notebook_paths import _iter_notebooks, sanitize_notebook  # noqa: E402
from strip_empty_cell_tags import paired_py_has_fossil, strip_text  # noqa: E402


def test_strip_empty_cell_tags_handles_a_metadata_only_tag():
    raw = '{\n "metadata": {\n  "tags": []\n },\n "source": []\n}\n'
    cleaned, count = strip_text(raw)
    assert count == 1
    assert '"tags"' not in cleaned


def test_no_machine_specific_paths_in_committed_notebooks() -> None:
    """Outputs and metadata only. A path in `source` may be load bearing.

    A leak the notebook's source also contains counts too. The sanitizer skips
    those rather than rewriting them, so leaving them out of this count would
    let a leak the reader can plainly see sit in a committed output while the
    gate reported the repository clean - the tool declining to fix something is
    not the same as there being nothing to fix.
    """
    offenders: list[str] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        _, n, skipped = sanitize_notebook(raw)
        if n or skipped:
            note = f"{n}" + (f", {len(skipped)} the sanitizer cannot rewrite" if skipped else "")
            offenders.append(f"{nb.relative_to(REPO_ROOT)} ({note})")
    assert not offenders, (
        "Notebooks leak machine-specific absolute paths in their committed "
        "outputs/metadata. Run `uv run python .github/scripts/sanitize_notebook_paths.py` "
        "to fix; a leak it reports as unrewritable has to be removed by hand or by "
        "re-executing the notebook:\n  " + "\n  ".join(offenders)
    )


def test_the_sanitizer_reports_a_string_it_cannot_safely_rewrite() -> None:
    """A leak the source shares is skipped, not rewritten, and it is named.

    The raw-text edit is what keeps the diff to the replaced paths, and it is
    also what makes a string that appears in both places ambiguous. Skipping is
    the safe answer; going quiet about it is not.
    """
    shared = "/home/someone/ml4t/code/data/prices.parquet"
    raw = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": [f'pl.read_parquet("{shared}")'],
                    "outputs": [{"output_type": "stream", "text": [shared]}],
                }
            ],
            "metadata": {},
        }
    )

    new, replaced, skipped = sanitize_notebook(raw)

    assert skipped == [shared]
    assert replaced == 0
    assert new == raw


# Debt list for notebooks still carrying the empty-tag fossil. Emptied when the
# case studies shipped: every entry below was a code-repo artifact that the
# released notebooks do not carry, so the list cleared rather than shrank one at
# a time. The list must only ever shrink, which the second test below enforces.
KNOWN_DESYNCED: frozenset[str] = frozenset()


def _empty_tag_offenders() -> dict[str, int]:
    """{relative path: count} for notebooks whose paired .py lacks the empty tags."""
    out: dict[str, int] = {}
    for nb in _iter_notebooks():
        if paired_py_has_fossil(nb):
            continue  # pair agrees; stripping one side is what would break it
        _, n = strip_text(nb.read_text(encoding="utf-8"))
        if n:
            out[str(nb.relative_to(REPO_ROOT))] = n
    return out


def test_no_empty_cell_tags_in_committed_notebooks() -> None:
    """Empty `tags: []` desyncs a notebook from its .py, so JupyterLab won't open it."""
    offenders = [f"{p} ({n})" for p, n in _empty_tag_offenders().items() if p not in KNOWN_DESYNCED]
    assert not offenders, (
        "Notebooks carry empty `tags: []` cell metadata their paired .py lacks, so "
        "JupyterLab shows a 'File Load Error' instead of the notebook (cf. public "
        "#372). Run `uv run python .github/scripts/strip_empty_cell_tags.py` to fix:\n  "
        + "\n  ".join(offenders)
    )


def test_known_desynced_list_has_no_stale_entries() -> None:
    """The debt list must only shrink: a fixed notebook has to leave it.

    Entries whose notebook is absent are ignored, not stale: this file is mirrored
    to the public repo, which ships only a subset of the case studies.
    """
    offenders = _empty_tag_offenders()
    stale = sorted(e for e in KNOWN_DESYNCED - set(offenders) if (REPO_ROOT / e).exists())
    assert not stale, (
        "These notebooks are listed in KNOWN_DESYNCED but are now clean. Remove them "
        "from the list in this file so it cannot silently mask a regression:\n  "
        + "\n  ".join(stale)
    )


# Notebooks whose committed outputs carry a plotly figure as a bare
# `application/json` payload with no `image/png`/`text/html` sibling, so GitHub
# shows a JSON tree instead of the chart. Each was executed under
# `PLOTLY_RENDERER=json`; the fix is to re-execute in its documented environment
# with the DEFAULT renderer (`utils.__init__` sets `plotly_mimetype+png`; kaleido
# is present in every relevant env, including the benchmark image), which emits
# the `image/png` GitHub needs. The list must only ever shrink, which the
# companion test below enforces. `_archive/` notebooks are not shipped to readers
# but are tracked here for consistency with the sibling debt list above.
KNOWN_UNRENDERABLE = frozenset(
    {
        "case_studies/crypto_perps_funding/_archive/11_autoencoder.ipynb",
        "case_studies/us_equities_panel/20_strategy_analysis.ipynb",
    }
)


def _is_plotly_figure_spec(payload: object) -> bool:
    """A plotly figure serialized to JSON is a dict carrying `data` and `layout`."""
    return isinstance(payload, dict) and "data" in payload and "layout" in payload


def _unrenderable_plotly_offenders() -> dict[str, int]:
    """{relative path: figure count} for plotly figures GitHub cannot render.

    A figure output is unrenderable if it is a plotly figure (a `data`+`layout`
    spec under `application/json`, or an `application/vnd.plotly.v1+json` mime)
    that carries neither an `image/png` nor a `text/html` sibling to fall back on.
    """
    out: dict[str, int] = {}
    for nb_path in _iter_notebooks():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        count = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" in data or "text/html" in data:
                    continue
                is_plotly = "application/vnd.plotly.v1+json" in data or _is_plotly_figure_spec(
                    data.get("application/json")
                )
                if is_plotly:
                    count += 1
        if count:
            out[str(nb_path.relative_to(REPO_ROOT))] = count
    return out


def test_no_unrenderable_plotly_figures_in_committed_notebooks() -> None:
    """Plotly figures stored as bare `application/json` do not render on GitHub."""
    offenders = [
        f"{p} ({n})"
        for p, n in _unrenderable_plotly_offenders().items()
        if p not in KNOWN_UNRENDERABLE
    ]
    assert not offenders, (
        "Notebooks commit plotly figures as a bare `application/json` payload with "
        "no `image/png`, so GitHub renders a JSON tree instead of the chart. Re-execute "
        "the notebook in its documented environment with the DEFAULT plotly renderer "
        "(do NOT set `PLOTLY_RENDERER=json`), which emits the `image/png` via kaleido:\n  "
        + "\n  ".join(offenders)
    )


def test_known_unrenderable_list_has_no_stale_entries() -> None:
    """The debt list must only shrink: a fixed notebook has to leave it.

    Entries whose notebook is absent are ignored, not stale: this file is mirrored
    to the public repo, which ships only a subset of the case studies.
    """
    offenders = _unrenderable_plotly_offenders()
    stale = sorted(e for e in KNOWN_UNRENDERABLE - set(offenders) if (REPO_ROOT / e).exists())
    assert not stale, (
        "These notebooks are listed in KNOWN_UNRENDERABLE but their plotly figures now "
        "render. Remove them from the list in this file so it cannot silently mask a "
        "regression:\n  " + "\n  ".join(stale)
    )
