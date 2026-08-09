"""Guards on what committed notebooks expose to readers.

Four hygiene defects have reached readers from committed ``.ipynb`` files:

* machine-specific absolute paths baked into cell outputs and papermill
  metadata - ``/home/<user>/...``, and a scratch root under ``/tmp``,
* an empty ``tags: []`` stamped on every cell by papermill, which desynced the
  notebook from its jupytext-paired ``.py`` and made JupyterLab refuse to open
  it (public issue #372), and
* a plotly figure serialized as a bare ``application/json`` payload with no
  ``image/png`` (and no ``text/html``) sibling, which GitHub's notebook viewer
  cannot render: the reader sees a collapsed JSON tree instead of the chart.
  This happens when a notebook is executed with ``PLOTLY_RENDERER=json`` (the
  headless/CI recipe) instead of the default ``plotly_mimetype+png`` renderer, and
* a figure destroyed by the sanitizer above, which deleted a chance ``/app/`` out
  of a base64 PNG payload and left an encoding that no longer decodes.

Each test scans every tracked ``.ipynb`` and names the script that fixes it.
"""

from __future__ import annotations

import base64
import binascii
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".github" / "scripts"))

from sanitize_notebook_paths import (  # noqa: E402
    BINARY_MIME,
    _iter_notebooks,
    sanitize_notebook,
    sanitize_text,
)
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


def test_a_scratch_root_under_tmp_is_rewritten() -> None:
    """An agent scratchpad and a staging notebook executed from /tmp are both leaks."""
    scratchpad = (
        "/tmp/claude-1000/-home-someone-ml4t-agents/"
        "adaaf42e-c336-4616-bea6-f139792daf17/scratchpad/nb.ipynb"
    )
    assert sanitize_text(scratchpad)[0] == "~/scratch/nb.ipynb"
    assert sanitize_text("/tmp/dpgan_final_out.ipynb")[0] == "~/scratch/dpgan_final_out.ipynb"


def test_the_documented_test_output_directory_is_not_rewritten() -> None:
    """`/tmp/ml4t-test-output` is real configuration, not one machine's layout.

    `AGENTS.md` ("Output isolation") makes tests write there, so a notebook
    printing it is telling the reader where its output went. Rewriting it would
    repeat the mistake the raw-text sanitizer made with the `/app` mount path in
    `02_financial_data_universe/16_provider_comparison`.
    """
    for path in (
        "/tmp/ml4t-test-output/ch04_kalshi/kalshi_features.parquet",
        "/tmp/ml4t-test-output-ch15/ch15_momentum_causal_trading/artifacts.json",
    ):
        assert sanitize_text(path) == (path, 0)


def test_an_ipython_cell_path_is_not_rewritten() -> None:
    """`/tmp/ipykernel_<pid>/<hash>.py` is what IPython calls a cell in any kernel.

    It carries a process id, not a user or a directory layout, and it appears
    inside tracebacks and warnings where the line number is the point. Rewriting
    the root would corrupt a location a reader may need to follow.
    """
    frame = "/tmp/ipykernel_790523/2252327757.py:76: FutureWarning"
    assert sanitize_text(frame) == (frame, 0)


def test_an_already_rewritten_path_is_not_rewritten_again() -> None:
    """The `/tmp` rules match a filesystem root, not the segment `tmp` anywhere.

    `~/.claude/jobs/<id>/tmp/run.ipynb` is what the `/home/<user>/` rule leaves
    behind. An unanchored `/tmp/` rule would splice a second `~` into the middle
    of it, which is why the rules carry a lookbehind.
    """
    nested = "~/.claude/jobs/7c96381e/tmp/dpgan_final_out.ipynb"
    assert sanitize_text(nested) == (nested, 0)


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


def test_the_sanitizer_leaves_an_image_payload_alone_when_it_encodes_app() -> None:
    """base64 contains ``/app/`` by chance, and deleting it destroys the figure.

    Built from the payload that was actually damaged: cell 33 of
    ``case_studies/etfs/05_evaluation`` carried a 244,684-character PNG holding one
    chance ``/app/``, and removing it left 244,679 characters, which is not a
    multiple of four and does not decode. Nothing raised - the page simply showed a
    broken image. The header is a real PNG so the payload is valid before the run,
    and the padding is chosen to keep the length divisible by four.
    """
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
    payload = png + "/app/" + "A" * 7
    assert len(payload) % 4 == 0
    base64.b64decode(payload, validate=True)

    raw = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["fig.show()"],
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {"image/png": payload, "text/plain": ["/app/figure.py"]},
                            "metadata": {},
                        }
                    ],
                }
            ],
            "metadata": {},
        }
    )

    new, replaced, skipped = sanitize_notebook(raw)

    assert skipped == []
    # The text/plain sibling is a genuine container path and is still rewritten.
    assert replaced == 1
    rewritten = json.loads(new)["cells"][0]["outputs"][0]["data"]
    assert rewritten["text/plain"] == ["figure.py"]
    assert rewritten["image/png"] == payload
    base64.b64decode(rewritten["image/png"], validate=True)


def test_the_sanitizer_leaves_an_image_payload_alone_when_it_encodes_tmp() -> None:
    """The `/tmp` rules carry the same exposure as the `/app/` one, by the same alphabet.

    `t`, `m` and `p` are as much a part of base64 as `a` and `p` are, and the
    filesystem-root lookbehind does not help: `+` and `/` are both in the alphabet
    and both satisfy it. So `/tmp/` occurs inside a long payload by chance exactly
    as `/app/` does, and rewriting it to `~/scratch/` would corrupt the image just
    as deleting five characters did. What prevents it is the MIME filter, not the
    anchor - which is why this is pinned separately rather than assumed from the
    `/app/` case.
    """
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
    payload = png + "A+/tmp/dpgan" + "A" * 4
    assert len(payload) % 4 == 0
    base64.b64decode(payload, validate=True)
    # The same string outside a payload is a leak, so the rule does fire on it.
    assert sanitize_text("/tmp/dpgan")[1] == 1

    raw = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["fig.show()"],
                    "outputs": [
                        {
                            "output_type": "display_data",
                            "data": {"image/png": payload, "text/plain": ["/tmp/dpgan.ipynb"]},
                            "metadata": {},
                        }
                    ],
                }
            ],
            "metadata": {},
        }
    )

    new, replaced, skipped = sanitize_notebook(raw)

    assert skipped == []
    assert replaced == 1
    rewritten = json.loads(new)["cells"][0]["outputs"][0]["data"]
    assert rewritten["text/plain"] == ["~/scratch/dpgan.ipynb"]
    assert rewritten["image/png"] == payload
    base64.b64decode(rewritten["image/png"], validate=True)


def test_no_committed_notebook_carries_an_image_that_stopped_decoding() -> None:
    """The damage the rule above caused is detectable, so it is checked for."""
    broken: list[str] = []
    for nb in _iter_notebooks():
        parsed = json.loads(nb.read_text(encoding="utf-8"))
        for index, cell in enumerate(parsed.get("cells", [])):
            for output in cell.get("outputs", []):
                for mime, value in (output.get("data") or {}).items():
                    if mime not in BINARY_MIME:
                        continue
                    text = value if isinstance(value, str) else "".join(value)
                    try:
                        base64.b64decode(text, validate=True)
                    except (binascii.Error, ValueError):
                        broken.append(f"{nb.relative_to(REPO_ROOT)} cell {index} {mime}")
    assert not broken, (
        f"{len(broken)} committed image payload(s) no longer decode: {broken[:5]}. "
        "Re-run the notebook; the payload cannot be repaired in place."
    )


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
