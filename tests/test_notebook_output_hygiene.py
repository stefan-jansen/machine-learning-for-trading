"""Guards on what committed notebooks expose to readers.

Two hygiene defects have reached readers from committed ``.ipynb`` files:

* machine-specific absolute paths (``/home/<user>/...``) baked into cell
  outputs and papermill metadata, and
* an empty ``tags: []`` stamped on every cell by papermill, which desynced the
  notebook from its jupytext-paired ``.py`` and made JupyterLab refuse to open
  it (public issue #372).

Each test scans every tracked ``.ipynb`` and names the script that fixes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sanitize_notebook_paths import (  # noqa: E402
    _iter_notebooks,
    sanitize_notebook_text,
)
from strip_empty_cell_tags import paired_py_has_fossil, strip_text  # noqa: E402


def test_no_machine_specific_paths_in_committed_notebooks() -> None:
    offenders: list[str] = []
    for nb in _iter_notebooks():
        raw = nb.read_text(encoding="utf-8")
        _, n = sanitize_notebook_text(raw)
        if n:
            offenders.append(f"{nb.relative_to(REPO_ROOT)} ({n})")
    assert not offenders, (
        "Notebooks leak machine-specific absolute paths in their committed "
        "outputs/metadata. Run `uv run python scripts/sanitize_notebook_paths.py` "
        "to fix:\n  " + "\n  ".join(offenders)
    )


def test_path_sanitizer_does_not_rewrite_cell_source() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ['path = "/home/reader/ml4t/code/data/file.parquet"\n'],
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["/home/reader/ml4t/code/data/file.parquet\n"],
                    }
                ],
            }
        ],
        "metadata": {},
    }

    clean, count = sanitize_notebook_text(json.dumps(notebook))
    clean_notebook = json.loads(clean)

    assert count == 1
    assert clean_notebook["cells"][0]["source"] == notebook["cells"][0]["source"]
    assert clean_notebook["cells"][0]["outputs"][0]["text"] == ["data/file.parquet\n"]


# Notebooks still carrying the fossil, all in chapters not yet shipped to readers
# (case studies -> Beat 5+). They are desynced for additional reasons too, so the
# empty tags cannot be stripped in isolation: doing so is churn that leaves the
# notebook just as unopenable. Clear these before the beats that ship them; the
# list must only ever shrink, which the second test below enforces.
KNOWN_DESYNCED = frozenset(
    {
        "case_studies/cme_futures/10a_pca.ipynb",
        "case_studies/cme_futures/10b_stochastic_discount_factor.ipynb",
        "case_studies/crypto_perps_funding/05_evaluation.ipynb",
        "case_studies/crypto_perps_funding/_archive/11_autoencoder.ipynb",
        "case_studies/etfs/11a_pca.ipynb",
        "case_studies/etfs/11b_ipca.ipynb",
        "case_studies/etfs/11c_conditional_autoencoder.ipynb",
        "case_studies/etfs/11d_stochastic_discount_factor.ipynb",
        "case_studies/etfs/11e_supervised_autoencoder.ipynb",
        "case_studies/fx_pairs/06_linear.ipynb",
        "case_studies/nasdaq100_microstructure/05_evaluation.ipynb",
        "case_studies/sp500_equity_option_analytics/05_evaluation.ipynb",
        "case_studies/sp500_equity_option_analytics/06_linear.ipynb",
        "case_studies/sp500_equity_option_analytics/08_tabular_dl.ipynb",
        "case_studies/sp500_equity_option_analytics/11a_pca.ipynb",
        "case_studies/sp500_equity_option_analytics/11b_ipca.ipynb",
        "case_studies/sp500_equity_option_analytics/11c_conditional_autoencoder.ipynb",
        "case_studies/sp500_equity_option_analytics/11d_stochastic_discount_factor.ipynb",
        "case_studies/sp500_equity_option_analytics/11e_supervised_autoencoder.ipynb",
        "case_studies/sp500_options/01_feasibility_analysis.ipynb",
        "case_studies/sp500_options/05_evaluation.ipynb",
        "case_studies/us_equities_panel/04_model_based_features.ipynb",
        "case_studies/us_equities_panel/05_evaluation.ipynb",
        "case_studies/us_firm_characteristics/04_evaluation.ipynb",
        "case_studies/us_firm_characteristics/08a_ipca.ipynb",
        "case_studies/us_firm_characteristics/08b_conditional_autoencoder.ipynb",
        "case_studies/us_firm_characteristics/08c_stochastic_discount_factor.ipynb",
        "case_studies/us_firm_characteristics/08d_supervised_autoencoder.ipynb",
    }
)


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
        "#372). Run `uv run python scripts/strip_empty_cell_tags.py` to fix:\n  "
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
