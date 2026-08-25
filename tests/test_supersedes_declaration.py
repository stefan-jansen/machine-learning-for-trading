"""The notebook parameter that says which causal identity a refit retires.

`_enforce_causal_supersedes` refuses to register a second current identity for a
label, and refuses at write time - after the DML fit and every placebo refit have
been paid for. The only way a notebook can satisfy that refusal is to declare the
predecessor, so the parameter has to exist, reach the request, and reject a
declaration that names nothing the notebook fits before the fit starts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from case_studies.research import supersedes_for

REPO = Path(__file__).resolve().parent.parent

# Every notebook that opens a causal request. A notebook here without the parameter
# cannot answer the write-time refusal and would lose its fit to it.
CAUSAL_NOTEBOOKS = [
    "case_studies/cme_futures/11_causal_dml.py",
    "case_studies/crypto_perps_funding/11_causal_dml.py",
    "case_studies/etfs/12_causal_dml.py",
    "case_studies/fx_pairs/11_causal_dml.py",
    "case_studies/us_equities_panel/14_causal_dml.py",
    # Numbered one lower than the other eight: this case study has no stage-04
    # notebook, so its causal notebook is 09 rather than 11.
    "case_studies/us_firm_characteristics/09_causal_dml.py",
]

# Four more notebooks open a causal request through register_causal_run and do not yet
# declare the parameter, so they cannot answer the write-time refusal:
# nasdaq100_microstructure/12, sp500_options/10, sp500_equity_option_analytics/12 and
# us_firm_characteristics/09. Each is owned by an active branch and takes the same patch
# from its own side; add the path here in the same commit that adds the parameter.


class TestTheDeclarationParser:
    def test_an_empty_declaration_retires_nothing(self) -> None:
        assert supersedes_for("", "fwd_ret_5d") is None
        assert supersedes_for(None, "fwd_ret_5d") is None
        assert supersedes_for("   ", "fwd_ret_5d") is None

    def test_a_bare_hash_is_the_single_label_form(self) -> None:
        assert supersedes_for("ab12cd34", "fwd_ret_5d", labels=["fwd_ret_5d"]) == "ab12cd34"

    def test_a_bare_hash_is_refused_when_the_notebook_fits_several(self) -> None:
        # One hash cannot say which of three identities is being retired, and guessing
        # would retire the wrong one.
        with pytest.raises(ValueError, match="bare hash"):
            supersedes_for("ab12cd34", "a", labels=["a", "b", "c"])

    def test_a_mapping_selects_this_label(self) -> None:
        text = '{"a": "hash_a", "b": "hash_b"}'
        assert supersedes_for(text, "a", labels=["a", "b"]) == "hash_a"
        assert supersedes_for(text, "b", labels=["a", "b"]) == "hash_b"

    def test_a_label_absent_from_the_mapping_retires_nothing(self) -> None:
        assert supersedes_for('{"a": "hash_a"}', "b", labels=["a", "b"]) is None

    def test_a_label_the_notebook_does_not_fit_is_a_typo_not_a_no_op(self) -> None:
        # Silently ignoring it means the run pays for the fit and then fails at
        # registration for the identity it meant to retire.
        with pytest.raises(ValueError, match="does not fit"):
            supersedes_for('{"typo": "hash"}', "a", labels=["a", "b"])

    def test_malformed_json_names_itself(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            supersedes_for('{"a": ', "a", labels=["a"])


@pytest.mark.parametrize("path", CAUSAL_NOTEBOOKS)
def test_the_notebook_declares_the_parameter(path: str) -> None:
    source = (REPO / path).read_text()
    assert "SUPERSEDES_CAUSAL" in source, f"{path} cannot answer the write-time refusal"


# The two shapes a notebook can use to reach the write, and the keyword each takes.
# `study.causal(supersedes=...)` is the request path; `register_causal_run(
# supersedes_hash=...)` is the direct one, and it is the function that actually
# performs the refusal - `_enforce_causal_supersedes` is called from inside it. A
# notebook on the direct path answers the refusal just as completely, so requiring the
# request path here would fail a notebook that is correct.
_SUPERSEDES_CALLS = {"causal": "supersedes", "register_causal_run": "supersedes_hash"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


@pytest.mark.parametrize("path", CAUSAL_NOTEBOOKS)
def test_the_parameter_reaches_the_request(path: str) -> None:
    """Declaring it and not passing it is the same as not declaring it."""
    source = (REPO / path).read_text()
    tree = ast.parse(source)
    passed = [
        keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in _SUPERSEDES_CALLS
        for keyword in node.keywords
        if keyword.arg == _SUPERSEDES_CALLS[_call_name(node)]
    ]
    assert passed, (
        f"{path} declares SUPERSEDES_CAUSAL but nothing takes it: no study.causal() call "
        "with supersedes= and no register_causal_run() call with supersedes_hash="
    )


@pytest.mark.parametrize("path", CAUSAL_NOTEBOOKS)
def test_the_parameter_is_in_the_parameters_cell(path: str) -> None:
    """Papermill only overrides names in the tagged cell; elsewhere it is a constant."""
    source = (REPO / path).read_text()
    cells = source.split("\n# %%")
    tagged = [cell for cell in cells if cell.startswith(' tags=["parameters"]')]
    assert tagged, f"{path} has no parameters cell"
    assert any("SUPERSEDES_CAUSAL" in cell for cell in tagged), (
        f"{path} declares SUPERSEDES_CAUSAL outside the parameters cell, so papermill "
        "cannot set it and the declaration can never be supplied"
    )
