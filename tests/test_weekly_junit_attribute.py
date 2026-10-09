"""The weekly issue opener must read the testcase ``name``, not ``classname``.

``/name="/`` matches the ``name="`` inside ``classname="..."``. Every failure
was filed as ``weekly-flake: tests.test_chapter_notebooks`` (#1046) or
``external-drift: tests.test_external_drift``, so the notebook or source that
actually failed never made it into the title.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "weekly-external.yml"

CHAPTER_BLOCK = (
    '<testcase classname="tests.test_chapter_notebooks" '
    'name="test_chapter_notebook[22_rag_financial_research::02_domain_embeddings_comparison.py]" '
    'time="1.2"><failure message="Failed">huggingface</failure></testcase>'
)
DRIFT_BLOCK = (
    '<testcase classname="tests.test_external_drift" '
    'name="test_etf_yahoo_reachable" time="0.4">'
    '<failure message="schema">moved</failure></testcase>'
)


def _attr_pattern() -> re.Pattern[str]:
    text = WORKFLOW.read_text()
    # The JS source is `\\b` inside a template literal, which is one backslash
    # in the file. Both parsers have to use that form.
    assert text.count(r'new RegExp(`\\b${name}="([^"]*)"`)') == 2
    return re.compile(r'\bname="([^"]*)"')


def test_the_chapter_name_is_the_parametrized_test_not_the_module() -> None:
    match = _attr_pattern().search(CHAPTER_BLOCK)
    assert match is not None
    name = match.group(1)
    assert name.startswith("test_chapter_notebook[")
    nb_id = re.sub(r"^test_[a-z_]+\[", "", name).removesuffix("]")
    assert nb_id == "22_rag_financial_research::02_domain_embeddings_comparison.py"
    assert "test_chapter_notebooks" not in nb_id


def test_the_drift_name_is_the_source_test_not_the_module() -> None:
    match = _attr_pattern().search(DRIFT_BLOCK)
    assert match is not None
    assert match.group(1) == "test_etf_yahoo_reachable"


def test_a_bare_name_pattern_is_what_filed_the_wrong_issue() -> None:
    """Documents the defect: the first ``name="`` in the tag is inside ``classname``."""
    bare = re.search(r'name="([^"]*)"', CHAPTER_BLOCK)
    assert bare is not None
    assert bare.group(1) == "tests.test_chapter_notebooks"
