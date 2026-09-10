"""Unit tests for the Chapter 24 research-agent helpers.

``extract_key_findings`` reads the items a model enumerated in its rationale. Models
pick the format themselves, and the two shapes that reach it - one item per line, and
``(1) ... (2) ...`` run together inside a sentence - are pinned here along with the
abbreviation case that decides where the last inline item ends.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "24_autonomous_agents"))

from agent_research import extract_key_findings  # noqa: E402


def test_line_leading_bullets_and_numbers_win():
    rationale = "Reasons:\n- Oil prices rose\n2) Yields increased\n* Growth slowed"
    assert extract_key_findings(rationale) == [
        "Oil prices rose",
        "Yields increased",
        "Growth slowed",
    ]


def test_inline_markers_are_split_into_items():
    rationale = "Key factors: (1) Oil prices rose; (2) Yields increased; (3) Growth slowed."
    assert extract_key_findings(rationale) == [
        "Oil prices rose",
        "Yields increased",
        "Growth slowed",
    ]


def test_the_last_inline_item_stops_at_its_own_sentence():
    rationale = (
        "Key factors: (1) Oil prices rose; (2) Growth slowed. "
        "However, uncertainty remains around the labor market."
    )
    assert extract_key_findings(rationale) == ["Oil prices rose", "Growth slowed"]


def test_an_abbreviation_does_not_end_the_last_item():
    rationale = "Reasons: (1) Oil prices rose; (2) U.S. Treasury yields increased."
    assert extract_key_findings(rationale) == [
        "Oil prices rose",
        "U.S. Treasury yields increased",
    ]


def test_a_lone_parenthesised_digit_is_not_a_list():
    assert extract_key_findings("The Fed cut once (1) last year, then held.") == []


def test_a_rationale_that_enumerates_nothing_yields_nothing():
    assert extract_key_findings("Rates are likely to stay where they are.") == []
