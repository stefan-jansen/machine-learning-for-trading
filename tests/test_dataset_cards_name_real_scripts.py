"""Every script a dataset card tells a reader to run must exist in the repository.

The nine `data/**/dataset_card.py` notebooks each end with a section that either
prints a dataset's profile or, when there is none, tells the reader how to make
one. Eight of them named `generate_profiles.py`, a script that has never existed
in this repository's history, and the instruction was shown precisely to the
reader who had no profile and wanted one (ml4t/agent-workspace#1149).

Nothing catches that: the string is in an `else` arm, so it renders only when the
profile is absent, and a card is not executed in CI. It is checkable against the
working tree, so check it.

Scope is the commands a reader is told to type, taken from both string literals
and fenced code blocks. It resolves each `.py` argument against the repository
root and against the card's own directory, and runs nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS = sorted((REPO_ROOT / "data").rglob("dataset_card.py"))

# `python <script>.py`, with any flags between, in a shell line or a string literal.
_PYTHON_INVOCATION = re.compile(r"python3?\s+(?:-\w+\s+)*([\w./-]+\.py)\b")


def _named_scripts(card: Path) -> set[str]:
    return set(_PYTHON_INVOCATION.findall(card.read_text()))


def test_there_are_cards_to_check():
    """A zero-length corpus would make every assertion below vacuous."""
    assert len(CARDS) >= 9, f"expected the nine dataset cards, found {len(CARDS)}"


@pytest.mark.parametrize("card", CARDS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_script_a_card_names_exists(card: Path):
    missing = [
        script
        for script in sorted(_named_scripts(card))
        if not (REPO_ROOT / script).exists() and not (card.parent / script).exists()
    ]
    assert not missing, (
        f"{card.relative_to(REPO_ROOT)} tells the reader to run "
        f"{', '.join(missing)}, which is not in the repository"
    )
