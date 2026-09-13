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


# A card's profile section offers the reader one of two things: "Written by: python
# <script>", or a statement that this dataset's downloader writes no profile. The first
# is a claim about what that script does, and a script that exists but never calls
# `save_dataset_profile` leaves the reader exactly where `generate_profiles.py` did -
# running something that does not produce the file they asked for. Three cards carried
# that form wrongly (etfs, and both microstructure datasets), and the test above passed
# on all three because the scripts are real.
_WRITTEN_BY = re.compile(r"Written by:\s*python3?\s+([\w./-]+\.py)")


def _resolve(card: Path, script: str) -> Path | None:
    for candidate in (REPO_ROOT / script, card.parent / script):
        if candidate.exists():
            return candidate
    return None


@pytest.mark.parametrize("card", CARDS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_a_card_credits_a_profile_only_to_a_script_that_writes_one(card: Path):
    credited = sorted(set(_WRITTEN_BY.findall(card.read_text())))
    wrong = []
    for script in credited:
        path = _resolve(card, script)
        if path is None or "save_dataset_profile" not in path.read_text():
            wrong.append(script)
    assert not wrong, (
        f"{card.relative_to(REPO_ROOT)} says the profile is written by "
        f"{', '.join(wrong)}, which never calls save_dataset_profile. Either credit the "
        f"script that does, or say the downloader writes no profile."
    )


def test_some_card_credits_a_writer_so_the_check_is_not_vacuous():
    """If no card used the "Written by" form, the test above could never fail."""
    crediting = [c for c in CARDS if _WRITTEN_BY.search(c.read_text())]
    assert crediting, "no card uses the 'Written by:' form, so the credit check is vacuous"
