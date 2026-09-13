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
# <script>", or a statement that this dataset's downloader writes no profile. Saying both
# is what the etfs card did - it told a reader with no profile that the downloader writes
# none, two cells above a section explaining that re-running that downloader refreshes it.
# One of the two is always wrong and a reader cannot tell which.
#
# This checks the contradiction and not the underlying claim, deliberately. Whether a
# script writes a profile is not decidable from its own text: data/etfs/market/download.py
# never names save_dataset_profile and writes one anyway, because ETFDataManager.save
# calls generate_profile and save_profile in ml4t.data.storage.data_profile. A grep-based
# check called that card wrong once already. Resolving it properly means following
# first-party delegation through an installed package, which a static test should not
# pretend to do; the honest check is the one on the card's own two statements.
_WRITTEN_BY = re.compile(r"Written by:\s*python3?\s+([\w./-]+\.py)")
_NO_WRITER = re.compile(r"downloader does not write one", re.IGNORECASE)


@pytest.mark.parametrize("card", CARDS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_a_card_does_not_both_credit_a_writer_and_deny_one(card: Path):
    text = card.read_text()
    credited = sorted(set(_WRITTEN_BY.findall(text)))
    denies = bool(_NO_WRITER.search(text))
    assert not (credited and denies), (
        f"{card.relative_to(REPO_ROOT)} tells the reader both that the profile is written "
        f"by {', '.join(credited)} and that this dataset's downloader writes none. "
        f"One of the two is wrong."
    )


def test_both_forms_are_in_use_so_the_check_is_not_vacuous():
    """If every card used one form, the contradiction above could never arise."""
    texts = [c.read_text() for c in CARDS]
    assert any(_WRITTEN_BY.search(t) for t in texts), "no card credits a writer"
    assert any(_NO_WRITER.search(t) for t in texts), "no card denies a writer"
