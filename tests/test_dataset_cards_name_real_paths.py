"""A dataset card must name a script that exists and a profile path the code agrees with.

The `data/**/dataset_card.py` notebooks each end with a section that prints the
dataset's profile, or says where one would be. Three ways that goes wrong, all of
them invisible at runtime because the card is not executed in CI and because the
wrong branch is the one a reader without the data sees:

- The card tells the reader to run a script the repository does not contain.
- The card builds a profile path by hand. `ml4t.data.storage.data_profile` writes
  `<stem>_profile.json` beside a file and `_profile.json` inside a directory, and
  never a bare `profile.json`, so a card checking for `profile.json` reports "not
  found" against every dataset forever.
- The card's summary table names a different path from the one its own code builds.

Each check has a companion that fails if its corpus or its pattern stops matching,
so none of them can pass by seeing nothing. Nothing here executes a card or reads a
dataset; all three are decidable from the working tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CARDS = sorted((REPO_ROOT / "data").rglob("dataset_card.py"))

# `python <script>.py`, with any flags between, in a shell line or a string literal.
_PYTHON_INVOCATION = re.compile(r"python3?\s+(?:-\w+\s+)*([\w./-]+\.py)\b")

# A path expression rooted at ML4T_DATA_PATH: the quoted segments that follow it.
_DATA_PATH_EXPR = re.compile(r"ML4T_DATA_PATH\s*((?:/\s*\"[^\"]+\"\s*)+)")
_SEGMENT = re.compile(r"\"([^\"]+)\"")

# A profile path written for the reader in a markdown cell, e.g.
# `# | Profile | `$ML4T_DATA_PATH/etfs/market/etf_universe_profile.json` |`
_PROSE_PROFILE_PATH = re.compile(r"\$ML4T_DATA_PATH/([\w./{}-]*profile\.json)")

# A quoted filename that `get_profile_path` never produces, however the path around it
# is assembled. `_profile.json` and `<stem>_profile.json` are the two it does produce.
_BARE_PROFILE_LITERAL = re.compile(r"\"((?<![\w-])profile\.json)\"")


def _named_scripts(text: str) -> set[str]:
    return set(_PYTHON_INVOCATION.findall(text))


def _code_profile_paths(text: str) -> list[tuple[str, ...]]:
    """The quoted segments of every path the card builds under ML4T_DATA_PATH.

    Both forms count: a literal `.. / "x_profile.json"`, and a directory handed to
    `get_profile_path`, which appends the `_profile.json` itself.
    """
    return [
        tuple(_SEGMENT.findall(match.group(1)))
        for match in _DATA_PATH_EXPR.finditer(text)
        if _SEGMENT.findall(match.group(1))
    ]


def _profile_dir(segments: tuple[str, ...]) -> str:
    """The directory the profile sits in, whichever of the two forms built it."""
    if segments[-1].endswith(".json"):
        segments = segments[:-1]
    return "/".join(segments)


def test_there_are_cards_to_check():
    """A zero-length corpus would make every assertion below vacuous."""
    assert len(CARDS) >= 9, f"expected at least the nine dataset cards, found {len(CARDS)}"


@pytest.mark.parametrize("card", CARDS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_script_a_card_names_exists(card: Path):
    missing = [
        script
        for script in sorted(_named_scripts(card.read_text()))
        if not (REPO_ROOT / script).exists() and not (card.parent / script).exists()
    ]
    assert not missing, (
        f"{card.relative_to(REPO_ROOT)} tells the reader to run "
        f"{', '.join(missing)}, which is not in the repository"
    )


def test_the_invocation_pattern_still_matches_a_real_card():
    """If no card names a script, the check above passes without deciding anything."""
    naming = [c for c in CARDS if _named_scripts(c.read_text())]
    assert naming, "no dataset card names a `python <script>.py` command any more"


@pytest.mark.parametrize("card", CARDS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_card_looks_for_a_profile_the_writer_never_writes(card: Path):
    text = card.read_text()
    bad = [
        "/".join(segments)
        for segments in _code_profile_paths(text)
        if segments[-1].endswith(".json") and not segments[-1].endswith("_profile.json")
    ]
    # A card can also assemble the path from a loop variable, which the expression
    # above cannot follow: `ML4T_DATA_PATH / "factors" / subdir / "profile.json"`.
    # The filename literal is the part that is wrong either way, so check it directly.
    bad += [f'a literal "{name}"' for name in _BARE_PROFILE_LITERAL.findall(text)]
    assert not bad, (
        f"{card.relative_to(REPO_ROOT)} looks for {', '.join(bad)}. "
        "ml4t.data.storage.data_profile.get_profile_path writes `<stem>_profile.json` "
        "beside a file and `_profile.json` inside a directory, so this path is never "
        "written and the card reports 'not found' against every dataset."
    )


def test_the_path_pattern_still_finds_profile_paths_in_the_cards():
    """If no card builds a path, the check above passes without deciding anything."""
    building = [c for c in CARDS if _code_profile_paths(c.read_text())]
    assert len(building) == len(CARDS), (
        f"{len(CARDS) - len(building)} of {len(CARDS)} card(s) build no ML4T_DATA_PATH "
        "path at all; the pattern above no longer matches how the cards are written"
    )


@pytest.mark.parametrize("card", CARDS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_a_profile_path_shown_to_the_reader_matches_the_one_the_code_builds(card: Path):
    text = card.read_text()
    code_dirs = {_profile_dir(segments) for segments in _code_profile_paths(text)}
    if not code_dirs:
        pytest.skip("this card builds no ML4T_DATA_PATH profile path")
    for prose in sorted(set(_PROSE_PROFILE_PATH.findall(text))):
        prose_dir = "/".join(prose.split("/")[:-1])
        assert prose_dir in code_dirs, (
            f"{card.relative_to(REPO_ROOT)} shows the reader "
            f"$ML4T_DATA_PATH/{prose} while its code reads from "
            f"{', '.join(sorted(code_dirs))}"
        )


def test_the_prose_pattern_still_finds_a_path_shown_to_a_reader():
    """If no card shows a path in prose, the check above decides nothing."""
    showing = [c for c in CARDS if _PROSE_PROFILE_PATH.search(c.read_text())]
    assert showing, "no dataset card shows a $ML4T_DATA_PATH profile path to the reader"


@pytest.mark.parametrize(
    ("text", "checker", "expected"),
    [
        (
            "run `python does_not_exist_anywhere.py` first",
            _named_scripts,
            {"does_not_exist_anywhere.py"},
        ),
        (
            'ML4T_DATA_PATH / "factors" / "aqr" / "profile.json"',
            _code_profile_paths,
            [("factors", "aqr", "profile.json")],
        ),
        (
            'ML4T_DATA_PATH / "factors" / subdir / "profile.json"',
            _BARE_PROFILE_LITERAL.findall,
            ["profile.json"],
        ),
        ('ML4T_DATA_PATH / "fx" / "market" / "4h_profile.json"', _BARE_PROFILE_LITERAL.findall, []),
    ],
    ids=["invocation", "hand-built-path", "bare-literal", "bare-literal-accepts-real-name"],
)
def test_each_pattern_sees_the_defect_it_exists_for(text, checker, expected):
    """A regex that stopped matching would make its check pass on a broken card."""
    assert checker(text) == expected
