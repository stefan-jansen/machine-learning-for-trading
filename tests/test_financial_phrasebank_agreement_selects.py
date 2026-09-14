"""`agreement` has to choose a corpus, and it was filtering a column no file carries.

`load_financial_phrasebank(agreement=...)` documented five agreement levels and implemented
them as `data.filter(pl.col("agreement") >= ...)` guarded by `"agreement" in data.columns`.
No distributed file carries that column - the corpus ships one file per level
(`sentences_allagree.parquet`, `sentences_75agree.parquet`, ...) - so the guard was always
false and the argument selected nothing.

It went unnoticed because the data directory holds exactly one of the five files, which made
"read every parquet here" and "read the 100% file" the same answer. The failure needs a second
file present, which is why these tests construct one: with two on disk the old loader
concatenated both and returned a corpus that is neither level, for every value of `agreement`.

Found while giving the CI fixture a builder: the fixture's `sentences_allagree.parquet` held
the 4,846-row 50% corpus under the 100% filename, and this argument is what should have
caught it.
"""

from __future__ import annotations

import polars as pl
import pytest

from data.alternative import loader
from data.exceptions import DataNotFoundError


def _write(root, name: str, sentences: list[str]) -> None:
    target = root / "alternative" / "text" / "financial_phrasebank" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"sentence": sentences, "label": [1] * len(sentences)},
    ).write_parquet(target)


@pytest.fixture
def two_levels(tmp_path, monkeypatch):
    """The 100% and 50% files side by side, as a full download leaves them."""
    _write(tmp_path, "sentences_allagree.parquet", ["unanimous one", "unanimous two"])
    _write(
        tmp_path,
        "sentences_50agree.parquet",
        ["unanimous one", "unanimous two", "contested three"],
    )
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)
    return tmp_path


def test_each_level_reads_its_own_file(two_levels) -> None:
    """The defect. The old loader answered both calls with all five rows concatenated."""
    assert loader.load_financial_phrasebank(agreement="100")["sentence"].to_list() == [
        "unanimous one",
        "unanimous two",
    ]
    assert loader.load_financial_phrasebank(agreement="50")["sentence"].to_list() == [
        "unanimous one",
        "unanimous two",
        "contested three",
    ]


def test_a_neighbouring_file_does_not_reach_the_result(two_levels) -> None:
    """Selecting a level is what keeps the other levels out, not a filter afterwards.

    Stated separately from the case above because it is the half that silently corrupts a
    result rather than merely widening it: a duplicated sentence in a fine-tuning corpus is
    not visible in a row count anyone checks.
    """
    unanimous = loader.load_financial_phrasebank(agreement="100")

    assert unanimous.height == unanimous["sentence"].n_unique()
    assert "contested three" not in unanimous["sentence"].to_list()


def test_a_level_that_was_never_downloaded_refuses(two_levels) -> None:
    """Returning a different level's corpus is the failure this argument exists to prevent.

    The old loader returned the concatenation of whatever happened to be present, so asking
    for a level absent from disk produced a full DataFrame and no signal at all.
    """
    with pytest.raises(DataNotFoundError, match="Financial Phrasebank"):
        loader.load_financial_phrasebank(agreement="66")


def test_all_tags_each_level_rather_than_merging_them(two_levels) -> None:
    """`agreement="all"` promised an `agreement_level` column and never produced one.

    The levels nest, so the concatenation repeats sentences by construction. That is the
    point of the column: without it the result is an unlabelled corpus with duplicates,
    which is what the old loader returned for every argument value.
    """
    combined = loader.load_financial_phrasebank(agreement="all")

    assert combined["agreement_level"].to_list() == ["100", "100", "50", "50", "50"]


def test_an_empty_directory_still_refuses(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    with pytest.raises(DataNotFoundError):
        loader.load_financial_phrasebank()
    with pytest.raises(DataNotFoundError):
        loader.load_financial_phrasebank(agreement="all")
