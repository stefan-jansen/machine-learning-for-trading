"""The ITCH loader resolves a path the parser notebook has not written yet.

`03_market_microstructure/01_itch_parser` is the notebook that creates the parsed
`messages/` directory, so on a clean start - raw binary downloaded, nothing parsed -
it has to be told where to write before anything is there. Every other Chapter 3
notebook reads that directory, and for them its absence is the download instruction.
"""

import pytest

from data.equities import loader
from data.exceptions import DataNotFoundError


def _clean_start(tmp_path):
    """A data root as the download script leaves it: raw present, messages absent."""
    raw = tmp_path / "equities" / "market" / "microstructure" / "nasdaq_itch" / "raw"
    raw.mkdir(parents=True)
    (raw / "S013020-v50.txt.gz").write_bytes(b"")
    return raw.parent / "messages"


def test_absent_messages_is_the_download_instruction_for_a_reader(tmp_path, monkeypatch) -> None:
    messages = _clean_start(tmp_path)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    with pytest.raises(DataNotFoundError):
        loader.load_nasdaq_itch(get_base_path=True)

    assert not messages.exists()


def test_the_parser_notebook_is_told_where_to_write(tmp_path, monkeypatch) -> None:
    messages = _clean_start(tmp_path)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    resolved = loader.load_nasdaq_itch(get_base_path=True, must_exist=False)

    assert resolved == messages
    resolved.mkdir(parents=True, exist_ok=True)
    assert (resolved.parent / "raw").exists()


def test_must_exist_false_without_get_base_path_is_refused(tmp_path, monkeypatch) -> None:
    _clean_start(tmp_path)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    with pytest.raises(ValueError, match="get_base_path"):
        loader.load_nasdaq_itch(must_exist=False)


def test_an_empty_messages_directory_still_refuses(tmp_path, monkeypatch) -> None:
    """A parse that wrote nothing must not silence the instruction for later notebooks.

    `01_itch_parser` used to create `messages/` where it resolved the path, before it
    checked for the raw binary, so a reader without the feed left an empty directory
    behind. Notebooks 02 to 07 ask only for the path, and an existence check alone
    handed it to them; they then failed on empty frames instead of being told to
    download. An empty directory is not parsed data.
    """
    messages = _clean_start(tmp_path)
    messages.mkdir(parents=True)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    with pytest.raises(DataNotFoundError):
        loader.load_nasdaq_itch(get_base_path=True)
    with pytest.raises(DataNotFoundError):
        loader.load_nasdaq_itch(message_types=["A"])

    # The parser still has to be able to resolve it in order to write into it.
    assert loader.load_nasdaq_itch(get_base_path=True, must_exist=False) == messages


def test_a_directory_holding_message_types_is_accepted(tmp_path, monkeypatch) -> None:
    """The empty-directory rule must not reject a real store."""
    messages = _clean_start(tmp_path)
    (messages / "A").mkdir(parents=True)
    monkeypatch.setattr(loader, "ML4T_DATA_PATH", tmp_path)

    assert loader.load_nasdaq_itch(get_base_path=True) == messages
