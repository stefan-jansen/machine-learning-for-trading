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
