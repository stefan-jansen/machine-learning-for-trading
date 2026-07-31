"""Exit-status tests for the crypto downloader (``data/crypto/market/download.py``).

The downloader makes roughly 700 Binance calls over 10-15 minutes against a
documented server-side rate limit, so coming back short is its expected failure
mode. It used to exit non-zero only when the downloaded frame was *entirely*
empty: 18 of 19 symbols could fail and the script still exited 0, which
``data/download_all.py`` reads as ``[OK] Crypto``.

These pin the status, not the summary text: a non-empty failed-symbol list on
either dataset must exit 1, and ``--allow-partial`` must be the only way to keep
what arrived and still exit 0.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

import data.download_all as da

DOWNLOAD_PY = Path(da.__file__).parent / "crypto" / "market" / "download.py"


def _load_download_module():
    """Load ``crypto/market/download.py`` (not a package) as a module."""
    spec = importlib.util.spec_from_file_location("crypto_market_download", DOWNLOAD_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _frame(symbol: str) -> pl.DataFrame:
    """One in-window row, the minimum the write path needs."""
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timestamp": [datetime(2021, 6, 1)],
            "close": [30000.0],
        }
    )


@pytest.fixture
def downloader(monkeypatch, tmp_path):
    """The real ``main()`` with the network and the profiler replaced."""
    mod = _load_download_module()

    class _StubProvider:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("ml4t.data.providers.binance_public.BinancePublicProvider", _StubProvider)
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(mod, "save_dataset_profile", lambda *a, **k: tmp_path / "profile.json")
    return mod


def _run(mod, monkeypatch, tmp_path, *, perps_failed, premium_failed, cli=()):
    monkeypatch.setattr(
        mod,
        "download_perps",
        lambda *a, **k: (_frame("BTCUSDT"), list(perps_failed)),
    )
    monkeypatch.setattr(
        mod,
        "download_premium",
        lambda *a, **k: (_frame("BTCUSDT"), list(premium_failed)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["download.py", "--data-path", str(tmp_path), "--symbol", "BTCUSDT", *cli],
    )
    mod.main()


def test_complete_download_exits_zero(downloader, monkeypatch, tmp_path):
    _run(downloader, monkeypatch, tmp_path, perps_failed=[], premium_failed=[])


def test_partial_perps_exits_nonzero(downloader, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(downloader, monkeypatch, tmp_path, perps_failed=["ETHUSDT"], premium_failed=[])
    assert exc.value.code == 1


def test_partial_premium_exits_nonzero(downloader, monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(downloader, monkeypatch, tmp_path, perps_failed=[], premium_failed=["ETHUSDT"])
    assert exc.value.code == 1


def test_allow_partial_keeps_what_arrived(downloader, monkeypatch, tmp_path, capsys):
    _run(
        downloader,
        monkeypatch,
        tmp_path,
        perps_failed=["ETHUSDT"],
        premium_failed=["ETHUSDT"],
        cli=("--allow-partial",),
    )
    assert "ETHUSDT" in capsys.readouterr().out


def test_failed_symbols_are_named(downloader, monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run(
            downloader,
            monkeypatch,
            tmp_path,
            perps_failed=["ETHUSDT", "SOLUSDT"],
            premium_failed=[],
        )
    out = capsys.readouterr().out
    assert "ETHUSDT" in out and "SOLUSDT" in out
