"""Exit-status tests for the crypto downloader (``data/crypto/market/download.py``).

The downloader makes roughly 700 Binance calls over 10-15 minutes against a
documented server-side rate limit, so coming back short is its expected failure
mode. It used to exit non-zero only when the downloaded frame was *entirely*
empty: 18 of 19 symbols could fail and the script still exited 0, which
``data/download_all.py`` reads as ``[OK] Crypto``.

These pin the status, not the summary text: a requested symbol absent from what
is now on disk must exit 1, and ``--allow-partial`` must be the only way to keep
what arrived and still exit 0.

On a plain run the status comes from the *merged* dataset rather than from the
symbols that failed in the last request. ``combine_existing()`` folds a retry
into what an earlier run wrote, so a symbol that fails on the retry is still on
disk — basing the status on the request would fail a download that is complete.

``--update`` inverts that, and both directions are pinned below. Every symbol is
already on disk by construction, so presence proves nothing about whether the
window was extended; there the request's failures are what did not arrive.
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


def _frame(symbols) -> pl.DataFrame:
    """One in-window row per symbol, the minimum the write path needs.

    An empty response is a frame with *no columns*, which is what the provider
    actually returns and what an earlier version of these tests got wrong: a
    zero-row frame that still carries named columns hides every crash on the
    empty path, because combine_existing() and sort() both need `timestamp`.
    """
    symbols = list(symbols)
    if not symbols:
        return pl.DataFrame()
    return pl.DataFrame(
        {
            "symbol": symbols,
            "timestamp": [datetime(2021, 6, 1)] * len(symbols),
            "close": [30000.0] * len(symbols),
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


@pytest.fixture
def configured_symbols():
    """The symbols a plain run requests, read from the shipped config."""
    from utils.downloading import flatten_group_values, load_section

    config = load_section(DOWNLOAD_PY.parent / "config.yaml", "crypto")
    symbols = flatten_group_values(config.get("symbols", {}), "symbols")
    assert len(symbols) > 3, "these tests need a multi-symbol universe"
    return symbols


def _run(mod, monkeypatch, tmp_path, *, arrived, failed=(), cli=()):
    """Run ``main()`` with both download functions replaced.

    *arrived* is what the provider returns this time. *failed* is what the
    request reports, which on a retry is deliberately allowed to disagree with
    what an earlier run already put on disk.
    """
    monkeypatch.setattr(mod, "download_perps", lambda *a, **k: (_frame(arrived), list(failed)))
    monkeypatch.setattr(mod, "download_premium", lambda *a, **k: (_frame(arrived), list(failed)))
    monkeypatch.setattr(sys, "argv", ["download.py", "--data-path", str(tmp_path), *cli])
    mod.main()


def test_complete_download_exits_zero(downloader, monkeypatch, tmp_path, configured_symbols):
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols)


def test_missing_symbol_exits_nonzero(downloader, monkeypatch, tmp_path, configured_symbols):
    with pytest.raises(SystemExit) as exc:
        _run(
            downloader,
            monkeypatch,
            tmp_path,
            arrived=configured_symbols[:1],
            failed=configured_symbols[1:],
        )
    assert exc.value.code == 1


def test_missing_symbols_are_named(downloader, monkeypatch, tmp_path, capsys, configured_symbols):
    with pytest.raises(SystemExit):
        _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols[:-2])
    out = capsys.readouterr().out
    for symbol in configured_symbols[-2:]:
        assert symbol in out


def test_allow_partial_keeps_what_arrived(
    downloader, monkeypatch, tmp_path, capsys, configured_symbols
):
    _run(
        downloader,
        monkeypatch,
        tmp_path,
        arrived=configured_symbols[:-1],
        cli=("--allow-partial",),
    )
    assert configured_symbols[-1] in capsys.readouterr().out


def test_a_retry_that_completes_the_dataset_exits_zero(
    downloader, monkeypatch, tmp_path, configured_symbols
):
    """The case a request-based status gets wrong.

    The first run comes back short. The second fetches the rest and reports the
    already-downloaded symbols as failures, which is what a rate-limited retry
    looks like. The merged dataset is complete, so the run has succeeded.
    """
    first, rest = configured_symbols[:2], configured_symbols[2:]
    with pytest.raises(SystemExit):
        _run(downloader, monkeypatch, tmp_path, arrived=first, failed=rest)

    # Nothing may raise: across the two runs every symbol is now on disk.
    _run(downloader, monkeypatch, tmp_path, arrived=rest, failed=first)


def test_a_retry_that_is_still_short_exits_nonzero(
    downloader, monkeypatch, tmp_path, configured_symbols
):
    """The converse, so the fix cannot degrade into always exiting 0."""
    first, second = configured_symbols[:2], configured_symbols[2:-1]
    with pytest.raises(SystemExit):
        _run(downloader, monkeypatch, tmp_path, arrived=first)
    with pytest.raises(SystemExit) as exc:
        _run(downloader, monkeypatch, tmp_path, arrived=second)
    assert exc.value.code == 1


def test_update_reports_a_failed_extension(downloader, monkeypatch, tmp_path, configured_symbols):
    """--update inverts what presence proves.

    Every symbol is already on disk by construction, so an update whose every
    incremental request failed would look complete to a presence check even
    though no new rows arrived. There the request's failures are the answer.
    """
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols)

    with pytest.raises(SystemExit) as exc:
        _run(
            downloader,
            monkeypatch,
            tmp_path,
            arrived=configured_symbols,
            failed=configured_symbols,
            cli=("--update",),
        )
    assert exc.value.code == 1


def test_update_that_extends_everything_exits_zero(
    downloader, monkeypatch, tmp_path, configured_symbols
):
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols)
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols, cli=("--update",))


def test_an_empty_retry_against_a_complete_dataset_exits_zero(
    downloader, monkeypatch, tmp_path, configured_symbols
):
    """A fully rate-limited retry is not a failed download.

    Nothing arrives, but everything is already on disk from the first run. This
    used to exit 1 before the status logic ran, and ignored --allow-partial too.
    """
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols)
    _run(downloader, monkeypatch, tmp_path, arrived=[], failed=configured_symbols)


@pytest.mark.parametrize("dataset", ["--perps", "--premium"])
def test_an_empty_forced_download_exits_nonzero(
    downloader, monkeypatch, tmp_path, configured_symbols, dataset
):
    """--force replaces rather than merges, so there is nothing to fall back to.

    A forced refresh that returned nothing has failed. Reporting success on the
    strength of the rows it was about to replace would present stale data as the
    result of the run.

    One dataset at a time: perps and premium have separate fallbacks, and a run
    covering both would exit 1 on either, so it could not tell which was fixed.
    """
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols, cli=(dataset,))
    with pytest.raises(SystemExit) as exc:
        _run(
            downloader,
            monkeypatch,
            tmp_path,
            arrived=[],
            failed=configured_symbols,
            cli=(dataset, "--force"),
        )
    assert exc.value.code == 1


def test_an_empty_first_run_exits_nonzero(
    downloader, monkeypatch, tmp_path, capsys, configured_symbols
):
    """The converse: nothing on disk and nothing arrived is still a failure.

    The message matters as much as the status here. A reader on their first run
    has no earlier download, so telling them the script is falling back to what
    is on disk describes something that does not exist.
    """
    with pytest.raises(SystemExit) as exc:
        _run(downloader, monkeypatch, tmp_path, arrived=[], failed=configured_symbols)
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "Nothing has been downloaded before either." in out
    assert "Falling back to what is already on disk." not in out


def test_an_empty_retry_says_it_is_using_what_is_on_disk(
    downloader, monkeypatch, tmp_path, capsys, configured_symbols
):
    _run(downloader, monkeypatch, tmp_path, arrived=configured_symbols)
    capsys.readouterr()
    _run(downloader, monkeypatch, tmp_path, arrived=[], failed=configured_symbols)
    assert "Falling back to what is already on disk." in capsys.readouterr().out


def test_update_starts_from_the_earliest_symbol(downloader, tmp_path):
    """An update must not step over a gap a previous partial update left.

    Symbols that succeeded carry the dataset maximum; the ones that failed sit
    behind it. Starting from the maximum would skip the gap permanently while
    reporting success, so the start comes from the earliest symbol.
    """
    output = tmp_path / "perps_1h.parquet"
    pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "timestamp": [datetime(2021, 6, 10), datetime(2021, 6, 1)],
            "close": [30000.0, 2000.0],
        }
    ).write_parquet(output)

    start = downloader.get_update_start(output, "2021-12-31", interval_hours=1)
    assert start == "2021-06-01", "must resume from the symbol that is furthest behind"


def test_update_reopens_the_window_for_a_symbol_with_no_history(downloader, tmp_path):
    """An absent symbol needs the configured start, not the recent gap.

    Resuming from the earliest *stored* symbol would give it only the tail, and
    its presence afterwards would then read as a complete download.
    """
    output = tmp_path / "perps_1h.parquet"
    pl.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "timestamp": [datetime(2021, 6, 10), datetime(2021, 6, 1)],
            "close": [30000.0, 2000.0],
        }
    ).write_parquet(output)

    start = downloader.get_update_start(
        output, "2021-12-31", 1, ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "2020-01-01"
    )
    assert start == "2020-01-01", "a symbol with no history reopens the configured window"


def test_a_single_requested_symbol_that_arrives_is_complete(downloader, monkeypatch, tmp_path):
    """--symbol narrows what was asked for, so nothing else can be missing."""
    _run(
        downloader,
        monkeypatch,
        tmp_path,
        arrived=["BTCUSDT"],
        failed=["ETHUSDT"],
        cli=("--symbol", "BTCUSDT"),
    )
