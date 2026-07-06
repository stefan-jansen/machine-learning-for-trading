"""Dispatch-logic tests for ``data/download_all.py``.

These run the real ``main()`` control flow with the network boundary
(``run_download_script``, which shells out to each downloader) replaced by a
recorder. No subprocesses, no network. They pin which datasets each mode
dispatches — the layer that silently broke in the #361 fixes:

- ``--free-only`` must skip the API-key datasets (macro/FX) but keep the core +
  factor + firm-char downloads;
- ``--skip-firm-characteristics`` must omit the 1.5 GB academic pull;
- firm characteristics must be invoked *plainly* (no ``--convert``, which used
  to run a conversion over data that had never been downloaded);
- every dispatched script must carry the selected ``--data-path``.
"""

from __future__ import annotations

import sys

import pytest

import data.download_all as da


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    """Replace run_download_script with a call recorder; neutralise dotenv."""
    calls: list[tuple[str, list]] = []

    def _record(script_name, extra_args=None):
        calls.append((script_name, list(extra_args or [])))
        return True

    monkeypatch.setattr(da, "run_download_script", _record)
    monkeypatch.setattr(da, "load_dotenv", lambda *a, **k: None)
    return calls


def _run(monkeypatch, tmp_path, *cli):
    monkeypatch.setattr(sys, "argv", ["download_all.py", "--data-path", str(tmp_path), *cli])
    da.main()


def _names(calls):
    return {name for name, _ in calls}


def test_free_only_dispatches_core_and_factors(recorder, monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, "--free-only")
    names = _names(recorder)
    # core case-study datasets + free factors + firm characteristics
    assert {
        "etfs.py",
        "crypto.py",
        "prediction_markets.py",
        "cot.py",
        "ff_factors.py",
        "aqr_factors.py",
        "firm_characteristics.py",
    } <= names
    # API-key / paid / large datasets must NOT be dispatched in free-only mode
    assert "macro.py" not in names
    assert "fx_pairs.py" not in names
    assert "us_equities.py" not in names


def test_skip_firm_characteristics(recorder, monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, "--free-only", "--skip-firm-characteristics")
    assert "firm_characteristics.py" not in _names(recorder)
    # the rest of the free tier still runs
    assert "etfs.py" in _names(recorder)


def test_firm_characteristics_invoked_without_convert(recorder, monkeypatch, tmp_path):
    """firm-char must download plainly — never with ``--convert`` alone."""
    _run(monkeypatch, tmp_path, "--free-only")
    fc = [args for name, args in recorder if name == "firm_characteristics.py"]
    assert len(fc) == 1
    assert "--convert" not in fc[0]
    assert "--data-path" in fc[0]


def test_core_mode_adds_api_key_datasets(recorder, monkeypatch, tmp_path):
    """Default (core) mode also dispatches the free-API-key datasets."""
    _run(monkeypatch, tmp_path)  # no --free-only
    names = _names(recorder)
    assert "macro.py" in names
    assert "fx_pairs.py" in names
    # but still not the --all-only historical equities pull
    assert "us_equities.py" not in names


def test_every_dispatch_carries_selected_data_path(recorder, monkeypatch, tmp_path):
    """No dispatched script may be left to guess the data root."""
    _run(monkeypatch, tmp_path, "--free-only")
    for name, args in recorder:
        assert str(tmp_path) in args, f"{name} did not receive the data path: {args}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
