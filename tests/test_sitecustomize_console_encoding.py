"""`sitecustomize` forces UTF-8 on stdout, which is what keeps Windows readers' scripts alive.

A Windows console interpreter encodes `print` through the ANSI code page - `cp1252` on a
default install - so the first non-ASCII character in a progress line raises
`UnicodeEncodeError` and takes the process down. The Reader install walk hit exactly that:
`data/futures/positioning/cot_download.py:135` prints an arrow, and the CFTC download died
reporting seven years of data it had already fetched and written.

The repro below is platform-independent: `PYTHONIOENCODING=cp1252` puts any interpreter in
the same position a Windows console puts one, so the failure and the fix are both provable
here rather than only on a Windows runner.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.test_sitecustomize_data_root import load_sitecustomize

sc = load_sitecustomize()

# The character that actually broke the run, not a stand-in.
ARROW = "→"


class _Stream:
    """A stdout that records what it was reconfigured to, the way `TextIOWrapper` accepts it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def reconfigure(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _run(code: str, encoding: str) -> subprocess.CompletedProcess:
    # `-S` skips `site`, which is what imports `sitecustomize`. Without it this checkout's
    # own hook would run first and apply the fix on Windows, so the test meant to reproduce
    # the failure would pass by never reaching it - green on every platform but the one
    # under repair. The environment is inherited rather than replaced because Windows needs
    # `SystemRoot` to start an interpreter at all.
    env = {**os.environ, "PYTHONIOENCODING": encoding}
    # Decode explicitly. `text=True` alone decodes with the *parent's* locale encoding,
    # which on Windows is the same cp1252 this test is about, so the arrow the second
    # subprocess deliberately writes as UTF-8 would come back as mojibake and the
    # assertion would fail on the platform being fixed.
    return subprocess.run(
        [sys.executable, "-S", "-c", code],
        capture_output=True,
        encoding="utf-8",
        env=env,
        check=False,
    )


def test_an_ansi_code_page_kills_a_script_on_the_first_non_ascii_character():
    """The defect itself. Without this the fix below is a change with nothing behind it."""
    result = _run(f"print('rows {ARROW} out.parquet')", encoding="cp1252")
    assert result.returncode != 0
    assert "UnicodeEncodeError" in result.stderr


def test_reconfiguring_to_utf8_is_what_lets_the_same_script_finish():
    result = _run(
        f"import sys; sys.stdout.reconfigure(encoding='utf-8'); print('rows {ARROW} out.parquet')",
        encoding="cp1252",
    )
    assert result.returncode == 0
    assert ARROW in result.stdout


def test_the_hook_reconfigures_both_streams_on_windows(monkeypatch):
    out, err = _Stream(), _Stream()
    monkeypatch.setattr(sc.sys, "platform", "win32")
    monkeypatch.setattr(sc.sys, "stdout", out)
    monkeypatch.setattr(sc.sys, "stderr", err)

    sc._force_utf8_console()

    for stream in (out, err):
        assert stream.calls == [{"encoding": "utf-8", "errors": "backslashreplace"}]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_the_hook_leaves_posix_streams_alone(monkeypatch, platform):
    """Posix already gives UTF-8, and reconfiguring a redirected stream there would be a
    change nobody asked for - a reader piping output to a file chose that encoding."""
    out, err = _Stream(), _Stream()
    monkeypatch.setattr(sc.sys, "platform", platform)
    monkeypatch.setattr(sc.sys, "stdout", out)
    monkeypatch.setattr(sc.sys, "stderr", err)

    sc._force_utf8_console()

    assert out.calls == []
    assert err.calls == []


def test_a_stream_that_cannot_be_reconfigured_is_not_an_error(monkeypatch):
    """Under papermill and under pytest's capture, stdout is not a `TextIOWrapper` and has
    no `reconfigure`. The hook runs at interpreter startup, so it must never raise."""

    class _NoReconfigure:
        pass

    monkeypatch.setattr(sc.sys, "platform", "win32")
    monkeypatch.setattr(sc.sys, "stdout", _NoReconfigure())
    monkeypatch.setattr(sc.sys, "stderr", _NoReconfigure())

    sc._force_utf8_console()
