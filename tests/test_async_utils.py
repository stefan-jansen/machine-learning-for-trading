"""Tests for chapter 25 asynchronous notebook execution."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path

CHAPTER_DIR = Path(__file__).parents[1] / "25_live_trading"
sys.path.insert(0, str(CHAPTER_DIR))

from async_utils import run_async  # noqa: E402


def test_run_async_without_running_loop() -> None:
    async def identify_loop() -> tuple[int, bool]:
        return threading.get_ident(), asyncio.get_running_loop().is_running()

    thread_id, loop_running = run_async(identify_loop())

    assert thread_id == threading.get_ident()
    assert loop_running is True


# The body of the nested-loop test, run in a fresh interpreter. See
# `test_run_async_completes_inside_a_running_loop` for why it cannot run in this one.
_NESTED_LOOP_DRIVER = """
import asyncio, sys
sys.path.insert(0, {chapter_dir!r})
from async_utils import run_async


async def outer():
    assert asyncio.get_running_loop().is_running()

    async def inner():
        await asyncio.sleep(0)
        return "inner-done"

    depth = 0

    async def counted():
        nonlocal depth
        depth += 1
        return depth

    return run_async(counted()), run_async(inner())


depth, value = asyncio.run(outer())
print("DEPTH:", depth)
print("VALUE:", value)
"""


def test_run_async_completes_inside_a_running_loop() -> None:
    """The case the helper exists for: a notebook kernel already runs a loop.

    Plain ``asyncio.run`` raises there. What ``run_async`` promises is that the
    awaitable still runs to completion and its value comes back - not which
    thread or loop it used, which differs between the nest_asyncio form the
    chapter ships and the thread-per-call form.

    Driven in a subprocess, and that is not tidiness. This is the one path that
    reaches ``nest_asyncio.apply``, which patches the interpreter rather than the
    loop it is handed: measured on Python 3.14, ``asyncio.run`` is replaced by
    ``nest_asyncio._patch_asyncio.<locals>.run`` and ``asyncio.Task`` is swapped
    from ``_asyncio.Task`` to the pure-Python ``asyncio.tasks.Task``. Nothing
    reverses either, so every later test in the same process inherits them, and
    anything driving a kernel through asyncio then dies with ``RuntimeError:
    Timeout should be used inside a task``. Reproduced against papermill on a
    one-cell notebook: it executes in a clean interpreter and raises that in one
    where this test has already run.

    The cost of letting it leak is paid by whoever runs ``pytest tests/``
    locally, where this file sorts before every notebook-executing test and
    takes all of them down. CI never saw it because ``test_case_studies.py`` and
    ``test_chapter_notebooks.py`` are quarantined out of ``test-unit`` and run
    alone in their own jobs.

    A fresh interpreter has no patched asyncio to inherit, and what is under test
    - that the awaitable completes and its value comes back - is the same either
    way. ``test_the_module_leaves_asyncio_unpatched`` is what fails if this is
    ever put back in-process.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _NESTED_LOOP_DRIVER.format(chapter_dir=str(CHAPTER_DIR))],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DEPTH: 1" in completed.stdout, completed.stdout + completed.stderr
    assert "VALUE: inner-done" in completed.stdout, completed.stdout + completed.stderr


def test_run_async_propagates_the_exception_the_awaitable_raises() -> None:
    class Boom(Exception):
        pass

    async def explode() -> None:
        raise Boom("from inside the awaitable")

    try:
        run_async(explode())
    except Boom as exc:
        assert str(exc) == "from inside the awaitable"
    else:  # pragma: no cover - the helper must not swallow it
        raise AssertionError("run_async swallowed the awaitable's exception")


def test_the_module_leaves_asyncio_unpatched() -> None:
    """Nothing in this file may patch the interpreter's asyncio for the rest of the run.

    This sorts last so it observes whatever the tests above left behind. It is the
    negative control on the subprocess in
    ``test_run_async_completes_inside_a_running_loop``: run that body in-process
    and this reds, naming what was replaced.

    The two names are the ones ``nest_asyncio`` actually rebinds. ``asyncio.run``
    becomes a closure in ``nest_asyncio``, and ``asyncio.Task`` is swapped from the
    C accelerator to the pure-Python class - the second is what makes
    ``asyncio.timeout`` raise inside papermill, because it tests the running task
    against the C type.
    """
    assert asyncio.run.__module__ == "asyncio.runners", (
        f"asyncio.run is {asyncio.run.__module__}.{asyncio.run.__qualname__}, not "
        "asyncio.runners.run - something in this file applied nest_asyncio in-process "
        "and every notebook-executing test after it will fail"
    )
    assert asyncio.Task.__module__ == "_asyncio", (
        f"asyncio.Task is {asyncio.Task.__module__}.{asyncio.Task.__name__}, not the C "
        "_asyncio.Task - nest_asyncio replaced it in-process, and asyncio.timeout will "
        "then raise 'Timeout should be used inside a task' under papermill"
    )
