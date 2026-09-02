"""Tests for chapter 25 asynchronous notebook execution."""

from __future__ import annotations

import asyncio
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


def test_run_async_completes_inside_a_running_loop() -> None:
    """The case the helper exists for: a notebook kernel already runs a loop.

    Plain ``asyncio.run`` raises there. What ``run_async`` promises is that the
    awaitable still runs to completion and its value comes back - not which
    thread or loop it used, which differs between the nest_asyncio form the
    chapter ships and the thread-per-call form.
    """

    async def outer() -> tuple[int, str]:
        assert asyncio.get_running_loop().is_running()

        async def inner() -> str:
            await asyncio.sleep(0)
            return "inner-done"

        depth = 0

        async def counted() -> int:
            nonlocal depth
            depth += 1
            return depth

        return run_async(counted()), run_async(inner())

    depth, value = asyncio.run(outer())

    assert depth == 1
    assert value == "inner-done"


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
