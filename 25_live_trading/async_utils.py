"""Helpers for running async demos inside notebook kernels."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable

import nest_asyncio


def run_async[T](awaitable: Awaitable[T]) -> T:
    """Run an awaitable in both scripts and notebook kernels."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    # nest_asyncio.apply reaches an asyncio accessor deprecated in 3.14. The filter belongs
    # here rather than in each notebook: it has to be in force at the call, and installing it
    # in a cell does not survive to the cell that calls this. Narrow to the one message.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*get_event_loop_policy.*",
        )
        nest_asyncio.apply(loop)
    return loop.run_until_complete(awaitable)
