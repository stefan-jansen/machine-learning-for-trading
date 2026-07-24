"""Helpers for running async demos inside notebook kernels."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import nest_asyncio


def run_async[T](awaitable: Awaitable[T]) -> T:
    """Run an awaitable in both scripts and notebook kernels."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    nest_asyncio.apply(loop)
    return loop.run_until_complete(awaitable)
