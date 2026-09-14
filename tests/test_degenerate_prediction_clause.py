"""A fold whose daily ICs cancel is a result; a fold whose predictions are constant is not.

`degenerate_prediction_sql` excludes a prediction set from every backtest and leaderboard,
so what it tests decides whether a real model can be silently dropped. It reads
`fold_metrics.ic`, which is the MEAN of the fold's per-cross-section Spearman ICs
(`case_studies/utils/registry/metrics.py:121`). A mean near zero has two causes that the
mean alone cannot tell apart: every cross-section ranked nothing, or the cross-sections
ranked well in both directions and cancelled. Only the first is degenerate.

The two cases separate on dispersion, so the clause tests `ic_std` alongside `ic`. These
fixtures are the two causes side by side, at magnitudes taken from the registries: the
denormal pair from `nasdaq100_microstructure`'s LASSO configurations, the cancelling row's
`ic_std` from the low end of what real folds carry.
"""

from __future__ import annotations

import sqlite3

import pytest

from case_studies.utils.notebook_contracts import degenerate_prediction_sql

# `ic` and `ic_std` for one fold of each shape.
CONSTANT_NULL = ("p_null", None, None)
CONSTANT_DENORMAL = ("p_denormal", 2.0048e-16, 1.9524e-17)
CANCELLING = ("p_cancelling", 3.0e-18, 0.104)
ORDINARY = ("p_ordinary", 0.0091, 0.187)


@pytest.fixture
def registry() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fold_metrics (prediction_hash TEXT, ic REAL, ic_std REAL)")
    conn.executemany(
        "INSERT INTO fold_metrics VALUES (?, ?, ?)",
        [CONSTANT_NULL, CONSTANT_DENORMAL, CANCELLING, ORDINARY],
    )
    conn.commit()
    return conn


def admitted(conn: sqlite3.Connection) -> set[str]:
    """The prediction sets the clause leaves selectable, through the clause as callers use it."""
    sql = (
        "SELECT DISTINCT prediction_hash FROM fold_metrics p WHERE 1=1"
        + degenerate_prediction_sql()
    )
    return {row[0] for row in conn.execute(sql)}


def test_constant_folds_are_excluded_in_both_stored_shapes(registry: sqlite3.Connection) -> None:
    """A NULL IC and a denormal IC are the same defect stored two ways."""
    assert CONSTANT_NULL[0] not in admitted(registry)
    assert CONSTANT_DENORMAL[0] not in admitted(registry)


def test_a_cancelling_fold_is_not_excluded(registry: sqlite3.Connection) -> None:
    """The regression this guards: an `ic` of 3e-18 is below any threshold on the mean alone.

    Its `ic_std` of 0.104 says the cross-sections ranked and disagreed, which is a model
    result. Testing `abs(ic)` without `ic_std` drops it.
    """
    assert CANCELLING[0] in admitted(registry)


def test_an_ordinary_fold_is_not_excluded(registry: sqlite3.Connection) -> None:
    assert ORDINARY[0] in admitted(registry)
