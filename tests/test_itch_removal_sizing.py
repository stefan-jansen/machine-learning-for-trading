"""A delete removes what is left of an order, not what the order was added with.

`03_market_microstructure/03_itch_lob_analysis` builds order flow imbalance by hand from
add and removal messages. ITCH `D` messages carry no share count, so their size has to be
derived. Deriving it as the original add double counts any order that was partially filled
or cancelled first, which inflates both sides of the imbalance.

The CI fixture cannot exercise this: its delete, cancel and execute reference numbers are
disjoint, so no fixture order is ever partially removed and then deleted. The function is
therefore read out of the notebook source and driven with frames that do cover it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import polars as pl
import pytest

NOTEBOOK = Path("03_market_microstructure/03_itch_lob_analysis.py")


@pytest.fixture(scope="module")
def enrich_removals() -> Any:
    """The notebook's own `_enrich_removals`, compiled out of the source it ships."""
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_enrich_removals"
    ]
    assert functions, f"_enrich_removals is no longer defined in {NOTEBOOK}"
    namespace: dict[str, Any] = {"pl": pl}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["_enrich_removals"]


def _registry() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_reference_number": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "side": ["B", "S", "B"],
            "shares": pl.Series([500, 300, 100], dtype=pl.UInt32),
        }
    )


def _removals(rows: list[tuple[int, int, int | None, str]]) -> pl.DataFrame:
    timestamps, refs, removed, events = zip(*rows)
    return pl.DataFrame(
        {
            "timestamp": pl.Series(timestamps, dtype=pl.Int64),
            "order_reference_number": pl.Series(refs, dtype=pl.UInt64),
            "shares_removed": pl.Series(removed, dtype=pl.Int64),
            "event_type": list(events),
        }
    )


def _removed_per_order(enriched: pl.DataFrame) -> dict[int, int]:
    totals = enriched.group_by("order_reference_number").agg(pl.col("shares_removed").sum())
    return dict(zip(*totals.to_dict(as_series=False).values()))


def test_a_partially_filled_order_does_not_remove_more_than_it_added(enrich_removals) -> None:
    enriched = enrich_removals(
        _removals(
            [
                (10, 1, 300, "execute"),
                (20, 1, None, "delete"),
                (30, 2, None, "delete"),
                (15, 3, 40, "cancel"),
                (25, 3, 60, "execute"),
            ]
        ),
        _registry(),
    )
    # Order 1 was added at 500 and filled for 300, so the delete takes the remaining 200.
    # Sizing it at the original would have removed 800 from a 500-share order.
    assert _removed_per_order(enriched) == {1: 500, 2: 300, 3: 100}
    delete = enriched.filter(
        (pl.col("order_reference_number") == 1) & (pl.col("event_type") == "delete")
    )
    assert delete["shares_removed"].to_list() == [200]


def test_a_delete_only_frame_is_sized_from_the_registry(enrich_removals) -> None:
    """`shares_removed` is absent entirely when no cancel or execution was loaded."""
    enriched = enrich_removals(
        pl.DataFrame(
            {
                "timestamp": pl.Series([30], dtype=pl.Int64),
                "order_reference_number": pl.Series([2], dtype=pl.UInt64),
                "event_type": ["delete"],
            }
        ),
        _registry(),
    )
    assert enriched["shares_removed"].to_list() == [300]


def test_a_delete_sorts_after_a_partial_that_shares_its_timestamp(enrich_removals) -> None:
    """Tied timestamps must not let the delete be sized before the fill it follows."""
    enriched = enrich_removals(
        _removals([(10, 1, None, "delete"), (10, 1, 300, "execute")]),
        _registry(),
    )
    assert _removed_per_order(enriched) == {1: 500}


def test_removals_exceeding_the_add_leave_the_delete_at_zero(enrich_removals) -> None:
    enriched = enrich_removals(
        _removals([(10, 3, 150, "execute"), (20, 3, None, "delete")]),
        _registry(),
    )
    assert _removed_per_order(enriched) == {3: 150}
