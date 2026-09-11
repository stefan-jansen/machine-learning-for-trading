"""A replaced order inherits its chain root's attributes, however long the chain.

`05_itch_trading_activity` attributes executions to a stock and a side by walking ITCH
`U` (Replace) messages back to the add that started the chain. On a full trading day the
longest chain is roughly ten thousand hops, because a market maker rewrites the same
quote all session, so the resolution has to cost a pass per doubling of the depth rather
than a pass per hop.

The CI fixture's chains are one hop deep, so these drive the function over chains deep
enough to tell the two costs apart.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "03_market_microstructure" / "05_itch_trading_activity.py"


@pytest.fixture(scope="module")
def apply_replacements() -> Any:
    """The notebook's own `_apply_replacements`, compiled out of the source it ships."""
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_apply_replacements")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "MAX_REPLACEMENT_PASSES" for t in node.targets
            )
        )
    ]
    assert len(wanted) == 2, f"_apply_replacements or its pass cap is missing from {NOTEBOOK}"
    namespace: dict[str, Any] = {"pl": pl, "Path": Path}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["_apply_replacements"]


def _write_u(tmp_path: Path, rows: list[tuple[int, int, float]]) -> Path:
    """Lay out a message directory holding just the U messages given."""
    originals, news, prices = zip(*rows) if rows else ((), (), ())
    u_dir = tmp_path / "U"
    u_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "original_order_reference_number": pl.Series(originals, dtype=pl.UInt64),
            "new_order_reference_number": pl.Series(news, dtype=pl.UInt64),
            "price": pl.Series(prices, dtype=pl.Float64),
        }
    ).write_parquet(u_dir / "part-000000.parquet")
    return tmp_path


def _adds(refs_sides_stocks: list[tuple[int, str, str]]) -> pl.DataFrame:
    refs, sides, stocks = zip(*refs_sides_stocks)
    # Column order matches `_build_order_attrs`, which the resolved rows are stacked onto.
    return pl.DataFrame(
        {
            "order_reference_number": pl.Series(refs, dtype=pl.UInt64),
            "price": pl.Series([10.0] * len(refs), dtype=pl.Float64),
            "buy_sell_indicator": list(sides),
            "stock": list(stocks),
        }
    )


def _attrs_of(frame: pl.DataFrame, ref: int) -> dict[str, Any]:
    row = frame.filter(pl.col("order_reference_number") == ref)
    assert row.height == 1, f"expected one row for {ref}, got {row.height}"
    return row.to_dicts()[0]


def test_a_deep_chain_resolves_to_the_add_that_started_it(apply_replacements, tmp_path) -> None:
    """One add, rewritten a thousand times, as a quoting strategy does all session."""
    depth = 1000
    rows = [(1000 + i, 1001 + i, 20.0 + i) for i in range(depth)]
    resolved = apply_replacements(_write_u(tmp_path, rows), _adds([(1000, "B", "AAPL")]))

    assert resolved.height == 1 + depth
    last = _attrs_of(resolved, 1000 + depth)
    # Side and ticker come from the add; the price is the one the replacement quotes.
    assert last["buy_sell_indicator"] == "B"
    assert last["stock"] == "AAPL"
    assert last["price"] == pytest.approx(20.0 + depth - 1)
    assert _attrs_of(resolved, 1500)["stock"] == "AAPL"


def test_order_references_stay_unique(apply_replacements, tmp_path) -> None:
    """The caller asserts uniqueness before enriching, so the resolution must preserve it."""
    rows = [(1000 + i, 1001 + i, 20.0) for i in range(50)]
    rows.append((1000, 1001, 99.0))  # the same new reference issued a second time
    resolved = apply_replacements(_write_u(tmp_path, rows), _adds([(1000, "S", "NVDA")]))

    assert resolved["order_reference_number"].n_unique() == resolved.height


def test_two_chains_do_not_cross(apply_replacements, tmp_path) -> None:
    rows = [(10, 11, 1.0), (11, 12, 2.0), (20, 21, 3.0), (21, 22, 4.0)]
    resolved = apply_replacements(
        _write_u(tmp_path, rows), _adds([(10, "B", "AAPL"), (20, "S", "MSFT")])
    )

    assert _attrs_of(resolved, 12)["stock"] == "AAPL"
    assert _attrs_of(resolved, 12)["buy_sell_indicator"] == "B"
    assert _attrs_of(resolved, 22)["stock"] == "MSFT"
    assert _attrs_of(resolved, 22)["buy_sell_indicator"] == "S"


def test_a_chain_with_no_add_is_reported_not_attributed(apply_replacements, tmp_path) -> None:
    """A parent outside the sample leaves its whole chain unresolved rather than guessed."""
    rows = [(10, 11, 1.0), (11, 12, 2.0), (900, 901, 3.0), (901, 902, 4.0)]
    resolved = apply_replacements(_write_u(tmp_path, rows), _adds([(10, "B", "AAPL")]))

    assert set(resolved["order_reference_number"].to_list()) == {10, 11, 12}
    assert resolved.filter(pl.col("order_reference_number").is_in([901, 902])).is_empty()


def test_a_cycle_terminates(apply_replacements, tmp_path) -> None:
    """A reference chain that loops must not spin the doubling forever."""
    rows = [(10, 11, 1.0), (11, 12, 2.0), (12, 10, 3.0)]
    resolved = apply_replacements(_write_u(tmp_path, rows), _adds([(99, "B", "AAPL")]))

    assert resolved["order_reference_number"].to_list() == [99]


def test_no_u_messages_leaves_the_adds_untouched(apply_replacements, tmp_path) -> None:
    adds = _adds([(10, "B", "AAPL")])
    assert apply_replacements(tmp_path, adds).equals(adds)
