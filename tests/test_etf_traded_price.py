"""The ETF traded-price sidecar carries a price a share actually changed hands at.

`etf_universe.parquet` is adjusted for splits and distributions, which a return needs and
a dollar amount must not use. Yahoo's `auto_adjust=False` close removes the distribution
half and leaves the split half, so `undo_splits` takes the rest off.
"""

from datetime import datetime

import polars as pl

from data.etfs.market.download import undo_splits


def _panel(rows: list[tuple[str, str, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "timestamp": [datetime.fromisoformat(r[1]) for r in rows],
            "close": [r[2] for r in rows],
            "volume": [r[3] for r in rows],
            "split": [r[4] for r in rows],
        }
    ).sort(["symbol", "timestamp"])


def test_forward_split_restores_the_pre_split_price():
    # IVW's shape: 4:1 on the third session, so the two before it traded at 4x.
    out = undo_splits(
        _panel(
            [
                ("IVW", "2020-10-15", 60.0, 400.0, 0.0),
                ("IVW", "2020-10-16", 60.0, 400.0, 0.0),
                ("IVW", "2020-10-19", 59.0, 1600.0, 4.0),
            ]
        )
    )
    assert out["close"].to_list() == [240.0, 240.0, 59.0]
    assert out["volume"].to_list() == [100.0, 100.0, 1600.0]


def test_reverse_split_restores_the_pre_split_price():
    # OIH's shape: 1:20, so the earlier history traded at a twentieth.
    out = undo_splits(
        _panel(
            [
                ("OIH", "2020-04-14", 100.0, 50.0, 0.0),
                ("OIH", "2020-04-15", 95.0, 1000.0, 0.05),
            ]
        )
    )
    assert out["close"].to_list() == [5.0, 95.0]
    assert out["volume"].to_list() == [1000.0, 1000.0]


def test_splits_compound_backwards():
    out = undo_splits(
        _panel(
            [
                ("SMH", "2023-05-04", 10.0, 100.0, 0.0),
                ("SMH", "2023-05-05", 10.0, 100.0, 2.0),
                ("SMH", "2024-03-07", 10.0, 100.0, 3.0),
            ]
        )
    )
    # The first session sits behind both splits, the second behind one, the third behind none.
    assert out["close"].to_list() == [60.0, 30.0, 10.0]


def test_turnover_is_invariant_under_the_correction():
    panel = _panel(
        [
            ("XLK", "2025-12-04", 250.0, 800.0, 0.0),
            ("XLK", "2025-12-05", 126.0, 1600.0, 2.0),
        ]
    )
    out = undo_splits(panel)
    before = (panel["close"] * panel["volume"]).to_list()
    after = (out["close"] * out["volume"]).to_list()
    assert before == after


def test_a_symbol_without_splits_is_returned_unchanged():
    panel = _panel(
        [
            ("SPY", "2006-01-03", 126.7, 73_256_700.0, 0.0),
            ("SPY", "2006-01-04", 127.3, 51_899_600.0, 0.0),
        ]
    )
    out = undo_splits(panel)
    assert out["close"].to_list() == panel["close"].to_list()
    assert out["volume"].to_list() == panel["volume"].to_list()


def test_each_symbol_carries_only_its_own_splits():
    out = undo_splits(
        _panel(
            [
                ("AAA", "2020-01-02", 10.0, 100.0, 0.0),
                ("AAA", "2020-01-03", 10.0, 100.0, 5.0),
                ("BBB", "2020-01-02", 10.0, 100.0, 0.0),
                ("BBB", "2020-01-03", 10.0, 100.0, 0.0),
            ]
        )
    )
    assert out.filter(pl.col("symbol") == "AAA")["close"].to_list() == [50.0, 10.0]
    assert out.filter(pl.col("symbol") == "BBB")["close"].to_list() == [10.0, 10.0]
