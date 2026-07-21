"""Contracts for the ALFRED initial-release materialization."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from data.macro.download_alfred import build_aligned_panel


def test_build_aligned_panel_starts_at_complete_point_in_time_coverage() -> None:
    raw = pl.DataFrame(
        {
            "series": ["dgs2", "dgs2", "dgs10", "dgs10", "vixcls", "vixcls"],
            "timestamp": [
                date(2020, 1, 2),
                date(2020, 1, 6),
                date(2020, 1, 2),
                date(2020, 1, 6),
                date(2020, 1, 3),
                date(2020, 1, 6),
            ],
            "vintage_date": [date(2020, 1, 3)] * 6,
            "value": [1.5, 1.6, 1.9, 2.0, 12.0, 13.0],
        }
    )
    panel = build_aligned_panel(
        raw,
        ["DGS2", "DGS10", "VIXCLS"],
        [{"name": "YIELD_CURVE_SLOPE", "formula": "DGS10 - DGS2"}],
    )

    assert panel["timestamp"].to_list() == [
        date(2020, 1, 3),
        date(2020, 1, 4),
        date(2020, 1, 5),
        date(2020, 1, 6),
    ]
    assert panel["dgs2"].to_list() == [1.5, 1.5, 1.5, 1.6]
    assert panel["vixcls"].to_list() == [12.0, 12.0, 12.0, 13.0]
    assert panel["YIELD_CURVE_SLOPE"].to_list() == pytest.approx([0.4, 0.4, 0.4, 0.4])
