"""Regression tests for case_studies/utils/backtest_runner.py helpers.

Pins the P2.4 fixes from roborev jobs #2904, #2501, #2502, #2500:
- ``_align_symbol_dtype`` surfaces case-study context on ticker-vs-id mismatches.
- ``substitute_continuous_return_for_classification`` raises on duplicate
  (timestamp, symbol) rows in the continuous-return parquet and on left-join
  height changes.
- ``apply_universe_filter`` collapses sub-daily timestamps to the date grain
  before computing the within-date rank.
- ``_MAX_NULL_RATE`` constant is wired through ``max_null_rate`` parameter.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import polars as pl
import pytest

from case_studies.utils.backtest_runner import (
    _MAX_NULL_RATE,
    _align_symbol_dtype,
    apply_universe_filter,
    substitute_continuous_return_for_classification,
)


def test_max_null_rate_constant_default() -> None:
    assert _MAX_NULL_RATE == 0.10


def test_align_symbol_dtype_same_dtype_passthrough() -> None:
    target = pl.DataFrame({"symbol": ["A", "B"]})
    other = pl.DataFrame({"symbol": ["C", "D"]})
    out = _align_symbol_dtype(target, other, case_study="x", target_side="t", other_side="o")
    assert out.schema["symbol"] == pl.Utf8
    # Returned frame is the original when dtypes match.
    assert out.equals(other)


def test_align_symbol_dtype_int_target_numeric_string_source() -> None:
    target = pl.DataFrame({"symbol": [1, 2]}, schema={"symbol": pl.UInt32})
    other = pl.DataFrame({"symbol": ["10", "20"]})
    out = _align_symbol_dtype(
        target, other, case_study="us_firm", target_side="weights", other_side="prices"
    )
    assert out.schema["symbol"] == pl.UInt32
    assert out["symbol"].to_list() == [10, 20]


def test_align_symbol_dtype_int_target_ticker_source_raises_with_context() -> None:
    target = pl.DataFrame({"symbol": [1, 2]}, schema={"symbol": pl.UInt32})
    other = pl.DataFrame({"symbol": ["AAPL", "MSFT"]})
    with pytest.raises(TypeError, match=r"case_study='broken'"):
        _align_symbol_dtype(
            target,
            other,
            case_study="broken",
            target_side="weights",
            other_side="prices",
        )


def test_align_symbol_dtype_int_source_to_string_target() -> None:
    target = pl.DataFrame({"symbol": ["A"]})
    other = pl.DataFrame({"symbol": [1, 2]}, schema={"symbol": pl.UInt32})
    out = _align_symbol_dtype(target, other, case_study="x", target_side="t", other_side="o")
    assert out.schema["symbol"] == pl.Utf8


def test_apply_universe_filter_collapses_intraday_to_date_grain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sub-daily bars share a date but rank should be within-date, not within-bar.

    Without the date-collapse fix, two intraday bars per (date, symbol) would
    produce a denominator of 2N instead of N for the daily rank, silently
    filtering against a within-bar universe.
    """
    cs = "sp500_options_test"
    cs_dir = tmp_path / cs / "config"
    cs_dir.mkdir(parents=True)
    (cs_dir / "setup.yaml").write_text(
        dedent(
            """
            backtest:
              sweep:
                htm_cost_cascade:
                  liquid_quantile: 0.50
            """
        ).strip()
    )
    import case_studies.utils.backtest_runner as br

    monkeypatch.setattr(br, "CASE_STUDIES_DIR", str(tmp_path), raising=False)
    # ``CASE_STUDIES_DIR`` is imported lazily inside the function, so also
    # patch the source module ``utils`` so the rebinding wins.
    import utils as _utils  # type: ignore

    monkeypatch.setattr(_utils, "CASE_STUDIES_DIR", str(tmp_path), raising=False)

    # Two intraday bars per (date, symbol). Without date-collapse, rank
    # denominator would be 4 (two bars × two symbols) and both symbols would
    # land at the 0.50 quantile; with date-collapse, denominator is 2 (two
    # symbols), and the tighter-spread symbol (A) is the unique survivor.
    d1 = datetime(2024, 1, 2)
    bar_open = datetime(2024, 1, 2, 9, 30)
    bar_close = datetime(2024, 1, 2, 16, 0)
    prices = pl.DataFrame(
        {
            "timestamp": [bar_open, bar_close, bar_open, bar_close],
            "symbol": ["A", "A", "B", "B"],
            "instr_rel_spread": [0.01, 0.012, 0.05, 0.06],
        }
    )
    predictions = pl.DataFrame(
        {
            "timestamp": [d1, d1],
            "symbol": ["A", "B"],
        }
    )
    out = apply_universe_filter(
        predictions, prices, case_study=cs, signal_config={"universe_filter": "liquid"}
    )
    # Only the tighter-spread symbol (A) survives the 0.50 quantile.
    assert out["symbol"].to_list() == ["A"]


def test_substitute_continuous_return_dedupe_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cs = "test_cs"
    cs_dir = tmp_path / cs
    (cs_dir / "config").mkdir(parents=True)
    (cs_dir / "labels").mkdir()
    (cs_dir / "config" / "setup.yaml").write_text(
        dedent(
            """
            labels:
              classification_eval_label:
                fwd_dir_1d: fwd_ret_1d
            """
        ).strip()
    )
    # Continuous-return parquet with a duplicate (timestamp, symbol) row.
    d1 = datetime(2024, 1, 2)
    eval_df = pl.DataFrame(
        {
            "timestamp": [d1, d1, d1],  # 2× (d1, "A") — duplicate!
            "symbol": ["A", "A", "B"],
            "fwd_ret_1d": [0.01, 0.02, 0.03],
        }
    )
    eval_df.write_parquet(cs_dir / "labels" / "fwd_ret_1d.parquet")

    predictions = pl.DataFrame(
        {
            "timestamp": [d1, d1],
            "symbol": ["A", "B"],
            "y_score": [0.1, 0.2],
            "y_true": [1, 0],
        }
    )

    import case_studies.utils.backtest_runner as br
    import utils as _utils  # type: ignore

    monkeypatch.setattr(_utils, "CASE_STUDIES_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(br, "CASE_STUDIES_DIR", str(tmp_path), raising=False)

    with pytest.raises(ValueError, match=r"duplicate \(timestamp, symbol\)"):
        substitute_continuous_return_for_classification(
            predictions, case_study=cs, label="fwd_dir_1d"
        )


def test_substitute_continuous_return_max_null_rate_param(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing ``max_null_rate=1.0`` allows callers in a legitimately high-null regime."""
    cs = "test_cs_nulls"
    cs_dir = tmp_path / cs
    (cs_dir / "config").mkdir(parents=True)
    (cs_dir / "labels").mkdir()
    (cs_dir / "config" / "setup.yaml").write_text(
        dedent(
            """
            labels:
              classification_eval_label:
                fwd_dir_1d: fwd_ret_1d
            """
        ).strip()
    )
    d1 = datetime(2024, 1, 2)
    d2 = datetime(2024, 1, 3)
    # Eval parquet only covers d1, not d2 — predictions on d2 will null-match.
    eval_df = pl.DataFrame({"timestamp": [d1], "symbol": ["A"], "fwd_ret_1d": [0.01]})
    eval_df.write_parquet(cs_dir / "labels" / "fwd_ret_1d.parquet")

    predictions = pl.DataFrame(
        {
            "timestamp": [d1, d2, d2, d2],
            "symbol": ["A", "A", "B", "C"],
            "y_score": [0.1, 0.2, 0.3, 0.4],
            "y_true": [1, 0, 1, 0],
        }
    )

    import case_studies.utils.backtest_runner as br
    import utils as _utils  # type: ignore

    monkeypatch.setattr(_utils, "CASE_STUDIES_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(br, "CASE_STUDIES_DIR", str(tmp_path), raising=False)

    # Default cap (10%) raises: 3/4 = 75% null rate.
    with pytest.raises(ValueError, match=r"exceeds max_null_rate"):
        substitute_continuous_return_for_classification(
            predictions, case_study=cs, label="fwd_dir_1d"
        )
    # Override loosens the cap; missing rows are dropped instead of raised.
    out = substitute_continuous_return_for_classification(
        predictions, case_study=cs, label="fwd_dir_1d", max_null_rate=1.0
    )
    assert out.height == 1
    assert out["y_true"].to_list() == [0.01]
