"""Tests for utils/data_quality.py.

Pins:
- apply_max_symbols: seed determinism + edge cases (no-op when max<=0 or >=N).
  Called by every loader; non-determinism would break reproducibility of tests
  and notebooks that depend on a sampled subset.
- check_ohlc_invariants: correct detection of OHLC violations, graceful
  handling of null values (TAQ no-trade bars), and missing-column tolerance.
"""

from __future__ import annotations

import polars as pl

from utils.data_quality import (
    apply_max_symbols,
    check_ohlc_invariants,
    describe_coverage,
    null_rate,
)


def _make_prices(symbols: list[str], n_rows: int = 3) -> pl.DataFrame:
    rows = []
    for s in symbols:
        for i in range(n_rows):
            rows.append(
                {
                    "symbol": s,
                    "timestamp": f"2024-01-{i + 1:02d}",
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1_000 * (i + 1),
                }
            )
    return pl.DataFrame(rows)


# -----------------------------------------------------------------------------
# apply_max_symbols
# -----------------------------------------------------------------------------


def test_apply_max_symbols_zero_is_passthrough() -> None:
    df = _make_prices(["A", "B", "C"])
    out = apply_max_symbols(df, 0)
    assert out.equals(df)


def test_apply_max_symbols_negative_is_passthrough() -> None:
    df = _make_prices(["A", "B", "C"])
    out = apply_max_symbols(df, -1)
    assert out.equals(df)


def test_apply_max_symbols_exceeds_universe_is_passthrough() -> None:
    df = _make_prices(["A", "B"])
    out = apply_max_symbols(df, 10)
    assert out.equals(df)


def test_apply_max_symbols_samples_requested_count() -> None:
    df = _make_prices(["A", "B", "C", "D", "E"])
    out = apply_max_symbols(df, 3)
    assert out["symbol"].n_unique() == 3


def test_apply_max_symbols_is_seed_deterministic() -> None:
    """Same seed → same subset; critical for reproducible tests."""
    df = _make_prices(["A", "B", "C", "D", "E", "F", "G", "H"])

    first = apply_max_symbols(df, 3, seed=42)["symbol"].unique().sort().to_list()
    second = apply_max_symbols(df, 3, seed=42)["symbol"].unique().sort().to_list()
    assert first == second


def test_apply_max_symbols_different_seed_yields_different_sample() -> None:
    df = _make_prices(["A", "B", "C", "D", "E", "F", "G", "H"])

    s42 = set(apply_max_symbols(df, 3, seed=42)["symbol"].unique().to_list())
    s7 = set(apply_max_symbols(df, 3, seed=7)["symbol"].unique().to_list())
    # At least one sample differs — very high probability for k=3, n=8
    assert s42 != s7


def test_apply_max_symbols_preserves_all_rows_per_symbol() -> None:
    df = _make_prices(["A", "B", "C", "D"], n_rows=5)
    out = apply_max_symbols(df, 2)
    # Each selected symbol should keep all its rows (function filters by symbol set)
    per_symbol = out.group_by("symbol").len()
    assert per_symbol["len"].to_list() == [5, 5]


def test_apply_max_symbols_sort_then_sample_is_order_invariant() -> None:
    """Shuffling the input before sampling must yield the same subset: the
    function sorts symbols before seeding the RNG so unstable loader order
    (e.g., parquet partition order) can't perturb the selection."""
    df_asc = _make_prices(["A", "B", "C", "D", "E"])
    df_desc = df_asc.sort("symbol", descending=True)

    s1 = apply_max_symbols(df_asc, 2, seed=42)["symbol"].unique().sort().to_list()
    s2 = apply_max_symbols(df_desc, 2, seed=42)["symbol"].unique().sort().to_list()
    assert s1 == s2


def test_apply_max_symbols_custom_symbol_col() -> None:
    df = pl.DataFrame({"product": ["X", "Y", "Z"], "value": [1, 2, 3]})
    out = apply_max_symbols(df, 2, symbol_col="product")
    assert out["product"].n_unique() == 2


def test_apply_max_symbols_with_lazyframe_returns_lazyframe() -> None:
    df = _make_prices(["A", "B", "C", "D"])
    lf = df.lazy()
    out = apply_max_symbols(lf, 2)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect()["symbol"].n_unique() == 2


# -----------------------------------------------------------------------------
# check_ohlc_invariants
# -----------------------------------------------------------------------------


def test_check_ohlc_invariants_clean_data_is_100_percent() -> None:
    df = _make_prices(["A"], n_rows=5)
    invariants = check_ohlc_invariants(df)
    # 6 checks expected: high_gte_low/open/close, low_lte_open/close, volume_non_negative
    assert invariants.height == 6
    assert (invariants["valid_pct"] == 100.0).all()


def test_check_ohlc_invariants_detects_high_below_low() -> None:
    df = pl.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [99.0, 105.0],  # first row violates high >= low
            "low": [100.0, 95.0],
            "close": [99.5, 102.0],
            "volume": [1_000, 1_000],
        }
    )
    invariants = check_ohlc_invariants(df)
    row = invariants.filter(pl.col("check") == "high_gte_low").row(0, named=True)
    assert row["valid_pct"] == 50.0
    assert row["applicable_rows"] == 2


def test_check_ohlc_invariants_ignores_null_rows() -> None:
    """Rows where any required col is null are excluded from the percentage."""
    df = pl.DataFrame(
        {
            "open": [100.0, None, 100.0],
            "high": [105.0, None, 102.0],
            "low": [95.0, None, 95.0],
            "close": [101.0, None, 101.0],
            "volume": [1_000, 1_000, 1_000],
        }
    )
    invariants = check_ohlc_invariants(df)
    row = invariants.filter(pl.col("check") == "high_gte_low").row(0, named=True)
    assert row["applicable_rows"] == 2  # 3 rows, 1 excluded for nulls
    assert row["valid_pct"] == 100.0


def test_check_ohlc_invariants_detects_negative_volume() -> None:
    df = pl.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [101.0],
            "volume": [-5],
        }
    )
    invariants = check_ohlc_invariants(df)
    row = invariants.filter(pl.col("check") == "volume_non_negative").row(0, named=True)
    assert row["valid_pct"] == 0.0


def test_check_ohlc_invariants_omits_volume_when_missing() -> None:
    df = pl.DataFrame({"open": [100.0], "high": [105.0], "low": [95.0], "close": [101.0]})
    invariants = check_ohlc_invariants(df)
    assert "volume_non_negative" not in invariants["check"].to_list()
    assert invariants.height == 5  # 5 price checks, no volume check


def test_check_ohlc_invariants_empty_df_returns_empty_result() -> None:
    df = pl.DataFrame({"x": []})
    invariants = check_ohlc_invariants(df)
    assert invariants.height == 0


# -----------------------------------------------------------------------------
# Smaller coverage helpers
# -----------------------------------------------------------------------------


def test_describe_coverage_basic_shape() -> None:
    df = _make_prices(["A", "B"], n_rows=3)
    cov = describe_coverage(df)
    assert cov["rows"] == 6
    assert cov["assets"] == 2
    assert cov["unique_times"] == 3


def test_null_rate_reports_per_column() -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    rates = null_rate(df)
    by_col = dict(zip(rates["column"], rates["null_pct"], strict=True))
    # 1/3 ≈ 33.33 for a, 2/3 ≈ 66.66 for b
    assert round(by_col["a"], 2) == 33.33
    assert round(by_col["b"], 2) == 66.67
