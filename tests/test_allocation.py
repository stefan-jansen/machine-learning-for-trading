"""Tests for case_studies/utils/allocation.py — portfolio weight contracts.

These allocators sit between model predictions and the backtest engine.
A silent regression here would corrupt every Ch17+ result.

The tests pin *structural* contracts, not exact numerical values:

- Long-only weights sum to 1 per timestamp
- Long-only weights are all non-negative
- Long/short weights are dollar-neutral (net ≈ 0) with gross leverage ≈ 2
  (inverse-vol / risk-parity / HRP) or gross ≈ 1 (MVO)
- Exactly ``top_k`` assets per side are selected when enough assets exist
- Output columns are ``[timestamp, symbol, weight]`` in the expected order

Exact MVO values come from SLSQP and may vary across scipy versions, so the
numeric pins are loose (gross, net, count) rather than per-asset weights.

The ``synthetic_panel`` fixture builds 8 symbols × 300 dates of random-walk
prices. MVO needs a full lookback window (126 days); HRP needs ``vol_window``
(63 days). The fixture gives both allocators enough runway before the
rebalance timestamps.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from case_studies.utils.allocation import (
    compute_hrp_weights,
    compute_inverse_vol_weights,
    compute_mvo_weights,
    compute_risk_parity_weights,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_panel() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (predictions, prices) for 8 symbols × 300 days with 3 rebalance dates.

    Prices: geometric random walk with asset-specific vol (so inverse-vol
    produces distinguishable weights). Scores are ascending by symbol id
    (S0..S7) so top_k picks deterministically.
    """
    rng = np.random.default_rng(42)
    n_symbols = 8
    n_dates = 300
    symbols = [f"S{i}" for i in range(n_symbols)]
    ts = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), "1d", eager=True)[:n_dates]

    vols = 0.005 + 0.005 * np.arange(n_symbols) / n_symbols  # 0.5% to ~1%
    shocks = rng.normal(0.0, vols[None, :], (n_dates, n_symbols))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))

    price_rows: list[dict] = []
    for i, t in enumerate(ts):
        for j, s in enumerate(symbols):
            price_rows.append({"timestamp": t, "symbol": s, "close": float(prices[i, j])})
    prices_df = pl.DataFrame(price_rows)

    pred_dates = ts[-3:]
    pred_rows: list[dict] = []
    for t in pred_dates:
        for j, s in enumerate(symbols):
            pred_rows.append({"timestamp": t, "symbol": s, "y_score": float(j)})
    predictions = pl.DataFrame(pred_rows)

    return predictions, prices_df


# -----------------------------------------------------------------------------
# Shared contract checks
# -----------------------------------------------------------------------------


def _assert_output_shape(out: pl.DataFrame) -> None:
    assert set(out.columns) == {"timestamp", "symbol", "weight"}


def _assert_long_only_sums_to_1(out: pl.DataFrame) -> None:
    per_date = out.group_by("timestamp").agg(pl.col("weight").sum().alias("s")).sort("timestamp")
    for s in per_date["s"].to_list():
        assert abs(s - 1.0) < 1e-6, per_date


def _assert_non_negative(out: pl.DataFrame) -> None:
    assert (out["weight"] < 0).sum() == 0


def _assert_dollar_neutral(out: pl.DataFrame, gross_target: float) -> None:
    per_date = (
        out.group_by("timestamp")
        .agg(
            net=pl.col("weight").sum(),
            gross=pl.col("weight").abs().sum(),
        )
        .sort("timestamp")
    )
    for net, gross in zip(per_date["net"].to_list(), per_date["gross"].to_list(), strict=True):
        assert abs(net) < 1e-6, f"long-short should net to 0, got {net}"
        assert abs(gross - gross_target) < 1e-6, f"expected gross={gross_target}, got {gross}"


def _assert_top_k_selected(out: pl.DataFrame, top_k: int) -> None:
    per_date = (
        out.group_by("timestamp").agg(n=pl.col("symbol").count()).sort("timestamp")["n"].to_list()
    )
    for n in per_date:
        assert n == top_k, f"expected {top_k} selected, got {n}"


# -----------------------------------------------------------------------------
# compute_inverse_vol_weights
# -----------------------------------------------------------------------------


def test_inverse_vol_long_only_contracts(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    out = compute_inverse_vol_weights(predictions, prices, top_k=4)
    _assert_output_shape(out)
    _assert_long_only_sums_to_1(out)
    _assert_non_negative(out)
    _assert_top_k_selected(out, top_k=4)


def test_inverse_vol_picks_top_k_by_score(synthetic_panel) -> None:
    """Scores ascending S0..S7 → top 4 should be S4..S7."""
    predictions, prices = synthetic_panel
    out = compute_inverse_vol_weights(predictions, prices, top_k=4)
    assert set(out["symbol"].unique().to_list()) == {"S4", "S5", "S6", "S7"}


def test_inverse_vol_long_short_is_dollar_neutral(synthetic_panel) -> None:
    """Long/short with top_k=3 → 3 longs @ +w_i, 3 shorts @ -w_j, gross≈2 (two sides of 1)."""
    predictions, prices = synthetic_panel
    out = compute_inverse_vol_weights(predictions, prices, top_k=3, long_short=True)
    _assert_dollar_neutral(out, gross_target=2.0)


def test_inverse_vol_produces_nonuniform_weights(synthetic_panel) -> None:
    """Weights are 1/σ-normalized — selected assets have heterogeneous vols,
    so weights must not collapse to equal-weight (0.25 for top_k=4).
    """
    predictions, prices = synthetic_panel
    out = compute_inverse_vol_weights(predictions, prices, top_k=4)
    last_ts = out["timestamp"].max()
    slice_ = out.filter(pl.col("timestamp") == last_ts)
    weights = np.array(slice_["weight"].to_list())
    # Range of weights should be nontrivial (> 1% spread)
    assert weights.max() - weights.min() > 0.01


def test_inverse_vol_deterministic(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    a = compute_inverse_vol_weights(predictions, prices, top_k=4)
    b = compute_inverse_vol_weights(predictions, prices, top_k=4)
    assert a.sort("timestamp", "symbol").equals(b.sort("timestamp", "symbol"))


# -----------------------------------------------------------------------------
# compute_risk_parity_weights
# -----------------------------------------------------------------------------


def test_risk_parity_long_only_contracts(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    out = compute_risk_parity_weights(predictions, prices, top_k=4)
    _assert_output_shape(out)
    _assert_long_only_sums_to_1(out)
    _assert_non_negative(out)
    _assert_top_k_selected(out, top_k=4)


def test_risk_parity_long_short_is_dollar_neutral(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    out = compute_risk_parity_weights(predictions, prices, top_k=3, long_short=True)
    _assert_dollar_neutral(out, gross_target=2.0)


def test_risk_parity_assigns_less_to_high_vol_than_inverse_vol(synthetic_panel) -> None:
    """Risk-parity uses 1/σ^1.5 (steeper penalty than inverse-vol's 1/σ).

    High-vol assets should be relatively *less* weighted under risk-parity
    than under inverse-vol.
    """
    predictions, prices = synthetic_panel
    iv = compute_inverse_vol_weights(predictions, prices, top_k=4)
    rp = compute_risk_parity_weights(predictions, prices, top_k=4)
    last_ts = iv["timestamp"].max()
    iv_weights = dict(
        zip(
            iv.filter(pl.col("timestamp") == last_ts)["symbol"].to_list(),
            iv.filter(pl.col("timestamp") == last_ts)["weight"].to_list(),
            strict=True,
        )
    )
    rp_weights = dict(
        zip(
            rp.filter(pl.col("timestamp") == last_ts)["symbol"].to_list(),
            rp.filter(pl.col("timestamp") == last_ts)["weight"].to_list(),
            strict=True,
        )
    )
    # S7 is the highest-vol asset selected — risk-parity should weight it lower than inverse-vol
    assert rp_weights["S7"] < iv_weights["S7"]


def test_risk_parity_deterministic(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    a = compute_risk_parity_weights(predictions, prices, top_k=4)
    b = compute_risk_parity_weights(predictions, prices, top_k=4)
    assert a.sort("timestamp", "symbol").equals(b.sort("timestamp", "symbol"))


# -----------------------------------------------------------------------------
# compute_hrp_weights
# -----------------------------------------------------------------------------


def test_hrp_long_only_contracts(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    out = compute_hrp_weights(predictions, prices, top_k=4)
    _assert_output_shape(out)
    _assert_long_only_sums_to_1(out)
    _assert_non_negative(out)
    _assert_top_k_selected(out, top_k=4)


def test_hrp_long_short_is_dollar_neutral(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    out = compute_hrp_weights(predictions, prices, top_k=3, long_short=True)
    _assert_dollar_neutral(out, gross_target=2.0)


def test_hrp_falls_back_to_equal_weight_on_short_history() -> None:
    """With <20 days of history, HRP cannot form a covariance matrix → equal-weight."""
    rng = np.random.default_rng(0)
    n_dates = 10  # well under the 20-obs floor
    ts = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10), "1d", eager=True)
    price_rows = []
    for i, t in enumerate(ts):
        for j, s in enumerate(["A", "B", "C", "D"]):
            price_rows.append({"timestamp": t, "symbol": s, "close": 100.0 + float(rng.normal())})
    prices = pl.DataFrame(price_rows)
    predictions = pl.DataFrame(
        {
            "timestamp": [ts[-1]] * 4,
            "symbol": ["A", "B", "C", "D"],
            "y_score": [0.0, 1.0, 2.0, 3.0],
        }
    )
    out = compute_hrp_weights(predictions, prices, top_k=4)
    # Equal-weight = 1/4 for each
    for w in out["weight"].to_list():
        assert abs(w - 0.25) < 1e-9


def test_hrp_deterministic(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    a = compute_hrp_weights(predictions, prices, top_k=4)
    b = compute_hrp_weights(predictions, prices, top_k=4)
    assert a.sort("timestamp", "symbol").equals(b.sort("timestamp", "symbol"))


# -----------------------------------------------------------------------------
# compute_mvo_weights
# -----------------------------------------------------------------------------


def test_mvo_long_only_contracts(synthetic_panel) -> None:
    predictions, prices = synthetic_panel
    out = compute_mvo_weights(predictions, prices, top_k=4, max_weight=0.5)
    _assert_output_shape(out)
    _assert_non_negative(out)
    # Some assets may be dropped from the output if their optimal weight is
    # below 1e-6; don't pin the count. Weights should still sum to ~1.
    per_date = out.group_by("timestamp").agg(pl.col("weight").sum().alias("s"))
    for s in per_date["s"].to_list():
        assert abs(s - 1.0) < 1e-6


def test_mvo_long_short_gross_normalized_to_1_and_dollar_neutral(synthetic_panel) -> None:
    """MVO long/short normalizes gross to 1 and uses a dollar-neutral constraint."""
    predictions, prices = synthetic_panel
    out = compute_mvo_weights(predictions, prices, top_k=3, long_short=True, max_weight=0.5)
    _assert_dollar_neutral(out, gross_target=1.0)


def test_mvo_respects_position_cap(synthetic_panel) -> None:
    """No weight should exceed max_weight after normalization (long-only path)."""
    predictions, prices = synthetic_panel
    out = compute_mvo_weights(predictions, prices, top_k=8, max_weight=0.15)
    # Small tolerance for renormalization + float error
    assert out["weight"].max() <= 0.15 + 5e-3


def test_mvo_falls_back_to_equal_weight_on_short_history() -> None:
    """With 20 dates of history but lookback=126, MVO falls back to equal weight."""
    rng = np.random.default_rng(0)
    ts = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 20), "1d", eager=True)
    price_rows = []
    for t in ts:
        for s in ["A", "B", "C", "D"]:
            price_rows.append({"timestamp": t, "symbol": s, "close": 100.0 + float(rng.normal())})
    prices = pl.DataFrame(price_rows)
    predictions = pl.DataFrame(
        {
            "timestamp": [ts[-1]] * 4,
            "symbol": ["A", "B", "C", "D"],
            "y_score": [0.0, 1.0, 2.0, 3.0],
        }
    )
    out = compute_mvo_weights(predictions, prices, top_k=4)
    for w in out["weight"].to_list():
        assert abs(w - 0.25) < 1e-9


def test_mvo_returns_empty_frame_when_fewer_than_3_assets_selected() -> None:
    """top_k=2 → <3 assets → MVO skips the date and emits an empty frame."""
    rng = np.random.default_rng(0)
    ts = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), "1d", eager=True)[:200]
    price_rows = []
    for t in ts:
        for s in ["A", "B"]:
            price_rows.append({"timestamp": t, "symbol": s, "close": 100.0 + float(rng.normal())})
    prices = pl.DataFrame(price_rows)
    predictions = pl.DataFrame(
        {"timestamp": [ts[-1]] * 2, "symbol": ["A", "B"], "y_score": [0.0, 1.0]}
    )
    out = compute_mvo_weights(predictions, prices, top_k=2)
    assert out.height == 0
    assert set(out.columns) == {"timestamp", "symbol", "weight"}


# -----------------------------------------------------------------------------
# Input flexibility: accept either 'close' or 'ret' column in prices
# -----------------------------------------------------------------------------


def test_inverse_vol_accepts_ret_column_directly(synthetic_panel) -> None:
    """If prices already carry 'ret', the allocator uses it instead of pct_change('close')."""
    predictions, prices = synthetic_panel
    ret_prices = (
        prices.sort("timestamp", "symbol")
        .with_columns(ret=pl.col("close").pct_change().over("symbol"))
        .select("timestamp", "symbol", "ret")
    )
    out = compute_inverse_vol_weights(predictions, ret_prices, top_k=4)
    _assert_long_only_sums_to_1(out)
    _assert_top_k_selected(out, top_k=4)
