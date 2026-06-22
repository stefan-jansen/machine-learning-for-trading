"""Tests for case_studies/utils/signals.py — prediction → weight contracts.

Signal construction sits on the critical path between every model and every
backtest. A silent behavior change here would corrupt every Ch16-20
strategy result. These tests pin the observable contracts:

- threshold / percentile cutoffs are applied with the documented
  inequality semantics (``>`` for fixed threshold, ``>=`` for cross-
  sectional percentile, ``>`` for rolling)
- long-short variants produce symmetric signals and weights
- equal-weight top-K weights sum to 1 (or 0 for excluded assets), and
  score-weighted weights sum to 1 with score-proportional magnitudes
- the config dispatcher routes every documented method and raises on
  unknowns
- ``direction=short_only`` is a pure sign flip of the weight column
- zero-weight rows are filtered from the output
- outputs are deterministic across repeated calls on the same input
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from case_studies.utils.signals import (
    build_target_weights,
    build_target_weights_from_config,
    cross_sectional_percentile_signal,
    fixed_threshold_signal,
    per_symbol_rolling_percentile_signal,
    rolling_percentile_signal,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def predictions_2d5s() -> pl.DataFrame:
    """2 timestamps × 5 symbols (A–E), y_score ascending per date."""
    return pl.DataFrame(
        {
            "timestamp": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
            "symbol": list("ABCDE") * 2,
            "y_score": [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


@pytest.fixture
def predictions_2d6s() -> pl.DataFrame:
    """2 timestamps × 6 symbols (A–F) for even-split top/bottom tests."""
    return pl.DataFrame(
        {
            "timestamp": ["2024-01-01"] * 6 + ["2024-01-02"] * 6,
            "symbol": list("ABCDEF") * 2,
            "y_score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0],
        }
    ).with_columns(pl.col("timestamp").str.to_date())


@pytest.fixture
def predictions_rolling() -> pl.DataFrame:
    """50 timestamps × 2 symbols for rolling-window tests."""
    rng = np.random.default_rng(42)
    ts = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 2, 19), "1d", eager=True)
    rows = [(t, s, float(rng.random())) for s in ("A", "B") for t in ts]
    return pl.DataFrame(rows, schema=["timestamp", "symbol", "y_score"], orient="row").sort(
        "timestamp", "symbol"
    )


# -----------------------------------------------------------------------------
# fixed_threshold_signal
# -----------------------------------------------------------------------------


def test_fixed_threshold_long_only_strict_greater_than(predictions_2d5s) -> None:
    """signal=1 iff score > threshold (strict). At-threshold scores get 0."""
    out = fixed_threshold_signal(predictions_2d5s, threshold=0.5, signal_type="long_only")
    # 0.5 → 0 (not strictly >); 0.7/0.9/0.6/0.8/1.0 → 1
    expected = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
    assert out["signal"].to_list() == expected


def test_fixed_threshold_signal_is_int8(predictions_2d5s) -> None:
    out = fixed_threshold_signal(predictions_2d5s, threshold=0.5)
    assert out["signal"].dtype == pl.Int8


def test_fixed_threshold_preserves_row_count_and_columns(predictions_2d5s) -> None:
    out = fixed_threshold_signal(predictions_2d5s, threshold=0.5)
    assert out.height == predictions_2d5s.height
    assert set(predictions_2d5s.columns) <= set(out.columns)


def test_fixed_threshold_long_short_uses_symmetric_mirror() -> None:
    """long_short with threshold=0.7 → above 0.7 → 1, below (1-0.7)=0.3 → -1."""
    df = pl.DataFrame({"y_score": [0.1, 0.4, 0.5, 0.6, 0.9]})
    out = fixed_threshold_signal(df, threshold=0.7, signal_type="long_short")
    assert out["signal"].to_list() == [-1, 0, 0, 0, 1]


def test_fixed_threshold_deterministic_across_calls(predictions_2d5s) -> None:
    a = fixed_threshold_signal(predictions_2d5s, threshold=0.5)
    b = fixed_threshold_signal(predictions_2d5s, threshold=0.5)
    assert_frame_equal(a, b)


# -----------------------------------------------------------------------------
# rolling_percentile_signal
# -----------------------------------------------------------------------------


def test_rolling_percentile_adds_threshold_column(predictions_rolling) -> None:
    out = rolling_percentile_signal(predictions_rolling, window=10, percentile=80.0)
    assert "rolling_threshold" in out.columns


def test_rolling_percentile_early_window_has_null_threshold(predictions_rolling) -> None:
    """First window-1 rows per asset have insufficient history → null threshold."""
    out = rolling_percentile_signal(predictions_rolling, window=10, percentile=80.0)
    # 2 symbols × (window-1=9) early rows = 18 nulls
    assert out["rolling_threshold"].null_count() == 18


def test_rolling_percentile_long_short_adds_both_thresholds(predictions_rolling) -> None:
    out = rolling_percentile_signal(
        predictions_rolling, window=10, percentile=80.0, signal_type="long_short"
    )
    assert "rolling_threshold" in out.columns
    assert "rolling_lower_threshold" in out.columns
    # Must produce at least one long and one short signal with random data
    counts = dict(out.group_by("signal").len().iter_rows())
    assert counts.get(1, 0) > 0
    assert counts.get(-1, 0) > 0


def test_rolling_percentile_per_asset_independence() -> None:
    """Each asset computes its own rolling quantile — asset ordering shouldn't
    change its own signal sequence."""
    ts = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 20), "1d", eager=True)
    rows_a = [(t, "A", float(i)) for i, t in enumerate(ts)]
    rows_b = [(t, "B", float(-i)) for i, t in enumerate(ts)]
    df = pl.DataFrame(rows_a + rows_b, schema=["timestamp", "symbol", "y_score"], orient="row")
    a_only_thresholds = rolling_percentile_signal(
        df.filter(pl.col("symbol") == "A"), window=5, percentile=80.0
    )["rolling_threshold"]
    with_both = rolling_percentile_signal(df, window=5, percentile=80.0).filter(
        pl.col("symbol") == "A"
    )["rolling_threshold"]
    assert a_only_thresholds.to_list() == with_both.to_list()


# -----------------------------------------------------------------------------
# cross_sectional_percentile_signal
# -----------------------------------------------------------------------------


def test_cs_percentile_long_only_at_or_above_cutoff(predictions_2d5s) -> None:
    """cs_percentile uses ``>=`` — score equal to the threshold gets a signal.

    At percentile=80 with 5 symbols, the 80th percentile interpolates to the
    second-highest score. With ascending scores D=0.7, E=0.9 for date 1,
    cs_threshold=0.7 and both D and E get signal=1.
    """
    out = cross_sectional_percentile_signal(predictions_2d5s, percentile=80.0).sort(
        "timestamp", "symbol"
    )
    # Per date, top 2 by score should be selected
    assert out.filter(pl.col("signal") == 1).height == 4  # 2 dates × 2 winners


def test_cs_percentile_threshold_differs_per_timestamp(predictions_2d5s) -> None:
    """Different dates have different score distributions → different thresholds."""
    out = cross_sectional_percentile_signal(predictions_2d5s, percentile=80.0)
    per_date = out.group_by("timestamp").agg(pl.col("cs_threshold").first()).sort("timestamp")
    thresholds = per_date["cs_threshold"].to_list()
    assert thresholds[0] != thresholds[1]


def test_cs_percentile_long_short_produces_both_signs(predictions_2d5s) -> None:
    out = cross_sectional_percentile_signal(
        predictions_2d5s, percentile=80.0, signal_type="long_short"
    )
    signs = set(out["signal"].to_list())
    assert 1 in signs
    assert -1 in signs


# -----------------------------------------------------------------------------
# build_target_weights — equal_weight_top_k
# -----------------------------------------------------------------------------


def test_equal_weight_top_k_long_only_weights_sum_to_1(predictions_2d5s) -> None:
    out = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    per_date = out.group_by("timestamp").agg(pl.col("weight").sum()).sort("timestamp")
    for w in per_date["weight"].to_list():
        assert abs(w - 1.0) < 1e-9


def test_equal_weight_top_k_selects_exactly_k_assets_per_date(predictions_2d5s) -> None:
    out = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    per_date = out.group_by("timestamp").agg(pl.col("symbol").count().alias("n")).sort("timestamp")
    assert per_date["n"].to_list() == [2, 2]


def test_equal_weight_top_k_picks_highest_scores(predictions_2d5s) -> None:
    """With ascending scores A..E, top 2 should always be D and E."""
    out = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    selected = set(out["symbol"].unique().to_list())
    assert selected == {"D", "E"}


def test_equal_weight_top_k_long_short_weights_are_symmetric(predictions_2d6s) -> None:
    """long_short top_k=2 with 6 symbols: 2 long @+0.5, 2 short @-0.5, 2 zero (dropped)."""
    out = build_target_weights(
        predictions_2d6s, method="equal_weight_top_k", top_k=2, long_short=True
    )
    longs = out.filter(pl.col("weight") > 0)
    shorts = out.filter(pl.col("weight") < 0)
    assert longs.height == 4  # 2 dates × 2 longs
    assert shorts.height == 4
    # Magnitudes equal
    assert all(abs(w - 0.5) < 1e-9 for w in longs["weight"])
    assert all(abs(w + 0.5) < 1e-9 for w in shorts["weight"])


def test_equal_weight_top_k_clamps_when_k_exceeds_n_assets(predictions_2d5s) -> None:
    """Asking for top_k=100 with 5 assets per date → selects all 5, weights = 1/5."""
    out = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=100)
    assert out.height == 10  # all rows survive
    assert all(abs(w - 0.2) < 1e-9 for w in out["weight"])


def test_equal_weight_top_k_filters_zero_weights(predictions_2d5s) -> None:
    """The helper strips zero-weight rows from the output."""
    out = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    assert (out["weight"] == 0.0).sum() == 0


def test_equal_weight_top_k_output_sorted_by_time_then_asset(predictions_2d5s) -> None:
    out = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    pairs = list(zip(out["timestamp"].to_list(), out["symbol"].to_list(), strict=True))
    assert pairs == sorted(pairs)


# -----------------------------------------------------------------------------
# build_target_weights — score_weighted_top_k
# -----------------------------------------------------------------------------


def test_score_weighted_top_k_long_only_weights_sum_to_1(predictions_2d5s) -> None:
    out = build_target_weights(predictions_2d5s, method="score_weighted_top_k", top_k=2)
    per_date = out.group_by("timestamp").agg(pl.col("weight").sum())
    for w in per_date["weight"].to_list():
        assert abs(w - 1.0) < 1e-9


def test_score_weighted_top_k_weight_proportional_to_abs_score(predictions_2d6s) -> None:
    """Top 2 of [0.8, 0.9] → weights 0.8/1.7 ≈ 0.4706 and 0.9/1.7 ≈ 0.5294."""
    out = build_target_weights(predictions_2d6s, method="score_weighted_top_k", top_k=2).sort(
        "timestamp", "symbol"
    )
    date1 = out.filter(pl.col("timestamp") == pl.date(2024, 1, 1)).sort("symbol")
    weights = dict(zip(date1["symbol"].to_list(), date1["weight"].to_list(), strict=True))
    assert abs(weights["E"] - 0.8 / 1.7) < 1e-9
    assert abs(weights["F"] - 0.9 / 1.7) < 1e-9


def test_score_weighted_top_k_deterministic(predictions_2d6s) -> None:
    a = build_target_weights(predictions_2d6s, method="score_weighted_top_k", top_k=2)
    b = build_target_weights(predictions_2d6s, method="score_weighted_top_k", top_k=2)
    assert_frame_equal(a.sort("timestamp", "symbol"), b.sort("timestamp", "symbol"))


# -----------------------------------------------------------------------------
# build_target_weights — inverse_vol (placeholder path: equal weight)
# -----------------------------------------------------------------------------


def test_inverse_vol_placeholder_uses_equal_weight(predictions_2d5s) -> None:
    """inverse_vol is documented as a placeholder — same output as equal_weight_top_k."""
    eq = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    iv = build_target_weights(predictions_2d5s, method="inverse_vol", top_k=2)
    assert_frame_equal(
        eq.sort("timestamp", "symbol"),
        iv.sort("timestamp", "symbol"),
    )


# -----------------------------------------------------------------------------
# build_target_weights_from_config — dispatcher
# -----------------------------------------------------------------------------


def test_from_config_equal_weight_top_k_matches_direct_call(predictions_2d5s) -> None:
    direct = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    via = build_target_weights_from_config(
        predictions_2d5s, {"method": "equal_weight_top_k", "top_k": 2}
    )
    assert_frame_equal(direct.sort("timestamp", "symbol"), via.sort("timestamp", "symbol"))


def test_from_config_decile_long_short_on_small_universe(predictions_2d6s) -> None:
    """6 symbols, decile → top_cutoff=floor(6/10)=0 clipped to 1 → 1 long, 1 short."""
    out = build_target_weights_from_config(predictions_2d6s, {"method": "decile_long_short"}).sort(
        "timestamp", "symbol"
    )
    # Per date: 1 long @ +1.0 (top score), 1 short @ -1.0 (bottom score)
    assert out.height == 4
    assert sorted(out["weight"].unique().to_list()) == [-1.0, 1.0]


def test_from_config_cross_sectional_percentile(predictions_2d5s) -> None:
    out = build_target_weights_from_config(
        predictions_2d5s,
        {"method": "cross_sectional_percentile", "percentile": 80.0},
    )
    # Top 2 assets per date → weights sum to 1 per date
    per_date = out.group_by("timestamp").agg(pl.col("weight").sum())
    for w in per_date["weight"].to_list():
        assert abs(w - 1.0) < 1e-9


def test_from_config_fixed_threshold_selects_above_cutoff(predictions_2d5s) -> None:
    out = build_target_weights_from_config(
        predictions_2d5s, {"method": "fixed_threshold", "threshold": 0.5}
    )
    # Per date: D (0.7), E (0.9) → 2 assets @ 0.5 each, sum to 1 on date 1.
    # Date 2: C (0.6), D (0.8), E (1.0) → 3 assets @ 1/3 each.
    date1 = out.filter(pl.col("timestamp") == pl.date(2024, 1, 1))
    date2 = out.filter(pl.col("timestamp") == pl.date(2024, 1, 2))
    assert date1.height == 2 and abs(date1["weight"].sum() - 1.0) < 1e-9
    assert date2.height == 3 and abs(date2["weight"].sum() - 1.0) < 1e-9


def test_from_config_short_only_negates_weights(predictions_2d5s) -> None:
    """direction=short_only flips signs; magnitudes identical to long_only."""
    long_w = build_target_weights_from_config(
        predictions_2d5s, {"method": "equal_weight_top_k", "top_k": 2}
    )
    short_w = build_target_weights_from_config(
        predictions_2d5s,
        {"method": "equal_weight_top_k", "top_k": 2, "direction": "short_only"},
    )
    # Sort and pair up, then verify the negation contract
    long_sorted = long_w.sort("timestamp", "symbol")
    short_sorted = short_w.sort("timestamp", "symbol")
    assert long_sorted["symbol"].to_list() == short_sorted["symbol"].to_list()
    for lw, sw in zip(
        long_sorted["weight"].to_list(), short_sorted["weight"].to_list(), strict=True
    ):
        assert abs(lw + sw) < 1e-9


def test_from_config_rejects_unknown_method(predictions_2d5s) -> None:
    with pytest.raises(ValueError, match="Unknown signal method"):
        build_target_weights_from_config(predictions_2d5s, {"method": "bogus"})


def test_from_config_rejects_unknown_direction(predictions_2d5s) -> None:
    with pytest.raises(ValueError, match="Unknown signal direction"):
        build_target_weights_from_config(
            predictions_2d5s,
            {"method": "equal_weight_top_k", "top_k": 2, "direction": "bogus"},
        )


def test_from_config_quintile_long_short_uses_5_buckets(predictions_2d5s) -> None:
    """quintile with 5 assets → top_cutoff=1 → 1 long, 1 short per date."""
    out = build_target_weights_from_config(predictions_2d5s, {"method": "quintile_long_short"})
    per_date = out.group_by("timestamp").agg(pl.col("symbol").count().alias("n"))
    assert per_date["n"].to_list() == [2, 2]  # 1 long + 1 short each date


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_cross_sectional_percentile_deterministic(predictions_2d5s) -> None:
    a = cross_sectional_percentile_signal(predictions_2d5s, percentile=80.0)
    b = cross_sectional_percentile_signal(predictions_2d5s, percentile=80.0)
    assert_frame_equal(a, b)


def test_rolling_percentile_deterministic(predictions_rolling) -> None:
    a = rolling_percentile_signal(predictions_rolling, window=10, percentile=80.0)
    b = rolling_percentile_signal(predictions_rolling, window=10, percentile=80.0)
    assert_frame_equal(a, b)


def test_build_target_weights_deterministic(predictions_2d5s) -> None:
    a = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    b = build_target_weights(predictions_2d5s, method="equal_weight_top_k", top_k=2)
    assert_frame_equal(a, b)


# -----------------------------------------------------------------------------
# per_symbol_rolling_percentile_signal — stay_q extension
# -----------------------------------------------------------------------------


@pytest.fixture
def per_symbol_intraday() -> pl.DataFrame:
    """30 days × 2 symbols × 14 bars/day; deterministic seeded scores."""
    from datetime import datetime, timedelta

    rng = np.random.default_rng(11)
    rows = []
    for d in range(30):
        for i in range(14):
            ts = datetime(2024, 1, 2, 9, 30) + timedelta(days=d, minutes=15 * i)
            for sym in ("AAA", "BBB"):
                rows.append((ts, sym, float(rng.standard_normal())))
    return pl.DataFrame(rows, schema=["timestamp", "symbol", "y_score"], orient="row").sort(
        "symbol", "timestamp"
    )


def test_per_symbol_default_excludes_stay_thresh(per_symbol_intraday) -> None:
    """When stay_q is None, stay_thresh column is NOT present (back-compat)."""
    out = per_symbol_rolling_percentile_signal(
        per_symbol_intraday,
        long_q=0.80,
        lookback_days=10,
        bars_per_day=14,
    )
    assert "stay_thresh" not in out.columns
    assert "signal" in out.columns


def test_per_symbol_stay_q_adds_stay_thresh(per_symbol_intraday) -> None:
    """When stay_q is set, stay_thresh column is added; non-null after warm-up."""
    out = per_symbol_rolling_percentile_signal(
        per_symbol_intraday,
        long_q=0.80,
        lookback_days=10,
        bars_per_day=14,
        stay_q=0.40,
    )
    assert "stay_thresh" in out.columns
    # After warm-up (~5 sessions = 70 bars per symbol), stay_thresh should be non-null
    by_sym = out.group_by("symbol").agg(pl.col("stay_thresh").is_not_null().sum().alias("n"))
    for n in by_sym["n"].to_list():
        assert n > 100  # well past warm-up of W//2 = 70


def test_per_symbol_stay_thresh_monotonic_in_stay_q(per_symbol_intraday) -> None:
    """stay_thresh must increase monotonically with stay_q, and thus stay below
    the entry threshold (long_q) for any stay_q < long_q.

    long_thresh is dropped from the output, and the function forbids
    ``stay_q == long_q``, so we cannot read long_thresh directly. Instead we
    verify the underlying invariant black-box: a lower stay_q must yield a
    stay_thresh at or below a higher stay_q's on the same row. Since the entry
    threshold is the long_q quantile, monotonicity transitively guarantees
    every ``stay_q < long_q`` threshold sits below it. This catches a sign flip
    between stay_thresh and long_thresh, unlike the previous tautological
    "score exceeds the threshold that fired it" check.
    """
    kw = dict(long_q=0.80, lookback_days=10, bars_per_day=14)
    lo = per_symbol_rolling_percentile_signal(per_symbol_intraday, stay_q=0.40, **kw)
    hi = per_symbol_rolling_percentile_signal(per_symbol_intraday, stay_q=0.79, **kw)

    joined = (
        lo.select(["symbol", "timestamp", "stay_thresh"])
        .join(
            hi.select(["symbol", "timestamp", pl.col("stay_thresh").alias("stay_thresh_hi")]),
            on=["symbol", "timestamp"],
            how="inner",
        )
        .filter(pl.col("stay_thresh").is_not_null() & pl.col("stay_thresh_hi").is_not_null())
    )

    assert joined.height > 100  # well past warm-up
    # q=0.40 quantile must never exceed the q=0.79 quantile (< the q=0.80 entry).
    assert (joined["stay_thresh"] - joined["stay_thresh_hi"]).max() <= 1e-9


def test_per_symbol_rejects_stay_q_at_or_above_long_q(per_symbol_intraday) -> None:
    with pytest.raises(ValueError, match="stay_q must be < long_q"):
        per_symbol_rolling_percentile_signal(
            per_symbol_intraday,
            long_q=0.60,
            lookback_days=10,
            bars_per_day=14,
            stay_q=0.60,
        )
