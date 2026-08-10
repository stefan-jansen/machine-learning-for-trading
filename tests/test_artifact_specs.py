from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.backtest_loaders import (
    get_backtest_config,
    load_backtest_prices,
)
from utils.modeling import load_modeling_dataset


def test_crypto_modeling_splits_use_label_clock_before_feature_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import utils.modeling as modeling

    case_dir = tmp_path / "crypto_perps_funding"
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "features").mkdir()
    (case_dir / "labels").mkdir()
    (case_dir / "config" / "setup.yaml").write_text(
        "labels:\n  primary: fwd_ret_8h\n  buffer: 8H\n"
    )

    full_ts = pl.datetime_range(datetime(2020, 1, 1), datetime(2021, 1, 3), "1d", eager=True)
    warm_ts = full_ts.filter(full_ts >= datetime(2021, 1, 1))
    pl.DataFrame(
        {
            "timestamp": full_ts,
            "symbol": ["AAA"] * len(full_ts),
            "fwd_ret_8h": [0.1] * len(full_ts),
        }
    ).write_parquet(case_dir / "labels" / "fwd_ret_8h.parquet")
    pl.DataFrame(
        {"timestamp": warm_ts, "symbol": ["AAA"] * len(warm_ts), "feature": [1.0] * len(warm_ts)}
    ).write_parquet(case_dir / "features" / "financial.parquet")

    captured = {}

    def capture_splits(frame, **_kwargs):
        captured["minimum"] = frame["timestamp"].min()
        return [{"fold": 0, "val_start": full_ts[0], "val_end": full_ts[-1]}]

    monkeypatch.setattr(modeling, "get_case_study_dir", lambda _case_id: case_dir)
    monkeypatch.setattr(modeling, "load_feature_spec", lambda *_args: {})
    monkeypatch.setattr(modeling, "load_label_spec", lambda *_args: {})
    monkeypatch.setattr(
        modeling,
        "resolve_storage_path",
        lambda _case_id, _spec, fallback: case_dir / fallback,
    )
    monkeypatch.setattr(modeling, "resolve_label_buffer", lambda *_args: "8H")
    monkeypatch.setattr(modeling, "generate_cv_splits", capture_splits)
    monkeypatch.setattr(modeling, "make_wf_config", lambda *_args, **_kwargs: None)

    result = modeling.load_modeling_dataset("crypto_perps_funding", "fwd_ret_8h")

    assert result.dataset["timestamp"].min() == datetime(2021, 1, 1)
    assert captured["minimum"] == datetime(2020, 1, 1)


def test_us_equities_pilot_helpers_preserve_current_outputs() -> None:
    bt = get_backtest_config("us_equities_panel")
    prices = load_backtest_prices("us_equities_panel", max_symbols=2)
    mds = load_modeling_dataset("us_equities_panel", "fwd_ret_1d", max_symbols=2)

    assert bt.primary_label == "fwd_ret_1d"
    assert bt.label_buffer == "1D"
    assert bt.calendar == "NYSE"
    assert bt.cadence == "daily_close"

    assert prices.columns == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert prices["symbol"].n_unique() == 2

    assert mds.label_col == "fwd_ret_1d"
    assert mds.date_col == "timestamp"
    assert mds.entity_cols == ["symbol"]
    assert mds.join_cols == ["symbol", "timestamp"]
    # 62 financial + 9 temporal. `03_financial_features` dropped `size_rank`, which was a
    # bit-for-bit duplicate of `liq_rank`, and both artifacts on disk carry the new counts.
    assert len(mds.feature_names) == 71
    assert len(mds.splits) == 16
    assert mds.label_buffer == "1D"
    assert mds.task_type == "regression"


def test_microstructure_pilot_helpers_preserve_current_outputs() -> None:
    bt = get_backtest_config("nasdaq100_microstructure")
    prices = load_backtest_prices("nasdaq100_microstructure", max_symbols=2)
    mds = load_modeling_dataset("nasdaq100_microstructure", "fwd_ret_15m", max_symbols=2)

    assert bt.primary_label == "fwd_ret_15m"
    assert bt.label_buffer == "15min"
    assert bt.calendar == "NYSE"
    assert bt.cadence == "15_minute"

    # Microstructure carries OHLCV + bid/ask OHLC so the backtest engine can
    # cost spread-aware fills.
    required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    assert all(c in prices.columns for c in required_cols)
    assert "bid_close" in prices.columns and "ask_close" in prices.columns
    assert prices["symbol"].n_unique() == 2

    assert mds.label_col == "fwd_ret_15m"
    assert mds.date_col == "timestamp"
    assert mds.entity_cols == ["symbol"]
    assert mds.join_cols == ["symbol", "timestamp"]
    assert len(mds.feature_names) == 88
    assert len(mds.splits) == 2
    assert mds.label_buffer == "15min"
    assert mds.task_type == "regression"


def test_universe_reduction_breaks_row_count_ties_on_entity_name() -> None:
    """Tied row counts must reduce to the same universe on every call.

    ``max_symbols`` picks the entities with the most rows. When counts tie at the
    cutoff, an unstable sort lets two callers reducing the same dataset to the
    same size pick different symbols - a reduced stage-04 run and the reduced
    model notebooks downstream of it, for instance. The symbols only one of them
    chose then carry null model-based features, which runs clean and is wrong.

    The panel gives every symbol the same row count, so the cutoff is decided
    entirely by the tie-break, and the frame is then reordered to show the
    reduction does not follow frame order.
    """
    import polars as pl

    from utils.modeling import reduce_to_top_entities

    dataset = pl.DataFrame(
        {
            "symbol": [s for s in ("DELTA", "ALPHA", "CHARLIE", "BRAVO") for _ in range(3)],
            "value": list(range(12)),
        }
    )

    def kept(frame: pl.DataFrame) -> list[str]:
        return sorted(reduce_to_top_entities(frame, "symbol", 2)["symbol"].unique())

    assert kept(dataset) == ["ALPHA", "BRAVO"]
    assert kept(dataset.sort("value", descending=True)) == ["ALPHA", "BRAVO"]
    assert kept(dataset.sample(fraction=1.0, shuffle=True, seed=7)) == ["ALPHA", "BRAVO"]


def test_universe_reduction_prefers_history_over_name() -> None:
    """The name is the tie-break, never the criterion: more rows still wins."""
    import polars as pl

    from utils.modeling import reduce_to_top_entities

    dataset = pl.DataFrame(
        {"symbol": ["ZULU"] * 5 + ["ALPHA"] * 2 + ["BRAVO"] * 2, "value": list(range(9))}
    )
    assert sorted(reduce_to_top_entities(dataset, "symbol", 2)["symbol"].unique()) == [
        "ALPHA",
        "ZULU",
    ]
