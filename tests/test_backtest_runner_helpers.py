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

from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import polars as pl
import pytest

from case_studies.utils.backtest_runner import (
    _MAX_NULL_RATE,
    _align_symbol_dtype,
    _target_weights_by_timestamp,
    apply_universe_filter,
    run_plumbing_test,
    substitute_continuous_return_for_classification,
)


def test_max_null_rate_constant_default() -> None:
    assert _MAX_NULL_RATE == 0.10


def test_vectorized_plumbing_test_runs_random_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import case_studies.utils.backtest_runner as br

    spec = {
        "version": 2,
        "strategy": {"rebalance": {"mode": "vectorized"}},
        "backtest_config": {},
    }
    predictions = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "symbol": ["A", "B"],
            "y_score": [0.8, 0.2],
            "y_true": [0.1, -0.1],
        }
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(br, "get_backtest_config", lambda _: object())
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kwargs: args[2])

    def fake_run_backtest(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metrics={"sharpe": 0.25})

    monkeypatch.setattr(br, "run_backtest", fake_run_backtest)

    observed = run_plumbing_test(
        "demo",
        pl.DataFrame(),
        spec,
        predictions=predictions,
        label="fwd_ret_1m",
        seed=7,
    )

    randomized = captured["predictions"]
    assert isinstance(randomized, pl.DataFrame)
    assert observed == 0.25
    assert captured["register"] is False
    assert randomized["y_true"].to_list() == predictions["y_true"].to_list()
    assert randomized["y_score"].to_list() != predictions["y_score"].to_list()


def test_vectorized_plumbing_test_resolves_primary_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import case_studies.utils.backtest_runner as br
    import case_studies.utils.registry as registry

    spec = {
        "version": 2,
        "strategy": {"rebalance": {"mode": "vectorized"}},
        "backtest_config": {},
    }
    predictions = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "symbol": ["A", "B"],
            "y_score": [0.8, 0.2],
            "y_true": [0.1, -0.1],
        }
    )
    requested: dict[str, object] = {}
    backtest_call: dict[str, object] = {}

    monkeypatch.setattr(
        br,
        "get_backtest_config",
        lambda _: SimpleNamespace(primary_label="fwd_ret_1d"),
    )
    monkeypatch.setattr(br, "ensure_backtest_spec", lambda *args, **kwargs: args[2])

    def fake_index(case_study, *, label, split):
        requested.update(case_study=case_study, label=label, split=split)
        return pl.DataFrame({"prediction_hash": ["abc123"]})

    monkeypatch.setattr(registry, "load_prediction_index", fake_index)
    monkeypatch.setattr(registry, "read_predictions", lambda *_args: predictions)

    def fake_backtest(*args, **kwargs):
        backtest_call["prediction_hash"] = args[1]
        backtest_call.update(kwargs)
        return SimpleNamespace(metrics={"sharpe": 0.0})

    monkeypatch.setattr(br, "run_backtest", fake_backtest)

    observed = run_plumbing_test("demo", pl.DataFrame(), spec)

    assert observed == 0.0
    assert requested == {
        "case_study": "demo",
        "label": "fwd_ret_1d",
        "split": "validation",
    }
    assert backtest_call["prediction_hash"] == "abc123"
    assert backtest_call["register"] is False


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


def test_target_weights_are_deterministic_across_input_order() -> None:
    timestamp = datetime(2024, 1, 2)
    weights = pl.DataFrame(
        {
            "timestamp": [timestamp, timestamp, timestamp],
            "symbol": ["C", "A", "B"],
            "weight": [0.2, 0.5, 0.3],
        }
    )

    expected = {timestamp: {"A": 0.5, "B": 0.3, "C": 0.2}}
    assert _target_weights_by_timestamp(weights) == expected
    assert _target_weights_by_timestamp(weights.reverse()) == expected


def test_target_weights_reject_duplicate_keys() -> None:
    timestamp = datetime(2024, 1, 2)
    weights = pl.DataFrame(
        {
            "timestamp": [timestamp, timestamp],
            "symbol": ["A", "A"],
            "weight": [0.5, -0.5],
        }
    )
    with pytest.raises(ValueError, match="duplicate timestamp-symbol"):
        _target_weights_by_timestamp(weights)


def test_precompute_weights_forwards_prediction_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    import case_studies.utils.backtest_runner as br

    timestamp = datetime(2024, 1, 2)
    predictions = pl.DataFrame({"timestamp": [timestamp], "symbol": ["BTC"], "y_score": [0.1]})
    base_weights = pl.DataFrame({"timestamp": [timestamp], "symbol": ["BTC"], "weight": [1.0]})
    captured = {}

    monkeypatch.setattr(br, "build_target_weights_from_config", lambda *_args: base_weights)

    def fake_apply(*_args, **kwargs):
        captured.update(kwargs)
        return base_weights

    monkeypatch.setattr(br, "_apply_allocation", fake_apply)
    result = br.precompute_weights(
        predictions,
        {
            "signal": {"method": "equal_weight_top_k"},
            "allocation": {"method": "conformal_weighted"},
        },
        pl.DataFrame(),
        label="fwd_ret_24h",
        case_study="crypto_perps_funding",
        prediction_hash="current_mae_pit",
    )

    assert result.equals(base_weights)
    assert captured["prediction_hash"] == "current_mae_pit"


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

    # A Path, not a str: `get_case_study_dir` joins this with `/`, which is how the labels
    # artifact is now resolved so that output isolation reaches it.
    monkeypatch.setattr(_utils, "CASE_STUDIES_DIR", Path(tmp_path), raising=False)
    monkeypatch.setattr(br, "CASE_STUDIES_DIR", Path(tmp_path), raising=False)

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

    # A Path, not a str: `get_case_study_dir` joins this with `/`, which is how the labels
    # artifact is now resolved so that output isolation reaches it.
    monkeypatch.setattr(_utils, "CASE_STUDIES_DIR", Path(tmp_path), raising=False)
    monkeypatch.setattr(br, "CASE_STUDIES_DIR", Path(tmp_path), raising=False)

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


def test_the_continuous_return_label_is_read_from_the_isolated_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label parquet is generated output, so ML4T_OUTPUT_DIR decides where it is read from.

    The configuration stays in the checkout - it is source - and only the artifact moves. The
    two directories are the same whenever artifacts sit beside the source, which is why reading
    the checkout was never wrong in production and always wrong under isolation: measured in CI
    on `crypto_perps_funding` 13_backtest, which raised FileNotFoundError for a label the same
    run had just written.
    """
    cs = "isolated_cs"
    source = tmp_path / "checkout"
    (source / cs / "config").mkdir(parents=True)
    (source / cs / "config" / "setup.yaml").write_text(
        dedent(
            """
            labels:
              classification_eval_label:
                fwd_dir_1d: fwd_ret_1d
            """
        ).strip()
    )
    # Deliberately no labels/ in the checkout: the only copy is in the output root.
    output = tmp_path / "output"
    (output / cs / "labels").mkdir(parents=True)
    d1 = datetime(2024, 1, 2)
    pl.DataFrame(
        {"timestamp": [d1, d1], "symbol": ["A", "B"], "fwd_ret_1d": [0.011, 0.031]}
    ).write_parquet(output / cs / "labels" / "fwd_ret_1d.parquet")

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

    monkeypatch.setattr(_utils, "CASE_STUDIES_DIR", source, raising=False)
    monkeypatch.setattr(br, "CASE_STUDIES_DIR", source, raising=False)
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(output))

    out = substitute_continuous_return_for_classification(
        predictions, case_study=cs, label="fwd_dir_1d"
    )

    assert out["y_true"].to_list() == [0.011, 0.031]


class TestAnAllocationThatProducedNoTargetIsRefused:
    """A run with no target weight at any rebalance is an absence, not a Sharpe of 0.0.

    `_refuse_an_allocation_that_produced_no_target` runs immediately before
    `register_backtest_run`, so the row never reaches the registry. It matters because nothing
    downstream filters it out of the trial count: `cohort_metrics` lists cohort members straight
    from `backtest_runs` with no zero-trade clause, so an absence is counted as a trial against
    every real candidate beside it (ml4t/agent-workspace#1004).
    """

    @staticmethod
    def _returns(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
                "daily_return": [0.0] * n,
            }
        )

    @staticmethod
    def _weights(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
                "symbol": ["EUR_USD"] * n,
                "weight": [1.0] * n,
            },
            schema={"timestamp": pl.Datetime, "symbol": pl.String, "weight": pl.Float64},
        )

    SPEC = {
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 2},
            "allocation": {"method": "mvo_ledoit_wolf", "lookback": 63},
            "rebalance": {"cadence": "daily_ny_close", "min_weight_change": 0.005},
        }
    }

    def test_an_empty_weight_frame_over_a_real_window_raises(self) -> None:
        from case_studies.utils.backtest_runner import (
            _refuse_an_allocation_that_produced_no_target,
        )

        with pytest.raises(ValueError, match="no target weight at any rebalance"):
            _refuse_an_allocation_that_produced_no_target(
                self._weights(0), self._returns(2063), self.SPEC
            )

    def test_a_run_with_targets_passes(self) -> None:
        from case_studies.utils.backtest_runner import (
            _refuse_an_allocation_that_produced_no_target,
        )

        _refuse_an_allocation_that_produced_no_target(
            self._weights(2063), self._returns(2063), self.SPEC
        )

    def test_a_short_fixture_panel_that_books_no_order_passes(self) -> None:
        # A one-bar CI fixture has a target and no later bar to fill it on under `next_bar`
        # execution, so it books zero orders legitimately. An earlier form of this guard tested
        # `num_trades == 0` and stopped eleven such fixture backtests across
        # `test_research_contract_execution` and `test_cme_futures_research`.
        from case_studies.utils.backtest_runner import (
            _refuse_an_allocation_that_produced_no_target,
        )

        _refuse_an_allocation_that_produced_no_target(self._weights(1), self._returns(1), self.SPEC)

    def test_an_empty_return_series_is_left_to_its_own_diagnosis(self) -> None:
        from case_studies.utils.backtest_runner import (
            _refuse_an_allocation_that_produced_no_target,
        )

        _refuse_an_allocation_that_produced_no_target(self._weights(0), self._returns(0), {})
