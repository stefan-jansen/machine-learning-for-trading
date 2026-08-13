from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from case_studies.utils.benchmark import (
    _decision_periods_per_year,
    _resolve_decision_cadence,
    build_equal_weight_benchmark,
    generate_benchmark,
    load_benchmark_metrics,
    load_benchmark_returns,
    write_benchmark,
)
from case_studies.utils.paired_metrics import _align_challenger_to_benchmark_periods


def _minute_labels() -> pl.DataFrame:
    rows = []
    for day, a_return, b_return in (
        (date(2021, 6, 30), 0.01, 0.03),
        (date(2021, 7, 1), 0.02, 0.04),
    ):
        start = datetime.combine(day, datetime.min.time()).replace(hour=9, minute=30)
        for offset in range(6):
            timestamp = start + timedelta(minutes=offset)
            rows.extend(
                (
                    {"timestamp": timestamp, "symbol": "A", "label": a_return},
                    {"timestamp": timestamp, "symbol": "B", "label": b_return},
                    {"timestamp": timestamp, "symbol": "EXTRA", "label": 0.99},
                )
            )
    return pl.DataFrame(rows)


def _build() -> tuple[pl.DataFrame, dict]:
    return build_equal_weight_benchmark(
        _minute_labels(),
        case_study="sample",
        label="label",
        symbols=["A", "B"],
        windows={
            "validation": (date(2021, 6, 30), date(2021, 6, 30)),
            "holdout": (date(2021, 7, 1), date(2021, 7, 1)),
        },
        cadence="1_minute",
        rebalance_step=4,
        calendar="NYSE",
        periods_per_year=252,
        label_digest="label-digest",
    )


def test_benchmark_uses_declared_roster_and_restarts_schedule_each_session() -> None:
    returns, metadata = _build()

    assert returns.to_dicts() == [
        {"timestamp": date(2021, 6, 30), "ew_return": pytest.approx(1.02**2 - 1)},
        {"timestamp": date(2021, 7, 1), "ew_return": pytest.approx(1.03**2 - 1)},
    ]
    assert metadata["n_symbols_in_universe"] == 2
    assert metadata["n_symbols_observed"] == 2
    assert metadata["by_period"]["validation"]["n_periods"] == 1
    assert metadata["by_period"]["holdout"]["n_periods"] == 1
    assert metadata["inputs"]["label_digest"] == "label-digest"
    assert metadata["configuration"]["rebalance_step"] == 4
    assert metadata["periods_per_year"] == 252.0
    daily_values = returns.get_column("ew_return").to_list()
    daily_mean = sum(daily_values) / len(daily_values)
    daily_std = abs(daily_values[1] - daily_values[0]) / math.sqrt(2.0)
    assert metadata["sharpe"] == pytest.approx(daily_mean / daily_std * math.sqrt(252.0))


def test_intraday_benchmark_pairs_as_daily_compounded_returns() -> None:
    benchmark, metadata = _build()
    challenger = pl.DataFrame(
        {
            "timestamp": [date(2021, 6, 29), date(2021, 6, 30), date(2021, 7, 1)],
            "ret": [0.50, 0.01, 0.02],
        }
    )

    aligned = _align_challenger_to_benchmark_periods(
        challenger,
        benchmark.rename({"ew_return": "ret"}),
        benchmark_periods_per_year=metadata["periods_per_year"],
        daily_periods_per_year=252.0,
    )

    assert aligned.to_dicts() == [
        {
            "timestamp": date(2021, 6, 30),
            "ret": 0.01,
            "ret_b": pytest.approx(1.02**2 - 1.0),
        },
        {
            "timestamp": date(2021, 7, 1),
            "ret": 0.02,
            "ret_b": pytest.approx(1.03**2 - 1.0),
        },
    ]


def test_benchmark_aligns_minute_labels_to_fifteen_minute_decision_grid() -> None:
    start = datetime(2021, 6, 30, 9, 30)
    labels = pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=offset) for offset in range(61)],
            "symbol": ["A"] * 61,
            "label": [0.01] * 61,
        }
    )

    returns, _ = build_equal_weight_benchmark(
        labels,
        case_study="sample",
        label="label",
        symbols=["A"],
        windows={"validation": (date(2021, 6, 30), date(2021, 6, 30))},
        cadence="15_minute",
        rebalance_step=4,
        calendar="NYSE",
        periods_per_year=252,
        label_digest="digest",
    )

    assert returns.to_dicts() == [
        {"timestamp": date(2021, 6, 30), "ew_return": pytest.approx(1.01**2 - 1)},
    ]


def test_sparse_daily_schedule_uses_decision_period_annualization() -> None:
    dates = pl.date_range(date(2021, 1, 1), date(2021, 3, 31), eager=True)
    labels = pl.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["A"] * len(dates),
            "label": [0.01] * len(dates),
        }
    )

    returns, metadata = build_equal_weight_benchmark(
        labels,
        case_study="sample",
        label="label",
        symbols=["A"],
        windows={"validation": (date(2021, 1, 1), date(2021, 3, 31))},
        cadence="daily_close",
        rebalance_step=21,
        calendar="NYSE",
        periods_per_year=252,
        label_digest="digest",
    )

    assert returns.height == 5
    assert metadata["periods_per_year"] == 12.0
    assert metadata["by_period"]["validation"]["periods_per_year"] == 12.0
    assert metadata["configuration"]["daily_periods_per_year"] == 252


@pytest.mark.parametrize(
    ("cadence", "step", "expected"),
    [
        ("monthly_month_end", 1, 12.0),
        ("weekly_friday", 2, 26.0),
        ("daily_close", 21, 12.0),
        ("8_hour_funding_aligned", 3, 365.0),
        ("1_minute", 14, 252.0),
    ],
)
def test_decision_annualization_matches_the_retained_schedule(
    cadence: str, step: int, expected: float
) -> None:
    assert (
        _decision_periods_per_year(
            cadence, step, 252 if cadence != "8_hour_funding_aligned" else 365
        )
        == expected
    )


def test_sparse_benchmark_alignment_compounds_daily_challenger_periods() -> None:
    challenger = pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2021, 1, 2), date(2021, 2, 12), eager=True),
            "ret": [0.01] * 42,
        }
    )
    benchmark = pl.DataFrame(
        {
            "timestamp": [date(2021, 1, 1), date(2021, 1, 22), date(2021, 2, 12)],
            "ret": [0.20, 0.21, 0.22],
        }
    )

    aligned = _align_challenger_to_benchmark_periods(
        challenger,
        benchmark,
        benchmark_periods_per_year=12.0,
        daily_periods_per_year=252.0,
    )

    assert aligned.to_dicts() == [
        {
            "timestamp": date(2021, 1, 1),
            "ret": pytest.approx(1.01**21 - 1.0),
            "ret_b": 0.20,
        },
        {
            "timestamp": date(2021, 1, 22),
            "ret": pytest.approx(1.01**21 - 1.0),
            "ret_b": 0.21,
        },
    ]


def test_paired_metric_path_uses_benchmark_frequency_and_compounded_periods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from case_studies.utils import paired_metrics

    decisions = [date(2021, 1, 1) + timedelta(days=3 * offset) for offset in range(8)]
    benchmark = pl.DataFrame({"timestamp": decisions, "ret": [0.02] * len(decisions)})
    challenger_dates = pl.date_range(decisions[0] + timedelta(days=1), decisions[-1], eager=True)
    challenger = pl.DataFrame(
        {"timestamp": challenger_dates, "ret": [0.01] * len(challenger_dates)}
    )
    observed: dict[str, object] = {}

    def compute(challenger_values, benchmark_values, *, periods_per_year, **kwargs):
        observed["challenger"] = challenger_values
        observed["benchmark"] = benchmark_values
        observed["computed_ppy"] = periods_per_year
        return {"sharpe_diff": 0.0, "n_obs": len(challenger_values)}

    def register(*args, periods_per_year, **kwargs):
        observed["registered_ppy"] = periods_per_year

    monkeypatch.setattr(paired_metrics, "compute_paired_uncertainty", compute)
    monkeypatch.setattr(paired_metrics, "register_paired_metrics", register)

    result = paired_metrics._populate_pair(
        "sample",
        "challenger",
        "benchmark",
        "equal_weight_side_artifact",
        challenger,
        benchmark,
        252,
        "label",
        comparison_periods_per_year=12.0,
        write_case_dir=tmp_path,
    )

    assert "skip" not in result
    assert observed["challenger"].tolist() == pytest.approx([1.01**3 - 1.0] * 7)
    assert observed["benchmark"].tolist() == pytest.approx([0.02] * 7)
    assert observed["computed_ppy"] == 12.0
    assert observed["registered_ppy"] == 12.0


def test_cme_benchmark_uses_product_as_the_panel_key() -> None:
    labels = _minute_labels().rename({"symbol": "product"})

    returns, metadata = build_equal_weight_benchmark(
        labels,
        case_study="cme_futures",
        label="label",
        symbols=["A", "B"],
        windows={"validation": (date(2021, 6, 30), date(2021, 7, 1))},
        cadence="1_minute",
        rebalance_step=4,
        calendar="CME",
        periods_per_year=252,
        label_digest="digest",
    )

    assert returns.height == 2
    assert metadata["configuration"]["entity_col"] == "product"
    assert metadata["n_symbols_observed"] == 2


def test_benchmark_cadence_uses_shared_configuration_precedence() -> None:
    assert (
        _resolve_decision_cadence(
            {
                "decision": {
                    "entry_cadence": "weekly_friday",
                    "cadence": "daily_close",
                    "bar_frequency": "1_minute",
                }
            }
        )
        == "weekly_friday"
    )


@pytest.mark.parametrize("key", ["entry_cadence", "cadence", "bar_frequency"])
def test_benchmark_cadence_accepts_each_configuration_key(key: str) -> None:
    assert _resolve_decision_cadence({"decision": {key: "1_minute"}}) == "1_minute"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda frame: pl.concat([frame, frame.head(1)]),
            "duplicate \\(timestamp, symbol\\)",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(float("inf"))
                .otherwise(pl.col("label"))
                .alias("label")
            ),
            "non-finite label",
        ),
    ],
)
def test_benchmark_rejects_invalid_label_rows(mutate, match: str) -> None:
    frame = mutate(_minute_labels())
    with pytest.raises(ValueError, match=match):
        build_equal_weight_benchmark(
            frame,
            case_study="sample",
            label="label",
            symbols=["A", "B"],
            windows={"validation": (date(2021, 6, 30), date(2021, 7, 1))},
            cadence="1_minute",
            rebalance_step=4,
            calendar="NYSE",
            periods_per_year=252,
            label_digest="digest",
        )


def test_writer_replaces_both_artifacts_without_temp_residue(tmp_path: Path) -> None:
    returns, metadata = _build()
    parquet_path, json_path = write_benchmark(
        returns,
        metadata,
        output_dir=tmp_path,
        label="label",
    )

    assert pl.read_parquet(parquet_path).equals(returns)
    assert json.loads(json_path.read_text())["output_digest"] == metadata["output_digest"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_generator_reads_declared_universe_and_label_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    case_dir = tmp_path / "sample"
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "labels").mkdir()
    setup = {
        "universe": {"symbols": ["A", "B"], "n_assets": 2},
        "decision": {"bar_frequency": "1_minute"},
        "evaluation": {"calendar": "NYSE", "periods_per_year": 252},
        "labels": {"rebalance_step": {"label": 4}},
    }
    (case_dir / "config" / "setup.yaml").write_text(yaml.safe_dump(setup))
    label_path = case_dir / "labels" / "label.parquet"
    _minute_labels().write_parquet(label_path)
    label_path.with_suffix(".parquet.digest.json").write_text(json.dumps({"digest": "source"}))

    parquet_path, json_path = generate_benchmark(
        "sample",
        "label",
        windows={
            "validation": (date(2021, 6, 30), date(2021, 6, 30)),
            "holdout": (date(2021, 7, 1), date(2021, 7, 1)),
        },
    )

    assert parquet_path == case_dir / "benchmark" / "label.parquet"
    assert json.loads(json_path.read_text())["inputs"]["label_digest"] == "source"
    assert load_benchmark_returns("sample", "label", period="validation").height == 1
    assert load_benchmark_returns("sample", "label", period="holdout").height == 1
    metrics = load_benchmark_metrics("sample", "label", period="overall")
    assert metrics is not None
    assert metrics["n_periods"] == 2
