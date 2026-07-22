from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils import conformal
from case_studies.utils.allocation import compute_conformal_weights
from case_studies.utils.registry.specs import backtest_hash_from_parts
from case_studies.utils.strategy_analysis import rank_returns_on_common_support


def _write_predictions(root: Path, rows: list[dict[str, object]]) -> None:
    pred_dir = root / "run_log" / "predictions" / "candidate"
    pred_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(pred_dir / "predictions.parquet")


def test_walk_forward_widths_never_use_current_or_future_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(
        case_dir,
        [
            {
                "timestamp": datetime(2020, 1, 1),
                "symbol": "A",
                "y_true": 10.0,
                "y_score": 0.0,
                "fold_id": 7,
            },
            {
                "timestamp": datetime(2020, 1, 2),
                "symbol": "A",
                "y_true": 10.0,
                "y_score": 0.0,
                "fold_id": 7,
            },
            {
                "timestamp": datetime(2020, 1, 1),
                "symbol": "B",
                "y_true": 1.0,
                "y_score": 0.0,
                "fold_id": 7,
            },
            {
                "timestamp": datetime(2020, 2, 1),
                "symbol": "A",
                "y_true": 100.0,
                "y_score": 0.0,
                "fold_id": 3,
            },
            {
                "timestamp": datetime(2020, 2, 1),
                "symbol": "B",
                "y_true": 100.0,
                "y_score": 0.0,
                "fold_id": 3,
            },
            {
                "timestamp": datetime(2020, 2, 2),
                "symbol": "A",
                "y_true": 100.0,
                "y_score": 0.0,
                "fold_id": 3,
            },
        ],
    )
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    widths = conformal.compute_conformal_widths(
        "demo", "candidate", min_calibration_n=2, write=False
    )

    assert widths.filter(pl.col("fold_id") == 7).is_empty()
    later = (
        widths.filter(pl.col("fold_id") == 3)
        .select("symbol", "width", "calibration_scope", "calibration_version")
        .unique()
        .sort("symbol")
    )
    assert later["symbol"].to_list() == ["A", "B"]
    assert later["calibration_scope"].to_list() == ["symbol", "pooled"]
    assert later["calibration_version"].unique().to_list() == ["walk_forward_v2"]
    assert later.filter(pl.col("symbol") == "B")["width"].item() == 20.0


def test_conformal_allocation_rejects_missing_selected_widths() -> None:
    predictions = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 2, 1), datetime(2020, 2, 1)],
            "symbol": ["A", "B"],
            "y_score": [2.0, 1.0],
        }
    )
    incomplete_widths = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 2, 1)],
            "symbol": ["A"],
            "width": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing widths for selected assets"):
        compute_conformal_weights(predictions, incomplete_widths, top_k=2)


def test_calibration_contract_changes_backtest_identity() -> None:
    legacy = {
        "strategy": {
            "signal": {"method": "equal_weight_top_k", "top_k": 10},
            "allocation": {"method": "conformal_weighted", "top_k": 10},
        }
    }
    corrected = conformal.ensure_conformal_calibration_identity(legacy)

    assert corrected["strategy"]["allocation"]["calibration_version"] == "walk_forward_v2"
    assert backtest_hash_from_parts("pred", legacy) != backtest_hash_from_parts("pred", corrected)


def test_legacy_width_artifact_must_be_preserved_before_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case_studies" / "demo"
    _write_predictions(
        case_dir,
        [
            {
                "timestamp": datetime(2020, 1, 1),
                "symbol": "A",
                "y_true": 1.0,
                "y_score": 0.0,
                "fold_id": 0,
            },
            {
                "timestamp": datetime(2020, 1, 2),
                "symbol": "A",
                "y_true": 1.0,
                "y_score": 0.0,
                "fold_id": 0,
            },
            {
                "timestamp": datetime(2020, 2, 1),
                "symbol": "A",
                "y_true": 1.0,
                "y_score": 0.0,
                "fold_id": 1,
            },
        ],
    )
    legacy_path = case_dir / "run_log" / "predictions" / "candidate" / "conformal_widths.parquet"
    pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1)],
            "symbol": ["A"],
            "fold_id": [0],
            "width": [2.0],
            "alpha": [0.2],
            "calibration_n": [2],
        }
    ).write_parquet(legacy_path)
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: case_dir)

    with pytest.raises(ValueError, match="preserve it in the pre-fix snapshot"):
        conformal.compute_conformal_widths("demo", "candidate", min_calibration_n=2, write=True)


def test_common_support_ranking_uses_identical_timestamps() -> None:
    full = pl.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "daily_return": [0.50, 0.01, -0.01, 0.01],
        }
    )
    strict = pl.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "daily_return": [0.02, 0.00, 0.01],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))

    ranked = rank_returns_on_common_support({"full": full, "strict": strict}, periods_per_year=252)

    assert ranked["n_periods"].unique().to_list() == [3]
    assert ranked["start"].unique().to_list() == [datetime(2020, 1, 2)]
    assert ranked["end"].unique().to_list() == [datetime(2020, 1, 4)]
