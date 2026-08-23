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


def test_conformal_weight_floor_never_uses_future_timestamps() -> None:
    """Appending future widths must not change already emitted allocation weights."""
    early = datetime(2024, 1, 1)
    future = datetime(2024, 2, 1)
    symbols = [f"S{index:03d}" for index in range(100)]
    predictions = pl.DataFrame(
        {
            "timestamp": [early] * 100 + [future] * 100,
            "symbol": symbols * 2,
            "y_score": [float(100 - index) for index in range(100)] * 2,
        }
    )
    widths = pl.DataFrame(
        {
            "timestamp": [early] * 100 + [future] * 100,
            "symbol": symbols * 2,
            "width": [1e-9] + [1.0] * 99 + [1e-12] * 100,
        }
    )

    prefix = compute_conformal_weights(
        predictions.filter(pl.col("timestamp") == early),
        widths.filter(pl.col("timestamp") == early),
        top_k=100,
    ).sort("timestamp", "symbol")
    full = (
        compute_conformal_weights(predictions, widths, top_k=100)
        .filter(pl.col("timestamp") == early)
        .sort("timestamp", "symbol")
    )

    max_difference = (
        prefix.join(full, on=["timestamp", "symbol"], suffix="_full")
        .select((pl.col("weight") - pl.col("weight_full")).abs().max())
        .item()
    )
    assert max_difference < 1e-12


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


def test_locked_width_retry_rejects_conflict_without_replacing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "conformal_widths.parquet"
    original = pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1)],
            "symbol": ["A"],
            "fold_id": [-1],
            "width": [2.0],
            "alpha": [0.2],
            "calibration_version": [conformal.CALIBRATION_VERSION],
        }
    )
    original.write_parquet(path)
    conflicting = original.with_columns(pl.lit(3.0).alias("width"))

    with pytest.raises(ValueError, match="locked conformal artifact conflicts"):
        conformal._write_widths(path, conflicting, 0.2, immutable=True)

    assert pl.read_parquet(path).equals(original)


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


def test_conformal_allocation_accepts_tz_aware_widths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widths keep the predictions parquet's time zone; weights are always tz-naive.

    ``normalize_prediction_columns`` strips the time zone from predictions before
    signal weights are built, while ``compute_conformal_widths`` reads
    predictions.parquet directly and preserves it. Case studies that store UTC-aware
    timestamps (crypto_perps_funding) therefore reach ``_apply_allocation`` with a
    tz-aware widths frame and a tz-naive weights frame, which raised
    ``SchemaError: datatypes of join keys don't match`` on the common-support join.
    """
    import case_studies.utils.backtest_loaders as loaders
    from case_studies.utils.backtest_runner import _apply_allocation

    stamps = [datetime(2024, 1, 1, 8), datetime(2024, 1, 1, 16)]
    naive = pl.DataFrame(
        {
            "timestamp": stamps * 2,
            "symbol": ["BTC", "BTC", "ETH", "ETH"],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
    weights = naive.with_columns(weight=pl.lit(0.5))
    predictions = naive.with_columns(y_score=pl.Series([2.0, 2.0, 1.0, 1.0]))
    widths = naive.with_columns(
        width=pl.lit(1.0),
        timestamp=pl.col("timestamp").dt.replace_time_zone("UTC"),
    )
    assert widths["timestamp"].dtype.time_zone == "UTC"

    monkeypatch.setattr(loaders, "get_rebalance_step", lambda *_args: 1)

    result = _apply_allocation(
        weights,
        predictions,
        naive.with_columns(close=pl.lit(100.0)),
        {"method": "conformal_weighted", "top_k": 2},
        label="fwd_ret_24h",
        case_study="crypto_perps_funding",
        prediction_hash="unused_widths_are_passed_in",
        conformal_widths=widths,
    )

    assert result.height == 4
    assert result["timestamp"].dtype == weights["timestamp"].dtype


def _overlay_registry(workspace: Path, source_root: Path, prediction_hash: str) -> None:
    """Record where a workspace's copied prediction row says its artifact stayed."""
    import sqlite3

    registry = workspace / "run_log" / "registry.db"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(registry) as db:
        db.execute(
            "CREATE TABLE overlay_references ("
            "result_hash TEXT, result_kind TEXT, source_root TEXT, created_at TEXT)"
        )
        db.execute(
            "INSERT INTO overlay_references VALUES (?,?,?,?)",
            (prediction_hash, "prediction", str(source_root), "2026-08-23T00:00:00+00:00"),
        )


def _two_fold_rows() -> list[dict[str, object]]:
    return (
        [
            {
                "timestamp": datetime(2020, 1, 1),
                "symbol": s,
                "y_true": t,
                "y_score": 0.0,
                "fold_id": 1,
            }
            for s, t in (("A", 10.0), ("B", 1.0))
        ]
        + [
            {
                "timestamp": datetime(2020, 1, 2),
                "symbol": s,
                "y_true": t,
                "y_score": 0.0,
                "fold_id": 1,
            }
            for s, t in (("A", 10.0), ("B", 1.0))
        ]
        + [
            {
                "timestamp": datetime(2020, 2, 1),
                "symbol": s,
                "y_true": 100.0,
                "y_score": 0.0,
                "fold_id": 2,
            }
            for s in ("A", "B")
        ]
    )


def test_widths_calibrate_from_a_prediction_the_workspace_only_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workspace run copies the registry rows and leaves the parquet where it was.

    Resolving the artifact from the active case directory alone finds nothing, so the
    conformal allocator used to fail on every prediction the caller did not fit themselves.
    """
    release = tmp_path / "release" / "demo"
    _write_predictions(release, _two_fold_rows())
    workspace = tmp_path / "workspace" / "demo"
    _overlay_registry(workspace, release, "candidate")
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: workspace)

    widths = conformal.compute_conformal_widths("demo", "candidate", min_calibration_n=2)

    assert not widths.is_empty()
    assert widths.filter(pl.col("fold_id") == 1).is_empty()
    local = workspace / "run_log" / "predictions" / "candidate" / "conformal_widths.parquet"
    released = release / "run_log" / "predictions" / "candidate" / "conformal_widths.parquet"
    assert local.is_file(), "derived widths belong in the run log doing the deriving"
    assert not released.exists(), "a workspace run must not write into the released run log"


def test_a_missing_prediction_still_names_the_directory_it_was_expected_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace" / "demo"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(conformal, "get_case_study_dir", lambda _: workspace)

    with pytest.raises(FileNotFoundError, match=str(workspace)):
        conformal.compute_conformal_widths("demo", "candidate", write=False)
