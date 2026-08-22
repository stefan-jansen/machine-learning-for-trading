"""Regression tests for cached TabM and sequence-DL result reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from case_studies.utils import cv_results, deep_learning


@dataclass
class _CompleteStatus:
    complete: bool = True
    partial: bool = False

    def summary(self) -> str:
        return "complete"


def _predictions(config: str, epoch: int, *, n_dates: int = 3) -> pl.DataFrame:
    rows = []
    for fold_id, day in enumerate(range(n_dates)):
        for symbol_id in range(5):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "symbol": f"S{symbol_id}",
                    "fold_id": fold_id,
                    "y_true": float(symbol_id),
                    "y_score": float(symbol_id) + epoch / 1000,
                    "config": config,
                    "epoch": epoch,
                }
            )
    return pl.DataFrame(rows)


def _patch_complete_registry(monkeypatch: pytest.MonkeyPatch, family: str) -> None:
    prediction_hash = f"{family}-pred"

    def build_spec(_family, config_name, _label, **_kwargs):
        return {"family": _family, "config_name": config_name}

    def prediction_sets(_case_study, *, training_hash, split):
        assert split == "validation"
        return pl.DataFrame(
            {
                "prediction_hash": [prediction_hash],
                "checkpoint_value": [10],
                "training_hash": [training_hash],
            }
        )

    def prediction_metrics(_case_study, *, prediction_hash: str):
        assert prediction_hash == f"{family}-pred"
        return pl.DataFrame(
            {
                "ic_mean": [0.03],
                "ic_std": [0.01],
                "ic_n_days": [3.0],
            }
        )

    predictions = _predictions("model", 10).drop("config", "epoch")

    import case_studies.utils.registry as registry

    monkeypatch.setattr(registry, "build_training_spec", build_spec)
    monkeypatch.setattr(registry, "training_hash_from_spec", lambda spec: spec["config_name"])
    monkeypatch.setattr(
        registry, "training_run_status", lambda *_args, **_kwargs: _CompleteStatus()
    )
    monkeypatch.setattr(registry, "load_prediction_sets", prediction_sets)
    monkeypatch.setattr(cv_results, "build_training_spec", build_spec)
    monkeypatch.setattr(cv_results, "training_hash_from_spec", lambda spec: spec["config_name"])
    monkeypatch.setattr(cv_results, "load_prediction_sets", prediction_sets)
    monkeypatch.setattr(cv_results, "load_prediction_metrics", prediction_metrics)
    monkeypatch.setattr(cv_results, "read_predictions", lambda *_args, **_kwargs: predictions)


def test_all_cached_dl_runner_returns_registered_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = "deep_learning"
    _patch_complete_registry(monkeypatch, family)
    config = {
        "family": family,
        "config_name": "model",
        "n_epochs": 10,
        "params": {"architecture": "lstm", "lookback": 2},
    }
    common = {
        "configs": [config],
        "feature_names": ["feature"],
        "label_col": "target",
        "date_col": "timestamp",
        "entity_col": "symbol",
        "register": True,
        "case_study": "example",
        "save_dir": tmp_path,
    }

    result = deep_learning.run_dl_cv(
        pd.DataFrame(),
        [{"fold": 0}],
        device="cpu",
        n_features=1,
        **common,
    )

    assert result["best_config_name"] == "model"
    assert result["best_epoch"] == 10
    assert result["best_ic"] == pytest.approx(0.03)
    assert result["all_predictions"].height == 15


def test_full_coverage_checkpoint_beats_higher_partial_ic() -> None:
    curves = pl.DataFrame(
        [
            {"config": "lstm", "epoch": 10, "ic_mean": 0.0301, "ic_n_days": 2016},
            {"config": "lstm", "epoch": 25, "ic_mean": 0.0521, "ic_n_days": 1536},
            {"config": "tsmixer", "epoch": 15, "ic_mean": 0.0290, "ic_n_days": 2016},
        ]
    )
    predictions = pl.concat(
        [
            _predictions("lstm", 10),
            _predictions("lstm", 25, n_dates=2),
            _predictions("tsmixer", 15),
        ]
    )

    result = cv_results.assemble_cv_result(
        curves,
        predictions,
        date_col="timestamp",
        entity_col="symbol",
    )

    assert result["best_config_name"] == "lstm"
    assert result["best_epoch"] == 10
    assert result["best_ic"] == pytest.approx(0.0301)
    assert result["full_coverage_days"] == 2016


def test_registry_rebuild_deduplicates_checkpoint_prediction_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv_results,
        "build_training_spec",
        lambda family, config_name, label, **kwargs: {"config_name": config_name},
    )
    monkeypatch.setattr(cv_results, "training_hash_from_spec", lambda spec: spec["config_name"])
    monkeypatch.setattr(
        cv_results,
        "load_prediction_sets",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "prediction_hash": ["duplicate-b", "duplicate-a"],
                "checkpoint_value": [10, 10],
            }
        ),
    )
    monkeypatch.setattr(
        cv_results,
        "load_prediction_metrics",
        lambda *args, **kwargs: pl.DataFrame(
            {"ic_mean": [0.03], "ic_std": [0.01], "ic_n_days": [3.0]}
        ),
    )
    monkeypatch.setattr(
        cv_results,
        "read_predictions",
        lambda case_study, prediction_hash: _predictions("model", 10).drop("config", "epoch"),
    )

    result = cv_results.rebuild_cv_result_from_registry(
        "example",
        [{"family": "deep_learning", "config_name": "model", "n_epochs": 10}],
        label_col="target",
        n_folds=1,
        prediction_split="validation",
        date_col="timestamp",
        entity_col="symbol",
    )

    assert result["all_learning_curves"].height == 1
    assert result["all_predictions"].height == 15


def test_combining_cached_and_fresh_results_keeps_both_configs() -> None:
    cached = cv_results.assemble_cv_result(
        [{"config": "cached", "epoch": 5, "ic_mean": 0.04, "ic_n_days": 3}],
        _predictions("cached", 5),
        date_col="timestamp",
        entity_col="symbol",
    )
    fresh = cv_results.assemble_cv_result(
        [{"config": "fresh", "epoch": 10, "ic_mean": 0.03, "ic_n_days": 3}],
        _predictions("fresh", 10),
        date_col="timestamp",
        entity_col="symbol",
    )

    result = cv_results.combine_cv_results(
        [cached, fresh],
        date_col="timestamp",
        entity_col="symbol",
    )

    assert {row["config_name"] for row in result["grid_results"]} == {"cached", "fresh"}
    assert result["best_config_name"] == "cached"


def test_result_assembly_preserves_backend_metadata() -> None:
    result = cv_results.assemble_cv_result(
        [{"config": "darts", "epoch": 10, "ic_mean": 0.03, "ic_n_days": 3}],
        _predictions("darts", 10),
        date_col="timestamp",
        entity_col="symbol",
        metadata={"darts": {"input_chunk_length": 60}},
    )

    assert result["grid_results"][0]["input_chunk_length"] == 60


def test_result_assembly_reports_and_rejects_invalid_checkpoint_scores() -> None:
    predictions = pl.concat(
        [
            _predictions("valid", 10),
            _predictions("invalid", 10).with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(float("inf"))
                .otherwise(pl.col("y_score"))
                .alias("y_score")
            ),
        ]
    )
    result = cv_results.assemble_cv_result(
        [
            {"config": "valid", "epoch": 10, "ic_mean": 0.03, "ic_n_days": 3},
            {"config": "invalid", "epoch": 10, "ic_mean": 0.90, "ic_n_days": 3},
        ],
        predictions,
        date_col="timestamp",
        entity_col="symbol",
    )

    assert result["best_config_name"] == "valid"
    assert result["grid_results"] == [
        {
            "config_name": "valid",
            "best_epoch": 10,
            "best_ic": pytest.approx(0.03),
            "ic_n_days": pytest.approx(3.0),
            "n_invalid": 0,
            "n_folds": 3,
            "selectable": True,
            "elapsed_s": 0.0,
            "started_at": None,
        }
    ]


def test_result_assembly_normalizes_checkpoint_key_types_before_join() -> None:
    curves = pl.DataFrame([{"config": "lstm", "epoch": 10, "ic_mean": 0.03, "ic_n_days": 3}])
    predictions = _predictions("lstm", 10).with_columns(pl.col("epoch").cast(pl.Int32))

    result = cv_results.assemble_cv_result(
        curves,
        predictions,
        date_col="timestamp",
        entity_col="symbol",
    )

    assert result["best_config_name"] == "lstm"
    assert result["best_epoch"] == 10


def test_result_assembly_excludes_curve_without_prediction_rows() -> None:
    curves = pl.DataFrame(
        [
            {"config": "lstm", "epoch": 10, "ic_mean": 0.03, "ic_n_days": 3},
            {"config": "lstm", "epoch": 25, "ic_mean": 0.90, "ic_n_days": 3},
        ]
    )

    result = cv_results.assemble_cv_result(
        curves,
        _predictions("lstm", 10),
        date_col="timestamp",
        entity_col="symbol",
    )

    assert result["best_epoch"] == 10


def test_result_assembly_rejects_mixed_known_and_unknown_coverage() -> None:
    curves = pl.DataFrame(
        [
            {"config": "cached", "epoch": 5, "ic_mean": 0.04, "ic_n_days": 3},
            {"config": "fresh", "epoch": 10, "ic_mean": 0.03, "ic_n_days": None},
        ]
    )

    with pytest.raises(ValueError, match="coverage is missing"):
        cv_results.assemble_cv_result(
            curves,
            pl.concat([_predictions("cached", 5), _predictions("fresh", 10)]),
            date_col="timestamp",
            entity_col="symbol",
        )
