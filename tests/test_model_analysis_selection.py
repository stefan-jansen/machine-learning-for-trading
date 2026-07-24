import numpy as np
import polars as pl

from case_studies.utils.model_analysis import (
    best_model_per_family_fast,
    load_fold_metrics_from_registry,
    load_metrics_from_registry,
    load_predictions,
    prediction_correlation_matrix,
)
from case_studies.utils.model_viz import _sorted_fold_columns
from case_studies.utils.registry import (
    register_fold_metrics,
    register_prediction_set,
    register_training_run,
)


def test_best_model_per_family_uses_daily_cross_sectional_ic() -> None:
    metrics = pl.DataFrame(
        {
            "family": ["linear", "linear", "gbm"],
            "config_name": ["pooled_winner", "daily_winner", "gbm_daily"],
            "ic_mean": [0.30, 0.10, 0.20],
            "ic_mean_daily": [0.05, 0.15, 0.12],
        }
    )

    selected = best_model_per_family_fast(metrics)

    assert selected.filter(pl.col("family") == "linear")["config_name"].item() == "daily_winner"


def test_registry_loader_exposes_daily_ic_as_selection_metric(tmp_path, monkeypatch) -> None:
    for config_name, fold_ic, daily_ic in (
        ("fold_winner", 0.30, 0.05),
        ("daily_winner", 0.10, 0.15),
    ):
        training_hash = register_training_run(
            "test",
            {
                "family": "latent_factors",
                "label": "fwd_ret_5d",
                "config_name": config_name,
                "params": {},
                "seed": 42,
            },
            case_dir=tmp_path,
        )
        register_prediction_set(
            "test",
            training_hash,
            checkpoint_value=1,
            split="validation",
            metrics={"ic_mean": fold_ic, "ic_mean_daily": daily_ic, "ic_std": 0.01},
            case_dir=tmp_path,
        )

    monkeypatch.setattr("case_studies.utils.model_analysis.get_case_study_dir", lambda _: tmp_path)
    metrics = load_metrics_from_registry("test", families=["latent_factors"])

    assert metrics["config_name"].to_list() == ["daily_winner", "fold_winner"]
    assert metrics["ic_mean"].to_list() == [0.15, 0.05]


def test_registry_loaders_preserve_exact_prediction_identity(tmp_path, monkeypatch) -> None:
    prediction_hashes = []
    for variant, daily_ic in (("old_input", 0.20), ("current_input", 0.10)):
        training_hash = register_training_run(
            "test",
            {
                "family": "deep_learning",
                "label": "fwd_ret_5d",
                "config_name": "same_display_config",
                "params": {"input_lineage": variant},
                "seed": 42,
            },
            case_dir=tmp_path,
        )
        prediction_hashes.append(
            register_prediction_set(
                "test",
                training_hash,
                checkpoint_value=5,
                checkpoint_kind="epoch",
                split="validation",
                predictions=pl.DataFrame(
                    {
                        "timestamp": ["2020-01-01"] * 5 + ["2020-01-02"] * 5,
                        "symbol": ["A", "B", "C", "D", "E"] * 2,
                        "y_true": list(range(5)) * 2,
                        "y_score": [daily_ic + rank for rank in range(5)] * 2,
                    }
                ),
                metrics={"ic_mean": daily_ic, "ic_mean_daily": daily_ic, "ic_std": 0.01},
                case_dir=tmp_path,
            )
        )
        register_fold_metrics(
            "test",
            prediction_hashes[-1],
            {0: {"ic": daily_ic}, 1: {"ic": daily_ic}},
            case_dir=tmp_path,
        )

    monkeypatch.setattr("case_studies.utils.model_analysis.get_case_study_dir", lambda _: tmp_path)
    metrics = load_metrics_from_registry("test", families=["deep_learning"])
    folds = load_fold_metrics_from_registry("test", families=["deep_learning"])
    exact = load_predictions("test", prediction_hash=prediction_hashes[1])

    assert metrics.height == 2
    assert set(metrics["prediction_hash"].to_list()) == set(prediction_hashes)
    assert metrics["ic_mean"].to_list() == [0.20, 0.10]
    assert set(folds["prediction_hash"].to_list()) == set(prediction_hashes)
    assert exact["prediction_hash"].unique().to_list() == [prediction_hashes[1]]


def test_prediction_correlation_averages_daily_cross_sectional_spearman() -> None:
    rows = []
    scores = {
        "2020-01-02": ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        "2020-01-03": ([100.0, 200.0, 300.0], [3.0, 2.0, 1.0]),
    }
    for timestamp, (left, right) in scores.items():
        for family, values in (("left", left), ("right", right)):
            rows.extend(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "family": family,
                    "config_name": "model",
                    "checkpoint_value": 1,
                    "y_score": value,
                }
                for symbol, value in zip(["A", "B", "C"], values, strict=True)
            )
    predictions = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_date())
    models = [
        {"family": "left", "config_name": "model", "checkpoint": 1},
        {"family": "right", "config_name": "model", "checkpoint": 1},
    ]

    matrix, labels = prediction_correlation_matrix(predictions, models=models)

    assert labels == ["left/model", "right/model"]
    assert np.isclose(matrix[0, 1], 0.0, atol=1e-12)


def test_fold_columns_are_sorted_numerically() -> None:
    assert _sorted_fold_columns(["10", "2", "1", "0"]) == ["0", "1", "2", "10"]
