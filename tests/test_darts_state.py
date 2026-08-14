from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import polars as pl
import pytest
from darts import TimeSeries
from darts.models import NBEATSModel, TSMixerModel

from case_studies.utils.darts_forecasting import (
    BASE_TARGET_COL,
    _attach_base_target,
    _attach_darts_target,
    _attach_expected_periods,
    _build_darts_model,
    _FoldSeries,
    _predict_fold,
    _prepare_fold_series,
    darts_checkpoint_path,
    darts_validation_keys,
    load_darts_checkpoint,
    run_darts_cv,
    validate_darts_checkpoint_population,
    write_darts_checkpoint,
)
from case_studies.utils.deep_learning import _select_sequence_observations, run_dl_cv
from case_studies.utils.registry import evaluate_prediction_coverage
from utils.modeling import load_configs


def _fit_tiny_tsmixer() -> tuple[TSMixerModel, TimeSeries]:
    series = TimeSeries.from_values(np.arange(24, dtype=np.float32))
    model = TSMixerModel(
        input_chunk_length=4,
        output_chunk_length=1,
        hidden_size=4,
        ff_size=4,
        num_blocks=1,
        n_epochs=1,
        batch_size=8,
        random_state=7,
        save_checkpoints=False,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "logger": False,
        },
    )
    model.fit(series, verbose=False)
    return model, series


def test_darts_checkpoint_reconstructs_identical_predictions(tmp_path) -> None:
    model, series = _fit_tiny_tsmixer()
    expected = model.predict(2, series=series).values()
    root = tmp_path / "models"
    path = darts_checkpoint_path(root, "tsmixer", 0, 1)

    write_darts_checkpoint(
        path,
        model=model,
        architecture="tsmixer",
        metadata={
            "config_name": "tsmixer",
            "fold": 0,
            "checkpoint_kind": "epoch",
            "checkpoint_value": 1,
        },
    )
    validate_darts_checkpoint_population(
        root,
        config_name="tsmixer",
        fold_ids=(0,),
        checkpoints=(1,),
        architecture="tsmixer",
    )
    restored, metadata = load_darts_checkpoint(path)
    actual = restored.predict(2, series=series).values()

    np.testing.assert_array_equal(actual, expected)
    assert metadata["checkpoint_value"] == 1


def test_darts_checkpoint_population_rejects_missing_weights(tmp_path) -> None:
    model, _series = _fit_tiny_tsmixer()
    root = tmp_path / "models"
    path = darts_checkpoint_path(root, "tsmixer", 0, 1)
    write_darts_checkpoint(
        path,
        model=model,
        architecture="tsmixer",
        metadata={
            "config_name": "tsmixer",
            "fold": 0,
            "checkpoint_kind": "epoch",
            "checkpoint_value": 1,
        },
    )
    path.with_suffix(".pt.ckpt").unlink()

    with pytest.raises(ValueError, match="population is incomplete"):
        validate_darts_checkpoint_population(
            root,
            config_name="tsmixer",
            fold_ids=(0,),
            checkpoints=(1,),
            architecture="tsmixer",
        )


def test_lagged_label_target_aligns_weekly_horizon_and_resets_after_gap() -> None:
    dataset = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-12",
                    "2024-01-19",
                    "2024-02-02",
                    "2024-02-09",
                    "2024-02-16",
                ]
            ),
            "symbol": ["S0"] * 6,
            "fwd_ret_5d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    dataset = _attach_expected_periods(
        dataset,
        date_col="timestamp",
        calendar_id="NYSE",
        case_study="us_equities_panel",
    )
    config = {
        "config_name": "nbeats_weekly",
        "params": {
            "darts_input_chunk_length": 12,
            "darts_output_chunk_length": 2,
            "darts_target": "lagged_label",
        },
    }

    attached = _attach_darts_target(
        dataset,
        case_study="us_equities_panel",
        date_col="timestamp",
        entity_col="symbol",
        label_col="fwd_ret_5d",
        config=config,
    )

    assert np.isnan(attached.loc[0, BASE_TARGET_COL])
    assert np.isnan(attached.loc[1, BASE_TARGET_COL])
    assert attached.loc[2, BASE_TARGET_COL] == pytest.approx(np.log1p(0.01))
    assert np.isnan(attached.loc[3, BASE_TARGET_COL])
    assert np.isnan(attached.loc[4, BASE_TARGET_COL])
    assert attached.loc[5, BASE_TARGET_COL] == pytest.approx(np.log1p(0.04))


def test_cadence_selected_darts_target_rejects_daily_return_fallback() -> None:
    dataset = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "symbol": ["S0", "S0"],
            "fwd_ret_5d": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="require an explicit cadence-aware target"):
        _attach_darts_target(
            dataset,
            case_study="us_equities_panel",
            date_col="timestamp",
            entity_col="symbol",
            label_col="fwd_ret_5d",
            config={"params": {"decision_cadence": "weekly_friday"}},
        )


@pytest.mark.parametrize(
    ("output_chunk_length", "labels", "match"),
    [
        (1, [0.01, 0.02, 0.03], "two-period forecast"),
        (2, [-1.0, 0.02, 0.03], "returns greater than -1"),
    ],
)
def test_lagged_label_target_rejects_unsafe_contracts(output_chunk_length, labels, match) -> None:
    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-05", periods=3, freq="W-FRI"),
            "symbol": ["S0"] * 3,
            "fwd_ret_5d": labels,
        }
    )
    dataset = _attach_expected_periods(
        dataset,
        date_col="timestamp",
        calendar_id="NYSE",
        case_study="us_equities_panel",
    )

    with pytest.raises(ValueError, match=match):
        _attach_darts_target(
            dataset,
            case_study="us_equities_panel",
            date_col="timestamp",
            entity_col="symbol",
            label_col="fwd_ret_5d",
            config={
                "config_name": "nbeats_weekly",
                "params": {
                    "darts_input_chunk_length": 12,
                    "darts_output_chunk_length": output_chunk_length,
                    "darts_target": "lagged_label",
                },
            },
        )


def test_lagged_label_forecast_scores_the_terminal_horizon() -> None:
    target = TimeSeries.from_values(np.arange(6, dtype=np.float32))
    covariates = TimeSeries.from_values(np.arange(6, dtype=np.float32))
    state = _FoldSeries(
        identity={"symbol": "S0"},
        full_target=target,
        full_covariates=covariates,
        train_target=None,
        train_covariates=None,
        prediction_start_pos=4,
        val_start_pos=3,
        val_end_pos=3,
        dates=pd.date_range("2024-01-05", periods=6, freq="W-FRI").to_numpy(),
        y_true=np.arange(6, dtype=np.float32) / 100,
        n_train_samples=0,
    )

    class _TwoPeriodModel:
        def historical_forecasts(self, _series, *, start, **_kwargs):
            return [
                TimeSeries.from_times_and_values(
                    pd.RangeIndex(start, start + 2),
                    np.log1p(np.array([0.1, 0.2], dtype=np.float32)),
                )
            ]

    predictions = _predict_fold(
        _TwoPeriodModel(),
        [state],
        0,
        "timestamp",
        "symbol",
        output_chunk_length=2,
        forecast_reduction="terminal",
    )

    assert predictions["y_score"].item() == pytest.approx(0.2)
    assert pd.Timestamp(predictions["timestamp"].item()) == pd.Timestamp(state.dates[3])


def test_weekly_nbeats_preset_builds_without_adapter_parameters() -> None:
    config = next(
        config
        for config in load_configs("us_equities_panel", "fwd_ret_5d", "deep_learning")
        if config["config_name"] == "nbeats_weekly"
    )

    model = _build_darts_model(
        config,
        device="cpu",
        fold_seed=7,
        input_chunk_length=12,
        output_chunk_length=2,
    )

    assert isinstance(model, NBEATSModel)


def test_darts_runner_persists_state_with_exact_gap_free_prediction_keys(tmp_path) -> None:
    dates = mcal.get_calendar("NYSE").valid_days("2024-01-02", "2024-02-15")[:20]
    dates = dates.tz_localize(None)
    dataset = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": f"S{symbol}",
                "feature": symbol + day / 10,
                "fwd_ret_1d": np.sin(day / 3) + symbol / 100,
            }
            for symbol in range(6)
            for day, timestamp in enumerate(dates)
        ]
    )
    missing_date = dates[12]
    dataset = dataset.loc[dataset["timestamp"] != missing_date].reset_index(drop=True)
    exact_lookback = pd.DataFrame(
        {
            "timestamp": dates[-7:],
            "symbol": "S6",
            "feature": np.arange(7, dtype=np.float32),
            "fwd_ret_1d": np.arange(7, dtype=np.float32) / 100,
        }
    )
    dataset = pd.concat([dataset, exact_lookback], ignore_index=True)
    config = {
        "family": "deep_learning",
        "library": "darts",
        "config_name": "tsmixer_probe",
        "params": {
            "architecture": "tsmixer",
            "lookback": 4,
            "hidden_dim": 4,
            "n_blocks": 1,
            "dropout": 0.0,
            "darts_target": "lagged_label",
            "darts_output_chunk_length": 2,
        },
        "n_epochs": 1,
        "batch_size": 32,
        "checkpoint_interval": 1,
    }
    split = {
        "fold": 0,
        "train_start": dates[0],
        "train_end": dates[9],
        "val_start": dates[10],
        "val_end": dates[-1],
    }
    model_root = tmp_path / "models"

    result = run_darts_cv(
        dataset,
        [split],
        configs=[config],
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=tmp_path / "run",
        max_train_sequences=0,
        register=False,
        case_study="etfs",
        notebook=None,
        checkpoint_root=model_root,
    )
    expected = darts_validation_keys(
        dataset,
        [split],
        config=config,
        feature_names=["feature"],
        label_col="fwd_ret_1d",
        date_col="timestamp",
        entity_col="symbol",
        case_study="etfs",
    )
    predictions = result["all_predictions"].rename({"fold_id": "fold"})

    assert evaluate_prediction_coverage(expected, predictions).complete
    observed_dates = set(expected["timestamp"].dt.date().to_list())
    assert missing_date.date() not in observed_dates
    assert not {date.date() for date in dates[13:18]} & observed_dates
    assert {date.date() for date in dates[18:]} <= observed_dates
    assert (
        expected.filter((pl.col("symbol") == "S6") & (pl.col("timestamp") == dates[-1])).height == 1
    )
    assert (
        len(
            validate_darts_checkpoint_population(
                model_root,
                config_name="tsmixer_probe",
                fold_ids=(0,),
                checkpoints=(1,),
                architecture="tsmixer",
            )
        )
        == 1
    )


def test_weekly_darts_runner_matches_resolved_eligible_keys(tmp_path) -> None:
    dates = mcal.get_calendar("NYSE").valid_days("2023-01-03", "2023-06-30")
    dates = dates.tz_localize(None)
    dataset = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": f"S{symbol}",
                "feature": symbol + day / 100,
                "fwd_ret_5d": np.sin(day / 5) / 10 + symbol / 1000,
            }
            for symbol in range(6)
            for day, timestamp in enumerate(dates)
        ]
    )
    config = {
        "family": "deep_learning",
        "library": "darts",
        "config_name": "nbeats_weekly",
        "params": {
            "architecture": "nbeats",
            "decision_cadence": "weekly_friday",
            "darts_output_chunk_length": 2,
            "darts_target": "lagged_label",
            "dropout": 0.0,
            "hidden_size": 4,
            "lookback": 3,
            "n_blocks": 1,
            "n_layers": 1,
        },
        "n_epochs": 1,
        "batch_size": 32,
        "checkpoint_interval": 1,
    }
    split = {
        "fold": 0,
        "train_start": dates[0],
        "train_end": dates[59],
        "val_start": dates[65],
        "val_end": dates[-1],
    }

    result = run_dl_cv(
        dataset,
        [split],
        configs=[config],
        n_features=1,
        feature_names=["feature"],
        label_col="fwd_ret_5d",
        date_col="timestamp",
        entity_col="symbol",
        device="cpu",
        save_dir=tmp_path / "run",
        case_study="us_equities_panel",
        checkpoint_root=tmp_path / "models",
    )
    weekly = _select_sequence_observations(
        dataset,
        date_col="timestamp",
        cadence="weekly_friday",
        calendar="NYSE",
    )
    assert isinstance(weekly, pd.DataFrame)
    expected = darts_validation_keys(
        weekly,
        [split],
        config=config,
        feature_names=["feature"],
        label_col="fwd_ret_5d",
        date_col="timestamp",
        entity_col="symbol",
        case_study="us_equities_panel",
    )
    predictions = result["all_predictions"].rename({"fold_id": "fold"})

    assert evaluate_prediction_coverage(expected, predictions).complete
    assert (
        predictions.select("symbol", "timestamp", "fold")
        .unique()
        .sort("symbol", "timestamp", "fold")
        .equals(expected)
    )


def test_darts_segments_and_predicts_each_cme_contract_position() -> None:
    dates = pd.date_range("2024-01-02", periods=8)
    dataset = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "product": product,
                "position": position,
                "feature": float(day + position),
                "fwd_ret_1d": float(day) / 100,
                BASE_TARGET_COL: np.log1p(float(day) / 100),
            }
            for product in ("ES", "ZC")
            for position in range(3)
            for day, timestamp in enumerate(dates if product == "ES" else dates.delete(2))
        ]
    )
    dataset = _attach_expected_periods(
        dataset, date_col="timestamp", calendar_id="CME_Equity", case_study="cme_futures"
    )
    split = {
        "fold": 0,
        "train_start": dates[0],
        "train_end": dates[3],
        "val_start": dates[4],
        "val_end": dates[-1],
    }

    states = _prepare_fold_series(
        dataset,
        split,
        ["feature"],
        "fwd_ret_1d",
        "timestamp",
        "product",
        4,
        1,
    )

    assert len(states) == 6
    assert {tuple(state.identity.items()) for state in states} == {
        (("product", product), ("position", position))
        for product in ("ES", "ZC")
        for position in range(3)
    }

    class _OneStepModel:
        def historical_forecasts(self, series, *, start, **_kwargs):
            return [
                TimeSeries.from_times_and_values(
                    pd.RangeIndex(start, start + 1), np.array([0.01], dtype=np.float32)
                )
            ]

    predictions = _predict_fold(
        _OneStepModel(), states, 0, "timestamp", "product", output_chunk_length=1
    )
    assert {"product", "position"} <= set(predictions.columns)
    assert predictions.select("product", "position").n_unique() == 6
    assert predictions.filter(pl.col("product") == "ZC")["timestamp"].unique().to_list() == [
        dates[4]
    ]


def test_cme_base_target_uses_only_finalized_panel_sessions(monkeypatch) -> None:
    raw = pl.DataFrame(
        {
            "session_date": [
                pd.Timestamp("2024-01-02").date(),
                pd.Timestamp("2024-01-03").date(),
                pd.Timestamp("2024-01-04").date(),
            ],
            "product": ["ES"] * 3,
            "tenor": [0] * 3,
            "adj_close": [100.0, 110.0, 121.0],
        }
    )
    monkeypatch.setattr("data.load_cme_futures", lambda: raw)
    finalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "product": ["ES", "ES"],
            "position": [0, 0],
        }
    )

    attached = _attach_base_target(finalized, "cme_futures", "timestamp")

    assert attached[BASE_TARGET_COL].isna().sum() == 1
    assert attached.loc[1, BASE_TARGET_COL] == pytest.approx(np.log(121.0 / 100.0))
