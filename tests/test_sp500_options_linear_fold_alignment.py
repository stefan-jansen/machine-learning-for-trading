"""Regression tests for fold-aware S&P 500 options linear modeling."""

from __future__ import annotations

import ast
import hashlib
import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from ml4t.diagnostic.metrics import compute_ic_uncertainty, cross_sectional_ic_series
from scipy import stats

from case_studies.utils.registry import prediction_hash_from_parts, training_hash_from_spec

NOTEBOOK = Path("case_studies/sp500_options/06_linear.py")
TEMPORAL_FEATURES = ["garch_cond_vol", "sv_vol"]


def _load_notebook_guards() -> dict:
    """Load pure guard functions without executing the notebook."""
    tree = ast.parse(NOTEBOOK.read_text())
    wanted = {
        "_align_temporal_fold",
        "_artifact_identity",
        "_assert_isolated_path",
        "_bind_training_identity",
        "_compute_complete_daily_ic",
        "_compute_ic_uncertainty_metrics",
        "_evaluation_identity",
        "_expected_prediction_contract",
        "_read_cached_result",
        "_require_isolated_registry",
        "_same_inode",
        "_seal_label_endpoints",
        "_validate_actual_values",
        "_validate_cached_metrics",
        "_validate_cv_splits",
        "_validate_prediction_keys",
        "_validate_temporal_keys",
    }
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace = {
        "CASE_STUDY_ID": "sp500_options",
        "FORCE_RETRAIN": False,
        "IC_DEPENDENCE_SESSIONS": 21,
        "IC_HAC_LAG": 20,
        "IC_UNCERTAINTY_ALIASES": {
            "mean_ic": "ic_mean_daily",
            "std_ic": "ic_std_daily",
            "n_days": "ic_n_days",
            "pct_positive": "ic_pct_positive",
            "se_naive": "ic_se_naive",
            "ci_naive_lower": "ic_naive_lo",
            "ci_naive_upper": "ic_naive_hi",
            "se_hac": "ic_se_hac",
            "ci_hac_lower": "ic_ci_lo",
            "ci_hac_upper": "ic_ci_hi",
            "t_hac": "ic_t_hac",
            "p_hac": "ic_p_hac",
            "hac_lag": "ic_hac_lag",
            "ci_boot_lower": "ic_boot_lo",
            "ci_boot_upper": "ic_boot_hi",
            "boot_block_size": "ic_boot_block",
        },
        "PREDICTION_SPLIT": "validation",
        "Path": Path,
        "cross_sectional_ic_series": cross_sectional_ic_series,
        "compute_ic_uncertainty": compute_ic_uncertainty,
        "hashlib": hashlib,
        "np": np,
        "norm": stats.norm,
        "os": os,
        "pd": pd,
        "pl": pl,
        "prediction_hash_from_parts": prediction_hash_from_parts,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), NOTEBOOK, "exec"), namespace)
    return namespace


def _split(fold: int = 0) -> dict:
    return {
        "fold": fold,
        "train_start": date(2018, 1, 5),
        "train_end": date(2019, 11, 12),
        "val_start": date(2020, 1, 6),
        "val_end": date(2020, 12, 31),
    }


def _base_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [date(2018, 6, 1), date(2020, 6, 1)],
            "symbol": ["AAA", "AAA"],
            "financial_feature": [1.0, 2.0],
            "ret_to_expiry": [0.1, 0.2],
        }
    )


def _temporal_panel(folds: list[int] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [date(2018, 6, 1), date(2020, 6, 1)],
            "symbol": ["AAA", "AAA"],
            "fold": folds or [0, 0],
            "garch_cond_vol": [10.0, 20.0],
            "sv_vol": [11.0, 21.0],
        }
    )


def test_fold_join_covers_training_and_validation_on_three_keys() -> None:
    guards = _load_notebook_guards()
    temporal = _temporal_panel()
    guards["_validate_temporal_keys"](temporal, "timestamp", "symbol", TEMPORAL_FEATURES)
    aligned = guards["_align_temporal_fold"](
        _base_panel(), temporal, _split(), "timestamp", "symbol", TEMPORAL_FEATURES
    ).sort("timestamp")

    assert aligned["fold"].to_list() == [0, 0]
    assert aligned["garch_cond_vol"].to_list() == [10.0, 20.0]
    assert aligned["sv_vol"].to_list() == [11.0, 21.0]


def test_old_fold_inversion_fails_instead_of_imputing() -> None:
    guards = _load_notebook_guards()
    inverted = _temporal_panel(folds=[1, 1])

    with pytest.raises(ValueError, match="missing 2 temporal row matches"):
        guards["_align_temporal_fold"](
            _base_panel(), inverted, _split(), "timestamp", "symbol", TEMPORAL_FEATURES
        )


def test_temporal_join_fails_on_duplicate_null_and_nonfinite_state() -> None:
    guards = _load_notebook_guards()
    duplicated = pl.concat([_temporal_panel(), _temporal_panel().head(1)])
    with pytest.raises(ValueError, match="duplicate fold-specific keys"):
        guards["_validate_temporal_keys"](duplicated, "timestamp", "symbol", TEMPORAL_FEATURES)

    invalid = _temporal_panel().with_columns(
        pl.when(pl.col("timestamp") == date(2020, 6, 1))
        .then(float("inf"))
        .otherwise(pl.col("sv_vol"))
        .alias("sv_vol")
    )
    with pytest.raises(ValueError, match="null or nonfinite temporal rows"):
        guards["_align_temporal_fold"](
            _base_panel(), invalid, _split(), "timestamp", "symbol", TEMPORAL_FEATURES
        )


def test_cv_contract_rejects_holdout_and_noncanonical_fold_ids() -> None:
    validate = _load_notebook_guards()["_validate_cv_splits"]
    validate([_split()], "2021-01-01")

    with pytest.raises(ValueError, match="0..N-1"):
        validate([_split(fold=1)], "2021-01-01")
    holdout_split = _split() | {"val_end": date(2021, 1, 4)}
    with pytest.raises(ValueError, match="sealed holdout"):
        validate([holdout_split], "2021-01-01")


def test_label_endpoint_purge_seals_actual_expiry() -> None:
    seal = _load_notebook_guards()["_seal_label_endpoints"]
    labels = pl.DataFrame(
        {
            "timestamp": [date(2020, 11, 20), date(2020, 12, 10)],
            "symbol": ["AAA", "AAA"],
            "dte_calendar": [35, 35],
            "ret_to_expiry": [0.1, 0.2],
        }
    )

    selected = seal(labels, "timestamp", "2021-01-01")
    assert selected["timestamp"].to_list() == [date(2020, 11, 20)]


def _prediction_fixture() -> tuple[pl.DataFrame, list[dict]]:
    timestamps = [date(2020, 1, 2)] * 5 + [date(2020, 1, 3)] * 5
    symbols = list("ABCDE") * 2
    prediction = [1.0, 2.0, 3.0, 4.0, 5.0] * 2
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbols,
            "fold": [0] * 10,
            "prediction": prediction,
            "actual": prediction,
        }
    )
    prepared = [
        {
            "fold": 0,
            "meta": frame.select("timestamp", "symbol").to_pandas(),
            "y_val": np.array(prediction),
        }
    ]
    return frame, prepared


def test_prediction_contract_requires_exact_keys_and_daily_dates() -> None:
    guards = _load_notebook_guards()
    predictions, prepared = _prediction_fixture()
    expected, dates = guards["_expected_prediction_contract"](
        prepared, ["timestamp", "symbol"], "timestamp", "symbol"
    )
    predictions = guards["_validate_prediction_keys"](
        predictions, expected, ["timestamp", "symbol"]
    )
    summary = guards["_compute_complete_daily_ic"](predictions, dates, "timestamp", "symbol")
    assert summary == {"ic_mean": 1.0, "ic_std": 0.0, "ic_n_days": 2}

    with pytest.raises(ValueError, match="missing=1, extra=0"):
        guards["_validate_prediction_keys"](predictions.head(9), expected, ["timestamp", "symbol"])
    wrong_actual = predictions.with_columns((pl.col("actual") + 0.1).alias("actual"))
    with pytest.raises(ValueError, match="differ from endpoint-sealed labels"):
        guards["_validate_prediction_keys"](wrong_actual, expected, ["timestamp", "symbol"])
    constant_date = predictions.with_columns(
        pl.when(pl.col("timestamp") == date(2020, 1, 3))
        .then(1.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    with pytest.raises(ValueError, match="Daily IC coverage differs: missing=1"):
        guards["_compute_complete_daily_ic"](constant_date, dates, "timestamp", "symbol")


def _manual_bartlett_hac_se(values: np.ndarray, lag: int) -> float:
    """Compute the constant-only Newey-West SE without an ML4T helper."""
    residual = values - values.mean()
    long_run_sum = float(residual @ residual)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1)
        long_run_sum += 2.0 * weight * float(residual[offset:] @ residual[:-offset])
    n_obs = len(values)
    corrected_variance = long_run_sum / n_obs**2 * n_obs / (n_obs - 1)
    return float(np.sqrt(corrected_variance))


def test_expiry_purge_and_hac_dependence_use_separate_horizons() -> None:
    source = NOTEBOOK.read_text()
    assert "IC_DEPENDENCE_SESSIONS = 21" in source
    assert "IC_HAC_LAG = IC_DEPENDENCE_SESSIONS - 1" in source
    evaluation = _load_notebook_guards()["_evaluation_identity"](
        [_split()], "2021-01-01", "2021-12-31", "35D"
    )
    assert evaluation["label_buffer"] == "35D"


def test_daily_ic_uncertainty_matches_independent_20_lag_oracle() -> None:
    timestamps = [date(2020, 1, 1) + pd.Timedelta(days=offset) for offset in range(60)]
    prediction, actual, dates, symbols = [], [], [], []
    base = np.arange(5, dtype=float)
    for offset, timestamp in enumerate(timestamps):
        dates.extend([timestamp] * 5)
        symbols.extend(list("ABCDE"))
        prediction.extend(base)
        actual.extend(np.roll(base, offset % 5))
    frame = pl.DataFrame(
        {"timestamp": dates, "symbol": symbols, "prediction": prediction, "actual": actual}
    )
    daily_values = []
    for daily in frame.partition_by("timestamp", maintain_order=True):
        pred_rank = stats.rankdata(daily["prediction"].to_numpy())
        actual_rank = stats.rankdata(daily["actual"].to_numpy())
        daily_values.append(float(np.corrcoef(pred_rank, actual_rank)[0, 1]))
    values = np.asarray(daily_values)
    manual_se = _manual_bartlett_hac_se(values, lag=20)
    manual_mean = float(values.mean())
    critical = float(stats.norm.ppf(0.975))

    metrics = _load_notebook_guards()["_compute_ic_uncertainty_metrics"](
        frame, set(timestamps), "timestamp", "symbol"
    )
    assert metrics["ic_hac_lag"] == 20
    assert metrics["ic_mean_daily"] == pytest.approx(manual_mean, abs=1e-12)
    assert metrics["ic_se_hac"] == pytest.approx(manual_se, abs=1e-12)
    assert metrics["ic_t_hac"] == pytest.approx(manual_mean / manual_se, abs=1e-12)
    assert metrics["ic_ci_lo"] == pytest.approx(manual_mean - critical * manual_se, abs=1e-12)
    assert metrics["ic_ci_hi"] == pytest.approx(manual_mean + critical * manual_se, abs=1e-12)
    assert metrics["ic_p_hac"] == pytest.approx(
        2 * stats.norm.sf(abs(manual_mean / manual_se)), abs=1e-12
    )


def test_cache_hit_requires_physical_predictions_and_exact_daily_metrics(tmp_path: Path) -> None:
    guards = _load_notebook_guards()
    predictions, prepared = _prediction_fixture()
    expected, dates = guards["_expected_prediction_contract"](
        prepared, ["timestamp", "symbol"], "timestamp", "symbol"
    )
    path = tmp_path / "predictions.parquet"
    predictions.write_parquet(path)
    prediction_hash = prediction_hash_from_parts("train-hash", None, "validation")
    uncertainty = {
        "ic_mean_daily": 1.0,
        "ic_std_daily": 0.0,
        "ic_n_days": 2.0,
        "ic_se_hac": 0.1,
        "ic_ci_lo": 0.8,
        "ic_ci_hi": 1.2,
        "ic_t_hac": 10.0,
        "ic_p_hac": 0.01,
        "ic_hac_lag": 20.0,
    }
    guards.update(
        {
            "date_col": "timestamp",
            "entity_col": "symbol",
            "expected_ic_dates": dates,
            "expected_ic_days": 2,
            "expected_prediction_keys": expected,
            "mds": SimpleNamespace(join_cols=["timestamp", "symbol"]),
            "training_hash_from_spec": lambda spec: "train-hash",
            "_compute_ic_uncertainty_metrics": lambda *args: uncertainty,
            "_prediction_cache_path": lambda prediction_hash: path,
            "load_prediction_sets": lambda *args, **kwargs: pl.DataFrame(
                {
                    "prediction_hash": [prediction_hash],
                    "training_hash": ["train-hash"],
                    "split": ["validation"],
                }
            ),
            "load_prediction_metrics": lambda *args, **kwargs: pl.DataFrame(
                {"prediction_hash": [prediction_hash]}
                | {name: [value] for name, value in uncertainty.items()}
            ),
        }
    )
    cfg = {"config_name": "ridge_test"}
    status = SimpleNamespace(complete=True)
    result = guards["_read_cached_result"](cfg, {"spec": "test"}, status)
    assert result["ic_n_days"] == 2
    assert result["cached"] is True

    guards["load_prediction_sets"] = lambda *args, **kwargs: pl.DataFrame(
        {
            "prediction_hash": ["forged-hash"],
            "training_hash": ["train-hash"],
            "split": ["validation"],
        }
    )
    with pytest.raises(ValueError, match="prediction hash is forged"):
        guards["_read_cached_result"](cfg, {"spec": "test"}, status)

    guards["load_prediction_sets"] = lambda *args, **kwargs: pl.DataFrame(
        {
            "prediction_hash": [prediction_hash],
            "training_hash": ["train-hash"],
            "checkpoint_value": [10],
            "checkpoint_kind": ["iteration"],
            "split": ["validation"],
        }
    )
    with pytest.raises(ValueError, match="not the final checkpoint"):
        guards["_read_cached_result"](cfg, {"spec": "test"}, status)

    guards["load_prediction_sets"] = lambda *args, **kwargs: pl.DataFrame(
        {
            "prediction_hash": [prediction_hash],
            "training_hash": ["train-hash"],
            "split": ["validation"],
        }
    )
    guards["load_prediction_metrics"] = lambda *args, **kwargs: pl.DataFrame(
        {"prediction_hash": [prediction_hash]}
        | {name: [1.0 if name == "ic_n_days" else value] for name, value in uncertainty.items()}
    )
    with pytest.raises(ValueError, match="count differs"):
        guards["_read_cached_result"](cfg, {"spec": "test"}, status)

    guards["load_prediction_metrics"] = lambda *args, **kwargs: pl.DataFrame(
        {"prediction_hash": [prediction_hash]}
        | {name: [34.0 if name == "ic_hac_lag" else value] for name, value in uncertainty.items()}
    )
    with pytest.raises(ValueError, match="21-session dependence contract"):
        guards["_read_cached_result"](cfg, {"spec": "test"}, status)

    guards["load_prediction_sets"] = lambda *args, **kwargs: pl.DataFrame()
    assert guards["_read_cached_result"](cfg, {"spec": "test"}, status) is None

    guards["load_prediction_sets"] = lambda *args, **kwargs: pl.DataFrame(
        {
            "prediction_hash": [prediction_hash],
            "training_hash": ["train-hash"],
            "split": ["validation"],
        }
    )
    guards["_prediction_cache_path"] = lambda prediction_hash: tmp_path / "missing.parquet"
    assert guards["_read_cached_result"](cfg, {"spec": "test"}, status) is None


def test_preservation_mode_requires_real_isolated_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require = _load_notebook_guards()["_require_isolated_registry"]
    monkeypatch.setenv("ML4T_PRESERVE_REGISTRY", "1")
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
    with pytest.raises(RuntimeError, match="require ML4T_OUTPUT_DIR"):
        require(tmp_path / "sp500_options")

    canonical = tmp_path / "canonical" / "sp500_options"
    (canonical / "run_log").mkdir(parents=True)
    monkeypatch.setenv("ML4T_CANONICAL_CASE_DIR", str(canonical))
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path / "isolated"))
    case_dir = tmp_path / "isolated" / "sp500_options"
    case_dir.mkdir(parents=True)
    require(case_dir)
    assert (case_dir / "run_log/training").is_dir()
    assert (case_dir / "run_log/predictions").is_dir()


def test_preservation_mode_rejects_registry_file_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require = _load_notebook_guards()["_require_isolated_registry"]
    canonical = tmp_path / "canonical" / "sp500_options"
    canonical_registry = canonical / "run_log/registry.db"
    canonical_registry.parent.mkdir(parents=True)
    canonical_registry.touch()
    output_root = tmp_path / "isolated"
    run_log = output_root / "sp500_options/run_log"
    run_log.mkdir(parents=True)
    (run_log / "registry.db").symlink_to(canonical_registry)
    monkeypatch.setenv("ML4T_PRESERVE_REGISTRY", "1")
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("ML4T_CANONICAL_CASE_DIR", str(canonical))

    with pytest.raises(RuntimeError, match="symlink"):
        require(output_root / "sp500_options")


def test_preservation_mode_rejects_output_root_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require = _load_notebook_guards()["_require_isolated_registry"]
    real_root = tmp_path / "real-output"
    (real_root / "sp500_options").mkdir(parents=True)
    alias_root = tmp_path / "output-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    canonical = tmp_path / "canonical" / "sp500_options"
    canonical.mkdir(parents=True)
    monkeypatch.setenv("ML4T_PRESERVE_REGISTRY", "1")
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(alias_root))
    monkeypatch.setenv("ML4T_CANONICAL_CASE_DIR", str(canonical))

    with pytest.raises(RuntimeError, match="resolves outside ML4T_OUTPUT_DIR"):
        require(alias_root / "sp500_options")


def test_preservation_mode_rejects_registry_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require = _load_notebook_guards()["_require_isolated_registry"]
    canonical = tmp_path / "canonical" / "sp500_options"
    canonical_registry = canonical / "run_log/registry.db"
    canonical_registry.parent.mkdir(parents=True)
    canonical_registry.write_bytes(b"registry")
    output_root = tmp_path / "isolated"
    run_log = output_root / "sp500_options/run_log"
    run_log.mkdir(parents=True)
    os.link(canonical_registry, run_log / "registry.db")
    monkeypatch.setenv("ML4T_PRESERVE_REGISTRY", "1")
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("ML4T_CANONICAL_CASE_DIR", str(canonical))

    with pytest.raises(RuntimeError, match="multiple hard links"):
        require(output_root / "sp500_options")


def test_preservation_mode_rejects_cross_name_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require = _load_notebook_guards()["_require_isolated_registry"]
    canonical = tmp_path / "canonical" / "sp500_options"
    source = canonical / "run_log/training/source/coefficients.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"coefficients")
    output_root = tmp_path / "isolated"
    target = output_root / "sp500_options/run_log/training/different/coefficients.parquet"
    target.parent.mkdir(parents=True)
    os.link(source, target)
    monkeypatch.setenv("ML4T_PRESERVE_REGISTRY", "1")
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("ML4T_CANONICAL_CASE_DIR", str(canonical))

    with pytest.raises(RuntimeError, match="multiple hard links"):
        require(output_root / "sp500_options")


def test_training_hash_binds_artifact_bytes_version_and_split_bounds(tmp_path: Path) -> None:
    guards = _load_notebook_guards()
    paths = {
        "financial": tmp_path / "financial.parquet",
        "model_based": tmp_path / "model_based.parquet",
        "label": tmp_path / "ret_to_expiry.parquet",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode())
    base = {"family": "linear", "label": "ret_to_expiry", "seed": 42}
    evaluation = guards["_evaluation_identity"]([_split()], "2021-01-01", "2021-12-31", "35D")
    artifacts = guards["_artifact_identity"](paths)
    bind = guards["_bind_training_identity"]
    baseline = bind(base, artifacts, evaluation, "sp500-options-linear-v3")

    paths["label"].write_bytes(b"ret_to_expirx")
    byte_mutation = bind(
        base, guards["_artifact_identity"](paths), evaluation, "sp500-options-linear-v3"
    )
    version_mutation = bind(base, artifacts, evaluation, "sp500-options-linear-v4")
    shifted = _split() | {"val_end": date(2020, 12, 30)}
    shifted_evaluation = guards["_evaluation_identity"](
        [shifted], "2021-01-01", "2021-12-31", "35D"
    )
    split_mutation = bind(base, artifacts, shifted_evaluation, "sp500-options-linear-v3")

    baseline_hash = training_hash_from_spec(baseline)
    assert training_hash_from_spec(byte_mutation) != baseline_hash
    assert training_hash_from_spec(version_mutation) != baseline_hash
    assert training_hash_from_spec(split_mutation) != baseline_hash


def test_notebook_uses_canonical_folds_daily_ic_and_exact_join() -> None:
    source = NOTEBOOK.read_text()
    assert "canonical_splits = generate_cv_splits(" in source
    assert 'keys = [date_col, entity_col, "fold"]' in source
    assert 'on=keys, how="left", validate="1:1"' in source
    assert "np.nanmean(fold_ics)" not in source
    assert "sort_values(" in source and "[date_col, entity_col]" in source
    assert 'r["ic_n_days"] == expected_ic_days' in source
    assert "max_ic_days" not in source
    assert "best = complete[0]" in source
    register = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_register_result"
    )
    register_source = ast.get_source_segment(source, register)
    assert register_source is not None
    assert register_source.count("_require_isolated_registry(CASE_DIR)") == 3


def test_notebook_has_no_frozen_result_claims_or_oversized_code_cells() -> None:
    source = NOTEBOOK.read_text()
    for stale_claim in (
        "linear signal here is faint",
        "best config barely clears zero",
        "declines monotonically",
        "aggressive L1 selection is worst",
    ):
        assert stale_claim not in source

    lines = source.splitlines()
    markers = [index for index, line in enumerate(lines) if line.startswith("# %%")]
    markers.append(len(lines))
    oversized = []
    for start, end in zip(markers, markers[1:], strict=False):
        if "[markdown]" not in lines[start] and end - start - 1 > 40:
            oversized.append((start + 1, end - start - 1))
    assert oversized == []
