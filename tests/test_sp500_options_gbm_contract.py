"""Focused fail-closed tests for the S&P 500 options GBM notebook."""

import ast
import copy
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy import stats

from case_studies.utils.registry import (
    build_training_spec,
    canonical_json,
    load_preset,
    training_hash_from_spec,
)

SOURCE = Path("case_studies/sp500_options/07_gbm.py")
CURVE_COLUMNS = [
    "config",
    "iteration",
    "ic_mean",
    "ic_std",
    "ic_n_days",
    "ic_se_hac",
    "ic_ci_lo",
    "ic_ci_hi",
    "ic_t_hac",
    "ic_p_hac",
    "ic_hac_lag",
]


def _load_functions(names: set[str], namespace: dict) -> dict:
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), str(SOURCE), "exec"), namespace)
    return namespace


def _curve_frame() -> pl.DataFrame:
    means = np.asarray([0.01, 0.02, *([0.0] * 8)])
    standard_errors = np.full(10, 0.1)
    t_stats = means / standard_errors
    critical = stats.norm.ppf(0.975)
    return pl.DataFrame(
        {
            "config": ["probe"] * 10,
            "iteration": list(range(50, 501, 50)),
            "ic_mean": means,
            "ic_std": [0.2] * 10,
            "ic_n_days": [484] * 10,
            "ic_se_hac": standard_errors,
            "ic_ci_lo": means - critical * standard_errors,
            "ic_ci_hi": means + critical * standard_errors,
            "ic_t_hac": t_stats,
            "ic_p_hac": 2 * stats.norm.sf(abs(t_stats)),
            "ic_hac_lag": [20] * 10,
        }
    ).select(CURVE_COLUMNS)


def test_normal_hac_reference_exact_values():
    namespace = _load_functions({"_normal_hac_inference"}, {"stats": stats})
    result = namespace["_normal_hac_inference"](
        0.004642116049637068,
        0.009453359475989015,
        0.49105464162531565,
    )
    assert result["ic_p_hac"] == pytest.approx(0.6233878012648169, abs=1e-18)
    assert result["ic_ci_lo"] == pytest.approx(-0.013886128056211842, abs=1e-18)
    assert result["ic_ci_hi"] == pytest.approx(0.023170360155485976, abs=1e-18)


def test_learning_curve_cache_rejects_physical_aliases(tmp_path, monkeypatch):
    isolated = tmp_path / "isolated"
    canonical = tmp_path / "canonical"
    external = tmp_path / "external"
    for directory in (isolated, canonical, external):
        directory.mkdir()
    monkeypatch.setenv("ML4T_PRESERVE_REGISTRY", "1")
    monkeypatch.setenv("ML4T_CANONICAL_CASE_DIR", str(canonical))
    namespace = _load_functions(
        {"_same_inode", "_assert_isolated_path", "_training_cache_path"},
        {"Path": Path, "os": os, "CASE_DIR": isolated},
    )
    external_file = external / "learning_curves.parquet"
    _curve_frame().write_parquet(external_file)
    for alias in ("symlink", "hardlink"):
        target_dir = isolated / "run_log/training" / alias
        target_dir.mkdir(parents=True)
        target = target_dir / "learning_curves.parquet"
        target.symlink_to(external_file) if alias == "symlink" else os.link(external_file, target)
        with pytest.raises(RuntimeError):
            namespace["_training_cache_path"](alias, "learning_curves.parquet")
    training_root = isolated / "run_log/training"
    (training_root / "parent_alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError):
        namespace["_training_cache_path"]("parent_alias", "learning_curves.parquet")
    with pytest.raises(RuntimeError):
        namespace["_training_cache_path"]("../../../external", "learning_curves.parquet")


def test_training_identity_binds_resolved_config_and_preset():
    namespace = _load_functions(
        {
            "_config_identity",
            "_resolved_runtime_config",
            "_assert_preset_matches",
            "_training_spec",
        },
        {
            "build_training_spec": build_training_spec,
            "canonical_json": canonical_json,
            "load_preset": load_preset,
            "label_col": "ret_to_expiry",
            "fold_data": [{"fold": 0}, {"fold": 1}],
            "MAX_BIN": 255,
            "SEED": 42,
            "TRAIN_SAMPLE_FRAC": 1.0,
            "TRAINING_IDENTITY_VERSION": "test-v2",
            "ARTIFACT_HASHES": {"financial": {"sha256": "b" * 64}},
            "splits": [{"fold": 0}, {"fold": 1}],
            "mds": type("MDS", (), {"label_buffer": "35D"})(),
            "DEVICE": "cpu",
            "HAC_LAGS": 20,
            "_evaluation_identity": lambda *_args: {"splits": [{"fold": 0}]},
            "_selection_identity": lambda: {"symbols_sha256": "a" * 64},
            "feature_names": ["feature_a", "feature_b"],
        },
    )
    base = load_preset("gbm", "leaves_15_mae")
    namespace["_assert_preset_matches"](base)
    base_hash = training_hash_from_spec(namespace["_training_spec"](base))
    for parameter, value in (("num_leaves", 127), ("objective", "huber"), ("learning_rate", 0.333)):
        changed = copy.deepcopy(base)
        changed["params"][parameter] = value
        changed_hash = training_hash_from_spec(namespace["_training_spec"](changed))
        assert changed_hash != base_hash
        with pytest.raises(ValueError):
            namespace["_assert_preset_matches"](changed)


def test_curve_validation_rejects_verifier_mutations():
    namespace = _load_functions(
        {
            "_require_array_close",
            "_expected_checkpoints",
            "_validate_curve_frame",
            "_validate_curve_selection",
            "_reconcile_curve_frames",
        },
        {
            "np": np,
            "pl": pl,
            "stats": stats,
            "CURVE_COLUMNS": CURVE_COLUMNS,
            "EXPECTED_CHECKPOINT_COUNT": 10,
            "expected_ic_days": 484,
            "HAC_LAGS": 20,
        },
    )
    cfg = {"config_name": "probe", "checkpoint_interval": 50, "max_iterations": 500}
    base = namespace["_validate_curve_frame"](cfg, _curve_frame())
    summary = base.filter(pl.col("iteration") == 100).row(0, named=True)
    namespace["_validate_curve_selection"](base, 100, summary)
    mutations = [
        base.with_columns(pl.lit("other_config").alias("config")),
        base.with_columns(
            pl.when(pl.col("iteration") == 50)
            .then(0.99)
            .otherwise(pl.col("ic_mean"))
            .alias("ic_mean")
        ),
        base.with_columns(
            pl.when(pl.col("iteration") == 50)
            .then(0.0)
            .otherwise(pl.col("ic_p_hac"))
            .alias("ic_p_hac")
        ),
        base.with_columns(
            pl.when(pl.col("iteration") == 50)
            .then(1)
            .otherwise(pl.col("ic_n_days"))
            .alias("ic_n_days"),
            pl.when(pl.col("iteration") == 50)
            .then(0)
            .otherwise(pl.col("ic_hac_lag"))
            .alias("ic_hac_lag"),
        ),
        base.drop("ic_ci_hi"),
    ]
    for mutation in mutations:
        with pytest.raises(ValueError):
            namespace["_validate_curve_frame"](cfg, mutation)
    physical = base.with_columns(
        pl.when(pl.col("iteration") == 50).then(0.3).otherwise(pl.col("ic_std")).alias("ic_std")
    )
    with pytest.raises(ValueError):
        namespace["_reconcile_curve_frames"](base, physical)


def test_cached_metrics_reconcile_every_emitted_field():
    summary = {
        "ic_mean": 0.004,
        "ic_std": 0.1,
        "ic_n_days": 484,
        "ic_se_hac": 0.01,
        "ic_ci_lo": -0.01559963984540054,
        "ic_ci_hi": 0.02359963984540054,
        "ic_t_hac": 0.4,
        "ic_p_hac": float(2 * stats.norm.sf(0.4)),
        "ic_hac_lag": 20,
    }
    holder = {}
    namespace = _load_functions(
        {"_metric_payload", "_validate_cached_metrics"},
        {
            "np": np,
            "CASE_STUDY_ID": "sp500_options",
            "load_prediction_metrics": lambda *_args, **_kwargs: pl.DataFrame([holder["row"]]),
        },
    )
    payload = namespace["_metric_payload"](summary)
    holder["row"] = {"prediction_hash": "prediction_hash", **payload}
    namespace["_validate_cached_metrics"]("prediction_hash", summary)
    for name in payload:
        forged = dict(holder["row"])
        forged[name] = float(forged[name]) + 0.5
        holder["row"] = forged
        with pytest.raises(ValueError):
            namespace["_validate_cached_metrics"]("prediction_hash", summary)


def test_fresh_checkpoint_grid_rejects_missing_extra_and_duplicate():
    fold_data = [{"fold": 0}, {"fold": 1}]
    namespace = _load_functions(
        {"_expected_checkpoints", "_validate_raw_checkpoint_grid"},
        {"fold_data": fold_data, "EXPECTED_CHECKPOINT_COUNT": 10},
    )
    cfg = {"checkpoint_interval": 50, "max_iterations": 500}
    complete = [
        {"n_trees": checkpoint, "fold": fold}
        for checkpoint in range(50, 501, 50)
        for fold in (0, 1)
    ]
    assert namespace["_validate_raw_checkpoint_grid"](cfg, {"predictions": complete}) == list(
        range(50, 501, 50)
    )
    variants = [
        complete[:-1],
        [*complete, {"n_trees": 550, "fold": 0}],
        [*complete[:-1], {"n_trees": 500, "fold": 0}],
    ]
    for predictions in variants:
        with pytest.raises(ValueError):
            namespace["_validate_raw_checkpoint_grid"](cfg, {"predictions": predictions})
