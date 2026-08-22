"""Run the reduced real-data proof for US-equities fold-major execution."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from case_studies.research import Study, run_models
from utils.paths import REPO_ROOT

CASE_STUDY = "us_equities_panel"
LABEL = "fwd_ret_1d"


def _seed_worktree_workspace(workspace: Path) -> None:
    target = workspace.resolve() / CASE_STUDY
    if target.exists():
        return
    release_case = REPO_ROOT / "case_studies" / CASE_STUDY
    if not (release_case / "run_log").is_symlink():
        return
    workspace.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "run_log").mkdir()
    shutil.copytree(release_case / "config", target / "config")
    if not (workspace / "config").exists():
        shutil.copytree(REPO_ROOT / "case_studies" / "config", workspace / "config")
    for name in ("features", "labels"):
        source = (release_case / name).resolve(strict=True)
        (target / name).symlink_to(source, target_is_directory=True)
    (target / ".study.json").write_text(
        json.dumps({"schema_version": 1, "case_study": CASE_STUDY}, sort_keys=True) + "\n"
    )


def _requests(
    study: Study,
    *,
    family: str,
    folds: list[int],
    max_symbols: int,
    train_sample_frac: float,
    device: str,
):
    common_reductions = {"folds": folds, "max_symbols": max_symbols}
    if family == "tabular_dl":
        common_reductions.update({"checkpoint_interval": 1, "n_epochs": 1})
        variants = ((64, max_symbols), (80, max_symbols), (96, max(1, max_symbols // 2)))
        return [
            study.model(
                family=family,
                label=LABEL,
                config_name="tabm_s",
                overrides={"device": device, "hidden_dim": hidden_dim, "num_threads": 8},
                execution_tier="preview",
                preview_reductions={**common_reductions, "max_symbols": symbols},
            )
            for hidden_dim, symbols in variants
        ]
    common_reductions["train_sample_frac"] = train_sample_frac
    if family == "gbm":
        common_reductions.update({"checkpoint_interval": 2, "max_iterations": 4})
    incompatible_reductions = {
        **common_reductions,
        "train_sample_frac": train_sample_frac / 2,
    }
    config_names = (
        ("ridge_a1.0", "ridge_a10.0", "ridge_a100.0")
        if family == "linear"
        else ("default_mse", "default_huber", "leaves_15_mse")
    )
    return [
        study.model(
            family=family,
            label=LABEL,
            config_name=config_name,
            execution_tier="preview",
            preview_reductions=(
                common_reductions if config_name != config_names[-1] else incompatible_reductions
            ),
        )
        for config_name in config_names
    ]


def _run_key(run, family: str) -> str:
    spec = run.training.spec()
    if family != "tabular_dl":
        return str(spec["config_name"])
    hidden_dim = spec["computation"]["model"]["params"]["hidden_dim"]
    return f"{spec['config_name']}-h{hidden_dim}"


def _assert_results(execution, family: str, folds: list[int]) -> dict[str, dict[str, Any]]:
    assert len(execution.runs) == 3
    expected_predictions = 6 if family == "gbm" else 3
    assert execution.catalog_rows.height == expected_predictions
    assert set(execution.catalog_rows["execution_tier"]) == {"preview"}
    diagnostics = {}
    for run in execution.runs:
        spec = run.training.spec()
        config_name = spec["config_name"]
        assert len(run.predictions) == (2 if family == "gbm" else 1)
        for prediction in run.predictions:
            frame = prediction.load()
            coverage = prediction.coverage()
            keys = frame.select("symbol", "timestamp", "fold")
            assert prediction.complete
            assert coverage["status"] == "complete"
            assert coverage["n_expected"] == coverage["n_actual"] == frame.height
            assert keys.n_unique(["symbol", "timestamp", "fold"]) == keys.height
            assert sorted(keys["fold"].unique().to_list()) == folds
            assert frame["prediction"].is_finite().all()
        model_dir = run.training.root / "run_log" / "training" / run.training.hash / "models"
        if family == "linear":
            fitted_folds = {path.stem for path in model_dir.glob("fold_*.joblib")}
        elif family == "gbm":
            fitted_folds = {path.stem for path in (model_dir / "boosters").glob("fold_*.txt")}
        else:
            fitted_folds = {
                path.parent.name
                for path in (model_dir / run.training.hash).glob("fold_*/epoch_0001.pt")
            }
        expected_fitted_folds = {
            f"fold_{fold:02d}" if family == "tabular_dl" else f"fold_{fold}" for fold in folds
        }
        assert fitted_folds == expected_fitted_folds
        diagnostics[_run_key(run, family)] = dict(run.diagnostics)
    config_names = list(diagnostics)
    assert (
        diagnostics[config_names[0]]["compatibility_group"]
        == diagnostics[config_names[1]]["compatibility_group"]
    )
    assert diagnostics[config_names[0]]["compatibility_group_size"] == 2
    assert diagnostics[config_names[2]]["compatibility_group_size"] == 1
    assert (
        diagnostics[config_names[0]]["compatibility_group"]
        != diagnostics[config_names[2]]["compatibility_group"]
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in diagnostics.values():
        groups.setdefault(str(item["compatibility_group"]), []).append(item)
    for group in groups.values():
        preparation_counts = {int(item["base_fold_preparations"]) for item in group}
        prepared_folds = {int(fold) for item in group for fold in item.get("fitted_folds", [])}
        assert preparation_counts == {len(prepared_folds)}
    assert all(item["disk_fold_cache"] is False for item in diagnostics.values())
    return diagnostics


def prove(
    workspace: Path,
    *,
    family: str,
    folds: list[int],
    max_symbols: int,
    train_sample_frac: float,
    device: str,
) -> dict[str, object]:
    _seed_worktree_workspace(workspace)
    study = Study.open(CASE_STUDY, workspace=workspace)
    execution = run_models(
        study,
        requests=_requests(
            study,
            family=family,
            folds=folds,
            max_symbols=max_symbols,
            train_sample_frac=train_sample_frac,
            device=device,
        ),
    )
    diagnostics = _assert_results(execution, family, folds)
    first_hashes = {
        _run_key(run, family): {
            "training": run.training.hash,
            "predictions": [prediction.hash for prediction in run.predictions],
        }
        for run in execution.runs
    }

    restarted_study = Study.open(CASE_STUDY, workspace=workspace)
    restarted = run_models(
        restarted_study,
        requests=_requests(
            restarted_study,
            family=family,
            folds=folds,
            max_symbols=max_symbols,
            train_sample_frac=train_sample_frac,
            device=device,
        ),
    )
    restarted_hashes = {
        _run_key(run, family): {
            "training": run.training.hash,
            "predictions": [prediction.hash for prediction in run.predictions],
        }
        for run in restarted.runs
    }
    assert restarted_hashes == first_hashes
    reuse_field = "reused" if family == "tabular_dl" else "cache_hit"
    assert all(run.diagnostics[reuse_field] is True for run in restarted.runs)
    assert all(run.diagnostics["base_fold_preparations"] == 0 for run in restarted.runs)
    prediction_hashes = [
        prediction_hash
        for record in first_hashes.values()
        for prediction_hash in record["predictions"]
    ]
    selected = restarted_study.predictions.table(include_preview=True).filter(
        pl.col("prediction_hash").is_in(prediction_hashes)
    )
    assert selected.height == len(prediction_hashes)
    assert set(selected["prediction_hash"]) == set(prediction_hashes)

    group_measurements = {}
    for item in diagnostics.values():
        group_measurements.setdefault(
            item["compatibility_group"],
            {
                "base_fold_preparation_s": item["base_fold_preparation_s"],
                "base_fold_preparations": item["base_fold_preparations"],
                "candidate_fit_s": 0.0,
                "group_size": item["compatibility_group_size"],
                "preparation_fraction": item["preparation_fraction"],
            },
        )
        group_measurements[item["compatibility_group"]]["candidate_fit_s"] += item[
            "candidate_fit_s"
        ]

    return {
        "family": family,
        "folds": folds,
        "groups": group_measurements,
        "hashes": first_hashes,
        "max_symbols": max_symbols,
        "selected_catalog_rows": selected.height,
        "train_sample_frac": train_sample_frac if family != "tabular_dl" else None,
        "workspace": str(study.storage_root("preview")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--family", choices=("gbm", "linear", "tabular_dl"), default="linear")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    args = parser.parse_args()
    print(
        json.dumps(
            prove(
                args.workspace,
                family=args.family,
                folds=sorted(set(args.folds)),
                max_symbols=args.max_symbols,
                train_sample_frac=args.train_sample_frac,
                device=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
