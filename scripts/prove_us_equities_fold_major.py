"""Run the reduced real-data proof for US-equities fold-major execution."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    folds: list[int],
    max_symbols: int,
    train_sample_frac: float,
):
    common_reductions = {
        "folds": folds,
        "max_symbols": max_symbols,
        "train_sample_frac": train_sample_frac,
    }
    incompatible_reductions = {
        **common_reductions,
        "train_sample_frac": train_sample_frac / 2,
    }
    return [
        study.model(
            family="linear",
            label=LABEL,
            config_name=config_name,
            execution_tier="preview",
            preview_reductions=(
                common_reductions if config_name != "ridge_a100.0" else incompatible_reductions
            ),
        )
        for config_name in ("ridge_a1.0", "ridge_a10.0", "ridge_a100.0")
    ]


def _assert_results(execution, folds: list[int]) -> dict[str, dict[str, object]]:
    assert len(execution.runs) == 3
    assert execution.catalog_rows.height == 3
    assert set(execution.catalog_rows["execution_tier"]) == {"preview"}
    diagnostics = {}
    for run in execution.runs:
        spec = run.training.spec()
        config_name = spec["config_name"]
        prediction = run.predictions[0]
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
        assert sorted(path.stem for path in model_dir.glob("fold_*.joblib")) == [
            f"fold_{fold}" for fold in folds
        ]
        diagnostics[config_name] = dict(run.diagnostics)
    assert (
        diagnostics["ridge_a1.0"]["compatibility_group"]
        == diagnostics["ridge_a10.0"]["compatibility_group"]
    )
    assert diagnostics["ridge_a1.0"]["compatibility_group_size"] == 2
    assert diagnostics["ridge_a100.0"]["compatibility_group_size"] == 1
    assert (
        diagnostics["ridge_a1.0"]["compatibility_group"]
        != diagnostics["ridge_a100.0"]["compatibility_group"]
    )
    for item in diagnostics.values():
        expected_preparations = 0 if item["cache_hit"] else len(folds)
        assert item["base_fold_preparations"] == expected_preparations
    assert all(item["disk_fold_cache"] is False for item in diagnostics.values())
    return diagnostics


def prove(
    workspace: Path,
    *,
    folds: list[int],
    max_symbols: int,
    train_sample_frac: float,
) -> dict[str, object]:
    _seed_worktree_workspace(workspace)
    study = Study.open(CASE_STUDY, workspace=workspace)
    execution = run_models(
        study,
        requests=_requests(
            study,
            folds=folds,
            max_symbols=max_symbols,
            train_sample_frac=train_sample_frac,
        ),
    )
    diagnostics = _assert_results(execution, folds)
    first_hashes = {
        run.training.spec()["config_name"]: {
            "training": run.training.hash,
            "prediction": run.predictions[0].hash,
        }
        for run in execution.runs
    }

    restarted_study = Study.open(CASE_STUDY, workspace=workspace)
    restarted = run_models(
        restarted_study,
        requests=_requests(
            restarted_study,
            folds=folds,
            max_symbols=max_symbols,
            train_sample_frac=train_sample_frac,
        ),
    )
    restarted_hashes = {
        run.training.spec()["config_name"]: {
            "training": run.training.hash,
            "prediction": run.predictions[0].hash,
        }
        for run in restarted.runs
    }
    assert restarted_hashes == first_hashes
    assert all(run.diagnostics["cache_hit"] is True for run in restarted.runs)
    assert all(run.diagnostics["base_fold_preparations"] == 0 for run in restarted.runs)
    selected = restarted_study.predictions.table(include_preview=True).filter(
        (pl.col("label") == LABEL)
        & (pl.col("family") == "linear")
        & pl.col("config_name").is_in(list(first_hashes))
        & (pl.col("execution_tier") == "preview")
        & pl.col("complete")
    )
    assert selected.height == 3

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
        "folds": folds,
        "groups": group_measurements,
        "hashes": first_hashes,
        "max_symbols": max_symbols,
        "selected_catalog_rows": selected.height,
        "train_sample_frac": train_sample_frac,
        "workspace": str(study.storage_root("preview")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    args = parser.parse_args()
    print(
        json.dumps(
            prove(
                args.workspace,
                folds=sorted(set(args.folds)),
                max_symbols=args.max_symbols,
                train_sample_frac=args.train_sample_frac,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
