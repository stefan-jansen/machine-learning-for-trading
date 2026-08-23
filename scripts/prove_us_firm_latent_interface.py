"""Run the reduced real-data IPCA proof for US firm characteristics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from case_studies.research import OfficialPopulation, Result, run_models
from case_studies.us_firm_characteristics.research_workflow import (
    completed_prediction_rows,
    expected_prediction_hashes,
    model_request_catalog,
    model_requests,
    open_study,
)
from case_studies.utils.artifact_digest import value_digest

CASE_STUDY = "us_firm_characteristics"
LABEL = "fwd_ret_1m"
CONFIG_NAME = "ipca"
RUNTIME_OVERRIDES = {"device": "cpu", "fold_workers": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_snapshot(model_dir: Path) -> dict[str, str]:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = dict(manifest["files"])
    if not files:
        raise AssertionError("IPCA manifest contains no fitted-state files")
    for relative, expected_digest in files.items():
        path = model_dir / relative
        assert path.is_file()
        assert _sha256(path) == expected_digest
    return files


def _prediction_keys(frame: pl.DataFrame) -> pl.DataFrame:
    fold = "fold" if "fold" in frame.columns else "fold_id"
    return frame.select("symbol", "timestamp", pl.col(fold).alias("fold")).sort(
        "symbol", "timestamp", "fold"
    )


def prove(
    workspace: Path,
    *,
    max_symbols: int = 50,
    max_iter: int = 100,
    require_notebook_results: bool = False,
) -> dict[str, Any]:
    preview_reductions = {
        "folds": [0],
        "max_iter": max_iter,
        "max_symbols": max_symbols,
        "n_factors": 2,
    }
    study = open_study(execution_tier="preview", workspace=workspace)
    request_catalog = model_request_catalog(
        "latent_factors",
        labels=(LABEL,),
        config_names=(CONFIG_NAME,),
    )
    helper_request = model_requests(
        study,
        request_catalog,
        execution_tier="preview",
        preview_reductions=preview_reductions,
    )[0]
    direct_request = study.model(
        family="latent_factors",
        label=LABEL,
        config_name=CONFIG_NAME,
        execution_tier="preview",
        overrides=RUNTIME_OVERRIDES,
        preview_reductions=preview_reductions,
    )
    helper_resolved = helper_request.resolve()
    direct_resolved = direct_request.resolve()
    assert helper_resolved.identity == direct_resolved.identity
    assert helper_resolved.spec == direct_resolved.spec

    computation = direct_resolved.spec["computation"]
    assert computation["preview_reductions"] == preview_reductions
    assert computation["runtime"]["device"] == "cpu"
    assert computation["runtime"]["fold_workers"] == 4
    assert computation["model"]["class"] == CONFIG_NAME
    assert computation["model"]["n_factors"] == 2
    assert computation["model"]["params"]["max_iter"] == max_iter
    assert computation["checkpoint_schedule"] == [{"kind": "epoch", "value": 0}]

    projected_hashes = expected_prediction_hashes((direct_resolved,))
    if require_notebook_results:
        notebook_rows = study.predictions.table(include_preview=True).filter(
            pl.col("prediction_hash").is_in(projected_hashes)
        )
        assert notebook_rows.height == len(projected_hashes)
        assert notebook_rows.get_column("complete").all()
        assert notebook_rows.get_column("artifact_available").all()

    execution = run_models(study, requests=(direct_request,))
    run = execution.runs[0]
    assert run.training.hash == direct_resolved.identity
    assert tuple(prediction.hash for prediction in run.predictions) == projected_hashes
    prediction = run.predictions[0]
    frame = prediction.load()
    actual_keys = _prediction_keys(frame)
    expected_keys = _prediction_keys(direct_resolved._context.expected_keys)
    key_columns = ["symbol", "timestamp", "fold"]
    assert actual_keys.height == actual_keys.n_unique(key_columns)
    assert actual_keys.join(expected_keys, on=key_columns, how="anti").is_empty()
    assert expected_keys.join(actual_keys, on=key_columns, how="anti").is_empty()
    assert actual_keys.get_column("fold").unique().to_list() == [0]
    assert 1 < actual_keys.get_column("symbol").n_unique() <= max_symbols
    assert frame.get_column("prediction").is_finite().all()
    assert frame.get_column("actual").is_finite().all()

    split = direct_resolved._context.case.splits[0]
    assert split["train_end"] < split["val_start"]
    first_timestamp = actual_keys.get_column("timestamp").min()
    last_timestamp = actual_keys.get_column("timestamp").max()
    assert split["val_start"] <= first_timestamp <= last_timestamp <= split["val_end"]
    coverage = prediction.coverage()
    assert coverage is not None and coverage["status"] == "complete"
    assert coverage["n_expected"] == coverage["n_actual"] == actual_keys.height

    training_dir = run.training.root / "run_log" / "training" / run.training.hash
    model_dir = training_dir / "models"
    manifest_before = _manifest_snapshot(model_dir)
    model_extras = model_dir / "fold_extras.json"
    public_extras = training_dir / "fold_extras.json"
    assert model_extras.read_bytes() == public_extras.read_bytes()
    fold_extras = json.loads(model_extras.read_text())
    assert [extra["fold_id"] for extra in fold_extras] == [0]
    assert all(extra.get("converged") is True for extra in fold_extras)
    assert all(0 < int(extra["iterations"]) <= max_iter for extra in fold_extras)

    restarted = run_models(study, requests=(direct_request,)).runs[0]
    assert restarted.training.hash == run.training.hash
    assert tuple(item.hash for item in restarted.predictions) == projected_hashes
    assert value_digest(restarted.predictions[0].load()) == value_digest(frame)
    assert _manifest_snapshot(model_dir) == manifest_before

    opened = Result.open(study, prediction.hash, include_preview=True)
    assert opened.complete
    assert value_digest(opened.load()) == value_digest(frame)
    catalog = completed_prediction_rows(study, projected_hashes)
    assert catalog.height == 1
    assert catalog.item(0, "execution_tier") == "preview"
    canonical_catalog = study.predictions.table(include_preview=False)
    if not canonical_catalog.is_empty():
        assert prediction.hash not in canonical_catalog.get_column("prediction_hash").to_list()
    try:
        OfficialPopulation.create(
            study,
            name="preview-ipca-must-not-enter-official-population",
            member_kind="prediction",
            members=[prediction.hash],
        )
    except ValueError as error:
        assert "preview" in str(error)
    else:
        raise AssertionError("preview IPCA prediction entered an official population")

    return {
        "config_name": CONFIG_NAME,
        "converged": True,
        "eligible_rows": actual_keys.height,
        "first_timestamp": str(first_timestamp),
        "iterations": [int(extra["iterations"]) for extra in fold_extras],
        "last_timestamp": str(last_timestamp),
        "manifest_files": len(manifest_before),
        "prediction_hash": prediction.hash,
        "symbols": actual_keys.get_column("symbol").n_unique(),
        "training_hash": run.training.hash,
        "workspace": str(study.storage_root("preview")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--require-notebook-results", action="store_true")
    args = parser.parse_args()
    summary = prove(
        args.workspace,
        max_symbols=args.max_symbols,
        max_iter=args.max_iter,
        require_notebook_results=args.require_notebook_results,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
