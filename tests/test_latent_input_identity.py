"""Training-input identity and hash-scoped latent artifact regressions."""

from pathlib import Path

from case_studies.utils.latent_factors import case_study
from case_studies.utils.latent_factors import cv as latent_cv
from case_studies.utils.registry.specs import build_training_spec, training_hash_from_spec


def test_feature_content_enters_latent_training_hash(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "etfs"
    (case_dir / "features").mkdir(parents=True)
    (case_dir / "labels").mkdir()
    (case_dir / "config").mkdir()
    (case_dir / "features" / "financial.parquet").write_bytes(b"financial-v1")
    (case_dir / "features" / "model_based.parquet").write_bytes(b"temporal-v1")
    (case_dir / "labels" / "fwd_ret_21d.parquet").write_bytes(b"labels-v1")
    (case_dir / "config" / "setup.yaml").write_text("version: 1\n")

    monkeypatch.setattr(case_study, "get_case_study_dir", lambda _case_study_id: case_dir)
    monkeypatch.setattr(case_study, "load_feature_spec", lambda *_args: None)
    monkeypatch.setattr(case_study, "load_label_spec", lambda *_args: None)
    monkeypatch.setattr(
        case_study,
        "resolve_storage_path",
        lambda _case_study_id, _spec, fallback: case_dir / fallback,
    )

    first = case_study._training_input_identity("etfs", "fwd_ret_21d")
    first_spec = {"family": "latent_factors", "label": "fwd_ret_21d", "seed": 42}
    first_spec["input_data"] = first

    (case_dir / "features" / "financial.parquet").write_bytes(b"financial-v2")
    second = case_study._training_input_identity("etfs", "fwd_ret_21d")
    second_spec = {**first_spec, "input_data": second}

    assert first["input_digest"] != second["input_digest"]
    assert training_hash_from_spec(first_spec) != training_hash_from_spec(second_spec)


def test_fold_extras_are_scoped_by_training_hash(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "etfs"
    first_path = case_dir / "run_log" / "training" / "training-a" / "fold_extras.json"
    second_path = case_dir / "run_log" / "training" / "training-b" / "fold_extras.json"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)
    latent_cv._save_fold_extras(first_path, [{"fold_id": 0, "converged": True}])
    latent_cv._save_fold_extras(second_path, [{"fold_id": 0, "converged": False}])

    monkeypatch.setattr("utils.paths.get_case_study_dir", lambda _case_study_id: case_dir)

    assert latent_cv.load_fold_extras("etfs", "training-a") == [{"fold_id": 0, "converged": True}]
    assert latent_cv.load_fold_extras("etfs", "training-b") == [{"fold_id": 0, "converged": False}]


def test_gbm_training_hash_binds_modeling_input_identity() -> None:
    first_input = {
        "version": "v1",
        "files": [{"role": "financial", "sha256": "sha256:first"}],
        "input_digest": "sha256:first",
    }
    second_input = {
        "version": "v1",
        "files": [{"role": "financial", "sha256": "sha256:second"}],
        "input_digest": "sha256:second",
    }
    common = {
        "family": "gbm",
        "config_name": "leaves_7_mae",
        "label": "fwd_ret_21d",
        "n_folds": 8,
        "max_bin": 63,
        "checkpoint_interval": 50,
    }

    first = build_training_spec(
        **common,
        extra_params={"input_data_spec": first_input},
    )
    second = build_training_spec(
        **common,
        extra_params={"input_data_spec": second_input},
    )

    assert first["params"]["input_data_spec"] == first_input
    assert second["params"]["input_data_spec"] == second_input
    assert training_hash_from_spec(first) != training_hash_from_spec(second)
