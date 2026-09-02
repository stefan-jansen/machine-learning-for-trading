from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from case_studies.research import CVSpec, LabelDefinition, Study, open_study
from case_studies.research.contracts import ExecutionTier
from case_studies.research.model_planning import ModelPlan, PlannedModel
from case_studies.utils import linear
from case_studies.utils.registry.store import _open_registry
from utils import modeling
from utils.artifact_specs import load_setup_config
from utils.paths import get_case_study_dir


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed_release(tmp_path: Path, *, marker: str = "release") -> Path:
    release = tmp_path / "release"
    case_dir = release / "case_studies" / "etfs"
    (case_dir / "run_log").mkdir(parents=True)
    (case_dir / "run_log" / "registry.db").write_bytes(b"")
    (case_dir / "config").mkdir()
    (case_dir / "config" / "setup.yaml").write_text(
        "\n".join(
            [
                f"marker: {marker}",
                "labels:",
                "  primary: fwd_ret_21d",
                "  buffer: 2D",
                "  horizons: {fwd_ret_21d: 2D}",
                "evaluation:",
                "  n_splits: 2",
                "  train_size: 4D",
                "  val_size: 2D",
                "  holdout_start: '2024-01-11'",
                "  holdout_end: '2024-01-12'",
                "  calendar: crypto",
                "decision:",
                "  cadence: daily_close",
                "  execution_delay: next_bar_open",
                "mapping:",
                "  position_state_space: long_only",
                "costs:",
                "  class: negligible",
            ]
        )
        + "\n"
    )
    shared = release / "case_studies" / "config" / "linear"
    shared.mkdir(parents=True)
    (shared / "ridge.yaml").write_text("family: linear\nparams: {}\n")
    return release


def test_crypto_preview_then_canonical_workspace_isolates_symlinked_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    from case_studies.crypto_perps_funding import research_workflow

    release = tmp_path / "release"
    case_root = release / "case_studies" / "crypto_perps_funding"
    artifacts = tmp_path / "artifacts"
    for name in ("features", "labels", "run_log"):
        source = artifacts / name
        source.mkdir(parents=True)
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / name).symlink_to(source, target_is_directory=True)
    (artifacts / "features" / "input.bin").write_bytes(b"release")
    (artifacts / "run_log" / "registry.db").write_bytes(b"released-run-log")
    (case_root / "config").mkdir()
    (case_root / "config" / "setup.yaml").write_text("evaluation: {}\n")
    (release / "case_studies" / "config").mkdir()
    monkeypatch.setattr(research_workflow, "REPO_ROOT", release)

    workspace = tmp_path / "workspace"
    preview_study = research_workflow.open_study(execution_tier="preview", workspace=workspace)
    preview_root = preview_study.activate("preview")
    assert preview_root == workspace / ".preview" / "crypto_perps_funding"
    assert preview_root.joinpath("features").resolve() == workspace.joinpath(
        "crypto_perps_funding", "features"
    )
    study = research_workflow.open_study(execution_tier="canonical", workspace=workspace)

    assert study.root == workspace / "crypto_perps_funding"
    assert study.root.joinpath("features").is_dir()
    assert not study.root.joinpath("features").is_symlink()
    study.root.joinpath("features", "input.bin").write_bytes(b"workspace")
    assert artifacts.joinpath("features", "input.bin").read_bytes() == b"release"
    assert study.root.joinpath("labels").is_dir()
    assert not study.root.joinpath("labels").is_symlink()
    assert study.root.joinpath("run_log").is_dir()
    assert not study.root.joinpath("run_log").is_symlink()
    assert study.root.joinpath("config", "setup.yaml").read_text() == "evaluation: {}\n"
    assert not study.root.joinpath("config").is_symlink()
    assert study.output_root.joinpath("config").is_dir()
    assert not study.output_root.joinpath("config").is_symlink()
    study.root.joinpath("config", "setup.yaml").write_text("evaluation: changed\n")
    assert case_root.joinpath("config", "setup.yaml").read_text() == "evaluation: {}\n"
    study.output_root.joinpath("config", "probe.yaml").write_text("workspace: true\n")
    assert not release.joinpath("case_studies", "config", "probe.yaml").exists()
    # The workspace starts an empty run log rather than inheriting the released one,
    # so a preview cannot read released results as if it had produced them.
    assert study.root.joinpath("run_log", "registry.db").read_bytes() != b"released-run-log"
    assert artifacts.joinpath("run_log", "registry.db").read_bytes() == b"released-run-log"
    assert study.manifest["baseline_source_commit"]
    assert study.manifest["baseline_manifest_sha256"]


def _planned(
    family: str,
    label: str,
    config_name: str,
    training_hash: str,
    prediction_hash: str,
) -> PlannedModel:
    return PlannedModel(
        family=family,
        label=label,
        config_name=config_name,
        training_hash=training_hash,
        checkpoint_kind="final",
        checkpoint_value=None,
        prediction_hash=prediction_hash,
        spec_json="{}",
    )


def test_crypto_model_population_is_frozen_before_the_first_fit(tmp_path, monkeypatch) -> None:
    from case_studies.crypto_perps_funding import research_workflow

    root = tmp_path / "crypto_perps_funding"
    root.mkdir()
    _open_registry(root).close()
    study = Study(
        case_study="crypto_perps_funding",
        root=root,
        release_root=tmp_path,
        output_root=tmp_path,
        read_only=False,
        manifest={},
    )

    class _FailingPlan(ModelPlan):
        def run(self):
            # The registry must already carry the complete declared population at the moment
            # the first fit is attempted, so a failed member cannot silently disappear.
            with sqlite3.connect(root / "run_log" / "registry.db") as db:
                row = db.execute(
                    "SELECT p.name, p.member_kind, m.member_hash "
                    "FROM official_populations AS p "
                    "JOIN official_population_members AS m USING (population_hash)"
                ).fetchone()
            assert row == (
                "crypto-linear-validation-predictions-v1",
                "prediction",
                "prediction-1",
            )
            raise RuntimeError("first fit failed")

    plan = _FailingPlan(
        study,
        (object(),),
        (_planned("linear", "fwd_ret_8h", "ols", "training-1", "prediction-1"),),
        ExecutionTier.CANONICAL,
        (),
    )
    monkeypatch.setattr(research_workflow, "plan_models", lambda *args, **kwargs: plan)

    with pytest.raises(RuntimeError, match="first fit failed"):
        research_workflow.run_model_catalog(
            study,
            pl.DataFrame({"family": ["linear"], "label": ["fwd_ret_8h"], "config_name": ["ols"]}),
            execution_tier="canonical",
            population_name="crypto-linear-validation-predictions-v1",
        )

    population = research_workflow.OfficialPopulation.one(
        study, name="crypto-linear-validation-predictions-v1"
    )
    assert population.members == ("prediction-1",)
    with pytest.raises(ValueError, match="prediction-1:missing"):
        population.require_complete()


def test_crypto_official_population_declares_gpu_and_imbalance_treatment(
    tmp_path, monkeypatch
) -> None:
    """A silent CPU fallback would change the training identity of every GPU family."""
    from case_studies.crypto_perps_funding import research_workflow

    study = Study(
        case_study="crypto_perps_funding",
        root=tmp_path / "crypto_perps_funding",
        release_root=tmp_path,
        output_root=tmp_path,
        read_only=False,
        manifest={},
    )
    monkeypatch.setattr(
        research_workflow,
        "load_configs",
        lambda case_study, label, family: [
            {
                "config_name": {
                    "linear": "ols",
                    "gbm": "default_mse",
                    "tabular_dl": "tabm_s",
                    "deep_learning": "lstm_h64",
                }[family]
            }
        ],
    )

    requests = research_workflow.official_model_requests(study)
    overrides = {request.family: request.overrides for request in requests}
    labels = {}
    for request in requests:
        labels.setdefault(request.family, set()).add(request.label)

    assert overrides["tabular_dl"]["device"] == "cuda"
    assert overrides["deep_learning"]["device"] == "cuda"
    # Unbalanced intraday direction is the point of the TabM path; losing this silently
    # trains on the majority class and still registers a complete-looking result.
    assert overrides["tabular_dl"]["class_weight"] == "balanced"
    # A CPU-only family must not carry a device override at all, or its identity moves too.
    assert overrides["linear"] == {}
    assert overrides["gbm"] == {}
    # Sequence families are declared for the regression labels only.
    assert labels["deep_learning"] == set(research_workflow.REGRESSION_LABELS)
    assert labels["linear"] == set(research_workflow.ALL_LABELS)
    assert all(request.execution_tier is ExecutionTier.CANONICAL for request in requests)


def test_crypto_official_population_equals_the_union_of_the_model_notebooks(tmp_path) -> None:
    """Nothing may be declared that no notebook produces, or produced that nothing declares."""
    from case_studies.crypto_perps_funding import research_workflow as rw

    study = Study(
        case_study="crypto_perps_funding",
        root=tmp_path / "crypto_perps_funding",
        release_root=tmp_path,
        output_root=tmp_path,
        read_only=False,
        manifest={},
    )

    # Restated independently of research_workflow: one entry per model notebook, matching the
    # catalog each one builds. If a notebook's slice moves, this disagrees with the official
    # population rather than letting the population quietly cover something nothing produces.
    notebook_slices = [
        ("linear", rw.ALL_LABELS, None),
        ("gbm", rw.ALL_LABELS, None),
        ("tabular_dl", rw.ALL_LABELS, "tabm"),
        ("deep_learning", rw.REGRESSION_LABELS, ("nlinear", "lstm")),
        ("deep_learning", rw.REGRESSION_LABELS, "tcn"),
    ]
    from_notebooks = {
        (family, row["label"], row["config_name"])
        for family, labels, prefix in notebook_slices
        for row in rw.model_request_catalog(family, labels=labels, config_prefix=prefix).iter_rows(
            named=True
        )
    }
    official = {
        (request.family, request.label, request.config_name)
        for request in rw.official_model_requests(study)
    }

    assert official == from_notebooks

    # And the union must exhaust the published menu apart from the members the sequence
    # runner cannot execute, which must be non-empty here or the rule is untested.
    assert rw.declared_menu() - official == rw.unsupported_requests()
    assert rw.unsupported_requests()
    # nlinear has published results in the released registry, so it must be covered, not excluded.
    assert ("deep_learning", "fwd_ret_8h", "nlinear") in official


def test_crypto_complete_model_population_is_frozen_before_family_execution(
    tmp_path, monkeypatch
) -> None:
    from case_studies.crypto_perps_funding import research_workflow

    root = tmp_path / "crypto_perps_funding"
    root.mkdir()
    _open_registry(root).close()
    study = Study(
        case_study="crypto_perps_funding",
        root=root,
        release_root=tmp_path,
        output_root=tmp_path,
        read_only=False,
        manifest={},
    )
    requests = (object(), object())
    plan = ModelPlan(
        study,
        requests,
        (
            _planned("linear", "fwd_ret_8h", "ols", "training-1", "prediction-1"),
            _planned("gbm", "fwd_dir_8h", "lgbm", "training-2", "prediction-2"),
        ),
        ExecutionTier.CANONICAL,
        (),
    )
    monkeypatch.setattr(research_workflow, "official_model_requests", lambda study: requests)
    monkeypatch.setattr(research_workflow, "plan_models", lambda *args, **kwargs: plan)

    population = research_workflow.freeze_official_model_population(study)

    assert population.name == research_workflow.OFFICIAL_POPULATION
    assert population.members == ("prediction-1", "prediction-2")
    with pytest.raises(ValueError, match="prediction-1:missing, prediction-2:missing"):
        population.require_complete()


def _seed_regeneration_release(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    release = _seed_release(tmp_path)
    case_dir = release / "case_studies" / "etfs"
    generated_root = release / "generated" / "etfs"
    targets = {name: generated_root / name for name in ("features", "labels", "run_log")}
    generated_root.mkdir(parents=True)
    (case_dir / "run_log").rename(targets["run_log"])
    targets["features"].mkdir()
    targets["labels"].mkdir()
    for name, target in targets.items():
        (case_dir / name).symlink_to(target, target_is_directory=True)

    (case_dir / "config" / "setup.yaml").write_text(
        "\n".join(
            [
                "labels:",
                "  primary: fwd_ret_1d",
                "  buffer: 0D",
                "  horizons: {fwd_ret_1d: 0D}",
                "evaluation:",
                "  n_splits: 2",
                "  train_size: 4D",
                "  val_size: 2D",
                "  holdout_start: '2024-01-11'",
                "  holdout_end: '2024-01-12'",
                "  calendar: crypto",
                "decision:",
                "  cadence: daily_close",
                "  execution_delay: next_bar_open",
                "mapping:",
                "  position_state_space: long_only",
                "costs:",
                "  class: negligible",
            ]
        )
        + "\n"
    )
    (release / "case_studies" / "config" / "linear" / "ridge.yaml").write_text(
        "\n".join(
            [
                "config_name: ridge",
                "family: linear",
                "library: sklearn",
                "model_class: Ridge",
                "params:",
                "  alpha: 1.0",
            ]
        )
        + "\n"
    )
    rows = []
    for date_index, timestamp in enumerate(
        pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 10), eager=True)
    ):
        for symbol_index in range(6):
            x1 = float(symbol_index - 2.5)
            x2 = float(date_index) + x1 / 10
            rows.append(
                {
                    "symbol": f"S{symbol_index}",
                    "timestamp": timestamp,
                    "x1": x1,
                    "x2": x2,
                    "fwd_ret_1d": 0.03 * x1 + 0.01 * x2,
                }
            )
    frame = pl.DataFrame(rows)
    frame.select("symbol", "timestamp", "x1", "x2").write_parquet(
        targets["features"] / "financial.parquet"
    )
    frame.select("symbol", "timestamp", "fwd_ret_1d").write_parquet(
        targets["labels"] / "fwd_ret_1d.parquet"
    )
    _open_registry(case_dir).close()
    return release, targets


def test_study_create_and_reopen_do_not_change_release_bytes(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    before = _tree_digest(release)

    study = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=release)

    assert study.root == reopened.root == tmp_path / "workspace" / "etfs"
    assert study.manifest == reopened.manifest
    assert _tree_digest(release) == before


def test_seeded_output_root_becomes_the_explicit_isolated_preview_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    release = _seed_release(tmp_path)
    release_before = _tree_digest(release)
    isolated = tmp_path / "isolated"
    seeded_case = isolated / "etfs"
    (seeded_case / "config").mkdir(parents=True)
    (seeded_case / "config" / "setup.yaml").write_text("marker: seeded\n")
    _open_registry(seeded_case).close()
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(isolated))

    study = Study.open("etfs", workspace=isolated, release_root=release)
    training = study.results.register_training(
        {
            "identity_version": 2,
            "execution_tier": "preview",
            "family": "linear",
            "label": "fwd_ret_21d",
            "config_name": "ridge",
            "seed": 42,
            "preview_reductions": {"folds": [0]},
        },
        execution_tier="preview",
    )

    preview_case = isolated / ".preview" / "etfs"
    assert study.root == seeded_case
    assert study.manifest["adopted_output_root"] is True
    assert training.root == preview_case
    assert (seeded_case / ".study.json").is_file()
    assert (seeded_case / "run_log" / "registry.db").is_file()
    assert (preview_case / "run_log" / "registry.db").is_file()
    assert (preview_case / "run_log" / "training" / training.hash / "spec.json").is_file()
    assert _tree_digest(release) == release_before
    assert not (tmp_path / "experiments").exists()


@pytest.mark.parametrize("regular_directory", ["features", "labels", "run_log"])
def test_regeneration_rejects_regular_generated_artifact_directories(
    tmp_path: Path, regular_directory: str
) -> None:
    release, _ = _seed_regeneration_release(tmp_path)
    path = release / "case_studies" / "etfs" / regular_directory
    path.unlink()
    path.mkdir()

    with pytest.raises(PermissionError, match=regular_directory):
        Study.regenerate("etfs", release_root=release)


def test_regeneration_writes_through_resolved_directory_symlinks(tmp_path: Path) -> None:
    release, targets = _seed_regeneration_release(tmp_path)
    study = Study.regenerate("etfs", release_root=release)

    assert study.root == release / "case_studies" / "etfs"
    assert study.root == get_case_study_dir("etfs")
    assert all(
        (study.root / name).resolve(strict=True) == target for name, target in targets.items()
    )

    feature_probe = pl.DataFrame(
        {"symbol": ["S0"], "timestamp": ["2024-01-01"], "value": [1.0]}
    ).with_columns(pl.col("timestamp").str.to_date())
    feature_probe.write_parquet(get_case_study_dir("etfs") / "features" / "probe.parquet")
    published = study.labels.publish(
        LabelDefinition("probe", "regression", "1D"),
        feature_probe.rename({"value": "probe"}),
    )
    training = study.results.register_training(
        {
            "identity_version": 2,
            "execution_tier": "canonical",
            "family": "linear",
            "label": "probe",
            "config_name": "ridge",
            "seed": 42,
        }
    )

    assert (targets["features"] / "probe.parquet").is_file()
    assert published.path.resolve().parent == targets["labels"]
    assert (targets["run_log"] / "training" / training.hash / "spec.json").is_file()


def test_regeneration_preview_runs_real_model_without_changing_canonical_registry(
    tmp_path: Path, monkeypatch
) -> None:
    release, targets = _seed_regeneration_release(tmp_path)
    study = Study.regenerate("etfs", release_root=release)
    canonical_registry = targets["run_log"] / "registry.db"
    canonical_bytes = canonical_registry.read_bytes()
    with sqlite3.connect(canonical_registry) as db:
        canonical_training_rows = db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
    loaded_from = None
    real_loader = modeling.load_modeling_dataset

    def observed_loader(*args, **kwargs):
        nonlocal loaded_from
        loaded_from = get_case_study_dir("etfs", create=False)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(linear, "load_modeling_dataset", observed_loader)
    run = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        execution_tier=ExecutionTier.PREVIEW,
        preview_reductions={"folds": [0], "max_symbols": 3, "train_sample_frac": 1.0},
    ).run()

    preview_case = release / "case_studies" / ".preview" / "etfs"
    assert loaded_from == preview_case
    assert (preview_case / "config").is_symlink()
    assert (preview_case / "config").resolve(strict=True) == study.root / "config"
    preview_shared_config = release / "case_studies" / ".preview" / "config"
    canonical_shared_config = release / "case_studies" / "config"
    assert preview_shared_config.is_symlink()
    assert preview_shared_config.resolve(strict=True) == canonical_shared_config
    assert (preview_case / "features").is_symlink()
    assert (preview_case / "features").resolve(strict=True) == targets["features"]
    assert (preview_case / "labels").is_symlink()
    assert (preview_case / "labels").resolve(strict=True) == targets["labels"]
    assert run.training.root == preview_case
    assert (
        preview_case / "run_log" / "training" / run.training.hash / "models" / "manifest.json"
    ).is_file()
    assert run.predictions[0].complete
    assert canonical_registry.read_bytes() == canonical_bytes
    with sqlite3.connect(canonical_registry) as db:
        assert (
            db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
            == canonical_training_rows
        )
    assert not (targets["run_log"] / "training").exists()
    assert not (targets["run_log"] / "predictions").exists()
    assert not (targets["run_log"] / "backtest").exists()

    preview_request = study.model(
        family="linear",
        label="fwd_ret_1d",
        config_name="ridge",
        execution_tier=ExecutionTier.PREVIEW,
        preview_reductions={"folds": [0], "max_symbols": 3, "train_sample_frac": 1.0},
    )
    before_change = preview_request.resolve()
    setup_path = study.root / "config" / "setup.yaml"
    setup_path.write_text(setup_path.read_text().replace("train_size: 4D", "train_size: 3D"))
    after_change = preview_request.resolve()

    assert before_change.spec["computation"]["cv"] != after_change.spec["computation"]["cv"]
    canonical_preset = canonical_shared_config / "linear" / "ridge.yaml"
    canonical_preset.unlink()
    assert not (preview_shared_config / "linear" / "ridge.yaml").exists()


def test_release_study_is_read_only(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)

    study = Study.open("etfs", release_root=release)

    assert study.read_only
    with pytest.raises(PermissionError, match="read-only"):
        study.labels.publish(
            LabelDefinition("custom", "regression", "1D"),
            pl.DataFrame(
                {
                    "symbol": ["A"],
                    "timestamp": [pl.date(2024, 1, 1)],
                    "custom": [0.1],
                }
            ),
        )


def test_switching_workspaces_invalidates_root_sensitive_config_cache(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    first = Study.open("etfs", workspace=tmp_path / "first", release_root=release)
    (first.root / "config" / "setup.yaml").write_text(
        (first.root / "config" / "setup.yaml").read_text().replace("release", "first")
    )
    assert load_setup_config("etfs")["marker"] == "first"

    second = Study.open("etfs", workspace=tmp_path / "second", release_root=release)
    (second.root / "config" / "setup.yaml").write_text(
        (second.root / "config" / "setup.yaml").read_text().replace("release", "second")
    )

    assert load_setup_config("etfs")["marker"] == "second"


def test_label_lookup_reactivates_its_owning_workspace(tmp_path: Path) -> None:
    release = _seed_release(tmp_path)
    first = Study.open("etfs", workspace=tmp_path / "first", release_root=release)
    frame = pl.DataFrame(
        {"symbol": ["A"], "timestamp": ["2024-01-01"], "custom": [0.1]}
    ).with_columns(pl.col("timestamp").str.to_date())
    published = first.labels.publish(LabelDefinition("custom", "regression", "1D"), frame)
    Study.open("etfs", workspace=tmp_path / "second", release_root=release)

    resolved = first.labels.get("custom")

    assert resolved.path == published.path
    assert get_case_study_dir("etfs") == first.root


def test_custom_classification_label_resolves_continuous_target(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame = pl.DataFrame(
        {
            "symbol": ["A", "B"],
            "timestamp": ["2024-01-01", "2024-01-01"],
            "up_90p_5d": [0, 1],
            "fwd_ret_5d": [-0.01, 0.03],
        }
    ).with_columns(pl.col("timestamp").str.to_date())

    published = study.labels.publish(
        LabelDefinition(
            name="up_90p_5d",
            task_type="classification",
            horizon="5D",
            continuous_eval_label="fwd_ret_5d",
        ),
        frame,
    )
    reopened = Study.open("etfs", workspace=tmp_path / "workspace", release_root=study.release_root)
    resolved = reopened.labels.get("up_90p_5d")

    assert published.digest == resolved.digest
    assert resolved.definition.continuous_eval_label == "fwd_ret_5d"
    assert resolved.load().get_column("fwd_ret_5d").to_list() == [-0.01, 0.03]


def test_invalid_label_publication_has_no_partial_writes(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    definition = LabelDefinition(
        name="up_90p_5d",
        task_type="classification",
        horizon="5D",
        continuous_eval_label="fwd_ret_5d",
    )
    invalid = pl.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": ["2024-01-01", "2024-01-01"],
            "up_90p_5d": [0, 1],
        }
    )

    with pytest.raises(ValueError):
        study.labels.publish(definition, invalid)

    assert not (study.root / "labels" / "up_90p_5d.parquet").exists()
    assert not (study.root / "labels" / "up_90p_5d.parquet.digest.json").exists()


def test_label_discovery_rejects_artifact_digest_mismatch(tmp_path: Path) -> None:
    study = Study.open(
        "etfs", workspace=tmp_path / "workspace", release_root=_seed_release(tmp_path)
    )
    frame = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": ["2024-01-01"],
            "custom": [0.1],
        }
    ).with_columns(pl.col("timestamp").str.to_date())
    published = study.labels.publish(LabelDefinition("custom", "regression", "1D"), frame)
    frame.with_columns(pl.lit(0.2).alias("custom")).write_parquet(published.path)

    with pytest.raises(ValueError, match="digest"):
        study.labels.get("custom")


def test_cvspec_resolves_stable_exact_boundaries_and_changes_with_protocol() -> None:
    timeline = pl.DataFrame(
        {"timestamp": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 12), eager=True)}
    )
    base = CVSpec.walk_forward(
        training_window="4D",
        validation_window="2D",
        retrain_every="2D",
        folds=range(2),
        horizon="1D",
        holdout_start="2024-01-11",
        holdout_end="2024-01-12",
        calendar=None,
    )

    first = base.resolve(timeline)
    second = base.resolve(timeline)
    reordered = base.with_changes(folds=(1, 0)).resolve(timeline)
    changed = base.with_changes(training_window="3D").resolve(timeline)

    assert first == second == reordered
    assert first.normalized_folds != changed.normalized_folds
    assert first.identity != changed.identity


def test_custom_cv_cannot_relabel_fold_scoped_temporal_features() -> None:
    from case_studies.research.cv import require_fold_scoped_temporal_compatibility

    artifact = [
        {
            "fold": 0,
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "val_start": "2021-01-01",
            "val_end": "2021-03-31",
        },
        {
            "fold": 1,
            "train_start": "2020-04-01",
            "train_end": "2021-03-31",
            "val_start": "2021-04-01",
            "val_end": "2021-06-30",
        },
    ]

    require_fold_scoped_temporal_compatibility([artifact[1]], artifact)
    changed = [{**artifact[1], "train_end": "2021-05-31"}]
    with pytest.raises(ValueError, match="incompatible with fold-scoped temporal features"):
        require_fold_scoped_temporal_compatibility(changed, artifact)


def test_preview_records_the_entry_point_when_generated_dirs_are_not_symlinks(
    tmp_path: Path,
) -> None:
    """A clean clone has regular generated directories, and its runs must still say who wrote them.

    The symlink branch of `open_study` is a maintainer-worktree convenience; the branch taken
    everywhere else must carry `entry_point` just the same, or the registry row loses the only
    column that names the notebook.
    """
    release = _seed_release(tmp_path)
    generated = release / "case_studies" / "etfs"
    assert not any((generated / name).is_symlink() for name in ("features", "labels", "run_log")), (
        "fixture must exercise the regular-directory branch"
    )

    study = open_study(
        "etfs",
        execution_tier=ExecutionTier.PREVIEW,
        workspace=tmp_path / "ws",
        release_root=release,
        entry_point="06_linear",
    )
    training = study.results.register_training(
        {
            "identity_version": 2,
            "execution_tier": "preview",
            "family": "linear",
            "label": "fwd_ret_21d",
            "config_name": "ridge",
            "seed": 42,
            "preview_reductions": {"folds": [0]},
        },
        execution_tier="preview",
    )

    with sqlite3.connect(training.root / "run_log" / "registry.db") as db:
        recorded = db.execute(
            "SELECT entry_point FROM training_runs WHERE training_hash = ?", (training.hash,)
        ).fetchone()
    assert recorded == ("06_linear",)


def test_a_second_study_previewing_into_one_workspace_repoints_the_input_links(
    tmp_path: Path,
) -> None:
    """Two studies, one workspace, in sequence: the second must not inherit the first's inputs.

    `activate` links `labels` and `features` into `<workspace>/.preview/<case>/` so a preview
    reads real inputs while writing only to the workspace. The link belongs to whichever study
    is active. When a second study with different input directories activates into the same
    workspace, the link has to follow it: leaving it is worse than any error, because the
    preview would then read the previous study's labels under the current study's name.

    This is the case that shipped. `_ensure_input_link` refused a link resolving elsewhere while
    `_ensure_config_link`, ten lines below, repaired exactly that situation for `config`. Two
    functions handling one situation two ways, and the refusal is the wrong half.

    It never surfaced in CI, which is the part worth keeping: a plain clone has regular
    generated directories, so every study routes through the same branch and resolves the same
    inputs. Only a maintainer worktree, whose `labels`/`features`/`run_log` are symlinks into
    shared data, reaches the branch where two studies disagree - and there the failure is
    ordered, so a notebook passes alone and fails after its predecessor in the same session.
    """
    first_release, _ = _seed_regeneration_release(tmp_path / "first")
    second_release, _ = _seed_regeneration_release(tmp_path / "second")
    workspace = tmp_path / "workspace"

    first = open_study(
        "etfs",
        execution_tier=ExecutionTier.PREVIEW,
        workspace=workspace,
        release_root=first_release,
    )
    first_labels = (first.root / "labels").resolve(strict=True)

    second = open_study(
        "etfs",
        execution_tier=ExecutionTier.PREVIEW,
        workspace=workspace,
        release_root=second_release,
    )
    second_labels = (second.root / "labels").resolve(strict=True)

    assert first_labels != second_labels, "fixture must give the two studies different inputs"

    link = workspace / ".preview" / "etfs" / "labels"
    assert link.is_symlink()
    assert link.resolve(strict=True) == second_labels


class TestTheSingleRootReadOnlyForm:
    """`Study.at`, the form an analysis notebook uses because it must not activate.

    A notebook that only reads has no writes to place, and every other way into a `Study` ends
    in `activate()`: it rewrites `ML4T_OUTPUT_DIR` for the whole process and clears the caches
    keyed on it, so every later `get_case_study_dir` answers for a different directory than the
    one the notebook resolved. On the preview tier that directory is `.preview/<case>`, whose
    registry `activate()` creates empty - which is why the failure this form prevents is not a
    crash but a comparison that reports on nothing and calls it success.
    """

    def test_it_does_not_move_the_active_output_root(self, tmp_path: Path) -> None:
        """The property the notebooks depend on, asserted in both directions."""
        release = _seed_release(tmp_path)
        case_dir = release / "case_studies" / "etfs"

        os.environ["ML4T_OUTPUT_DIR"] = str(tmp_path / "chosen")
        Study.at(case_dir, case_study="etfs")
        assert os.environ["ML4T_OUTPUT_DIR"] == str(tmp_path / "chosen")

        os.environ.pop("ML4T_OUTPUT_DIR", None)
        Study.at(case_dir, case_study="etfs")
        assert "ML4T_OUTPUT_DIR" not in os.environ

    def test_activating_it_points_the_path_helpers_at_its_own_root(self, tmp_path: Path) -> None:
        """`Study.at` never activates, but the things it hands out do.

        `LabelCatalog.get` calls `study.activate(tier)` before resolving the artifact, so a
        read-only study does reach `activate()`. That branch used to pop `ML4T_OUTPUT_DIR`,
        and `get_case_study_dir` falls back to the repo's own `case_studies/` when the
        variable is absent - so every later lookup answered for the repo instead of the
        directory the notebook resolved, for the rest of the process. Under a redirected
        output root the repo holds none of these artifacts and the failure reads as a
        missing label rather than as a moved root.
        """
        release = _seed_release(tmp_path)
        case_dir = release / "case_studies" / "etfs"
        os.environ["ML4T_OUTPUT_DIR"] = str(tmp_path / "elsewhere")

        study = Study.at(case_dir, case_study="etfs")
        assert study.activate() == case_dir.resolve()

        from utils.paths import get_case_study_dir

        assert get_case_study_dir("etfs", create=False) == case_dir.resolve()

    def test_both_roots_are_the_directory_it_was_given(self, tmp_path: Path) -> None:
        """`OfficialPopulation.one` reads `root`; `Result.open` reads `release_case_root`.

        They have to be the same directory, or the notebook resolves a population from one
        registry and loads its members' artifacts from another.
        """
        release = _seed_release(tmp_path)
        case_dir = release / "case_studies" / "etfs"

        study = Study.at(case_dir, case_study="etfs")

        assert study.root == case_dir.resolve()
        assert study.release_case_root == case_dir.resolve()
        assert study.read_only
        assert study.output_root is None

    def test_it_reads_the_root_it_is_given_and_not_the_repo_checkout(self, tmp_path: Path) -> None:
        """`release_root` still answers for the repo, so provenance lookups keep working.

        The directory a fixture or an output tree hands over is not `<repo>/case_studies/<name>`
        and cannot be derived from `release_root`, which is the whole reason this form carries
        the case directory explicitly rather than deriving it.
        """
        release = _seed_release(tmp_path)
        case_dir = release / "case_studies" / "etfs"

        study = Study.at(case_dir, case_study="etfs")

        assert study.release_case_root != study.release_root / "case_studies" / "etfs"
        assert study.release_case_root == case_dir.resolve()

    def test_the_case_study_name_defaults_to_the_directory_name(self, tmp_path: Path) -> None:
        release = _seed_release(tmp_path)
        case_dir = release / "case_studies" / "etfs"

        assert Study.at(case_dir).case_study == "etfs"
