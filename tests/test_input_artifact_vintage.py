"""A training run may not join a population fitted on a different vintage of its inputs.

A training run pins each input artifact by whole-file sha256 and then fits on whatever is on
disk, and until this guard nothing compared the two. Regenerate a stage-03 or stage-04
artifact and the next run registers against a vintage no prior member of the population was
fitted under, silently, and the mixture is found later by comparing the registry to the disk
by hand - if at all.

Measured across the nine production registries on 2026-09-07, before the guard existed:
`fx_pairs` holds 138 training runs pinning one `model_based` sha, and **zero** of them match
the file on disk. That state is safe only for as long as nobody runs a modelling notebook in
that worktree, and nothing enforced it.

ml4t/agent-workspace#987.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from case_studies.utils.registry.registration import (
    declare_artifact_supersession,
    register_training_run,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _spec(*, model_based: str, config_name: str = "ridge_default", label: str = "fwd_ret_21d"):
    """A resolved training spec in the shape six of the seven producers build.

    ``config_name`` also varies a hashed model parameter: the training identity is the
    computation, and a second run distinguished only by its config name resolves to the same
    hash and is served from the registry rather than registered.
    """
    return {
        "identity_version": 2,
        "execution_tier": "canonical",
        "family": "linear",
        "label": label,
        "seed": 42,
        "config_name": config_name,
        "computation": {
            "feature_names": ["momentum"],
            "model": {"class": "Ridge", "config": config_name},
            "input_data_spec": {
                "schema_version": 1,
                "artifacts": {
                    "financial": {"sha256": "f" * 64, "size": 11},
                    "label": {"sha256": "1" * 64, "size": 12},
                    "model_based": {"sha256": model_based, "size": 13},
                },
                "fingerprint": model_based[:16],
            },
        },
    }


def _registered(case_dir: Path) -> int:
    with closing(sqlite3.connect(case_dir / "run_log" / "registry.db")) as db:
        return db.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]


def test_a_run_on_the_registered_vintage_registers(tmp_path: Path) -> None:
    """The control. Nothing about the guard blocks fitting on the artifact on disk."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    register_training_run(
        "etfs", _spec(model_based=SHA_A, config_name="ridge_wide"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 2


def test_a_run_on_a_regenerated_artifact_is_refused(tmp_path: Path) -> None:
    """The defect: the second run fits on a file the first was never fitted on."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)

    with pytest.raises(ValueError, match="model_based") as raised:
        register_training_run(
            "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
        )

    # The message has to name both shas, because the author's next step is deciding whether
    # the regeneration was deliberate - and that is answered by which file is on disk.
    assert SHA_A in str(raised.value)
    assert SHA_B in str(raised.value)
    assert _registered(tmp_path) == 1, "the refusal must leave no row behind"


def test_the_refusal_lands_before_anything_is_written(tmp_path: Path) -> None:
    """It refuses before the fit, so nothing about the new vintage reaches disk.

    `register_training_run` writes an immutable `spec.json` under the training directory
    before it inserts. A guard that ran after that would leave an orphaned artifact
    directory for a run that was refused, which the next registration then reads as a
    conflicting spec.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    before = sorted(p.name for p in (tmp_path / "run_log" / "training").iterdir())

    with pytest.raises(ValueError, match="model_based"):
        register_training_run(
            "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
        )

    assert sorted(p.name for p in (tmp_path / "run_log" / "training").iterdir()) == before


def test_a_declared_supersession_lets_the_new_vintage_register(tmp_path: Path) -> None:
    """Superseding an artifact on purpose is legitimate, so the refusal has an override."""
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    declare_artifact_supersession(
        "etfs", "model_based", sha256=SHA_B, supersedes_sha256=SHA_A, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 2


def test_a_declaration_naming_an_unregistered_sha_is_refused(tmp_path: Path) -> None:
    """A mistyped predecessor must not be recorded as though it had unblocked something.

    This is a function an author calls by hand with a hash copied out of an error message,
    which is the same reason `declare_causal_supersedes` validates its predecessor.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    with pytest.raises(ValueError, match="nothing to supersede"):
        declare_artifact_supersession(
            "etfs", "model_based", sha256=SHA_B, supersedes_sha256="c" * 64, case_dir=tmp_path
        )


def test_a_declaration_does_not_wave_through_a_third_vintage(tmp_path: Path) -> None:
    """The declaration names one sha, so it retires that one and nothing else.

    Without this the override would be a bypass with a hash attached: declare once and every
    later regeneration of the same artifact registers unremarked.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    declare_artifact_supersession(
        "etfs", "model_based", sha256=SHA_B, supersedes_sha256=SHA_A, case_dir=tmp_path
    )
    register_training_run(
        "etfs", _spec(model_based=SHA_B, config_name="ridge_wide"), case_dir=tmp_path
    )

    with pytest.raises(ValueError, match="model_based"):
        register_training_run(
            "etfs", _spec(model_based="c" * 64, config_name="ridge_deep"), case_dir=tmp_path
        )


def test_another_label_is_compared_against_its_own_runs(tmp_path: Path) -> None:
    """The comparison is per label, because `label` names a different file for each one.

    `labels/<label>.parquet` is registered under the artifact name `label`, so a comparison
    across labels would refuse the first run of every new label on the grounds that its own
    label artifact is not the previous label's.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)

    other = _spec(model_based=SHA_A, label="fwd_ret_5d")
    other["computation"]["input_data_spec"]["artifacts"]["label"] = {"sha256": "2" * 64, "size": 9}
    register_training_run("etfs", other, case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def test_a_run_pinning_no_artifacts_is_not_blocked(tmp_path: Path) -> None:
    """The latent adapter records a `files` list rather than `artifacts` (#891).

    Reaching this guard with nothing to compare has to be a weaker check for that family
    rather than a refusal it can never satisfy.
    """
    register_training_run("etfs", _spec(model_based=SHA_A), case_dir=tmp_path)
    latent = _spec(model_based=SHA_A, config_name="pca_5")
    latent["family"] = "latent_factors"
    latent["computation"]["input_data_spec"] = {
        "files": [{"role": "financial", "sha256": "f" * 64}]
    }
    register_training_run("etfs", latent, case_dir=tmp_path)
    assert _registered(tmp_path) == 2


def test_recording_a_supersession_declares_it_in_the_registry(tmp_path: Path) -> None:
    """One author action leaves both records, so the refusal names a command that finishes.

    `scripts/record_artifact_supersession.py` establishes that a new artifact extended the
    old one rather than replacing it, fold by fold, while both files are still on disk. That
    is exactly the case where a new training run should be allowed to join the population
    fitted on the old vintage, so the script now declares it here too.
    """
    import hashlib
    import subprocess
    import sys

    import polars as pl

    from case_studies.utils.artifact_digest import write_artifact

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _frame(folds: range) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "fold": [f for f in folds for _ in range(2)],
                "symbol": [s for _ in folds for s in ("AAA", "BBB")],
                "feature": [float(f * 10 + i) for f in folds for i in range(2)],
            }
        )

    features = tmp_path / "features"
    features.mkdir()
    old = features / "superseded.parquet"
    write_artifact(
        _frame(range(3)), old, keys=["fold", "symbol"], written_by="t", fold_column="fold"
    )
    new = features / "model_based.parquet"
    write_artifact(
        _frame(range(4)), new, keys=["fold", "symbol"], written_by="t", fold_column="fold"
    )

    register_training_run("etfs", _spec(model_based=_sha(old)), case_dir=tmp_path)

    repo_root = Path(__file__).resolve().parent.parent
    done = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "record_artifact_supersession.py"),
            "--superseded",
            str(old),
            "--current",
            str(new),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert done.returncode == 0, done.stderr
    assert "supersedes" in done.stdout, done.stdout

    # The point of the wiring: no second command between the recorder and a run that fits
    # on the file it just established as an extension.
    register_training_run(
        "etfs", _spec(model_based=_sha(new), config_name="ridge_wide"), case_dir=tmp_path
    )
    assert _registered(tmp_path) == 2
