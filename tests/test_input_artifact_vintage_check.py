"""The vintage pre-flight reports the state registration would refuse, and only that.

Both outcomes matter and for different reasons. A checker that never reports a conflict is
the state ml4t/agent-workspace#1123 describes - the run stopping is the only warning. One
that reports every second vintage is worse than useless on a repository where regenerating an
artifact is a normal, declared operation: it would be red on `us_equities_panel` today, where
`model_based` carries two vintages and a declaration retires the older.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.check_input_artifact_vintage import check_case_study, main

CASE_STUDY = "vintage_fixture"
LABEL = "fwd_ret_1d"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _registry(db_path: Path, *, artifact_shas: dict[str, str], supersessions=()) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE training_runs (training_hash TEXT, label TEXT, spec_json TEXT)")
    con.execute(
        "CREATE TABLE artifact_supersessions "
        "(artifact_name TEXT, sha256 TEXT, supersedes_sha256 TEXT)"
    )
    spec = {
        "label": LABEL,
        "computation": {
            "input_data_spec": {
                "artifacts": {
                    name: {"sha256": sha, "size": 1} for name, sha in artifact_shas.items()
                }
            }
        },
    }
    con.execute("INSERT INTO training_runs VALUES (?,?,?)", ("hash-1", LABEL, json.dumps(spec)))
    for name, sha, supersedes in supersessions:
        con.execute("INSERT INTO artifact_supersessions VALUES (?,?,?)", (name, sha, supersedes))
    con.commit()
    con.close()


@pytest.fixture
def artifacts_root(tmp_path: Path) -> Path:
    root = tmp_path / "case_studies"
    features = root / CASE_STUDY / "features"
    labels = root / CASE_STUDY / "labels"
    features.mkdir(parents=True)
    labels.mkdir(parents=True)
    (features / "financial.parquet").write_bytes(b"financial-v1")
    (features / "model_based.parquet").write_bytes(b"model-based-v2")
    (labels / f"{LABEL}.parquet").write_bytes(b"label-v1")
    return root


def _on_disk(root: Path) -> dict[str, str]:
    base = root / CASE_STUDY
    return {
        "financial": _sha256_of(base / "features" / "financial.parquet"),
        "model_based": _sha256_of(base / "features" / "model_based.parquet"),
        "label": _sha256_of(base / "labels" / f"{LABEL}.parquet"),
    }


def _statuses(findings) -> dict[str, str]:
    return {f.artifact: f.status for f in findings}


def test_a_registry_fitted_on_what_is_on_disk_reports_current(artifacts_root):
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(db, artifact_shas=_on_disk(artifacts_root))

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)

    assert _statuses(findings) == {
        "financial": "current",
        "model_based": "current",
        "label": "current",
    }
    assert not any(f.is_failure for f in findings)


def test_a_regenerated_artifact_with_no_declaration_is_reported(artifacts_root):
    """The `us_equities_panel/06_linear` state of 2026-09-10, before the declaration."""
    shas = _on_disk(artifacts_root) | {"model_based": "0" * 64}
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(db, artifact_shas=shas)

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)

    assert _statuses(findings)["model_based"] == "undeclared"
    failure = next(f for f in findings if f.is_failure)
    assert failure.pinned == ("0" * 64,)
    assert failure.on_disk == _on_disk(artifacts_root)["model_based"]
    assert "declare_artifact_supersession" in failure.detail
    assert main(["--case-study", CASE_STUDY, "--artifacts-root", str(artifacts_root)]) == 1


def test_a_declared_supersession_makes_the_second_vintage_legitimate(artifacts_root):
    disk = _on_disk(artifacts_root)
    retired = "0" * 64
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(
        db,
        artifact_shas=disk | {"model_based": retired},
        supersessions=[("model_based", disk["model_based"], retired)],
    )

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)

    # The run pins only the retired sha, so the disk vintage is not among the pinned set and
    # the declaration is the whole reason this is not a conflict.
    assert not any(f.is_failure for f in findings)
    assert main(["--case-study", CASE_STUDY, "--artifacts-root", str(artifacts_root)]) == 0


def test_an_artifact_the_registry_pins_and_disk_lacks_is_reported_not_ignored(artifacts_root):
    shas = _on_disk(artifacts_root) | {"model_based": "0" * 64}
    (artifacts_root / CASE_STUDY / "features" / "model_based.parquet").unlink()
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(db, artifact_shas=shas)

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)

    assert _statuses(findings)["model_based"] == "absent"


def test_an_artifact_name_that_cannot_be_located_is_reported_not_skipped(artifacts_root):
    """A checker that answers green for what it did not look at is this defect class."""
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(db, artifact_shas=_on_disk(artifacts_root) | {"latent_files": "0" * 64})

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)

    assert _statuses(findings)["latent_files"] == "unresolved"
    assert "unchecked" in next(f for f in findings if f.artifact == "latent_files").detail
