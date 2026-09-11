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


def _registry(
    db_path: Path, *, artifact_shas: dict[str, str], supersessions=(), extra_runs=()
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE training_runs (training_hash TEXT, family TEXT, label TEXT, spec_json TEXT)"
    )
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
    con.execute(
        "INSERT INTO training_runs VALUES (?,?,?,?)", ("hash-1", "gbm", LABEL, json.dumps(spec))
    )
    for training_hash, family, run_spec in extra_runs:
        con.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?)",
            (training_hash, family, LABEL, json.dumps(run_spec)),
        )
    for name, sha, supersedes in supersessions:
        con.execute("INSERT INTO artifact_supersessions VALUES (?,?,?)", (name, sha, supersedes))
    con.commit()
    con.close()


# The two shapes the vintage rule does not read, copied from the live registries rather than
# invented: an etfs latent_factors run and an etfs deep_learning run, 2026-09-11.
LATENT_SPEC = {
    "label": LABEL,
    "computation": {
        "input_data_spec": {
            "files": [
                {"role": "financial", "sha256": "sha256:" + "a" * 64},
                {"role": "label", "sha256": "sha256:" + "b" * 64},
            ],
            "input_digest": "sha256:" + "c" * 64,
            "version": "v2",
        }
    },
}
DEEP_LEARNING_SPEC = {
    "label": LABEL,
    "computation": {
        "input_data_spec": {
            "input_data_spec": {"artifacts": {"financial": {"sha256": "d" * 64, "size": 1}}},
            "lookback": 60,
            "max_train_sequences": 0,
        }
    },
}


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
    # the declaration is the whole reason this is not a conflict. It is NOT "current": no run
    # was fitted on the file that is on disk, and saying so hid the replacement under --quiet.
    assert not any(f.is_failure for f in findings)
    assert _statuses(findings)["model_based"] == "accepted_replacement"
    detail = next(f for f in findings if f.artifact == "model_based").detail
    assert "no run for this label was fitted on the file now on disk" in detail
    assert main(["--case-study", CASE_STUDY, "--artifacts-root", str(artifacts_root)]) == 0


def test_two_pinned_vintages_are_only_superseded_when_a_declaration_retires_the_other(
    artifacts_root,
):
    """Two shas alone do not establish the retirement, and the rule stays silent either way."""
    disk = _on_disk(artifacts_root)
    other = "0" * 64
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    second = {
        "label": LABEL,
        "computation": {
            "input_data_spec": {"artifacts": {"model_based": {"sha256": other, "size": 1}}}
        },
    }
    _registry(db, artifact_shas=disk, extra_runs=[("hash-2", "gbm", second)])

    mixed = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)
    assert _statuses(mixed)["model_based"] == "mixed"
    assert not any(f.is_failure for f in mixed)

    _registry(
        db,
        artifact_shas=disk,
        extra_runs=[("hash-2", "gbm", second)],
        supersessions=[("model_based", disk["model_based"], other)],
    )
    retired = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)
    assert _statuses(retired)["model_based"] == "superseded"


def test_the_two_shapes_that_used_to_be_unchecked_are_now_checked(artifacts_root):
    """What ml4t/agent-workspace#1137 changed, from this end.

    These two specs used to come back `unchecked`, one per family, because
    `_input_artifact_shas` read `computation.input_data_spec.artifacts` and nothing else -
    so 119 registered runs across the nine live registries were vintage-checked by neither
    registration nor this pre-flight. The rule reads all three shapes now, so both specs
    produce ordinary per-artifact findings and neither family appears as unchecked.

    Asserting the findings rather than only the absence: a widening that made
    `_input_artifact_shas` return `{}` faster would also empty the unchecked list, and would
    look identical here.
    """
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(
        db,
        artifact_shas=_on_disk(artifacts_root),
        extra_runs=[
            ("hash-latent", "latent_factors", LATENT_SPEC),
            ("hash-dl", "deep_learning", DEEP_LEARNING_SPEC),
        ],
    )

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)

    assert not [f for f in findings if f.status == "unchecked"]
    # The latent spec pins `financial` at a...a and `label` at b...b, the sequence spec pins
    # `financial` at d...d, and none of those is what is on disk. Being reported at all is the
    # change; the `sha256:` prefix on the latent side being stripped is why its pins compare
    # against the same digests every other family records.
    pinned = {sha for f in findings for sha in f.pinned}
    assert {"a" * 64, "b" * 64, "d" * 64} <= pinned


def test_a_spec_carrying_none_of_the_three_shapes_is_still_reported(artifacts_root):
    """The unchecked category has to stay able to fire.

    With all three shapes read nothing in the live registries reaches it, and a category that
    cannot fire is the failure this whole check exists to end - it would report a registry of
    nothing but unreadable runs as entirely clean. So: a spec recording its inputs nowhere the
    rule looks is still counted and named.
    """
    db = artifacts_root / CASE_STUDY / "run_log" / "registry.db"
    _registry(
        db,
        artifact_shas=_on_disk(artifacts_root),
        extra_runs=[
            (
                "hash-opaque",
                "some_future_family",
                {"label": LABEL, "computation": {"input_data_spec": {"digest": "x" * 64}}},
            )
        ],
    )

    findings = check_case_study(CASE_STUDY, artifacts_root=artifacts_root)
    unchecked = {f.artifact: f.detail for f in findings if f.status == "unchecked"}

    assert set(unchecked) == {"some_future_family"}
    assert "no artifact shas recorded" in unchecked["some_future_family"]
    # Reported, not fatal: registration accepts such a run, and a pre-flight stricter than the
    # rule it previews would refuse a chain that would in fact register.
    assert not any(f.is_failure for f in findings)


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
