"""The audit must say which tree it walked, and refuse to walk the wrong one.

An uncommitted version of this audit reported `crypto_perps_funding/financial` missing for
144 fits while the file sat at the canonical path under its registered hash, untouched for
two months. A case study's `features/` is a symlink into the artifacts root, so an audit
that does not resolve it, or that runs where the link is absent, compares the registry
against a tree the fits never read. These tests pin the two behaviours that stop that:
resolution with an explicit root, and a partial scope that cannot read as a clean one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "artifact_audit", REPO_ROOT / "scripts" / "artifact_audit.py"
)
assert _spec and _spec.loader
audit_module = importlib.util.module_from_spec(_spec)
sys.modules["artifact_audit"] = audit_module
_spec.loader.exec_module(audit_module)


def write_parquet_like(path: Path, payload: bytes) -> str:
    """A file the audit will hash. Its bytes need not be parquet; only the suffix and the
    hash matter, and using real parquet would test polars rather than the audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def make_registry(db_path: Path, specs: list[dict]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE training_runs (training_hash TEXT, spec_json TEXT)")
    connection.executemany(
        "INSERT INTO training_runs VALUES (?, ?)",
        [(f"h{i:04d}", json.dumps(spec)) for i, spec in enumerate(specs)],
    )
    connection.commit()
    connection.close()


def spec_with(artifacts) -> dict:
    return {"computation": {"feature_artifacts": artifacts}}


class TestRegisteredArtifacts:
    def test_dict_shape(self):
        spec = json.dumps(spec_with({"financial": {"sha256": "abc", "size": 7}}))
        assert audit_module.registered_artifacts(spec) == [("financial", "abc", 7)]

    def test_list_shape_strips_the_sha256_prefix(self):
        spec = json.dumps(spec_with([{"role": "label", "sha256": "sha256:def"}]))
        assert audit_module.registered_artifacts(spec) == [("label", "def", None)]

    def test_a_spec_with_no_artifacts_yields_nothing(self):
        assert audit_module.registered_artifacts(json.dumps({"computation": {}})) == []
        assert audit_module.registered_artifacts(json.dumps({})) == []

    def test_an_entry_without_a_hash_is_skipped(self):
        spec = json.dumps(spec_with({"financial": {"size": 7}}))
        assert audit_module.registered_artifacts(spec) == []


class TestScope:
    """The failure that produced the false alarm, and the guard against it."""

    @pytest.fixture
    def tree(self, tmp_path, monkeypatch):
        artifacts = tmp_path / "artifacts"
        checkout = tmp_path / "checkout"
        (checkout / "cs").mkdir(parents=True)
        (artifacts / "cs" / "features").mkdir(parents=True)
        (artifacts / "cs" / "labels").mkdir(parents=True)
        (checkout / "cs" / "features").symlink_to(artifacts / "cs" / "features")
        (checkout / "cs" / "labels").symlink_to(artifacts / "cs" / "labels")
        monkeypatch.setattr(audit_module, "get_case_study_dir", lambda name: checkout / name)
        monkeypatch.setattr(audit_module, "CASE_STUDIES", ("cs",))
        return artifacts, checkout

    def test_walk_follows_the_symlink_and_finds_the_artifact(self, tree):
        artifacts, _ = tree
        sha = write_parquet_like(artifacts / "cs" / "features" / "financial.parquet", b"x")
        on_disk, scope, _absent = audit_module.walk("cs", (artifacts).resolve())
        assert sha in on_disk
        assert str((artifacts / "cs" / "features").resolve()) in " ".join(scope)

    def test_a_directory_outside_the_declared_root_is_a_hard_error(self, tmp_path, monkeypatch):
        """The exact mis-scope: the checkout holds its OWN features dir, not a link."""
        elsewhere = tmp_path / "throwaway"
        (elsewhere / "cs" / "features").mkdir(parents=True)
        write_parquet_like(elsewhere / "cs" / "features" / "financial.parquet", b"y")
        monkeypatch.setattr(audit_module, "get_case_study_dir", lambda name: elsewhere / name)
        with pytest.raises(SystemExit) as caught:
            audit_module.walk("cs", (tmp_path / "artifacts").resolve())
        assert "never read" in str(caught.value)

    def test_the_root_defaults_to_where_the_artifacts_live_not_the_checkout(self, tree):
        artifacts, checkout = tree
        write_parquet_like(artifacts / "cs" / "features" / "financial.parquet", b"z")
        root = audit_module.default_artifacts_root(("cs",))
        assert root == artifacts.resolve()
        assert root != (checkout).resolve()

    def test_a_present_artifact_is_not_reported_missing(self, tree, capsys):
        artifacts, checkout = tree
        sha = write_parquet_like(artifacts / "cs" / "features" / "financial.parquet", b"present")
        make_registry(
            checkout / "cs" / "run_log" / "registry.db",
            [spec_with({"financial": {"sha256": sha, "size": 7}})] * 3,
        )
        total, unaudited, unseen = audit_module.audit(("cs",), artifacts.resolve())
        assert (total, unaudited) == (0, [])
        assert "every named artifact present" in capsys.readouterr().out

    def test_an_absent_artifact_is_reported_with_its_fit_count(self, tree, capsys):
        artifacts, checkout = tree
        write_parquet_like(artifacts / "cs" / "features" / "other.parquet", b"other")
        make_registry(
            checkout / "cs" / "run_log" / "registry.db",
            [spec_with({"model_based": {"sha256": "deadbeef" * 8, "size": 11}})] * 31,
        )
        total, unaudited, unseen = audit_module.audit(("cs",), artifacts.resolve())
        assert (total, unaudited) == (31, [])
        assert "named by 31 fits" in capsys.readouterr().out


class TestPartialScopeCannotReadAsClean:
    def test_a_missing_registry_is_named_and_returned(self, tmp_path, monkeypatch, capsys):
        artifacts = tmp_path / "artifacts"
        checkout = tmp_path / "checkout"
        (artifacts / "cs" / "features").mkdir(parents=True)
        (checkout / "cs").mkdir(parents=True)
        (checkout / "cs" / "features").symlink_to(artifacts / "cs" / "features")
        monkeypatch.setattr(audit_module, "get_case_study_dir", lambda name: checkout / name)
        total, unaudited, unseen = audit_module.audit(("cs",), artifacts.resolve())
        assert (total, unaudited, unseen) == (0, ["cs"], [])
        assert "NOT AUDITED" in capsys.readouterr().out

    def test_main_exits_2_when_the_scope_was_partial(self, tmp_path, monkeypatch, capsys):
        """0 missing over an incomplete scope must NOT exit 0: that is the whole defect."""
        artifacts = tmp_path / "artifacts"
        checkout = tmp_path / "checkout"
        (artifacts / "cs" / "features").mkdir(parents=True)
        (checkout / "cs").mkdir(parents=True)
        (checkout / "cs" / "features").symlink_to(artifacts / "cs" / "features")
        monkeypatch.setattr(audit_module, "get_case_study_dir", lambda name: checkout / name)
        monkeypatch.setattr(audit_module, "CASE_STUDIES", ("cs",))
        monkeypatch.setattr(sys, "argv", ["artifact_audit.py"])
        assert audit_module.main() == 2
        assert "PARTIAL" in capsys.readouterr().out


class TestAnAbsentDirectoryIsNotACleanScope:
    """The half that was silent, and is the reason a wrong answer looked authoritative.

    An unaudited *case study* printed PARTIAL and named itself. An absent artifact
    *directory* inside an audited one printed a single ABSENT line in the middle of the
    scope block, hashed nothing, and then counted every fit in that case study as naming a
    missing artifact - finishing with a bare total and exit 1. Measured 2026-09-14 on a
    checkout whose `us_equities_panel` had `run_log` but no `features/`: 393 fits reported
    missing, while the named `financial.parquet` sat on disk at exactly the
    4,478,156,899 bytes the MISSING line quoted.
    """

    @pytest.fixture
    def blind_tree(self, tmp_path, monkeypatch):
        artifacts = tmp_path / "artifacts"
        checkout = tmp_path / "checkout"
        (artifacts / "cs" / "labels").mkdir(parents=True)
        (checkout / "cs").mkdir(parents=True)
        # labels/ is linked; features/ is not there at all, which is the normal state of a
        # worktree whose gitignored symlinks were never created.
        (checkout / "cs" / "labels").symlink_to(artifacts / "cs" / "labels")
        make_registry(
            checkout / "cs" / "run_log" / "registry.db",
            [spec_with({"financial": {"sha256": "deadbeef", "size": 4_478_156_899}})] * 393,
        )
        monkeypatch.setattr(audit_module, "get_case_study_dir", lambda name: checkout / name)
        monkeypatch.setattr(audit_module, "CASE_STUDIES", ("cs",))
        return artifacts, checkout

    def test_the_case_study_is_not_counted_at_all(self, blind_tree, capsys):
        artifacts, _ = blind_tree
        total, unaudited, unseen = audit_module.audit(("cs",), artifacts.resolve())
        assert total == 0, "a case study whose tree was not fully seen must contribute nothing"
        assert unaudited == []
        assert len(unseen) == 1 and unseen[0].startswith("cs/features")
        assert "393" not in capsys.readouterr().out

    def test_main_prints_no_total_and_exits_2(self, blind_tree, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["artifact_audit.py"])
        assert audit_module.main() == 2
        out = capsys.readouterr().out
        assert "PARTIAL" in out
        assert "cs/features" in out
        # The whole point: no number that could be read as the answer.
        assert "fits name an artifact that is not on disk" not in out
