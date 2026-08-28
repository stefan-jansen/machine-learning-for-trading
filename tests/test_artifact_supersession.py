"""One digest per fold, and the record that says an artifact extended another rather than replacing it.

A stage-04 temporal artifact is pinned into every training identity fitted on it by
whole-file sha256, so appending the holdout fold a holdout retrain needs moves the pin over
folds whose every value is unchanged. Whether a new file extended the old one or rewrote it
is exactly what a whole-file digest cannot say, and it can only be established while both
files are still on disk. These tests fix what the answer depends on.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from case_studies.utils.artifact_digest import (
    digest_sidecar,
    fold_digests,
    read_digest,
    value_digest,
    write_artifact,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDER = REPO_ROOT / "scripts" / "record_artifact_supersession.py"


def _frame(folds: range, *, bump: int | None = None) -> pl.DataFrame:
    """A small fold-scoped frame; ``bump`` perturbs one fold's values and no other's."""
    return pl.DataFrame(
        {
            "fold": [f for f in folds for _ in range(3)],
            "symbol": [s for _ in folds for s in ("AAA", "BBB", "CCC")],
            "feature": [
                float(f * 10 + i) + (100.0 if bump == f else 0.0) for f in folds for i in range(3)
            ],
        }
    )


def _write(frame: pl.DataFrame, path: Path) -> Path:
    write_artifact(frame, path, keys=["fold", "symbol"], written_by="test", fold_column="fold")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(old: Path, new: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RECORDER), "--superseded", str(old), "--current", str(new), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


# --------------------------------------------------------------------------- fold digests


def test_fold_digest_moves_only_for_the_fold_whose_values_moved():
    before = fold_digests(_frame(range(3)))
    after = fold_digests(_frame(range(3), bump=1))
    assert set(before) == set(after) == {"0", "1", "2"}
    assert before["1"] != after["1"]
    assert before["0"] == after["0"]
    assert before["2"] == after["2"]


def test_fold_digest_is_invariant_to_row_order():
    frame = _frame(range(2))
    assert fold_digests(frame) == fold_digests(frame.reverse())


def test_appending_a_fold_leaves_every_earlier_fold_digest_alone():
    two = fold_digests(_frame(range(2)))
    three = fold_digests(_frame(range(3)))
    assert {fold: three[fold] for fold in two} == two
    assert set(three) - set(two) == {"2"}


def test_a_single_fold_frame_digests_to_the_frame_digest():
    """So the per-fold and whole-frame definitions cannot drift apart."""
    frame = _frame(range(1))
    assert fold_digests(frame) == {"0": value_digest(frame)}


def test_the_sidecar_records_fold_digests_only_when_a_producer_asks():
    frame = _frame(range(2))
    assert "fold_digests" not in digest_sidecar(frame, keys=["fold"], written_by="test")
    asked = digest_sidecar(frame, keys=["fold"], written_by="test", fold_column="fold")
    assert asked["fold_digests"] == fold_digests(frame)


def test_fold_digest_refuses_a_column_the_frame_does_not_have():
    with pytest.raises(KeyError, match="missing_fold"):
        fold_digests(_frame(range(2)), fold_column="missing_fold")


# ------------------------------------------------------------------ recording a supersession


def test_an_appended_fold_is_recorded_as_a_supersession(tmp_path):
    old = _write(_frame(range(2)), tmp_path / "old.parquet")
    old_sha, old_folds = _sha256(old), fold_digests(pl.read_parquet(old))
    new = _write(_frame(range(3)), tmp_path / "new.parquet")

    result = _record(old, new, "--expect-sha256", old_sha)
    assert result.returncode == 0, result.stderr

    supersedes = read_digest(new)["supersedes"]
    assert supersedes == {"sha256": old_sha, "fold_digests": old_folds}
    assert "2" not in supersedes["fold_digests"]


def test_a_rewritten_fold_is_refused_and_nothing_is_written(tmp_path):
    """Gaining fold 2 while fold 1 changes is a replacement, and no lock may treat it as one."""
    old = _write(_frame(range(2)), tmp_path / "old.parquet")
    new = _write(_frame(range(3), bump=1), tmp_path / "new.parquet")
    before = read_digest(new)

    result = _record(old, new)
    assert result.returncode == 1
    assert "hold different values" in result.stdout
    assert read_digest(new) == before


def test_a_dropped_fold_is_refused(tmp_path):
    old = _write(_frame(range(3)), tmp_path / "old.parquet")
    new = _write(_frame(range(2)), tmp_path / "new.parquet")

    result = _record(old, new)
    assert result.returncode == 1
    assert "are in old.parquet and not in new.parquet" in result.stdout
    assert "supersedes" not in read_digest(new)


def test_the_expected_sha256_is_what_ties_the_record_to_one_pin(tmp_path):
    """Paste the digest from the lock, and a differently-vintaged file cannot be recorded."""
    old = _write(_frame(range(2)), tmp_path / "old.parquet")
    new = _write(_frame(range(3)), tmp_path / "new.parquet")

    result = _record(old, new, "--expect-sha256", "0" * 64)
    assert result.returncode == 1
    assert "not the artifact the lock pins" in result.stderr
    assert "supersedes" not in read_digest(new)


def test_a_dry_run_reports_without_writing(tmp_path):
    old = _write(_frame(range(2)), tmp_path / "old.parquet")
    new = _write(_frame(range(3)), tmp_path / "new.parquet")

    result = _record(old, new, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "added:      ['2']" in result.stdout
    assert "supersedes" not in read_digest(new)


def test_an_artifact_cannot_supersede_itself(tmp_path):
    only = _write(_frame(range(2)), tmp_path / "only.parquet")
    result = _record(only, only)
    assert result.returncode == 1
    assert "cannot supersede itself" in result.stderr


def test_the_recorded_block_survives_json_round_tripping(tmp_path):
    old = _write(_frame(range(2)), tmp_path / "old.parquet")
    new = _write(_frame(range(3)), tmp_path / "new.parquet")
    assert _record(old, new).returncode == 0
    sidecar = json.loads((tmp_path / "new.parquet.digest.json").read_text())
    assert sidecar["fold_digests"] == fold_digests(pl.read_parquet(new))
    assert set(sidecar["supersedes"]["fold_digests"]) == {"0", "1"}
