"""A label's identity is its values, with or without a digest sidecar beside it.

`LabelCatalog.get` digested the artifact's *bytes* when no sidecar was present and its
*values* when one was. Every production label carries a sidecar; the CI fixtures, written
before sidecars existed, do not - so the two answered differently for the same data, and
the sidecar-less answer additionally forked whenever a parquet default moved. A whole-file
digest of a container is not an identity of its contents.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from case_studies.research.labels import LabelCatalog
from case_studies.research.workspace import Study
from case_studies.utils.artifact_digest import value_digest

LABEL = "fwd_ret_21d"


def _case_dir(tmp_path: Path) -> Path:
    case_dir = tmp_path / "release" / "case_studies" / "etfs"
    (case_dir / "run_log").mkdir(parents=True)
    (case_dir / "run_log" / "registry.db").write_bytes(b"")
    (case_dir / "config").mkdir()
    (case_dir / "config" / "setup.yaml").write_text(
        "labels:\n"
        f"  primary: {LABEL}\n"
        "  buffer: 2D\n"
        f"  horizons: {{{LABEL}: 2D}}\n"
        "evaluation:\n"
        "  n_splits: 2\n"
        "  train_size: 4D\n"
        "  val_size: 2D\n"
        "  holdout_start: '2024-01-11'\n"
        "  holdout_end: '2024-01-12'\n"
        "  calendar: crypto\n"
    )
    (case_dir / "labels").mkdir()
    return case_dir


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"] * 50,
            "timestamp": list(range(100)),
            LABEL: [i / 7 for i in range(100)],
        }
    )


@pytest.fixture(autouse=True)
def _no_output_redirect(monkeypatch):
    monkeypatch.delenv("ML4T_OUTPUT_DIR", raising=False)
    yield
    os.environ.pop("ML4T_OUTPUT_DIR", None)


def _digest_for(case_dir: Path, **write_kwargs) -> str:
    _frame().write_parquet(case_dir / "labels" / f"{LABEL}.parquet", **write_kwargs)
    study = Study.at(case_dir, case_study="etfs")
    return LabelCatalog(study).get(LABEL).digest


def test_the_same_values_under_two_codecs_have_one_identity(tmp_path: Path) -> None:
    """Repeated writes are byte-stable, so only a settings change or an upgrade bites."""
    default = _digest_for(_case_dir(tmp_path / "a"))
    lz4 = _digest_for(_case_dir(tmp_path / "b"), compression="lz4")
    row_groups = _digest_for(_case_dir(tmp_path / "c"), row_group_size=10)

    assert default == lz4 == row_groups


def test_the_identity_is_the_value_digest_the_sidecar_would_have_carried(
    tmp_path: Path,
) -> None:
    assert _digest_for(_case_dir(tmp_path)) == value_digest(_frame())
