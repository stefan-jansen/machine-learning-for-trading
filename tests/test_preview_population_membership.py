"""A preview run must not be asked whether the canonical registry publishes its own fits.

`Study.activate` sends every write into `.preview/<case>` and deliberately leaves `study.root`
on the released case directory, so `split_unpublished_members` asked the CANONICAL registry
about hashes that only exist in the workspace. The answer is empty by construction, and empty
is what six etfs notebooks - 13, 14, 15, 16, 17 and 20 - refused on when the smoke chain reached
them on 2026-09-05: 247 prediction sets registered in the preview workspace, none of them
listed by any released population, "nothing published to sweep".

It bites on the maintainer path rather than in CI. `open_study` returns an isolated study, whose
root IS the workspace, when the case study's generated directories are not symlinks - every CI
checkout and every clean clone. A worktree built by `new-worktree.sh --case-study` symlinks them,
so it takes the regeneration path, where root stays canonical. That is the path every smoke run
executes on.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

from case_studies.research.contracts import ExecutionTier
from case_studies.research.population import split_unpublished_members
from case_studies.research.workspace import Study
from case_studies.utils.registry.store import _open_registry

RELEASED = ["released_a", "released_b"]
FITTED_IN_PREVIEW = ["preview_x", "preview_y"]


def _publish(case_dir: Path, members: list[str]) -> None:
    """Give a registry one population generation listing ``members``."""
    _open_registry(case_dir).close()
    db = sqlite3.connect(case_dir / "run_log" / "registry.db")
    db.execute(
        "INSERT INTO official_populations "
        "(population_hash, name, member_kind, snapshot_json, supersedes_hash, created_at) "
        "VALUES (?,?,?,?,?,?)",
        ("gen1", "etfs:models", "prediction", "{}", None, "2026-09-05T00:00:00Z"),
    )
    db.executemany(
        "INSERT INTO official_population_members (population_hash, member_hash, ordinal) "
        "VALUES (?,?,?)",
        [("gen1", m, i) for i, m in enumerate(members)],
    )
    db.commit()
    db.close()


def _study(tmp_path: Path, tier: ExecutionTier) -> Study:
    """The shape `open_study` builds on the regeneration path: root canonical, writes elsewhere."""
    release_case = tmp_path / "release" / "etfs"
    (release_case / "run_log").mkdir(parents=True)
    _publish(release_case, RELEASED)

    output_root = tmp_path / "workspace"
    preview_case = output_root / ".preview" / "etfs"
    preview_case.mkdir(parents=True)
    _open_registry(preview_case).close()

    return Study(
        case_study="etfs",
        root=release_case,
        release_root=tmp_path / "release",
        output_root=output_root,
        read_only=False,
        manifest={"schema_version": 1, "case_study": "etfs"},
        execution_tier=tier,
    )


def test_a_preview_keeps_the_predictions_it_just_fitted(tmp_path: Path) -> None:
    """The preview registry declares no population, so membership cannot be asked and nothing
    is filtered. Before the fix this returned an empty `live` and the notebook refused."""
    study = _study(tmp_path, ExecutionTier.PREVIEW)
    index = pl.DataFrame({"prediction_hash": FITTED_IN_PREVIEW})

    split = split_unpublished_members(study, index)

    assert split.live["prediction_hash"].to_list() == FITTED_IN_PREVIEW
    assert split.retired.height == 0


def test_the_canonical_tier_still_filters_to_what_is_published(tmp_path: Path) -> None:
    """The negative control. If the fix had simply stopped asking, this would pass vacuously:
    the released registry publishes two hashes and must still exclude everything else."""
    study = _study(tmp_path, ExecutionTier.CANONICAL)
    index = pl.DataFrame({"prediction_hash": [*RELEASED, "never_published"]})

    split = split_unpublished_members(study, index)

    assert split.live["prediction_hash"].to_list() == RELEASED
    assert split.retired["prediction_hash"].to_list() == ["never_published"]
