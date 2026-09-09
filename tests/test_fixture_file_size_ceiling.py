"""No fixture file may come within the margin of GitHub's per-file limit.

`intermediates/nasdaq100_microstructure/features/financial.parquet` reached 98.8 MB
against a hard 100 MiB, and the way that surfaced was a rejected push in an unrelated
lane: the fixture repo uses no Git LFS, so the limit is enforced by the remote at the
moment someone tries to publish a regeneration (ml4t/agent-workspace#1019). Nothing in
either repo measured it.

The immediate cause there is fixed - `tests/overrides.yaml` set `MAX_SYMBOLS: 5` for
nasdaq's 01, 02, 04, 05 and 07 and not for `03_financial_features`, whose notebook
default applies no reduction, and the missing line takes that panel to 50.5 MB. What was
not fixed is that nothing checks. This does.

The ceiling is GitHub's limit less a margin, not the limit itself. A check that fired at
100 MiB would fire for the first time on the commit that is already unpublishable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: GitHub rejects a push carrying a file at or above this size, on every repository,
#: with no per-repository setting. `ml4t/third-edition-test-data` tracks its parquets
#: directly rather than through Git LFS, so this applies to every one of them.
GITHUB_FILE_LIMIT_BYTES = 100 * 1024 * 1024

#: What is left for one regeneration's growth. The binding file today is
#: `intermediates/etfs/features/financial.parquet` at 93,815,698 bytes - 89.5 MiB, 89.5%
#: of the limit - so the margin has to be small enough to admit it and large enough that
#: the failure arrives before the push does. Ten mebibytes is roughly one added feature
#: block on a panel of this width.
MARGIN_BYTES = 10 * 1024 * 1024

CEILING_BYTES = GITHUB_FILE_LIMIT_BYTES - MARGIN_BYTES


def test_no_fixture_file_is_within_the_margin_of_the_push_limit(intermediates_dir):
    """Measured over the fixture as committed, not over what a generation would write.

    A generation writes into whatever `--output` names, and the file that gets rejected
    is the one in the repo, so the repo is what this reads.
    """
    if intermediates_dir is None:
        pytest.skip("no test-data intermediates on this checkout")
    oversized = sorted(
        (path.stat().st_size, path)
        for path in intermediates_dir.rglob("*")
        if path.is_file() and path.stat().st_size > CEILING_BYTES
    )
    assert not oversized, (
        "fixture file(s) within "
        f"{MARGIN_BYTES / 1024 / 1024:.0f} MiB of GitHub's {GITHUB_FILE_LIMIT_BYTES / 1024 / 1024:.0f} MiB "
        "per-file limit, so the next regeneration's push is rejected:\n  "
        + "\n  ".join(
            f"{path.relative_to(intermediates_dir)} {size / 1024 / 1024:.1f} MiB"
            for size, path in oversized
        )
        + "\nThree ways out, and they are not equivalent: put `intermediates/**/*.parquet` "
        "behind Git LFS (fixes the ceiling and stops history growth, and every consumer - "
        "CI checkout, both workstations - then needs LFS); narrow this case study's fixture "
        "window (cheapest, and `evaluation.n_splits` has to still resolve to the same fold "
        "count); or cut its symbol count (cheapest of all, and it starves every reader the "
        "fixture feeds). ml4t/agent-workspace#1019."
    )


def test_the_margin_leaves_room_for_the_file_that_is_closest_to_it(intermediates_dir):
    """The ceiling is only a warning if something is measured against it.

    A margin set so wide that no file approaches it would pass forever and check nothing,
    which is the state this file exists to leave. Recorded so that a later change to
    `MARGIN_BYTES` has to face what it makes true.
    """
    if intermediates_dir is None:
        pytest.skip("no test-data intermediates on this checkout")
    largest = max(
        (path.stat().st_size for path in intermediates_dir.rglob("*") if path.is_file()),
        default=0,
    )
    assert largest > CEILING_BYTES / 2, (
        f"the largest fixture file is {largest / 1024 / 1024:.1f} MiB against a "
        f"{CEILING_BYTES / 1024 / 1024:.0f} MiB ceiling. Nothing is near it, so the ceiling "
        "is measuring nothing; either the fixture shrank a great deal or this file is "
        "reading the wrong tree."
    )
