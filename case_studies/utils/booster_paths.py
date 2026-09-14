"""Where a GBM training run's saved LightGBM boosters are on disk.

Path logic only, so it imports nothing beyond the standard library. That is the
point rather than tidiness: `case_studies/utils/model_analysis.py` and
`case_studies/utils/insight_chapter.py` both resolve these paths, and both import
lightgbm and torch at module scope for an OpenMP and CUDA load-order race that
has nothing to do with locating a directory. A test over the layout table had to
install gigabytes to reach three `joinpath` calls, and on `test-unit`, which
installs neither, it failed at collection.
"""

from __future__ import annotations

from pathlib import Path

# Ordered most recent first. The first is what the training stage writes today
# (`case_studies/utils/gbm.py:2334`); the other two are kept because run logs
# predating that move still carry them. Each entry is (first path segment under
# run_log, then the segments that follow the training hash).
BOOSTER_LAYOUTS: tuple[tuple[str, ...], ...] = (
    ("training", "models", "boosters"),
    ("training", "boosters"),
    ("models", "boosters"),
)


def booster_dir(case_dir: Path, training_hash: str) -> Path | None:
    """The first layout in `BOOSTER_LAYOUTS` that exists for this run, else None.

    Checked rather than declared, so a case study on an older layout is
    unaffected: a directory that is not there cannot match. The order decides
    only where both exist, which is what a migration looks like half way through.

    None means "this run saved no boosters", and the callers read it that way -
    as "this family emits no importances". It is also what a wrong path returns,
    which is why the table is tested: the loader checked only the two older
    layouts and so found nothing on `nasdaq100_microstructure`, where all 50 gbm
    runs carry the first, and `13_model_analysis` drew its feature figure from
    the correlation fallback while its prose described gain-based importance.
    """
    for parts in BOOSTER_LAYOUTS:
        candidate = case_dir.joinpath("run_log", parts[0], training_hash, *parts[1:])
        if candidate.is_dir():
            return candidate
    return None
