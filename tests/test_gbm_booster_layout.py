"""Where `load_gbm_feature_importance` looks for LightGBM boosters."""

from __future__ import annotations

from pathlib import Path

import pytest

from case_studies.utils.model_analysis import _BOOSTER_LAYOUTS, _booster_dir


@pytest.mark.parametrize("parts", _BOOSTER_LAYOUTS)
def test_every_declared_layout_is_found(tmp_path: Path, parts: tuple[str, ...]) -> None:
    """Each layout in the table resolves, so adding one cannot silently do nothing."""
    t_hash = "abc123"
    target = tmp_path.joinpath("run_log", parts[0], t_hash, *parts[1:])
    target.mkdir(parents=True)
    (target / "fold_0.txt").write_text("")

    assert _booster_dir(tmp_path, t_hash) == target


def test_the_current_trainer_layout_wins_over_an_older_one(tmp_path: Path) -> None:
    """A case study holding both gets the one the trainer writes today.

    The order matters only where both exist, which is what a migration looks like
    half way through. `run_log/training/{hash}/models/boosters` is what
    `nasdaq100_microstructure` carries on all 50 of its gbm runs.
    """
    t_hash = "abc123"
    current = tmp_path / "run_log" / "training" / t_hash / "models" / "boosters"
    older = tmp_path / "run_log" / "training" / t_hash / "boosters"
    for d in (current, older):
        d.mkdir(parents=True)
        (d / "fold_0.txt").write_text("")

    assert _booster_dir(tmp_path, t_hash) == current


def test_absent_boosters_return_none(tmp_path: Path) -> None:
    """The state the caller reads as "this family emits no importances".

    It is also the state a wrong path produces, which is why the layout table is
    checked above rather than trusted: on `nasdaq100_microstructure` the loader
    returned `None` for every one of 50 gbm runs that had saved its boosters, and
    `13_model_analysis` drew its feature figure from the correlation fallback while
    its prose described gain-based importance.
    """
    (tmp_path / "run_log" / "training" / "abc123").mkdir(parents=True)

    assert _booster_dir(tmp_path, "abc123") is None
