"""Where the GBM importance loaders look for LightGBM boosters.

Imports `case_studies.utils.booster_paths` rather than the two modules that
call it. Both of those import lightgbm and torch at module scope for a load-
order race, and `test-unit` installs neither, so importing either one here
failed the required job at collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from case_studies.utils.booster_paths import BOOSTER_LAYOUTS, booster_dir


@pytest.mark.parametrize("parts", BOOSTER_LAYOUTS)
def test_every_declared_layout_is_found(tmp_path: Path, parts: tuple[str, ...]) -> None:
    """Each layout in the table resolves, so adding one cannot silently do nothing."""
    t_hash = "abc123"
    target = tmp_path.joinpath("run_log", parts[0], t_hash, *parts[1:])
    target.mkdir(parents=True)
    (target / "fold_0.txt").write_text("")

    assert booster_dir(tmp_path, t_hash) == target


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

    assert booster_dir(tmp_path, t_hash) == current


def test_absent_boosters_return_none(tmp_path: Path) -> None:
    """The state the caller reads as "this family emits no importances".

    It is also the state a wrong path produces, which is why the layout table is
    checked above rather than trusted: on `nasdaq100_microstructure` the loader
    returned `None` for every one of 50 gbm runs that had saved its boosters, and
    `13_model_analysis` drew its feature figure from the correlation fallback while
    its prose described gain-based importance.
    """
    (tmp_path / "run_log" / "training" / "abc123").mkdir(parents=True)

    assert booster_dir(tmp_path, "abc123") is None


def test_recurrence_counts_folds_not_rows(capsys, tmp_path, monkeypatch) -> None:
    """A feature in one fold's top five, under many configs, is not persistent.

    `importance_df` carries one row per (config, fold, feature). Counting those rows
    let a fold's five highest raw rows be five configs agreeing on one feature, and
    with two folds the ">= 75% of folds" bar is 1.5 - so two configs inside a single
    fold cleared a threshold that is supposed to mean "in most folds".

    `one_fold_hog` here is top in fold 0 under all three configs and absent from fold
    1; `steady` is second in both folds. Only `steady` is persistent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import polars as pl

    from case_studies.utils.model_viz import plot_feature_importance_heatmap

    rows = []
    for config in ("a", "b", "c"):
        rows += [
            {
                "config_name": config,
                "fold_id": 0,
                "feature": "one_fold_hog",
                "importance_norm": 1.0,
            },
            {"config_name": config, "fold_id": 0, "feature": "steady", "importance_norm": 0.9},
            {"config_name": config, "fold_id": 0, "feature": "filler0", "importance_norm": 0.1},
            {"config_name": config, "fold_id": 1, "feature": "steady", "importance_norm": 0.9},
            {"config_name": config, "fold_id": 1, "feature": "filler1", "importance_norm": 0.1},
        ]

    plot_feature_importance_heatmap(pl.DataFrame(rows), top_n=5)
    printed = capsys.readouterr().out

    assert "Persistent features" in printed, printed
    assert "steady" in printed, printed
    assert "one_fold_hog" not in printed, printed
