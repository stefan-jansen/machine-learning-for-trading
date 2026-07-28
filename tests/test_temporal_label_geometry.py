"""Tests for per-label fold geometry in temporal feature artifacts.

A per-fold temporal artifact (``features/model_based.parquet``) describes one
label's walk-forward geometry. Case studies configure variant labels with their
own ``label_buffer``, so those labels seal a different gap between ``train_end``
and ``val_start`` and their folds cover different windows. An artifact built for
the primary label alone leaves variant labels reading temporal features fit on
the wrong training window, and missing the tail of their validation window.

Covers:

1. ``configured_labels`` enumerates primary + variants, primary first.
2. ``modeling_fold_boundaries_by_label`` returns genuinely different boundaries
   for labels whose buffers differ.
3. ``load_modeling_dataset`` selects the rows matching the label it loads and
   raises an actionable error when the artifact has no rows for that label.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

PRIMARY = "fwd_ret_21d"
VARIANT = "fwd_ret_5d"


@pytest.fixture
def isolated_case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect get_case_study_dir to tmp_path and clear cv_window's caches."""
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    from case_studies.utils import cv_window

    def _clear() -> None:
        cv_window._fold_splits.cache_clear()
        cv_window._load_setup_yaml.cache_clear()
        cv_window._holdout_window.cache_clear()

    _clear()
    yield tmp_path
    _clear()


def _seed_case_study(cs_dir: Path) -> None:
    """A two-label case study: 21D primary, 5D variant."""
    (cs_dir / "config").mkdir(parents=True, exist_ok=True)
    setup = {
        "strategy_id": cs_dir.name,
        "labels": {
            "primary": PRIMARY,
            "buffer": "21D",
            "variants": [VARIANT],
            "variant_buffers": {VARIANT: "5D"},
        },
        "evaluation": {
            "n_splits": 2,
            "train_size": "1Y",
            "val_size": "6M",
            "holdout_start": "2023-01-01",
            "holdout_end": "2023-12-31",
            "calendar": "NYSE",
            "periods_per_year": 252,
        },
    }
    (cs_dir / "config" / "setup.yaml").write_text(yaml.safe_dump(setup))

    dates = pl.date_range(date(2020, 1, 1), date(2023, 12, 31), interval="1d", eager=True)
    labels_dir = cs_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for label in (PRIMARY, VARIANT):
        pl.DataFrame(
            {"timestamp": dates, "symbol": ["AAA"] * len(dates), label: [0.01] * len(dates)}
        ).write_parquet(labels_dir / f"{label}.parquet")


def test_configured_labels_lists_primary_first(isolated_case_study: Path) -> None:
    from utils.cv_splits import configured_labels

    cs = "test_cs_geometry_labels"
    _seed_case_study(isolated_case_study / cs)

    assert configured_labels(cs) == [PRIMARY, VARIANT]


def test_configured_labels_requires_a_primary(isolated_case_study: Path) -> None:
    from utils.cv_splits import configured_labels

    cs = "test_cs_geometry_no_primary"
    cs_dir = isolated_case_study / cs
    _seed_case_study(cs_dir)
    setup = yaml.safe_load((cs_dir / "config" / "setup.yaml").read_text())
    setup["labels"] = {}
    (cs_dir / "config" / "setup.yaml").write_text(yaml.safe_dump(setup))

    with pytest.raises(ValueError, match="No labels configured"):
        configured_labels(cs)


def test_variant_buffer_produces_different_fold_windows(isolated_case_study: Path) -> None:
    """A shorter buffer seals less, so the variant's train window runs longer."""
    from case_studies.utils.cv_window import modeling_fold_boundaries_by_label

    cs = "test_cs_geometry_windows"
    _seed_case_study(isolated_case_study / cs)

    by_label = modeling_fold_boundaries_by_label(cs)
    assert set(by_label) == {PRIMARY, VARIANT}

    primary_fold0 = by_label[PRIMARY][0]
    variant_fold0 = by_label[VARIANT][0]
    assert variant_fold0["train_end"] > primary_fold0["train_end"]


def test_load_modeling_dataset_selects_rows_for_its_label(
    isolated_case_study: Path,
) -> None:
    """The 5D label must get the 5D rows, not the 21D ones."""
    from utils.modeling import load_modeling_dataset

    cs = "test_cs_geometry_select"
    cs_dir = isolated_case_study / cs
    _seed_case_study(cs_dir)
    _seed_features_and_temporal(cs_dir, labels=(PRIMARY, VARIANT))

    for label, marker in ((PRIMARY, 1.0), (VARIANT, 2.0)):
        mds = load_modeling_dataset(cs, label)
        assert "cv_label" not in mds.dataset.columns
        assert mds.temporal_by_fold["marker"].unique().tolist() == [marker]


def test_load_modeling_dataset_raises_when_label_absent_from_artifact(
    isolated_case_study: Path,
) -> None:
    """An artifact built for the primary alone must not silently serve variants."""
    from utils.modeling import load_modeling_dataset

    cs = "test_cs_geometry_missing"
    cs_dir = isolated_case_study / cs
    _seed_case_study(cs_dir)
    _seed_features_and_temporal(cs_dir, labels=(PRIMARY,))

    with pytest.raises(ValueError, match=rf"carries no rows for label '{VARIANT}'"):
        load_modeling_dataset(cs, VARIANT)


def test_load_temporal_features_selects_rows_for_its_label(
    isolated_case_study: Path,
) -> None:
    """load_temporal_features must select the same rows load_modeling_dataset does."""
    from utils.modeling import load_temporal_features

    cs = "test_cs_geometry_temporal_features"
    cs_dir = isolated_case_study / cs
    _seed_case_study(cs_dir)
    _seed_features_and_temporal(cs_dir, labels=(PRIMARY, VARIANT))

    for label, marker in ((PRIMARY, 1.0), (VARIANT, 2.0)):
        temporal = load_temporal_features(cs, label)
        assert "cv_label" not in temporal.columns
        assert temporal["marker"].unique().to_list() == [marker]


def test_load_temporal_features_raises_when_label_absent_from_artifact(
    isolated_case_study: Path,
) -> None:
    from utils.modeling import load_temporal_features

    cs = "test_cs_geometry_temporal_features_missing"
    cs_dir = isolated_case_study / cs
    _seed_case_study(cs_dir)
    _seed_features_and_temporal(cs_dir, labels=(PRIMARY,))

    with pytest.raises(ValueError, match=rf"carries no rows for label '{VARIANT}'"):
        load_temporal_features(cs, VARIANT)


def test_load_temporal_features_passes_through_when_no_cv_label_column(
    isolated_case_study: Path,
) -> None:
    """A legacy artifact with no cv_label column is returned unfiltered."""
    from utils.modeling import load_temporal_features

    cs = "test_cs_geometry_temporal_features_legacy"
    cs_dir = isolated_case_study / cs
    _seed_case_study(cs_dir)

    dates = pl.date_range(date(2020, 1, 1), date(2023, 12, 31), interval="1d", eager=True)
    features_dir = cs_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"timestamp": dates, "symbol": ["AAA"] * len(dates), "fold": [0] * len(dates)}
    ).write_parquet(features_dir / "model_based.parquet")

    temporal = load_temporal_features(cs, PRIMARY)
    assert temporal.height == len(dates)


def _seed_features_and_temporal(cs_dir: Path, *, labels: tuple[str, ...]) -> None:
    """Write a financial-feature parquet plus a per-label temporal artifact.

    Each label's temporal rows carry a distinct ``marker`` value so a test can
    tell which label's rows were selected.
    """
    from case_studies.utils.cv_window import modeling_fold_boundaries_by_label

    dates = pl.date_range(date(2020, 1, 1), date(2023, 12, 31), interval="1d", eager=True)
    features_dir = cs_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"timestamp": dates, "symbol": ["AAA"] * len(dates), "feat": [0.5] * len(dates)}
    ).write_parquet(features_dir / "financial.parquet")

    by_label = modeling_fold_boundaries_by_label(cs_dir.name)
    frames = []
    for marker, label in enumerate(labels, start=1):
        for fold in by_label[label]:
            window = dates.filter(
                (dates >= fold["train_start"]) & (dates <= fold["val_end"]),
            )
            frames.append(
                pl.DataFrame(
                    {
                        "timestamp": window,
                        "symbol": ["AAA"] * len(window),
                        "fold": [fold["fold"]] * len(window),
                        "cv_label": [label] * len(window),
                        "marker": [float(marker)] * len(window),
                    }
                )
            )
    pl.concat(frames).write_parquet(features_dir / "model_based.parquet")
