"""Tests for case_studies/utils/cv_window.py P2.6 fixes (#2471).

Covers:

1. ``_fold_splits`` raises ``ValueError`` with the actionable
   "Add buffer to labels.buffer..." hint when ``label_buffer`` is
   missing from setup.yaml — restores the loud-fail contract that
   matches ``utils.modeling.load_modeling_dataset``.
2. ``_fold_splits`` detects the time column from the parquet schema
   (``timestamp`` else ``date``), so legacy parquets that haven't
   migrated to the canonical ``timestamp`` name don't crash with
   ``ColumnNotFoundError``.
3. ``_fold_splits`` returns ``None`` when the label parquet doesn't
   exist (unchanged contract).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml


@pytest.fixture
def isolated_case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect get_case_study_dir to tmp_path via ML4T_OUTPUT_DIR.

    Also clears the _fold_splits / _load_setup_yaml / _holdout_window
    lru caches so tests don't leak case-study state across runs.
    """
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))

    from case_studies.utils import cv_window

    cv_window._fold_splits.cache_clear()
    cv_window._load_setup_yaml.cache_clear()
    cv_window._holdout_window.cache_clear()
    yield tmp_path
    cv_window._fold_splits.cache_clear()
    cv_window._load_setup_yaml.cache_clear()
    cv_window._holdout_window.cache_clear()


def _seed_setup_yaml(cs_dir: Path, *, with_buffer: bool, label: str) -> None:
    cs_dir.mkdir(parents=True, exist_ok=True)
    cfg = cs_dir / "config"
    cfg.mkdir(exist_ok=True)
    setup: dict = {
        "strategy_id": cs_dir.name,
        "labels": {"primary": label},
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
    if with_buffer:
        setup["labels"]["buffer"] = "21D"
    (cfg / "setup.yaml").write_text(yaml.safe_dump(setup))


def _seed_label_parquet(cs_dir: Path, *, label: str, date_col: str) -> None:
    """Write a minimal label parquet with the given time column name."""
    labels_dir = cs_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2023, 12, 31), interval="1d", eager=True)
    df = pl.DataFrame(
        {
            date_col: dates,
            "symbol": ["AAA"] * len(dates),
            label: [0.01] * len(dates),
        }
    )
    df.write_parquet(labels_dir / f"{label}.parquet")


def test_missing_label_buffer_raises_with_actionable_hint(
    isolated_case_study: Path,
) -> None:
    """Setup.yaml without labels.buffer must raise loudly."""
    from case_studies.utils.cv_window import _fold_splits

    cs = "test_cs_missing_buffer"
    cs_dir = isolated_case_study / cs
    _seed_setup_yaml(cs_dir, with_buffer=False, label="fwd_ret_21d")
    _seed_label_parquet(cs_dir, label="fwd_ret_21d", date_col="timestamp")

    with pytest.raises(ValueError, match=r"No explicit label buffer found for 'fwd_ret_21d'"):
        _fold_splits(cs, "fwd_ret_21d")


def test_missing_label_parquet_returns_none(isolated_case_study: Path) -> None:
    """No parquet means 'no folds derivable' — still a None return."""
    from case_studies.utils.cv_window import _fold_splits

    cs = "test_cs_no_parquet"
    cs_dir = isolated_case_study / cs
    _seed_setup_yaml(cs_dir, with_buffer=True, label="fwd_ret_21d")
    # NB: no parquet written

    assert _fold_splits(cs, "fwd_ret_21d") is None


def test_schema_detection_picks_timestamp_column(isolated_case_study: Path) -> None:
    """Canonical-schema parquet with 'timestamp' column resolves folds."""
    from case_studies.utils.cv_window import _fold_splits

    cs = "test_cs_ts"
    cs_dir = isolated_case_study / cs
    _seed_setup_yaml(cs_dir, with_buffer=True, label="fwd_ret_21d")
    _seed_label_parquet(cs_dir, label="fwd_ret_21d", date_col="timestamp")

    splits = _fold_splits(cs, "fwd_ret_21d")
    assert splits is not None
    assert len(splits) >= 1
    fold_id, val_start, val_end = splits[0]
    assert fold_id == 0
    assert isinstance(val_start, date) and isinstance(val_end, date)
    assert val_start <= val_end


def test_schema_detection_falls_back_to_date_column(
    isolated_case_study: Path,
) -> None:
    """Legacy 'date'-column parquet still works — no ColumnNotFoundError."""
    from case_studies.utils.cv_window import _fold_splits

    cs = "test_cs_date"
    cs_dir = isolated_case_study / cs
    _seed_setup_yaml(cs_dir, with_buffer=True, label="fwd_ret_21d")
    _seed_label_parquet(cs_dir, label="fwd_ret_21d", date_col="date")

    splits = _fold_splits(cs, "fwd_ret_21d")
    assert splits is not None
    assert len(splits) >= 1


def test_schema_without_timestamp_or_date_raises(
    isolated_case_study: Path,
) -> None:
    """A parquet with neither 'timestamp' nor 'date' must raise actionably."""
    from case_studies.utils.cv_window import _fold_splits

    cs = "test_cs_no_time_col"
    cs_dir = isolated_case_study / cs
    _seed_setup_yaml(cs_dir, with_buffer=True, label="fwd_ret_21d")
    # Write a parquet with neither column.
    labels_dir = cs_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": ["AAA"], "fwd_ret_21d": [0.01]}).write_parquet(
        labels_dir / "fwd_ret_21d.parquet"
    )

    with pytest.raises(ValueError, match=r"neither 'timestamp' nor 'date'"):
        _fold_splits(cs, "fwd_ret_21d")
