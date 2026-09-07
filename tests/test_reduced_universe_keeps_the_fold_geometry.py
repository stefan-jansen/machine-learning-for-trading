"""A universe reduction must not move where the folds fall.

``load_modeling_dataset`` derives the walk-forward geometry from the label frame, and
``generate_cv_splits`` reads the unique timestamps of whatever frame it is handed. Since #780
pushed the ``max_symbols`` reduction into the scans, that frame was the reduced one, so a preview
got a different calendar from the canonical run it is supposed to be a subset of.

Nothing downstream follows the shift. ``canonical_window`` always reads the whole label parquet,
so ``load_backtest_prices_for`` windows prices on the canonical geometry while the preview's
predictions carry sessions outside it, and ``Strategy._decision_weights`` refuses the decision
artifact with "decision artifact contains keys outside the backtest price grid". Measured on
cme_futures on 2026-09-06: the full 30-product universe puts fold 0's validation window at
2019-01-03..2020-01-02, a five-product preview at 2018-12-31..2019-12-30, and `cs-cme_futures`
failed at `13_backtest` on the two sessions between them.

The reduction here is built so the dropped entity owns dates the kept ones do not, which is what
makes the two timelines differ. ``top_entities`` keeps the entities with the most rows, so the
short-history entity is the one that goes - and it is the one holding the tail of the calendar.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

CASE = "reduction_geometry_cs"


def _seed(tmp_path: Path) -> None:
    case_dir = tmp_path / CASE
    (case_dir / "config").mkdir(parents=True)
    (case_dir / "features").mkdir()
    (case_dir / "labels").mkdir()
    (case_dir / "config" / "setup.yaml").write_text(
        yaml.safe_dump(
            {
                "strategy_id": CASE,
                "labels": {"primary": "fwd_ret_1d", "buffer": "1D"},
                "evaluation": {
                    "n_splits": 2,
                    "train_size": "1Y",
                    "val_size": "6M",
                    "calendar": "NYSE",
                    "periods_per_year": 252,
                },
            }
        )
    )

    long_days = pl.date_range(date(2018, 1, 1), date(2021, 6, 30), interval="1d", eager=True)
    tail_days = pl.date_range(date(2021, 7, 1), date(2021, 12, 31), interval="1d", eager=True)

    rows: dict[str, list] = {"timestamp": [], "symbol": []}
    # Four long-history entities, kept by the reduction, stopping mid-2021.
    for symbol in ("AAA", "BBB", "CCC", "DDD"):
        rows["timestamp"].extend(long_days.to_list())
        rows["symbol"].extend([symbol] * len(long_days))
    # One short-history entity, dropped by the reduction, owning the last six months.
    rows["timestamp"].extend(tail_days.to_list())
    rows["symbol"].extend(["ZZZ"] * len(tail_days))

    frame = pl.DataFrame(rows)
    frame.with_columns(pl.lit(0.01).alias("fwd_ret_1d")).write_parquet(
        case_dir / "labels" / "fwd_ret_1d.parquet"
    )
    frame.with_columns(pl.lit(1.0).alias("feature")).write_parquet(
        case_dir / "features" / "financial.parquet"
    )


@pytest.fixture
def seeded_case_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ML4T_OUTPUT_DIR", str(tmp_path))
    _seed(tmp_path)

    import utils.modeling as modeling
    from case_studies.utils import cv_window

    monkeypatch.setattr(modeling, "load_feature_spec", lambda *_args: {})
    monkeypatch.setattr(modeling, "load_label_spec", lambda *_args: {})
    monkeypatch.setattr(
        modeling,
        "resolve_storage_path",
        lambda _case_id, _spec, fallback: tmp_path / CASE / fallback,
    )
    cv_window._fold_splits.cache_clear()
    cv_window._load_setup_yaml.cache_clear()
    yield tmp_path
    cv_window._fold_splits.cache_clear()
    cv_window._load_setup_yaml.cache_clear()


def _windows(splits: list[dict]) -> list[tuple]:
    return [(s["fold"], s["val_start"], s["val_end"]) for s in splits]


def test_the_dropped_entity_really_does_shorten_the_timeline(seeded_case_study: Path) -> None:
    """Without this the test below could pass on a fixture where the reduction changes nothing."""
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")
    reduced = load_modeling_dataset(CASE, "fwd_ret_1d", max_symbols=4)

    entity = full.entity_cols[0]
    assert full.dataset[entity].n_unique() == 5
    assert reduced.dataset[entity].n_unique() == 4
    assert reduced.dataset["timestamp"].max() < full.dataset["timestamp"].max()


def test_a_reduced_load_keeps_the_full_universe_fold_geometry(seeded_case_study: Path) -> None:
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")
    reduced = load_modeling_dataset(CASE, "fwd_ret_1d", max_symbols=4)

    assert _windows(reduced.splits) == _windows(full.splits)


def test_a_reduced_load_agrees_with_the_window_the_backtest_prices_on(
    seeded_case_study: Path,
) -> None:
    """canonical_window reads the label parquet whole; a preview has to land on the same dates."""
    from case_studies.utils.cv_window import canonical_window
    from utils.modeling import load_modeling_dataset

    reduced = load_modeling_dataset(CASE, "fwd_ret_1d", max_symbols=4)
    window = canonical_window(CASE, "fwd_ret_1d")

    assert window is not None
    starts = [s["val_start"].date() for s in reduced.splits]
    ends = [s["val_end"].date() for s in reduced.splits]
    assert (min(starts), max(ends)) == window
