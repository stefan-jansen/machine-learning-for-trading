"""A column request has to narrow the scans, not the frame the caller is handed.

Every ``load_modeling_dataset`` caller used to materialize the whole joined panel and
project afterwards, so a narrow consumer paid the full width. Measured on
``us_equities_panel`` with label ``fwd_ret_1d``: the join returns 74 columns and 5.88 GB,
the DML estimand reads 7 of them worth 0.47 GB, and the notebook peaked at 17.6 GiB.
``max_symbols`` cannot help - it narrows rows, and a caller that needs the whole universe
cannot use it at all.

The test that matters is the third one. A projection applied to the returned frame passes
"the right columns came back" and "an unknown name raises" exactly as well as a projection
pushed into the scan, and buys none of the memory. So the probe watches every
materialization the call makes and asserts none of them ever held a dropped column, with
the unprojected load as the control that says the probe can see one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

CASE = "column_projection_cs"
FEATURES = ["alpha", "beta", "gamma", "delta"]
TEMPORAL = ["latent_1", "latent_2"]


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

    days = pl.date_range(date(2018, 1, 1), date(2021, 6, 30), interval="1d", eager=True)
    rows: dict[str, list] = {"timestamp": [], "symbol": []}
    for symbol in ("AAA", "BBB", "CCC"):
        rows["timestamp"].extend(days.to_list())
        rows["symbol"].extend([symbol] * len(days))
    frame = pl.DataFrame(rows)

    # Distinct values per column, so a test that compares them cannot pass by their
    # happening to agree.
    frame.with_columns(
        [pl.lit(float(i + 1)).alias(name) for i, name in enumerate(FEATURES)]
    ).write_parquet(case_dir / "features" / "financial.parquet")
    frame.with_columns(
        [pl.lit(float(10 + i)).alias(name) for i, name in enumerate(TEMPORAL)]
    ).write_parquet(case_dir / "features" / "model_based.parquet")
    frame.with_columns(pl.lit(0.01).alias("fwd_ret_1d")).write_parquet(
        case_dir / "labels" / "fwd_ret_1d.parquet"
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


@pytest.fixture
def materialized_columns(monkeypatch: pytest.MonkeyPatch) -> list[set[str]]:
    """Every column set that reached a ``collect()`` during the call."""
    seen: list[set[str]] = []
    original = pl.LazyFrame.collect

    def recording_collect(self: pl.LazyFrame, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(set(self.collect_schema().names()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", recording_collect)
    return seen


def test_a_projected_load_returns_the_requested_columns_and_the_keys(
    seeded_case_study: Path,
) -> None:
    from utils.modeling import load_modeling_dataset

    mds = load_modeling_dataset(CASE, "fwd_ret_1d", columns=["beta", "latent_2"])

    assert set(mds.dataset.columns) == {"timestamp", "symbol", "beta", "latent_2", "fwd_ret_1d"}
    assert set(mds.feature_names) == {"beta", "latent_2"}


def test_a_projected_load_carries_the_same_values_as_the_full_one(
    seeded_case_study: Path,
) -> None:
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")
    projected = load_modeling_dataset(CASE, "fwd_ret_1d", columns=["beta", "latent_2"])

    keys = ["timestamp", "symbol"]
    kept = keys + ["beta", "latent_2", "fwd_ret_1d"]
    assert projected.dataset.sort(keys).equals(full.dataset.select(kept).sort(keys))


def test_no_materialization_ever_holds_a_dropped_column(
    seeded_case_study: Path, materialized_columns: list[set[str]]
) -> None:
    """The claim the parameter exists for: narrowed in the scan, not in the result."""
    from utils.modeling import load_modeling_dataset

    load_modeling_dataset(CASE, "fwd_ret_1d", columns=["beta", "latent_2"])

    dropped = {"alpha", "gamma", "delta", "latent_1"}
    held = [cols & dropped for cols in materialized_columns if cols & dropped]
    assert held == [], f"dropped columns were materialized: {held}"


def test_the_probe_can_see_a_dropped_column_when_nothing_is_projected(
    seeded_case_study: Path, materialized_columns: list[set[str]]
) -> None:
    """Control for the test above: without it, a probe that observes nothing also passes."""
    from utils.modeling import load_modeling_dataset

    load_modeling_dataset(CASE, "fwd_ret_1d")

    dropped = {"alpha", "gamma", "delta", "latent_1"}
    assert any(cols & dropped for cols in materialized_columns)


def test_a_column_no_artifact_carries_raises_and_names_it(seeded_case_study: Path) -> None:
    from utils.modeling import load_modeling_dataset

    with pytest.raises(ValueError, match="no artifact carries.*not_a_column"):
        load_modeling_dataset(CASE, "fwd_ret_1d", columns=["beta", "not_a_column"])


def test_passing_nothing_keeps_the_full_width(seeded_case_study: Path) -> None:
    from utils.modeling import load_modeling_dataset

    full = load_modeling_dataset(CASE, "fwd_ret_1d")

    assert set(FEATURES) | set(TEMPORAL) <= set(full.dataset.columns)
