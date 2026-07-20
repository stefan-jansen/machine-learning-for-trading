"""Regression tests for the CME futures downloader (``data/futures/market/download.py``).

Two classes of bug are pinned here, both reader-facing and both surfaced by a
student who hit ``422 symbology_invalid_request / None of the symbols could be
resolved`` on ``uv run python data/futures/market/download.py``:

1. **Config vs availability** — a product whose ``start`` predates its actual
   GLBX.MDP3 history makes the per-year cost estimator request continuous
   symbols over empty windows, which DataBento rejects with 422. RTY (E-mini
   Russell 2000) relisted on Globex 2017-07-10 (was on ICE before) and LBR
   (Lumber) launched 2022-08-08 (replaced the delisted Random Length Lumber
   LBS); neither has history back to the old 2011 start. These tests fail if a
   future edit walks either start date back before the contract existed.

2. **Download + update round-trip** — the raw-DBN → canonical-schema transform,
   the Hive partition write, and the incremental merge/dedup branch (re-running
   an existing partition must not double rows). Exercised against the *real*
   ``download_full_product`` with only DataBento's ``get_range`` stubbed, so it
   is deterministic and network-free (no API key, no credits).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

import data.download_all as da

DOWNLOAD_PY = Path(da.__file__).parent / "futures" / "market" / "download.py"
CONFIG_YAML = DOWNLOAD_PY.parent / "config.yaml"


def _load_download_module():
    """Load ``futures/market/download.py`` (not a package) as a module.

    Registered in ``sys.modules`` before execution so its ``@dataclass``
    definitions can resolve ``__module__`` on Python 3.14.
    """
    spec = importlib.util.spec_from_file_location("futures_market_download", DOWNLOAD_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dl = _load_download_module()


# ---------------------------------------------------------------------------
# 1. Config start dates must not predate the contract's GLBX history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("product", "earliest_allowed"),
    [
        ("RTY", "2017-07-01"),  # E-mini Russell 2000 relisted on Globex 2017-07-10
        ("LBR", "2022-08-01"),  # Lumber launched 2022-08-08 (replaced LBS)
    ],
)
def test_config_start_not_before_contract_existed(product, earliest_allowed):
    """RTY/LBR starts must stay at/after their CME launch, never back at 2011.

    A regression here re-introduces the 422 symbology error for readers.
    """
    config = dl.FuturesConfig.load(CONFIG_YAML)
    start = config.get_product_start(product)
    assert start >= earliest_allowed, (
        f"{product} start={start} predates its GLBX.MDP3 history (must be "
        f">= {earliest_allowed}); this re-creates the 422 symbology error."
    )


# ---------------------------------------------------------------------------
# 2. Download + update round-trip (get_range stubbed — network-free)
# ---------------------------------------------------------------------------


class _FakeData:
    """Stands in for a DataBento ``DBNStore``; only ``to_parquet`` is used."""

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def to_parquet(self, path):
        self._df.write_parquet(path)


class _FakeClient:
    """Records requests and returns synthetic RAW-DBN bars — no network/credits."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.timeseries = self

    def get_range(self, dataset, symbols, schema, start, end, stype_in):
        self.calls.append((tuple(symbols), start, end, stype_in))
        base = dt.datetime.fromisoformat(start + "T23:00:00")
        rows = []
        for sym in symbols:  # e.g. "RTY.v.0"
            for h in range(2):
                rows.append(
                    dict(
                        ts_event=base + dt.timedelta(hours=h),
                        symbol=sym,
                        open=100.0 + h,
                        high=101.0 + h,
                        low=99.0 + h,
                        close=100.5 + h,
                        volume=1000 + h,
                        rtype=34,
                        publisher_id=1,
                        instrument_id=42,
                    )
                )
        df = pl.DataFrame(rows).with_columns(pl.col("ts_event").dt.cast_time_unit("ns"))
        return _FakeData(df)


@pytest.fixture
def stub_databento(monkeypatch):
    """Patch ``databento.Historical`` so the real downloader uses the fake client."""
    import databento

    fake = _FakeClient()
    monkeypatch.setattr(databento, "Historical", lambda: fake)
    return fake


def test_new_download_writes_canonical_partition(tmp_path, stub_databento):
    """A fresh download parses raw DBN -> canonical schema and Hive-partitions it."""
    config = dl.FuturesConfig.load(CONFIG_YAML)
    rows, msg = dl.download_full_product("RTY", config, tmp_path, dry_run=False)

    # Requested the corrected start date, as a continuous query.
    assert stub_databento.calls[-1][1] == config.get_product_start("RTY") == "2017-07-01"
    assert stub_databento.calls[-1][3] == "continuous"

    # 3 tenors x 2 synthetic bars = 6 rows, written under year=2017.
    part = dl.get_partition_path(tmp_path, "RTY", 2017)
    assert part.exists(), f"expected partition at {part}"
    df = pl.read_parquet(part)
    assert rows == df.height == 6
    # Canonical schema: ts_event -> timestamp, tenor extracted from symbol, product tagged.
    assert {"timestamp", "open", "high", "low", "close", "volume", "tenor", "product"} <= set(
        df.columns
    )
    assert sorted(df["tenor"].unique().to_list()) == [0, 1, 2]
    assert df["product"].unique().to_list() == ["RTY"]


def test_update_merges_without_duplicating(tmp_path, stub_databento):
    """Re-running over an existing partition must merge/dedup, not double rows."""
    config = dl.FuturesConfig.load(CONFIG_YAML)
    part = dl.get_partition_path(tmp_path, "RTY", 2017)

    dl.download_full_product("RTY", config, tmp_path, dry_run=False)
    first = pl.read_parquet(part).height

    # Second identical run exercises the merge-with-existing branch.
    dl.download_full_product("RTY", config, tmp_path, dry_run=False)
    second = pl.read_parquet(part).height

    assert first == 6
    assert second == first, "incremental update duplicated rows instead of deduping"


def test_dry_run_writes_nothing(tmp_path, stub_databento):
    """``--dry-run`` must not touch the API or write partitions."""
    config = dl.FuturesConfig.load(CONFIG_YAML)
    rows, msg = dl.download_full_product("RTY", config, tmp_path, dry_run=True)
    assert rows == 0
    assert stub_databento.calls == []
    assert not dl.get_partition_path(tmp_path, "RTY", 2017).exists()
