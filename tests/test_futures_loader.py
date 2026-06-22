"""Unit tests for data/futures/loader.py — list_cme_products().

The loader binds ML4T_DATA_PATH at import time (`from utils import
ML4T_DATA_PATH`), so monkeypatching the env var is insufficient —
we patch the module symbol directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.exceptions import DataNotFoundError
from data.futures import loader as futures_loader


def _build_hive_fixture(root: Path, products: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for p in products:
        (root / f"product={p}").mkdir()


def _build_individual_fixture(root: Path, products: list[str], include_empty: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for p in products:
        pdir = root / p
        pdir.mkdir()
        (pdir / "data.parquet").write_bytes(b"fake-parquet")
    if include_empty:
        (root / "EMPTY_DIR").mkdir()  # no data.parquet — must be skipped


@pytest.fixture
def isolated_data_path(tmp_path, monkeypatch):
    """Redirect the loader's bound ML4T_DATA_PATH to a temp directory."""
    monkeypatch.setattr(futures_loader, "ML4T_DATA_PATH", tmp_path)
    return tmp_path


def test_list_cme_products_hourly_returns_sorted_unique(isolated_data_path) -> None:
    hive_root = isolated_data_path / "futures" / "market" / "continuous" / "hourly"
    _build_hive_fixture(hive_root, ["ES", "NQ", "CL", "GC", "6E"])

    products = futures_loader.list_cme_products()
    assert products == ["6E", "CL", "ES", "GC", "NQ"]


def test_list_cme_products_ignores_non_product_subdirs(isolated_data_path) -> None:
    hive_root = isolated_data_path / "futures" / "market" / "continuous" / "hourly"
    hive_root.mkdir(parents=True, exist_ok=True)
    (hive_root / "product=ES").mkdir()
    (hive_root / "_metadata").mkdir()
    (hive_root / "random_other_dir").mkdir()
    (hive_root / "readme.txt").write_text("x")

    assert futures_loader.list_cme_products() == ["ES"]


def test_list_cme_products_individual_requires_data_parquet(isolated_data_path) -> None:
    ind_root = isolated_data_path / "futures" / "market" / "individual"
    _build_individual_fixture(ind_root, ["ES", "CL"], include_empty=True)

    assert futures_loader.list_cme_products(frequency="individual") == ["CL", "ES"]


def test_list_cme_products_raises_when_hourly_root_missing(isolated_data_path) -> None:
    with pytest.raises(DataNotFoundError, match="CME Futures Hourly"):
        futures_loader.list_cme_products()


def test_list_cme_products_raises_when_individual_root_missing(isolated_data_path) -> None:
    with pytest.raises(DataNotFoundError, match="CME Futures Individual"):
        futures_loader.list_cme_products(frequency="individual")


def test_list_cme_products_rejects_unknown_frequency(isolated_data_path) -> None:
    with pytest.raises(ValueError, match="frequency must be"):
        futures_loader.list_cme_products(frequency="daily")


def test_list_cme_products_returns_empty_list_for_empty_hive(isolated_data_path) -> None:
    (isolated_data_path / "futures" / "market" / "continuous" / "hourly").mkdir(parents=True)
    assert futures_loader.list_cme_products() == []
