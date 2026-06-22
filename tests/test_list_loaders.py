"""Unit tests for list_etfs / list_crypto_perps / list_fx_pairs.

Parallel to test_futures_loader.py — each loader binds ML4T_DATA_PATH at
import time, so we monkeypatch the module symbol rather than the env var.

The helpers enumerate the symbol universe from each dataset's parquet
and are the canonical way to answer "what's available locally?" for
marketing / data-inventory / Ch2 EDA notebooks.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from data.crypto import loader as crypto_loader
from data.etfs import loader as etfs_loader
from data.exceptions import DataNotFoundError
from data.fx import loader as fx_loader


def _write_symbol_parquet(path: Path, symbols: list[str]) -> None:
    """Minimal parquet with a ``symbol`` column plus a dummy value column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": symbols, "close": [1.0] * len(symbols)}).write_parquet(path)


# -----------------------------------------------------------------------------
# list_etfs
# -----------------------------------------------------------------------------


@pytest.fixture
def etfs_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(etfs_loader, "ML4T_DATA_PATH", tmp_path)
    return tmp_path


def test_list_etfs_returns_sorted_unique_symbols(etfs_isolated) -> None:
    _write_symbol_parquet(
        etfs_isolated / "etfs" / "market" / "etf_universe.parquet",
        # deliberately unsorted with a duplicate
        ["SPY", "QQQ", "AGG", "SPY", "IWM"],
    )
    assert etfs_loader.list_etfs() == ["AGG", "IWM", "QQQ", "SPY"]


def test_list_etfs_raises_when_parquet_missing(etfs_isolated) -> None:
    with pytest.raises(DataNotFoundError, match="ETF Universe"):
        etfs_loader.list_etfs()


# -----------------------------------------------------------------------------
# list_crypto_perps
# -----------------------------------------------------------------------------


@pytest.fixture
def crypto_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(crypto_loader, "ML4T_DATA_PATH", tmp_path)
    return tmp_path


def test_list_crypto_perps_returns_sorted_unique_symbols(crypto_isolated) -> None:
    _write_symbol_parquet(
        crypto_isolated / "crypto" / "market" / "perps_1h.parquet",
        ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BTCUSDT"],
    )
    assert crypto_loader.list_crypto_perps() == ["ADAUSDT", "BTCUSDT", "ETHUSDT"]


def test_list_crypto_perps_raises_when_parquet_missing(crypto_isolated) -> None:
    with pytest.raises(DataNotFoundError, match="Crypto Perpetuals"):
        crypto_loader.list_crypto_perps()


# -----------------------------------------------------------------------------
# list_fx_pairs
# -----------------------------------------------------------------------------


@pytest.fixture
def fx_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(fx_loader, "ML4T_DATA_PATH", tmp_path)
    return tmp_path


def test_list_fx_pairs_default_probes_daily(fx_isolated) -> None:
    _write_symbol_parquet(
        fx_isolated / "fx" / "market" / "daily.parquet",
        ["EUR_USD", "GBP_USD", "USD_JPY"],
    )
    assert fx_loader.list_fx_pairs() == ["EUR_USD", "GBP_USD", "USD_JPY"]


def test_list_fx_pairs_4h_uses_separate_parquet(fx_isolated) -> None:
    _write_symbol_parquet(
        fx_isolated / "fx" / "market" / "4h.parquet",
        ["AUD_JPY", "AUD_NZD"],
    )
    assert fx_loader.list_fx_pairs(frequency="4h") == ["AUD_JPY", "AUD_NZD"]


def test_list_fx_pairs_raises_when_parquet_missing(fx_isolated) -> None:
    with pytest.raises(DataNotFoundError, match="FX Pairs"):
        fx_loader.list_fx_pairs()


def test_list_fx_pairs_rejects_unknown_frequency(fx_isolated) -> None:
    with pytest.raises(ValueError, match="frequency must be"):
        fx_loader.list_fx_pairs(frequency="hourly")


# -----------------------------------------------------------------------------
# Re-export from data/__init__.py
# -----------------------------------------------------------------------------


def test_list_helpers_are_exported_from_data_package() -> None:
    """All three list_*() helpers must be importable at ``from data import ...``
    so marketing/inventory consumers don't need to know submodule layout.
    """
    import data

    assert hasattr(data, "list_etfs")
    assert hasattr(data, "list_crypto_perps")
    assert hasattr(data, "list_fx_pairs")
    assert "list_etfs" in data.__all__
    assert "list_crypto_perps" in data.__all__
    assert "list_fx_pairs" in data.__all__
