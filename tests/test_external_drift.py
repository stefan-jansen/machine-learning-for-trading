"""Phase 3 — live external drift smoke (weekly, flake-tolerant, auto-issue).

One tiny **real** pull per FREE data source. Each test asserts the source is
reachable, returns non-empty data, and still carries the schema our download
scripts depend on. When an upstream provider moves, renames a file, or changes
its payload shape, the matching test fails and the weekly workflow opens a
deduped GitHub issue — this is the drift detector.

Design rules (mirror ``work/2026-07-06-public-test-suite/PLAN.md``):
- **Tiny props only** — 2 symbols / a short window / a single product-year.
  Never a full-universe or full-history pull.
- **No billed APIs.** Only free sources run. Key-gated free sources (FRED,
  OANDA) skip cleanly when their free key is absent; geo-restricted sources
  (Kalshi / Polymarket) skip when blocked rather than fail.
- **Provider-level, not script-level.** We call the same providers the
  ``data/**/download.py`` scripts use, so a provider/API change surfaces here
  without writing files or pulling gigabytes.

Marked ``@pytest.mark.drift`` and collected only by the weekly ``drift`` job —
never per-PR (they touch the network).

``DRIFT_SOURCES`` is the public contract that ``tests/test_download_coverage.py``
(Phase 4) ratchets against: every free dataset must keep a drift smoke here.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.drift

# The free data sources this module keeps a live drift smoke for. Phase 4's
# coverage ratchet asserts every free ``data/**/download.py`` maps to an entry
# here, so a newly shipped free dataset can't land without a drift test.
DRIFT_SOURCES: tuple[str, ...] = (
    "etf",  # data/etfs/market/download.py            (Yahoo Finance)
    "crypto",  # data/crypto/market/download.py          (Binance public)
    "fama_french",  # data/factors/ff_download.py             (Ken French library)
    "aqr",  # data/factors/aqr_download.py            (AQR research)
    "cot",  # data/futures/positioning/cot_download.py (CFTC)
    "fred",  # data/macro/download.py                   (FRED — free key)
    "firm_characteristics",  # data/equities/firm_characteristics/download.py (Google Drive)
    "fx",  # data/fx/market/download.py               (OANDA — free key)
    "prediction_markets",  # data/prediction_markets/download.py     (Kalshi + Polymarket)
)

# A short, recent window keeps every pull tiny and deterministic in size.
_START = "2024-01-02"
_END = "2024-01-10"


def _assert_schema(df, required: set[str], source: str) -> None:
    """A source is healthy only if it returns non-empty data with our columns."""
    import polars as pl

    assert isinstance(df, pl.DataFrame), f"{source}: expected a polars DataFrame"
    assert not df.is_empty(), f"{source}: reachable but returned zero rows (drift?)"
    missing = required - set(df.columns)
    assert not missing, f"{source}: missing expected column(s) {sorted(missing)} — schema drift"


# ---------------------------------------------------------------------------
# No-key free sources — always run in the weekly job.
# ---------------------------------------------------------------------------


def test_etf_yahoo_reachable():
    """ETF universe download source (Yahoo Finance) — canonical OHLCV schema."""
    from ml4t.data.providers.yahoo import YahooFinanceProvider

    provider = YahooFinanceProvider()
    try:
        df = provider.fetch_ohlcv("SPY", _START, _END, "daily")
    finally:
        provider.close()
    _assert_schema(df, {"timestamp", "symbol", "open", "high", "low", "close"}, "etf/yahoo")


def test_crypto_binance_reachable():
    """Crypto premium-index source (Binance public) — canonical premium schema."""
    from ml4t.data.providers.binance_public import BinancePublicProvider

    provider = BinancePublicProvider(market="futures")
    try:
        df = provider.fetch_premium_index(
            "BTCUSDT", start="2024-01-01", end="2024-01-05", interval="8h"
        )
    finally:
        provider.close()
    _assert_schema(df, {"timestamp", "symbol", "premium_index_close"}, "crypto/binance")


def test_fama_french_reachable():
    """Fama-French factor source (Ken French library) — ff3 factors present."""
    from ml4t.data.providers.fama_french import FamaFrenchProvider

    provider = FamaFrenchProvider()
    try:
        df = provider.fetch("ff3", frequency="monthly", start="2023-01-01", end="2023-06-30")
    finally:
        provider.close()
    # Ken French renames/reformats his CSV zips periodically; pin the columns
    # the book's factor pipeline reads.
    _assert_schema(df, {"timestamp", "Mkt-RF", "SMB", "HML", "RF"}, "fama_french")


def test_aqr_reachable(tmp_path):
    """AQR factor source — live download of the monthly QMJ workbook still parses.

    Unlike the other providers, ``AQRFactorProvider`` splits the live pull from
    the local read: ``download()`` fetches the Excel workbook from aqr.com, and
    ``__init__``/``fetch`` only read a pre-populated data directory. Mirror
    ``data/factors/aqr_download.py`` — a tiny one-dataset download into a temp
    dir, then read it back — so this smoke actually exercises AQR's website. A
    moved/renamed workbook or layout change surfaces here; a bare ``fetch`` would
    only read an ambient local cache and never touch the network.
    """
    from ml4t.data.providers.aqr import AQRFactorProvider

    # No date filter: the monthly QMJ panel is already small, and the provider's
    # date-string filter path is brittle. We only need reachability + that the
    # Excel workbook still parses to a timestamped frame.
    AQRFactorProvider.download(output_path=tmp_path, datasets=["qmj_factors"])
    provider = AQRFactorProvider(data_path=tmp_path)
    try:
        df = provider.fetch("qmj_factors")
    finally:
        provider.close()
    _assert_schema(df, {"timestamp"}, "aqr")
    assert df.width >= 2, "aqr: QMJ workbook parsed but carries no factor columns — layout drift"


def test_cot_cftc_reachable(tmp_path, monkeypatch):
    """CFTC Commitment-of-Traders source — one product-year panel."""
    from ml4t.data.cot import COTConfig, COTFetcher

    # cot_reports writes a scratch .txt into the CWD; keep it out of the repo.
    monkeypatch.chdir(tmp_path)
    fetcher = COTFetcher(COTConfig(products=["ES"], start_year=2024, end_year=2024))
    df = fetcher.fetch_product("ES")
    _assert_schema(df, {"report_date", "open_interest"}, "cot")


def test_firm_characteristics_gdrive_listing():
    """Firm-characteristics source (Google Drive folder) — listing still resolves.

    Lists the folder without downloading (``skip_download=True``), catching a
    moved/renamed folder or a gdown API change — the class of bug that broke the
    firm-char download — for a fraction of a second and zero of the ~1.5 GB.
    """
    import os

    import gdown

    from data.equities.firm_characteristics.download import GDRIVE_FOLDER_URL

    listing = gdown.download_folder(GDRIVE_FOLDER_URL, skip_download=True, quiet=True)
    assert listing, "firm_characteristics: Google Drive folder listing is empty (moved/renamed?)"
    # The raw folder holds the source files the downloader fetches, then extracts:
    # RetChar.csv (the ~1.1 GB characteristics table the converter reads) and
    # datasets.zip (the char/macro/RF numpy splits). A rename of either breaks the
    # download, so pin their presence by basename rather than the extracted layout.
    names = {os.path.basename(getattr(f, "path", str(f))) for f in listing}
    for required in ("RetChar.csv", "datasets.zip"):
        assert required in names, (
            f"firm_characteristics: '{required}' missing from Drive folder "
            f"(found {sorted(names)}) — upstream dataset renamed/moved"
        )


# ---------------------------------------------------------------------------
# Key-gated free sources — skip cleanly when the free key is absent (no charge).
# ---------------------------------------------------------------------------


def test_fred_reachable():
    """Treasury/macro source (FRED) — free API key required, skip if unset."""
    if not os.getenv("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set — free-key source, nothing to charge")

    from ml4t.data.providers.fred import FREDProvider

    provider = FREDProvider()
    try:
        # DGS10 = 10-Year Treasury constant maturity, a stable free series.
        df = provider.fetch_ohlcv("DGS10", _START, _END, "daily")
    finally:
        provider.close()
    _assert_schema(df, {"timestamp"}, "fred")


def test_fx_oanda_reachable():
    """FX source (OANDA) — free API key required, skip if unset."""
    if not os.getenv("OANDA_API_KEY"):
        pytest.skip("OANDA_API_KEY not set — free-key source, nothing to charge")

    from ml4t.data.providers.oanda import OandaProvider

    provider = OandaProvider(api_key=os.environ["OANDA_API_KEY"])
    try:
        df = provider.fetch_ohlcv("EUR_USD", _START, _END, "daily")
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    _assert_schema(df, {"timestamp", "open", "high", "low", "close"}, "fx/oanda")


# ---------------------------------------------------------------------------
# Geo-restricted free source — skip (not fail) when the region blocks access.
# ---------------------------------------------------------------------------


def test_prediction_markets_reachable():
    """Prediction-markets source (Kalshi) — geo-restricted, skip when blocked.

    Kalshi and Polymarket restrict API access by jurisdiction, so a listing can
    fail at the network layer from some regions even though the endpoints are
    up. This dataset is optional; a geo/network block skips rather than fails so
    the weekly job doesn't file spurious drift issues from a blocked runner.
    """
    try:
        from ml4t.data.providers.kalshi import KalshiProvider
    except ImportError:
        pytest.skip("Kalshi provider not installed")

    provider = KalshiProvider()
    try:
        markets = provider.list_markets(limit=1)
    except Exception as exc:  # network/geo block — optional dataset
        pytest.skip(f"Kalshi unreachable from this runner (likely geo-restricted): {exc}")
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    if not markets:
        pytest.skip("Kalshi reachable but returned no markets (geo-filtered)")
    assert isinstance(markets, list) and markets[0].get("ticker"), (
        "prediction_markets: Kalshi listing shape changed (no 'ticker') — schema drift"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "drift"]))
