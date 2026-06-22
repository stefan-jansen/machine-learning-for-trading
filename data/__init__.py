"""Data loading utilities for ML4T third edition.

This package provides canonical data loaders for all datasets used in the book.

Usage:
    from data import load_etfs
    etf_data = load_etfs()

Available loaders:

Reference data:
- load_sp500_index: S&P 500 index OHLCV (1980-present)

Downloaded data (requires ML4T_DATA_PATH):
- load_etfs: 100-ETF cross-asset case study
- load_crypto_premium: Crypto funding rate premium index
- load_crypto_perps: Crypto perpetual futures hourly OHLCV
- load_macro: FRED macroeconomic indicators
- load_us_equities: US equities (NASDAQ Data Link, 1962-2018)
- load_ff_factors: Fama-French factors
- load_aqr_factors: AQR research factors
- load_cme_futures: CME futures (frequency="daily" default, or "hourly")
- load_cot: CFTC Commitment of Traders weekly positioning per product
- load_fx_pairs: FX pairs (OANDA + Yahoo)
- load_firm_characteristics: Firm characteristics for ML asset pricing (Chen-Pelger-Zhu)
- load_nasdaq100_bars: AlgoSeek NASDAQ-100 bars (minute default, any frequency; optional quotes or full microstructure)
- load_sp500_daily_bars: AlgoSeek daily OHLCV (S&P 500)
- load_sp500_options_eda: AlgoSeek options chains for Chapter 2 / 8 demos
  (8 symbols × 2019-2020, full schema)
- load_sp500_options_straddles_raw: ATM-band raw chains for the
  sp500_options case study (2017-2021, lifecycle-preserving)
- load_sp500_options_surface: Daily IV surface summary (derived)
- load_sp500_options_straddles: Daily 30D ATM straddles (derived)
- load_nasdaq100_taq: AlgoSeek TAQ tick data (March 2020)
- load_mbo_data: DataBento MBO tick data (NVDA, 10 trading days in Nov 2024)
- load_nasdaq_itch: NASDAQ ITCH parsed messages
- load_fnspid: FNSPID financial news dataset (HuggingFace)
- load_financial_phrasebank: Financial Phrasebank sentiment dataset
- load_sec_filings: SEC 10-K / 10-Q / 8-K filings (canonical schema)
- iter_sec_filings: per-record generator over SEC filings (replaces resolve_sec_filings_dir)
- load_sp500_10q_mda: convenience wrapper for load_sec_filings("10-Q", "sp500")
- load_sec_xbrl_fundamentals: SEC XBRL fundamentals panel (Frames + Submissions APIs)
- load_institutional_holdings_13f: Institutional holdings (13F, per-cik)
- load_13f_bulk_holdings: Institutional holdings (13F, quarterly bulk universe)
- load_defillama_chain_tvl: DeFi Total Value Locked (DefiLlama)
- load_coingecko_ohlcv: Daily prices/volume for a coin (CoinGecko)
- load_13f_stock_features: Stock-level features from 13F data
- load_13f_edges: Institution-stock edge list (for graph construction)
- load_kalshi: Kalshi prediction market OHLCV (CFTC-regulated, free)
- load_polymarket: Polymarket prediction market OHLCV (crypto, free)


Exceptions:
    DataNotFoundError: Raised when required data is missing
    DownloadError: Raised when a download fails
    MissingDependencyError: Raised when a required package is not installed
"""

# Alternative (cross-asset third-party — news, text corpora)
from data.alternative.loader import (
    load_bloomberg_news,
    load_financial_phrasebank,
    load_fnspid,
)

# Crypto (market + on-chain)
from data.crypto.loader import (
    list_crypto_perps,
    load_coingecko_ohlcv,
    load_crypto_perps,
    load_crypto_premium,
    load_defillama_chain_tvl,
)

# Equities (market + fundamentals + positioning + firm_characteristics)
from data.equities.loader import (
    iter_sec_filings,
    load_13f_bulk_holdings,
    load_13f_edges,
    load_13f_stock_features,
    load_firm_characteristics,
    load_iex_hist,
    load_institutional_holdings_13f,
    load_mbo_data,
    load_nasdaq100_bars,
    load_nasdaq100_taq,
    load_nasdaq_itch,
    load_sec_filings,
    load_sec_xbrl_fundamentals,
    load_sp500_10q_mda,
    load_sp500_daily_bars,
    load_sp500_index,
    load_sp500_options,
    load_sp500_options_eda,
    load_sp500_options_straddles,
    load_sp500_options_straddles_raw,
    load_sp500_options_surface,
    load_us_equities,
)

# ETFs
from data.etfs.loader import list_etfs, load_etfs

# Exceptions
from data.exceptions import (
    DataNotFoundError,
    DownloadError,
    MissingDependencyError,
)

# Factors
from data.factors.loader import load_aqr_factors, load_ff_factors

# Futures
from data.futures.loader import list_cme_products, list_cot_products, load_cme_futures, load_cot

# FX
from data.fx.loader import list_fx_pairs, load_fx_pairs

# Macro
from data.macro.loader import load_macro, load_macro_metadata

# Prediction Markets
from data.prediction_markets.loader import load_kalshi, load_polymarket

__all__ = [
    # Reference data
    "load_sp500_index",
    # ETFs
    "list_etfs",
    "load_etfs",
    # Crypto
    "list_crypto_perps",
    "load_crypto_premium",
    "load_crypto_perps",
    # Macro
    "load_macro",
    "load_macro_metadata",
    # Equities
    "load_us_equities",
    "load_sp500_daily_bars",
    "load_sp500_options",
    "load_sp500_options_eda",
    "load_sp500_options_straddles_raw",
    "load_sp500_options_surface",
    "load_sp500_options_straddles",
    "load_firm_characteristics",
    # Microstructure
    "load_nasdaq100_bars",
    "load_nasdaq100_taq",
    "load_mbo_data",
    "load_nasdaq_itch",
    "load_iex_hist",
    # Factors
    "load_ff_factors",
    "load_aqr_factors",
    # Futures
    "list_cme_products",
    "load_cme_futures",
    "list_cot_products",
    "load_cot",
    # FX
    "list_fx_pairs",
    "load_fx_pairs",
    # Alternative / NLP
    "load_institutional_holdings_13f",
    "load_13f_bulk_holdings",
    "load_13f_stock_features",
    "load_13f_edges",
    "load_fnspid",
    "load_bloomberg_news",
    "load_sec_filings",
    "iter_sec_filings",
    "load_sec_xbrl_fundamentals",
    "load_sp500_10q_mda",
    "load_financial_phrasebank",
    "load_defillama_chain_tvl",
    "load_coingecko_ohlcv",
    # Prediction Markets
    "load_kalshi",
    "load_polymarket",
    # Exceptions
    "DataNotFoundError",
    "DownloadError",
    "MissingDependencyError",
]
