"""Alternative data loaders: cross-asset third-party datasets.

Narrowed scope — SEC filings (10-K/10-Q/8-K/XBRL) and 13F moved to
`data/equities/loader.py`; on-chain datasets moved to
`data/crypto/loader.py`; CFTC Commitment of Traders moved to
`data/futures/loader.py`. This module now hosts only datasets that
are genuinely cross-asset or not tied to a single asset class:

- news/       — Bloomberg archive, FNSPID financial headlines
- text/       — Financial Phrasebank and other sentiment/NLP corpora
"""

from typing import Literal

import polars as pl

from data.exceptions import DataNotFoundError
from utils import ML4T_DATA_PATH


def load_fnspid(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load FNSPID (Financial News and Stock Price Integration Dataset).

    Financial news headlines linked to stock tickers for text-to-market
    signal research. Dataset contains 15.7M news records for 4,775 S&P 500
    companies from 1999-2023.

    Args:
        symbols: Optional list of stock symbols to filter (e.g., ["AAPL", "MSFT"])
        start_date: Optional start date (YYYY-MM-DD format)
        end_date: Optional end date (YYYY-MM-DD format)

    Returns:
        DataFrame with columns including: ticker, date, title/headline, source

    Source:
        HuggingFace: Zihan1004/FNSPID
        GitHub: https://github.com/Zdong104/FNSPID_Financial_News_Dataset

    Coverage: 1999-2023, 4,775 S&P 500 companies, 15.7M news records
    """
    base_path = ML4T_DATA_PATH / "alternative" / "news" / "fnspid"

    if not base_path.exists() or not list(base_path.glob("fnspid*.parquet")):
        raise DataNotFoundError(
            dataset_name="FNSPID Financial News Dataset",
            path=base_path,
            download_script="data/alternative/news/fnspid_download.py",
            readme="data/alternative/news/README.md",
        )

    parquet_files = sorted(base_path.glob("fnspid*.parquet"))
    data = pl.read_parquet(parquet_files[-1])

    col_map = {}
    for col in data.columns:
        col_lower = col.lower()
        if "ticker" in col_lower or "symbol" in col_lower:
            col_map[col] = "ticker"
        elif col_lower == "date" or "time" in col_lower:
            col_map[col] = "timestamp"
    if col_map:
        data = data.rename(col_map)

    if symbols:
        ticker_col = "ticker" if "ticker" in data.columns else "symbol"
        if ticker_col in data.columns:
            data = data.filter(pl.col(ticker_col).is_in(symbols))

    if "timestamp" in data.columns:
        if start_date:
            try:
                data = data.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_date())
            except Exception:
                data = data.filter(pl.col("timestamp") >= start_date)
        if end_date:
            try:
                data = data.filter(pl.col("timestamp") <= pl.lit(end_date).str.to_date())
            except Exception:
                data = data.filter(pl.col("timestamp") <= end_date)

    return data


def load_bloomberg_news(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    """Load the Bloomberg news archive (~470k headlines + article bodies).

    Produced by ``data/alternative/news/bloomberg_download.py`` from the
    HuggingFace-mirrored archive. Used by Chapter 22 as a secondary news
    corpus for ESG retrieval experiments.

    Args:
        start_date: Optional ``date`` filter (YYYY-MM-DD).
        end_date: Optional ``date`` filter (YYYY-MM-DD).

    Returns:
        DataFrame with columns: headline, journalists, date, link, article.

    Note:
        Bloomberg owns the underlying text; the HuggingFace mirror is
        distributed for research use only. Do not redistribute commercially.
    """
    path = ML4T_DATA_PATH / "alternative" / "news" / "bloomberg" / "bloomberg_news.parquet"
    if not path.exists():
        raise DataNotFoundError(
            dataset_name="Bloomberg News Archive",
            path=path,
            download_script="data/alternative/news/bloomberg_download.py",
            readme="data/alternative/news/README.md",
        )

    data = pl.read_parquet(path)

    if start_date:
        data = data.filter(pl.col("date") >= pl.lit(start_date).str.to_datetime())
    if end_date:
        data = data.filter(pl.col("date") <= pl.lit(end_date).str.to_datetime())

    return data


def load_financial_phrasebank(
    agreement: Literal["all", "50", "66", "75", "100"] = "100",
) -> pl.DataFrame:
    """Load Financial Phrasebank sentiment dataset.

    Academic dataset of financial sentences labeled with sentiment (positive,
    negative, neutral) by human annotators. Standard benchmark for financial
    sentiment analysis.

    Args:
        agreement: Minimum annotator agreement level:
            - "100": All annotators agree (most reliable, ~2,264 sentences)
            - "75": 75%+ agreement (~3,453 sentences)
            - "66": 66%+ agreement (~4,217 sentences)
            - "50": 50%+ agreement (~4,846 sentences)
            - "all": Load all agreement levels combined

    Returns:
        DataFrame with columns: sentence, sentiment/label, agreement_level (if 'all')

    Source: Malo et al. (2014) "Good debt or bad debt: Detecting semantic
            orientations in economic texts"
    Coverage: ~4,800 sentences from financial news articles
    """
    base_path = ML4T_DATA_PATH / "alternative" / "text" / "financial_phrasebank"
    parquet_files = list(base_path.glob("*.parquet")) if base_path.exists() else []

    if not parquet_files:
        raise DataNotFoundError(
            dataset_name="Financial Phrasebank",
            path=base_path,
            readme="data/alternative/text/README.md",
            instructions=(
                "Download from HuggingFace (takala/financial_phrasebank) and save as\n"
                f"  {base_path}/sentences_{{agreement}}.parquet\n\n"
                "See data/alternative/text/README.md for the download script."
            ),
        )

    data = pl.read_parquet(parquet_files)

    if agreement != "all" and "agreement" in data.columns:
        min_agreement = int(agreement) / 100
        data = data.filter(pl.col("agreement") >= min_agreement)

    return data
