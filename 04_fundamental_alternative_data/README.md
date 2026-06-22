# Chapter 4: Fundamental and Alternative Data

The chapter turns point-in-time correctness from a principle into an implementation discipline. It shows why restatements, amended filings, taxonomy drift, and corporate actions can silently leak future information into a backtest, and it gives readers the operational tools to prevent that leakage through bitemporal storage, as-of queries, and source-specific timestamp authority. This matters because a fundamentals pipeline is only as good as its historical eligibility logic.

## Learning Objectives

* Explain why point-in-time correctness and entity consistency are the core engineering constraints for fundamental and alternative data.
* Implement bitemporal storage and as-of query patterns for revision-prone financial datasets.
* Build a point-in-time corporate fundamentals pipeline from SEC EDGAR and XBRL filing histories.
* Design time-valid entity, security, and contract mapping workflows using deterministic, probabilistic, and embedding-based resolution methods with appropriate QA gates.
* Apply point-in-time alignment rules to macro, commodity, and on-chain datasets, including release timestamps, vintages, contract mapping, and finality policies.
* Evaluate alternative datasets for incremental signal, data quality, legal and compliance risk, and commercial or engineering feasibility.
* Extract, clean, and store SEC filing text as an auditable point-in-time corpus for downstream NLP feature engineering.

## Sections

### 4.1 The Point-in-Time Pipeline

This section turns point-in-time correctness from a principle into an implementation discipline. It shows why restatements, amended filings, taxonomy drift, and corporate actions can silently leak future information into a backtest, and it gives readers the operational tools to prevent that leakage through bitemporal storage, as-of queries, and source-specific timestamp authority. This matters because a fundamentals pipeline is only as good as its historical eligibility logic.

- [`01_academic_characteristics`](01_academic_characteristics.ipynb) — This notebook introduces the Chen-Pelger-Zhu (2020) academic dataset, which provides a standardized benchmark for comparing ML models in asset pricing. With ~1.2M stock-month observations and 46 firm characteristics, this anonymized dataset enables reproducible research without requiring WRDS access.

### 4.2 Entity Resolution and Mapping

This section explains why multi-source financial research fails if issuer, security, and contract identities are not resolved correctly over time. It moves from deterministic joins to fuzzy matching, embedding-based matching, and QA controls, while emphasizing that resolution is not just a name-matching problem but a time-valid mapping problem across layers of the capital structure. Readers should care because a single wrong join can contaminate an entire research pipeline.

- [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) — This notebook demonstrates EdgarTools, a high-level Python library for interactive SEC EDGAR analysis. EdgarTools excels at company exploration, financial statement extraction, and working with structured filing data like Form 4 and 13F.
- [`03_sec_form4_insider_transactions`](03_sec_form4_insider_transactions.ipynb) — This notebook demonstrates parsing and analyzing SEC Form 4 insider transaction filings. Form 4 reports must be filed within 2 business days of an insider trade, making them valuable for detecting informed trading activity.
- [`04_sec_xbrl_fundamentals`](04_sec_xbrl_fundamentals.ipynb) — This notebook fetches quarterly fundamental data from the SEC EDGAR XBRL API for use in downstream factor engineering (Chapter 6).
- [`05_entity_resolution`](05_entity_resolution.ipynb) — Entity resolution is the keystone problem in multi-source data integration. Before any data can be combined, we must correctly link disparate real-world names like "IBM Corp" and "International Business Machines" to the same unique security identifier.

### 4.3 Fundamentals Across the Asset-Class Spectrum

This section broadens the idea of fundamentals beyond equities and shows that the same PIT discipline applies to macro data, commodities, and crypto, even though release mechanics and tradable instruments differ. It gives readers a practical sense of how timestamp authority, revision histories, contract mapping, and chain finality vary by asset class. The payoff is a reusable framework for building time-consistent features across very different domains.

- [`06_fred_macro_eda`](06_fred_macro_eda.ipynb) — This notebook provides first contact with macroeconomic time series from the Federal Reserve Economic Data (FRED) database. Understanding mixed-frequency data handling is critical for building point-in-time correct features.
- [`07_macro_data_alignment`](07_macro_data_alignment.ipynb) — Macroeconomic data presents unique challenges for trading models: different release cadences (monthly CPI, weekly claims, quarterly GDP), revision histories, and the critical requirement of point-in-time correctness. This notebook demonstrates how to align multi-frequency macro data for daily trading models using pre-downloaded FRED data.
- [`08_futures_positioning`](08_futures_positioning.ipynb) — This notebook demonstrates how to access CFTC Commitment of Traders (COT) data for tracking institutional positioning in futures markets. COT reports provide weekly snapshots of trader positioning, offering valuable sentiment signals for futures trading strategies and contrarian indicators.

### 4.4 Alternative Data: From Evaluation to Integration

This section reframes alternative data as an acquisition and engineering decision, not a buzzword category. It gives a concrete due-diligence framework around incremental signal, data quality, legal risk, and operational cost, and makes clear that many datasets fail not because they are uninteresting but because they are not defensible, reproducible, or deployable. Readers should care because most alternative-data mistakes are expensive and predictable.

- [`09_onchain_fundamentals`](09_onchain_fundamentals.ipynb) — Digital assets provide unprecedented transparency: all transactions are public and verifiable. This "radical transparency" enables analysis of protocol metrics and ecosystem health that would be impossible in traditional markets.
- [`10_institutional_holdings_13f`](10_institutional_holdings_13f.ipynb) — This notebook demonstrates how to work with SEC Form 13F institutional holdings using the official SEC bulk data sets. Form 13F requires institutional investment managers with >$100M in qualifying securities to disclose their equity holdings quarterly - valuable for tracking "smart money" positioning.
- [`11_defi_tvl_evaluation`](11_defi_tvl_evaluation.ipynb) — This notebook demonstrates a rigorous alternative data evaluation framework using real data: DeFi Llama's Total Value Locked (TVL) metrics. Rather than theoretical checklists, we compute actual signal quality, assess real data gaps, and calculate whether this free dataset justifies integration into a trading pipeline.
- [`12_kalshi_prediction_markets`](12_kalshi_prediction_markets.ipynb) — Kalshi is the first CFTC-regulated prediction market in the US, offering binary contracts on economic, market, and policy events. This notebook loads real Kalshi OHLCV data and demonstrates how to build event probability indicators for ML feature engineering and regime detection.
- [`13_polymarket_prediction_markets`](13_polymarket_prediction_markets.ipynb) — Polymarket is the world's largest prediction market by trading volume, operating on the Polygon blockchain with USDC settlement. This notebook loads pre-downloaded Polymarket OHLCV data from the centralized data pipeline and compares it with the Kalshi data from the previous notebook to illustrate cross-platform differences in liquidity, pricing, and market structure.

### 4.5 Case Study: Text Data for NLP Features

This section provides a concrete pipeline for turning SEC filing text into a model-ready corpus. It focuses on document selection, section extraction, cleaning, and PIT-correct storage, deliberately stopping short of featurization so the engineering foundation is clear before later NLP chapters build on it. Its significance is that text features only become credible once the extraction and storage layer is auditable and time-correct.

- [`14_text_data_extraction`](14_text_data_extraction.ipynb) — Corporate filings contain valuable information locked in unstructured text. This notebook demonstrates how to extract and structure high-value text blocks (MD&A, Risk Factors) from 10-K and 10-Q filings, creating clean datasets ready for NLP analysis in later chapters.

## Running the Notebooks

```bash
# From the repository root
uv run python 04_fundamental_alternative_data/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "04_fundamental_alternative_data"
```

### Required environment variables

Some Chapter 4 notebooks hit external APIs and need credentials or
identification headers:

- `EDGAR_IDENTITY` — SEC EDGAR mandates a `User-Agent` of the form
  `"<Name> <email>"` (e.g. `"ML4T Research stefan@applied-ai.com"`).
  Required by `02_sec_filing_explorer`, `03_sec_form4_insider_transactions`,
  `04_sec_xbrl_fundamentals`, `10_institutional_holdings_13f`, and
  `14_text_data_extraction`.
- `FRED_API_KEY` — only needed for live FRED downloads; the in-repo
  parquet snapshots used by `06_fred_macro_eda` and
  `07_macro_data_alignment` do not require it at notebook-execution time.

All Chapter 4 notebooks complete in under 20 seconds with peak memory
under 4 GB; no long-running or high-memory callouts apply.
