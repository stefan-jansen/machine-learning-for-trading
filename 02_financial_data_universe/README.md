# Chapter 2: The Financial Data Universe

The chapter gives readers the conceptual map they need before touching any dataset. Its key contribution is not just the market / fundamental / alternative taxonomy, but the claim that every dataset embeds definitions about timestamps, adjustments, identifiers, and revisions, and that these choices determine what the data actually means in research.

## Learning Objectives

* Distinguish among market, fundamental, and alternative data, and explain how dataset definitions shape what each source means in research and trading applications
* Compare the observability, conventions, and engineering constraints of major asset classes, and identify how market structure changes what can be measured and modeled
* Apply a financial data quality framework to diagnose common failure modes, especially point-in-time violations, survivorship bias, corporate action errors, and identifier mismatches
* Conduct vendor due diligence across data quality, legal and compliance, and technical and commercial dimensions
* Choose storage and query architectures that fit research and production needs, including when to use partitioned files, embedded analytical databases, or server-based systems

## Sections

### 2.1 A Modern Taxonomy of Financial Data

This section gives readers the conceptual map they need before touching any dataset. Its key contribution is not just the market / fundamental / alternative taxonomy, but the claim that every dataset embeds definitions about timestamps, adjustments, identifiers, and revisions, and that these choices determine what the data actually means in research.

### 2.2 The Asset-Class Market Data Landscape

This section broadens the discussion from data categories to the practical reality that "price," "liquidity," and even "the dataset" mean different things across equities, ETPs, futures, options, digital assets, FX, fixed income, swaps, and commodities. Its value is comparative: it helps readers understand why engineering choices are inseparable from market structure.

- [`01_us_equities_eda`](01_us_equities_eda.ipynb) — This notebook introduces the Wiki Prices dataset - a survivorship-bias-free collection of US equity prices. Understanding survivorship bias is critical for realistic backtesting.
- [`02_corporate_actions`](02_corporate_actions.ipynb) — This notebook demonstrates how stock splits and dividends break historical price series, and shows the industry-standard backward adjustment methodology used by major data vendors. Correctly adjusting for corporate actions is essential for any ML model using return-based features.
- [`03_etfs_eda`](03_etfs_eda.ipynb) — This notebook introduces the 50-ETF universe that serves as the foundation for the ETF Rotational Momentum case study throughout the book. We explore the schema, coverage, categories, and data quality characteristics.
- [`04_cme_futures_eda`](04_cme_futures_eda.ipynb) — This notebook introduces the CME futures dataset shipped with the book. It demonstrates the data structure, coverage, and key concepts for working with futures data.
- [`05_futures_session_aggregation`](05_futures_session_aggregation.ipynb) — This notebook converts hourly continuous futures data (stored in UTC) to session-aware daily bars. CME futures sessions end at 4:00 PM Central Time, so daily bars must respect this boundary—not midnight UTC.
- [`06_futures_continuous`](06_futures_continuous.ipynb) — This notebook tackles one of the most critical challenges in futures analysis: creating continuous price series from individual expiring contracts. We implement roll detection algorithms and adjustment methods (Panama, ratio) to eliminate artificial price gaps while preserving accurate return characteristics.
- [`07_sp500_options_eda`](07_sp500_options_eda.ipynb) — This notebook provides a comprehensive exploration of the AlgoSeek S&P 500 Options Analytics dataset. Options data is fundamentally different from spot market data—it contains forward-looking information about expected volatility, directional sentiment, and tail risk that isn't directly observable in underlying prices.
- [`08_options_greeks_computation`](08_options_greeks_computation.ipynb) — This notebook derives and implements the Black-Scholes option pricing framework from first principles. We compute implied volatility and all Greeks, then validate our calculations against the pre-computed values in the AlgoSeek options data.
- [`09_options_continuous`](09_options_continuous.ipynb) — Options are time-decaying instruments. Unlike equities or futures, an option's price reflects both the value of the underlying exposure and the remaining time to expiration.
- [`10_crypto_perps_eda`](10_crypto_perps_eda.ipynb) — This notebook introduces the cryptocurrency dataset from Binance Futures. We explore hourly OHLCV data and the Premium Index (perpetual futures vs spot spread) that forms the basis for the Crypto Premium Arbitrage case study.
- [`11_crypto_premium_analysis`](11_crypto_premium_analysis.ipynb) — This notebook demonstrates how to work with Binance perpetual futures premium index data - the foundation for funding rate arbitrage strategies. We load, explore, and analyze premium dynamics across major cryptocurrencies to identify potential arbitrage opportunities.
- [`12_fx_pairs_eda`](12_fx_pairs_eda.ipynb) — This notebook introduces the FX dataset from OANDA. FX markets are OTC with no centralized exchange, so prices aggregate from multiple liquidity providers.

### 2.3 Data Sourcing: The Due Diligence Framework

This is the chapter's risk-control core. It explains that many apparent research successes are manufactured by data defects, then organizes due diligence around general quality dimensions, finance-specific failure modes, vendor evaluation, and internal governance. The section turns abstract warnings into operational rules: point-in-time correctness, survivorship handling, corporate action methodology, identifier integrity, legal rights, and reproducible versioning.

- [`13_data_quality_framework`](13_data_quality_framework.ipynb) — The ml4t-data library provides purpose-built tools for financial data quality. This notebook demonstrates the complete data quality workflow: Uses us_equities data.
- [`14_point_in_time_validation`](14_point_in_time_validation.ipynb) — Point-in-time correctness is essential for valid backtesting. Using information that wasn't available at decision time creates lookahead bias - making backtests look better than they would perform in live trading.
- [`15_survivorship_bias_detection`](15_survivorship_bias_detection.ipynb) — Survivorship bias is arguably the most dangerous form of data contamination in quantitative finance. This notebook uses real historical data from the US equities dataset (US Equities, originally Quandl WIKI) to demonstrate, detect, and quantify survivorship bias.
- [`16_provider_comparison`](16_provider_comparison.ipynb) — ML4T Third Edition - Chapter 2: The Financial Data...

### 2.4 Data Storage

This section translates data discipline into infrastructure decisions. Rather than promoting a single stack, it explains how storage choice depends on access patterns, scale, concurrency, and operational maturity, and it benchmarks the trade-offs among file formats, embedded engines, and server databases. It gives readers a practical default architecture for modern research workflows while also teaching when more complex systems are justified.

- [`17_complete_pipeline`](17_complete_pipeline.ipynb) — This notebook demonstrates end-to-end data pipelines, bringing together concepts from this chapter: Uses crypto_perps, wiki_provider data.
- [`18_data_management`](18_data_management.ipynb) — Previous notebooks fetched and validated data. This notebook shows how to manage it at scale using ml4t-data's production features: Uses universe data.
- [`19_incremental_updates`](19_incremental_updates.ipynb) — The previous notebook introduced DataManager and HiveStorage. This notebook focuses on the update workflow — the core reason ml4t-data exists: Uses all, treasury_yields data.
- [`20_storage_benchmark_file`](20_storage_benchmark_file.ipynb) — Focus: Pure file format comparison (no query engines) Technologies: CSV, Parquet, Feather (Arrow IPC), HDF5 Operations: Write, Read (with forced materialization), Columnar...
- [`21_storage_benchmark_database`](21_storage_benchmark_database.ipynb) — > Docker required: This notebook depends on the benchmark environment and > database services.
- [`22_pandas_polars_benchmark`](22_pandas_polars_benchmark.ipynb) — DataFrame-engine comparison (pandas vs Polars) across read, filter, groupby, join, and lazy operations on synthetic financial data at S/M/L scales. Backs the in-memory engine-choice recommendation in §2.4.

## Running the Notebooks

```bash
# From the repository root
uv run python 02_financial_data_universe/<notebook>.py

# Test mode (reduced data via Papermill)
uv run pytest tests/test_notebooks.py -v -k "02_financial_data_universe"
```

### NB21 storage_benchmark_database prerequisites

`21_storage_benchmark_database` exercises 7 storage engines (DuckDB, SQLite, TimescaleDB, ClickHouse, QuestDB, PostgreSQL, Polars Parquet). Of these, **DuckDB and SQLite run locally** with no additional setup; the other five require the benchmark service stack to be running first:

```bash
docker compose --profile benchmark up -d
```

Without these services, NB21 silently falls back to the 2-engine subset (DuckDB + SQLite only) and the comparative server-database results in §2.4 will not be reproduced. Bring the stack down with `docker compose --profile benchmark down` when finished.

## References

- **William Beaver et al.** (2007). [Delisting returns and their effect on accounting-based market anomalies](https://doi.org/10.1016/j.jacceco.2006.12.002). *Journal of Accounting and Economics*.
- **Florian Berg et al.** (2022). [Aggregate Confusion: The Divergence of ESG Ratings*](https://doi.org/10.1093/rof/rfac033). *Review of Finance*.
- **Mark M. Carhart et al.** (2002). [Mutual Fund Survivorship](https://doi.org/10.1093/rfs/15.5.1439). *The Review of Financial Studies*.
- **Lin William Cong et al.** (2023). [Crypto Wash Trading](https://doi.org/10.1287/mnsc.2021.02709). *Management Science*.
- **David Easley et al.** (2021). [Microstructure in the Machine Age](https://doi.org/10.1093/rfs/hhaa078). *The Review of Financial Studies*.
- **B. Espen Eckbo and Markus Lithell** (2025). [Merger-Driven Listing Dynamics](https://doi.org/10.1017/S0022109023001394). *Journal of Financial and Quantitative Analysis*.
- **Gene Ekster and Petter N. Kolm** (2020). [Alternative Data in Investment Management: Usage, Challenges and Valuation](https://doi.org/10.2139/ssrn.3715828).
- **Edwin J. Elton et al.** (1996). [Survivor Bias and Mutual Fund Performance](https://doi.org/10.1093/rfs/9.4.1097). *The Review of Financial Studies*.
- **Kingsley Y L Fong et al.** (2017). [What Are the Best Liquidity Proxies for Global Research?*](https://doi.org/10.1093/rof/rfx003). *Review of Finance*.
- **Songrun He et al.** (2024). [Fundamentals of Perpetual Futures](https://doi.org/10.48550/arXiv.2212.06888).
- **Jacques Joubert et al.** (2024). [The Three Types of Backtests](https://doi.org/10.2139/ssrn.4897573).
- **Nina Karnaukh et al.** (2015). [Understanding FX Liquidity](https://doi.org/10.2139/ssrn.2329738).
- **Gueorgui S. Konstantinov** (2025). [On Systematic Currency Management](https://doi.org/10.3905/jpm.2025.1.724). *The Journal of Portfolio Management*.
- **John Lehoczky and Mark Schervish** (2018). [Overview and History of Statistics for Equity Markets](https://doi.org/10.1146/annurev-statistics-031017-100518). *Annual Review of Statistics and Its Application*.
- **Alex Lipton and Marcos Lopez de Prado** (2020). [Three Quant Lessons from COVID-19](https://doi.org/10.2139/ssrn.3580185).
- **Tim Loughran and Bill McDonald** (2020). [Textual Analysis in Finance](https://doi.org/10.1146/annurev-financial-012820-032249). *Annual Review of Financial Economics*.
- **Yin Luo et al.** (2014). Seven Sins of Quantitative Investing.
- **Igor Makarov and Antoinette Schoar** (2020). [Trading and arbitrage in cryptocurrency markets](https://doi.org/10.1016/j.jfineco.2019.07.001). *Journal of Financial Economics*.
- **Hunter Ng et al.** (2025). [Price Discovery and Trading in Prediction Markets](https://doi.org/10.2139/ssrn.5331995).
- **Maureen O’Hara** (2015). [High frequency market microstructure](https://doi.org/10.1016/j.jfineco.2015.01.003). *Journal of Financial Economics*.
- **Marcos Lopez de Prado** (2018). Advances in Financial Machine Learning. *John Wiley & Sons*.
- **SEC** (2020). [Staff Report on Algorithmic Trading in U.S. Capital Markets](https://www.sec.gov/file/staff-report-algorithmic-trading-us-capital-markets).
- **Tyler Shumway** (1997). [The Delisting Bias in CRSP Data](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x). *The Journal of Finance*.
- **David Vidal-Tomás** (2022). [Which cryptocurrency data sources should scholars use?](https://doi.org/10.1016/j.irfa.2022.102061). *International Review of Financial Analysis*.
