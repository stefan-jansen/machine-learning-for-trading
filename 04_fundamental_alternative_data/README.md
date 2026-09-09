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

- [`01_academic_characteristics`](01_academic_characteristics.ipynb) — Reads the Chen, Pelger and Zhu (2021) panel: fifty years of monthly US equity observations, each carrying 46 rank-normalized firm characteristics and the following month's excess return, published free of the WRDS subscription the underlying data usually needs. Establishes the split boundaries, the normalization and the reach of the anonymous identifiers that every model in the `us_firm_characteristics` case study inherits.

### 4.2 Entity Resolution and Mapping

This section explains why multi-source financial research fails if issuer, security, and contract identities are not resolved correctly over time. It moves from deterministic joins to fuzzy matching, embedding-based matching, and QA controls, while emphasizing that resolution is not just a name-matching problem but a time-valid mapping problem across layers of the capital structure. Readers should care because a single wrong join can contaminate an entire research pipeline.

- [`02_sec_filing_explorer`](02_sec_filing_explorer.ipynb) — Works through the EDGAR surface once, through the EdgarTools library: finding a filer by its Central Index Key, filtering its submissions by form, pulling the three financial statements out of a 10-K's XBRL tags, and reading a Form 4 trade and a Form 13F holdings report.
- [`03_sec_form4_insider_transactions`](03_sec_form4_insider_transactions.ipynb) — Parses Form 4 insider trades from the raw XML, reconciles the extracted rows against the transaction blocks in the files, and separates the trades that reflect a decision from the compensation events that make up most of the volume.
- [`04_sec_xbrl_fundamentals`](04_sec_xbrl_fundamentals.ipynb) — Reads a quarterly fundamentals panel assembled from the SEC's XBRL Frames API, measures how long after each period end its filing arrived, and builds the as-of query that returns only what was public on a given date.
- [`05_entity_resolution`](05_entity_resolution.ipynb) — Builds the three stages of a name-to-identifier mapping — an identifier join, a fuzzy string score, and a sentence embedding — and scores all three on one labelled set, so that the acceptance threshold comes from a measured precision and recall rather than a rule of thumb.

### 4.3 Fundamentals Across the Asset-Class Spectrum

This section broadens the idea of fundamentals beyond equities and shows that the same PIT discipline applies to macro data, commodities, and crypto, even though release mechanics and tradable instruments differ. It gives readers a practical sense of how timestamp authority, revision histories, contract mapping, and chain finality vary by asset class. The payoff is a reusable framework for building time-consistent features across very different domains.

- [`06_fred_macro_eda`](06_fred_macro_eda.ipynb) — First contact with the FRED macro panel: what grid its rows sit on, and how to recover each series' release frequency from a file that has already carried every value forward onto one daily grid.
- [`07_macro_data_alignment`](07_macro_data_alignment.ipynb) — Re-dates each macro observation from the period it measures to the day it was published, rebuilds the daily panel with an as-of join so no date carries an unreleased number, and measures the revisions that remain using an archive of first-published values.
- [`08_futures_positioning`](08_futures_positioning.ipynb) — Turns the CFTC's weekly Commitment of Traders reports into a positioning signal: resolving the several contract markets that share one product code, standardizing a net position against its own year, and joining the reports onto trading sessions at the date they became public.

### 4.4 Understanding Alternative Data

This section reframes alternative data as an acquisition and engineering decision, not a buzzword category. It gives a concrete due-diligence framework around incremental signal, data quality, legal risk, and operational cost, and makes clear that many datasets fail not because they are uninteresting but because they are not defensible, reproducible, or deployable. Readers should care because most alternative-data mistakes are expensive and predictable.

- [`09_onchain_fundamentals`](09_onchain_fundamentals.ipynb) — Takes total value locked, the closest thing decentralized finance has to a fundamental, and tests whether it predicts ether returns — mostly by establishing how little a year of daily observations of a monthly horizon can say, which is the usual outcome of an honest alternative-data test.
- [`10_institutional_holdings_13f`](10_institutional_holdings_13f.ipynb) — Reads a whole quarter of Form 13F filings — several million positions from around seven thousand managers — and does the screening an aggregate over that file needs first: one filing per manager, filings whose numbers are internally inconsistent set aside, and every grouping on an identifier rather than a typed name.
- [`11_defi_tvl_evaluation`](11_defi_tvl_evaluation.ipynb) — Runs the four-question alternative-data evaluation on one real dataset — signal, data, legal, commercial — keeping the questions that can block on their own separate from the ones that can only rank, and reaching a decision from the gate that fails rather than from a weighted score.
- [`12_kalshi_prediction_markets`](12_kalshi_prediction_markets.ipynb) — Reads the first CFTC-designated prediction market: what a binary contract's price means and which price the feed actually carries, how a ladder of thresholds on one event prices a whole distribution, and how much trading sits behind a quote before a feature is built on it.
- [`13_polymarket_prediction_markets`](13_polymarket_prediction_markets.ipynb) — The unregulated counterpart, settled in a stablecoin on the Polygon blockchain and closed to US persons. Sets out which differences between the two venues change the data rather than the trading, and finds that one of the two feeds' volume columns is not a volume at all.

### 4.5 Using Text Data for NLP Features

This section provides a concrete pipeline for turning SEC filing text into a model-ready corpus. It focuses on document selection, section extraction, cleaning, and PIT-correct storage, deliberately stopping short of featurization so the engineering foundation is clear before later NLP chapters build on it. Its significance is that text features only become credible once the extraction and storage layer is auditable and time-correct.

- [`14_text_data_extraction`](14_text_data_extraction.ipynb) — Gets from a filing to a section reliably: locating an item heading that appears in the contents, in cross-references and once as the section itself, checking the extraction rather than assuming it, and measuring what changed between two consecutive filings of the same company.

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

Every Chapter 4 notebook completes in well under a minute with peak
memory under 3 GB; none needs a long-running or high-memory callout.
