## How to work with Fundamental data

Fundamental data pertains to the economic drivers that determine the value of securities. The nature of the data depends on the asset class:
- For equities and corporate credit, it includes corporate financials as well as industry and economy-wide data.
- For government bonds, it includes international macro-data and foreign exchange.
- For commodities, it includes asset-specific supply-and-demand determinants, such as weather data for crops. 

We will focus on equity fundamentals for the US, where data is easier to access. There are some 13,000+ public companies worldwide that generate 2 million pages of annual reports and 30,000+ hours of earnings calls. In algorithmic trading, fundamental data and features engineered from this data may be used to derive trading signals directly, for example as value indicators, and are an essential input for predictive models, including machine learning models.


### Financial statement data

The Securities and Exchange Commission (SEC) requires US issuers, that is, listed companies and securities, including mutual funds to file three quarterly financial statements (Form 10-Q) and one annual report (Form 10-K), in addition to various other regulatory filing requirements.

Since the early 1990s, the SEC made these filings available through its Electronic Data Gathering, Analysis, and Retrieval (EDGAR) system. They constitute the primary data source for the fundamental analysis of equity and other securities, such as corporate credit, where the value depends on the business prospects and financial health of the issuer. 

#### Automated processing using XBRL markup

Automated analysis of regulatory filings has become much easier since the SEC introduced XBRL, a free, open, and global standard for the electronic representation and exchange of business reports. XBRL is based on XML; it relies on [taxonomies](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) that define the meaning of the elements of a report and map to tags that highlight the corresponding information in the electronic version of the report. One such taxonomy represents the US Generally Accepted Accounting Principles (GAAP).

The SEC introduced voluntary XBRL filings in 2005 in response to accounting scandals before requiring this format for all filers since 2009 and continues to expand the mandatory coverage to other regulatory filings. The SEC maintains a website that lists the current taxonomies that shape the content of different filings and can be used to extract specific items.

There are several avenues to track and access fundamental data reported to the SEC:
- As part of the [EDGAR Public Dissemination Service]((https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)) (PDS), electronic feeds of accepted filings are available for a fee. 
- The SEC updates [RSS feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings) every 10 minutes, which list structured disclosure submissions.
- There are public [index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm) for the retrieval of all filings through FTP for automated processing.
- The financial statement (and notes) datasets contain parsed XBRL data from all financial statements and the accompanying notes.

The SEC also publishes log files containing the [internet search traffic](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) for EDGAR filings through SEC.gov, albeit with a six-month delay.


#### Building a fundamental data time series

The scope of the data in the [Financial Statement and Notes](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html) datasets consists of numeric data extracted from the primary financial statements (Balance sheet, income statement, cash flows, changes in equity, and comprehensive income) and footnotes on those statements. The data is available as early as 2009.


The folder [03_sec_edgar](03_sec_edgar) contains the notebook [edgar_xbrl](03_sec_edgar/edgar_xbrl.ipynb) to download and parse EDGAR data in XBRL format, and create fundamental metrics like the P/E ratio by combining financial statement and price data.

### Other fundamental data sources

- [Compilation of macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-and-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)


### Efficient data storage with pandas

We'll be using many different data sets in this book, and it's worth comparing the main formats for efficiency and performance. In particular, we compare the following:

- CSV: Comma-separated, standard flat text file format.
- HDF5: Hierarchical data format, developed initially at the National Center for Supercomputing, is a fast and scalable storage format for numerical data, available in pandas using the PyTables library.
- Parquet: A binary, columnar storage format, part of the Apache Hadoop ecosystem, that provides efficient data compression and encoding and has been developed by Cloudera and Twitter. It is available for pandas through the pyarrow library, led by Wes McKinney, the original author of pandas.

The notebook [storage_benchmark](04_storage_benchmark/storage_benchmark.ipynb) in the directory [04_storage_benchmark](04_storage_benchmark) compares the performance of the preceding libraries.
