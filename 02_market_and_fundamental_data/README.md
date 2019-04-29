# Chapter 02: Market & Fundamental Data

This chapter introduces market and fundamental data sources and the environment in which they are created. Familiarity with various types of orders and the trading infrastructure matters because they affect backtest simulations of a trading strategy. We also illustrate how to use Python to access and work with trading and financial statement data.
 
In particular, this chapter will cover the following topics:
- How market microstructure shapes market data
- How to reconstruct the order book from tick data using Nasdaq ITCH 
- How to summarize tick data using various types of bars
- How to work with eXtensible Business Reporting Language (XBRL)-encoded electronic filings
- How to parse and combine market and fundamental data to create a P/E series
- How to access various market and fundamental data sources using Python

## How to work with Market Data

Market data results from the placement and processing of buy and sell orders in the course of the trading of financial instruments on the many marketplaces. The data reflects the institutional environment of trading venues, including the rules and regulations that govern orders, trade execution, and price formation.

Algorithmic traders use ML algorithms to analyze the flow of buy and sell orders and the resulting volume and price statistics to extract trade signals or features that capture insights into, for example, demand-supply dynamics or the behavior of certain market participants.

This section reviews institutional features that impact the simulation of a trading strategy during a backtest. Then, we will take a look at how tick data can be reconstructed from the order book source. Next, we will highlight several methods that regularize tick data and aim to maximize the information content. Finally, we will illustrate how to access various market data provider interfaces and highlight several providers.

###  Market microstructure

Market microstructure is the branch of financial economics that investigates the trading process and the organization of related markets. The following references provide insights into institutional details that can be quite complex and diverse across asset classes and their derivatives, trading venues, and geographies, as well as data about the trading activities on various exchanges around the world

- [Trading and Exchanges - Market Microstructure for Practitioners](https://global.oup.com/ushe/product/trading-and-exchanges-9780195144703?cc=us&lang=en&), Larry Harris, Oxford University Press, 2002
- [World Federation of Exchanges](https://www.world-exchanges.org/our-work/statistics)

### Working with Order Book data
The primary source of market data is the order book, which is continuously updated in real-time throughout the day to reflect all trading activity. Exchanges typically offer this data as a real-time service and may provide some historical data for free. 

#### The FIX protocol

The trading activity is reflected in numerous messages about trade orders sent by market participants. These messages typically conform to the electronic Financial Information eXchange (FIX) communications protocol for real-time exchange of securities transactions and market data or a native exchange protocol. 

- [FIX Trading Standards](https://www.fixtrading.org/standards/)
- Python: [Simplefix](https://github.com/da4089/simplefix)
- C++ version: [quickfixengine](http://www.quickfixengine.org/)
- Interactive Brokers [interface](https://www.interactivebrokers.com/en/index.php?f=4988)

#### Nasdaq TotalView-ITCH data

While FIX has a dominant large market share, exchanges also offer native protocols. The Nasdaq offers a TotalView ITCH direct data-feed protocol that allows subscribers to track individual orders for equity instruments from placement to execution or cancellation.

- The ITCH [Specifications](http://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
- [Sample Files](ftp://emi.nasdaq.com/ITCH/)

#### Code Examples

- The folder [NASDAQ TotalView ITCH Order Book](01_NASDAQ_TotalView-ITCH_Order_Book) contains the notebooks to
    - download NASDAQ Total View sample tick data,
    - parse the messages from the binary source data
    - reconstruct the order book for a given stock
    - visualize order flow data
    - normalize tick data
- Binary Data services: the `struct` [module](https://docs.python.org/3/library/struct.html)

#### Other protocols

 - Native exchange protocols [around the world](https://en.wikipedia.org/wiki/List_of_electronic_trading_protocols_

### Access to Market Data

There are several options to access market data via API using Python. In this chapter, we first present a few sources built into the [`pandas`](https://pandas.pydata.org/) library. Then we briefly introduce the trading platform [Quantopian](https://www.quantopian.com/posts), the data provider [Quandl](https://www.quandl.com/) (acquired by NASDAQ in 12/2018) and the backtesting library [`zipline`](https://github.com/quantopian/zipline) that we will use later in the book, and list several additional options to access various types of market data. The directory [data_providers](02_data_providers) contains several notebooks that illustrate the usage of these options.

#### Remote data access using pandas

- read_html [docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_html.html?highlight=pandas%20io%20read_html)
- S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- `pandas-datareader`[docs](https://pandas-datareader.readthedocs.io/en/latest/index.html)

#### Code Examples

The folder [data providers](02_data_providers) contains examples to use various data providers.

Relevant sources include:

- Quandl [docs](https://docs.quandl.com/docs) and Python [API](https://www.quandl.com/tools/python﻿)
- [Quantopian](https://www.quantopian.com/posts)
- [Zipline](http://www.zipline.io/﻿)
- [LOBSTER](https://lobsterdata.com/)
- [The Investor Exchange](https://iextrading.com/﻿)
- [Money.net](https://www.money.net/)
- [Trading Economic](https://tradingeconomics.com/)
- [Barchart](https://www.barchart.com/)
- [Alpha Vantage](https://www.alphavantage.co/﻿)
- [Alpha Trading Labs](https://www.alphatradinglabs.com/)

News
- [Bloomberg and Reuters lose data share to smaller rivals](https://www.ft.com/content/622855dc-2d31-11e8-9b4b-bc4b9f08f381), FT, 2018


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
