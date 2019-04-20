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

This chapter reviews institutional features that impact the simulation of a trading strategy during a backtest. Then, we will take a look at how tick data can be reconstructed from the order book source. Next, we will highlight several methods that regularize tick data and aim to maximize the information content. Finally, we will illustrate how to access various market data provider interfaces and highlight several providers.

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

### Financial statement data

#### Automated processing using XBRL markup

- [SEC Standard EDGAR Taxonomies](https://www.sec.gov/info/edgar/edgartaxonomies.shtml)
- [ EDGAR Public Dissemination Service (PDS)](https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)
- [SEC Really Simple Syndication (RSS) Feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings)
- [SEC EDGAR index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm)
- [SEC seach traffic log files](https://www.sec.gov/dera/data/edgar-log-file-data-set.html)


#### Building a fundamental data time series
- [SEC Financial Statements & Notes Data Set](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html)

#### Code Examples

The folder EDGAR contains the code to download and parse EDGAR data in XBRL format.

### Other fundamental data sources

- [Compilation of macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-and-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)

### Efficient data storage with pandas

#### HDF Format

- Pandas [HDF5](http://pandas.pydata.org/pandas-docs/version/0.22/io.html#hdf5-pytables)
- [HDF Support Portal](http://portal.hdfgroup.org/display/support)
- PyTables [docs](https://www.pytables.org/)

#### Parquet Format

- Apache Parquet [docs](https://parquet.apache.org/)
- PyArrow: Parquet for Python [docs](https://arrow.apache.org/docs/python/parquet.html)
- Development update: High speed Apache Parquet in Python with Apache Arrow by Wes McKinney [blog](http://wesmckinney.com/blog/python-parquet-update/)


Latest test: 03-06-2019

