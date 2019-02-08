# Chapter 02: Market & Fundamental Data

## How to work with Market Data
###  Market microstructure

- [Trading and Exchanges - Market Microstructure for Practitioners](https://global.oup.com/ushe/product/trading-and-exchanges-9780195144703?cc=us&lang=en&), Larry Harris, Oxford University Press, 2002
- [World Federation of Exchanges](https://www.world-exchanges.org/our-work/statistics)

### Working with Order Book data

#### The FIX protocol

- [FIX Trading Standards](https://www.fixtrading.org/standards/)
- Python: [Simplefix](https://github.com/da4089/simplefix)
- C++ version: [quickfixengine](http://www.quickfixengine.org/)
- Interactive Brokers [interface](https://www.interactivebrokers.com/en/index.php?f=4988)

#### Nasdaq TotalView-ITCH data

- The ITCH [Specifications](http://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
- [Sample Files](ftp://emi.nasdaq.com/ITCH/)

#### Code Examples

- The folder `NASDAQ TotalView ITCH Order Book` contains the notebooks to
    - download NASDAQ Total View sample tick data,
    - parse the messages from the binary source data
    - reconstruct the order book for a given stock
    - visualize order flow data
    - normalize tick data
- Binary Data services: the `struct` [module](https://docs.python.org/3/library/struct.html)


#### Other protocols

 - Native exchange protocols [around the world](https://en.wikipedia.org/wiki/List_of_electronic_trading_protocols_

### Access to Market Data

#### Remote data access using pandas

- read_html [docs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_html.html?highlight=pandas%20io%20read_html)
- S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- `pandas-datareader`[docs](https://pandas-datareader.readthedocs.io/en/latest/index.html)

#### Code Examples

The folder `data providers` contains examples to use various data providers.

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

