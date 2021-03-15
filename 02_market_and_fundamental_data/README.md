# Market & Fundamental Data: Sources and Techniques

Data has always been an essential driver of trading, and traders have long made efforts to gain an advantage from access to superior information. These efforts date back at least to the rumors that the House of Rothschild benefited handsomely from bond purchases upon advance news about the British victory at Waterloo carried by pigeons across the channel.

Today, investments in faster data access take the shape of the Go West consortium of leading **high-frequency trading** (HFT) firms that connects the Chicago Mercantile Exchange (CME) with Tokyo. The round-trip latency between the CME and the BATS exchanges in New York has dropped to close to the theoretical limit of eight milliseconds as traders compete to exploit arbitrage opportunities. At the same time, regulators and exchanges have started to introduce speed bumps that slow down trading to limit the adverse effects on competition of uneven access to information.

Traditionally, investors mostly relied on **publicly available market and fundamental data**.  Efforts to create or acquire private datasets, for example through proprietary surveys, were limited. Conventional strategies focus on equity fundamentals and build financial models on reported financials, possibly combined with industry or macro data to project earnings per share and stock prices. Alternatively, they leverage technical analysis to extract signals from market data using indicators computed from price and volume information.

**Machine learning (ML) algorithms** promise to exploit market and fundamental data more efficiently than human-defined rules and heuristics, in particular when combined with alternative data, the topic of the next chapter. We will illustrate how to apply ML algorithms ranging from linear models to recurrent neural networks (RNNs) to market and fundamental data and generate tradeable signals.

This chapter introduces market and fundamental data sources and explains how they reflect the environment in which they are created. The details of the **trading environment** matter not only for the proper interpretation of market data but also for the design and execution of your strategy and the implementation of realistic backtesting simulations. We also illustrate how to access and work with trading and financial statement data from various sources using Python. 
 
## Content

1. [Market data reflects the trading environment](#market-data-reflects-the-trading-environment)
    * [Market microstructure: The nuts and bolts of trading](#market-microstructure-the-nuts-and-bolts-of-trading)
2. [Working with high-frequency market data](#working-with-high-frequency-market-data)
    * [How to work with NASDAQ order book data](#how-to-work-with-nasdaq-order-book-data)
    * [How trades are communicated: The FIX protocol](#how-trades-are-communicated-the-fix-protocol)
    * [The NASDAQ TotalView-ITCH data feed](#the-nasdaq-totalview-itch-data-feed)
        - [Code Example: Parsing and normalizing tick data ](#code-example-parsing-and-normalizing-tick-data-)
        - [Additional Resources](#additional-resources)
    * [AlgoSeek minute bars: Equity quote and trade data](#algoseek-minute-bars-equity-quote-and-trade-data)
        - [From the consolidated feed to minute bars](#from-the-consolidated-feed-to-minute-bars)
        - [Code Example: How to process AlgoSeek intraday data](#code-example-how-to-process-algoseek-intraday-data)
3. [API Access to Market Data](#api-access-to-market-data)
    * [Remote data access using pandas](#remote-data-access-using-pandas)
    * [Code Examples](#code-examples)
    * [Data sources](#data-sources)
    * [Industry News](#industry-news)
4. [How to work with Fundamental data](#how-to-work-with-fundamental-data)
    * [Financial statement data](#financial-statement-data)
    * [Automated processing using XBRL markup](#automated-processing-using-xbrl-markup)
    * [Code Example: Building a fundamental data time series](#code-example-building-a-fundamental-data-time-series)
    * [Other fundamental data sources](#other-fundamental-data-sources)
5. [Efficient data storage with pandas](#efficient-data-storage-with-pandas)
    * [Code Example](#code-example)
 
## Market data reflects the trading environment

Market data is the product of how traders place orders for a financial instrument directly or through intermediaries on one of the numerous marketplaces and how they are processed and how prices are set by matching demand and supply. As a result, the data reflects the institutional environment of trading venues, including the rules and regulations that govern orders, trade execution, and price formation. See [Harris](https://global.oup.com/ushe/product/trading-and-exchanges-9780195144703?cc=us&lang=en&) (2003) for a global overview and [Jones](https://www0.gsb.columbia.edu/faculty/cjones/papers/2018.08.31%20US%20Equity%20Market%20Data%20Paper.pdf) (2018) for details on the US market.

Algorithmic traders use algorithms, including ML, to analyze the flow of buy and sell orders and the resulting volume and price statistics to extract trade signals that capture insights into, for example, demand-supply dynamics or the behavior of certain market participants. This section reviews institutional features that impact the simulation of a trading strategy during a backtest before we start working with actual tick data created by one such environment, namely the NASDAQ.

### Market microstructure: The nuts and bolts of trading

Market microstructure studies how the institutional environment affects the trading process and shapes outcomes like the price discovery, bid-ask spreads and quotes, intraday trading behavior, and transaction costs. It is one of the fastest-growing fields of financial research, propelled by the rapid development of algorithmic and electronic trading.  

Today, hedge funds sponsor in-house analysts to track the rapidly evolving, complex details and ensure execution at the best possible market prices and design strategies that exploit market frictions. This section provides a brief overview of key concepts, namely different market places and order types, before we dive into the data generated by trading.

- [Trading and Exchanges - Market Microstructure for Practitioners](https://global.oup.com/ushe/product/trading-and-exchanges-9780195144703?cc=us&lang=en&), Larry Harris, Oxford University Press, 2003
- [Understanding the Market for Us Equity Market Data](https://www0.gsb.columbia.edu/faculty/cjones/papers/2018.08.31%20US%20Equity%20Market%20Data%20Paper.pdf), Charles Jones, NYSE, 2018 
- [World Federation of Exchanges](https://www.world-exchanges.org/our-work/statistics)
- [Econophysics of Order-driven Markets](https://www.springer.com/gp/book/9788847017658), Abergel et al, 2011
    - Presents the ideas and research from various communities (physicists, economists, mathematicians, financial engineers) on the  modelling and analyzing order-driven markets. Of primary interest in these studies are the mechanisms leading to the statistical regularities of price statistics. Results pertaining to other important issues such as market impact, the profitability of trading strategies, or mathematical models for microstructure effects, are also presented.

## Working with high-frequency market data

Two categories of market data cover the thousands of companies listed on US exchanges that are traded under Reg NMS: The consolidated feed combines trade and quote data from each trading venue, whereas each individual exchange offers proprietary products with additional activity information for that particular venue.

In this section, we will first present proprietary order flow data provided by the NASDAQ that represents the actual stream of orders, trades, and resulting prices as they occur on a tick-by-tick basis. Then, we demonstrate how to regularize this continuous stream of data that arrives at irregular intervals into bars of a fixed duration. Finally, we introduce AlgoSeek’s equity minute bar data that contains consolidated trade and quote information. In each case, we illustrate how to work with the data using Python so you can leverage these sources for your trading strategy.

### How to work with NASDAQ order book data

The primary source of market data is the order book, which updates in real-time throughout the day to reflect all trading activity. Exchanges typically offer this data as a real-time service for a fee but may provide some historical data for free. 

In the United States, stock markets provide quotes in three tiers, namely Level I, II and III that offer increasingly granular information and capabilities:
- Level I: real-time bid- and ask-price information, as available from numerous online sources
- Level II: adds information about bid and ask prices by specific market makers as well as size and time of recent transactions for better insights into the liquidity of a given equity.
- Level III: adds the ability to enter or change quotes, execute orders, and confirm trades and is only available to market makers and exchange member firms. Access to Level III quotes permits registered brokers to meet best execution requirements.

The trading activity is reflected in numerous messages about orders sent by market participants. These messages typically conform to the electronic Financial Information eXchange (FIX) communications protocol for real-time exchange of securities transactions and market data or a native exchange protocol. 

- [The Limit Order Book](https://arxiv.org/pdf/1012.0349.pdf)
- [Feature Engineering for Mid-Price Prediction With Deep Learning](https://arxiv.org/abs/1904.05384)
- [Price jump prediction in Limit Order Book](https://arxiv.org/pdf/1204.1381.pdf)
- [Handling and visualizing order book data](https://github.com/0b01/recurrent-autoencoder/blob/master/Visualizing%20order%20book.ipynb) by Ricky Han

### How trades are communicated: The FIX protocol

The trading activity is reflected in numerous messages about trade orders sent by market participants. These messages typically conform to the electronic Financial Information eXchange (FIX) communications protocol for real-time exchange of securities transactions and market data or a native exchange protocol. 

- [FIX Trading Standards](https://www.fixtrading.org/standards/)
- Python: [Simplefix](https://github.com/da4089/simplefix)
- C++ version: [quickfixengine](http://www.quickfixengine.org/)
- Interactive Brokers [interface](https://www.interactivebrokers.com/en/index.php?f=4988)

### The NASDAQ TotalView-ITCH data feed

While FIX has a dominant large market share, exchanges also offer native protocols. The Nasdaq offers a TotalView ITCH direct data-feed protocol that allows subscribers to track individual orders for equity instruments from placement to execution or cancellation.

- The ITCH [Specifications](http://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
- [Sample Files](ftp://emi.nasdaq.com/ITCH/)

#### Code Example: Parsing and normalizing tick data 

- The folder [NASDAQ TotalView ITCH Order Book](01_NASDAQ_TotalView-ITCH_Order_Book) contains the notebooks to
    - download NASDAQ Total View sample tick data,
    - parse the messages from the binary source data
    - reconstruct the order book for a given stock
    - visualize order flow data
    - normalize tick data
- Binary Data services: the `struct` [module](https://docs.python.org/3/library/struct.html)
 
#### Additional Resources
 
 - Native exchange protocols [around the world](https://en.wikipedia.org/wiki/List_of_electronic_trading_protocols_
 - [High-frequency trading in a limit order book](https://www.math.nyu.edu/faculty/avellane/HighFrequencyTrading.pdf), Avellaneda and Stoikov, Quantitative Finance, Vol. 8, No. 3, April 2008, 217–224
 - [Using a Simulator to Develop Execution Algorithms](http://www.math.ualberta.ca/~cfrei/PIMS/Almgren5.pdf), Robert Almgren, quantitative brokers, 2016
 - [Backtesting Microstructure Strategies](https://rickyhan.com/jekyll/update/2019/12/22/how-to-simulate-market-microstructure.html), Ricky Han, 2019
- [Optimal High-Frequency Market Making](http://stanford.edu/class/msande448/2018/Final/Reports/gr5.pdf), Fushimi et al, 2018
- [Simulating and analyzing order book data: The queue-reactive model](https://arxiv.org/pdf/1312.0563.pdf), Huan et al, 2014
- [How does latent liquidity get revealed in the limit order book?](https://arxiv.org/pdf/1808.09677.pdf), Dall’Amico et al, 2018

### AlgoSeek minute bars: Equity quote and trade data

AlgoSeek provides historical intraday data at the quality previously available only to institutional investors. The AlgoSeek Equity bars provide a very detailed intraday quote and trade data in a user-friendly format aimed at making it easy to design and backtest intraday ML-driven strategies. As we will see, the data includes not only OHLCV information but also information on the bid-ask spread and the number of ticks with up and down price moves, among others.
AlgoSeek has been so kind as to provide samples of minute bar data for the NASDAQ 100 stocks from 2013-2017 for demonstration purposes and will make a subset of this data available to readers of this book.

#### From the consolidated feed to minute bars

AlgoSeek minute bars are based on data provided by the Securities Information Processor (SIP) that manages the consolidated feed mentioned at the beginning of this section. You can find the documentation at https://www.algoseek.com/data-drive.html.

Quote and trade data fields
The minute bar data contain up to 54 fields. There are eight fields for the open, high, low, and close elements of the bar, namely:
- The timestamp for the bar and the corresponding trade 
- The price and the size for the prevailing bid-ask quote and the relevant trade

There are also 14 data points with volume information for the bar period:
- The number of shares and corresponding trades
- The trade volumes at or below the bid, between the bid quote and the midpoint, at the midpoint, between the midpoint and the ask quote, and at or above the ask, as well as for crosses
- The number of shares traded with up- or downticks, i.e., when the price rose or fell, as well as when the price did not change, differentiated by the previous direction of price movement

#### Code Example: How to process AlgoSeek intraday data

The directory [algoseek_intraday](02_algoseek_intraday) contains instructions on how to download sample data from AlgoSeek. 

- This information will be made available shortly.

## API Access to Market Data

There are several options to access market data via API using Python. In this chapter, we first present a few sources built into the [`pandas`](https://pandas.pydata.org/) library. Then we briefly introduce the trading platform [Quantopian](https://www.quantopian.com/posts), the data provider [Quandl](https://www.quandl.com/) (acquired by NASDAQ in 12/2018) and the backtesting library [`zipline`](https://github.com/quantopian/zipline) that we will use later in the book, and list several additional options to access various types of market data. The directory [data_providers](03_data_providers) contains several notebooks that illustrate the usage of these options.

### Remote data access using pandas

- read_html [docs](https://pandas.pydata.org/pandas-docs/stable/)
- S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- `pandas-datareader`[docs](https://pandas-datareader.readthedocs.io/en/latest/index.html)

### Code Examples

The folder [data providers](03_data_providers) contains examples to use various data providers.
1. Remote data access using [pandas DataReader](03_data_providers/01_pandas_datareader_demo.ipynb)
2. Downloading market and fundamental data with [yfinance](03_data_providers/02_yfinance_demo.ipynb)
3. Parsing Limit Order Tick Data from [LOBSTER](03_data_providers/03_lobster_itch_data.ipynb)
4. Quandl [API Demo](03_data_providers/04_quandl_demo.ipynb)
5. Zipline [data access](03_data_providers/05_zipline_data_demo.ipynb)

### Data sources

- Quandl [docs](https://docs.quandl.com/docs) and Python [API](https://www.quandl.com/tools/python﻿)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Quantopian](https://www.quantopian.com/posts)
- [Zipline](https://zipline.ml4trading.io/﻿)
- [LOBSTER](https://lobsterdata.com/)
- [The Investor Exchange](https://iextrading.com/﻿)
- [IEX Cloud](https://iexcloud.io/) financial data infrastructure
- [Money.net](https://www.money.net/)
- [Trading Economic](https://tradingeconomics.com/)
- [Barchart](https://www.barchart.com/)
- [Alpha Vantage](https://www.alphavantage.co/﻿)
- [Alpha Trading Labs](https://www.alphatradinglabs.com/)
- [Tiingo](https://www.tiingo.com/) stock market tools

### Industry News

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

### Automated processing using XBRL markup

Automated analysis of regulatory filings has become much easier since the SEC introduced XBRL, a free, open, and global standard for the electronic representation and exchange of business reports. XBRL is based on XML; it relies on [taxonomies](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) that define the meaning of the elements of a report and map to tags that highlight the corresponding information in the electronic version of the report. One such taxonomy represents the US Generally Accepted Accounting Principles (GAAP).

The SEC introduced voluntary XBRL filings in 2005 in response to accounting scandals before requiring this format for all filers since 2009 and continues to expand the mandatory coverage to other regulatory filings. The SEC maintains a website that lists the current taxonomies that shape the content of different filings and can be used to extract specific items.

There are several avenues to track and access fundamental data reported to the SEC:
- As part of the [EDGAR Public Dissemination Service]((https://www.sec.gov/oit/announcement/public-dissemination-service-system-contact.html)) (PDS), electronic feeds of accepted filings are available for a fee. 
- The SEC updates [RSS feeds](https://www.sec.gov/structureddata/rss-feeds-submitted-filings) every 10 minutes, which list structured disclosure submissions.
- There are public [index files](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm) for the retrieval of all filings through FTP for automated processing.
- The financial statement (and notes) datasets contain parsed XBRL data from all financial statements and the accompanying notes.

The SEC also publishes log files containing the [internet search traffic](https://www.sec.gov/dera/data/edgar-log-file-data-set.html) for EDGAR filings through SEC.gov, albeit with a six-month delay.

### Code Example: Building a fundamental data time series

The scope of the data in the [Financial Statement and Notes](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html) datasets consists of numeric data extracted from the primary financial statements (Balance sheet, income statement, cash flows, changes in equity, and comprehensive income) and footnotes on those statements. The data is available as early as 2009.

The folder [04_sec_edgar](04_sec_edgar) contains the notebook [edgar_xbrl](04_sec_edgar/edgar_xbrl.ipynb) to download and parse EDGAR data in XBRL format, and create fundamental metrics like the P/E ratio by combining financial statement and price data.

### Other fundamental data sources

- [Compilation of macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-and-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)

## Efficient data storage with pandas

We'll be using many different data sets in this book, and it's worth comparing the main formats for efficiency and performance. In particular, we compare the following:

- CSV: Comma-separated, standard flat text file format.
- HDF5: Hierarchical data format, developed initially at the National Center for Supercomputing, is a fast and scalable storage format for numerical data, available in pandas using the PyTables library.
- Parquet: A binary, columnar storage format, part of the Apache Hadoop ecosystem, that provides efficient data compression and encoding and has been developed by Cloudera and Twitter. It is available for pandas through the pyarrow library, led by Wes McKinney, the original author of pandas.

### Code Example

The notebook [storage_benchmark](05_storage_benchmark/storage_benchmark.ipynb) in the directory [05_storage_benchmark](05_storage_benchmark) compares the performance of the preceding libraries.
