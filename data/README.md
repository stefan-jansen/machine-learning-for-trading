# Data Sources used in the book

We will use freely available historical data from market, fundamental and alternative sources. Chapter 2, Market and Fundamental Data and Chapter 3, Alternative Data for Finance  cover characteristics and access to these data sources and introduce key providers that we will use throughout the book. 

A few sample data sources that we will source and work with include, among others:
- Quandl daily prices and other data points for over 3,000 US stocks
- Algoseek minute bar trade and quote price data for NASDAQ 100 stocks
- Stooq daily price data on Japanese equities and US ETFs and stocks
- Yahoo finance daily price data and fundamentals for US stocks  
- NASDAQ ITCH order book data
- Electronic Data Gathering, Analysis, and Retrieval (EDGAR) SEC filings
- Earnings call transcripts from Seeking Alpha
- Various macro fundamental data from the Federal Reserve and others
- Financial news data from Reuters, etc.
- Twitter sentiment data
- Yelp business reviews sentiment data

## How to source the Data

There are several notebooks that guide you through the data sourcing process:
- The notebook [create_datasets](create_datasets.ipynb) contains information on downloading the **Quandl Wiki stock prices** and a few other sources that we use throughout the book, such as S&P500 benchmark, and US equities metadata.
- The notebook [create_stooq_data](create_stooq_data.ipynb) demonstrates how to download historical prices for Japanese stocks and US stocks and ETFs from STOOQ.
  > Please note that STOOQ will disable automatic downloads and require CAPTCHA starting Dec 10, 2020 so that the code that downloads and unpacks the zip files will no longer work; please navigate to their website for manual download.
- The notebook [create_yelp_review_data](create_yelp_review_data.ipynb) combines text data with additional numerical features for sentiment analysis from Yelp user reviews. 
- The notebook [glove_word_vectors](glove_word_vectors.ipynb) downloads pre-trained word vectors.
- The notebook [twitter_sentiment](twitter_sentiment.ipynb) downloads and extracts twitter data for sentiment analysis.

In addition, instructions to obtain data sources for specific applications are provided in the relevant directories and notebooks. 

