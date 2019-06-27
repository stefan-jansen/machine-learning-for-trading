#!/usr/bin/env python
# coding: utf-8

# Remote data access using pandas

import os
import pandas_datareader.data as web
from datetime import datetime
import pandas as pd

# Download html table with SP500 constituents

sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_constituents = pd.read_html(sp_url, header=0)[0]

print(sp500_constituents.info())

print(sp500_constituents.head())

# pandas-datareader for Market Data
# Yahoo Finance

start = '2014'
end = datetime(2017, 5, 24)

yahoo = web.DataReader('FB', 'yahoo', start=start, end=end)
print(yahoo.info())

# IEX

start = datetime(2015, 2, 9)
# end = datetime(2017, 5, 24)

# iex = web.DataReader('FB', 'iex', start)
# iex.info()
#
# print(iex.tail())

# Book Data
# DEEP is used to receive real-time depth of book quotations direct from IEX. The depth of book quotations received via DEEP provide an aggregated size of resting displayed orders at a price and side, and do not indicate the size or number of individual orders at any price level. Non-displayed orders and non-displayed portions of reserve orders are not represented in DEEP.
# DEEP also provides last trade price and size information. Trades resulting from either displayed or non-displayed orders matching on IEX will be reported. Routed executions will not be reported.
# Only works on trading days.

book = web.get_iex_book('AAPL')
print(list(book.keys()))

orders = pd.concat([pd.DataFrame(book[side]).assign(side=side) for side in ['bids', 'asks']])
print(orders.head())

for key in book.keys():
    try:
        print(f'\n{key}')
        print(pd.DataFrame(book[key]))
    except:
        print(book[key])

print(pd.DataFrame(book['trades']).head())

# Quandl

symbol = 'FB.US'

quandl = web.DataReader(symbol, 'quandl', '2015-01-01')
print(quandl.info())
# else:
#     from pprint import pprint
#     print('Please obtain API key from Quandl and store as environment variable')
#     pprint(os.environ)
# FRED

start = datetime(2010, 1, 1)
end = datetime(2013, 1, 27)
gdp = web.DataReader('GDP', 'fred', start, end)
print(gdp.info())

inflation = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start, end)
print(inflation.info())

# Fama/French

from pandas_datareader.famafrench import get_available_datasets

print(get_available_datasets())

ds = web.DataReader('5_Industry_Portfolios', 'famafrench')
print(ds['DESCR'])

# World Bank

from pandas_datareader import wb

gdp_variables = wb.search('gdp.*capita.*const')
gdp_variables.head()

wb_data = wb.download(indicator='NY.GDP.PCAP.KD',
                      country=['US', 'CA', 'MX'],
                      start=1990,
                      end=2019)
print(wb_data.head())

# OECD

df = web.DataReader('TUD', 'oecd', end='2015')
print(df[['Japan', 'United States']])

# EuroStat

df = web.DataReader('tran_sf_railac', 'eurostat')
print(df.head())

# Stooq
# Google finance stopped providing common index data download. The Stooq site had this data for download for a while but is currently broken, awaiting release of [fix](https://github.com/pydata/pandas-datareader/issues/594)


index_url = 'https://stooq.com/t/'
ix = pd.read_html(index_url)
len(ix)

f = web.DataReader('^SPX', 'stooq', start='20000101')
print(f.info())
print(f.head())

# NASDAQ Symbols

from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

symbols = get_nasdaq_symbols()
print(symbols.info())

url = 'https://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ'
res = pd.read_html(url)
print(len(res))

for r in res:
    print(r.info())

# Tiingo

# Requires [signing up](https://api.tiingo.com/) and storing API key in environment

df = web.get_data_tiingo('GOOG', api_key=os.getenv('TIINGO_API_KEY'))
print(df.info())
