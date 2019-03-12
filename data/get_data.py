#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
import pandas_datareader.data as web

pd.set_option('display.expand_frame_repr', False)


def get_wiki_prices():
    """source: https://www.quandl.com/api/v3/datatables/WIKI/PRICES?qopts.export=true&api_key=<API_KEY>
        Download and rename to wiki_prices.csv
    """

    df = pd.read_csv('wiki_prices.csv',
                     parse_dates=['date'],
                     index_col=['date', 'ticker'],
                     infer_datetime_format=True)

    print(df.info(null_counts=True))
    with pd.HDFStore('assets.h5') as store:
        store.put('quandl/wiki/prices', df)


def get_wiki_constitutents():
    """source: https://www.quandl.com/api/v3/databases/WIKI/codes?api_key=<API_KEY>
        Download and rename to wiki_stocks.csv
    """
    df = pd.read_csv('wiki_stocks.csv', header=None)
    df = pd.concat([df[0].str.split('/', expand=True)[1].str.strip(),
                    df[1].str.split('(', expand=True)[0].str.strip()], axis=1)
    df.columns = ['symbol', 'name']
    print(df.info(null_counts=True))
    with pd.HDFStore('assets.h5') as store:
        store.put('quandl/wiki/prices', df)


def get_sp500_prices():
    """Download historical S&P 500 prices from stooq"""
    df = pd.read_csv('https://stooq.com/q/d/l/?s=^spx&i=d', parse_dates=['Date'])
    df = df.rename(columns=str.lower).set_index('date')
    print(df.info())
    with pd.HDFStore('assets.h5') as store:
        store.put('sp500/prices', df)


def get_sp500_constituents():
    """Download current S&P 500 constituents from Wikipedia"""
    df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', header=0)[0]
    df.columns = ['ticker', 'name', 'sec_filings', 'gics_sector', 'gics_sub_industry',
                  'location', 'first_added', 'cik', 'founded']
    df = df.drop('sec_filings', axis=1).set_index('ticker')
    print(df.info())
    with pd.HDFStore('assets.h5') as store:
        store.put('sp500/stocks', df)


def get_nasdaq_companies():
    """Download list of companies traded on NASDAQ, AMEX and NYSE"""
    url = 'https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange={}&render=download'
    exchanges = ['NASDAQ', 'AMEX', 'NYSE']
    df = pd.concat([pd.read_csv(url.format(ex)) for ex in exchanges]).dropna(how='all', axis=1)
    df = df.rename(columns=str.lower).set_index('symbol').drop('summary quote', axis=1)
    print(df.info())
    with pd.HDFStore('assets.h5') as store:
        store.put('us_equities/stocks', df)


def get_fred():
    """Download bond index data from FRED"""
    securities = {'BAMLCC0A0CMTRIV'   : 'US Corp Master TRI',
                  'BAMLHYH0A0HYM2TRIV': 'US High Yield TRI',
                  'BAMLEMCBPITRIV'    : 'Emerging Markets Corporate Plus TRI',
                  'GOLDAMGBD228NLBM'  : 'Gold (London, USD)',
                  'DGS10'             : '10-Year Treasury CMR',
                  }

    df = web.DataReader(name=list(securities.keys()), data_source='fred', start=2000)
    df = df.rename(columns=securities).dropna(how='all').resample('B').mean()

    with pd.HDFStore('assets.h5') as store:
        store.put('fred/assets', df)


def get_treasury_index():
    name = 'S&P U.S. Treasury Bond Current 10-Year Index'
    df = pd.read_excel('treasury_10y.xls')
    df.Data = pd.to_datetime(df.Date)
    return df.set_index('Date').Index.resample('B').mean().to_frame('Treasury Index')


def get_bcom():
    bcom = pd.read_csv('BCOM.csv', parse_dates=['Date'])
    return bcom.set_index('Date').Price.resample('B').mean().to_frame('BCOM')


def get_stock_sample():
    data_dir = Path('..', '00_data')
    with pd.HDFStore(str(data_dir / 'assets.h5')) as store:
        df = store.get(join('quandl', 'wiki', 'prices'))
        close = df.adj_close.unstack().loc[str(start):str(end)]
        open = df.adj_open.unstack().loc[str(start):str(end)]

    nobs = close.count()
    close = close.loc[:, nobs[nobs == nobs.quantile(.9)].index]
    print(close.info(null_counts=True))
    open = open.loc[:, close.columns]
    print(open.info())
    with pd.HDFStore('alpha_factors.h5') as store:
        store.put('prices/open', open)
        store.put('prices/close', close)
