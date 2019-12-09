#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

idx = pd.IndexSlice

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

download_timestamp = '2019-10-04T15;31;36.973526'
data_path = Path('~', 'projects', 'machine-learning-for-trading', 'data', 'nasdaq100').expanduser()
zipline_path = Path('~', '.zipline', 'data', 'quandl', download_timestamp).expanduser()

adj_db_path = zipline_path / 'adjustments.sqlite'
equities_db_path = zipline_path / 'assets-7.sqlite'


def read_sqlite(table, con):
    return pd.read_sql("SELECT * FROM " + table, con=con).dropna(how='all', axis=1)


def get_equities():
    nasdaq100 = pd.read_hdf(data_path / 'data.h5', '1min_trades')

    equities_con = sqlite3.connect(equities_db_path.as_posix())
    equities = read_sqlite('equity_symbol_mappings', equities_con)

    all_tickers = nasdaq100.index.get_level_values('ticker').unique()
    tickers_with_meta = np.sort(all_tickers.intersection(pd.Index(equities.symbol)))

    nasdaq_info = (get_nasdaq_symbols()
                   .reset_index()
                   .rename(columns=lambda x: x.lower().replace(' ', '_'))
                   .loc[:, ['symbol', 'security_name']]
                   .rename(columns={'security_name': 'asset_name'}))
    nasdaq_tickers = pd.DataFrame({'symbol': tickers_with_meta}).merge(nasdaq_info, how='left')
    nasdaq_sids = (equities.loc[equities.symbol.isin(nasdaq_tickers.symbol),
                                ['symbol', 'sid']])
    nasdaq_tickers = (nasdaq_tickers.merge(nasdaq_sids, how='left')
                      .reset_index()
                      .rename(columns={'sid': 'quandl_sid', 'index': 'sid'}))
    nasdaq_tickers.to_hdf('algoseek.h5', 'equities')


def get_dividends():
    equities = pd.read_hdf('algoseek.h5', 'equities')

    adjustments_con = sqlite3.connect(adj_db_path.as_posix())
    div_cols = ['sid', 'ex_date', 'declared_date', 'pay_date', 'record_date', 'amount']

    dividends = read_sqlite('dividend_payouts', adjustments_con)[['sid', 'ex_date', 'amount']]
    dividends = (dividends.rename(columns={'sid': 'quandl_sid'})
                 .merge(equities[['quandl_sid', 'sid']])
                 .drop('quandl_sid', axis=1))
    print(dividends.loc[:, div_cols].info())
    dividends.reindex(div_cols, axis=1).to_hdf('algoseek.h5', 'dividends')


def get_splits():
    split_cols = ['sid', 'effective_date', 'ratio']
    equities = pd.read_hdf('algoseek.h5', 'equities')
    adjustments_con = sqlite3.connect(adj_db_path.as_posix())
    splits = read_sqlite('splits', adjustments_con)[split_cols]
    splits = (splits.rename(columns={'sid': 'quandl_sid'})
              .merge(equities[['quandl_sid', 'sid']])
              .drop('quandl_sid', axis=1)
              )
    print(splits.loc[:, split_cols].info())
    splits.loc[:, split_cols].to_hdf('algoseek.h5', 'splits')


def get_ohlcv_by_ticker():
    equities = pd.read_hdf('algoseek.h5', 'equities')
    col_dict = {'first': 'open', 'last': 'close'}
    nasdaq100 = (pd.read_hdf(data_path / 'data.h5', '1min_trades')
                 .loc[idx[equities.symbol, :], :]
                 .rename(columns=col_dict))
    print(nasdaq100.info())

    symbol_dict = equities.set_index('symbol').sid.to_dict()
    for symbol, data in nasdaq100.groupby(level='ticker'):
        print(symbol)
        data.reset_index('ticker', drop=True).to_hdf('algoseek.h5', '{}'.format(symbol_dict[symbol]))

    equities.drop('quandl_sid', axis=1).to_hdf('algoseek.h5', 'equities')
