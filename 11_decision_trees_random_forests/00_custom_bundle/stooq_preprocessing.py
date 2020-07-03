#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = Path('..', '..', 'data')
idx = pd.IndexSlice


def create_split_table():
    with pd.HDFStore('stooq.h5') as store:
        store.put('jp/splits', pd.DataFrame(columns=['sid', 'effective_date', 'ratio'],
                                            data=[[1, pd.to_datetime('2010-01-01'), 1.0]]), format='t')


def load_prices():
    df = pd.read_hdf(DATA_DIR / 'assets.h5', 'stooq/jp/tse/stocks/prices')

    return (df.loc[idx[:, '2014': '2019'], :]
            .unstack('ticker')
            .sort_index()
            .tz_localize('UTC')
            .ffill(limit=5)
            .dropna(axis=1)
            .stack('ticker')
            .swaplevel())


def load_symbols(tickers):
    df = pd.read_hdf(DATA_DIR / 'assets.h5', 'stooq/jp/tse/stocks/tickers')
    return (df[df.ticker.isin(tickers)]
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index': 'sid'}))


if __name__ == '__main__':
    prices = load_prices()
    print(prices.info(null_counts=True))
    tickers = prices.index.unique('ticker')

    symbols = load_symbols(tickers)
    print(symbols.info(null_counts=True))
    symbols.to_hdf('stooq.h5', 'jp/equities', format='t')

    dates = prices.index.unique('date')
    start_date = dates.min()
    end_date = dates.max()

    for sid, symbol in symbols.set_index('sid').symbol.items():
        p = prices.loc[symbol]
        p.to_hdf('stooq.h5', 'jp/{}'.format(sid), format='t')

    with pd.HDFStore('stooq.h5') as store:
        print(store.info())

    create_split_table()
