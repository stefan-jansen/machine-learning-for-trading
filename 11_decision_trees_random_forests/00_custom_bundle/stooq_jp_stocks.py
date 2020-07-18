#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import os
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


zipline_root = None

try:
    zipline_root = os.environ['ZIPLINE_ROOT']
except KeyError:
    print('Please ensure a ZIPLINE_ROOT environment variable is defined and accessible '
          '(or alter the script and manually set the path')
    exit()

custom_data_path = Path(zipline_root, 'custom_data')

# custom_data_path = Path('~/.zipline/custom_data').expanduser()


def load_equities():
    return pd.read_hdf(custom_data_path / 'stooq.h5', 'jp/equities')


def ticker_generator():
    """
    Lazily return (sid, ticker) tuple
    """
    return (v for v in load_equities().values)


def data_generator():
    for sid, symbol, asset_name in ticker_generator():
        df = pd.read_hdf(custom_data_path / 'stooq.h5', 'jp/{}'.format(sid))

        start_date = df.index[0]
        end_date = df.index[-1]

        first_traded = start_date.date()
        auto_close_date = end_date + pd.Timedelta(days=1)
        exchange = 'XTKS'

        yield (sid, df), symbol, asset_name, start_date, end_date, first_traded, auto_close_date, exchange


def metadata_frame():
    dtype = [
        ('symbol', 'object'),
        ('asset_name', 'object'),
        ('start_date', 'datetime64[ns]'),
        ('end_date', 'datetime64[ns]'),
        ('first_traded', 'datetime64[ns]'),
        ('auto_close_date', 'datetime64[ns]'),
        ('exchange', 'object'), ]
    return pd.DataFrame(np.empty(len(load_equities()), dtype=dtype))


def stooq_jp_to_bundle(interval='1d'):
    def ingest(environ,
               asset_db_writer,
               minute_bar_writer,
               daily_bar_writer,
               adjustment_writer,
               calendar,
               start_session,
               end_session,
               cache,
               show_progress,
               output_dir
               ):
        metadata = metadata_frame()

        def daily_data_generator():
            return (sid_df for (sid_df, *metadata.iloc[sid_df[0]]) in data_generator())

        daily_bar_writer.write(daily_data_generator(), show_progress=True)

        metadata.dropna(inplace=True)
        asset_db_writer.write(equities=metadata)
        # empty DataFrame
        adjustment_writer.write(splits=pd.read_hdf(custom_data_path / 'stooq.h5', 'jp/splits'))

    return ingest
