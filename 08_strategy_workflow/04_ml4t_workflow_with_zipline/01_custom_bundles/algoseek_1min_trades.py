#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

# create symbolic link to algoseek.h5 in ~/.zipline/custom_data directory
# create symbolic link to algoseek_1min_trades.py in ~/.zipline directory
# in ~/.zipline/:
# ln -s /path/to/machine-learning-for-trading/08_strategy_workflow/04_ml4t_workflow_with_zipline/algoseek_1min_trades.py .
custom_data_path = Path('~/.zipline/custom_data').expanduser()


def load_equities():
    return pd.read_hdf(custom_data_path / 'algoseek.h5', 'equities')


def ticker_generator():
    """
    Lazily return (sid, ticker) tuple
    """
    return (v for v in load_equities().values)


def data_generator():
    for sid, symbol, asset_name in ticker_generator():
        df = (pd.read_hdf(custom_data_path / 'algoseek.h5', str(sid))
              .tz_localize('US/Eastern')
              .tz_convert('UTC'))

        start_date = df.index[0]
        end_date = df.index[-1]

        first_traded = start_date.date()
        auto_close_date = end_date + pd.Timedelta(days=1)
        exchange = 'AlgoSeek'

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


def algoseek_to_bundle(interval='1m'):
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

        def minute_data_generator():
            return (sid_df for (sid_df, *metadata.iloc[sid_df[0]]) in data_generator())

        minute_bar_writer.write(minute_data_generator(), show_progress=True)

        metadata.dropna(inplace=True)
        asset_db_writer.write(equities=metadata)
        adjustment_writer.write(splits=pd.read_hdf(custom_data_path / 'algoseek.h5', 'splits'))
        # dividends do not work
        # adjustment_writer.write(dividends=pd.read_hdf(custom_data_path / 'algoseek.h5', 'dividends'))

    return ingest
