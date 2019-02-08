#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd


def timing_results():
    path = Path('timings')
    df = pd.concat([pd.read_csv(f) for f in path.glob('*.csv')])
    print(df.info())
    print(df.sort_values('workers'))


timing_results()
exit()


def experiment_results():
    experiment_path = Path('experiments')

    # dtm params
    min_dfs = [50, 100, 250, 500]
    max_dfs = [.1, .25, .5, 1.0]
    binarys = [True, False]

    perplexity = pd.DataFrame()
    coherence = pd.DataFrame()
    for min_df in min_dfs:
        for max_df in max_dfs:
            for binary in binarys:
                vocab_path = experiment_path / str(min_df) / str(max_df) / str(int(binary))
                try:
                    # perplexity = pd.concat([perplexity,
                    #                         pd.read_csv(vocab_path / 'perplexity.csv')])
                    df = (pd.melt(pd.read_csv(vocab_path / 'coherence.csv',
                                              header=[0, 1]),
                                  var_name=['num_topics', 'passes'],
                                  value_name='coherence')
                          .dropna()
                          .assign(min_df=min_df,
                                  max_df=max_df,
                                  binary=binary))

                    coherence = pd.concat([coherence,
                                           df])
                except FileNotFoundError:
                    print('Missing:', min_df, max_df, binary)

    with pd.HDFStore('results.h5') as store:
        store.put('perplexity', perplexity)
        store.put('coherence', coherence)
