#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd

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
                df = pd.read_csv(vocab_path / 'coherence.csv',
                                 header=[0, 1]).stack()
                df.index.names = ['topic', 'passes']
                df = (pd.melt(df.reset_index(),
                              id_vars=['topic', 'passes'],
                              var_name=['num_topics'],
                              value_name='coherence')
                      .dropna()
                      .assign(min_df=min_df,
                              max_df=max_df,
                              binary=binary))
                coherence = pd.concat([coherence,
                                       df])
            except FileNotFoundError:
                print('Missing:', min_df, max_df, binary)

# print(perplexity.info())
print(coherence.info())
with pd.HDFStore('results.h5') as store:
    # store.put('perplexity', perplexity)
    store.put('coherence', coherence)
