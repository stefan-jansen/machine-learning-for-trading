#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

from pathlib import Path
from time import time
import warnings
from collections import Counter
import logging

import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

ANALOGIES_PATH = Path().cwd().parent / 'data' / 'analogies' / 'analogies-en.txt'


def eval_analogies(w2v, max_vocab=15000):
    accuracy = w2v.wv.accuracy(ANALOGIES_PATH,
                               restrict_vocab=max_vocab,
                               case_insensitive=True)
    return (pd.DataFrame([[c['section'],
                           len(c['correct']),
                           len(c['incorrect'])] for c in accuracy],
                         columns=['category', 'correct', 'incorrect'])
            .assign(average=lambda x:
    x.correct.div(x.correct.add(x.incorrect)))).fillna(0)


model_path = Path('models', 'trial_5')
accuracies = pd.DataFrame()
totals = {}
for model_file in model_path.glob('*.bin'):
    _, size = model_file.stem.split('_')
    model = KeyedVectors.load_word2vec_format(model_file.as_posix(),
                                              binary=True,
                                              unicode_errors='ignore')
    accuracy = eval_analogies(model).set_index('category')
    total = (accuracy.loc['total',
                          ['correct', 'incorrect']]
             .sum().astype(int))
    totals[size] = total
    print(size, '\t', f"{total:,d} | {accuracy.loc['total', 'average']:.2%}")

    accuracies[size] = accuracy.average
totals = pd.Series(totals)
print(totals)
totals.to_csv(model_path / 'totals.csv')
accuracies.to_csv(model_path / 'accuracies.csv')
