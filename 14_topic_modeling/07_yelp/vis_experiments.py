#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
from itertools import zip_longest
import numpy as np
import pandas as pd
from gensim.models import LdaModel, LdaMulticore
from gensim.matutils import Sparse2Corpus
from scipy import sparse
from itertools import product
from time import time
from gensim.corpora import Dictionary
import pyLDAvis
from pyLDAvis.gensim import prepare

np.random.seed(42)


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f'{h:>02.0f}:{m:>02.0f}:{s:>02.0f}'


experiment_path = Path('experiments')
vis_path = Path('ldavis')
if not vis_path.exists():
    vis_path.mkdir(exist_ok=True)

# dtm params
min_dfs = [.001, .005, .01]
max_dfs = [.1, .25, .5, 1.0]
binarys = [True, False]
dtm_params = list(product(*[min_dfs, max_dfs, binarys]))

topics = [3, 5, 7, 10, 15, 20, 25, 50]
passes = 1
start = time()
for i, (min_df, max_df, binary) in enumerate(dtm_params, 1):

    print(min_df, max_df, binary)

    vocab_path = experiment_path / str(min_df) / str(max_df) / str(int(binary))
    try:
        dtm = sparse.load_npz(vocab_path / f'dtm.npz')
        tokens = pd.read_csv(vocab_path / f'tokens.csv', header=None, squeeze=True)
    except FileNotFoundError:
        print('missing')
        continue
    corpus = Sparse2Corpus(dtm, documents_columns=False)
    id2word = tokens.to_dict()
    dictionary = Dictionary.from_corpus(corpus, id2word)

    for num_topics in topics:
        print(num_topics, end=' ')
        model_path = vocab_path / str(num_topics) / str(passes) / 'lda'
        if model_path.exists():
            lda = LdaModel.load(model_path.as_posix())
        else:
            continue
        start = time()
        vis = prepare(lda, corpus, dictionary, mds='tsne')
        terms = vis.topic_info
        terms = terms[terms.Category != 'Default']
        pyLDAvis.save_html(vis, (model_path / 'ldavis.html').as_posix())
        terms.to_csv(model_path / 'relevant_terms.csv', index=False)
        duration = time() - start
        print(format_time(duration))
