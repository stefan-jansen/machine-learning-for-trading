#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
from itertools import zip_longest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import LdaModel, LdaMulticore
from gensim.matutils import Sparse2Corpus
from scipy import sparse
from itertools import product
from random import shuffle
from time import time
import logging

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

logging.basicConfig(
        filename='gensim.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f'{h:>02.0f}:{m:>02.0f}:{s:>02.0f}'


# experiment setup
cols = ['vocab_size', 'test_vocab', 'min_df', 'max_df', 'binary', 'num_topics', 'passes', 'perplexity']
experiment_path = Path('experiments')

docs = Path('clean_reviews.txt').read_text().split('\n')
shuffle(docs)
print('\n', len(docs))
# train_docs, test_docs = train_test_split(docs, test_size=.1)

# dtm params
min_dfs = [.001, .005, .01]
max_dfs = [.1, .25, .5, 1.0]
binarys = [True, False]
dtm_params = list(product(*[min_dfs, max_dfs, binarys]))
n = len(dtm_params)
shuffle(dtm_params)

topicss = [3, 5, 7, 10, 15, 20, 25, 50]
passess = [1]
model_params = list(product(*[topicss, passess]))

# corpus = id2word = train_corpus = train_tokens = test_corpus = vocab_size = test_vocab = None
start = time()
for i, (min_df, max_df, binary) in enumerate(dtm_params, 1):
    print(min_df, max_df, binary)

    vocab_path = experiment_path / str(min_df) / str(max_df) / str(int(binary))
    coherence_path = vocab_path / 'coherence.csv'
    perplexity_path = vocab_path / 'result.csv'
    if all([coherence_path.exists(), perplexity_path.exists()]):
        continue
    if not vocab_path.exists():
        vocab_path.mkdir(exist_ok=True, parents=True)
    dtm_path = vocab_path / f'dtm.npz'
    token_path = vocab_path / f'tokens.csv'
    start = time()
    if all([dtm_path.exists() and token_path.exists()]):
        print('Loading vectorized docs')
        dtm = sparse.load_npz(vocab_path / f'dtm.npz')
        tokens = pd.read_csv(vocab_path / f'tokens.csv', header=None, squeeze=True)
        print('Loading done', format_time(time() - start))
    else:
        print('Vectorizing docs')
        vectorizer = CountVectorizer(min_df=min_df,
                                     max_df=max_df,
                                     binary=binary)
        dtm = vectorizer.fit_transform(docs)
        tokens = pd.Series(vectorizer.get_feature_names())
        sparse.save_npz(dtm_path, dtm)
        tokens.to_csv(token_path, index=False)
        print('Vectorizing done', format_time(time() - start))

    corpus = Sparse2Corpus(dtm, documents_columns=False)
    id2word = tokens.to_dict()
    vocab_size = len(tokens)

    assert vocab_size == dtm.shape[1], print(dtm.shape, vocab_size)

    train_dtm, test_dtm = train_test_split(dtm, test_size=.1)
    assert vocab_size == train_dtm.shape[1] == test_dtm.shape[1], \
        print(vocab_size, train_dtm.shape[1], test_dtm.shape[1])
    assert train_dtm.shape[0] + test_dtm.shape[0] == dtm.shape[0]
    train_corpus = Sparse2Corpus(train_dtm, documents_columns=False)
    test_corpus = Sparse2Corpus(test_dtm, documents_columns=False)
    timing = []
    for workers in [8, 16]:
        for num_topics in [10, 50]:
            print('start', workers, num_topics, end=' ')
            start = time()
            lda = LdaMulticore(corpus=train_corpus,
                               num_topics=num_topics,
                               id2word=id2word,
                               chunksize=1000,
                               passes=1,
                               eval_every=None,
                               workers=workers,
                               random_state=42)
            duration = time() - start
            test_perplexity = 2 ** (-lda.log_perplexity(test_corpus))
            timing.append([workers, num_topics, duration, test_perplexity])
            print(format_time(duration), test_perplexity)
            pd.DataFrame(timing, columns=['workers',
                                          'num_topics',
                                          'duration',
                                          'test_perplexity']).to_csv(f'timings_{workers}.csv', index=False)
    exit()

    test_vocab = test_dtm.count_nonzero()
    perplexity, coherence = [], []
    for num_topics, passes in model_params:
        model_path = vocab_path / str(num_topics) / str(passes)
        if not model_path.exists():
            model_path.mkdir(exist_ok=True, parents=True)
        print((num_topics, passes), end=' ', flush=True)
        lda = LdaMulticore(corpus=train_corpus,
                           num_topics=num_topics,
                           id2word=id2word,
                           passes=passes,
                           eval_every=None,
                           workers=72,
                           random_state=42)
        test_perplexity = 2 ** (-lda.log_perplexity(test_corpus))
        lda.update(corpus=test_corpus)
        lda.save((model_path / 'lda').resolve().as_posix())

        topic_coherence = lda.top_topics(corpus=corpus, coherence='u_mass', topn=20)
        coherence.append([c[1] for c in topic_coherence])

        perplexity.append([vocab_size, test_vocab, min_df, max_df,
                           binary, num_topics, passes, test_perplexity])

    elapsed = time() - start
    print(f'\nDone: {i / n:.2%} | Duration: {format_time(elapsed)} | To Go: {format_time(elapsed / i * (n - i))}\n')
    perplexity = pd.DataFrame(perplexity, columns=cols).sort_values('perplexity')
    print(perplexity)
    perplexity.to_csv(perplexity_path, index=False)
    pd.DataFrame((_ for _ in zip_longest(*coherence))).to_csv(coherence_path, index=False)
