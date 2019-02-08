#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
from scipy import sparse
from itertools import product
from random import shuffle
from time import time
import spacy
import logging

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)
nlp = spacy.load('en')

logging.basicConfig(
        filename='gensim.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f'{h:>02.0f}:{m:>02.0f}:{s:>02.0f}'


clean_text = Path('clean_text.txt')

# experiment setup
cols = ['vocab_size', 'test_vocab', 'min_df', 'max_df', 'binary', 'num_topics', 'passes', 'perplexity']
experiment_path = Path('experiments')

# get text files
clean_docs = clean_text.read_text().split('\n')

print('\n', len(clean_docs))
train_docs, test_docs = train_test_split(clean_docs, test_size=.1)

# dtm params
min_dfs = [50, 100, 250, 500]
max_dfs = [.1, .25, .5, 1.0]
binarys = [True, False]
dtm_params = list(product(*[min_dfs, max_dfs, binarys]))
n = len(dtm_params)
shuffle(dtm_params)

topicss = [3, 5, 7, 10, 15, 20, 25, 50]
passess = [1, 25]
model_params = list(product(*[topicss, passess]))

corpus = id2word = train_corpus = train_tokens = test_corpus = vocab_size = test_vocab = None
start = time()
for i, (min_df, max_df, binary) in enumerate(dtm_params, 1):
    print(min_df, max_df, binary)
    result = []

    vocab_path = experiment_path / str(min_df) / str(max_df) / str(int(binary))
    if vocab_path.exists():
        continue
    else:
        vocab_path.mkdir(exist_ok=True, parents=True)
        vectorizer = CountVectorizer(min_df=min_df,
                                     max_df=max_df,
                                     binary=binary)
        train_dtm = vectorizer.fit_transform(train_docs)
        train_corpus = Sparse2Corpus(train_dtm, documents_columns=False)
        train_tokens = vectorizer.get_feature_names()

        test_dtm = vectorizer.transform(test_docs)
        test_corpus = Sparse2Corpus(test_dtm, documents_columns=False)
        test_vocab = test_dtm.count_nonzero()

        dtm = vectorizer.fit_transform(clean_docs)
        sparse.save_npz(vocab_path / f'dtm.npz', dtm)
        tokens = vectorizer.get_feature_names()
        vocab_size = len(tokens)
        pd.Series(tokens).to_csv(vocab_path / f'tokens.csv', index=False)

        id2word = pd.Series(tokens).to_dict()
        corpus = Sparse2Corpus(dtm, documents_columns=False)

    coherence = pd.DataFrame()
    for num_topics, passes in model_params:
        model_path = vocab_path / str(num_topics) / str(passes)
        if not model_path.exists():
            model_path.mkdir(exist_ok=True, parents=True)
        print((num_topics, passes), end=' ', flush=True)
        lda = LdaModel(corpus=corpus,
                       num_topics=num_topics,
                       id2word=id2word,
                       passes=passes,
                       eval_every=None,
                       random_state=42)

        doc_topics = pd.DataFrame()
        model_file = (model_path / 'lda').resolve().as_posix()
        lda.save(model_file)
        train_lda = LdaModel(corpus=train_corpus,
                             num_topics=num_topics,
                             id2word=pd.Series(train_tokens).to_dict(),
                             passes=passes,
                             eval_every=None,
                             random_state=42)

        test_perplexity = 2 ** (-train_lda.log_perplexity(test_corpus))
        coherence = pd.concat([coherence, (pd.Series([c[1] for c in lda.top_topics(corpus=corpus,
                                                                                   coherence='u_mass',
                                                                                   topn=20)])
                                           .to_frame((num_topics, passes)))], axis=1)
        result.append([vocab_size,
                       test_vocab,
                       min_df,
                       max_df,
                       binary,
                       num_topics,
                       passes,
                       test_perplexity])

    elapsed = time() - start
    print(f'\nDone: {i / n:.2%} | Duration: {format_time(elapsed)} | To Go: {format_time(elapsed / i * (n - i))}\n')
    results = pd.DataFrame(result, columns=cols).sort_values('perplexity')
    print(results.head(10))
    results.to_csv(vocab_path / 'perplexity.csv', index=False)
    coherence.to_csv(vocab_path / 'coherence.csv', index=False)
