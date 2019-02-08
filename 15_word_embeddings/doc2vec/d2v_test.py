#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import logging
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

logging.basicConfig(
        filename='test.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')

df = pd.read_csv('yelp_sample.csv')


def train_model():
    sentences = []
    for i, (stars, text) in df.iterrows():
        sentences.append(TaggedDocument(words=text.split(), tags=[i]))

    print('start training')
    model = Doc2Vec(vector_size=300, window=5, min_count=5, workers=8, epochs=1)
    print('build vocab')
    model.build_vocab(sentences)

    print('keep training')

    for epoch in range(10):
        print(epoch, end=' ', flush=True)
        shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    print(model.most_similar('good'))
    # save model
    model.save('test.model')


model = Doc2Vec.load('test.model')
X = np.zeros(shape=(len(df), 300))
y = np.zeros(shape=len(df))
for i in range(len(df)):
    X[i] = model[i]
    y[i] = df.loc[i, 'stars']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log = LogisticRegression()
rf = RandomForestClassifier(n_jobs=-1, n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mode = pd.Series(y_train).mode()
print(accuracy_score(y_true=y_test, y_pred=y_pred))
print(accuracy_score(y_true=np.full_like(y_test, fill_value=mode), y_pred=y_pred))
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
