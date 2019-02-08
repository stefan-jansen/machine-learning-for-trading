#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
from collections import Counter
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.callbacks import Callback, TensorBoard

np.random.seed(42)

LANGUAGES = ['en', 'es']

SOURCES = ['ted', 'euro']
SOURCE_LABELS = ['Ted', 'Europarliament']
source_dict = dict(zip(SOURCES, SOURCE_LABELS))

parser = ArgumentParser(description='Run Keras word2vec model')
parser.add_argument('-l', '--language', choices=LANGUAGES, help='language', default='en')
parser.add_argument('-s', '--source', choices=SOURCES, help='data source', default='euro')
parser.add_argument('-m', '--model', choices=[1, 2, 3], help='model', default=1)

args = parser.parse_args()
LANGUAGE = args.language
MODEL = 'ngrams_{}'.format(args.model)
SOURCE = source_dict[args.source]

PROJECT_DIR = Path('/home/stefan/projects/odsc_2018/word2vec-translation')


def get_vocab_stats():
    with pd.HDFStore(Path('vocab', SOURCE, 'vocab.h5').name) as store:
        df = store['{}/vocab'.format(LANGUAGE)]

    wc = df['count'].value_counts().sort_index(ascending=False).reset_index()
    wc.columns = ['word_count', 'freq']
    wc['n_words'] = wc.word_count.mul(wc.freq)

    wc['corpus_share'] = wc.n_words.div(wc.n_words.sum())
    wc['coverage'] = wc.corpus_share.cumsum()
    wc['vocab_size'] = wc.freq.cumsum()
    return wc


# wc = get_vocab_stats()
# print('# words: {:,d}'.format(wc.n_words.sum()))
# print(wc.loc[:, ['word_count', 'freq', 'n_words', 'vocab_size', 'coverage']].tail(10))

### Model Settings
MIN_FREQ = 5
WINDOW_SIZE = 5
EMBEDDING_SIZE = 300
EPOCHS = 1
BATCH_SIZE = 100

PATH = Path('.', SOURCE, LANGUAGE, MODEL)
TB_PATH = PATH / 'tensorboard'
if not TB_PATH.exists():
    TB_PATH.mkdir(parents=True, exist_ok=True)

VALID_SIZE = 15  # Random set of words to evaluate similarity on.
VALID_WINDOW = 250  # Evaluation samples from most frequent words
NN = 10  # Nearest neighbors for evaluation

valid_examples = np.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)


def build_data(language, ngrams=1):
    path = PROJECT_DIR / 'vocab' / SOURCE / language / 'ngrams_{}.txt'.format(ngrams)
    words = path.read_text().split()

    token_counts = [t for t in Counter(words).most_common() if t[1] >= MIN_FREQ]
    tokens, counts = list(zip(*token_counts))

    id_to_token = pd.Series(tokens).to_dict()
    id_to_token.update({-1: 'UNK'})
    token_to_id = {t: i for i, t in id_to_token.items()}
    data = [token_to_id.get(word, -1) for word in words]
    return data, token_to_id, id_to_token


data, token_to_id, id_to_token = build_data(LANGUAGE, ngrams=1)

vocab_size = len(token_to_id) - 1


def save_meta(d):
    s = pd.Series(d).value_counts().reset_index()
    s.columns = ['id', 'count']
    s['token'] = s.id.map(id_to_token)
    s[s.id >= 0].sort_values('id').token.dropna().to_csv(TB_PATH / 'meta.tsv', index=False)


save_meta(data)


# #### Process Analogies
def get_analogies(lang):
    analogies = pd.read_csv(Path('..', 'data', 'analogies', 'analogies-{}.txt'.format(lang)),
                            header=None, names=['analogies'], squeeze=True)
    cats = analogies.apply(lambda x: x if x.startswith(':') else np.nan).ffill().str.strip(':').str.strip().to_frame(
            'cats')
    analogies = analogies[~analogies.str.startswith(':')].str.split(expand=True)
    analogies.columns = list('abcd')
    analogies = cats.merge(analogies, left_index=True, right_index=True)
    df['cats'], idx = pd.factorize(df.cats)
    return analogies, pd.Series(idx)


analogies, categories = get_analogies('en')
analogies_id = analogies.apply(lambda x: x.map(token_to_id))

test_set = analogies_id.dropna().astype(int)
a, b, c, actual = test_set.values.T
actual = actual.reshape(-1, 1)
n_analogies = len(actual)

sampling_table = sequence.make_sampling_table(vocab_size)

couples, labels = skipgrams(sequence=data,
                            vocabulary_size=vocab_size,
                            window_size=WINDOW_SIZE,
                            sampling_table=sampling_table,
                            negative_samples=1.0,
                            shuffle=True)

target_word, context_word = np.array(couples, dtype=np.int32).T
labels = np.array(labels, dtype=np.int8)
del couples

with pd.HDFStore(PATH / 'data.h5') as store:
    store.put('id_to_token', pd.Series(id_to_token))
    store.put('analogies', test_set)


def model_graph():
    #### Scalar Input Variables
    input_target = Input((1,), name='target_input')
    input_context = Input((1,), name='context_input')

    #### Shared Embedding Layer
    embedding = Embedding(input_dim=vocab_size,
                          output_dim=EMBEDDING_SIZE,
                          input_length=1,
                          name='embedding_layer')

    #### Select Embedding Vectors
    target = embedding(input_target)
    target = Reshape((EMBEDDING_SIZE, 1), name='target_embedding')(target)

    context = embedding(input_context)
    context = Reshape((EMBEDDING_SIZE, 1), name='context_embedding')(context)

    #### Compute Similarity (not normalized)
    dot_product = Dot(axes=1)([target, context])
    dot_product = Reshape((1,), name='similarity')(dot_product)

    #### Sigmoid Output Layer
    output = Dense(units=1, activation='sigmoid', name='output')(dot_product)

    # #### Training Model
    model = Model(inputs=[input_target, input_context], outputs=output)

    # Validation Model (Cosine Similarity)
    similarity = Dot(normalize=True,
                     axes=1,
                     name='cosine_similarity')([target, context])
    valid_model = Model(inputs=[input_target, input_context], outputs=similarity)

    return model, valid_model, embedding


train_model, valid_model, embedding = model_graph()
train_model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print(train_model.summary())
print(valid_model.summary())


#### Evaluation: Nearest Neighors & Analogies
class EvalCallback(Callback):
    def on_train_begin(self, logs=None):
        print('\n\t{} nearest neighbors:'.format(NN))
        for i in range(VALID_SIZE):
            valid_word = id_to_token[valid_examples[i]]
            sim = self._get_similiarity(valid_examples[i]).reshape(-1)
            nearest = (-sim).argsort()[1:NN + 1]
            neighbors = [id_to_token[nearest[n]] for n in range(NN)]
            print('\t\t{}: {}'.format(valid_word, ', '.join(neighbors)))

    def on_train_end(self, logs=None):
        print('\n\t{} nearest neighbors:'.format(NN))
        for i in range(VALID_SIZE):
            valid_word = id_to_token[valid_examples[i]]
            sim = self._get_similiarity(valid_examples[i]).reshape(-1)
            nearest = (-sim).argsort()[1:NN + 1]
            neighbors = [id_to_token[nearest[n]] for n in range(NN)]
            print('\t\t{}: {}'.format(valid_word, ', '.join(neighbors)))

    def on_epoch_end(self, eppch, logs=None):
        print('\n\tAnalogy Accuracy:\n\t\t', end='')
        print(self.test_analogies())

    @staticmethod
    def test_analogies():
        embeddings = embedding.get_weights()[0]
        target = embeddings[c] + embeddings[b] - embeddings[a]
        neighbors = np.argsort(cdist(target, embeddings, metric='cosine'))
        match_id = np.argwhere(neighbors == actual)[:, 1]
        return '\n\t\t'.join(['Top {}: {:.2%}'.format(i, (match_id < i).sum() / n_analogies) for i in [1, 5, 10]])

    @staticmethod
    def _get_similiarity(valid_word_idx):
        target = np.full(shape=vocab_size, fill_value=valid_word_idx)
        context = np.arange(vocab_size)
        return valid_model.predict([target, context])


evaluation = EvalCallback()

# ##### Tensorboard Callback
tensorboard = TensorBoard(log_dir=str(TB_PATH), histogram_freq=0,
                          batch_size=32, write_graph=True,
                          embeddings_freq=100,
                          embeddings_metadata=str((TB_PATH / 'meta.tsv').resolve()))

loss = train_model.fit(x=[target_word, context_word], y=labels,
                       shuffle=True,
                       batch_size=BATCH_SIZE, epochs=EPOCHS,
                       callbacks=[evaluation, tensorboard])

train_model.save(str(PATH / 'skipgram_model.h5'))
