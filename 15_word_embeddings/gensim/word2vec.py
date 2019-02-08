# coding: utf-8

from pathlib import Path
from argparse import ArgumentParser
from time import time
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

np.random.seed(42)

LANGUAGES = ['en', 'es']

SOURCES = ['ted', 'euro']
SOURCE_LABELS = ['Ted', 'Europarliament']
source_dict = dict(zip(SOURCES, SOURCE_LABELS))

parser = ArgumentParser(description='Run Keras word2vec model')
parser.add_argument('-l', '--language', choices=LANGUAGES, help='language', default='es')
parser.add_argument('-s', '--source', choices=SOURCES, help='data source', default='euro')
parser.add_argument('-m', '--model', type=int, choices=[1, 2, 3], help='model', default=1)

args = parser.parse_args()
LANGUAGE = args.language
MODEL = 'ngrams_{}'.format(args.model)
SOURCE = source_dict[args.source]

PROJECT_DIR = Path('/home/stefan/projects/odsc_2018/word2vec-translation')

print('\nLanguage: {} | Source: {} | Model: {}'.format(LANGUAGE, SOURCE, MODEL))


def get_accuracy(acc, detail=False):
    results = [[c['section'], len(c['correct']), len(c['incorrect'])] for c in acc]
    results = pd.DataFrame(results, columns=['category', 'correct', 'incorrect'])
    results['average'] = results.correct.div(results[['correct', 'incorrect']].sum(1))
    results.sort_values('average', ascending=False)
    if detail:
        print(results)
    return results.iloc[-1, 1:].tolist()


ANALOGIES_PATH = PROJECT_DIR / 'data' / 'analogies' / 'analogies-{}.txt'.format(LANGUAGE)
gensim_path = PROJECT_DIR / 'gensim' / SOURCE / LANGUAGE / MODEL
if not gensim_path.exists():
    gensim_path.mkdir(parents=True, exist_ok=True)

sentence_path = PROJECT_DIR / 'vocab' / SOURCE / LANGUAGE / 'ngrams_{}.txt'.format(1)
sentences = LineSentence(str(sentence_path))
start = time()
model = Word2Vec(sentences,
                 sg=1,
                 size=300,
                 window=5,
                 min_count=5,
                 negative=10,
                 workers=8,
                 iter=1,
                 alpha=0.05)

print('Duration: {:,.1f}s'.format(time() - start))

model.wv.save(str(gensim_path / 'word_vectors.bin'))

acc = get_accuracy(model.wv.accuracy(str(ANALOGIES_PATH), case_insensitive=True))
print('Base Accuracy: Correct {:,d} | Wrong {:,d} | Avg {:,.2%}\n'.format(*acc))

accuracies = [acc]
for i in range(1, 11):
    start = time()
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    accuracies.append(get_accuracy(model.wv.accuracy(str(ANALOGIES_PATH))))
    print('{} | Duration: {:,.1f} | Accuracy: {:.2%} '.format(i, time() - start, accuracies[-1][-1]))

pd.DataFrame(accuracies, columns=['correct', 'wrong', 'average']).to_csv(gensim_path / 'accuracies.csv', index=False)
model.wv.save(str(gensim_path / 'word_vectors_final.bin'))
