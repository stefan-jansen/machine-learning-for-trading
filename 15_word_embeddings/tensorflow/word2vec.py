"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'. These need to be compiled.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using true SGD.

"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
import threading
from math import sqrt
from os import environ
from pathlib import Path
from time import time, sleep

import numpy as np
import pandas as pd

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin
from tensorflow.contrib.tensorboard.plugins import projector


PROJECT_DIR = Path().cwd().resolve()
VOCAB_DIR = PROJECT_DIR / 'vocab'
DATA_DIR = PROJECT_DIR / 'data'

# Load custom ops
word2vec = tf.load_op_library(str(PROJECT_DIR / 'tensorflow' / 'word2vec_ops.so'))

# Define command line options
flags = tf.app.flags
flags.DEFINE_string('language', 'en', 'Document language.')
flags.DEFINE_string('file', 'ngrams_1', 'Input doc.')
flags.DEFINE_string('source', 'TED', 'Data source.')
flags.DEFINE_integer('epochs_to_train', 1, 'Number of epochs to train. ')
flags.DEFINE_integer('embedding_size', 200, 'The embedding dimension size.')
flags.DEFINE_float('starter_lr', 0.05, 'Initial learning rate.')
flags.DEFINE_float('target_lr', 0.05, 'Final learning rate.')
flags.DEFINE_integer('num_neg_samples', 10, 'Negative samples per training example.')
flags.DEFINE_integer('batch_size', 500, 'No. of training examples each step processes (no minibatching).')
flags.DEFINE_integer('concurrent_steps', 8, 'The number of concurrent training steps.')
flags.DEFINE_integer('window_size', 5, "No of words to predict to the left and right of target.")
flags.DEFINE_integer('min_count', 5, 'Minimum no of occurrences for word to enter vocabulary.')
flags.DEFINE_float('subsample', 1e-3, 'Words with higher frequency will be randomly down-sampled. 0 to disable.')
flags.DEFINE_boolean('custom_freq', False, 'Use language-specific subsample threshold.')
flags.DEFINE_integer('words_to_project', 10000, 'Words to project using Tensorboard.')
FLAGS = flags.FLAGS


def time_diff(t):
    m, s = divmod(time() - t, 60)
    h, m = divmod(m, 60)
    return ['{:0>2.0f}'.format(x) for x in [h, m, s]]


class Options(object):
    """Flags to options used by word2vec model."""

    def __init__(self):
        self.freq_map = dict(en=0.003,
                             es=0.014)  # custom subsample thresholds to filter out 0.1% most frequent words
        self.lang = FLAGS.language
        self.file = FLAGS.file
        self.source = FLAGS.source
        self.emb_dim = FLAGS.embedding_size
        self.train_data = Path(VOCAB_DIR, self.source, self.lang, self.file + '.txt')
        self.num_samples = FLAGS.num_neg_samples
        self.starter_lr = FLAGS.starter_lr
        self.target_lr = FLAGS.target_lr
        self.epochs_to_train = FLAGS.epochs_to_train
        self.concurrent_steps = FLAGS.concurrent_steps
        self.batch_size = FLAGS.batch_size
        self.window_size = FLAGS.window_size
        self.min_count = FLAGS.min_count
        self.custom_freq = FLAGS.custom_freq
        if self.custom_freq:
            self.subsample = self.freq_map[self.lang]
        else:
            self.subsample = FLAGS.subsample
        self.words_to_project = FLAGS.words_to_project
        self.save_path = Path(PROJECT_DIR, self.source, self.lang, self.file , '{}_{}_{}_{}_{}_{}_{}'.format(self.emb_dim, self.num_samples, int(self.starter_lr * 100), int(self.target_lr * 100), self.batch_size, self.window_size, self.min_count))
        self.tensor_board_path = self.save_path / 'tensorboard'
        if not self.tensor_board_path.exists():
            self.tensor_board_path.mkdir(parents=True, exist_ok=True)

        self.analogy_path = DATA_DIR / 'analogies' / 'analogies-{}.txt'.format(self.lang)


class Word2Vec(object):
    """Word2Vec model (Skipgram)."""

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()
        self.accuracies = []

    def read_analogies(self):
        """Reads through the analogy question file.

        Returns:
          questions: a [n, 4] numpy array containing the analogy question's word ids.
          questions_skipped: questions skipped due to unknown words.
        """
        questions = []
        questions_skipped = 0
        with open(self._options.analogy_path, 'rb') as analogy_f:
            for line in analogy_f:
                if line.startswith(b':'):  # Skip comments.
                    continue
                words = line.strip().lower().split()
                ids = [self._word2id.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print('\nEval analogy file: ', self._options.analogy_path.stem)
        print('Questions: ', len(questions))
        print('Skipped: ', questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)

    def build_graph(self):
        """Build the model graph."""
        opts = self._options

        # The training data. A text file.
        with tf.name_scope('input'):
            words, counts, words_per_epoch, current_epoch, total_words_processed, center_words, target_words = \
                word2vec.skipgram_word2vec(filename=str(opts.train_data),
                                           batch_size=opts.batch_size,
                                           window_size=opts.window_size,
                                           min_count=opts.min_count,
                                           subsample=opts.subsample)

        opts.vocab_words, opts.vocab_counts, opts.words_per_epoch = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)

        print('Data file: ', opts.file)
        print('Vocab size: {:,} + UNK'.format(opts.vocab_size - 1))
        print('Words per epoch: {:,}'.format(opts.words_per_epoch))

        self._id2word = opts.vocab_words
        self._word2id = {w: i for i, w in enumerate(self._id2word)}

        # Input words embedding: [vocab_size, emb_dim]
        with tf.name_scope('embedding'):
            embeddings = tf.Variable(
                    tf.random_uniform([opts.vocab_size, opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
                    name='vectors')
            tf.summary.histogram('histogram', embeddings)

        with tf.name_scope('output'):
            embed = tf.nn.embedding_lookup(embeddings, center_words)
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                        tf.truncated_normal([opts.vocab_size, opts.emb_dim], stddev=1.0 / sqrt(opts.emb_dim)),
                        name='weights')
                tf.summary.histogram('histogram', nce_weights)
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([opts.vocab_size]), name='biases')
                tf.summary.histogram('histogram', nce_biases)

        with tf.name_scope('nce_loss'):
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=tf.reshape(target_words, shape=(-1, 1)),
                                   inputs=embed,
                                   num_sampled=opts.num_samples,
                                   num_classes=opts.vocab_size), name='loss')
            ema = tf.train.ExponentialMovingAverage(decay=0.99)
            update_loss_ema = ema.apply([loss])
            loss_ema = ema.average(loss)
            tf.summary.scalar('exp_moving_avg', loss_ema)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
            learning_rate = opts.starter_lr * tf.maximum(opts.target_lr, 1.0 - tf.cast(total_words_processed,
                                                                                       tf.float32) / words_to_train)
            tf.summary.scalar('learning_rate', learning_rate)

            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)

        self._update_loss_ema = update_loss_ema
        self._loss_ema = loss_ema
        self._embeddings = embeddings
        self._learning_rate = learning_rate
        self._train_step = train_step
        self.global_step = global_step
        self._epoch = current_epoch
        self._words = total_words_processed

    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options

        vocab = []
        for i in range(opts.vocab_size):
            vocab.append([tf.compat.as_text(opts.vocab_words[i]).encode('utf-8'), opts.vocab_counts[i]])
        vocab = pd.DataFrame(vocab, columns=['word', 'count'])
        vocab.word = vocab.word.astype(str)

        with pd.HDFStore(opts.save_path / 'results.h5') as store:
            store.put('/'.join([opts.lang, 'vocab']), vocab, format='t')

        meta = vocab.word.str[1:].str.strip("'").iloc[:opts.words_to_project]
        meta.to_csv(opts.tensor_board_path / 'metadata.tsv', header=None, sep='\t', index=False)

    def build_eval_graph(self):
        """Build the evaluation graph."""
        # Eval graph
        opts = self._options

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.
        with tf.name_scope('eval'):
            # The eval feeds three vectors of word ids for a, b, c, each of
            # which is of size N, where N is the number of analogies we want to
            # evaluate in one batch.
            analogy_a = tf.placeholder(dtype=tf.int32, name='analogy_a')  # [N]
            analogy_b = tf.placeholder(dtype=tf.int32, name='analogy_b')  # [N]
            analogy_c = tf.placeholder(dtype=tf.int32, name='analogy_c')  # [N]

            # Normalized word embeddings of shape [vocab_size, emb_dim].
            nemb = tf.nn.l2_normalize(self._embeddings, 1)

            # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
            # They all have the shape [N, emb_dim]
            a_emb = tf.gather(nemb, analogy_a, name='analogy_a_emb')  # a's embs
            b_emb = tf.gather(nemb, analogy_b, name='analogy_a_emb')  # b's embs
            c_emb = tf.gather(nemb, analogy_c, name='analogy_a_emb')  # c's embs

            # We expect that d's embedding vectors on the unit hyper-sphere is
            # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
            target = c_emb + (b_emb - a_emb)

            # Compute cosine distance between each pair of target and vocab.
            # dist has shape [N, vocab_size].
            dist = tf.matmul(target, nemb, transpose_b=True, name='target_dist')

            # For each question (row in dist), find the top 4 words.
            _, pred_idx = tf.nn.top_k(dist, 4)

            # Nodes for computing neighbors for a given word according to
            # their cosine distance.
            nearby_word = tf.placeholder(dtype=tf.int32, name='nearby_word')  # word id
            nearby_emb = tf.gather(nemb, nearby_word, name='nearby_emb')
            nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True, name='nearby_dist')
            nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, opts.vocab_size), name='k_nn')

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

        self._merged = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter(str(opts.tensor_board_path / 'train'), graph=self._session.graph)
        self._test_writer = tf.summary.FileWriter(str(opts.tensor_board_path / 'test'), graph=self._session.graph)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, _, epoch = self._session.run([self._update_loss_ema, self._train_step, self._epoch])
            if epoch != initial_epoch:
                break

    def train(self):
        """Train the model."""
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        workers = []
        for _ in range(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time, start = initial_words, time(), time()
        rates = []
        while True:
            sleep(2)  # Reports our progress once a while.
            epoch, step, words, loss_ema, summary = self._session.run(
                    [self._epoch, self.global_step, self._words, self._loss_ema, self._merged])
            self._train_writer.add_summary(summary=summary, global_step=step)

            now = time()
            last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
            rates.append(rate)

            h, m, s = time_diff(start)
            print(
                    '\r{}:{}:{}\tEpoch {:>2}\tStep {:>8,}\twords/sec: {:>8,.0f}\tloss: {:>8.4f}'.format(
                            h, m, s, epoch, step, sum(rates) / len(rates), loss_ema), end='')
            sys.stdout.flush()

            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()

    def _predict(self, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx

    def eval(self):
        """Evaluate analogy questions and reports accuracy."""

        # How many questions we get right at precision@1.
        correct = 0

        try:
            total = self._analogy_questions.shape[0]
        except AttributeError:
            raise AttributeError('Need to read analogy questions.')

        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in range(sub.shape[0]):
                for j in range(4):
                    if idx[question, j] == sub[question, 3]:
                        # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracies', simple_value=correct / total), ])
        step = self.global_step.eval(self._session)
        self._test_writer.add_summary(summary, global_step=step)

        self.accuracies.append(correct / total)

        print('\n\tEval {:4d}/{}\t\taccuracy = {:.1%}'.format(correct, total, correct / total))

    def analogy(self, w0, w1, w2):
        """Predict word w3 as in w0:w1 vs w2:w3."""
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                print(c)
                break
        print('unknown')

    def nearby(self, words, num=20):
        """Prints out nearby words given a list of words."""
        ids = np.array([self._word2id.get(x, 0) for x in words])
        vals, idx = self._session.run(
                [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in range(len(words)):
            print("\n%s\n=====================================" % (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                print("%-20s %6.4f" % (self._id2word[neighbor], distance))


def main(_):
    """Train a word2vec model."""
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device('/cpu:0'):
            model = Word2Vec(opts, session)
            model.read_analogies()  # Read analogy questions
        for e in range(opts.epochs_to_train):
            print()
            model.train()  # Process one epoch
            model.eval()  # Eval analogies.


        model.saver.save(session, opts.tensor_board_path / 'model.ckpt', global_step=model.global_step)
        final_embeddings = tf.nn.l2_normalize(model._embeddings, 1).eval(session=session)

        writer = tf.summary.FileWriter(str(opts.tensor_board_path), session.graph)

        embedding_var = tf.Variable(final_embeddings[:opts.words_to_project], name='embedding_result')
        session.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        embedded = config.embeddings.add()
        embedded.tensor_name = embedding_var.name
        embedded.metadata_path = str((opts.tensor_board_path / 'metadata.tsv').resolve())
        projector.visualize_embeddings(writer, config)

        tf.global_variables_initializer().run()
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(session, opts.tensor_board_path / 'skip-gram.ckpt', 1)

        params = dict(emb_dim=opts.emb_dim,
                      num_samples=opts.num_samples,
                      starter_lr=opts.starter_lr,
                      target_lr=opts.target_lr,
                      epochs_to_train=opts.epochs_to_train,
                      window_size=opts.window_size,
                      min_count=opts.min_count,
                      subsample=opts.subsample)

        with pd.HDFStore(opts.save_path / 'results.h5') as store:
            store.put('/'.join([opts.lang, 'embeddings']), pd.DataFrame(final_embeddings))
            store.put('/'.join([opts.lang, 'accuracies']), pd.Series(model.accuracies))
            store.put('/'.join([opts.lang, 'params']), pd.Series(params))


if __name__ == "__main__":
    tf.app.run()
