# Chapter 18: Recurrent Neural Networks

The major innovation of RNN is that each output is a function of both previous output and new data. As a result, RNN gain the ability to incorporate information on previous observations into the computation it performs on a new feature vector, effectively creating a model with memory. This recurrent formulation enables parameter sharing across a much deeper computational graph that includes cycles. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that aim to overcome the challenge of vanishing gradients associated with learning long-range dependencies, where errors need to be propagated over many connections. 

RNNs have been successfully applied to various tasks that require mapping one or more input sequences to one or more output sequences and are particularly well suited to natural language. RNN can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in [Chapter 15](15_word_embeddings) to classify the sentiment expressed in documents. Most specifically, this chapter addresses:
- How to unroll and analyze the computational graph for an RNN
- How gated units learn to regulate an RNN’s memory from data to enable long-range dependencies
- How to design and train RNN for univariate and multivariate time series in Python
- How to leverage word embeddings for sentiment analysis with RNN


## How RNN work

RNNs assume that data is sequential so that previous data points impact the current observation and are relevant for predictions of subsequent elements in the sequence.
They allow for more complex and diverse input-output relationships than feedforward networks (FFNN) and convolutional nets that are designed to map one input to one output vector, usually of fixed size and using a given number of computational steps. RNN, in contrast, can model data for tasks where the input, the output or both are best represented as a sequence of vectors. 

Note that input and output sequences can be of arbitrary lengths because the recurrent transformation that is fixed but learned from the data can be applied as many times as needed.
Just as CNN easily scale to large images and some CNN can process images of variable size, RNN scale to much longer sequences than networks not tailored to sequence-based tasks. Most RNN can also process sequences of variable length.

### Backpropagation through Time

 RNNs are called recurrent because they apply the same transformations to every element of a sequence in a way that the output depends on the outcome of prior iterations. As a result, RNNs maintain an internal state that captures information about previous elements in the sequence akin to a memory.

The backpropagation algorithm that updates the weight parameters based on the gradient of the loss function with respect to the parameters involves a forward pass from left to right along the unrolled computational graph, followed by backward pass in the opposite direction.

- [Sequence Modeling: Recurrent and Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), Deep Learning Book, Chapter 10, Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf), Alex Graves, 2013
- [Tutorial on LSTM Recurrent Networks](http://people.idsia.ch/~juergen/lstm/sld001.htm), Juergen Schmidhuber, 2003
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Alternative RNN Architectures

RNNs can be designed in a variety of ways to best capture the functional relationship and dynamic between input and output data. In addition to the recurrent connections between the hidden states, there are several alternative approaches, including recurrent output relationships, bidirectional RNN, and encoder-decoder architectures.

### Long-Short Term Memory

RNNs with an LSTM architecture have more complex units that maintain an internal state and contain gates to keep track of dependencies between elements of the input sequence and regulate the cell’s state accordingly. These gates recurrently connect to each other instead of the usual hidden units we encountered above. They aim to address the problem of vanishing and exploding gradients by letting gradients pass through unchanged.

A typical LSTM unit combines four parameterized layers that interact with each other and the cell state by transforming and passing along vectors. These layers usually involve an input gate, an output gate, and a forget gate, but there are variations that may have additional gates or lack some of these mechanisms

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), Christopher Olah, 2015
- [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf), Rafal Jozefowicz, Ilya Sutskever, et al, 2015

### Gated Recurrent Units

Gated recurrent units (GRU) simplify LSTM units by omitting the output gate. They have been shown to achieve similar performance on certain language modeling tasks but do better on smaller datasets.

- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), Kyunghyun Cho, Yoshua Bengio, et al 2014
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555), Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio, 2014

## How to build and train an RNN using Python

We illustrate how to build RNN using the Keras library for various scenarios. The first set of models includes regression and classification of univariate and multivariate time series. The second set of tasks focuses on text data for sentiment analysis using text data converted to word embeddings (see [Chapter 15](../15_word_embeddings)). 

- [Keras documentation](https://keras.io/getting-started/sequential-model-guide/)
- [LSTM documentation](https://keras.io/layers/recurrent/)
- [Keras-recommended approach for RNNs](https://keras.io/optimizers/) (use RMSProp)


### Univariate Time Series Regression

The notebook [univariate_time_series_regression](01_univariate_time_series_regression.ipynb) demonstrates how to get data into the requisite shape and how to forecast the S&P 500 index values using a Recurrent Neural Network. 

### Stacked LSTMs for time series classification

We'll now build a slightly deeper model by stacking two LSTM layers using the Quandl stock price data (see the [stacked_lstm_with_feature_embeddings](02_stacked_lstm_with_feature_embeddings.ipynb) notebook for implementation details). Furthermore, we will include features that are not sequential in nature, namely indicator variables that identify the ticker and time periods like month and year.

### Multivariate Time Series Regression

So far, we have limited our modeling efforts to single time series. RNNs are naturally well suited to multivariate time series and represent a non-linear alternative to the Vector Autoregressive (VAR) models we covered in [Chapter 8, Time Series Models](../08_time_series_models).

The notebook [multivariate_timeseries](03_multivariate_timeseries.ipynb) demonstrates the application of RNNs to modeling and forecasting several time series using the same dataset we used for the [VAR example](../08_time_series_models/03_vector_autoregressive_model.ipynb), namely monthly data on consumer sentiment, and industrial production from the Federal Reserve's FRED service.

### LSTM & Word Embeddings for Sentiment Classification

RNNs are commonly applied to various natural language processing tasks. We've already encountered sentiment analysis using text data in part three of [this book](https://www.amazon.com/Hands-Machine-Learning-Algorithmic-Trading-ebook/dp/B07JLFH7C5/ref=sr_1_2?ie=UTF8&qid=1548455634&sr=8-2&keywords=machine+learning+algorithmic+trading).

The notebook [sentiment_analysis](04_sentiment_analysis.ipynb) illustrates how to apply an RNN model to text data to detect positive or negative sentiment (which can easily be extended to a finer-grained sentiment scale). We are going to use word embeddings to represent the tokens in the documents. We covered word embeddings in [Chapter 15, Word Embeddings](../15_word_embeddings). They are an excellent technique to convert text into a continuous vector representation such that the relative location of words in the latent space encodes useful semantic aspects based on the words' usage in context.

In this example, we again use Keras' built-in embedding layer that allows us to train vector representations specific to the task at hand. In the next example, we use pretrained vectors instead.


### How to use pre-trained word embeddings

In [Chapter 15, Word Embeddings](../15_word_embeddings), we showed how to learn domain-specific word embeddings. Word2vec, and related learning algorithms, produce high-quality word vectors, but require large datasets. Hence, it is common that research groups share word vectors trained on large datasets, similar to the weights for pretrained deep learning models that we encountered in the section on transfer learning in the [previous chapter](../17_convolutional_neural_nets).

The notebook [sentiment_analysis_pretrained_embeddings](05_sentiment_analysis_pretrained_embeddings.ipynb) illustrates how to use pretrained Global Vectors for Word Representation (GloVe) provided by the Stanford NLP group with the IMDB review dataset.

- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), Stanford AI Group
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/), Stanford NLP