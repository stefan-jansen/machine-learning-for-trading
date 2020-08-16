# RNN for Trading: Multivariate Time Series and Text Data

The major innovation of RNN is that each output is a function of both previous output and new data. As a result, RNN gain the ability to incorporate information on previous observations into the computation it performs on a new feature vector, effectively creating a model with memory. This recurrent formulation enables parameter sharing across a much deeper computational graph that includes cycles. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that aim to overcome the challenge of vanishing gradients associated with learning long-range dependencies, where errors need to be propagated over many connections. 

RNNs have been successfully applied to various tasks that require mapping one or more input sequences to one or more output sequences and are particularly well suited to natural language. RNN can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in [Chapter 16](16_word_embeddings) to classify the sentiment expressed in documents.

## Content

1. [How recurrent neural nets work](#how-recurrent-neural-nets-work)
    * [Backpropagation through Time](#backpropagation-through-time)
    * [Alternative RNN Architectures](#alternative-rnn-architectures)
        - [Long-Short Term Memory](#long-short-term-memory)
        - [Gated Recurrent Units](#gated-recurrent-units)
2. [RNN for financial time series with TensorFlow 2](#rnn-for-financial-time-series-with-tensorflow-2)
    * [Code example: Univariate time-series regression: predicting the S&P 500](#code-example-univariate-time-series-regression-predicting-the-sp-500)
    * [Code example: Stacked LSTM for predicting weekly stock price moves and returns](#code-example-stacked-lstm-for-predicting-weekly-stock-price-moves-and-returns)
    * [Code example: Predicting returns instead of directional price moves](#code-example-predicting-returns-instead-of-directional-price-moves)
    * [Code example: Multivariate time-series regression for macro data](#code-example-multivariate-time-series-regression-for-macro-data)
3. [RNN for text data: sentiment analysis and return prediction](#rnn-for-text-data-sentiment-analysis-and-return-prediction)
    * [Code example: LSTM with custom word embeddings for sentiment classification](#code-example-lstm-with-custom-word-embeddings-for-sentiment-classification)
    * [Code example: Sentiment analysis with pretrained word vectors](#code-example-sentiment-analysis-with-pretrained-word-vectors)
    * [Code example: SEC filings for a bidirectional RNN GRU to predict weekly returns](#code-example-sec-filings-for-a-bidirectional-rnn-gru-to-predict-weekly-returns)

## How recurrent neural nets work

RNNs assume that the input data has been generated as a sequence such that previous data points impact the current observation and are relevant for predicting subsequent values. Thus, they allow for more complex input-output relationships than FFNNs and CNNs, which are designed to map one input vector to one output vector using a given number of computational steps. 
RNNs, in contrast, can model data for tasks where the input, the output, or both, are best represented as a sequence of vectors. 

For a thorough overview, see [chapter 10](https://www.deeplearningbook.org/contents/rnn.html in [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville (2016).

### Backpropagation through Time

 RNNs are called recurrent because they apply the same transformations to every element of a sequence in a way that the output depends on the outcome of prior iterations. As a result, RNNs maintain an internal state that captures information about previous elements in the sequence akin to a memory.

The backpropagation algorithm that updates the weight parameters based on the gradient of the loss function with respect to the parameters involves a forward pass from left to right along the unrolled computational graph, followed by backward pass in the opposite direction.

- [Sequence Modeling: Recurrent and Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), Deep Learning Book, Chapter 10, Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf), Alex Graves, 2013
- [Tutorial on LSTM Recurrent Networks](http://people.idsia.ch/~juergen/lstm/sld001.htm), Juergen Schmidhuber, 2003
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Alternative RNN Architectures

RNNs can be designed in a variety of ways to best capture the functional relationship and dynamic between input and output data. In addition to the recurrent connections between the hidden states, there are several alternative approaches, including recurrent output relationships, bidirectional RNN, and encoder-decoder architectures.

#### Long-Short Term Memory

RNNs with an LSTM architecture have more complex units that maintain an internal state and contain gates to keep track of dependencies between elements of the input sequence and regulate the cell’s state accordingly. These gates recurrently connect to each other instead of the usual hidden units we encountered above. They aim to address the problem of vanishing and exploding gradients by letting gradients pass through unchanged.

A typical LSTM unit combines four parameterized layers that interact with each other and the cell state by transforming and passing along vectors. These layers usually involve an input gate, an output gate, and a forget gate, but there are variations that may have additional gates or lack some of these mechanisms

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), Christopher Olah, 2015
- [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf), Rafal Jozefowicz, Ilya Sutskever, et al, 2015

#### Gated Recurrent Units

Gated recurrent units (GRU) simplify LSTM units by omitting the output gate. They have been shown to achieve similar performance on certain language modeling tasks but do better on smaller datasets.

- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), Kyunghyun Cho, Yoshua Bengio, et al 2014
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555), Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio, 2014

## RNN for financial time series with TensorFlow 2

We illustrate how to build RNN using the Keras library for various scenarios. The first set of models includes regression and classification of univariate and multivariate time series. The second set of tasks focuses on text data for sentiment analysis using text data converted to word embeddings (see [Chapter 15](../15_word_embeddings)). 

- [Recurrent Neural Networks (RNN) with Keras](https://www.tensorflow.org/guide/keras/rnn)
- [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras documentation](https://keras.io/getting-started/sequential-model-guide/)
- [LSTM documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Working with RNNs](https://keras.io/guides/working_with_rnns/) by Scott Zhu and Francois Chollet

### Code example: Univariate time-series regression: predicting the S&P 500

The notebook [univariate_time_series_regression](01_univariate_time_series_regression.ipynb) demonstrates how to get data into the requisite shape and how to forecast the S&P 500 index values using a Recurrent Neural Network. 

### Code example: Stacked LSTM for predicting weekly stock price moves and returns

We'll now build a slightly deeper model by stacking two LSTM layers using the Quandl stock price data. Furthermore, we will include features that are not sequential in nature, namely indicator variables that identify the ticker and time periods like month and year.
- See the [stacked_lstm_with_feature_embeddings](02_stacked_lstm_with_feature_embeddings.ipynb) notebook for implementation details.

### Code example: Predicting returns instead of directional price moves

The notebook [stacked_lstm_with_feature_embeddings_regression](03_stacked_lstm_with_feature_embeddings_regression.ipynb) illustrates how to adapt the model to the regression task of predicting returns rather than binary price changes.

### Code example: Multivariate time-series regression for macro data

So far, we have limited our modeling efforts to single time series. RNNs are naturally well suited to multivariate time series and represent a non-linear alternative to the Vector Autoregressive (VAR) models we covered in [Chapter 9, Time Series Models](../09_time_series_models).

The notebook [multivariate_timeseries](04_multivariate_timeseries.ipynb) demonstrates the application of RNNs to modeling and forecasting several time series using the same dataset we used for the [VAR example](../09_time_series_models/04_vector_autoregressive_model.ipynb), namely monthly data on consumer sentiment, and industrial production from the Federal Reserve's FRED service.

## RNN for text data: sentiment analysis and return prediction

### Code example: LSTM with custom word embeddings for sentiment classification

RNNs are commonly applied to various natural language processing tasks. We've already encountered sentiment analysis using text data in part three of [this book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=VMKJPZC4N36TTZZCWATP&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=8f331266-0d21-4c76-a3eb-d2e61d23bb31&pd_rd_w=kVGNF&pd_rd_wg=LYLKH&ref_=pd_gw_ci_mcx_mr_hp_d).

This example shows how to learn custom embedding vectors while training an RNN on the classification task. This differs from the word2vec model that learns vectors while optimizing predictions of neighboring tokens, resulting in their ability to capture certain semantic relationships among words (see Chapter 16). Learning word vectors with the goal of predicting sentiment implies that embeddings will reflect how a token relates to the outcomes it is associated with.

The notebook [sentiment_analysis_imdb](05_sentiment_analysis_imdb.ipynb) illustrates how to apply an RNN model to text data to detect positive or negative sentiment (which can easily be extended to a finer-grained sentiment scale). We are going to use word embeddings to represent the tokens in the documents. We covered word embeddings in [Chapter 15, Word Embeddings](../15_word_embeddings). They are an excellent technique to convert text into a continuous vector representation such that the relative location of words in the latent space encodes useful semantic aspects based on the words' usage in context.

### Code example: Sentiment analysis with pretrained word vectors

In [Chapter 15, Word Embeddings](../15_word_embeddings), we showed how to learn domain-specific word embeddings. Word2vec, and related learning algorithms, produce high-quality word vectors, but require large datasets. Hence, it is common that research groups share word vectors trained on large datasets, similar to the weights for pretrained deep learning models that we encountered in the section on transfer learning in the [previous chapter](../17_convolutional_neural_nets).

The notebook [sentiment_analysis_pretrained_embeddings](06_sentiment_analysis_pretrained_embeddings.ipynb) illustrates how to use pretrained Global Vectors for Word Representation (GloVe) provided by the Stanford NLP group with the IMDB review dataset.

- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), Stanford AI Group
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/), Stanford NLP

### Code example: SEC filings for a bidirectional RNN GRU to predict weekly returns

In Chapter 16, we discussed important differences between product reviews and financial text data. While the former was useful to illustrate important workflows, in this section, we will tackle more challenging but also more relevant financial documents. 

More specifically, we will use the SEC filings data introduced in [Chapter 16](../16_word_embeddings) to learn word embeddings tailored to predicting the return of the ticker associated with the disclosures from before publication to one week after.

The notebook [sec_filings_return_prediction](07_sec_filings_return_prediction.ipynb) contains the code examples for this application. 

See the notebook [sec_preprocessing](../16_word_embeddings/06_sec_preprocessing.ipynb) in Chapter 16 and instructions in the data folder on GitHub on how to obtain the data.
