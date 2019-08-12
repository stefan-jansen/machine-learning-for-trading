# Machine Learning for Algorithmic Trading - 1st Edition

This book provides a comprehensive introduction to how ML can add value to trading strategies. It was [published](https://www.amazon.com/Hands-Machine-Learning-Algorithmic-Trading-ebook/dp/B07JLFH7C5/ref=sr_1_2?ie=UTF8&qid=1548455634&sr=8-2&keywords=machine+learning+algorithmic+trading) in January 2019 by [Stefan Jansen](https://www.linkedin.com/in/applied-ai/).

**Update**: I've started working on the **second edition** that will add more end-to-end backtesting examples using zipline to complement the current illustrations how to generate signals using ML models. This should hopefully become available around the end of this year.   

The book provides a comprehensive introduction on how to use ML to add value to trading strategies. It is organized in four parts that cover different aspects of the data sourcing and strategy development process, as well as different solutions to various ML challenges.

Chapters 16-20 of Part IV, Deep and Reinforcement Learning are available as pdfs in the corresponding directories:

- Chapter 16: [Deep Learning](16_deep_learning/16_Deep_Learning.pdf)
- Chapter 17: [Convolutional Neural Networks](17_convolutional_neural_nets/17_Convolutional_Neural_Networks.pdf)
- Chapter 18: [Recurrent Neural Networks](18_recurrent_neural_nets/18_Recurrent_Neural_Networks.pdf)
- Chapter 19: [Autoencoders & GANs](19_deep_unsupervised_learning/19_Autoencoders_and_GANs.pdf)
- Chapter 20: [Reinforcement Learning](20_reinforcement_learning/20_Reinforcement_Learning.pdf)

## What to expect

This book aims to equip you with the strategic perspective, conceptual understanding, and practical tools to add value from applying ML to the trading and investment process. To this end, it covers ML as an important element in a process rather than a standalone exercise.

First and foremost, it illustrates how a broad range of supervised, unsupervised, and reinforcement learning algorithms can be used to extract signals from a diverse set of data sources relevant to different asset classes. It demonstrates how to develop an end-to-end trading strategy and presents ML models as building blocks that extract or combine alpha factors using a systematic workflow. The modular approach implies that not every algorithm is shown as part of a complete strategy. This allows to develop the mathematical and statistical background that in turn facilitate the tuning of an algorithm or the interpretation of the results.

Investors can extract value from third-party data more than other industries. As a consequence, the book covers not only how to work with market and fundamental data but also how to source, evaluate, process, and model alternative data sources such as unstructured text and image data.

It should not be a surprise that this book does not provide investment advice or ready-made trading algorithms. Instead, it presents building blocks required to identify, evaluate, and combine datasets suitable for a given investment objective, to select and apply ML algorithms to this data, and to develop and test algorithmic trading strategies based on the results.

## Who should read this book

You should find the book informative if you are an analyst, data scientist, or ML engineer with an understanding of financial markets and interest in how to apply ML to trading strategies. You should also find value as an investment professional who aims to leverage ML to make better decisions.

If your background is software and ML, you may be able to just skim or skip some introductory material on ML. Similarly, if your expertise is in investment, you will likely be familiar with some or all of the financial context. You will likely find the book most useful as a survey of key algorithms, building blocks and use cases than for specialized coverage of a particular algorithm or strategy. However, the book assumes you are interested in continuing to learn about this very dynamic area. To this end, it references numerous resources to support your journey towards customized trading strategies that leverage and build on the fundamental methods and tools it covers.

You should be comfortable using Python and various scientific computing libraries like `numpy`, `pandas`, or `scipy` and be interested in picking up numerous others along the way. Some experience with ML and `scikit-learn` would be helpful, but we briefly cover the basic workflow and reference various resources to fill gaps or dive deeper.

## Installation and Data Sources

- For instructions on setting up a `conda` environment and installing the packages used in the notebooks, see [here](./installation.md).
- To download and preprocess many of the data sources used in this book see [create_datasets](data/create_datasets.ipynb).

# How the book is organized

It is organized in four parts that cover different aspects of the data sourcing and strategy development process, as well as different solutions to various ML challenges.

## Part 1: How to Design a Trading Strategy

The first part provides a framework for the development of trading strategies driven by machine learning (ML). It focuses on the data that power the ML algorithms and strategies discussed in this book, outlines how ML can be used to derive trading signals, and how to deploy and evaluate strategies as part of a portfolio.

### Chapter 01: Machine Learning for Trading

This [chapter](01_machine_learning_for_trading) summarizes how and why ML became central to investment, describes the trading process and outlines how ML can add value. It covers:

- How to read this book
- The rise of ML in the Investment Industry 
- Design and execution of a trading strategy
- ML and algorithmic trading strategies: use cases

### Chapter 02: Market & Fundamental Data

This [chapter](02_market_and_fundamental_data) introduces market and fundamental data sources and the environment in which they are created. Familiarity with various types of orders and the trading infrastructure matters because they affect backtest simulations of a trading strategy. We also illustrate how to use Python to access and work with trading and financial statement data. 

In particular, this chapter will cover the following topics:
- How market microstructure shapes market data
- How to reconstruct the order book from tick data using Nasdaq ITCH 
- How to summarize tick data using various time, volume and dollar bars
- How to work with eXtensible Business Reporting Language (XBRL)-encoded electronic filings
- How to parse and combine market and fundamental data to create a P/E series
- How to access various market and fundamental data sources using Python

### Chapter 03: Alternative Data for Finance

This [chapter](03_alternative_data) outlines categories and describes criteria to assess the exploding number of alternative data sources and providers. It also demonstrates how to create alternative data sets by scraping websites, for example to collect earnings call transcripts for use with natural language processing (NLP) and sentiment analysis algorithms in the second part of the book. More specifically, this chapter covers:

- How the alternative data revolution has unleashed new sources of information
- How individuals, business processes, and sensors generate alternative data
- How to evaluate the proliferating supply of alternative data used for algorithmic trading
- How to work with alternative data in Python, such as by scraping the internet
- Important categories and providers of alternative data

### Chapter 04: Research & Evaluation of Alpha Factors

[Chapter 4](04_alpha_factor_research) provides a framework for understanding how factors work and how to measure their performance, for example using the information coefficient (IC). It demonstrates how to engineer alpha factors from data using Python libraries offline and on the Quantopian platform. It also introduces the `zipline` library to backtest factors and the `alphalens` library to evaluate their predictive power. More specifically, this chapter covers:

- How to characterize, justify and measure key types of alpha factors
- How to create alpha factors using financial feature engineering
- How to use `zipline` offline to test individual alpha factors
- How to use `zipline` on Quantopian to combine alpha factors and identify more sophisticated signals
- How the information coefficient (IC) measures an alpha factor's predictive performance
- How to use `alphalens` to evaluate predictive performance and turnover
 
### Chapter 05: Strategy Evaluation & Portfolio Management

Testing a strategy requires simulating the portfolios generated by an algorithm to verify its performance under market conditions. Strategy evaluation includes backtesting against historical data to optimize the strategy's parameters, and forward-testing to validate the in-sample performance against new, out-of-sample data and avoid false discoveries from tailoring a strategy to specific past circumstances. This [chapter](05_strategy_evaluation) introduces several approaches to optimizing portfolios that include the application of machine learning (ML) to learn hierarchical relationships among assets. 

More specifically, in this [chapter](05_strategy_evaluation), we cover 
- How to build and test a portfolio based on alpha factors using zipline
- How to measure portfolio risk and return
- How to evaluate portfolio performance using pyfolio
- How to manage portfolio weights using mean-variance optimization and alternatives
- How to use machine learning to optimize asset allocation in a portfolio context

## Part 2: Machine Learning Fundamentals

The second part covers the fundamental supervised and unsupervised learning algorithms and illustrates their application to trading strategies. It also introduces the Quantopian platform where you can leverage and combine the data and ML techniques developed in this book to implement algorithmic strategies that execute trades in live markets.

### Chapter 06: The Machine Learning Process

this [chapter](06_machine_learning_process) sets the stage by outlining how to formulate, train, tune and evaluate the predictive performance of ML models as a systematic workflow. It covers:

- How supervised and unsupervised learning using data works
- How to apply the ML workflow
- How to formulate loss functions for regression and classification
- How to train and evaluate supervised learning models
- How the bias-variance trade-off impacts prediction errors
- How to diagnose and address prediction errors 
- How to train a model using cross-validation to manage the bias-variance trade-off 
- How to implement cross-validation using scikit-learn
- Why the nature of financial data requires different approaches to out-of-sample testing

### Chapter 07: Linear Models for Regression & Classification

Linear models are applied to regression and classification problems with the goals of inference and prediction. Numerous asset pricing models that have been developed by academic and industry researchers leverage linear regression. Applications include the identification of significant factors that drive asset returns, for example, as a basis for risk management, as well as the prediction of returns over various time horizons. Classification problems, on the other hand, include directional price forecasts. [Chapter 07](07_linear_models) covers the following topics:

- How linear regression works and which assumptions it makes
- How to train and diagnose linear regression models
- How to use linear regression to predict future returns
- How use regularization to improve the predictive performance
- How logistic regression works
- How to convert a regression into a classification problem
- How to design a trading algorithm based on price predictions generated by a ML model

### Chapter 08: Linear Time Series Models

 This [chapter](08_time_series_models) focuses on models that extract signals from previously observed data to predict future values for the same time series. The time dimension of trading makes the application of time series models to market, fundamental, and alternative data very popular. 
 
 We present tools to diagnose time series characteristics, including stationarity, and extract features that capture potential patterns. Then it introduces univariate and multivariate time series models and how to apply them to forecast macro data and volatility patterns. It concludes with the concept of cointegration and how to apply it to develop a pairs trading strategy.

In particular, we will cover the following topics:
- How to use time series analysis to diagnose diagnostic statistics that inform the modeling process
- How to estimate and diagnose autoregressive and moving-average time series models
- How to build Autoregressive Conditional Heteroskedasticity (ARCH) models to predict volatility
- How to build vector autoregressive models
- How to use cointegration for a pairs trading strategy

### Chapter 09: Bayesian Machine Learning

This [chapter](09_bayesian_machine_learning) introduces how Bayesian approaches to machine learning add value when developing and evaluating trading strategies due to their different perspective on uncertainty. More specifically, this chapter covers:

- How Bayesian statistics apply to machine learning
- How to use probabilistic programming with PyMC3
- How to define and train machine learning models
- How to run state-of-the-art sampling methods to conduct approximate inference
- How to apply Bayesian machine learning to compute dynamic Sharpe ratios, build Bayesian classifiers, and estimate stochastic volatility

### Chapter 10: Decision Trees & Random Forests

This [chapter](10_decision_trees_random_forests) shows how decision trees and random forests can be used for trading. We will see how decision trees learn rules from data that encodes non-linear relationships between the input and the output variables. We also  introduce ensemble models that combine multiple individual models to produce a single aggregate prediction with lower prediction-error variance. In short, in this chapter, we will cover:
- How to use decision trees for regression and classification
- How to gain insights from decision trees and visualize the decision rules learned from the data
- Why ensemble models tend to deliver superior results
- How bootstrap aggregation addresses the overfitting challenges of decision trees
- How to train, tune, and interpret random forests


### Chapter 11: Gradient Boosting Machines

This [chapter](11_gradient_boosting_machines) explores boosting, an alternative tree-based ensemble algorithm that often produces better results. The key difference is that boosting modifies the data that is used to train each tree based on the cumulative errors made by the model before adding the new tree. In contrast to random forests, which train many trees independently from each other using different versions of the training set, boosting proceeds sequentially using reweighted versions of the data. State-of-the-art boosting implementations also adopt the randomization strategies of random forests. 

More specifically, in this chapter we will cover the following topics:
- How boosting works, and how it compares to bagging
- How boosting has evolved from adaptive to gradient boosting (GB)
- How to use and tune AdaBoost and GB models with sklearn
- How state-of-the-art GB implementations speed up computation
- How to prevent overfitting of GB models
- How to build, tune, and evaluate GB models using xgboost, lightgbm, and catboost
- How to interpret and gain insights from GM models using [SHAP](https://github.com/slundberg/shap) values

### Chapter 12: Unsupervised Learning

Dimensionality reduction and clustering are the main tasks for unsupervised learning: 
- Dimensionality reduction transforms the existing features into a new, smaller set while minimizing the loss of information. A broad range of algorithms exists that differ by how they measure the loss of information, whether they apply linear or non-linear transformations or the constraints they impose on the new feature set. 
- Clustering algorithms identify and group similar observations or features instead of identifying new features. Algorithms differ in how they define the similarity of observations and their assumptions about the resulting groups.

More specifically, this [chapter](12_unsupervised_learning) covers:
- how principal and independent component analysis perform linear dimensionality reduction
- how to apply PCA to identify risk factors and eigen portfolios from asset returns 
- how to use non-linear manifold learning to summarize high-dimensional data for effective visualization
- how to use T-SNE and UMAP to explore high-dimensional alternative image data
- how k-Means, hierarchical, and density-based clustering algorithms work
- how to apply agglomerative clustering to build robust portfolios according to hierarchical risk parity

## Part 3: Natural Language Processing

Text data are rich in content, yet unstructured in format and hence require more preprocessing so that a machine learning algorithm can extract the potential signal. The key challenge consists in converting text into a numerical format for use by an algorithm, while simultaneously expressing the semantics or meaning of the content. We will cover several techniques that capture nuances of language readily understandable to humans so that they can be used as input for machine learning algorithms.

### Chapter 13:	Working with Text Data

This [chapter](13_working_with_text_data) introduces text feature extraction techniques that focus on individual semantic units, i.e. words or short groups of words called tokens. We will show how to represent documents as vectors of token counts by creating a document-term matrix that in turn serves as input for text classification and sentiment analysis. We will also introduce the Naive Bayes algorithm that is popular for this purpose. 

In particular, in this chapter covers:
- What the NLP workflow looks like
- How to build a multilingual feature extraction pipeline using spaCy and Textblob
- How to perform NLP tasks like parts-of-speech tagging or named entity recognition
- How to convert tokens to numbers using the document-term matrix
- How to classify text using the Naive Bayes model
- How to perform sentiment analysis

### Chapter 14:	Topic Modeling

This chapter uses unsupervised learning to model latent topics and extract hidden themes from documents. These themes can produce detailed insights into a large body of documents in an automated way. They are very useful to understand the haystack itself and permit the concise tagging of documents because using the degree of association of topics and documents. 

Topic models permit the extraction of sophisticated, interpretable text features that can be used in various ways to extract trading signals from large collections of documents. They speed up the review of documents, help identify and cluster similar documents, and can be annotated as a basis for predictive modeling. Applications include the identification of key themes in company disclosures or earnings call transcripts, customer reviews or contracts, annotated using, e.g., sentiment analysis or direct labeling with subsequent asset returns. More specifically, this chapter covers:
- What topic modeling achieves, why it matters and how it has evolved
- How Latent Semantic Indexing (LSI) reduces the dimensionality of the DTM
- How probabilistic Latent Semantic Analysis (pLSA) uses a generative model to extract topics
- How Latent Dirichlet Allocation (LDA) refines pLSA and why it is the most popular topic model
- How to visualize and evaluate topic modeling results
- How to implement LDA using sklearn and gensim
- How to apply topic modeling to collections of earnings calls and Yelp business reviews

### Chapter 15:	Word Vector Embeddings

This chapter introduces uses neural networks to learn a vector representation of individual semantic units like a word or a paragraph. These vectors are dense rather than sparse as in the bag-of-words model and have a few hundred real-valued rather than tens of thousand binary or discrete entries. They are called embeddings because they assign each semantic unit a location in a continuous vector space.
 
Embeddings result from training a model to relate tokens to their context with the benefit that similar usage implies a similar vector. As a result, the embeddings encode semantic aspects like relationships among words by means of their relative location. They are powerful features for use in the deep learning models that we will introduce in the following chapters. More specifically, in this chapter, we will cover:
- What word embeddings are, how they work and capture semantic information
- How to use trained word vectors
- Which network architectures are useful to train word2vec models
- How to train a word2vec model using keras, gensim, and TensorFlow
- How to visualize and evaluate the quality of word vectors
- How to train a word2vec model using SEC filings
- How doc2vec extends word2vec


## Part 4: Deep & Reinforcement Learning

### Chapter 16:	Deep Learning

The chapter presents feedforward neural networks (NN) to demonstrate how to efficiently train large models using backpropagation, and manage the risks of overfitting. It also shows how to use of the frameworks Keras, TensorFlow 2.0, and PyTorch.

In the following chapters, we will build on this foundation to design and train a variety of architectures suitable for different investment applications with a particular focus on alternative data sources. These include recurrent NN tailored to sequential data like time series or natural language and convolutional NN particularly well suited to image data. We will also cover deep unsupervised learning, including Generative Adversarial Networks (GAN) to create synthetic data and reinforcement learning to train agents that interactively learn from their environment. In particular, this chapter will cover
- How DL solves AI challenges in complex domains
- How key innovations have propelled DL to its current popularity
- How feed-forward networks learn representations from data
- How to design and train deep neural networks in Python
- How to implement deep NN using Keras, TensorFlow, and PyTorch
- How to build and tune a deep NN to predict asset price moves

### Chapter 17:	Convolutional Neural Networks

CNNs are named after the linear algebra operation called convolution that replaces the general matrix multiplication typical of feed-forward networks. Research into CNN architectures has proceeded very rapidly and new architectures that improve performance on some benchmark continue to emerge frequently. CNNs are designed to learn hierarchical feature representations from grid-like data. One of their shortcomings is that they do not learn spatial relationships, i.e., the relative positions of these features. In the last section, we will outline how Capsule Networks work that have emerged to overcome these limitations. 

More specifically, this [chapter](17_convolutional_neural_nets) covers

- How CNNs use key building blocks to efficiently model grid-like data
- How to design CNN architectures using Keras and PyTorch
- How to train, tune and regularize CNN for various data types
- How to use transfer learning to streamline CNN, even with fewer data
- How Capsule Networks improve on CNN and may enable a new wave of innovation


### Chapter 18:	Recurrent Neural Networks

The major innovation of RNN is that each output is a function of both previous output and new data. As a result, RNN gain the ability to incorporate information on previous observations into the computation it performs on a new feature vector, effectively creating a model with memory. This recurrent formulation enables parameter sharing across a much deeper computational graph that includes cycles. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that aim to overcome the challenge of vanishing gradients associated with learning long-range dependencies, where errors need to be propagated over many connections. 

RNNs have been successfully applied to various tasks that require mapping one or more input sequences to one or more output sequences and are particularly well suited to natural language. RNN can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in [Chapter 15](15_word_embeddings) to classify the sentiment expressed in documents. Most specifically, this chapter addresses:
- How to unroll and analyze the computational graph for an RNN
- How gated units learn to regulate an RNN’s memory from data to enable long-range dependencies
- How to design and train RNN for univariate and multivariate time series in Python
- How to leverage word embeddings for sentiment analysis with RNN


### Chapter 19:	Autoencoders & Generative Adversarial Networks

This [chapter](19_deep_unsupervised_learning) presents two unsupervised learning techniques that leverage deep learning: autoencoders, which have been around for decades, and Generative Adversarial Networks (GANs), which were introduced by Ian Goodfellow in 2014 and which Yann LeCun has called the most exciting idea in AI in the last ten years. 
- An autoencoder is a neural network trained to reproduce the input while learning a new representation of the data, encoded by the parameters of a hidden layer. Autoencoders have long been used for nonlinear dimensionality reduction and manifold learning. More recently, autoencoders have been designed as generative models that learn probability distributions over observed and latent variables. A variety of designs leverage the feedforward network, Convolutional Neural Network (CNN), and recurrent neural network (RNN) architectures we covered in the last three chapters.
- GANs are a recent innovation that train two neural nets—a generator and a discriminator—in a competitive setting. The generator aims to produce samples that the discriminator is unable to distinguish from a given class of training data. The result is a generative model capable of producing new (fake) samples that are representative of a certain target distribution. GANs have produced a wave of research and can be successfully applied in many domains. An example from the medical domain that could potentially be highly relevant for trading is the generation of time-series data that simulates alternative trajectories and can be used to train supervised or reinforcement algorithms.

More specifically, this chapter covers:

- Which types of autoencoders are of practical use and how they work
- How to build and train autoencoders using Python
- How GANs work, why they're useful, and how they could be applied to trading
- How to build GANs using Python


### Chapter 20:	Reinforcement Learning

Reinforcement Learning (RL) is a computational approach to goal-directed learning performed by an agent that interacts with a typically stochastic environment which the agent has incomplete information about. RL aims to automate how the agent makes decisions to achieve a long-term objective by learning the value of states and actions from a reward signal. The ultimate goal is to derive a policy that encodes behavioral rules and maps states to actions.

This [chapter](20_reinforcement_learning) shows how to formulate an RL problem and how to apply various solution methods. It covers model-based and model-free methods, introduces the [OpenAI Gym](https://gym.openai.com/) environment, and combines deep learning with RL to train an agent that navigates a complex environment. Finally, we'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts with the financial market while trying to optimize an objective function. 

More specifically,this chapter will cover:

- How to define a Markov Decision Problem (MDP)
- How to use Value and Policy Iteration to solve an MDP
- How to apply Q-learning in an environment with discrete states and actions
- How to build and train a deep Q-learning agent in a continuous environment
- How to use OpenAI Gym to train an RL trading agent