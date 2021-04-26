# ML for Trading - 2<sup>nd</sup> Edition

This [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d) aims to show how ML can add value to algorithmic trading strategies in a practical yet comprehensive way. It covers a broad range of ML techniques from linear regression to deep reinforcement learning and demonstrates how to build, backtest, and evaluate a trading strategy driven by model predictions.  

In four parts with **23 chapters plus an appendix**, it covers on **over 800 pages**:
- important aspects of data sourcing, **financial feature engineering**, and portfolio management, 
- the design and evaluation of long-short **strategies based on supervised and unsupervised ML algorithms**,
- how to extract tradeable signals from **financial text data** like SEC filings, earnings call transcripts or financial news,
- using **deep learning** models like CNN and RNN with market and alternative data, how to generate synthetic data with generative adversarial networks, and training a trading agent using deep reinforcement learning

<p align="center">
<a href="https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d">
<img src="https://ml4t.s3.amazonaws.com/assets/cover_toc_gh.png" width="75%">
</a>
</p>

This repo contains **over 150 notebooks** that put the concepts, algorithms, and use cases discussed in the book into action. They provide numerous examples that show:
- how to work with and extract signals from market, fundamental and alternative text and image data, 
- how to train and tune models that predict returns for different asset classes and investment horizons, including how to replicate recently published research, and 
- how to design, backtest, and evaluate trading strategies.

> We **highly recommend** reviewing the notebooks while reading the book; they are usually in an executed state and often contain additional information not included due to space constraints.  

In addition to the information in this repo, the book's [website](ml4trading.io) contains chapter summary and additional information.

## Join the ML4T Community!

To make it easy for readers to ask questions about the book's content and code examples, as well as the development and implementation of their own strategies and industry developments, we are hosting an online [platform](https://exchange.ml4trading.io/).

Please [join](https://exchange.ml4trading.io/) our community and connect with fellow traders interested in leveraging ML for trading strategies, share your experience, and learn from each other! 

## What's new in the 2<sup>nd</sup> Edition?

First and foremost, this [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=VMKJPZC4N36TTZZCWATP&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=8f331266-0d21-4c76-a3eb-d2e61d23bb31&pd_rd_w=kVGNF&pd_rd_wg=LYLKH&ref_=pd_gw_ci_mcx_mr_hp_d) demonstrates how you can extract signals from a diverse set of data sources and design trading strategies for different asset classes using a broad range of supervised, unsupervised, and reinforcement learning algorithms. It also provides relevant mathematical and statistical knowledge to facilitate the tuning of an algorithm or the interpretation of the results. Furthermore, it covers the financial background that will help you work with market and fundamental data, extract informative features, and manage the performance of a trading strategy.

From a practical standpoint, the 2nd edition aims to equip you with the conceptual understanding and tools to develop your own ML-based trading strategies. To this end, it frames ML as a critical element in a process rather than a standalone exercise, introducing the end-to-end ML for trading workflow from data sourcing, feature engineering, and model optimization to strategy design and backtesting.

More specifically, the ML4T workflow starts with generating ideas for a well-defined investment universe, collecting relevant data, and extracting informative features. It also involves designing, tuning, and evaluating ML models suited to the predictive task. Finally, it requires developing trading strategies to act on the models' predictive signals, as well as simulating and evaluating their performance on historical data using a backtesting engine. Once you decide to execute an algorithmic strategy in a real market, you will find yourself iterating over this workflow repeatedly to incorporate new information and a changing environment.

<p align="center">
<img src="https://i.imgur.com/kcgItgp.png" width="75%">
</p>

The [second edition](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d)'s emphasis on the ML4t workflow translates into a new chapter on [strategy backtesting](08_ml4t_workflow), a new [appendix](24_alpha_factor_library) describing over 100 different alpha factors, and many new practical applications. We have also rewritten most of the existing content for clarity and readability. 

The trading applications now use a broader range of data sources beyond daily US equity prices, including international stocks and ETFs. It also demonstrates how to use ML for an intraday strategy with minute-frequency equity data. Furthermore, it extends the coverage of alternative data sources to include SEC filings for sentiment analysis and return forecasts, as well as satellite images to classify land use. 

Another innovation of the second edition is to replicate several trading applications recently published in top journals: 
- [Chapter 18](18_convolutional_neural_nets) demonstrates how to apply convolutional neural networks to time series converted to image format for return predictions based on [Sezer and Ozbahoglu](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach) (2018). 
- [Chapter 20](20_autoencoders_for_conditional_risk_factors) shows how to extract risk factors conditioned on stock characteristics for asset pricing using autoencoders based on [Autoencoder Asset Pricing Models](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) by Shihao Gu, Bryan T. Kelly, and Dacheng Xiu (2019), and 
- [Chapter 21](21_gans_for_synthetic_time_series) shows how to create synthetic training data using generative adversarial networks based on [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar (2019).

All applications now use the latest available (at the time of writing) software versions such as pandas 1.0 and TensorFlow 2.2. There is also a customized version of Zipline that makes it easy to include machine learning model predictions when designing a trading strategy.


## Installation, data sources and bug reports

The code examples rely on a wide range of Python libraries from the data science and finance domains. To facilitate installation, we use [Docker](https://www.docker.com/get-started) to provide containerized [conda](https://docs.conda.io/en/latest/) environments.

> Update April 2021: with the update of [Zipline](https://zipline.ml4trading.io), it is no longer necessary to use Docker. The installation instructions now refer to OS-specific environment files that should simplify your running of the notebooks.

> Update Februar 2021: code sample release 2.0 updates the conda environments provided by the Docker image to Python 3.8, Pandas 1.2, and TensorFlow 1.2, among others; the Zipline backtesting environment with now uses Python 3.6.

- The [installation](installation/README.md) directory contains detailed instructions on setting up and using a Docker image to run the notebooks. It also contains configuration files for setting up various `conda` environments and install the packages used in the notebooks directly on your machine if you prefer (and, depending on your system, are prepared to go the extra mile).
- To download and preprocess many of the data sources used in this book, see the instructions in the [README](data/README.md) file alongside various notebooks in the [data](data) directory.

> If you have any difficulties installing the environments, downloading the data or running the code, please raise a **GitHub issue** in the repo ([here](https://github.com/stefan-jansen/machine-learning-for-trading/issues)). Working with GitHub issues has been described [here](https://guides.github.com/features/issues/).

> **Update**: You can download the **[algoseek](https://www.algoseek.com)** data used in the book [here](https://www.algoseek.com/ml4t-book-data.html). See instructions for preprocessing in [Chapter 2](02_market_and_fundamental_data/02_algoseek_intraday/README.md) and an intraday example with a gradient boosting model in [Chapter 12](12_gradient_boosting_machines/10_intraday_features.ipynb).  

> **Update**: The [figures](figures) directory contains color versions of the charts used in the book. 

# Outline & Chapter Summary

The [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d) has four parts that address different challenges that arise when sourcing and working with market, fundamental and alternative data sourcing, developing ML solutions to various predictive tasks in the trading context, and designing and evaluating a trading strategy that relies on predictive signals generated by an ML model.

> The directory for each chapter contains a README with additional information on content, code examples and additional resources.  

[Part 1: From Data to Strategy Development](#part-1-from-data-to-strategy-development)
* [01 Machine Learning for Trading: From Idea to Execution](#01-machine-learning-for-trading-from-idea-to-execution)
* [02 Market & Fundamental Data: Sources and Techniques](#02-market--fundamental-data-sources-and-techniques)
* [03 Alternative Data for Finance: Categories and Use Cases](#03-alternative-data-for-finance-categories-and-use-cases)
* [04 Financial Feature Engineering: How to research Alpha Factors](#04-financial-feature-engineering-how-to-research-alpha-factors)
* [05 Portfolio Optimization and Performance Evaluation](#05-portfolio-optimization-and-performance-evaluation)

[Part 2: Machine Learning for Trading: Fundamentals](#part-2-machine-learning-for-trading-fundamentals)
* [06 The Machine Learning Process](#06-the-machine-learning-process)
* [07 Linear Models: From Risk Factors to Return Forecasts](#07-linear-models-from-risk-factors-to-return-forecasts)
* [08 The ML4T Workflow: From Model to Strategy Backtesting](#08-the-ml4t-workflow-from-model-to-strategy-backtesting)
* [09 Time Series Models for Volatility Forecasts and Statistical Arbitrage](#09-time-series-models-for-volatility-forecasts-and-statistical-arbitrage)
* [10 Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading](#10-bayesian-ml-dynamic-sharpe-ratios-and-pairs-trading)
* [11 Random Forests: A Long-Short Strategy for Japanese Stocks](#11-random-forests-a-long-short-strategy-for-japanese-stocks)
* [12 Boosting your Trading Strategy](#12-boosting-your-trading-strategy)
* [13 Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning](#13-data-driven-risk-factors-and-asset-allocation-with-unsupervised-learning)

[Part 3: Natural Language Processing for Trading](#part-3-natural-language-processing-for-trading)
* [14 Text Data for Trading: Sentiment Analysis](#14-text-data-for-trading-sentiment-analysis)
* [15 Topic Modeling: Summarizing Financial News](#15-topic-modeling-summarizing-financial-news)
* [16 Word embeddings for Earnings Calls and SEC Filings](#16-word-embeddings-for-earnings-calls-and-sec-filings)

[Part 4: Deep & Reinforcement Learning](#part-4-deep--reinforcement-learning)
* [17 Deep Learning for Trading](#17-deep-learning-for-trading)
* [18 CNN for Financial Time Series and Satellite Images](#18-cnn-for-financial-time-series-and-satellite-images)
* [19 RNN for Multivariate Time Series and Sentiment Analysis](#19-rnn-for-multivariate-time-series-and-sentiment-analysis)
* [20 Autoencoders for Conditional Risk Factors and Asset Pricing](#20-autoencoders-for-conditional-risk-factors-and-asset-pricing)
* [21 Generative Adversarial Nets for Synthetic Time Series Data](#21-generative-adversarial-nets-for-synthetic-time-series-data)
* [22 Deep Reinforcement Learning: Building a Trading Agent](#22-deep-reinforcement-learning-building-a-trading-agent)
* [23 Conclusions and Next Steps](#23-conclusions-and-next-steps)
* [24 Appendix - Alpha Factor Library](#24-appendix---alpha-factor-library)

## Part 1: From Data to Strategy Development

The first part provides a framework for developing trading strategies driven by machine learning (ML). It focuses on the data that power the ML algorithms and strategies discussed in this book, outlines how to engineer and evaluates features suitable for ML models, and how to manage and measure a portfolio's performance while executing a trading strategy.

### 01 Machine Learning for Trading: From Idea to Execution

This [chapter](01_machine_learning_for_trading) explores industry trends that have led to the emergence of ML as a source of competitive advantage in the investment industry. We will also look at where ML fits into the investment process to enable algorithmic trading strategies. 

More specifically, it covers the following topics:
- Key trends behind the rise of ML in the investment industry
- The design and execution of a trading strategy that leverages ML
- Popular use cases for ML in trading

### 02 Market & Fundamental Data: Sources and Techniques

This [chapter](02_market_and_fundamental_data) shows how to work with market and fundamental data and describes critical aspects of the environment that they reflect. For example, familiarity with various order types and the trading infrastructure matter not only for the interpretation of the data but also to correctly design backtest simulations. We also illustrate how to use Python to access and manipulate trading and financial statement data.  

Practical examples demonstrate how to work with trading data from NASDAQ tick data and Algoseek minute bar data with a rich set of attributes capturing the demand-supply dynamic that we will later use for an ML-based intraday strategy. We also cover various data provider APIs and how to source financial statement information from the SEC.

<p align="center">
<img src="https://i.imgur.com/enaSo0C.png" title="Order Book" width="50%"/>
</p>
In particular, this chapter covers:

- How market data reflects the structure of the trading environment
- Working with intraday trade and quotes data at minute frequency
- Reconstructing the **limit order book** from tick data using NASDAQ ITCH 
- Summarizing tick data using various types of bars
- Working with eXtensible Business Reporting Language (XBRL)-encoded **electronic filings**
- Parsing and combining market and fundamental data to create a P/E series
- How to access various market and fundamental data sources using Python

### 03 Alternative Data for Finance: Categories and Use Cases

This [chapter](03_alternative_data) outlines categories and use cases of alternative data, describes criteria to assess the exploding number of sources and providers, and summarizes the current market landscape. 

It also demonstrates how to create alternative data sets by scraping websites, such as collecting earnings call transcripts for use with natural language processing (NLP) and sentiment analysis algorithms in the third part of the book.
 
More specifically, this chapter covers:

- Which new sources of signals have emerged during the alternative data revolution
- How individuals, business, and sensors generate a diverse set of alternative data
- Important categories and providers of alternative data
- Evaluating how the burgeoning supply of alternative data can be used for trading
- Working with alternative data in Python, such as by scraping the internet

### 04 Financial Feature Engineering: How to research Alpha Factors

If you are already familiar with ML, you know that feature engineering is a crucial ingredient for successful predictions. It matters at least as much in the trading domain, where academic and industry researchers have investigated for decades what drives asset markets and prices, and which features help to explain or predict price movements.

<p align="center">
<img src="https://i.imgur.com/UCu4Huo.png" width="70%">
</p>

This [chapter](04_alpha_factor_research) outlines the key takeaways of this research as a starting point for your own quest for alpha factors. It also presents essential tools to compute and test alpha factors, highlighting how the NumPy, pandas, and TA-Lib libraries facilitate the manipulation of data and present popular smoothing techniques like the wavelets and the Kalman filter that help reduce noise in data. After reading it, you will know about:
- Which categories of factors exist, why they work, and how to measure them,
- Creating alpha factors using NumPy, pandas, and TA-Lib,
- How to de-noise data using wavelets and the Kalman filter,
- Using Zipline to test individual and multiple alpha factors,
- How to use [Alphalens](https://github.com/quantopian/alphalens) to evaluate predictive performance.
 
### 05 Portfolio Optimization and Performance Evaluation

Alpha factors generate signals that an algorithmic strategy translates into trades, which, in turn, produce long and short positions. The returns and risk of the resulting portfolio determine whether the strategy meets the investment objectives.
<p align="center">
<img src="https://i.imgur.com/E2h63ZB.png" width="65%">
</p>

There are several approaches to optimize portfolios. These include the application of machine learning (ML) to learn hierarchical relationships among assets and treat them as complements or substitutes when designing the portfolio's risk profile. This [chapter](05_strategy_evaluation) covers:
- How to measure portfolio risk and return
- Managing portfolio weights using mean-variance optimization and alternatives
- Using machine learning to optimize asset allocation in a portfolio context
- Simulating trades and create a portfolio based on alpha factors using Zipline
- How to evaluate portfolio performance using [pyfolio](https://quantopian.github.io/pyfolio/)

## Part 2: Machine Learning for Trading: Fundamentals

The second part covers the fundamental supervised and unsupervised learning algorithms and illustrates their application to trading strategies. It also introduces the Quantopian platform that allows you to leverage and combine the data and ML techniques developed in this book to implement algorithmic strategies that execute trades in live markets.

### 06 The Machine Learning Process

This [chapter](06_machine_learning_process) kicks off Part 2 that illustrates how you can use a range of supervised and unsupervised ML models for trading. We will explain each model's assumptions and use cases before we demonstrate relevant applications using various Python libraries. 

There are several aspects that many of these models and their applications have in common. This chapter covers these common aspects so that we can focus on model-specific usage in the following chapters. It sets the stage by outlining how to formulate, train, tune, and evaluate the predictive performance of ML models as a systematic workflow. The content includes:

<p align="center">
<img src="https://i.imgur.com/5qisClE.png" width="65%">
</p>

- How supervised and unsupervised learning from data works
- Training and evaluating supervised learning models for regression and classification tasks
- How the bias-variance trade-off impacts predictive performance
- How to diagnose and address prediction errors due to overfitting
- Using cross-validation to optimize hyperparameters with a focus on time-series data
- Why financial data requires additional attention when testing out-of-sample

### 07 Linear Models: From Risk Factors to Return Forecasts

Linear models are standard tools for inference and prediction in regression and classification contexts. Numerous widely used asset pricing models rely on linear regression. Regularized models like Ridge and Lasso regression often yield better predictions by limiting the risk of overfitting. Typical regression applications identify risk factors that drive asset returns to manage risks or predict returns. Classification problems, on the other hand, include directional price forecasts.

<p align="center">
<img src="https://i.imgur.com/3Ph6jma.png" width="65%">
</p>

[Chapter 07](07_linear_models) covers the following topics:

- How linear regression works and which assumptions it makes
- Training and diagnosing linear regression models
- Using linear regression to predict stock returns
- Use regularization to improve the predictive performance
- How logistic regression works
- Converting a regression into a classification problem

### 08 The ML4T Workflow: From Model to Strategy Backtesting

This [chapter](08_ml4t_workflow) presents an end-to-end perspective on designing, simulating, and evaluating a trading strategy driven by an ML algorithm. 
We will demonstrate in detail how to backtest an ML-driven strategy in a historical market context using the Python libraries [backtrader](https://www.backtrader.com/) and [Zipline](https://zipline.ml4trading.io/index.html). 
The ML4T workflow ultimately aims to gather evidence from historical data that helps decide whether to deploy a candidate strategy in a live market and put financial resources at risk. A realistic simulation of your strategy needs to faithfully represent how security markets operate and how trades execute. Also, several methodological aspects require attention to avoid biased results and false discoveries that will lead to poor investment decisions.

<p align="center">
<img src="https://i.imgur.com/R9O0fn3.png" width="65%">
</p>

More specifically, after working through this chapter you will be able to:

- Plan and implement end-to-end strategy backtesting
- Understand and avoid critical pitfalls when implementing backtests
- Discuss the advantages and disadvantages of vectorized vs event-driven backtesting engines
- Identify and evaluate the key components of an event-driven backtester
- Design and execute the ML4T workflow using data sources at minute and daily frequencies, with ML models trained separately or as part of the backtest
- Use Zipline and backtrader to design and evaluate your own strategies 

### 09 Time Series Models for Volatility Forecasts and Statistical Arbitrage

This [chapter](09_time_series_models) focuses on models that extract signals from a time series' history to predict future values for the same time series. 
Time series models are in widespread use due to the time dimension inherent to trading. It presents tools to diagnose time series characteristics such as stationarity and extract features that capture potentially useful patterns. It also introduces univariate and multivariate time series models to forecast macro data and volatility patterns. 
Finally, it explains how cointegration identifies common trends across time series and shows how to develop a pairs trading strategy based on this crucial concept. 

<p align="center">
<img src="https://i.imgur.com/cglLgJ0.png" width="90%">
</p>

In particular, it covers:
- How to use time-series analysis to prepare and inform the modeling process
- Estimating and diagnosing univariate autoregressive and moving-average models
- Building autoregressive conditional heteroskedasticity (ARCH) models to predict volatility
- How to build multivariate vector autoregressive models
- Using cointegration to develop a pairs trading strategy

### 10 Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading

Bayesian statistics allows us to quantify uncertainty about future events and refine estimates in a principled way as new information arrives. This dynamic approach adapts well to the evolving nature of financial markets. 
Bayesian approaches to ML enable new insights into the uncertainty around statistical metrics, parameter estimates, and predictions. The applications range from more granular risk management to dynamic updates of predictive models that incorporate changes in the market environment. 

<p align="center">
<img src="https://i.imgur.com/qOUPIDV.png" width="80%">
</p>

More specifically, this [chapter](10_bayesian_machine_learning) covers: 
- How Bayesian statistics applies to machine learning
- Probabilistic programming with PyMC3
- Defining and training machine learning models using PyMC3
- How to run state-of-the-art sampling methods to conduct approximate inference
- Bayesian ML applications to compute dynamic Sharpe ratios, dynamic pairs trading hedge ratios, and estimate stochastic volatility


### 11 Random Forests: A Long-Short Strategy for Japanese Stocks

This [chapter](11_decision_trees_random_forests) applies decision trees and random forests to trading. Decision trees learn rules from data that encode nonlinear input-output relationships. We show how to train a decision tree to make predictions for regression and classification problems, visualize and interpret the rules learned by the model, and tune the model's hyperparameters to optimize the bias-variance tradeoff and prevent overfitting.

The second part of the chapter introduces ensemble models that combine multiple decision trees in a randomized fashion to produce a single prediction with a lower error. It concludes with a long-short strategy for Japanese equities based on trading signals generated by a random forest model.

<p align="center">
<img src="https://i.imgur.com/S4s0rou.png" width="80%">
</p>

In short, this chapter covers:
- Use decision trees for regression and classification
- Gain insights from decision trees and visualize the rules learned from the data
- Understand why ensemble models tend to deliver superior results
- Use bootstrap aggregation to address the overfitting challenges of decision trees
- Train, tune, and interpret random forests
- Employ a random forest to design and evaluate a profitable trading strategy


### 12 Boosting your Trading Strategy

Gradient boosting is an alternative tree-based ensemble algorithm that often produces better results than random forests. The critical difference is that boosting modifies the data used to train each tree based on the cumulative errors made by the model. While random forests train many trees independently using random subsets of the data, boosting proceeds sequentially and reweights the data.
This [chapter](12_gradient_boosting_machines) shows how state-of-the-art libraries achieve impressive performance and apply boosting to both daily and high-frequency data to backtest an intraday trading strategy. 

<p align="center">
<img src="https://i.imgur.com/Re0uI0H.png" width="70%">
</p>

More specifically, we will cover the following topics:
- How does boosting differ from bagging, and how did gradient boosting evolve from adaptive boosting,
- Design and tune adaptive and gradient boosting models with scikit-learn,
- Build, optimize, and evaluate gradient boosting models on large datasets with the state-of-the-art implementations XGBoost, LightGBM, and CatBoost,
- Interpreting and gaining insights from gradient boosting models using [SHAP](https://github.com/slundberg/shap) values, and
- Using boosting with high-frequency data to design an intraday strategy.

### 13 Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning

Dimensionality reduction and clustering are the main tasks for unsupervised learning: 
- Dimensionality reduction transforms the existing features into a new, smaller set while minimizing the loss of information. A broad range of algorithms exists that differ by how they measure the loss of information, whether they apply linear or non-linear transformations or the constraints they impose on the new feature set. 
- Clustering algorithms identify and group similar observations or features instead of identifying new features. Algorithms differ in how they define the similarity of observations and their assumptions about the resulting groups.

<p align="center">
<img src="https://i.imgur.com/Rfk7uCM.png" width="70%">
</p>

More specifically, this [chapter](13_unsupervised_learning) covers:
- How principal and independent component analysis (PCA and ICA) perform linear dimensionality reduction
- Identifying data-driven risk factors and eigenportfolios from asset returns using PCA
- Effectively visualizing nonlinear, high-dimensional data using manifold learning
- Using T-SNE and UMAP to explore high-dimensional image data
- How k-means, hierarchical, and density-based clustering algorithms work
- Using agglomerative clustering to build robust portfolios with hierarchical risk parity


## Part 3: Natural Language Processing for Trading

Text data are rich in content, yet unstructured in format and hence require more preprocessing so that a machine learning algorithm can extract the potential signal. The critical challenge consists of converting text into a numerical format for use by an algorithm, while simultaneously expressing the semantics or meaning of the content. 

The next three chapters cover several techniques that capture language nuances readily understandable to humans so that machine learning algorithms can also interpret them.

### 14 Text Data for Trading: Sentiment Analysis

Text data is very rich in content but highly unstructured so that it requires more preprocessing to enable an ML algorithm to extract relevant information. A key challenge consists of converting text into a numerical format without losing its meaning.
This [chapter](14_working_with_text_data) shows how to represent documents as vectors of token counts by creating a document-term matrix that, in turn, serves as input for text classification and sentiment analysis. It also introduces the Naive Bayes algorithm and compares its performance to linear and tree-based models.

In particular, in this chapter covers:
- What the fundamental NLP workflow looks like
- How to build a multilingual feature extraction pipeline using spaCy and TextBlob
- Performing NLP tasks like part-of-speech tagging or named entity recognition
- Converting tokens to numbers using the document-term matrix
- Classifying news using the naive Bayes model
- How to perform sentiment analysis using different ML algorithms

### 15 Topic Modeling: Summarizing Financial News

This [chapter](15_topic_modeling) uses unsupervised learning to model latent topics and extract hidden themes from documents. These themes can generate detailed insights into a large corpus of financial reports.
Topic models automate the creation of sophisticated, interpretable text features that, in turn, can help extract trading signals from extensive collections of texts. They speed up document review, enable the clustering of similar documents, and produce annotations useful for predictive modeling.
Applications include identifying critical themes in company disclosures, earnings call transcripts or contracts, and annotation based on sentiment analysis or using returns of related assets. 

<p align="center">
<img src="https://i.imgur.com/VVSnTCa.png" width="60%">
</p>

More specifically, it covers:
- How topic modeling has evolved, what it achieves, and why it matters
- Reducing the dimensionality of the DTM using latent semantic indexing
- Extracting topics with probabilistic latent semantic analysis (pLSA)
- How latent Dirichlet allocation (LDA) improves pLSA to become the most popular topic model
- Visualizing and evaluating topic modeling results -
- Running LDA using scikit-learn and gensim
- How to apply topic modeling to collections of earnings calls and financial news articles

### 16 Word embeddings for Earnings Calls and SEC Filings

This [chapter](16_word_embeddings) uses neural networks to learn a vector representation of individual semantic units like a word or a paragraph. These vectors are dense with a few hundred real-valued entries, compared to the higher-dimensional sparse vectors of the bag-of-words model. As a result, these vectors embed or locate each semantic unit in a continuous vector space.

Embeddings result from training a model to relate tokens to their context with the benefit that similar usage implies a similar vector. As a result, they encode semantic aspects like relationships among words through their relative location. They are powerful features that we will use with deep learning models in the following chapters.

<p align="center">
<img src="https://i.imgur.com/v8w9XLL.png" width="80%">
</p>

 More specifically, in this chapter, we will cover:
- What word embeddings are and how they capture semantic information
- How to obtain and use pre-trained word vectors
- Which network architectures are most effective at training word2vec models
- How to train a word2vec model using TensorFlow and gensim
- Visualizing and evaluating the quality of word vectors
- How to train a word2vec model on SEC filings to predict stock price moves
- How doc2vec extends word2vec and helps with sentiment analysis
- Why the transformer’s attention mechanism had such an impact on NLP
- How to fine-tune pre-trained BERT models on financial data

## Part 4: Deep & Reinforcement Learning

Part four explains and demonstrates how to leverage deep learning for algorithmic trading. 
The powerful capabilities of deep learning algorithms to identify patterns in unstructured data make it particularly suitable for alternative data like images and text. 

The sample applications show, for exapmle, how to combine text and price data to predict earnings surprises from SEC filings, generate synthetic time series to expand the amount of training data, and train a trading agent using deep reinforcement learning.
Several of these applications replicate research recently published in top journals.

### 17 Deep Learning for Trading

This [chapter](17_deep_learning) presents feedforward neural networks (NN) and demonstrates how to efficiently train large models using backpropagation while managing the risks of overfitting. It also shows how to use TensorFlow 2.0 and PyTorch and how to optimize a NN architecture to generate trading signals.
In the following chapters, we will build on this foundation to apply various architectures to different investment applications with a focus on alternative data. These include recurrent NN tailored to sequential data like time series or natural language and convolutional NN, particularly well suited to image data. We will also cover deep unsupervised learning, such as how to create synthetic data using Generative Adversarial Networks (GAN). Moreover, we will discuss reinforcement learning to train agents that interactively learn from their environment.

<p align="center">
<img src="https://i.imgur.com/5cet0Fi.png" width="70%">
</p>

In particular, this chapter will cover
- How DL solves AI challenges in complex domains
- Key innovations that have propelled DL to its current popularity
- How feedforward networks learn representations from data
- Designing and training deep neural networks (NNs) in Python
- Implementing deep NNs using Keras, TensorFlow, and PyTorch
- Building and tuning a deep NN to predict asset returns
- Designing and backtesting a trading strategy based on deep NN signals

### 18 CNN for Financial Time Series and Satellite Images

CNN architectures continue to evolve. This chapter describes building blocks common to successful applications, demonstrates how transfer learning can speed up learning, and how to use CNNs for object detection.
CNNs can generate trading signals from images or time-series data. Satellite data can anticipate commodity trends via aerial images of agricultural areas, mines, or transport networks. Camera footage can help predict consumer activity; we show how to build a CNN that classifies economic activity in satellite images.
CNNs can also deliver high-quality time-series classification results by exploiting their structural similarity with images, and we design a strategy based on time-series data formatted like images. 

<p align="center">
<img src="https://i.imgur.com/PlLQV0M.png" width="60%">
</p>

More specifically, this [chapter](18_convolutional_neural_nets) covers:

- How CNNs employ several building blocks to efficiently model grid-like data
- Training, tuning and regularizing CNNs for images and time series data using TensorFlow
- Using transfer learning to streamline CNNs, even with fewer data
- Designing a trading strategy using return predictions by a CNN trained on time-series data formatted like images
- How to classify economic activity based on satellite images

### 19 RNN for Multivariate Time Series and Sentiment Analysis

Recurrent neural networks (RNNs) compute each output as a function of the previous output and new data, effectively creating a model with memory that shares parameters across a deeper computational graph. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that address the challenges of learning long-range dependencies.
RNNs are designed to map one or more input sequences to one or more output sequences and are particularly well suited to natural language. They can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in Chapter 16 to classify the sentiment expressed in documents.

<p align="center">
<img src="https://i.imgur.com/E9fOApg.png" width="60%">
</p>

More specifically, this chapter addresses:
- How recurrent connections allow RNNs to memorize patterns and model a hidden state
- Unrolling and analyzing the computational graph of RNNs
- How gated units learn to regulate RNN memory from data to enable long-range dependencies
- Designing and training RNNs for univariate and multivariate time series in Python
- How to learn word embeddings or use pretrained word vectors for sentiment analysis with RNNs
- Building a bidirectional RNN to predict stock returns using custom word embeddings

### 20 Autoencoders for Conditional Risk Factors and Asset Pricing

This [chapter](20_autoencoders_for_conditional_risk_factors) shows how to leverage unsupervised deep learning for trading. We also discuss autoencoders, namely, a neural network trained to reproduce the input while learning a new representation encoded by the parameters of a hidden layer. Autoencoders have long been used for nonlinear dimensionality reduction, leveraging the NN architectures we covered in the last three chapters.
We replicate a recent AQR paper that shows how autoencoders can underpin a trading strategy. We will use a deep neural network that relies on an autoencoder to extract risk factors and predict equity returns, conditioned on a range of equity attributes.

<p align="center">
<img src="https://i.imgur.com/aCmE0UD.png" width="60%">
</p>

More specifically, in this chapter you will learn about:
- Which types of autoencoders are of practical use and how they work
- Building and training autoencoders using Python
- Using autoencoders to extract data-driven risk factors that take into account asset characteristics to predict returns

### 21 Generative Adversarial Nets for Synthetic Time Series Data

This chapter introduces generative adversarial networks (GAN). GANs train a generator and a discriminator network in a competitive setting so that the generator learns to produce samples that the discriminator cannot distinguish from a given class of training data. The goal is to yield a generative model capable of producing synthetic samples representative of this class.
While most popular with image data, GANs have also been used to generate synthetic time-series data in the medical domain. Subsequent experiments with financial data explored whether GANs can produce alternative price trajectories useful for ML training or strategy backtests. We replicate the 2019 NeurIPS Time-Series GAN paper to illustrate the approach and demonstrate the results.

<p align="center">
<img src="https://i.imgur.com/W1Rp89K.png" width="60%">
</p>

More specifically, in this chapter you will learn about:
- How GANs work, why they are useful, and how they could be applied to trading
- Designing and training GANs using TensorFlow 2
- Generating synthetic financial data to expand the inputs available for training ML models and backtesting

### 22 Deep Reinforcement Learning: Building a Trading Agent

Reinforcement Learning (RL) models goal-directed learning by an agent that interacts with a stochastic environment. RL optimizes the agent's decisions concerning a long-term objective by learning the value of states and actions from a reward signal. The ultimate goal is to derive a policy that encodes behavioral rules and maps states to actions.
This [chapter](22_deep_reinforcement_learning) shows how to formulate and solve an RL problem. It covers model-based and model-free methods, introduces the OpenAI Gym environment, and combines deep learning with RL to train an agent that navigates a complex environment. Finally, we'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts with the financial market while trying to optimize an objective function.

<p align="center">
<img src="https://i.imgur.com/lg0ofbZ.png" width="60%">
</p>

More specifically,this chapter will cover:

- Define a Markov decision problem (MDP)
- Use value and policy iteration to solve an MDP
- Apply Q-learning in an environment with discrete states and actions
- Build and train a deep Q-learning agent in a continuous environment
- Use the OpenAI Gym to design a custom market environment and train an RL agent to trade stocks

### 23 Conclusions and Next Steps

In this concluding chapter, we will briefly summarize the essential tools, applications, and lessons learned throughout the book to avoid losing sight of the big picture after so much detail.
We will then identify areas that we did not cover but would be worth focusing on as you expand on the many machine learning techniques we introduced and become productive in their daily use.

In sum, in this chapter, we will
- Review key takeaways and lessons learned
- Point out the next steps to build on the techniques in this book
- Suggest ways to incorporate ML into your investment process

### 24 Appendix - Alpha Factor Library

Throughout this book, we emphasized how the smart design of features, including appropriate preprocessing and denoising, typically leads to an effective strategy. This appendix synthesizes some of the lessons learned on feature engineering and provides additional information on this vital topic.

To this end, we focus on the broad range of indicators implemented by TA-Lib (see [Chapter 4](04_alpha_factor_research)) and WorldQuant's [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) paper (Kakushadze 2016), which presents real-life quantitative trading factors used in production with an average holding period of 0.6-6.4 days.

This chapter covers: 
- How to compute several dozen technical indicators using TA-Lib and NumPy/pandas,
- Creating the formulaic alphas describe in the above paper, and
- Evaluating the predictive quality of the results using various metrics from rank correlation and mutual information to feature importance, SHAP values and Alphalens. 
