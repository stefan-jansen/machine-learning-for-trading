# Chapter 23 - Next Steps

In this concluding chapter, we will briefly summarize the key tools, applications, and lessons learned throughout the book to avoid losing sight of the big picture after so much detail. We will then identify areas that we did not cover but would be worthwhile to focus on as you expand on the many machine learning techniques we introduced and become productive in their daily use.
In sum, in this chapter, we will
- Review key takeaways and lessons learned
- Point out the next steps to build on the techniques in this book
- Suggest ways to incorporate ML into your investment process

## Content

1. [Key Takeaways and Lessons Learned](#key-takeaways-and-lessons-learned)
    * [Data is the single most important ingredient](#data-is-the-single-most-important-ingredient)
    * [Domain expertise: separate the signal from the noise](#domain-expertise-separate-the-signal-from-the-noise)
    * [ML is a toolkit for solving problems with data](#ml-is-a-toolkit-for-solving-problems-with-data)
    * [Beware of backtest overfitting](#beware-of-backtest-overfitting)
    * [How to gain insights from black-box models](#how-to-gain-insights-from-black-box-models)
2. [Machine Learning for Trading in Practice](#machine-learning-for-trading-in-practice)
    * [Data management technologies](#data-management-technologies)
    * [Machine learning tools](#machine-learning-tools)
    * [Online trading platforms](#online-trading-platforms)

## Key Takeaways and Lessons Learned

Important insights to keep in mind as you proceed to the practice of machine learning for trading include:
- Data is the single most important ingredient that requires careful sourcing and handling
- Domain expertise is key to realizing the value contained in data and avoiding some of the pitfalls of using ML.
- ML offers tools that you can adapt and combine to create solutions for your use case.
- The choices of model objectives and performance diagnostics are key to productive iterations towards an optimal system.
- Backtest overfitting is a huge challenge that requires significant attention.
- Transparency of black-box models can help build confidence and facilitate the adoption of ML by skeptics.

### Data is the single most important ingredient

A key insight is that state-of-the-art ML techniques like deep neural networks are successful because their predictive performance continues to improve with more data. On the flip side, model and data complexity need to match to balance the bias-variance trade-off, which becomes more challenging the higher the noise-to-signal ratio of the data. Managing data quality and integrating data sets are key steps in realizing the potential value.

### Domain expertise: separate the signal from the noise

We emphasized that informative data is a necessary condition for successful ML applications. However, domain expertise is equally essential to define the strategic direction, select relevant data, engineer informative features, and design robust models.

### ML is a toolkit for solving problems with data

Machine learning offers algorithmic solutions and techniques that can be applied to many use cases. Parts 2, 3 and 4 of the book have presented machine learning as a diverse set of tools that can add value to various steps of the strategy process, including
- Idea generation and alpha factor research
- Signal aggregation and portfolio optimization
- Strategy testing
- Trade execution
- Strategy evaluation

### Beware of backtest overfitting

We covered the risks of false discoveries due to overfitting to historical data repeatedly throughout the book. Chapter 5, on strategy evaluation, lays out the main drivers and potential remedies. The low noise-to-signal ratio and relatively small datasets (compared to web-scale image or text data) make this challenge particularly serious in the trading domain. Awareness is critical since the ease of access to data and tools to apply ML increases the risks significantly.

There are no easy answers because the risks are inevitable. However, we presented methods to adjust backtest metrics to account for repeated trials such as the deflated Sharpe ratio. When working towards a live trading strategy, staged paper-trading, and closely monitored performance during execution in the market need to be part of the implementation process.

### How to gain insights from black-box models

Deep neural networks and complex ensembles can raise suspicion when they are considered impenetrable black-box models, in particular in light of the risks of backtest overfitting. We introduced several methods to gain insights into how these models make predictions in Chapter 12, Boosting Your Trading Strategy.

In addition to conventional measures of feature importance, the recent game-theoretic innovation of SHapley Additive exPlanations (SHAP) is a significant step towards understanding the mechanics of complex models. SHAP values allow for the exact attribution of features and their values to predictions so that it becomes easier to validate the logic of a model in the light of specific theories about market behavior for a given investment target. Besides justification, exact feature importance scores and attribution of predictions allow for deeper insights into the drivers of the investment outcome of interest.

## Machine Learning for Trading in Practice

As you proceed to integrate the numerous tools and techniques into your investment and trading process, there are numerous things you can focus your efforts on. If your goal is to make better decisions, you should select projects that are realistic yet ambitious given your current skill set. This will help you to develop an efficient workflow underpinned by productive tools and gain practical experience.

### Data management technologies

The central role of data in the ML4T process requires familiarity with a range of technologies to store, transform, and analyze data at scale, including the use of cloud-based services like Amazon Web Services, Microsoft Azure, and Google Cloud.

### Machine learning tools

We covered many libraries of the Python ecosystem in this book. Python has evolved to become the language of choice for data science and machine learning. The set of open-source libraries continues to both diversify and mature, and are built on the robust core of scientific computing libraries NumPy and SciPy. 

There are several providers that aim to facilitate the machine learning workflow:
- H2O.ai offers the H2O platform that integrates cloud computing with machine learning automation. It allows users to fit thousands of potential models to their data to explore patterns in the data. It has interfaces in Python as well as R and Java.
- Datarobot aims to automate the model development process by providing a platform to rapidly build and deploy predictive models in the cloud or on-premise.
- Dataiku is a collaborative data science platform designed to help analysts and engineers explore, prototype, build, and deliver their own data products.

There are also several open-source initiatives led by companies that build on and expand the Python ecosystem:
- The quantitative hedge fund [Two Sigma](https://www.twosigma.com/) contributes quantitative analysis tools to the Jupyter Notebook environment under the [BeakerX](https://github.com/twosigma/beakerx) project.
- Bloomberg has integrated the Jupyter Notebook into its terminal to facilitate the interactive analysis of its financial data.

### Online trading platforms

The main options to develop trading strategies that use machine learning are online platforms, which often look for and allocate capital to successful trading strategies. 

Popular solutions include 
- [Quantopian](https://www.quantopian.com/), 
- [Quantconnect](https://www.quantconnect.com/), and 
- [QuantRocket](https://www.quantrocket.com/). 

In addition, [Interactive Brokers](https://www.interactivebrokers.com/en/home.php) offers a [Python API](https://www.interactivebrokers.com/en/index.php?f=44094) that you can use to develop your own trading solution. 

[Alpaca](https://alpaca.markets/algotrading?gclid=EAIaIQobChMInNybkbug6wIV1f_jBx1Z9AayEAAYASAAEgLu5fD_BwE) offers commission-free execution of algorithmic trading strategies. Several libraries provide integration:
- [pipeline-live](https://github.com/alpacahq/pipeline-live): Zipline Pipeline Extension for Live Trading
- [pylivetrader](https://github.com/alpacahq/pylivetrader): a simple python live trading framework with zipline interface

[Backtrader](https://www.backtrader.com/) is intended for both backtesting and trading with multiple broker integrations.