# From Volatility Forecasts to Statistical Arbitrage: Linear Time Series Models

In this chapter, we will build dynamic linear models to explicitly represent time and include variables observed at specific intervals or lags. A key characteristic of time-series data is their sequential order: rather than random samples of individual observations as in the case of cross-sectional data, our data are a single realization of a stochastic process that we cannot repeat.

Our goal is to identify **systematic patterns in time series** that help us predict how the time series will behave in the future. More specifically, we focus on models that extract signals from a historical sequence of the output and, optionally, other contemporaneous or lagged input variables to predict future values of the output. For example, we might try to predict future returns for a stock using past returns, combined with historical returns of a benchmark or macroeconomic variables. We focus on linear time-series models before turning to nonlinear models like recurrent or convolutional neural networks in Part 4. 

Time-series models are very popular given the time dimension inherent to trading. Key applications include the **prediction of asset returns and volatility**, as well as the identification of co-movements of asset price series. Time-series data are likely to become more prevalent as an ever-broader array of connected devices collects regular measurements with potential signal content.

We first introduce tools to diagnose time-series characteristics and to extract features that capture potential patterns. Then we introduce univariate and multivariate time-series models and apply them to forecast macro data and volatility patterns. We conclude with the concept of **cointegration** and how to apply it to develop a **pairs trading strategy**.

## Content

1. [Tools for diagnostics and feature extraction](#tools-for-diagnostics-and-feature-extraction)
    * [How to decompose time series patterns](#how-to-decompose-time-series-patterns)
    * [Rolling window statistics and moving averages](#rolling-window-statistics-and-moving-averages)
    * [How to measure autocorrelation](#how-to-measure-autocorrelation)
    * [How to diagnose and achieve stationarity](#how-to-diagnose-and-achieve-stationarity)
    * [How to apply time series transformations](#how-to-apply-time-series-transformations)
    * [How to diagnose and address unit roots](#how-to-diagnose-and-address-unit-roots)
    * [Code example: working with time series data](#code-example-working-with-time-series-data)
    * [Resources](#resources)
2. [Univariate Time Series Models](#univariate-time-series-models)
    * [How to build autoregressive models](#how-to-build-autoregressive-models)
    * [How to build moving average models](#how-to-build-moving-average-models)
    * [How to build ARIMA models and extensions](#how-to-build-arima-models-and-extensions)
    * [Code example: forecasting macro fundamentals with ARIMA and SARIMAX models](#code-example-forecasting-macro-fundamentals-with-arima-and-sarimax-models)
    * [How to use time series models to forecast volatility](#how-to-use-time-series-models-to-forecast-volatility)
    * [How to build a volatility-forecasting model](#how-to-build-a-volatility-forecasting-model)
    * [Code examples: volatility forecasts](#code-examples-volatility-forecasts)
    * [Resources](#resources-2)
3. [Multivariate Time Series Models](#multivariate-time-series-models)
    * [The vector autoregressive (VAR) model](#the-vector-autoregressive-var-model)
    * [Code example: How to use the VAR model for macro fundamentals forecasts](#code-example-how-to-use-the-var-model-for-macro-fundamentals-forecasts)
    * [Resources](#resources-3)
4. [Cointegration – time series with a common trend](#cointegration--time-series-with-a-common-trend)
    * [Pairs trading: Statistical arbitrage with cointegration](#pairs-trading-statistical-arbitrage-with-cointegration)
    * [Alternative approaches to selecting and trading comoving assets](#alternative-approaches-to-selecting-and-trading-comoving-assets)
    * [Code example: Pairs trading in practice](#code-example-pairs-trading-in-practice)
        - [Computing distance-based heuristics to identify cointegrated pairs](#computing-distance-based-heuristics-to-identify-cointegrated-pairs)
        - [Precomputing the cointegration tests](#precomputing-the-cointegration-tests)
    * [Resources](#resources-4)

## Tools for diagnostics and feature extraction

Most of the examples in this section use data provided by the Federal Reserve that you can access using the pandas datareader that we introduced in [Chapter 2, Market and Fundamental Data](../02_market_and_fundamental_data). 

### How to decompose time series patterns

Time series data typically contains a mix of various patterns that can be decomposed into several components, each representing an underlying pattern category. In particular, time series often consist of the systematic components trend, seasonality and cycles, and unsystematic noise. These components can be combined in an additive, linear model, in particular when fluctuations do not depend on the level of the series, or in a non-linear, multiplicative model. 

- `pandas` Time Series and Date functionality [docs](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
- [Forecasting - Principles & Practice, Hyndman, R. and Athanasopoulos, G., ch.6 'Time Series Decomposition'](https://otexts.org/fpp2/decomposition.html)

### Rolling window statistics and moving averages

The pandas library includes very flexible functionality to define various window types, including rolling, exponentially weighted and expanding windows.

- `pandas` window function [docs](https://pandas.pydata.org/pandas-docs/stable/computation.html#window-functions)

### How to measure autocorrelation

Autocorrelation (also called serial correlation) adapts the concept of correlation to the time series context: just as the correlation coefficient measures the strength of a linear relationship between two variables, the autocorrelation coefficient measures the extent of a linear relationship between time series values separated by a given lag.

We present the following tools to measure autorcorrelation:
- autocorrelation function (ACF)
- partial autocorrelation function (PACF)
- correlogram as a plot of ACF or PACF against the number of lags.

### How to diagnose and achieve stationarity

The statistical properties, such as the mean, variance, or autocorrelation, of a stationary time series are independent of the period, that is, they don't change over time. Hence, stationarity implies that a time series does not have a trend or seasonal effects and that descriptive statistics, such as the mean or the standard deviation, when computed for different rolling windows, are constant or do not change much over time.

### How to apply time series transformations

To satisfy the stationarity assumption of linear time series models, we need to transform the original time series, often in several steps. Common transformations include the application of the (natural) logarithm to convert an exponential growth pattern into a linear trend and stabilize the variance, or differencing.

### How to diagnose and address unit roots

Unit roots pose a particular problem for determining the transformation that will render a time series stationary. In practice, time series of interest rates or asset prices are often not stationary, for example, because there does not exist a price level to which the series reverts. The most prominent example of a non-stationary series is the random walk.

The defining characteristic of a unit-root non-stationary series is long memory: since current values are the sum of past disturbances, large innovations persist for much longer than for a mean-reverting, stationary series. Identifying the correct transformation, and in particular, the appropriate number and lags for differencing is not always clear-cut. We present a few heuristics to guide the process.

Statistical unit root tests are a common way to determine objectively whether (additional) differencing is necessary. These are statistical hypothesis tests of stationarity that are designed to determine whether differencing is required.

### Code example: working with time series data

- The notebook [tsa_and_stationarity](01_tsa_and_stationarity.ipynb) illustrates the concepts discussed in this section.

### Resources

- [Analysis of Financial Time Series, 3rd Edition, Ruey S. Tsay](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354)
- [Quantitative Equity Investing: Techniques and Strategies, Frank J. Fabozzi, Sergio M. Focardi, Petter N. Kolm](https://www.wiley.com/en-us/Quantitative+Equity+Investing%3A+Techniques+and+Strategies-p-9780470262474)
- `statsmodels` Time Series Analysis [docs](https://www.statsmodels.org/dev/tsa.html)

## Univariate Time Series Models

Univariate time series models relate the value of the time series at the point in time of interest to a linear combination of lagged values of the series and possibly past disturbance terms.

While exponential smoothing models are based on a description of the trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data. ARIMA(p, d, q) models require stationarity and leverage two building blocks:
- Autoregressive (AR) terms consisting of p-lagged values of the time series
- Moving average (MA) terms that contain q-lagged disturbances

### How to build autoregressive models

An AR model of order p aims to capture the linear dependence between time series values at different lags. It closely resembles a multiple linear regression on lagged values of the outcome.

### How to build moving average models

An MA model of order q uses q past disturbances rather than lagged values of the time series in a regression-like model. Since we do not observe the white-noise disturbance values, MA(q) is not a regression model like the ones we have seen so far. Rather than using least squares, MA(q) models are estimated using maximum likelihood (MLE).

### How to build ARIMA models and extensions

Autoregressive integrated moving-average ARIMA(p, d, q) models combine AR(p) and MA(q) processes to leverage the complementarity of these building blocks and simplify model development by using a more compact form and reducing the number of parameters, in turn reducing the risk of overfitting.

- statsmodels State-Space Models [docs](https://www.statsmodels.org/dev/statespace.html)

### Code example: forecasting macro fundamentals with ARIMA and SARIMAX models

We will build a SARIMAX model for monthly data on an industrial production time series for the 1988-2017 period. See notebook [arima_models](02_arima_models.ipynb) for implementation details.

### How to use time series models to forecast volatility

A particularly important area of application for univariate time series models is the prediction of volatility. The volatility of financial time series is usually not constant over time but changes, with bouts of volatility clustering together. Changes in variance create challenges for time series forecasting using the classical ARIMA models.

### How to build a volatility-forecasting model

The development of a volatility model for an asset-return series consists of four steps:
1. Build an ARMA time series model for the financial time series based on the serial dependence revealed by the ACF and PACF.
2. Test the residuals of the model for ARCH/GARCH effects, again relying on the ACF and PACF for the series of the squared residual.
3. Specify a volatility model if serial correlation effects are significant, and jointly estimate the mean and volatility equations.
4. Check the fitted model carefully and refine it if necessary.

### Code examples: volatility forecasts

The notebook [arch_garch_models](03_arch_garch_models.ipynb) demonstrates the usage of the ARCH library to estimate time series models for volatility foreccasting with NASDAQ data.

### Resources

- NYU Stern [VLAB](https://vlab.stern.nyu.edu/)
- ARCH Library
    - [docs](https://arch.readthedocs.io/en/latest/index.html) 
    - [examples](http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb)

## Multivariate Time Series Models

Multivariate time series models are designed to capture the dynamic of multiple time series simultaneously and leverage dependencies across these series for more reliable predictions.

Univariate time-series models like the ARMA approach are limited to statistical relationships between a target variable and its lagged values or lagged disturbances and exogenous series in the ARMAX case. In contrast, multivariate time-series models also allow for lagged values of other time series to affect the target. This effect applies to all series, resulting in complex interactions.

In addition to potentially better forecasting, multivariate time series are also used to gain insights into cross-series dependencies. For example, in economics, multivariate time series are used to understand how policy changes to one variable, such as an interest rate, may affect other variables over different horizons. 

- [New Introduction to Multiple Time Series Analysis, Lütkepohl, Helmut, Springer, 2005](https://www.springer.com/us/book/9783540401728)

### The vector autoregressive (VAR) model

The vector autoregressive VAR(p) model extends the AR(p) model to k series by creating a system of k equations where each contains p lagged values of all k series.

VAR(p) models also require stationarity, so that the initial steps from univariate time-series modeling carry over. First, explore the series and determine the necessary transformations, and then apply the augmented Dickey-Fuller test to verify that the stationarity criterion is met for each series and apply further transformations otherwise. It can be estimated with OLS conditional on initial information or with MLE, which is equivalent for normally distributed errors but not otherwise.

If some or all of the k series are unit-root non-stationary, they may be cointegrated (see next section). This extension of the unit root concept to multiple time series means that a linear combination of two or more series is stationary and, hence, mean-reverting. 

### Code example: How to use the VAR model for macro fundamentals forecasts

The notebook [vector_autoregressive_model](04_vector_autoregressive_model.ipynb) demonstrates how to use `statsmodels` to estimate a VAR model for macro fundamentals time series.

### Resources

- `statsmodels` Vector Autoregression [docs](https://www.statsmodels.org/dev/vector_ar.html)
- [Time Series Analysis in Python with statsmodels](https://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf), Wes McKinney, Josef Perktold, Skipper Seabold, SciPY Conference 2011

## Cointegration – time series with a common trend

The concept of an integrated multivariate series is complicated by the fact that all the component series of the process may be individually integrated but the process is not jointly integrated in the sense that one or more linear combinations of the series exist that produce a new stationary series.

In other words, a combination of two co-integrated series has a stable mean to which this linear combination reverts. A multivariate series with this characteristic is said to be co-integrated. This also applies when the individual series are integrated of a higher order and the linear combination reduces the overall order of integration. 

We demonstrate two major approaches to testing for cointegration:
- The Engle–Granger two-step method
- The Johansen procedure

### Pairs trading: Statistical arbitrage with cointegration

Statistical arbitrage refers to strategies that employ some statistical model or method to take advantage of what appears to be relative mispricing of assets while maintaining a level of market neutrality.

Pairs trading is a conceptually straightforward strategy that has been employed by algorithmic traders since at least the mid-eighties (Gatev, Goetzmann, and Rouwenhorst 2006). The goal is to find two assets whose prices have historically moved together, track the spread (the difference between their prices), and, once the spread widens, buy the loser that has dropped below the common trend and short the winner. If the relationship persists, the long and/or the short leg will deliver profits as prices converge and the positions are closed. 

This approach extends to a multivariate context by forming baskets from multiple securities and trade one asset against a basket of two baskets against each other.

In practice, the strategy requires two steps: 
1. Formation phase: Identify securities that have a long-term mean-reverting relationship. Ideally, the spread should have a high variance to allow for frequent profitable trades while reliably reverting to the common trend.
2. Trading phase: Trigger entry and exit trading rules as price movements cause the spread to diverge and converge.

Several approaches to the formation and trading phases have emerged from increasingly active research in this area across multiple asset classes over the last several years. The next subsection outlines the key differences before we dive into an example application.

### Alternative approaches to selecting and trading comoving assets

A recent comprehensive survey of pairs trading strategies [Statistical Arbitrage Pairs Trading Strategies: Review
and Outlook](https://www.iwf.rw.fau.de/files/2016/03/09-2015.pdf), Krauss (2017) identifies four different methodologies plus a number of other more recent approaches, including ML-based forecasts:

- **Distance** approach: The oldest and most-studied method identifies candidate pairs with distance metrics like correlation and uses non-parametric thresholds like Bollinger Bands to trigger entry and exit trades. The computational simplicity allows for large-scale applications with demonstrated profitability across markets and asset classes for extended periods of time since Gatev, et al. (2006). However, performance has decayed more recently.
- **Cointegration** approach: As outlined previously, this approach relies on an econometric model of a long-term relationship among two or more variables and allows for statistical tests that promise more reliability than simple distance metrics. Examples in this category use the Engle-Granger and Johansen procedures to identify pairs and baskets of securities as well as simpler heuristics that aim to capture the concept (Vidyamurthy 2004). Trading rules often resemble the simple thresholds used with distance metrics.
- **Time-series** approach: With a focus on the trading phase, strategies in this category aim to model the spread as a mean-reverting stochastic process and optimize entry and exit rules accordingly (Elliott, Hoek, and Malcolm 2005). It assumes promising pairs have already been identified.
- **Stochastic control** approach: Similar to the time-series approach, the goal is to optimize trading rules using stochastic control theory to find value and policy functions to arrive at an optimal portfolio (Liu and Timmermann 2013). We will address this type of approach in Chapter 21, Reinforcement Learning.
- **Other approaches**: Besides pair identification based on unsupervised learning like principal component analysis (see Chapter 13, Unsupervised Learning) and statistical models like copulas (Patton 2012), machine learning has become popular more recently to identify pairs based on their relative price or return forecasts (Huck 2019). We will cover several ML algorithms that can be used for this purpose and illustrate corresponding multivariate pairs trading strategies in the coming chapters.

### Code example: Pairs trading in practice

The **distance approach** identifies pairs using the correlation of (normalized) asset prices or their returns and is simple and orders of magnitude less computationally intensive than cointegration tests. 
- The notebook [cointegration_tests](05_cointegration_tests.ipynb) illustrates this for a sample of ~150 stocks with four years of daily data: it takes ~30ms to compute the correlation with the returns of an ETF, compared to 18 seconds for a suite of cointegration tests (using statsmodels) - 600x slower.

The speed advantage is particularly valuable because the number of potential pairs is the product of the number of candidates to be considered on either side so that evaluating combinations of 100 stocks and 100 ETFs requires comparing 10,000 tests (we’ll discuss the challenge of multiple testing bias below).

On the other hand, distance metrics do not necessarily select the most profitable pairs: correlation is maximized for perfect co-movement that in turn eliminates actual trading opportunities. Empirical studies confirm that the volatility of the price spread of cointegrated pairs is almost twice as high as the volatility of the price spread of distance pairs (Huck and Afawubo 2015).

To balance the tradeoff between computational cost and the quality of the resulting pairs, Krauss (2017) recommends a procedure that combines both approaches based on his literature review:
1. Select pairs with a stable spread that shows little drift to reduce the number of candidates
2. Test the remaining pairs with the highest spread variance for cointegration

This process aims to select cointegrated pairs with lower divergence risk while ensuring more volatile spreads that in turn generate higher profit opportunities.

A large number of tests introduce data snooping bias as discussed in Chapter 6, The Machine Learning Workflow: multiple testing is likely to increase the number of false positives that mistakenly reject the null hypothesis of no cointegration. While statistical significance may not be necessary for profitable trading (Chan 2008), a study of commodity pairs (Cummins and Bucca 2012) shows that controlling the familywise error rate to improve the tests’ power according to Romano and Wolf (2010) can lead to better performance.

#### Computing distance-based heuristics to identify cointegrated pairs

- The notebook [cointegration_tests](05_cointegration_tests.ipynb) takes a closer look at how predictive various heuristics for the degree of comovement of asset prices are for the result of cointegration tests. The example code uses a sample of 172 stocks and 138 ETFs traded on the NYSE and NASDAQ with daily data from 2010 - 2019 provided by Stooq. 

The securities represent the largest average dollar volume over the sample period in their respective class; highly correlated and stationary assets have been removed. See the notebook [create_datasets](../data/create_datasets.ipynb) in the data folder of the GitHub repository for downloading for instructions on how to obtain the data and the notebook cointegration_tests for the relevant code and additional preprocessing and exploratory details.

#### Precomputing the cointegration tests

The notebook [statistical_arbitrage_with_cointegrated_pairs](06_statistical_arbitrage_with_cointegrated_pairs.ipynb) implements a statistical arbitrage strategy based on cointegration for the sample of stocks and ETFs and the 2017-2019 period.

It first generates and stores the cointegration tests for all candidate pairs and the resulting trading signals before we backtest a strategy based on these signals given the computational intensity of the process.

### Resources

- Quantopian offers various resources on pairs trading:
    - [Introduction to Pairs Trading](https://www.quantopian.com/lectures/introduction-to-pairs-trading)
    - [Quantopian Johansen](https://www.quantopian.com/posts/trading-baskets-co-integrated-with-spy)
    - [Quantopian PT](https://www.quantopian.com/posts/how-to-build-a-pairs-trading-strategy-on-quantopian)
    - [Pairs Trading Basics: Correlation, Cointegration And Strategy](https://blog.quantinsti.com/pairs-trading-basics/)
- Additional blog posts include:
    - [Pairs Trading using Data-Driven Techniques: Simple Trading Strategies Part 3](https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a)
    - [Pairs Trading Johansen & Kalman](https://letianzj.github.io/kalman-filter-pairs-trading.html)
    - [Copulas](https://twiecki.io/blog/2018/05/03/copulas/) by Thomas Wiecki
