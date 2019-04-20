# Chapter 08: Linear Time Series Models

This chapter covers:
- How to use time series analysis to diagnose diagnostic statistics that inform the modeling process
- How to estimate and diagnose autoregressive and moving-average time series models
- How to build Autoregressive Conditional Heteroskedasticity (ARCH) models to predict volatility
- How to build vector autoregressive models
- How to use cointegration for a pairs trading strategy


## Analytical tools for diagnostics and feature extraction

Most of the examples in this chapter use data provided by the Federal Reserve that you can access using the pandas datareader that we introduced in [Chapter 2, Market and Fundamental Data](../02_market_and_fundamental_data). 

The code examples for this section are available in the notebook [](01_stationarity_and_arima.ipynb).

### How to decompose time series patterns

Time series data typically contains a mix of various patterns that can be decomposed into several components, each representing an underlying pattern category. In particular, time series often consist of the systematic components trend, seasonality and cycles, and unsystematic noise. These components can be combined in an additive, linear model, in particular when fluctuations do not depend on the level of the series, or in a non-linear, multiplicative model. 

- `pandas` Time Series and Date functionality [docs](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
- [Forecasting - Principles & Practice, Hyndman, R. and Athanasopoulos, G., ch.6 'Time Series Decomposition'](https://otexts.org/fpp2/decomposition.html)

### How to compute rolling window statistics

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

## Univariate Time Series Models

 Univariate time series models relate the value of the time series at the point in time of interest to a linear combination of lagged values of the series and possibly past disturbance terms.

While exponential smoothing models are based on a description of the trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data. ARIMA(p, d, q) models require stationarity and leverage two building blocks:
- Autoregressive (AR) terms consisting of p-lagged values of the time series
- Moving average (MA) terms that contain q-lagged disturbances

- [Analysis of Financial Time Series, 3rd Edition, Ruey S. Tsay](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354)

- [Quantitative Equity Investing: Techniques and Strategies, Frank J. Fabozzi, Sergio M. Focardi, Petter N. Kolm](https://www.wiley.com/en-us/Quantitative+Equity+Investing%3A+Techniques+and+Strategies-p-9780470262474)

- `statsmodels` Time Series Analysis [docs](https://www.statsmodels.org/dev/tsa.html)

### How to build autoregressive models

An AR model of order p aims to capture the linear dependence between time series values at different lags. It closely resembles a multiple linear regression on lagged values of the outcome.

### How to build moving average models

An MA model of order q uses q past disturbances rather than lagged values of the time series in a regression-like model. Since we do not observe the white-noise disturbance values, MA(q) is not a regression model like the ones we have seen so far. Rather than using least squares, MA(q) models are estimated using maximum likelihood (MLE).

### How to build ARIMA models and extensions

Autoregressive integrated moving-average ARIMA(p, d, q) models combine AR(p) and MA(q) processes to leverage the complementarity of these building blocks and simplify model development by using a more compact form and reducing the number of parameters, in turn reducing the risk of overfitting.

- statsmodels State-Space Models [docs](https://www.statsmodels.org/dev/statespace.html)

### How to forecast macro fundamentals

We will build a SARIMAX model for monthly data on an industrial production time series for the 1988-2017 period. See notebook [stationarity_and_arima](01_stationarity_and_arima.ipynb) for implementation details.

### How to use time series models to forecast volatility

A particularly important area of application for univariate time series models is the prediction of volatility. The volatility of financial time series is usually not constant over time but changes, with bouts of volatility clustering together. Changes in variance create challenges for time series forecasting using the classical ARIMA models.

- NYU Stern [VLAB](https://vlab.stern.nyu.edu/)

### How to build a volatility-forecasting model

The development of a volatility model for an asset-return series consists of four steps:
1. Build an ARMA time series model for the financial time series based on the serial dependence revealed by the ACF and PACF.
2. Test the residuals of the model for ARCH/GARCH effects, again relying on the ACF and PACF for the series of the squared residual.
3. Specify a volatility model if serial correlation effects are significant, and jointly estimate the mean and volatility equations.
4. Check the fitted model carefully and refine it if necessary.

The notebook [arch_garch_models](02_arch_garch_models.ipynb) demonstrates the usage of the ARCH library to estimate time series models for volatility foreccasting with NASDAQ data.  

- ARCH Library [examples](http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb)


## Multivariate Time Series Models

Multivariate time series models are designed to capture the dynamic of multiple time series simultaneously and leverage dependencies across these series for more reliable predictions.

- [New Introduction to Multiple Time Series Analysis, Lütkepohl, Helmut, Springer, 2005](https://www.springer.com/us/book/9783540401728)

### The vector autoregressive (VAR) model

- `statsmodels` Vector Autoregression [docs](https://www.statsmodels.org/dev/vector_ar.html)

- [Time Series Analysis in Python with statsmodels](https://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf), Wes McKinney, Josef Perktold, Skipper Seabold, SciPY Conference 2011

### How to use the VAR model for macro fundamentals forecasts

The notebook [vector_autoregressive_model](03_vector_autoregressive_model.ipynb) demonstrates how to use `statsmodels` to estimate a VAR model for macro fundamentals time series.

### Cointegration – time series with a common trend

The concept of an integrated multivariate series is complicated by the fact that all the component series of the process may be individually integrated but the process is not jointly integrated in the sense that one or more linear combinations of the series exist that produce a new stationary series.

In other words, a combination of two co-integrated series has a stable mean to which this linear combination reverts. A multivariate series with this characteristic is said to be co-integrated. This also applies when the individual series are integrated of a higher order and the linear combination reduces the overall order of integration. 

We demonstrate two major approaches to testing for cointegration:
- The Engle–Granger two-step method
- The Johansen procedure

### How to use cointegration for a pairs-trading strategy

Pairs-trading relies on a stationary, mean-reverting relationship between two asset prices. In other words, the ratio or difference between the two prices, also called the spread, may over time diverge but should ultimately return to the same level. Given such a pair, the strategy consists of going long (that is, purchasing) the under-performing asset because it would require a period of outperformance to close the gap. At the same time, one would short the asset that has moved away from the price anchor in the positive direction to fund the purchase.

In practice, given a universe of assets, a pairs-trading strategy will search for co-integrated pairs by running a statistical test on each pair. The key challenge here is to account for multiple testing biases, as outlined in [Chapter 6, Machine Learning Workflow](../06_machine_learning_process).

- [Introduction to Pairs Trading](https://www.quantopian.com/lectures/introduction-to-pairs-trading)