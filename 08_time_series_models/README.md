# Chapter 08: Linear Time Series Models

## Analytical tools for diagnostics and feature extraction

- `pandas` Time Series and Date functionality [docs](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)


### How to decompose time series patterns

- [Forecasting - Principles & Practice, Hyndman, R. and Athanasopoulos, G., ch.6 'Time Series Decomposition'](https://otexts.org/fpp2/decomposition.html)

### How to compute rolling window statistics

- `pandas` window function [docs](https://pandas.pydata.org/pandas-docs/stable/computation.html#window-functions)
### How to measure autocorrelation

### How to diagnose and achieve stationarity

### How to apply time series transformations

### Code Examples

The code examples for this section are available in the notebook `tsa_and_arima`

## Univariate Time Series Models

- [Analysis of Financial Time Series, 3rd Edition, Ruey S. Tsay](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354)

- [Quantitative Equity Investing: Techniques and Strategies, Frank J. Fabozzi, Sergio M. Focardi, Petter N. Kolm](https://www.wiley.com/en-us/Quantitative+Equity+Investing%3A+Techniques+and+Strategies-p-9780470262474)

- `statsmodels` Time Series Analysis [docs](https://www.statsmodels.org/dev/tsa.html)
### How to build autoregressive models

### How to build moving average models

### How to build ARIMA models and extensions

- statsmodels State-Space Models [docs](https://www.statsmodels.org/dev/statespace.html)

### How to forecast macro fundamentals

### How to use time series models to forecast volatility

- NYU Stern [VLAB](https://vlab.stern.nyu.edu/)

#### Code Example

- ARCH Library [examples](http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb)

The code examples for this section are available in the notebook `arch_garch_models`

## Multivariate Time Series Models

- [New Introduction to Multiple Time Series Analysis, Lütkepohl, Helmut, Springer, 2005](https://www.springer.com/us/book/9783540401728)

### The vector autoregressive (VAR) model

- `statsmodels` Vector Autoregression [docs](https://www.statsmodels.org/dev/vector_ar.html)

- [Time Series Analysis in Python with statsmodels](https://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf), Wes McKinney, Josef Perktold, Skipper Seabold, SciPY Conference 2011

### How to use the VAR model for macro fundamentals forecasts

#### Code Example

The code examples for this section are available in the notebook `vector_autoregressive_models`

### Cointegration – time series with a common trend

### How to use cointegration for a pairs-trading strategy

- [Introduction to Pairs Trading](https://www.quantopian.com/lectures/introduction-to-pairs-trading)