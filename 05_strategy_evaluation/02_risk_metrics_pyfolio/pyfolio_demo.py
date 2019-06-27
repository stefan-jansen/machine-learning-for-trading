#!/usr/bin/env python
# coding: utf-8

# # From `zipline` to `pyfolio`

# [Pyfolio](http://quantopian.github.io/pyfolio/) facilitates the analysis of portfolio performance and risk in-sample and out-of-sample using many standard metrics. It produces tear sheets covering the analysis of returns, positions, and transactions, as well as event risk during periods of market stress using several built-in scenarios, and also includes Bayesian out-of-sample performance analysis.
# 
# * Open-source backtester by Quantopian Inc.
# * Powers Quantopian.com
# * State-of-the-art portfolio and risk analytics
# * Various models for transaction costs and slippage.
# * Open source and free: Apache v2 license
# * Can be used:
#    - stand alone
#    - with Zipline
#    - on Quantopian

# Run this note the following from the command line to create a `conda` environment with `zipline` and `pyfolio`: 
# ```
# conda env create -f environment.yml
# ```
# This assumes you have miniconda3 installed.

# ## Imports & Settings

# In[1]:


from pathlib import Path
import warnings
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from pyfolio.utils import extract_rets_pos_txn_from_zipline
from pyfolio.plotting import (plot_perf_stats,
                              show_perf_stats,
                              plot_rolling_beta,
                              plot_rolling_fama_french,
                              plot_rolling_returns,
                              plot_rolling_sharpe,
                              plot_rolling_volatility,
                              plot_drawdown_periods,
                              plot_drawdown_underwater)

from pyfolio.timeseries import perf_stats, extract_interesting_date_ranges
# from pyfolio.tears import create_returns_tear_sheet


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# ## Converting data from zipline to pyfolio

# In[6]:


with pd.HDFStore('../01_trading_zipline/backtests.h5') as store:
    backtest = store['backtest']
backtest.info()


# `pyfolio` relies on portfolio returns and position data, and can also take into account the transaction costs and slippage losses of trading activity. The metrics are computed using the empyrical library that can also be used on a standalone basis. The performance DataFrame produced by the zipline backtesting engine can be translated into the requisite pyfolio input.

# In[7]:


returns, positions, transactions = extract_rets_pos_txn_from_zipline(backtest)


# In[8]:


returns.head().append(returns.tail())


# In[9]:


positions.columns = [c.symbol for c in positions.columns[:-1]] + ['cash']
positions.index = positions.index.normalize()
positions.info()


# In[10]:


transactions.symbol = transactions.symbol.apply(lambda x: x.symbol)


# In[11]:


transactions.head().append(transactions.tail())


# In[16]:


HDF_PATH = Path('..', '..', 'data', 'assets.h5')


# ### Sector Map

# In[17]:


assets = positions.columns[:-1]
with pd.HDFStore(HDF_PATH) as store:
    df = store.get('us_equities/stocks')['sector'].dropna()
    df = df[~df.index.duplicated()]
sector_map = df.reindex(assets).fillna('Unknown').to_dict()


# ### Benchmark

# In[18]:


with pd.HDFStore(HDF_PATH) as store:
    benchmark_rets = store['sp500/prices'].close.pct_change()
benchmark_rets.name = 'S&P500'
benchmark_rets = benchmark_rets.tz_localize('UTC').filter(returns.index)
benchmark_rets.tail()


# In[19]:


perf_stats(returns=returns, factor_returns=benchmark_rets, positions=positions, transactions=transactions)


# In[20]:


plot_perf_stats(returns=returns, factor_returns=benchmark_rets)


# ## Returns Analysis

# Testing a trading strategy involves backtesting against historical data to fine-tune alpha factor parameters, as well as forward-testing against new market data to validate that the strategy performs well out of sample or if the parameters are too closely tailored to specific historical circumstances.
# 
# Pyfolio allows for the designation of an out-of-sample period to simulate walk-forward testing. There are numerous aspects to take into account when testing a strategy to obtain statistically reliable results, which we will address here. 

# In[21]:


oos_date = '2017-01-01'


# In[22]:


show_perf_stats(returns=returns, 
                factor_returns=benchmark_rets, 
                positions=positions, 
                transactions=transactions, 
                live_start_date=oos_date)


# ### Rolling Returns OOS

# The `plot_rolling_returns` function displays cumulative in and out-of-sample returns against a user-defined benchmark (we are using the S&P 500):
# 

# In[23]:


plot_rolling_returns(returns=returns, factor_returns=benchmark_rets, live_start_date=oos_date, cone_std=(1.0, 1.5, 2.0))
plt.gcf().set_size_inches(14, 8)


# The plot includes a cone that shows expanding confidence intervals to indicate when out-of-sample returns appear unlikely given random-walk assumptions. Here, our strategy did not perform well against the benchmark during the simulated 2017 out-of-sample period

# ## Summary Performance Statistics

# pyfolio offers several analytic functions and plots. The perf_stats summary displays the annual and cumulative returns, volatility, skew, and kurtosis of returns and the SR. The following additional metrics (which can also be calculated individually) are most important:
# - Max drawdown: Highest percentage loss from the previous peak
# - Calmar ratio: Annual portfolio return relative to maximal drawdown
# - Omega ratio: The probability-weighted ratio of gains versus losses for a return target, zero per default
# - Sortino ratio: Excess return relative to downside standard deviation
# - Tail ratio: Size of the right tail (gains, the absolute value of the 95th percentile) relative to the size of the left tail (losses, abs. value of the 5th percentile) 
# - Daily value at risk (VaR): Loss corresponding to a return two standard deviations below the daily mean
# - Alpha: Portfolio return unexplained by the benchmark return
# - Beta: Exposure to the benchmark
# 

# ### Rolling Sharpe

# In[24]:


plot_rolling_sharpe(returns=returns)
plt.gcf().set_size_inches(14, 8);


# ### Rolling Beta

# In[25]:


plot_rolling_beta(returns=returns, factor_returns=benchmark_rets)
plt.gcf().set_size_inches(14, 8);


# ## Drawdown Periods

# The plot_drawdown_periods(returns) function plots the principal drawdown periods for the portfolio, and several other plotting functions show the rolling SR and rolling factor exposures to the market beta or the Fama French size, growth, and momentum factors:

# In[26]:


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = ax.flatten()

plot_drawdown_periods(returns=returns, ax=axes[0])
plot_rolling_beta(returns=returns, factor_returns=benchmark_rets, ax=axes[1])
plot_drawdown_underwater(returns=returns, ax=axes[2])
plot_rolling_sharpe(returns=returns)
plt.tight_layout();


# This plot, which highlights a subset of the visualization contained in the various tear sheets, illustrates how pyfolio allows us to drill down into the performance characteristics and exposure to fundamental drivers of risk and returns.

# ## Modeling Event Risk

# Pyfolio also includes timelines for various events that you can use to compare the performance of a portfolio to a benchmark during this period, for example, during the fall 2015 selloff following the Brexit vote.

# In[27]:


interesting_times = extract_interesting_date_ranges(returns=returns)
(interesting_times['Fall2015']
 .to_frame('pf').join(benchmark_rets)
 .add(1).cumprod().sub(1)
 .plot(lw=2, figsize=(14, 6), title='Post-Brexit Turmoil'))
plt.tight_layout()

