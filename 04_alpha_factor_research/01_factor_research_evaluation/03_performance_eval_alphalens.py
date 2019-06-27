#!/usr/bin/env python
# coding: utf-8

# # Separating signal and noise – how to use alphalens

# Quantopian has open sourced the Python library, alphalens, for the performance analysis of predictive stock factors that integrates well with the backtesting library zipline and the portfolio performance and risk analysis library pyfolio that we will explore in the next chapter.
# alphalens facilitates the analysis of the predictive power of alpha factors concerning the:
# - Correlation of the signals with subsequent returns
# - Profitability of an equal or factor-weighted portfolio based on a (subset of) the signals
# - Turnover of factors to indicate the potential trading costs
# - Factor-performance during specific events
# - Breakdowns of the preceding by sector
# 
# The analysis can be conducted using tearsheets or individual computations and plots.

# ## Imports & Settings

# In[1]:


import re
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import *
from alphalens.plotting import *
from alphalens.tears import *


# In[2]:


warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# ## Creating forward returns and factor quantiles

# To utilize `alpahalens`, we need to provide signals for a universe of assets like those returned by the ranks of the MeanReversion factor, and the forward returns earned by investing in an asset for a given holding period. .
# 
# We will recover the prices from the single_factor.pickle file as follows (factor_data accordingly):

# In[3]:


performance = pd.read_pickle('single_factor.pickle')


# In[4]:


prices = pd.concat([df.to_frame(d) for d, df in performance.prices.items()],axis=1).T
prices.columns = [re.findall(r"\[(.+)\]", str(col))[0] for col in prices.columns]
prices.index = prices.index.normalize()
prices.info()


# In[5]:


factor_data = pd.concat([df.to_frame(d) for d, df in performance.factor_data.items()],axis=1).T
factor_data.columns = [re.findall(r"\[(.+)\]", str(col))[0] for col in factor_data.columns]
factor_data.index = factor_data.index.normalize()
factor_data = factor_data.stack()
factor_data.index.names = ['date', 'asset']
factor_data.head()


# In[10]:


with pd.HDFStore('../../data/assets.h5') as store:
    sp500 = store['sp500/prices'].close
sp500 = sp500.resample('D').ffill().tz_localize('utc').filter(prices.index.get_level_values(0))
sp500.head()


# We can create the alphalens input data in the required format using the `get_clean_factor_and_forward_returns` utility function that also returns the signal quartiles and the forward returns for the given holding periods:

# In[11]:


HOLDING_PERIODS = (5, 10, 21, 42)
QUANTILES = 5
alphalens_data = get_clean_factor_and_forward_returns(factor=factor_data,
                                                      prices=prices,
                                                      periods=HOLDING_PERIODS,
                                                      quantiles=QUANTILES)


# The `alphalens_data` `DataFrame` contains the returns on an investment in the given asset on a given date for the indicated holding period, as well as the factor value, that is, the asset's `MeanReversion` ranking on that date, and the corresponding quantile value:

# In[12]:


alphalens_data.head()


# The forward returns and the signal quantiles are the basis for evaluating the predictive power of the signal. Typically, a factor should deliver markedly different returns for distinct quantiles, such as negative returns for the bottom quintile of the factor values and positive returns for the top quantile.

# ## Summary Tear Sheet

# In[13]:


create_summary_tear_sheet(alphalens_data)


# ## Predictive performance by factor quantiles -  Returns Analysis

# As a first step, we would like to visualize the average period return by factor quantile. We can use the built-in function mean_return_by_quantile from the performance and plot_quantile_returns_bar from the plotting modules

# In[14]:


mean_return_by_q, std_err = mean_return_by_quantile(alphalens_data)
mean_return_by_q_norm = mean_return_by_q.apply(lambda x: x.add(1).pow(1/int(x.name[:-1])).sub(1))


# ### Mean Return by Holding Period and Quintile

# The result is a bar chart that breaks down the mean of the forward returns for the four different holding periods based on the quintile of the factor signal. As you can see, the bottom quintiles yielded markedly more negative results than the top quintiles, except for the longest holding period:
# 

# In[27]:


plot_quantile_returns_bar(mean_return_by_q)
# plt.savefig('mean_return', dpi=300);


# The 10D holding period provides slightly better results for the first and fourth quartiles. We would also like to see the performance over time of investments driven by each of the signal quintiles. 
# 
# We will calculate daily, as opposed to average returns for the 5D holding period, and alphalens will adjust the period returns to account for the mismatch between daily signals and a longer holding period (for details, see docs):

# In[28]:


mean_return_by_q_daily, std_err = mean_return_by_quantile(alphalens_data, by_date=True)


# ### Cumulative 5D Return

# The resulting line plot shows that, for most of this three-year period, the top two quintiles significantly outperformed the bottom two quintiles. However, as suggested by the previous plot, signals by the fourth quintile produced a better performance than those by the top quintile

# In[ ]:


plot_cumulative_returns_by_quantile(mean_return_by_q_daily['5D'], period='5D', freq='Day')


# ### Return Distribution by Holding Period and Quintile

# This distributional plot highlights that the range of daily returns is fairly wide and, despite different means, the separation of the distributions is very limited so that, on any given day, the differences in performance between the different quintiles may be rather limited:

# In[22]:


plot_quantile_returns_violin(mean_return_by_q_daily)
# plt.savefig('mean_ret', dpi=300);


# ## Information Coefficient

# Most of this book is about the design of alpha factors using ML models. ML is about optimizing some predictive objective, and in this section, we will introduce the key metrics used to measure the performance of an alpha factor. We will define alpha as the average return in excess of a benchmark.
# This leads to the information ratio (IR) that measures the average excess return per unit of risk taken by dividing alpha by the tracking risk. When the benchmark is the risk-free rate, the IR corresponds to the well-known Sharpe ratio, and we will highlight crucial statistical measurement issues that arise in the typical case when returns are not normally distributed. We will also explain the fundamental law of active management that breaks the IR down into a combination of forecasting skill and a strategy's ability to effectively leverage the forecasting skills.

# ### 5D Information Coefficient (Rolling Average)

# The goal of alpha factors is the accurate directional prediction of future returns. Hence, a natural performance measure is the correlation between an alpha factor's predictions and the forward returns of the target assets. 
# 
# It is better to use the non-parametric Spearman rank correlation coefficient that measures how well the relationship between two variables can be described using a monotonic function, as opposed to the Pearson correlation that measures the strength of a linear relationship. 
# 
# We can obtain the information coefficient using alphalens, which relies on `scipy.stats.spearmanr` under the hood. 
# 
# The `factor_information_coefficient` function computes the period-wise correlation and plot_ic_ts creates a time-series plot with one-month moving average:

# In[23]:


ic = factor_information_coefficient(alphalens_data)
plot_ic_ts(ic[['5D']])
# plt.savefig('violin', dpi=300);


# #### Information Coefficient by Holding Period

# This time series plot shows extended periods with significantly positive moving-average IC. An IC of 0.05 or even 0.1 allows for significant outperformance if there are sufficient opportunities to apply this forecasting skill, as the fundamental law of active management will illustrate:
# 
# A plot of the annual mean IC highlights how the factor's performance was historically uneven:

# In[24]:


ic = factor_information_coefficient(alphalens_data)
ic_by_year = ic.resample('A').mean()
ic_by_year.index = ic_by_year.index.year
ic_by_year.plot.bar(figsize=(14, 6))
# plt.savefig('ic', dpi=300);


# ### Summary Tear Sheet

# In[25]:


create_summary_tear_sheet(alphalens_data);


# ### Turnover Tear Sheet

# Factor turnover measures how frequently the assets associated with a given quantile change, that is, how many trades are required to adjust a portfolio to the sequence of signals. More specifically, it measures the share of assets currently in a factor quantile that was not in that quantile in the last period.

# In[27]:


create_turnover_tear_sheet(alphalens_data);


# In[ ]:




