#!/usr/bin/env python
# coding: utf-8

# How to transform data into factors

import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_datareader.data as web
from pyfinance.ols import PandasRollingOLS

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
idx = pd.IndexSlice

# Get Data
DATA_STORE = '../../data/assets.h5'

with pd.HDFStore(DATA_STORE) as store:
    prices = store['quandl/wiki/prices'].loc[idx['2000':'2018', :], 'adj_close'].unstack('ticker')
    stocks = store['us_equities/stocks'].loc[:, ['marketcap', 'ipoyear', 'sector']]

# Keep data with stock info
# Remove `stocks` duplicates and align index names for later joining.
stocks = stocks[~stocks.index.duplicated()]
stocks.index.name = 'ticker'

# Get tickers with both price information and metdata
shared = prices.columns.intersection(stocks.index)
stocks = stocks.loc[shared, :]
print(stocks.info())

prices = prices.loc[:, shared]
print(prices.info())

assert prices.shape[1] == stocks.shape[0]

# ## Create monthly return series
# To reduce training time and experiment with strategies for longer time horizons,
# we convert the business-daily data to month-end frequency using the available adjusted close price:
monthly_prices = prices.resample('M').last()

# To capture time series dynamics that reflect, for example, momentum patterns,
# we compute historical returns using the method `.pct_change(n_periods)`,
# that is, returns over various monthly periods as identified by lags.
# We then convert the wide result back to long format with the `.stack()` method,
# use `.pipe()` to apply the `.clip()` method to the resulting `DataFrame`,
# and winsorize returns at the [1%, 99%] levels; that is, we cap outliers at these percentiles.
# Finally, we normalize returns using the geometric average. After using `.swaplevel()`
# to change the order of the `MultiIndex` levels,
# we obtain compounded monthly returns for six periods ranging from 1 to 12 months:
outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    data[f'return_{lag}m'] = (monthly_prices
                              .pct_change(lag)
                              .stack()
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1 - outlier_cutoff)))
                              .add(1)
                              .pow(1 / lag)
                              .sub(1)
                              )
data = data.swaplevel().dropna()
print(data.info())

# ## Drop stocks with less than 10 yrs of returns
min_obs = 120
nobs = data.groupby(level='ticker').size()
keep = nobs[nobs > min_obs].index

data = data.loc[idx[keep, :], :]
print(data.info())
print(data.describe())

# cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues')

# We are left with 1,775 tickers.
print(data.index.get_level_values('ticker').nunique())

# Rolling Factor Betas

# We will introduce the Fama—French data to estimate the exposure of assets to common risk factors
# using linear regression in [Chapter 8, Time Series Models]([](../../08_time_series_models)).

# Use Fama-French research factors to estimate the factor exposures of the stock in the dataset
# to the 5 factors market risk, size, value, operating profitability and investment.
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2000')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
print(factor_data.info())

factor_data = factor_data.join(data['return_1m']).sort_index()
print(factor_data.info())

T = 24
betas = (factor_data
    .groupby(level='ticker', group_keys=False)
    .apply(
        lambda x: PandasRollingOLS(window=min(T, x.shape[0] - 1), y=x.return_1m, x=x.drop('return_1m', axis=1)).beta))

print(betas.describe().join(betas.sum(1).describe().to_frame('total')))

cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(betas.corr(), annot=True, cmap=cmap, center=0)
plt.show()

data = (data
        .join(betas
              .groupby(level='ticker')
              .shift()))
print(data.info())

# Impute mean for missing factor betas
data.loc[:, factors] = data.groupby('ticker')[factors].apply(lambda x: x.fillna(x.mean()))
print(data.info())

# Momentum factors

# We can use these results to compute momentum factors based on the difference between returns over longer periods
# and the most recent monthly return, as well as for the difference between 3 and 12 month returns as follows:
for lag in [2, 3, 6, 9, 12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
data[f'momentum_3_12'] = data[f'return_12m'].sub(data.return_3m)

# Date Indicators
dates = data.index.get_level_values('date')
data['year'] = dates.year
data['month'] = dates.month

# Lagged returns
# To use lagged values as input variables or features associated with the current observations,
# we use the .shift() method to move historical returns up to the current period:

for t in range(1, 7):
    data[f'return_1m_t-{t}'] = data.groupby(level='ticker').return_1m.shift(t)
data.info()

# Target: Holding Period Returns
# Similarly, to compute returns for various holding periods, we use the normalized period returns
# computed previously and shift them back to align them with the current financial features

for t in [1, 2, 3, 6, 12]:
    data[f'target_{t}m'] = data.groupby(level='ticker')[f'return_{t}m'].shift(-t)

cols = ['target_1m',
        'target_2m',
        'target_3m', 'return_1m',
        'return_2m',
        'return_3m',
        'return_1m_t-1',
        'return_1m_t-2',
        'return_1m_t-3']

print(data[cols].dropna().sort_index().head(10))
print(data.info())

# Create age proxy
# We use quintiles of IPO year as a proxy for company age.
data = (data
        .join(pd.qcut(stocks.ipoyear, q=5, labels=list(range(1, 6)))
              .astype(float)
              .fillna(0)
              .astype(int)
              .to_frame('age')))
data.age = data.age.fillna(-1)

# Create dynamic size proxy
# We use the marketcap information from the NASDAQ ticker info to create a size proxy.
print(stocks.info())

# Market cap information is tied to currrent prices
# We create an adjustment factor to have the values reflect lower historical prices for each individual stock:

size_factor = (monthly_prices
               .loc[data.index.get_level_values('date').unique(),
                    data.index.get_level_values('ticker').unique()]
               .sort_index(ascending=False)
               .pct_change()
               .fillna(0)
               .add(1)
               .cumprod())
print(size_factor.info())

msize = (size_factor
         .mul(stocks
              .loc[size_factor.columns, 'marketcap'])).dropna(axis=1, how='all')

# Create Size indicator as deciles per period
# Compute size deciles per month:
data['msize'] = (msize
                 .apply(lambda x: pd.qcut(x, q=10, labels=list(range(1, 11)))
                        .astype(int), axis=1)
                 .stack()
                 .swaplevel())
data.msize = data.msize.fillna(-1)

# Combine data
data = data.join(stocks[['sector']])
data.sector = data.sector.fillna('Unknown')
print(data.info())

# Store data
with pd.HDFStore(DATA_STORE) as store:
    store.put('engineered_features', data.sort_index().loc[idx[:, :datetime(2018, 3, 1)], :])
    print(store.info())

# Create Dummy variables
# For most models, we need to encode categorical variables as 'dummies' (one-hot encoding):
dummy_data = pd.get_dummies(data,
                            columns=['year', 'month', 'msize', 'age', 'sector'],
                            prefix=['year', 'month', 'msize', 'age', ''],
                            prefix_sep=['_', '_', '_', '_', ''])
dummy_data = dummy_data.rename(columns={c: c.replace('.0', '') for c in dummy_data.columns})
dummy_data.info()
