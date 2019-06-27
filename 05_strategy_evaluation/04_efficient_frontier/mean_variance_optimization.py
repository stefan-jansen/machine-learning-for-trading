#!/usr/bin/env python
# coding: utf-8

# Mean-Variance Optimization

# Imports & Settings

import warnings

import pandas as pd
import numpy as np
from numpy.random import random, uniform, normal, dirichlet, choice
from numpy.linalg import inv

from scipy.optimize import minimize

import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
np.random.seed(42)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')

cmap = sns.diverging_palette(10, 240, n=9, as_cmap=True)

# Prepare Data

with pd.HDFStore('../../data/assets.h5') as store:
    sp500_stocks = store['sp500/stocks'].index
    prices = store['quandl/wiki/prices'].adj_close.unstack('ticker').filter(sp500_stocks).sample(n=50, axis=1)

# Compute Returns
start = 1988
end = 2017

# Create month-end monthly returns and drop dates that have no observations:

monthly_returns = prices.loc[f'{start}':f'{end}'].resample('M').last().pct_change().dropna(how='all')
monthly_returns = monthly_returns.dropna(axis=1)
print(monthly_returns.info())

# Set  Parameters
stocks = monthly_returns.columns
n_obs, n_assets = monthly_returns.shape
print(n_assets, n_obs)

NUM_PF = 100000  # no of portfolios to simulate

x0 = uniform(-1, 1, n_assets)
x0 /= np.sum(np.abs(x0))

# Annualization Factor
periods_per_year = round(monthly_returns.resample('A').size().mean())
print(periods_per_year)

# Compute Covariance

mean_returns = monthly_returns.mean()
cov_matrix = monthly_returns.cov()
precision_matrix = pd.DataFrame(inv(cov_matrix), index=stocks, columns=stocks)

treasury_10yr_monthly = (web.DataReader('DGS10', 'fred', start, end)
                         .resample('M')
                         .last()
                         .div(periods_per_year)
                         .div(100)
                         .squeeze())

# Risk-Free Rate
rf_rate = treasury_10yr_monthly.mean()

alpha = np.full(shape=n_assets, fill_value=.01)
weights = dirichlet(alpha=alpha, size=NUM_PF)
weights *= choice([-1, 1], size=weights.shape)
print(weights.shape)


# Simulate Random Portfolios
def simulate_portfolios(mean_ret, cov, rf_rate=rf_rate, short=True):
    alpha = np.full(shape=n_assets, fill_value=.025)
    weights = dirichlet(alpha=alpha, size=NUM_PF)
    weights *= choice([-1, 1], size=weights.shape)

    returns = weights @ mean_ret.values + 1
    returns = returns ** periods_per_year - 1
    std = (weights @ monthly_returns.T).std(1)
    std *= np.sqrt(periods_per_year)
    sharpe = (returns - rf_rate) / std

    return pd.DataFrame({'Annualized Standard Deviation': std,
                         'Annualized Returns'           : returns,
                         'Sharpe Ratio'                 : sharpe}), weights


simul_perf, simul_wt = simulate_portfolios(mean_returns, cov_matrix, short=False)

# Plot Simulated Portfolios

# pandas 0.24 will fix bug with colorbars: https://github.com/pandas-dev/pandas/pull/20446
ax = simul_perf.plot.scatter(x=0, y=1, c=2,
                             cmap='RdBu',
                             alpha=0.5, figsize=(14, 9), colorbar=False,
                             title=f'{NUM_PF:,d} Simulated Portfolios')

max_sharpe_idx = simul_perf.iloc[:, 2].idxmax()
sd, r = simul_perf.iloc[max_sharpe_idx, :2].values
ax.scatter(sd, r, marker='*', color='darkblue',
           s=500, label='Max. Sharpe Ratio')

min_vol_idx = simul_perf.iloc[:, 0].idxmin()
sd, r = simul_perf.iloc[min_vol_idx, :2].values
ax.scatter(sd, r, marker='*', color='green', s=500, label='Min Volatility')
plt.legend(labelspacing=1, loc='upper left')
plt.tight_layout()
plt.show()


# Compute Annualized PF Performance

def portfolio_std(wt, rt=None, cov=None):
    """Annualized PF standard deviation"""
    return np.sqrt(wt @ cov @ wt * periods_per_year)


def portfolio_returns(wt, rt=None, cov=None):
    """Annualized PF returns"""
    return (wt @ rt + 1) ** periods_per_year - 1


def portfolio_performance(wt, rt, cov):
    """Annualized PF returns & standard deviation"""
    r = portfolio_returns(wt, rt=rt)
    sd = portfolio_std(wt, cov=cov)
    return r, sd


# Max Sharpe PF

def neg_sharpe_ratio(weights, mean_ret, cov):
    r, sd = portfolio_performance(weights, mean_ret, cov)
    return -(r - rf_rate) / sd


weight_constraint = {'type': 'eq',
                     'fun' : lambda x: np.sum(np.abs(x)) - 1}


def max_sharpe_ratio(mean_ret, cov, short=True):
    return minimize(fun=neg_sharpe_ratio,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=((-1 if short else 0, 1),) * n_assets,
                    constraints=weight_constraint,
                    options={'tol': 1e-10, 'maxiter': 1e4})


# Compute Efficient Frontier

def min_vol_target(mean_ret, cov, target, short=True):
    def ret_(wt):
        return portfolio_returns(wt, mean_ret)

    constraints = [{'type': 'eq', 'fun': lambda x: ret_(x) - target},
                   weight_constraint]

    bounds = ((-1 if short else 0, 1),) * n_assets
    return minimize(portfolio_std,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'tol': 1e-10, 'maxiter': 1e4})


# Min Volatility Portfolio
def min_vol(mean_ret, cov, short=True):
    bounds = ((-1 if short else 0, 1),) * n_assets

    return minimize(fun=portfolio_std,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=weight_constraint,
                    options={'tol': 1e-10, 'maxiter': 1e4})


def efficient_frontier(mean_ret, cov, ret_range, short=True):
    return [min_vol_target(mean_ret, cov, ret) for ret in ret_range]


# Get random PF

simul_perf, simul_wt = simulate_portfolios(mean_returns, cov_matrix, short=True)
print(simul_perf.describe())

simul_max_sharpe = simul_perf.iloc[:, 2].idxmax()
print(simul_perf.iloc[simul_max_sharpe])

# Get Max Sharpe PF

max_sharpe_pf = max_sharpe_ratio(mean_returns, cov_matrix, short=False)
max_sharpe_perf = portfolio_performance(max_sharpe_pf.x, mean_returns, cov_matrix)
r, sd = max_sharpe_perf
print(pd.Series({'ret': r, 'sd': sd, 'sr': (r - rf_rate) / sd}))

# Get Min Vol PF
min_vol_pf = min_vol(mean_returns, cov_matrix, short=False)
min_vol_perf = portfolio_performance(min_vol_pf.x, mean_returns, cov_matrix)

# Get Efficent PFs
ret_range = np.linspace(0, simul_perf.iloc[:, 1].max() * 1.1, 25)
eff_pf = efficient_frontier(mean_returns, cov_matrix, ret_range, short=False)

# Plot Result

fig, ax = plt.subplots()
simul_perf.plot.scatter(x=0, y=1, c=2, ax=ax,
                        cmap='RdBu',
                        alpha=0.5, figsize=(14, 9), colorbar=False,
                        title='Mean-Variance Efficient Frontier')

r, sd = max_sharpe_perf
ax.scatter(sd, r, marker='*', color='r', s=500, label='Max Sharpe Ratio PF')

r, sd = min_vol_perf
ax.scatter(sd, r, marker='*', color='g', s=500, label='Min Volatility PF')

kelly_wt = precision_matrix.dot(mean_returns).values
kelly_wt /= np.sum(np.abs(kelly_wt))
r, sd = portfolio_performance(kelly_wt, mean_returns, cov_matrix)
ax.scatter(sd, r, marker='*', color='darkgrey', s=500, label='Kelly PF')

std = monthly_returns.std()
std /= std.sum()
r, sd = portfolio_performance(std, mean_returns, cov_matrix)
ax.scatter(sd, r, marker='*', color='y', s=500, label='Risk Parity PF')

r, sd = portfolio_performance(np.full(n_assets, 1 / n_assets), mean_returns, cov_matrix)
ax.scatter(sd, r, marker='*', color='blue', s=500, label='1/n PF')

ax.plot([p['fun'] for p in eff_pf], ret_range,
        linestyle='--', lw=3, color='darkgrey', label='Efficient Frontier')
ax.legend(labelspacing=0.8)
fig.tight_layout()
plt.show()

std = monthly_returns.std()
std /= std.sum()
r, sd = portfolio_performance(std, mean_returns, cov_matrix)
print(r, sd)
