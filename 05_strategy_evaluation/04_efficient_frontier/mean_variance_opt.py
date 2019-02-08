# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.random import dirichlet

plt.style.use('fivethirtyeight')
np.random.seed(42)
pd.set_option('display.expand_frame_repr', False)

N_PORTFOLIOS = 2500000  # number of pf to simulate
RF_RATE = 0.0178
TRADING_DAYS = 252


with pd.HDFStore('mv.h5') as store:
    factor = store.get('factor').loc['2016':]
    prices = store.get('prices').loc['2010': '2015']

print(factor.info())
print(prices.info())

baseline = factor.iloc[0].dropna()
print(baseline.describe())
start = baseline.name
print(start)
base_pf = baseline.div(baseline.abs().sum())
print(base_pf.abs().sum())


assets = baseline.index.tolist()
n_assets = len(assets)  # number of assets to allocate

returns = prices.loc[:, assets].pct_change()
x0 = np.full(n_assets, 1 / n_assets)
mean_asset_ret = returns.mean()
asset_cov = returns.cov()


def pf_vol(weights, cov):
    return np.sqrt(weights.T @ (cov @ weights) * TRADING_DAYS)


def pf_ret(weights, mean_ret):
    return (weights @ mean_ret + 1) ** TRADING_DAYS - 1


def pf_performance(weights, mean_ret, cov):
    r = pf_ret(weights, mean_ret)
    sd = pf_vol(weights, cov)
    return r, sd


def simulate_pf(mean_ret, cov):
    perf, weights = [], []
    for i in range(N_PORTFOLIOS):
        if i % 50000 == 0:
            print(i)
        weights = dirichlet([.08] * n_assets)
        weights /= np.sum(weights)

        r, sd = pf_performance(weights, mean_ret, cov)
        perf.append([r, sd, (r - RF_RATE) / sd])
    perf_df = pd.DataFrame(perf, columns=['ret', 'vol', 'sharpe'])
    return perf_df, weights


def get_ret_vol(perf, idx):
    r = perf.loc[idx, 'ret']
    std = perf.loc[idx, 'vol']
    return r, std


def simulate_alloc(mean_ret, cov):
    perf, weights = simulate_pf(mean_ret, cov)

    df = pd.DataFrame()
    alloc = pd.DataFrame()
    max_sharpe_ix = perf.sharpe.idxmax()
    df['Max Sharpe'] = perf.loc[max_sharpe_ix, ['ret', 'vol']]
    alloc['Max Sharpe'] = pd.Series(weights[max_sharpe_ix], index=assets)

    min_std_idx = perf.vol.idxmin()
    df['Min Vol'] = perf.loc[min_std_idx, ['ret', 'vol']]
    alloc['Min Vol'] = pd.Series(weights[min_std_idx], index=assets)
    return perf, alloc, df


def simulate_efficient_frontier(mean_ret, cov):
    perf, alloc, df = simulate_alloc(mean_ret, cov)

    perf.plot.scatter(x='vol', y='ret', c='sharpe',
                      cmap='YlGnBu', marker='o', s=10,
                      alpha=0.3, figsize=(10, 7), colorbar=True,
                      title='PF Simulation')

    r, sd = df['Max Sharpe'].values
    plt.scatter(sd, r, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    r, sd = df['Min Vol'].values
    plt.scatter(sd, r, marker='*', color='g', s=500, label='Minimum volatility')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing=0.8)
    plt.savefig('Simulated EF.png')
    plt.close()

    alloc.sort_values('Max Sharpe', ascending=False).plot.bar(figsize=(12, 6))
    plt.savefig('allocations.png')


def neg_sharpe_ratio(weights, mean_ret, cov):
    r, sd = pf_performance(weights, mean_ret, cov)
    return -(r - RF_RATE) / sd


def max_sharpe_ratio(mean_ret, cov):
    args = (mean_ret, cov)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = ((0.0, 1.0),) * n_assets
    return minimize(fun=neg_sharpe_ratio,
                    x0=x0,
                    args=args,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints)


def pf_volatility(w, r, c):
    return pf_performance(w, r, c)[1]


def min_variance(mean_ret, cov):
    args = (mean_ret, cov)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = ((0.0, 1.0),) * n_assets

    return minimize(fun=pf_volatility,
                    x0=x0,
                    args=args,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints)


def efficient_return(mean_ret, cov, target):
    args = (mean_ret, cov)

    def ret_(weights):
        return pf_ret(weights, mean_ret)

    constraints = [{'type': 'eq', 'fun': lambda x: ret_(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    bounds = ((0.0, 1.0),) * n_assets

    # noinspection PyTypeChecker
    return minimize(pf_volatility,
                    x0=x0,
                    args=args, method='SLSQP',
                    bounds=bounds,
                    constraints=constraints)


def efficient_frontier(mean_ret, cov, ret_range):
    efficient_pf = []
    for ret in ret_range:
        efficient_pf.append(efficient_return(mean_ret, cov, ret))
    return efficient_pf


def calculate_efficient_frontier(mean_ret, cov):
    perf, wt = simulate_pf(mean_ret, cov)
    print(pd.DataFrame(wt).stack().describe())

    max_sharpe = max_sharpe_ratio(mean_ret, cov)
    max_sharpe_perf = pf_performance(max_sharpe.x, mean_ret, cov)
    wmax = max_sharpe.x
    print(np.sum(wmax))

    min_vol = min_variance(mean_ret, cov)
    min_vol_perf = pf_performance(min_vol['x'], mean_ret, cov)

    pf = ['Max Sharpe', 'Min Vol']
    alloc = pd.DataFrame(dict(zip(pf, [max_sharpe.x, min_vol.x])), index=assets)
    selected_pf = pd.DataFrame(dict(zip(pf, [max_sharpe_perf, min_vol_perf])),
                               index=['ret', 'vol'])

    print(selected_pf)
    print(perf.describe())

    perf.plot.scatter(x='vol', y='ret', c='sharpe',
                      cmap='YlGnBu', marker='o', s=10,
                      alpha=0.3, figsize=(10, 7), colorbar=True,
                      title='PF Simulation')

    r, sd = selected_pf['Max Sharpe'].values
    plt.scatter(sd, r, marker='*', color='r', s=500, label='Max Sharpe Ratio')
    r, sd = selected_pf['Min Vol'].values
    plt.scatter(sd, r, marker='*', color='g', s=500, label='Min volatility')
    plt.xlabel('Annualised Volatility')
    plt.ylabel('Annualised Returns')
    plt.legend(labelspacing=0.8)

    rmin = selected_pf.loc['ret', 'Min Vol']
    rmax = returns.add(1).prod().pow(1 / len(returns)).pow(TRADING_DAYS).sub(1).max()
    ret_range = np.linspace(rmin, rmax, 50)
    # ret_range = np.linspace(rmin, .22, 50)
    efficient_portfolios = efficient_frontier(mean_asset_ret, cov, ret_range)

    plt.plot([p['fun'] for p in efficient_portfolios], ret_range, linestyle='-.', color='black',
             label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    plt.tight_layout()
    plt.savefig('Calculated EF.png')


calculate_efficient_frontier(mean_asset_ret, asset_cov)
