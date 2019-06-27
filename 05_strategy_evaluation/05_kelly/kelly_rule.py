#!/usr/bin/env python
# coding: utf-8

# # How to size your bets - The Kelly Rule

# he Kelly rule has a long history in gambling because it provides guidance on how much to stake on each of an (infinite) sequence of bets with varying (but favorable) odds to maximize terminal wealth. It was published as A New Interpretation of the Information Rate in 1956 by John Kelly who was a colleague of Claude Shannon's at Bell Labs. He was intrigued by bets placed on candidates at the new quiz show The $64,000 Question, where a viewer on the west coast used the three-hour delay to obtain insider information about the winners. 
# 
# Kelly drew a connection to Shannon's information theory to solve for the bet that is optimal for long-term capital growth when the odds are favorable, but uncertainty remains. His rule maximizes logarithmic wealth as a function of the odds of success of each game, and includes implicit bankruptcy protection since log(0) is negative infinity so that a Kelly gambler would naturally avoid losing everything.

# ## Imports

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sympy import symbols, solve, log, diff
from scipy.optimize import minimize_scalar, newton, minimize
from scipy.integrate import quad
from scipy.stats import norm
import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import dirichlet
import warnings


# In[3]:


warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
np.random.seed(42)


# ## The optimal size of a bet

# Kelly began by analyzing games with a binary win-lose outcome. The key variables are:
# - b: The odds define the amount won for a \\$1 bet. Odds = 5/1 implies a \\$5 gain if the bet wins, plus recovery of the \\$1 capital.
# - p: The probability defines the likelihood of a favorable outcome.
# - f: The share of the current capital to bet.
# - V: The value of the capital as a result of betting.
# 
# The Kelly rule aims to maximize the value's growth rate, G, of infinitely-repeated bets (see Chapter 5 for background).
# $$G=\lim_{N\rightarrow\infty}=\frac{1}{N}\log\frac{V_N}{V_0}$$

# We can maximize the rate of growth G by maximizing G with respect to f, as illustrated using sympy as follows:

# In[4]:


share, odds, probability = symbols('share odds probability')
Value = probability * log(1 + odds * share) + (1 - probability) * log(1 - share)
solve(diff(Value, share), share)


# In[5]:


f, p = symbols('f p')
y = p * log(1 + f) + (1 - p) * log(1 - f)
solve(diff(y, f), f)


# ## Get S&P 500 Data

# In[6]:


with pd.HDFStore('../../data/assets.h5') as store:
    sp500 = store['sp500/prices'].close


# ### Compute Returns & Standard Deviation

# In[7]:


annual_returns = sp500.resample('A').last().pct_change().to_frame('sp500')


# In[8]:


return_params = annual_returns.sp500.rolling(25).agg(['mean', 'std']).dropna()


# In[9]:


return_ci = (return_params[['mean']]
                .assign(lower=return_params['mean'].sub(return_params['std'].mul(2)))
                .assign(upper=return_params['mean'].add(return_params['std'].mul(2))))


# In[10]:


return_ci.plot(lw=2, figsize=(14, 8));


# ### Kelly Rule for a Single Asset - Index Returns

# In a financial market context, both outcomes and alternatives are more complex, but the Kelly rule logic does still apply. It was made popular by Ed Thorp, who first applied it profitably to gambling (described in Beat the Dealer) and later started the successful hedge fund Princeton/Newport Partners.
# 
# With continuous outcomes, the growth rate of capital is defined by an integrate over the probability distribution of the different returns that can be optimized numerically.
# We can solve this expression (see book) for the optimal f* using the `scipy.optimize` module:

# In[15]:


def norm_integral(f, mean, std):
    val, er = quad(lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std), 
                               mean - 3 * std, 
                               mean + 3 * std)
    return -val


# In[16]:


def norm_dev_integral(f, mean, std):
    val, er = quad(lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std), m-3*std, mean+3*std)
    return val


# In[17]:


def get_kelly_share(data):
    solution = minimize_scalar(norm_integral, 
                        args=(data['mean'], data['std']), 
                        bounds=[0, 2], 
                        method='bounded') 
    return solution.x


# In[18]:


annual_returns['f'] = return_params.apply(get_kelly_share, axis=1)


# In[19]:


return_params.plot(subplots=True, lw=2, figsize=(14, 8));


# In[20]:


annual_returns.tail()


# ### Performance Evaluation

# In[21]:


(annual_returns[['sp500']]
 .assign(kelly=annual_returns.sp500.mul(annual_returns.f.shift()))
 .dropna()
 .loc['1900':]
 .add(1)
 .cumprod()
 .sub(1)
 .plot(lw=2));


# In[22]:


annual_returns.f.describe()


# In[23]:


return_ci.head()


# ### Compute Kelly Fraction

# In[22]:


m = .058
s = .216


# In[24]:


# Option 1: minimize the expectation integral
sol = minimize_scalar(norm_integral, args=(m, s), bounds=[0., 2.], method='bounded')
print('Optimal Kelly fraction: {:.4f}'.format(sol.x))


# In[25]:


# Option 2: take the derivative of the expectation and make it null
x0 = newton(norm_dev_integral, .1, args=(m, s))
print('Optimal Kelly fraction: {:.4f}'.format(x0))


# ## Kelly Rule for Multiple Assets

# We will use an example with various equities. [E. Chan (2008)](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889) illustrates how to arrive at a multi-asset application of the Kelly Rule, and that the result is equivalent to the (potentially levered) maximum Sharpe ratio portfolio from the mean-variance optimization. 
# 
# The computation involves the dot product of the precision matrix, which is the inverse of the covariance matrix, and the return matrix:

# In[24]:


with pd.HDFStore('../../data/assets.h5') as store:
    sp500_stocks = store['sp500/stocks'].index 
    prices = store['quandl/wiki/prices'].adj_close.unstack('ticker').filter(sp500_stocks)


# In[25]:


prices.info()


# In[26]:


monthly_returns = prices.loc['1988':'2017'].resample('M').last().pct_change().dropna(how='all').dropna(axis=1)
stocks = monthly_returns.columns
monthly_returns.info()


# ### Compute Precision Matrix

# In[41]:


cov = monthly_returns.cov()
precision_matrix = pd.DataFrame(inv(cov), index=stocks, columns=stocks)


# In[42]:


kelly_allocation = monthly_returns.mean().dot(precision_matrix)


# In[43]:


kelly_allocation.describe()


# In[44]:


kelly_allocation.sum()


# ### Largest Portfolio Allocation

# The plot shows the tickers that receive an allocation weight > 5x their value:

# In[75]:


kelly_allocation[kelly_allocation.abs()>5].sort_values(ascending=False).plot.barh(figsize=(8, 10))
plt.yticks(fontsize=12)
plt.tight_layout()


# ### Performance vs SP500

# The Kelly rule does really well. But it has also been computed from historical data..

# In[88]:


ax = monthly_returns.loc['2010':].mul(kelly_allocation.div(kelly_allocation.sum())).sum(1).to_frame('Kelly').add(1).cumprod().sub(1).plot();
sp500.loc[monthly_returns.loc['2010':].index].pct_change().add(1).cumprod().sub(1).to_frame('SP500').plot(ax=ax, legend=True);


# In[ ]:




