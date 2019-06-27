#!/usr/bin/env python
# coding: utf-8

# ## zipline installation

# Please follow latest instructions [here](https://www.zipline.io/install.html)

# It is recommended to install Zipline in an isolated conda environment. Installing Zipline in conda environments will not interfere your default Python deployment or site-packages, which will prevent any possible conflict with your global libraries.

# Use the `environment.yml`in this directory to install `zipline` and dependencies in a separate `conda` environment.

# ```
# conda env create -f environment.yml
# ```

# ## Ingest Data

# Get QUANDL API key and follow instructions to download zipline bundles [here](http://www.zipline.io/bundles.html).

# ## Imports

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from zipline.api import order_target, record, symbol
from zipline.examples import buyapple


# In[2]:


get_ipython().run_line_magic('load_ext', 'zipline')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## zipline Demo

# Zipline is the algorithmic trading library that powers the Quantopian backtesting and live-trading platform. It is also available offline to develop a strategy using a limited number of free data bundles that can be ingested and used to test the performance of trading ideas before porting the result to the online Quantopian platform for paper and live trading.

# ### Data access using zipline

# See `zipline` [docs](http://www.zipline.io/bundles.html) on the download and management of data bundles used to simulate backtests. 
# 
# The following commandline instruction lists the available bundles (store per default in `~/.zipline`. 

# In[3]:


get_ipython().system('zipline bundles')


# The following code illustrates how zipline permits us to access daily stock data for a range of companies. You can run zipline scripts in the Jupyter Notebook using the magic function of the same name.

# First, you need to initialize the context with the desired security symbols. We'll also use a counter variable. Then zipline calls handle_data, where we use the `data.history()` method to look back a single period and append the data for the last day to a .csv file:

# In[4]:


get_ipython().run_cell_magic('zipline', '--start 2010-1-1 --end 2018-1-1 --data-frequency daily', 'from zipline.api import order_target, record, symbol\nimport pandas as pd\n\ndef initialize(context):\n    context.i = 0\n    context.assets = [symbol(\'FB\'), symbol(\'GOOG\'), symbol(\'AMZN\')]\n    \ndef handle_data(context, data):\n    df = data.history(context.assets, fields=[\'price\', \'volume\'], bar_count=1, frequency="1d")\n    df = df.to_frame().reset_index()\n    \n    if context.i == 0:\n        df.columns = [\'date\', \'asset\', \'price\', \'volume\']\n        df.to_csv(\'stock_data.csv\', index=False)\n    else:\n        df.to_csv(\'stock_data.csv\', index=False, mode=\'a\', header=None)\n    context.i += 1')


# We can plot the data as follows:

# In[5]:


df = pd.read_csv('stock_data.csv')
df.date = pd.to_datetime(df.date)
df.set_index('date').groupby('asset').price.plot(lw=2, legend=True, figsize=(14, 6))


# ### Simple moving average strategy

# The following code example illustrates a [Dual Moving Average Cross-Over Strategy](https://www.zipline.io/beginner-tutorial.html#access-to-previous-prices-using-history) to demonstrate zipline in action:

# In[6]:


get_ipython().run_cell_magic('zipline', '--start 2014-1-1 --end 2018-1-1 -o dma.pickle', 'from zipline.api import order_target, record, symbol\nimport matplotlib.pyplot as plt\n\ndef initialize(context):\n    context.i = 0\n    context.asset = symbol(\'AAPL\')\n\n\ndef handle_data(context, data):\n    # Skip first 300 days to get full windows\n    context.i += 1\n    if context.i < 300:\n        return\n\n    # Compute averages\n    # data.history() has to be called with the same params\n    # from above and returns a pandas dataframe.\n    short_mavg = data.history(context.asset, \'price\', bar_count=100, frequency="1d").mean()\n    long_mavg = data.history(context.asset, \'price\', bar_count=300, frequency="1d").mean()\n\n    # Trading logic\n    if short_mavg > long_mavg:\n        # order_target orders as many shares as needed to\n        # achieve the desired number of shares.\n        order_target(context.asset, 100)\n    elif short_mavg < long_mavg:\n        order_target(context.asset, 0)\n\n    # Save values for later inspection\n    record(AAPL=data.current(context.asset, \'price\'),\n           short_mavg=short_mavg,\n           long_mavg=long_mavg)\n\n\ndef analyze(context, perf):\n    fig = plt.figure()\n    ax1 = fig.add_subplot(211)\n    perf.portfolio_value.plot(ax=ax1)\n    ax1.set_ylabel(\'portfolio value in $\')\n\n    ax2 = fig.add_subplot(212)\n    perf[\'AAPL\'].plot(ax=ax2)\n    perf[[\'short_mavg\', \'long_mavg\']].plot(ax=ax2)\n\n    perf_trans = perf.ix[[t != [] for t in perf.transactions]]\n    buys = perf_trans.ix[[t[0][\'amount\'] > 0 for t in perf_trans.transactions]]\n    sells = perf_trans.ix[\n        [t[0][\'amount\'] < 0 for t in perf_trans.transactions]]\n    ax2.plot(buys.index, perf.short_mavg.ix[buys.index],\n             \'^\', markersize=10, color=\'m\')\n    ax2.plot(sells.index, perf.short_mavg.ix[sells.index],\n             \'v\', markersize=10, color=\'k\')\n    ax2.set_ylabel(\'price in $\')\n    plt.legend(loc=0)\n    plt.show() ')


# In[ ]:




