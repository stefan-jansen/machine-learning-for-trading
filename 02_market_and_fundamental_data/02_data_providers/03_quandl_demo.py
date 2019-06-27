#!/usr/bin/env python
# coding: utf-8

# # Quandl

# Quandl uses a very straightforward API to make its free and premium data available. See [documentation](https://www.quandl.com/tools/api) for more details.

# In[1]:


import quandl
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
oil = quandl.get('EIA/PET_RWTC_D').squeeze()
oil.plot(lw=2, title='WTI Crude Oil Price')
plt.show()
