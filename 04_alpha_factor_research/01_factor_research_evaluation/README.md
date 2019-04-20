## Seeking signals – how to use zipline

Historically, alpha factors used a single input and simple heuristics, thresholds or quantile cutoffs to identify buy or sell signals. ML has proven quite effective in extracting signals from a more diverse and much larger set of input data, including other alpha factors based on the analysis of historical patterns. As a result, algorithmic trading strategies today leverage a large number of alpha signals, many of which may be weak individually but can yield reliable predictions when combined with other model-driven or traditional factors by an ML algorithm.

The open source [zipline](http://www.zipline.io/index.html) library is an event-driven backtesting system maintained and used in production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development and live-trading. 

It automates the algorithm's reaction to trade events and provides it with current and historical point-in-time data that avoids look-ahead bias.

### The architecture – event-driven trading simulation

- A zipline algorithm will run for a specified period after an initial setup and executes its trading logic when specific events occur. 
- These events are driven by the trading frequency and can also be scheduled by the algorithm, and result in zipline calling certain methods. 
- The algorithm maintains state through a context dictionary and receives actionable information through a data variable containing point-in-time (PIT) current and historical data. 
- The algorithm returns a DataFrame containing portfolio performance metrics if there were any trades, as well as user-defined metrics that can be used to record, for example, the factor values.

The Pipeline API facilitates the definition and computation of alpha factors for a cross-section of securities from historical data. A pipeline defines computations that produce columns in a table with PIT values for a set of securities. It needs to be registered with the initialize() method and can then be executed on an automatic or custom schedule. The library provides numerous built-in computations such as moving averages or Bollinger Bands that can be used to quickly compute standard factors but also allows for the creation of custom factors as we will illustrate next. 

Most importantly, the Pipeline API renders alpha factor research modular because it separates the alpha factor computation from the remainder of the algorithm, including the placement and execution of trade orders and the bookkeeping of portfolio holdings, values, and so on.

### A single alpha factor from market data

We are first going to illustrate the `zipline` alpha factor research workflow in an offline environment. 

The notebook [single_factor_zipline](01_single_factor_zipline.ipynb) develops and test a simple mean-reversion factor that measures how much recent performance has deviated from the historical average. Short-term reversal is a common strategy that takes advantage of the weakly predictive pattern that stock price increases are likely to mean-revert back down over horizons from less than a minute to one month.

### Combining factors from diverse data sources

The Quantopian research environment is tailored to the rapid testing of predictive alpha factors. The process is very similar because it builds on `zipline`, but offers much richer access to data sources. 

The notebook [multiple_factors_quantopian_research](02_multiple_factors_quantopian_research.ipynb) illustrates how to compute alpha factors not only from market data as previously but also from fundamental and alternative data.

## Separating signal and noise – how to use alphalens

Quantopian has open sourced the Python library [alphalens](https://github.com/quantopian/alphalens) for the performance analysis of predictive stock factors that integrates well with the backtesting library `zipline` and the portfolio performance and risk analysis library `pyfolio` that we will explore in the next chapter.

`alphalens` facilitates the analysis of the predictive power of alpha factors concerning the:
- Correlation of the signals with subsequent returns
- Profitability of an equal or factor-weighted portfolio based on a (subset of) the signals
- Turnover of factors to indicate the potential trading costs
- Factor-performance during specific events
- Breakdowns of the preceding by sector

The analysis can be conducted using `tearsheets` or individual computations and plots. The tearsheets are illustrated in the online repo to save some space.





 