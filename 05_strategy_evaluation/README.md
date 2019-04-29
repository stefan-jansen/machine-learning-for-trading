# Chapter 05: Strategy Evaluation & Portfolio Management

This chapter covers:

- How to build and test a portfolio based on alpha factors using zipline
- How to measure portfolio risk and return
- How to avoid the pitfalls of backtesting
- How to evaluate portfolio performance using pyfolio
- How to manage portfolio weights using mean-variance optimization and alternatives
- How to use machine learning to optimize asset allocation in a portfolio context

## How to build and test a portfolio with `zipline`

In [Chapter 4](../04_alpha_factor_research), we introduced `zipline` to simulate the computation of alpha factors from trailing cross-sectional market, fundamental, and alternative data. Now we will exploit the alpha factors to derive and act on buy and sell signals. 

We will postpone optimizing the portfolio weights until later in this chapter, and for now, just assign positions of equal value to each holding. 

The code for this section is in the subdirectory [trading_zipline](01_trading_zipline) subdirectory; the notebook [alpha_factor_zipline_with_trades](01_trading_zipline/alpha_factor_zipline_with_trades.ipynb) simulates the trading decisions that build a portfolio based on the simple MeanReversion alpha factor from the last chapter using zipline.

## How to measure performance with `pyfolio`

ML is about optimizing objective functions. In algorithmic trading, the objectives are the return and the risk of the overall investment portfolio, typically relative to a benchmark (which may be cash, the risk-free interest rate, or an asset price index like the S&P 500).

### The Sharpe Ratio

The ex-ante Sharpe Ratio (SR) compares the portfolio's expected excess portfolio to the volatility of this excess return, measured by its standard deviation. It measures the compensation as the average excess return per unit of risk taken. It can be estimated from data.

Financial returns often violate the iid assumptions. Andrew Lo has derived the necessary adjustments to the distribution and the time aggregation for returns that are stationary but autocorrelated. This is important because the time-series properties of investment strategies (for example, mean reversion, momentum, and other forms of serial correlation) can have a non-trivial impact on the SR estimator itself, especially when annualizing the SR from higher-frequency data.

- [The Statistics of Sharpe Ratios](https://www.jstor.org/stable/4480405?seq=1#page_scan_tab_contents), Andrew Lo, Financial Analysts Journal, 2002

### The Fundamental Law of Active Management

A high Information Ratio (IR) implies attractive out-performance relative to the additional risk taken. The Fundamental Law of Active Management breaks the IR down into the information coefficient (IC) as a measure of forecasting skill, and the ability to apply this skill through independent bets. It summarizes the importance to play both often (high breadth) and to play well (high IC).

The IC measures the correlation between an alpha factor and the forward returns resulting from its signals and captures the accuracy of a manager's forecasting skills. The breadth of the strategy is measured by the independent number of bets an investor makes in a given time period, and the product of both values is proportional to the IR, also known as appraisal risk (Treynor and Black).

The fundamental law is important because it highlights the key drivers of outperformance: both accurate predictions and the ability to make independent forecasts and act on these forecasts matter. In practice, estimating the breadth of a strategy is difficult given the cross-sectional and time-series correlation among forecasts. 

- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [How to Use Security Analysis to Improve Portfolio Selection](https://econpapers.repec.org/article/ucpjnlbus/v_3a46_3ay_3a1973_3ai_3a1_3ap_3a66-86.htm), Jack L Treynor and Fischer Black, Journal of Business, 1973
- [Portfolio Constraints and the Fundamental Law of Active Management](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf), Clarke et al 2002

### In- and out-of-sample performance with `pyfolio`

Pyfolio facilitates the analysis of portfolio performance and risk in-sample and out-of-sample using many standard metrics. It produces tear sheets covering the analysis of returns, positions, and transactions, as well as event risk during periods of market stress using several built-in scenarios, and also includes Bayesian out-of-sample performance analysis.

#### Code Examples

The directory [risk_metrics_pyfolio](02_risk_metrics_pyfolio) contains the notebook [pyfolio_demo](02_risk_metrics_pyfolio/pyfolio_demo.ipynb) that illustrates how to extract the `pyfolio` input from the backtest conducted in the previous folder. It then proceeds to calcuate several performance metrics and tear sheets using `pyfolio`

## How to avoid the pitfalls of backtesting

### Data Challenges

Backtesting simulates an algorithmic strategy using historical data with the goal of identifying patterns that generalize to new market conditions. In addition to the generic challenges of predicting an uncertain future in changing markets, numerous factors make mistaking positive in-sample performance for the discovery of true patterns very likely. 

These factors include aspects of the data, the implementation of the strategy simulation, and flaws with the statistical tests and their interpretation. The risks of false discoveries multiply with the use of more computing power, bigger datasets, and more complex algorithms that facilitate the identification of apparent patterns in the noise.

### Data-snooping and backtest overfitting

The most prominent challenge to backtest validity, including to published results, relates to the discovery of spurious patterns due to multiple testing during the strategy-selection process. Selecting a strategy after testing different candidates on the same data will likely bias the choice because a positive outcome is more likely to be due to the stochastic nature of the performance measure itself. In other words, the strategy is overly tailored, or overfit, to the data at hand and produces deceptively positive results.

[Marcos Lopez de Prado](http://www.quantresearch.info/) has published extensively on the risks of backtesting, and how to detect or avoid it. This includes an [online simulator of backtest-overfitting](http://datagrid.lbl.gov/backtest/).


#### The deflated Sharpe Ratio

De Lopez Prado and Bailey (2014) derive a deflated SR to compute the probability that the SR is statistically significant while controlling for the inflationary effect of multiple testing, non-normal returns, and shorter sample lengths.

The pyton script [deflated_sharpe_ratio](03_multiple_testing/deflated_sharpe_ratio.py) in the directory [multiple_testing](03_multiple_testing) contains the Python implementation with references for the derivation of the related formulas. 

#### References

- [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), Bailey, David and Lopez de Prado, Marcos, Journal of Portfolio Management, 2013
- [Backtest Overfitting: An Interactive Example](http://datagrid.lbl.gov/backtest/)
- [Backtesting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2606462), Lopez de Prado, Marcos, 2015
- [Secretary Problem (Optimal Stopping)](https://www.geeksforgeeks.org/secretary-problem-optimal-stopping-problem/)
- [Optimal Stopping and Applications](https://www.math.ucla.edu/~tom/Stopping/Contents.html), Ferguson, Math Department, UCLA
- [Advances in Machine Learning Lectures 4/10 - Backtesting I](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257420), Marcos Lopez de Prado, 2018
- [Advances in Machine Learning Lectures 5/10 - Backtesting II](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257497), Marcos Lopez de Prado, 2018

## How to Manage Portfolio Risk & Return

- [Portfolio Selection](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf), Harry Markowitz, The Journal of Finance, 1952
- [The Capital Asset Pricing Model: Theory and Evidence](http://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf), Eugene F. Fama and Kenneth R. French, Journal of Economic Perspectives, 2004

### Mean-variance optimization

MPT solves for the optimal portfolio weights to minimize volatility for a given expected return, or maximize returns for a given level of volatility. The key requisite input are expected asset returns, standard deviations, and the covariance matrix. 

#### Code Examples

We can calculate an efficient frontier using scipy.optimize.minimize and the historical estimates for asset returns, standard deviations, and the covariance matrix. 

The directory [efficient_frontier](04_efficient_frontier) contains the notebook [mean_variance_optimization](04_efficient_frontier/mean_variance_optimization.ipynb) to compute the efficient frontier in python.

### Alternatives to mean-variance optimization

#### The Black-Litterman approach

- [Global Portfolio Optimization](http://www.sef.hku.hk/tpg/econ6017/2011/black-litterman-1992.pdf), Black, Fischer; Litterman, Robert
Financial Analysts Journal, 1992

#### The Kelly Rule

- [A New Interpretation of Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf), John Kelly, 1956
- [Beat the Dealer: A Winning Strategy for the Game of Twenty-One](https://www.amazon.com/Beat-Dealer-Winning-Strategy-Twenty-One/dp/0394703103), Edward O. Thorp,1966
- [Beat the Market: A Scientific Stock Market System](https://www.researchgate.net/publication/275756748_Beat_the_Market_A_Scientific_Stock_Market_System) , Edward O. Thorp,1967
- [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889/ref=sr_1_2?s=books&ie=UTF8&qid=1545525861&sr=1-2), Ernie Chan, 2008

##### Code Example

The directory [kelly](05_kelly) Kelly Rule contains the notebooks [kelly_rule](05_kelly/kelly_rule.ipynb) to compute the Kelly rule portfolio.

#### Hierarchical Risk Parity

- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Thomas Raffinot, 2016