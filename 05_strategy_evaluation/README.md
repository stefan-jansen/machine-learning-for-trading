# Portfolio Optimization and Performance Evaluation

Alpha factors generate signals that an algorithmic strategy translates into trades, which, in turn, produce long and short positions. The returns and risk of the resulting portfolio determine the success of the strategy.

To test a strategy prior to implementation under market conditions, we need to simulate the trades that the algorithm would make and verify their performance. Strategy evaluation includes backtesting against historical data to optimize the strategy's parameters and forward-testing to validate the in-sample performance against new, out-of-sample data. The goal is to avoid false discoveries from tailoring a strategy to specific past circumstances.

In a portfolio context, positive asset returns can offset negative price movements. Positive price changes for one asset are more likely to offset losses on another the lower the correlation between the two positions. Based on how portfolio risk depends on the positions’ covariance, Harry Markowitz developed the theory behind modern portfolio management based on diversification in 1952. The result is mean-variance optimization that selects weights for a given set of assets to minimize risk, measured as the standard deviation of returns for a given expected return.

The capital asset pricing model (CAPM) introduces a risk premium, measured as the expected return in excess of a risk-free investment, as an equilibrium reward for holding an asset. This reward compensates for the exposure to a single risk factor—the market—that is systematic as opposed to idiosyncratic to the asset and thus cannot be diversified away. 

Risk management has evolved to become more sophisticated as additional risk factors and more granular choices for exposure have emerged. The Kelly criterion is a popular approach to dynamic portfolio optimization, which is the choice of a sequence of positions over time; it has been famously adapted from its original application in gambling to the stock market by Edward Thorp in 1968.

As a result, there are several approaches to optimize portfolios that include the application of machine learning (ML) to learn hierarchical relationships among assets and treat their holdings as complements or substitutes with respect to the portfolio risk profile.

In this chapter, we will cover the following topics:
- How to measure portfolio risk and return
- Managing portfolio weights using mean-variance optimization and alternatives
- Using machine learning to optimize asset allocation in a portfolio context
- Simulating trades and create a portfolio based on alpha factors using Zipline
- How to evaluate portfolio performance using pyfolio

## How to build and test a portfolio with `zipline`

The open source [zipline](http://www.zipline.io/index.html) library is an event-driven backtesting system maintained and used in production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development and live-trading. It automates the algorithm's reaction to trade events and provides it with current and historical point-in-time data that avoids look-ahead bias. [Chapter 8 - The ML4T Workflow](../08_strategy_workflow) has a more detailed, dedicated introduction to backtesting using both `zipline` and `backtrader`. 

In [Chapter 4](../04_alpha_factor_research), we introduced `zipline` to simulate the computation of alpha factors from trailing cross-sectional market, fundamental, and alternative data. Now we will exploit the alpha factors to derive and act on buy and sell signals. 

We will postpone optimizing the portfolio weights until later in this chapter, and for now, just assign positions of equal value to each holding. 

The code for this section is in the subdirectory [trading_zipline](01_trade_simulation_pf_management_zipline): 
- The notebook [backtest_with_trades](01_trade_simulation_pf_management_zipline/01_backtest_with_trades.ipynb) simulates the trading decisions that build a portfolio based on the simple MeanReversion alpha factor from the last chapter using zipline.
- The notebook [backtest_with_pf_optimization](01_trade_simulation_pf_management_zipline/02_backtest_with_pf_optimization.ipynb) demonstrates how to use PF optimization as part of a simple strategy backtest. 

### Installation

- The current release 1.3 has a few shortcomings such as the [dependency on benchmark data from the IEX exchange](https://github.com/quantopian/zipline/issues/2480) and limitations for importing features beyond the basic OHLCV data points.
- To enable the use of `zipline`, I've provided a [patched version](https://github.com/stefan-jansen/zipline) that works for the purposes of this book.
    - Create a virtual environment based on Python 3.5, for instance using [pyenv](https://github.com/pyenv/pyenv)
    - After activating the virtual environment, run `pip install -U pip Cython`
    - Install the patched `zipline` version by cloning the repo, `cd` into the packages' root folder and run `pip install -e`
    - Run `pip install jupyter pyfolio`

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