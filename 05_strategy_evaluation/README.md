# Portfolio Optimization and Performance Evaluation

To test a strategy prior to implementation under market conditions, we need to simulate the trades that the algorithm would make and verify their performance. Strategy evaluation includes backtesting against historical data to optimize the strategy's parameters and forward-testing to validate the in-sample performance against new, out-of-sample data. The goal is to avoid false discoveries from tailoring a strategy to specific past circumstances.

In a portfolio context, positive asset returns can offset negative price movements. Positive price changes for one asset are more likely to offset losses on another the lower the correlation between the two positions. Based on how portfolio risk depends on the positions’ covariance, Harry Markowitz developed the theory behind modern portfolio management based on diversification in 1952. The result is mean-variance optimization that selects weights for a given set of assets to minimize risk, measured as the standard deviation of returns for a given expected return.

The capital asset pricing model (CAPM) introduces a risk premium, measured as the expected return in excess of a risk-free investment, as an equilibrium reward for holding an asset. This reward compensates for the exposure to a single risk factor—the market—that is systematic as opposed to idiosyncratic to the asset and thus cannot be diversified away. 

Risk management has evolved to become more sophisticated as additional risk factors and more granular choices for exposure have emerged. The Kelly criterion is a popular approach to dynamic portfolio optimization, which is the choice of a sequence of positions over time; it has been famously adapted from its original application in gambling to the stock market by Edward Thorp in 1968.

As a result, there are several approaches to optimize portfolios that include the application of machine learning (ML) to learn hierarchical relationships among assets and treat their holdings as complements or substitutes with respect to the portfolio risk profile. This chapter will cover the following topics:

## Content

1. [How to measure portfolio performance](#how-to-measure-portfolio-performance)
    * [The (adjusted) Sharpe Ratio](#the-adjusted-sharpe-ratio)
    * [The fundamental law of active management](#the-fundamental-law-of-active-management)
2. [How to manage Portfolio Risk & Return](#how-to-manage-portfolio-risk--return)
    * [The evolution of modern portfolio management](#the-evolution-of-modern-portfolio-management)
    * [Mean-variance optimization](#mean-variance-optimization)
        - [Code Examples: Finding the efficient frontier in Python](#code-examples-finding-the-efficient-frontier-in-python)
    * [Alternatives to mean-variance optimization](#alternatives-to-mean-variance-optimization)
        - [The 1/N portfolio](#the-1n-portfolio)
        - [The minimum-variance portfolio](#the-minimum-variance-portfolio)
        - [The Black-Litterman approach](#the-black-litterman-approach)
        - [How to size your bets – the Kelly rule](#how-to-size-your-bets--the-kelly-rule)
        - [Alternatives to MV Optimizatino with Python](#alternatives-to-mv-optimizatino-with-python)
    * [Hierarchical Risk Parity](#hierarchical-risk-parity)
3. [Trading and managing a portfolio with `Zipline`](#trading-and-managing-a-portfolio-with-zipline)
    * [Code Examples: Backtests with trades and portfolio optimization ](#code-examples-backtests-with-trades-and-portfolio-optimization-)
4. [Measure backtest performance with `pyfolio`](#measure-backtest-performance-with-pyfolio)
    * [Code Example: `pyfolio` evaluation from a `Zipline` backtest](#code-example-pyfolio-evaluation-from-a-zipline-backtest)

## How to measure portfolio performance

To evaluate and compare different strategies or to improve an existing strategy, we need metrics that reflect their performance with respect to our objectives. In investment and trading, the most common objectives are the **return and the risk of the investment portfolio**.

The return and risk objectives imply a trade-off: taking more risk may yield higher returns in some circumstances, but also implies greater downside. To compare how different strategies navigate this trade-off, ratios that compute a measure of return per unit of risk are very popular. We’ll discuss the **Sharpe ratio** and the **information ratio** (IR) in turn.

### The (adjusted) Sharpe Ratio

The ex-ante Sharpe Ratio (SR) compares the portfolio's expected excess portfolio to the volatility of this excess return, measured by its standard deviation. It measures the compensation as the average excess return per unit of risk taken. It can be estimated from data.

Financial returns often violate the iid assumptions. Andrew Lo has derived the necessary adjustments to the distribution and the time aggregation for returns that are stationary but autocorrelated. This is important because the time-series properties of investment strategies (for example, mean reversion, momentum, and other forms of serial correlation) can have a non-trivial impact on the SR estimator itself, especially when annualizing the SR from higher-frequency data.

- [The Statistics of Sharpe Ratios](https://www.jstor.org/stable/4480405?seq=1#page_scan_tab_contents), Andrew Lo, Financial Analysts Journal, 2002

### The fundamental law of active management

It’s a curious fact that Renaissance Technologies (RenTec), the top-performing quant fund founded by Jim Simons that we mentioned in [Chapter 1](../01_machine_learning_for_trading), has produced similar returns as Warren Buffet despite extremely different approaches. Warren Buffet’s investment firm Berkshire Hathaway holds some 100-150 stocks for fairly long periods, whereas RenTec may execute 100,000 trades per day. How can we compare these distinct strategies?

ML is about optimizing objective functions. In algorithmic trading, the objectives are the return and the risk of the overall investment portfolio, typically relative to a benchmark (which may be cash, the risk-free interest rate, or an asset price index like the S&P 500).

A high Information Ratio (IR) implies attractive out-performance relative to the additional risk taken. The Fundamental Law of Active Management breaks the IR down into the information coefficient (IC) as a measure of forecasting skill, and the ability to apply this skill through independent bets. It summarizes the importance to play both often (high breadth) and to play well (high IC).

The IC measures the correlation between an alpha factor and the forward returns resulting from its signals and captures the accuracy of a manager's forecasting skills. The breadth of the strategy is measured by the independent number of bets an investor makes in a given time period, and the product of both values is proportional to the IR, also known as appraisal risk (Treynor and Black).

The fundamental law is important because it highlights the key drivers of outperformance: both accurate predictions and the ability to make independent forecasts and act on these forecasts matter. In practice, estimating the breadth of a strategy is difficult given the cross-sectional and time-series correlation among forecasts. 

- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [How to Use Security Analysis to Improve Portfolio Selection](https://econpapers.repec.org/article/ucpjnlbus/v_3a46_3ay_3a1973_3ai_3a1_3ap_3a66-86.htm), Jack L Treynor and Fischer Black, Journal of Business, 1973
- [Portfolio Constraints and the Fundamental Law of Active Management](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf), Clarke et al 2002

## How to manage Portfolio Risk & Return

Portfolio management aims to pick and size positions in financial instruments that achieve the desired risk-return trade-off regarding a benchmark. As a portfolio manager, in each period, you select positions that optimize diversification to reduce risks while achieving a target return. Across periods, these positions may require rebalancing to account for changes in weights resulting from price movements to achieve or maintain a target risk profile.

### The evolution of modern portfolio management

Diversification permits us to reduce risks for a given expected return by exploiting how imperfect correlation allows for one asset's gains to make up for another asset's losses. Harry Markowitz invented modern portfolio theory (MPT) in 1952 and provided the mathematical tools to optimize diversification by choosing appropriate portfolio weights.
 
### Mean-variance optimization

Modern portfolio theory solves for the optimal portfolio weights to minimize volatility for a given expected return, or maximize returns for a given level of volatility. The key requisite inputs are expected asset returns, standard deviations, and the covariance matrix. 
- [Portfolio Selection](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf), Harry Markowitz, The Journal of Finance, 1952
- [The Capital Asset Pricing Model: Theory and Evidence](http://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf), Eugene F. Fama and Kenneth R. French, Journal of Economic Perspectives, 2004

#### Code Examples: Finding the efficient frontier in Python

We can calculate an efficient frontier using scipy.optimize.minimize and the historical estimates for asset returns, standard deviations, and the covariance matrix. 
- The notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb) to compute the efficient frontier in python.

### Alternatives to mean-variance optimization

The challenges with accurate inputs for the mean-variance optimization problem have led to the adoption of several practical alternatives that constrain the mean, the variance, or both, or omit return estimates that are more challenging, such as the risk parity approach that we discuss later in this section.

#### The 1/N portfolio

Simple portfolios providae useful benchmarks to gauge the added value of complex models that generate the risk of overfitting. The simplest strategy—an equally-weighted portfolio—has been shown to be one of the best performers.

#### The minimum-variance portfolio

Another alternative is the global minimum-variance (GMV) portfolio, which prioritizes the minimization of risk. It is shown in the efficient frontier figure and can be calculated as follows by minimizing the portfolio standard deviation using the mean-variance framework.

#### The Black-Litterman approach

The Global Portfolio Optimization approach of Black and Litterman (1992) combines economic models with statistical learning and is popular because it generates estimates of expected returns that are plausible in many situations.
The technique assumes that the market is a mean-variance portfolio as implied by the CAPM equilibrium model. It builds on the fact that the observed market capitalization can be considered as optimal weights assigned to each security by the market. Market weights reflect market prices that, in turn, embody the market’s expectations of future returns.

- [Global Portfolio Optimization](http://www.sef.hku.hk/tpg/econ6017/2011/black-litterman-1992.pdf), Black, Fischer; Litterman, Robert
Financial Analysts Journal, 1992

#### How to size your bets – the Kelly rule

The Kelly rule has a long history in gambling because it provides guidance on how much to stake on each of an (infinite) sequence of bets with varying (but favorable) odds to maximize terminal wealth. It was published as A New Interpretation of the Information Rate in 1956 by John Kelly who was a colleague of Claude Shannon's at Bell Labs. He was intrigued by bets placed on candidates at the new quiz show The $64,000 Question, where a viewer on the west coast used the three-hour delay to obtain insider information about the winners. 

Kelly drew a connection to Shannon's information theory to solve for the bet that is optimal for long-term capital growth when the odds are favorable, but uncertainty remains. His rule maximizes logarithmic wealth as a function of the odds of success of each game, and includes implicit bankruptcy protection since log(0) is negative infinity so that a Kelly gambler would naturally avoid losing everything.

- [A New Interpretation of Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf), John Kelly, 1956
- [Beat the Dealer: A Winning Strategy for the Game of Twenty-One](https://www.amazon.com/Beat-Dealer-Winning-Strategy-Twenty-One/dp/0394703103), Edward O. Thorp,1966
- [Beat the Market: A Scientific Stock Market System](https://www.researchgate.net/publication/275756748_Beat_the_Market_A_Scientific_Stock_Market_System) , Edward O. Thorp,1967
- [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889/ref=sr_1_2?s=books&ie=UTF8&qid=1545525861&sr=1-2), Ernie Chan, 2008

#### Alternatives to MV Optimizatino with Python

- The notebook [kelly_rule](05_kelly_rule.ipynb) demonstrates the application for the single and multiple asset case. 
- The latter result is also included in the notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb), along with several other alternative approaches.

### Hierarchical Risk Parity

This novel approach developed by [Marcos Lopez de Prado](http://www.quantresearch.org/) aims to address three major concerns of quadratic optimizers, in general, and Markowitz’s critical line algorithm (CLA), in particular: 
- instability, 
- concentration, and 
- underperformance. 

Hierarchical Risk Parity (HRP) applies graph theory and machine-learning to build a diversified portfolio based on the information contained in the covariance matrix. However, unlike quadratic optimizers, HRP does not require the invertibility of the covariance matrix. In fact, HRP can compute a portfolio on an ill-degenerated or even a singular covariance matrix—an impossible feat for quadratic optimizers. Monte Carlo experiments show that HRP delivers lower out-ofsample variance than CLA, even though minimum variance is CLA’s optimization objective. HRP also produces less risky portfolios out of sample compared to traditional risk parity methods. We will discuss HRP in more detail in [Chapter 13](../13_unsupervised_learning) when we discuss applications of unsupervised learning, including hiearchical clustering, to trading.

- [Building diversified portfolios that outperform out of sample](https://jpm.pm-research.com/content/42/4/59.short), Marcos López de Prado, The Journal of Portfolio Management 42, no. 4 (2016): 59-69.
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Thomas Raffinot, 2016

We demonstrate how to implement HRP and compare it to alternatives in Chapter 13 on [Unsupervised Learning](../13_unsupervised_learning) where we also introduce hierarchical clustering.

## Trading and managing a portfolio with `Zipline`

The open source [zipline](http://www.zipline.io/index.html) library is an event-driven backtesting system maintained and used in production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development and live-trading. It automates the algorithm's reaction to trade events and provides it with current and historical point-in-time data that avoids look-ahead bias. [Chapter 8 - The ML4T Workflow](../08_strategy_workflow) has a more detailed, dedicated introduction to backtesting using both `zipline` and `backtrader`. 

In [Chapter 4](../04_alpha_factor_research), we introduced `zipline` to simulate the computation of alpha factors from trailing cross-sectional market, fundamental, and alternative data. Now we will exploit the alpha factors to derive and act on buy and sell signals. 

### Code Examples: Backtests with trades and portfolio optimization 

The code for this section lives in the following two notebooks: 
- The notebooks in this section rely on the `conda` environment `ml4t-zipline`. For installation, please see instructions provided [here](../installation).
- The notebook [backtest_with_trades](01_backtest_with_trades.ipynb) simulates the trading decisions that build a portfolio based on the simple MeanReversion alpha factor from the last chapter using Zipline. We not explicitly optimize the portfolio weights and just assign positions of equal value to each holding.
- The notebook [backtest_with_pf_optimization](02_backtest_with_pf_optimization.ipynb) demonstrates how to use PF optimization as part of a simple strategy backtest. 

## Measure backtest performance with `pyfolio`

Pyfolio facilitates the analysis of portfolio performance and risk in-sample and out-of-sample using many standard metrics. It produces tear sheets covering the analysis of returns, positions, and transactions, as well as event risk during periods of market stress using several built-in scenarios, and also includes Bayesian out-of-sample performance analysis.

### Code Example: `pyfolio` evaluation from a `Zipline` backtest

The notebook [pyfolio_demo](03_pyfolio_demo.ipynb) illustrates how to extract the `pyfolio` input from the backtest conducted in the previous folder. It then proceeds to calcuate several performance metrics and tear sheets using `pyfolio`

- The notebook relies on the `conda` environment `ml4t-zipline`. For installation, please see instructions provided [here](../installation).
