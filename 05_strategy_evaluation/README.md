# Chapter 05: Strategy Evaluation & Portfolio Management

This chapter covers:

- How to build and test a portfolio based on alpha factors using zipline
- How to measure portfolio risk and return
- How to evaluate portfolio performance using pyfolio
- How to manage portfolio weights using mean-variance optimization and alternatives
- How to use machine learning to optimize asset allocation in a portfolio context

## How to build and test a portfolio with `zipline`

### Code Examples

The directory Trading with `zipline` contains a python file with the code required to simulate the trading decisions that build a portfolio based on the simple alpha factor from the last chapter using zipline.

## How to measure performance with `pyfolio`

### The Sharpe Ratio

- [The Statistics of Sharpe Ratios](https://www.jstor.org/stable/4480405?seq=1#page_scan_tab_contents), Andrew Lo, Financial Analysts Journal, 2002

### The Fundamental Law of Active Management

- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [How to Use Security Analysis to Improve Portfolio Selection](https://econpapers.repec.org/article/ucpjnlbus/v_3a46_3ay_3a1973_3ai_3a1_3ap_3a66-86.htm), Jack L Treynor and Fischer Black, Journal of Business, 1973
- [Portfolio Constraints and the Fundamental Law of Active Management](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf), Clarke et al 2002

### In- and out-of-sample performance with `pyfolio`

#### Code Examples

The directory Risk Metris with `pyfolio` contains a jupyter notebook that illustrates how to extract the `pyfolio` input from the backtest conducted in the previous folder. It then proceeds to calcuate several performance metrics and tear sheets using `pyfolio`

## How to avoid the pitfalls of backtesting

### Data Challenges

### Implementation Issues

### Data-snooping and backtest overfitting

- [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), Bailey, David and Lopez de Prado, Marcos, Journal of Portfolio Management, 2013
- [Backtest Overfitting: An Interactive Example](http://datagrid.lbl.gov/backtest/)
- [Backtesting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2606462), Lopez de Prado, Marcos, 2015
- [Secretary Problem (Optimal Stopping)](https://www.geeksforgeeks.org/secretary-problem-optimal-stopping-problem/)
- [Optimal Stopping and Applications](https://www.math.ucla.edu/~tom/Stopping/Contents.html), Ferguson, Math Department, UCLA
- [Advances in Machine Learning Lectures 4/10 - Backtesting I](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257420), Marcos Lopez de Prado, 2018
- [Advances in Machine Learning Lectures 5/10 - Backtesting II](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257497), Marcos Lopez de Prado, 2018
- [Backtest overfitting simulator](http://datagrid.lbl.gov/backtest/)
- [quantresearch](http://www.quantresearch.info/) by Marcos Lopez de Prado

#### Code Examples

The directory (Multiple) Backtesting contains the implementation of the Deflated Sharpe Ratio by [Marcos Lopez de Prado](http://www.quantresearch.info/Software.htm).

## How to Manage Portfolio Risk & Return

- [Portfolio Selection](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf), Harry Markowitz, The Journal of Finance, 1952
- [The Capital Asset Pricing Model: Theory and Evidence](http://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf), Eugene F. Fama and Kenneth R. French, Journal of Economic Perspectives, 2004

### Mean-variance optimization

#### Code Examples

The directory Efficient Frontier contains the notebook to compute the efficient frontier in python.

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

The directory Kelly Rule contains the notebooks to compute the Kelly rule portfolio.

#### Hierarchical Risk Parity

- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Thomas Raffinot, 2016