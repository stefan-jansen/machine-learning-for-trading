# Chapter 01: From Idea to Execution

## The rise of ML in the investment industry

The investment industry has evolved dramatically over the last several decades and continues to do so amid increased competition, technological advances, and a challenging economic environment. This section reviews several key trends that have shaped the investment environment in general, and the context for algorithmic trading more specifically, and related themes that will recur throughout this book.

The trends that have propelled algorithmic trading and ML to current prominence include:
- Changes in the market microstructure, such as the spread of electronic trading and the integration of markets across asset classes and geographies
- The development of investment strategies framed in terms of risk-factor exposure, as opposed to asset classes
- The revolutions in computing power, data-generation and management, and analytic methods
- The outperformance of the pioneers in algorithmic traders relative to human, discretionary investors

In addition, the financial crises of 2001 and 2008 have affected how investors approach diversification and risk management and have given rise to low-cost passive investment vehicles in the form of exchange-traded funds (ETFs). Amid low yield and low volatility after the 2008 crisis, cost-conscious investors shifted $2 trillion from actively-managed mutual funds into passively managed ETFs. Competitive pressure is also reflected in lower hedge fund fees that dropped from the traditional 2% annual management fee and 20% take of profits to an average of 1.48% and 17.4%, respectively, in 2017.

### From electronic to high-frequency trading

Electronic trading has advanced dramatically in terms of capabilities, volume, coverage of asset classes, and geographies since networks started routing prices to computer terminals in the 1960s.

- [High Frequency Trading: Overview of Recent Developments](https://fas.org/sgp/crs/misc/R44443.pdf), Congressional Research Service, 2016
### Factor investing and smart beta funds

The return provided by an asset is a function of the uncertainty or risk associated with the financial investment. An equity investment implies, for example, assuming a company's business risk, and a bond investment implies assuming default risk.

To the extent that specific risk characteristics predict returns, identifying and forecasting the behavior of these risk factors becomes a primary focus when designing an investment strategy. It yields valuable trading signals and is the key to superior active-management results. The industry's understanding of risk factors has evolved very substantially over time and has impacted how ML is used for algorithmic trading.

The factors that explained returns above and beyond the CAPM were incorporated into investment styles that tilt portfolios in favor of one or more factors, and assets began to migrate into factor-based portfolios. The 2008 financial crisis underlined how asset-class labels could be highly misleading and create a false sense of diversification when investors do not look at the underlying factor risks, as asset classes came crashing down together.

Over the past several decades, quantitative factor investing has evolved from a simple approach based on two or three styles to multifactor smart or exotic beta products. Smart beta funds have crossed $1 trillion AUM in 2017, testifying to the popularity of the hybrid investment strategy that combines active and passive management. Smart beta funds take a passive strategy but modify it according to one or more factors, such as cheaper stocks or screening them according to dividend payouts, to generate better returns. This growth has coincided with increasing criticism of the high fees charged by traditional active managers as well as heightened scrutiny of their performance.

The ongoing discovery and successful forecasting of risk factors that, either individually or in combination with other risk factors, significantly impact future asset returns across asset classes is a key driver of the surge in ML in the investment industry and will be a key theme throughout this book.

### Algorithmic pioneers outperform humans at scale

The track record and growth of Assets Under Management (AUM) of firms that spearheaded algorithmic trading has played a key role in generating investor interest and subsequent industry efforts to replicate their success.

Systematic strategies that mostly or exclusively rely on algorithmic decision-making were most famously introduced by mathematician James Simons who founded Renaissance Technologies in 1982 and built it into the premier quant firm. Its secretive Medallion Fund, which is closed to outsiders, has earned an estimated annualized return of 35% since 1982.

DE Shaw, Citadel, and Two Sigma, three of the most prominent quantitative hedge funds that use systematic strategies based on algorithms, rose to the all-time top-20 performers for the first time in 2017 in terms of total dollars earned for investors, after fees, and since inception.

#### ML driven funds attract $1 trillion AUM

Morgan Stanley estimated in 2017 that algorithmic strategies have grown at 15% per year over the past six years and control about $1.5 trillion between hedge funds, mutual funds, and smart beta ETFs. Other reports suggest the quantitative hedge fund industry was about to exceed $1 trillion AUM, nearly doubling its size since 2010 amid outflows from traditional hedge funds. In contrast, total hedge fund industry capital hit $3.21 trillion according to the latest global Hedge Fund Research report.

#### The emergence of quantamental funds

Two distinct approaches have evolved in active investment management: systematic (or quant) and discretionary investing. Systematic approaches rely on algorithms for a repeatable and data-driven approach to identify investment opportunities across many securities; in contrast, a discretionary approach involves an in-depth analysis of a smaller number of securities. These two approaches are becoming more similar as fundamental managers take more data-science-driven approaches.

Even fundamental traders now arm themselves with quantitative techniques, accounting for $55 billion of systematic assets, according to Barclays. Agnostic to specific companies, quantitative funds trade patterns and dynamics across a wide swath of securities. Quants now account for about 17% of total hedge fund assets, data compiled by Barclays shows.


### ML and alternative data

Hedge funds have long looked for alpha through informational advantage and the ability to uncover new uncorrelated signals. Historically, this included things such as proprietary surveys of shoppers, or voters ahead of elections or referendums. Occasionally, the use of company insiders, doctors, and expert networks to expand knowledge of industry trends or companies crosses legal lines: a series of prosecutions of traders, portfolio managers, and analysts for using insider information after 2010 has shaken the industry.

In contrast, the informational advantage from exploiting conventional and alternative data sources using ML is not related to expert and industry networks or access to corporate management, but rather the ability to collect large quantities of data and analyze them in real-time.

Three trends have revolutionized the use of data in algorithmic trading strategies and may further shift the investment industry from discretionary to quantitative styles:
- The exponential increase in the amount of digital data 
- The increase in computing power and data storage capacity at lower cost
- The advances in ML methods for analyzing complex datasets

- [Can We Predict the Financial Markets Based on Google's Search Queries?](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2446), Perlin, et al, 2016, Journal of Forecasting

## Design and execution of a trading strategy

ML can add value at multiple steps in the lifecycle of a trading strategy, and relies on key infrastructure and data resources. Hence, this book aims to addresses how ML techniques fit into the broader process of designing, executing, and evaluating strategies.

An algorithmic trading strategy is driven by a combination of alpha factors that transform one or several data sources into signals that in turn predict future asset returns and trigger buy or sell orders. Chapter 2, Market and Fundamental Data and Chapter 3, Alternative Data for Finance cover the sourcing and management of data, the raw material and the single most important driver of a successful trading strategy.  

[Chapter 4, Alpha Factor Research](../04_alpha_factor_research) outlines a methodologically sound process to manage the risk of false discoveries that increases with the amount of data. [Chapter 5, Strategy Evaluation](../05_strategy_evaluation) provides the context for the execution and performance measurement of a trading strategy.

The following subsections outline these steps, which we will discuss in depth throughout the book.

### Sourcing and managing data

The dramatic evolution of data in terms of volume, variety, and velocity is both a necessary condition for and driving force of the application of ML to algorithmic trading. The proliferating supply of data requires active management to uncover potential value, including the following steps:
- Identify and evaluate market, fundamental, and alternative data sources containing alpha signals that do not decay too quickly.
- Deploy or access cloud-based scalable data infrastructure and analytical tools like Hadoop or Spark Sourcing to facilitate fast, flexible data access

Carefully manage and curate data to avoid look-ahead bias by adjusting it to the desired frequency on a point-in-time (PIT) basis. This means that data may only reflect information available and know at the given time. ML algorithms trained on distorted historical data will almost certainly fail during live trading.

### Alpha factor research and evaluation

Alpha factors are designed to extract signals from data to predict asset returns for a given investment universe over the trading horizon. A factor takes on a single value for each asset when evaluated, but may combine one or several input variables. The process involves the steps outlined in the following figure:

The Research phase of the trading strategy workflow includes the design, evaluation, and combination of alpha factors. ML plays a large role in this process because the complexity of factors has increased as investors react to both the signal decay of simpler factors and the much richer data available today.

### Portfolio optimization and risk management

Alpha factors emit entry and exit signals that lead to buy or sell orders, and order execution results in portfolio holdings. The risk profiles of individual positions interact to create a specific portfolio risk profile. Portfolio management involves the optimization of position weights to achieve the desired portfolio risk and return a profile that aligns with the overall investment objectives. This process is highly dynamic to incorporate continuously-evolving market data.

### Strategy backtesting

The incorporation of an investment idea into an algorithmic strategy requires extensive testing with a scientific approach that attempts to reject the idea based on its performance in alternative out-of-sample market scenarios. Testing may involve simulated data to capture scenarios deemed possible but not reflected in historic data.

## ML and algorithmic trading strategies

Quantitative strategies have evolved and become more sophisticated in three waves: 
- In the 1980s and 1990s, signals often emerged from academic research and used a single or very few inputs derived from market and fundamental data. These signals are now largely commoditized and available as ETF, such as basic mean-reversion strategies.
- In the 2000s, factor-based investing proliferated. Funds used algorithms to identify assets exposed to risk factors like value or momentum to seek arbitrage opportunities. Redemptions during the early days of the financial crisis triggered the quant quake of August 2007 that cascaded through the factor-based fund industry. These strategies are now also available as long-only smart-beta funds that tilt portfolios according to a given set of risk factors.
- The third era is driven by investments in ML capabilities and alternative data to generate profitable signals for repeatable trading strategies. Factor decay is a major challenge: the excess returns from new anomalies have been shown to drop by a quarter from discovery to publication, and by over 50% after publication due to competition and crowding.

There are several categories of trading strategies that use algorithms to execute trading rules:
- Short-term trades that aim to profit from small price movements, for example, due to arbitrage
- Behavioral strategies that aim to capitalize on anticipating the behavior of other market participants
- Programs that aim to optimize trade execution, and
- A large group of trading based on predicted pricing

### Use Cases of ML for Trading 

ML extracts signals from a wide range of market, fundamental, and alternative data, and can be applied at all steps of the algorithmic trading-strategy process. Key applications include:
- Data mining to identify patterns and extract features
- Supervised learning to generate risk factors or alphas and create trade ideas
- Aggregation of individual signals into a strategy
- Allocation of assets according to risk profiles learned by an algorithm
- The testing and evaluation of strategies, including through the use of synthetic data
- The interactive, automated refinement of a strategy using reinforcement learning

## References

- [The fundamental law of active management](http://jpm.iijournals.com/content/15/3/30), Richard C. Grinold, The Journal of Portfolio Management Spring 1989, 15 (3) 30-37
- [The relationship between return and market value of common stocks](https://www.sciencedirect.com/science/article/pii/0304405X81900180), Rolf Banz,Journal of Financial Economics, March 1981
- [The Arbitrage Pricing Theory: Some Empirical Results](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1981.tb00444.x), Marc Reinganum, Journal of Finance, 1981
- [The Relationship between Earnings' Yield, Market Value and Return for NYSE Common Stock](https://pdfs.semanticscholar.org/26ab/311756099c8f8c4e528083c9b90ff154f98e.pdf), Sanjoy Basu, Journal of Financial Economics, 1982

### News

- [The Rise of the Artificially Intelligent Hedge Fund](https://www.wired.com/2016/01/the-rise-of-the-artificially-intelligent-hedge-fund/#comments), Wired, 25-01-2016
- [Crowd-Sourced Quant Network Allocates Most Ever to Single Algo](https://www.bloomberg.com/news/articles/2018-08-02/crowd-sourced-quant-network-allocates-most-ever-to-single-algo), Bloomberg, 08-02-2018
- [Goldman Sachs’ lessons from the ‘quant quake’](https://www.ft.com/content/fdfd5e78-0283-11e7-aa5b-6bb07f5c8e12), Financial Times, 03-08-2017
- [Lessons from the Quant Quake resonate a decade later](https://www.ft.com/content/a7a04d4c-83ed-11e7-94e2-c5b903247afd), Financial Times, 08-18-2017
- [Smart beta funds pass $1tn in assets](https://www.ft.com/content/bb0d1830-e56b-11e7-8b99-0191e45377ec), Financial Times, 12-27-2017
- [BlackRock bets on algorithms to beat the fund managers](https://www.ft.com/content/e689a67e-2911-11e8-b27e-cc62a39d57a0), Financial Times, 03-20-2018
- [Smart beta: what’s in a name?](https://www.ft.com/content/d1bdabaa-a9f0-11e7-ab66-21cc87a2edde), Financial Times, 11-27-2017
- [Computer-driven hedge funds join industry top performers](https://www.ft.com/content/9981c870-e79a-11e6-967b-c88452263daf), Financial Times, 02-01-2017
- [Quants Rule Alpha’s Hedge Fund 100 List](https://www.institutionalinvestor.com/article/b1505pmf2v2hg3/quants-rule-alphas-hedge-fund-100-list), Institutional Investor, 06-26-2017
- [The Quants Run Wall Street Now](https://www.wsj.com/articles/the-quants-run-wall-street-now-1495389108), Wall Street Journal, 05-21-2017
- ['We Don’t Hire MBAs': The New Hedge Fund Winners Will Crunch The Better Data Sets](https://www.cbinsights.com/research/algorithmic-hedge-fund-trading-winners/), cbinsights, 06-28-2018
- [Artificial Intelligence: Fusing Technology and Human Judgment?](https://blogs.cfainstitute.org/investor/2017/09/25/artificial-intelligence-fusing-technology-and-human-judgment/), CFA Institute, 09-25-2017
- [The Hot New Hedge Fund Flavor Is 'Quantamental'](https://www.bloomberg.com/news/articles/2017-08-25/the-hot-new-hedge-fund-flavor-is-quantamental-quicktake-q-a), Bloomberg, 08-25-2017
- [Robots Are Eating Money Managers’ Lunch](https://www.bloomberg.com/news/articles/2017-06-20/robots-are-eating-money-managers-lunch), Bloomberg, 06-20-2017
- [Rise of Robots: Inside the World's Fastest Growing Hedge Funds](https://www.bloomberg.com/news/articles/2017-06-20/rise-of-robots-inside-the-world-s-fastest-growing-hedge-funds), Bloomberg, 06-20-2017
- [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
- [BlackRock bulks up research into artificial intelligence](https://www.ft.com/content/4f5720ce-1552-11e8-9376-4a6390addb44), Financial Times, 02-19-2018
- [AQR to explore use of ‘big data’ despite past doubts](https://www.ft.com/content/3a8f69f2-df34-11e7-a8a4-0a1e63a52f9c), Financial Times, 12-12-2017
- [Two Sigma rapidly rises to top of quant hedge fund world](https://www.ft.com/content/dcf8077c-b823-11e7-9bfb-4a9c83ffa852), Financial Times, 10-24-2017
- [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
- [Artificial intelligence (AI) in finance - six warnings from a central banker](https://www.bundesbank.de/en/press/speeches/artificial-intelligence--ai--in-finance--six-warnings-from-a-central-banker-711602), Deutsche Bundesbank, 02-27-2018
- [Fintech: Search for a super-algo](https://www.ft.com/content/5eb91614-bee5-11e5-846f-79b0e3d20eaf), Financial Times, 01-20-2016
- [Barron’s Top 100 Hedge Funds](https://www.barrons.com/articles/top-100-hedge-funds-1524873705)
- [How high-frequency trading hit a speed bump](https://www.ft.com/content/d81f96ea-d43c-11e7-a303-9060cb1e5f44), FT, 01-01-2018

### Books

- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086), Marcos Lopez de Prado, 2018
- [Quantresearch](http://www.quantresearch.info/index.html) by Marcos López de Prado
- [Quantitative Trading](http://epchan.blogspot.com/), Ernest Chan
#### Machine Learning

- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Tom Mitchell, McGraw Hill, 1997
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), Gareth James et al.
    - Excellent reference for essential machine learning concepts, available free online
- [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf), Barber, D., Cambridge University Press, 2012 (updated version available on author's website)

### Courses

- [Algorithmic Trading](http://personal.stevens.edu/~syang14/fe670.htm), Prof. Steve Yang, Stevens Institute of Technology
- [Machine Learning](https://www.coursera.org/learn/machine-learning), Andrew Ng, Coursera
- [](http://deeplearning.ai/), Andrew Ng
    - Andrew Ng’s introductory deep learning course

### Python Libraries

- matplotlib [docs]( <https://github.com/matplotlib/matplotlib)
-  numpy [docs](https://github.com/numpy/numpy)
-  pandas [docs](https://github.com/pydata/pandas)
-  scipy [docs](https://github.com/scipy/scipy)
-  seaborn [docs](https://github.com/mwaskom/seaborn)
-  statsmodels [docs](https://github.com/statsmodels/statsmodels)
- [Boosting numpy: Why BLAS Matters](http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)

