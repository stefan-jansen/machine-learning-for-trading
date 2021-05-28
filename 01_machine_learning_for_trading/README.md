# Machine Learning for Trading: From Idea to Execution

Algorithmic trading relies on computer programs that execute algorithms to automate some or all elements of a trading strategy. **Algorithms** are a sequence of steps or rules designed to achieve a goal. They can take many forms and facilitate optimization throughout the investment process, from idea generation to asset allocation, trade execution, and risk management.

**Machine learning** (ML) involves algorithms that learn rules or patterns from data to achieve a goal such as minimizing a prediction error. The examples in this book will illustrate how ML algorithms can extract information from data to support or automate key investment activities. These activities include observing the market and analyzing data to form expectations about the future and decide on placing buy or sell orders, as well as managing the resulting portfolio to produce attractive returns relative to the risk.

Ultimately, the goal of active investment management is to generate alpha, defined as portfolio returns in excess of the benchmark used for evaluation. The **fundamental law of active management** postulates that the key to generating alpha is having accurate return forecasts combined with the ability to act on these forecasts (Grinold 1989; Grinold and Kahn 2000).

It defines the **information ratio** (IR) to express the value of active management as the ratio of the return difference between the portfolio and a benchmark to the volatility of those returns. It further approximates the IR as the product of
- The **information coefficient** (IC), which measures the quality of forecast as their rank correlation with the outcomes
- The square root of the **breadth of a strategy** expressed as the number of independent bets on these forecasts

The competition of sophisticated investors in financial markets implies that making precise predictions to generate alpha requires superior information, either through access to better data, a superior ability to process it, or both. This is where ML comes in: applications of **ML for trading (ML4T)** typically aim to make more efficient use of a rapidly diversifying range of data to produce both better and more actionable forecasts, thus improving the quality of investment decisions and results.

Historically, algorithmic trading used to be more narrowly defined as the automation of trade execution to minimize the costs offered by the sell-side. This book takes a more comprehensive perspective since the use of algorithms in general and ML, in particular, has come to impact a broader range of activities from generating ideas and extracting signals from data to asset allocation, position-sizing, and testing and evaluating strategies.

This chapter looks at industry trends that have led to the emergence of ML as a source of competitive advantage in the investment industry. We will also look at where ML fits into the investment process to enable algorithmic trading strategies. 

## Content

1. [The rise of ML in the investment industry](#the-rise-of-ml-in-the-investment-industry)
    * [From electronic to high-frequency trading](#from-electronic-to-high-frequency-trading)
    * [Factor investing and smart beta funds](#factor-investing-and-smart-beta-funds)
    * [Algorithmic pioneers outperform humans](#algorithmic-pioneers-outperform-humans)
        - [ML driven funds attract $1 trillion AUM](#ml-driven-funds-attract-1-trillion-aum)
        - [The emergence of quantamental funds](#the-emergence-of-quantamental-funds)
    * [ML and alternative data](#ml-and-alternative-data)
2. [Designing and executing an ML-driven strategy](#designing-and-executing-an-ml-driven-strategy)
    * [Sourcing and managing data](#sourcing-and-managing-data)
    * [From alpha factor research to portfolio management](#from-alpha-factor-research-to-portfolio-management)
    * [Strategy backtesting](#strategy-backtesting)
3. [ML for trading in practice: strategies and use cases](#ml-for-trading-in-practice-strategies-and-use-cases)
    * [The evolution of algorithmic strategies](#the-evolution-of-algorithmic-strategies)
    * [Use cases of ML for trading](#use-cases-of-ml-for-trading)
        - [Data mining for feature extraction and insights](#data-mining-for-feature-extraction-and-insights)
        - [Supervised learning for alpha factor creation and aggregation](#supervised-learning-for-alpha-factor-creation-and-aggregation)
        - [Asset allocation](#asset-allocation)
        - [Testing trade ideas](#testing-trade-ideas)
        - [Reinforcement learning](#reinforcement-learning)
4. [Resources & References](#resources--references)
    * [Academic Research](#academic-research)
    * [Industry News](#industry-news)
    * [Books](#books)
        - [Machine Learning](#machine-learning)
    * [Courses](#courses)
    * [ML Competitions & Trading](#ml-competitions--trading)
    * [Python Libraries](#python-libraries)

## The rise of ML in the investment industry

The investment industry has evolved dramatically over the last several decades and continues to do so amid increased competition, technological advances, and a challenging economic environment. This section reviews key trends that have shaped the overall investment environment overall and the context for algorithmic trading and the use of ML more specifically.

The trends that have propelled algorithmic trading and ML to current prominence include:
- Changes in the market microstructure, such as the spread of electronic trading and the integration of markets across asset classes and geographies
- The development of investment strategies framed in terms of risk-factor exposure, as opposed to asset classes
- The revolutions in computing power, data generation and management, and statistical methods, including breakthroughs in deep learning
- The outperformance of the pioneers in algorithmic trading relative to human, discretionary investors

In addition, the financial crises of 2001 and 2008 have affected how investors approach diversification and risk management. One outcome is the rise to low-cost passive investment vehicles in the form of exchange-traded funds (ETFs). Amid low yields and low volatility following the 2008 crisis that triggered large-scale asset purchases by leading central banks, cost-conscious investors shifted over $3.5 trillion from actively managed mutual funds into passively managed ETFs. 

Competitive pressure is also reflected in lower hedge fund fees that dropped from the traditional 2 percent annual management fee and 20 percent take of profits to an average of 1.48 percent and 17.4 percent, respectively, in 2017.

### From electronic to high-frequency trading

Electronic trading has advanced dramatically in terms of capabilities, volume, coverage of asset classes, and geographies since networks started routing prices to computer terminals in the 1960s.

- [Dark Pool Trading & Finance](https://www.cfainstitute.org/en/advocacy/issues/dark-pools), CFA Institute
- [Dark Pools in Equity Trading: Policy Concerns and Recent Developments](https://crsreports.congress.gov/product/pdf/R/R43739), Congressional Research Service, 2014
- [High Frequency Trading: Overview of Recent Developments](https://fas.org/sgp/crs/misc/R44443.pdf), Congressional Research Service, 2016

### Factor investing and smart beta funds

The return provided by an asset is a function of the uncertainty or risk associated with the financial investment. An equity investment implies, for example, assuming a company's business risk, and a bond investment implies assuming default risk.

To the extent that specific risk characteristics predict returns, identifying and forecasting the behavior of these risk factors becomes a primary focus when designing an investment strategy. It yields valuable trading signals and is the key to superior active-management results. The industry's understanding of risk factors has evolved very substantially over time and has impacted how ML is used for algorithmic trading.

The factors that explained returns above and beyond the CAPM were incorporated into investment styles that tilt portfolios in favor of one or more factors, and assets began to migrate into factor-based portfolios. The 2008 financial crisis underlined how asset-class labels could be highly misleading and create a false sense of diversification when investors do not look at the underlying factor risks, as asset classes came crashing down together.

Over the past several decades, quantitative factor investing has evolved from a simple approach based on two or three styles to multifactor smart or exotic beta products. Smart beta funds have crossed $1 trillion AUM in 2017, testifying to the popularity of the hybrid investment strategy that combines active and passive management. Smart beta funds take a passive strategy but modify it according to one or more factors, such as cheaper stocks or screening them according to dividend payouts, to generate better returns. This growth has coincided with increasing criticism of the high fees charged by traditional active managers as well as heightened scrutiny of their performance.

The ongoing discovery and successful forecasting of risk factors that, either individually or in combination with other risk factors, significantly impact future asset returns across asset classes is a key driver of the surge in ML in the investment industry and will be a key theme throughout this book.

### Algorithmic pioneers outperform humans

The track record and growth of Assets Under Management (AUM) of firms that spearheaded algorithmic trading has played a key role in generating investor interest and subsequent industry efforts to replicate their success.

Systematic strategies that mostly or exclusively rely on algorithmic decision-making were most famously introduced by mathematician James Simons who founded Renaissance Technologies in 1982 and built it into the premier quant firm. Its secretive Medallion Fund, which is closed to outsiders, has earned an estimated annualized return of 35% since 1982.

DE Shaw, Citadel, and Two Sigma, three of the most prominent quantitative hedge funds that use systematic strategies based on algorithms, rose to the all-time top-20 performers for the first time in 2017 in terms of total dollars earned for investors, after fees, and since inception.

#### ML driven funds attract $1 trillion AUM

Morgan Stanley estimated in 2017 that algorithmic strategies have grown at 15% per year over the past six years and control about $1.5 trillion between hedge funds, mutual funds, and smart beta ETFs. Other reports suggest the quantitative hedge fund industry was about to exceed $1 trillion AUM, nearly doubling its size since 2010 amid outflows from traditional hedge funds. In contrast, total hedge fund industry capital hit $3.21 trillion according to the latest global Hedge Fund Research report.

- [Global Algorithmic Trading Market to Surpass US$ 21,685.53 Million by 2026](https://www.bloomberg.com/press-releases/2019-02-05/global-algorithmic-trading-market-to-surpass-us-21-685-53-million-by-2026)
- [The stockmarket is now run by computers, algorithms and passive managers](https://www.economist.com/briefing/2019/10/05/the-stockmarket-is-now-run-by-computers-algorithms-and-passive-managers), Economist, Oct 5, 2019

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

## Designing and executing an ML-driven strategy

ML can add value at multiple steps in the lifecycle of a trading strategy, and relies on key infrastructure and data resources. Hence, this book aims to addresses how ML techniques fit into the broader process of designing, executing, and evaluating strategies.

An algorithmic trading strategy is driven by a combination of alpha factors that transform one or several data sources into signals that in turn predict future asset returns and trigger buy or sell orders. Chapter 2, Market and Fundamental Data and Chapter 3, Alternative Data for Finance cover the sourcing and management of data, the raw material and the single most important driver of a successful trading strategy.  

[Chapter 4, Alpha Factor Research](../04_alpha_factor_research) outlines a methodologically sound process to manage the risk of false discoveries that increases with the amount of data. [Chapter 5, Strategy Evaluation](../05_strategy_evaluation) provides the context for the execution and performance measurement of a trading strategy.

The following subsections outline these steps, which we will discuss in depth throughout the book.

### Sourcing and managing data

The dramatic evolution of data availability in terms of volume, variety, and velocity is a key complement to the application of ML to trading, which in turn has boosted industry spending on the acquisition of new data sources. However, the proliferating supply of data requires careful selection and management to uncover the potential value, including the following steps:

1. Identify and evaluate market, fundamental, and alternative data sources containing alpha signals that do not decay too quickly.
2. Deploy or access a cloud-based scalable data infrastructure and analytical tools like Hadoop or Spark to facilitate fast, flexible data access.
3. Carefully manage and curate data to avoid look-ahead bias by adjusting it to the desired frequency on a point-in-time basis. This means that data should reflect only information available and known at the given time. ML algorithms trained on distorted historical data will almost certainly fail during live trading.

We will cover these aspects in practical detail in Chapter 2, Market and Fundamental Data: Sources and Techniques, and Chapter 3, Alternative Data for Finance: Categories and Use Cases.

### From alpha factor research to portfolio management

Alpha factors are designed to extract signals from data to predict asset returns for a given investment universe over the trading horizon. A factor takes on a single value for each asset when evaluated, but may combine one or several input variables. The process involves the steps outlined in the following figure:

The Research phase of the trading strategy workflow includes the design, evaluation, and combination of alpha factors. ML plays a large role in this process because the complexity of factors has increased as investors react to both the signal decay of simpler factors and the much richer data available today.

Alpha factors emit entry and exit signals that lead to buy or sell orders, and order execution results in portfolio holdings. The risk profiles of individual positions interact to create a specific portfolio risk profile. Portfolio management involves the optimization of position weights to achieve the desired portfolio risk and return a profile that aligns with the overall investment objectives. This process is highly dynamic to incorporate continuously-evolving market data.

### Strategy backtesting

The incorporation of an investment idea into an algorithmic strategy requires extensive testing with a scientific approach that attempts to reject the idea based on its performance in alternative out-of-sample market scenarios. Testing may involve simulated data to capture scenarios deemed possible but not reflected in historic data.

## ML for trading in practice: strategies and use cases

In practice, we apply ML to trading in the context of a specific strategy to meet a certain business goal. In this section, we briefly describe how trading strategies have evolved and diversified, and outline real-world examples of ML applications, highlighting how they relate to the content covered in this book.

### The evolution of algorithmic strategies

Quantitative strategies have evolved and become more sophisticated in three waves:

1. In the 1980s and 1990s, signals often emerged from academic research and used a single or very few inputs derived from market and fundamental data. AQR, one of the largest quantitative hedge funds today, was founded in 1998 to implement such strategies at scale. These signals are now largely commoditized and available as ETF, such as basic mean-reversion strategies.
2. In the 2000s, factor-based investing proliferated based on the pioneering work by Eugene Fama and Kenneth French and others. Funds used algorithms to identify assets exposed to risk factors like value or momentum to seek arbitrage opportunities. Redemptions during the early days of the financial crisis triggered the quant quake of August 2007 that cascaded through the factor-based fund industry. These strategies are now also available as long-only smart beta funds that tilt portfolios according to a given set of risk factors.
3. The third era is driven by investments in ML capabilities and alternative data to generate profitable signals for repeatable trading strategies. Factor decay is a major challenge: the excess returns from new anomalies have been shown to drop by a quarter from discovery to publication, and by over 50 percent after publication due to competition and crowding.

Today, traders pursue a range of different objectives when using algorithms to execute rules:
- Trade execution algorithms that aim to achieve favorable pricing
- Short-term trades that aim to profit from small price movements, for example, due to arbitrage
- Behavioral strategies that aim to anticipate the behavior of other market participants
- Trading strategies based on absolute and relative price and return predictions

### Use cases of ML for trading

ML extracts signals from a wide range of market, fundamental, and alternative data, and can be applied at all steps of the algorithmic trading-strategy process. Key applications include:
- Data mining to identify patterns, extract features and generate insights
- Supervised learning to generate risk factors or alphas and create trade ideas
- Aggregation of individual signals into a strategy
- Allocation of assets according to risk profiles learned by an algorithm
- The testing and evaluation of strategies, including through the use of synthetic data
- The interactive, automated refinement of a strategy using reinforcement learning

We briefly highlight some of these applications and identify where we will demonstrate their use in later chapters.

#### Data mining for feature extraction and insights

The cost-effective evaluation of large, complex datasets requires the detection of signals at scale. There are several examples throughout the book:
- **Information theory** helps estimate a signal content of candidate features is thus useful for extracting the most valuable inputs for an ML model. In Chapter 4, Financial Feature Engineering: How to Research Alpha Factors, we use mutual information to compare the potential values of individual features for a supervised learning algorithm to predict asset returns. Chapter 18 in De Prado (2018) estimates the information content of a price series as a basis for deciding between alternative trading strategies.
- **Unsupervised learning** provides a broad range of methods to identify structure in data to gain insights or help solve a downstream task. We provide several examples: 
    - In Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning/README.md), we introduce clustering and dimensionality reduction to generate features from high-dimensional datasets. 
    - In Chapter 15, [Topic Modeling for Earnings Calls and Financial News](../15_topic_modeling/README.md), we apply Bayesian probability models to summarize financial text data.
    - In Chapter 20: [Autoencoders for Conditional Risk Factors](../20_autoencoders_for_conditional_risk_factors), we used deep learning to extract non-linear risk factors conditioned on asset characteristics and predict stock returns based on [Kelly et. al.](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) (2020).
- **Model transparency**: we emphasize model-specific ways to gain insights into the predictive power of individual variables and introduce a novel game-theoretic approach called SHapley Additive exPlanations (SHAP). We apply it to gradient boosting machines with a large number of input variables in Chapter 12, Boosting your Trading Strategy and the Appendix.

#### Supervised learning for alpha factor creation and aggregation

The most familiar rationale for applying ML to trading is to obtain predictions of asset fundamentals, price movements, or market conditions. A strategy can leverage multiple ML algorithms that build on each other:

- **Downstream models** can generate signals at the portfolio level by integrating predictions about the prospects of individual assets, capital market expectations, and the correlation among securities. 
- Alternatively, ML predictions can inform **discretionary trades** as in the quantamental approach outlined previously. 

ML predictions can also **target specific risk factors**, such as value or volatility, or implement technical approaches, such as trend-following or mean reversion:
- In Chapter 3, [Alternative Data for Finance: Categories and Use Cases](../03_alternative_data/README.md), we illustrate how to work with fundamental data to create inputs to ML-driven valuation models.
- In Chapter 14, [Text Data for Trading: Sentiment Analysis](../14_working_with_text_data/README.md), Chapter 15, [Topic Modeling for Earnings Calls and Financial News](../15_topic_modeling/README.md), and Chapter 16, [Extracting Better Features: Word Embeddings for Earnings Calls and SEC Filings](../16_word_embeddings/README.md), we use alternative data on business reviews that can be used to project revenues for a company as an input for a valuation exercise.
- In Chapter 9, [From Volatility Forecasts to Statistical Arbitrage: Time Series Models](../09_time_series_models/README.md), we demonstrate how to forecast macro variables as inputs to market expectations and how to forecast risk factors such as volatility
- In Chapter 19, [RNNs for Trading: Multivariate Return Series and Text Data](../19_recurrent_neural_nets/README.md), we introduce recurrent neural networks that achieve superior performance with nonlinear time series data.

#### Asset allocation
ML has been used to allocate portfolios based on decision-tree models that compute a hierarchical form of risk parity. As a result, risk characteristics are driven by patterns in asset prices rather than by asset classes and achieve superior risk-return characteristics.

- In Chapter 5, [Portfolio Optimization and Performance Evaluation](../05_strategy_evaluation/README.md), and Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning/README.md), we illustrate how hierarchical clustering extracts data-driven risk classes that better reflect correlation patterns than conventional asset class definition (see Chapter 16 in De Prado, 2018).

#### Testing trade ideas

Backtesting is a critical step to select successful algorithmic trading strategies. Cross-validation using synthetic data is a key ML technique to generate reliable out-of-sample results when combined with appropriate methods to correct for multiple testing. The time-series nature of financial data requires modifications to the standard approach to avoid look-ahead bias or otherwise contaminate the data used for training, validation, and testing. In addition, the limited availability of historical data has given rise to alternative approaches that use synthetic data:
We will demonstrate various methods to test ML models using market, fundamental, and alternative that obtain sound estimates of out-of-sample errors.
In Chapter 21, [Generative Adversarial Networks for Synthetic Training Data](../21_gans_for_synthetic_time_series/README.md), we present generative adversarial networks (GANs) that are capable of producing high-quality synthetic data.

#### Reinforcement learning

Trading takes place in a competitive, interactive marketplace. Reinforcement learning aims to train agents to learn a policy function based on rewards; it is often considered as one of the most promising areas in financial ML. See, e.g. Hendricks and Wilcox (2014) and Nevmyvaka, Feng, and Kearns (2006) for applications to trade execution.
- In Chapter 22, [Deep Reinforcement Learning: Building a Trading Agent](../22_deep_reinforcement_learning/README.md), we present key reinforcement algorithms like Q-learning to demonstrate the training of reinforcement algorithms for trading using OpenAI's Gym environment.

## Resources & References

### Academic Research

- [The fundamental law of active management](http://jpm.iijournals.com/content/15/3/30), Richard C. Grinold, The Journal of Portfolio Management Spring 1989, 15 (3) 30-37
- [The relationship between return and market value of common stocks](https://www.sciencedirect.com/science/article/pii/0304405X81900180), Rolf Banz,Journal of Financial Economics, March 1981
- [The Arbitrage Pricing Theory: Some Empirical Results](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1981.tb00444.x), Marc Reinganum, Journal of Finance, 1981
- [The Relationship between Earnings' Yield, Market Value and Return for NYSE Common Stock](https://pdfs.semanticscholar.org/26ab/311756099c8f8c4e528083c9b90ff154f98e.pdf), Sanjoy Basu, Journal of Financial Economics, 1982
- [Bridging the divide in financial market forecasting: machine learners vs. financial economists](http://www.sciencedirect.com/science/article/pii/S0957417416302585), Expert Systems with Applications, 2016 
- [Financial Time Series Forecasting with Deep Learning : A Systematic Literature Review: 2005-2019](http://arxiv.org/abs/1911.13288), arXiv:1911.13288 [cs, q-fin, stat], 2019 
- [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009), The Review of Financial Studies, 2020 
- [The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns](http://academic.oup.com/rfs/article/30/12/4389/3091648), The Review of Financial Studies, 2017 
- [Characteristics are covariances: A unified model of risk and return](http://www.sciencedirect.com/science/article/pii/S0304405X19301151), Journal of Financial Economics, 2019 
- [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests](https://doi.org/10.1080/01621459.2017.1319839), Journal of the American Statistical Association, 2018 
- [An Empirical Study of Machine Learning Algorithms for Stock Daily Trading Strategy](https://www.hindawi.com/journals/mpe/2019/7816154/), Mathematical Problems in Engineering, 2019 
- [Predicting stock market index using fusion of machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414006551), Expert Systems with Applications, 2015 
- [Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414004473), Expert Systems with Applications, 2015 
- [Deep Learning for Limit Order Books](http://arxiv.org/abs/1601.01987), arXiv:1601.01987 [q-fin], 2016 
- [Trading via Image Classification](http://arxiv.org/abs/1907.10046), arXiv:1907.10046 [cs, q-fin], 2019 
- [Algorithmic trading review](http://doi.org/10.1145/2500117), Communications of the ACM, 2013 
- [Assessing the impact of algorithmic trading on markets: A simulation approach](https://www.econstor.eu/handle/10419/43250), , 2008 
- [The Efficient Market Hypothesis and Its Critics](http://www.aeaweb.org/articles?id=10.1257/089533003321164958), Journal of Economic Perspectives, 2003 
- [The Arbitrage Pricing Theory Approach to Strategic Portfolio Planning](https://doi.org/10.2469/faj.v40.n3.14), Financial Analysts Journal, 1984 

### Industry News

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
- [Machine Learning in Finance](https://www.springer.com/gp/book/9783030410674), Dixon, Matthew F., Halperin, Igor, Bilokon, Paul, Springer, 2020

#### Machine Learning

- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Tom Mitchell, McGraw Hill, 1997
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), Gareth James et al.
    - Excellent reference for essential machine learning concepts, available free online
- [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf), Barber, D., Cambridge University Press, 2012 (updated version available on author's website)

### Courses

- [Algorithmic Trading](http://personal.stevens.edu/~syang14/fe670.htm), Prof. Steve Yang, Stevens Institute of Technology
- [Machine Learning](https://www.coursera.org/learn/machine-learning), Andrew Ng, Coursera
- [Deep Learning Specialization](http://deeplearning.ai/), Andrew Ng
    - Andrew Ng’s introductory deep learning course
- Machine Learning for Trading Specialization, [Coursera](https://www.coursera.org/specializations/machine-learning-trading)
- Machine Learning for Trading, Georgia Tech CS 7646, [Udacity](https://www.udacity.com/course/machine-learning-for-trading--ud501
- Introduction to Machine Learning for Trading, [Quantinsti](https://quantra.quantinsti.com/course/introduction-to-machine-learning-for-trading)

### ML Competitions & Trading

- [IEEE Investment Ranking Challenge](https://www.crowdai.org/challenges/ieee-investment-ranking-challenge)
    - [Investment Ranking Challenge : Identifying the best performing stocks based on their semi-annual returns](https://arxiv.org/pdf/1906.08636.pdf)
- [Two Sigma Financial Modeling Challenge](https://www.kaggle.com/c/two-sigma-financial-modeling)
- [Two Sigma: Using News to Predict Stock Movements](https://www.kaggle.com/c/two-sigma-financial-news)
- [The Winton Stock Market Challenge](https://www.kaggle.com/c/the-winton-stock-market-challenge)
- [Algorithmic Trading Challenge](https://www.kaggle.com/c/AlgorithmicTradingChallenge)
   
### Python Libraries

- matplotlib [docs](https://github.com/matplotlib/matplotlib)
- numpy [docs](https://github.com/numpy/numpy)
- pandas [docs](https://github.com/pydata/pandas)
- scipy [docs](https://github.com/scipy/scipy)
- scikit-learn [docs](https://scikit-learn.org/stable/user_guide.html)
- LightGBM [docs](https://lightgbm.readthedocs.io/en/latest/)
- CatBoost [docs](https://catboost.ai/docs/concepts/about.html)
- TensorFlow [docs](https://www.tensorflow.org/guide)
- PyTorch [docs](https://pytorch.org/docs/stable/index.html)
- Machine Learning Financial Laboratory (mlfinlab) [docs](https://mlfinlab.readthedocs.io/en/latest/)
- seaborn [docs](https://github.com/mwaskom/seaborn)
- statsmodels [docs](https://github.com/statsmodels/statsmodels)
- [Boosting numpy: Why BLAS Matters](http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)



















































