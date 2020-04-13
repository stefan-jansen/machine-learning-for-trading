# Alternative Data for Trading

This chapter explains how individuals, business processes, and sensors produce alternative data. It also provides a framework to navigate and evaluate the proliferating supply of alternative data for investment purposes. 

It demonstrates the workflow, from acquisition to preprocessing and storage using Python for data obtained through web scraping to set the stage for the application of ML. It concludes by providing examples of sources, providers, and applications.

 More specifically, this chapter covers:

- How the alternative data revolution has unleashed new sources of information
- How individuals, business processes, and sensors generate alternative data
- How to evaluate the proliferating supply of alternative data used for algorithmic trading
- How to work with alternative data in Python, such as by scraping the internet
- Important categories and providers of alternative data

## The Alternative Data Revolution

For algorithmic trading, new data sources offer an informational advantage if they provide access to information unavailable from traditional sources, or provide access sooner. Following global trends, the investment industry is rapidly expanding beyond market and fundamental data to alternative sources to reap alpha through an informational edge. Annual spending on data, technological capabilities, and related talent are expected to increase from the current $3 billion by 12.8% annually through 2020.

Today, investors can access macro or company-specific data in real-time that historically has been available only at a much lower frequency. Use cases for new data sources include the following:
- Online price data on a representative set of goods and services can be used to measure inflation
- The number of store visits or purchases permits real-time estimates of company or industry-specific sales or economic activity
- Satellite images can reveal agricultural yields, or activity at mines or on oil rigs before this information is available elsewhere 

Useful references include:
- [The Digital Universe in 2020](https://www.emc.com/collateral/analyst-reports/idc-the-digital-universe-in-2020.pdf)
- [Big data: The next frontier for innovation, competition, and productivity](https://www.mckinsey.com/business-functions/digital-mckinsey/our-insights/big-data-the-next-frontier-for-innovation), McKinsey 2011
- [McKinsey on Artificial Intelligence](https://www.mckinsey.com/featured-insights/artificial-intelligence)


## Evaluating Alternative Datasets

The ultimate objective of alternative data is to provide an informational advantage in the competitive search for trading signals that produce alpha, namely positive, uncorrelated investment returns. In practice, the signals extracted from alternative datasets can be used on a standalone basis or combined with other signals as part of a quantitative strategy.

Key reference: 

- [Big Data and AI Strategies](http://valuesimplex.com/articles/JPM.pdf), Kolanovic, M. and Krishnamachari, R., JP Morgan, May 2017

## The Market for Alternative Data

The investment industry is going to spend an estimated $2bn-3bn on data services in 2018, and this number is expected to grow at double digits per year in line with other industries. This expenditure includes the acquisition of alternative data, investments in related technology, and the hiring of qualified talent.

 - [Alternative Data](https://alternativedata.org/)

## Working with Alternative Data

This section illustrates the acquisition of alternative data using web scraping, targeting first OpenTable restaurant data, and then move to earnings call transcripts hosted by Seeking Alpha.

- [Quantifying Trading Behavior in Financial Markets Using Google Trends](https://www.nature.com/articles/srep01684), Preis, Moat and Stanley, Nature, 2013
- [Quantifying StockTwits semantic terms’ trading behavior in financial markets: An effective application of decision tree algorithms](https://www.sciencedirect.com/science/article/pii/S0957417415005473), Al Nasseri et al, Expert Systems with Applications, 2015

#### Open Table Web Scraping

This subfolder [01_opentable](01_opentable) contains the script [opentable_selenium](01_opentable/opentable_selenium.py) to scrape OpenTable data using Scrapy and Selenium.

- [How to View the Source Code of a Web Page in Every Browser](https://www.lifewire.com/view-web-source-code-4151702)

#### SeekingAlpha Earnings Transcripts

The subfolder [02_earnings_calls](02_earnings_calls) contains the script [sa_selenium](02_earnings_calls/sa_selenium.py) to scrape earnings call transcripts from the [SeekingAlpha](www.seekingalpha.com) website.

#### Python Libraries & Documentation
- requests [docs](http://docs.python-requests.org/en/master/)
- beautifulsoup [docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/﻿)
- Selenium [docs](https://www.seleniumhq.org/﻿)
- Scrapy [docs](https://scapy.readthedocs.io/en/latest/)

