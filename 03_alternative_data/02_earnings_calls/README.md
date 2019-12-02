## How to Scrape Earnings Call Transcripts

Textual data is an essential alternative data source. One example of textual information is transcripts of earnings calls where executives do not only present the latest financial results, but also respond to questions by financial analysts. Investors utilize transcripts to evaluate changes in sentiment, emphasis on particular topics, or style of communication.

We will illustrate the scraping and parsing of earnings call transcripts from the popular trading website [www.seekingalpha.com](www.seekingalpha.com).

### Instructions

This section contains code to retrieve earnings call transcripts from Seeking Alpha.

Run `python sa_selenium.py` file to scrape transcripts and store the result under transcipts/parts and the company's symbol in csv files, named by the aspect of the earnings call they capture:
- content: statements and Q&A content
- participants: as listed by seeking alpha
- earnings: date and company the earnings the call is referring to

This requires [geckodriver](https://github.com/mozilla/geckodriver/releases) and [Firefox](https://www.mozilla.org/en-US/firefox/new/). 

- On macOS, you can use ```brew install geckodriver```.
- See [here](https://askubuntu.com/questions/870530/how-to-install-geckodriver-in-ubuntu) for Ubuntu.






