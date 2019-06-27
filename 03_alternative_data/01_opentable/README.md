## Scraping OpenTable data

Typical sources of alternative data are review websites such as Glassdoor or Yelp that convey insider insights using employee comments or guest reviews. This data provides valuable input for ML models that aim to predict a business' prospects or directly its market value to obtain trading signals.

The data needs to be extracted from the HTML source, barring any legal obstacles. To illustrate the web scraping tools that Python offers, we'll retrieve information on restaurant bookings from OpenTable. Data of this nature could be used to forecast economic activity by geography, real estate prices, or restaurant chain revenues.

### Building a dataset of restaurant bookings

With the browser automation tool [Selenium](https://www.seleniumhq.org/), you can follow the links to the next pages and quickly build a dataset of over 10,000 restaurants in NYC that you could then update periodically to track a time series.

To set up selenium, run 
```bash
./selenium_setup.sh
```
with suitable permission, i.e., after running `chmod +x selenium_setup.sh`.

The script [opentable_selenium](opentable_selenium.py) illustrates how to scrape and store the data. Simply run as 
```python
python opentable_selenium.py
```

Since websites change frequently, this code may stop working at any moment.

### One step further – Scrapy and splash

Scrapy is a powerful library to build bots that follow links, retrieve the content, and store the parsed result in a structured way. In combination with the headless browser splash, it can also interpret JavaScript and becomes an efficient alternative to Selenium. 

You can run the spider using the `scrapy crawl opentable` command in the 01_opentable directory where the results are logged to spider.log.




