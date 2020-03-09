# coding: utf-8


import re
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


def parse_html(html):
    """Parse content from various tags from OpenTable restaurants listing"""
    data, item = pd.DataFrame(), {}
    soup = BeautifulSoup(html, 'lxml')
    for i, resto in enumerate(soup.find_all('div', class_='rest-row-info')):
        item['name'] = resto.find('span', class_='rest-row-name-text').text

        booking = resto.find('div', class_='booking')
        item['bookings'] = re.search('\d+', booking.text).group() if booking else 'NA'

        rating = resto.find('div', class_='star-rating-score')
        item['rating'] = float(rating['aria-label'].split()[0]) if rating else 'NA'

        reviews = resto.find('span', class_='underline-hover')
        item['reviews'] = int(re.search('\d+', reviews.text).group()) if reviews else 'NA'

        item['price'] = int(resto.find('div', class_='rest-row-pricing').find('i').text.count('$'))
        item['cuisine'] = resto.find('span', class_='rest-row-meta--cuisine rest-row-meta-text sfx1388addContent').text
        item['location'] = resto.find('span', class_='rest-row-meta--location rest-row-meta-text sfx1388addContent').text
        data[i] = pd.Series(item)
    return data.T


# Start selenium and click through pages until reach end
# store results by iteratively appending to csv file
driver = webdriver.Firefox()
url = "https://www.opentable.com/new-york-restaurant-listings"
driver.get(url)
page = collected = 0
while True:
    sleep(1)
    new_data = parse_html(driver.page_source)
    if new_data.empty:
        break
    if page == 0:
        new_data.to_csv('results.csv', index=False)
    elif page > 0:
        new_data.to_csv('results.csv', index=False, header=None, mode='a')
    page += 1
    collected += len(new_data)
    print(f'Page: {page} | Downloaded: {collected}')
    driver.find_element_by_link_text('Next').click()

driver.close()
restaurants = pd.read_csv('results.csv')
print(restaurants)
