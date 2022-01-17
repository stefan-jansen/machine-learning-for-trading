#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import re
from pathlib import Path
from random import random
from time import sleep
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from furl import furl
from selenium import webdriver
import calendar

transcript_path = Path('transcripts')


def store_result(meta, participants, content):
    """Save parse content to csv"""
    path = transcript_path / 'parsed' / meta['symbol']
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(content, columns=['speaker', 'q&a', 'content']).to_csv(path / 'content.csv', index=False)
    pd.DataFrame(participants, columns=['type', 'name']).to_csv(path / 'participants.csv', index=False)
    pd.Series(meta).to_csv(path / 'earnings.csv')


def parse_html(html):
    """Main html parser function"""
    date_pattern = re.compile(r'(^\w+).\s(\d{2}),\s(\d{4})')
    quarter_pattern = re.compile(r'(\bQ\d\b)')
    soup = BeautifulSoup(html, 'lxml')

    meta, participants, content = {}, [], []
    h1 = soup.find('h1', attrs={"data-test-id": "post-title"})
    if h1 is None:
        return
    h1 = h1.text
    meta['company'] = h1[:h1.find('(')].strip()
    meta['symbol'] = h1[h1.find('(') + 1:h1.find(')')]

    match = quarter_pattern.search(h1)
    if match:
        meta['quarter'] = match.group(0)
    
    data_span = soup.find('span', attrs={"data-test-id": "post-date"})
    if data_span is not None:
        date_string = data_span.text
        match = date_pattern.search(date_string)
        if match:
            m, d, y = match.groups()
            meta['month'] = int(list(calendar.month_abbr).index(m))
            meta['day'] = int(d)
            meta['year'] = int(y)

    qa = 0
    speaker_types = ['Executive', 'Analysts', 'President']
    for header in [p.parent for p in soup.find_all('strong')]:
        checks = [soup.find(name='p', string=re.compile(r'(' + header.text + ' - ' + speaker_type + ').*')) is not None
                  for speaker_type in speaker_types]
        text = header.text.strip()
        if text.lower().startswith('copyright'):
            continue
        elif text.lower().startswith('question-and'):
            qa = 1
            continue
        elif any(checks):
            if header.text not in [p[1] for p in participants]:
                participants.append([speaker_types[checks.index(True)], header.text])
        else:
            p = []
            for participant in header.find_next_siblings('p'):
                if participant.find('strong'):
                    break
                else:
                    p.append(participant.text)
            content.append([header.text, qa, '\n'.join(p)])
    return meta, participants, content


SA_URL = 'https://seekingalpha.com/'
TRANSCRIPT = re.compile('Earnings Call Transcript')

next_page = True
page = 1
driver = webdriver.Firefox()
while next_page:
    print(f'Page: {page}')
    url = f'{SA_URL}/earnings/earnings-call-transcripts?page={page}'
    driver.get(urljoin(SA_URL, url))
    sleep(8 + (random() - .5) * 2)
    response = driver.page_source
    page += 1
    soup = BeautifulSoup(response, 'lxml')
    links = soup.find_all(name='a', string=TRANSCRIPT)
    if len(links) == 0:
        next_page = False
    else:
        for link in links:
            transcript_url = link.attrs.get('href')
            article_url = furl(urljoin(SA_URL, transcript_url)).add({'part': 'single'})
            driver.get(article_url.url)
            html = driver.page_source
            result = parse_html(html)
            if result is not None:
                meta, participants, content = result
                meta['link'] = link
                store_result(meta, participants, content)
                sleep(8 + (random() - .5) * 2)

driver.close()

