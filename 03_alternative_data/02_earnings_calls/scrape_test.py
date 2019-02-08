#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'
from bs4 import BeautifulSoup
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from time import sleep
import pickle
import requests
from lxml import html

from os import environ

EMAIL = environ['SEEKING_ALPHA_USER']
PASS = environ['SEEKING_ALPHA_PWD']

driver = webdriver.Chrome()
url = 'http://seekingalpha.com/account/login'
# url = 'https://seekingalpha.com/'
driver.get(url)

driver.find_element_by_id("sign-in").click()
# box = 'alphabox-modal-window'

sleep(1)

try:
    email = driver.find_element_by_id("authentication_login_email")
    email.send_keys(EMAIL)
except Exception as e:
    print(e)

try:
    password = driver.find_element_by_id('authentication_login_password')
    password.send_keys(PASS)
except Exception as e:
    print(e)

try:
    driver.find_element_by_xpath("//input[@value='Sign in' and @class='c']").click()
    # WebDriverWait(driver, 10).until(expected_conditions.title_contains("home"))

except Exception as e:
    print(e)
# html = driver.page_source
sleep(10)
cookies = driver.get_cookies()

pickle.dump(cookies, open('SA_cookies.pkl', 'wb'))
driver.close()

# exit()
# WebDriverWait(driver, 10).until(expected_conditions.title_contains("home"))


sessionRequests = requests.Session()


# This is the form data that the page sends when logging in
loginData = {
    'slugs[]'              : None,
    'rt'                   : None,
    'user[url_source]'     : None,
    'user[location_source]': 'orthodox_login',
    'user[email]'          : keys['username'],
    'user[password]'       : keys['password'],

}
# Authenticate
r = sessionRequests.post(loginUrl, data=loginData, headers={"Referer"   : "http://seekingalpha.com/",
                                                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"})
