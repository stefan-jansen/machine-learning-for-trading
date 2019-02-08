#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from opentable.items import OpentableItem
from scrapy import Spider
from scrapy_splash import SplashRequest


class OpenTableSpider(Spider):
    name = 'opentable'
    start_urls = ['https://www.opentable.com/new-york-restaurant-listings']

    def start_requests(self):
        for url in self.start_urls:
            yield SplashRequest(url=url,
                                callback=self.parse,
                                endpoint='render.html',
                                args={'wait': 1},
                                )

    def parse(self, response):
        item = OpentableItem()
        for resto in response.css('div.rest-row-info'):
            item['name'] = resto.css('span.rest-row-name-text::text').extract()
            item['bookings'] = resto.css('div.booking::text').re(r'\d+')
            item['rating'] = resto.css('div.all-stars::attr(style)').re_first('\d+')
            item['reviews'] = resto.css('span.star-rating-text--review-text::text').re_first(r'\d+')
            item['price'] = len(resto.css('div.rest-row-pricing > i::text').re('\$'))
            item['cuisine'] = resto.css('span.rest-row-meta--cuisine::text').extract()
            item['location'] = resto.css('span.rest-row-meta--location::text').extract()
            yield item
