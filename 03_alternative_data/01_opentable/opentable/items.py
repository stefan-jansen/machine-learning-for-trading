# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Field, Item


class OpentableItem(Item):
    name = Field()
    price = Field()
    bookings = Field()
    cuisine = Field()
    location = Field()
    reviews = Field()
    rating = Field()
