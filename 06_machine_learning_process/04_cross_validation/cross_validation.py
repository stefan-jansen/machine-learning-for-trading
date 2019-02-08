#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import warnings
import pandas as pd
import numpy as np
from numpy.random import rand, randn, normal
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedKFold, \
    TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data = list(range(1, 11))
print(data)

print(train_test_split(data, train_size=.8))

kf = KFold(n_splits=5)
for train, validate in kf.split(data):
    print(train, validate)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train, validate in kf.split(data):
    print(train, validate)


loo = LeaveOneOut()
for train, validate in loo.split(data):
    print(train, validate)


lpo = LeavePOut(p=2)
for train, validate in lpo.split(data):
    print(train, validate)


ss = ShuffleSplit(n_splits=3, test_size=2, random_state=0)
for train, validate in ss.split(data):
    print(train, validate)


tscv = TimeSeriesSplit(n_splits=5)
for train, validate in tscv.split(data):
    print(train, validate)