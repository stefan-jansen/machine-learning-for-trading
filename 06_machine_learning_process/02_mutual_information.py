#!/usr/bin/env python
# coding: utf-8

# Using information theory to evaluate features

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
idx = pd.IndexSlice

# Get Data


with pd.HDFStore('../data/assets.h5') as store:
    data = store['engineered_features']

# Create Dummy variables

dummy_data = pd.get_dummies(data,
                            columns=['year', 'month', 'msize', 'age', 'sector'],
                            prefix=['year', 'month', 'msize', 'age', ''],
                            prefix_sep=['_', '_', '_', '_', ''])
dummy_data = dummy_data.rename(columns={c: c.replace('.0', '') for c in dummy_data.columns})
print(dummy_data.info())

# Mutual Information

target_labels = [f'target_{i}m' for i in [1, 2, 3, 6, 12]]
targets = data.dropna().loc[:, target_labels]

features = data.dropna().drop(target_labels, axis=1)
features.sector = pd.factorize(features.sector)[0]

cat_cols = ['year', 'month', 'msize', 'age', 'sector']
discrete_features = [features.columns.get_loc(c) for c in cat_cols]

mutual_info = pd.DataFrame()
for label in target_labels:
    mi = mutual_info_classif(X=features,
                             y=(targets[label] > 0).astype(int),
                             discrete_features=discrete_features,
                             random_state=42
                             )
    mutual_info[label] = pd.Series(mi, index=features.columns)

print(mutual_info.sum())

# Normalized MI Heatmap
fig, ax = plt.subplots(figsize=(15, 4))
sns.heatmap(mutual_info.div(mutual_info.sum()).T, ax=ax, cmap='Blues')
plt.show()

# Dummy Data


target_labels = [f'target_{i}m' for i in [1, 2, 3, 6, 12]]
dummy_targets = dummy_data.dropna().loc[:, target_labels]

dummy_features = dummy_data.dropna().drop(target_labels, axis=1)
cat_cols = [c for c in dummy_features.columns if c not in features.columns]
discrete_features = [dummy_features.columns.get_loc(c) for c in cat_cols]

mutual_info_dummies = pd.DataFrame()
for label in target_labels:
    mi = mutual_info_classif(X=dummy_features,
                             y=(dummy_targets[label] > 0).astype(int),
                             discrete_features=discrete_features,
                             random_state=42
                             )
    mutual_info_dummies[label] = pd.Series(mi, index=dummy_features.columns)

print(mutual_info_dummies.sum())

fig, ax = plt.subplots(figsize=(4, 20))
sns.heatmap(mutual_info_dummies.div(mutual_info_dummies.sum()), ax=ax, cmap='Blues')
plt.show()
