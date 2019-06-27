#!/usr/bin/env python
# coding: utf-8

# Basic Walk-Through: k-nearest Neighbors

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, validation_curve, learning_curve, GridSearchCV
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler, scale
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from yellowbrick.model_selection import ValidationCurve, LearningCurve

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')

# Get the Data

# Data from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)
#
# ```kaggle datasets download -d harlfoxem/housesalesprediction```

DATA_PATH = Path('..', 'data')

house_sales = pd.read_csv(DATA_PATH / 'kc_house_data.csv')
house_sales = house_sales.drop(
        ['id', 'zipcode', 'lat', 'long', 'date'], axis=1)
print(house_sales.info())

# Select & Transform Features

# Asset Prices often have long tails
sns.distplot(house_sales.price)
plt.show()

# Use log-transform
# Useful for dealing with [skewed data](http://onlinestatbook.com/2/transformations/log.html).
X_all = house_sales.drop('price', axis=1)
y = np.log(house_sales.price)

# Mutual information regression
mi_reg = pd.Series(mutual_info_regression(X_all, y),
                   index=X_all.columns).sort_values(ascending=False)
print(mi_reg)

X = X_all.loc[:, mi_reg.iloc[:10].index]

# Bivariate Scatter Plots
sns.pairplot(X.assign(price=y), y_vars=['price'], x_vars=X.columns)
plt.show()

# Explore Correlations
correl = (X
          .apply(lambda x: spearmanr(x, y))
          .apply(pd.Series, index=['r', 'pval']))

correl.r.sort_values().plot.barh();

# KNN Regression
X_scaled = scale(X)
model = KNeighborsRegressor()
model.fit(X=X_scaled, y=y)
y_pred = model.predict(X_scaled)
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_squared_log_error,
                             median_absolute_error,
                             explained_variance_score,
                             r2_score)

# Computing the prediction error
error = (y - y_pred).rename('Prediction Errors')

scores = dict(
        rmse=np.sqrt(mean_squared_error(y_true=y, y_pred=y_pred)),
        rmsle=np.sqrt(mean_squared_log_error(y_true=y, y_pred=y_pred)),
        mean_ae=mean_absolute_error(y_true=y, y_pred=y_pred),
        median_ae=median_absolute_error(y_true=y, y_pred=y_pred),
        r2score=explained_variance_score(y_true=y, y_pred=y_pred)
)
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
sns.scatterplot(x=y, y=y_pred, ax=axes[0])
axes[0].set_xlabel('Log Price')
axes[0].set_ylabel('Predictions')
axes[0].set_ylim(11, 16)
sns.distplot(error, ax=axes[1])
pd.Series(scores).plot.barh(ax=axes[2], title='Error Metrics')
fig.suptitle('In-Sample Regression Errors', fontsize=24)
fig.tight_layout()
plt.subplots_adjust(top=.8)
plt.show()


# Cross-Validation

def rmse(y_true, pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=pred))


rmse_score = make_scorer(rmse)

cv_rmse = {}
n_neighbors = [1] + list(range(5, 51, 5))
for n in n_neighbors:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('knn', KNeighborsRegressor(n_neighbors=n))])
    cv_rmse[n] = cross_val_score(pipe,
                                 X=X,
                                 y=y,
                                 scoring=rmse_score,
                                 cv=5)

cv_rmse = pd.DataFrame.from_dict(cv_rmse, orient='index')
best_n, best_rmse = cv_rmse.mean(1).idxmin(), cv_rmse.mean(1).min()
cv_rmse = cv_rmse.stack().reset_index()
cv_rmse.columns = ['n', 'fold', 'RMSE']

ax = sns.lineplot(x='n', y='RMSE', data=cv_rmse)
ax.set_title(f'Cross-Validation Results KNN | Best N: {best_n:d} | Best RMSE: {best_rmse:.2f}')
plt.show()

# Actuals vs Predicted

pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsRegressor(n_neighbors=best_n))])
y_pred = cross_val_predict(pipe, X, y, cv=5)

ax = sns.scatterplot(x=y, y=y_pred)
y_range = list(range(int(y.min() + 1), int(y.max() + 1)))
pd.Series(y_range, index=y_range).plot(ax=ax, lw=2, c='darkred');

# Cross-Validation Errors

error = (y - y_pred).rename('Prediction Errors')

scores = dict(
        rmse=np.sqrt(mean_squared_error(y_true=y, y_pred=y_pred)),
        rmsle=np.sqrt(mean_squared_log_error(y_true=y, y_pred=y_pred)),
        mean_ae=mean_absolute_error(y_true=y, y_pred=y_pred),
        median_ae=median_absolute_error(y_true=y, y_pred=y_pred),
        r2score=explained_variance_score(y_true=y, y_pred=y_pred)
)

fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
sns.scatterplot(x=y, y=y_pred, ax=axes[0])
axes[0].set_xlabel('Log Price')
axes[0].set_ylabel('Predictions')
axes[0].set_ylim(11, 16)
sns.distplot(error, ax=axes[1])
pd.Series(scores).plot.barh(ax=axes[2], title='Error Metrics')
fig.suptitle('Cross-Validation Regression Errors', fontsize=24)
fig.tight_layout()
plt.subplots_adjust(top=.8)
plt.show()

# GridSearchCV with Pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsRegressor())])

n_folds = 5
n_neighbors = tuple(range(5, 101, 5))
param_grid = {'knn__n_neighbors': n_neighbors}
estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         scoring=rmse_score,
                         #                          n_jobs=-1
                         )
estimator.fit(X=X, y=y)

cv_results = estimator.cv_results_

test_scores = pd.DataFrame({fold: cv_results[f'split{fold}_test_score'] for fold in range(n_folds)},
                           index=n_neighbors).stack().reset_index()
test_scores.columns = ['k', 'fold', 'RMSE']

mean_rmse = test_scores.groupby('k').RMSE.mean()
best_k, best_score = mean_rmse.idxmin(), mean_rmse.min()

sns.pointplot(x='k', y='RMSE', data=test_scores, scale=.3, join=False, errwidth=2)
plt.title('Cross Validation Results')
plt.tight_layout()
plt.gcf().set_size_inches(10, 5)
plt.show()

# Train & Validation Curves mit yellowbricks
fig, ax = plt.subplots(figsize=(16, 9))
val_curve = ValidationCurve(KNeighborsRegressor(),
                            param_name='n_neighbors',
                            param_range=n_neighbors,
                            cv=5,
                            scoring=rmse_score,
                            #                       n_jobs=-1,
                            ax=ax)
val_curve.fit(X, y)
val_curve.poof()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(16, 9))
l_curve = LearningCurve(KNeighborsRegressor(n_neighbors=best_k),
                        train_sizes=np.arange(.1, 1.01, .1),
                        scoring=rmse_score,
                        cv=5,
                        #                         n_jobs=-1,
                        ax=ax)
l_curve.fit(X, y)
l_curve.poof()
fig.tight_layout()
plt.show()

# Binary Classification
y_binary = (y > y.median()).astype(int)
n_neighbors = tuple(range(5, 151, 10))
n_folds = 5
scoring = 'roc_auc'

pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsClassifier())])

param_grid = {'knn__n_neighbors': n_neighbors}

estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         scoring=scoring,
                         #                          n_jobs=-1
                         )
estimator.fit(X=X, y=y_binary)

best_k = estimator.best_params_['knn__n_neighbors']

fig, ax = plt.subplots(figsize=(16, 9))
val_curve = ValidationCurve(KNeighborsClassifier(),
                            param_name='n_neighbors',
                            param_range=n_neighbors,
                            cv=n_folds,
                            scoring=scoring,
                            #                       n_jobs=-1,
                            ax=ax)
val_curve.fit(X, y_binary)
val_curve.poof()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(16, 9))
l_curve = LearningCurve(KNeighborsClassifier(n_neighbors=best_k),
                        train_sizes=np.arange(.1, 1.01, .1),
                        scoring=scoring,
                        cv=5,
                        #                         n_jobs=-1,
                        ax=ax)
l_curve.fit(X, y_binary)
l_curve.poof()
fig.tight_layout()
plt.show()

# Classification Metrics

from sklearn.metrics import (classification_report,
                             accuracy_score,
                             zero_one_loss,
                             auc,
                             roc_auc_score,
                             roc_curve,
                             brier_score_loss,
                             cohen_kappa_score,
                             confusion_matrix,
                             fbeta_score,
                             hamming_loss,
                             hinge_loss,
                             jaccard_similarity_score,
                             log_loss,
                             matthews_corrcoef,
                             f1_score,
                             precision_recall_fscore_support,
                             average_precision_score,
                             precision_recall_curve,
                             precision_score,
                             recall_score)

y_score = cross_val_predict(KNeighborsClassifier(best_k),
                            X=X,
                            y=y_binary,
                            cv=5,
                            #                            n_jobs=-1,
                            method='predict_proba')[:, 1]

# Using Predicted Probabilities
pred_scores = dict(y_true=y_binary, y_score=y_score)

# ROC AUC
roc_auc_score(**pred_scores)
cols = ['False Positive Rate', 'True Positive Rate', 'threshold']
roc = pd.DataFrame(dict(zip(cols, roc_curve(**pred_scores))))

# Precision-Recall
precision, recall, ts = precision_recall_curve(y_true=y_binary, probas_pred=y_score)
pr_curve = pd.DataFrame({'Precision': precision, 'Recall': recall})

# F1 Score - Optimize Threshold
f1 = pd.Series({t: f1_score(y_true=y_binary, y_pred=y_score > t) for t in ts})
best_threshold = f1.idxmax()

# Plot
fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

ax = sns.scatterplot(x='False Positive Rate', y='True Positive Rate', data=roc, size=5, legend=False, ax=axes[0])
axes[0].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='k', ls='--', lw=1)
axes[0].fill_between(y1=roc['True Positive Rate'], x=roc['False Positive Rate'], alpha=.5, color='darkred')
axes[0].set_title('Receiver Operating Characteristic')

sns.scatterplot(x='Recall', y='Precision', data=pr_curve, ax=axes[1])
axes[1].set_ylim(0, 1)
axes[1].set_title('Precision-Recall Curve')

f1.plot(ax=axes[2], title='F1 Scores', ylim=(0, 1))
axes[2].set_xlabel('Threshold')
axes[2].axvline(best_threshold, lw=1, ls='--', color='k')
axes[2].text(text=f'Max F1 @ {best_threshold:.2f}', x=.75, y=.95, s=5)
fig.tight_layout()
plt.show()

# Average Precision
average_precision_score(y_true=y_binary, y_score=y_score)

# Brier Score
brier_score_loss(y_true=y_binary, y_prob=y_score)

# Use Predictions after thresholding
y_pred = y_score > best_threshold
scores = dict(y_true=y_binary, y_pred=y_pred)

# F-beta Score
fbeta_score(**scores, beta=1)
print(classification_report(**scores))

# Confusion Matrix
print(confusion_matrix(**scores))

# Accuracy
print(accuracy_score(**scores))

# Zero-One Loss
print(zero_one_loss(**scores))

# Hamming Loss
# Fraction of labels that are incorrectly predicted
print(hamming_loss(**scores))

# Cohen's Kappa
# Score that expresses the level of agreement between two annotators on a classification problem.
print(cohen_kappa_score(y1=y_binary, y2=y_pred))

# Hinge Loss
print(hinge_loss(y_true=y_binary, pred_decision=y_pred))

# Jaccard Similarity
print(jaccard_similarity_score(**scores))

# Log Loss / Cross Entropy Loss
print(log_loss(**scores))

# Matthews Correlation Coefficient
print(matthews_corrcoef(**scores))

# Multi-Class
y_multi = pd.qcut(y, q=3, labels=[0, 1, 2])
n_neighbors = tuple(range(5, 151, 10))
n_folds = 5
scoring = 'accuracy'

pipe = Pipeline([('scaler', StandardScaler()),
                 ('knn', KNeighborsClassifier())])

param_grid = {'knn__n_neighbors': n_neighbors}

estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         n_jobs=-1
                         )
estimator.fit(X=X, y=y_multi)

y_pred = cross_val_predict(estimator.best_estimator_,
                           X=X,
                           y=y_multi,
                           cv=5,
                           n_jobs=-1,
                           method='predict')

print(classification_report(y_true=y_multi, y_pred=y_pred))
