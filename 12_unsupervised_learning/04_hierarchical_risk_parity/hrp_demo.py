#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

# Hierarchical Risk Parity
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
import random
import numpy as np
import pandas as pd


def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = list(range(0, sortIx.shape[0] * 2, 2))  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index;
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = list(range(sortIx.shape[0]))  # re-index
    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        print(cItems)
        cItems = [i[j:k] for i in cItems for j, k in ((0, int(len(i) / 2)), (int(len(i) / 2), len(i))) if
                  len(i) > 1]  # bi-section

    for i in range(0, len(cItems), 2):  # parse in pairs
        cItems0 = cItems[i]  # cluster 1
        cItems1 = cItems[i + 1]  # cluster 2
        cVar0 = getClusterVar(cov, cItems0)
        cVar1 = getClusterVar(cov, cItems1)
        alpha = 1 - cVar0 / (cVar0 + cVar1)
        w[cItems0] *= alpha  # weight 1
        w[cItems1] *= 1 - alpha  # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    return dist


def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.savefig(path)
    mpl.clf()
    mpl.close()  # reset pylab


def generateData(nObs, size0, size1, sigma1):
    # Time series of correlated variables

    # 1) generating some uncorrelated data
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))  # each row is a variable

    # 2) creating correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=list(range(1, x.shape[1] + 1)))
    return x, cols


def main():
    # 1) Generate correlated data
    nObs, size0, size1, sigma1 = 10000, 5, 5, .25
    x, cols = generateData(nObs, size0, size1, sigma1)
    print([(j + 1, size0 + i) for i, j in enumerate(cols, 1)])

    # 2) compute and plot correl matrix
    cov, corr = x.cov(), x.corr()
    plotCorrMatrix('HRP3_corr0.png', corr, labels=corr.columns)

    # 3) cluster
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    df0 = corr.loc[sortIx, sortIx]  # reorder
    plotCorrMatrix('HRP3_corr1.png', df0, labels=df0.columns)
    # 4) Capital allocation
    hrp = getRecBipart(cov, sortIx)
    print('Allocation', hrp, sep='\n')
    return


if __name__ == '__main__':
    main()
