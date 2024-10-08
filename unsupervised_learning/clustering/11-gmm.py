#!/usr/bin/env python3
"""Task 11"""

import sklearn.mixture


def gmm(X, k):
    """Placeholder"""
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
