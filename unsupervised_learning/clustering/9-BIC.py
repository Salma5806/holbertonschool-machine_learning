#!/usr/bin/env python3
"""Task 9"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a
    GMM using the Bayesian Information Criterion"""
    if kmax is None:
        kmax = X.shape[0]
    n, d = X.shape
    l_like = np.empty(kmax - kmin + 1)
    b = np.empty_like(l_like)
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None:
            return None, None, None, None
        p = k * d * (d + 1) / 2 + d * k + k - 1
        l_like[k - kmin] = log_l
        b[k - kmin] = p * np.log(n) - 2 * log_l
        results.append((pi, m, S))
    best_k = np.argmin(b) + kmin
    best_result = results[best_k - kmin]
    return best_k, best_result, l_like, b
