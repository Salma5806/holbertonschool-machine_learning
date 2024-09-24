#!/usr/bin/env python3
"""Task 7"""

import numpy as np


def maximization(X, g):
    """Placeholder"""
    try:
        k, n = g.shape
        n, d = X.shape

        # Calculate updated priors (pi)
        pi = np.sum(g, axis=1) / n

        # Calculate updated centroid means (m)
        m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

        # Calculate updated covariance matrices (S)
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            S[i] = np.dot((g[i][:, np.newaxis] * diff).T, diff) / np.sum(g[i])

        return pi, m, S

    except Exception:
        return None, None, None
