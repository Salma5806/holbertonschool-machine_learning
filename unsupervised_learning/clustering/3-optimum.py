#!/usr/bin/env python3
"""Task 3"""

import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Placeholder"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax >= X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    results = []
    d_vars = []
    vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        vars.append(variance(X, C))

    for var in vars:
        d_vars.append(vars[0] - var)

    return results, d_vars
