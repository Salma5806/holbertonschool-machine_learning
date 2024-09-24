#!/usr/bin/env python3
"""Task 4"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model.

    Args:
        X (np.ndarray): Data set of shape (n, d).
        k (int): Number of clusters.

    Returns:
            - pi: Prior probabilities for each cluster, shape (k,).
            - m: Centroid means for each cluster, shape (k, d).
            - S: Covariance matrices for each cluster, shape (k, d, d)."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    # Initialize pi evenly
    pi = np.full(k, 1 / k)

    # Initialize m
    m, _ = kmeans(X, k)

    # Initialize S as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
