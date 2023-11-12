#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def pca(X, ndim):
    """Placeholder"""
    # Standardize data (mean centering)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Calculate e-vals & e-vecs of covariance matrix
    _, _, Vt = np.linalg.svd(X_centered)

    # Project data onto new feature space
    T = np.dot(X_centered, Vt[:ndim].T)

    return T
