#!/usr/bin/env python3
"""Mean and Covariance Module"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0)
    X_minus_mean = X - mean
    cov = np.dot(X_minus_mean.T, X_minus_mean) / (X.shape[0] - 1)

    return np.reshape(mean, (mean.shape[0], 1)), cov
