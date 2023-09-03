#!/usr/bin/env python3
"""  calculates the normalization (standardization) constants of a matrix """

import numpy as np


def normalization_constants(X):
    """
    Calculate the mean and standard deviation of each feature in matrix X.

    Args:
    - X: numpy.ndarray of shape (m, nx) to calculate normalization constants
      - m: number of data points
      - nx: number of features

    Returns:
    - A tuple containing the mean and standard 
    deviation of each feature,respectively.
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    return mean, std_dev
