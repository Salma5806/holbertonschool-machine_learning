#!/usr/bin/env python3
"""  shuffles the data points in two matrices the same way """

import numpy as np

def shuffle_data(X, Y):
    """
    Shuffle the data points in two matrices X and Y the same way.

    Args:
    - X: numpy.ndarray of shape (m, nx) to shuffle
      - m: number of data points
      - nx: number of features in X
    - Y: numpy.ndarray of shape (m, ny) to shuffle
      - m: number of data points (should be the same as in X)
      - ny: number of features in Y

    Returns:
    - A tuple containing the shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
