#!/usr/bin/env python3
"""  Normalize (standardize) a matrix X  """


def normalize(X, m, s):
    """
    Normalize (standardize) a matrix X.

    Args:
    - X: numpy.ndarray of shape (d, nx) to normalize
      - d: number of data points
      - nx: number of features
    - m: numpy.ndarray of shape (nx,) containing the mean of all features of X
    - s: numpy.ndarray of shape (nx,) containing the standard
    deviation of all features of X

    Returns:
    - The normalized X matrix
    """

    assert X.shape[1] == m.shape[0] == s.shape[0]
    normalized_X = (X - m) / s

    return normalized_X
