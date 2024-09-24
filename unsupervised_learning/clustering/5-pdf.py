#!/usr/bin/env python3
"""Task 5"""

import numpy as np


def pdf(X, m, S):
    """Placeholder"""
    if (
        type(X) is not np.ndarray or
        type(m) is not np.ndarray or
        type(S) is not np.ndarray
    ):
        return None

    if (
        len(X.shape) != 2 or
        len(S.shape) != 2 or
        len(m.shape) != 1
    ):
        return None

    _, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    if X.ndim != 2:
        return None

    if not np.allclose(S, S.T):
        return None

    det_S = np.linalg.det(S)
    if det_S <= 0:
        return None

    inv_S = np.linalg.inv(S)
    diff = X - m

    normalization = 1.0 / (np.sqrt((2 * np.pi) ** d * det_S))
    exponent = -0.5 * np.sum(np.dot(diff, inv_S) * diff, axis=1)
    P = normalization * np.exp(exponent)

    P = np.maximum(P, 1e-300)  # Set a minimum value

    return P
