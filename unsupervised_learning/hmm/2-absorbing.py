#!/usr/bin/env python3
"""Task 2"""

import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if np.min(P) < 0 or np.max(P) > 1:
        return False
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return False

    n = P.shape[0]
    D = np.diag(np.diag(P))
    if np.any(D == 1):
        return True

    Q = P.copy()
    np.fill_diagonal(Q, 0)
    for _ in range(n):
        Q = np.linalg.matrix_power(Q, 2)
        if np.any(np.diag(Q) == 1):
            return True

    return False
