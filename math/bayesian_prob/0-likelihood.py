#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def comb(n, k):
    """
    Calculate the binomial coefficient C(n, k).

    Parameters:
    - n: Total number of items.
    - k: Number of items to choose.

    Returns:
    - Binomial coefficient C(n, k).
    """
    if 0 <= k <= n:
        return np.math.factorial(n) // (
            np.math.factorial(k) * np.math.factorial(n - k))
    else:
        return 0


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihoods = np.array([
        comb(n, x) * (p**x) * ((
            1 - p)**(n - x)) for p in P])

    return likelihoods
