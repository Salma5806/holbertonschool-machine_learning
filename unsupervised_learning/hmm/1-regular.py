#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def regular(P):
    """Determines the steady state
    probabilities of a regular markov chain"""
    # Check if P is a square 2D numpy.ndarray
    if type(P) is not (
        np.ndarray or
        len(P.shape) != 2 or
        P.shape[0] != P.shape[1]
    ):
        return None

    # Check if P is a regular transition matrix
    n = P.shape[0]
    if not (np.linalg.matrix_power(P, n*n) > 0).all():
        return None

    # Compute the steady state probabilities
    w, v = np.linalg.eig(P.T)
    steady_state = np.real(v[:, np.isclose(w, 1)].flatten())
    steady_state /= steady_state.sum()

    return steady_state[np.newaxis]
