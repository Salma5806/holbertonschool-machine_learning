#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chain
    being in a particular state after a specified
    number of iterations"""
    # Check if P is a square 2D numpy.ndarray
    if type(P) is not (
        np.ndarray or
        len(P.shape) != 2 or
        P.shape[0] != P.shape[1]
    ):
        return None

    # Check if s is a numpy.ndarray of shape (1, n)
    if type(s) is not (
        np.ndarray or
        len(s.shape) != 2 or
        s.shape[0] != 1 or
        s.shape[1] != P.shape[0]
    ):
        return None

    # Check if t is an integer & greater than 0
    if type(t) is not int or t < 1:
        return None

    # Compute the state probabilities after t iterations
    state_prob = np.dot(s, np.linalg.matrix_power(P, t))

    return state_prob
