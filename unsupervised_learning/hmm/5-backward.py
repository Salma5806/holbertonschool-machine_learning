#!/usr/bin/env python3
"""Task 5"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs backward algorithm for a hidden Markov model"""
    # Shape of the Emission array
    N, M = Emission.shape

    # Number of observations
    T = Observation.shape[0]

    # Initialize backward path probabilities array B
    B = np.zeros((N, T))

    # Initial probabilities for last observation
    B[:, -1] = 1

    # Loop over the rest of observations in reverse order
    for t in range(T - 2, -1, -1):
        for n in range(N):
            Transitions = Transition[n, :] * Emission[:, Observation[t + 1]]
            B[n, t] = np.sum(Transitions * B[:, t + 1])

    # Likelihood of observations given the model
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
