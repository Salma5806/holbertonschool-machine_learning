#!/usr/bin/env python3
"""Task 3"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs forward algorithm for a hidden Markov model"""
    # Shape of the Emission array
    N, M = Emission.shape

    # Number of observations
    T = Observation.shape[0]

    # Initialize forward path probabilities array F
    F = np.zeros((N, T))

    # Initial probabilities for the first observation
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Loop over the rest of observations
    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n] * F[:, t - 1]
            Emission_prob = Emission[n, Observation[t]]
            F[n, t] = np.sum(Transitions * Emission_prob)

    # Likelihood of the observations given the model
    P = np.sum(F[:, -1])

    return P, F
