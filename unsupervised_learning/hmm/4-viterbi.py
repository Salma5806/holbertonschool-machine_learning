#!/usr/bin/env python3
"""Task 4"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of
    hidden states for a hidden Markov model"""
    # Shape of the Emission array
    N, M = Emission.shape

    # Number of observations
    T = Observation.shape[0]

    # Initialize Viterbi path probs array V & backpointers array B
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    # Initial probabilities for the first observation
    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Loop over the rest of observations
    for t in range(1, T):
        for n in range(N):
            Transitions = Transition[:, n] * V[:, t - 1]
            Emission_prob = Emission[n, Observation[t]]
            V[n, t] = np.max(Transitions * Emission_prob)
            B[n, t] = np.argmax(Transitions)

    # Backtrack to find the most likely path
    path = [np.argmax(V[:, -1])]
    for t in range(T - 1, 0, -1):
        path.append(B[path[-1], t])
    path = path[::-1]

    # Probability of obtaining the path sequence
    P = np.max(V[:, -1])

    return path, P
