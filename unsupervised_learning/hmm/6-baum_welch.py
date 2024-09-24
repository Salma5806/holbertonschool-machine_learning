#!/usr/bin/env python3
"""Task 6"""

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
    return F


def backward(Observation, Emission, Transition):
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
    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs Baum-Welch algorithm for a hidden Markov model"""
    # Shape of the Emission array
    N, M = Emission.shape

    # Number of observations
    T = Observations.shape[0]

    # Loop over the number of iterations
    for _ in range(iterations):

        # Compute forward and backward probabilities
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition)

        xi = np.zeros((N, N, T - 1))

        # Compute xi and gamma values
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(F[:, t].T, Transition) * Emission[
                    :, Observations[t + 1]].T, B[:, t + 1])
            for i in range(N):
                numerator = F[
                    i, t] * Transition[i, :] * Emission[
                        :, Observations[t + 1]].T * B[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        # Update transition matrix
        Transition = np.sum(xi, axis=2) / np.sum(
            gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((
            gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = np.zeros((N, M))

        # Update emission matrix
        for n in range(N):
            for m in range(M):
                K[n, m] = np.sum(
                    gamma[n, Observations == m]) / np.sum(gamma[n, :])

        Emission = K

    return Transition, Emission
