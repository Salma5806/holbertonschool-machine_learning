#!/usr/bin/env python3
"""
Policy Gradient: compute Monte-Carlo policy gradient
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix.

    matrix: numpy.ndarray of shape (1, n) representing the state
    weight: numpy.ndarray of shape (n, m) representing the weights

    Returns: numpy.ndarray of shape (1, m) representing action probabilities
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes Monte-Carlo policy gradient.

    state: numpy.ndarray of shape (n,) representing the current observation
    weight: numpy.ndarray of shape (n, m) representing the weights

    Returns: action, gradient
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    probs = probs.flatten()
    action = np.random.choice(len(probs), p=probs)
    action_one_hot = np.zeros_like(probs)
    action_one_hot[action] = 1
    grad = np.dot(state.T, (action_one_hot - probs).reshape(1, -1))
    return action, grad
