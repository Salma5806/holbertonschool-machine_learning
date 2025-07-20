#!/usr/bin/env python3
"""
Applying the simple
change from weights
and state matrix to
probabilities for the agent
using softmax function
"""
import numpy as np

def policy(matrix, weight):
    """
    - computing the scores to have the
    unnormalized probability of each action given a certain state
    - normalize
    - applying the softmax function to get probabilities
    """
    scores = np.dot(matrix, weight)  # (n_states x 1)
    # Softmax with numerical stability
    z = np.exp(scores - np.max(scores))
    probs = z / np.sum(z)
    return probs.reshape(1, -1)  # Keep 2D format for compatibility

def policy_gradient(state, weight):
    """
    Computing the needed gradient for
    the Monte Carlo policy gradient
    REINFORCE
    """
    probs = policy(state, weight)  # shape (1, n_actions)
    action = np.random.choice(len(probs[0]), p=probs[0])
    probs[0, action] -= 1  # derivative of log-softmax
    grad = np.dot(state.T, probs)
    return action, grad
