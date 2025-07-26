#!/usr/bin/env python3
"""
Defines function that updates weights using gradient descent with Dropout regularization
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a neural network using gradient descent with dropout.

    Args:
        Y: numpy.ndarray of shape (classes, m), one-hot true labels
        weights: dict of weights and biases
        cache: dict containing activations and dropout masks
        alpha: learning rate
        keep_prob: probability of keeping a node active during dropout
        L: number of layers in the network
    """
    m = Y.shape[1]
    # derivative at output layer (softmax + cross-entropy)
    dZ = cache['A' + str(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            W = weights['W' + str(l)]
            dA_prev = np.matmul(W.T, dZ)

            # Apply dropout mask and scale the gradient
            dA_prev *= cache['D' + str(l - 1)]
            dA_prev /= keep_prob

            # Derivative of tanh activation: 1 - A_prev^2
            dZ = dA_prev * (1 - A_prev ** 2)
