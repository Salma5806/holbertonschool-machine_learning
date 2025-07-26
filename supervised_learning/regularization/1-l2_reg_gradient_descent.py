#!/usr/bin/env python3
"""
Defines a function that updates the weights and biases
using gradient descent with L2 Regularization
"""

import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent with L2 regularization.

    Parameters:
    - Y: one-hot numpy.ndarray of shape (classes, m), true labels
    - weights: dict of weights and biases
    - cache: dict of outputs of each layer
    - alpha: learning rate
    - lambtha: L2 regularization parameter
    - L: number of layers in the network

    Updates weights in place.
    """
    m = Y.shape[1]

    # Initialize dZ for the last layer (softmax)
    A_final = cache['A' + str(L)]
    dZ = A_final - Y  # shape: (classes, m)

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]

        # Gradient of weights with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = weights['b' + str(l)] - alpha * db

        if l > 1:
            # Derivative of tanh activation: 1 - tanh^2(z) = 1 - A_l^2
            A_curr = cache['A' + str(l)]
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.power(A_curr, 2))
