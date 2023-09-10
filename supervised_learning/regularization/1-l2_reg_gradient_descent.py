#!/usr/bin/env python3
"""
Defines a function that updates the weights and biases
using gradient descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2 regularization"""
    m = Y.shape[1]

    for i in range(L, 0, -1):
        Ai = cache['A' + str(i)]
        Ai_minus_1 = cache['A' + str(i - 1)]
        Wi = weights['W' + str(i)]
        bi = weights['b' + str(i)]
        Zi = np.dot(Wi, Ai_minus_1) + bi
        if i == L:
            dZi = Ai - Y
        else:
            dZi = np.dot(weights['W' + str(i + 1)].T, dZi) * (1 - np.power(Ai, 2))

        dWi = np.dot(dZi, Ai_minus_1.T) / m + (lambtha / m) * Wi
        dbi = np.sum(dZi, axis=1, keepdims=True) / m

        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * dbi
