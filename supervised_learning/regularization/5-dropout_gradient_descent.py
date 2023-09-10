#!/usr/bin/env python3
"""
Defines function that updates the weights with Dropout regularization
using gradient descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights with Dropout regularization using gradient descent"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        if i > 1:
            D = cache['D' + str(i - 1)]
            dA = np.dot(W.T, dZ)
            dA *= D
            dA /= keep_prob
            dZ = dA * (1 - np.tanh(cache['A' + str(i - 1)]) ** 2)
        
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
