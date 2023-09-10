#!/usr/bin/env python3
"""
Defines a function that updates the weights and biases
using gradient descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using 
    gradient descent with L2 regularization"""
    m = Y.shape[1]

    for l in range(L, 0, -1):
        Al = cache["A" + str(l)]
        Al_1 = cache["A" + str(l - 1)]
        Wl = weights["W" + str(l)]
        bl = weights["b" + str(l)]
        Al_1 = cache["A" + str(l - 1)]

        if l == L:
            dZl = Al - Y
        else:
            dZl = dAl * (1 - (Al ** 2))  # Derivative of tanh

        dWl = (1 / m) * np.matmul(dZl, Al_1.T) + (lambtha / m) * Wl
        dbl = (1 / m) * np.sum(dZl, axis=1, keepdims=True)

        dAl = np.matmul(Wl.T, dZl)
        weights["W" + str(l)] -= alpha * dWl
        weights["b" + str(l)] -= alpha * dbl
