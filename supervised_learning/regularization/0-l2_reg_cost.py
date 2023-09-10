#!/usr/bin/env python3
"""
Defines a function that calculates the cost of a neural network
using L2 Regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    regularization_cost = 0

    for i in range(1, L + 1):
        weight = weights['W' + str(i)]
        regularization_cost += np.sum(np.square(weight))

    L2_regularization_cost = (lambtha / (2 * m)) * regularization_cost
    total_cost = cost + L2_regularization_cost

    return total_cost
